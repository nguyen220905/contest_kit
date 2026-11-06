"""
[d8_dones_calc] = d3_hybrid_title + ONE change, only for IFMIF-DONES-domain questions (_DONES_RE,
same regex as d6/d7; AI-paper questions are answered exactly as D3):
  after D3's answer, a calculation check —
    turn 2: "write ONE Python block recomputing every derived number (or deriving the asked
            quantity from values in the context), else reply NO_CALC";
    no code / NO_CALC / code rejected / error / timeout -> D3's answer unchanged;
    else the code runs in a separate `python -I` process (10 s timeout, math-only imports) and
    turn 3 asks for the final answer using the printed values.
  Why: D7 showed D3 gets only 7/15 DONES vs 29/35 AI; 4/10 DONES sample questions need arithmetic
  (volume from dimensions, flow ratio, decay constant) and the judge rejects 0.01306 for 0.01308.

[d3_hybrid_title] = d1_02_safe + ONE change, the retrieval stack measured as config c9 in
eval/retrieval_eval.py (bundled dsqa/ package, same code as the lab):
  - chunks: dsqa.chunk.chunk_pages on the same Docling markdown (800/150; paragraphs longer
    than 800 chars are split on lines), each chunk indexed with its document title prefixed
    ('[title]\\n' + text) -- the prompt still gets the plain chunk text;
  - ranking: hybrid = Reciprocal Rank Fusion (k=60) of dense MiniLM (same model, CPU) and
    BM25 (rank_bm25), best-ranked first (D2 showed best-first ordering matters).
  Lab (sample25 quotes, gold corpus): recall@1 0.30 -> 0.40, recall@20 0.60 -> 0.75, MRR 0.366 -> 0.470.

[d1_02_safe] = starter_kit/02_rag/main.py + robustness patches only:
  1. PDFs found recursively and case-insensitively (rglob, suffix.lower() == ".pdf").
  2. Model load and index build are separated and cached INCLUDING failure:
     a failure is never retried on later questions (no reload-per-question -> OOM).
       - index build fails or yields 0 chunks -> index None -> answered with no
         context (zero-shot, same system prompt);
       - model load fails -> every question returns "" immediately.
  3. answer() wraps everything in try/except: on error it empties the CUDA cache,
     prints the error to stderr and returns "" (always a str).
Everything else (prompt, chunking, budget, generation, embedder) is unchanged.

RAG baseline — IEEE-CIS Technical Challenge on Domain-Specific Question
Answering with LLMs.

Blind retrieval over the whole corpus: every PDF in corpus_dir is parsed with
Docling, chunked and embedded once per run, then for each question every
chunk is ranked by cosine similarity and the best-ranked ones are packed
into a fixed context budget before asking the base model. No document_id is
known to this code — the same "search has to find it" design as a real
participant would face.

Needs a local copy of Docling's layout model and of the embedding model next
to this file (see build.sh): the evaluation environment has no internet
access, so neither can be downloaded there the first time they're used.
"""

import re
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))  # bundled dsqa/ package
from dsqa.chunk import chunk_pages  # noqa: E402
from dsqa.retrieve import Retriever  # noqa: E402

RETRIEVE_K = 200  # ranked candidates handed to the packer (10k tokens ~ 50 chunks)

SYSTEM_PROMPT = (
    "You are a scientific expert. Answer the following question as accurately "
    "and concisely as possible based on your knowledge."
)

MAX_NEW_TOKENS = 2048
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# A local copy, not the "sentence-transformers/all-MiniLM-L6-v2" repo id:
# same reason as the Docling model below — small, but still a model the
# evaluation environment doesn't have pre-downloaded, and there's no
# internet there to fetch it by name (see build.sh).
EMBED_MODEL_PATH = str(Path(__file__).parent / "embed_model")

# Fixed context budget, in tokens. Deliberately NOT "as much as fits the
# model's real context window" (Olmo-3.1-32B-Instruct's is 65536): measured
# directly, filling anywhere near that runs the GPU (80 GB) out of memory
# during prefill, and even a context that does fit paired with a long real
# answer can take minutes for a single question. 10,000 keeps every
# question fast and safe — tested end to end against real questions.
MAX_CONTEXT_TOKENS = 10_000

# Loaded once and reused across every question in the run (answer() is
# called once per question, in the same process).
_cache = {}


def _docling_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(
        artifacts_path=str(Path(__file__).parent / "docling_models"),
        do_ocr=False,              # the corpus is born-digital text, not scans
        do_table_structure=False,  # avoids needing cv2/libGL, not in the image
    )
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})


def _chunk_markdown(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Greedily pack paragraphs into ~chunk_size-char chunks, so cuts land on
    paragraph boundaries instead of mid-sentence, carrying the tail of each
    chunk into the next one for a bit of overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def _build_index(corpus_dir: str):
    """Parse + chunk + index (dense + BM25) every PDF in corpus_dir. Runs once per submission."""
    converter = _docling_converter()

    all_chunks = []
    pdfs = sorted(p for p in Path(corpus_dir).rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    for pdf_path in pdfs:
        try:
            result = converter.convert(str(pdf_path))
            text = result.document.export_to_markdown()
        except Exception as e:
            # One bad PDF shouldn't take down the whole run.
            print(f"[rag] skipping {pdf_path.name}: {e}")
            continue
        all_chunks.extend(chunk_pages(pdf_path.stem, [{"page": None, "text": text}],
                                      CHUNK_SIZE, CHUNK_OVERLAP, prefix_title=True))

    print(f"[rag] parsed {len(pdfs)} PDFs into {len(all_chunks)} chunks")
    if not all_chunks:
        return None
    retriever = Retriever(all_chunks, mode="hybrid", embed_model=EMBED_MODEL_PATH, device="cpu")
    return all_chunks, retriever


_FAILED = "failed"


def _get_index(corpus_dir: str):
    """Index tuple or None (build failed / empty corpus). Built at most once."""
    key = ("index", corpus_dir)
    if key not in _cache:
        try:
            _cache[key] = _build_index(corpus_dir)
        except Exception:
            print("[rag] index build failed, answering without context:", file=sys.stderr)
            traceback.print_exc()
            _cache[key] = None
    return _cache[key]


def _get_model(model_id: str):
    """(tokenizer, model) or None if loading failed. Loaded at most once."""
    key = ("model", model_id)
    if key not in _cache:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="cuda",
            )
            model.eval()
            _cache[key] = (tokenizer, model)
        except Exception:
            print("[rag] model load failed, all answers will be empty:", file=sys.stderr)
            traceback.print_exc()
            _cache[key] = _FAILED
    return None if _cache[key] == _FAILED else _cache[key]


def _build_context(question, chunks, retriever, tokenizer, max_tokens=MAX_CONTEXT_TOKENS):
    """As many of the top-ranked chunks (blind, corpus-wide, hybrid rank) as fit in
    max_tokens, best first. Falls back to a truncated slice of the single best chunk
    when even that alone doesn't fit."""
    order = [i for i, _ in retriever.search(question, k=RETRIEVE_K)]

    selected, used = [], 0
    for i in order:
        chunk = chunks[i]["text"]
        n = len(tokenizer.encode(chunk))
        if used + n > max_tokens:
            if not selected and n > 0:
                ratio = max_tokens / n
                selected.append(chunk[: int(len(chunk) * ratio)])
            break
        selected.append(chunk)
        used += n

    if not selected:
        return ""
    body = "\n\n".join(f"[Fragment {i}]\n{c}" for i, c in enumerate(selected, 1))
    return f"Context extracted from a blind search over the corpus:\n\n{body}\n\n---\n\n"


# IFMIF-DONES-domain questions (all 10 DONES sample questions match, none of the 15 AI ones) — same as D6/D7
_DONES_RE = re.compile(r"IFMIF|DONES|LIPAc|lithium|Li loop|deuteron|neutron|irradiat|Be-?7|tritium|beam(?! search)",
                       re.IGNORECASE)

CALC_CHECK_PROMPT = (
    "Check the numbers in your answer. If your answer contains a number you obtained by calculation "
    "(not copied verbatim from the context), or if the question asks for a quantity that is not stated "
    "explicitly but can be derived from values in the context or standard constants (for example a volume "
    "from dimensions, a ratio of two flow rates, a decay constant from a half-life, a volumetric flow from a "
    "mass flow and a density), write ONE Python code block that computes each such quantity from its input "
    "values and prints it with a label and units. Use only the Python standard library (math).\n"
    "If no calculation is involved, reply exactly: NO_CALC"
)
REVISE_PROMPT = (
    "Output of your code:\n```\n{out}\n```\n"
    "Now give your final answer to the original question. Use these computed values (rounded sensibly) "
    "wherever they differ from your previous answer, and keep everything else that was correct."
)
CALC_CHECK_TOKENS = 1024
_CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S | re.I)
_ALLOWED_IMPORTS = {"math", "fractions", "decimal", "statistics"}
_FORBIDDEN = re.compile(r"__|\bopen\s*\(|\bexec\s*\(|\beval\s*\(|\bcompile\s*\(|\binput\s*\(|\bglobals\b|\blocals\b"
                        r"|\bgetattr\b|\bsetattr\b|\bdelattr\b|\bvars\s*\(|\bbreakpoint\b|\bos\b|\bsys\b")


def _generate(tokenizer, model, messages, max_new_tokens=MAX_NEW_TOKENS) -> str:
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()


def _safe_code(text: str):
    """The single Python block of a calc-check reply, or None (NO_CALC / no block / disallowed code)."""
    blocks = _CODE_RE.findall(text or "")
    if not blocks:
        return None
    code = blocks[0].strip()
    mods = re.findall(r"^[ \t]*(?:from[ \t]+(\w+)|import[ \t]+([\w \t,.]+))", code, re.M)
    names = set()
    for frm, imp in mods:  # "import math as m, fractions" -> {"math", "fractions"}
        names |= {frm} if frm else {n.split()[0].split(".")[0] for n in imp.split(",") if n.strip()}
    if not code or "print" not in code or names - _ALLOWED_IMPORTS or _FORBIDDEN.search(code):
        return None
    return code


def _run_code(code: str):
    """stdout of `python -I -c code` (10 s timeout, ≤ 2000 chars) or None on error/empty."""
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    out = (r.stdout or "").strip()
    return out[:2000] if r.returncode == 0 and out else None


def _calc_check(tokenizer, model, messages, first: str):
    """Revised answer after running the model's own calculation code, or None to keep `first`."""
    msgs = messages + [{"role": "assistant", "content": first}, {"role": "user", "content": CALC_CHECK_PROMPT}]
    reply = _generate(tokenizer, model, msgs, CALC_CHECK_TOKENS)
    code = _safe_code(reply)
    if code is None:
        return None
    out = _run_code(code)
    if out is None:
        return None
    msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": REVISE_PROMPT.format(out=out)}]
    final = _generate(tokenizer, model, msgs)
    print(f"[calc] code ran ({len(code)} chars) -> revised answer", file=sys.stderr)
    return final or None


def answer(question: str, corpus_dir: str, model_id: str) -> str:
    try:
        loaded = _get_model(model_id)
        if loaded is None:
            return ""
        tokenizer, model = loaded
        index = _get_index(corpus_dir)
        context = ""
        if index is not None:
            chunks, retriever = index
            context = _build_context(question, chunks, retriever, tokenizer)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context + question},
        ]
        first = _generate(tokenizer, model, messages)
        if not _DONES_RE.search(question or ""):
            return first
        try:
            return _calc_check(tokenizer, model, messages, first) or first
        except Exception:
            print("[calc] check failed, keeping first answer:", file=sys.stderr)
            traceback.print_exc()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return first
    except Exception:
        print("[rag] answer() failed for one question:", file=sys.stderr)
        traceback.print_exc()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return ""
