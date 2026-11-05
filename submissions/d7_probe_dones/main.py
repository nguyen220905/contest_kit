"""
[d7_probe_dones] = d3_hybrid_title, but questions NOT matching _DONES_RE (the same regex as
d6_gated_rerank) return "" without generating. Diagnostic submission: LB = 50*a/15 if every
DONES question matches and no AI question does -> a = D3's number of correct DONES answers.
Answers to DONES questions are byte-identical to D3 (same code path, greedy).

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


# IFMIF-DONES-domain questions (all 10 DONES sample questions match, none of the 15 AI ones) — same as D6
_DONES_RE = re.compile(r"IFMIF|DONES|LIPAc|lithium|Li loop|deuteron|neutron|irradiat|Be-?7|tritium|beam(?! search)",
                       re.IGNORECASE)


def answer(question: str, corpus_dir: str, model_id: str) -> str:
    if not _DONES_RE.search(question or ""):
        return ""  # probe: AI-domain questions deliberately unanswered
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
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[-1]
        return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
    except Exception:
        print("[rag] answer() failed for one question:", file=sys.stderr)
        traceback.print_exc()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return ""
