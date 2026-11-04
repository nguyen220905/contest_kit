"""Tập dev tổng hợp từ corpus C200 (gold 18 + nhiễu 182 PDF): mỗi câu có đáp án + quote nguyên văn.

    .venv\\Scripts\\python scripts/make_syn_dev.py --limit 6          # thử vài câu
    .venv\\Scripts\\python scripts/make_syn_dev.py                    # đủ: AI 1 câu/tài liệu, DONES 2 câu/tài liệu

Quy trình mỗi ứng viên (tất định: seed cố định, LLM temperature 0, cache ở data/work/llm_cache.jsonl):
  1. Chọn cửa sổ ~3500 ký tự trong thân bài (bỏ phần References) của text Docling đã cache.
  2. Generator (mặc định GPT qua OpenRouter) viết 1 câu hỏi kiểu chuyên gia theo question_type được bốc thăm,
     tự đứng được (nêu tên paper/hệ thống vì retrieval mù), kèm reference_short/long + 1–3 quote nguyên văn.
  3. Lọc: quote phải có trong passage; verifier (DeepSeek) trả lời từ passage + đánh giá standalone/well_posed;
     judge-v2 so câu trả lời verifier với reference_short → sai = câu mơ hồ → loại.
Ra: data/work/syn/<name>.jsonl (câu giữ, cùng schema sample25) + <name>_all.jsonl (mọi ứng viên + lý do loại).
"""
import argparse
import csv
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import judge as J  # noqa: E402

GEN_VERSION = "syn-gen-v1"
VER_VERSION = "syn-ver-v1"
WIN_CHARS = 3500
HEAD_CHARS = 1500
TYPES = {  # trọng số theo sample25 (DONES 10 câu, AI 15 câu)
    "DONES": {"factual": 3, "procedural": 4, "causal": 1, "comparative": 1, "predictive": 1},
    "AI Papers": {"factual": 4, "procedural": 1, "causal": 6, "comparative": 2, "predictive": 2},
}
FALSE_PREMISE_P = 0.08  # sample25: 2/25
TYPE_DEF = {
    "factual": "asks for a specific fact, value, name, setting or definition stated in the text",
    "procedural": "asks how something is done or computed, or asks for a value obtained by applying the described "
                  "procedure (may need a short calculation, unit conversion or ratio from numbers in the text)",
    "causal": "asks WHY something happens, or the reason/motivation for a design choice or a result",
    "comparative": "asks for the difference, trade-off or advantage between two things discussed in the text",
    "predictive": "asks what would happen / which value or choice follows under a stated condition or hypothetical "
                  "(e.g. applying a rule, heuristic or trend from the text to a new case)",
}

GEN_SYSTEM = ("You write hard, expert-level evaluation questions for a domain question-answering benchmark. "
              "You follow the constraints exactly and reply with one JSON object only.")
GEN_USER = """We are building a benchmark like the IEEE CIS "Domain-Specific QA with LLMs" challenge. A QA system will search a \
library of several hundred PDFs (accelerator / fusion-facility reports and AI research papers) WITHOUT knowing which document \
a question comes from, then answer in free text. A judge compares its answer to your reference answer.

DOCUMENT OPENING (for the title / system name only):
<<<
{head}
>>>

PASSAGE (the question must be answerable from this passage):
<<<
{passage}
>>>

Write ONE question of type "{qtype}": it {tdef}.{fp}

Rules:
- Standalone: the question must identify its context so it can be found by search. {ctx_rule} Never write "this paper", \
"the passage", "the authors", "the document" without naming it.
- Expert style, 1-2 sentences, like a researcher asking a colleague. Paraphrase: do not copy more than 5 consecutive words \
from the passage, except names, technical terms, symbols and numbers.
- The answer must be specific and checkable (numbers with units, named methods/components, explicit reasons), and must \
follow from the passage plus at most basic general knowledge (arithmetic, unit conversion, standard constants).
- Do not ask about authors, affiliations, dates of publication, figure/table/section numbers, citations or funding.
- If the passage has no substantive technical content (e.g. only references, author lists, boilerplate), set "skip": true.
- quotes: 1-3 excerpts copied EXACTLY, character for character, from the PASSAGE (each one sentence or one table row, \
at most 250 characters) that together support the answer.

Reply with JSON only:
{{"skip": false, "question": "...", "reference_short": "<one sentence with the key fact(s)>", \
"reference_long": "<2-5 sentences>", "quotes": ["..."], "calc": <true if a calculation is needed>, \
"false_premise": {fp_bool}}}"""

FP_TEXT = ("\nThis must be a FALSE-PREMISE question: it presupposes a specific claim that the passage contradicts "
           "(e.g. a component, value or property that does not exist or is different). reference_short must start with "
           "\"premise is false:\" and state the correct fact.")
CTX_AI = "Name the paper (short title) or its named method/model/framework, e.g. \"In the paper 'X', ...\" or \"In X, ...\"."
CTX_ACC = ("Name the facility/system/subsystem, e.g. \"In IFMIF-DONES, ...\", \"For the LIPAc RFQ, ...\", \"At the ESS "
           "linac, ...\"; a general physics/technique question is fine if it is self-contained.")

VER_SYSTEM = "You check candidate questions for a QA benchmark. Reply with one JSON object only."
VER_USER = """PASSAGE (from a document whose opening is: "{head}"):
<<<
{passage}
>>>

QUESTION: {question}

Tasks:
1. answer: answer the question using the passage (plus basic arithmetic / general knowledge). If the question rests on a \
false premise, say so and give the correct fact.
2. standalone: would a reader who has NOT seen the passage know which paper / facility / system the question is about, \
well enough to find it in a library of several hundred papers? (true/false; a fully general physics/ML question counts as true)
3. well_posed: does the question have one clear, checkable answer in the passage? (true/false)

Reply with JSON only: {{"answer": "...", "standalone": true|false, "well_posed": true|false, "issue": "<short, or empty>"}}"""

REF_RE = re.compile(r"\n#{1,6}\s*(?:[\dIVX]+\.?\s*)?(references|bibliography|acknowledge?ments?)\b", re.I)


def norm(s):  # = eval/retrieval_eval.norm
    s = (s or "").lower().replace("-", "")
    return re.sub(r"[\W_]+", " ", s).strip()


def load_docs():
    """-> [{doc_id, domain, pages}] cho 200 tài liệu C200 (bỏ bản trùng 'dup-of')."""
    dom_of_sample = {}
    for l in (ROOT / "data" / "splits" / "sample25.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            dom_of_sample[str(r["id"])] = r["domain"]
    docs = []
    for name in ("corpus_gold.csv", "corpus_distractors.csv"):
        with open(ROOT / "data" / "manifests" / name, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if "dup-of" in r["note"]:
                    continue
                if name == "corpus_gold.csv":
                    ids = [x for x in re.split(r"[,\s]+", r["sample_ids"]) if x]
                    domain = dom_of_sample.get(ids[0], "AI Papers") if ids else "AI Papers"
                else:
                    domain = "AI Papers" if r["note"].startswith("ai") else "DONES"
                cf = ROOT / "data" / "work" / "parse_cache" / "docling" / f"{r['sha256']}.json"
                pages = json.loads(cf.read_text(encoding="utf-8"))["pages"]
                docs.append({"doc_id": r["doc_id"], "domain": domain, "pages": pages})
    return sorted(docs, key=lambda d: d["doc_id"])


def windows(pages, n, rng):
    """n cửa sổ không chồng nhau (theo đoạn) trong thân bài, vị trí 5%–85%."""
    text = re.sub(r"<!-- image -->\s*", "", "\n\n".join(p.get("text") or "" for p in pages))
    m = REF_RE.search(text)
    if m and m.start() > 0.4 * len(text):
        text = text[: m.start()]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []
    offs, o = [], 0
    for p in paras:
        offs.append(o)
        o += len(p) + 2
    total = max(o, 1)
    cand = [i for i, x in enumerate(offs) if 0.05 * total <= x <= 0.85 * total] or list(range(len(paras)))
    out = []
    for k in range(n):
        part = cand[k * len(cand) // n: (k + 1) * len(cand) // n] or cand
        for _ in range(8):  # PDF toán có thể ra text rác (mã glyph) → thử vị trí khác
            i = rng.choice(part)
            buf = []
            while i < len(paras) and sum(len(b) + 2 for b in buf) + len(paras[i]) <= WIN_CHARS:
                buf.append(paras[i])
                i += 1
            if not buf:
                buf = [paras[i][:WIN_CHARS]]
            w = "\n\n".join(buf)
            if len(w) >= 800 and wordiness(w) >= 0.45:
                out.append(w)
                break
    return out


def wordiness(text):
    """Tỉ lệ token là từ chữ thường ≥ 3 ký tự (văn xuôi tiếng Anh ~0.6; text rác ~0.1)."""
    toks = text.split()
    return sum(bool(re.fullmatch(r"[a-z]{3,}[.,;:]?", t)) for t in toks) / max(1, len(toks))


def quote_ok(q, passage_n):
    from rapidfuzz import fuzz
    qn = norm(q)
    if len(qn) < 40:
        return False
    return qn in passage_n or fuzz.partial_ratio(qn, passage_n) >= 95


def page_of(q, pages):
    qn = norm(q)[:120]
    for p in pages:
        if qn and qn in norm(p.get("text")):
            return p.get("page")
    return None


def max_copy(question, passage):
    """Chuỗi từ dài nhất của câu hỏi chép nguyên từ passage."""
    qw, pn = norm(question).split(), " " + norm(passage) + " "
    best = 0
    for i in range(len(qw)):
        for j in range(i + best + 1, len(qw) + 1):
            if " " + " ".join(qw[i:j]) + " " in pn:
                best = j - i
            else:
                break
    return best


def make_tasks(docs, seed, per_domain):
    rng = random.Random(seed)
    tasks = []
    for d in docs:
        for w_i, passage in enumerate(windows(d["pages"], per_domain[d["domain"]], rng)):
            types = TYPES[d["domain"]]
            qtype = rng.choices(list(types), weights=list(types.values()))[0]
            fp = rng.random() < FALSE_PREMISE_P
            tasks.append(dict(doc=d, w=w_i, passage=passage, qtype=qtype, fp=fp))
    for i, t in enumerate(tasks):
        t["id"] = 1001 + i
    return tasks


def run_one(t, gen, ver):
    d, passage = t["doc"], t["passage"]
    head = "\n\n".join(p.get("text") or "" for p in d["pages"])[:HEAD_CHARS]
    rec = {"id": t["id"], "domain": d["domain"], "doc_id": d["doc_id"], "window": t["w"], "question_type": t["qtype"],
           "fp_requested": t["fp"], "keep": False, "reject": ""}
    user = GEN_USER.format(head=head, passage=passage, qtype=t["qtype"], tdef=TYPE_DEF[t["qtype"]],
                           fp=FP_TEXT if t["fp"] else "", fp_bool="true" if t["fp"] else "false",
                           ctx_rule=CTX_AI if d["domain"] == "AI Papers" else CTX_ACC)
    try:
        out = J.chat(gen, [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}], GEN_VERSION)
    except Exception as e:
        rec["reject"] = f"gen_api: {e}"
        return rec
    g = J.last_json(out, "question") or J.last_json(out, "skip") or {}
    if g.get("skip") or not g.get("question"):
        rec["reject"] = "skip" if g.get("skip") else "gen_parse"
        return rec
    rec.update(question=g["question"].strip(), reference_short=str(g.get("reference_short", "")).strip(),
               reference_long=str(g.get("reference_long", "")).strip(), calc=bool(g.get("calc")),
               false_premise=bool(g.get("false_premise")))
    pn = norm(passage)
    quotes = [q for q in (g.get("quotes") or []) if isinstance(q, str) and quote_ok(q, pn)]
    rec["quotes"] = quotes
    rec["max_copy"] = max_copy(rec["question"], passage)
    if not quotes:
        rec["reject"] = "quote_not_in_passage"
        return rec
    if not rec["reference_short"]:
        rec["reject"] = "no_reference"
        return rec
    vuser = VER_USER.format(head=head[:300].replace("\n", " "), passage=passage, question=rec["question"])
    try:
        vout = J.chat(ver, [{"role": "system", "content": VER_SYSTEM}, {"role": "user", "content": vuser}], VER_VERSION)
        v = J.last_json(vout, "answer") or {}
        rec.update(ver_answer=str(v.get("answer", "")), standalone=bool(v.get("standalone")),
                   well_posed=bool(v.get("well_posed")), ver_issue=str(v.get("issue", "")))
        j = J.judge(rec["question"], rec["reference_short"], rec["ver_answer"], ver)
        rec.update(ver_correct=bool(j["verdict"]), ver_reason=j["reason"])
    except Exception as e:
        rec["reject"] = f"ver_api: {e}"
        return rec
    reasons = [r for r, bad in (("not_standalone", not rec["standalone"]), ("ill_posed", not rec["well_posed"]),
                                ("verifier_disagrees", not rec["ver_correct"])) if bad]
    rec["reject"] = ",".join(reasons)
    rec["keep"] = not reasons
    rec["evidence"] = [{"doc_id": d["doc_id"], "page": page_of(q, d["pages"]), "quote": q} for q in quotes]
    return rec


def to_split_row(r, gen_model):
    return {"id": r["id"], "domain": r["domain"], "question_type": r["question_type"], "question": r["question"],
            "reference_short": r["reference_short"], "reference_long": r["reference_long"], "parts": [],
            "evidence": r["evidence"], "evidence_status": "found", "evidence_note": "",
            "flags": {"multi_part": False, "false_premise": r["false_premise"], "calc": r["calc"],
                      "outside_doc": False, "names_paper": r["domain"] == "AI Papers", "closed_book": None},
            "source": f"syn-v1:{gen_model}", "max_copy": r["max_copy"]}


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="syn_v1")
    ap.add_argument("--gen", default="openai", help="provider trong tools/council_config.json")
    ap.add_argument("--ver", default="deepseek")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--ai-per-doc", type=int, default=1)
    ap.add_argument("--dones-per-doc", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ứng viên đầu (rải đều 2 domain)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    docs = load_docs()
    tasks = make_tasks(docs, a.seed, {"AI Papers": a.ai_per_doc, "DONES": a.dones_per_doc})
    print(f"[syn] {len(docs)} tài liệu → {len(tasks)} ứng viên "
          f"(AI {sum(t['doc']['domain'] == 'AI Papers' for t in tasks)}, "
          f"DONES {sum(t['doc']['domain'] == 'DONES' for t in tasks)})")
    if a.limit:
        ai = [t for t in tasks if t["doc"]["domain"] == "AI Papers"]
        dn = [t for t in tasks if t["doc"]["domain"] == "DONES"]
        tasks = ai[: (a.limit + 1) // 2] + dn[: a.limit // 2]

    gen_model = J.provider_cfg(a.gen)["model"]
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(lambda t: run_one(t, a.gen, a.ver), tasks))

    out_dir = ROOT / "data" / "work" / "syn"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{a.limit}" if a.limit else ""
    with open(out_dir / f"{a.name}{suffix}_all.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = [r for r in recs if r["keep"]]
    with open(out_dir / f"{a.name}{suffix}.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(to_split_row(r, gen_model), ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"[syn] giữ {len(kept)}/{len(recs)}: " + ", ".join(
        f"{d} {sum(r['domain'] == d for r in kept)}/{sum(r['domain'] == d for r in recs)}" for d in ("DONES", "AI Papers")))
    print("[syn] lý do loại:", dict(Counter(x for r in recs if not r["keep"] for x in r["reject"].split(","))))
    print("[syn] loại câu (giữ):", dict(Counter(r["question_type"] for r in kept)),
          "| false_premise", sum(r.get("false_premise", False) for r in kept),
          "| max_copy>=8:", sum(r.get("max_copy", 0) >= 8 for r in kept))
    print(f"[syn] API mới {J._stats['calls']}, cache {J._stats['hits']} → {out_dir / (a.name + suffix + '.jsonl')}")


if __name__ == "__main__":
    main()
