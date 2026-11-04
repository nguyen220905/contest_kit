"""Phòng thí nghiệm retrieval (không LLM).

    .venv\\Scripts\\python eval/retrieval_eval.py --corpus data/corpus/gold --split sample25 \\
        --configs c0,c1,c2,c3,c4,c5,c6,c7 --run results/retr_gold_v1
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsqa.chunk import chunk_pages  # noqa: E402
from dsqa.parse import find_pdfs, parse_docling, parse_pymupdf  # noqa: E402
from dsqa.retrieve import Retriever  # noqa: E402

WORK = ROOT / "data" / "work"
MODELS = WORK / "models"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
KS = [1, 3, 5, 10, 20]

# model name -> (repo id, kind, query prefix)
MODEL_SPECS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "st", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", "st", BGE_PREFIX),
    "bge-base": ("BAAI/bge-base-en-v1.5", "st", BGE_PREFIX),
    "msmarco": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "ce", ""),
    "mxbai-xs": ("mixedbread-ai/mxbai-rerank-xsmall-v1", "ce", ""),
    "mxbai-base": ("mixedbread-ai/mxbai-rerank-base-v1", "ce", ""),
    "gte-rerank": ("Alibaba-NLP/gte-reranker-modernbert-base", "ce", ""),
    "bge-rerank": ("BAAI/bge-reranker-base", "ce", ""),
    # bản fp16 tự lưu từ gte-rerank (299 MB, để zip < 1 GB) — chỉ có ở local, không tải được theo tên
    "gte-rerank-fp16": ("local/gte-reranker-modernbert-base-fp16", "ce", ""),
}

# Mỗi cấu hình đổi 1 thứ so với cái trước (xem docstring từng dòng).
CONFIGS = {
    "c0": dict(desc="D1: docling + dense minilm", parser="docling", mode="dense", embed="minilm"),
    "c1": dict(desc="pymupdf + dense minilm", parser="pymupdf", mode="dense", embed="minilm"),
    "c2": dict(desc="pymupdf + bm25", parser="pymupdf", mode="bm25"),
    "c3": dict(desc="pymupdf + hybrid(minilm,bm25)", parser="pymupdf", mode="hybrid", embed="minilm"),
    "c4": dict(desc="c3 with bge-small", parser="pymupdf", mode="hybrid", embed="bge-small"),
    "c5": dict(desc="c3 with bge-base", parser="pymupdf", mode="hybrid", embed="bge-base"),
    "c6": dict(desc="c4 + rerank msmarco top50", parser="pymupdf", mode="hybrid", embed="bge-small",
               rerank="msmarco", rerank_n=50),
    "c7": dict(desc="c4 + prefix_title", parser="pymupdf", mode="hybrid", embed="bge-small",
               prefix_title=True),
    # tách từng yếu tố trên chunk docling của D1 (ứng viên D3 = D1 + đúng 1 thay đổi retrieval)
    "c8": dict(desc="docling + dense minilm + prefix_title", parser="docling", mode="dense", embed="minilm",
               prefix_title=True),
    "c9": dict(desc="docling + hybrid(minilm,bm25) + prefix_title", parser="docling", mode="hybrid",
               embed="minilm", prefix_title=True),
    "c10": dict(desc="docling + hybrid(bge-small,bm25) + prefix_title", parser="docling", mode="hybrid",
                embed="bge-small", prefix_title=True),
    "c11": dict(desc="pymupdf + hybrid(minilm,bm25) + prefix_title", parser="pymupdf", mode="hybrid",
                embed="minilm", prefix_title=True),
    # 2/11: định tuyến theo tiêu đề + kích thước chunk (trên c11, pymupdf cho nhanh)
    "c13": dict(desc="c11 + title_rrf (3rd RRF list: chunks of top-3 title-matching docs)", parser="pymupdf",
                mode="hybrid", embed="minilm", prefix_title=True, title_rrf=True),
    "c14": dict(desc="c11 with chunks 1200/200", parser="pymupdf", mode="hybrid", embed="minilm",
                prefix_title=True, size=1200, overlap=200),
    "c15": dict(desc="c11 with chunks 1600/250", parser="pymupdf", mode="hybrid", embed="minilm",
                prefix_title=True, size=1600, overlap=250),
    "c16": dict(desc="c11 with chunks 400/80", parser="pymupdf", mode="hybrid", embed="minilm",
                prefix_title=True, size=400, overlap=80),
    "c17": dict(desc="c9 (docling) + title_rrf", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, title_rrf=True),
    "c18": dict(desc="c9 (docling) with chunks 1200/200", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, size=1200, overlap=200),
    "c19": dict(desc="c14 (pymupdf 1200/200) with bge-small", parser="pymupdf", mode="hybrid",
                embed="bge-small", prefix_title=True, size=1200, overlap=200),
    # 3/11: c9 (D3) + cross-encoder rerank top 50 (chunk kèm tiêu đề), đuôi danh sách giữ nguyên
    "c20": dict(desc="c9 + rerank mxbai-xsmall top50", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="mxbai-xs", rerank_n=50),
    "c21": dict(desc="c9 + rerank mxbai-base top50", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="mxbai-base", rerank_n=50),
    "c22": dict(desc="c9 + rerank gte-modernbert top50", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="gte-rerank", rerank_n=50),
    "c23": dict(desc="c9 + rerank bge-reranker-base top50", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="bge-rerank", rerank_n=50),
    "c24": dict(desc="c9 + rerank msmarco top50", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="msmarco", rerank_n=50),
    # Olmo làm reranker (dsqa/llm_rerank.py): lab dùng 7B 4-bit thay cho 32B của server
    "c25": dict(desc="c9 + rerank Olmo-7B yes/no top30", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="olmo7b", rerank_n=30),
    "c26": dict(desc="c9 + rerank Olmo-7B yes/no top50", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="olmo7b", rerank_n=50),
    "c27": dict(desc="c22 with fp16 weights (D5)", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="gte-rerank-fp16", rerank_n=50),
    "c28": dict(desc="c27 top20", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="gte-rerank-fp16", rerank_n=20),
    "c29": dict(desc="c27 top100", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="gte-rerank-fp16", rerank_n=100),
    "c30": dict(desc="c27 top200", parser="docling", mode="hybrid", embed="minilm",
                prefix_title=True, rerank="gte-rerank-fp16", rerank_n=200),
}
BUDGETS = [2000, 5000, 10000]  # token budget của context (D1/D3 ghép tới 10k)
SEARCH_K = 200


_TOK = None


def ntok(text):
    """Số token Olmo (cùng vocab 7B/32B) — giống bộ ghép context của D1."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct")
    return len(_TOK.encode(text))


def dir_mb(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6


def ensure_model(repo_id, dest, kind="st"):
    """Tải repo_id về dest (nếu chưa có) bằng SentenceTransformer/CrossEncoder.save()."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if kind == "ce":
        from sentence_transformers import CrossEncoder
        CrossEncoder(repo_id).save(str(dest))
    else:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(repo_id).save(str(dest))
    print(f"[model] saved {repo_id} -> {dest} ({dir_mb(dest):.1f} MB)")
    return str(dest)


def model_path(name):
    repo, kind, _ = MODEL_SPECS[name]
    return ensure_model(repo, MODELS / repo.split("/")[-1], kind)


_LLM = {}


def llm_reranker(device):
    """Olmo-3-7B 4-bit (như dev_shim) làm reranker — gần đúng cho Olmo-32B trên server."""
    if "olmo7b" not in _LLM:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from dsqa.llm_rerank import LLMReranker
        repo = "allenai/Olmo-3-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(repo, device_map=device, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        _LLM["olmo7b"] = LLMReranker(model, AutoTokenizer.from_pretrained(repo))
    return _LLM["olmo7b"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(s):
    s = (s or "").lower().replace("-", "")
    return re.sub(r"[\W_]+", " ", s).strip()


def load_doc_map():
    m = {}
    p = ROOT / "data" / "manifests" / "corpus_gold.csv"
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                m[Path(r["file"]).name.lower()] = r["doc_id"]
    return m


def load_split(name):
    p = Path(name) if name.endswith(".jsonl") else ROOT / "data" / "splits" / f"{name}.jsonl"
    p = p if p.is_absolute() else ROOT / p
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- parse (cached)
def parse_corpus(parser, pdfs, docling_models):
    """-> ({pdf: pages}, wall_s_this_run, {'pages': n, 'secs': original parse secs})"""
    out, t0, tot_pages, tot_secs = {}, time.time(), 0, 0.0
    cdir = WORK / "parse_cache" / parser
    cdir.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        cf = cdir / f"{sha256_file(pdf)}.json"
        if cf.exists():
            d = json.loads(cf.read_text(encoding="utf-8"))
        else:
            t = time.time()
            try:
                pages = parse_pymupdf(pdf) if parser == "pymupdf" else parse_docling(pdf, docling_models)
            except Exception as e:  # one bad PDF must not kill the run
                print(f"[parse] {parser} failed on {pdf.name}: {e}", file=sys.stderr)
                pages = []
            secs = time.time() - t
            try:
                import fitz
                with fitz.open(str(pdf)) as dd:
                    npages = len(dd)
            except Exception:
                npages = len(pages)
            d = {"pages": pages, "secs": secs, "npages": npages}
            cf.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        out[pdf] = d["pages"]
        tot_pages += d["npages"]
        tot_secs += d["secs"]
    return out, time.time() - t0, {"pages": tot_pages, "secs": tot_secs}


# ---------------------------------------------------------------- eval
def evaluate(retr, chunks, rows, doc_ids_known):
    from rapidfuzz import fuzz
    t0 = time.time()
    per_q = []
    for r in rows:
        ev = r.get("evidence") or []
        if r.get("evidence_status") not in ("found", "partial") or not ev:
            continue
        quotes = [q for q in (norm(e.get("quote"))[:300] for e in ev) if q]
        docs = {e.get("doc_id") for e in ev if e.get("doc_id")}
        res = retr.search(r["question"], SEARCH_K)
        hit_rank = doc_rank = hit_ctx_end = None
        used = 0  # token đã ghép vào context (bộ ghép D1: thêm chunk nguyên vẹn tới khi hết ngân sách)
        for rank, (ci, _) in enumerate(res, 1):
            c = chunks[ci]
            if c.get("_ntok") is None:
                c["_ntok"] = ntok(c["text"])
            used += c["_ntok"]
            if doc_rank is None and c["doc"] in docs:
                doc_rank = rank
            if hit_rank is None:
                ct = norm(c["text"])
                if any(len(ct) >= len(q) and fuzz.partial_ratio(q, ct) >= 90 for q in quotes):
                    hit_rank, hit_ctx_end = rank, used
            if (hit_rank and doc_rank) or used > max(BUDGETS):
                break
        per_q.append(dict(id=r["id"], domain=r.get("domain", ""), hit_rank=hit_rank,
                          doc_rank=doc_rank, hit_ctx_end=hit_ctx_end, doc_in_corpus=bool(docs & doc_ids_known)))
    t_query = time.time() - t0

    def agg(qs):
        n = len(qs)
        if not n:
            return {"n": 0}
        o = {"n": n}
        for k in KS:
            o[f"quote_recall@{k}"] = sum(1 for q in qs if q["hit_rank"] and q["hit_rank"] <= k) / n
        for k in (1, 5):
            o[f"doc_hit@{k}"] = sum(1 for q in qs if q["doc_rank"] and q["doc_rank"] <= k) / n
        o["mrr"] = sum(1 / q["hit_rank"] for q in qs if q["hit_rank"]) / n
        for b in BUDGETS:
            o[f"in_ctx@{b // 1000}k"] = sum(1 for q in qs if q["hit_ctx_end"] and q["hit_ctx_end"] <= b) / n
        return o

    by_dom = defaultdict(list)
    for q in per_q:
        by_dom[q["domain"]].append(q)
    return agg(per_q), {d: agg(v) for d, v in sorted(by_dom.items())}, per_q, t_query


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--split", default="sample25")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--docling-models", default=None, help="artifacts_path của docling (mặc định: cache HF)")
    a = ap.parse_args()

    device = a.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(a.run if Path(a.run).is_absolute() else ROOT / a.run)
    run.mkdir(parents=True, exist_ok=True)
    (WORK / "emb_cache").mkdir(parents=True, exist_ok=True)

    pdfs = []
    for d in a.corpus:
        pdfs += find_pdfs(d if Path(d).is_absolute() else ROOT / d)
    if not pdfs:
        raise SystemExit("Không tìm thấy PDF nào trong --corpus")
    dmap = load_doc_map()
    doc_of = {p: dmap.get(p.name.lower(), p.stem) for p in pdfs}
    rows = load_split(a.split)
    print(f"[corpus] {len(pdfs)} PDFs; split {a.split}: {len(rows)} câu; device={device}")

    parsed, chunk_memo, emb_memo, summary = {}, {}, {}, []
    for name in [c.strip() for c in a.configs.split(",") if c.strip()]:
        if name not in CONFIGS:
            raise SystemExit(f"cấu hình không có: {name} (có: {','.join(CONFIGS)})")
        cfg = CONFIGS[name]
        parser = cfg["parser"]
        if parser not in parsed:
            parsed[parser] = parse_corpus(parser, pdfs, a.docling_models)
        pages_by_pdf, t_parse, pinfo = parsed[parser]

        ck = (parser, bool(cfg.get("prefix_title")), cfg.get("size", 800), cfg.get("overlap", 150))
        if ck not in chunk_memo:
            ch = []
            for pdf in pdfs:
                ch += chunk_pages(doc_of[pdf], pages_by_pdf[pdf], ck[2], ck[3], prefix_title=ck[1])
            chunk_memo[ck] = ch
        chunks = chunk_memo[ck]

        emb, t_embed, embed_path, prefix = None, 0.0, None, ""
        if cfg.get("embed"):
            embed_path = model_path(cfg["embed"])
            prefix = MODEL_SPECS[cfg["embed"]][2]
            key = hashlib.sha256((cfg["embed"] + "\n" + "\x00".join(c["etext"] for c in chunks))
                                 .encode("utf-8")).hexdigest()
            npy = WORK / "emb_cache" / f"{cfg['embed']}_{key[:32]}.npy"
            if key in emb_memo:
                emb = emb_memo[key]
            elif npy.exists():
                import numpy as np
                emb = np.load(npy)
            emb_memo_key = key
        rr = cfg.get("rerank")
        rerank_path = model_path(rr) if rr and rr != "olmo7b" else None
        rerank_fn = llm_reranker(device) if rr == "olmo7b" else None

        retr = Retriever(chunks, mode=cfg["mode"], embed_model=embed_path, query_prefix=prefix,
                         device=device, rerank_model=rerank_path, rerank_n=cfg.get("rerank_n", 50),
                         embeddings=emb, title_rrf=cfg.get("title_rrf", False), rerank_fn=rerank_fn)
        if cfg.get("embed"):
            if emb is None:
                import numpy as np
                np.save(npy, retr.embeddings)
                t_embed = retr.t_embed
            emb_memo[emb_memo_key] = retr.embeddings

        overall, by_dom, per_q, t_query = evaluate(retr, chunks, rows, set(doc_of.values()))
        res = dict(config=name, desc=cfg["desc"], n_chunks=len(chunks), overall=overall,
                   by_domain=by_dom, t_parse_s=round(t_parse, 2), t_embed_s=round(t_embed, 2),
                   t_query_s=round(t_query, 2),
                   parse_orig_secs=round(pinfo["secs"], 2), parse_pages=pinfo["pages"],
                   parse_s_per_page=round(pinfo["secs"] / max(1, pinfo["pages"]), 4), per_question=per_q)
        (run / f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        srow = dict(config=name, desc=cfg["desc"], n=overall.get("n"), chunks=len(chunks))
        for k in KS:
            srow[f"recall@{k}"] = round(overall.get(f"quote_recall@{k}", 0), 3)
        srow.update({"doc_hit@1": round(overall.get("doc_hit@1", 0), 3),
                     "doc_hit@5": round(overall.get("doc_hit@5", 0), 3),
                     "mrr": round(overall.get("mrr", 0), 3)})
        for b in BUDGETS:
            srow[f"in_ctx@{b // 1000}k"] = round(overall.get(f"in_ctx@{b // 1000}k", 0), 3)
        for d, v in by_dom.items():
            srow[f"{d}:n"] = v["n"]
            srow[f"{d}:recall@5"] = round(v.get("quote_recall@5", 0), 3)
            srow[f"{d}:doc_hit@5"] = round(v.get("doc_hit@5", 0), 3)
            srow[f"{d}:mrr"] = round(v.get("mrr", 0), 3)
        srow.update(t_parse=res["t_parse_s"], t_embed=res["t_embed_s"], t_query=res["t_query_s"],
                    parse_s_per_page=res["parse_s_per_page"])
        summary.append(srow)
        print(f"[{name}] {cfg['desc']}: " + " ".join(f"{k}={v}" for k, v in srow.items()
                                                      if k not in ("config", "desc")))

    cols = []
    for r in summary:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(run / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    print(f"-> {run / 'summary.csv'}")
    for name, (repo, kind, _) in MODEL_SPECS.items():
        p = MODELS / repo.split("/")[-1]
        if p.exists():
            print(f"[model size] {repo}: {dir_mb(p):.1f} MB")


if __name__ == "__main__":
    main()
