"""Chạy bài nộp (main.py với answer()) tuần tự, 1 process, giống server; ghi predictions.csv + run_info.json.

    .venv\\Scripts\\python eval/run_pipeline.py --submission starter_kit/01_zero_shot --run results/exp001_zs
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = ["id", "domain", "question_type", "question", "reference_short", "reference_long",
        "answer", "latency_s", "n_new_tokens", "error", "peak_vram_gib", "debug"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--split", default="sample25")
    ap.add_argument("--run", required=True)
    ap.add_argument("--model-id", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--budget-s", type=float, default=3600)
    ap.add_argument("--n-public", type=int, default=50)
    ap.add_argument("--n-private", type=int, default=100)
    ap.add_argument("--private-budget-s", type=float, default=7200)
    return ap.parse_args()


def load_split(name):
    p = Path(name)
    path = p if p.suffix == ".jsonl" else ROOT / "data" / "splits" / f"{name}.jsonl"
    if path.exists():
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if name == "sample25":
        raw = json.loads((ROOT / "starter_kit" / "sample" / "sample_qa.json").read_text(encoding="utf-8"))
        return [{"id": r["id"], "domain": r["domain"], "question_type": r.get("question_type", ""),
                 "question": r["question"], "reference_short": "", "reference_long": r["answer"],
                 "evidence": [], "source": "sample"} for r in raw]
    raise SystemExit(f"Không thấy split: {path}")


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def _ver(name):
    try:
        import importlib.metadata as m
        return m.version(name)
    except Exception:
        return None


def main():
    for s in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, không in được tiếng Việt
        s.reconfigure(encoding="utf-8", errors="replace")
    args = parse()
    if args.offline:  # phải đặt TRƯỚC khi import transformers/torch
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run = Path(args.run)
    if (run / "predictions.csv").exists() and not args.overwrite:
        raise SystemExit(f"{run}/predictions.csv đã có; dùng --overwrite để ghi đè")
    run.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus) if args.corpus else run / "empty_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    rows_in = load_split(args.split)
    if args.limit:
        rows_in = rows_in[: args.limit]

    import torch
    import transformers
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / "eval"))
    import dev_shim
    dev_shim.install(None if args.quant == "none" else args.quant)

    sub = Path(args.submission).resolve()
    main_py = sub / "main.py"
    sys.path.insert(0, str(sub))
    spec = importlib.util.spec_from_file_location("submission_main", main_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    cuda = torch.cuda.is_available()
    started = datetime.now().isoformat(timespec="seconds")
    t_start = time.perf_counter()
    out = []
    # ghi từng dòng + flush: run dài bị kill giữa chừng vẫn giữ được các câu đã xong
    f = open(run / "predictions.csv", "w", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for r in rows_in:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        if hasattr(module, "LAST_DEBUG"):
            module.LAST_DEBUG = None  # tránh mang debug của câu trước sang
        err, ans = "", ""
        t0 = time.perf_counter()
        try:
            res = module.answer(r["question"], str(corpus), args.model_id)
            if isinstance(res, str):
                ans = res
            else:
                err = f"non-str: {type(res).__name__}"
        except Exception as e:
            err = f"{type(e).__name__}: {e} | {traceback.format_exc().strip().splitlines()[-1]}"
        lat = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
        ntok = len(tok.encode(ans, add_special_tokens=False)) if ans else 0
        dbg = ""
        if getattr(module, "LAST_DEBUG", None) is not None:
            dbg = json.dumps(module.LAST_DEBUG, default=str, ensure_ascii=False)
        out.append({"id": r["id"], "domain": r.get("domain", ""), "question_type": r.get("question_type", ""),
                    "question": r["question"], "reference_short": r.get("reference_short", ""),
                    "reference_long": r.get("reference_long", ""), "answer": ans,
                    "latency_s": round(lat, 3), "n_new_tokens": ntok, "error": err,
                    "peak_vram_gib": round(peak, 3), "debug": dbg})
        w.writerow(out[-1])
        f.flush()
        print(f"[{len(out)}/{len(rows_in)}] id={r['id']} {lat:.1f}s {ntok} tok" + (f" ERR {err}" if err else ""),
              flush=True)
    f.close()
    total = time.perf_counter() - t_start

    lats = [o["latency_s"] for o in out]
    rest = lats[1:]
    if rest:
        overhead = max(0.0, lats[0] - statistics.median(rest))
        mean = statistics.mean(rest)
    else:
        overhead = lats[0] if lats else 0.0
        mean = 0.0
    toks = [o["n_new_tokens"] for o in out]
    proj_pub = overhead + mean * args.n_public
    proj_priv = overhead + mean * args.n_private
    if proj_priv <= args.private_budget_s:
        within = args.n_private
    else:
        within = int(max(0.0, args.private_budget_s - overhead) // mean) if mean > 0 else 0
    info = {
        "args": vars(args),
        "main_py_sha256": hashlib.sha256(main_py.read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
        "bitsandbytes": _ver("bitsandbytes"),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "quant": args.quant, "model_id": args.model_id,
        "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
        "n_questions": len(out), "total_s": round(total, 2), "overhead_s": round(overhead, 2),
        "latency_mean_s": round(mean, 2) if rest else None,
        "latency_p50_s": pct(rest, 0.5), "latency_p95_s": pct(rest, 0.95),
        "tokens_p50": pct(toks, 0.5), "tokens_p95": pct(toks, 0.95), "tokens_max": max(toks) if toks else None,
        "n_errors": sum(1 for o in out if o["error"]),
        "n_nonstr": sum(1 for o in out if o["error"].startswith("non-str")),
        "projected_public_s": round(proj_pub, 1), "projected_private_s": round(proj_priv, 1),
        "public_over_budget": proj_pub > args.budget_s,
        "private_questions_within_budget": within,
    }
    (run / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nXong {len(out)} câu trong {total:.0f}s | overhead {overhead:.0f}s | lỗi {info['n_errors']} "
          f"| tokens p50/p95/max {info['tokens_p50']}/{info['tokens_p95']}/{info['tokens_max']}")
    print(f"Ước tính public ({args.n_public} câu): {proj_pub:.0f}s | private ({args.n_private}): {proj_priv:.0f}s "
          f"(xong {within} câu trong {args.private_budget_s:.0f}s)")
    if info["public_over_budget"]:
        print(f"VƯỢT GIỚI HẠN public: {proj_pub:.0f}s > {args.budget_s:.0f}s")


if __name__ == "__main__":
    main()
