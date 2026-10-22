"""Eval chuẩn: (run_pipeline) -> judge -> metrics.json.

    .venv\\Scripts\\python eval/evaluate.py --submission <dir> --split sample25 --corpus <dir> --run results/<run>
    .venv\\Scripts\\python eval/evaluate.py --run results/<run> --judge-only [--refs data/work/sample25_refs.jsonl]
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import judge as J  # noqa: E402

csv.field_size_limit(2**31 - 1)
WINDOW = 4096


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def acc(rows):
    return 100.0 * sum(int(r["verdict"]) for r in rows) / len(rows) if rows else None


def group_acc(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r.get(key) or ""].append(r)
    return {k: round(acc(v), 2) for k, v in sorted(g.items())}


def retrieval_metrics(rows, evidence):
    """quote_recall@k + in_window_hit; None khi thiếu debug hoặc evidence."""
    from rapidfuzz import fuzz
    ks = [1, 3, 5, "all"]
    hits, n, win = {str(k): 0 for k in ks}, 0, 0
    for r in rows:
        quotes = [norm(e.get("quote")) for e in evidence.get(str(r["id"]), []) if e.get("quote")]
        if not quotes or not r.get("debug"):
            continue
        try:
            dbg = json.loads(r["debug"])
            frags = dbg["fragments"]
        except Exception:
            continue
        n += 1
        match = [any(fuzz.partial_ratio(q, norm(fr.get("text"))) >= 90 for q in quotes) for fr in frags]
        for k in ks:
            sub = match if k == "all" else match[:k]
            hits[str(k)] += int(any(sub))
        pt = dbg.get("prompt_tokens")
        if pt is not None and any(m and fr.get("tok_end", 0) > pt - WINDOW for m, fr in zip(match, frags)):
            win += 1
    if n == 0:
        return None, None, 0
    return {k: round(v / n, 4) for k, v in hits.items()}, round(win / n, 4), n


def load_evidence(split, path):
    ev = {}
    try:
        import run_pipeline
        for r in run_pipeline.load_split(split):
            if r.get("evidence"):
                ev[str(r["id"])] = r["evidence"]
    except SystemExit:
        pass
    if path and Path(path).exists():
        for l in Path(path).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                ev[str(r["id"])] = r.get("evidence", [])
    return ev


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge-only", action="store_true")
    ap.add_argument("--provider", choices=["deepseek", "openai"], default="deepseek")
    ap.add_argument("--ref", choices=["auto", "short", "long"], default="auto")
    ap.add_argument("--refs", default=None)
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--submission")
    ap.add_argument("--split", default="sample25")
    ap.add_argument("--corpus")
    ap.add_argument("--quant")
    ap.add_argument("--limit")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--model-id")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if not a.judge_only:
        if not a.submission:
            raise SystemExit("Thiếu --submission (hoặc dùng --judge-only)")
        cmd = [sys.executable, str(ROOT / "eval" / "run_pipeline.py"), "--submission", a.submission,
               "--split", a.split, "--run", a.run]
        for flag, v in (("--corpus", a.corpus), ("--quant", a.quant), ("--limit", a.limit), ("--model-id", a.model_id)):
            if v:
                cmd += [flag, str(v)]
        cmd += [f for f, on in (("--offline", a.offline), ("--overwrite", a.overwrite)) if on]
        subprocess.run(cmd, check=True)

    run = Path(a.run)
    rows = J.judge_run(run, a.ref, a.provider, a.refs)
    info = {}
    if (run / "run_info.json").exists():
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    ev = load_evidence(a.split, a.evidence)
    recall, in_win, n_ret = retrieval_metrics(rows, ev)

    dom = group_acc(rows, "domain")
    ref_counts = defaultdict(int)
    for r in rows:
        ref_counts[r["judge_ref"]] += 1
    m = {
        "n": len(rows),
        "balanced_acc": round(sum(dom.values()) / len(dom), 2) if dom else None,
        "acc_by_domain": dom,
        "acc_by_question_type": group_acc(rows, "question_type"),
        "n_correct": sum(int(r["verdict"]) for r in rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "judge_provider": a.provider,
        "judge_ref_counts": dict(ref_counts),
        "overhead_s": info.get("overhead_s"),
        "latency_p50_s": info.get("latency_p50_s"), "latency_p95_s": info.get("latency_p95_s"),
        "projected_public_s": info.get("projected_public_s"), "projected_private_s": info.get("projected_private_s"),
        "tokens_p50": info.get("tokens_p50"), "tokens_p95": info.get("tokens_p95"),
        "quote_recall_at_k": recall, "in_window_hit": in_win, "retrieval_n": n_ret,
    }
    (run / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"run {run.name}: n={m['n']} correct={m['n_correct']} errors={m['n_errors']} "
          f"balanced_acc={m['balanced_acc']}")
    print(f"  domain: {dom}\n  type: {m['acc_by_question_type']}")
    print(f"  time: overhead={m['overhead_s']} p50={m['latency_p50_s']} p95={m['latency_p95_s']} "
          f"public={m['projected_public_s']} private={m['projected_private_s']} tok p50/p95={m['tokens_p50']}/{m['tokens_p95']}")
    print(f"  retrieval: recall={recall} in_window={in_win}  | API mới {J._stats['calls']}, cache {J._stats['hits']}")


if __name__ == "__main__":
    main()
