"""So sánh 2 run theo id: compare_runs.py RUN_A RUN_B [--show N]"""
import argparse
import csv
from pathlib import Path


def load(run):
    p = Path(run)
    p = p / "predictions.csv" if p.is_dir() else p
    with open(p, encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    A, B = load(a.run_a), load(a.run_b)
    ids = [i for i in A if i in B]
    diff = [i for i in ids if A[i]["answer"].strip() != B[i]["answer"].strip()]
    print(f"câu chung: {len(ids)} | answer khác chữ: {len(diff)}")
    print(f"câu lỗi: A={sum(1 for r in A.values() if r['error'])} B={sum(1 for r in B.values() if r['error'])}")
    for col in ("n_new_tokens", "latency_s"):
        ma = mean([float(A[i][col] or 0) for i in ids])
        mb = mean([float(B[i][col] or 0) for i in ids])
        print(f"{col}: A={ma:.2f} B={mb:.2f} (B-A={mb - ma:+.2f})")
    if ids and "verdict" in A[ids[0]] and "verdict" in B[ids[0]]:
        flips = [i for i in ids if A[i]["verdict"] != B[i]["verdict"]]
        print(f"verdict lật: {len(flips)}")
    for i in diff[: a.show]:
        print(f"\n--- id={i}\nA: {A[i]['answer'].strip()[:300]}\nB: {B[i]['answer'].strip()[:300]}")


if __name__ == "__main__":
    main()
