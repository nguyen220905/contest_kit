"""Corpus gold cho 25 câu mẫu (Stage 3): tải PDF open access + kiểm quote bằng chứng.

  fetch.py add   --url URL --doc-id ID --site SITE --license TEXT --samples 3,7 [--note ...] [--force]
  fetch.py check --evidence data/work/sample25_evidence.jsonl
  fetch.py find  --doc-id ID --pattern REGEX [--context 200]   (tìm đoạn để chép quote)

PDF -> data/corpus/gold/<doc_id>.pdf ; manifest -> data/manifests/corpus_gold.csv.
Quote được so với text trang (PyMuPDF) sau chuẩn hoá: NFKC, bỏ gạch nối cuối dòng, gộp khoảng trắng,
chữ thường; rapidfuzz partial_ratio >= 90 -> found.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import pymupdf
import requests
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "data" / "corpus" / "gold"
MANIFEST = ROOT / "data" / "manifests" / "corpus_gold.csv"
COLUMNS = ["doc_id", "file", "url", "site", "sha256", "pages", "license", "sample_ids", "note"]
UA = "ieee-dsqa-corpus-research/0.1 (open-access PDF fetch for a QA benchmark; polite, 1 req/3s)"
POLITE_DELAY_S = 3.0
THRESHOLD = 90.0
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("­", "").translate(_DASHES)
    t = re.sub(r"(?<=\w)-[ \t]*\n\s*(?=\w)", "", t)  # gạch nối cuối dòng
    return re.sub(r"\s+", " ", t).strip().lower()


def page_texts(pdf_path: Path) -> list[str]:
    with pymupdf.open(pdf_path) as doc:
        return [normalize(p.get_text()) for p in doc]


def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def cmd_add(a) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.doc_id):
        sys.exit(f"doc_id không hợp lệ: {a.doc_id}")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    pdf = GOLD_DIR / f"{a.doc_id}.pdf"
    if pdf.exists() and not a.force:
        print(f"đã có {pdf.name}, không tải lại (dùng --force để tải lại)")
    else:
        time.sleep(POLITE_DELAY_S)  # lệnh chạy tuần tự -> >= 3 s giữa các request
        r = requests.get(a.url, headers={"User-Agent": UA}, timeout=120, allow_redirects=True)
        r.raise_for_status()
        if b"%PDF" not in r.content[:1024]:
            sys.exit(f"không phải PDF (content-type={r.headers.get('content-type')}, {len(r.content)} B): {a.url}")
        pdf.write_bytes(r.content)
    data = pdf.read_bytes()
    if b"%PDF" not in data[:1024]:
        sys.exit(f"file không phải PDF: {pdf}")
    with pymupdf.open(pdf) as doc:
        pages = doc.page_count
    rows = read_manifest()
    old = next((r for r in rows if r["doc_id"] == a.doc_id), None)
    samples = {s.strip() for s in a.samples.split(",") if s.strip()}
    if old:
        samples |= {s for s in old["sample_ids"].split(",") if s}
    row = {
        "doc_id": a.doc_id,
        "file": pdf.relative_to(ROOT).as_posix(),
        "url": a.url,
        "site": a.site,
        "sha256": hashlib.sha256(data).hexdigest(),
        "pages": pages,
        "license": a.license,
        "sample_ids": ",".join(sorted(samples, key=lambda s: (len(s), s))),
        "note": a.note if a.note is not None else (old or {}).get("note", ""),
    }
    rows = [row if r["doc_id"] == a.doc_id else r for r in rows] if old else rows + [row]
    write_manifest(rows)
    print(f"{a.doc_id}: {pages} trang, {len(data)/1e6:.2f} MB, sha256 {row['sha256'][:12]}… -> {row['file']}")
    return 0


def cmd_find(a) -> int:
    texts = page_texts(GOLD_DIR / f"{a.doc_id}.pdf")
    pat = re.compile(a.pattern, re.I)
    for i, t in enumerate(texts, 1):
        for m in pat.finditer(t):
            s, e = max(0, m.start() - a.context), min(len(t), m.end() + a.context)
            print(f"--- p{i} @{m.start()}: …{t[s:e]}…")
    return 0


def cmd_check(a) -> int:
    recs = [json.loads(l) for l in Path(a.evidence).read_text(encoding="utf-8").splitlines() if l.strip()]
    cache: dict[str, list[str]] = {}
    tot: dict[str, list[int]] = {}
    print(f"{'id':>3} {'domain':<9} {'status':<9} {'doc_id':<28} {'page':>4} {'score':>6} result")
    for r in recs:
        dom = r.get("domain", "?")
        ok_any = False
        for ev in r.get("evidence", []):
            path = GOLD_DIR / f"{ev['doc_id']}.pdf"
            if not path.exists():
                print(f"{r['id']:>3} {dom:<9} {r.get('status',''):<9} {ev['doc_id']:<28} {ev['page']:>4} {'-':>6} MISSING_PDF")
                continue
            if ev["doc_id"] not in cache:
                cache[ev["doc_id"]] = page_texts(path)
            texts = cache[ev["doc_id"]]
            q = normalize(ev["quote"])
            p = int(ev["page"])
            score = fuzz.partial_ratio(q, texts[p - 1]) if 1 <= p <= len(texts) else 0.0
            ok = score >= THRESHOLD
            ok_any |= ok
            extra = ""
            if not ok:
                best = max(range(len(texts)), key=lambda i: fuzz.partial_ratio(q, texts[i]))
                extra = f" (best p{best+1}: {fuzz.partial_ratio(q, texts[best]):.0f})"
            print(f"{r['id']:>3} {dom:<9} {r.get('status',''):<9} {ev['doc_id']:<28} {p:>4} {score:6.1f} "
                  f"{'found' if ok else 'NOT_FOUND'}{extra}")
        if not r.get("evidence"):
            print(f"{r['id']:>3} {dom:<9} {r.get('status',''):<9} {'-':<28} {'-':>4} {'-':>6} no_evidence")
        t = tot.setdefault(dom, [0, 0])
        t[0] += ok_any
        t[1] += 1
    print("---")
    for dom, (ok, n) in tot.items():
        print(f"{dom}: {ok}/{n} câu có >= 1 quote xác nhận")
    ok_all = sum(v[0] for v in tot.values())
    n_all = sum(v[1] for v in tot.values())
    print(f"TỔNG: {ok_all}/{n_all} câu có >= 1 quote xác nhận (partial_ratio >= {THRESHOLD:.0f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("--url", required=True)
    p.add_argument("--doc-id", required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--samples", required=True, help="vd 3,7")
    p.add_argument("--note")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("--evidence", default=str(ROOT / "data" / "work" / "sample25_evidence.jsonl"))
    p = sub.add_parser("find")
    p.add_argument("--doc-id", required=True)
    p.add_argument("--pattern", required=True)
    p.add_argument("--context", type=int, default=200)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    return {"add": cmd_add, "check": cmd_check, "find": cmd_find}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
