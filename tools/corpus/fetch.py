"""Corpus gold cho 25 câu mẫu (Stage 3) + corpus nhiễu (Stage 4): tải PDF open access + kiểm quote bằng chứng.

  fetch.py add   --url URL --doc-id ID --site SITE --license TEXT --samples 3,7 [--note ...] [--force]
                 [--dest DIR --manifest CSV]   (mặc định: gold)
  fetch.py check --evidence data/work/sample25_evidence.jsonl
  fetch.py find  --doc-id ID --pattern REGEX [--context 200]   (tìm đoạn để chép quote)
  fetch.py arxiv-search --query 'all:"state space" AND cat:cs.LG' [--max 20] [--topic T --out cand.jsonl]
  fetch.py batch --list items.jsonl|items.csv [--dest DIR --manifest CSV]   (mặc định: distractors)
                 mỗi dòng: url, doc_id, site, license ("auto-arxiv" -> lấy qua OAI-PMH), note
  fetch.py stats [--manifest CSV]   (số PDF theo chủ đề/site, trang, MB, % mở được, % không có text)

PDF -> data/corpus/gold/<doc_id>.pdf ; manifest -> data/manifests/corpus_gold.csv.
Nhiễu -> data/corpus/distractors/<doc_id>.pdf ; data/manifests/corpus_distractors.csv. batch bỏ qua tài liệu
trùng gold (arXiv id hoặc sha256) và trùng sha256 trong manifest đích; >= 3 s giữa các request cùng host.
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
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pymupdf
import requests
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "data" / "corpus" / "gold"
MANIFEST = ROOT / "data" / "manifests" / "corpus_gold.csv"
DIST_DIR = ROOT / "data" / "corpus" / "distractors"
DIST_MANIFEST = ROOT / "data" / "manifests" / "corpus_distractors.csv"
COLUMNS = ["doc_id", "file", "url", "site", "sha256", "pages", "license", "sample_ids", "note"]
UA = "ieee-dsqa-corpus-research/0.1 (open-access PDF fetch for a QA benchmark; polite, 1 req/3s)"
POLITE_DELAY_S = 3.0
MAX_PDF_BYTES = 80_000_000
THRESHOLD = 90.0
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?")
_ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
_last_req: dict[str, float] = {}


def _host_key(url: str) -> str:
    host = urlparse(url).hostname or ""
    return ".".join(host.split(".")[-2:])  # export.arxiv.org ~ arxiv.org


def polite_get(url: str, **kw) -> requests.Response:
    """GET có User-Agent, >= POLITE_DELAY_S giữa các request cùng host trong 1 process."""
    key = _host_key(url)
    wait = POLITE_DELAY_S - (time.monotonic() - _last_req.get(key, -1e9))
    if wait > 0:
        time.sleep(wait)
    try:
        return requests.get(url, headers={"User-Agent": UA}, timeout=120, allow_redirects=True, **kw)
    finally:
        _last_req[key] = time.monotonic()


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("­", "").translate(_DASHES)
    t = re.sub(r"(?<=\w)-[ \t]*\n\s*(?=\w)", "", t)  # gạch nối cuối dòng
    return re.sub(r"\s+", " ", t).strip().lower()


def page_texts(pdf_path: Path) -> list[str]:
    with pymupdf.open(pdf_path) as doc:
        return [normalize(p.get_text()) for p in doc]


def read_manifest(path: Path = MANIFEST) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def arxiv_base_id(text: str) -> str | None:
    """'arxiv-2312.00752v2' / 'https://arxiv.org/pdf/2312.00752v2' -> '2312.00752' (None nếu không có)."""
    i = text.lower().find("arxiv")
    if i < 0:
        return None
    m = _ARXIV_ID.search(text[i:])
    return m.group(1) if m else None


def gold_keys() -> tuple[set[str], set[str]]:
    """(arXiv id gốc, sha256) của toàn bộ tài liệu gold."""
    ids, shas = set(), set()
    for r in read_manifest(MANIFEST):
        shas.add(r["sha256"])
        for t in (r["doc_id"], r["url"]):
            if (i := arxiv_base_id(t)):
                ids.add(i)
    return ids, shas


def fetch_pdf(url: str, doc_id: str, site: str, license_: str, note: str | None, samples: str,
              dest: Path, manifest: Path, force: bool = False, guard_gold: bool = False) -> dict:
    """Tải 1 PDF vào dest/<doc_id>.pdf và upsert dòng manifest. guard_gold: từ chối bản trùng gold."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", doc_id):
        raise ValueError(f"doc_id không hợp lệ: {doc_id}")
    if guard_gold:
        gids, gshas = gold_keys()
        if (aid := arxiv_base_id(url) or arxiv_base_id(doc_id)) and aid in gids:
            raise ValueError(f"trùng gold (arXiv {aid})")
    dest.mkdir(parents=True, exist_ok=True)
    pdf = dest / f"{doc_id}.pdf"
    if pdf.exists() and not force:
        print(f"đã có {pdf.name}, không tải lại (dùng --force để tải lại)")
    else:
        r = polite_get(url)
        r.raise_for_status()
        if b"%PDF" not in r.content[:1024]:
            raise ValueError(f"không phải PDF (content-type={r.headers.get('content-type')}, {len(r.content)} B): {url}")
        if len(r.content) > MAX_PDF_BYTES:
            raise ValueError(f"PDF quá lớn ({len(r.content)/1e6:.0f} MB > {MAX_PDF_BYTES/1e6:.0f}): {url}")
        sha = hashlib.sha256(r.content).hexdigest()
        if guard_gold and sha in gold_keys()[1]:
            raise ValueError(f"trùng gold (sha256 {sha[:12]})")
        pdf.write_bytes(r.content)
    data = pdf.read_bytes()
    if b"%PDF" not in data[:1024]:
        raise ValueError(f"file không phải PDF: {pdf}")
    with pymupdf.open(pdf) as doc:
        pages = doc.page_count
    rows = read_manifest(manifest)
    old = next((r for r in rows if r["doc_id"] == doc_id), None)
    samp = {s.strip() for s in samples.split(",") if s.strip()}
    if old:
        samp |= {s for s in old["sample_ids"].split(",") if s}
    row = {
        "doc_id": doc_id,
        "file": pdf.relative_to(ROOT).as_posix(),
        "url": url,
        "site": site,
        "sha256": hashlib.sha256(data).hexdigest(),
        "pages": pages,
        "license": license_,
        "sample_ids": ",".join(sorted(samp, key=lambda s: (len(s), s))),
        "note": note if note is not None else (old or {}).get("note", ""),
    }
    rows = [row if r["doc_id"] == doc_id else r for r in rows] if old else rows + [row]
    write_manifest(rows, manifest)
    print(f"{doc_id}: {pages} trang, {len(data)/1e6:.2f} MB, sha256 {row['sha256'][:12]}… -> {row['file']}")
    return row


def cmd_add(a) -> int:
    dest = Path(a.dest) if a.dest else GOLD_DIR
    manifest = Path(a.manifest) if a.manifest else MANIFEST
    if not (dest / f"{a.doc_id}.pdf").exists() or a.force:
        time.sleep(POLITE_DELAY_S)  # lệnh chạy tuần tự -> >= 3 s giữa các request
    try:
        fetch_pdf(a.url, a.doc_id, a.site, a.license, a.note, a.samples, dest, manifest, a.force,
                  guard_gold=dest.resolve() != GOLD_DIR.resolve())
    except (ValueError, requests.RequestException) as e:
        sys.exit(str(e))
    return 0


_LICENSES = {
    "creativecommons.org/licenses/by/4.0": "CC BY 4.0",
    "creativecommons.org/licenses/by-sa/4.0": "CC BY-SA 4.0",
    "creativecommons.org/licenses/by-nc-sa/4.0": "CC BY-NC-SA 4.0",
    "creativecommons.org/licenses/by-nc-nd/4.0": "CC BY-NC-ND 4.0",
    "creativecommons.org/licenses/by-nc/4.0": "CC BY-NC 4.0",
    "creativecommons.org/licenses/by/3.0": "CC BY 3.0",
    "creativecommons.org/licenses/by-nc-sa/3.0": "CC BY-NC-SA 3.0",
    "creativecommons.org/publicdomain/zero/1.0": "CC0 1.0",
    "arxiv.org/licenses/nonexclusive-distrib/1.0": "arXiv non-exclusive distribution 1.0",
    "arxiv.org/licenses/assumed-1991-2003": "arXiv assumed 1991-2003",
}


def arxiv_license(aid: str) -> str:
    """License của bài arXiv qua OAI-PMH (metadataPrefix=arXiv)."""
    r = polite_get("https://export.arxiv.org/oai2",
                   params={"verb": "GetRecord", "identifier": f"oai:arXiv.org:{aid}", "metadataPrefix": "arXiv"})
    r.raise_for_status()
    m = re.search(r"<license>\s*(.*?)\s*</license>", r.text)
    if not m:
        return "arXiv (no explicit license; arXiv default distribution)"
    url = re.sub(r"^https?://|/$", "", m.group(1))
    return _LICENSES.get(url, m.group(1))


def cmd_arxiv_search(a) -> int:
    r = polite_get("https://export.arxiv.org/api/query",
                   params={"search_query": a.query, "start": a.start, "max_results": a.max,
                           "sortBy": a.sort, "sortOrder": "descending"})
    r.raise_for_status()
    gids, _ = gold_keys()
    feed = ET.fromstring(r.content)
    out = open(a.out, "a", encoding="utf-8") if a.out else None
    n = 0
    for e in feed.findall("a:entry", _ATOM):
        m = _ARXIV_ID.search(e.findtext("a:id", "", _ATOM).rsplit("/abs/", 1)[-1])
        if not m:
            continue
        rec = {
            "arxiv_id": m.group(1), "version": m.group(2) or "",
            "title": re.sub(r"\s+", " ", e.findtext("a:title", "", _ATOM)).strip(),
            "primary": (e.find("x:primary_category", _ATOM).get("term") if e.find("x:primary_category", _ATOM) is not None else ""),
            "published": e.findtext("a:published", "", _ATOM)[:10],
            "authors": [x.findtext("a:name", "", _ATOM) for x in e.findall("a:author", _ATOM)][:6],
            "doi": e.findtext("x:doi", "", _ATOM), "journal_ref": e.findtext("x:journal_ref", "", _ATOM),
            "gold": m.group(1) in gids, "topic": a.topic or "", "query": a.query,
        }
        n += 1
        print(f"{rec['arxiv_id']}{rec['version']:<4} {rec['primary']:<16} {rec['published'][:4]} "
              f"{'GOLD ' if rec['gold'] else ''}{rec['title'][:110]}")
        if out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if out:
        out.close()
    total = feed.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults", "?")
    print(f"--- {n} kết quả (tổng {total})")
    return 0


def _read_items(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_batch(a) -> int:
    dest, manifest = Path(a.dest), Path(a.manifest)
    items = _read_items(Path(a.list))
    ok, skipped, failed = 0, [], []
    for it in items:
        doc_id = it["doc_id"]
        have = {r["doc_id"]: r for r in read_manifest(manifest)}
        if doc_id in have and (dest / f"{doc_id}.pdf").exists() and not a.force:
            skipped.append((doc_id, "đã có"))
            continue
        try:
            lic = it.get("license") or ""
            if lic == "auto-arxiv":
                aid = arxiv_base_id(it["url"]) or arxiv_base_id(doc_id)
                lic = arxiv_license(aid) if aid else "unknown"
            row = fetch_pdf(it["url"], doc_id, it["site"], lic, it.get("note", ""), it.get("sample_ids", ""),
                            dest, manifest, a.force, guard_gold=True)
            dups = [r["doc_id"] for r in read_manifest(manifest) if r["sha256"] == row["sha256"] and r["doc_id"] != doc_id]
            if dups:  # file giống hệt (byte) tài liệu đã có -> bỏ
                (dest / f"{doc_id}.pdf").unlink()
                write_manifest([r for r in read_manifest(manifest) if r["doc_id"] != doc_id], manifest)
                skipped.append((doc_id, f"trùng sha256 với {dups[0]}"))
                continue
            ok += 1
        except (ValueError, requests.RequestException, RuntimeError) as e:
            failed.append((doc_id, str(e)[:200]))
            print(f"LỖI {doc_id}: {str(e)[:200]}")
    print(f"--- tải/ghi {ok}, bỏ qua {len(skipped)}, lỗi {len(failed)}")
    for d, why in skipped + failed:
        print(f"  {d}: {why}")
    return 0


def cmd_stats(a) -> int:
    rows = read_manifest(Path(a.manifest))
    n_open = n_notext = pages = 0
    size = 0
    for r in rows:
        p = ROOT / r["file"]
        size += p.stat().st_size if p.exists() else 0
        try:
            with pymupdf.open(p) as doc:
                n_open += 1
                pages += doc.page_count
                k = min(doc.page_count, 3)
                if sum(len(doc[i].get_text().strip()) for i in range(k)) < 100 * k:
                    n_notext += 1
                    print(f"  ít/không text: {r['doc_id']}")
        except Exception as e:  # noqa: BLE001 — PDF hỏng: đếm là không mở được
            print(f"  KHÔNG mở được: {r['doc_id']}: {e}")
    n = len(rows)
    print(f"PDF: {n} | mở được {n_open}/{n} ({100*n_open/max(n,1):.1f}%) | ít/không text {n_notext} "
          f"({100*n_notext/max(n_open,1):.1f}%) | tổng trang {pages} | TB {pages/max(n_open,1):.1f} trang/PDF "
          f"| {size/1e6:.1f} MB")
    print("theo chủ đề:", dict(Counter(r["note"].split("|")[0].strip() for r in rows).most_common()))
    print("theo topic:", dict(Counter(" | ".join(x.strip() for x in r["note"].split("|")[:2]) for r in rows).most_common()))
    print("theo site:", dict(Counter(r["site"] for r in rows).most_common()))
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
    p.add_argument("--samples", default="", help="vd 3,7 (rỗng cho tài liệu nhiễu)")
    p.add_argument("--note")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dest", help=f"thư mục PDF (mặc định {GOLD_DIR.relative_to(ROOT).as_posix()})")
    p.add_argument("--manifest", help=f"manifest CSV (mặc định {MANIFEST.relative_to(ROOT).as_posix()})")
    p = sub.add_parser("check")
    p.add_argument("--evidence", default=str(ROOT / "data" / "work" / "sample25_evidence.jsonl"))
    p = sub.add_parser("find")
    p.add_argument("--doc-id", required=True)
    p.add_argument("--pattern", required=True)
    p.add_argument("--context", type=int, default=200)
    p = sub.add_parser("arxiv-search")
    p.add_argument("--query", required=True, help="cú pháp arXiv API, vd 'all:\"mamba\" AND cat:cs.LG'")
    p.add_argument("--max", type=int, default=20)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--sort", default="relevance", choices=["relevance", "submittedDate", "lastUpdatedDate"])
    p.add_argument("--topic", help="nhãn chủ đề ghi vào --out")
    p.add_argument("--out", help="nối kết quả vào file JSONL")
    p = sub.add_parser("batch")
    p.add_argument("--list", required=True, help="JSONL/CSV: url, doc_id, site, license, note")
    p.add_argument("--dest", default=str(DIST_DIR))
    p.add_argument("--manifest", default=str(DIST_MANIFEST))
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("stats")
    p.add_argument("--manifest", default=str(DIST_MANIFEST))
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    return {"add": cmd_add, "check": cmd_check, "find": cmd_find, "arxiv-search": cmd_arxiv_search,
            "batch": cmd_batch, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
