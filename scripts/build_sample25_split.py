"""Ghép split cố định data/splits/sample25.jsonl từ: câu hỏi + gold (starter_kit), reference_short đã duyệt
(data/work/sample25_refs.jsonl) và evidence đã kiểm (data/work/sample25_evidence.jsonl). Chạy 1 lần; split được bảo vệ.

    .venv\\Scripts\\python scripts/build_sample25_split.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
gold = json.loads((ROOT / "starter_kit/sample/sample_qa.json").read_text(encoding="utf-8"))
refs = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/work/sample25_refs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
evid = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/work/sample25_evidence.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

out = []
for g in gold:
    i = g["id"]
    r, e = refs[i], evid[i]
    flags = dict(r.get("flags", {}))
    flags["closed_book"] = None  # điền sau khi đo closed-book (Stage 3b-e)
    out.append({
        "id": i, "domain": g["domain"], "question_type": g.get("question_type", ""), "question": g["question"],
        "reference_short": r["reference_short"], "reference_long": g["answer"], "parts": r.get("parts", []),
        "evidence": e.get("evidence", []), "evidence_status": e.get("status"), "evidence_note": e.get("note", ""),
        "flags": flags, "source": "sample",
    })

dst = ROOT / "data/splits/sample25.jsonl"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
print(f"{len(out)} câu -> {dst}; evidence found/partial/not_found: "
      + "/".join(str(sum(1 for x in out if x['evidence_status'] == s)) for s in ("found", "partial", "not_found")))
