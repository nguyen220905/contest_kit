"""Hiệu chỉnh judge J1 trên 25 câu mẫu.

    calibrate_judge.py --make-refs        # nháp reference_short (openai) -> data/work/sample25_refs_draft.jsonl
    calibrate_judge.py --make-variants    # biến thể gold (openai) -> data/work/calib_variants.jsonl
    calibrate_judge.py [--refs file]      # chạy 4 test với J1 -> results/calibration/<ts>/
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import judge as J  # noqa: E402

WORK = ROOT / "data" / "work"
DRAFT = WORK / "sample25_refs_draft.jsonl"
VARIANTS = WORK / "calib_variants.jsonl"
GEN = "openai"  # khác provider judge để tránh tự thiên vị
PV_REFS, PV_VAR = "calib-refs-v1", "calib-variants-v2"


def sample():
    return json.loads((ROOT / "starter_kit" / "sample" / "sample_qa.json").read_text(encoding="utf-8"))


def jl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def ask_json(prompt, pv, need):
    msgs = [{"role": "system", "content": "You are a careful assistant. Reply with a single JSON object only."},
            {"role": "user", "content": prompt}]
    for i in range(2):
        out = J.chat(GEN, msgs if i == 0 else msgs + [{"role": "user", "content": "Reply ONLY with valid JSON."}], pv)
        obj = J.last_json(out, need)
        if obj is not None:
            return obj
    raise RuntimeError("không parse được JSON từ model sinh")


def make_refs():
    def one(r):
        p = f"""Question: {r['question']}

Gold answer (long): {r['answer']}

Condense the gold answer into an ATOMIC reference claim: the key value(s) + unit(s) + main entity, nothing else.
If the question has several parts, list every part. If the question has a false premise, write "premise is false: ..." and what is true.
Return JSON: {{"reference_short": str, "flags": {{"multi_part": bool, "false_premise": bool, "calc": bool (needs a calculation), "outside_doc": bool (answer not derivable from a document, general knowledge), "names_paper": bool (question names a specific paper/document)}}, "parts": [str] (the parts if multi_part else [])}}"""
        o = ask_json(p, PV_REFS, "reference_short")
        return {"id": r["id"], "reference_short": o["reference_short"], "flags": o.get("flags", {}),
                "parts": o.get("parts", [])}
    with ThreadPoolExecutor(6) as ex:
        out = list(ex.map(one, sample()))
    WORK.mkdir(parents=True, exist_ok=True)
    DRAFT.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    print(f"ghi {len(out)} dòng -> {DRAFT}; multi_part={sum(bool(o['flags'].get('multi_part')) for o in out)}")


def make_variants(refs_path):
    flags = {str(r["id"]): r for r in jl(refs_path)}
    have = {(str(v["id"]), v["kind"]) for v in jl(VARIANTS)}
    new = []

    def one(r):
        res, i = [], str(r["id"])
        if (i, "para") not in have:
            o = ask_json(f"Question: {r['question']}\n\nGold answer: {r['answer']}\n\nRewrite the gold answer in different words, "
                         f"keeping EXACTLY the same facts, numbers, units and entities. Return JSON {{\"text\": str}}", PV_VAR, "text")
            res.append({"id": r["id"], "kind": "para", "text": o["text"]})
        fl_ref = flags.get(i, {}).get("reference_short", "")
        if (i, "wrong1") not in have or (i, "wrong2") not in have:
            o = ask_json(f"Question: {r['question']}\n\nGold answer: {r['answer']}\n\nProduce TWO different corrupted versions of the gold answer, "
                         f"each fluent and confident but WRONG on the key point: change a key number, a unit, or the main entity "
                         f"(a different change in each). Keep everything else the same. Return JSON {{\"variants\": [str, str]}}", PV_VAR, "variants")
            for k, t in zip(("wrong1", "wrong2"), o["variants"][:2]):
                res.append({"id": r["id"], "kind": k, "text": t})
        fl = flags.get(i, {})
        if fl.get("flags", {}).get("multi_part") and (i, "drop") not in have:
            o = ask_json(f"Question: {r['question']}\n\nGold answer: {r['answer']}\n\nParts of the answer: {json.dumps(fl.get('parts', []), ensure_ascii=False)}\n\n"
                         f"Rewrite the gold answer omitting EXACTLY ONE part (completely drop it, do not hint at it), keeping the other parts intact. "
                         f"Return JSON {{\"text\": str}}", PV_VAR, "text")
            res.append({"id": r["id"], "kind": "drop", "text": o["text"]})
        return res

    with ThreadPoolExecutor(6) as ex:
        for res in ex.map(one, sample()):
            new += res
    with open(VARIANTS, "a", encoding="utf-8") as f:
        for v in new:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    allv = jl(VARIANTS)
    cnt = {k: sum(v["kind"] == k for v in allv) for k in ("para", "wrong1", "wrong2", "drop")}
    print(f"thêm {len(new)} biến thể; tổng {cnt}")


def calibrate(refs_path, provider):
    qa = sample()
    by_id = {str(r["id"]): r for r in qa}
    refs = J.load_refs(refs_path)
    variants = jl(VARIANTS)
    if not variants:
        raise SystemExit("Chưa có biến thể: chạy --make-variants trước")
    items = [dict(id=str(v["id"]), kind=v["kind"], text=v["text"]) for v in variants]
    for k, r in enumerate(qa):  # lạc đề: gold của câu kế tiếp (xoay vòng)
        items.append(dict(id=str(r["id"]), kind="offtopic", text=qa[(k + 1) % len(qa)]["answer"]))

    def run(it, mode):
        q = by_id[it["id"]]
        row = {"id": it["id"], "reference_short": "", "reference_long": q["answer"]}
        ref, used = J.pick_ref(row, mode, refs)
        j = J.judge(q["question"], ref, it["text"], provider)
        return int(j["verdict"]), j["reason"], used

    with ThreadPoolExecutor(8) as ex:
        main = list(ex.map(lambda it: run(it, "auto"), items))
        long_ = list(ex.map(lambda it: run(it, "long"), items)) if refs else None
    for i, it in enumerate(items):
        it["verdict"], it["reason"], it["ref"] = main[i]
        if long_:
            it["verdict_long"] = long_[i][0]

    # kiểm tính hợp lệ của biến thể "sai" (model sinh = openai, độc lập với judge): biến thể có thực sự làm
    # sai/đổi nội dung của tham chiếu không? Nhiều lần model chỉ đổi chi tiết ngoài tham chiếu -> judge bác là sai.
    def valid(it):
        q = by_id[it["id"]]
        ref, _ = J.pick_ref({"id": it["id"], "reference_short": "", "reference_long": q["answer"]}, "auto", refs)
        o = ask_json(f"Reference answer: {ref}\n\nCandidate answer: {it['text']}\n\nDoes the candidate state something that "
                     f"CONTRADICTS or differs from a specific fact, number, unit, symbol, or entity stated in the reference answer? "
                     f"Differences in details the reference does not mention do NOT count. Return JSON {{\"contradicts\": bool, \"why\": str}}",
                     "calib-valid-v1", "contradicts")
        return bool(o["contradicts"])
    wr = [it for it in items if it["kind"] in ("wrong1", "wrong2")]
    with ThreadPoolExecutor(8) as ex:
        for it, v in zip(wr, ex.map(valid, wr)):
            it["valid_wrong"] = int(v)

    def sel(*kinds):
        return [it for it in items if it["kind"] in kinds]
    para, wrong_all, off, drop = sel("para"), sel("wrong1", "wrong2"), sel("offtopic"), sel("drop")
    wrong = [i for i in wrong_all if i.get("valid_wrong")]
    raw_rej = sum(1 - i["verdict"] for i in wrong_all)
    t = {
        "1_paraphrase_correct": dict(value=sum(i["verdict"] for i in para), n=len(para), need="correct >= 24/25",
                                     ok=sum(i["verdict"] for i in para) >= 24 and len(para) == 25),
        "2_wrong_rejected": dict(raw_all=f"{raw_rej}/{len(wrong_all)}", value=sum(1 - i["verdict"] for i in wrong), n=len(wrong), need="rejected >= 90% (on variants confirmed to contradict the reference)",
                                 ok=bool(wrong) and sum(1 - i["verdict"] for i in wrong) / len(wrong) >= 0.9),
        "3_offtopic_rejected": dict(value=sum(1 - i["verdict"] for i in off), n=len(off), need="rejected 25/25",
                                    ok=len(off) == 25 and all(i["verdict"] == 0 for i in off)),
        "4_missing_part_rejected": dict(value=sum(1 - i["verdict"] for i in drop), n=len(drop), need="rejected >= 80%",
                                        ok=bool(drop) and sum(1 - i["verdict"] for i in drop) / len(drop) >= 0.8),
    }
    for v in t.values():
        v["result"] = "PASS" if v["ok"] else "FAIL"
    flip = None
    if long_:
        ab = sel("para") + wrong
        flip = {"n": len(ab), "flips": sum(i["verdict"] != i["verdict_long"] for i in ab)}
        flip["rate"] = round(flip["flips"] / len(ab), 4)
    rep = {"prompt_version": J.PROMPT_VERSION, "provider": provider, "refs": str(refs_path) if refs else None,
           "tests": t, "flip_long_vs_short_on_1_2": flip, "all_pass": all(v["ok"] for v in t.values()),
           "api_calls_new": J._stats["calls"], "cache_hits": J._stats["hits"]}
    out = ROOT / "results" / "calibration" / datetime.now().strftime("%Y%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    cols = ["id", "kind", "ref", "verdict", "verdict_long", "valid_wrong", "reason", "text"]
    with open(out / "variants_judged.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)
    for k, v in t.items():
        print(f"{v['result']}  {k}:{' raw_all=' + v['raw_all'] if 'raw_all' in v else ''} {v['value']}/{v['n']}  ({v['need']})")
    print(f"flip long vs short: {flip}\nall_pass={rep['all_pass']}  -> {out}")
    return rep


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-refs", action="store_true")
    ap.add_argument("--make-variants", action="store_true")
    ap.add_argument("--refs", default=None)
    ap.add_argument("--provider", default="deepseek")
    a = ap.parse_args()
    if a.make_refs:
        make_refs()
    elif a.make_variants:
        refs = a.refs or (ROOT / "data" / "work" / "sample25_refs.jsonl"
                          if (WORK / "sample25_refs.jsonl").exists() else DRAFT)
        make_variants(refs)
    else:
        calibrate(a.refs, a.provider)
    print(f"API mới {J._stats['calls']}, cache {J._stats['hits']}")


if __name__ == "__main__":
    main()
