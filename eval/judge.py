"""Judge nhị phân (J1 = DeepSeek) + client API chung (cache theo hash).

    .venv\\Scripts\\python eval/judge.py --run results/<run> [--ref auto|short|long] [--provider deepseek|openai] [--refs file.jsonl]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "work"
CACHE_PATH = WORK / "llm_cache.jsonl"
PROMPT_VERSION = "judge-v2"

_lock = threading.Lock()
_cache = None
_stats = {"calls": 0, "hits": 0}
csv.field_size_limit(2**31 - 1)


def load_env():
    p = ROOT / "tools" / ".env"
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_PATH.exists():
            for l in CACHE_PATH.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    try:
                        r = json.loads(l)
                        _cache[r["key"]] = r["text"]
                    except Exception:
                        pass
    return _cache


def provider_cfg(name):
    cfg = json.loads((ROOT / "tools" / "council_config.json").read_text(encoding="utf-8"))
    return cfg["providers"][name]


def chat(provider, messages, prompt_version, max_tokens=16000, timeout=600):
    """Gọi chat completions (không search, temperature 0), có cache. Trả về text."""
    load_env()
    p = provider_cfg(provider)
    key = hashlib.sha256(json.dumps([provider, p["model"], prompt_version, messages],
                                    ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    with _lock:
        cache = _load_cache()
        if key in cache:
            _stats["hits"] += 1
            return cache[key]
    body = {"model": p["model"], "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    body.update(p.get("thinking_body") or {})
    if p.get("thinking"):
        body["thinking"] = p["thinking"]
    if p.get("reasoning_effort"):
        body["reasoning_effort"] = p["reasoning_effort"]
    headers = {"Content-Type": "application/json", "User-Agent": "ieee-dsqa-judge/1.0",
               "Authorization": f"Bearer {os.environ.get(p['api_key_env'], '')}"}
    url = p["base_url"].rstrip("/") + "/chat/completions"
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
            text = resp["choices"][0]["message"].get("content") or ""
            if isinstance(text, list):
                text = "".join(x.get("text", "") for x in text if isinstance(x, dict))
            if not text.strip():
                raise RuntimeError("empty content")
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {e.code}: {msg}"
            if e.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(last) from None
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, KeyError, IndexError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"API lỗi sau 3 lần: {last}")
    with _lock:
        _stats["calls"] += 1
        _cache[key] = text
        WORK.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "provider": provider, "model": p["model"],
                                "prompt_version": prompt_version, "text": text}, ensure_ascii=False) + "\n")
    return text


def last_json(text, need=None):
    """Khối JSON (dict) cuối cùng trong text; need = khoá bắt buộc."""
    dec, best, i = json.JSONDecoder(), None, 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return best
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict) and (need is None or need in obj):
                best = obj
            i = end
        except ValueError:
            i += 1


SYSTEM = (
    "You are a strict but fair grader of expert question answering. You compare a candidate answer against a "
    "reference answer. Respond with a JSON object only."
)
RUBRIC = """Grading rubric. Mark the candidate "correct" only if ALL hold:
1. It contains the key fact(s)/number(s) of the reference (reasonable rounding and equivalent units are fine).
2. It addresses every part the reference requires (if the question has several parts, all must be covered).
3. It does not contradict the reference.
Special cases:
- If the question rests on a false premise and the reference rejects it, the candidate is correct only if it also rejects the premise.
- Extra information that does not contradict the reference is NOT penalized.
- An empty, evasive, or "I don't know" answer is incorrect.
- Judge against the reference, not against your own knowledge.
- Be exacting: first list every concrete item in the reference (numbers, unit prefixes such as m/u/k, symbols and subscripts, entity or method names, quantifiers such as "single" vs "two", direction/sign). Then check the candidate states each of them identically. If the candidate gives a DIFFERENT value, unit, symbol, name or quantity for any of them, the answer is incorrect even if the rest is right and even if it sounds plausible.

Reply with JSON only: {"verdict": "correct" | "incorrect", "reason": "<one or two sentences>"}"""


def _msgs(question, reference, answer):
    user = (f"{RUBRIC}\n\n### Question\n{question}\n\n### Reference answer\n{reference}\n\n"
            f"### Candidate answer\n{answer}\n")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def judge(question, reference, answer, provider="deepseek"):
    if not (answer or "").strip():
        return {"verdict": False, "reason": "empty_answer"}
    msgs = _msgs(question, reference, answer)
    for attempt in range(2):
        m = msgs if attempt == 0 else msgs + [
            {"role": "user", "content": 'Your previous reply could not be parsed. Reply ONLY with JSON: '
                                        '{"verdict": "correct"|"incorrect", "reason": "..."}'}]
        out = chat(provider, m, PROMPT_VERSION)
        obj = last_json(out, "verdict")
        v = str((obj or {}).get("verdict", "")).strip().lower()
        if v in ("correct", "incorrect"):
            return {"verdict": v == "correct", "reason": str(obj.get("reason", ""))}
        # JSON hỏng do LaTeX trong reason (vd "\(" là escape không hợp lệ) → đọc verdict bằng regex
        m = re.findall(r'"verdict"\s*:\s*"(correct|incorrect)"', out)
        if m:
            r = re.findall(r'"reason"\s*:\s*"(.*?)"\s*}', out, re.S)
            return {"verdict": m[-1] == "correct", "reason": r[-1] if r else "regex_parse"}
    return {"verdict": False, "reason": "parse_error"}


def load_refs(path):
    refs = {}
    if path and Path(path).exists():
        for l in Path(path).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("reference_short"):
                    refs[str(r["id"])] = r["reference_short"]
    return refs


def pick_ref(row, mode, refs):
    short = (row.get("reference_short") or "").strip() or refs.get(str(row["id"]), "")
    long_ = row.get("reference_long") or ""
    if mode == "short":
        return (short, "short") if short else (long_, "long")
    if mode == "long":
        return long_, "long"
    return (short, "short") if short else (long_, "long")


def judge_run(run, ref="auto", provider="deepseek", refs_path=None, workers=8):
    """Ghi verdict/judge_reason/judge_ref/judge_provider vào predictions.csv của run. Trả về rows."""
    run = Path(run)
    pred = run / "predictions.csv"
    with open(pred, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        rows, cols = list(rd), list(rd.fieldnames)
    refs = load_refs(refs_path)

    def one(r):
        reference, used = pick_ref(r, ref, refs)
        r["judge_ref"], r["judge_provider"] = used, provider
        if r.get("error") or not (r.get("answer") or "").strip():
            r["verdict"], r["judge_reason"] = 0, "no_call:error_or_empty"
            return
        try:
            j = judge(r["question"], reference, r["answer"], provider)
        except Exception as e:
            j = {"verdict": False, "reason": f"api_error: {e}"}
        r["verdict"], r["judge_reason"] = int(j["verdict"]), j["reason"]

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, rows))
    for c in ("verdict", "judge_reason", "judge_ref", "judge_provider"):
        if c not in cols:
            cols.append(c)
    tmp = pred.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, pred)
    return rows


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", choices=["auto", "short", "long"], default="auto")
    ap.add_argument("--provider", choices=["deepseek", "openai"], default="deepseek")
    ap.add_argument("--refs", default=None)
    a = ap.parse_args()
    rows = judge_run(a.run, a.ref, a.provider, a.refs)
    print(f"judge xong {len(rows)} câu, đúng {sum(int(r['verdict']) for r in rows)}; "
          f"API mới {_stats['calls']}, cache {_stats['hits']}")


if __name__ == "__main__":
    main()
