"""Smoke test môi trường dev: load model Olmo (mặc định 7B, 4-bit NF4) và trả lời vài câu mẫu.

Kiểm: torch thấy GPU, transformers đúng version server, model load + generate được,
câu trả lời là str, tốc độ token/s và VRAM đỉnh.
Đo trên RTX 5060 Ti (2026-10-15, 7B): 4bit 9 tok/s · 4.9 GiB | none (bf16) 11 tok/s · 13.6 GiB | 8bit 2.8 tok/s · 7.6 GiB.

    .venv\\Scripts\\python scripts/smoke_test.py --n 3
"""

import argparse
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "starter_kit" / "sample" / "sample_qa.json"
SERVER_TRANSFORMERS = "5.17.0"
SYSTEM_PROMPT = (
    "You are a scientific expert. Answer the following question as accurately "
    "and concisely as possible based on your knowledge."
)


def load(model_id: str, quant: str):
    kwargs = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--quant", choices=["8bit", "4bit", "none"], default="4bit")
    ap.add_argument("--n", type=int, default=3, help="số câu mẫu")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    print(f"torch {torch.__version__} (cuda {torch.version.cuda}) | transformers {transformers.__version__}")
    if transformers.__version__ != SERVER_TRANSFORMERS:
        print(f"!! transformers khác server ({SERVER_TRANSFORMERS})")
    assert torch.cuda.is_available(), "torch không thấy GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

    t0 = time.time()
    tokenizer, model = load(args.model, args.quant)
    print(f"load {args.model} [{args.quant}]: {time.time() - t0:.1f}s | VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    questions = json.loads(SAMPLE.read_text(encoding="utf-8"))[: args.n]
    for q in questions:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q["question"]}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        dt = time.time() - t0
        new = out[0][inputs["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new, skip_special_tokens=True).strip()
        assert isinstance(answer, str)
        print(f"\n[q{q['id']} {q['domain']} | {len(new)} tok, {dt:.1f}s, {len(new) / dt:.1f} tok/s]")
        print("Q:", q["question"][:200])
        print("A:", answer[:400])
        print("GOLD:", q["answer"][:200])

    print(f"\nVRAM đỉnh: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB — OK")


if __name__ == "__main__":
    main()
