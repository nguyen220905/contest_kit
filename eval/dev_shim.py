"""Shim CHỈ DÙNG LOCAL (server chấm không có): ép bài nộp load Olmo 4-bit/8-bit để vừa 16 GB VRAM.

Vá AutoModelForCausalLM.from_pretrained để tự chèn quantization_config nếu lời gọi chưa có.
"""

_MARK = "_dev_shim_wrapped"


def install(quant):
    if quant not in ("4bit", "8bit"):
        return
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    current = AutoModelForCausalLM.__dict__.get("from_pretrained")
    if current is not None and getattr(getattr(current, "__func__", current), _MARK, False):
        return  # đã vá (idempotent)
    orig = AutoModelForCausalLM.from_pretrained  # bound method gốc

    def wrapper(*args, **kwargs):
        if kwargs.get("quantization_config") is None:
            if quant == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print(f"[dev_shim] chèn quantization_config={quant}")
        return orig(*args, **kwargs)

    setattr(wrapper, _MARK, True)
    AutoModelForCausalLM.from_pretrained = staticmethod(wrapper)
