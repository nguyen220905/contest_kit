# REFERENCES — round_06 (chỉ nguồn MỚI so với round_01/02)
> Nhãn: **đã mở** = researcher mở bằng WebFetch ngày 22/10/2026; **qua Perplexity** = có trong báo cáo `docs/research/20261022-*.md`, researcher chưa tự mở; **chưa mở** = do model hội đồng trích, chưa kiểm.

| # | Trích dẫn (kiểu IEEE) | URL | Nhãn | Dùng cho phần nào của paper |
|---|---|---|---|---|
| [1] | Allen Institute for AI, "Olmo-3.1-32B-Instruct — config.json," Hugging Face, 2025. | https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/raw/main/config.json | đã mở — `sliding_window 4096`, 48/64 lớp sliding + 16 full, GQA 40/8, YaRN ×8 (gốc 8192), `use_cache: false` | Phương pháp: lý do xếp evidence-last / giới hạn context |
| [2] | Allen Institute for AI, "Olmo-3.1-32B-Instruct — generation_config.json," Hugging Face, 2025. | https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/raw/main/generation_config.json | đã mở — `do_sample true, temperature 0.6, top_p 0.95, max_new_tokens 32768` | Thiết lập thí nghiệm: ép greedy để tái lập |
| [3] | Allen Institute for AI, "Olmo-3.1-32B-Instruct — chat_template.jinja," Hugging Face, 2025. | https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/raw/main/chat_template.jinja | đã mở — khi thiếu system message thì tự chèn "You are Olmo… date cutoff December 2024…" | Thiết lập: prompt luôn có system |
| [4] | IEEE CIS GPAIS Task Force, "Evaluation — Technical Challenge on Domain-Specific QA with LLMs," 2026. | https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/evaluation/ | đã mở — judge LLM ẩn, verdict nhị phân, domain-balanced accuracy, public 50%, cuối kỳ tính lại 100% | Bài toán & metric |
| [5] | NVIDIA, "RTX PRO 6000 Blackwell Server Edition," datasheet web, 2025. | https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/ | đã mở — 96 GB GDDR7, 1597 GB/s, BF16 "1 PFLOP" (không ghi dense/sparse), 600 W | Phụ lục phần cứng: quy đổi thời gian |
| [6] | NVIDIA, "H100 Tensor Core GPU," datasheet web, 2023. | https://www.nvidia.com/en-us/data-center/h100/ | đã mở — SXM 80 GB, 3.35 TB/s, BF16 1979 TFLOPS **có sparsity** (dense 989) | Phụ lục phần cứng |
| [7] | Ollama, "FAQ — How can I specify the context window size?," docs, 2026. | https://docs.ollama.com/faq | đã mở — "By default, Ollama uses a context window size of 4096 tokens"; `OLLAMA_CONTEXT_LENGTH` / `num_ctx` | Công cụ dev (không dùng trong bài nộp) |
| [8] | NVIDIA, "A100 Tensor Core GPU," datasheet web, 2020. | https://www.nvidia.com/en-us/data-center/a100/ | qua Perplexity — 40 GB SXM 1.555 TB/s, 80 GB SXM 2.039 TB/s, BF16 dense 312 | Phụ lục phần cứng |
| [9] | Hugging Face, "Big Model Inference (Accelerate) — device_map='auto'," docs. | https://huggingface.co/docs/accelerate/usage_guides/big_modeling | qua Perplexity/brief — model parallel tuần tự, batch 1 thì mỗi lúc chỉ 1 GPU chạy | Lý do không chọn 2× A100 |
| [10] | G. Gerganov et al., "llama.cpp — quantize README (Q8_0 / Q4_K_M PPL)," GitHub. | https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md | qua Perplexity — Q8_0 ≈ +0.003 PPL, Q4_K_M ≈ +0.18 (Llama-3-8B) | Không dùng Ollama/GGUF cho quyết định |
| [11] | Docling Project, "GPU support / performance," docs + issue #1364. | https://docling-project.github.io/docling/usage/gpu/ | qua Perplexity/brief — RTX 5090 / L40S có số; không có số cho A100/H100; GPU dùng thấp | Thời gian parse |
| [12] | M. Park, "Is the exact test better than the asymptotic test for testing marginal homogeneity in 2×2 tables?," *Biometrical J.*, 44(5), 2002. | https://doi.org/10.1002/1521-4036(200207)44:5<571::AID-BIMJ571>3.0.CO;2-P | chưa mở (M2 trích) — McNemar exact cho n nhỏ | Phương pháp thống kê: so cặp trên 25 câu |
| [13] | Y. Dubois et al., "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators," arXiv:2404.04475, 2024. | https://arxiv.org/abs/2404.04475 | chưa mở (M1 trích; paper có thật) — thiên lệch độ dài của LLM judge | Thảo luận: probe độ dài trên LB |
| [14] | Oumi, "Olmo3 32B inference recipe," GitHub. | https://github.com/oumi-ai/oumi/blob/main/configs/recipes/olmo3/inference/32b_infer.yaml | chưa mở (M7 trích) — tham số sampling khuyến nghị | Không cần cho paper |

Ghi chú:
- arXiv 2512.13961 (Olmo 3 technical report) đã có ở round_01. Đã mở lại abstract: đúng là Team Olmo, "Olmo 3". Câu M7 viện dẫn từ paper này ("chỉ lớp cuối full attention") **sai** so với config [1] (cứ 4 lớp có 1 lớp full).
- Không phát hiện link bịa trong các nguồn mà top đề xuất dựa vào. Các trích dẫn "hệ số A100→H100 0.90/0.42/0.62" (M6) không có nguồn.
