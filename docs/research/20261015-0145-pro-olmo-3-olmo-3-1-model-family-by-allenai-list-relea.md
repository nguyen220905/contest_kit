# Olmo 3 / Olmo 3.1 model family by AllenAI: list released sizes and variants (7B, 32B Instruct, Think), context length, tokenizer and chat template: do Olmo-3-7B-Instruct and Olmo-3.1-32B-Instruct share the same tokenizer and chat template? bf16 VRAM, transformers support version, known generation speed on H100 with HF transformers, recommended sampling settings.

> Perplexity pro · 2026-10-15 01:45. Kiểm chứng link quan trọng trước khi dựa vào.

## Tóm tắt phát hiện

- Dòng **OLMo 3** gồm các model dense decoder-only ở hai quy mô **7B** và **32B**, với các biến thể chính **Base**, **Instruct** và **Think**. Các checkpoint liên quan gồm `Olmo-3-7B-Instruct`, `Olmo-3-7B-Think`, `Olmo-3-32B-Think`, `Olmo-3.1-32B-Instruct` và `Olmo-3.1-32B-Think`.[1][2]
- **Context length là 65.536 token (64K)** ở các checkpoint đã kiểm tra; cấu hình của 7B Instruct và 32B Instruct đều đặt `max_position_embeddings`/`model_max_length` bằng `65536`.[3][4]
- Hai model **Olmo-3-7B-Instruct** và **Olmo-3.1-32B-Instruct** dùng cùng họ tokenizer OLMo 3 Instruct và cùng định dạng chat release về mặt thiết kế, nhưng không nên suy luận rằng hai thư mục checkpoint hoàn toàn byte-identical. Cần tải tokenizer từ chính checkpoint tương ứng; Ai2 ghi rõ các bản release có thể khác template ở system prompt và giai đoạn huấn luyện.[5]
- Transformers chính thức hỗ trợ OLMo 3 từ **`transformers>=4.57.0`**.[6]
- Model card của Ai2 khuyến nghị **`temperature=0.6`, `top_p=0.95`, `max_tokens/max_new_tokens=32768`** cho generation.[7][8]
- Chưa tìm thấy benchmark sơ cấp đáng tin cậy công bố **tokens/s trên một H100 khi chạy riêng bằng `transformers.generate()`** cho hai model này. Con số Ai2 công bố **7,7K token/device/s trên H100** là throughput *huấn luyện* của OLMo 3 Base 7B, không phải tốc độ suy luận.[9] Vì vậy, đội nên benchmark trực tiếp trên workload thực tế thay vì dùng con số này để dự đoán inference speed.
- Ước tính trọng số BF16, chưa tính KV cache và overhead:
  - **7B:** khoảng \(7\times10^9\times2\) byte ≈ **14 GB**; thực tế nên dành khoảng **16–20 GB VRAM** cho model + runtime ở context vừa phải.
  - **32B:** khoảng \(32\times10^9\times2\) byte ≈ **64 GB**; model card/Hub hiển thị khoảng **64,5 GB**, nên H100 80GB có thể chứa trọng số nhưng biên an toàn cho context dài hoặc batch lớn khá nhỏ.[10]
  - Đây là ước tính kỹ thuật từ số tham số và kiểu BF16, không phải mức VRAM tối thiểu được Ai2 bảo đảm.

## Kỹ thuật / ý tưởng đáng thử (mỗi ý: là gì · bằng chứng · điều kiện áp dụng · nguồn)

- **Dùng model card làm cấu hình mặc định cho generation** · là gì: chạy sampling với `do_sample=True`, `temperature=0.6`, `top_p=0.95`; đặt `max_new_tokens` theo ngân sách, tối đa khuyến nghị 32.768 thay vì luôn dùng toàn bộ context 65.536 · bằng chứng: model cards của 7B Instruct và 32B Instruct đều nêu cùng bộ tham số này; ví dụ inference của Ai2 cũng dùng `top_k=50`, `top_p=0.95` · điều kiện áp dụng: phù hợp chat, trả lời mở và bài toán cần đa dạng; với output cần tái lập hoặc format nghiêm ngặt, thử greedy/temperature thấp hơn và kiểm tra chất lượng · nguồn: [7][8]

- **Dùng `apply_chat_template()` thay vì tự ghép chuỗi prompt** · là gì: nạp tokenizer cùng checkpoint rồi gọi `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")` · bằng chứng: ví dụ chính thức của Hugging Face/Ai2 dùng đúng API này cho OLMo 3. Template chứa các token hội thoại đặc thù và generation prompt; tự ghép theo template của Llama/Qwen có thể làm lệch phân phối prompt · điều kiện áp dụng: bắt buộc khi dùng Instruct/Think, đặc biệt với tool calling hoặc multi-turn · nguồn: [11][12]

- **So sánh hai model bằng tokenizer đi kèm từng repo, nhưng kiểm tra template trước khi kết luận “giống nhau”** · là gì: tải riêng `AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct")` và `AutoTokenizer.from_pretrained("allenai/Olmo-3.1-32B-Instruct")`, rồi so sánh `vocab_size`, special-token IDs, `chat_template`, token hóa cùng một bộ prompt và chuỗi do `apply_chat_template()` tạo ra · bằng chứng: Ai2 mô tả tokenizer Instruct 7B/32B cùng dựa trên `olmo-3-tokenizer-instruct-dev`, trong khi bản release dùng `olmo-3-tokenizer-instruct-release`, có khác biệt liên quan system prompt; repo 32B hiện khai báo `GPT2Tokenizer` và `model_max_length=65536` · điều kiện áp dụng: cần thiết khi chuyển dataset, cache token hoặc prompt giữa 7B và 32B; không dùng tokenizer của model này với model kia nếu chưa xác minh ID và template · nguồn: [5][13]

- **Benchmark inference theo prefill và decode riêng biệt trên H100** · là gì: đo thời gian tải model, time-to-first-token, decode tokens/s và peak VRAM cho các độ dài input 512/4K/16K/64K, với batch 1 và batch lớn; cố định `torch_dtype=torch.bfloat16`, `use_cache=True`, cùng `max_new_tokens` · bằng chứng: không có số liệu công khai đáng tin cậy cho đúng tổ hợp “H100 + HF Transformers + OLMo 3”; tốc độ phụ thuộc mạnh vào context, batch, attention kernel, phiên bản PyTorch/Transformers và CPU-tokenization · điều kiện áp dụng: nên làm trước khi chọn 7B hay 32B cho cuộc thi; báo cáo cả median/p95 và peak VRAM, không chỉ tokens/s · nguồn: [6][9]

- **Chọn 7B Instruct cho latency và throughput; chọn 32B Instruct khi chất lượng quan trọng hơn** · là gì: dùng 7B làm model chính hoặc baseline nhanh, 32B làm reranker/judge/second pass cho các mẫu khó · bằng chứng: Ai2 định vị 7B Instruct cho chat và phản hồi nhanh, còn 32B Instruct là model chat/tool-use lớn hơn và mạnh nhất của họ trong nhóm open chat tại thời điểm phát hành · điều kiện áp dụng: phù hợp pipeline hai tầng nếu ngân sách H100 cho phép; cần đo chất lượng trên dữ liệu cuộc thi vì benchmark công khai không thay thế validation nội bộ · nguồn: [1]

- **Dùng quantization cho 7B nếu VRAM hoặc concurrency là nút thắt** · là gì: thử 8-bit/4-bit hoặc backend tối ưu thay vì BF16 toàn phần · bằng chứng: model card 7B chính thức nêu phương án `load_in_8bit=True` với bitsandbytes; trọng số BF16 7B đã chiếm khoảng 14 GB trước KV cache · điều kiện áp dụng: cần kiểm tra suy giảm điểm số, độ ổn định format và hỗ trợ kernel; không giả định 4-bit luôn nhanh hơn BF16 trên H100 · nguồn: [14]

- **Với Think, giới hạn ngân sách reasoning riêng khỏi context tối đa** · là gì: đặt `max_new_tokens` thấp hơn 32.768 cho vòng thử nghiệm, sau đó tăng cho bài toán khó; tách thời gian reasoning khỏi câu trả lời cuối khi chấm latency · bằng chứng: Ai2 mô tả Think là biến thể hiển thị các bước suy luận trung gian và nhấn mạnh lợi ích cho math/code; model card vẫn dùng khuyến nghị tối đa 32.768 token · điều kiện áp dụng: thích hợp cho bài toán reasoning; không dùng Think cho mọi request chat nếu latency là tiêu chí chính · nguồn: [1][15]

## Bẫy và cảnh báo đã biết

- **Không nhầm 64K context với 32K output.** `model_max_length/max_position_embeddings=65536` là giới hạn tổng ngữ cảnh; model card khuyến nghị `max_tokens=32768` cho generation. Input dài sẽ làm giảm số token output còn lại.[3][4][7]
- **Không dùng tốc độ huấn luyện 7,7K token/s để ước tính tốc độ suy luận.** Đây là throughput huấn luyện OLMo 3 Base 7B trên thiết lập nhiều H100, không phải decode speed của `transformers.generate()` trên một GPU.[9]
- **32B BF16 trên H100 80GB có thể “load được” nhưng chưa chắc chạy thoải mái.** Khoảng 64–64,5 GB chỉ là trọng số; KV cache, temporary tensors, allocator fragmentation và context dài có thể khiến OOM.[10]
- **Chat template có biến thể theo giai đoạn/model.** Tài liệu Ai2 ghi nhận khác biệt giữa `dev` và `release`, giữa SFT/DPO/RL, cũng như system prompt nhận dạng “Olmo”. Vì vậy, không nên copy template từ một model sang model khác chỉ vì cùng họ OLMo.[5]
- **`chat_template.jinja` có thể được ưu tiên hơn trường `chat_template` trong `tokenizer_config.json`.** Nếu hai nơi không đồng bộ, Transformers ưu tiên file `.jinja`; nên pin revision và kiểm tra file thực tế trong checkpoint.[5]
- **EOS/stop behavior cần kiểm thử.** Các model OLMo 3 dùng nhiều token đặc biệt cho message boundaries; nếu server không nhận đúng `eos_token_id` hoặc stop token, model có thể tiếp tục sinh, in token điều khiển hoặc vượt ngân sách.[11]
- **Không đánh đồng Instruct và Think.** Think được huấn luyện để dành token cho reasoning; dùng prompt/chat template hoặc stop rule của Instruct một cách máy móc có thể cắt mất phần suy luận hoặc tạo output không đúng format.[15]
- **Ví dụ `top_k=50` không nhất thiết là toàn bộ khuyến nghị.** Model card nêu rõ `temperature=0.6`, `top_p=0.95`, `max_tokens=32768`; `top_k=50` xuất hiện trong ví dụ generation. Đội nên A/B test có và không có `top_k` trên validation set.[7][8]
- **Không trộn tokenizer cache giữa các revision.** Hugging Face repo có thể được cập nhật tokenizer/template sau khi checkpoint phát hành; nên pin revision cụ thể cho model, tokenizer và code inference để kết quả tái lập.[13]
- **Tốc độ thực tế phụ thuộc backend.** Kết quả với HF eager/SDPA, FlashAttention, `torch.compile`, vLLM hoặc TensorRT-LLM không thể so sánh trực tiếp; nếu cuộc thi chấm qua Transformers thuần, benchmark đúng backend đó.

## Nguồn

1. Olmo 3: Charting a path through the model flow to lead open-source AI — https://allenai.org/blog/olmo3  
2. allenai/Olmo-3.1-32B-Instruct — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct  
3. config.json của Olmo-3-7B-Instruct — https://huggingface.co/allenai/Olmo-3-7B-Instruct/commit/d42dc9de51ee481e02b8a35d8222b86c6d7c5b20  
4. tokenizer_config.json của Olmo-3.1-32B-Instruct — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/blame/main/tokenizer_config.json  
5. Olmo 3 & Olmo Hybrid — Open Instruct — https://allenai.github.io/open-instruct/olmo3/  
6. allenai/Olmo-3-7B-Instruct — https://huggingface.co/allenai/Olmo-3-7B-Instruct  
7. Model Card for Olmo 3 7B Instruct — https://huggingface.co/allenai/Olmo-3-7B-Instruct  
8. Model Card for Olmo 3.1 32B Instruct — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/blob/d72759eb9ecd8b6b89b79b8b6bec7506f8955152/README.md  
9. Olmo 3: Efficient training stack — https://allenai.org/blog/olmo3?s=09  
10. Instructions to use allenai/Olmo-3.1-32B-Instruct — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/tree/2a97cd4ffbb9146e4a918d050522685b07eb5da2  
11. allenai/Olmo-3-7B-Instruct — https://huggingface.co/allenai/Olmo-3-7B-Instruct  
12. allenai/Olmo-3.1-32B-Instruct — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct  
13. settings và tokenizer configuration của Olmo-3.1-32B-Instruct — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/commit/48bf1d6041accbd9327f3dfe5e7f95dff1033876  
14. allenai/Olmo-3-7B-Instruct — https://huggingface.co/allenai/Olmo-3-7B-Instruct  
15. allenai/Olmo-3-7B-Think — https://huggingface.co/allenai/Olmo-3-7B-Think

## Nguồn trả về từ API
- allenai/Olmo-3.1-32B-Instruct at main — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/tree/main
- README.md · allenai/Olmo-3.1-32B-Instruct at ... — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/blob/61593cf594616a3b3a49e493a8efd0cffbdd74c1/README.md
- README.md · allenai/Olmo-3.1-32B-Instruct at refs/pr/2 — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/blame/refs%2Fpr%2F2/README.md
- README.md · allenai/Olmo-3.1-32B-Instruct-DPO at main — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-DPO/blame/main/README.md
- allenai/Olmo-3.1-32B-Instruct - Hugging Face — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct
- unsloth/Olmo-3.1-32B-Instruct — https://huggingface.co/unsloth/Olmo-3.1-32B-Instruct
- README.md · allenai/Olmo-3.1-32B-Instruct-DPO at ... - Hugging Face — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-DPO/blob/98b9c0783087193b63ebb5e661d16ac376bceb68/README.md
- Instructions to use allenai/Olmo-3.1-32B- ... — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/tree/2a97cd4ffbb9146e4a918d050522685b07eb5da2
- settings (#1) · allenai/Olmo-3.1-32B-Instruct at 48bf1d6 — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/commit/48bf1d6041accbd9327f3dfe5e7f95dff1033876
- initial commit · allenai/Olmo-3.1-32B-Instruct at 4688cda — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/commit/4688cda5259293eb4a4245639e250df4fb18fdfe
- raw — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-SFT/raw/main/README.md
- README.md · allenai/Olmo-3.1-32B-Think at ... — https://huggingface.co/allenai/Olmo-3.1-32B-Think/blame/832c3f543499af8fe68b88359501de9cb7840544/README.md
- Olmo 3: Charting a path through the model flow to lead open-source AI — https://allenai.org/blog/olmo3
- Add OLMo 3 citation · allenai/Olmo-3.1-32B-Instruct-DPO at be41c63 — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-DPO/commit/be41c634e3610a721876ef90c432897d5c5d5d05
- allenai/Olmo-3.1-32B-Instruct · settings — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/discussions/1
- allenai/Olmo-3-7B-Instruct · Hugging Face — https://huggingface.co/allenai/Olmo-3-7B-Instruct
- Tiny Olmo3ForCausalLM — Hardware Requirements & Compatibility — https://llmrun.dev/model/trl-internal-testing-tiny-olmo3forcausallm
- allenai/Olmo-3.1-32B-Instruct-SFT — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-SFT
- unsloth/Olmo-3-7B-Instruct-GGUF — https://huggingface.co/unsloth/Olmo-3-7B-Instruct-GGUF
- README.md · allenai/Olmo-3-7B-Instruct at refs/pr/2 — https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/refs%2Fpr%2F2/README.md
- Olmo 3 Local Deployment: VRAM Requirements per Variant | NexGPU — https://nexgpu.net/en/models/olmo/
- README.md · allenai/Olmo-3.1-32B-Instruct-SFT at ... - Hugging Face — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-SFT/blob/6066108a231f1475986c85c46df02815c665ec62/README.md
- Olmo 3.1 32B Instruct - Intelligence, Performance & Price ... — https://artificialanalysis.ai/models/olmo-3-1-32b-instruct
- Olmo 3: Charting a path through the model flow to lead open-source AI | Ai2 — https://allenai.org/blog/olmo3?s=09
- Olmo 3.1 - a allenai Collection - Hugging Face — https://huggingface.co/collections/allenai/olmo-31
- Olmo 3 32B Think API Endpoint & Discounted Pricing — https://llmapi.ai/models/allenai-olmo-3-32b-think/
- README.md · allenai/Olmo-3.1-32B-Instruct at ... - Hugging Face — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/blob/d72759eb9ecd8b6b89b79b8b6bec7506f8955152/README.md
- Update README.md · allenai/Olmo-3.1-32B-Instruct-SFT at 0a149c6 — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-SFT/commit/0a149c688b99d534af23f3f3b0a255ce754dc383
- huggingface.co · allenai · Olmo-3README.md · allenai/Olmo-3.1-32B-Instruct-SFT at main — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-SFT/blob/main/README.md
- allenai/Olmo-3.1-32B-Think - Hugging Face — https://huggingface.co/allenai/Olmo-3.1-32B-Think
- README.md · allenai/Olmo-3-7B-Think at ... — https://huggingface.co/allenai/Olmo-3-7B-Think/blame/4afa9e9e3bb4512ca2932c87dc9faeb7440dd0cb/README.md
- README.md · allenai/Olmo-3-7B-Instruct-DPO at refs/pr/6 — https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO/blame/refs%2Fpr%2F6/README.md
- Update tokenizer_config.json · allenai/Olmo-3.1-32B- ... — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/commit/d72759eb9ecd8b6b89b79b8b6bec7506f8955152
- Olmo 3 & Olmo Hybrid - Open Instruct - allenai.github.io — https://allenai.github.io/open-instruct/olmo3/
- Tool Calling - vLLM — https://docs.vllm.ai/en/v0.29.0/features/tool_calling/
- docs.vllm.ai · en · v0Tool Calling - vLLM — https://docs.vllm.ai/en/v0.25.1/features/tool_calling/
- allenai/Olmo-3-7B-Instruct at cdf1ec0af76d8f4f551aaf6da4b05493e0b90786 — https://huggingface.co/allenai/Olmo-3-7B-Instruct/tree/cdf1ec0af76d8f4f551aaf6da4b05493e0b90786
- JetBrains-Research/Olmo-3-Think-imend-eos · Hugging Face — https://huggingface.co/JetBrains-Research/Olmo-3-Think-imend-eos
- allenai/Olmo-3-7B-Instruct at 51c1f0b28546f9cf537583ea5b9f5da4fed23ebe — https://huggingface.co/allenai/Olmo-3-7B-Instruct/tree/51c1f0b28546f9cf537583ea5b9f5da4fed23ebe
- Upload checkpoint from step 450 · allenai/Olmo-3-7B- ... — https://huggingface.co/allenai/Olmo-3-7B-Instruct/commit/d42dc9de51ee481e02b8a35d8222b86c6d7c5b20
- allenai/Olmo-3-32B-Think — https://huggingface.co/allenai/Olmo-3-32B-Think
- tokenizer_config.json · allenai/Olmo-3.1-32B-Instruct at main — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/blame/main/tokenizer_config.json
- Upload folder using huggingface_hub · allenai/Olmo-3.1-32B-Instruct at e06fa64 — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/commit/e06fa647c4ba4c0a8374d3f83dada1c285a5f4ba
- config.json · allenai/Olmo-3-7B-Think at ... — https://huggingface.co/allenai/Olmo-3-7B-Think/blob/fd68d4681f006a58de0dc54aeb17e8bdfe55c94d/config.json
- tokenizer.json · allenai/Olmo-3-7B-Instruct at ... — https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/d42dc9de51ee481e02b8a35d8222b86c6d7c5b20/tokenizer.json
- allenai/Olmo-3-7B-Instruct at main - Hugging Face — https://huggingface.co/allenai/Olmo-3-7B-Instruct/tree/main
- Update README.md · allenai/Olmo-3-7B-Instruct at d93ec82 — https://huggingface.co/allenai/Olmo-3-7B-Instruct/commit/d93ec829d9a40710b4ad95b8d0a277203a57ae2a
- Update the number of parameters (#13) · allenai/Olmo-3-7B-Instruct at 6e5971d — https://huggingface.co/allenai/Olmo-3-7B-Instruct/commit/6e5971d9eba42665f5bd5a0fcf047f299ce1dccc
- allenai/Olmo-3-7B-Instruct · Update the number of parameters — https://huggingface.co/allenai/Olmo-3-7B-Instruct/discussions/13
- OLMo 3 - AI Wiki — https://aiwiki.ai/wiki/olmo_3
- allenai/Olmo-3-7B-Instruct-DPO - Hugging Face — https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO
- allenai/Olmo-3-1025-7B - Hugging Face — https://huggingface.co/allenai/Olmo-3-1025-7B
- olmo-3.1:32b-think — https://ollama.com/library/olmo-3.1:32b-think
- 3 Ai2 OLMo – Open Labs — https://tecunningham.github.io/open-labs/chapters/12-olmo.html
- allenai/Olmo-3-1125-32B - Hugging Face — https://huggingface.co/allenai/Olmo-3-1125-32B
- Mungert/Olmo-3-1025-7B-GGUF — https://huggingface.co/Mungert/Olmo-3-1025-7B-GGUF
- How to Deploy Olmo 3.1 32B Instruct on AMD and NVIDIA GPUs ... — https://docs.vultr.com/models/allen-ai-olmo-31/olmo-3.1-32b-instruct
- OLMo — https://huggingface.co/docs/transformers/model_doc/olmo
- olmo-3.1:32b-think-fp16 - Ollama — https://ollama.com/library/olmo-3.1:32b-think-fp16
- Reasoning Shift: How Context Silently Shortens LLM Reasoning — https://arxiv.org/pdf/2604.01161v3
- Olmo-3.1-32B-Think API & Inference Endpoint — https://friendli.ai/models/allenai/Olmo-3.1-32B-Think
- (không tiêu đề) — https://huggingface.co/allenai/Olmo-3-7B-Think
