# Ollama/llama.cpp GGUF quantization (Q4_K_M, Q8_0, F16) vs HuggingFace transformers bf16 outputs: how different are greedy outputs and accuracy for the same model (e.g. OLMo 3 7B Instruct)? Chat template differences in Ollama Modelfile vs HF chat_template, sampling defaults (temperature, repeat penalty, num_ctx default 2048/4096 truncation), determinism with temperature 0 and seed, sliding window attention support for OLMo 3 in llama.cpp. Practical guidance for using Ollama as a fast local dev backend or local LLM judge while the production uses transformers. Give sources.

> Perplexity pro · 2026-10-22 12:53. Kiểm chứng link quan trọng trước khi dựa vào.

## Tóm tắt phát hiện

Với **cùng checkpoint, tokenizer, chat template và decoding**, Q8_0 thường rất gần F16/bf16; Q4_K_M thường vẫn đủ tốt cho phát triển local nhưng có thể làm thay đổi token greedy sớm, từ đó khiến toàn bộ phần sinh tiếp theo khác hẳn. Perplexity thấp không bảo đảm chuỗi greedy giống nhau hoặc accuracy trên một task cụ thể giống nhau.[48][57]

Sai khác thực tế thường đến từ bốn nguồn lớn hơn bản thân lượng tử hóa:

- prompt được serialize khác nhau giữa HF và Ollama;
- Ollama mặc định `temperature=0.8`, `num_ctx=2048`, trong khi Transformers mặc định greedy với `do_sample=False` và temperature hiệu dụng thường là 1.0 nhưng không có tác dụng khi không sampling;[1][95][106][108]
- context bị cắt hoặc context shift;
- backend, phiên bản thư viện, GPU/CPU và thứ tự tính toán khác nhau.

**Khuyến nghị thực dụng:** dùng Ollama Q4_K_M làm backend phát triển nhanh và smoke test; dùng Q8_0 hoặc F16/bf16 khi cần đối chiếu nghiêm ngặt, đánh giá cuối cùng hoặc làm judge quan trọng. Production Transformers nên giữ làm chuẩn; Ollama chỉ được xem là tương đương sau khi đã kiểm thử trực tiếp trên bộ prompt và metric của đội.

Olmo 3 7B đặc biệt cần chú ý: model được huấn luyện với context 8192, có sliding-window attention ở 3/4 số layer với cửa sổ 4096 và layer cuối full attention.[122] Vì vậy không nên mặc định rằng mọi engine sẽ xử lý dài-context giống Transformers; cần kiểm tra metadata GGUF, phiên bản llama.cpp/Ollama và log runtime.

## Kỹ thuật / ý tưởng đáng thử (mỗi ý: là gì · bằng chứng · điều kiện áp dụng · nguồn)

- **Thiết lập một “golden parity harness” giữa Transformers và Ollama** · là gì: chạy cùng một tập prompt cố định qua bf16/F16, Q8_0 và Q4_K_M; lưu prompt token IDs, output token IDs, text, log-prob nếu có, latency, context counts và accuracy. Bắt đầu bằng prompt thuần văn bản trước khi kiểm tra chat. · bằng chứng: Transformers hỗ trợ chat serialization qua `apply_chat_template(..., add_generation_prompt=True)`, còn GGUF/llama.cpp có thể mang metadata chat template; chỉ cần serialize khác là token đầu vào đã khác.[17][27][42] · điều kiện áp dụng: phải cố định revision model/tokenizer, tokenizer, system prompt, BOS/EOS, stop strings, `max_new_tokens`, context size, batch size và backend; so sánh cả exact-match output lẫn task accuracy. · nguồn: [17][27][42]

- **So sánh theo từng token thay vì chỉ so sánh câu cuối** · là gì: ở mỗi bước, ghi token argmax và top-k logits của bf16/F16/Q8_0/Q4_K_M; tìm “first divergence token”, margin giữa hạng nhất và hạng nhì, rồi đo tỷ lệ prompt có divergence trước token 1, 5, 20. · bằng chứng: greedy decoding chọn token xác suất cao nhất ở mỗi bước; một sai khác số học nhỏ tại bước có margin thấp có thể đổi token, sau đó làm trạng thái KV và mọi bước tiếp theo khác đi.[106][112] Nghiên cứu gần đây cũng báo cáo greedy decoding không bất biến theo precision, ngay cả giữa BF16 và FP16 trên cùng phần cứng.[53] · điều kiện áp dụng: cần backend cho phép trả logits; nếu Ollama API không trả đủ logits, dùng llama.cpp/llama-server hoặc một harness riêng để kiểm tra, rồi dùng Ollama cho kiểm thử text-level. · nguồn: [53][106][112]

- **Dùng Q8_0 làm baseline local gần F16/bf16** · là gì: tạo GGUF F16 từ checkpoint HF, sau đó quantize sang Q8_0 và Q4_K_M; dùng Q8_0 làm mốc fidelity, Q4_K_M làm mốc tốc độ/bộ nhớ. · bằng chứng: tài liệu llama.cpp mô tả Q8_0 là gần bản gốc, với ví dụ tăng perplexity khoảng `+0.003`, còn Q4_K_M khoảng `+0.18` trên Llama-3-8B; đây là chỉ báo chung, không phải số đo cho Olmo 3.[48] Một benchmark task-level khác báo cáo Q4_K_M giảm 2,5 điểm phần trăm so với F16 trong cấu hình cụ thể, nhưng Q8_0 cũng giảm 2,5 điểm, cho thấy accuracy phụ thuộc task và nhiễu benchmark, không thể suy ra chỉ từ bit-width.[46] · điều kiện áp dụng: phải đo chính model/task của đội; không dùng con số Llama làm cam kết cho Olmo 3. Nếu judge cần ổn định, ưu tiên Q8_0/F16 và rubric có cấu trúc. · nguồn: [46][48][57]

- **Quantize từ checkpoint chính xác và kiểm tra chuyển đổi trước khi benchmark** · là gì: chuyển checkpoint HF sang GGUF F16 bằng công cụ chuyển đổi tương ứng, kiểm tra architecture, tokenizer, special tokens, RoPE/context metadata, rồi mới tạo Q8_0/Q4_K_M. · bằng chứng: GGUF là định dạng tối ưu cho các executor như GGML và checkpoint PyTorch/HF cần được chuyển đổi trước khi chạy; llama.cpp cung cấp luồng `convert_hf_to_gguf.py` rồi `llama-quantize`.[42][45] Olmo 3 GGUF cộng đồng hiện được ghi nhận dùng architecture `olmo2`, nên cần xác minh converter và runtime hỗ trợ đúng biến thể thay vì chỉ dựa vào tên file.[31] · điều kiện áp dụng: pin commit/version của converter và runtime; kiểm tra checksum; không trộn GGUF của model khác, biến thể DPO/SFT khác hoặc tokenizer khác. · nguồn: [31][42][45]

- **Làm cho chat template của Ollama giống hệt HF** · là gì: lấy chuỗi được HF tạo bằng `apply_chat_template`, so token-by-token với prompt mà Ollama gửi; nếu cần, đặt `TEMPLATE` trong Modelfile hoặc gọi API với prompt đã serialize sẵn, tránh để Ollama tự chèn template lần thứ hai. · bằng chứng: HF khuyến nghị `add_generation_prompt=True`; model card Olmo 3 Instruct chứa system message mặc định, các marker `<|im_start|>`, `<|im_end|>` và `<|endoftext|>`.[17][27] Ollama cho phép định nghĩa `TEMPLATE`, `SYSTEM`, `.Messages`, `.Prompt`, `.Response` và các role khác trong Modelfile.[95][100] · điều kiện áp dụng: đặc biệt quan trọng với Olmo 3 Instruct, vì system prompt mặc định của model card không phải là “không có system prompt”. Phải quyết định rõ dùng system prompt mặc định hay bỏ nó ở cả hai backend. · nguồn: [17][95][100]

- **Chạy Ollama ở chế độ sampling và greedy tương đương với HF, không dùng defaults** · là gì: với greedy, đặt rõ `temperature=0`, `top_k`/`top_p` phù hợp, không repeat penalty, cùng stop và max output; với sampling, truyền cùng `temperature`, `top_p`, `top_k`, seed và penalty ở cả hai bên. · bằng chứng: Ollama mặc định `temperature=0.8`, `repeat_penalty=1.0` theo tài liệu hiện tại, `seed=0`, và `num_ctx=2048`.[1][95] llama.cpp CLI hiện nêu temperature mặc định 0.8 và seed mặc định -1, tức seed ngẫu nhiên nếu không đặt.[76][78] Transformers mặc định dùng greedy khi `do_sample=False`; temperature chỉ ảnh hưởng khi sampling.[106][111] · điều kiện áp dụng: “temperature 0 + seed” không biến hai engine thành cùng một hệ số học. Greedy vẫn có thể khác do lượng tử hóa, tie-breaking và số học; seed chỉ có tác dụng khi có RNG/sampling. · nguồn: [1][76][78][106][111]

- **Tách kiểm thử “determinism trong một backend” khỏi “parity giữa backend”** · là gì: chạy cùng Ollama request 10 lần, cùng seed và temperature 0; sau đó chạy cùng prompt qua HF; báo cáo riêng tỷ lệ lặp lại nội bộ và tỷ lệ giống Ollama–HF. · bằng chứng: Ollama mô tả seed cố định là cách tạo text lặp lại cho cùng prompt.[1][95] Transformers cung cấp `set_seed` cho reproducibility, nhưng tài liệu cũng lưu ý reproducibility đầy đủ phụ thuộc môi trường; các phép toán GPU/backend vẫn có thể khác.[111][115] Nghiên cứu về precision cho thấy greedy output có thể thay đổi do khác biệt số học.[53] · điều kiện áp dụng: pin driver, GPU, phiên bản PyTorch/CUDA/llama.cpp/Ollama; bật deterministic algorithms trong HF khi cần, chấp nhận khả năng giảm tốc; ghi lại hardware và batch size. · nguồn: [1][53][111][115]

- **Đặt `num_ctx` rõ ràng và kiểm tra prompt truncation** · là gì: cấu hình `num_ctx`/`OLLAMA_CONTEXT_LENGTH` đủ lớn cho toàn bộ prompt cộng phần output dự kiến; log `prompt_eval_count`, số token tokenizer và cảnh báo truncation. · bằng chứng: Ollama Modelfile ghi default `num_ctx=2048`.[1][95] Tài liệu/issue Ollama ghi nhận prompt vượt buffer có thể bị cắt, thường giữ phần cuối; một issue khác minh họa prompt bị cắt khi không truyền `num_ctx`.[94][96] llama.cpp CLI hiện có default context phụ thuộc phiên bản/đường chạy, với tài liệu CLI mới nêu `0 = loaded from model`, còn tài liệu cũ nêu 4096; vì vậy không nên suy luận default chỉ từ tên engine.[76][78] · điều kiện áp dụng: không so sánh HF 8192 với Ollama 2048/4096; với hội thoại dài, kiểm tra cả truncation ban đầu và context shift giữa các lượt. Context bị cắt có thể làm output khác dù temperature bằng 0.[97] · nguồn: [1][76][78][94][96][97]

- **Kiểm tra SWA của Olmo 3 thay vì tắt hoặc giả lập tùy tiện** · là gì: xác nhận GGUF chứa metadata sliding-window, chạy phiên bản llama.cpp/Ollama mới có hỗ trợ architecture tương ứng, rồi benchmark các prompt 4K, 8K, 16K; so sánh với Transformers trên cùng chiều dài. · bằng chứng: báo cáo Olmo 3 nói model dùng context huấn luyện 8192, cửa sổ SWA 4096 ở 3/4 layer và full attention ở layer cuối.[122] Tài liệu hệ sinh thái llama.cpp mô tả SWA/KV-cache và các tùy chọn context checkpoint; llama.cpp cũng có cơ chế context shift để trượt cửa sổ khi đầy.[37][40][43] · điều kiện áp dụng: không đồng nhất “context shift” với SWA của kiến trúc. Context shift là cơ chế loại bỏ token cũ khi buffer đầy; SWA là pattern attention được huấn luyện trong model. Nếu runtime không hiểu metadata/kiến trúc, kết quả dài-context có thể sai hoặc khác đáng kể. · nguồn: [37][40][43][122]

- **Dùng Ollama như backend phát triển, nhưng judge phải có calibration và kiểm soát bias** · là gì: dùng Ollama Q4_K_M để lặp nhanh, chạy judge cục bộ trên rubric JSON cố định, rồi calibration trên tập có nhãn người hoặc đáp án chuẩn; các quyết định cuối cùng chạy lại bằng Q8_0/F16 hoặc Transformers. · bằng chứng: các harness local đã dùng Ollama cho LLM-as-a-judge và nhấn mạnh judge có thể chấm sai hoặc thiên lệch; một walkthrough ghi nhận cùng model có thể tự chấm câu trả lời sai ở ngưỡng pass ổn định qua nhiều lần chạy.[61][63] Các repo đánh giá local hỗ trợ so sánh accuracy, latency và judge score qua nhiều cấu hình.[66][75] · điều kiện áp dụng: không dùng một judge cùng model/quantization để tự đánh giá chính nó mà không có mẫu đối chứng; bắt buộc lưu raw output, rubric, seed, model tag, quantization và phiên bản. · nguồn: [61][63][66][75]

- **Thử một ma trận benchmark nhỏ nhưng có sức phân biệt** · là gì: chọn 100–300 mẫu gồm exact-match, multiple-choice, code/unit tests, long-context retrieval và open-ended; chạy F16/bf16, Q8_0, Q4_K_M ở greedy và sampling; báo cáo accuracy, first-divergence rate, độ dài, JSON validity, latency và RAM. · bằng chứng: các benchmark quantization cho thấy task accuracy có thể khác nhau theo nhóm nhiệm vụ ngay khi thay đổi quantization.[46][58] Perplexity chỉ đo độ phù hợp với văn bản, không thay thế accuracy ứng dụng.[48][59] · điều kiện áp dụng: bootstrap confidence interval hoặc lặp nhiều seed cho sampling; không kết luận từ vài prompt hội thoại. · nguồn: [46][48][58][59]

- **Chọn cấu hình theo vai trò** · là gì: Q4_K_M cho interactive dev và test nhanh; Q8_0 cho local regression/judge khi đủ RAM; F16 hoặc Transformers bf16 cho reference và final production evaluation. · bằng chứng: tài liệu llama.cpp xếp Q4_K_M là cân bằng tốc độ/chất lượng, Q8_0 là near-original/reference, còn F16 là baseline float.[6][7][48] · điều kiện áp dụng: đây là heuristic, không phải bảo đảm cho Olmo 3; nếu task nhạy với token exact-match, JSON schema hoặc code, benchmark riêng trước khi chọn Q4. · nguồn: [6][7][48]

## Bẫy và cảnh báo đã biết

- **Sai model hoặc sai biến thể:** `Olmo-3-7B-Instruct`, `Olmo-3-7B-Instruct-SFT`, DPO và các bản OP﻿SA/Code không thể coi là cùng checkpoint; model card và template có thể khác.[16][17][20][22]

- **Chat template bị áp dụng hai lần:** HF đã serialize role markers, nhưng wrapper Ollama lại có thể chèn `TEMPLATE`; hoặc ngược lại, prompt raw bị gửi vào một template đã kỳ vọng `.Messages`. Kết quả là khác ngay từ token đầu.[17][95][100]

- **System prompt mặc định bị bỏ sót:** model card Olmo 3 Instruct chứa system prompt mặc định về trợ lý function-calling. Một Modelfile chỉ có user message có thể không tương đương ví dụ HF.[17]

- **“Temperature 0” không đồng nghĩa bitwise deterministic giữa engine:** nó loại sampling hoặc gần greedy, nhưng không loại khác biệt do quantization, floating-point accumulation, GPU kernel, tie-breaking, KV-cache và batch scheduling.[53][106][111]

- **Seed không sửa được prompt khác nhau:** seed chỉ kiểm soát RNG; nó không làm Q4_K_M thành bf16 và không khôi phục token đã bị truncation. Ollama và llama.cpp còn có default seed khác nhau.[1][76]

- **Repeat penalty âm thầm làm đổi output:** Ollama có `repeat_penalty`, `repeat_last_n` và các sampling option riêng; HF generate mặc định không áp dụng cùng pipeline penalty nếu đội không truyền rõ.[1][95][91]

- **Default context không thống nhất:** Ollama Modelfile hiện ghi 2048; llama.cpp CLI thay đổi theo công cụ/version và có thể lấy context từ model metadata.[1][76][78] Đừng dùng “Ollama mặc định 4096” hoặc “llama.cpp luôn 2048” như giả định cố định.

- **Truncation có thể khó phát hiện:** output vẫn hợp lệ nhưng mất phần đầu của tài liệu/hội thoại. Kiểm tra token count, log, `prompt_eval_count` và một sentinel đặt ở đầu prompt.[92][94][96][99]

- **Context shift không phải long-context miễn phí:** khi cửa sổ đầy, token cũ có thể bị loại; câu trả lời sau đó không còn tương đương với Transformers đang giữ toàn bộ context. llama.cpp có cơ chế context shift, nhưng hành vi và cờ bật mặc định phụ thuộc công cụ/version.[41][43]

- **SWA của Olmo 3 không giống sliding context thông thường:** SWA là một phần kiến trúc được huấn luyện; context shift chỉ là quản lý KV/prompt. Với Olmo 3, 3/4 layer dùng cửa sổ 4096 nhưng layer cuối full attention, nên benchmark dài-context phải kiểm tra đúng implementation.[122]

- **Perplexity có thể đánh lạc hướng:** chênh lệch PPL nhỏ không đảm bảo exact-match, reasoning, code hoặc judge agreement không đổi; ngược lại, một khác biệt token sớm có thể làm text khác hoàn toàn dù chất lượng con người tương đương.[46][48][53][57]

- **Judge tự đánh giá chính nó:** local judge có thể có bias theo model, format hoặc câu trả lời; cần blind comparison, calibration với người/ground truth và kiểm tra self-preference.[61][63]

- **Sampling defaults của Ollama và ví dụ HF Olmo 3 khác nhau:** model card Olmo 3 khuyến nghị temperature 0.6, top-p 0.95 và max token lớn, trong khi Ollama mặc định temperature 0.8; nếu chỉ gọi hai API “mặc định”, đó không phải phép so sánh model.[18][95]

- **Phiên bản runtime là một biến thực nghiệm:** hỗ trợ chat template Minja, architecture Olmo/Olmo2, SWA, quantization kernel và context handling có thể thay đổi; phải lưu phiên bản Ollama, llama.cpp, Transformers, PyTorch, CUDA/Metal và commit converter.[31][85]

## Nguồn (đánh số: tiêu đề — URL)

[1] Modelfile Reference — https://docs.ollama.com/modelfile

[6] Quantizing Models — https://ggml-org-llama-cpp.mintlify.app/models/quantizing-models

[7] Performance Tuning — https://ggml-org-llama-cpp.mintlify.app/advanced/performance-tuning

[16] allenai/Olmo-3-7B-Instruct-DPO — https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO

[17] allenai/Olmo-3-7B-Instruct README — https://huggingface.co/allenai/Olmo-3-7B-Instruct/raw/main/README.md

[18] allenai/Olmo-3-7B-Instruct Inference Example — https://huggingface.co/allenai/Olmo-3-7B-Instruct?inference_api=true

[20] allenai/Olmo-3-7B-Instruct-DPO README — https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO

[27] Templates for Chat Models — https://huggingface.co/docs/transformers/v4.43.0/en/chat_templating

[31] Olmo-3-7B-Instruct GGUF — https://huggingface.co/ramgpt/Olmo-3-7B-Instruct-OPSA-Code-GGUF

[37] llama.cpp Architecture — https://mintlify.wiki/ggml-org/llama.cpp/concepts/architecture

[40] LlamaContextOptions, SWA — https://node-llama-cpp.withcat.ai/api/type-aliases/LlamaContextOptions

[41] llama.cpp KV Cache and Context — https://agentwikis.com/wiki/llama-cpp/wiki/concepts/kv-cache-and-context.md

[42] GGUF — https://huggingface.co/docs/hub/en/gguf

[43] Context Sliding Window — https://github.com/ggml-org/llama.cpp/discussions/11845

[45] GGUF conversion and quantization workflow — https://www.datacamp.com/tutorial/gguf-format-a-complete-guide

[46] llm-quantization-bench — https://github.com/tkarim45/llm-quantization-bench

[48] Quantizing Models — https://ggml-org-llama-cpp.mintlify.app/models/quantizing-models

[53] Greedy Decoding Is Not Precision-Invariant — https://arxiv.org/pdf/2609.26621v1

[57] Which Quantization Should I Use? — https://arxiv.org/html/2601.14277v1

[58] Optimizing LLM Inference with Quantization — https://github.com/aws-samples/optimizing-llm-inference-with-quantization

[59] llama.cpp quantization perplexity example — https://github.com/ninjahawk/qwen3/blob/main/docs/source/quantization/llama.cpp.md

[61] LLM Evaluation for Engineers — https://github.com/amitbad/llm-evaluation

[63] LLM Evaluation Harness Walkthrough — https://sbezjak.github.io/llm-eval-harness/docs/walkthrough.html

[66] Simple Local LLM Evaluation Harness — https://github.com/hakanotal/llm-eval-harness

[75] Evaluation-Driven Iteration for LLM Applications — https://arxiv.org/html/2601.22025v1

[76] llama.cpp examples/main — https://github.com/ggml-org/llama.cpp/tree/master/examples/main

[78] llama.cpp CLI Reference — https://github.com/ggml-org/llama.cpp/tree/master/tools/cli

[85] llama.cpp guide: adding model architectures — https://github.com/ggml-org/llama.cpp/discussions/16770

[91] Ollama Generate API — https://docs.ollama.com/api/generate

[92] Ollama issue: prompt truncation and num_ctx — https://github.com/ollama/ollama/issues/17889

[94] Ollama issue: input truncation — https://github.com/ollama/ollama/issues/10012

[95] Ollama Modelfile Reference — https://docs.ollama.com/modelfile

[97] Ollama issue: fixed seed and temperature 0 — https://github.com/ollama/ollama/issues/586

[100] Ollama Template Documentation — https://docs.ollama.com/template

[106] Transformers Text Generation Strategies — https://huggingface.co/docs/transformers/v4.28.0/en/generation_strategies

[108] Transformers Generation — https://huggingface.co/docs/transformers/main_classes/text_generation

[111] Transformers Generation Utilities — https://huggingface.co/docs/transformers/v5.5.1/en/internal/generation_utils

[115] Transformers set_seed — https://huggingface.co/docs/transformers/internal/trainer_utils

[122] Olmo 3 — https://arxiv.org/html/2512.13961v2

## Nguồn trả về từ API
- Modelfile Reference - Ollama documentation — https://docs.ollama.com/modelfile
- Modelfile 参考 - ollama — https://ollama.atomgit.com/modelfile.html
- ollama/docs/modelfile.md at v0.1.29 · ollama/ollama — https://github.com/ollama/ollama/blob/v0.1.29/docs/modelfile.md
- คู่มืออ้างอิง Modelfile — https://lzw.me/docs/ollama/th/modelfile.html
- ollama-local/docs/modelfile.mdx at main · LlamaWritesCode/ollama-local — https://github.com/LlamaWritesCode/ollama-local/blob/main/docs/modelfile.mdx
- Quantizing Models - llama.cpp — https://ggml-org-llama-cpp.mintlify.app/models/quantizing-models
- Performance Tuning - llama.cpp — https://ggml-org-llama-cpp.mintlify.app/advanced/performance-tuning
- Modelfileリファレンス | Ollama — https://lzw.me/docs/ollama/ja/modelfile.html
- ollama-model-train-guide/docs/modelfile-reference.md ... — https://github.com/manzolo/ollama-model-train-guide/blob/main/docs/modelfile-reference.md
- llama.cpp - Liquid Docs — https://docs.liquid.ai/deployment/on-device/llama-cpp
- qwen3/docs/source/quantization/llama.cpp.md at main ... - GitHub — https://github.com/ninjahawk/qwen3/blob/main/docs/source/quantization/llama.cpp.md
- raw.githubusercontent.com — https://raw.githubusercontent.com/ggml-org/llama.cpp/HEAD/tools/quantize/README.md
- llama-cpp/examples/quantize/README.md at master - GitHub — https://github.com/Jaid/llama-cpp/blob/master/examples/quantize/README.md
- documentation/ollama-modelfile-guide.md at main · caavme ... — https://github.com/caavme/documentation/blob/main/ollama-modelfile-guide.md
- github.com · blob · mainLLM-Qwen3/docs/source/quantization/llama.cpp.md at main - GitHub — https://github.com/CSPSolutions-Dev/LLM-Qwen3/blob/main/docs/source/quantization/llama.cpp.md
- allenai/Olmo-3-7B-Instruct-DPO - Hugging Face — https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO
- README.md — https://huggingface.co/allenai/Olmo-3-7B-Instruct/raw/main/README.md
- allenai/Olmo-3-7B-Instruct · Hugging Face — https://huggingface.co/allenai/Olmo-3-7B-Instruct?inference_api=true&inference_provider=publicai
- README.md · allenai/Olmo-3-7B-Instruct at refs/pr/2 — https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/refs%2Fpr%2F2/README.md
- README.md · allenai/Olmo-3-7B-Instruct-DPO at refs/pr/6 — https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO/blame/refs%2Fpr%2F6/README.md
- allenai/OLMo-7B-Instruct — https://huggingface.co/allenai/OLMo-7B-Instruct
- allenai/Olmo-3-7B-Instruct at main - Hugging Face — https://huggingface.co/allenai/Olmo-3-7B-Instruct/tree/main
- Upload checkpoint from step 450 · allenai/Olmo-3-7B- ... — https://huggingface.co/allenai/Olmo-3-7B-Instruct/commit/d42dc9de51ee481e02b8a35d8222b86c6d7c5b20
- Update the number of parameters (#13) · allenai/Olmo-3-7B-Instruct at 6e5971d — https://huggingface.co/allenai/Olmo-3-7B-Instruct/commit/6e5971d9eba42665f5bd5a0fcf047f299ce1dccc
- allenai/Olmo-3-7B-Instruct-SFT at f528b031aedd886f655315fb7fbc60ba0fa3bd6b — https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT/tree/f528b031aedd886f655315fb7fbc60ba0fa3bd6b
- allenai/Olmo-3-7B-Instruct · Update inference examples to use the ... — https://huggingface.co/allenai/Olmo-3-7B-Instruct/discussions/7/files
- Templates for Chat Models · Hugging Face — https://huggingface.co/docs/transformers/v4.43.0/en/chat_templating
- allenai/Olmo-3-7B-Instruct-SFT at main - Hugging Face — https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT/tree/main
- Update README.md · allenai/Olmo-3-7B-Instruct-SFT at 3e0b824 — https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT/commit/3e0b8245bc8ec203cfd5f6838f2037abf3af5927
- allenai/Olmo-3-7B-Instruct at 51c1f0b28546f9cf537583ea5b9f5da4fed23ebe — https://huggingface.co/allenai/Olmo-3-7B-Instruct/tree/51c1f0b28546f9cf537583ea5b9f5da4fed23ebe
- ramgpt/Olmo-3-7B-Instruct-OPSA-Code-GGUF · Hugging Face — https://huggingface.co/ramgpt/Olmo-3-7B-Instruct-OPSA-Code-GGUF
- Mungert/Olmo-3-7B-Instruct-GGUF - Hugging Face — https://huggingface.co/Mungert/Olmo-3-7B-Instruct-GGUF
- mradermacher/Olmo-3-7B-Instruct-OPSA-Code-i1-GGUF at 6af30edcd99745ec9c289a266f98bb507d9322a9 — https://huggingface.co/mradermacher/Olmo-3-7B-Instruct-OPSA-Code-i1-GGUF/tree/6af30edcd99745ec9c289a266f98bb507d9322a9
- uploaded from leia · mradermacher/Olmo-3-7B-Instruct-OPSA-Code-i1-GGUF at aeec9c1 — https://huggingface.co/mradermacher/Olmo-3-7B-Instruct-OPSA-Code-i1-GGUF/commit/aeec9c12256525cf4ec8620d772e8a0e31a88527
- eaddario/Olmo-3-7B-Instruct-GGUF · Hugging Face — https://huggingface.co/eaddario/Olmo-3-7B-Instruct-GGUF?local-app=docker-model-runner
- Commits · mradermacher/Olmo-3-7B-Instruct-OPSA-Code-i1-GGUF — https://huggingface.co/mradermacher/Olmo-3-7B-Instruct-OPSA-Code-i1-GGUF/commits/6af30edcd99745ec9c289a266f98bb507d9322a9/Olmo-3-7B-Instruct-OPSA-Code.i1-IQ3_XXS.gguf
- architecture — https://mintlify.wiki/ggml-org/llama.cpp/concepts/architecture
- unsloth/Olmo-3-7B-Instruct-GGUF — https://huggingface.co/unsloth/Olmo-3-7B-Instruct-GGUF
- Mungert/Olmo-3-1025-7B-GGUF — https://huggingface.co/Mungert/Olmo-3-1025-7B-GGUF
- Type Alias: LlamaContextOptions - node-llama-cpp — https://node-llama-cpp.withcat.ai/api/type-aliases/LlamaContextOptions
- agentwikis.com · wiki · llama-cppKV Cache and Context — llama.cpp — Agent Wikis — https://agentwikis.com/wiki/llama-cpp/wiki/concepts/kv-cache-and-context.md
- GGUF · Hugging Face — https://huggingface.co/docs/hub/en/gguf
- Context Sliding Window #11845 - ggml-org llama.cpp — https://github.com/ggml-org/llama.cpp/discussions/11845
- Advanced Features | Notes - Vasu Grover — https://notes.itsvasugrover.com/kb/ai/llama-cpp/advanced-features/
- GGUF Format: A Complete Guide to Local LLM Inference — https://www.datacamp.com/tutorial/gguf-format-a-complete-guide
- llm-quantization-bench - GitHub — https://github.com/tkarim45/llm-quantization-bench
- Quantization Amplifies Cache-Induced Divergence in LLM Serving — https://arxiv.org/html/2609.04748
- Best Quantization Methods for llama.cpp - Sesame Disk — https://sesamedisk.com/best-quantization-methods-for-llama-cpp/
- claude-skills/llama-cpp/references/quantization-guide.md at ... — https://github.com/maystudios/claude-skills/blob/main/llama-cpp/references/quantization-guide.md
- Understanding and Mitigating Numerical Sources of ... - alphaXiv — https://www.alphaxiv.org/abs/2506.09501
- OpenVINO Backend for llama.cpp - benchmarks - GitHub — https://github.com/Liquid4All/benchmarks-llama.cpp/blob/benchmarks/docs/backend/OPENVINO.md
- Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in LLM Inference — https://arxiv.org/pdf/2609.26621v1
- Efficient On-Device LLM Inference on Apple Silicon: From ... — https://atomgradient.github.io/apple-silicon-llm-inference/paper.pdf
- deepwiki.com · ggml-org · llamaQuantization Techniques | ggml-org/llama.cpp | DeepWiki — https://deepwiki.com/ggml-org/llama.cpp/7.3-quantization-techniques
- Which Quantization Should I Use? A Unified Evaluation of llama.cpp ... — https://arxiv.org/html/2601.14277v1
- aws-samples/optimizing-llm-inference-with-quantization — https://github.com/aws-samples/optimizing-llm-inference-with-quantization
- Evaluation Methods Covered — https://github.com/amitbad/llm-evaluation/blob/main/README.md
- GitHub - HusneShabbir/rhdh-Lightspeed-Evaluation: This repository contains automated tests for evaluating RAG (Retrieval-Augmented Generation) responses from RHDH Lightspeed using various quality metrics including relevancy, faithfulness, bias, and hallucination. — https://github.com/HusneShabbir/rhdh-Lightspeed-Evaluation
- Walkthrough | llm-eval-harness — https://sbezjak.github.io/llm-eval-harness/docs/walkthrough.html
- Open-source EvalLLM for LLM Evaluation | Wil Hurley posted on the ... — https://www.linkedin.com/posts/wil-hurley_github-wilhurleyevalllm-open-source-llm-activity-7441144207320203264-bEAL
- Judges - LLM Jury Documentation — https://netraj-patil.github.io/LLM-Jury/docs/judges/
- github.com · hakanotal · llm-eval-harnessGitHub - hakanotal/llm-eval-harness: Simple Local LLM ... — https://github.com/hakanotal/llm-eval-harness
- Build software better, together — https://github.com/topics/llmevaluation
- Throughput and output-correctness benchmark data for local LLM inference on a repurposed enterprise workstation — https://zenodo.org/records/22079196
- Build software better, together — https://github.com/topics/llm-as-judge?l=typescript
- llm-as-judge · GitHub Topics — https://github.com/topics/llm-as-judge?l=python&o=desc&s=updated
- local-llm-agent — https://github.com/topics/local-llm-agent
- Hydroxide538/qwen-agentworld-35b-a3b - Ollama — https://ollama.com/Hydroxide538/qwen-agentworld-35b-a3b
- LLMJudge - LLM4Eval — https://llm4eval.github.io/LLMJudge/
- Build software better, together — https://github.com/topics/local-llm-gguf
- When “Better” Prompts Hurt: Evaluation-Driven Iteration for LLM ... — https://arxiv.org/html/2601.22025v1
- llama.cpp/examples/main at master · ggml-org/llama.cpp — https://github.com/ggml-org/llama.cpp/tree/master/examples/main
- llama.cpp/tools/cli/README.md at 7e8324f5feba1d8be3f966e1791e84107e40cdd7 · ggml-org/llama.cpp — https://github.com/ggml-org/llama.cpp/blob/7e8324f5/tools/cli/README.md?plain=1
- llama.cpp/tools/cli/README.md at 5cdadfc326163ed1fa66cf609fb1761f7928900c · ggml-org/llama.cpp — https://github.com/ggml-org/llama.cpp/blob/5cdadfc3/tools/cli/README.md?plain=1
- llama.cpp/tools/cli at master · ggml-org/llama.cpp · GitHub — https://github.com/ggml-org/llama.cpp/tree/master/tools/cli
- llama.cpp/tools/server/README.md at b11390 · ggml-org/llama.cpp — https://github.com/ggml-org/llama.cpp/blob/b11390/tools/server/README.md
- llama.cpp/docs/preset.md at 5cdadfc326163ed1fa66cf609fb1761f7928900c · ggml-org/llama.cpp — https://github.com/ggml-org/llama.cpp/blob/5cdadfc3/docs/preset.md?plain=1
- llama.cpp/examples/diffusion/README.md at master · ggml ... - GitHub — https://github.com/ggml-org/llama.cpp/blob/master/examples/diffusion/README.md
- guide: llama-cli help reformatted, organized, fleshed out ... — https://github.com/ggml-org/llama.cpp/discussions/15709
- Update llama-cli documentation (#8315) · ggml-org/llama.cpp@a8db2a9 — https://github.com/ggml-org/llama.cpp/commit/a8db2a9ce64cd4417f6a312ab61858f17f0f8584
- (không tiêu đề) — https://huggingface.co/allenai/Olmo-3-7B-Instruct?inference_api=true
- (không tiêu đề) — https://github.com/amitbad/llm-evaluation
- (không tiêu đề) — https://github.com/ggml-org/llama.cpp/discussions/16770
- (không tiêu đề) — https://docs.ollama.com/api/generate
- (không tiêu đề) — https://github.com/ollama/ollama/issues/17889
- (không tiêu đề) — https://github.com/ollama/ollama/issues/10012
- (không tiêu đề) — https://github.com/ollama/ollama/issues/586
- (không tiêu đề) — https://docs.ollama.com/template
- (không tiêu đề) — https://huggingface.co/docs/transformers/v4.28.0/en/generation_strategies
- (không tiêu đề) — https://huggingface.co/docs/transformers/main_classes/text_generation
- (không tiêu đề) — https://huggingface.co/docs/transformers/v5.5.1/en/internal/generation_utils
- (không tiêu đề) — https://huggingface.co/docs/transformers/internal/trainer_utils
- (không tiêu đề) — https://arxiv.org/html/2512.13961v2
