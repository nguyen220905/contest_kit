# SYNTHESIS — council round_06: chấm lại plan P2+P3 v2 (`docs/STAGES.md`)
> Researcher, 22/10/2026. Hội đồng: 8/8 model, 2 vòng, không lỗi provider (status.json: 16 lượt gọi, 986k token vào + 217k token ra, 0 lượt search, 175 nguồn liệt kê).
> Kết quả hội đồng là tín hiệu; mọi nhãn bên dưới đã đối chiếu code (`starter_kit/`, `eval/`), `results/`, `docs/DISCORD.md` và các trang đã mở bằng WebFetch.

## 0. Cảnh báo cần xử lý NGAY

1. **Công thức điểm local sai ở 3 model (M1 minimax, M2 openai, M6 gemini): "0.3·acc_DONES + 0.7·acc_AI"** — SAI. LB = 50·(a/15 + b/35) = 100·(acc_D + acc_A)/2, tức là **trung bình accuracy 2 domain** (macro). `evaluate.py` phải tính `balanced = (acc_DONES + acc_AI)/2`. Trên sample25, 1 câu DONES = 5 điểm, 1 câu AI = 3.33 điểm. STAGES.md dòng 126 ghi "1 câu = 4 điểm" cũng sai theo cách tính cân bằng.
2. **Chạy lại local KHÔNG đo được nhiễu.** `compare_runs.py results/exp001_zs_4bit_a results/exp001_zs_4bit_b` → **0/25 câu khác chữ**, token giống hệt (p50/p95/max 597/1738/1984), chỉ thời gian lệch (89.2 với 77.8 s/câu). 7B 4-bit greedy chạy local là tất định, nên tiêu chí S6 "chạy lại 02 holdout lần 2 → đếm câu lật = nhiễu" (STAGES dòng 145) sẽ luôn cho 0. Nhiễu thật nằm ở (a) tập câu nhỏ (thống kê), (b) server H100 (Jaime 16/10: nộp lại cùng zip thì điểm khác). Cần thay bằng McNemar/ngưỡng câu ròng + nhiễu LB.
3. **Lượt chạy bf16 của S1 rỗng**: `results/exp001_zs_bf16/predictions.csv` có 0 dòng và không có run_info.json. Mục "lệch shim (bf16 1 lần)" của S1 chưa xong.
4. **"Private bị giới hạn cứng 1 h" (M7 qwen vòng 1) là sai giả định.** Theo Discord (Jaime, 3/10), private 2 h là **giới hạn mềm**: câu nào xong trong 2 h thì được chấm. Trang evaluation (đã mở) ghi private chấm lại **toàn bộ 100% benchmark**, nghĩa là gồm cả 50 câu public. Các đề xuất "probe P để tránh fail private" vì vậy mất phần lớn lý do (chi tiết ở §2).
5. **Bác ý "cố tình raise ở dòng cuối để lấy số trang corpus qua thông báo lỗi" (M7-A vòng 1).** Đây là dùng kênh lỗi để lấy dữ liệu ẩn mà BTC đã nói không cho truy cập (7/10), nên có rủi ro bị loại. M5, M2, M6, M1 đều phản đối; M7 cũng bỏ ý này ở vòng 2. Không làm.
6. Đã kiểm (từ M5 grok), là sự thật nhưng **code BTC đã xử lý**:
   - `generation_config.json` của 32B đặt `do_sample: true, temperature 0.6, top_p 0.95`. 01/02 đều truyền `do_sample=False` tường minh (`starter_kit/02_rag/main.py:158`, `01_zero_shot/main.py:46`), nên vẫn greedy. **Code mới của mình bắt buộc giữ dòng này.**
   - `config.json` 32B đặt `use_cache: false`. LB 2080 s cho thấy KV cache vẫn bật lúc generate. Nên thêm `use_cache=True` tường minh (rẻ).
   - Chat template 32B tự chèn system "You are Olmo… date cutoff December 2024…" **khi không có system message**. 01/02 luôn có system nên không bị. Code mới phải luôn có system message.

## 1. Bảng điểm plan v2

| Model (vai) | Vòng 1 | Vòng 2 | Kết luận |
|---|---|---|---|
| M1 minimax (data_detective) | không chấm | không chấm | (ngầm: sửa nhỏ — thêm cổng nhãn) |
| M2 openai (plan_critic) | 8 | **7.5** | sửa nhỏ rồi làm |
| M3 kimi (alternative) | không chấm | không chấm | (ngầm: đảo trục sang LB, gần "sửa lớn" về thứ tự) |
| M4 deepseek (root_cause) | không chấm | không chấm | (ngầm: sửa nhỏ — thêm oracle) |
| M5 grok (red_team) | 6 | **6.5** | sửa nhỏ rồi làm ("3 sửa là chặn") |
| M6 gemini (pragmatist) | 7.5 | **7.0** | sửa nhỏ rồi làm |
| M7 qwen (competition_veteran) | 6 | (giữ 6) | sửa nhỏ rồi làm |
| M8 glm (simplifier) | 7.5 | **7.0** | sửa nhỏ rồi làm |

- 5 model có điểm (lấy điểm cuối): **trung bình 6.8, trung vị 7.0**, khoảng 6–7.5. Cả 5 kết luận **"sửa nhỏ rồi làm"**; không model nào chọn "đủ tốt để làm".
- So với round_02 (chỉ DeepSeek, plan v1): 5/10 và 6/10 (trung bình 5.5) → v2 tăng khoảng +1.3. Các sửa v1→v2 được công nhận là đúng hướng: cách ly sample25, synthetic không dùng để xếp hạng generator, C500 không chặn.
- Phê bình chung: **đúng phương pháp, sai thứ tự**. Plan dành 5–6 ngày dựng eval trước khi nộp bài đầu tiên. S5, J2/kappa và 15–20 câu người viết là thừa trên critical path. Thước đo retrieval (doc-hit trên quote PyMuPDF) đo sai chỗ.

## 2. Các sửa đề xuất — đã kiểm chứng

| # | Đề xuất (ID) | Ủng hộ (Borda) | Nhãn code | Nhãn nguồn | Bằng chứng | Ưu tiên |
|---|---|---|---|---|---|---|
| 1 | **M7-A** Dùng LB làm thiết bị đo: D1 = 02 + vá an toàn; probe P, probe prefill, probe DONES-null | M1 M8 M4 M3 M5 (14) | **Hợp lệ một phần.** Bài D1 là đúng. Probe P/prefill: giá trị thấp, vì 2 mốc BTC đã cho gần đủ thông tin (xem dưới), và lý do "private 1 h" là sai giả định. DONES-null cần tự phân loại domain trong `answer()` (hàm chỉ nhận question, corpus_dir, model_id), nên probe có thể sai. Lưới nghiệm còn thiếu (3,33). | Trang evaluation đã mở: không có câu "private 1 h"; Discord là nguồn chuẩn | 7a+3b=120 cho 5 nghiệm (3,33),(6,26),(9,19),(12,12),(15,5). Mốc 01 = 29.05 cho 7a+3b=61, tức (1,18),(4,11),(7,4). Ước P từ mốc BTC: T₀₂−T₀₁ = 2080−733 = 1347 s ≈ parse+embed (+ prefill 50×10k + chênh độ dài trả lời). Suy ra T₁₀₀(02) ≈ 2080 + (733 − load) + prefill ≈ 2.7–3.4k s, dưới xa 7200 s mềm | **Cao cho D1**; probe P chỉ khi D1 > 2700 s (như M3 vòng 2); DONES-null để sau (≥ lượt 8) |
| 2 | **M5-B** Đo "in-window hit" (quote nằm trong 4096 token cuối) thay doc-hit trên quote PyMuPDF | M1 M4 M3 M6 M7 (12) | **Hợp lệ, và mạnh hơn hội đồng nghĩ.** `config.json` 32B (đã mở): `sliding_window 4096`, **48/64 lớp sliding, 16 lớp full** (mỗi lớp thứ 4). 02 xếp chunk tốt nhất **lên đầu** (`main.py:124-141`) trong ctx 10k (`:49`). Vì vậy khi corpus đủ lấp 10k token, Fragment 1 luôn cách cuối prompt khoảng 6k token, **ngoài cửa sổ của 48 lớp**. Không cần đo cho 02; chỉ cần đo cho cách xếp mới. Thêm: 02 **không lưu doc/page** (`all_chunks` chỉ là chuỗi, `:98`) → doc-hit@k của S3b không đo được nếu không có bản sao có gắn provenance | config.json HF: xác nhận. M5/M8 cẩn thận đúng: 48/64 trước đó chỉ là ngoại suy, nay đã xác nhận. M7 sai một chi tiết ("chỉ lớp cuối full") | Hiệu ứng lên accuracy **chưa chứng minh**: 16 lớp full vẫn thấy toàn bộ context. Phải A/B evidence-last | **Cao**: A/B evidence-last (1 nút, 0 chi phí thời gian chạy) |
| 3 | **M4-A** Oracle-chunk trên sample25 để tách retrieval khỏi generator | M1 M8 M3 M2 M5 (8) | **Hợp lệ, phụ thuộc S3.** `sample_qa.json` chỉ có `id, domain, question, question_type, answer`, chưa có evidence/quote → oracle chỉ làm được sau S3 (≥ 20/25 quote). Plan có bảng lỗi 4 nhóm ở S3b nhưng **chưa có oracle** | — (thiết kế thí nghiệm, không dựa nguồn ngoài) | Chi phí: 7B 4-bit 01 ≈ 78–89 s/câu (đo) → 25 câu × 2–3 biến thể ≈ 1.5–2 h/biến thể với 02 | **Cao**, đặt trong S3b |
| 4 | **M8-A** Cắt S5 khỏi critical path; S2 còn 3 test; LB đóng vai holdout | M6 M2 M7 (7) | **Hợp lệ.** S5 (STAGES:125-139) nằm trong Tiêu chí tổng (holdout C200), nên phải sửa cả Tiêu chí tổng. 15–20 câu người viết "chưa có người phụ trách" (STAGES:138). Bảng tương quan local↔LB cần ≥ 5–6 lượt nộp mới có nghĩa | van Elburg 2508.11758 (round_02 đã mở): synthetic xếp hạng đúng retriever, không xếp được generator | — | **Cao** |
| 5 | **M3-A** Đảo trục: LB tất định làm dev harness, probe khẩu vị độ dài của judge ẩn | M4 M7 (3) | **Hợp lệ về ý.** Nhưng bản gốc (3 lượt format trước khi có sàn nhiễu) bị M2/M5/M8 phê đúng: phải có bài đối chứng D1 trước | Thiên lệch độ dài của LLM judge có 2 chiều (Zheng 2306.05685, round_01 đã mở) | Gộp vào lịch 3 bài nộp (§4 câu 2) | Trung bình |
| 6 | M1-A Kiểm nhãn `reference_short` 25 câu (atomic claim + cờ outside_doc/false_premise/closed_book) | M2 (1) | Hợp lệ, rẻ (~2 h). Gold dài 81–2677 ký tự | Krumdick 2503.05061 (round_01/02 đã mở) | — | Cao (trong S2) |
| 7 | M7-B Ctx ≤ 3.8k + evidence-last + cap 384–512 + answer-first | M6 (1) | Hợp lệ làm hướng P4, nhưng gộp 4 nút; tách ra từng nút | config.json xác nhận SWA | — | Trung bình (P4) |
| 8 | M6-C Thuê "A100 80GB" 2.5 h, quy đổi 3.35/2.04 | — | **Không áp dụng**: EzyCloudX không có A100-80GB. Hệ số 0.90/0.42/0.62 của M6 vòng 1 là độ chính xác giả (M1, M3, M5, M8 cũng chỉ ra) | — | — | Bỏ |

**Đã kiểm, giữ lại từ nhóm còn lại:** nộp mỗi lần một nút (M2-A); build zip trên Linux/WSL và kiểm "1 GB sau giải nén" (M5, M8); McNemar exact với n nhỏ cần 6–0 cặp lệch thì mới p < 0.05 (M2, M7; 2·0.5⁶ = 0.031, còn 5–0 cho 0.0625); kappa paradox khiến cổng kappa ≥ 0.8 dễ báo động giả (M7).

## 3. Ý kiến thiểu số có lý

- **M5 grok: "7B 4-bit và Ollama không được dùng để chấp nhận hay bác prompt"** (chỉ 7B bf16 với p95 token < 0.5×trần mới được). Cứng, nhưng có cơ sở: 7B MHA ≠ 32B GQA, 4-bit đổi token greedy. Nên giữ ở mức nhẹ hơn: thay đổi generator chỉ được *đề cử* bằng 7B, còn *chấp nhận* phải dựa trên 32B bf16 (phiên thuê) hoặc LB.
- **M5 grok: bài vô tình may mắn có thể khoá private.** Private chấm lại bài public điểm cao nhất. Một biến thể thắng nhờ nhiễu sẽ được chạy lại trên 100 câu với output mới. Quy tắc nên dùng: biến thể chỉ được coi là "best mới" khi hơn best cũ ≥ max(sàn nhiễu, 2.86). Cơ chế "best tự động" không chặn được điều này, nên đây là kỷ luật diễn giải chứ không phải cấu hình.
- **M1 minimax: đếm closed-book trên sample25 trước.** Nếu 32B/7B tự trả đúng ≥ 60% câu DONES khi không có corpus, giả định "retrieval là nút thắt DONES" yếu đi. Rẻ (25 lượt gọi) và nên có trong S3b.
- **M3 kimi: selection-first (LLM chỉ chọn câu trích, ≤ 5 token sinh).** Decode gần 0, runtime giảm mạnh. Nhưng gãy với câu tính toán và tiền đề sai (M6), và có thể hại Phase 2 (chuyên gia chấm chất lượng). Để làm ý tưởng P5/P6, không đưa vào P2/P3.
- **M7 qwen: Phase 2 = 30% accuracy + 30% chuyên gia + 40% innovation.** Đừng đốt hết lượt cho +1–2 điểm LB; giữ thời gian cho mô-đun CI và report.

## 4. Trả lời câu hỏi trong brief mục 8

**C1. Điểm + 3 sửa còn cần.** 6.8/10 (trung vị 7), "sửa nhỏ rồi làm". Ba sửa có đồng thuận cao nhất: (i) cắt S5/J2/kappa/câu người viết khỏi critical path, S2 gọn lại; (ii) S3b thêm oracle-chunk + in-window/evidence-last + bản sao 02 có provenance; (iii) nộp D1 ngay khi S1 xong, không đợi S6. Không sửa v1→v2 nào bị coi là sai. Thừa: J2 kappa ≥ 0.8 làm cổng; bảng tương quan local↔LB làm tiêu chí S6 (quá ít điểm); "chạy lại 02 lần 2 = nhiễu" (local tất định, §0.2).

**C2. Ba bài nộp đầu tiên** (mỗi bài 1 nút, build bằng `build.sh` trên Linux/WSL, kiểm unzip ≤ 1 GB, không có `eval/`/`dev_shim`):
1. **D1 `02-safe`**: 02 nguyên bản + chỉ vá không đổi hành vi trên corpus phẳng: `rglob` lọc `suffix.lower()==".pdf"`; try/except từng câu với fallback trả chuỗi (không load lại model); `use_cache=True` tường minh; giữ `do_sample=False`, prompt, 10k, 2048. Mục tiêu: kiểm đóng gói, có một bài hợp lệ làm best public, và **một mẫu nhiễu miễn phí** (|D1 − 57.14|). Không cần nộp lại cùng zip lần 2 (M5) vì D1 so với mốc BTC đã là một lần quay.
2. **D2 `02-evidence-last`**: D1 + đảo thứ tự fragment (chunk tốt nhất nằm sát câu hỏi). Cơ chế đã xác nhận trong code + config; thời gian chạy không đổi; rủi ro thấp.
3. **D3 `02-answer-first-512`**: tốt nhất của D1/D2 + prompt answer-first ("1–3 câu, số + đơn vị trước, trả lời đủ mọi vế, bác tiền đề sai") + `max_new_tokens` 512. Đây là probe khẩu vị độ dài của judge ẩn, đồng thời giảm thời gian. Nếu tụt ≥ 2.86 thì judge ưa dài, hoàn tác.
- Quy tắc đọc kết quả: nếu |D1 − 57.14| ≥ 2.86, delta của một nút ở D2/D3 không đọc được → dựa nhiều hơn vào phiên thuê 32B trên sample25. Probe P chỉ nộp nếu D1 > 2700 s. DONES-null để sau.

**C3. Ollama.** (a) **Có, hẹp**: backend nhanh cho vòng lặp retrieval/packing khi generator giữ cố định. Nhanh hơn rất nhiều so với HF bnb 4-bit (đo được 9 tok/s; 01 mất 78–89 s/câu). Điều kiện: `temperature 0`, seed, `num_ctx` ≥ prompt thật (mặc định chỉ 4096, đã mở FAQ: "By default, Ollama uses a context window size of 4096 tokens" → 02 với 10k sẽ **bị cắt im lặng**), template kiểm bằng raw mode. Kiểm độ lệch: chạy cùng 25 câu × prompt 02 qua HF 4-bit và Ollama Q4_K_M, đếm verdict lật; chấp nhận nếu ≤ 2/25 và không đảo thứ tự 2 biến thể. **Cấm dùng Ollama cho thí nghiệm vị trí evidence/cửa sổ 4096** cho tới khi xác nhận GGUF/llama.cpp cài đặt đúng sliding window của Olmo3 (Perplexity cũng cảnh báo điều này). (b) Judge thứ 2: không — DeepSeek API mạnh hơn 7B Q4, rẻ. (c) Lọc closed-book: được (rẻ), nhưng chỉ là cờ, không phải cổng. (d) 32B Q4 offload: không — 19 GB > 16 GB, chậm, Q4 ≠ bf16, thuê PRO 6000 rẻ hơn thời gian bỏ ra. (e) Mọi quyết định chấp nhận generator: không dùng Ollama.

**C4. GPU thuê: chọn 1× RTX PRO 6000 Blackwell 96 GB**, không chọn 2× A100-40GB.
- Lý do chính: zip chạy **nguyên trạng** (`device_map="cuda"`, 1 GPU, ~64 GB + KV). Với 2× A100-40GB, `device_map="cuda"` sẽ OOM, phải sửa sang `auto` (model parallel tuần tự), tức là đổi đường code so với bài nộp. Chênh tiền ~60k CP/giờ (3 h: 315k vs 142.5k) là nhỏ so với rủi ro đo sai.
- Thông số đã mở: PRO 6000 Server Edition 1597 GB/s (Workstation 1792 GB/s theo Perplexity), BF16 "1 PFLOP" (trang NVIDIA không ghi rõ dense hay sparse, khả năng là sparse, dense ≈ 500); H100 SXM 3.35 TB/s, BF16 1979 TFLOPS **có sparsity** → dense 989. Arch sm_120 giống card local 5060 Ti, nơi torch 2.8 cu128 đã chạy.
- Quy đổi về H100: decode ×(1.597–1.792)/3.35 → **H100 nhanh hơn ~1.9–2.1×**; prefill ~2× (lý thuyết); Docling/PyMuPDF/embedding CPU **không quy đổi được** (EPYC 9654 48 nhân vs 32 CPU server) → đo riêng, ghi ±50%. Cách hiệu chỉnh thực nghiệm: chạy **01_zero_shot** (không phụ thuộc corpus) trên sample25 với 32B, lấy load + token/câu + tok/s, rồi so với mốc LB 733 s/50 câu để suy ra hệ số decode thật của server. Sau đó chạy 02 trên corpus gold để tách prefill/decode/parse. Đặt mục tiêu dự phóng public ≤ 2400 s.
- **Thuê lúc nào**: không cần trước D1/D2 (hành vi giống 02 BTC). Thuê 2–3 h **sau S3 (đã có tài liệu gold) và trước D3/bất kỳ bài đổi generator nào**, khoảng 25–27/10. Đo: (1) load, tok/s decode, prefill ở 4k/10k; (2) 01 và 02 trên sample25 + corpus gold, mỗi cái 2 lần (đếm câu lật = nhiễu GPU thật); (3) evidence-first vs last, prompt 02 vs answer-first trên 32B; (4) phân bố token trả lời của 32B; (5) unzip + chạy offline end-to-end bài D1 trên Linux. Phải `docker cp` toàn bộ `results/` về trước khi xoá container.

**C5. Synthetic + ngưỡng.** Quy tắc "synthetic chỉ chọn retrieval" đúng (van Elburg). Nhưng chưa cần synthetic trong tuần đầu: retrieval-only có thể đo trên quote của sample25 + nhiễu C50/C200. Ngưỡng với 25 câu: đề cử lên LB khi **thắng ròng ≥ 3 câu và thua ≤ 1** (cùng retrieval, cùng judge); "chắc chắn" (McNemar exact p < 0.05) cần 6–0. Báo cáo điểm cân bằng (macro), không dùng micro hay 0.3/0.7. Chấp nhận trên LB khi Δ ≥ max(2.86, sàn nhiễu đo từ D1).

**C6. Judge.** Thiết kế đúng ý, quá nhiều cổng. Giữ 4 test: diễn đạt lại → đúng ≥ 24/25; đổi số/đơn vị/thực thể → bác ≥ 90%; lạc đề → bác 25/25; **thiếu 1 vế của câu nhiều vế → bác ≥ 80%** (mới, M7). Bỏ cổng kappa (chỉ ghi bất đồng); test long/short reference chỉ để báo cáo. `reference_short` viết dạng atomic claim (giá trị + đơn vị + thực thể + cách xử lý tiền đề sai), người duyệt, giới hạn 2 h.

**C7. Kiểm chứng còn thiếu.** (1) Build/unzip/chạy trên Linux (harness đang chạy Windows; thiếu triton). (2) Bản sao 02 có provenance (doc, page, vị trí token trong prompt) cho `retrieve()`/`LAST_DEBUG`. (3) Code mới luôn có system message, `do_sample=False`, `use_cache=True`. (4) Fallback trong except không được load lại model (ca OOM Discord 16/10). (5) Watchdog theo thời gian: khi còn < 15% ngân sách thì hạ `max_new_tokens`. (6) Chạy xong lượt bf16 7B của S1 (đang rỗng). (7) Tính token trong harness bằng tokenizer của model_id local; khi chạy 32B thì token khác nhưng dùng chung vocab (100278), nên không lệch.

## 5. DANH SÁCH SỬA CHO STAGES.md v3 (đề xuất — chưa sửa)

1. **Tiêu chí tổng** (dòng 44-51): bỏ yêu cầu `--split holdout --corpus C200` và J2 kappa. Thay bằng: `evaluate.py --submission 02_rag --split sample25 --corpus gold(+C50)` → metrics có **balanced = mean(acc_D, acc_A)**, theo domain/type, thời gian, token, `quote_recall@k`, `in_window_hit`. Holdout/synthetic chuyển xuống "P2b sau ≥ 3 bài nộp".
2. **Stage 1** (dòng 70): bỏ "4-bit chạy 2 lần = nhiễu" (đã đo: 0/25 khác chữ → ghi "local tất định"). Hoàn tất lượt bf16 (đang rỗng). Thêm tiêu chí: build zip D1 bằng `build.sh` trên WSL/Linux, unzip ≤ 1 GB, không có `eval/`, chạy `HF_HUB_OFFLINE=1` → **nộp D1** ngay khi qua.
3. **Stage 2** (dòng 82-87): còn 4 test (§4 C6); bỏ cổng kappa; `reference_short` dạng atomic claim + cờ `outside_doc/false_premise/closed_book`, giới hạn 2 h; sửa "1 câu = 4 điểm" thành 5 (DONES) / 3.33 (AI).
4. **Stage 3b** (dòng 102-110): thêm (a) bản sao `02_rag_instrumented` lưu (doc, page, token offset), (b) `quote_recall@k` + `in_window_hit` (4096 token cuối), (c) **oracle-chunk** (context thật vs đoạn chứa quote), (d) **evidence-first vs evidence-last**, (e) closed-book trên sample25 theo domain. Quyết định: oracle − thật ≥ 3 câu → ưu tiên retrieval; ≤ 1 → ưu tiên generator/format.
5. **Stage 5**: chuyển khỏi critical path (sau ≥ 3 bài nộp, chỉ retrieval-only, không chạy LLM sinh câu trả lời trên synthetic); bỏ 15–20 câu người viết cho tới khi có người phụ trách.
6. **Stage 6** (dòng 141-150): bỏ "chạy lại = nhiễu" và "bảng tương quan" làm tiêu chí. Thay bằng: ngưỡng quyết định = thắng ròng ≥ 3 và thua ≤ 1 trên sample25; ngưỡng LB = max(2.86, |D1−57.14|, nhiễu 32B đo trên GPU thuê).
7. **Thêm Stage "P1b-sớm"** (sau S3, khoảng 25–27/10): thuê RTX PRO 6000 2–3 h theo §4 C4; hiệu chỉnh hệ số thời gian bằng 01 vs LB 733 s.
8. **Lịch nộp** (dòng 157): ghi rõ D1/D2/D3 như §4 C2 và quy tắc "best mới chỉ khi Δ ≥ max(2.86, sàn nhiễu)". Probe P chỉ khi D1 > 2700 s.
9. **Thời gian**: S1 + D1 hôm nay–23/10; S2 gọn + S3 23–24/10; S3b + thuê GPU 25–27/10; S4 (C50, rồi C200 chỉ cho thời gian/retrieval) song song.
