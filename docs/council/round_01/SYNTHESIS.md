# SYNTHESIS — council round_01 (phiên mở đầu: PLAN tổng thể) — 2026-10-15
> Researcher tổng hợp + kiểm chứng. Hội đồng: 8 model trả lời vòng 1; **vòng 2 chỉ còn deepseek** (7 model OpenRouter lỗi HTTP 403 "Key limit exceeded (daily limit)"). Vì vậy bảng phiếu Borda chính thức (M8-B 3đ, M8-A 2đ, M8-C 1đ) chỉ có **1 người bầu** → không đại diện. Bảng dưới dùng thêm cột "số model đề xuất cùng ý ở vòng 1" (đếm tay từ r1/).
> Không có code của đội trong repo (chỉ `starter_kit/`), nên nhãn code đối chiếu với `starter_kit/02_rag/main.py` + dữ liệu thật (sample_qa.json, LB Codabench).

## 0. Cảnh báo cần kiểm tra NGAY
1. **Brief sai giả định "chưa có thông tin LB"** — LB public đã có 36 bài (lấy qua `https://www.codabench.org/api/phases/30064/get_leaderboard/`, đã tự gọi): Zero-shot BTC **29.05 / 733 s**, Simple RAG BTC (= 02_rag) **57.14 / 2080 s**, top **69.05 / 1288 s**, 68.57 / 668 s, 68.10 / 605 s. LB **có hiển thị cột thời gian** → mỗi lượt nộp cho biết runtime (profiling miễn phí). ⇒ KHÔNG tốn lượt nộp 01_zero_shot / 02_rag nguyên bản.
2. **Cấu trúc điểm public**: mọi điểm LB khớp đúng công thức 50·(a/15 + b/35) → public = 50 câu, chia **15 / 35** giữa 2 domain (chưa biết domain nào 15; tỉ lệ mẫu 10 DONES/15 AI gợi ý DONES = 15). Bước nhảy nhỏ nhất 0.476 điểm; 1 câu ở domain nhỏ = **3.33 điểm**, ở domain lớn = 1.43 điểm. Private = toàn bộ 100 câu (evaluation page). Nhiễu lấy mẫu public→private ước ~±7.6 điểm (SE nhị thức, p≈0.6) → chênh public < ~4 điểm không đáng tin.
3. **Trọng số Phase 2** (evaluation page, đã mở): điểm cuối = 30% accuracy Phase 1 + 30% chất lượng câu trả lời do chuyên gia xếp hạng + **40% technical innovation** ("appropriate use of Computational Intelligence techniques", "originality"). Top 20 vào Phase 2. ⇒ thành phần CI không phải "trang trí" mà là trọng số lớn nhất; và câu trả lời quá cộc có thể bị chuyên gia chấm thấp ở 30% chất lượng.
4. **BGE-M3 và bge-reranker-v2-m3 không vừa zip 1 GB**: đã kiểm qua HF API — mỗi model safetensors 2.27 GB (fp32); fp16 ≈ 1.14 GB vẫn vượt. Brief mục 7 gợi ý chúng → phải bỏ.
5. **Docling không bật được table structure trên server**: `starter_kit/02_rag/main.py:60` ghi `do_table_structure=False  # avoids needing cv2/libGL, not in the image`. ⇒ bảng phải lấy bằng PyMuPDF `find_tables()` / pdfplumber, không trông vào Docling.
6. **Gold mẫu dùng kiến thức ngoài tài liệu và có câu tiền đề sai**: q6 tính lưu lượng thể tích bằng khối lượng riêng Li ≈505 kg/m³ (không từ bảng); q3 tính λ=ln2/T½; q7 "on-line nitrogen trap" — gold: không có bẫy on-line (bẫy off-line 500–600 °C); q9 gold dùng kích thước toà nhà **IFMIF** 137×111×40.5 m. ⇒ prompt "chỉ dùng số có trong fragment" (M8-A) sẽ làm hỏng câu kiểu q3/q6; cần "ưu tiên số trong tài liệu, được dùng hằng số/kiến thức chuẩn để suy ra, nói rõ khi tiền đề sai".
7. Robustness baseline: `02_rag/main.py:89` chỉ `glob("*.pdf")` (không đệ quy, phân biệt hoa thường) và `answer()` không bọc try/except → một lỗi = mất câu/mất bài. Luôn `return str`.
8. Thời gian private: Codabench API `execution_time_limit` public 3600 s, private **7200 s** cho 100 câu (gấp đôi số câu) → per-question budget giữ nguyên ~như public; watchdog theo thời gian/câu là bắt buộc.

## 1. Bảng đề xuất (top 5 + phụ) — đã kiểm chứng
Ưu tiên = (Kỳ vọng × Tự tin) / Chi phí; K=điểm % kỳ vọng, T=tự tin (0–1), C: S=1, M=2, L=3.

| # | Đề xuất | Model ủng hộ (r1) · phiếu Borda | Nhãn code | Nhãn nguồn | Bằng chứng | (K×T)/C |
|---|---|---|---|---|---|---|
| 1 | **Pipeline runtime nhanh + an toàn**: PyMuPDF đa tiến trình (32 CPU) + `find_tables`, embed GPU, cap `max_new_tokens` ~512–768, context ~3–6k, try/except mọi tầng, watchdog, zip < 900 MB | grok-B, gemini-A, qwen-C/A, glm-A, deepseek-C(r2), minimax-B (6/8) · M8-B **3đ** | hợp lệ, chưa làm — baseline đang: Docling `02_rag:56-61`, embed CPU `:86`, ctx 10k `:49`, 2048 token `:33` | xác nhận: PyMuPDF benchmark (0.474 vs 11.0 s/trang, bên PyMuPDF tự chạy) ; LB (605–1288 s cho top) | RAG BTC 2080 s vs zero-shot 733 s ⇒ ~22 phút là parse+prefill; top LB chạy 10–20 phút | (+5 × 0.8)/1 = **4.0** |
| 2 | **Định tuyến tài liệu theo tiêu đề + hybrid BM25+dense (RRF) + header ngữ cảnh mỗi chunk** `[Doc title | section | page]` | openai-B, gemini-A, qwen-A, glm-B, kimi-A, deepseek-B, grok-B (7/8) | hợp lệ, chưa làm — baseline chỉ cosine MiniLM `02_rag:124`, không BM25/không title | xác nhận: Contextual Retrieval (−49%/−67% lỗi top-20), 2604.01733 (hybrid+rerank tốt nhất, BM25 > dense trên bảng/số), WattBot (BM25 > semantic), ERC2 winner (router + LLM rerank) | 13/15 câu AI mẫu nêu tên paper/phương pháp (đếm tay sample_qa.json) | (+8 × 0.6)/2 = **2.4** |
| 3 | **Hợp đồng câu trả lời cho judge nhị phân**: câu đầu = đáp án trực tiếp (số + đơn vị), xử lý tiền đề sai, 2–5 câu giải thích có nguồn; không độn | grok-A, qwen-B, kimi-C, glm-C, minimax-C (5/8) · M8-A **2đ** | hợp lệ một phần — system prompt BTC "based on your knowledge" `02_rag:28-31` không dùng context; **sai giả định** ở vế "chỉ dùng số trong fragment" (xem cảnh báo 6) | xác nhận: overview "Reference answers are deliberately concise"; evaluation "binary verdict"; Zheng 2023 (bias độ dài) | Gold mẫu q7 (tiền đề sai), q25 (81 ký tự) | (+5 × 0.6)/1 = **3.0** |
| 4 | **Eval harness trước mọi lượt nộp**: corpus local có gold-evidence, đo coverage/oracle, Recall@k/doc-hit@1, judge local hiệu chỉnh trên 25 gold, ngưỡng nhiễu | deepseek-A, minimax-A/C, qwen-C, openai-A/C, grok-C (6/8) · M8-C **1đ** | hợp lệ, chưa làm (`eval/` chưa có — AGENTS.md "Eval chuẩn CHƯA dựng") | xác nhận: evaluation page (50% public, nhị phân) | Cảnh báo 2: bậc 0.476, 1 câu = 1.43/3.33 điểm | hạ tầng (bắt buộc, không tính K) |
| 5 | **Mô-đun CI cho paper**: gộp tín hiệu retrieval (BM25, dense, title-match, numeric-overlap) bằng hàm thuộc mờ / Choquet, trọng số tối ưu bằng tiến hoá (DE/CMA-ES) offline; + tiến hoá prompt (GEPA/EvoPrompt) offline, bundle JSON | gemini-C, qwen-A5, kimi-C (3/8) | hợp lệ, chưa làm; 0 s runtime thêm | xác nhận: EvoPrompt (ICLR'24), GEPA (ICLR'26), Promptbreeder, Allo-Self-RAG (FIS + SA, IJFS 2026), RRF (SIGIR'09) | Phase 2: 40% innovation (cảnh báo 3) | (+3 × 0.5)/2 = 0.75 điểm LB, nhưng giá trị Phase 2 cao nhất |
| 6 | PoT có cổng (regex) cho câu tính toán: Olmo sinh code Python, `exec` sandbox (chỉ `math`), chèn kết quả | gemini-C, qwen-B, kimi-B (3/8) | hợp lệ; Olmo template có `<functions>`/`<function_calls>` (đã mở tokenizer_config) | xác nhận: PoT (TMLR'23, +12% vs CoT) | q3, q6, q9, q15 cần phép tính | (+2 × 0.5)/2 = 0.5 |
| 7 | Không nộp 01/02 nguyên bản | grok-B (đối lập: openai, gemini, qwen, glm muốn nộp) | — | **xác nhận** LB đã có 2 baseline của BTC | cảnh báo 1 | tiết kiệm 2 lượt |

Đề xuất bị hạ / loại:
- BGE-M3, bge-reranker-v2-m3 (brief mục 7): **không áp dụng được** — vượt 1 GB (HF API). Thay: bge-small-en-v1.5 (133 MB), bge-base-en-v1.5 (438 MB fp32 / ~220 fp16), reranker mxbai-rerank-base-v1 (369 MB) hoặc gte-reranker-modernbert-base (598 MB) hoặc bge-reranker-base (~556 MB fp16). Kiểm tra ngân sách: Docling layout + embedder + reranker < 900 MB.
- Speculative decoding với draft LLM (brief A): **vùng xám luật** ("only model you may load") và chỉ phục vụ tie-break → thay bằng **prompt lookup decoding** (`prompt_lookup_num_tokens`, không cần model phụ, hợp với RAG vì câu trả lời chép n-gram từ context; có trong docs transformers v5.14 — đã mở; không hỗ trợ batch). Cần đo trên Olmo.
- LoRA/adapter domain: không có dữ liệu train, không chạy được 32B local, rủi ro cao → loại cho Phase 1 (không model nào ủng hộ).
- HyDE/multi-query: 2604.01733 cho thấy lợi ích hạn chế với câu số liệu; thêm 1 lượt generate/câu → để sau, chỉ bật khi retrieval kém tin cậy.
- Self-consistency k>1: nhân thời gian; tie-break phạt chậm → chỉ cân nhắc cho câu tính toán ngắn.

## 2. Kiểm chứng nguồn mà top đề xuất dựa vào
| Nguồn (do model nêu) | Nhãn | Ghi chú |
|---|---|---|
| LB Codabench (grok) 29.05/57.14/68.1 | **xác nhận** | Đã gọi API; còn 69.05 & 68.57 mới hơn |
| evaluation page (grok/glm): judge ẩn nhị phân, 50% public, top 20, tie-break thời gian | **xác nhận** | + trọng số 30/30/40 |
| overview "Reference answers are deliberately concise" (grok, minimax) | **xác nhận** | |
| join-the-challenge: 1h gồm load model/index/preprocess; không truy cập corpus eval (qwen, glm) | **xác nhận** | |
| Contest Rules PDF (qwen): Sponsor "review and verify ... compliance" | **xác nhận** | Không có câu nào cụ thể về model phụ |
| HF bge-m3 / reranker-v2-m3 2.27 GB (gemini, grok) | **xác nhận** (HF API) | |
| pymupdf.io/benchmark (glm) | **xác nhận**, nguồn có lợi ích (vendor) | cần tự đo |
| S-WinStat ari-dasci (gemini) | **xác nhận** repo của ARI/DaSCI | không nêu tác giả cụ thể; tin tức IFMIF-DONES xác nhận Aguilera-Martos trong BTC |
| ifmif-dones.es tin tức (qwen) | **xác nhận** | 100 câu, chuyên gia IFMIF-DONES + DaSCI soạn ~1 năm, ~20 đội vào Phase 2 |
| 2604.01733 (openai, deep research) | **xác nhận** | |
| WattBot 5th (openai, kimi) | **xác nhận** | BM25 > semantic; chất lượng trích PDF quan trọng hơn retrieval phức tạp |
| ERC2 "How we won" (kimi, link besthub.dev) | **xác nhận một phần** | besthub là bản đăng lại; timetoact xác nhận winner dùng router + dense + LLM rerank |
| QD for RAG ACL-SRW 2025 (kimi) | **xác nhận** | +36.7% MRR@10, +11.6% F1 |
| 2510.12462 (qwen: "factual 9.12→4.00, verbosity −0.3") | **xác nhận bài tồn tại; số liệu không thấy trong abstract** | hạ tự tin vào con số |
| 2410.02736 CALM (qwen) | **xác nhận** | |
| researchgate DONES buildings L-shape 160×75×32 (grok) | **không mở được** | không dựa vào |
| cis.ieee.org activities page (qwen) | **không mở được** (HTTP 418) | |

## 3. Ý kiến bất đồng đáng chú ý
- **Nộp baseline trước hay không?** openai/gemini/qwen/glm: nộp 02_rag ngay để có mốc. grok (thiểu số): không cần — và **grok đúng** vì LB đã có mốc BTC. Tuy vậy, ý của qwen "nộp cùng 1 zip 2 lần để đo nhiễu judge" vẫn có giá trị — làm với bài của mình, không phải baseline.
- **Atomic vs giải thích**: grok muốn câu trả lời atomic; qwen muốn giữ độ dài như gold; kimi muốn tiến hoá định dạng. Bằng chứng: tham chiếu ẩn "deliberately concise" + judge nhị phân ủng hộ "đáp án trước"; nhưng 30% Phase 2 là chuyên gia chấm chất lượng → đề xuất dung hoà: **Answer-first + giải thích ngắn có nguồn**, đo bằng A/B local.
- **Document-first đọc cả paper (kimi-A, thiểu số)**: có lý cho câu AI nêu tên paper (đọc 15–30k token của đúng 1 paper tránh mất bảng/ký hiệu), nhưng prefill 32B dài tốn thời gian (BTC: gần 65k OOM). Đáng giữ như nhánh thí nghiệm có điều kiện: chỉ khi doc-hit@1 cao, giới hạn ~12–16k token theo section.
- **Agent tìm kiếm lặp ReAct (kimi-B, thiểu số)**: phù hợp câu comparative/causal nhưng rủi ro thời gian/format; để sau khi pipeline 1-lượt ổn.
- **Corpus mục tiêu theo nhóm ra đề (gemini-B)**: hợp lý (WinStat thuộc ARI/DaSCI; câu GPAIS, FCM là chủ đề nhóm Granada) — dùng để dựng corpus dev, KHÔNG được giả định corpus thật chỉ chứa nhóm này.
- **Thuê H100**: gemini muốn hoãn tới đầu 11; deepseek-r2 muốn 1 giờ sớm cho thí nghiệm oracle 32B. Vì LB trả runtime, có thể hoãn thuê đến khi cần đo hành vi 32B (format/PoT).

## 4. Trả lời các vấn đề mở C–G
**C. Luật model.** Văn bản Codabench (đã mở): `model_id` là "The **only** model you may load". Nhưng bài 02_rag của chính BTC bundle MiniLM + layout Docling (`starter_kit/02_rag/build.sh`), nên cách hiểu thực tế = "LLM duy nhất". Câu Discord "use a different model in your internal pipeline **to try**" mơ hồ (có thể chỉ là dev local). Kết luận an toàn: (1) Olmo-3.1-32B là LM duy nhất sinh văn bản trên server; (2) embedder/reranker nhỏ, không sinh văn bản: rủi ro thấp (tiền lệ BTC) nhưng hỏi xác nhận; (3) **không** dùng draft LLM/rewriter LLM khác — dùng prompt-lookup decoding và chính Olmo để rewrite. Câu nên hỏi (kênh Discord, lưu ảnh):
> "To confirm the 'only model you may load' rule for the FINAL submission: (1) Is it OK to bundle and load small non-generative models (a sentence-transformers embedder and a cross-encoder reranker, like MiniLM/Docling in 02_rag) as long as all answer text is generated by allenai/Olmo-3.1-32B-Instruct? (2) Is loading any second causal LM (e.g., a small draft model for speculative decoding or a query rewriter) forbidden? (3) May Olmo-3.1-32B be loaded quantized (bitsandbytes 8-bit) or with a PEFT/LoRA adapter we bundle?"

**D. Dev 16 GB.** Olmo-3-7B-Instruct cùng họ tokenizer (đã mở: GPT2Tokenizer, cùng special tokens `<|im_start|>/<|im_end|>/<|endoftext|>`, chat template cùng cấu trúc; khác system prompt mặc định — 32B tự chèn "You are Olmo… date cutoff December 2024" nếu không có system message). 7B bf16 ≈ 14.6 GB → quá sát 16 GB khi có context; dùng bnb 8-bit (model card chỉ dẫn) hoặc 4-bit NF4. torch 2.8.0+cu128 hỗ trợ sm_120 (Blackwell). 7B chỉ để so sánh tương đối retrieval/prompt; hành vi 32B (format, PoT, độ dài) đo bằng LB (có cột thời gian) hoặc thuê H100 2–4 giờ khi cần (sau khi có harness, ~cuối 10; lần 2 trước khi đóng băng). Ước lượng thời gian: generation 32B batch-1 ≈ 20–35 tok/s (ước tính theo băng thông, CHƯA đo) — đối chiếu: zero-shot BTC 50 câu tối đa 2048 token = 733 s tổng.

**E. Corpus + dev.** (1) Tải đúng các PDF cho 25 câu mẫu (tên paper có sẵn; DONES: scipub.euro-fusion.org, KIT, JFE open access PMC9596351, JACoW/IPAC); (2) thêm nhiễu cùng chủ đề từ sites.txt (bản trùng arXiv vs publisher là nhiễu thật); (3) sinh 50–100 câu tổng hợp bằng LLM ngoài ở bước prep (có internet, chỉ cho eval local), mỗi câu kèm gold-evidence (doc, page) và đủ 5 loại câu; (4) tách holdout **theo tài liệu**; 25 câu mẫu giữ làm holdout cuối, không tuning trên nó. Đo retrieval bằng Recall@k/doc-hit@1 (không nhiễu judge) trước, rồi mới đo accuracy.

**F. Ngân sách 1 h.** Dữ kiện: RAG BTC 2080 s vs zero-shot 733 s ⇒ Docling + prefill 10k tốn ~22 phút; top LB 605–1288 s. Kế hoạch: PyMuPDF multiprocess (0.47 s/trang/process theo benchmark vendor → 32 process) + bảng `find_tables` chỉ ở trang có bảng; embed bằng GPU trước khi load Olmo hoặc song song; context 3–6k token; `max_new_tokens` 512–768 với dừng sớm; prompt lookup decoding thử nghiệm. Không batch được giữa các câu (answer() tuần tự); index xây 1 lần ở câu đầu. Zip: MiniLM 91 MB / bge-small 133 MB / bge-base 438 MB (fp32) + reranker ≤ 600 MB; kiểm `du` bản giải nén < 900 MB.

**G. Judge local.** Mô phỏng judge ẩn: verdict nhị phân so với tham chiếu ngắn. Dùng 2 judge (1 LLM mạnh qua API chỉ cho eval local + Olmo-7B) với rubric "đúng nếu khớp sự kiện/số chính (sai số làm tròn hợp lý), không mâu thuẫn tham chiếu". Hiệu chỉnh: (a) chấm 25 gold → phải ≥ 24/25 đúng; (b) chấm bản gold bị làm sai số/đơn vị → phải bác ≥ 90%; (c) chấm bản "gold rút gọn 1 câu" → phải vẫn đúng (kiểm tính chịu độ ngắn). Nhiễu: đo bất đồng 2 judge và chấm lặp 3 lần; ngưỡng ra quyết định local ≥ 3 câu lật trên holdout; trên LB coi Δ < 4 điểm là nhiễu.

## 5. Đề xuất mục mới cho PLAN (chưa sửa PLAN.md)
| ID gợi ý | Việc | Nguồn | Kỳ vọng | Tự tin | Chi phí |
|---|---|---|---|---|---|
| P1' | Venv khớp server (torch 2.8.0+cu128, transformers 5.17.0) + smoke test `answer()` với Olmo-3-7B 8-bit; harness eval: predictions.csv + judge nhị phân hiệu chỉnh (G) | setup + council-r01 (M4, M5, M1) | hạ tầng | cao | S |
| P2' | Corpus dev: PDF của 25 câu mẫu + DONES công khai + nhiễu sites.txt + 50–100 câu tổng hợp có gold-evidence; đo Recall@k/doc-hit@1 của baseline | council-r01 (M1, M7, M3) | hạ tầng | cao | M |
| P3' | Bài nộp #1 "fast-safe": PyMuPDF đa tiến trình + find_tables, MiniLM GPU, BM25+dense RRF, title boost, header chunk, ctx ~5k, prompt answer-first, max_new_tokens 640, try/except + watchdog, zip < 900 MB. Bác bỏ nếu LB ≤ 57.14 hoặc runtime > 1500 s | council-r01 (M8-B, M2-A, M3-A, M5-A) | +5…+10 | vừa | M |
| P4' | A/B định dạng câu trả lời (answer-first vs dài kiểu gold vs atomic) trên judge local, rồi PoT có cổng cho câu tính toán | council-r01 (M8-A, M5-B, M6-C) | +3…+6 | vừa–thấp | S |
| P5' | Mô-đun CI: fuzzy fusion tín hiệu retrieval, trọng số tối ưu bằng DE/CMA-ES offline (so với RRF), + GEPA/EvoPrompt tiến hoá prompt offline; ghi log hội tụ cho paper | council-r01 (M3-C, M5-A5, M6-C) + evaluation page 40% | +1…+4 LB; cao cho Phase 2 | vừa | M |

Hỏi BTC trên Discord (câu ở mục C) trước khi đưa reranker vào P3'.

## 6. Chi phí phiên
Hội đồng: 758,651 token vào + 99,369 token ra = **858,020 token** (grok riêng 567,898 vào); `searches_total` trong status.json = 0 (bộ đếm không ghi nhận, nhưng 171 nguồn được trả về). Perplexity: 1× sonar-pro-search (243/3,068 token) + 1× sonar-deep-research (358/17,072 token). OpenRouter đã chạm hạn mức ngày ($8.76) → vòng 2 hụt 7/8 model.
