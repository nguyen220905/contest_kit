# EXPERIMENTS — nhật ký thí nghiệm
Quy ước: 1 thí nghiệm = 1 thay đổi. Ghi cả thất bại. Kết luận phải trả lời "VÌ SAO".
Local = balanced acc = mean(acc_DONES, acc_AI) trên sample25 (judge chưa có → chưa có điểm local). LB = public 50 câu (15/35);
best mới chỉ khi Δ ≥ max(2.86, |D1 − 57.14|) (xem STAGES.md "Ngưỡng quyết định").

| ID | Ngày | Giả thuyết → thay đổi | Local | Δ vs best | LB | Kết luận | Commit |
|---|---|---|---|---|---|---|---|
| BTC-01 | 27/9 | Baseline BTC zero-shot (`starter_kit/01_zero_shot`) | — | — | 29.05 / 733 s | mốc, không nộp lại | — |
| BTC-02 | 27/9 | Baseline BTC RAG (`starter_kit/02_rag`) | — | — | 57.14 / 2080 s | mốc, không nộp lại | — |
| exp-001 | 22/10 | Harness Stage 1: 01_zero_shot, Olmo-3-7B 4-bit, sample25, 2 lượt | **13.34** (3/25; DONES 20.0, AI 6.67) | — | — | local tất định (0/25 khác chữ); 4-bit ≠ bf16 (3/3 khác chữ); 7B zero-shot trả lời chung chung, thiếu số/chi tiết | 057ff50 |
| D1 | 22/10 | `02-safe`: 02_rag + vá an toàn (PDF đệ quy/hoa thường, không reload khi lỗi, try/except) | 38.34 (exp-002) | — | **57.14** (nộp 29/10) | = đúng baseline BTC → server tất định, nhiễu LB với cùng code = 0 | d31608b |
| exp-002 | 22/10 | D1 (`02-safe`) trên corpus gold (18 PDF/366 trang), sample25, 7B 4-bit | **38.34** (10/25; DONES 30.0, AI 46.67) | +25.0 vs exp-001 | — | RAG > zero-shot: +7 câu, 0 câu mất (verdict lật 7, đều lên) → tiêu chí "02 > 01" đạt | (commit này) |
| exp-003 | 2/11 | Phòng thí nghiệm retrieval (không LLM): 12 cấu hình, sample25 (20 câu có quote), corpus gold 18 PDF | MRR c0(D1) 0.366 → c9 0.470 | — | — | hybrid(MiniLM+BM25)+tiêu đề tốt nhất trên chunk docling; rerank ms-marco tệ (0.191); n=20 → chênh 1–3 câu | (commit này) |
| D3 | 2/11 | `d3_hybrid_title`: D1 + retrieval c9 (dsqa: chunk tách đoạn dài + tiêu đề tài liệu khi index + hybrid RRF MiniLM/BM25), giữ thứ tự tốt nhất trước | lab: r@1 .30→.40, r@20 .60→.75; C200 in_ctx@10k .55→.70 | **+7.62** | **64.76** / 2088 s (nộp 2/11) | XÁC NHẬN: retrieval là đòn bẩy lớn; lab dự đoán đúng hướng. Hạng ~19–21/49 | 6a55065 |
| exp-004/005 | 2/11 | Local corpus gold, 7B 4-bit: D3 (exp-004) vs D4 (exp-005) | D3 **35.0** (9/25) · D4 **26.66** (7/25) | D4 −2 câu, 0 câu được | — | Local dự đoán ĐÚNG hướng D4 tụt (LB −10.95). Lưu ý: local gold (không nhiễu) KHÔNG thấy lợi của retrieval (D3 9/25 ≈ D1 10/25, LB +7.62) | (commit này) |
| D4 | 3/11 | `d4_context_prompt`: D3 + system prompt mới (dùng fragment làm nguồn chính, tiền đề sai, answer-first) + user turn `context + "Question: "` | 26.66 (exp-005) | **−10.95** | **53.81** (nộp 3/11) | BÁC BỎ: prompt answer-first/"step" làm câu trả lời dài hơn (+157 token) và sai số ở bước suy luận (local: lưu lượng 32 L/s thay vì 97.5; trần ⌈L/3⌉=171 thay vì 170). Giữ nguyên prompt 02_rag | — |
| exp-006 | 3/11 | Lab rerank (cross-encoder top-N trên c9 Docling, chunk kèm tiêu đề): mxbai-xs / mxbai-base / gte-modernbert / bge-reranker-base / msmarco; gte top 20/50/100/200 | C200 MRR c9 .407 → **c29 gte top100 .535**; in_ctx@2k/5k/10k .60/.80/.85 → .75/.90/.95 · gold MRR .477 → .584 | — | — | Chỉ gte-reranker-modernbert thắng (mxbai/bge/msmarco hoà hoặc tệ hơn). top100 > top50 > top20; top200 ≈ top100. fp16 = fp32 (lệch điểm ~3e-4) | (commit này) |
| D5 | 3/11 | `d5_gte_rerank`: D3 + rerank top-100 bằng gte-reranker-modernbert-base fp16 (GPU), đuôi danh sách giữ thứ tự hybrid; prompt/generation giữ nguyên | lab c29 (trên) | chờ | chờ nộp | Giả thuyết: top fragment đúng hơn → 32B (bám fragment đầu) trả lời đúng hơn. Bác bỏ nếu LB ≤ 64.76 + 1.43 | (commit này) |
| D2 | 22/10 | `02-evidence-last`: D1 + đảo thứ tự fragment (tốt nhất nằm cuối, trong cửa sổ 4096 của 48/64 lớp sliding của 32B) | chưa chạy | **−8.09** | **49.05** (nộp 30/10) | BÁC BỎ: 32B dựa vào fragment ĐẦU context; độ chính xác top fragment là đòn bẩy lớn | 5e6a7a8 |

## Chi tiết
### exp-001 — 01_zero_shot local (kiểm harness, đo tất định + lệch shim)
- Lệnh: `.venv\Scripts\python eval/run_pipeline.py --submission starter_kit/01_zero_shot --split sample25 --run results/exp001_zs_4bit_{a,b} --quant 4bit --offline`
  và `--quant none --run results/exp001_zs_bf16` (dừng sau 3 câu: VRAM tràn sang RAM do GPU dùng chung, ~2.5 tok/s).
- Kết quả: 25/25 câu trả str, 0 lỗi. Token sinh p50/p95/max 597/1738/1984. 77–89 s/câu (lệch ~15% giữa 2 lượt; GPU dùng chung).
  `compare_runs` a vs b: 0/25 câu khác chữ. 4-bit vs bf16 (3 câu): 3/3 khác chữ, token 533 vs 528.
- Chấm (judge-v2 DeepSeek, refs đã duyệt `data/work/sample25_refs.jsonl`): balanced 13.34 = 3/25 đúng (#1, #2, #22).
  Theo loại: causal 0, comparative 33.3, factual 14.3, predictive 25, procedural 0. Với refs nháp cũng 13.34 → không do refs khắt khe.
- Nhận xét — vì sao: greedy + cùng cấu hình → tất định, nên "chạy lại để đo nhiễu" vô nghĩa ở local. Lượng tử 4-bit đổi cách diễn đạt
  → local 4-bit chỉ để so sánh tương đối; thay đổi generator phải kiểm bằng bf16/32B (GPU thuê). Olmo-7B rất dài dòng (p95 1738 token).

### exp-002 — D1 trên corpus gold (Stage 3b-a: 02 vs 01)
- Lệnh: `eval/evaluate.py --submission submissions/d1_02_safe --split sample25 --corpus data/corpus/gold --run results/exp002_d1_gold --quant 4bit --offline --refs data/work/sample25_refs.jsonl`
  — bị dừng ở giới hạn 2h sau 22/25 câu; 3 câu còn lại chạy riêng (results/exp002_d1_gold_rest) rồi gộp (run_note.json), chấm lại đủ 25.
- Index: 18 PDF → 2013 chunk. Thời gian local 90–790 s/câu (prefill 10k trên 7B 4-bit + GPU dùng chung) → không đại diện server.
- Kết quả: 10/25 đúng, balanced 38.34 (DONES 3/10, AI 7/15). So exp-001 (zero-shot): 7 câu lật, cả 7 từ sai → đúng.
  Theo loại: causal 50, comparative 33.3, factual 28.6, predictive 75, procedural 20.
- Nhận xét — vì sao: có tài liệu đúng trong corpus thì RAG giúp rõ (giống LB 57 vs 29). DONES vẫn thấp (3/10): q1, q2, q4 không có
  tài liệu nguồn trong corpus; procedural (tính toán) 20% → ứng viên PoT/format. Chưa có retrieval metrics (D1 không ghi LAST_DEBUG) → cần bản instrumented.

### exp-003 — phòng thí nghiệm retrieval (`eval/retrieval_eval.py`, thư viện `dsqa/`)
- Lệnh: `.venv\Scripts\python eval/retrieval_eval.py --corpus data/corpus/gold --split sample25 --configs c0,...,c11 --run results/retr_gold_v1`
  Hit = quote evidence nằm trong top-k chunk (rapidfuzz partial_ratio ≥ 90). 20 câu có quote → 1 câu = 0.05.
- Kết quả (r@1 / r@3 / r@5 / r@20 / MRR):
  c0 D1 docling+MiniLM .30/.40/.45/.60/.366 · c1 pymupdf+MiniLM .30/.35/.35/.55/.351 · c2 pymupdf+BM25 .10/.30/.55/.70/.257 ·
  c3 pymupdf+hybrid .30/.45/.45/.65/.379 · c4 +bge-small .20/.30/.45/.70/.306 · c5 +bge-base .15/.35/.35/.75/.276 ·
  c6 c4+rerank ms-marco .10/.15/.25/.60/.191 · c7 c4+tiêu đề .35/.50/.60/.70/.448 ·
  c8 docling+MiniLM+tiêu đề .20/.40/.55/.75/.337 · **c9 docling+hybrid(MiniLM,BM25)+tiêu đề .40/.50/.50/.75/.470** ·
  c10 docling+hybrid(bge-small)+tiêu đề .30/.35/.40/.70/.359 · c11 pymupdf+hybrid(MiniLM)+tiêu đề .25/.45/.55/.60/.370.
- PyMuPDF parse 0.009 s/trang (rất nhanh so với Docling). Model: MiniLM 92 MB, bge-small 134 MB, bge-base 439 MB, ms-marco CE 92 MB.
- Nhận xét — vì sao: BM25 bắt tên paper/thuật ngữ chính xác; gắn tiêu đề giúp câu nêu tên paper (13/15 câu AI). bge không hơn MiniLM ổn định
  (n nhỏ). Rerank ms-marco (huấn luyện trên web search) hạ thứ hạng đoạn khoa học → loại. Corpus chỉ 18 tài liệu → cần kiểm lại với nhiễu C200.

### exp-003b — phòng thí nghiệm trên corpus có nhiễu C200 (gold 18 + nhiễu 182 PDF, ~3780 trang, 20.8k chunk pymupdf)
- Thêm thước đo `in_ctx@Bk` = quote evidence nằm trọn trong B token đầu của context (bộ ghép kiểu D1, token Olmo).
- Kết quả (in_ctx@2k / @5k / @10k / MRR): c1 dense MiniLM .35/.45/.55/.308 · c3 hybrid .45/.55/.70/.284 · c11 hybrid+tiêu đề .55/.55/.70/.304 ·
  c13 +title_rrf .50/.55/.65/.272 · **c14 c11+chunk 1200/200 .50/.65/.75/.356** · c15 chunk 1600/250 .50/.60/.75/.361 ·
  c16 chunk 400/80 .35/.40/.50/.118 · c7 bge-small+tiêu đề .55/.65/.70/.332.
- Nhận xét — vì sao: có nhiễu, hybrid+tiêu đề vẫn đưa evidence vào context 10k ở 70% câu (dense 55%) → hướng D3 đúng. Chunk lớn hơn (1200)
  giữ đoạn bằng chứng liền mạch → in_ctx@5k +0.10. Định tuyến theo tiêu đề (RRF list 3) và chunk nhỏ làm tệ → loại.
  Còn ~25% câu evidence không vào được 10k token → giới hạn trên của pipeline hiện tại. Docling trên C200 (c0/c9/c18) đang chạy.

### exp-003c — Docling trên C200 (cấu hình thật của D3), 20 câu có quote
- in_ctx@2k/@5k/@10k / MRR: c0 D1 dense .50/.60/.60/.301 · **c9 D3 .60/.80/.85/.407** · c18 c9+chunk 1200/200 .55/.80/.85/.333 ·
  c19 pymupdf 1200/200 + bge-small .45/.65/.75/.327.
- Vì sao: Docling tách đoạn sạch hơn PyMuPDF → c9 Docling vượt mọi cấu hình PyMuPDF (.85 vs .75 @10k). Chunk 1200 không thêm gì trên Docling
  (MRR giảm) → bỏ ý D5 "chunk 1200". Đổi sang PyMuPDF mất ~1/10 câu evidence → chỉ đáng làm nếu thiếu thời gian, không vì điểm.
- Giới hạn trên còn lại: 15% câu evidence không vào context 10k; 40% câu evidence không nằm trong 2k token đầu (vùng model bám nhiều nhất, xem D2).

### D5 — `submissions/d5_gte_rerank`
- Thay đổi so với D3: `Retriever(..., rerank_model=rerank_model/, rerank_n=100, rerank_device="cuda")`; reranker lỗi khi load → thứ tự hybrid D3;
  lỗi khi chấm → tắt rerank cho các câu sau. Vá `dsqa/retrieve.py`: rerank giữ phần đuôi sau top-N (trước đây bị cắt còn N kết quả)
  và chấm trên chunk kèm tiêu đề.
- Build: zip 674 MB, giải nén 737 MB (< 1 GB). Smoke (bản giải nén, offline, corpus gold): 2633 chunk = D3, CrossEncoder trên cuda,
  hit_rank khớp lab c29 20/20, MRR .584. Không chạy được LLM local (GPU bị dự án khác chiếm 13.4/16 GB) — phần sinh giống hệt D3.
- Thời gian dự kiến: D3 2088 s + ~1 s/câu rerank trên H100.

### D3 — `submissions/d3_hybrid_title`
- D1 + đúng khối retrieval c9, dùng nguyên `dsqa/` (bundle trong zip). Bản giải nén zip: 2633 chunk và doc_rank khớp lab c9.
- Build: `submissions/d3_hybrid_title.zip` 399 MB (giải nén 435 MB). Không thêm model (MiniLM sẵn có, BM25 = rank_bm25 trên server).
- Smoke (bản giải nén zip, 7B 4-bit, offline, corpus gold 18 PDF): 2633 chunk, 2/2 câu có context trả lời, 0 lỗi, VRAM 11 GB.
- Giả thuyết: top fragment chính xác hơn → +≥ 2.86 trên LB so với D1 (57.14). Bác bỏ nếu ≤ 57.14 + 1.43.

### D1 — `submissions/d1_02_safe` (bài nộp đối chứng)
- Build: `.venv\Scripts\python submissions/d1_02_safe/build.py` → `submissions/d1_02_safe.zip` 399 MB, giải nén 435 MB
  (heron + heron-onnx layout 2×171 MB, MiniLM 91 MB), `main.py` ở gốc, không có `eval/`/`dev_shim`.
- Smoke test (từ bản giải nén zip, 7B 4-bit, offline, corpus 2 PDF arXiv có thư mục con + đuôi `.PDF`): index 177 chunk, 3/3 câu trả str, 0 lỗi,
  VRAM đỉnh 11 GB. Thư mục rỗng → index None, không crash (worker kiểm). Thời gian local không đại diện server.
- Kỳ vọng LB: ≈ 57.14 (cùng prompt/retrieval/generation với 02_rag trên corpus phẳng). Kết quả: <chờ nộp>.

### D2 — `submissions/d2_evidence_last` (evidence-last)
- Giả thuyết: Olmo-3.1-32B có 48/64 lớp sliding-window 4096; 02_rag đặt fragment hạng 1 ở ĐẦU context ~10k → 48 lớp không thấy khi sinh.
  Đặt fragment hạng 1 sát câu hỏi → model dùng evidence tốt hơn. Bác bỏ nếu LB(D2) − LB(D1) ≤ −max(2.86, |D1 − 57.14|).
- Thay đổi: đúng 1 dòng `selected.reverse()` trong `_build_context` (cùng tập fragment, cùng ngân sách). Worker kiểm bằng CPU: D2 = đảo ngược D1.
- Build: `submissions/d2_evidence_last.zip` 399 MB (giải nén 435 MB). Smoke (bản giải nén, 7B 4-bit, offline, 2 PDF): 2/2 câu OK, 0 lỗi;
  khác chữ D1 2/2, token trung bình 348 vs 435 (local 7B, không kết luận).
- **Kết quả LB (30/10): 49.05 vs D1 57.14 → −8.09 (vượt ngưỡng 2.86) → bác bỏ.** Đặt fragment hạng thấp ở đầu context và hạng 1 sát câu hỏi
  làm 32B trả lời kém hơn: model bám vào phần ĐẦU context (primacy) mạnh hơn hiệu ứng sliding-window. Hệ quả cho các bước sau:
  giữ thứ tự hạng giảm dần; tăng độ chính xác của vài fragment đầu (hybrid/rerank) và bớt fragment nhiễu là hướng đáng thử.
