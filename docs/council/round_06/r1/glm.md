# M8 = glm (z-ai/glm-5.3) — vai: simplifier — vòng 1

## 1. Chẩn đoán nhanh
Kết quả đang bằng 0 vì 5–6 ngày tới bị tiêu vào xây cơ sở hạ tầng eval, trong khi khoản cách tới top-20 chỉ **1.91 điểm = +2 câu AI (mỗi câu 1.43) hoặc +1 câu DONES (3.33)**, và LB thực chất là một eval harness miễn phí: 50 câu thật, 1 lượt/ngày, nộp kém không bị phạt. Các quyết định sinh điểm thực sự (sống sót trên corpus thật, độ dài câu trả lời vs judge "concise", ngân sách 1h) đều không cần judge 7-test + 60–80 câu synthetic; chính brief trích van Elburg: synthetic xếp hạng generator τ≈0. Plan v2: **7.5/10 → "sửa nhỏ rồi làm"** — giữ S1+S3+S3b, thu hẹp S2, cắt S5 khỏi đường tới nộp bài.

## 2. Đề xuất

### Đề xuất A: Cắt S5 khỏi critical path; S2 rút còn 3 test; LB đóng vai trò holdout
- **Giả thuyết:** Không quyết định nào trước ~05/11 phụ thuộc vào câu synthetic/human; chất lượng retrieval đo trực tiếp bằng doc-hit@k trên tài liệu gold của sample25 + nhiễu thô.
- **Bằng chứng trong brief:** van Elburg (arXiv 2508.11758) — synthetic xếp hạng retriever ổn, generator thì không; quy tắc dùng split của chính plan cũng thừa nhận generator phải qua sample25 + LB; 15–20 câu human còn thiếu người phụ trách; 39 lượt nộp vs 5–6 ngày infra.
- **Cách làm cụ thể:** Giữ S1 (thêm: chạy trong WSL2/docker Linux vì server là Linux, không phải Windows) → S3 (≥20/25 tài liệu gold) → S3b (01 vs 02 + doc-hit@k + bảng 4 nhóm lỗi) → S2 chỉ còn 3 test: paraphrase→đúng ≥24/25, đổi số/thực thể→bác ≥90%, lạc đề→bác 25/25. S4 nhiễu = chính các PDF đã tải (~C100), bỏ thiết kế arXiv-dupes. S5, J2/kappa, test long/short reference: chuyển sang "làm sau khi có ≥3 bài nộp". Cache parse/embedding theo hash corpus để vòng lặp local không parse lại.
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Chạy S3b: nếu doc-hit@5 < 60% → retrieval cần S4 đầy đủ (giữ S4, vẫn cắt S5); nếu 02 ≤ 01 trên sample25 → pipeline hỏng ở thượng nguồn, mọi eval sau vô nghĩa.
- **Kỳ vọng:** nộp bài thật từ ~24–25/10 thay vì 28/10 (+4–5 slot thông tin) | **Tự tin: cao** | **Chi phí: S**

### Đề xuất B: 3 bài nộp đầu = chuỗi A/B một-nút-knob (trả lời câu hỏi 2)
- **Giả thuyết:** LB 50 câu thật mạnh hơn mọi split synthetic cho đúng loại quyết định cần; mỗi bài nộp 1 biến thể đơn lẻ → delta khả quy.
- **Bằng chứng:** 02_rag đã chạy ổn trên server (2080 s < 3600 s); rủi ro đã biết của 02 (glob không đệ quy, không try/except); smoke test 7B chạm trần 256 token → sinh dài; judge tham chiếu "deliberately concise".
- **Cách làm cụ thể:** **Ngày 1** = 02_rag + chỉ vá an toàn (glob đệ quy, try/except từng câu, ép UTF-8, watchdog mỗi câu) → kỳ vọng ≈57.14, đo dao động tất định. **Ngày 2** = + prompt answer-first + max_new_tokens ~512–640 (kiểm qua sample25 trước theo đúng quy tắc split của plan). **Ngày 3** = thay Docling bằng PyMuPDF đa tiến trình (issue #1364: A100 util 0–20%, chỉ ~2× CPU → "GPU-auto" không đáng giá).
- **Tiêu chí bác bỏ:** Ngày 2 giảm ≥ ~4 điểm vs Ngày 1 → judge ưa câu dài, hoàn tác cap; Ngày 3 giảm rõ → Docling có giá trị layout, giữ.
- **Kỳ vọng:** **+3.3 đến +7 điểm (1–2 câu AI)** từ nút độ dài/format; kèm tiết kiệm ~1000–1500 token sinh/câu cho ngân sách 1h | **Tự tin: vừa** | **Chi phí: S**

### Đề xuất C: Duyệt thuê A100 80GB 2–3h ngay sau S1, một kịch bản đo cố định (trả lời câu hỏi 4)
- **Giả thuyết:** Ẩn số lớn nhất (32B bf16 prefill/decode tại ctx 2k–12k, OOM margin, Docling thật, tính tất định) đo được trong 1 buổi; quy đổi A100→H100 bằng tỉ lệ băng thông HBM (~1.6× decode bs=1, ~3× prefill) — số NVIDIA do brief tự trích, tôi không tự kiểm chứng được.
- **Bằng chứng:** 7B 4-bit local không suy ra 32B (GQA/sliding-window khác); chưa test được 32B bf16 trước khi nộp = rủi ro cháy slot.
- **Cách làm cụ thể:** Script 5 phép đo: (1) microbench 32B bf16 sdpa, ctx {2k,5k,10k,12k}, prefill p50/p95 + decode tok/s; (2) chạy 02_rag trên C100 hai lần → đếm câu lật (= đo dao động); (3) tổng thời gian × hệ số 1.6 so với LB 2080 s → ước lượng kích thước corpus thật; (4) Docling vs PyMuPDF trang/s; (5) sanity chat template 32B.
- **Tiêu chí bác bỏ:** Nếu 50 câu × (12k prefill + 512 gen) ≫ 3600 s trên A100×1.6 → phải hạ ctx (~5k, hợp thức hoá P4); nếu 02_rag A100 quy đổi lệch > 25% so 2080 s → giả định corpus sai → chỉnh lại quy mô S4.
- **Kỳ vọng:** 0 điểm trực tiếp, chặn cả nhánh thiết kế sai trước 30/11 | **Tự tin: cao** | **Chi phí: S**

## 3. Điểm đáng ngờ / thông tin còn thiếu
- **Kết quả nộp lại 01/02 nguyên bản không được báo:** nếu điểm lặp lại chính xác → generation tất định trên server → mọi delta LB ≥1 câu là tín hiệu thật, điều kiện tiên quyết để diễn giải chuỗi A/B. Cần số này ngay.
- Gold sample25 (trung vị ~670 ký tự) ≠ tham chiếu ẩn "concise" → mọi ngưỡng tuyệt đối của J1 là proxy-bọc-proxy; chỉ tin so sánh tương đối (plan đã nói, nhưng các ngưỡng "≥24/25" vẫn phụ thuộc rubric tự dựng).
- 25 câu quá ít cho McNemar (sức thống kê gần như bằng 0): quy tắc ra quyết định nên là "≥2–3 câu lật cùng chiều + không tệ hơn ở domain kia", chấp nhận ồn, rồi xác nhận bằng LB.
- Ollama (câu 3): chọn gần như **(e) không dùng cho quyết định** — Q4_K_M (+~0.18 PPL theo số llama.cpp trong brief), temp mặc định 0.6, num_ctx 4096, template khác → quyết định không chuyển dịch sang server HF bf16 greedy. Tôi không kiểm chứng được tag `olmo-3.1:32b` (hết lượt tìm).
- Plan không nêu chạy S1 trên Linux (WSL2/docker) dù thiếu triton trên Windows — khe lệch lớn nhất giữa local và server.

## 4. Nguồn đã tra cứu
- IEEE CIS Contest Rules — https://cis.taskforce.ieee.org/gpais/wp-content/uploads/sites/6/2026/09/IEEE-CIS-Technical-Challenge-on-Domain-Specific-Question-Answering-with-LLMs-Contest-Rules.pdf — xác nhận entry period 28/09–30/11/2026, winners tuần 14/12, Phase 2 do chuyên gia.
- Join the Challenge — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/join-the-challenge/ — ngân giờ bao gồm load model + preprocessing + inference; vượt hạn = chấm dứt.
- van Elburg et al., Can we Evaluate RAGs with Synthetic Data? — https://arxiv.org/abs/2508.11758 — synthetic xếp hạng ổn cấu hình retriever, KHÔNG xếp hạng được generator.
- Docling GPU docs — https://docling-project.github.io/docling/usage/gpu/ — RTX 5090 7.9 trang/s vs CPU-only 1.5; speedup 5–6× trên GPU nhanh.
- Docling issue #1364 — https://github.com/docling-project/docling/issues/1364 — A100 util 0–20%, chỉ ~2× so CPU → Docling chặn bởi CPU.
- Docling discussion #3442 — https://github.com/docling-project/docling/discussions/3442 — batch size/concurrency không đổi throughput, GPU util ≤29%, stage-bound.
- Không tra cứu được (hết hạn mức tìm kiếm): specs Olmo-3.1-32B, số so sánh H100 vs A100, YourBench/Krumdick/Thakur — các mục này dựa vào trích dẫn trong brief.
