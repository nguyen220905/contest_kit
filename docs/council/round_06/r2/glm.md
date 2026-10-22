# M8 = glm (z-ai/glm-5.3) — vai: simplifier — vòng 2

## 1. Nhận xét từng thành viên
- **M1**: Điểm mạnh nhất: đề xuất A soi `reference_short` (proxy-bọc-proxy vì gold dài ≠ tham chiếu ẩn concise) bằng ma trận lật 3 chiều — đúng chỗ rủi ro nhãn, rẻ. | Lỗ hổng: chỉ thêm cổng phân tích, không giải phóng kênh tín hiệu miễn phí đang bỏ trống (LB); đề xuất B đòi "≥8 câu/type" tăng khối lượng synthetic mà chính trích dẫn van Elburg nói không xếp hạng được generator.
- **M2**: Điểm mạnh nhất: A — nộp song song từ S1, mỗi bài 1 knob, đúng với dữ kiện mới "đồng ý nộp song song"; B đúng khi hạ judge xuống công cụ so sánh tương đối. | Lỗ hổng: A đổi knob (max_new 640) trước khi xác minh lại bản build gốc (chưa có điểm tái lập 57.14 → delta không quy được cho knob hay lỗi đóng gói); C thuê A100 "ngay sau S1" hơi sớm.
- **M3**: Điểm mạnh nhất: A — đảo trục dùng LB tất định làm thiết bị đo, đo sàn nhiễu bằng nộp lại bản gốc; ý thức verbosity bias hai chiều. | Lỗ hổng: B (extractive-first) là viết lại toàn bộ pipeline khi chưa có bằng chứng nào rằng generator là nút thắt (thiếu bước oracle); A đốt 3 slot thuần đo trước khi có biến thể an toàn nào.
- **M4**: Điểm mạnh nhất: A oracle-chunk là thí nghiệm triage rẻ–quyết định nhất toàn hội đồng; B so Docling vs PyMuPDF bằng quote vàng cũng rẻ và quyết. | Lỗ hổng: chuỗi phụ thuộc nặng ("sau khi J1 hiệu chỉnh", "sau S3"), mâu thuẫn với thực tế 39 slot LB đang cháy mà chưa có bài; ngưỡng của C vẫn dựa 7B 4-bit.
- **M5**: Điểm mạnh nhất: A quy tắc "biến thể phải thắng cả 2 mốc neo" + đo nhiễu server; B phát hiện metric doc-hit đo sai chỗ (cửa sổ trượt 4096, ctx 10k) — góc nhìn mới thật. | Giả định chưa kiểm chứng: "48/64 lớp SWA của 32B" (brief chỉ xác nhận 24/32 của 7B — cùng hướng nhưng tôi không kiểm được); ngày-1 đã đổi max_new 384 nên mất điểm neo build gốc.
- **M6**: Điểm mạnh nhất: B — danh sách cắt cụ thể nhất (không chạy LLM trên synthetic, judge 2 test, quy tắc net-win ≥ max(2, k_noise+1)), cộng số arity S6 ≈ 460 generation = 4–29h GPU lộ vỡ kế hoạch. | Lỗ hổng: A gộp 2–3 knob/bài nộp (PyMuPDF + max_new 512) → delta không quy được; câu "3600 token kích hoạt đủ 32/32 lớp" sai về kiến trúc (SWA là cố định); hệ số quy đổi 3 chữ số thập phân của C là giả chính xác.
- **M7**: Điểm mạnh nhất: A bộ probe (P, prefill, lattice) phân rã đúng các ẩn số không đo được ở local (P, kích thước corpus, chi phí prefill trên H100 thật); điểm 9 và 6 (kappa paradox) chính xác. | Lỗ hổng: 4 slot probe trước khi có biến thể cải thiện điểm; trộn nguồn đã tra (trang cuộc thi) với nguồn tôi không kiểm được (paper 2512.13961, corpus private=public — đánh dấu chưa kiểm chứng); bộ nghiệm lattice thiếu cặp (3,33).

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)
- **Chấm v2: 7/10 — "sửa nhỏ rồi làm"**, với điều kiện các sửa dưới đây.
- **Giữ** phần tôi cũ: cắt S5 khỏi critical path, S2 gọn còn 3 test. **Sửa**: thay ngưỡng quyết định mơ hồ của tôi ("≥2–3 câu lật") bằng quy tắc của M6-B/M5-A: generator đổi chỉ tin khi net-win ≥ max(2, k_noise+1) trên sample25, rồi xác nhận LB với |Δ| ≥ max(2×nhiễu, 2.86).
- **Bỏ** đề xuất cũ thuê A100 "ngay sau S1": tôi bị M7-th.10 thuyết phục — LB cho sẵn số H100 trên corpus thật, thứ rental không tái tạo được; check template 7B/32B chỉ cần tải file tokenizer vài MB (M7-th.4, chi phí 0). A100 chỉ còn value sau khi có probe.
- **Bỏ** ngày-3 "đổi PyMuPDF" trong chuỗi nộp cũ của tôi: đổi parser là một quyết định kiến trúc, phải qua quote-containment local (M4-B) trước, không đốt slot để thử.
- **Tôi không theo** M3-B (extractive-first) cho tới khi oracle-chunk (M4-A) chứng minh generator là nút thắt — bằng chứng hiện tại chỉ là smoke test 3 câu.
- **Ollama**: giữ quan điểm gần-(e), nhưng chấp nhận 2 dụng cụ hẹp của M6-C/M7-th.11: (c) lọc cờ `closed_book_easy`, (a) smoke raw-mode nếu ép `num_ctx` ≥ prompt thật + temperature 0. Cấm làm judge, cấm proxy 32B.

## 3. Đề xuất cuối (tối đa 3)

### Đề xuất A: Chuỗi 4 slot "đo trước, tối ưu sau" (dựa trên: M7-A, M2-A, M5-A)
- **Giả thuyết:** Ba ẩn số lớn nhất (lỗi đóng gói, cấu trúc corpus, P = load+parse+index) đo được từng cái một bằng slot LB tất định; mỗi bài chỉ 1 thay đổi.
- **Cách làm cụ thể:** Ngày 1: build lại **02_rag nguyên văn** → kỳ vọng ≈57.14/2080 s. Ngày 2: + **chỉ** bản vá an toàn (rglob không phân biệt hoa/thường, try/except từng câu, UTF-8, watchdog từng câu). Ngày 3: **P-probe** — giữ nguyên load/parse/index, `answer()` trả chuỗi hằng (điểm ~0, không bao giờ thành best-public vì 57.14 đã tồn tại → rủi ro private = 0); wall-time = P trên corpus thật → T_private ≈ 2×2080 − P. Ngày 4: 1 knob format (prompt answer-first + max_new_tokens ~512, đã kiểm không thua trên sample25 theo đúng quy tắc của plan).
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Ngày 1 lệch ≥ 2.86 → build/Linux/offline hỏng, dừng mọi biến thể. Ngày 2 giữ ≈57.14 → corpus phẳng; tăng ≥ 3 điểm → corpus lồng nhau (sửa retrieval scope trong P4). Ngày 3: P < 560 s → bản tốt nhất hiện tại **có nguy cơ fail private 100 câu** nếu quy định 1h cứng áp dụng (xung đột M7-th.1 vs brief "2h mềm" — phải đọc lại rule trước ngày 3) → P4 phải cắt ctx/max_new/parser trước khi tối ưu accuracy. Ngày 4 mất ≥ 3 câu AI → judge ưa dài, hoàn tác.
- **Kỳ vọng:** chặn rủi ro private = 0 điểm | **Tự tin: cao** | **Chi phí: S** (4 slot, ~2h code)

### Đề xuất B: Cắt eval + triage bằng oracle-chunk (dựa trên: M6-B, M2-B, M4-A, M7-th.9)
- **Giả thuyết:** S2/S5/S6 hiện có số lượng cổng vượt quá những gì 25 câu + judge proxy hỗ trợ được; synthetic chỉ đủ tin cho retrieval (van Elburg: τ retriever 0.44–0.75, generator ≈0).
- **Cách làm cụ thể:** (i) J1 còn 3 test tự động (paraphrase ≥24/25; đổi số/đơn vị → bác ≥90%; lạc đề → bác 25/25), **bỏ cổng kappa ≥0.8** (M7-th.6: kappa thấp vì cấu trúc biên, không vì judge kém), J2 chỉ là báo cáo; (ii) synthetic **chỉ chạy hook `retrieve()`**: quote-recall@5 + doc-hit@k + thời gian index/retrieve, **không generation LLM**; (iii) LLM chỉ chạy trên sample25: 01 vs 02 + **oracle-chunk** (context thật vs đúng chunk chứa quote) → bảng lỗi 4 nhóm quyết định hướng P4; (iv) cache parse+embedding theo hash corpus; (v) **xóa**: 15–20 câu human (không người phụ trách), C500, bảng tương quan local↔LB (<6 điểm LB không ước lượng được tương quan).
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Oracle − retrieved ≥ 15 điểm % và "đúng tài liệu sai đoạn" ≥ 50% lỗi → ưu tiên retrieval (P4 hybrid BM25+RRF, chọn bằng quote-recall trên tài liệu gold, không tốn LLM). Oracle ≈ retrieved (≤5 điểm) → ưu tiên generator/format. Nếu 02 ≤ 01 trên sample25 → hỏng thượng nguồn, dừng eval.
- **Kỳ vọng:** tiết kiệm ~3 ngày + ~14h GPU; bài thật từ ~24–25/10 | **Tự tin: cao** | **Chi phí: S**

### Đề xuất C: Kiểm parser + cửa sổ 4096 không tốn slot (dựa trên: M4-B, M5-B)
- **Giả thuyết:** Docling có thể làm mất quote (bảng/công thức, `do_table_structure=False` vì thiếu cv2) và ctx 10k đặt evidence ngoài cửa sổ 4096 của ~75% lớp SWA (khớp hướng 24/32 của 7B trong brief; con số 48/64 của M5 chưa kiểm chứng) → doc-hit và chất lượng sinh đều đo lệch.
- **Cách làm cụ thể:** Trên ≥20 tài liệu gold (S3): trích text bằng PyMuPDF và Docling (cùng cờ với 02), đo tỉ lệ quote chứa được (fuzzy ≥0.9) từng bên; với bản sao có instrument provenance của 02 (không phải bài nộp), tính **in_window_hit** = tỉ lệ quote của chunk tốt nhất nằm trong 4096 token cuối ở ctx 10k; ablation 7B: chunk tốt nhất đặt đầu vs cuối.
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Docling ≥90% chứa **và** ≥80% in-window → giữ Docling, vấn đề chỉ còn tốc độ (đã có số từ P-probe). Docling <70% → P4 đổi PyMuPDF (có dữ kiện, không đoán). Chunk-cuối thắng ≥5 câu → P4 phải đảo thứ tự context.
- **Kỳ vọng:** tránh chọn sai parser/thứ tự = ±1–2 câu | **Tự tin: vừa** | **Chi phí: S** (không LLM, không slot)

## 4. Bình chọn
VOTES: M7-A, M4-A, M6-B
