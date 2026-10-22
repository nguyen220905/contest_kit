# M4 = deepseek (deepseek-v4-pro) — vai: root_cause — vòng 1

## 1. Chẩn đoán nhanh
Tôi chưa có prediction local nên chưa thể chốt nguyên nhân; với LB 57.14 của 02_rag, tôi đặt 5 giả thuyết cạnh tranh: (1) retrieval đưa sai chunk/tài liệu do chunk 800 ký tự, ctx 10k token, MiniLM không domain, không reranker; (2) Docling làm mất evidence ở bảng/công thức nên retrieval mù; (3) generator sinh quá dài/không concise, lệch giọng judge ẩn; (4) một nhóm câu cần kiến thức ngoài PDF hoặc bác tiền đề sai nên RAG bị trần; (5) judge/reference_short chưa khớp phong cách ẩn làm tín hiệu local sai. Thí nghiệm rẻ nhất để phân biệt là oracle-chunk (H1 vs H3/H4) và A/B độ dài lời giải (H3).

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)

### Đề xuất A: Oracle-chunk trên sample25 để tách retrieval khỏi generator
- Giả thuyết: Retrieval (đặc biệt chunking/embedding) là nguyên nhân chính khiến 02_rag đứng ở 57; nếu đưa đúng đoạn evidence, accuracy sẽ tăng rõ.
- Dựa trên bằng chứng nào trong brief: S3b chỉ mới phân loại lỗi 4 nhóm, chưa chạy context vàng; 02_rag dùng chunk 800 ký tự, ctx 10k, không reranker; sample25 có trường `evidence[{doc_id,page,quote}]` dự kiến.
- Cách làm cụ thể: Sau khi S3 lấy được ≥ 20/25 tài liệu gold + quote, chọn 10–12 câu sample25 có evidence rõ. Với mỗi câu, chạy 7B 4-bit greedy với hai loại context: (1) chunk 02_rag trả về, (2) đúng chunk chứa `quote` (oracle). Giữ nguyên prompt; thử cả `max_new_tokens=2048` và `640`. Chấm bằng J1 đã hiệu chỉnh; phân loại lỗi theo 4 nhóm + tách `false_premise`, `calc`.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu oracle-chunk accuracy > retrieval accuracy ≥ 15 điểm phần trăm trên cùng tập con, đồng thời lỗi “đúng tài liệu sai đoạn” chiếm ≥ 50% lỗi retrieval → retrieval là nút thắt. Nếu chênh ≤ 5 điểm hoặc lỗi chủ yếu “đúng đoạn sinh sai” → bác bỏ H1, chuyển sang prompt/model. Nếu oracle-chunk vẫn thấp ở câu `false_premise`/`calc` → ghi nhận H4 (coverage/kiến thức ngoài tài liệu).
- Kỳ vọng: +10–20 điểm accuracy sample25 nếu H1 đúng; gần như 0 nếu sai. | Tự tin: vừa | Chi phí: M

### Đề xuất B: Đối chiếu Docling vs PyMuPDF bằng quote vàng
- Giả thuyết: Docling đang làm mất hoặc bóp méo text quan trọng (bảng, công thức, chú thích) khiến retrieval không thể tìm thấy evidence; chuyển parser sớm sẽ gỡ trần.
- Dựa trên bằng chứng nào trong brief: 02_rag dùng Docling GPU-auto; issue Docling bị nghẽn CPU/pipeline trên GPU; server thiếu cv2 nên table structure có thể lỗi; S4 có tiêu chí PyMuPDF nhưng chưa đo text coverage của Docling; manifest gold có `quote`.
- Cách làm cụ thể: Với 20–25 tài liệu gold lấy được, trích text thô bằng PyMuPDF và Docling (local Windows, CPU đủ cho 20–25 PDF). Với mỗi `quote`, kiểm tra fuzzy containment ≥ 0.9 trong văn bản trích. Thống kê tỉ lệ quote tìm thấy của mỗi parser, số trang có text, % PDF không text, số bảng/công thức bị mất (soi thủ công 5 trang). Nếu Docling tệ hơn hẳn, lặp lại với 5 tài liệu nhiễu từ C50.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu Docling tìm thấy < 70% quote trong khi PyMuPDF > 90% trên cùng bộ → parser là nút thắt, P4 nên ưu tiên PyMuPDF. Nếu cả hai ≥ 90% và không mất bảng/công thức quan trọng → bác bỏ H2, tập trung embedding/chunk. Nếu PyMuPDF cũng thấp → tài liệu sai/bản scan.
- Kỳ vọng: Không trực tiếp + điểm; loại/giữ H2 và chốt parser cho P4. | Tự tin: vừa | Chi phí: S/M (chỉ 20–25 PDF, không cần LLM)

### Đề xuất C: A/B max_new_tokens + answer-first trên sample25
- Giả thuyết: Generator/phong cách trả lời là nguyên nhân: `max_new_tokens=2048` khiến câu bị loãng, lệch tiêu chí “deliberately concise”, judge ẩn bác; giảm token và cấu trúc câu trả lời sẽ cải thiện.
- Dựa trên bằng chứng nào trong brief: Smoke test 7B chạm trần 256 token và sinh dài; 02_rag greedy với `max_new_tokens=2048`; hidden reference “deliberately concise”; P4 dự kiến `max_new_tokens ~640` + answer-first nhưng chưa chạy.
- Cách làm cụ thể: Sau khi J1 qua hiệu chỉnh, chạy 02_rag 7B 4-bit trên sample25 (hoặc 15 câu đã có retrieval) với ba biến: (i) giữ nguyên 02_rag; (ii) chỉ giảm `max_new_tokens` xuống 640; (iii) giảm 640 + prompt yêu cầu trả lời một đoạn, nêu số/sự kiện trước, không lặp câu hỏi. Ghi token sinh p50/p95, J1 verdict; so sánh accuracy và % câu bị cắt ngang.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu biến (iii) tăng ≥ 8 điểm accuracy so với (i) trên cùng câu, đồng thời token trung vị giảm ≥ 40% → phong cách/độ dài là nút thắt. Nếu cả ba biến chênh ≤ 2 điểm → bác bỏ H3, dồn chi phí cho retrieval/parser. Lưu ý: kết quả 7B chỉ định hướng; trước khi đổi prompt phải xác nhận bằng 32B trên H100, vì 7B có thể không đại diện.
- Kỳ vọng: +5–10 điểm sample25 nếu H3 đúng; gần như 0 nếu sai. | Tự tin: thấp-vừa | Chi phí: S (3×25 câu 7B 4-bit, ước 2–4 giờ GPU local)

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Chưa có prediction local, chưa đo nhiễu, chưa biết phân bố lỗi thật; mọi chẩn đoán ở trên là giả thuyết cần falsify bằng J1 và sample25.
- Không rõ corpus public thật có phẳng hay lồng nhau; brief nói nếu không phẳng thì 02_rag ≈ zero-shot, nhưng 57.14 > 29.05 nên cần xác minh `glob("*.pdf")` có đúng không.
- Chưa có J1 hiệu chỉnh; mọi thí nghiệm dùng J1 phải chờ S2. Hơn nữa judge ẩn tất định không trả output, nên việc lặp nộp cùng zip để đo dao động generation rất tốn lượt (1 bài/ngày) nhưng cần làm sớm.
- 7B 4-bit có thể lệch nhiều so với 32B bf16, đặc biệt ở prompt/format; các kết luận từ 7B cần xác nhận bằng 32B trước khi dùng cho LB.
- Thiếu số đo thời gian decode 32B bf16 trên H100 với ctx 10k token, nên nguy cơ P4 vượt ngưỡng 1500 s chưa được đánh giá.

## 4. Nguồn đã tra cứu
- Không tra cứu nguồn mới; dùng các tài liệu đã dẫn trong brief: van Elburg 2025 (synthetic xếp hạng retriever nhưng không generator), Krumdick 2025 (reference đúng nâng kappa judge), Thakur 2024 (judge dễ dãi, cần kappa). — không có URL mới.
