# M1 = deepseek (deepseek-v4-pro) — vai: root_cause — vòng 1

## 1. Chẩn đoán nhanh
Điểm 57.14 mới là sàn của 02_rag, chưa đủ dữ liệu để gọi là plateau thật; nhưng nếu không cải được, cạnh tranh nhất là 4 nhóm sau. (1) Retrieval recall thấp: Docling+MiniLM CPU, `glob("*.pdf")` không đệ quy, ctx 10k làm trôi bằng chứng. (2) Generator lệch judge: `max_new_tokens=2048`, greedy, không answer-first nên câu trả lời dài/mâu thuẫn, trong khi judge nhị phân ưa “concise”. (3) Có câu vốn không thể trả lời từ corpus vì cần kiến thức ngoài/tiền đề sai (q7, ln2, Li density), 02_rag không có cơ chế bác tiền đề. (4) Lệch domain: public 15 DONES/35 AI, mỗi câu DONES nặng 3.33 điểm; nếu DONES thiếu nguồn hoặc retrieval kém thì điểm bị chặn.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)

### Đề xuất A: Phân loại lỗi 02_rag trên 25 câu mẫu với corpus vàng/C50 + hook retrieval
- Giả thuyết: Điểm kẹt do retrieval (H1), generator/prompt (H2), hay do câu hỏi không thể trả lời từ tài liệu (H3).
- Dựa trên bằng chứng nào trong brief: Chưa có predictions; 02_rag public 57.14; dev có 25 câu gold với `question_type`, `flags{calc,false_premise,closed_book_easy}`; hook `retrieve()` đã được plan để đo doc-hit; local 7B 4-bit đủ chạy ~30–60 s/câu.
- Cách làm cụ thể:
  1. Ngay sau S3 có ≥20 PDF nguồn, dựng corpus chỉ chứa tài liệu gold (hoặc C50 nếu S4 xong) — không cần chờ C200.
  2. Chạy `02_rag` qua harness với `DSQA_DEV_QUANT=4bit`, ghi: `retrieve` top-k/doc-hit@5, câu trả lời, độ dài token, thời gian.
  3. Dùng J1 đã hiệu chỉnh sơ bộ chấm từng câu; người phân loại lỗi thành 4 nhóm: retrieval miss, retrieval hit nhưng sai/verbose, judge lệch, câu cần ngoài/false-premise.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ:
  - Nếu H1 đúng: ≥60% câu sai có doc-hit@5 = 0 hoặc quote không khớp, và doc-hit@5 toàn mẫu <70%.
  - Nếu H2 đúng: doc-hit@5 ≥70% nhưng ≥60% câu sai rơi vào nhóm “retrieval hit, answer sai”, median answer length >300 token.
  - Nếu H3 đúng: ≥4/25 câu sai thuộc nhóm cần ngoài/false premise dù retrieval hit.
- Kỳ vọng: không trực tiếp + điểm LB; chọn đúng hướng P4 thay vì thử mù 3 hướng. Nếu H1 đúng, P4 có thể +6–12 điểm public nhờ sửa retrieval; nếu H2, +4–10 nhờ prompt/rút sinh; nếu H3, cần module bác tiền đề/kiến thức ngoài. | Tự tin: cao | Chi phí: M (sau S3; 1–2 h chạy 7B + phân loại)

### Đề xuất B: A/B answer-first + giảm `max_new_tokens` từ 2048 xuống 512–640
- Giả thuyết: Một phần lớn điểm mất do generator sinh dài, lạc đề, không trả lời thẳng; judge ẩn “deliberately concise” phạt.
- Dựa trên bằng chứng nào trong brief: 02_rag dùng `max_new_tokens=2048` greedy; gold trung vị ~670 ký tự nhưng tham chiếu ẩn concise; top public 69.05 so với 57.14 có thể chỉ nhờ định dạng câu trả lời.
- Cách làm cụ thể:
  1. Dùng đúng corpus/retrieval của 02_rag, chỉ monkey-patch prompt và `max_new_tokens`.
  2. Biến A: giữ prompt gốc, `max_new_tokens=2048`.
  3. Biến B: thêm “Answer in one short sentence. If the premise is false, say that first. Do not explain unless asked.”, `max_new_tokens=512` (nếu câu nào answer bị cụt, thử 640).
  4. Chấm J1 đã hiệu chỉnh; đo median length token, accuracy 25 câu.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu B ≥ A + 10 điểm accuracy và median length giảm ≥30%, H2 được xác nhận. Nếu B không hơn A ≥3 điểm hoặc làm sai các câu tính toán/cần số, bác bỏ H2.
- Kỳ vọng: nếu đúng, P4 được thiết lập sẵn prompt + độ dài; có thể +4–10 điểm LB. | Tự tin: vừa | Chi phí: S (2 lần chạy 25 câu, <1 h 7B, chạy song song với A)

### Đề xuất C: Đối chiếu 01_zero_shot và 02_rag theo domain/question_type trên 25 câu mẫu
- Giả thuyết: Điểm public 57.14 là trung bình cộng domain bị kéo xuống bởi DONES (retrieval nguồn kém hoặc câu hỏi ngoài tài liệu), trong khi AI có thể đã khá.
- Dựa trên bằng chứng nào trong brief: Public LB chia 15/35, DONES mỗi câu 3.33 điểm, AI 1.43; dev có 10 DONES/15 AI; không có per-domain LB; bài báo AI dễ truy hồi theo tên paper, DONES nguồn có thể không công khai.
- Cách làm cụ thể:
  1. Chạy `01_zero_shot` và `02_rag` trên 25 câu mẫu, cùng harness/J1.
  2. Ghi accuracy theo `domain` và `question_type`; với 02_rag, ghi doc-hit@5 theo domain.
  3. Kiểm tra: ACC(DONES) so với ACC(AI); 02_rag có hơn zero-shot ở DONES không.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu ACC(DONES) ≤40% trong khi ACC(AI) ≥70%, hoặc 02_rag không hơn zero-shot ở DONES nhưng hơn rõ ở AI, kết luận domain bottleneck (H3/H1 theo domain). Nếu chênh lệch domain <15 điểm và doc-hit hai domain tương đương, bác bỏ H3, chuyển nguồn lực sang generator/retrieval chung.
- Kỳ vọng: nếu đúng, P4 ưu tiên xử lý DONES trước, có thể +6–10 điểm public với ít thay đổi AI. | Tự tin: vừa | Chi phí: M (tái dùng lần chạy ở A, thêm 01; <1 h 7B)

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Không có per-domain/public per-question outputs nên không biết 57.14 do DONES hay AI; đây là lỗ hổng thông tin lớn nhất.
- Không có mô tả chính xác prompt 02_rag, thứ tự chunk, số lượng chunk, cách đưa context; khó mô phỏng đúng H2.
- Không có PDF nguồn cho 25 câu mẫu, nên chưa thể chạy A/C ngay; đang phụ thuộc S3.
- Nguy cơ rò rỉ: 25 câu mẫu vừa dùng làm few-shot sinh câu vừa là holdout cuối; cần tách câu mẫu ra khỏi holdout trước khi sinh (nếu chưa sửa plan).
- Chưa rõ độ tin cậy ban đầu của J1/J2 trước khi hiệu chỉnh; nếu chưa hiệu chỉnh xong thì các thí nghiệm phân loại có thể nhiễu.

## 4. Nguồn đã tra cứu
- không tra cứu
