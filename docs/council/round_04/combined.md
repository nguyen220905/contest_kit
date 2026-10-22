Hội đồng model — round_04 — 2026-10-22 12:24 — 1 vòng

> Các model KHÔNG thấy code, chỉ thấy brief. Đồng thuận là tín hiệu, không phải bằng chứng — kiểm chứng với code/log thật trước khi đưa vào PLAN.

## Thành viên
- M1 = deepseek (deepseek-v4-pro) — vai: red_team

## Câu trả lời vòng cuối (vòng 1)

---

# M1 = deepseek (deepseek-v4-pro) — vai: red_team

## 1. Chẩn đoán nhanh
Plan có cấu trúc tốt nhưng khâu eval chưa đáng tin vì cùng 25 câu mẫu bị dùng làm ba việc: hiệu chỉnh judge, few-shot sinh câu tổng hợp, và holdout cuối. Judge tự chế J1/J2 chưa được chứng minh tương quan với judge ẩn, trong khi brief đã dẫn chứng synthetic kém khi so generator. Ngoài ra chưa thấy kế hoạch đóng gói model/embedder trong 1 GB và kiểm tra môi trường no-internet/Linux — rủi ro fail trắng lần nộp đầu.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)

### Đề xuất A: Tách 25 câu mẫu: không dùng cùng tập cho hiệu chỉnh judge, few-shot sinh câu và holdout cuối
- Giả thuyết: Việc tái dùng 25 câu gốc làm calibrate set, few-shot sinh câu tổng hợp và đồng thời là sample25/holdout cuối sẽ làm nhiễm holdout, khiến mọi lựa chọn P4–P6 bị chọn trên tập đã thấy, local metrics không phản ánh server.
- Dựa trên bằng chứng nào trong brief: S2 dùng 25 câu mẫu + biến thể để hiệu chỉnh J1/J2; S5 dùng 25 câu mẫu làm few-shot sinh câu; S6/S1 chấm sample25 như holdout. Ba lần tái sử dụng cùng 25 câu, đồng thời sample25 được gọi là "holdout cuối, không tune".
- Cách làm cụ thể: Chia 25 câu gốc thành 2 phần ngẫu nhiên theo tài liệu (seed 2026): `calibrate_set` 10 câu (dùng cho S2 và few-shot S5) và `true_holdout` 15 câu (không đụng vào bất kỳ hiệu chỉnh/few-shot nào). Các biến thể dùng hiệu chỉnh judge phải sinh từ `calibrate_set`. Nếu cần ≥25 mẫu để hiệu chỉnh judge, bổ sung câu thật viết tay từ tài liệu tìm được ở S3, không lấy từ `true_holdout`. Mọi so sánh generator về sau dùng `true_holdout` riêng biệt và báo cáo riêng.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Chạy 02_rag trên `true_holdout` lần 2; nếu số câu lật >1/15 (≈6.7%) thì true_holdout quá nhỏ/nhiễu. Nếu J1–J2 đạt ≥90% đồng ý trên `calibrate_set` nhưng không có dữ liệu để kiểm trên `true_holdout` vì không có gold, phải coi judge chưa hiệu chỉnh. Bác bỏ nếu không tách được vì quá ít câu; khi đó phải ghi rõ "holdout bị nhiễm" và không dùng để quyết định generator.
- Kỳ vọng: 0 điểm LB trực tiếp; ngăn mất ≥5 điểm do overfit holdout | Tự tin: cao | Chi phí: S

### Đề xuất B: Không dùng J1/J2 làm trục chính cho thay đổi generator; chỉ tin synthetic cho retrieval
- Giả thuyết: LLM judge tự chế không phải judge ẩn; các tài liệu brief đã dẫn (van Elburg: generator τ≈0/âm; Thakur: judge dễ dãi, ưa câu dài/mơ hồ) cho thấy dùng số local judge để A/B prompt/format/PoT sẽ chọn thay đổi cải thiện judge tự chế nhưng không cải thiện điểm ẩn.
- Dựa trên bằng chứng nào trong brief: Brief ghi van Elburg "sai khi so generator/LLM (τ≈0 hoặc âm)"; Thakur "judge dễ dãi, ưa câu dài/mơ hồ"; tuy S2 nói "chỉ để so sánh tương đối" nhưng P4–P6 dùng để chọn format/PoT/CI.
- Cách làm cụ thể: Giới hạn synthetic evaluation chỉ dùng cho retrieval (doc-hit, index time, coverage). Cho generator, dùng 15 câu `true_holdout` (sau khi tách ở Đề xuất A) với exact-match hoặc judge người; nếu dùng J1/J2 thì chỉ báo cáo, không lấy làm tiêu chí dừng. Các thay đổi generator phải được kiểm bằng nộp thật LB (1 bài/ngày) hoặc bằng proxy có tương quan đã đo với LB: chạy 01/02 32B trên H100 và đối chiếu điểm LB 29.05/57.14 trước khi tin.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Khi có ≥3 bài nộp LB, tính Spearman giữa J1/J2 và LB; nếu |Spearman| < 0.5 thì bác bỏ việc dùng J1/J2 cho generator. Trên `true_holdout`, nếu một thay đổi cải thiện điểm J1/J2 nhưng không cải thiện exact-match/format/time, coi như không có bằng chứng. Đo trung bình token câu trả lời J1/J2 cho; nếu J1/J2 ưu tiên câu dài hơn reference_short, bác bỏ rubric hiện tại.
- Kỳ vọng: 0 điểm LB trực tiếp; tránh chọn nhầm generator | Tự tin: cao | Chi phí: M (cần H100 hoặc thêm lần nộp)

### Đề xuất C: Kiểm tra đóng gói no-internet + Linux cho 02_rag và toàn bộ model/embedder trước khi tin bất kỳ số local nào
- Giả thuyết: Server không internet và chạy Linux; nếu 02_rag cần tải Docling/MiniLM/tokenizer lúc chạy, hoặc code có lỗi `glob("*.pdf")` không đệ quy, hoặc phụ thuộc Windows, bài nộp #1 sẽ fail/timeout, khiến toàn bộ eval local đo trên môi trường khác trở nên vô nghĩa.
- Dựa trên bằng chứng nào trong brief: Brief nói "Không internet", "Server 1×H100 ... 32 CPU", "02_rag cần tải model (bản Windows)" là rủi ro S1, "02_rag dễ vỡ: glob('*.pdf') không đệ quy, answer() không try/except", zip ≤1 GB chưa có kế hoạch bundle model.
- Cách làm cụ thể: Tạo một bản zip thử nghiệm đúng cấu trúc nộp (main.py ở gốc), chạy trong container Linux (Docker/WSL2) với venv torch/transformers như server, không mạng, corpus 10 PDF nhiễu. Chạy `--budget-s 3600` cho 50 câu giả để đo thời gian và lỗi import. Kiểm tra corpus_dir có thư mục con, PDF hỏng, tên file Unicode; parse đệ quy bằng PyMuPDF. Đo dung lượng zip sau nén; nếu >1 GB thì phải cắt MiniLM/docling hoặc chọn mô hình nhẹ hơn.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu `python main.py` trong container không thể import/load model mà không có mạng → lỗi đóng gói. Nếu thời gian 50 câu vượt 3600s trong container → cần tối ưu. Nếu zip >1 GB → phải cắt. Nếu 02_rag không tìm thấy PDF trong thư mục con → đúng bug `glob`. Bác bỏ nếu mọi thứ chạy dưới 1h và zip ≤1 GB; khi đó rủi ro môi trường được loại trừ.
- Kỳ vọng: 0 điểm LB trực tiếp; tránh mất trắng lần nộp | Tự tin: vừa/cao | Chi phí: S/M (cần Docker/WSL hoặc thuê H100 sớm)

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Chưa thấy kế hoạch đóng gói model/embedder/tokenizer trong 1 GB và test no-internet, Linux; rủi ro fail trắng ở lần nộp đầu vẫn còn nguyên.
- Không rõ `sample25`/holdout có phải là 25 câu mẫu gốc không, và chúng có bị dùng cho hiệu chỉnh judge và few-shot sinh câu hay không; cần manifest chi tiết.
- Không có cách đo tương quan giữa J1/J2 và LB; vì server không trả output, không thể hiệu chỉnh judge nếu không nộp thật hoặc chạy đúng 32B.
- Chưa thấy cách xử lý câu hỏi cần kiến thức ngoài corpus (ln2, khối lượng riêng Li) và câu tiền đề sai trong synthetic; nếu synthetic thiếu các loại này thì không đại diện.
- Thời gian 5–6 ngày có thể không đủ vì S3 tìm tài liệu DONES công khai có thể thất bại, S5 sinh/kiểm 120 câu + người duyệt, S6 chạy baseline nhiều lần trên máy 7B 4-bit tốn nhiều giờ.

## 4. Nguồn đã tra cứu
- Không tra cứu web trong vòng này.

_Nguồn tra cứu: 0 link từ 0 lượt search — xem sources.md_
