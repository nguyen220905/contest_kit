# M4 = deepseek (deepseek-v4-pro) — vai: root_cause — vòng 2

## 1. Nhận xét từng thành viên
- M1: điểm mạnh — cộng trọng số LB 15/35 và audit `reference_short` trước khi sinh câu tổng hợp là rất cần. | lỗ hổng — vẫn xoay quanh judge local; giả định có thể đo lật `reference_short ↔ concise` bằng J1 trước khi J1 được neo vào LB, nên rủi ro vẫn là proxy-bọc-proxy.
- M2: điểm mạnh — nộp song song + không biến hiệu chỉnh judge thành dự án riêng là đúng. | lỗ hổng — ba nộp đầu chỉ đổi `max_new_tokens`/ctx nhưng mỗi ngày không có đối chứng cùng điều kiện; khó tách nguyên nhân nếu điểm lệch do nhiễu generation.
- M3: điểm mạnh — dùng LB làm dev harness và đo “khẩu vị” độ dài judge ẩn là mũi thí nghiệm sắc nhất. | lỗ hổng — đốt 3 lượt LB ngay khi chưa đo nhiễu nền; extractive-first có thể hại Phase 2 (chất lượng chuyên gia) và câu cần suy luận.
- M5: điểm mạnh — chỉ ra sliding-window 4096 và “evidence phải nằm trong 4096 token cuối” rất cụ thể, bác được doc-hit@k ảo. | lỗ hổng — cấm tuyệt đối 7B 4-bit để chọn generator hơi cứng; ba nộp đề xuất (max 384, nộp lại zip, đảo chunk) chưa đo được format judge.
- M6: điểm mạnh — cắt S5 khỏi critical path, ra ngưỡng paired net-win rõ ràng, ba nộp cụ thể. | lỗ hổng — kỳ vọng +5–9 điểm LB từ ba chỉnh đơn là quá cao; hệ số 0.42 prefill/0.62 decode cho A100→H100 chưa được đo thực nghiệm, chủ yếu lý thuyết.
- M7: điểm mạnh — probe LB P/prefill/DONES-null dùng lattice điểm để tách thời gian và điểm theo domain rất thông minh; cảnh báo 1h vs 2h và private variance quan trọng. | lỗ hổng — bốn probe tốn 4/39 lượt; “raise lỗi cuối để exfiltrate số trang corpus” có thể phạm luật; citation “arXiv 2512.13961” cho kiến trúc Olmo cần đánh dấu chưa kiểm chứng.
- M8: điểm mạnh — cắt S5, giữ S1/S3/S3b, ba bài nộp one-knob, kịch bản A100 cố định. | lỗ hổng — ngưỡng doc-hit@5 < 60% chưa có cơ sở; C100 quá nhỏ để chọn retrieval; chưa tách được length vs parser trong ba nộp.

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)
Tôi giữ ba giả thuyết gốc của mình (retrieval, Docling mất evidence, generator không concise) nhưng **sửa trọng số thời gian**: trước đây tôi đợi J1 hiệu chỉnh rồi chạy local; nay tôi theo M3/M7/M8: **LB là thí nghiệm rẻ nhất và quyết định nhất**, phải có 1–3 nộp sớm để đo length/nhiễu, còn local chỉ để chọn nhanh trước khi nộp.

Tôi **bỏ** việc bắt đầu bằng `A/B max_new_tokens` thuần local (quá phụ thuộc J1); thay bằng cặp nộp LB ngắn-vs-dài. Tôi **thêm** cửa sổ 4096 vào thí nghiệm oracle-chunk của mình, theo M5, vì nếu quote nằm ngoài cửa sổ thì mọi doc-hit@k không có ý nghĩa. Tôi **giảm** kỳ vọng vào J1: dùng J1 ở chế độ “so sánh cùng reference”, không dùng để chốt generator; tham khảo M6/M8 bỏ bớt S5/J2.

## 3. Đề xuất cuối (tối đa 3)
### Đề xuất A: Thí nghiệm local tách ba nguyên nhân bằng “vị trí evidence + oracle chunk” (dựa trên: M4-A, M5-B)
- Giả thuyết: 57.14 do một trong ba: (H1) retrieval đưa sai chunk/tài liệu; (H2) evidence đưa đúng tài liệu nhưng nằm ngoài 4096 token cuối nên 75% lớp sliding-window không thấy; (H3) generator sinh dài/lệch “concise” nên judge bác.
- Cách làm cụ thể: Trên 15–25 câu sample25 đã có `quote`, chạy 7B 4-bit greedy với đúng prompt 02_rag ở ba biến: (1) chunk-best đầu context như 02_rag; (2) đảo chunk-best xuống cuối, sát câu hỏi; (3) oracle: nhét đúng đoạn chứa `quote` vào 4096 token cuối. Đo `in_window_hit` (quote có nằm trong 4096 token cuối không), `n_new_tokens`, và verdict J1 (cùng `reference_short`). Không dùng 7B để chốt điểm tuyệt đối, chỉ để định hướng trước khi nộp 32B.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu (3) thắng (1) ≥ 4–5/25 câu cùng chiều và lỗi “đúng tài liệu sai đoạn” chiếm ưu thế → H1. Nếu trong (1), ≥ 30% quote ngoài cửa sổ và (2) thắng (1) ≥ 5 câu mà thua ≤ 1 → H2 (lúc đó bỏ doc-hit, chọn theo `in_window_hit`). Nếu cả ba chênh ≤ 2 câu và lỗi chủ yếu “đúng đoạn sinh sai” → H3. Bác bỏ H1/H2 nếu không đạt ngưỡng trên.
- Kỳ vọng: Không trực tiếp + điểm, nhưng chọn đúng nhánh sửa cho LB trong 1 buổi. | Tự tin: vừa | Chi phí: M

### Đề xuất B: Ba lần nộp LB “control / concise / verbose” để đo khẩu vị độ dài judge ẩn (dựa trên: M3-A, M6-A, M8-B)
- Giả thuyết: Judge ẩn có verbosity bias một trong hai chiều; đây là ẩn số lớn nhất chưa đo, và chỉ LB đo được.
- Cách làm cụ thể: Chuẩn bị zip 02_rag với vá an toàn tối thiểu (glob đệ quy, try/except, UTF-8, watchdog) gọi là `base`. Ngày 1 nộp `base` giữ `max_new_tokens=2048`, prompt gốc — đó là control để đo nhiễu so với 57.14 đã biết. Ngày 2 nộp cùng `base`, chỉ sửa `max_new_tokens=512` + prompt answer-first yêu cầu 1–3 câu, số/đơn vị trước. Ngày 3 nộp cùng `base`, chỉ sửa `max_new_tokens=1600` + prompt yêu cầu giải thích có căn cứ. Mọi thứ khác giữ nguyên. Lưu predictions local nếu có (server không trả output), ghi thời gian.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu |điểm concise − verbose| ≤ nhiễu nền (ước từ `base` vs 57.14 hoặc từ chênh runtime/điểm cùng zip) → độ dài không phải nút thắt, bác bỏ H3, dồn chi phí retrieval/parser. Nếu concise thắng ≥ 3.33 điểm → judge ưa ngắn, mọi P4/P5 theo hướng cắt token. Nếu verbose thắng ≥ 3.33 điểm và `base` thấp hơn concise → judge ưa dài, phải ngược lại. Nếu concise bị tụt > 4 điểm so với `base` đồng thời verbose cao hơn `base` → xác nhận ưa dài.
- Kỳ vọng: +3.3 đến +6.7 nếu chọn đúng format; ít nhất loại/giữ H3 chỉ trong 3 ngày. | Tự tin: vừa | Chi phí: S

### Đề xuất C: Đo quote-recall Docling vs PyMuPDF trước khi đổi parser trên LB (dựa trên: M4-B, M1-C)
- Giả thuyết: Docling có thể làm mất bảng/công thức/text khiến retrieval sai dù embedding tăng; nhưng chưa được đo nên đừng đốt một lượt LB để thử parser.
- Cách làm cụ thể: Với 20–25 tài liệu gold lấy được, trích text bằng PyMuPDF và Docling. Với mỗi câu sample25 có `quote`, kiểm fuzzy containment ≥ 0.9 trong từng parser; đếm quote-recall, số trang không text, số bảng/công thức bị mất ở 5 trang soi tay. Nếu cần, lặp trên 5 PDF nhiễu.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu Docling tìm < 70% quote trong khi PyMuPDF > 90% trên cùng bộ → parser là nút thắt, đưa PyMuPDF vào nộp LB sau Đề xuất B. Nếu cả hai ≥ 90% → bác bỏ giả thuyết parser, tập trung embedding/chunk/cửa sổ. Nếu PyMuPDF cũng < 70% → tài liệu sai hoặc bản scan, quay lại S3.
- Kỳ vọng: Không trực tiếp + điểm; quyết định có nên dùng một lượt LB cho parser hay không. | Tự tin: vừa | Chi phí: S–M

## 4. Bình chọn
VOTES: M7-A, M3-A, M5-B
