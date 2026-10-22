# Discord BTC — các câu trả lời chính thức (đọc ngày 2026-10-22)
> Server "IEEE-CIS Technical Challenge on Domain-Specific Question Answering with LLMs" (https://discord.gg/q6H7cJVkQQ).
> BTC trả lời: **Jaime Uclés** (tác giả starter kit, `jucls`), **Isaac** (Isaac Triguero). Ghi lại theo ý, trích nguyên văn ngắn khi cần.
> Kênh đã đọc: welcome, announcements (trống), forum general (trống), start-kit, support/general, submission-issues, dependencies.
> Cập nhật file này mỗi lần đọc lại Discord; ghi ngày đọc ở tiêu đề.

## Thay đổi luật quan trọng nhất
- **Model (Jaime, 3/10 và 19/10):** được bundle **bất kỳ model nào, sinh văn bản hay không**; Olmo-3.1-32B-Instruct
  **không bắt buộc là LLM duy nhất**, thậm chí có thể không dùng LM nào. Câu "the only model you may load" đã bị **xoá** khỏi
  trang Codabench (19/10). Code của mình tự load `model_id` (nếu dùng). Olmo được cung cấp sẵn chỉ vì giới hạn 1 GB làm việc
  upload model lớn khác không thực tế. Ràng buộc còn lại: kích thước zip + thời gian.
- **Dev local (Isaac, 29/9):** không chạy được 32B thì thử bằng model khác trong pipeline, miễn **cùng thư viện Transformers**.
  (Người dùng dẫn thêm 15/10: BTC xác nhận dùng model cùng kiến trúc, không nhất thiết 32B.)

## Thời gian & chấm điểm
- **Private LB (Jaime, 3/10):** giới hạn **2 giờ** nhưng **không phải giới hạn cứng**: BTC chạy lại bài public tốt nhất (đã hợp lệ)
  nên không thể fail vì thời gian; **chỉ câu trả lời xong trong 2 giờ mới được chấm**. BTC giả định thời gian tăng tuyến tính
  theo số câu (50 → 100) nhưng không đảm bảo. Public vẫn là giới hạn cứng 1 giờ (quá giờ = bài không hợp lệ).
- **Bài dùng cho private (Jaime, 16/10):** tự động lấy bài public điểm cao nhất, không tự chọn.
- **Judge (Jaime, 16/10):** verdict **nhất quán** — output giống hệt → điểm giống hệt. Nộp lại cùng zip mà điểm khác là do
  **output sinh ra khác** (không tất định trên GPU), không phải do judge.
- **VRAM (Jaime, 16/10):** H100 80 GB; tự tính footprint của Olmo khi chọn độ dài context. BTC không chia sẻ thông tin bài đội khác.

## Corpus & dữ liệu
- **Corpus (Jaime, 7/10):** không được truy cập corpus đánh giá, **kích thước không công bố**. Gợi ý: tự dựng corpus từ các
  site trong `sites.txt`, bắt đầu nhỏ rồi tăng dần để xem cách tiếp cận chịu được corpus lớn thế nào.

## Chẩn đoán bài nộp
- **Mọi output bị tắt (Jaime, 16/10):** nút tải prediction/scoring bị khoá có chủ ý; chẩn đoán duy nhất là **thông báo lỗi ở
  status** khi bài fail. Hỗ trợ lỗi bài nộp được trả lời công khai ở #submission-issues.
- Ví dụ thông báo lỗi: `Error 20, no question produced an answer (TimeoutError, line 78)`.
- Ca lỗi thật (16/10): code truy cập `model.hf_device_map` khi không load bằng `device_map` → lỗi → **load lại model ở mỗi câu**
  → CUDA OOM cho các câu còn lại. BTC khuyên load model giống starter kit.

## Thủ tục
- **Dependencies (Jaime, 29/9):** xin package mới bằng **private thread** trong #dependencies: tag @Organization, ghi
  `package==version` + lý do. Package thêm vào dùng chung cho mọi đội, thông báo định kỳ ở #dependencies.
  Đến 22/10: **chưa có thông báo package mới** nào.
- **Đăng ký (Jaime, 30/9):** **mọi thành viên** đều phải điền form. Đội = organisation trên Codabench; vào đội khi nộp bài dưới
  tên organisation đó.
- **Metadata / fact sheet (Jaime, 8/10):** bỏ vào zip như file phụ, dùng từ code của mình.
- **Điều kiện tham gia (Jaime, 4/10):** mở cho mọi người; chỉ loại thành viên ban tổ chức / hội đồng chấm.
- **Codabench trắng trang (Jaime, 15/10):** mạng nền tảng chập chờn, tải lại sau ít phút.
