# AGENTS.md — Luật chung cho MỌI AI agent trong repo này
> Claude Code đọc file này qua CLAUDE.md. Antigravity / Codex / Cursor / Gemini… cũng phải đọc file này TRƯỚC khi làm.

## 0. Nguồn sự thật (đọc trước khi làm bất cứ gì)
1. `docs/PROGRESS.md` — đang ở đâu, làm dở gì, bước tiếp theo
2. `docs/PLAN.md` — backlog đã xếp hạng (mục đầu tiên = việc tiếp theo)
3. `docs/CODEMAP.md` — file nào làm gì (đọc cái này thay vì đọc cả codebase)
4. `docs/EXPERIMENTS.md` — đã thử gì, kết quả ra sao (không lặp lại)

## 1. Bài toán (ĐIỀN VÀO)
- Cuộc thi: <tên> — <mô tả 1–2 câu: input → output>
- Metric: <vd macro-F1>, càng <cao/thấp> càng tốt. Cách tính chính xác: <...>
- Hạn nộp: <ngày giờ + múi giờ>. Ràng buộc: <số lần nộp/ngày, số bài cuối được chọn, BTC có kiểm code/checkpoint không>
- Luật cứng: <điều bị cấm / dễ bị loại, vd không dùng dữ liệu ngoài, không nối dữ liệu giữa các track>
- Tài nguyên: <GPU/máy nào, dùng chung với ai, cách chạy (hàng đợi, scheduler), quota cloud>
- Lệnh chuẩn (chạy từ <thư mục gốc>, Python = `<đường dẫn python/venv>`):
  - Cài đặt: `<...>`
  - Train: `<...>`
  - **Eval chuẩn** (dùng để so sánh MỌI thí nghiệm): `python eval/evaluate.py --run results/<run>`
  - Tạo file nộp: `<...>`
  - Test nhanh / smoke test: `<...>`
- Nhiễu đã đo: <điểm public ±..., chênh lệch giữa 2 bài ±...> → không chọn model theo chênh lệch LB nhỏ hơn mức này.

### Hợp đồng eval (bắt buộc, làm đầu tiên)
- Split cố định lưu ở `data/splits/` (không sinh lại), seed cố định.
- Mỗi lần eval ghi ra `results/<run>/metrics.json` (metric chính + metric phụ + theo từng lớp)
  và `results/<run>/predictions.csv` (id, y_true, y_pred, score/prob, + cột meta hữu ích: độ dài, nguồn…).
- Không có predictions từng mẫu thì KHÔNG thể tìm nguyên nhân lỗi → không được bỏ qua.
- Khi eval đã chốt: bật dòng `eval/**` trong `.claude/protected_paths.txt`.

## 2. Phạm vi sửa code (quan trọng nhất)
- Trước khi sửa, ghi rõ: **"Sẽ sửa: file1, file2 — lý do"**. Mặc định ≤ 3 file mỗi bước.
- Chỉ sửa đúng những file đã nêu. Cần thêm file → DỪNG, nói lý do, chờ đồng ý.
- Không refactor, đổi tên, format lại, "dọn dẹp" ngoài yêu cầu. Không xoá code/comment không liên quan.
- Không đụng đường dẫn được bảo vệ (`.claude/protected_paths.txt`): dữ liệu gốc, eval đã chốt, split, file `.env`.
- Ưu tiên diff nhỏ nhất làm được việc.

## 3. Quy trình thí nghiệm
- Mỗi thí nghiệm thay đổi MỘT thứ. Viết giả thuyết + tiêu chí bác bỏ TRƯỚC khi chạy.
- Luôn so bằng lệnh eval chuẩn, cùng split, cùng seed. Chênh lệch nhỏ hơn nhiễu giữa các seed = "không kết luận".
- Kết quả tốt bất thường → nghi leakage/bug trước khi tin.
- Không cải thiện → KHÔNG đổi hyperparameter mò mẫm. Phân tích lỗi từ `predictions.csv` (nhóm lỗi, đếm, xem ví dụ), rồi đặt giả thuyết dựa trên bằng chứng.
- 3 thí nghiệm liên tiếp không cải thiện → dừng, đề xuất phân tích lỗi + hội đồng model hoặc tranh luận giả thuyết.
- Ghi MỌI thí nghiệm (kể cả thất bại) vào `docs/EXPERIMENTS.md`.

## 4. Theo dõi tiến độ
- Xong mỗi bước: cập nhật `docs/PROGRESS.md` (Đang làm / Đã xong / Bước tiếp theo). Ngắn, ghi đè — không viết dài thêm.
- Mỗi bước chạy được: git commit nhỏ, message dạng `[exp-012] thêm focal loss` hoặc `[web] trang đăng nhập`.
- Không commit: data lớn, checkpoint, `.env`, `node_modules`.

## 5. Codebase lớn
- Dùng `docs/CODEMAP.md` để định vị. Chỉ mở file cần thiết; tìm bằng grep thay vì đọc toàn bộ.
- Thêm/đổi module → cập nhật CODEMAP.md (1–2 dòng).

## 6. Giao tiếp
- Trả lời bằng tiếng Việt, ngắn gọn. Kết thúc mỗi bước bằng: đã làm gì → kết quả → bước tiếp theo.
- Không chắc → hỏi 1 câu cụ thể thay vì đoán.
- Bị hook chặn → không tìm cách lách, báo người dùng.
