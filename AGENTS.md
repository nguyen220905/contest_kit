# AGENTS.md — Luật chung cho MỌI AI agent trong repo này
> Claude Code đọc file này qua CLAUDE.md. Antigravity / Codex / Cursor / Gemini… cũng phải đọc file này TRƯỚC khi làm.

## 0. Nguồn sự thật (đọc trước khi làm bất cứ gì)
1. `docs/PROGRESS.md` — đang ở đâu, làm dở gì, bước tiếp theo
2. `docs/PLAN.md` — backlog đã xếp hạng (mục đầu tiên = việc tiếp theo)
3. `docs/CODEMAP.md` — file nào làm gì (đọc cái này thay vì đọc cả codebase)
4. `docs/EXPERIMENTS.md` — đã thử gì, kết quả ra sao (không lặp lại)

## 1. Bài toán
- Cuộc thi: **IEEE CIS Technical Challenge on Domain-Specific Question Answering with LLMs** — Codabench #17954
  (https://www.codabench.org/competitions/17954/, chi tiết: https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/, Discord: https://discord.gg/q6H7cJVkQQ).
  Câu hỏi chuyên gia (text) → câu trả lời tự do (text), 2 domain: **IFMIF-DONES** (máy gia tốc hạt) và **bài báo nghiên cứu AI**.
  Bài nộp = 1 file `.zip`, `main.py` nằm ở GỐC zip, có `answer(question: str, corpus_dir: str, model_id: str) -> str`;
  được gọi 1 lần/câu trong CÙNG process (cache model/index giữa các câu). `corpus_dir` = thư mục PDF (tài liệu thật + tài liệu nhiễu),
  retrieval "mù" (không biết câu hỏi thuộc tài liệu nào). Trả về không phải `str` = không trả lời.
- Metric: **domain-balanced accuracy** (cân bằng giữa 2 domain), chấm tự động bằng LLM judge; càng cao càng tốt.
  Hoà điểm → bài chạy nhanh hơn thắng. BTC KHÔNG trả output, chỉ status + điểm (lỗi: hover status xem lý do ngắn).
- Hạn nộp: đăng ký + lập đội đóng **1/11/2026**; Public LB đóng **30/11/2026 23:55 UTC** (theo phase Codabench; trang Timeline ghi 23:59)
  = 06:55 ngày 1/12 giờ VN. Private LB 1–3/12: BTC tự chấm lại bài public tốt nhất, không nộp thêm. 3/12 công bố Phase 1;
  đội được chọn viết paper IEEE 4 trang + chuyên gia chấm câu trả lời (Phase 2) → kết quả 14/12/2026.
  Ràng buộc: **1 bài/ngày/đội** (tối đa 60 ở public). Không tự chọn bài cuối — bài có điểm public cao nhất được dùng. Đội ≤ 4 người.
- Luật cứng:
  - LLM của challenge: `model_id` = `allenai/Olmo-3.1-32B-Instruct`, load kiểu `AutoModelForCausalLM.from_pretrained(model_id)`
    (đã cache sẵn trên server, offline). Luôn dùng `corpus_dir` được truyền vào, không hard-code đường dẫn.
  - BTC trả lời trên Discord (user dẫn lại, 15/10/2026): "you may use a different model in your internal pipeline to try.
    Just make sure you use the same Transformers library"; sau đó BTC xác nhận thêm: dùng model **cùng kiến trúc** với model thi,
    không nhất thiết 32B → dev/test dùng `allenai/Olmo-3-7B-Instruct` (cùng `Olmo3ForCausalLM`, vocab 100278, ctx 65536,
    sliding window 4096; khác: 7B có 32/32 head, 32B có 40/8 GQA → bộ nhớ/thời gian của 7B KHÔNG suy ra 32B), miễn cùng `transformers==5.17.0`.
    Bài nộp thực tế vẫn sinh bằng Olmo-3.1-32B (chỉ model này có sẵn trên server; 7B ~14.6 GB không vừa zip 1 GB).
    Model phụ (embedder, layout Docling…) phải bundle trong zip — `starter_kit/02_rag` làm đúng như vậy.
  - Không internet lúc chạy; chỉ dùng package trong `starter_kit/requirements-participant.txt` (đúng version). Thiếu → xin ở kênh
    #dependencies trên Discord. Không có vLLM, flash-attn (chỉ transformers/accelerate, sdpa/eager).
  - Dữ liệu ngoài được phép nếu trích dẫn trong tài liệu nộp kèm. Index/embedding/prompt chuẩn bị trước (có internet) được bundle.
  - **Cấm** tấn công/jailbreak LLM judge → loại ngay.
- Tài nguyên:
  - Server chấm: 1×H100 80GB, 32 CPU, 128GB RAM. Giới hạn wall-clock tính cả load model: **1h** (public), ≤ 2h (private).
    Zip giải nén ≤ **1GB**. Chạy lần lượt (thời gian chờ hàng đợi không tính).
    Tham chiếu: `02_rag` mất ~36 phút cho cả corpus + 50 câu → parse PDF + generate là nút thắt thời gian.
  - Máy local: RTX 5060 Ti 16GB. Python 3.11 hệ thống đang có torch 2.10 **CPU-only** + transformers 5.3.0 → cần venv riêng khớp server
    (torch 2.8.0 bản CUDA, transformers 5.17.0). Olmo-32B không chạy được local (bf16 ~64GB; 4-bit vẫn > 16GB)
    → dev local bằng Olmo-3-7B-Instruct 8-bit.
  - GPU cloud: ngân sách thuê **thoải mái** (user, 15/10/2026) → thuê 1×H100 80GB giống server để chạy Olmo-3.1-32B thật
    (đo thời gian end-to-end, hành vi/format câu trả lời) trước khi tiêu lượt nộp.
- Dữ liệu dev: `starter_kit/sample/sample_qa.json` — 25 câu có đáp án mẫu (DONES 10, AI Papers 15; trường `question_type`).
  KHÔNG có corpus PDF; `starter_kit/sample/sites.txt` liệt kê các nguồn mà corpus được lấy từ đó.
  Dữ liệu/corpus bổ sung: **chỉ đào từ nguồn mở** (open access: arXiv, JACoW, EUROfusion, PMC…), ghi nguồn để trích dẫn khi nộp.
- `starter_kit/` = bản sao upstream `jucls/IEEE-CIS-DSQA-Starter-Kit` @ `5e596be` (2026-09-29) — chỉ đọc, code của mình để chỗ khác.
- Lệnh chuẩn (chạy từ gốc repo, Python = `.venv\Scripts\python` — Python 3.11, torch 2.8.0+cu128, transformers 5.17.0):
  - Cài đặt: `python -m venv .venv` → `.venv\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128`
    → `pip install -r` bản sao `starter_kit/requirements-participant.txt` BỎ dòng `triton==3.4.0` (chỉ có wheel Linux). Model dev: `allenai/Olmo-3-7B-Instruct` (cache HF ở C:).
  - Train: không bắt buộc (adapter/LoRA là tuỳ chọn)
  - **Eval chuẩn** (dùng để so sánh MỌI thí nghiệm): `python eval/evaluate.py --run results/<run>` — CHƯA dựng (chạy `answer()` trên tập dev + LLM judge)
  - Tạo file nộp: `<chưa có — mẫu: starter_kit/01_zero_shot/build.sh>`
  - Test nhanh / smoke test môi trường: `.venv\Scripts\python scripts/smoke_test.py --n 3` (Olmo-3-7B 4-bit, in tok/s + VRAM).
    Smoke test bài nộp (`answer()` trả str, thời gian): `<chưa có — làm cùng P4>`
- Nhiễu đã đo: <chưa đo> → không chọn model theo chênh lệch LB nhỏ hơn mức này.

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
