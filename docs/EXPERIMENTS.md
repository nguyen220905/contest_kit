# EXPERIMENTS — nhật ký thí nghiệm
Quy ước: 1 thí nghiệm = 1 thay đổi. Ghi cả thất bại. Kết luận phải trả lời "VÌ SAO".
Local = balanced acc = mean(acc_DONES, acc_AI) trên sample25 (judge chưa có → chưa có điểm local). LB = public 50 câu (15/35);
best mới chỉ khi Δ ≥ max(2.86, |D1 − 57.14|) (xem STAGES.md "Ngưỡng quyết định").

| ID | Ngày | Giả thuyết → thay đổi | Local | Δ vs best | LB | Kết luận | Commit |
|---|---|---|---|---|---|---|---|
| BTC-01 | 27/9 | Baseline BTC zero-shot (`starter_kit/01_zero_shot`) | — | — | 29.05 / 733 s | mốc, không nộp lại | — |
| BTC-02 | 27/9 | Baseline BTC RAG (`starter_kit/02_rag`) | — | — | 57.14 / 2080 s | mốc, không nộp lại | — |
| exp-001 | 22/10 | Harness Stage 1: 01_zero_shot, Olmo-3-7B 4-bit, sample25, 2 lượt | chưa chấm | — | — | local tất định (0/25 khác chữ); 4-bit ≠ bf16 (3/3 khác chữ) | 057ff50 |
| D1 | 22/10 | `02-safe`: 02_rag + vá an toàn (PDF đệ quy/hoa thường, không reload khi lỗi, try/except) | smoke 3/3 OK | — | chờ nộp | kỳ vọng ≈ 57.14; |D1 − 57.14| = mẫu nhiễu LB đầu tiên | (commit này) |

## Chi tiết
### exp-001 — 01_zero_shot local (kiểm harness, đo tất định + lệch shim)
- Lệnh: `.venv\Scripts\python eval/run_pipeline.py --submission starter_kit/01_zero_shot --split sample25 --run results/exp001_zs_4bit_{a,b} --quant 4bit --offline`
  và `--quant none --run results/exp001_zs_bf16` (dừng sau 3 câu: VRAM tràn sang RAM do GPU dùng chung, ~2.5 tok/s).
- Kết quả: 25/25 câu trả str, 0 lỗi. Token sinh p50/p95/max 597/1738/1984. 77–89 s/câu (lệch ~15% giữa 2 lượt; GPU dùng chung).
  `compare_runs` a vs b: 0/25 câu khác chữ. 4-bit vs bf16 (3 câu): 3/3 khác chữ, token 533 vs 528.
- Nhận xét — vì sao: greedy + cùng cấu hình → tất định, nên "chạy lại để đo nhiễu" vô nghĩa ở local. Lượng tử 4-bit đổi cách diễn đạt
  → local 4-bit chỉ để so sánh tương đối; thay đổi generator phải kiểm bằng bf16/32B (GPU thuê). Olmo-7B rất dài dòng (p95 1738 token).

### D1 — `submissions/d1_02_safe` (bài nộp đối chứng)
- Build: `.venv\Scripts\python submissions/d1_02_safe/build.py` → `submissions/d1_02_safe.zip` 399 MB, giải nén 435 MB
  (heron + heron-onnx layout 2×171 MB, MiniLM 91 MB), `main.py` ở gốc, không có `eval/`/`dev_shim`.
- Smoke test (từ bản giải nén zip, 7B 4-bit, offline, corpus 2 PDF arXiv có thư mục con + đuôi `.PDF`): index 177 chunk, 3/3 câu trả str, 0 lỗi,
  VRAM đỉnh 11 GB. Thư mục rỗng → index None, không crash (worker kiểm). Thời gian local không đại diện server.
- Kỳ vọng LB: ≈ 57.14 (cùng prompt/retrieval/generation với 02_rag trên corpus phẳng). Kết quả: <chờ nộp>.
