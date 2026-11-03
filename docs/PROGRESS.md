# PROGRESS — cập nhật: 2026-11-03 bởi claude
> File này là "bộ nhớ" của dự án. Hook SessionStart tự nạp nó sau mỗi lần mở phiên / compact / clear.
> Giữ ngắn (< 80 dòng). Ghi đè thay vì viết thêm.

## Mục tiêu hiện tại
- Người dùng (2/11): **điểm cao nhất, thời gian là phụ** (public ≤ ~55 phút dưới 1h cứng). LB 2/11: top 72.86, top 20 = 64.76.

## Điểm tốt nhất
| Metric local | LB public | File nộp | Thí nghiệm | Ghi chú |
|---|---|---|---|---|
| lab C200 Docling in_ctx@10k .85, MRR .407 | **64.76** / 2088 s | d3_hybrid_title.zip (2/11) | D3 | tài khoản `snute`; D1 57.14 · D2 49.05 · D4 53.81 (đều thua) |

## Đang làm
- Lab rerank trên C200 Docling: c9 (D3) vs c20–c24 (mxbai-xs / mxbai-base / gte-modernbert / bge-reranker-base / msmarco, top 50)
  → `results/retr_c200_rerank/summary.csv`. Thắng nếu MRR + in_ctx@2k tăng rõ (≥ 2/20 câu) → D5 = D3 + rerank (prompt giữ nguyên).
- Git Windows mất (ổ E:) → commit bằng WSL git (memory git-via-wsl).

## Đã xong (mới nhất ở trên, tối đa 10 dòng)
- 2026-11-03 — D4 (prompt mới) LB **53.81 (−10.95)** → bác bỏ; local 7B gold đã báo trước (7/25 vs D3 9/25). Sửa judge: verdict bằng regex khi JSON hỏng vì LaTeX
- 2026-11-02 — Lab Docling C200: c9 .60/.80/.85 (in_ctx@2k/5k/10k) > PyMuPDF .75 @10k; chunk 1200 không thêm gì → loại D5 PyMuPDF/chunk 1200
- 2026-11-02 — D3 LB **64.76 (+7.62)**: retrieval hybrid MiniLM+BM25 + tiêu đề (lab dự đoán đúng)
- 2026-11-02 — D2 evidence-last 49.05 (−8.09) → 32B bám fragment ĐẦU context
- 2026-10-29 — D1 02-safe 57.14 = baseline BTC → server tất định
- 2026-10-22 — Stage 1–3: harness (`eval/run_pipeline.py`, `evaluate.py`, judge DeepSeek), corpus gold 18 PDF + nhiễu 182 PDF, sample25 split

## Bước tiếp theo (1–3 việc, có lệnh chạy)
1. Đọc lab rerank → nếu thắng: build `submissions/d5_rerank` (D3 + CrossEncoder, kiểm zip giải nén < 1 GB) + smoke
   `.venv\Scripts\python eval/run_pipeline.py --submission data/work/unzip/d5_rerank --split sample25 --run results/s1_d5_smoke --offline --limit 2`
2. Thử Olmo làm reranker (LLM chấm liên quan top-30 chunk) trong lab bằng 7B — mô-đun "CI" cho paper
3. Thuê GPU (RTX PRO 6000 96 GB) để chạy 32B thật: đổi generation chỉ nộp khi 32B trên sample25 không thua

## Vướng mắc / câu hỏi mở
- GPU thuê: chưa thuê — mọi thay đổi khâu sinh câu trả lời (prompt, tự kiểm) đang mù với 32B (D2, D4 đều tụt 8–11 điểm).
- Domain nào có 15 câu trong public chưa rõ.

## Bẫy đã biết (đừng lặp lại)
- **Không đổi prompt/generation khi chưa có số**: D4 tụt 10.95. Local gold 7B đoán đúng hướng đổi generation → chờ local xong rồi mới nộp.
- Local gold (18 PDF, không nhiễu) KHÔNG thấy lợi của retrieval (D3 ≈ D1 local nhưng +7.62 LB) → retrieval phải đo bằng lab C200.
- 32B bám fragment ĐẦU (D2) → mọi thay đổi xếp hạng phải tăng độ chính xác top fragment (MRR, in_ctx@2k).
- Đừng ép "chỉ dùng số trong context": gold dùng cả kiến thức chuẩn (ln2, khối lượng riêng Li) và có câu tiền đề sai.
- Zip giải nén ≤ 1 GB (D3 đang 435 MB). Không dùng BGE-M3 / bge-reranker-v2-m3. Docling table structure không chạy trên server.
- bnb 8-bit rất chậm → dev 4-bit. Tốc độ/bộ nhớ 7B (MHA) không suy ra 32B (GQA 40/8).
- GPU local dùng chung với dự án khác → KHÔNG kill tiến trình của dự án khác; thời gian local nhiễu.
- Judge local: reason chứa LaTeX làm JSON hỏng → đã có fallback regex (eval/judge.py).
