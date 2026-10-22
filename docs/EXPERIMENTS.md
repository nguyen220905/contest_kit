# EXPERIMENTS — nhật ký thí nghiệm
Quy ước: 1 thí nghiệm = 1 thay đổi. Ghi cả thất bại. Kết luận phải trả lời "VÌ SAO".
Local = balanced acc = mean(acc_DONES, acc_AI) trên sample25 (judge chưa có → chưa có điểm local). LB = public 50 câu (15/35);
best mới chỉ khi Δ ≥ max(2.86, |D1 − 57.14|) (xem STAGES.md "Ngưỡng quyết định").

| ID | Ngày | Giả thuyết → thay đổi | Local | Δ vs best | LB | Kết luận | Commit |
|---|---|---|---|---|---|---|---|
| BTC-01 | 27/9 | Baseline BTC zero-shot (`starter_kit/01_zero_shot`) | — | — | 29.05 / 733 s | mốc, không nộp lại | — |
| BTC-02 | 27/9 | Baseline BTC RAG (`starter_kit/02_rag`) | — | — | 57.14 / 2080 s | mốc, không nộp lại | — |
| exp-001 | 22/10 | Harness Stage 1: 01_zero_shot, Olmo-3-7B 4-bit, sample25, 2 lượt | **13.34** (3/25; DONES 20.0, AI 6.67) | — | — | local tất định (0/25 khác chữ); 4-bit ≠ bf16 (3/3 khác chữ); 7B zero-shot trả lời chung chung, thiếu số/chi tiết | 057ff50 |
| D1 | 22/10 | `02-safe`: 02_rag + vá an toàn (PDF đệ quy/hoa thường, không reload khi lỗi, try/except) | smoke 3/3 OK | — | chờ nộp | kỳ vọng ≈ 57.14; |D1 − 57.14| = mẫu nhiễu LB đầu tiên | d31608b |
| exp-002 | 22/10 | D1 (`02-safe`) trên corpus gold (18 PDF/366 trang), sample25, 7B 4-bit | **38.34** (10/25; DONES 30.0, AI 46.67) | +25.0 vs exp-001 | — | RAG > zero-shot: +7 câu, 0 câu mất (verdict lật 7, đều lên) → tiêu chí "02 > 01" đạt | (commit này) |
| D2 | 22/10 | `02-evidence-last`: D1 + đảo thứ tự fragment (tốt nhất nằm cuối, trong cửa sổ 4096 của 48/64 lớp sliding của 32B) | smoke 2/2 OK | — | chờ nộp (sau D1) | 1 thay đổi so với D1; đọc kết quả với ngưỡng max(2.86, |D1 − 57.14|) | (commit này) |

## Chi tiết
### exp-001 — 01_zero_shot local (kiểm harness, đo tất định + lệch shim)
- Lệnh: `.venv\Scripts\python eval/run_pipeline.py --submission starter_kit/01_zero_shot --split sample25 --run results/exp001_zs_4bit_{a,b} --quant 4bit --offline`
  và `--quant none --run results/exp001_zs_bf16` (dừng sau 3 câu: VRAM tràn sang RAM do GPU dùng chung, ~2.5 tok/s).
- Kết quả: 25/25 câu trả str, 0 lỗi. Token sinh p50/p95/max 597/1738/1984. 77–89 s/câu (lệch ~15% giữa 2 lượt; GPU dùng chung).
  `compare_runs` a vs b: 0/25 câu khác chữ. 4-bit vs bf16 (3 câu): 3/3 khác chữ, token 533 vs 528.
- Chấm (judge-v2 DeepSeek, refs đã duyệt `data/work/sample25_refs.jsonl`): balanced 13.34 = 3/25 đúng (#1, #2, #22).
  Theo loại: causal 0, comparative 33.3, factual 14.3, predictive 25, procedural 0. Với refs nháp cũng 13.34 → không do refs khắt khe.
- Nhận xét — vì sao: greedy + cùng cấu hình → tất định, nên "chạy lại để đo nhiễu" vô nghĩa ở local. Lượng tử 4-bit đổi cách diễn đạt
  → local 4-bit chỉ để so sánh tương đối; thay đổi generator phải kiểm bằng bf16/32B (GPU thuê). Olmo-7B rất dài dòng (p95 1738 token).

### exp-002 — D1 trên corpus gold (Stage 3b-a: 02 vs 01)
- Lệnh: `eval/evaluate.py --submission submissions/d1_02_safe --split sample25 --corpus data/corpus/gold --run results/exp002_d1_gold --quant 4bit --offline --refs data/work/sample25_refs.jsonl`
  — bị dừng ở giới hạn 2h sau 22/25 câu; 3 câu còn lại chạy riêng (results/exp002_d1_gold_rest) rồi gộp (run_note.json), chấm lại đủ 25.
- Index: 18 PDF → 2013 chunk. Thời gian local 90–790 s/câu (prefill 10k trên 7B 4-bit + GPU dùng chung) → không đại diện server.
- Kết quả: 10/25 đúng, balanced 38.34 (DONES 3/10, AI 7/15). So exp-001 (zero-shot): 7 câu lật, cả 7 từ sai → đúng.
  Theo loại: causal 50, comparative 33.3, factual 28.6, predictive 75, procedural 20.
- Nhận xét — vì sao: có tài liệu đúng trong corpus thì RAG giúp rõ (giống LB 57 vs 29). DONES vẫn thấp (3/10): q1, q2, q4 không có
  tài liệu nguồn trong corpus; procedural (tính toán) 20% → ứng viên PoT/format. Chưa có retrieval metrics (D1 không ghi LAST_DEBUG) → cần bản instrumented.

### D1 — `submissions/d1_02_safe` (bài nộp đối chứng)
- Build: `.venv\Scripts\python submissions/d1_02_safe/build.py` → `submissions/d1_02_safe.zip` 399 MB, giải nén 435 MB
  (heron + heron-onnx layout 2×171 MB, MiniLM 91 MB), `main.py` ở gốc, không có `eval/`/`dev_shim`.
- Smoke test (từ bản giải nén zip, 7B 4-bit, offline, corpus 2 PDF arXiv có thư mục con + đuôi `.PDF`): index 177 chunk, 3/3 câu trả str, 0 lỗi,
  VRAM đỉnh 11 GB. Thư mục rỗng → index None, không crash (worker kiểm). Thời gian local không đại diện server.
- Kỳ vọng LB: ≈ 57.14 (cùng prompt/retrieval/generation với 02_rag trên corpus phẳng). Kết quả: <chờ nộp>.

### D2 — `submissions/d2_evidence_last` (evidence-last)
- Giả thuyết: Olmo-3.1-32B có 48/64 lớp sliding-window 4096; 02_rag đặt fragment hạng 1 ở ĐẦU context ~10k → 48 lớp không thấy khi sinh.
  Đặt fragment hạng 1 sát câu hỏi → model dùng evidence tốt hơn. Bác bỏ nếu LB(D2) − LB(D1) ≤ −max(2.86, |D1 − 57.14|).
- Thay đổi: đúng 1 dòng `selected.reverse()` trong `_build_context` (cùng tập fragment, cùng ngân sách). Worker kiểm bằng CPU: D2 = đảo ngược D1.
- Build: `submissions/d2_evidence_last.zip` 399 MB (giải nén 435 MB). Smoke (bản giải nén, 7B 4-bit, offline, 2 PDF): 2/2 câu OK, 0 lỗi;
  khác chữ D1 2/2, token trung bình 348 vs 435 (local 7B, không kết luận).
