# PROGRESS — cập nhật: 2026-10-22 bởi claude
> File này là "bộ nhớ" của dự án. Hook SessionStart tự nạp nó sau mỗi lần mở phiên / compact / clear.
> Giữ ngắn (< 80 dòng). Ghi đè thay vì viết thêm.

## Mục tiêu hiện tại
- Vào top 20 Public LB (ngưỡng hiện 59.05; top 69.05) trước 2026-10-26, rồi đẩy lên ≥ 69 với mô-đun CI cho paper Phase 2.

## Điểm tốt nhất
| Metric local | LB public | File nộp | Thí nghiệm | Ghi chú |
|---|---|---|---|---|
| — | — | — | — | chưa nộp; mốc BTC: zero-shot 29.05 / 733 s, RAG 57.14 / 2080 s |

## Đang làm
- Việc: Stage 1 của `docs/STAGES.md` v3 — harness xong (commit `057ff50`); còn lượt bf16 (đang chạy, chậm do GPU dùng chung) + bài nộp D1 `02-safe`
- Trạng thái: chờ lượt bf16; D1 chưa bắt đầu

## Đã xong (mới nhất ở trên, tối đa 10 dòng)
- 2026-10-22 — Council round_06 (đủ 8 model) chấm STAGES v2: TB 6.8/10, trung vị 7, "sửa nhỏ rồi làm" → STAGES v3 (D1 nộp sớm, Stage 5 hoãn, Stage 3b thêm oracle-chunk/evidence-last)
- 2026-10-22 — Stage 1 harness: 01_zero_shot 7B 4-bit sample25 ×2 → local tất định (0/25 khác chữ), token p50/p95 597/1738, 77–89 s/câu
- 2026-10-22 — Council round_02 chấm plan P2+P3: chỉ DeepSeek trả lời (key OpenRouter HẾT HẠN, 401) → 5/10 và 6/10, "sửa nhỏ rồi làm"; STAGES.md v2
- 2026-10-22 — Đọc Discord BTC → `docs/DISCORD.md`: được bundle model bất kỳ (kể cả LLM khác), judge tất định, private 2h là giới hạn mềm, corpus không công bố kích thước
- 2026-10-15 — P1: `.venv` khớp server (torch 2.8.0+cu128 thấy sm_120, transformers 5.17.0, pip check sạch; bỏ triton vì chỉ có Linux). Smoke test Olmo-3-7B: 4bit 9 tok/s · 4.9 GiB; bf16 11 tok/s · 13.6 GiB; 8bit 2.8 tok/s
- 2026-10-15 — Council round_01 (8 model vòng 1; vòng 2 chỉ deepseek do OpenRouter hết hạn mức ngày) → SYNTHESIS + REFERENCES (40 nguồn đã mở), PLAN cập nhật P0–P6
- 2026-10-15 — Chép kit v10, clone starter kit BTC vào `starter_kit/` (@5e596be), điền AGENTS.md mục 1

## Bước tiếp theo (1–3 việc, có lệnh chạy)
1. P0 (người dùng): đăng ký trước 1/11
2. P2 + P3: eval harness (judge nhị phân, predictions.csv) + corpus dev nguồn mở; kiểm môi trường: `.venv\Scripts\python scripts/smoke_test.py --n 3`
3. P4: cải tiến baseline → rồi mới thuê H100 (P1b) để đo 32B trước khi nộp

## Vướng mắc / câu hỏi mở
- Đã rõ (BTC 15/10): dev dùng model cùng kiến trúc (Olmo-3-7B-Instruct), cùng transformers 5.17.0. Ngân sách thuê GPU thoải mái. Dữ liệu thêm chỉ từ nguồn mở.
- Đã rõ (Discord 19/10): được bundle model bất kỳ, kể cả LLM khác Olmo — chỉ giới hạn zip 1 GB + thời gian.
- GPU thuê: EzyCloudX Docker (không có H100/A100-80GB) → đề xuất 1x RTX PRO 6000 Blackwell 95 GB (105k CP/giờ); quy đổi thời gian bằng 02_rag vs LB 2080 s.
- `data/splits/**` bị hook bảo vệ → Stage 3 cần người dùng cho phép tạm để ghi split lần đầu.
- Domain nào có 15 câu trong public (đoán DONES theo tỉ lệ mẫu, chưa chắc).
- Key OpenRouter **hết hạn** (22/10, HTTP 401) → 7/8 model hội đồng + Perplexity không chạy được; người dùng cần thay key trong `tools/.env`.
  Trước đó key còn giới hạn $7/ngày → khi thay nên nâng hạn mức.

## Bẫy đã biết (đừng lặp lại)
- Đừng ép "chỉ dùng số trong context": gold dùng cả kiến thức chuẩn (ln2, khối lượng riêng Li) và có câu tiền đề sai.
- Đừng chọn model > 1 GB để bundle (BGE-M3, bge-reranker-v2-m3). Đừng trông vào Docling table structure trên server.
- Một lượt nộp = 1 ngày: luôn chạy smoke test + kiểm zip giải nén < 900 MB trước khi nộp.
- bnb 8-bit (LLM.int8) rất chậm (2.8 tok/s) → dev dùng 4-bit. Tốc độ/bộ nhớ của 7B (MHA 32/32) KHÔNG suy ra được 32B (GQA 40/8).
- `triton` không có wheel Windows → venv local thiếu triton; kiểm lại trên Linux (H100 thuê) trước khi nộp.
- GPU local dùng chung với dự án khác (project_r2ai, UNDERWATER-IMAGING, project_pnpl) → thời gian local nhiễu; KHÔNG kill tiến trình của dự án khác.
- Olmo-32B: 48/64 lớp sliding 4096 → fragment đặt đầu context dài bị 48 lớp "không thấy" khi sinh (giả thuyết D2 evidence-last).
