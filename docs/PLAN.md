# PLAN — backlog đã xếp hạng
Ưu tiên = (Kỳ vọng × Tự tin) / Chi phí. Mục "todo" đầu tiên = việc tiếp theo.
Nguồn: setup | error-analysis | council-rXX/<model> | debate | ý tưởng của tôi
Chi tiết + bằng chứng: `docs/council/round_01/SYNTHESIS.md` · nguồn cho paper: `docs/council/round_01/REFERENCES.md`

## Dữ kiện đã kiểm chứng (council-r01, phiên chính tự kiểm lại 2026-10-15)
- Public LB = 50 câu, chia **15/35** giữa 2 domain: cả 36/36 điểm LB khớp `50·(a/15 + b/35)`, không split nào khác (10–25) khớp.
  1 câu = 3.33 điểm (domain 15 câu) hoặc 1.43 điểm (domain 35 câu). Chênh LB < ~4 điểm = nhiễu. Private = đủ 100 câu.
- LB 2026-10-15 (34 đội): top 69.05 (1288 s) · 68.57 (668 s) · 68.10 (605 s); **ngưỡng top 20 = 59.05**.
  Baseline BTC đã có trên LB: zero-shot 29.05 / 733 s, RAG 02_rag 57.14 / 2080 s → không tốn lượt nộp lại. LB có cột thời gian (= profiling miễn phí).
- Phase 2 (trang evaluation): điểm cuối = **30%** accuracy Phase 1 + **30%** chuyên gia chấm chất lượng câu trả lời + **40%** technical innovation (CI); top 20 vào Phase 2.
  Judge ẩn chấm nhị phân, không điểm từng phần; hoà → tổng thời gian ngắn hơn thắng.
- Gold mẫu dùng cả kiến thức ngoài tài liệu (q3 ln2, q6 khối lượng riêng Li) và có câu tiền đề sai (q7) → KHÔNG ép prompt "chỉ dùng số trong fragment".
- Zip 1 GB: BGE-M3 / bge-reranker-v2-m3 (2.27 GB) không vừa. Docling không bật table structure được trên server (`starter_kit/02_rag/main.py:60`).
- Baseline 02_rag dễ vỡ: `glob("*.pdf")` không đệ quy (`02_rag/main.py:89`), `answer()` không try/except → luôn bọc và luôn trả `str`.

## Dữ kiện từ Discord BTC (đọc 2026-10-22, chi tiết `docs/DISCORD.md`)
- **Được bundle model bất kỳ** (sinh hay không), Olmo không bắt buộc là LLM duy nhất; "only model you may load" đã xoá. Còn lại: zip 1 GB + thời gian.
- Judge **tất định** → nộp lại cùng zip mà điểm đổi là do generation không tất định. Nhiễu LB = nhiễu sinh câu trả lời, không phải judge.
- Private 2h là giới hạn **mềm**: câu nào chưa xong khi hết 2h thì mất điểm câu đó → giữ thời gian public ≤ ~45 phút.
- Kích thước corpus thật không công bố → harness phải đo được thời gian theo kích thước corpus (corpus local tăng dần).
- Không có output nào từ server; chỉ có thông báo lỗi ngắn khi fail. Lỗi không được kéo theo load lại model (ca thật: reload mỗi câu → OOM).

## Mốc
- M0: đăng ký xong (form + Codabench cùng email) (hạn: **2026-10-31**, đăng ký đóng 1/11)
- M1: venv khớp server (local + H100 thuê) + eval harness + corpus dev, đo Recall@k của retrieval baseline (hạn: 2026-10-22)
- M2: bài nộp #1 "fast-safe" > 57.14 và vào top 20 (hiện 59.05), runtime < 1500 s (hạn: 2026-10-26)
- M3: định dạng câu trả lời + mô-đun CI (cho paper), mục tiêu ≥ top hiện tại 69 (hạn: 2026-11-20)
- M4: chốt bài nộp cuối, đóng băng (hạn: **2026-11-30 23:55 UTC**); Phase 2 paper nếu vào top 20

## Backlog
| ID | Thí nghiệm / việc | Nguồn | Kỳ vọng | Tự tin | Chi phí | Trạng thái |
|---|---|---|---|---|---|---|
| P0 | (người dùng) Hoàn tất đăng ký — **mọi thành viên** điền form; đội = organisation trên Codabench. Luật model đã rõ (Discord 19/10), không cần hỏi thêm | setup + BTC Discord | — | cao | S | todo |
| P1 | Venv local khớp server (torch 2.8.0+cu128, transformers 5.17.0, đúng requirements-participant.txt) + smoke test với Olmo-3-7B-Instruct (dev mặc định 4-bit: 9 tok/s, 4.9 GiB) | setup + BTC 15/10 | hạ tầng | cao | S | **xong 15/10** |
| P1b | "Server giả lập" trên 1×H100 80GB thuê: cùng requirements, Olmo-3.1-32B bf16, script chạy `answer()` cho cả corpus dev + đo thời gian end-to-end (gồm load model, parse PDF); hiệu chỉnh bằng 02_rag (LB 2080 s). Mọi bài nộp phải qua đây trước | user 15/10 (ngân sách thuê thoải mái) | hạ tầng; tránh mất lượt nộp | cao | M | hoãn — user: thuê sau khi cải tiến baseline. council-r02 đề xuất 1 phiên 2–3 h ngay sau Stage 1+3 (đo Docling/prefill 32B thật để kiểm giả thuyết kích thước corpus, chạy Linux offline) — **chờ người dùng** |
| P2+P3 | **Plan chi tiết: `docs/STAGES.md` v2** (6 stage + 3b, đã qua council-r02) — chờ người dùng duyệt | council-r02 | hạ tầng | cao | M | chờ duyệt |
| P2 | Eval harness: `results/<run>/predictions.csv` + `metrics.json` (domain-balanced kiểu 15/35), judge nhị phân hiệu chỉnh: chấm đúng ≥ 24/25 gold, bác ≥ 90% gold bị làm sai số/đơn vị; ngưỡng quyết định ≥ 3 câu lật trên holdout | council-r01/M8-C + G | hạ tầng | cao | S | todo |
| P3 | Corpus dev (CHỈ nguồn mở, ghi nguồn để trích dẫn): PDF của 25 câu mẫu + tài liệu DONES công khai + nhiễu từ sites.txt + 50–100 câu tổng hợp có gold-evidence; holdout theo tài liệu (25 câu mẫu = holdout cuối); đo Recall@k / doc-hit@1 của retrieval 02_rag | council-r01/E | hạ tầng | cao | M | todo |
| P4 | Bài nộp #1 "fast-safe": PyMuPDF đa tiến trình + `find_tables`, embed GPU, BM25+dense (RRF) + title boost + header ngữ cảnh mỗi chunk, ctx ~5k, prompt answer-first, `max_new_tokens` ~640, try/except + watchdog, zip < 900 MB. Chạy qua P1b trước khi nộp. Bác bỏ nếu LB ≤ 57.14 hoặc runtime > 1500 s. council-r02: làm song song P2/P3, mỗi ngày nộp 1 biến thể nhỏ từ khi harness Stage 1 xong (không phí lượt nộp) — **chờ người dùng** | council-r01/M8-B + #2 + #3, council-r02 | +5…+10 | vừa | M | todo |
| P5 | A/B định dạng câu trả lời (answer-first + giải thích ngắn vs dài kiểu gold vs atomic) trên judge local; sau đó PoT có cổng cho câu tính toán | council-r01/M8-A + #6 | +3…+6 | vừa–thấp | S | todo |
| P6 | Mô-đun CI: fuzzy fusion tín hiệu retrieval, trọng số tối ưu DE/CMA-ES offline (so với RRF); tiến hoá prompt GEPA/EvoPrompt offline; ghi log hội tụ cho paper | council-r01/#5 | +1…+4 LB; cao cho Phase 2 (40%) | vừa | M | todo |

Hoãn (chỉ làm khi có bằng chứng cần): HyDE / multi-query (council-r01, lợi ích thấp với câu số liệu) · self-consistency k>1 (nhân thời gian) ·
document-first đọc cả paper 12–16k token khi doc-hit@1 cao (kimi, thiểu số) · agent ReAct nhiều bước (kimi, thiểu số) ·
prompt lookup decoding (giảm thời gian, cần đo trên Olmo-32B ở P1b) ·
LLM phụ bundle ≤ 1 GB (draft model cho speculative decoding, rewrite query, reranker kiểu LLM) — **được phép** từ Discord 19/10;
Olmo không có bản nhỏ cùng tokenizer vừa 1 GB → draft khác tokenizer cần universal assisted decoding, phải đo lợi thời gian thật ·
LoRA/adapter cho Olmo-32B — luật cho phép, thuê H100 được; vẫn thiếu dữ liệu train (chỉ có câu tổng hợp) → cân nhắc sau P4.

## Đã loại (và lý do)
- Nộp lại 01_zero_shot / 02_rag nguyên bản — LB đã có điểm của BTC (29.05 / 57.14), tốn lượt vô ích (council-r01/grok, đã kiểm API).
- BGE-M3, bge-reranker-v2-m3 — 2.27 GB, fp16 ~1.14 GB, vượt zip 1 GB. Thay bằng bge-small/bge-base, mxbai-rerank-base / bge-reranker-base.
- Docling table structure — server thiếu cv2/libGL (`starter_kit/02_rag/main.py:60`); bảng lấy bằng PyMuPDF `find_tables` / pdfplumber.
