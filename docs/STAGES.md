# STAGES — P2 + P3: eval harness + corpus dev — v3 sau council round_06 (đủ 8 model)
> /orchestrate tạo và cập nhật file này. Hook Stop đọc dòng "Trạng thái" và mục "Tiêu chí tổng" để quyết định vòng lặp chạy tiếp hay dừng.
> Chỉ criteria-checker đánh [x], kèm bằng chứng. Góp ý: `docs/council/round_06/SYNTHESIS.md` (8 model, 6.8/10 trung bình, "sửa nhỏ rồi làm"),
> `docs/council/round_02/SYNTHESIS.md` (v1, chỉ DeepSeek).

Trạng thái: lập kế hoạch
<!-- Các giá trị: lập kế hoạch · đang chạy · chờ người dùng — <câu hỏi> · dừng — <lý do> · xong -->

## Bối cảnh (2026-10-22)
- Điểm hiện tại: chưa nộp. LB: top 69.05, ngưỡng top 20 = 59.05; mốc BTC zero-shot 29.05 / 733 s, RAG 57.14 / 2080 s.
- Mục tiêu: thước đo local đủ tin để chọn thay đổi; **LB là thước đo chính** (judge ẩn tất định) → nộp sớm, mỗi ngày 1 bài có đối chứng.
- Đã đo (Stage 1): Olmo-3-7B 4-bit, 01_zero_shot trên sample25: **local tất định** (2 lượt, 0/25 câu khác chữ); token sinh p50/p95/max
  597/1738/1984; 77–89 s/câu (lệch ~15% giữa 2 lượt; GPU dùng chung với tiến trình dự án khác → thời gian local chỉ để tham khảo).
- Olmo-3.1-32B (config): 48/64 lớp sliding-window 4096 + 16 lớp full. 02_rag đặt fragment tốt nhất ĐẦU context 10k (`02_rag/main.py:124-141`)
  → fragment đó nằm ngoài cửa sổ của 48 lớp sliding khi sinh. 02_rag không lưu doc/page của chunk (`main.py:98`).
- Điểm số: balanced = mean(acc_DONES, acc_AI). Public 50 câu (15/35): 1 câu = 3.33 hoặc 1.43 điểm. Sample25 (10 DONES/15 AI): 1 câu DONES = 5 điểm, AI = 3.33.
- Đã loại (đừng lặp): nộp lại 01/02 nguyên bản; BGE-M3 / bge-reranker-v2-m3 (> 1 GB); "chạy lại local = nhiễu" (local tất định);
  dò số trang corpus bằng cách cố tình gây lỗi (rủi ro bị loại); Ollama làm judge hay proxy 32B.

## Quyền tự quyết & ngân sách
- Tự điều chỉnh stage chưa làm: có
- Tự đổi hướng (khi chạm dấu hiệu hướng sai): không — hỏi người dùng
- Tối đa 3 lượt sửa / stage · tối đa 40 lượt cho cả vòng lặp
- API: judge + sinh câu hỏi = DeepSeek gọi thẳng. OpenRouter (key mới 22/10) cho judge phụ / hội đồng.
- Nộp bài: người dùng tự upload zip lên Codabench (1 lượt/ngày); phiên chính chuẩn bị zip + smoke test.

## Thiết kế chung
- Bài nộp là thư mục có `main.py` (giống zip). Harness `eval/run_pipeline.py` import đúng `main.py` đó, gọi `answer()` tuần tự 1 process.
  Local: `eval/dev_shim.py` chèn bnb 4-bit + `model_id=allenai/Olmo-3-7B-Instruct`. Server không có shim.
- Hook chẩn đoán tuỳ chọn (server bỏ qua): `retrieve(question, corpus_dir) -> list[dict(doc, page, score)]` và `LAST_DEBUG`
  (doc, page, token offset của từng fragment trong prompt) → đo `quote_recall@k` và `in_window_hit` (evidence có nằm trong 4096 token cuối không).
- Định dạng split (`data/splits/*.jsonl`): `id, domain, question_type, question, reference_short, reference_long,
  evidence: [{doc_id, page, quote}], flags: {calc, false_premise, names_paper, outside_doc, closed_book}, source`.
- **Quy tắc dùng split:** synthetic (nếu có) chỉ để chọn retrieval/chunking/thời gian; thay đổi generator (prompt, format, PoT…) phải
  thắng trên sample25 rồi xác nhận bằng LB (van Elburg et al., arXiv 2508.11758).
- **Ngưỡng quyết định:** local sample25 — đề cử lên LB khi **thắng ròng ≥ 3 câu và thua ≤ 1** (cùng retrieval, cùng judge);
  LB — coi là best mới chỉ khi **Δ ≥ max(2.86, |D1 − 57.14|, nhiễu 32B đo trên GPU thuê)**.
- Mọi I/O `encoding="utf-8"`; mỗi lượt chạy = 1 process mới.
- Git: `data/` đang bị ignore → đổi thành `data/*` + `!data/splits/` + `!data/manifests/`. Ghi `data/splits/` lần đầu cần người dùng
  tạm bỏ dòng bảo vệ (hook chặn). Sau Stage 2: bật `eval/**` trong `.claude/protected_paths.txt`.

## Tiêu chí tổng
- [ ] `.venv\Scripts\python eval/evaluate.py --submission <02 instrumented> --split sample25 --corpus data/raw/corpus/gold(+C50) --run results/<run>`
      → `predictions.csv` + `metrics.json` có: **balanced = mean(acc_D, acc_A)**, acc theo domain và question_type, số lỗi, thời gian
      (overhead, p50/p95, dự phóng public/private), token sinh p50/p95, `quote_recall@k`, `in_window_hit`.
- [ ] Judge J1 qua 4 test hiệu chỉnh (Stage 2).
- [ ] D1 đã nộp, có điểm + runtime trên LB; kết quả ghi `docs/EXPERIMENTS.md`.
- [ ] Stage 3b xong: oracle-chunk + evidence-first/last + closed-book theo domain → quyết định ưu tiên retrieval hay generator, có số.

## Dấu hiệu hướng sai (viết TRƯỚC khi làm)
- Judge J1 không qua hiệu chỉnh sau 2 lần sửa rubric → đổi model judge / cách chấm (hỏi người dùng).
- 02 KHÔNG hơn 01 trên sample25 với corpus gold (LB: +28 điểm) → bug harness/corpus.
- ≥ 6/25 câu mẫu không tìm được tài liệu nguồn mở → corpus local không tái tạo được đề.
- |D1 − 57.14| ≥ 2.86 → nhiễu LB lớn, delta 1 bài nộp không đọc được → dựa nhiều hơn vào phiên thuê 32B.
- Cùng một lỗi lặp lại 2 lần trong một stage.

## Stage 1: Harness chạy bài + bài nộp D1 — đang làm (người dùng duyệt 22/10)
- Mục tiêu: chạy bài như server (1 process, tuần tự, offline, đo thời gian); đóng gói + nộp D1 làm bài đối chứng.
- File: `eval/run_pipeline.py`, `eval/dev_shim.py`, `eval/compare_runs.py` (đã có, commit `057ff50`); `submissions/d1_02_safe/` + script build zip.
- Tiêu chí:
  - [x] 01_zero_shot sample25 → 25 dòng, answer là str, thời gian + VRAM + token (exp001_zs_4bit_a/b)
  - [x] Chạy `--offline` không lỗi
  - [x] Bài giả ném exception / trả non-str → ghi lỗi, chạy tiếp
  - [x] Dự phóng public/private + cảnh báo vượt giới hạn có trong run_info.json
  - [x] Local tất định: 2 lượt 4-bit 0/25 câu khác chữ
  - [~] Lệch shim: bf16 dừng sau 3 câu (VRAM tràn, GPU dùng chung) → 3/3 câu khác chữ so với 4-bit; đo đủ + verdict lật trên GPU thuê / sau Stage 2
  - [ ] D1 `02-safe` = 02 nguyên bản + chỉ vá không đổi hành vi trên corpus phẳng: tìm PDF đệ quy, không phân biệt hoa thường;
        try/except từng câu trả chuỗi fallback, KHÔNG load lại model khi lỗi (cache cả trạng thái lỗi); giữ prompt, 10k, 2048, greedy
        — code + zip xong (`submissions/d1_02_safe.zip`), chờ người dùng nộp
  - [x] Zip D1: `main.py` ở gốc, giải nén 435 MB, không chứa `eval/`/`dev_shim`; harness `--offline` từ bản giải nén, 2 PDF thật → 3/3 câu trả str (results/s1_d1_smoke)
- Bằng chứng: exp001_zs_4bit_a/b (run_info.json), compare_runs 0/25.

## Stage 2: Judge + metrics (gọn) — xong 22/10 (chờ ghi refs vào data/splits khi người dùng mở bảo vệ)
- File dự kiến: `eval/judge.py`, `eval/evaluate.py` (= run_pipeline + judge + metrics; `--judge-only`), `eval/calibrate_judge.py`
- Thiết kế: J1 = DeepSeek, temperature 0, JSON {verdict, reason}, cache theo hash. Input: câu hỏi + `reference_short` + câu trả lời.
  `reference_short` = atomic claim (giá trị + đơn vị + thực thể + cách xử lý tiền đề sai) + cờ `outside_doc / false_premise / closed_book`;
  LLM rút gọn từ gold dài + người duyệt, giới hạn 2 giờ.
- Tiêu chí (4 test hiệu chỉnh trên sample25):
  - [x] gold diễn đạt lại → đúng ≥ 24/25 — 25/25 (judge-v2, results/calibration/20261022-1705)
  - [x] gold đổi số/đơn vị/thực thể → bác ≥ 90% — 37/38 = 97% trên biến thể ĐƯỢC XÁC NHẬN có mâu thuẫn (GPT kiểm);
        thô 39/50 = 78% (12 biến thể không thật sự mâu thuẫn tham chiếu). Định nghĩa cổng làm rõ: chỉ tính biến thể hợp lệ — báo người dùng 22/10
  - [x] trả lời lạc đề → bác 25/25
  - [x] thiếu 1 vế của câu nhiều vế → bác ≥ 80% — 4/4 (chỉ 4 câu multi_part)
  - [x] Báo cáo: lật verdict long vs short 1/63 = 1.6%
  - reference_short: GPT soạn nháp → phiên chính duyệt, sửa 5 câu (#5, #16, #17, #21, #23 nhồi chi tiết ngoài trọng tâm) → `data/work/sample25_refs.jsonl`
  - Judge: `eval/judge.py` (DeepSeek, prompt judge-v2, cache `data/work/llm_cache.jsonl`), `eval/evaluate.py`, `eval/calibrate_judge.py`
- Rủi ro: judge ẩn khác hẳn → số local chỉ để so sánh tương đối.

## Stage 3: Tài liệu nguồn cho 25 câu mẫu — xong 22/10 (commit addac62)
- Kết quả: 18 PDF / 366 trang / 61 MB ở `data/corpus/gold/` (không commit; corpus ở `data/corpus/` vì `data/raw/**` được bảo vệ cho dữ liệu gốc BTC).
  `fetch.py check`: 20/25 câu có quote xác nhận (DONES 7/10, AI 13/15). Không tìm được bản mở: q1, q2, q4 (DONES), q18, q21 (AI);
  partial: q5 (chi tiết FISPACT-II ở bài đóng Frisoni 2020), q12, q23. Evidence: `data/work/sample25_evidence.jsonl`.
- Lệch gold vs tài liệu: q6 gold 250–273 °C, tài liệu 2025 ghi 318→300 °C; q9 gold dùng kích thước toà nhà IFMIF cũ (DONES 159 m);
  q3, q15 phần tính (ln2, chia L) ngoài tài liệu. Một số nguồn là bản OA ở kho trường (KIT, IRB, UGR, Tilburg) hoặc Wayback (MDPI, UHasselt).
- File dự kiến: `tools/corpus/fetch.py`, `data/manifests/corpus_gold.csv`, `evidence` trong `data/splits/sample25.jsonl`
- Cách làm: AI → arXiv/OpenReview/ACL; DONES → EUROfusion scipub, JACoW, PMC, KIT. Chỉ domain trong `sites.txt` (hoặc arXiv cùng bài), ghi license/OA.
- Tiêu chí:
  - [ ] ≥ 20/25 câu mẫu có tài liệu + `quote` tìm thấy trong text PyMuPDF (fuzzy ≥ 0.9)
  - [ ] Manifest: url, site, sha256, số trang, license
- Rủi ro: tài liệu DONES không công khai; q9 dùng số liệu toà nhà IFMIF cũ.

## Stage 3b: Chẩn đoán trên sample25 — chưa làm
- Mục tiêu: biết điểm mất ở retrieval hay generator trước khi tối ưu; kiểm cơ chế sliding-window.
- File dự kiến: `submissions/02_instrumented/main.py` (bản sao 02 lưu doc, page, token offset vào `LAST_DEBUG`), `docs/EXPERIMENTS.md`
- Cách làm (corpus gold + C50 nếu có, 7B 4-bit local; lặp lại trên 32B ở phiên thuê):
  (a) 01 vs 02 · (b) `quote_recall@k` + `in_window_hit` · (c) **oracle-chunk**: context thật vs chỉ đoạn chứa quote ·
  (d) **evidence-first vs evidence-last** · (e) closed-book theo domain.
- Tiêu chí:
  - [x] 02 > 01 trên sample25 — D1 38.34 (10/25) vs zero-shot 13.34 (3/25), +7 câu, 0 mất (exp-002)
  - [ ] Quyết định có số: oracle − thật ≥ 3 câu → ưu tiên retrieval; ≤ 1 câu → ưu tiên generator/format
  - [ ] Bảng phân loại lỗi (không tìm đúng tài liệu / sai đoạn / đúng đoạn nhưng sinh sai / judge sai) theo domain trong EXPERIMENTS.md

## Stage P1b-sớm: Phiên GPU thuê 2–3 giờ — chưa làm (cần người dùng thuê)
- Khi: sau Stage 3, khoảng 25–27/10. Máy: EzyCloudX 1x RTX PRO 6000 Blackwell 95 GB (105k CP/giờ), zip chạy nguyên `device_map="cuda"`.
- Việc: cài đúng requirements (Linux, có triton) → chạy 01 trên sample25 để hiệu chỉnh hệ số thời gian (so LB 733 s; decode H100 nhanh hơn ~1.9–2.1×)
  → chạy Stage 3b (a)–(d) bằng Olmo-3.1-32B → đo nhiễu 32B (2 lượt cùng zip).
- Tiêu chí: [ ] hệ số thời gian ghi vào AGENTS.md · [ ] kết quả 32B của Stage 3b trong EXPERIMENTS.md

## Stage 4: Corpus nhiễu — chưa làm (song song, không chặn)
- C50 ⊂ C200 chứa toàn bộ tài liệu gold; chỉ dùng cho thời gian index + retrieval-only. Nguồn: arXiv API, JACoW, EUROfusion, PMC OA; tải ≥ 3 s/request.
- Tiêu chí: [ ] PyMuPDF đọc ≥ 98% PDF · [ ] trang/PDF, % PDF không text, trang/giây PyMuPDF vs Docling + ngoại suy 5k/10k/20k trang

## Stage 5: Câu hỏi tổng hợp — HOÃN (P2b, sau ≥ 3 bài nộp)
- Chỉ retrieval-only (không chạy LLM sinh câu trả lời trên synthetic); không sinh từ tài liệu gold của sample25; `doc(dev) ∩ doc(sample25) = ∅`.
- Câu người viết: hoãn tới khi có người phụ trách.

## Lịch nộp (mỗi ngày 1 bài, người dùng upload)
- **D1 `02-safe`** (Stage 1): kiểm đóng gói, có bài hợp lệ, 1 mẫu nhiễu miễn phí (|D1 − 57.14|).
- **D2 `02-evidence-last`**: D1 + đảo thứ tự fragment (chunk tốt nhất sát câu hỏi). Thời gian không đổi.
- **D3 `answer-first-512`**: tốt nhất của D1/D2 + prompt answer-first (1–3 câu, số + đơn vị trước, đủ mọi vế, bác tiền đề sai)
  + `max_new_tokens` 512 → probe khẩu vị độ dài của judge ẩn + giảm thời gian; tụt ≥ 2.86 → hoàn tác.
- Probe số trang corpus chỉ nộp nếu D1 > 2700 s.

## Ước lượng thời gian
- Stage 1 + D1: 22–23/10 · Stage 2 gọn + Stage 3: 23–24/10 · Stage 3b + phiên GPU thuê: 25–27/10 · Stage 4 song song.

## Nhật ký quyết định
| Lượt | Sự kiện | Tiêu chí đạt | Quyết định | Lý do + bằng chứng |
|---|---|---|---|---|
| 0 | Lập plan v1, gửi hội đồng chấm | — | chờ council + người dùng | người dùng yêu cầu 22/10 |
| 1 | Council round_02: chỉ DeepSeek (key OpenRouter hết hạn) → 5/10, 6/10 | — | v2: cách ly sample25, thêm Stage 3b, quy tắc dùng split, judge thêm test | round_02 SYNTHESIS §5; rò rỉ đúng |
| 2 | Người dùng duyệt Stage 1 + nộp song song | — | worker code harness; phiên chính review, sửa ghi CSV từng dòng | commit `057ff50` |
| 3 | Council round_06 đủ 8 model: 5 model chấm 6–7.5 (TB 6.8, trung vị 7), cả 5 "sửa nhỏ rồi làm" | Stage 1: 5/7 | v3: D1 nộp ngay sau Stage 1; Stage 2 còn 4 test; Stage 5 hoãn; Stage 3b thêm oracle-chunk + evidence-last + in-window; thêm phiên GPU thuê sớm | round_06 SYNTHESIS §5; phiên chính kiểm: 32B có 48/64 lớp sliding 4096 (config.json); local 4-bit 0/25 câu khác |
