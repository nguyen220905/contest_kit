# STAGES — P2 + P3: eval harness + corpus dev (nguồn mở) — v2 sau council round_02
> /orchestrate tạo và cập nhật file này. Hook Stop đọc dòng "Trạng thái" và mục "Tiêu chí tổng" để quyết định vòng lặp chạy tiếp hay dừng.
> Chỉ criteria-checker đánh [x], kèm bằng chứng. Vòng trước: chưa có. Góp ý: `docs/council/round_02/SYNTHESIS.md` (chỉ DeepSeek trả lời — key OpenRouter hết hạn).

Trạng thái: lập kế hoạch
<!-- Các giá trị: lập kế hoạch · đang chạy · chờ người dùng — <câu hỏi> · dừng — <lý do> · xong -->

## Bối cảnh (2026-10-22)
- Điểm hiện tại: chưa nộp. LB: top 69.05, ngưỡng top 20 = 59.05; mốc BTC zero-shot 29.05 / 733 s, RAG 57.14 / 2080 s.
- Mục tiêu: có một thước đo local **tin được** để chọn thay đổi ở P4–P6 bằng số, không phí lượt nộp (1 bài/ngày, server không trả output).
- Ràng buộc: dev local Olmo-3-7B-Instruct 4-bit (9 tok/s, 4.9 GiB, RTX 5060 Ti 16 GB). bf16 (13.6 GiB) chỉ dùng cho 01_zero_shot:
  với 02_rag (context ~12k) KV cache ~3 GiB (24 lớp sliding 4096 + 8 lớp full) → vượt 16 GB. Olmo-32B chỉ chạy được trên H100 thuê.
  Dữ liệu thêm CHỈ từ nguồn mở. Judge chính thức ẩn nhưng tất định, chấm nhị phân, tham chiếu "deliberately concise".
- Đã loại (đừng lặp): nộp lại 01/02 nguyên bản; BGE-M3 / bge-reranker-v2-m3 (> 1 GB).
- Giả thuyết kích thước corpus thật (CHƯA kiểm): 02_rag tốn hơn zero-shot 2080 − 733 ≈ 1350 s; trừ prefill 10k token × 50 câu (~100–200 s)
  → ~1100 s parse Docling + embed MiniLM CPU → nếu Docling ~2–4 trang/s thì corpus ≈ 2.5–4.5k trang ≈ 150–400 PDF. Chỉ kiểm được bằng
  tốc độ Docling thật trên H100 (P1b).

## Quyền tự quyết & ngân sách
- Tự điều chỉnh stage chưa làm: có
- Tự đổi hướng (khi chạm dấu hiệu hướng sai): không — hỏi người dùng
- Tối đa 3 lượt sửa / stage · tối đa 40 lượt cho cả vòng lặp
- API: J1 + sinh câu hỏi = DeepSeek gọi thẳng (còn credit). J2 tuỳ chọn: OpenRouter khi key còn hạn (**key hết hạn 22/10 — người dùng cần thay**),
  nếu không thì DeepSeek với prompt/temperature khác.

## Thiết kế chung
- Bài nộp là thư mục có `main.py` (giống zip). Harness import đúng `main.py` đó và gọi `answer()` tuần tự trong 1 process như server.
  Không sửa code bài nộp để chạy local: `eval/dev_shim.py` vá `AutoModelForCausalLM.from_pretrained` khi `DSQA_DEV_QUANT=4bit`
  (chèn `BitsAndBytesConfig`) và harness truyền `model_id=allenai/Olmo-3-7B-Instruct`. Trên server không có shim.
- Hook tuỳ chọn cho chẩn đoán (server bỏ qua): `main.py` có thể có `retrieve(question, corpus_dir) -> list[dict(doc, page, score)]`
  (đo retrieval KHÔNG cần LLM → vòng lặp nhanh) và biến `LAST_DEBUG` (tài liệu đã dùng cho câu vừa trả lời).
- Định dạng split (`data/splits/*.jsonl`): `id, domain (DONES|AI Papers), question_type, question, reference_short, reference_long,
  evidence: [{doc_id, page, quote}], flags: {calc, false_premise, names_paper, closed_book_easy}, source (sample|synthetic|human)`.
- **Quy tắc dùng split (bắt buộc):** dev/holdout tổng hợp chỉ dùng để chọn **retrieval / chunking / thời gian**. Thay đổi ở **generator**
  (prompt, format, PoT…) phải thắng trên sample25 VÀ không thua trên holdout tổng hợp, rồi xác nhận bằng LB.
  Lý do: câu tổng hợp xếp hạng generator gần như ngẫu nhiên (van Elburg et al., arXiv 2508.11758, τ≈0).
- **Cách ly sample25:** tài liệu gold của 25 câu mẫu KHÔNG được dùng sinh câu dev; few-shot sinh câu dùng template phong cách trừu tượng,
  không dán nguyên văn câu/đáp án mẫu.
- Mọi I/O `encoding="utf-8"`; mỗi lượt chạy = 1 process mới.
- Git: `data/` đang bị ignore → đổi thành `data/*` + `!data/splits/` + `!data/manifests/` (PDF không commit, manifest + split commit).
  Sau Stage 6: bật `eval/**` trong `.claude/protected_paths.txt` (data/raw/** và data/splits/** đã bảo vệ sẵn).

## Tiêu chí tổng
- [ ] `.venv\Scripts\python eval/evaluate.py --submission starter_kit/02_rag --split holdout --corpus data/raw/corpus/C200 --run results/exp002_rag_c200`
      → `results/exp002_rag_c200/predictions.csv` + `metrics.json` có: domain-balanced acc, acc theo domain và theo question_type,
      số lỗi, thời gian (load, index, mỗi câu p50/p95, tổng, dự phóng public 50 câu / private 100 câu), số token sinh p50/p95, doc-hit@1/5/10.
- [ ] J1 qua hiệu chỉnh (Stage 2); J2 (nếu có) Cohen's kappa với J1 ≥ 0.8 trên 25 câu mẫu.
- [ ] `data/manifests/corpus_C{50,200}.csv` + `data/splits/{dev,holdout,sample25}.jsonl` đã commit; holdout tách theo TÀI LIỆU với dev;
      `doc(dev) ∩ doc(sample25) = ∅`.
- [ ] Baseline đo xong: 01_zero_shot và 02_rag trên dev/holdout/sample25 (7B 4-bit); ngưỡng nhiễu ghi vào AGENTS.md "Nhiễu đã đo";
      exp-001/exp-002 trong `docs/EXPERIMENTS.md`; bảng theo dõi tương quan local ↔ LB lập sẵn trong EXPERIMENTS.md.

## Dấu hiệu hướng sai (viết TRƯỚC khi làm)
- Judge J1 không qua hiệu chỉnh sau 2 lần sửa rubric → đổi model judge / cách chấm (hỏi người dùng).
- 02_rag KHÔNG hơn 01_zero_shot trên sample25 với corpus gold (Stage 3b; LB: +28 điểm) → bug harness/corpus.
- ≥ 6/25 câu mẫu không tìm được tài liệu nguồn mở → corpus local không tái tạo được đề; dựa nhiều hơn vào câu tổng hợp.
- Tỉ lệ closed-book đúng của câu tổng hợp lệch > 15 điểm % so với tỉ lệ đó trên sample25 (cùng model, cùng quy trình) → câu không giống đề.
- Thời gian index/trang local > 4× số liệu benchmark PyMuPDF/Docling → lỗi môi trường, dừng đo thời gian.
- Cùng một lỗi lặp lại 2 lần trong một stage.

## Stage 1: Harness chạy bài — chưa làm
- Mục tiêu: chạy bất kỳ thư mục bài nộp nào trên một split, đúng như server (1 process, tuần tự, offline, đo thời gian), ghi predictions.
- File dự kiến: `eval/run_pipeline.py`, `eval/dev_shim.py`
- Tiêu chí (lệnh → kết quả mong đợi):
  - [ ] `eval/run_pipeline.py --submission starter_kit/01_zero_shot --split sample25 --run results/exp001_zs` → 25 dòng, mọi answer là str,
        có thời gian load + từng câu, VRAM đỉnh, `n_new_tokens` + p50/p95 token sinh
  - [ ] Chạy với `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` không lỗi (giống server không internet)
  - [ ] Bài giả ném exception ở câu 3 → harness ghi lỗi câu 3, chạy tiếp, answer rỗng cho câu đó
  - [ ] `--budget-s 3600` báo "VƯỢT GIỚI HẠN" khi tổng thời gian dự phóng vượt; báo cả dự phóng private (100 câu, 7200 s mềm)
  - [ ] Lệch do shim: 01_zero_shot trên sample25 chạy 4-bit 2 lần (đếm câu khác chữ = nhiễu) + bf16 1 lần (đếm verdict lật sau Stage 2) → ghi "lệch shim"
- Rủi ro: starter kit load bf16 `device_map="cuda"` → shim phải chèn quant mà không đổi code; Docling/embedder 02_rag cần tải model về như `build.sh` (bản Windows).
  Olmo-7B sinh dài (smoke test: cả 3 câu chạm trần 256 token); trần 2048 ở 9 tok/s = 227 s/câu → lên lịch Stage 3b/6 theo token đo được.
- Bằng chứng:

## Stage 2: Judge + metrics + hiệu chỉnh — chưa làm
- Mục tiêu: chấm nhị phân giống judge ẩn nhất có thể, tất định, có số đo độ tin cậy.
- File dự kiến: `eval/judge.py`, `eval/evaluate.py` (= run_pipeline + judge + metrics; `--judge-only` để chấm lại), `eval/calibrate_judge.py`
- Thiết kế: Input: câu hỏi + `reference_short` + câu trả lời. Rubric: đúng nếu chứa đúng sự kiện/số chính của tham chiếu (làm tròn hợp lý,
  đơn vị tương đương), không mâu thuẫn; câu tiền đề sai được tính đúng khi bác tiền đề như tham chiếu. temperature 0, output JSON {verdict, reason},
  cache theo hash → chấm lại không tốn tiền. `reference_short` của 25 câu mẫu: LLM rút gọn từ gold dài + người duyệt.
- Tiêu chí:
  - [ ] Hiệu chỉnh J1 trên 25 câu mẫu: gold diễn đạt lại → đúng ≥ 24/25; gold bị đổi số/đơn vị/thực thể (2 biến thể/câu) → bác ≥ 90%;
        bản chỉ 1 câu trả lời thẳng → đúng ≥ 23/25; câu trả lời dài lạc đề → bác 25/25; câu né/mơ hồ → bác ≥ 90%;
        câu dài có đáp án đúng + 1 chi tiết mâu thuẫn → bác ≥ 80%
  - [ ] Cùng câu trả lời chấm với `reference_long` vs `reference_short` → verdict lật ≤ 10%
  - [ ] J2 (nếu có): Cohen's kappa với J1 ≥ 0.8; mọi câu bất đồng ghi vào `results/<run>/judge_disagreements.csv`
  - [ ] `eval/evaluate.py --run results/exp001_zs --judge-only` → metrics.json đủ trường như Tiêu chí tổng
- Rủi ro: judge ẩn có thể khác hẳn (model, rubric) → số local chỉ dùng để SO SÁNH TƯƠNG ĐỐI, không dự đoán điểm LB tuyệt đối.
- Bằng chứng:

## Stage 3: Tài liệu nguồn cho 25 câu mẫu — chưa làm
- Mục tiêu: corpus "gold" — tìm bản nguồn mở chứa đáp án của từng câu mẫu, định vị trang + trích đoạn.
- File dự kiến: `tools/corpus/fetch.py` (tải + sha256 + kiểm OA + ghi manifest), `data/manifests/corpus_gold.csv`, cập nhật `evidence` của `data/splits/sample25.jsonl`
- Cách làm: 13/15 câu AI nêu tên paper → bản arXiv/OpenReview/ACL; DONES → EUROfusion scipub, JACoW (IPAC/LINAC/IBIC), PMC (J. Fusion Energy), KIT.
  Chỉ tải từ domain trong `sites.txt` (hoặc bản arXiv của cùng bài), có ghi license/OA.
- Tiêu chí:
  - [ ] ≥ 20/25 câu mẫu có tài liệu nguồn + `quote` tìm thấy trong text PyMuPDF của PDF (fuzzy ≥ 0.9)
  - [ ] Mọi dòng manifest có url, site ∈ sites.txt (hoặc arXiv), sha256, số trang, license
- Rủi ro: một số tài liệu DONES không công khai; gold q9 dùng kích thước toà nhà IFMIF (có thể từ tài liệu IFMIF cũ, không phải DONES).
- Bằng chứng:

## Stage 3b: Baseline sớm trên sample25 — chưa làm
- Mục tiêu: kiểm hợp lý harness + corpus sớm, không đợi Stage 6; biết lỗi của 02_rag nằm ở retrieval hay generator.
- File dự kiến: `docs/EXPERIMENTS.md` (exp-001, exp-002)
- Cách làm: 01_zero_shot và 02_rag (code BTC nguyên bản, qua shim) trên sample25 với corpus = chỉ tài liệu gold (+ C50 nếu đã có);
  retrieval-only doc-hit@k; phân loại lỗi 02_rag theo 4 nhóm (không tìm đúng tài liệu / tìm đúng nhưng sai đoạn / đúng đoạn nhưng sinh sai / judge sai) và theo domain.
- Tiêu chí:
  - [ ] 02_rag > 01_zero_shot trên sample25 (kiểm "dấu hiệu hướng sai")
  - [ ] Bảng phân loại lỗi trong EXPERIMENTS.md
- Bằng chứng:

## Stage 4: Corpus nhiễu + quy mô — chưa làm
- Mục tiêu: corpus giống đề (tài liệu thật + nhiễu cùng chủ đề) ở 2 quy mô lồng nhau C50 ⊂ C200 (đều chứa toàn bộ tài liệu gold);
  C500 chỉ để đo thời gian index, không chặn Tiêu chí tổng.
- File dự kiến: `tools/corpus/fetch.py` (thêm nguồn: arXiv API, JACoW, EUROfusion, PMC OA), `data/manifests/corpus_C{50,200}.csv`
- Cách làm: nhiễu cùng chủ đề (máy gia tốc/LIPAc/ESS/chẩn đoán chùm, nhiệt hạch WPENS; LLM/CI/fuzzy/tiến hoá, nhóm DaSCI Granada),
  có cả bản trùng arXiv vs publisher (nhiễu thật). Tải lịch sự (arXiv ≥ 3 s/request), chỉ open access.
- Tiêu chí:
  - [ ] PyMuPDF đọc được ≥ 98% PDF; ghi trang/PDF trung bình, % PDF không có text
  - [ ] Trang/giây của PyMuPDF và Docling ở C50/C200 + ngoại suy thời gian index tới 5k/10k/20k trang
  - [ ] Thống kê domain/site của từng quy mô trong manifest
- Rủi ro: tốn thời gian tải; PDF scan (không có text) → loại hoặc đánh dấu.
- Bằng chứng:

## Stage 5: Câu hỏi tổng hợp + split — chưa làm
- Mục tiêu: 60–80 câu tổng hợp đã kiểm + 15–20 câu người viết, đủ để chọn retrieval/chunking (25 câu mẫu quá ít: 1 câu = 4 điểm).
- File dự kiến: `tools/corpus/synth_qa.py`, `tools/corpus/validate_qa.py`, `data/splits/{dev,holdout,sample25}.jsonl`
- Cách làm: LLM mạnh (DeepSeek) đọc 1 section/bảng của tài liệu trong C200 **ngoài tài liệu gold của sample25**, few-shot bằng template
  phong cách trừu tượng; phân bố question_type + tỉ lệ "nêu tên paper" (AI ~13/15) giống mẫu; có câu tính toán, ~5% câu tiền đề sai.
  Kiểm: (1) quote có trong text PDF; (2) model thứ 2 trả lời chỉ từ đoạn evidence → judge "đúng"; (3) closed-book: model mạnh không có tài liệu
  → nếu đúng thì gắn `closed_book_easy`; (4) người duyệt 20 câu ngẫu nhiên; (5) loại câu có cosine > 0.85 với bất kỳ câu mẫu nào.
  Split seed 2026: dev / holdout tách theo TÀI LIỆU; sample25 = holdout cuối, không tune trên nó.
- Tiêu chí:
  - [ ] ≥ 60 câu tổng hợp qua kiểm (1)+(2)+(5); phân bố question_type và "nêu tên paper" lệch ≤ 10 điểm % so với mẫu
  - [ ] Tỉ lệ `closed_book_easy` lệch ≤ 15 điểm % so với tỉ lệ closed-book của cùng model trên sample25
  - [ ] Script kiểm: `doc(dev) ∩ doc(holdout) = ∅` và `doc(dev) ∩ doc(sample25) = ∅`
- Rủi ro: câu do LLM sinh dễ hơn / khác phong cách câu chuyên gia → chỉ dùng cho retrieval (quy tắc ở Thiết kế chung); kiểm lại khi có 32B (P1b).
  Câu người viết: cần người dùng/đội viết hoặc duyệt — chưa có người phụ trách.
- Bằng chứng:

## Stage 6: Đo baseline đầy đủ + chốt eval — chưa làm
- Mục tiêu: số baseline + ngưỡng nhiễu + thời gian theo quy mô corpus; khoá thước đo.
- File dự kiến: `docs/EXPERIMENTS.md`, `AGENTS.md` (mục Nhiễu đã đo + lệnh chuẩn), `.claude/protected_paths.txt` (bật `eval/**`)
- Cách làm: chạy 01_zero_shot và 02_rag trên dev/holdout/sample25 với C200 (lên lịch theo số token đo ở Stage 1; chạy đêm);
  chạy lại 02_rag trên holdout lần 2 không đổi gì → đếm câu lật (nhiễu generation); retrieval-only (doc-hit@k) không cần LLM.
- Tiêu chí:
  - [ ] 02_rag > 01_zero_shot trên cả 3 split
  - [ ] Ngưỡng ra quyết định (số câu lật) ghi vào AGENTS.md; exp-001/002 trong EXPERIMENTS.md
  - [ ] Bảng tương quan local ↔ LB trong EXPERIMENTS.md (cập nhật sau mỗi bài nộp: điểm local sample25/holdout vs LB)
  - [ ] `eval/**` bật trong protected_paths
- Sau đó (P1b, khi thuê H100): chạy 01/02 bằng 32B trên sample25 + holdout → nếu thứ tự/khoảng cách lệch xa LB (29 / 57) thì xem lại bộ câu tổng hợp.
- Bằng chứng:

## Ước lượng thời gian
- Stage 1–2: 1 ngày · Stage 3 + 3b: 1–1.5 ngày · Stage 4: 1 ngày · Stage 5: 1 ngày · Stage 6: 1 ngày + máy chạy (7B 4-bit 30–227 s/câu
  tuỳ độ dài sinh → có thể nhiều giờ/lượt; chạy đêm hoặc trên H100). Tổng ~5–6 ngày → xong khoảng 28/10.
- Song song (đề xuất, cần người dùng đồng ý — xem PLAN): từ khi Stage 1 xong, mỗi ngày nộp 1 biến thể nhỏ của P4 để không phí lượt nộp.

## Nhật ký quyết định
| Lượt | Sự kiện | Tiêu chí đạt | Quyết định | Lý do + bằng chứng |
|---|---|---|---|---|
| 0 | Lập plan P2+P3 v1, gửi hội đồng chấm | — | chờ council round_02 + người dùng duyệt | người dùng yêu cầu 22/10 |
| 1 | Council round_02: chỉ DeepSeek (key OpenRouter hết hạn); 4 vai chấm 5/10, 6/10, 2 vai không chấm; researcher 6.5 → "sửa nhỏ rồi làm" | — | Sửa thành v2: cách ly sample25, thêm Stage 3b, quy tắc dùng split, test offline/token/lệch shim, judge thêm kappa + leniency, C500 tuỳ chọn, Stage 5 giảm còn 60–80 câu | SYNTHESIS round_02 §5; phiên chính tự kiểm: rò rỉ đúng (v1 dòng sinh câu từ tài liệu gold); KV cache 7B bf16 thực ~3 GiB ở 12k (24 lớp sliding) — vẫn vượt 16 GB |
