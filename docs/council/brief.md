# BRIEF cho hội đồng model — round 02 (CHẤM plan P2+P3: eval harness + corpus dev)
> Hôm nay 22/10/2026. Chưa có code eval, chưa có corpus, chưa nộp bài nào. Nhiệm vụ của hội đồng: chấm plan dưới đây, chỉ ra sửa gì trước khi triển khai.

## 1. Bài toán
- Cuộc thi IEEE CIS Technical Challenge on Domain-Specific QA with LLMs (Codabench #17954). Nộp zip, `main.py` ở gốc có `answer(question, corpus_dir, model_id) -> str`, gọi tuần tự 1 lần/câu trong CÙNG process (được cache model/index). Corpus = thư mục PDF (tài liệu thật + nhiễu), retrieval mù. 2 domain: IFMIF-DONES (máy gia tốc deuteron/bia lithium) và bài báo AI.
- Metric: domain-balanced accuracy, LLM judge ẩn chấm **nhị phân** so với tham chiếu "deliberately concise". Judge **tất định** (BTC xác nhận: output giống → điểm giống). Hoà điểm → bài nhanh hơn thắng. Server **không trả output nào**, chỉ điểm + thời gian + lỗi ngắn.
- Public LB = 50 câu, chia **15/35** giữa 2 domain (mọi điểm LB khớp 50·(a/15+b/35)) → 1 câu = 3.33 hoặc 1.43 điểm. Private = 100 câu.
- LB 22/10: top 69.05; ngưỡng top 20 = 59.05. Mốc BTC: zero-shot 29.05 / 733 s; RAG (02_rag: Docling + MiniLM CPU + ctx 10k + max_new_tokens 2048, greedy) 57.14 / 2080 s.
- Ràng buộc: 1 bài/ngày; public đóng 30/11; 1h cứng (public, gồm load model + parse PDF); private 2h mềm (câu chưa xong = mất điểm). Server 1×H100 80GB, 32 CPU. Zip ≤ 1 GB. Không internet. LLM cấp sẵn Olmo-3.1-32B-Instruct; được bundle model bất kỳ (kể cả LLM khác) trong 1 GB. Kích thước corpus thật KHÔNG công bố; BTC gợi ý tự dựng từ `sites.txt` (32 domain: arxiv, jacow, scipub.euro-fusion, pmc, openreview, springer…) và tăng dần.
- Dev: 25 câu mẫu có gold (DONES 10, AI 15; loại: causal 6, factual 7, procedural 5, predictive 4, comparative 3). Gold dài 81–2677 ký tự (trung vị ~670) — KHÁC tham chiếu ẩn "concise". 13/15 câu AI nêu tên paper. Gold dùng cả kiến thức ngoài tài liệu (ln2, khối lượng riêng Li), có câu tiền đề sai (q7 "on-line nitrogen trap" — thực ra off-line). Không có PDF kèm.
- Local: RTX 5060 Ti 16 GB, venv khớp server (torch 2.8.0+cu128, transformers 5.17.0, Windows). Olmo-3-7B-Instruct 4-bit NF4: 9 tok/s, 4.9 GiB; bf16 11 tok/s, 13.6 GiB. Olmo-32B chỉ chạy trên H100 thuê (người dùng có ngân sách nhưng muốn thuê SAU khi cải tiến baseline). OpenRouter không có Olmo. API: DeepSeek gọi thẳng (còn credit); OpenRouter hạn mức $7/ngày.
- Phase 2 (top 20): 30% accuracy + 30% chuyên gia chấm chất lượng + 40% technical innovation (Computational Intelligence).

## 2. Plan cần chấm — thiết kế chung (STAGES P2+P3)
- Mục tiêu: thước đo local **tin được** để chọn mọi thay đổi ở P4–P6 bằng số, không tốn lượt nộp.
- Harness import đúng thư mục bài nộp (`main.py`), gọi `answer()` tuần tự trong 1 process như server. Không sửa code bài nộp: `eval/dev_shim.py` vá `AutoModelForCausalLM.from_pretrained` khi `DSQA_DEV_QUANT=4bit` (chèn BitsAndBytesConfig), harness truyền `model_id=Olmo-3-7B-Instruct`.
- Hook chẩn đoán tuỳ chọn (server bỏ qua): `retrieve(question, corpus_dir) -> [{doc, page, score}]` để đo retrieval không cần LLM; biến `LAST_DEBUG`.
- Split jsonl: `id, domain, question_type, question, reference_short, reference_long, evidence[{doc_id,page,quote}], flags{calc,false_premise,closed_book_easy}, source(sample|synthetic)`.
- Git: commit manifest + split, không commit PDF. Sau Stage 6 khoá `eval/**`.
- **Tiêu chí tổng**: (1) `evaluate.py --submission 02_rag --split holdout --corpus C200` ra predictions.csv + metrics.json (balanced acc, theo domain/question_type, số lỗi, thời gian load/index/p50/p95/tổng + dự phóng 50/100 câu, doc-hit@1/5/10); (2) cả 2 judge qua hiệu chỉnh, đồng ý J1–J2 ≥ 90% trên 25 câu mẫu; (3) manifest C50/C200/C500 + split dev/holdout/sample25 commit, holdout tách theo TÀI LIỆU; (4) baseline 01/02 trên 3 split (7B 4-bit), ngưỡng nhiễu ghi vào AGENTS.md.
- **Dấu hiệu hướng sai** (viết trước): judge không qua hiệu chỉnh sau 2 lần sửa rubric; 02_rag không hơn 01 trên holdout; ≥ 6/25 câu mẫu không tìm được tài liệu nguồn mở; > 60% câu tổng hợp trả lời được closed-book; thời gian index/trang > 4× benchmark; cùng lỗi lặp 2 lần.
- Giả thuyết corpus thật (chưa kiểm): 02_rag tốn hơn zero-shot ≈ 1350 s; trừ prefill 10k×50 (~100–200 s) → ~1100 s Docling + embed CPU → nếu Docling 2–4 trang/s thì ≈ 2.5–4.5k trang ≈ 150–400 PDF.

## 3. Sáu stage (tiêu chí = lệnh → kết quả)
**S1 Harness chạy bài** (`eval/run_pipeline.py`, `dev_shim.py`): 01_zero_shot trên sample25 → 25 dòng, mọi answer là str, thời gian load + từng câu, VRAM đỉnh; bài giả ném exception ở câu 3 → ghi lỗi, chạy tiếp; `--budget-s 3600` báo vượt + dự phóng private (100 câu/7200 s). Rủi ro: starter load bf16 `device_map="cuda"`; Docling/embedder 02_rag cần tải model (bản Windows).

**S2 Judge + metrics + hiệu chỉnh** (`judge.py`, `evaluate.py` có `--judge-only`, `calibrate_judge.py`): J1 = DeepSeek (API thẳng), J2 = 1 model OpenRouter (GPT/Gemini). Input: câu hỏi + `reference_short` + câu trả lời. Rubric: đúng nếu chứa đúng sự kiện/số chính (làm tròn hợp lý, đơn vị tương đương), không mâu thuẫn; tiền đề sai → đúng khi bác tiền đề. temperature 0, JSON {verdict, reason}, cache theo hash. `reference_short` của 25 câu mẫu: LLM rút gọn gold + người duyệt. Tiêu chí cho cả J1, J2: gold diễn đạt lại → đúng ≥ 24/25; gold đổi số/đơn vị/thực thể (2 biến thể/câu) → bác ≥ 90%; bản 1 câu trả lời thẳng → đúng ≥ 23/25; câu dài lạc đề → bác 25/25; đồng ý J1–J2 ≥ 90%, bất đồng ghi csv. Chỉ dùng để SO SÁNH TƯƠNG ĐỐI, không dự đoán điểm LB.

**S3 Tài liệu nguồn cho 25 câu mẫu** (`tools/corpus/fetch.py`, manifest gold): AI → arXiv/OpenReview/ACL theo tên paper; DONES → EUROfusion scipub, JACoW, PMC, KIT. Chỉ domain trong sites.txt (hoặc arXiv cùng bài), ghi sha256/license. Tiêu chí: ≥ 20/25 câu có tài liệu + `quote` khớp text PyMuPDF (fuzzy ≥ 0.9). Rủi ro: tài liệu DONES không công khai; gold q9 dùng kích thước toà nhà IFMIF cũ.

**S4 Corpus nhiễu + 3 quy mô lồng nhau C50 ⊂ C200 ⊂ C500**, đều chứa toàn bộ tài liệu gold; nhiễu cùng chủ đề (LIPAc/ESS/chẩn đoán chùm/WPENS; LLM/fuzzy/tiến hoá/nhóm DaSCI Granada), có bản trùng arXiv vs publisher. Tiêu chí: PyMuPDF đọc ≥ 98% PDF; tổng trang mỗi quy mô; thống kê domain/site.

**S5 Câu hỏi tổng hợp + split** (`synth_qa.py`, `validate_qa.py`): DeepSeek đọc 1 section/bảng của tài liệu gold + ~60 tài liệu khác trong C200 (coi là "tài liệu thật"), **few-shot bằng 25 câu mẫu**; phân bố question_type giống mẫu; có câu tính toán, ~5% tiền đề sai. Kiểm: (1) quote có trong PDF; (2) model thứ 2 trả lời chỉ từ evidence → judge đúng; (3) closed-book đúng → gắn `closed_book_easy`; (4) người duyệt 20 câu. Split seed 2026: dev ~100 / holdout ~50 tách theo tài liệu; **sample25 = holdout cuối**, không tune. Tiêu chí: ≥ 120 câu qua (1)+(2); closed_book_easy ≤ 40%; question_type lệch ≤ 10 điểm %; dev ∩ holdout không chung tài liệu.

**S6 Baseline + chốt eval**: 01 và 02 (code BTC nguyên bản, qua shim) trên dev/holdout/sample25 với C200; thời gian index C50/C200/C500; chạy lại 02 trên holdout lần 2 → đếm câu lật (nhiễu); retrieval-only doc-hit@k. Tiêu chí: 02 > 01 trên cả 3 split; ngưỡng (số câu lật) ghi AGENTS.md; khoá `eval/**`. Sau đó (P1b, khi thuê H100): chạy 01/02 bằng 32B trên sample25 + holdout, lệch xa LB (29/57) → xem lại câu tổng hợp.

**Thời gian**: S1–2 1 ngày · S3 1 · S4 1 · S5 1–2 · S6 1 + máy chạy (7B 4-bit ~30–60 s/câu khi sinh 300–500 token → ~150 câu ≈ 1.5–2.5 h/lượt, chạy đêm). Tổng 5–6 ngày → xong ~28/10; bài nộp #1 (P4) ~30/10. (Lưu ý dữ kiện: 01/02 của BTC đặt max_new_tokens = 2048, greedy.)

## 4. Đã thử
Chưa có thí nghiệm. Đã loại: nộp lại 01/02 nguyên bản (LB đã có điểm BTC); BGE-M3 / bge-reranker-v2-m3 (> 1 GB); Docling table structure (server thiếu cv2).

## 5. Phân tích lỗi
Chưa có predictions. Baseline 02_rag dễ vỡ: `glob("*.pdf")` không đệ quy, `answer()` không try/except.

## 6. Plan sau P2+P3 (để hội đồng thấy bối cảnh)
P4 bài nộp #1 "fast-safe" (PyMuPDF đa tiến trình + find_tables, embed GPU, BM25+dense RRF, title boost, header chunk, ctx ~5k, prompt answer-first, max_new_tokens ~640, watchdog; bác bỏ nếu LB ≤ 57.14 hoặc > 1500 s) → P5 A/B định dạng câu trả lời + PoT cho câu tính toán → P6 mô-đun CI (fuzzy fusion tín hiệu retrieval, trọng số DE/CMA-ES offline; GEPA/EvoPrompt) → P1b thuê H100 chạy 32B trước khi nộp.

## 7. Tài liệu tham khảo (researcher đã mở)
- van Elburg et al., Can we Evaluate RAGs with Synthetic Data? (arXiv 2508.11758) — câu tổng hợp xếp hạng đúng cấu hình **retriever** (Kendall τ 0.44–0.75) nhưng **sai khi so generator/LLM** (τ ≈ 0 hoặc âm) do lệch tác vụ + bias phong cách.
- Krumdick et al., No Free Labels (arXiv 2503.05061) — tham chiếu đúng nâng kappa judge 0.14 → 0.78 trên câu khó; tham chiếu sai hại hơn không có; 35% tham chiếu MT-Bench toán sai.
- Thakur et al., Judging the Judges (GEM 2025, arXiv 2406.12624) — dùng Cohen's kappa, không chỉ % đồng ý; judge dễ dãi, ưa câu dài/mơ hồ; judge yếu vẫn xếp hạng hệ đúng.
- Badshah & Sajjad, Reference-Guided Verdict (arXiv 2408.09235) — ensemble nhiều judge ổn định hơn 1 judge.
- Shashidhar et al., YourBench (arXiv 2504.01833) — sinh câu từ tài liệu + lọc citation grounding; giữ thứ hạng MMLU, ~85% câu đạt khi người duyệt.
- Saad-Falcon et al., ARES (NAACL 2024) — judge tự động + prediction-powered inference với ít nhãn người.
- Rahmani et al., Synthetic Test Collections for Retrieval Evaluation (SIGIR 2024) — query/judgment tổng hợp xếp hạng hệ gần test collection người làm.
- Filice et al., DataMorgana (ACL 2025 Industry) — cấu hình phân bố loại câu/người hỏi để câu đa dạng.

## 8. Câu hỏi cụ thể cho hội đồng
1. **CHẤM plan thang 1–10** và kết luận đúng một trong ba: "đủ tốt để làm" / "sửa nhỏ rồi làm" / "sửa lớn". Nêu 3 sửa quan trọng nhất (stage nào, thay gì, cụ thể).
2. Phạm vi & thứ tự: gì thừa/cắt được để nộp bài #1 sớm hơn ~30/10? Có nên làm song song P4? Nên bỏ/đơn giản hoá gì (2 judge? C500? 150 câu tổng hợp?).
3. Câu tổng hợp do LLM sinh có đại diện cho đề chuyên gia không? Kiểm tính đại diện thế nào khi chưa chạy được Olmo-32B? Phương án tốt hơn (mở rộng thủ công, cách lấy câu khác)? Rủi ro rò rỉ khi dùng 25 câu mẫu làm few-shot sinh câu trong khi 25 câu mẫu là holdout cuối?
4. Judge: thiết kế (J1 DeepSeek + J2 OpenRouter, tham chiếu ngắn tự rút gọn, rubric, cache), ngưỡng hiệu chỉnh có hợp lý không? Cách mô phỏng judge ẩn sát hơn?
5. Giả thuyết corpus thật 150–400 PDF và thang C50/C200/C500 — hợp lý không, nên đo gì thêm?
6. Có nên thuê H100 SỚM (trong P2/P3) để (a) chạy eval nhanh hơn 7B 4-bit 9 tok/s, (b) đối chiếu dev set với LB bằng 32B (29.05 / 57.14) — so với chờ đến sau P4? Nêu lợi/hại có số.
7. Bước kiểm chứng còn thiếu, rủi ro bỏ sót (bug harness, Windows vs Linux server, shim 4-bit làm sai hành vi so với bf16, độ dài sinh 2048 token…)?
8. 5–6 ngày có thực tế không?
Trả lời cụ thể, có số; không nhắc lại brief.
