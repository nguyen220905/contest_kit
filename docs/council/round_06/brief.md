# BRIEF cho hội đồng model — round 06 (CHẤM LẠI plan P2+P3 v2: eval harness + corpus dev)
> Hôm nay 22/10/2026 (bản cập nhật: thêm EzyCloudX, Ollama 0.40.8). Chưa có code eval, chưa có corpus, chưa nộp bài nào. Plan v1 đã được 1 model (DeepSeek, 4 vai) chấm 5/10 và 6/10 → sửa thành v2 (9 sửa, xem mục 6). Nhiệm vụ: chấm v2, chỉ ra sửa gì còn cần trước khi làm.

## 1. Bài toán
- IEEE CIS Technical Challenge on Domain-Specific QA with LLMs (Codabench #17954). Nộp zip, `main.py` ở gốc có `answer(question, corpus_dir, model_id) -> str`, gọi tuần tự 1 lần/câu trong CÙNG process (được cache model/index). Corpus = thư mục PDF (tài liệu thật + nhiễu), retrieval mù. 2 domain: IFMIF-DONES (máy gia tốc deuteron 40 MeV/125 mA, bia lithium) và bài báo AI.
- Metric: domain-balanced accuracy, LLM judge ẩn chấm **nhị phân** so với tham chiếu "deliberately concise". Judge **tất định** (output giống → điểm giống); dao động giữa 2 lần nộp cùng zip = do generation không tất định. Hoà → bài nhanh hơn thắng. Server **không trả output**, chỉ điểm + thời gian + lỗi ngắn.
- Public LB = 50 câu chia **15/35** giữa 2 domain (mọi điểm LB khớp 50·(a/15+b/35)) → 1 câu = 3.33 hoặc 1.43 điểm. Private = 100 câu, BTC tự chấm lại bài public tốt nhất.
- LB 22/10: top 69.05; ngưỡng top 20 = 59.05. Mốc BTC: zero-shot 29.05 / 733 s; 02_rag (Docling GPU-auto + MiniLM CPU + chunk 800 ký tự + ctx 10k token + max_new_tokens 2048, greedy, system prompt "based on your knowledge") 57.14 / 2080 s.
- Ràng buộc: **1 bài/ngày**, public đóng 30/11 (~39 lượt còn lại); bài cuối = bài public điểm cao nhất (nộp bài kém không bị trừ). 1h cứng public (gồm load model + parse PDF); private 2h mềm (câu chưa xong = mất điểm). Server 1×**H100 80GB**, 32 CPU, 128 GB RAM, Linux, không internet. Zip ≤ 1 GB. LLM cấp sẵn Olmo-3.1-32B-Instruct (bf16, transformers 5.17.0, sdpa/eager, không vLLM/flash-attn); được bundle model bất kỳ trong 1 GB. Kích thước corpus thật KHÔNG công bố; BTC gợi ý tự dựng từ `sites.txt` (32 domain: arxiv, jacow, scipub.euro-fusion, pmc, openreview, springer…) và tăng dần.
- Dev: 25 câu mẫu có gold (DONES 10, AI 15; causal 6, factual 7, procedural 5, predictive 4, comparative 3). Gold dài 81–2677 ký tự (trung vị ~670) — KHÁC tham chiếu ẩn "concise". 13/15 câu AI nêu tên paper; câu DONES chung chung. Gold dùng cả kiến thức ngoài tài liệu (ln2, khối lượng riêng Li), có câu tiền đề sai. Không có PDF kèm.
- Phase 2 (top 20): 30% accuracy + 30% chuyên gia chấm chất lượng + 40% technical innovation (Computational Intelligence).

## 2. Môi trường & dữ kiện mới (22/10)
- Local: RTX 5060 Ti 16 GB, Windows, `.venv` khớp server (torch 2.8.0+cu128, transformers 5.17.0; thiếu triton vì không có wheel Windows). Dev bằng Olmo-3-7B-Instruct (cùng `Olmo3ForCausalLM`, vocab 100278, ctx 65536; 24 lớp sliding-window 4096 + 8 lớp full; 7B MHA 32/32 head, 32B GQA 40/8 → bộ nhớ/tốc độ 7B KHÔNG suy ra 32B). HF + bitsandbytes 4-bit NF4: **9 tok/s**, 4.9 GiB; bf16 11 tok/s, 13.6 GiB → 02_rag (~12k token) ở bf16 vượt 16 GB nên phải 4-bit. Smoke test: cả 3 câu chạm trần 256 token → 7B sinh dài; trần 2048 ở 9 tok/s = 227 s/câu.
- **Nộp song song: người dùng ĐỒNG Ý** — từ khi harness Stage 1 chạy được, mỗi ngày nộp 1 biến thể nhỏ của P4.
- **Ollama 0.40.8** vừa cài trên máy local (chưa tải model). Thư viện Ollama có `olmo-3:7b-instruct` (Q4_K_M 4.5 GB, mặc định temperature 0.6/top_p 0.95) và `olmo-3.1:32b-instruct` (Q4, 19 GB — không vừa trọn 16 GB VRAM, phải offload CPU). Context mặc định Ollama 4096 (phải đặt `num_ctx`); template chat của Ollama riêng, khác HF `apply_chat_template` có thể khác.
- **Thuê GPU: EzyCloudX** (Docker GPU, tính giờ, dữ liệu mất khi xoá container). KHÔNG có H100 hay A100 80GB. Gói liên quan (CP/giờ, CP≈VND): 1× RTX PRO 6000 Blackwell 95 GB (EPYC 9654 48 core, 108–190 GB RAM) 105k · 2× A100-SXM4-40GB (80 GB tổng, EPYC 7642 48 core, 120 GB RAM) 47.5k · 1× A100-40GB 24k · RTX PRO 6000 MIG 48 GB 50.5k · RTX 5090 32 GB 52.5k · 3× V100-32GB 32k. Olmo-3.1-32B bf16 ~64 GB → chỉ RTX PRO 6000 (1 GPU) hoặc 2× A100-40GB (chia lớp, `device_map="auto"` tuần tự, khác `device_map="cuda"` của server). Thông số (datasheet): H100 SXM 3.35 TB/s, BF16 dense 989 TFLOPS · RTX PRO 6000 Workstation 1.79 TB/s / Server Edition ~1.6 TB/s, BF16 dense ~500 · A100-40GB SXM 1.555 TB/s, 312. Trần decode bs=1 (đọc 64 GB/token): H100 ~52 tok/s, PRO 6000 ~25–28, 2×A100 ~24 (tuần tự). torch 2.8.0+cu128 có sm_120. Phiên thuê sớm 2–3 h **chưa duyệt**.
- API: OpenRouter key mới (không có Olmo trên OpenRouter); DeepSeek gọi thẳng còn credit. Local HF+bnb 4-bit 7B chỉ 9 tok/s.

## 3. Kết quả
Chưa có kết quả local nào, chưa nộp. Nhiễu chưa đo.

## 4. Đã thử / đã loại
Nộp lại 01/02 nguyên bản (LB đã có điểm BTC); BGE-M3 / bge-reranker-v2-m3 (> 1 GB); Docling table structure (server thiếu cv2); bnb 8-bit (2.8 tok/s).

## 5. Phân tích lỗi
Chưa có predictions. 02_rag dễ vỡ: `glob("*.pdf")` không đệ quy, `answer()` không try/except. (Corpus thật phẳng: nếu không thì 02 đã ≈ zero-shot.)

## 6. Plan v2 cần chấm (STAGES P2+P3)
**Thiết kế chung.** Harness import đúng thư mục bài nộp, gọi `answer()` tuần tự trong 1 process mới mỗi run. Không sửa code bài: `eval/dev_shim.py` vá `AutoModelForCausalLM.from_pretrained` khi `DSQA_DEV_QUANT=4bit` (chèn BitsAndBytesConfig), harness truyền `model_id=Olmo-3-7B-Instruct`. Hook tuỳ chọn: `retrieve(question, corpus_dir) -> [{doc,page,score}]` (đo retrieval không cần LLM), `LAST_DEBUG`. Split jsonl: `id, domain, question_type, question, reference_short, reference_long, evidence[{doc_id,page,quote}], flags{calc,false_premise,names_paper,closed_book_easy}, source(sample|synthetic|human)`. **Quy tắc dùng split:** dev/holdout tổng hợp chỉ để chọn **retrieval/chunking/thời gian**; thay đổi **generator** (prompt/format/PoT) phải thắng trên sample25 VÀ không thua trên holdout tổng hợp, rồi xác nhận bằng LB (lý do: van Elburg 2025, synthetic xếp hạng generator τ≈0). **Cách ly sample25:** tài liệu gold của 25 câu mẫu không dùng sinh câu dev; few-shot sinh câu = template phong cách trừu tượng. UTF-8 mọi I/O. Commit manifest + split, không commit PDF; khoá `eval/**` sau Stage 6.

**Tiêu chí tổng:** (1) `evaluate.py --submission 02_rag --split holdout --corpus C200` → predictions.csv + metrics.json (balanced acc, theo domain/question_type, số lỗi, thời gian load/index/p50/p95/tổng + dự phóng 50/100 câu, token sinh p50/p95, doc-hit@1/5/10); (2) J1 qua hiệu chỉnh; J2 (nếu có) Cohen's kappa ≥ 0.8 trên 25 câu mẫu; (3) manifest C50/C200 + split dev/holdout/sample25 commit, holdout tách theo TÀI LIỆU, `doc(dev) ∩ doc(sample25) = ∅`; (4) baseline 01/02 trên 3 split (7B 4-bit), ngưỡng nhiễu ghi AGENTS.md, bảng tương quan local ↔ LB lập sẵn.

**Dấu hiệu hướng sai:** J1 không qua hiệu chỉnh sau 2 lần sửa rubric; 02 KHÔNG hơn 01 trên sample25 với corpus gold (Stage 3b); ≥ 6/25 câu mẫu không tìm được tài liệu nguồn mở; tỉ lệ closed-book đúng của câu tổng hợp lệch > 15 điểm % so với trên sample25 (cùng model); thời gian index/trang > 4× benchmark; cùng lỗi lặp 2 lần.

**S1 Harness chạy bài** (`run_pipeline.py`, `dev_shim.py`): 01 trên sample25 → 25 dòng, mọi answer là str, thời gian load + từng câu, VRAM đỉnh, `n_new_tokens` p50/p95; chạy với `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; bài giả ném exception ở câu 3 → ghi lỗi, chạy tiếp; `--budget-s 3600` báo vượt + dự phóng private; **lệch shim**: 01 4-bit chạy 2 lần (đếm câu khác chữ) + bf16 1 lần (đếm verdict lật sau S2).

**S2 Judge + metrics + hiệu chỉnh** (`judge.py`, `evaluate.py --judge-only`, `calibrate_judge.py`): J1 = DeepSeek API thẳng; J2 tuỳ chọn (OpenRouter hoặc DeepSeek prompt/temperature khác). Input: câu hỏi + `reference_short` + câu trả lời; đúng nếu chứa đúng sự kiện/số chính (làm tròn hợp lý, đơn vị tương đương), không mâu thuẫn; tiền đề sai → đúng khi bác tiền đề. temperature 0, JSON {verdict, reason}, cache theo hash. `reference_short` của 25 câu mẫu: LLM rút gọn gold + người duyệt. Ngưỡng J1: gold diễn đạt lại → đúng ≥ 24/25; gold đổi số/đơn vị/thực thể (2 biến thể/câu) → bác ≥ 90%; bản 1 câu trả lời thẳng → đúng ≥ 23/25; câu dài lạc đề → bác 25/25; câu né/mơ hồ → bác ≥ 90%; câu dài đúng + 1 chi tiết mâu thuẫn → bác ≥ 80%; chấm cùng câu với `reference_long` vs `reference_short` → lật ≤ 10%; J2 kappa ≥ 0.8, bất đồng ghi csv. Chỉ dùng SO SÁNH TƯƠNG ĐỐI.

**S3 Tài liệu nguồn cho 25 câu mẫu** (`tools/corpus/fetch.py`, manifest gold): AI → arXiv/OpenReview/ACL theo tên paper; DONES → EUROfusion scipub, JACoW, PMC, KIT; chỉ domain trong sites.txt (hoặc arXiv cùng bài), sha256/license. Tiêu chí: ≥ 20/25 câu có tài liệu + `quote` khớp text PyMuPDF (fuzzy ≥ 0.9).

**S3b Baseline sớm (MỚI)**: 01 và 02 (code BTC nguyên bản, qua shim) trên sample25 với corpus = chỉ tài liệu gold (+ C50 nếu có); retrieval-only doc-hit@k; phân loại lỗi 02 theo 4 nhóm (sai tài liệu / đúng tài liệu sai đoạn / đúng đoạn sinh sai / judge sai) và theo domain. Tiêu chí: 02 > 01; bảng lỗi.

**S4 Corpus nhiễu C50 ⊂ C200** (đều chứa toàn bộ gold; C500 chỉ đo thời gian, không chặn): nhiễu cùng chủ đề (LIPAc/ESS/chẩn đoán chùm/WPENS; LLM/fuzzy/tiến hoá/nhóm DaSCI Granada), có bản trùng arXiv vs publisher. Tiêu chí: PyMuPDF đọc ≥ 98% PDF; trang/PDF, % PDF không text; trang/giây PyMuPDF và Docling ở C50/C200 + ngoại suy tới 5k/10k/20k trang; thống kê domain/site.

**S5 Câu tổng hợp + split** (`synth_qa.py`, `validate_qa.py`): 60–80 câu tổng hợp (DeepSeek đọc 1 section/bảng của tài liệu C200 NGOÀI gold sample25) + 15–20 câu người viết (chưa có người phụ trách); phân bố question_type + tỉ lệ nêu tên paper giống mẫu; có câu tính toán, ~5% tiền đề sai. Kiểm: (1) quote có trong PDF; (2) model thứ 2 trả lời chỉ từ evidence → judge đúng; (3) closed-book đúng → `closed_book_easy`; (4) người duyệt 20 câu; (5) loại câu cosine > 0.85 với câu mẫu. Split seed 2026, dev/holdout tách theo tài liệu; sample25 = holdout cuối. Tiêu chí: ≥ 60 câu qua (1)(2)(5); phân bố lệch ≤ 10 điểm %; closed_book_easy lệch ≤ 15 điểm % so với sample25; kiểm giao tài liệu.

**S6 Baseline đầy đủ + chốt eval**: 01 và 02 trên dev/holdout/sample25 với C200 (lên lịch theo token đo ở S1, chạy đêm); chạy lại 02 holdout lần 2 → đếm câu lật = nhiễu; doc-hit@k. Tiêu chí: 02 > 01 trên 3 split; ngưỡng (số câu lật) ghi AGENTS.md; bảng tương quan local ↔ LB; khoá `eval/**`. Sau đó (P1b, H100 thuê): 01/02 bằng 32B trên sample25 + holdout; lệch xa LB (29/57) → xem lại câu tổng hợp.

**Thời gian:** S1–2 1 ngày · S3+3b 1–1.5 · S4 1 · S5 1 · S6 1 + máy chạy (7B 4-bit 30–227 s/câu). Tổng ~5–6 ngày → ~28/10.
**Sửa từ v1 → v2:** cách ly sample25 (v1 sinh câu dev từ chính tài liệu gold + few-shot nguyên văn câu mẫu); thêm S3b; quy tắc dùng split; S1 thêm test offline/token/lệch shim; judge thêm kappa + test leniency + test long/short reference; J2 tuỳ chọn; C500 không chặn; S5 giảm 150 → 60–80 câu + 15–20 câu người; closed_book_easy so với mốc sample25 thay ngưỡng tuyệt đối 40%.

**Sau P2+P3 (bối cảnh):** P4 "fast-safe" (PyMuPDF đa tiến trình + find_tables, embed GPU, BM25+dense RRF, title boost, header chunk, ctx ~5k, prompt answer-first, max_new_tokens ~640, watchdog; bác bỏ nếu LB ≤ 57.14 hoặc > 1500 s) → P5 A/B định dạng câu trả lời + PoT → P6 mô-đun CI (fuzzy fusion, DE/CMA-ES, GEPA).

## 7. Tài liệu tham khảo (researcher đã mở/tra)
- van Elburg et al., Can we Evaluate RAGs with Synthetic Data? (arXiv 2508.11758) — synthetic xếp hạng đúng **retriever** (Kendall τ 0.44–0.75), sai với **generator** (τ≈0/âm).
- Krumdick et al., No Free Labels (arXiv 2503.05061) — tham chiếu đúng nâng kappa judge 0.14 → 0.78; tham chiếu sai hại hơn không có.
- Thakur et al., Judging the Judges (arXiv 2406.12624) — dùng Cohen's kappa; judge dễ dãi, ưa câu dài; judge yếu vẫn xếp hạng hệ đúng.
- Shashidhar et al., YourBench (arXiv 2504.01833) — sinh câu từ tài liệu + lọc citation grounding; ~85% câu đạt khi người duyệt.
- Accelerate big-model docs: `device_map="auto"` = model parallel tuần tự, batch 1 chỉ 1 GPU chạy tại một thời điểm. Hyperstack (vLLM 70B batch 64) A100→H100 2.8× — không phải bs=1.
- Docling GPU docs — layout dùng GPU; RTX 5090 7.9 trang/s, L40S 3.1 trang/s; không có số A100/H100. Issue #1364: GPU util 0–20% trên A100, chỉ ~2× so CPU → Docling bị chặn bởi CPU/pipeline.
- Ollama FAQ/library: ctx mặc định 4096, template riêng; llama.cpp: Q8_0 ~+0.003 PPL, Q4_K_M ~+0.18 (Llama-3-8B) → Q4 lật token greedy sớm.

## 8. Câu hỏi cụ thể cho hội đồng
1. **CHẤM plan v2 thang 1–10** + kết luận đúng một trong ba: "đủ tốt để làm" / "sửa nhỏ rồi làm" / "sửa lớn". Nêu tối đa 3 sửa còn cần (stage nào, thay gì, cụ thể). Các sửa v1 → v2 có đúng hướng không, có sửa nào thừa/sai?
2. Phạm vi & thứ tự còn gì cắt được để nộp sớm? Với quyết định nộp song song: **3 bài nộp đầu tiên** (mỗi ngày 1 bài) nên là biến thể gì để học được nhiều nhất từ LB (điểm + runtime), rủi ro thấp? (Chưa thuê H100: bài phải chạy đúng với 32B bf16 mà chưa test được 32B.)
3. Dùng **Ollama** vào đâu cho đáng: (a) backend dev nhanh cho vòng lặp prompt/retrieval, (b) judge local thứ 2 thay OpenRouter, (c) sinh/kiểm câu tổng hợp, (d) `olmo-3.1:32b-instruct` Q4 offload làm proxy hành vi 32B, (e) không dùng — vì số học GGUF/llama.cpp khác HF transformers bf16 trên server? Mỗi cách: lợi, hại, cách kiểm độ lệch so với transformers.
4. Thuê gói nào trên EzyCloudX để chạy 32B bf16 thay H100: **1× RTX PRO 6000 95 GB** (1 GPU giống server, băng thông ~1.6–1.8 TB/s) hay **2× A100-40GB** (rẻ hơn 2.2×, phải chia 2 GPU)? Quy đổi thời gian về H100 thế nào (vd chạy lại 02_rag rồi so với LB 2080 s; tách prefill/decode/Docling)? Lúc nào thật sự cần H100? Có nên thuê sớm 2–3 h, đặt ở đâu trong plan, đo gì?
5. Câu tổng hợp + quy tắc "synthetic chỉ để chọn retrieval, generator phải thắng trên sample25 + LB" — hợp lý chưa? Ngưỡng ra quyết định với 25 câu (1 câu = 4 điểm) có đủ không; nên đặt ngưỡng thế nào (số câu lật, McNemar…)?
6. Judge: thiết kế + ngưỡng hiệu chỉnh v2 có hợp lý không (quá nhiều/quá ít test)?
7. Bước kiểm chứng còn thiếu, rủi ro bị bỏ sót (bug harness, Windows vs Linux, shim 4-bit, chat template 7B vs 32B, độ dài sinh…).
Trả lời cụ thể, có số; không nhắc lại brief.
