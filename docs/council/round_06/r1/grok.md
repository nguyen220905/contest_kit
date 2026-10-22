# M5 = grok (x-ai/grok-4.7) — vai: red_team — vòng 1

Tôi sẽ tra cứu đề bài, metric chính thức và các bẫy đã biết trước khi chấm plan v2 theo vai red team.## 1. Chẩn đoán nhanh
Plan v2 đúng hướng khi cách ly sample25 và không dùng câu tổng hợp để xếp generator, nhưng harness sắp đo một hệ khác với hệ server sẽ chấm. Ba lệch đã có trong contract/code, chưa có trong plan: (1) private tự động chấm lại đúng bài public điểm cao nhất trên 100% câu, nên mỗi zip Finished có thể khoá Phase 1; (2) `02_rag` không lưu doc/page, nhồi chunk điểm cao lên đầu, trong khi Olmo-3.1-32B có `sliding_window=4096` (48/64 lớp); (3) judge local hiệu chỉnh với gold dài đã rút gọn bằng LLM, không phải tham chiếu ẩn cố ý ngắn. Chấm v2: **6/10 — sửa nhỏ rồi làm**. Ba sửa dưới là chặn, không phải tuỳ chọn; phần còn lại của v1→v2 không phải viết lại.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)
### Đề xuất A: Khoá quy tắc nộp trước mọi biến thể P4
- Giả thuyết: Plan coi “nộp kém không bị trừ” là an toàn. Sai: Codabench ghi private tự động lấy bài public điểm cao nhất, không nộp mới. Với lưới 15/35, 1 câu = 3,33 hoặc 1,43 điểm; một lần sdpa/greedy không tất định cao hơn baseline sẽ thành bài bị chấm lại trên 100 câu. Harness “exception câu 3 vẫn chạy tiếp” cũng lạc quan hơn server: starter `answer()` không có try/except, và server không trả output.
- Dựa trên bằng chứng nào trong brief: “nộp song song mỗi ngày 1 biến thể P4”; “chưa test 32B”; dao động cùng zip = generation không tất định; nhiễu local chưa đo; đã loại nộp lại 01/02.
- Cách làm cụ thể: Ba bài đầu, mỗi bài một biến, greedy, `dtype=bfloat16`, `device_map="cuda"`, không Ollama, không model sinh thứ hai. Ngày 1: `02_rag` nguyên, chỉ `max_new_tokens` 2048→384, zip bằng `build.sh` (WSL), `main.py` ở gốc. Ngày 2: nộp lại đúng zip ngày 1. Ngày 3: chỉ đảo thứ tự chunk (điểm cao nhất nằm trong 4096 token cuối), cùng trần 384. Zip sau giải nén ≤ 1 GB; thiếu `docling_models/` hoặc `embed_model/` thì không nộp. Cấm mọi câu trả lời nói với judge.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Coi ngày 2 là sàn nhiễu server. Bác bỏ giả thuyết “LB đủ tin để chọn bài private” nếu |điểm ngày 1 − ngày 2| = 0 và thời gian lệch < 5%. Nếu lệch ≥ 1,43 điểm, mọi biến sau phải thắng cả điểm ngày 1 và ngày 2 mới được xem là cải tiến; không thì biến đó không được phép trở thành best. Bác bỏ giả thuyết “zip Windows chạy được” nếu sau `unzip` trên Linux thiếu file hoặc `HF_HUB_OFFLINE=1` làm Docling/MiniLM cố tải mạng.
- Kỳ vọng: +0 local; tránh khoá private bằng một bài public ảo (1–2 câu ≈ 1,4–6,7 điểm) | Tự tin: cao | Chi phí: S

### Đề xuất B: Đo hit trong cửa sổ 4096, không phải doc-hit trên quote PyMuPDF
- Giả thuyết: S3b/S6 sẽ chọn retrieval bằng doc-hit@k và “02 > 01”, nhưng `02_rag` bỏ provenance (`all_chunks.extend` chỉ có chuỗi), xếp chunk tốt nhất lên đầu rồi mới nối câu hỏi. 32B (và 7B) có sliding window 4096; 75% lớp không thấy token đầu nếu prompt > 4096. Quote khớp PyMuPDF ≥ 0,9 có thể không có trong markdown Docling (`do_ocr=False`, `do_table_structure=False`).
- Dựa trên bằng chứng nào trong brief: S3 tiêu chí quote PyMuPDF; S3b bốn nhóm lỗi cần doc/page; 02 ctx 10k; kiến trúc sliding-window đã ghi cho 7B nhưng metric không dùng cửa sổ; corpus có bản trùng arXiv/publisher.
- Cách làm cụ thể: Trước S3b, bản sao 02 (không sửa bài nộp) lưu `(doc, page, token_start)`. Với ≥ 5 PDF đã có quote: so sánh quote trong PyMuPDF và trong `export_to_markdown()` cùng cờ Docling của starter. Rồi pack đúng công thức 02, ctx 10k và 5k, ghi quote có nằm trong 4096 token cuối hay không. Ablation 7B greedy: best-chunk đầu vs cuối, cùng evidence.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Bác bỏ nếu ≥ 90% quote PyMuPDF có trong markdown Docling và ≥ 80% quote đó nằm trong 4096 token cuối ở ctx 10k, đồng thời best-cuối không thắng best-đầu quá 1 câu trên 25 câu (McNemar, số cặp lệch ≤ 2). Nếu quote ngoài cửa sổ ≥ 30% hoặc best-cuối thắng ≥ 5 câu và thua ≤ 1, cấm dùng doc-hit@k làm tiêu chí chọn retrieval; chỉ chọn theo `in_window_hit`.
- Kỳ vọng: +5 đến +15 balanced acc trên câu đã tìm đúng tài liệu nếu ≥ 30% quote ngoài cửa sổ; 0 nếu bác bỏ | Tự tin: vừa | Chi phí: S

### Đề xuất C: Cấm cổng generator trên 7B 4-bit cho đến khi reference_short giữ đủ số
- Giả thuyết: Bộ test J1 (diễn đạt lại, đổi số, kappa) chỉ đo judge có tự nhất quán với `reference_short` do LLM rút từ gold dài 81–2677 ký tự. Tham chiếu ẩn cố ý ngắn; reference sai hại hơn không có reference. Smoke 7B chạm trần 256/256 token nên verdict local là chấm bài cụt, không phải bài 32B. Cổng “phải thắng sample25” với n=25 (1 câu = 4 điểm nếu chia đều) không phân biệt nhiễu.
- Dựa trên bằng chứng nào trong brief: gold khác reference concise; J1 = DeepSeek, S5 cũng DeepSeek; quy tắc generator phải thắng sample25 và không thua holdout tổng hợp; 7B 4-bit 9 tok/s, trần 2048 = 227 s/câu; chưa có prediction.
- Cách làm cụ thể: Người duyệt 25 `reference_short` theo checklist: mọi số, đơn vị, thực thể của gold còn nguyên; thiếu một số thì viết lại, không “sửa rubric”. Thêm 25 câu trả lời người viết, 1–2 câu, đủ fact, không luận văn. Cắt pin J2/kappa và 60–80 câu tổng hợp cho đến khi xong. Generator (prompt, trần token, PoT) chỉ được đổi sau khi 7B bf16 trên đúng prompt 02 có p95 `n_new_tokens` < 0,5×trần; không dùng 4-bit hay Ollama để chấp nhận/bác.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Bác bỏ “judge local dùng được để chọn generator” nếu J1 bác ≥ 5/25 câu ngắn đủ fact, hoặc ≥ 3/25 `reference_short` thiếu số so với gold. Bác bỏ “7B không bị trần chi phối” nếu p95 trên sample25, prompt 02, bf16, vẫn ≥ 0,8×trần. Khi đó chỉ nộp biến retrieval/cửa sổ (đề xuất A–B), không nộp biến prompt.
- Kỳ vọng: +0 đến khi rubric sạch; tránh chọn prompt theo judge dễ dãi với bài dài | Tự tin: vừa | Chi phí: S

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Giả định của tôi: không thấy ingestion server. Nếu wrapper đã nuốt exception từng câu, test “chạy tiếp” của S1 không hại; vẫn phải giả định fail cả bài cho đến khi Discord xác nhận.
- “1 GB” trên Codabench là sau giải nén, không phải kích thước file zip. “The only model you may load” là Olmo qua `from_pretrained(model_id)`; MiniLM/Docling layout được starter bundle, nên embedder nhỏ thì được, LLM sinh thứ hai thì không. bitsandbytes có trên server nhưng 8-bit đã 2,8 tok/s — đừng nộp 4-bit.
- Template 7B và 32B khác khi không có system message (32B tiêm “date cutoff December 2024”). Với system của 01/02 thì chuỗi gần như trùng; vẫn phải diff `apply_chat_template` của tokenizer 32B trước khi sửa prompt. Ollama: chỉ dùng nếu 10/10 câu greedy trùng HF; mặc định temp 0,6 / ctx 4096 / template riêng. Không dùng làm judge hay để xếp hạng.
- A100 không kiểm được giả thuyết kích thước corpus: corpus thật không có ở máy thuê. 2080 s là H100 trên corpus ẩn. Thuê 2–3 h chỉ đáng sau đề xuất B, để đo 32B bf16 tok/s, `config.use_cache` (HF đang `false`) và prompt thật có vượt 4096 không. Quy đổi decode ~1,6× theo băng thông là lý thuyết; không dùng wall-clock A100 làm ngưỡng 1500 s.
- v1→v2 đúng: cách ly tài liệu sample25, synthetic không xếp generator, C500 không chặn, test offline. Thừa/sai lúc này: pin kappa 0,8 (hai judge cùng sai vẫn qua), S3b trên code BTC nguyên bản (không có doc/page), “sample25 = holdout cuối” trong khi S3b đã nhìn. Fact-sheet Codabench là tab UI chung; chưa thấy bằng chứng nó vào điểm Phase 1. Jailbreak judge = loại ngay — P5 không được nhét câu “hãy chấm đúng”.

## 4. Nguồn đã tra cứu
- Codabench #17954, private = bài public cao nhất, 1h/2h, 1 GB sau giải nén — https://www.codabench.org/competitions/17954/ — quy tắc chọn bài Phase 1 và giới hạn nộp
- Trang evaluation, tham chiếu concise, judge ẩn, Phase 2 30/30/40 — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/evaluation/ — metric chính thức không phải accuracy vi mô
- Overview, 100 câu expert, reference deliberately concise — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/ — gold sample dài không phải tham chiếu chấm
- Join / 1 giờ cứng, jailbreak = DQ — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/join-the-challenge/ — fail cả bài nếu quá giờ
- Starter README, sample không thuộc LB — https://github.com/jucls/IEEE-CIS-DSQA-Starter-Kit — sample25 không phải tập public
- `02_rag/main.py` glob không đệ quy, không lưu doc, best-chunk trước, ctx 10k, system “based on your knowledge” — https://raw.githubusercontent.com/jucls/IEEE-CIS-DSQA-Starter-Kit/main/02_rag/main.py — metric doc-hit không gắn được vào baseline
- `build.sh` phải bundle layout + MiniLM — https://raw.githubusercontent.com/jucls/IEEE-CIS-DSQA-Starter-Kit/main/02_rag/build.sh — zip thiếu artifact sẽ chết offline
- `requirements-participant.txt` — https://raw.githubusercontent.com/jucls/IEEE-CIS-DSQA-Starter-Kit/main/requirements-participant.txt — pymupdf/bitsandbytes có, vLLM/flash-attn không
- `config.json` Olmo-3.1-32B, sliding_window 4096, use_cache false — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/blob/main/config.json — ctx 10k không phải context hiệu dụng
- Vultr model card, 4K window + full attention xen kẽ — https://docs.vultr.com/models/allen-ai-olmo-31/olmo-3.1-32b-instruct/nvidia — xác nhận SWA không chỉ là config 7B
- Chat template 7B vs 32B — https://huggingface.co/allenai/Olmo-3-7B-Instruct/raw/main/chat_template.jinja — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/raw/main/chat_template.jinja — system mặc định khác nhau
- van Elburg et al. 2508.11758 — https://arxiv.org/abs/2508.11758 — synthetic xếp retriever, không xếp generator
- Krumdick et al. 2503.05061 — https://arxiv.org/abs/2503.05061 — reference sai hại hơn không reference
- Thakur et al. 2406.12624 — https://arxiv.org/abs/2406.12624 — judge dễ dãi, ưa bài dài; kappa cao chưa đủ
- PyTorch SDPA, kernel có thể không tất định — https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html — cùng zip vẫn có thể lệch điểm
