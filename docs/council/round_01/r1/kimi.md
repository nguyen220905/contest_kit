# M6 = kimi (moonshotai/kimi-k3) — vai: alternative — vòng 1

## 1. Chẩn đoán nhanh

Chưa có điểm nên "điểm dừng" ở đây là rủi ro của *plan*, không phải của kết quả. Ba điểm nghẽn lớn nhất: (1) pipeline baseline là RAG chunk một lượt — trong khi ~60% câu AI nêu thẳng tên paper, tức bài toán thực chất là *chọn đúng tài liệu* rồi đọc kỹ, không phải xếp hạng chunk; (2) câu DONES hỏi số liệu trong bảng mà baseline tắt table-structure của Docling và chunk 800 ký tự sẽ cắt đứt bảng/đơn vị; (3) metric là LLM judge với rubric ẩn + đáp án mẫu dài giải thích — nghĩa là *phía output* (định dạng, độ đầy đủ, trích dẫn) có thể đáng giá nhiều điểm như phía retrieval, nhưng plan hiện tại gần như không đụng tới. Plan cũng thiếu một hướng "khác hẳn" để đối chứng: nếu RAG cải tiến hoá ra sai hướng (ví dụ corpus thực tế nhỏ, vài chục PDF), toàn bộ công sức dồn vào chunking/rerank là lãng phí.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)

### Đề xuất A: Document-first RAG — định tuyến tài liệu rồi "đọc cả tài liệu", thay vì chunk top-k

- Giả thuyết: Với domain AI (~60% câu nêu tên paper), bước quyết định là khớp câu hỏi → đúng PDF; một khi đúng tài liệu, đưa *toàn bộ* (hoặc toàn bộ section liên quan) vào context 65k của Olmo-32B sẽ thắng mọi pipeline chunk+rerank, vì không mất bảng, không mất ngữ cảnh định nghĩa ký hiệu, không bị "lost in the middle" ở mức chunk lẻ.
- Dựa trên bằng chứng nào trong brief: quan sát (a) của brief; luồng "query→database routing" là thành phần then chốt trong bài thắng ERC2 (giảm không gian tìm kiếm 100×, xem nguồn ở mục 4); model card Olmo-3.1-32B xác nhận ctx 65k. Lưu ý BTC đo 65k prefill bị OOM/chậm — nên giới hạn ~30–40k token/tài liệu, đủ cho 1 paper ngắn + câu hỏi.
- Cách làm cụ thể: (i) parse PDF lấy title + trang đầu (PyMuPDF, rẻ); (ii) tầng 1: matcher tên paper — regex/fuzzy (rapidfuzz có sẵn) trên chuỗi in nghiêng/trích dẫn trong câu hỏi, fallback embed title bằng MiniLM; (iii) tầng 2: nếu khớp 1–2 tài liệu → nhồi full text (cắt theo section, ưu tiên abstract/intro/tables/conclusion nếu quá dài) vào Olmo; nếu không khớp → fallback pipeline hybrid BM25+dense hiện có. Giữ budget: câu route thành công ~1 lần generate, không reranker.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Trên 15 câu AI mẫu + ~30 câu tổng hợp từ paper do mình tải: đo (a) tỉ lệ route đúng tài liệu, (b) accuracy của full-doc vs chunk-top10 trên các câu route đúng, dùng judge local. Bác bỏ nếu route đúng <70% câu có tên paper, hoặc full-doc không vượt chunk-RAG ≥5 điểm trên tập dev.
- Kỳ vọng: +8–15 điểm ở domain AI | Tự tin: vừa | Chi phí: M (2–3 ngày, chủ yếu là matcher + chiến lược cắt section)

### Đề xuất B: Đổi bài toán thành agent tìm kiếm lặp (ReAct-style), thay vì pipeline retrieval một lượt

- Giả thuyết: Các câu comparative/causal/predictive và câu tính toán cần nhiều mảnh bằng chứng ở nhiều tài liệu; một vòng retrieve duy nhất không bao giờ đủ. Cho Olmo-32B tự lặp thought→search(query)→observation (tối đa 3–4 vòng) khắc phục cả lỗi retrieval lẫn lỗi lập luận trong cùng một cơ chế.
- Dựa trên bằng chứng nào trong brief: 5 loại câu hỏi gồm comparative/causal/predictive + câu tính toán (Be7, "nếu L=512 thì window size?"); decomposition + rerank được chứng minh +36.7% MRR@10 và +11.6% F1 trên MultiHop-RAG/HotpotQA (nguồn mục 4); search lặp có feedback cải thiện đáng kể so với một lượt (Agent-UniRAG, DeepRAG — mục 4).
- Cách làm cụ thể: dùng `outlines`/`lm-format-enforcer` (có sẵn) ép output JSON {action: search|answer, query, ...}; mỗi vòng search chạy trên index hybrid đã build; dừng khi model chọn answer hoặc hết 4 vòng. Với 50 câu và ~35–40 phút budget sinh, mỗi câu được ~40s → đủ cho 3–4 lần generate ngắn (thought + query ngắn, chỉ lần cuối sinh dài). Kèm tool `compute` (chạy python eval cho biểu thức số) để phủ câu tính toán thay vì PoT tách riêng.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Chạy local với Olmo-3-7B-Instruct (16GB vừa ở bf16/8bit) trên dev mở rộng: so với RAG một lượt giữ nguyên retrieval. Đo cả wall-clock/câu. Bác bỏ nếu: trung bình >2.5 vòng/câu (vượt budget khi scale lên 32B), hoặc accuracy không tăng ≥4 điểm — lưu ý decomposition có thể phản tác dụng với câu đã cụ thể (hạn chế đã nêu trong paper nguồn).
- Kỳ vọng: +5–10 điểm tổng (mạnh nhất ở comparative/causal) | Tự tin: vừa | Chi phí: M–L (rủi ro chính là độ ổn định format của Olmo khi đóng vai agent)

### Đề xuất C: Tối ưu phía OUTPUT như một bài toán tối ưu rời rạc (tiến hoá prompt/template theo judge mô phỏng) — khác hẳn vì tối ưu đối tượng tối ưu, không tối ưu kiến trúc

- Giả thuyết: Judge là LLM với rubric ẩn, đáp án mẫu dài 80–2700 ký tự dạng giải thích → một phần lớn variance điểm nằm ở *cách trình bày* (cấu trúc: kết luận trước–giải thích sau, số liệu kèm đơn vị, trích dẫn tên tài liệu, độ dài). Đây là biến rẻ nhất để tối ưu bằng thuật toán tiến hoá, và vừa khớp yêu cầu "Computational Intelligence" cho Phase 2.
- Dựa trên bằng chứng nào trong brief: judge + rubric không công bố, cấm jailbreak nhưng không cấm tối ưu giọng văn; challenge nhấn mạnh PEFT/prompt engineering/fuzzy thay vì scale; GEPA (arXiv 2507.19457) vượt GRPO với ít hơn 35× rollout — tức tối ưu prompt bằng tiến hoá phản hồi là khả thi với ngân sách local nhỏ; challenge tiền nhiệm FLAME 2024 đã thưởng cho hướng LLM+tiến hoá (hạng 2 LLM4AD).
- Cách làm cụ thể: (i) build judge local (Olmo-7B hoặc một model 8B mạnh như Qwen2.5-7B/14B-AWQ nếu luật cho phép dùng offline — *chỉ dùng local, không bundle*) hiệu chỉnh trên 25 đáp án mẫu bằng cách chấm tương đồng với gold answer; (ii) khởi tạo quần thể 8–12 template trả lời (biến: thứ tự kết luận/giải thích, có/không mục "Source:", độ dài mục tiêu, có/không restate con số); (iii) tiến hoá 10–20 thế hệ trên dev mở rộng ~80 câu, fitness = điểm judge local trung bình 2 domain; (iv) bundle prompt tốt nhất (vài KB) vào zip. Chạy hoàn toàn offline trước khi nộp, thời gian chạy server không đổi.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Kiểm chứng hai lớp: (1) judge local phải xếp hạng đúng cặp (gold vs câu trả lời nhiễu ngẫu nhiên) ≥90% trên 25 mẫu — nếu không, bỏ toàn bộ hướng này vì fitness vô nghĩa; (2) prompt tối ưu phải thắng prompt tay trên tập dev hold-out (không dùng trong tiến hoá) ≥3 điểm, và xác nhận bằng 1 lượt nộp LB sớm. Bác bỏ nếu hold-out không cải thiện (overfit judge local).
- Kỳ vọng: +3–8 điểm | Tự tin: vừa | Chi phí: S–M (1 GPU 16GB chạy nền vài ngày; là hướng có giá trị paper Phase 2 cao nhất trong 3 đề xuất)

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief

- **Kích thước và danh sách corpus thật**: chưa biết số PDF, tỉ lệ nhiễu. Nếu corpus chỉ vài chục PDF thì Đề xuất A trở thành hiển nhiên đúng và mọi đầu tư vào chunking/rerank là thừa — cần thu thập thông tin này từ Discord/starter kit trước khi chốt plan.
- **Mâu thuẫn luật model**: trang luật ghi Olmo-3.1-32B là "the only model you may load" nhưng starter kit bundle embedder + layout model, và BTC trên Discord nói được dùng model khác "trong internal pipeline". Đề xuất B/C phụ thuộc vào việc này (model phụ cho judge local thì an toàn vì chạy offline; model phụ *chạy trên server* thì cần hỏi lại BTC bằng văn bản). Câu hỏi nên gửi BTC: "Are auxiliary models (embedders, rerankers, small LLMs for query rewriting) permitted at inference time, provided the final answer is generated by Olmo-3.1-32B-Instruct?"
- **Hành vi judge**: không biết judge chấm độc lập từng câu hay so với gold; không biết có phạt câu dài. Mọi tối ưu output (Đề xuất C) chỉ chắc qua lượt nộp LB thật — nên dành 1–2 lượt nộp đầu chỉ để probe định dạng câu trả lời.
- **Giả định của tôi** (brief không nêu): corpus_dir ở eval giống phân phối với corpus tự dựng từ sites.txt; answer() được gọi tuần tự nên có thể cache kết quả parse/index sau câu đầu; 50 câu public chia đều 2 domain.

## 4. Nguồn đã tra cứu

- Join the Challenge — GPAIS/IEEE CIS (luật chính thức: 1h H100, 1GB zip, 1 bài/ngày, corpus không được truy cập trực tiếp) — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/join-the-challenge/
- Model card allenai/Olmo-3.1-32B-Instruct (ctx, cách load, transformers ≥4.57) — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/blob/main/README.md
- How We Won the RAG Challenge (ERC2): query→database routing và LLM reranking là thành phần đóng góp lớn nhất — nền cho Đề xuất A — https://www.besthub.dev/articles/how-we-won-the-rag-challenge-multi-router-dynamic-knowledge-base-techniques-revealed-b9bb4df2e8d1
- Question Decomposition for RAG (ACL 2025 SRW): QD+rerank +36.7% MRR@10, +11.6% F1, đồng thời cảnh báo phản tác dụng với câu đã cụ thể — nền và rủi ro của Đề xuất B — https://aclanthology.org/2025.acl-srw.32.pdf
- Agent-UniRAG (arXiv 2505.22571): search lặp có evidence-feedback vượt retrieval một lượt — nền cho Đề xuất B — https://arxiv.org/abs/2505.22571v1
- WattBot 2025 5th place write-up: BM25 > dense trên PDF kỹ thuật, neighbor chunk giúp — gợi ý hybrid là baseline tối thiểu — https://github.com/anubhavmaity/wattbot
- GEPA (đã có trong brief, không tra thêm được do giới hạn lượt search): arXiv 2507.19457 — dùng theo trích dẫn của brief cho Đề xuất C.
