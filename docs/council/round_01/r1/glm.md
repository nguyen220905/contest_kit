# M2 = glm (z-ai/glm-5.3) — vai: simplifier — vòng 1

## 1. Chẩn đoán nhanh

Rủi ro lớn nhất hiện tại không phải thuật toán mà là (i) chưa có mốc đo nào — chưa nộp 01/02 nguyên bản nên mọi tối ưu sau này đều mò trong bóng tối, (ii) rủi ro vượt ngân sách 1h: theo ParseBench (2.078 docs), Docling chạy ~11 s/page còn PyMuPDF chỉ ~0,47 s/page — baseline BTC đang tiêu phần lớn ngân sách cho việc chậm nhất có thể, (iii) **trang chính thức khẳng định người tham gia KHÔNG có quyền truy cập corpus đánh giá** ("Participants do not have direct access to the evaluation corpus", https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/join-the-challenge/) → mọi ý tưởng "pre-bake index của corpus thật" đều chết từ đầu; phải parse lúc chấm. Cuối cùng, ~60% câu AI nêu thẳng tên paper → một baseline định tuyến theo tiêu đề cực kỳ rẻ sẽ là thước đo mà mọi stack phức tạp (hybrid, HyDE, GEPA, KG, fuzzy) phải vượt qua mới đáng làm.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)

### Đề xuất A: Đổi Docling → PyMuPDF (kèm `find_tables`) cho parse runtime, multiprocess 32 CPU
- Giả thuyết: Parse bằng PyMuPDF + table finder cho chất lượng ngang/thắng Docling layout nhưng nhanh ~20×, giải phóng 15–25 phút ngân sách 1h.
- Dựa trên bằng chứng nào trong brief: Docling 2.127 và pymupdf 1.28 đều có sẵn trên server; BTC tự đo 36 phút cho "toàn corpus + 50 câu"; luật 1h chấm kể cả load model + parse.
- Cách làm cụ thể: `main.py` lần gọi đầu duyệt `corpus_dir`, mỗi PDF dùng `page.get_text()` + `page.find_tables()`, chạy 32 process, chunk, giữ bảng dạng text có căn cột, cache trong process (luật cho phép cache index).
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: dựng corpus xấp xỉ từ sites.txt; đo p50 s/page và tổng thời gian; với các câu DONES cần số liệu bảng, so sánh số đọc được PyMuPDF vs Docling trên chính các bảng đó. **Bác bỏ nếu PyMuPDF sai/tách cột sai ở >20% bảng kiểm → chỉ giữ Docling cho PDF chứa bảng (hybrid) hoặc bác bỏ A.**
- Kỳ vọng: +0% trực tiếp nhưng giảm mạnh rủi ro fail wall-clock (mục tiêu tổng <40 phút/100 câu) | Tự tin: cao | Chi phí: S
- Nguồn: https://pymupdf.io/benchmark (PyMuPDF 0,474 s/page vs Docling 11 s/page; overall 53,57 vs 50,72; tables 72,0 vs 69,1 — bảng này là benchmark bên của PyMuPDF chạy ParseBench, cần tự tái lập trên corpus xấp xỉ); https://github.com/docling-project/docling-parse/blob/main/docs/performance_benchmarks.md

### Đề xuất B: Baseline "định tuyến tiêu đề + BM25 trong đúng paper" — thước đo tối thiểu
- Giả thuyết: phần lớn câu AI được giải đúng chỉ bằng việc chọn đúng paper (tên paper đã nằm trong câu hỏi), không cần dense retrieval toàn corpus.
- Dựa trên bằng chứng nào trong brief: ~60% câu AI nêu tên paper ("WinStat", "ZeroDiff", "Mamba"); rank-bm25, rapidfuzz có sẵn; package nhẹ, không tốn chỗ zip.
- Cách làm cụ thể: bóc tiêu đề từ trang 1 mỗi PDF; match câu hỏi ↔ tiêu đề bằng rapidfuzz/BM25; nếu match thì retrieve (BM25 + MiniLM) chỉ trong paper đó, kèm bảng; không match thì mới đi nhánh dense toàn corpus.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: trên 15 câu AI của 25 câu mẫu (dựng lại PDF tương ứng), đo **doc-hit@1**. **Bác bỏ nếu <70% chọn đúng paper** → phải chuyển sang hybrid dense toàn corpus.
- Kỳ vọng: +2–5% tổng điểm (2 domain) | Tự tin: vừa | Chi phí: S

### Đề xuất C: Vòng đo local rẻ + cap sinh token + đóng băng danh sách "chưa làm"
- Giả thuyết: muốn 46 lượt nộp có giá trị, mỗi thí nghiệm phải qua local trước; đồng thời sinh 2048 token/câu là phung phí (hoà điểm → nhanh hơn thắng, luật chấm 1h/2h).
- Dựa trên bằng chứng nào trong brief: đáp án mẫu 80–2.700 ký tự; judge không công bố; BTC đo baseline có câu "mất vài phút".
- Cách làm cụ thể: (1) cache parse→chunk→embed (pickle/parquet), eval 25 câu < 10 phút/lượt; Olmo-3-7B (cùng họ tokenizer, model card: https://huggingface.co/allenai/Olmo-3.1-32B-Instruct) cho prototyping trên 16GB, thuê H100 2–4 giờ để hiệu chỉnh prompt/rủi ro của 32B thật; (2) hạ `max_new_tokens` 2048→~512, prompt "câu trả lời ngắn: số liệu + đơn vị + 1–2 câu giải thích"; (3) **freeze đến khi có ≥3 mốc LB**: speculative decoding, HyDE, multi-step RAG, LoRA, GEPA, knowledge graph, fuzzy fusion.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: A/B trên dev local với judge local (rubric tự dựng theo 25 đáp án mẫu). **Bác bỏ cap 512 token nếu câu ngắn bị judge chấm thấp hơn ở ≥5/25 câu.**
- Kỳ vọng: 0–3% từ định dạng, chủ yếu hạ rủi ro thời gian/tie-break | Tự tin: vừa | Chi phí: S

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief

- **Corpus không truy cập được** (trang chính thức) → phải tự ước số PDF/loại nhiễu từ sites.txt; mọi số "pre-bake" phải hiểu là bundle *model phụ + prompt*, không phải index corpus thật.
- Zip ≤1GB: BGE-M3/BGE-reranker-v2-m3 (568M tham số, ~1,1GB fp16) **có thể không vừa** khi cộng thêm index/prompt — cần đo dung lượng thật trước khi cam kết dùng reranker.
- Mâu thuẫn luật chưa khép: "only model you may load" (trang luật) vs Discord 15/10 "có thể dùng model khác trong internal pipeline". Cần hỏi BTC đúng 2 câu: (a) "Auxiliary models (embedder/reranker/query rewriter) are allowed as long as the final answer text is generated by Olmo-3.1-32B — correct?"; (b) "Is an 8-bit/quantized load of `allenai/Olmo-3.1-32B-Instruct` allowed?" (model card chính thức chỉ dẫn bnb 8-bit). Model card ví dụ dùng `do_sample=True, top_p=0.95` — khác với greedy của baseline, đáng đưa vào A/B.
- Rubric judge không công bố → mọi tối ưu định dạng câu trả lời là giả định; BTC baseline không trích dẫn nguồn nhưng rubric có thể thưởng trích dẫn.
- Chưa tìm được write-up cuộc thi RAG tương tự (hết lượt tìm kiếm) — thiếu bằng chứng chéo về tầm hiệu quả từng hướng.

## 4. Nguồn đã tra cứu

- IEEE CIS "Join the Challenge" (starter kit, 25 cặp mẫu, **không truy cập corpus đánh giá**, 1 H100, 1h, 1GB, 1 nộp/ngày, không internet) — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/join-the-challenge/ — nền tảng cho đề xuất A (phải parse runtime) và phisode kế hoạch.
- IEEE CIS "Evaluation" (top-20 sang expert jury; điểm cuối = domain-balanced accuracy + 2 thành phần chuyên gia) — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/evaluation/ — giá trị của việc giữ pipeline đơn giản, giải phóng thời gian cho paper Phase 2.
- Official Rules PDF (timeline 28/09–30/11/2026, 2-phase, giải thưởng) — https://cis.taskforce.ieee.org/gpais/wp-content/uploads/sites/6/2026/09/IEEE-CIS-Technical-Challenge-on-Domain-Specific-Question-Answering-with-LLMs-Contest-Rules.pdf
- Model card Olmo-3.1-32B-Instruct (HF transformers chuẩn, **bnb load_in_8bit là cách dùng chính thức**, ví dụ sinh top_k=50/top_p=0.95) — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct
- PyMuPDF benchmark (ParseBench 2026-09-16, 2.078 docs: PyMuPDF 0,474 s/page vs Docling 11 s/page; overall 53,57 vs 50,72; tables 72,0 vs 69,1) — https://pymupdf.io/benchmark — bằng chứng chính cho đề xuất A (lưu ý: do PyMuPDF duyệt, cần tự tái lập trên corpus của mình).
- PyMuPDF blog so sánh Docling (53,6 vs 50,72; 0,47 vs 11 s/page; lưu ý chất lượng phụ thuộc phiên bản Docling) — https://pymupdf.io/blog/playing-an-away-game-with-the-new-pymupdf-benchmark-page
- docling-parse performance docs (so sánh parser + threading) — https://github.com/docling-project/docling-parse/blob/main/docs/performance_benchmarks.md
- Write-up Kaggle/cuộc thi RAG tương tự: **không tra cứu được** (hết lượt tìm kiếm web trong vòng này).
