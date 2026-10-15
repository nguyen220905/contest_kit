# BRIEF cho hội đồng model — round 01 (phiên mở đầu: lên PLAN tổng thể)
> Chưa có code, chưa có thí nghiệm, chưa có điểm LB. Mọi số liệu bên dưới lấy từ luật cuộc thi + starter kit chính thức.

## 1. Bài toán
- Cuộc thi: IEEE CIS Technical Challenge on Domain-Specific Question Answering with LLMs (Codabench #17954). Nộp 1 file .zip, `main.py` ở gốc có `answer(question, corpus_dir, model_id) -> str`, gọi tuần tự 1 lần/câu trong CÙNG process (cache model/index được). Trả về không phải str = không trả lời.
- Input: câu hỏi chuyên gia (text). Output: câu trả lời tự do (text). 2 domain: **IFMIF-DONES** (nguồn neutron/máy gia tốc deuteron 40 MeV 125 mA, bia lithium lỏng, đang xây ở Granada) và **bài báo nghiên cứu AI** (nhiều câu nêu thẳng tên paper: "In the paper WinStat…", "In ZeroDiff…", "In the paper of Mamba…"; có bài nghiêng fuzzy cognitive maps / GPAIS — có thể là bài của nhóm tổ chức ở Granada).
- `corpus_dir`: thư mục PDF (tài liệu thật + tài liệu nhiễu), retrieval "mù" (không biết câu hỏi thuộc tài liệu nào). Nguồn corpus: ~32 domain (arxiv, jacow/accelconf, ieeexplore, sciencedirect, openreview, springer, iopscience, scipub.euro-fusion.org, mdpi, nature, …). Số PDF chưa biết.
- Metric: **domain-balanced accuracy** (trung bình accuracy 2 domain), chấm tự động bằng LLM judge (judge model + rubric KHÔNG công bố). Hoà điểm → bài chạy nhanh hơn thắng. BTC không trả output, chỉ status + điểm.
- Benchmark tổng: 100 câu QA do chuyên gia soạn (5 loại: factual, comparative, procedural, causal, predictive). Starter kit nói đã test "toàn corpus + 50 câu" → public LB nhiều khả năng ~50 câu (1 câu ≈ 2–4 điểm % trong 1 domain).
- Dữ liệu dev: 25 câu mẫu có đáp án (DONES 10, AI 15), KHÔNG kèm PDF. Đáp án mẫu dạng giải thích 80–2700 ký tự; có câu tính toán (vd "hằng số phân rã Be7 nếu làm tròn chu kỳ bán rã theo ngày"), câu tra số liệu thiết kế (lưu lượng thể tích bộ trao đổi nhiệt sơ cấp, nhiệt độ bẫy nitơ, bypass ratio nhánh ICS, thể tích toà nhà chính), câu suy luận "nếu L=512 thì window size bao nhiêu" (WinStat), câu phản biện ("so sánh có bất công không?").
- Luật cứng:
  - Trang luật: `model_id` = `allenai/Olmo-3.1-32B-Instruct` là "The **only** model you may load". Nhưng starter kit chính thức 02_rag tự bundle embedder MiniLM + layout model Docling trong zip. BTC trả lời Discord (15/10): "you may use a different model in your internal pipeline to try. Just make sure you use the same Transformers library".
  - Không internet lúc chạy; chỉ package có sẵn: torch 2.8.0, transformers 5.17.0, accelerate, bitsandbytes, peft, optimum, kernels, sentence-transformers 6.0.1, FlagEmbedding 1.4.2, faiss-cpu, chromadb, rank-bm25, whoosh, rapidfuzz, docling 2.127, pymupdf 1.28, pdfplumber, pypdf, spacy, nltk, outlines, lm-format-enforcer, llama-index, langchain, scikit-learn, networkx… KHÔNG vLLM, KHÔNG flash-attn (sdpa/eager).
  - Được bundle index/embedding/prompt/adapter chuẩn bị trước (có internet). Dữ liệu ngoài được phép nếu trích dẫn. Cấm tấn công/jailbreak judge.
- Tài nguyên chấm: 1×H100 80GB, 32 CPU, 128GB RAM; wall-clock **1h public** (tính cả load model, parse PDF), ≤2h private; zip giải nén ≤ **1GB**.
- Thời gian: hôm nay 15/10/2026; Public LB đóng 30/11 23:55 UTC → ~46 lượt nộp (1/ngày). Bài public điểm cao nhất tự động chấm lại ở private (không tự chọn). Top đội vào Phase 2: paper IEEE 4 trang, hội đồng khoa học chấm "technical innovation" + chuyên gia chấm chất lượng câu trả lời. Challenge nhấn mạnh Computational Intelligence: thích nghi LLM KHÔNG dựa vào scale model / fine-tune nặng / nhiều dữ liệu (PEFT, prompt engineering, knowledge graph, fuzzy modeling). Challenge tiền nhiệm FLAME 2024: hạng 2 LLM4AD (LLM + tiến hoá), hạng 3 CiTIUS (fuzzy/XAI).
- Máy local: 1× RTX 5060 Ti 16GB. Olmo-32B không chạy local (bf16 ~64GB).

## 2. Pipeline hiện tại (baseline 02_rag của BTC — chưa chạy lại)
1. Lần gọi đầu: Docling (no OCR, no table structure) parse mọi PDF → markdown → chunk 800 ký tự, overlap 150 theo đoạn.
2. Embed all-MiniLM-L6-v2 trên **CPU**, cosine top-k, nhồi tối đa **10.000 token** context (BTC đo: ctx gần 65k → OOM khi prefill trên 80GB, và câu dài mất vài phút).
3. Olmo-32B bf16, greedy, max_new_tokens 2048, system prompt chung "You are a scientific expert…". Không rerank, không BM25, không xử lý bảng, không trích dẫn.
4. BTC báo: ~36 phút cho toàn corpus + 50 câu (không tách parse vs generate). Baseline 01_zero_shot: không retrieval.

## 3. Kết quả
- Chưa có (chưa nộp lần nào, chưa có eval local). Nhiễu: chưa đo. Độ lệch local–LB: chưa biết.

## 4. Đã thử
| Thay đổi | Δ metric | Ghi chú |
|---|---|---|
| — | — | Chưa có thí nghiệm nào |

## 5. Phân tích lỗi gần nhất
- Chưa có. Quan sát từ 25 câu mẫu: (a) ~60% câu AI nêu tên paper/phương pháp → định tuyến theo tiêu đề tài liệu có thể rất mạnh; (b) câu DONES hỏi số liệu thiết kế cụ thể nằm trong bảng/báo cáo kỹ thuật → cần giữ bảng và đơn vị; (c) có câu cần tính toán; (d) đáp án mẫu dài, giải thích → độ dài/định dạng câu trả lời có thể ảnh hưởng judge.

## 6. Plan hiện tại (khung, chưa duyệt)
1. Dựng venv khớp server (torch 2.8 CUDA + transformers 5.17) + smoke test answer() với model nhỏ.
2. Dựng corpus local (tải PDF theo chủ đề 25 câu mẫu + nhiễu từ sites.txt) + tập dev mở rộng + LLM judge local.
3. Nộp 01_zero_shot và 02_rag nguyên bản để có mốc LB.
4. Cải tiến retrieval (parse nhanh, hybrid, rerank) và prompt.
5. Thành phần CI cho paper.

## 7. Tài liệu tham khảo (researcher đã tra cứu, đã mở)
- Olmo 3 tech report (arXiv 2512.13961) + model card Olmo-3.1-32B-Instruct — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct — họ 7B/32B cùng họ tokenizer, ctx 65k, transformers ≥4.57; khuyến nghị temp 0.6/top_p 0.95.
- Anthropic Contextual Retrieval (2024) — https://www.anthropic.com/news/contextual-retrieval — thêm ngữ cảnh tài liệu vào chunk + BM25 giảm 49% lỗi top-20; + rerank giảm 67%.
- BGE-M3 (arXiv 2402.03216) + bge-reranker-v2-m3 (0.6B, Apache-2.0) — dense+sparse trong 1 model; cross-encoder rerank, chạy được bằng FlagEmbedding có sẵn.
- Docling Technical Report (arXiv 2408.09869) — layout model chính xác nhưng chậm; benchmark bên thứ ba ~0.6 trang/s vs PyMuPDF ~4+ trang/s.
- Program of Thoughts (arXiv 2211.12588) — tách tính toán ra Python, +12% so CoT trên bài số/bảng tài chính.
- GEPA (arXiv 2507.19457, ICLR 2026) / EvoPrompt (ICLR 2024) — tiến hoá prompt bằng LLM, GEPA vượt GRPO với ít hơn 35× rollout.
- Allo-Self-RAG (Iranian J. Fuzzy Systems 2026) — hệ suy diễn mờ gộp tín hiệu RAG, simulated annealing tối ưu membership function.
- Lost in the Middle (TACL 2023) — đặt bằng chứng quan trọng ở đầu/cuối context.

## 8. Câu hỏi cụ thể cho hội đồng
A. Hướng giải tổng thể tốt nhất trong ràng buộc trên: so sánh và XẾP HẠNG các hướng — RAG cải tiến (parse PDF nhanh, chunking theo cấu trúc, hybrid BM25+dense, reranker, định tuyến theo tiêu đề paper, query rewriting/HyDE, đa bước), prompt/định dạng câu trả lời tối ưu cho LLM judge, xử lý câu tính toán (đọc bảng, PoT/python), adapter LoRA/kiến thức domain đóng gói sẵn, self-consistency/verification, speculative decoding… Mỗi hướng: kỳ vọng tăng điểm, rủi ro, chi phí thời gian chạy trong 1h.
B. Hướng nào vừa tăng điểm vừa đủ "Computational Intelligence" để có giá trị đổi mới cho paper Phase 2 (tối ưu prompt/pipeline bằng thuật toán tiến hoá, fuzzy fusion cho retrieval, knowledge graph…)? Nêu nguồn cụ thể.
C. Luật model: hiểu lời BTC thế nào — có được dùng LLM khác Olmo để SINH câu trả lời không? Model phụ (embedder, reranker, draft model cho speculative decoding, model nhỏ rewrite query) có an toàn không? Nên hỏi BTC đúng câu gì?
D. Dev/eval khi local chỉ 16GB: model thay thế (Olmo-3-7B-Instruct? lượng tử hoá?), có nên thuê H100 cloud chạy Olmo-32B thật (bao nhiêu giờ, khi nào), ước lượng thời gian thế nào?
E. Không có corpus: dựng corpus local + tập dev ra sao (tải PDF từ sites.txt theo chủ đề 25 câu mẫu, tài liệu DONES công khai, sinh câu hỏi tổng hợp từ PDF), tránh overfit 25 câu.
F. Ngân sách 1h: parse PDF (PyMuPDF vs pdfplumber vs Docling, đa tiến trình 32 CPU), embed trên GPU, thời gian load Olmo-32B, độ dài sinh, batching nhiều câu (answer() gọi tuần tự — có nên tiền xử lý/cached?), zip 1GB chứa được embedder/reranker nào.
G. LLM judge local: chọn judge nào, rubric, hiệu chỉnh với 25 đáp án mẫu, ước lượng nhiễu (50 câu public).
H. Lộ trình theo mốc tới 30/11 với 1 bài/ngày: nộp gì trước để học được nhiều nhất từ LB (ablation nào đáng tốn 1 lượt nộp).
Mỗi đề xuất ghi: kỳ vọng (điểm %), tự tin, chi phí, và cách KIỂM CHỨNG local trước khi tốn lượt nộp.
