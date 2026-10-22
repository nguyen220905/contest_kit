# Nguồn researcher tự tra cứu (bước 0) — 2026-10-15
> Mỗi dòng: tiêu đề — URL — điều rút ra. [đã mở] = đã WebFetch và khớp.

## Cuộc thi / luật
- Codabench #17954 (API JSON: pages How to participate / Timeline / Terms) — https://www.codabench.org/api/competitions/17954/ — [đã mở] model_id "The **only** model you may load"; 1h public (execution_time_limit 3600), 2h private (7200); zip ≤1GB giải nén; 1 bài/ngày; external data OK nếu trích dẫn; nộp kèm "draft description of your methodology"; Public LB end 2026-11-30T23:55Z.
- IEEE CIS Technical Challenge on DSQA — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/ — [đã mở] benchmark = **100 câu QA** chuyên gia (factual/comparative/procedural/causal/predictive), chưa công bố; CI = "adaptation strategies that don't rely on scaling, extensive fine-tuning, or large data": PEFT, prompt engineering, knowledge graphs, fuzzy modeling.
- FLAME 2024 (challenge tiền nhiệm) — https://citius.gal/en/news/news/international-podium-for-citiuss-explainable-artificial-intelligence/ — [chỉ search] hạng 3 CiTIUS (fuzzy/XAI), hạng 2 LLM4AD (CityU, tiến hoá thuật toán bằng LLM) → hội đồng khoa học chuộng LLM+EC/fuzzy.
- Triguero et al., GPAIS: Properties, definition, taxonomy…, Information Fusion 103 (2024) — https://nottingham-repository.worktribe.com/output/27363287 — [search] bài của nhóm tổ chức (Granada); 1 câu mẫu hỏi đúng bài này → corpus AI có thể chứa bài của nhóm BTC (FCM, fuzzy, GPAIS).

## Model
- allenai/Olmo-3.1-32B-Instruct model card — https://huggingface.co/allenai/Olmo-3.1-32B-Instruct — [đã mở] transformers ≥4.57; sibling Olmo-3-7B-Instruct; trích dẫn arXiv 2512.13961.
- Team Olmo, "Olmo 3", arXiv:2512.13961 (2025) — https://arxiv.org/abs/2512.13961 — [đã mở] họ 7B/32B fully-open, long-context.
- Perplexity: docs/research/20261015-0145-pro-olmo-3-...md — 7B & 32B cùng họ tokenizer/chat template (cần tự so `chat_template`), ctx 65536, khuyến nghị temp 0.6/top_p 0.95.

## Retrieval / RAG
- Lewis et al., RAG, NeurIPS 2020 — https://arxiv.org/abs/2005.11401 — [đã mở] baseline khái niệm.
- Chen et al., BGE M3-Embedding, 2024 — https://arxiv.org/abs/2402.03216 — [đã mở] dense+sparse+multi-vector trong 1 model, 8192 token.
- BAAI/bge-reranker-v2-m3 — https://huggingface.co/BAAI/bge-reranker-v2-m3 — [đã mở] cross-encoder 0.6B, Apache-2.0, dùng qua FlagEmbedding/ST (đều có trong requirements).
- Anthropic, Contextual Retrieval (2024) — https://www.anthropic.com/news/contextual-retrieval — [đã mở] contextual emb + BM25 giảm 49% lỗi top-20, + rerank giảm 67%.
- Gao et al., HyDE, ACL 2023 — https://arxiv.org/abs/2212.10496 — [đã mở] sinh tài liệu giả rồi embed.
- Rackauckas, RAG-Fusion, 2024 — https://arxiv.org/abs/2402.03367 — [đã mở] multi-query + RRF; rủi ro lệch chủ đề.
- Sarthi et al., RAPTOR, ICLR 2024 — https://arxiv.org/abs/2401.18059 — [đã mở] cây tóm tắt; tốn LLM lúc index → khó trong 1h.
- Liu et al., Lost in the Middle, TACL 2023 — https://arxiv.org/abs/2307.03172 — [đã mở] đặt chunk tốt nhất ở đầu/cuối context.
- Skarlinski et al., PaperQA2, 2024 — https://arxiv.org/abs/2409.13740 — [search] agent RAG cho bài báo khoa học (rerank+contextual summary).
- Auer et al., Docling Technical Report, 2024 — https://arxiv.org/abs/2408.09869 — [đã mở] layout model; chậm hơn PyMuPDF nhiều (benchmark bên thứ ba: ~0.6 vs ~4 trang/s).

## Suy luận / giải mã
- Wang et al., Self-Consistency, ICLR 2023 — https://arxiv.org/abs/2203.11171 — [đã mở].
- Chen et al., Program of Thoughts, TMLR 2023 — https://arxiv.org/abs/2211.12588 — [đã mở] tách tính toán ra Python (+12% so CoT) → câu Be7.
- Leviathan et al., Speculative Decoding, ICML 2023 — https://arxiv.org/abs/2211.17192 — [đã mở] 2–3× nhanh, output y hệt; cần draft model (vướng luật "only model").

## Computational Intelligence cho paper
- Guo et al., EvoPrompt, ICLR 2024 — https://arxiv.org/abs/2309.08532 — [đã mở].
- Fernando et al., Promptbreeder, 2023 — https://arxiv.org/abs/2309.16797 — [đã mở].
- Agrawal et al., GEPA, ICLR 2026 — https://arxiv.org/abs/2507.19457 — [đã mở] tiến hoá prompt phản tư + Pareto, ít rollout.
- Khattab et al., DSPy, 2023 — https://arxiv.org/abs/2310.03714 — [đã mở].
- Hosseini & Eftekhari, Allo-Self-RAG (fuzzy aggregation + SA), Iranian J. Fuzzy Systems 2026 — https://ijfs.usb.ac.ir/article_9936.html — [đã mở] FIS gộp tín hiệu RAG, SA tối ưu membership.

## Đánh giá
- Zheng et al., Judging LLM-as-a-Judge, NeurIPS 2023 D&B — https://arxiv.org/abs/2306.05685 — [đã mở] bias vị trí/độ dài/tự ưu.

## Domain IFMIF-DONES
- Bernardi et al., The IFMIF-DONES Project: Design Status…, J. Fusion Energy 41:24 (2022), doi 10.1007/s10894-022-00337-5 — https://pmc.ncbi.nlm.nih.gov/articles/PMC9596351/ — [đã mở] open access; 40 MeV/125 mA, Li 25mm 15 m/s.
- Tài liệu Li loop/Be7/tritium (KIT, EUROfusion scipub, Springer s10894-023-00378-4) — https://scipub.euro-fusion.org/wp-content/uploads/eurofusion/WPENSCPR18_19806_submitted-4.pdf — [search] khớp chủ đề câu mẫu (Be7, ICS, heat exchanger); scipub.euro-fusion.org nằm trong sites.txt.

# Bổ sung 2026-10-22 (round_02: chấm plan P2+P3 eval harness + corpus dev)
> Perplexity `--ask` KHÔNG chạy được: OpenRouter trả HTTP 401 "API key expired" (cả 7 model OpenRouter của hội đồng cũng 401; chỉ deepseek gọi thẳng còn sống). Nguồn dưới tự tra bằng WebSearch/WebFetch.

## Judge (LLM-as-judge có tham chiếu)
- Thakur et al., Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges, GEM 2025 (arXiv 2406.12624) — https://arxiv.org/abs/2406.12624 — [đã mở] dùng Cohen's kappa thay % đồng ý (đồng ý % cao vẫn lệch điểm tới 5 điểm); judge có xu hướng dễ dãi (leniency), ưa câu dài/thiếu cụ thể; judge nhỏ/lexical vẫn XẾP HẠNG đúng dù điểm tuyệt đối lệch.
- Krumdick et al., No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding, 2025 (arXiv 2503.05061) — https://arxiv.org/abs/2503.05061 — [đã mở] tham chiếu đúng nâng kappa 0.14 → 0.78 trên câu judge tự không trả lời được; tham chiếu SAI hại hơn không có tham chiếu; 35% tham chiếu MT-Bench toán sai → phải duyệt `reference_short`.
- Badshah & Sajjad, Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form QA, WiNLP (arXiv 2408.09235) — https://arxiv.org/abs/2408.09235 — [đã mở] ensemble nhiều judge tăng độ tin cậy so với 1 judge.
- Zheng et al., Judging LLM-as-a-Judge (đã có ở round_01).

## Câu hỏi tổng hợp / đánh giá RAG
- van Elburg, van der Putten, Marx, Can we Evaluate RAGs with Synthetic Data?, 2025 (arXiv 2508.11758) — https://arxiv.org/abs/2508.11758 — [đã mở] synthetic benchmark xếp hạng ĐÚNG cấu hình retriever (Kendall τ 0.44–0.75) nhưng SAI khi so generator/LLM (τ ≈ 0 hoặc âm) do lệch tác vụ + bias phong cách → dùng câu tổng hợp cho retrieval, cẩn trọng với A/B prompt/format.
- Shashidhar et al., YourBench: Easy Custom Evaluation Sets for Everyone, 2025 (arXiv 2504.01833) — https://arxiv.org/abs/2504.01833 — [đã mở] sinh câu từ tài liệu + lọc citation grounding; giữ thứ hạng MMLU (Spearman 1.0), ~85% câu đạt chất lượng khi người duyệt.
- Saad-Falcon et al., ARES, NAACL 2024 (arXiv 2311.09476) — https://arxiv.org/abs/2311.09476 — [đã mở] judge tự động + prediction-powered inference với vài trăm nhãn người để hiệu chỉnh khoảng tin cậy.
- Rahmani et al., Synthetic Test Collections for Retrieval Evaluation, SIGIR 2024 (arXiv 2405.07767) — https://arxiv.org/abs/2405.07767 — [đã mở] query + judgment tổng hợp cho kết quả xếp hạng gần test collection người làm; bias với hệ cùng LLM nhỏ.
- Filice et al., Generating Diverse Q&A Benchmarks for RAG Evaluation with DataMorgana, ACL 2025 Industry (arXiv 2501.12789) — https://arxiv.org/abs/2501.12789 — [search] cấu hình phân bố loại câu/người hỏi → câu đa dạng hơn; dùng cho SIGIR LiveRAG 2025.
