# REFERENCES — nguồn tham khảo cho paper Phase 2 (round_01, 2026-10-15)
> Gom từ docs/council/sources_researcher.md + round_01/sources.md + docs/research/*.md.
> Nhãn: **[đã mở]** = researcher tự WebFetch/đọc file và thấy khớp tiêu đề/tác giả/nội dung · **[mở 1 phần]** = mở được nhưng không thấy hết chi tiết được trích · **[không mở được]** · **[chưa kiểm]**.
> Venue ghi theo trang đã mở; nếu chỉ thấy arXiv thì ghi arXiv (cần tra lại venue cuối cùng trước khi nộp paper).
> Dùng cho: RW = related work · M = method · B = baseline/setting · E = evaluation.

## A. Dùng được (đã tự mở, khớp)

### A1. Cuộc thi, model, domain
| # | Trích dẫn kiểu IEEE | URL | Nhãn | Dùng cho |
|---|---|---|---|---|
| 1 | IEEE CIS Task Force on LLMs & CI for GPAIS, "IEEE CIS Technical Challenge on Domain-Specific Question Answering with LLMs," 2026 (Overview / Join the Challenge / Evaluation). | https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/ · …/join-the-challenge/ · …/evaluation/ | [đã mở] | B, E (luật, metric, trọng số 30/30/40) |
| 2 | Codabench, "IEEE CIS Technical Challenge on DSQA with LLMs," competition #17954, 2026. | https://www.codabench.org/competitions/17954/ (API: /api/competitions/17954/) | [đã mở qua API] | B (giới hạn 1 h/2 h, 1 GB, LB baselines 29.05/57.14) |
| 3 | Challenge organizers (GitHub user jucls), "IEEE-CIS DSQA Starter Kit," GitHub, commit 5e596be, 2026. | https://github.com/jucls/IEEE-CIS-DSQA-Starter-Kit | [đã đọc bản sao local starter_kit/] | B (01_zero_shot, 02_rag) |
| 4 | Team OLMo, "Olmo 3," arXiv:2512.13961, 2025. | https://arxiv.org/abs/2512.13961 | [đã mở] | B (model sinh) |
| 5 | Allen Institute for AI, "Olmo-3.1-32B-Instruct" (model card, tokenizer_config, generation_config), Hugging Face, 2025. | https://huggingface.co/allenai/Olmo-3.1-32B-Instruct | [đã mở] | B (ctx 65,536; temp 0.6/top_p 0.95; date cutoff Dec 2024) |
| 6 | D. Bernardi et al., "The IFMIF-DONES project: Design status and main achievements within the EUROfusion FP8 work programme," *J. Fusion Energy*, vol. 41, no. 2, art. 24, 2022, doi:10.1007/s10894-022-00337-5. | https://pmc.ncbi.nlm.nih.gov/articles/PMC9596351/ | [đã mở] | Giới thiệu domain DONES |
| 7 | F. S. Nitti et al., "Lithium loop and purification system of DONES: Preliminary design," preprint for 30th Symp. Fusion Technology (SOFT), EUROfusion WPENS-CPR(18) 19806, 2018. | https://scipub.euro-fusion.org/wp-content/uploads/eurofusion/WPENSCPR18_19806_submitted-4.pdf | [đã mở PDF] | Corpus dev DONES (Li loop, ICS, bẫy nitơ) |
| 8 | I. Triguero, D. Molina, J. Poyatos, J. Del Ser, F. Herrera, "General purpose artificial intelligence systems (GPAIS): Properties, definition, taxonomy, societal implications and responsible governance," *Information Fusion*, vol. 103, 2024. | https://arxiv.org/abs/2307.14283 | [đã mở] | RW (bối cảnh GPAIS); corpus dev AI |

### A2. Retrieval-augmented generation
| # | Trích dẫn | URL | Nhãn | Dùng cho |
|---|---|---|---|---|
| 9 | P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Proc. NeurIPS*, 2020. | https://arxiv.org/abs/2005.11401 | [đã mở] | RW |
| 10 | G. V. Cormack, C. L. A. Clarke, S. Büttcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in *Proc. ACM SIGIR*, 2009. | https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf | [đã mở PDF] | M (baseline fusion để so với fuzzy fusion) |
| 11 | Anthropic, "Introducing Contextual Retrieval," blog, Sep. 2024. | https://www.anthropic.com/news/contextual-retrieval | [đã mở] | M (header ngữ cảnh cho chunk) |
| 12 | M. Akarsu, R. K. Karaman, C. Mierbach, "From BM25 to corrective RAG: Benchmarking retrieval strategies for text-and-table documents," arXiv:2604.01733, 2026. | https://arxiv.org/abs/2604.01733 | [đã mở] | RW/M (hybrid+rerank, BM25 mạnh ở số/bảng) |
| 13 | L. Gao, X. Ma, J. Lin, J. Callan, "Precise zero-shot dense retrieval without relevance labels" (HyDE), arXiv:2212.10496, 2022. | https://arxiv.org/abs/2212.10496 | [đã mở] | RW |
| 14 | Z. Rackauckas, "RAG-Fusion: A new take on retrieval-augmented generation," arXiv:2402.03367, 2024. | https://arxiv.org/abs/2402.03367 | [đã mở] | RW |
| 15 | P. Sarthi et al., "RAPTOR: Recursive abstractive processing for tree-organized retrieval," arXiv:2401.18059, 2024. | https://arxiv.org/abs/2401.18059 | [đã mở] | RW |
| 16 | N. F. Liu et al., "Lost in the middle: How language models use long contexts," *TACL*, 2023. | https://arxiv.org/abs/2307.03172 | [đã mở] | M (thứ tự chunk) |
| 17 | X. Li, Y. Cao, Y. Ma, A. Sun, "Long context vs. RAG for LLMs: An evaluation and revisits," arXiv:2501.01880, 2024. | https://arxiv.org/abs/2501.01880 | [đã mở] | RW (document-first vs chunk) |
| 18 | A. Asai, Z. Wu, Y. Wang, A. Sil, H. Hajishirzi, "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," arXiv:2310.11511, 2023. | https://arxiv.org/abs/2310.11511 | [đã mở] | RW |
| 19 | M. D. Skarlinski et al., "Language agents achieve superhuman synthesis of scientific knowledge" (PaperQA2), arXiv:2409.13740, 2024. | https://arxiv.org/abs/2409.13740 | [đã mở] | RW (QA trên bài báo khoa học) |
| 20 | H. Chen et al., "SciRerankBench: Benchmarking rerankers towards scientific retrieval-augmented generation with LLMs," arXiv:2508.08742, 2025. | https://arxiv.org/abs/2508.08742 | [đã mở] | RW (rerank cho QA khoa học) |
| 21 | P. J. L. Ammann, J. Golde, A. Akbik, "Question decomposition for retrieval-augmented generation," in *Proc. ACL Student Research Workshop*, 2025. | https://aclanthology.org/2025.acl-srw.32/ | [đã mở] | RW (multi-hop) |
| 22 | J. Chen et al., "M3-Embedding (BGE-M3)…," arXiv:2402.03216, 2024. | https://arxiv.org/abs/2402.03216 | [đã mở] | RW (lý do KHÔNG dùng: 2.27 GB > 1 GB) |
| 23 | S. Xiao et al., "C-Pack: Packed resources for general Chinese embeddings" (BGE), in *Proc. SIGIR*, 2024. | https://arxiv.org/abs/2309.07597 | [đã mở] | M (nếu dùng bge-small/base) |
| 24 | C. Auer et al., "Docling technical report," arXiv:2408.09869, 2024. | https://arxiv.org/abs/2408.09869 | [đã mở] | B (parser của baseline) |
| 25 | Artifex, "PyMuPDF parser benchmark (ParseBench, 2,078 docs)," 2026 — nguồn của nhà phát triển PyMuPDF. | https://pymupdf.io/benchmark | [đã mở] | M (lý do đổi parser; ghi rõ là benchmark của vendor) |

### A3. Suy luận, giải mã
| # | Trích dẫn | URL | Nhãn | Dùng cho |
|---|---|---|---|---|
| 26 | X. Wang et al., "Self-consistency improves chain of thought reasoning in language models," in *Proc. ICLR*, 2023. | https://arxiv.org/abs/2203.11171 | [đã mở] | RW/M |
| 27 | W. Chen, X. Ma, X. Wang, W. W. Cohen, "Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks," *TMLR*, 2023. | https://arxiv.org/abs/2211.12588 | [đã mở] | M (câu tính toán) |
| 28 | Y. Leviathan, M. Kalman, Y. Matias, "Fast inference from transformers via speculative decoding," in *Proc. ICML*, 2023. | https://arxiv.org/abs/2211.17192 | [đã mở] | RW (lý do không dùng draft model) |
| 29 | Hugging Face, "Assisted decoding" (prompt lookup decoding, `prompt_lookup_num_tokens`), Transformers v5.14 docs. | https://huggingface.co/docs/transformers/v5.14.0/assisted_decoding | [đã mở] | M (tăng tốc không cần model phụ) |

### A4. Computational Intelligence (đóng góp chính cho 40% innovation)
| # | Trích dẫn | URL | Nhãn | Dùng cho |
|---|---|---|---|---|
| 30 | Q. Guo et al., "EvoPrompt: Connecting LLMs with evolutionary algorithms yields powerful prompt optimizers," in *Proc. ICLR*, 2024. | https://arxiv.org/abs/2309.08532 | [đã mở] | M/RW |
| 31 | C. Fernando et al., "Promptbreeder: Self-referential self-improvement via prompt evolution," arXiv:2309.16797, 2023. | https://arxiv.org/abs/2309.16797 | [đã mở] | RW |
| 32 | L. A. Agrawal et al., "GEPA: Reflective prompt evolution can outperform reinforcement learning," in *Proc. ICLR*, 2026. | https://arxiv.org/abs/2507.19457 | [đã mở] | M/RW |
| 33 | O. Khattab et al., "DSPy: Compiling declarative language model calls into self-improving pipelines," arXiv:2310.03714, 2023. | https://arxiv.org/abs/2310.03714 | [đã mở] | RW |
| 34 | F. Hosseini, M. Eftekhari, "Allo-Self-RAG: Fuzzy aggregation of internal and external critique signals for improved Self-RAG evaluation," *Iranian J. Fuzzy Systems*, 2026. | https://ijfs.usb.ac.ir/article_9936.html | [đã mở] | RW (fuzzy + SA cho RAG — tiền lệ gần nhất) |

### A5. Đánh giá bằng LLM judge
| # | Trích dẫn | URL | Nhãn | Dùng cho |
|---|---|---|---|---|
| 35 | L. Zheng et al., "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena," in *Proc. NeurIPS Datasets & Benchmarks*, 2023. | https://arxiv.org/abs/2306.05685 | [đã mở] | E |
| 36 | J. Ye et al., "Justice or prejudice? Quantifying biases in LLM-as-a-judge," arXiv:2410.02736, 2024. | https://arxiv.org/abs/2410.02736 | [đã mở] | E |

### A6. Lời giải cuộc thi tương tự (tham khảo kỹ thuật, không phải paper)
| # | Nguồn | URL | Nhãn | Dùng cho |
|---|---|---|---|---|
| 37 | A. Maity, "WattBot 2025 — RAG system for technical document Q&A" (5th private, Kaggle), GitHub. | https://github.com/anubhavmaity/wattbot | [đã mở] | RW/M (BM25 > dense; chất lượng trích PDF quan trọng) |
| 38 | TIMETOACT, "Winners of the Enterprise RAG Challenge, Edition 2," Feb. 2025 (winner: router + dense + LLM rerank). | https://www.timetoact-group.com/content-share/tta-at/insights/the-winner-of-the-enterprise-rag-challenge-78846 | [mở 1 phần — qua kết quả tìm kiếm] | RW |
| 39 | BAAI, "bge-reranker-v2-m3" model card (0.6B, Apache-2.0). | https://huggingface.co/BAAI/bge-reranker-v2-m3 | [đã mở] | B (loại vì > 1 GB) |
| 40 | IFMIF-DONES España, news: "UGR and IFMIF-DONES challenge the international scientific community…," ifmif-dones.es, 2026. | https://ifmif-dones.es/dones-updates/ugr-and-ifmif-dones-challenge-the-international-scientific-community-to-develop-an-expert-ai-that-works-with-limited-resources-and-without-sending-data-off-site/ | [đã mở] | Giới thiệu (100 câu do chuyên gia soạn) |

## B. Chưa kiểm chứng / không mở được (KHÔNG trích dẫn trước khi tự mở)
| Nguồn | URL | Nhãn | Ghi chú |
|---|---|---|---|
| J. Gao et al., "Evaluating and mitigating LLM-as-a-judge bias in communication systems," arXiv:2510.12462 | https://arxiv.org/abs/2510.12462 | [mở 1 phần] | Bài tồn tại; con số "9.12→4.00" do qwen trích KHÔNG thấy trong abstract |
| Verbosity bias in preference labeling by LLMs, arXiv:2310.10076 | https://ar5iv.labs.arxiv.org/html/2310.10076 | [chưa kiểm] | do qwen nêu |
| Explaining length bias in LLM-based preference evaluations, arXiv:2407.01085 | https://arxiv.org/html/2407.01085v3 | [chưa kiểm] | do qwen nêu |
| Agent-UniRAG, arXiv:2505.22571 | https://arxiv.org/abs/2505.22571v1 | [chưa kiểm] | do kimi nêu |
| MODE: Mixture of document experts for RAG, arXiv:2509.00100 | https://arxiv.org/html/2509.00100v1 | [chưa kiểm] | Perplexity deep |
| Multi-HyDE financial RAG, arXiv:2509.16369 | https://arxiv.org/html/2509.16369v1 | [chưa kiểm] | Perplexity deep |
| Vision-guided chunking, arXiv:2506.16035 | https://arxiv.org/html/2506.16035v1 | [chưa kiểm] | Perplexity deep |
| RAG vs long context (NEPA), arXiv:2407.07321; RAG or LC hybrid, arXiv:2407.16833 | https://arxiv.org/html/2407.07321v1 · https://arxiv.org/html/2407.16833v1 | [chưa kiểm] | Perplexity deep |
| "How we won the RAG challenge" (bản đăng lại) | https://www.besthub.dev/articles/how-we-won-the-rag-challenge-multi-router-dynamic-knowledge-base-techniques-revealed-b9bb4df2e8d1 | [chưa kiểm] | dùng #38 thay thế |
| IFMIF-DONES buildings and plant services (Luque et al.) | https://www.researchgate.net/publication/389721709 | [không mở được] | grok dùng để nói gold q9 lệch |
| IFMIF-DONES Li loop / Be7 / tritium (KIT, Springer s10894-023-00378-4) | https://publikationen.bibliothek.kit.edu/1000174828/154977142 · https://link.springer.com/article/10.1007/s10894-023-00378-4 | [không mở được] (Springer chuyển hướng đăng nhập) | ứng viên corpus dev |
| IEEE CIS activities page (mục tiêu CI, giải thưởng) | https://cis.ieee.org/activities/technical-activities/2026-technical-challenge-on-domain-specific-question-answering-with-llms | [không mở được] (HTTP 418) | dùng #1 thay thế |
| Contest Rules PDF | https://cis.taskforce.ieee.org/gpais/wp-content/uploads/sites/6/2026/09/IEEE-CIS-Technical-Challenge-on-Domain-Specific-Question-Answering-with-LLMs-Contest-Rules.pdf | [đã mở PDF] | không nói về model phụ; chỉ "verify compliance" |
| FLAME 2024 kết quả (CiTIUS hạng 3, LLM4AD hạng 2) | https://citius.gal/en/news/news/international-podium-for-citiuss-explainable-artificial-intelligence/ | [chưa kiểm — chỉ kết quả tìm kiếm] | |
| Các nguồn còn lại trong round_01/sources.md (171 link, chủ yếu từ grok) | docs/council/round_01/sources.md | [chưa kiểm] | |

## C. Gợi ý cấu trúc trích dẫn trong paper 4 trang
- Introduction / setting: #1, #2, #4, #6, #40.
- Related work: RAG #9, #18, #19; hybrid/fusion #10, #11, #12; chunk/context #11, #16, #17; prompt evolution #30–#33; fuzzy RAG #34; judge #35–#36.
- Method: parser #24/#25, fusion #10 (baseline) vs fuzzy-evolutionary fusion (đóng góp), PoT #27, prompt lookup #29, prompt evolution #30/#32.
- Baseline: 02_rag của BTC (#3) + điểm LB 57.14 (#2).
