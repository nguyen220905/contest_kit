# REFERENCES — round_02 (chấm plan eval harness + corpus dev) — 2026-10-22
> Chỉ nguồn CHƯA có trong round_01/REFERENCES.md. Nhãn: **[đã mở]** = researcher WebFetch trang arXiv abs/html, khớp tiêu đề/tác giả/nội dung · **[search]** = chỉ thấy qua kết quả tìm kiếm.
> Hội đồng round_02 không tra web (`--no-search`) và không nêu URL nào → mọi nguồn dưới do researcher tự tìm. Venue ghi theo trang đã mở; tra lại venue cuối trước khi nộp paper.
> Dùng cho: E = evaluation/phương pháp đánh giá trong paper · RW = related work.

| # | Trích dẫn kiểu IEEE | URL | Nhãn | Dùng cho |
|---|---|---|---|---|
| 1 | J. van Elburg, P. van der Putten, and M. Marx, "Can we evaluate RAGs with synthetic data?," arXiv:2508.11758, 2025. | https://arxiv.org/abs/2508.11758 | [đã mở] | E — lý do dùng câu tổng hợp cho chọn retriever, không cho chọn generator (τ 0.44–0.75 vs ≈0) |
| 2 | M. Krumdick, C. Lovering, V. Reddy, S. Ebner, and C. Tanner, "No free labels: Limitations of LLM-as-a-judge without human grounding," arXiv:2503.05061, 2025. | https://arxiv.org/abs/2503.05061 | [đã mở] | E — judge cần tham chiếu đúng, người duyệt `reference_short` (kappa 0.14 → 0.78) |
| 3 | A. S. Thakur, K. Choudhary, V. S. Ramayapally, S. Vaidyanathan, and D. Hupkes, "Judging the judges: Evaluating alignment and vulnerabilities in LLMs-as-judges," in Proc. GEM Workshop, 2025, arXiv:2406.12624. | https://arxiv.org/abs/2406.12624 | [đã mở] | E — báo Cohen's kappa, test leniency/độ dài của judge local |
| 4 | S. Badshah and H. Sajjad, "Reference-guided verdict: LLMs-as-judges in automatic evaluation of free-form QA," in Proc. Widening NLP Workshop, arXiv:2408.09235, 2024. | https://arxiv.org/abs/2408.09235 | [đã mở] | E — dùng nhiều judge (J1/J2) |
| 5 | S. Shashidhar, C. Fourrier, A. Lozovskia, T. Wolf, G. Tur, and D. Hakkani-Tür, "YourBench: Easy custom evaluation sets for everyone," arXiv:2504.01833, 2025. | https://arxiv.org/abs/2504.01833 | [đã mở] | E/RW — sinh câu từ tài liệu + lọc citation grounding (giống kiểm quote ở S5) |
| 6 | J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, "ARES: An automated evaluation framework for retrieval-augmented generation systems," in Proc. NAACL, 2024, arXiv:2311.09476. | https://arxiv.org/abs/2311.09476 | [đã mở] | E/RW — judge tự động + ít nhãn người (prediction-powered inference) |
| 7 | H. A. Rahmani, N. Craswell, E. Yilmaz, B. Mitra, and D. Campos, "Synthetic test collections for retrieval evaluation," in Proc. SIGIR, 2024, arXiv:2405.07767. | https://arxiv.org/abs/2405.07767 | [đã mở] | E — đo retrieval (doc-hit@k) bằng câu tổng hợp là có cơ sở |
| 8 | S. Filice et al., "Generating diverse Q&A benchmarks for RAG evaluation with DataMorgana," in Proc. ACL Industry Track, 2025, arXiv:2501.12789. | https://arxiv.org/abs/2501.12789 | [search] | RW — điều khiển phân bố loại câu khi sinh |

Ghi chú: Perplexity không dùng được phiên này (OpenRouter 401 "API key expired") nên không có báo cáo mới trong docs/research/.
