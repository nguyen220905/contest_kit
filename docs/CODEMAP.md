# CODEMAP — bản đồ codebase
> Chạy /map để subagent code-mapper cập nhật tự động. Agent nên đọc file này thay vì đọc toàn bộ code.

## Tổng quan
Chưa có pipeline của đội. `starter_kit/` = code mẫu của BTC (chỉ đọc).

## Điểm vào & lệnh
- `scripts/smoke_test.py` — kiểm môi trường dev (GPU, transformers 5.17.0, load + generate Olmo-3-7B).

## Module
| Đường dẫn | Trách nhiệm | Phụ thuộc | Ghi chú |
|---|---|---|---|
| `starter_kit/01_zero_shot/main.py` | Baseline BTC không retrieval | transformers | LB 29.05 / 733 s |
| `starter_kit/02_rag/main.py` | Baseline BTC: Docling + MiniLM CPU + cosine, ctx 10k | docling, sentence-transformers | LB 57.14 / 2080 s |
| `starter_kit/sample/` | 25 câu mẫu có gold + danh sách site nguồn corpus | — | dev data |
| `scripts/smoke_test.py` | Smoke test môi trường | transformers, bitsandbytes | `--quant 4bit` mặc định |
| `eval/run_pipeline.py` | Harness: import `<submission>/main.py`, gọi `answer()` tuần tự 1 process như server → `results/<run>/predictions.csv` (ghi từng dòng) + `run_info.json` (thời gian, token, VRAM, dự phóng public/private) | torch, transformers, `eval/dev_shim.py` | `--split sample25` đọc thẳng starter_kit khi chưa có `data/splits/` |
| `eval/dev_shim.py` | CHỈ local: vá `AutoModelForCausalLM.from_pretrained` chèn bnb 4/8-bit | bitsandbytes | server không có |
| `submissions/d1_02_safe/main.py` | Bài nộp D1: 02_rag + vá an toàn (PDF đệ quy/hoa thường, cache cả trạng thái lỗi, try/except) | docling, sentence-transformers | build: `submissions/d1_02_safe/build.py` → `submissions/d1_02_safe.zip` |
| `eval/judge.py` | Judge nhị phân (DeepSeek, prompt judge-v2) + client API chung (nạp tools/.env, cache `data/work/llm_cache.jsonl`); CLI thêm cột verdict vào predictions.csv | tools/council_config.json | `--refs data/work/sample25_refs.jsonl` |
| `eval/evaluate.py` | = run_pipeline (subprocess) + judge + `metrics.json` (balanced = mean acc 2 domain, theo type, thời gian, quote_recall@k, in_window_hit) | judge.py, run_pipeline.py | `--judge-only` để chấm lại |
| `eval/calibrate_judge.py` | `--make-refs` (nháp reference_short, GPT), `--make-variants`, chạy 4 test hiệu chỉnh → results/calibration/ | judge.py | |
| `dsqa/parse.py`, `dsqa/chunk.py`, `dsqa/retrieve.py` | Thư viện dùng chung (bundle vào bài nộp): parse PyMuPDF/Docling, chunk có tiêu đề, Retriever dense/BM25/hybrid RRF/rerank | sentence-transformers, rank_bm25 | lab = bài nộp (cùng code) |
| `eval/retrieval_eval.py` | Phòng thí nghiệm retrieval không LLM: quote_recall@k, doc_hit@k, MRR cho cấu hình c0..c11; cache parse/embedding ở `data/work/` | dsqa | model ở `data/work/models/` |
| `submissions/d3_hybrid_title/` | D3 = D1 + retrieval c9 (bundle dsqa/) | dsqa | build.py → d3_hybrid_title.zip |
| `dsqa/llm_rerank.py` | LLMReranker: Olmo chấm Yes/No từng chunk (dùng lại model sinh), truyền vào `Retriever(rerank_fn=...)` | transformers | lab c25/c26 (7B) |
| `submissions/d6_gated_rerank/` | D6 = D5 nhưng câu khớp `_DONES_RE` giữ thứ tự hybrid D3 (tách tác dụng rerank theo domain) | dsqa | build.py → d6_gated_rerank.zip |
| `submissions/d9_dones_ctx20k/` | D9 = D3 + câu DONES context 20k token (câu AI 10k như D3) | dsqa | build.py → d9_dones_ctx20k.zip |
| `submissions/d8_dones_calc/` | D8 = D3 + câu DONES: lượt tự kiểm phép tính (model viết Python → chạy `python -I` có timeout → trả lời lại); `_safe_code` chỉ cho import math/fractions/decimal/statistics | dsqa | build.py → d8_dones_calc.zip |
| `submissions/d7_probe_dones/` | D7 thăm dò = D3 nhưng câu không khớp `_DONES_RE` trả "" (đo số câu DONES đúng của D3) | dsqa | build.py → d7_probe_dones.zip (399 MB) |
| `submissions/d5_gte_rerank/` | D5 = D3 + CrossEncoder gte-reranker-modernbert-base fp16 rerank top-100 (lab c29), fallback về thứ tự hybrid | dsqa | build.py → d5_gte_rerank.zip (giải nén 737 MB) |
| `scripts/make_syn_dev.py` | Tập dev tổng hợp từ text Docling đã cache của C200: chọn passage → generator viết câu + đáp án + quote → lọc (quote có thật, verifier + judge) → `data/work/syn/<name>.jsonl` (schema sample25) | eval/judge.py | `--gen/--ver` = provider; `--limit N` thử; cache LLM nên chạy lại miễn phí |
| `eval/compare_runs.py` | So 2 run theo id: số câu khác chữ, lỗi, token, latency (+ verdict lật khi có) | — | đo nhiễu / lệch shim |

## Luồng dữ liệu

## Muốn sửa X thì vào đâu

## Vùng nguy hiểm
- data/raw/, data/splits/, eval/ (sau khi chốt) — được bảo vệ bởi hook
