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
| `eval/compare_runs.py` | So 2 run theo id: số câu khác chữ, lỗi, token, latency (+ verdict lật khi có) | — | đo nhiễu / lệch shim |

## Luồng dữ liệu

## Muốn sửa X thì vào đâu

## Vùng nguy hiểm
- data/raw/, data/splits/, eval/ (sau khi chốt) — được bảo vệ bởi hook
