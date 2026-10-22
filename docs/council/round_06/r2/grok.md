# M5 = grok (x-ai/grok-4.7) — vai: red_team — vòng 2

## 1. Nhận xét từng thành viên
- M1: điểm mạnh — audit `reference_short` và điểm quy đổi 0,3·DONES+0,7·AI bắt đúng hai chỗ local có thể đảo 01/02 so với LB. | lỗ hổng — không đụng ngân sách giờ và zip nào được phép thành best public; Wilson trên slice 3–4 câu rộng đến mức không bác được giả thuyết nào.
- M2: điểm mạnh — control trước khi đụng C200/synthetic, và không nhân wall-clock A100 thành runtime H100; McNemar 6–0 là ngưỡng đúng cho n nhỏ. | lỗ hổng — ngày 2 đổi `max_new_tokens` khi chưa có sàn nhiễu cùng zip và chưa biết 2080 s nằm ở parse hay decode.
- M3: điểm mạnh — “khẩu vị” độ dài của judge ẩn chỉ đo được trên LB, J1 không neo được. | lỗ hổng — ba probe format trước sàn nhiễu; zip may mắn thành best public thì private khóa đúng zip đó. Trích RankGPT/GPT-4 không suy ra Olmo pointwise. Đề xuất B là viết lại, không phải check rẻ.
- M4: điểm mạnh — oracle-chunk tách retrieval khỏi generator, falsifiable, không cần C200. | lỗ hổng — vẫn chấm 7B 4-bit bằng J1; “oracle” lấy từ PyMuPDF có thể không phải text Docling mà model nhìn, và chưa kiểm cửa sổ 4096.
- M6: điểm mạnh — bẫy `generation_config` không ép greedy, `glob` phân biệt hoa/thường, fork sau CUDA, shim lọt zip. | lỗ hổng — mỗi bài gộp nhiều nút; +5 đến +9 điểm không có cơ chế quy được. 24/32 lớp SWA là số của 7B, không phải 32B. Công thức 0,90/0,42/0,62 là độ chính xác giả.
- M7: điểm mạnh — probe P và hệ thức \(T \approx P+2(T_{50}-P)\): nếu private cũng 1 h và corpus cùng cỡ thì 02 chỉ an toàn khi \(P \ge 560\) s. Lattice 57,14 hữu ích. | lỗ hổng — cố tình raise để kéo số trang corpus qua “lỗi ngắn” là lấy dữ liệu ẩn qua kênh lỗi, rủi ro DQ; tôi không nhận. Lattice thiếu nghiệm (3, 33). “1 h” (M7) mâu thuẫn “private 2 h” trong brief và entry 30/11 của M8 — chưa phân xử.
- M8: điểm mạnh — cắt S5 khỏi đường tới nộp, harness phải chạy Linux, một nút mỗi bài. | lỗ hổng — ngày 1 vẫn gộp glob đệ quy + try/except + watchdog; +3,3 đến +7 từ độ dài chưa có sàn nhiễu. Đòi số nộp lại 01/02 trong khi brief đã loại việc đó.

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)
Giữ: doc-hit trên quote PyMuPDF không phải thứ 02 nhìn thấy; `reference_short` rút từ gold dài không được làm cổng generator; 7B 4-bit/Ollama Q4 không được chấp nhận hay bác prompt; zip khác nhau mà thắng nhờ nhiễu sẽ bị private chấm lại.

Bỏ: ngày 1 cắt 2048→384 (làm hỏng control — M2-A đúng hơn). Bỏ khẳng định 48/64 lớp: đó là ngoại suy từ 24+8 của 7B. Bỏ ý “nộp lại cùng zip khóa một pipeline khác”: cùng zip chỉ khóa cùng code; hại là điểm public bị thổi và private là một lần quay mới (M7). Đảo chunk không được nộp trước khi đo cửa sổ.

Sửa điểm: **6,5/10 — sửa nhỏ rồi làm**, chưa đủ tốt. v1→v2 đúng hướng (cách ly sample25, synthetic không xếp generator, C500 không chặn). Thừa/sai lúc này: kappa ≥ 0,8 làm cổng, S3b bắt doc-hit trên code không có doc/page, chạy LLM trên câu tổng hợp. Ba mục dưới là chặn, không phải tuỳ chọn.

## 3. Đề xuất cuối (tối đa 3)
### Đề xuất A: Control happy-path rồi probe P, cấm khóa zip (dựa trên: M2-A, M7-A)
- Giả thuyết: 2080 s chưa tách được parse khỏi decode; nếu private bị 1 h (nguồn M7, mâu thuẫn brief — coi là rủi ro cho đến khi đọc lại trang evaluation) và corpus cùng cỡ, pipeline 2048 token fail cả bài. Zip khác 02 mà vô tình thành best public sẽ bị chấm private.
- Cách làm cụ thể: Preflight trên Linux (WSL/docker), không phải Windows: unzip, `main.py` ở gốc, không có `eval/` hay `dev_shim.py`, không load LLM thứ hai, không 4-bit. Ép `do_sample=False`, `temperature=None` trong `generate` (không tin `generation_config`). Diff `chat_template` 7B vs 32B, chỉ file. Ngày 1: 02 nguyên + try/except từng câu chỉ khi exception (fallback cố định); không đổi glob, prompt, thứ tự chunk, `max_new_tokens`. Ngày 2: đúng zip ngày 1. Ngày 3, chỉ sau khi ngày 1 Finished và điểm ≥ 29: sau khi build index, `answer()` trả chuỗi hằng. Không raise, không in số trang/tên file — chỉ lấy thời gian server trả về.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Bác “happy-path khớp BTC” nếu ngày 1 không Finished, hoặc |điểm − 57,14| ≥ 1,43, hoặc |thời gian − 2080| / 2080 > 15%. Sàn nhiễu = |ngày 1 − ngày 2|. Bác “1 câu LB là tín hiệu” nếu sàn ≥ 1,43. Giả định corpus private = public: \(T_{100} \approx P + 2(T_{ngày1}-P)\). Bác “02 an toàn giờ” nếu \(T_{100} > 0{,}8\times\) ngân sách đang tranh chấp (2880 s nếu 1 h, 5760 s nếu 2 h). Zip mới chỉ được phép vượt best nếu điểm ≥ best + sàn nhiễu; probe P cố ý điểm thấp nên không được nộp khi chưa có Finished ≥ 29.
- Kỳ vọng: +0 điểm; tránh mất cả private hoặc khóa zip sai (1 câu DONES = 3,33 điểm LB) | Tự tin: cao | Chi phí: S

### Đề xuất B: in-window hit + oracle, không dùng doc-hit làm cổng (dựa trên: M4-A, M4-B, M1-C)
- Giả thuyết: S3b/S6 sẽ chọn retrieval bằng doc-hit trên quote PyMuPDF, trong khi 02 nhồi chunk điểm cao lên đầu, ctx 10k, và Olmo có `sliding_window` 4096 (đã có trong config 7B; pattern lớp của 32B chưa đọc). Quote có thể vắng trong markdown Docling. Near-dup arXiv/publisher làm hit ảo.
- Cách làm cụ thể: Ngay khi có ≥ 5 PDF gold, không đợi 20/25. (1) Đọc `config.json` 32B: `sliding_window` và danh sách loại lớp — không cần GPU. (2) Cùng cờ Docling của starter, fuzzy ≥ 0,9 quote trong markdown vs PyMuPDF. (3) Pack đúng 02 (best trước, ctx 10k, câu hỏi cuối); ghi quote có nằm trong 4096 token cuối hay không. (4) Trên 10 câu có quote: context retrieved vs context chỉ là đoạn quote; so khớp số/đơn vị trong `reference`, chưa dùng điểm J1. Đếm near-dup Jaccard ≥ 0,9 trên text đã trích (M1-C) trước khi tin hit@k.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Bác giả thuyết cửa sổ/parser nếu config 32B không có SWA, hoặc ≥ 90% quote có trong markdown Docling và ≥ 80% số đó nằm trong 4096 token cuối. Bác “retrieval là nút thắt” nếu oracle không tăng khớp số thêm ≥ 3/10 câu. Nếu ≥ 30% quote ngoài cửa sổ, hoặc near-dup > 10% PDF và hit@10 lệch > 5 điểm % sau dedup: cấm cổng “02 > 01” / doc-hit@k; chỉ chọn theo `in_window_hit`. Ablation 7B không được phép đổi prompt nộp.
- Kỳ vọng: +0 nếu bác; +5 đến +15 balanced acc local trên câu đã đúng tài liệu nếu ≥ 30% quote ngoài cửa sổ và oracle xác nhận | Tự tin: vừa | Chi phí: S

### Đề xuất C: Khóa cổng generator cho đến khi nhãn concise giữ đủ số (dựa trên: M1-A, M2-B)
- Giả thuyết: Bộ test J1 chỉ đo tự nhất quán với `reference_short` do LLM rút từ gold 81–2677 ký tự, không phải tham chiếu ẩn concise. Krumdick (nguồn M1/M2): reference sai hại hơn không reference. Smoke 7B 4-bit chạm trần 256/256 nên verdict local đang chấm bài cụt. Kappa ≥ 0,8 có thể fail vì paradox dù agreement cao (M7) — không được là cổng.
- Cách làm cụ thể: Trước S5 và trước mọi đổi prompt/trần/PoT: người viết `reference_concise` 1–2 câu cho 25 mẫu, giữ mọi số, đơn vị, thực thể; gắn `outside_doc`, `false_premise`, `closed_book` (DeepSeek không corpus, người duyệt). Ma trận lật trên cùng bộ câu trả lời: gold / `reference_short` / `reference_concise`. Bỏ kappa làm cổng; chỉ ghi bảng bất đồng. Generator chỉ được đổi sau đề xuất A (đã có sàn nhiễu) và chỉ khi 7B **bf16**, prompt 02, context vừa 16 GB, có p95 `n_new_tokens` < 0,5×trần. 4-bit và Ollama không được chấp nhận hay bác.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Bác “J1 dùng được để chọn generator” nếu lật short↔concise ≥ 4/25, hoặc thứ tự 01 vs 02 đảo, hoặc J1 bác ≥ 5/25 câu người viết đủ fact và không mâu thuẫn. Bác “7B không bị trần chi phối” nếu p95 bf16 ≥ 0,8×trần. Khi bác: không nộp biến prompt; chỉ nộp biến đã qua A/B. Nếu lật ≤ 2/25 và thứ tự 01/02 không đổi: giữ S2, đề xuất này dừng.
- Kỳ vọng: +0 trực tiếp; tránh lệch hướng ±2–4 điểm local trên n=25 | Tự tin: vừa | Chi phí: S

## 4. Bình chọn
VOTES: M7-A, M4-A, M2-A
