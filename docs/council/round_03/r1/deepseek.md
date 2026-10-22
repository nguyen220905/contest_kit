# M1 = deepseek (deepseek-v4-pro) — vai: plan_critic — vòng 1

## 1. Chẩn đoán nhanh
Chấm plan: **5/10 — sửa lớn**. Kết quả đang dừng ở mức baseline 57.14 vì chưa có harness/corpus để đo local và vì đội chưa làm bất kỳ thay đổi nào lên code 01/02; mọi cải tiến P4–P6 còn là giả định. Ba lỗi lớn nhất của plan: (1) dùng câu tổng hợp làm holdout để chọn thay đổi **generator** trong khi chính tài liệu tham khảo van Elburg nói synthetic rank retriever ổn nhưng rank generator gần như ngẫu nhiên; (2) bắt buộc shim 4-bit dù 7B bf16 vừa 16 GB, tạo thêm nguồn nhiễu khi so với server; (3) xây corpus C50/C200/C500 + 120 câu tổng hợp trước khi chạy baseline lần đầu, dễ tốn 5–6 ngày cho một bộ eval chưa biết có tương quan với LB hay không.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)
### Đề xuất A: Đảo baseline lên trước; chạy 7B bf16 thay vì shim 4-bit
- Giả thuyết: 7B bf16 vừa VRAM 16 GB (13.6 GiB) và gần hành vi server hơn 4-bit; việc chạy baseline 01/02 lên đầu giúp phát hiện sớm lỗi harness, đo thời gian thực, kiểm tra 02 > 01 trước khi đổ công sức vào corpus/synthetic.
- Dựa trên bằng chứng nào trong brief: Brief ghi RTX 5060 Ti 16 GB, bf16 13.6 GiB, 4-bit 4.9 GiB; plan đặt `DSQA_DEV_QUANT=4bit` mặc định; chưa có thí nghiệm nào; Stage 6 mới chạy baseline sau 5 ngày.
- Cách làm cụ thể: Ngày đầu viết harness tối thiểu để chạy `01_zero_shot` trên `sample25`, so sánh 4-bit vs bf16 cùng `max_new_tokens=2048, greedy`, ghi VRAM đỉnh/tok/s/độ dài answer. Khi S3 có tài liệu gold, chạy `02_rag` trên `sample25` ngay với C50, so sánh 4-bit/bf16 và đo retrieval. Không xây C200/C500 hay sinh câu tổng hợp trước bước này.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu >3/25 câu trả lời khác nhau giữa 4-bit và bf16, hoặc balanced accuracy lệch >2 điểm trên sample25, thì bỏ 4-bit cho toàn bộ eval. Nếu bf16 OOM khi chạy 02 cùng Docling/embedder trong một process, mới phải dùng 4-bit và ghi rõ lý do. Nếu 02 không vượt 01 trên sample25 đã có gold tài liệu, dừng xây eval đầy đủ để sửa baseline trước.
- Kỳ vọng: giảm lệch đánh giá do quantization từ khoảng ±10 điểm xuống <2 điểm trên 25 câu; không trực tiếp tăng LB nhưng tránh chọn sai hướng ở P4–P5. | Tự tin: cao | Chi phí: S

### Đề xuất B: Không dùng câu tổng hợp làm holdout cho generator; chỉ dùng cho retrieval
- Giả thuyết: Câu tổng hợp sinh từ DeepSeek + few-shot bằng 25 câu mẫu sẽ bị lệch tác vụ, phong cách, và có thể nhiễm do dùng chính sample25 làm prompt; dùng nó để so sánh prompt/chunking/output format ở P5–P6 sẽ cho thứ hạng generator gần nhiễu (theo van Elburg τ≈0). Holdout dùng cho quyết định generator phải là câu thật, mở rộng thủ công.
- Dựa trên bằng chứng nào trong brief: Brief trích van Elburg et al. “synthetic xếp hạng retriever đúng nhưng sai khi so generator/LLM (τ≈0 hoặc âm)”; đồng thời 13/15 câu AI nêu tên paper, nên câu tổng hợp dễ bỏ qua/thiên về tài liệu tạo ra.
- Cách làm cụ thể: S5 chỉ tạo ≤60 câu tổng hợp, dùng cho retrieval-only và sanity-check. Giữ `sample25` làm holdout cuối cho generator, nhưng **không** dùng 25 câu này làm few-shot sinh câu. Thay vào đó tự mở rộng 15–20 câu thật từ các paper khác trong C200 (không trùng với sample25), có gold do người viết, để làm holdout generator. Mọi quyết định về prompt/output format ở P5 phải căn cứ trên `sample25` + câu thật mở rộng; synthetic không được dùng để chốt A/B.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Chạy 01 vs 02 (hoặc 02 vs một biến thể prompt đơn giản) trên `sample25` và trên tập synthetic. Nếu synthetic xếp hạng hai hệ ngược dấu so với `sample25`/câu thật, bác bỏ hoàn toàn việc dùng synthetic cho generator. Nếu sau 2 lần hiệu chỉnh rubric mà synthetic vẫn không phân biệt được hệ đã biết (ví dụ 02 không >01), loại synthetic khỏi eval generator.
- Kỳ vọng: tránh lãng phí ~1 ngày sinh 120 câu + 20 câu duyệt; giảm rủi ro đi sai hướng P5/P6. | Tự tin: vừa | Chi phí: M

### Đề xuất C: Cắt C500 và giảm khối lượng synthetic; đưa H100 lên chạy baseline 32B sớm có điều kiện
- Giả thuyết: C200 đủ để đại diện corpus thật; C500 chỉ làm tăng thời gian index và chạy 7B, không thêm nhiều tín hiệu. H100 chạy 32B sớm sẽ giúp phát hiện lệch giữa dev local và LB trước khi nộp P4, thay vì chờ đến P1b.
- Dựa trên bằng chứng nào trong brief: Giả thuyết corpus thật 150–400 PDF; 02_rag mất 2080 s; H100 budget có sẵn nhưng plan để cuối. C500 chứa 500 PDF trong khi mọi tài liệu gold đã nằm trong C200.
- Cách làm cụ thể: Chỉ xây C50 và C200; bỏ C500 ở P2/P3. Sau khi S1/S3 chạy được `02_rag` trên sample25 với C200, thuê H100 1–2 h để chạy 01/02 bằng 32B trên sample25 (và nếu rẻ thì holdout ~50). So sánh kết quả local với LB: 32B zero-shot 29.05, 32B RAG 57.14. Nếu lệch tuyệt đối >10 điểm so với hai mốc đó, xem lại judge/corpus/question distribution trước khi nộp.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu C200 không đạt ≥20/25 câu có doc-hit@1 trong retrieval-only, tăng lên C300 nhưng vẫn không cần C500. Nếu H100 chạy 32B cho 25 câu vượt quá 1 h, bỏ bớt holdout, chỉ chạy sample25. Nếu 32B local khớp LB trong ±5 điểm, có thể dùng 7B để chọn P4–P5 nhưng vẫn phải chạy lại 32B trước khi nộp.
- Kỳ vọng: rút ngắn 1–2 ngày so với plan; tăng độ tin cậy quyết định P4 bằng cách đối chiếu 32B sớm. | Tự tin: vừa | Chi phí: M

**Thứ tự top-5 thay thế:**
1. S1 + baseline 01/02 trên `sample25` — phát hiện sớm lỗi harness + so 4-bit/bf16.
2. S3 gold docs + C50 — đủ để chạy 02_rag, kiểm tra retrieval.
3. Judge calibration trên 25 câu gốc + 15–20 câu thật mở rộng — không dùng synthetic làm trọng tài generator.
4. H100 chạy 01/02 32B — đối chiếu dev/LB trước khi nộp bài #1.
5. S4 C200 + ≤60 câu synthetic retrieval-only — phục vụ A/B retrieval, không phải generator.

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Chưa thấy code `main.py` của 01/02; cần biết cách gọi model, có cache không, có lỗi nào khác ngoài `glob("*.pdf")` không đệ quy và `answer()` không try/except.
- Giả thuyết corpus thật 150–400 PDF mới chỉ suy từ thời gian 2080 s, chưa có phép đo page/throughput thực tế của Docling trên server; cần chạy một lần với corpus đã biết trang để hiệu chỉnh.
- `reference_short` tự rút gọn từ gold dài có thể không khớp tham chiếu “deliberately concise” ẩn, chưa có cách kiểm chứng.
- Chưa có ước lượng chi phí API cho J1 DeepSeek, J2 OpenRouter và 120 câu tổng hợp; nếu vượt hạn mức $7/ngày, whole S5/J2 đổ vỡ.
- Chưa biết độ dài sinh thực tế của 02_rag trên sample25; `max_new_tokens=2048` có thể gây câu trả lời dư thừa hoặc chậm ngoài dự kiến, ảnh hưởng ngưỡng 1 h.
- Điều kiện “02 > 01 trên cả 3 split” đang phụ thuộc vào sample25/holdout tự tạo, trong khi LB chỉ cho điểm tổng 50 câu và không trả output, nên khó biết local eval có tương quan với server hay không.

## 4. Nguồn đã tra cứu
- Không tra cứu thêm ngoài brief; các dẫn chứng van Elburg, Krumdick, Thakur lấy từ mục 7 của brief. Không viết URL từ trí nhớ.
