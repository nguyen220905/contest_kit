# M1 = deepseek (deepseek-v4-pro) — vai: simplifier — vòng 1

## 1. Chẩn đoán nhanh
Kết quả chưa có vì kế hoạch P2+P3 đang đầu tư 5–6 ngày cho hạ tầng eval (2 judge, 3 corpus, 150 câu tổng hợp) trước khi có bất kỳ biến thể sinh câu đáng nộp nào. Vấn đề chính không phải thiếu corpus mà là chưa có baseline đơn giản an toàn: 02_rag sửa `glob` đệ quy + `try/except` + sinh ngắn. Ngoài ra, dùng 25 câu mẫu làm few-shot sinh câu rồi lại lấy 25 câu đó làm holdout cuối là rò rỉ trực tiếp; và theo trích dẫn trong brief, câu tổng hợp không xếp hạng tốt generator. Tôi chấm plan 6/10, kết luận: sửa nhỏ rồi làm — cắt phần nặng, thêm một baseline rẻ.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)

### Đề xuất A: Cắt 2 judge + 150 câu tổng hợp; thay bằng grader nhanh trên 25 gold + 1 LLM judge nếu cần
- Giả thuyết: Để chọn thay đổi ở P4, không cần judge đôi hiệu chỉnh đồng thuận 90%, cũng không cần holdout tổng hợp. Một grader bám theo key facts của 25 gold là đủ cho so sánh tương đối local 7B.
- Dựa trên bằng chứng nào trong brief: Brief mục 2 và 5: 25 câu mẫu có gold; S5 dùng 25 câu đó làm few-shot sinh câu, nhưng S5 cũng ghi “sample25 = holdout cuối” → rò rỉ; phần 7 trích van Elburg et al. cho thấy câu tổng hợp xếp hạng retriever nhưng sai khi so generator; câu hỏi 2 hỏi nên bỏ gì để nộp sớm hơn.
- Cách làm cụ thể: Tạo `gold_facts.jsonl`: mỗi câu gồm 3–7 sự kiện/số/quan hệ then chốt từ gold. Chấm không cần LLM: normalize lowercase, làm tròn số 2 chữ số có nghĩa, quy đổi đơn vị; đúng nếu chứa đủ key facts và không mâu thuẫn; câu tiền đề sai phải bác tiền đề. Nếu cần phân biệt cách diễn đạt, gọi DeepSeek temperature=0, cache theo hash, chỉ 25 câu, không cần J2 OpenRouter. Xoá S5 `synth_qa.py`, bỏ hiệu chỉnh 2 judge; không dựng C200/C500 ở vòng này. Nếu muốn thêm câu hỏi thật, mở rộng thủ công từ 3–5 paper AI/DONES khác và giữ tách tài liệu, không sinh LLM.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Chạy 01 vs 02 trên 25 gold bằng grader này; 02 phải > 01, khớp với LB 57.14 > 29.05. Bác bỏ grader nếu nó chấm 02 ≤ 01, hoặc chấm gold diễn đạt lại đúng dưới 20/25. Nếu sau 2 lần sửa rubric vẫn không tách được 01/02, mới nâng cấp lên 1 LLM judge; chưa cần 2 judge.
- Kỳ vọng: tiết kiệm 2–3 ngày, có bài nộp #1 sớm ~27–28/10; không hứa +accuracy | Tự tin: vừa | Chi phí: S

### Đề xuất B: Thêm baseline bắt buộc 03 = 02_rag sửa 3 dòng (glob đệ quy + try/except + sinh 512)
- Giả thuyết: Lợi ích an toàn nhất là không bỏ sót PDF lồng thư mục, không crash giữa chừng, và sinh ngắn để tiết kiệm thời gian. Các module P4 phức tạp hơn chỉ nên thêm sau khi 03 vượt 02.
- Dựa trên bằng chứng nào trong brief: Phần 5 ghi rõ 02_rag dùng `glob("*.pdf")` không đệ quy, `answer()` không try/except; phần 1 nói judge ẩn tham chiếu concise và tie-break thời gian; BTC baseline 02 dùng `max_new_tokens=2048`; local 7B chỉ 9–11 tok/s; P4 liệt kê nhiều module chưa kiểm chứng.
- Cách làm cụ thể: Copy 02_rag thành `03_fast_safe`: đổi `glob` sang `rglob`, bọc `answer()` try/except trả `"[error]"`, đặt `max_new_tokens=512` và prompt yêu cầu câu trả lời 1 câu. Dùng harness tối thiểu từ đề xuất A chạy 01/02/03 trên 25 gold, cache parse/embed theo corpus. Báo cáo balanced accuracy, thời gian p50/p95/tổng, doc-hit@k.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: 03 phải không thua 02 về doc-hit@5 và accuracy; thời gian p95/tổng phải giảm ≥ 20% so với 02 do sinh 512 thay 2048 token. Bác bỏ việc nộp 03 nếu 03 không hơn 02 ở cả retrieval lẫn accuracy, hoặc thời gian không giảm. Nếu qua, nộp 03 ngay làm bài #1.
- Kỳ vọng: +0–3.33 điểm LB nếu `rglob` lấy thêm tài liệu bị bỏ; giảm ~30–50% thời gian sinh; rủi ro thấp | Tự tin: cao | Chi phí: S

### Đề xuất C: Đóng băng P5/P6; chạy local 7B bf16, bỏ shim 4-bit; chỉ thuê H100 một lần để calibrate 32B
- Giả thuyết: Tăng tốc eval và giảm bug quan trọng hơn xây P5/P6 trước bài #1. 7B bf16 chạy được trên GPU local và nhanh hơn 4-bit; shim 4-bit là thêm một tầng rủi ro hành vi. Một lần thuê H100 ngắn đủ để biết local 7B có xếp hạng giống 32B server hay không.
- Dựa trên bằng chứng nào trong brief: Local 7B bf16 13.6 GiB, 11 tok/s, vừa GPU 16 GB; `dev_shim.py` vá `AutoModelForCausalLM.from_pretrained` thêm BitsAndBytes; câu hỏi 6 hỏi lợi/hại thuê H100 sớm; kế hoạch P1b để thuê 32B.
- Cách làm cụ thể: Bỏ `dev_shim` 4-bit, chạy 7B bf16, giữ `max_new_tokens=512`. Cache PyMuPDF/embed theo corpus để mỗi lần chạy không parse lại. Tạo 5 câu smoke (2 DONES, 3 AI, đủ các question_type); chỉ chạy full 25 câu khi smoke pass. Đóng băng P5/P6 tới sau bài nộp #1. Sau khi 03 sẵn sàng, thuê 1h H100 chạy 01/02/03 bằng Olmo-32B trên smoke + 25 gold, so với LB 29/57.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Chạy 5 câu smoke bằng 7B bf16 và 4-bit; nếu khác >1/5 verdict key facts thì chọn bf16, không hồi phục shim. Thuê 1h H100; nếu 7B bf16 không xếp hạng 03 so với 02 cùng chiều 32B trên ≥4/5 smoke, thì local 7B không đủ để tune P4 — chuyển sang dùng H100 sớm hoặc tăng tần suất thuê. Nếu 32B cũng không cho 03 > 02, quay lại sửa retrieval trước P4.
- Kỳ vọng: không trực tiếp +điểm, nhưng giảm bug shim, tăng tốc eval, giảm nguy cơ tune sai do local bias | Tự tin: vừa | Chi phí: S (local) / M (1h H100)

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Chưa có predictions của 01/02 local trên 25 gold, nên chưa biết 7B local có tách được 29.05 vs 57.14 không — đây là bước 0 bắt buộc.
- 25 gold có `reference_long` 81–2677 ký tự, trong khi judge ẩn “deliberately concise”; việc tự rút gọn ra `reference_short` có thể lệch khỏi tham chiếu thật, chưa được kiểm chứng.
- Câu hỏi 3: dùng 25 câu mẫu làm few-shot sinh câu rồi lại giữ 25 câu đó làm holdout là rò rỉ trực tiếp; kể cả không trùng câu, các câu tổng hợp từ tài liệu gold dễ trùng sự kiện.
- Giả thuyết corpus thật 150–400 PDF chỉ là ước lượng mỏng; cần khi fetch xong C50 phải ghi page count, tỷ lệ PDF scan/searchable, số trang trung bình để kiểm tra trước khi dựng C200/C500.
- 5–6 ngày chỉ thực tế nếu cắt S2 và S5; nếu giữ nguyên 2 judge + 150 câu tổng hợp + C500 thì khả năng trễ cao. Chưa có dữ liệu về bug Windows/Linux (multiprocessing spawn vs fork, dtype bf16 khi server chạy 32B).

## 4. Nguồn đã tra cứu
- Không tra cứu web vòng này. Dùng trích dẫn sẵn trong brief: van Elburg et al., *Can we Evaluate RAGs with Synthetic Data?* (arXiv 2508.11758) — ủng hộ việc không dùng câu tổng hợp để chọn generator.
- Thakur et al., *Judging the Judges* (GEM 2025, arXiv 2406.12624) — cảnh báo nên dùng Cohen's kappa, nhưng trong vòng này ta tránh 2 judge để đơn giản hoá.
