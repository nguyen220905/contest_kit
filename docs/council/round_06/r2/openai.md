# M2 = openai (openai/gpt-6-luna) — vai: plan_critic — vòng 2

## 1. Nhận xét từng thành viên
- M1: Điểm mạnh là chỉ ra cần chấm theo trọng số 30/70 của LB và kiểm tra tham chiếu ngắn trước khi tin J1 | Lỗ hổng là thêm một vòng kiểm nhãn có thể kéo dài plan; ngưỡng lật 2–4/25 chưa đủ để kết luận chắc về judge ẩn.
- M2: Điểm mạnh là đưa S1 lên trước và tận dụng submission sớm để thu tín hiệu thật | Lỗ hổng là đặt S2 khá đầy đủ trước S3b, dù retrieval-only không cần judge; cap 640 cũng hơi mạo hiểm khi 7B không đại diện cho 32B.
- M3: Điểm mạnh là nhấn mạnh LB là tín hiệu trực tiếp duy nhất về judge ẩn, không thể thay bằng hiệu chỉnh J1 | Lỗ hổng là dành cả ba lượt đầu cho format và lặp baseline; với nhiễu generation chưa đo, khó quy chênh lệch điểm cho đúng nguyên nhân.
- M4: Điểm mạnh là oracle-context giúp tách lỗi retrieval khỏi lỗi sinh bằng một phép thử có sức chẩn đoán | Lỗ hổng là ngưỡng 15 điểm và kết luận từ 7B/J1 dễ quá cứng; cần xem kết quả như định hướng, không phải quyết định cuối.
- M5: Điểm mạnh là yêu cầu đo evidence thực sự có trong prompt, không chỉ doc-hit, và kiểm tra artifact chạy offline | Lỗ hổng là một số kết luận về cửa sổ attention và hậu quả lên accuracy dựa trên chi tiết kiến trúc chưa có trong brief; phải đo thay vì suy ra.
- M6: Điểm mạnh là cắt chạy generator trên synthetic và đẩy việc nộp sớm | Lỗ hổng là kỳ vọng tăng điểm và hệ số quy đổi A100→H100 quá cụ thể so với dữ liệu; các lượt nộp đề xuất cũng gộp nhiều thay đổi.
- M7: Điểm mạnh là chú ý ngân sách wall-clock và giá trị của việc đo runtime sớm | Lỗ hổng lớn là khẳng định giới hạn private 1h trái với brief, và ý tưởng khai thác lỗi để lộ thông tin corpus không nên làm.
- M8: Điểm mạnh là nhận ra S5/J2 không nên chặn submission và nêu rõ cần kiểm tra môi trường Linux | Lỗ hổng là dùng thời gian A100 để suy ra kích thước corpus ẩn không đáng tin; hệ số quy đổi chung không thể thay phép đo theo từng phần.

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)
Tôi hạ đánh giá trước đó từ **8/10** xuống **7.5/10 — “sửa nhỏ rồi làm”**. Tôi giữ S1 ở đầu, nhưng đổi ý về việc làm S2 đầy đủ trước S3b: chỉ cần một kiểm tra J1 gọn trước các quyết định về generator; retrieval-only và kiểm tra harness không cần chờ judge. Tôi cũng đồng ý với M8 rằng S5 không nên nằm trên critical path, và với M4 rằng oracle-context có giá trị chẩn đoán cao. Không theo M3 ở việc dùng cả ba submission đầu để dò format: trước hết cần một biến thể kiểm soát rủi ro, rồi mỗi lượt chỉ đổi một yếu tố.

**Thứ tự top-5 đề xuất:**
1. **S1 tối giản** — kiểm tra harness, shim 4-bit, offline, đóng gói và ghi runtime từ lúc process khởi động.
2. **Nộp biến thể an toàn đầu tiên ngay khi S1 qua** — lấy score/runtime LB sớm, không chờ xong synthetic corpus.
3. **S3 + S3b** — lấy tài liệu nguồn và dùng oracle-context để xác định retrieval có phải nút thắt không.
4. **S2 gọn** — audit `reference_short` và kiểm tra J1 trước khi dùng nó để chọn generator; không cần J2 làm cổng.
5. **S4 nhỏ rồi S6 chọn lọc** — ưu tiên retrieval-only; hoãn full S5/J2/C500 và chỉ thuê A100 sau khi đã rõ câu hỏi cần đo.

## 3. Đề xuất cuối (tối đa 3)

### Đề xuất A: Nộp sớm, giữ mỗi lượt một thay đổi (dựa trên: M8-A, M2-A)
- **Giả thuyết:** Submission sớm cho tín hiệu LB có giá trị hơn việc chờ hoàn tất eval nội bộ; các lượt chỉ đổi một biến giúp diễn giải score và runtime.
- **Cách làm cụ thể:** Sau smoke test S1, submission đầu dùng 02 với các vá an toàn không đổi hành vi trả lời: quét PDF đệ quy, không phân biệt hoa thường, và bảo đảm mỗi câu luôn trả về chuỗi. Submission thứ hai chỉ hạ `max_new_tokens` xuống 1024, nhưng chỉ nếu chạy sample25 không có câu bị cắt; submission thứ ba chỉ đổi prompt sang answer-first, giữ các thiết lập còn lại như bản tốt nhất. Không đưa `dev_shim` vào zip; không dùng Ollama hay 32B Q4 làm căn cứ chốt hành vi 32B bf16.
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Trước mỗi submission, kiểm tra zip trên môi trường Linux/offline và chạy đủ sample25. Nếu biến thể lỗi, vượt 3600 giây public, hoặc cap 1024 cắt câu cần nhiều ý trên sample25, bác bỏ biến thể đó và không nộp. So từng score/runtime với baseline 57.14/2080 giây; giữ bản điểm cao nhất như quy định.
- **Kỳ vọng:** +0 điểm được bảo đảm; lợi ích là có tín hiệu LB sớm và giảm rủi ro mất lượt vì lỗi đóng gói | Tự tin: cao | Chi phí: S

### Đề xuất B: Chạy oracle-context trước khi đầu tư vào C200 (dựa trên: M4-A, M5-B)
- **Giả thuyết:** Nếu đưa đúng quote vào context làm tăng verdict đúng rõ rệt so với context retrieval, ưu tiên sửa retrieval/chunking; nếu không, mở rộng corpus nhiễu trước là lãng phí.
- **Cách làm cụ thể:** Trên các câu sample25 có quote nguồn, ghi lại riêng tài liệu/đoạn được retrieval chọn và kiểm tra quote có thực sự lọt vào prompt gửi generator hay không. So sánh cùng model/prompt với context retrieval hiện tại và oracle-context chứa quote; dùng J1 đã kiểm tra sơ bộ hoặc adjudication thủ công cho các cặp khác biệt. Diagnostic sidecar không được thay đổi code submission.
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Cần tối thiểu 20 câu có evidence usable. Nếu oracle-context cải thiện ít nhất 3 câu và lỗi sai tài liệu/sai đoạn chiếm phần đáng kể các lỗi hiện tại, tiếp tục S4/P4 cho retrieval. Nếu cải thiện không quá 1 câu hoặc lỗi chủ yếu là sinh sai dù có evidence, bác bỏ giả thuyết retrieval là nút thắt chính và chuyển trọng tâm sang generator.
- **Kỳ vọng:** +0 điểm trực tiếp; có thể tránh một vòng xây corpus không giải quyết đúng lỗi | Tự tin: vừa | Chi phí: S–M

### Đề xuất C: Thu gọn cổng judge và dùng đúng trọng số LB (dựa trên: M1-A, M1-B, M8-A)
- **Giả thuyết:** J1 có thể hữu ích để so sánh tương đối, nhưng bộ test dài và kappa J2 không chứng minh được tương quan với judge ẩn; sai tham chiếu ngắn có thể dẫn tới chọn nhầm generator.
- **Cách làm cụ thể:** Người duyệt kiểm tra `reference_short` của cả 25 câu, nhất là số/đơn vị/thực thể và tiền đề sai. Với các so sánh local, báo cáo thêm metric LB-style `0.3 × acc_DONES + 0.7 × acc_AI`, không chỉ accuracy macro. Giữ một nhóm nhỏ test J1 gồm paraphrase đúng, đổi số/thực thể, trả lời lạc đề và câu dài có mâu thuẫn. Chuyển synthetic sang retrieval-only; không bắt buộc J2/kappa hay 15–20 câu người viết để được nộp.
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Nếu người duyệt phát hiện trên 2/25 tham chiếu ngắn thiếu fact trọng yếu, hoặc J1 bác trên 2/25 câu trả lời đủ fact đã được duyệt, không dùng J1 làm cổng chọn prompt/generator; dùng nó để phân loại lỗi và để LB quyết định. Nếu vượt các kiểm tra này, vẫn chỉ dùng kết quả local như so sánh tương đối.
- **Kỳ vọng:** +0 điểm trực tiếp; giảm nguy cơ tối ưu theo proxy sai và phản ánh đúng trọng số miền của LB | Tự tin: vừa | Chi phí: S

## 4. Bình chọn
VOTES: M8-A, M4-A, M1-A
