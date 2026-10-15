# M4 = openai (openai/gpt-6-luna) — vai: plan_critic — vòng 1

## 1. Chẩn đoán nhanh
Điểm nghẽn hiện chưa phải một lỗi mô hình đã quan sát được; rủi ro lớn nhất là chưa biết baseline có chạy đúng hợp đồng và kịp giới hạn thời gian trên corpus thật hay không. Plan hiện tại trì hoãn baseline cho đến sau khi dựng corpus và mở rộng dev, nên có thể tốn công trước khi biết mức nền. Các thay đổi retrieval có cơ sở trong brief, nhưng chưa nên đưa reranker, parser mới và CI vào cùng một vòng thử vì sẽ khó quy kết nguyên nhân. Tôi giả định corpus chấm có thể khác tập PDF nhóm tự tải; nếu không đúng, việc kiểm tra trên corpus địa phương sẽ kém giá trị hơn.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)
### Đề xuất A: Đổi thứ tự thành 5 cổng kiểm chứng
- Giả thuyết: Ưu tiên kiểm tra hợp đồng, runtime và baseline sẽ giảm rủi ro nộp bài lỗi, đồng thời khiến các thử nghiệm sau có đối chứng.
- Dựa trên bằng chứng nào trong brief: Baseline của BTC mất khoảng 36 phút cho corpus + 50 câu, trong khi giới hạn là 1 giờ; chưa tách thời gian parse, nạp model và sinh đáp án.
- Cách làm cụ thể:
  1. **Smoke test gói nộp theo đúng harness**, rồi chạy lạnh một lượt với corpus gần giống corpus thi và đo riêng parse, nạp model, retrieval, sinh đáp án.
  2. **Nộp zero-shot và RAG gốc** để lấy mốc trước khi sửa hệ thống.
  3. **Lập dev có nguồn đối chiếu**: với mỗi câu, ghi PDF/trang hoặc đoạn chứng cứ; tách holdout theo tài liệu, không chỉ tách ngẫu nhiên câu.
  4. **Thử từng thay đổi retrieval riêng biệt** trên holdout; lưu top-k và điểm trả lời, phân nhóm câu nêu tên paper và câu hỏi số liệu/bảng.
  5. **Chỉ sau khi pipeline ổn định mới thử CI**, với một ablation rõ ràng; bỏ nếu không thắng cấu hình tốt nhất trên holdout.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Đo thời gian đầu-cuối trên corpus và 50 câu; bác bỏ việc tiếp tục tối ưu parser trước nếu gói gốc đã quá sát/qua 1 giờ hoặc lỗi đến từ nạp model/harness. Với mỗi thay đổi, chỉ giữ nếu điểm trên holdout tăng mà runtime vẫn đạt giới hạn; nếu chỉ tăng local judge nhưng không cải thiện các câu đối chiếu độc lập, coi là chưa xác nhận.
- Kỳ vọng: **+0 điểm trực tiếp từ đổi thứ tự; lợi ích là tránh lượt nộp vô hiệu và giảm thử nghiệm sai hướng** | Tự tin: cao | Chi phí: S

### Đề xuất B: Ưu tiên định tuyến tiêu đề + hybrid retrieval, chưa rerank ngay
- Giả thuyết: Exact title/metadata matching và BM25 có thể bổ sung cho dense retrieval, nhất là với câu nêu tên paper và câu cần số liệu chính xác; reranker nên chờ đến khi biết còn dư thời gian.
- Dựa trên bằng chứng nào trong brief: Khoảng 60% câu AI nêu tên paper, còn câu DONES số liệu thiết kế thường nằm trong bảng/báo cáo. Một benchmark 2026 trên tài liệu có cả text và bảng báo cáo hybrid retrieval tốt cho recall, BM25 mạnh ở truy vấn số chính xác, còn query expansion có lợi hạn chế trong nhóm này; đó là kết quả trên dữ liệu tài chính, nên chỉ là bằng chứng gợi ý chứ không đảm bảo cho corpus cuộc thi (https://arxiv.org/html/2604.01733v1). Một write-up của hệ thống đứng thứ 5 ở WattBot 2025 cũng dùng BM25 + dense retrieval và reranking cho QA từ PDF kỹ thuật, nhưng không chứng minh các thành phần đó sẽ tăng điểm ở cuộc thi này (https://github.com/anubhavmaity/wattbot).
- Cách làm cụ thể: Trên cùng một tập câu/corpus, so dense gốc với (i) exact title boost, rồi (ii) dense + BM25 hợp nhất bằng RRF. Giữ cùng parser, chunking và prompt để chỉ thay một yếu tố mỗi lần. Chưa thêm HyDE, query rewriting hay reranker trong lượt này.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Báo riêng Recall@5 của chứng cứ và điểm trả lời cho nhóm nêu tên paper, nhóm hỏi số/bảng, và toàn tập. Bác bỏ thành phần nếu không tăng Recall@5 hoặc không giúp điểm trả lời trên holdout; nếu có tăng retrieval nhưng không tăng trả lời, không nên coi retrieval đó là cải thiện cuộc thi.
- Kỳ vọng: **0–4 điểm % (ước lượng rất sơ bộ; chỉ có ý nghĩa nếu sửa được 1–2 câu trên tập chấm nhỏ)** | Tự tin: thấp | Chi phí: M

### Đề xuất C: Chặn rủi ro dev nhỏ và luật model trước khi đầu tư thêm
- Giả thuyết: Dev 25 câu và local judge có thể giúp sàng lọc, nhưng nếu câu hỏi và tài liệu nguồn không độc lập với quá trình chỉnh pipeline thì dễ chọn cấu hình overfit; diễn giải luật model mơ hồ còn có thể khiến một phương án bị loại.
- Dựa trên bằng chứng nào trong brief: Dev chỉ có 25 câu và không kèm PDF; judge cuộc thi không công bố rubric. Brief cũng ghi luật nêu Olmo là model duy nhất được load, nhưng Discord được thuật lại là cho phép model phụ trong pipeline.
- Cách làm cụ thể: Trước khi thử model phụ, hỏi BTC bằng văn bản: “Có được dùng model khác ngoài `model_id` để sinh câu trả lời cuối không? Có được nạp model phụ chỉ để embedding/reranking/query rewrite không? Có yêu cầu Transformers cho mọi model không?” Trong dev, khóa trước một holdout theo tài liệu nguồn; dùng LLM judge như công cụ sàng lọc, không làm tiêu chí duy nhất.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Lưu output cho từng cấu hình và so sánh từng câu trên holdout; nếu một cải tiến chỉ thắng trên các câu dùng để phát triển hoặc thay đổi quyết định khi đổi judge/rubric, chưa đủ cơ sở để nộp. Không triển khai model phụ cho bài thi nếu chưa có xác nhận rõ từ BTC.
- Kỳ vọng: **+0 điểm trực tiếp; giảm nguy cơ chọn sai cấu hình hoặc vi phạm luật** | Tự tin: vừa | Chi phí: S

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Chưa rõ số PDF, tổng số trang, tỷ lệ scan/bảng, và liệu corpus có sẵn để benchmark trước khi nộp hay chỉ được cấp trong harness.
- Mốc 36 phút gộp parse và generation chưa đủ để quyết định dùng Docling, PyMuPDF hay reranker; cần đo trên đúng đường chạy và corpus.
- Chưa có thư xác nhận chính thức về model phụ; nội dung luật và câu trả lời Discord được thuật lại trong brief đang tạo cách hiểu khác nhau.
- Cần giữ tách biệt điểm local judge, recall chứng cứ và điểm leaderboard: chưa có dữ liệu chứng minh local judge dự đoán được judge ẩn.
- Brief ghi “hôm nay 15/10/2026” và trích Discord ngày 15/10; ngày hiện tại theo chỉ dẫn phiên này là **14/10/2026**. Nên xác minh lại mốc thời gian/ghi chú Discord trước khi dùng chúng làm căn cứ.

## 4. Nguồn đã tra cứu
- IEEE CIS Technical Challenge on Domain-Specific Question Answering with LLMs — https://www.codabench.org/competitions/17954/ — trang Codabench nêu hợp đồng `answer(question, corpus_dir, model_id)`, cách gọi theo từng câu và model_id hiện tại.
- IEEE-CIS DSQA Starter Kit — https://github.com/jucls/IEEE-CIS-DSQA-Starter-Kit — kho starter kit chính thức/được liên kết cho các submission mẫu; dùng để đối chiếu điểm vào triển khai.
- From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents — https://arxiv.org/html/2604.01733v1 — benchmark text/bảng hữu ích để hình thành giả thuyết hybrid/BM25; khác miền tài liệu nên không phải dự báo điểm cuộc thi.
- WattBot 2025 — RAG System for Technical Document Q&A — https://github.com/anubhavmaity/wattbot — write-up của một submission xếp hạng cao trong một cuộc thi QA tài liệu kỹ thuật, nêu hybrid search và reranking; chỉ dùng làm ví dụ gần bài toán.
