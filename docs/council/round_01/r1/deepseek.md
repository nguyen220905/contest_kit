# M7 = deepseek (deepseek-v4-pro) — vai: root_cause — vòng 1

## 1. Chẩn đoán nhanh
Chưa có điểm nào để nói “plateau”, nên tôi coi đây là dự báo cho baseline 02_rag. Tôi nghiêng về 3 nguyên nhân cạnh tranh: (1) retrieval/chunking kém, đặc biệt câu AI nêu thẳng tên paper cần định tuyến theo tiêu đề tài liệu; (2) Docling no table structure làm mất bảng số liệu DONES, hỏng câu hỏi thiết kế/tính toán; (3) generator dùng context kém do nhồi 10k token, greedy decode và prompt chung, chưa tối ưu cho LLM judge. Cần phân biệt bằng các thí nghiệm oracle rẻ trước khi tốn lượt nộp.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)

### Đề xuất A: Oracle-context test để cô lập tầng lỗi
- Giả thuyết: Nếu generator có sẵn context chuẩn vẫn trả lời sai nhiều, bottleneck là generator/định dạng câu trả lời/LLM judge; nếu trả lời đúng với context chuẩn nhưng sai với baseline/no-context thì bottleneck là retrieval/parser.
- Dựa trên bằng chứng nào trong brief: 25 câu mẫu có đáp án nhưng chưa có thí nghiệm nào đo tầng nào lỗi; 02_rag chưa được chạy lại; nhiều câu đáp án dài, hỏi số liệu/tính toán, và LLM judge rubric không công bố.
- Cách làm cụ thể:
  1. Chọn 10 câu từ 25 câu mẫu: 5 câu AI nêu tên paper rõ ràng, 5 câu DONES hỏi số liệu thiết kế/tính toán.
  2. Với mỗi câu, tạo 3 prompt: (a) không context; (b) oracle-snippet chứa câu trả lời/nội dung gần đáp án (cắt từ đáp án mẫu hoặc tài liệu nguồn công khai, 500–1.500 ký tự); (c) nếu đã tải được PDF liên quan, thêm baseline-retrieval context, nếu chưa thì bỏ qua bước này.
  3. Chạy local Olmo-7B-Instruct 8-bit (nếu 16GB quá sát thì 4-bit) với prompt giống 02_rag, greedy, `max_new_tokens` 512–1024. Chấm bằng một LLM judge local đơn giản + so khớp số/phrase cho câu numeric.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu `acc(oracle) < 0.4` hoặc không cao hơn rõ rệt so với `acc(no-context)` thì generator/format là bottleneck, chưa nên ưu tiên retrieval. Nếu `acc(oracle) >= 0.7` và `acc(baseline)` thấp hơn oracle ≥ 0.3 thì bottleneck nằm ở retrieval/parser. Nếu cả hai đều thấp, model proxy 7B quá yếu so với 32B, cần thuê H100 để lặp lại thí nghiệm rồi mới kết luận.
- Kỳ vọng: không trực tiếp tăng điểm; giúp tránh 1–2 lượt nộp sai hướng, định hướng các Đề xuất B/C. | Tự tin: vừa | Chi phí: S

### Đề xuất B: Đo recall@k và title routing trên mini corpus
- Giả thuyết: Lỗi chính ở retrieval vì chunk 800 ký tự làm mất ngữ cảnh tiêu đề paper/mục, embedder MiniLM yếu trên văn bản kỹ thuật, và pipeline chưa có title routing; đặc biệt 60% câu AI nêu thẳng tên paper.
- Dựa trên bằng chứng nào trong brief: Baseline 02_rag chỉ dùng cosine similarity MiniLM, nhồi 10k token, không BM25, không rerank, không dùng tiêu đề tài liệu; 25 câu mẫu có ~60% câu AI nêu tên paper/phương pháp.
- Cách làm cụ thể:
  - Xây mini corpus 15–20 PDF: tải các paper AI theo tên trong 25 câu mẫu, vài tài liệu DONES công khai, thêm 5 PDF nhiễu.
  - Gán nhãn chunk/trang vàng cho 8–10 câu (tốn 1–2 giờ).
  - Chạy 3 cấu hình retrieval: (a) baseline 800-char + MiniLM; (b) chunk theo section + prepend tiêu đề paper/section; (c) BM25 title + dense fusion. Đo `recall@5` và `nDCG@5`.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu `recall@5` của (a) < 0.3 và (b/c) > 0.7 → đầu tư title routing/chunking theo cấu trúc. Nếu cả ba > 0.8 nhưng câu trả lời vẫn sai theo Đề xuất A → retrieval không phải bottleneck, chuyển sang parser/generator. Ngưỡng này cần đọc cùng bootstrap CI vì chỉ có 8–10 câu.
- Kỳ vọng: nếu đúng, định tuyến theo tiêu đề có thể +5–15 điểm domain AI trên LB. | Tự tin: vừa | Chi phí: S

### Đề xuất C: Ablation parser/bảng cho câu DONES
- Giả thuyết: Docling no table structure làm mất các bảng số liệu thiết kế DONES; các câu hỏi dạng “lưu lượng thể tích”, “nhiệt độ bẫy nitơ”, “bypass ratio” thất bại vì context không chứa bảng/đơn vị, không phải vì retrieval hay generator.
- Dựa trên bằng chứng nào trong brief: 25 câu mẫu có nhiều câu DONES hỏi số liệu thiết kế cụ thể nằm trong bảng/báo cáo kỹ thuật; 02_rag dùng Docling no OCR/no table structure; Docling được biết chậm hơn PyMuPDF nhiều lần.
- Cách làm cụ thể:
  - Lấy 2–3 tài liệu DONES công khai chứa bảng thiết kế (hoặc dùng đoạn chứa đáp án mẫu nếu không tải được PDF).
  - Với 5 câu DONES dạng số, tạo 3 phiên bản context: (a) Docling no table; (b) PyMuPDF/plain text giữ dòng bảng; (c) pdfplumber extract_table hoặc markdown. Dùng 7B local trả lời, chấm bằng numeric match + LLM judge.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu độ chính xác DONES của (b/c) cao hơn (a) ≥ 20 điểm % trên 5 câu → parser/bảng là bottleneck, cần ưu tiên thay parser hoặc tái tạo bảng trước khi nhúng. Nếu không khác biệt đáng kể → vấn đề nằm ở retrieval/generator, không nên tốn thêm chi phí parser sâu.
- Kỳ vọng: nếu đúng, giữ bảng có thể +5–10 điểm domain DONES. | Tự tin: vừa | Chi phí: S

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Chưa có điểm LB, chưa biết độ lệch local–LB, chưa biết số câu public/private thực tế và phân bố DONES vs AI.
- Chưa có corpus thực và chưa rõ mức độ nhiễu; các trích đoạn oracle ở Đề xuất A có thể quá sạch so với retrieval thật.
- Chưa rõ độ tương đương giữa Olmo-7B local và Olmo-32B server khi kết luận về generator; nên coi 7B là proxy yếu, chỉ dùng để so sánh tương đối giữa các điều kiện.
- Luật model chưa rõ ràng: câu trả lời AI của BTC nói “you may use a different model in your internal pipeline to try” nhưng luật cứng ghi Olmo-32B là model duy nhất; cần hỏi BTC đúng câu để xác nhận model phụ sinh đáp án có được phép không, hay chỉ được dùng cho rewrite/embed/rerank.

## 4. Nguồn đã tra cứu
- Không tra cứu thêm ngoài brief. Các nguồn được nhắc đến trong phân tích (Olmo 3.1 model card, Anthropic Contextual Retrieval, Docling Technical Report, Lost in the Middle) đều lấy từ brief, chưa mở lại trong vòng này.
