# M1 = minimax (xiaomi/mimo-v2.6-flash) — vai: data_detective — vòng 1

## 1. Chẩn đoán nhanh
Brief chưa có bất kỳ số liệu corpus nào (số PDF, tỉ lệ trùng lặp, tỉ lệ bảng mất định dạng), nên rủi ro lớn nhất hiện tại là "cải tiến retrieval trên một corpus chưa biết là đủ hay không" — nếu 25 câu dev không tìm được đáp án vàng trong corpus local thì mọi rerank/hybrid đều vô nghĩa. Dev 25 câu (DONES 10, AI 15) quá nhỏ và lệch so với metric domain-balanced: mỗi câu DONES sai = 10 điểm domain đó → độ lệch chuẩn local ~±15% → không phân biệt được thay đổi nhỏ. Có mâu thuẫn dữ liệu cần giải quyết: trang chính thức nói tham chiếu "deliberately concise" (https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/) nhưng đáp án mẫu dài tới 2700 ký tự — nếu tham chiếu ẩn ngắn mà mình sinh dài (hoặc ngược lại) thì judge penalty là lỗi nhãn, không phải lỗi model. Corpus gồm PDF từ ~32 site cùng chủ đề → gần như chắc chắn có bản trùng (arXiv vs publisher vs JACoW) chiếm slot top-k và ngân sách 10k token. Cuối cùng, các câu tính toán/bảng trong dev chỉ 1–2 câu mỗi loại → slice hiếm không đo được trên 25 câu, dễ bị tối ưu hóa mù.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)

### Đề xuất A: Kiểm toán "oracle coverage" — đáp án vàng có nằm trong corpus local không?
- Giả thuyết: bottleneck hiện tại là độ phủ corpus (thiếu PDF đúng tài liệu), không phải quality retrieval; nếu phủ đủ, oracle context sẽ cho accuracy cao ngay với prompt baseline.
- Dựa trên bằng chứng nào trong brief: "không có corpus" (mục 3), corpus phải tự dựng từ sites.txt (câu hỏi E), 25 câu mẫu KHÔNG kèm PDF; câu DONES hỏi số liệu thiết kế cụ thể trong bảng/báo cáo (mục 5b).
- Cách làm cụ thể: (1) tải corpus theo sites.txt cho 2 chủ đề; (2) với từng câu trong 25 câu dev, trích "gold evidence" thủ công (chuỗi số/đơn vị/tên paper/câu văn trả lời); (3) tìm trong corpus bằng regex + rapidfuzz (có sẵn) → bảng coverage theo domain × loại câu; (4) với câu tìm được, nhét đúng chunk chứa evidence vào Olmo (mô hình 7B local hoặc Olmo-32B cloud 1 lần) + prompt baseline → "oracle accuracy".
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Bác bỏ giả thuyết nếu coverage ≥90% VÀ oracle accuracy ≥80% → corpus không phải bottleneck, chuyển trọng tâm sang retrieval/prompt. Ngược lại coverage <70% ở DONES → đổi plan: ưu tiên mở rộng corpus (báo cáo thiết kế DONES công khai, JACoW proceedings đầy đủ) thay vì hybrid/rerank; kỳ vọng phục hồi tới +5–15 điểm domain DONES.
- Kỳ vọng: +0–15 điểm (phụ thuộc kết quả đo) | Tự tin: vừa | Chi phí: S–M

### Đề xuất B: Kiểm toán trùng lặp + toàn vẹn bảng trong chunk
- Giả thuyết: ≥20% top-k bị chiếm bởi near-duplicate cùng bài báo, và/hoặc gold number/unit bị chunk 800 ký tự làm đứt → mất điểm oan ở câu bảng/số liệu.
- Dựa trên bằng chứng nào trong brief: corpus từ ~32 site (arXiv + ieeexplore + springer + sciencedirect cùng chủ đề → bản trùng lịch sử là phổ biến); baseline chunk 800 ký tự theo đoạn, "no table structure"; câu DONES tra bảng (bypass ratio, lưu lượng bộ trao đổi nhiệt).
- Cách làm cụ thể: (1) gộp tiêu đề+tóm tắt từng PDF, phân cụm near-dup bằng rapidfuzz/MinHash → đếm % PDF trùng; (2) với 25 câu dev có gold evidence dạng số, tìm chuỗi số+đơn vị trong corpus → "gold-value intact rate" trước/sau chunking; (3) đếm số chunk trùng lặp trong top-10 của 25 truy vấn.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu >30% chunk top-k là trùng lặp HOẶC >30% gold number bị mất sau chunk → thêm bước dedup khi build index + parse bảng giữ nguyên dòng/cột (thử PyMuPDF layout mode thay Docling cho bảng); re-trên 25 câu dev. Bác bỏ nếu dup <10% top-k VÀ gold-value intact >90% → bỏ qua, đừng tốn thời gian.
- Kỳ vọng: +1–4 điểm (chủ yếu DONES) | Tự tin: vừa | Chi phí: S

### Đề xuất C: Kiểm toán dev + tự kiểm judge (đo nhiễu nhãn trước khi tin LB)
- Giả thuyết: (i) phân bố 25 câu lệch benchmark ẩn (loại câu × domain), (ii) judge local mis-score đáp án vàng do format/độ dài → mọi Δ local nhỏ hơn độ nhiễu đo được.
- Dạy trên bằng chứng nào trong brief: benchmark 100 câu, 5 loại câu, "reference answers deliberately concise" (https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/) vs đáp án mẫu 80–2700 ký tự (mục 1); dev lệch 10/15; mỗi câu ≈2–4% trong 1 domain; câu tính toán có thể có đáp án làm tròn khác nhau.
- Cách làm cụ thể: (1) đếm bảng chéo 25 câu: domain × 5 loại câu × độ dài đáp án × có/không tính toán; (2) "gold-answer self-test": cho judge local chấm chính 25 đáp án mẫu → phải ≥90%; (3) "concise-ablation": cắt đáp án vàng còn 1 câu chính + số then chốt, chấm lại → nếu điểm không giảm thì format câu trả lời nên là ngắn gọn; (4) ước tính noise: mỗi câu = 4–10 điểm % trong 1 domain → tính khoảng tin cậy của Δ local.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu judge chấm gold <90% → SAI judge/rubric, sửa trước mọi experiment (không được tin LB delta <5 điểm). Nếu dev thiếu ≥2/5 loại câu ở mỗi domain → sinh thêm câu hỏi tổng hợp từ corpus local theo loại thiếu để eval, tránh tối ưu hóa factual-only.
- Kỳ vọng: +0 điểm trực tiếp, nhưng chặn mọi thay đổi sai | Tự tin: cao | Chi phí: S

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Số PDF corpus, số trang, tỉ lệ bảng/figure — không biết → không ước lượng được budget 1h hay tỉ lệ trùng lặp.
- Mâu thuẫn "reference answers deliberately concise" (trang chính thức) vs đáp án mẫu tới 2700 ký tự: mẫu là gold hay chỉ là minh hoạ? Ảnh hưởng trực tiếp format trả lời.
- Phân bố 5 loại câu × 2 domain trong benchmark 100 câu (dev 10/15 có khớp domain-balanced 50/50 không?).
- Judge chấm semantic hay strict theo số/đơn vị/làm tròn? Rubric không công bố → nhiễu judge chưa ước lượng được.
- sites.txt liệt kê site nào cụ thể (brief chỉ ghi "~32 domain sources") → chưa xác định được nguồn trùng lặp khả năng cao (arXiv vs publisher).
- 25 câu dev có câu nào trùng near-duplicate với câu khác (cùng PDF sinh ra nhiều câu) không → nếu có, effective N < 25.

## 4. Nguồn đã tra cứu
- Overview – IEEE CIS GPAIS Challenge — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/ — benchmark 100 cặp QA, 5 loại câu, "Reference answers are deliberately concise" → cơ sở phát hiện mâu thuẫn độ dài đáp án mẫu.
- Join the Challenge – IEEE CIS GPAIS — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/join-the-challenge/ — 25 câu mẫu cùng methodology/domain, danh sách site corpus công khai, không truy cập corpus eval trực tiếp → cơ sở đề xuất A (phải tự dựng corpus nên coverage là rủi ro #1).
- How Much Can We Forget about Data Contamination? — https://arxiv.org/html/2410.03249v4 — benchmark có câu near-duplicate/đáp án câu này nằm trong ngữ cảnh câu kia; xử lý bằng fuzzy match rapidfuzz → hỗ trợ việc kiểm dev trùng lặp và làm sạch.
- Query near-duplication & train-test leakage (ORCAS/Robust04) — https://export.arxiv.org/pdf/2206.14759v2.pdf — 69% truy vấn có near-dup trong corpus khác → bằng chứng định lượng rằng trùng lặp lọt retrieval là vấn đề thật, không phải lý thuyết.
- Investigating Data Contamination in Modern Benchmarks — https://aclanthology.org/2024.naacl-long.482.pdf — phương pháp 13-gram/retrieval đo overlap corpus–benchmark → nền cho kiểm toán coverage/leakage local.
- Evaluation data contamination in LLMs (ConTAM) — https://arxiv.org/html/2411.03923v1 — n-gram-based metric cần ngưỡng theo từng model/task, đa số gây false negative nếu ngẩu hứng → cảnh báo khi tự đo "đáp án có trong corpus".
- Thí nghiệm LLM-judge verbosity/length bias — **không tra cứu được** (công cụ tìm kiếm đạt giới hạn lượt trong vòng này) → đề xuất C dựa trên mâu thuẫn độ dài ngay trong brief, không có nguồn web backing.
