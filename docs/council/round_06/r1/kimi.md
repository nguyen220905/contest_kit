# M3 = kimi (moonshotai/kimi-k3) — vai: alternative — vòng 1

## 1. Chẩn đoán nhanh

Kết quả đang "dừng ở 0" không phải vì pipeline yếu, mà vì plan đầu tư 5–6 ngày vào một hạ harness đo bằng **hai proxy chưa được neo vào sự thật**: judge local J1 = DeepSeek (không phải judge ẩn — ban tổ chức khẳng định danh tính judge chỉ công bố sau khi kết thúc, https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/evaluation/) và câu hỏi tổng hợp mà chính brief thừa nhận xếp hạng **generator** với τ≈0 (van Elburg). Trong khi đó, tín hiệu duy nhất chắc chắn đúng — public LB tất định, 1 câu = 1.43–3.33 điểm, nộp bài kém không bị trừ, còn ~39 lượt — lại bị bỏ trống đến ~28/10. Nút thắt thật sự chưa được đo: judge ẩn thích câu trả lời **dài hay ngắn** (verbosity bias của LLM judge là hai chiều tùy model, https://arxiv.org/abs/2310.10076), và chưa ai tách được 57.14 của 02_rag đến từ retrieval hay từ kiến thức sẵn có của 32B (prompt "based on your knowledge" gợi ý phần lớn là closed-book).

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)

### Đề xuất A: Đảo trục — dùng LB tất định làm dev harness, 3 probe đầu đo "khẩu vị" judge ẩn
- Giả thuyết: khác biệt lớn nhất chưa biết là **định dạng/độ dài câu trả lời mà judge ẩn thưởng**, không phải retrieval; và nó chỉ đo được trên LB thật, không đo được bằng J1.
- Dựa trên bằng chứng nào trong brief: judge tất định + server chỉ trả điểm → mỗi lần nộp là 1 thí nghiệm tái lặp được; nộp kém không trừ; J1 hiệu chỉnh kỹ đến đâu cũng không kiểm chứng được tương quan với judge ẩn; mốc 29.05 (zero-shot) vs 57.14 (02_rag) đã cho thấy 3 mức hành vi rất khác nhau trên cùng judge.
- Cách làm cụ thể: thu nhỏ S2/S5 xuống mức tối thiểu (smoke + runtime + judge thô), và 3 bài nộp đầu — mỗi bài chỉ khác nhau ở **format**, giữ nguyên 02_rag nguyên bản làm xương: (1) 02_rag + prompt answer-first tối đa ~80 token; (2) 02_rag + prompt trả lời dài có giải thích (max ~400 token); (3) 02_rag nguyên bản chạy lại để đo nhiễu generation (độ lệch điểm của cùng zip). Tổng 3 ngày, thu được: hệ số judge theo độ dài + sàn nhiễu LB.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: nếu |(1) − (2)| ≤ |nhiễu đo ở (3)| → format không phải nút thắt, bác bỏ giả thuyết và quay lại plan v2 đầy đủ (lúc đó đã mất chỉ 3 ngày, không mất gì thêm vì các lượt này vốn bỏ trống). Nếu chênh ≥ 3–5 điểm → toàn bộ P4/P5 phải thiết kế quanh format thắng.
- Kỳ vọng: tiết kiệm 3–4 ngày đi sai hướng + chọn đúng format ngay tuần đầu | Tự tin: cao | Chi phí: S

### Đề xuất B: Đổi bài toán từ generative QA sang extractive-first (chọn câu văn nguyên văn + 1 dòng tổng hợp)
- Giả thuyết: với tham chiếu "deliberately concise" và judge nhị phân kiểm chứa-sự-kiện, câu trả lời dạng **trích dẫn nguyên văn** thắng vì không thể ảo giác/mâu thuẫn, ngắn → decode nhanh → dư ngân sách 1h cho retrieval.
- Dựa trên bằng chứng nào trong brief: smoke test 3/3 câu chạm trần 256 token (model sinh dài vô hạn, rủi ro lạc đề/mâu thuẫn — chính nhóm lỗi mà rubric S2 của bạn định phạt "dài lạc đề → bác 25/25"); 02_rag dùng max_new_tokens 2048 với 2080 s/50 câu — decode đang chi phối runtime; gold dài trung vị ~670 ký tự nhưng tham chiếu ẩn "concise".
- Cách làm cụ thể: pipeline "retrieve top-k → 32B chọn 1–3 câu nguyên văn (chỉ sinh số thứ tự câu, ~5 token) → ghép thành câu trả lời, tùy chọn thêm 1 dòng chốt". Đây là formulation khác hẳn: selection thay generation.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: trên sample25 + corpus gold của S3b, so 3 chế độ {generative, quote-only, quote+1-line} bằng J1 + p95 token sinh; bác bỏ nếu quote-only thua generative > 2/25 câu trên J1 **và** thua trên 1 probe LB. Lưu ý phản chứng cần kiểm: judge ẩn có thể ưa câu dài (verbosity bias, https://arxiv.org/abs/2306.05685v2) — đúng lý do phải cặp với probe LB ở Đề xuất A.
- Kỳ vọng: +5–10 điểm từ khử hallucination + giảm mạnh runtime | Tự tin: vừa | Chi phí: S–M

### Đề xuất C: Rerank bằng chính LLM đã load (pointwise), thay vì đóng gói cross-encoder
- Giả thuyết: nút thắt của 02_rag là retrieval (glob không đệ quy, MiniLM CPU trên chunk 800 ký tự); dùng 32B/Olmo đang load sẵn làm reranker zero-shot cho top-20 BM25 cho hit@1 cao hơn mà **không tốn 1 byte zip** nào (BGE-M3/bge-reranker đã bị loại vì >1 GB).
- Dựa trên bằng chứng nào trong brief: đã loại 2 reranker ngoài vì giới hạn zip; S3b đã có hook `retrieve()` đo doc-hit@k không cần LLM. Bên ngoài brief: LLM được prompt đúng cách rerank zero-shot ngang hoặc hơn reranker supervised (GPT-4 +2.3 nDCG@10 so monoT5-3B trên BEIR; pointwise rẻ hơn listwise 10–100×, https://arxiv.org/html/2304.09542v3 và https://doi.org/10.48550/arxiv.2311.01555).
- Cách làm cụ thể: sau RRF, lấy top-20, với mỗi chunk hỏi "Relevant to the question? yes/no" (1 token decode, prefill ~200 token/chunk), giữ thứ tự RRF làm tie-break; chỉ đưa top-5 đã rerank vào context.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: trong S3b dùng 7B local làm proxy, đo doc-hit@1/5 trên tài liệu gold: RRF vs RRF+LLM-rerank; bác bỏ nếu hit@1 không tăng ≥ 10% tương đối, hoặc khi ngoại suy prefill 32B (đo trên phiên A100/H100 thuê) vượt ~10 s/câu. Nếu 7B-proxy không tái hiện được thứ hạng 32B thì đây là điểm duy nhất thật sự cần máy thuê sớm.
- Kỳ vọng: +2–5 điểm nếu retrieval là nút thắt | Tự tin: vừa | Chi phí: M

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Thiếu bước **neo J1 vào LB**: lưu predictions của mọi bài nộp, chấm bằng J1, so với điểm LB thật → đây là cách duy nhất định lượng độ lệch judge; plan v2 hiệu chỉnh J1 rất kỹ nhưng không có bước này.
- Điểm 57.14 của 02_rag chưa được phân rã: bao nhiêu câu đúng nhờ retrieval, bao nhiêu nhờ kiến thức sẵn có (zero-shot đã 29.05) — quyết định xem nên đầu tư retrieval hay prompt/kiến thức.
- Giả định "corpus phẳng" chỉ suy ngược từ điểm LB, chưa kiểm trực tiếp; 02_rag `glob("*.pdf")` không đệ quy nên nếu giả định sai thì 57.14 là *sàn*, không phải trần.
- Tie-break bằng thời gian chỉ có giá trị khi hòa điểm — plan chưa nói rõ ngưỡng runtime nào đáng hy sinh accuracy (theo tôi: không đáng, trừ khi chênh < 1 câu).
- Brief không nêu judge ẩn có bị verbosity bias chiều nào; rubric S2 tự đặt "dài lạc đề → bác" là **giả định về judge ẩn**, không phải sự thật đã đo.

## 4. Nguồn đã tra cứu
- Evaluation — IEEE CIS TC on Domain-Specific QA with LLMs — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/evaluation/ — xác nhận judge ẩn nhị phân, danh tính chỉ công bố sau kết thúc, tie-break theo thời gian, trọng số Phase 2 (30/30/40).
- Overview — cùng giải — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/ — xác nhận tham chiếu "deliberately concise" và 5 loại câu hỏi.
- Sun et al., Is ChatGPT Good at Search? (RankGPT) — https://arxiv.org/html/2304.09542v3 — LLM zero-shot rerank vượt supervised reranker trên BEIR; nền cho Đề xuất C.
- Instruction Distillation for zero-shot ranking — https://doi.org/10.48550/arxiv.2311.01555 — pointwise rẻ hơn listwise 10–100× mà vẫn hiệu quả; chọn pointwise cho Đề xuất C.
- Zheng et al., Judging LLM-as-a-judge with MT-Bench — https://arxiv.org/abs/2306.05685v2 — verbosity bias: judge có thể thiên vị câu dài; phản chứng phải đo cho Đề xuất B.
- Saito et al., Verbosity Bias in Preference Labeling — https://arxiv.org/abs/2310.10076 — bias độ dài là hai chiều tùy model → không thể giả định judge ẩn thích ngắn; củng cố Đề xuất A.
