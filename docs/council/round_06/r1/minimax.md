# M1 = minimax (xiaomi/mimo-v2.6-flash) — vai: data_detective — vòng 1

## 1. Chẩn đoán nhanh
Plan v2 vẫn neo toàn bộ quyết định vào sample25 (25 câu) — bộ nhỏ, lệch cấu trúc so với LB (dev 40/60 domain so với LB 30/70; comparative chỉ 3 câu, predictive 4 câu → mọi "phân bố question_type lệch ≤10 điểm %" trên 25 câu thì 1 câu đã = 4 điểm %, ngưỡng tự mâu thuẫn với mẫu). Gold dài (trung vị ~670 ký tự, tối đa 2677) + chứa kiến thức ngoài tài liệu + câu tiền đề sai, nhưng tham chiếu ẩn là bản "deliberately concise" → `reference_short` rút từ gold là nơi rủi ro nhãn sai tập trung nhất và đang được đo bằng một ngưỡng mỏng (flip ≤10% = 2,5 câu). Corpus nhiễu có bản trùng arXiv/publisher cố ý + giả định corpus phẳng chưa kiểm → doc-hit@k đo trên C200 có thể bị phóng đại hoặc làm méo việc chọn chunking. Chưa câu nào trong 25 câu được kiểm "đóng cửa có trả lời được không" → cờ `closed_book_easy` và ngưỡng lệch 15 điểm % chưa có mẫu để ước lượng.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)

### Đề xuất A: Kiểm nhãn & tham chiếu 25 câu trước khi xây S5
- Giả thuyết: `reference_short` suy từ gold (dài, outside-knowledge, tiền đề sai) sẽ cho verdict lật đáng kể so với một tham chiếu viết tay kiểu concise 1–2 câu, đặc biệt trên câu trả lời dài đúng + chi tiết thừa → judge nội bộ không proxy judge ẩn.
- Dựa trên bằng chứng nào trong brief: gold 81–2677 ký tự, "KHÁC tham chiếu ẩn concise"; "có câu tiền đề sai"; gold dùng ln2/khối lượng riêng Li ngoài tài liệu; S2 đã có test long/short nhưng ngưỡng "lật ≤10%" chưa có con số nền.
- Cách làm cụ thể: với từng câu trong 25, gán 3 nhãn: (i) outside_doc (có/không), (ii) false_premise (khoanh đúng câu nào — brief chỉ nói "có câu", không nêu số), (iii) closed_book (DeepSeek trả lời không corpus, người duyệt đúng/sai). Viết `reference_concise` tay cho 25 câu. Chạy judge trên cùng bộ câu trả lời mẫu với gold / reference_short / reference_concise, đếm ma trận lật 3 chiều.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: nếu % lật (reference_short ↔ reference_concise) ≤ 2/25 và không đổi verdict ròng của baseline 01 vs 02 → giữ nguyên S2, A bị bác. Nếu ≥ 4/25 lật hoặc baseline 01 > 02 đảo chiều → sửa ngay cách sinh `reference_short` trước S5.
- Kỳ vọng: tránh ±2–4 điểm metric local sai hướng | Tự tin: vừa | Chi phí: S (1 buổi, chỉ API judge + con người đọc 25 câu)

### Đề xuất B: Cân lại metric local theo công thức LB + kiểm slices hiếm
- Giả thuyết: xếp hạng 01 vs 02 trên split không cân lại 15/35 sẽ lệch so với LB; các slices question_type hiếm (3–4 câu) không đủ để ra quyết định kiểu "lệch ≤10 điểm %".
- Dựa trên bằng chứng nào trong brief: LB = 50·(a/15+b/35), 1 câu DONES = 3.33 điểm vs AI = 1.43; sample25 = 10/15; phân bố type causal 6 / factual 7 / procedural 5 / predictive 4 / comparative 3.
- Cách làm cụ thể: `evaluate.py` in thêm 2 số: (1) điểm quy đổi LB-style = 0.3·acc_DONES + 0.7·acc_AI trên mọi split; (2) Wilson 95% CI cho từng slice. Với split tổng hợp, ấn định mẫu tối thiểu theo type (ví dụ ≥8 câu/type) thay ngưỡng "lệch ≤10 điểm %" tính trên 25 câu.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: nếu điểm quy đổi 30/70 và điểm macro không đảo thứ tự 01/02 trên bất kỳ split nào → giữ bản macro, B chỉ là bổ sung. Nếu đảo thứ tự → mọi quyết định S3b/S6 phải dùng điểm quy đổi, tính lại.
- Kỳ vọng: quyết định local khớp LB ±1 câu DONES | Tự tin: cao | Chi phí: S

### Đề xuất C: Audit trùng lặp & trích xuất PDF trên C50/C200 trước khi tin doc-hit
- Giả thuyết: bản trùng arXiv vs publisher + PDF không có text layer làm doc-hit@k và quote-check ảo (giả định corpus phẳng chưa kiểm) → tuning retrieval trên C200 sai lệch so với corpus thật.
- Dựa trên bằng chứng nào trong brief: S4 chủ đích "có bản trùng arXiv vs publisher"; ngưỡng "PyMuPDF đọc ≥98%"; `glob("*.pdf")` không đệ hội; corpus thật không công bố kích thước.
- Cách làm cụ thể: trên C50/C200 đếm: (1) cặp near-dup bằng sha256 + MinHash trên text đã trích (ngưỡng Jaccard ≥0.9); (2) % PDF có <50 ký tự text sau PyMuPDF; (3) % evidence quote khớp ở cả 2 bản trùng; (4) chạy glob phẳng vs đệ quy trên chính C200, đếm PDF bị bỏ sót.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: nếu near-dup >10% số PDF HOẶC doc-hit@10 lệch >5 điểm % giữa bản có/không dedup → phải thêm bước dedup vào P4 và ghi lại baseline. Nếu cả hai dưới ngưỡng và glob phẳng không bỏ sót → C bị bác, giữ nguyên plan.
- Kỳ vọng: ±1–3 điểm doc-hit thực | Tự tin: vừa | Chi phí: S–M (chỉ hashing + 1 script, không chạy LLM)

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Không nêu có bao nhiêu/ câu nào trong 25 mang tiền đề sai và outside_doc — hai biến số này quyết định cả việc tách split lẫn cách viết judge rubric nhưng đang mờ.
- Không biết 25 câu DONES/AI phân bố trên bao nhiêu tài liệu nguồn khác nhau; nếu 8/10 câu DONES đến từ ≤2 PDF thì "split tách theo TÀI LIỆU" không có ý nghĩa với holdout.
- Chưa có tỉ lệ closed-book đúng nền trên sample25 → ngưỡng "lệch >15 điểm % so với sample25" chưa có mẫu để so.
- Giả định "corpus thật phẳng" suy từ "02 đã ≈ zero-shot nếu không phẳng" là suy luận một chiều: có thể corpus phẳng NHƯNG 02 vẫn trượt vì lý do khác — chưa có bằng chứng loại trừ.
- `reference_short` sinh bằng LLM + "người duyệt" nhưng không nói ai chấm gold đó đúng/không với gold gốc; sai số nhãn ở bước này chưa được đo.
- Số liệu split dev 40/60 vs LB 30/70 chưa được chuyển thành chênh lệch điểm kỳ vọng (thiếu bảng quy đổi).
- Chưa thấy dataset câu tổng hợp có trích (doc_id, page, quote) được kiểm sau khi corpus đổi từ C50 lên C200 — quote có thể vỡ khi parse khác phiên bản PDF.

## 4. Nguồn đã tra cứu
- Can we Evaluate RAGs with Synthetic Data? (van Elburg et al.) — https://arxiv.org/abs/2508.11758 — synthetic xếp hạng retriever ổn định nhưng xếp hạng generator kém; RAG trên human benchmark thường khó hơn synthetic → ủng hộ quy tắc "synthetic chỉ chọn retrieval, generator phải thắng sample25+LB".
- No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding (Krumdick et al.) — https://arxiv.org/abs/2503.05061 — reference sai làm kappa judge sụp (tham chiếu "Wrong" kappa ≈0.21); reference chất lượng cao hơn model judge mạnh hơn → giải thích vì sao việc audit `reference_short` (Đề xuất A) phải làm trước S5.
- Judging LLM-as-a-judge with MT-Bench (Zheng et al.) — https://arxiv.org/html/2306.05685v2 — verbosity bias: judge ưa câu dài kể cả khi không thêm thông tin → gold dài 81–2677 ký tự so với tham chiếu concise ẩn là rủi ro lật verdict có thật.
- Length-Controlled AlpacaEval — https://arxiv.org/html/2404.04475v1 — chỉ đổi lệnh "verbose/concise" làm win-rate dao động 22.9%→64.3% → kiểm long/short trong S2 không phải tùy chọn.
- Các truy vấn khác (corpus near-dup retrieval, McNemar n=25, PDF text-layer failure rate, write-up Codabench #17954): **không tra cứu được** — giới hạn 3 lượt tìm kiếm của tool đã đạt sau 3 truy vấn đầu.
