# M7 = deepseek (deepseek-v4-pro) — vai: root_cause — vòng 2

## 1. Nhận xét từng thành viên
- M1: điểm mạnh: đưa ra các phép đo oracle/phủ corpus, trùng lặp, nguyên vẹn bảng — đúng trọng tâm chẩn đoán nguyên nhân và rẻ. | lỗ hổng: chưa dùng mốc LB công khai (nếu M8 đúng) để đặt ngưỡng; có thể đốt nhiều công vào việc tải toàn bộ corpus trước khi cần.
- M2: điểm mạnh: thay Docling bằng PyMuPDF và title routing là “benchmark tối thiểu” cần vượt; tách runtime khỏi chất lượng. | lỗ hổng: benchmark tốc độ do PyMuPDF công bố cần tái lập trên corpus của mình; đề xuất nộp 01/02 nguyên bản sẽ đáng phí nếu LB đã lộ điểm baseline.
- M3: điểm mạnh: cảnh báo BGE-M3/reranker v2-m3 vượt trần 1GB là quan trọng; targeted corpus từ nhóm ra đề hợp lý. | lỗ hổng: kỳ vọng +12–20% sau vài ngày là phi thực tế; nhiều số liệu model/zip chưa xác minh; ít thiết kế kiểm chứng tách tầng lỗi.
- M4: điểm mạnh: nguyên tắc cổng kiểm chứng và holdout theo tài liệu, thay đổi từng yếu tố. | lỗ hổng: thiếu phép đo cụ thể để nhanh chóng cô lập retrieval vs parser vs output; bỏ qua thông tin LB mới.
- M5: điểm mạnh: bảo hiểm runtime + probe N/nhiễu LB + đo recall@k là các phép đo cần thiết. | lỗ hổng: nhiều nguồn chưa kiểm chứng do hết lượt; chưa tận dụng mốc LB công khai; probe ngủ có thể tốn lượt nộp không cần thiết nếu LB đã cho đủ tín hiệu.
- M6: điểm mạnh: document-first routing và tối ưu output bằng tiến hoá khớp hai đặc điểm: câu AI nêu tên paper và cần CI novelty. | lỗ hổng: nhồi cả tài liệu 30–40k token có rủi ro OOM/runtime với 32B; agent lặp khó kham 1h; đây là hướng giải pháp, chưa phải thí nghiệm chẩn đoán.
- M7: điểm mạnh: oracle-context test và tách parser/retrieval rẻ, đúng kiểu root cause. | lỗ hổng: dựa vào 7B proxy dễ kết luận sai về generator; chưa cập nhật LB; chưa tính tới khả năng gold mẫu có câu sai/mâu thuẫn.
- M8: điểm mạnh: đưa ra mốc LB cụ thể (Zero 29.05/733s, RAG 57.14/2080s, top 68.1/605s), cảnh báo trần zip BGE, và nghi vấn vài gold mẫu — thông tin thay đổi hẳn kế hoạch. | lỗ hổng: các con số và phân tích gold chưa được xác minh lại; giải pháp “atomic answer” chưa có A/B judge; cần tránh coi giả định là sự thật tuyệt đối.

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)
- Giữ: ý tưởng oracle-context và tách parser/retrieval, nhưng nâng cấp: phải chạy 32B trên các điều kiện then chốt, không chỉ 7B; thêm biến thể output atomic-vs-dài để test phía judge.
- Bỏ: kế hoạch nộp `01_zero_shot`/`02_rag` nguyên bản để lấy mốc nếu thông tin LB của M8 được xác minh; bỏ việc dựng toàn bộ corpus 32 domain làm điều kiện tiên quyết trước khi chẩn đoán.
- Sửa: dùng mốc 57.14→68.1 làm sàn để đặt ngưỡng cải thiện; lượt nộp đầu tiên nên là pipeline an toàn tối thiểu, không phải baseline nguyên bản.

Các giả thuyết cạnh tranh chính tôi muốn phân biệt: (H1) retrieval/title routing chưa đưa evidence vào context; (H2) parser/chunk làm mất bảng số liệu DONES; (H3) output/judge làm mất điểm dù evidence đúng; (H4) generator 32B chưa đủ năng lực hoặc dùng context kém; (H5) LB quá nhiễu/nhỏ khiến các Δ<4 điểm không có ý nghĩa.

## 3. Đề xuất cuối (tối đa 3)

### Đề xuất A: Thí nghiệm “trần năng lực” 10 câu với 7B + 32B để tách H1/H2/H3/H4 (dựa trên: M1-A, M7-A/C, M4-A)
- Giả thuyết: Nếu 32B có context chuẩn vẫn sai nhiều, nguyên nhân nằm ở generator/output/judge; nếu chỉ sai khi context là baseline chunk hoặc mất bảng, nguyên nhân nằm ở retrieval/parser.
- Cách làm cụ thể:
  1. Chọn 10 câu từ 25 mẫu: 5 câu AI nêu thẳng tên paper, 5 câu DONES hỏi số liệu/tính toán.
  2. Mỗi câu tạo các biến thể context: (a) không context; (b) oracle-snippet thủ công chứa đáp án/tài liệu gốc; (c) mô phỏng baseline chunk 800 ký tự; (d) parse giữ bảng/dòng bằng PyMuPDF/pdfplumber.
  3. Chạy trước trên Olmo-3-7B để gỡ lỗi; sau đó chạy 32B trên H100 một giờ với cùng biến thể, batch-1, `max_new_tokens` ngắn.
  4. Chấm tay mù theo checklist: đúng số/đơn vị, đúng đối tượng, xử lý tiền đề sai. Không dùng LLM judge làm tiêu chí chính.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ:
  - Bác bỏ H4 nếu `oracle(32B)` đạt ≥7/10 và hơn `no-context` ≥2 câu; ngược lại oracle <6/10 thì ưu tiên sửa generator/output.
  - Bác bỏ H2 nếu biến thể giữ bảng (d) không thắng (c) ở ≥2/5 câu DONES.
  - Bác bỏ H3 nếu atomic answer không thắng câu trả lời dài kiểu gold ở ≥2/10 câu.
- Kỳ vọng: không tăng LB trực tiếp; định hướng chọn 1–2 hướng đúng thay vì mò 4 hướng. | Tự tin: vừa | Chi phí: S-M (một giờ H100 + một ngày chuẩn bị)

### Đề xuất B: Đo title routing + recall@k không cần LLM để kiểm tra H1 (dựa trên: M2-B, M6-A, M5-A)
- Giả thuyết: Nếu matcher tiêu đề không chọn đúng paper và recall top-5 baseline thấp, retrieval là bottleneck; nếu recall đã cao nhưng điểm vẫn thấp, H1 bị loại.
- Cách làm cụ thể:
  - Dựng mini corpus 15–20 PDF: các paper có tên trong câu AI, vài tài liệu DONES công khai, thêm 5 PDF nhiễu.
  - Nhãn thủ công trang/chunk chứa đáp án cho 10–12 câu.
  - Đo `doc-hit@1` dùng rapidfuzz title match; đo `Recall@5` cho baseline MiniLM chunk 800, BM25+dense, title-boost routing.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ:
  - Nếu `doc-hit@1` < 70% trên câu AI có tên paper → title routing không đủ, cần hybrid phức hợp hơn.
  - Nếu mọi biến thể `Recall@5` ≥ 0.8 nhưng điểm kỳ vọng vẫn < 68 → H1 bị loại, chuyển trọng tâm sang H2/H3.
  - Đọc kèm bootstrap CI vì mẫu chỉ 10–12 câu.
- Kỳ vọng: cô lập H1 rẻ nhất; nếu đúng, đầu tư title routing trong 1 ngày. | Tự tin: vừa | Chi phí: S

### Đề xuất C: Một lượt nộp “pipeline an toàn tối thiểu” để thử tổ hợp H1+H2+H3, không nộp baseline (dựa trên: M8-A/B, M2-A, M4-A)
- Giả thuyết: Nếu PyMuPDF giữ bảng + BM25/title routing + atomic answer + watchdog vẫn không vượt mức RAG công khai, thì nguyên nhân chính nằm ở generator/judge, không phải retrieval/parser.
- Cách làm cụ thể:
  - Zip < 900 MB: chỉ MiniLM + code + prompt; không nhét BGE-M3/v2-m3; index build runtime trên `corpus_dir`.
  - PyMuPDF đa tiến trình, `try/except` từng PDF, giữ bảng bằng `find_tables`; hybrid BM25 + cosine + title bonus; top-5–6 chunk; `max_new_tokens` 512; prompt atomic; watchdog theo clock.
  - Trước khi nộp: kiểm tra giải nén < 900 MB, wall-clock giả lập < 20 phút parse/embed, không exception trên 30 PDF nhiễu.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ:
  - Nếu LB ≤ 57.14 → tổ hợp H1/H2/H3 tối thiểu không đủ, quay sang kiểm tra judge/output kỹ hơn, không tinh chỉnh retrieval/parser nhỏ thêm.
  - Nếu LB ≥ 68.1 hoặc runtime < 700s → có tín hiệu nhưng chưa quy kết được thành phần; dùng A/B local để ablation tiếp.
  - Nếu Failed/timeout dù local đạt → rủi ro harness/môi trường, xem lại khác biệt server.
- Kỳ vọng: +0 đến +11 điểm so với RAG hiện tại; chủ yếu là thí nghiệm phân biệt “retrieval/parser/output có phải nguyên nhân chính không”. | Tự tin: vừa | Chi phí: M (1–2 ngày + 1 lượt LB)

## 4. Bình chọn
VOTES: M8-B, M8-A, M8-C
