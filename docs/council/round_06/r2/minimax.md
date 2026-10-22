# M1 = minimax (xiaomi/mimo-v2.6-flash) — vai: data_detective — vòng 2

## 1. Nhận xét từng thành viên
- **M2**: điểm mạnh = thứ tự top-5 stage rõ ràng, thứ tự nộp A/B từng nút-knob đúng nguyên tắc một biến/lemma, và số liệu 1 câu = 5 điểm % DONES / 3,33 điểm % AI trên sample25 đã sửa được lỗi tính của tôi ("1 câu = 4 điểm" chỉ đúng khi chia đều). | lỗ hổng = chưa chỉ ra ai chấm `reference_short` — vẫn để "người duyệt" mập mờ, đúng chỗ tôi nêu nhưng M2 không có biện pháp đo.
- **M3**: điểm mạnh = yêu cầu neo J1 vào điểm LB thật (chấm predictions local bằng J1 rồi so điểm LB) — đây là lỗ hổng tôi chưa nêu. | lỗ hổng = Đề xuất B (extractive quote-only) giả định judge ẩn chấm theo kiểu factual-extractive, nhưng chính M3 thừa định hướng verbosity bias chưa đo → mâu thuẫn nội tại; và đánh đổi toàn bộ S5/S2 lấy 3 probe chỉ hợp lý nếu chấp nhận mất 3/39 lượt cho probe.
- **M4**: điểm mạnh = oracle-chunk là thí nghiệm đắt nhất/đẻ ra nhiều thông tin nhất trong mọi đề xuất — tách retrieval khỏi generator bằng đúng evidence có sẵn, không cần thêm dữ liệu. | lỗ hổng = giả định sample25 đã có `evidence[{doc_id,page,quote}]` đầy đủ (brief nói schema dự kiến, chưa có dữ liệu thật); nếu evidence chỉ có 5/25 câu thì oracle-chunk không đủ mẫu.
- **M5**: điểm mạnh = trích `02_rag` không lưu provenance + sliding-window 4096 → đề xuất `in_window_hit` thay `doc-hit@k` là đóng góp data-centric sắc nhất round này, đúng chỗ tôi chỉ "doc-hit@k có thể ảo" nhưng cụ thể hơn tôi. | lỗ hổng = cổng "≥90% quote nằm trong 4096 token cuối" và "best-cuối thắng ≥5 câu" bị đặt quá gần nhau — 2 tiêu chí này dễ sprzịch; và giả định config 32B có 48/64 lớp SWA chưa ai tự kiểm (brief chỉ ghi SWA cho 7B).
- **M6**: điểm mạnh = lịch 72h + cắt S5/J2 khỏi critical path thẳng thắn, và phát hiện `generation_config.json` có thể bật sampling (0 chi phí kiểm). | lỗ hổng = đề xuất A trộn 3–4 thay đổi mỗi bài (PyMuPDF + rglob + trần 512 + prompt) rồi vẫn gọi "mỗi ngày 1 biến thể" — không thể quy delta cho nút nào; công thức quy đổi A100→H100 với 3 hệ số cụ thể (0.90/0.42/0.62) không có nguồn, tự bịa độ chính xác.
- **M7**: điểm mạnh = probe DONES-null để xác nhận lại ma trận metric 15/35 là thí nghiệm data-check rẻ nhất round; số học lattice 57.14 ∈ {(6,26),(9,19),(12,12),(15,5)} rất sắc. | lỗ hổng = đề xuất exfiltrate số trang qua "lỗi ngắn" cuối dòng — phản kháng không cần thiết và có rủi ro DQ; và chú ý "1h cứng áp cho private?" nhưng vẫn dùng T_private ≈ 2·2080 mà không chờ xác minh.
- **M8**: điểm mạnh = yêu cầu số điểm của lần nộp lại 01/02 nguyên bản (đã nộp, brief không nêu kết quả) — điều kiện tiên quyết để diễn giải mọi delta LB. | lỗ hổng = coi S4 chỉ là "chính các PDF đã tải ~C100, bỏ arXiv-dupes" — nhưng bản trùng arXiv/publisher là nhiễu có chủ đích của plan và liên quan trực tiếp tới việc split theo tài liệu có rò rỉ không; bỏ luôn thì mất kiểm tra này.
- **(về chính mình)** vòng trước tôi để ngưỡng "phân bố lệch ≤10 điểm %" trên n=25 tự mâu thuẫn — M2 và M7 chỉ ra đúng, tôi giữ nguyên nhận định đó.

## 2. Tôi đổi quan điểm thế nào
- **Giữ**: 3 đề xuất A/B/C vòng trước — cả 3 chưa bị ai phản bác trực tiếp; M5/M7 củng cố đề xuất C (gần-dup + quote PyMuPDF vs markdown) bằng lớp kiểm mới `in_window_hit`, tôi sẽ gộp vào.
- **Bỏ**: ngưỡng "phân bố lệch ≤10 điểm %" tính trên 25 câu — M2 chứng minh nó không phân giải được (1 câu DONES đã = 5 điểm %). Thay bằng: slices hiếm không dùng làm cổng quyết định, chỉ báo cáo kèm Wilson CI.
- **Sửa**: trong đề xuất B vòng trước, tôi chốt "điểm quy đổi 30/70" nhưng M7–M8 cho thấy chưa xác nhận được metric 15/35 thực sự vận hành thế (probe DONES-null của M7). Tôi hạ kỳ vọng: bảng quy đổi chỉ được dùng *sau* probe DONES-null hoặc ít nhất sau khi lấy lại điểm nộp lại 01/02 (M8). Tức đề xuất B của tôi cần phụ thuộc vào dữ kiện này — tôi nói rõ điều đó thay vì giữ nguyên.
- **Thêm quan điểm mới**: M5 (cửa sổ 4096 trong metric) + M3 (neo J1 vào LB) là hai thứ tôi chưa có; tôi sẽ không tự lặp nhưng sẽ bình chọn.

## 3. Đề xuất cuối (tối đa 3)

### Đề xuất A: Gộp "audit trùng + in-window hit" thành 1 script đo retrieval trung thực (dựa trên: M5-B, M1-C vòng trước)
- Giả thuyết: `doc-hit@k` + quote-PyMuPDF hiện tại cao ảo vì (i) bản trùng arXiv/publisher không canonicalize → split theo tài liệu rò rỉ, (ii) quote khớp PyMuPDF nhưng không có trong markdown Docling, (iii) quote có trong context nhưng nằm ngoài cửa sổ 4096 token của 75% lớp SWA → retrieval "đúng" mà generator không thấy.
- Cách làm cụ thể: trên C50/C200 + ≥20 tài liệu gold, chạy 1 script không LLM ghi bảng 4 cột/quote: `in_pymupdf`, `in_docling_md` (fuzzy ≥0.9), `in_window_4096` (tính theo đúng cách pack của 02, ctx 10k và 5k), `near_dup_group` (sha256 + MinHash Jaccard ≥0.9, gộp cùng work-ID nếu tra được); kèm đếm cặp câu DEV/HOLDOUT chia sẻ cùng `near_dup_group`.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: nếu `in_docling_md` <90% **hoặc** `in_window_4096` <80% **hoặc** có ≥1 cặp dev/holdout cùng near-dup group → plan phải thêm bước canonicalize + đổi metric sang `in_window_hit@k` trước S3b. Nếu cả 3 ≥ ngưỡng và 0 cặp rò rỉ → đề xuất bị bác, giữ `doc-hit@k`.
- Kỳ vọng: ±1–3 câu DONES khỏi false-positive retrieval | Tự tin: vừa | Chi phí: S (1 script, không LLM)

### Đề xuất B: Bảng atomic-claim cho 25 `reference_short` + đếm outside_doc/false_premise/closed_book (sửa đề xuất A vòng trước theo M5-C, M7-§7)
- Giả thuyết: `reference_short` sinh bằng LLM tóm từ gold 81–2677 ký tự là nhãn yếu nhất của toàn pipeline; viết lại theo kiểu atomic claim (giá trị + đơn vị + thực thể) và gán 3 cờ nền sẽ làm thay đổi verdict judge đủ để đổi thứ tự 01/02.
- Cách làm cụ thể: người duyệt 25 câu ghi bảng: mỗi câu 1 danh sách claim bắt buộc; gán `outside_doc` / `false_premise` (khoanh đúng câu nào) / `closed_book` (DeepSeek không corpus trả đúng hay sai). Chạy judge trên cùng bộ câu trả lời mẫu với gold / bản LLM-tóm / bản atomic → ma trận lật 3×3.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: nếu lật ≤2/25 và thứ tự 01 vs 02 không đổi → giữ `reference_short` hiện tại, đề xuất bị bác. Nếu ≥4/25 lật hoặc 01/02 đảo chiều → phải viết lại reference trước bất kỳ quyết định generator nào.
- Kỳ vọng: tránh ±2–4 điểm local sai hướng | Tự tin: vừa | Chi phí: S

### Đề xuất C: Đếm closed_book trên sample25 để vô hiệu hoá ngưỡng 15 điểm % chưa có mẫu (mới, từ lỗ hổng tôi nêu vòng trước + M8-§3)
- Giả thuyết: ngưỡng "tỉ lệ closed_book_easy lệch >15 điểm % so với sample25" không kiểm chứng được vì sample25 chưa có tỉ lệ nền; và nếu closed-book 32B tự trả đúng ≥60% câu DONES thì đầu tư retrieval cho DONES bị đánh giá sai.
- Cách làm cụ thể: với 25 câu sample, chạy DeepSeek (không corpus) trả lời, người duyệt đúng/sai; tách theo domain và question_type. Cộng với điểm của lần nộp lại 01/02 nguyên bản (đã làm, brief chưa nêu — M8) để ước lượng đóng góp closed-book của 02_rag = 57.14 − 29.05.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: nếu closed-book đúng ≥60% trên DONES → giả định "retrieval là nút thắt DONES" (của M7-C, M4-A) bị suy giảm mạnh, phải đo oracle-chunk trước khi tin. Nếu ≤30% → giữ nguyên hướng đầu tư retrieval DONES.
- Kỳ vọng: định hướng đúng chỗ nên +3.33 điểm | Tự tin: vừa | Chi phí: S (25 câu, 1 API call + người duyệt)

## 4. Bình chọn
VOTES: M5-B, M7-A, M4-A
