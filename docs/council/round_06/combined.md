Hội đồng model — round_06 — 2026-10-22 13:18 — 2 vòng

> Các model KHÔNG thấy code, chỉ thấy brief. Đồng thuận là tín hiệu, không phải bằng chứng — kiểm chứng với code/log thật trước khi đưa vào PLAN.

## Thành viên
- M1 = minimax (xiaomi/mimo-v2.6-flash) — vai: data_detective
- M2 = openai (openai/gpt-6-luna) — vai: plan_critic
- M3 = kimi (moonshotai/kimi-k3) — vai: alternative
- M4 = deepseek (deepseek-v4-pro) — vai: root_cause
- M5 = grok (x-ai/grok-4.7) — vai: red_team
- M6 = gemini (google/gemini-4-argon) — vai: pragmatist
- M7 = qwen (qwen/qwen3.8-max-0902) — vai: competition_veteran
- M8 = glm (z-ai/glm-5.3) — vai: simplifier

## Bảng xếp hạng đồng thuận (phiếu vòng 2 cho đề xuất vòng 1)

| Hạng | ID | Đề xuất | Của | Điểm | Người bầu |
|---|---|---|---|---|---|
| 1 | M7-A | Dùng LB làm thiết bị đo — 3 probe "hi sinh" (P, prefill, tách điểm DONES/AI) | qwen | 14 | M1, M8, M4, M3, M5 |
| 2 | M5-B | Đo hit trong cửa sổ 4096, không phải doc-hit trên quote PyMuPDF | grok | 12 | M1, M4, M3, M6, M7 |
| 3 | M4-A | Oracle-chunk trên sample25 để tách retrieval khỏi generator | deepseek | 8 | M1, M8, M3, M2, M5 |
| 4 | M8-A | Cắt S5 khỏi critical path; S2 rút còn 3 test; LB đóng vai trò holdout | glm | 7 | M6, M2, M7 |
| 5 | M3-A | Đảo trục — dùng LB tất định làm dev harness, 3 probe đầu đo "khẩu vị" judge ẩn | kimi | 3 | M4, M7 |
| 6 | M1-A | Kiểm nhãn & tham chiếu 25 câu trước khi xây S5 | minimax | 1 | M2 |
| 7 | M2-A | Chạy submission song song, rút gọn thứ tự các stage | openai | 1 | M5 |
| 8 | M6-B | Cắt gọn 3 điểm thừa trong Plan v2 (S2, S5, S6) & Thiết lập quy tắc quyết định ghép cặp | gemini | 1 | M8 |
| 9 | M7-B | Context ≤ ~3.8k token theo sliding-window + "evidence-last" + cap 384–512 token + answer-first | qwen | 1 | M6 |

## Chi tiết top 5 đề xuất được bầu

**M7-A** (qwen)

### Đề xuất A: Dùng LB làm thiết bị đo — 3 probe "hi sinh" (P, prefill, tách điểm DONES/AI)
- **Giả thuyết**: Vì server trả về *điểm + thời gian* và bài điểm thấp không bị trừ, mỗi submission là một phép đo. (i) Probe **P**: giữ nguyên 02_rag nhưng `answer()` trả chuỗi hằng ngay sau khi build index → runtime = load + parse + index. (ii) Probe **prefill**: như trên nhưng cho sinh 24 token → (T₂ − P)/50 = chi phí prefill 10k token/câu trên H100. (iii) Probe **DONES-null**: trả lời thật cho 15 câu DONES, trả "I cannot determine this from the corpus." cho 35 câu AI → điểm = 50·a/15.
- **Dựa trên bằng chứng trong brief**: "nộp bài kém không bị trừ", "bài cuối = bài public điểm cao nhất", "server không trả output, chỉ điểm + thời gian", "chưa đo nhiễu", "kích thước corpus thật KHÔNG công bố". Thêm đòn bẩy số học: điểm nằm trên lattice 50·(a/15+b/35) nên **57.14 chỉ có thể là (a,b) ∈ {(6,26), (9,19), (12,12), (15,5)}** và probe DONES-null phải trả về đúng {20.00, 30.00, 40.00, 50.00}.
- **Cách làm cụ thể**: Ngày 1 nộp 02_rag + *chỉ* lưới an toàn (try/except mỗi câu → fallback closed-book, `glob("**/*.pdf")`, watchdog còn 15% budget thì giảm `max_new_tokens` xuống 128 rồi 32) — kỳ vọng ≈57.14, đồng thời cho **mẫu nhiễu LB miễn phí** (chênh so với 57.14 của BTC). Ngày 2: probe P. Ngày 3: probe prefill. Ngày 4: probe DONES-null. Mọi probe đều `print()` thống kê (số PDF, số trang, giây/parser, token p50/p95) ra stdout — nếu trường "lỗi ngắn" là stderr/exception tail thì probe có chủ đích raise ở *dòng cuối* sẽ exfiltrate được số trang corpus (kiểm tra rule trước khi làm).
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ**: Nếu P ≥ 560 s → 02_rag-like an toàn với giới hạn 1h, được phép tăng k/context. Nếu P < 560 s → **mọi pipeline greedy 2048 token là rủi ro fail private**, phải cắt parse (PyMuPDF thay Docling) + cắt `max_new_tokens` trước khi tối ưu độ chính xác. Nếu điểm DONES-null ∉ {20,30,40,50} → giả thuyết "balanced acc + judge nhị phân + 15/35" sai, phải suy diễn lại metric trước khi tune bất cứ thứ gì. Nếu a ≥ 12 → dừng đầu tư DONES, chuyển sang AI (1.43/câu) + chất lượng câu trả lời cho Phase 2.
- **Kỳ vọng**: +0 điểm trực tiếp, nhưng loại rủi ro −toàn bộ điểm private và quyết định đúng chỗ cần +3.33 điểm | **Tự tin: cao** | **Chi phí: S** (4 ngày submission, ~2 h code)

**M5-B** (grok)

### Đề xuất B: Đo hit trong cửa sổ 4096, không phải doc-hit trên quote PyMuPDF
- Giả thuyết: S3b/S6 sẽ chọn retrieval bằng doc-hit@k và “02 > 01”, nhưng `02_rag` bỏ provenance (`all_chunks.extend` chỉ có chuỗi), xếp chunk tốt nhất lên đầu rồi mới nối câu hỏi. 32B (và 7B) có sliding window 4096; 75% lớp không thấy token đầu nếu prompt > 4096. Quote khớp PyMuPDF ≥ 0,9 có thể không có trong markdown Docling (`do_ocr=False`, `do_table_structure=False`).
- Dựa trên bằng chứng nào trong brief: S3 tiêu chí quote PyMuPDF; S3b bốn nhóm lỗi cần doc/page; 02 ctx 10k; kiến trúc sliding-window đã ghi cho 7B nhưng metric không dùng cửa sổ; corpus có bản trùng arXiv/publisher.
- Cách làm cụ thể: Trước S3b, bản sao 02 (không sửa bài nộp) lưu `(doc, page, token_start)`. Với ≥ 5 PDF đã có quote: so sánh quote trong PyMuPDF và trong `export_to_markdown()` cùng cờ Docling của starter. Rồi pack đúng công thức 02, ctx 10k và 5k, ghi quote có nằm trong 4096 token cuối hay không. Ablation 7B greedy: best-chunk đầu vs cuối, cùng evidence.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Bác bỏ nếu ≥ 90% quote PyMuPDF có trong markdown Docling và ≥ 80% quote đó nằm trong 4096 token cuối ở ctx 10k, đồng thời best-cuối không thắng best-đầu quá 1 câu trên 25 câu (McNemar, số cặp lệch ≤ 2). Nếu quote ngoài cửa sổ ≥ 30% hoặc best-cuối thắng ≥ 5 câu và thua ≤ 1, cấm dùng doc-hit@k làm tiêu chí chọn retrieval; chỉ chọn theo `in_window_hit`.
- Kỳ vọng: +5 đến +15 balanced acc trên câu đã tìm đúng tài liệu nếu ≥ 30% quote ngoài cửa sổ; 0 nếu bác bỏ | Tự tin: vừa | Chi phí: S

**M4-A** (deepseek)

### Đề xuất A: Oracle-chunk trên sample25 để tách retrieval khỏi generator
- Giả thuyết: Retrieval (đặc biệt chunking/embedding) là nguyên nhân chính khiến 02_rag đứng ở 57; nếu đưa đúng đoạn evidence, accuracy sẽ tăng rõ.
- Dựa trên bằng chứng nào trong brief: S3b chỉ mới phân loại lỗi 4 nhóm, chưa chạy context vàng; 02_rag dùng chunk 800 ký tự, ctx 10k, không reranker; sample25 có trường `evidence[{doc_id,page,quote}]` dự kiến.
- Cách làm cụ thể: Sau khi S3 lấy được ≥ 20/25 tài liệu gold + quote, chọn 10–12 câu sample25 có evidence rõ. Với mỗi câu, chạy 7B 4-bit greedy với hai loại context: (1) chunk 02_rag trả về, (2) đúng chunk chứa `quote` (oracle). Giữ nguyên prompt; thử cả `max_new_tokens=2048` và `640`. Chấm bằng J1 đã hiệu chỉnh; phân loại lỗi theo 4 nhóm + tách `false_premise`, `calc`.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu oracle-chunk accuracy > retrieval accuracy ≥ 15 điểm phần trăm trên cùng tập con, đồng thời lỗi “đúng tài liệu sai đoạn” chiếm ≥ 50% lỗi retrieval → retrieval là nút thắt. Nếu chênh ≤ 5 điểm hoặc lỗi chủ yếu “đúng đoạn sinh sai” → bác bỏ H1, chuyển sang prompt/model. Nếu oracle-chunk vẫn thấp ở câu `false_premise`/`calc` → ghi nhận H4 (coverage/kiến thức ngoài tài liệu).
- Kỳ vọng: +10–20 điểm accuracy sample25 nếu H1 đúng; gần như 0 nếu sai. | Tự tin: vừa | Chi phí: M

**M8-A** (glm)

### Đề xuất A: Cắt S5 khỏi critical path; S2 rút còn 3 test; LB đóng vai trò holdout
- **Giả thuyết:** Không quyết định nào trước ~05/11 phụ thuộc vào câu synthetic/human; chất lượng retrieval đo trực tiếp bằng doc-hit@k trên tài liệu gold của sample25 + nhiễu thô.
- **Bằng chứng trong brief:** van Elburg (arXiv 2508.11758) — synthetic xếp hạng retriever ổn, generator thì không; quy tắc dùng split của chính plan cũng thừa nhận generator phải qua sample25 + LB; 15–20 câu human còn thiếu người phụ trách; 39 lượt nộp vs 5–6 ngày infra.
- **Cách làm cụ thể:** Giữ S1 (thêm: chạy trong WSL2/docker Linux vì server là Linux, không phải Windows) → S3 (≥20/25 tài liệu gold) → S3b (01 vs 02 + doc-hit@k + bảng 4 nhóm lỗi) → S2 chỉ còn 3 test: paraphrase→đúng ≥24/25, đổi số/thực thể→bác ≥90%, lạc đề→bác 25/25. S4 nhiễu = chính các PDF đã tải (~C100), bỏ thiết kế arXiv-dupes. S5, J2/kappa, test long/short reference: chuyển sang "làm sau khi có ≥3 bài nộp". Cache parse/embedding theo hash corpus để vòng lặp local không parse lại.
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Chạy S3b: nếu doc-hit@5 < 60% → retrieval cần S4 đầy đủ (giữ S4, vẫn cắt S5); nếu 02 ≤ 01 trên sample25 → pipeline hỏng ở thượng nguồn, mọi eval sau vô nghĩa.
- **Kỳ vọng:** nộp bài thật từ ~24–25/10 thay vì 28/10 (+4–5 slot thông tin) | **Tự tin: cao** | **Chi phí: S**

**M3-A** (kimi)

### Đề xuất A: Đảo trục — dùng LB tất định làm dev harness, 3 probe đầu đo "khẩu vị" judge ẩn
- Giả thuyết: khác biệt lớn nhất chưa biết là **định dạng/độ dài câu trả lời mà judge ẩn thưởng**, không phải retrieval; và nó chỉ đo được trên LB thật, không đo được bằng J1.
- Dựa trên bằng chứng nào trong brief: judge tất định + server chỉ trả điểm → mỗi lần nộp là 1 thí nghiệm tái lặp được; nộp kém không trừ; J1 hiệu chỉnh kỹ đến đâu cũng không kiểm chứng được tương quan với judge ẩn; mốc 29.05 (zero-shot) vs 57.14 (02_rag) đã cho thấy 3 mức hành vi rất khác nhau trên cùng judge.
- Cách làm cụ thể: thu nhỏ S2/S5 xuống mức tối thiểu (smoke + runtime + judge thô), và 3 bài nộp đầu — mỗi bài chỉ khác nhau ở **format**, giữ nguyên 02_rag nguyên bản làm xương: (1) 02_rag + prompt answer-first tối đa ~80 token; (2) 02_rag + prompt trả lời dài có giải thích (max ~400 token); (3) 02_rag nguyên bản chạy lại để đo nhiễu generation (độ lệch điểm của cùng zip). Tổng 3 ngày, thu được: hệ số judge theo độ dài + sàn nhiễu LB.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: nếu |(1) − (2)| ≤ |nhiễu đo ở (3)| → format không phải nút thắt, bác bỏ giả thuyết và quay lại plan v2 đầy đủ (lúc đó đã mất chỉ 3 ngày, không mất gì thêm vì các lượt này vốn bỏ trống). Nếu chênh ≥ 3–5 điểm → toàn bộ P4/P5 phải thiết kế quanh format thắng.
- Kỳ vọng: tiết kiệm 3–4 ngày đi sai hướng + chọn đúng format ngay tuần đầu | Tự tin: cao | Chi phí: S

## Câu trả lời vòng cuối (vòng 2)

---

# M1 = minimax (xiaomi/mimo-v2.6-flash) — vai: data_detective

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

---

# M2 = openai (openai/gpt-6-luna) — vai: plan_critic

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

---

# M3 = kimi (moonshotai/kimi-k3) — vai: alternative

## 1. Nhận xét từng thành viên
- M1: điểm mạnh — audit `reference_short` bằng ma trận lật 3 chiều trước S5 là đúng chỗ đau (Krumdick: reference sai hại hơn không có), và đề xuất quy đổi điểm LB-style 30/70 là chi tiết cả M2/M6 đều bỏ qua | lỗ hổng — cả 3 đề xuất đều là cổng *trước* S5, không đụng tới vấn đề lớn nhất: 5–6 ngày không nộp bài; "1 câu = 4 điểm" trên sample25 sai trọng số (DONES=5đ, AI=3.33đ).
- M2: điểm mạnh — thứ tự top-5 rõ ràng, hiệu chỉnh đúng McNemar (6–0 mới p<0.05), A100 chỉ để kiểm tương thích không lấy wall-clock làm chuẩn | lỗ hổng — chấm 8/10 quá tay khi plan khóa 5–6 ngày trước bài nộp đầu; 2 biến thể nộp (max_new 640, ctx 5k) là đoán mò khẩu vị judge chưa đo — đúng thứ probe format của tôi/M7 đo được rẻ hơn.
- M4: điểm mạnh — oracle-chunk là thí nghiệm rẻ nhất phân tách retrieval/generator/parser, đúng tinh thần "đo headroom trước khi đầu tư" | lỗ hổng — phụ thuộc J1 chưa hiệu chỉnh (vòng tròn với S2), và kết luận generator từ 7B 4-bit chính M4 thừa nhận không đại diện 32B; test Docling vs PyMuPDF trên 20–25 PDF lẫn giữa lỗi parser và lỗi chunking.
- M5: điểm mạnh — đóng góp bằng chứng cứng nhất hội đồng: `sliding_window=4096` của 32B + 02_rag đặt chunk tốt nhất *lên đầu* prompt 10k → 75% lớp không thấy evidence; `use_cache=false`; private tự động chấm lại bài public cao nhất (bẫy variance-lock) | lỗ hổng — chuỗi 3 bài nộp (384 token, đảo chunk) vẫn giả định khẩu vị judge thay vì đo; một số chi tiết starter code tôi không kiểm chứng được dù có trích raw URL.
- M6: điểm mạnh — ngân sách 72h cụ thể, quy tắc net-win `Δ ≥ max(2, k_noise+1)` thay McNemar cứng, bẫy spawn/fork CUDA Windows↔Linux rất thực | lỗ hổng — Sub2 đổi 4 nút cùng lúc (ctx, thứ tự chunk, max_new, prompt) tự mâu thuẫn với logic A/B một-nút của chính hội đồng; hệ số quy đổi 0.90/0.42/0.62 trình bày như đã hiệu chỉnh nhưng chưa có một điểm đo A100 nào.
- M7: điểm mạnh — ý tưởng mật độ thông tin cao nhất: dùng lattice 50·(a/15+b/35) để giải mã 57.14 ∈ {(6,26),(9,19),(12,12),(15,5)} bằng 1 probe DONES-null; chỉ ra mâu thuẫn 1h/2h private và kappa paradox | lỗ hổng — 4 ngày probe (2 slot "hi sinh" ~0 điểm) hơi xa xỉ khi M8 chỉ ra khoảng cách top-20 chỉ 1.91 điểm; giả định corpus private = corpus public chưa kiểm chứng.
- M8: điểm mạnh — framing sắc nhất (1.91 điểm = 1 câu DONES), cắt S5 khỏi critical path, bắt đúng lỗi plan không nói chạy S1 trên Linux/WSL2 | lỗ hổng — Ngày 2 đổi prompt + max_new cùng lúc (2 nút), giả định judge thích concise trước khi đo; hỏi "kết quả nộp lại 01/02 đâu" là đúng nhưng brief ghi rõ chúng nằm ở mục *đã loại* = chưa chạy, không phải mất dữ liệu.

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)
- **Sửa M3-A (probe format)**: M7-A cho thấy cách đó gốc của tôi thô hơn số học lattice. Tôi nhập hẳn: probe DONES-null giải mã (a,b) và probe P đo load+parse là thông tin *không mua được* bằng biến thể có điểm. Nhưng tôi nén 4 probe của M7 xuống còn 2 slot hi sinh + làm P-probe có điều kiện (xem Đề xuất A).
- **Sửa M3-B (extractive-first)**: phát hiện sliding-window 4096 của M5 làm đề xuất của tôi mạnh hơn *và* bị ràng buộc hơn: quote-picking chỉ hoạt động nếu evidence nằm trong 4096 token cuối → "evidence-last packing" từ M5-B/M7-B trở thành tiền đề bắt buộc, không còn là biến thể độc lập.
- **Giáng cấp M3-C (LLM reranker)**: M4-A thuyết phục tôi — đo oracle headroom trước, nếu headroom nhỏ thì reranker vô nghĩa dù miễn phí zip. Reranker từ "đề xuất" xuống "hậu phương án chờ kết quả chẩn đoán".
- **Đổi quan điểm về judge**: tôi giữ nguyên "J1 không neo được vào judge ẩn", và giờ có thêm đạn từ M5 (reference_short thiếu số → judge proxy sai lệch hệ thống) + M7 (kappa paradox làm cổng J2 ≥0.8 vô nghĩa). Kết luận chung của hội đồng về S2/S5 rút gọn là đúng, tôi không cần đề xuất riêng nữa.

## 3. Đề xuất cuối (tối đa 3)
### Đề xuất A: Chiến dịch LB-đo-lường 4 ngày: 2 slot có điểm + 2 slot hi sinh (dựa trên: M7-A, M3-A, M5-A, M8-B)
- Giả thuyết: 3 ẩn số quyết định toàn bộ P4–P6 — (i) nhiễu LB của cùng 1 zip, (ii) phân rã điểm theo domain (a,b) trên lattice, (iii) khẩu vị độ dài của judge ẩn — chỉ đo được bằng LB, và đo xong trong 4 ngày với 2 slot ~0 điểm.
- Cách làm cụ thể: **Ngày 1**: 02_rag nguyên bản + chỉ vá sống còn (glob đệ quy `suffix.lower()`, try/except từ câu, greedy tường minh, watchdog) → đo |điểm − 57.14| = nhiễu LB; nếu runtime vượt ~2700 s thì kích hoạt **Ngày 2 có điều kiện**: probe P (`answer()` trả chuỗi hằng sau khi index xong) → tách load+parse+index khỏi generation, quyết định ngân sách 1h/2h private (mâu thuẫn M7 chỉ ra). Nếu runtime an toàn, bỏ probe P, dời lên: **Ngày 2/3**: probe format — cùng retrieval, chỉ đổi output: bản answer-first ≤80 token vs bản giải thích ~400 token (giữ max_new 2048). **Ngày 4**: probe DONES-null (trả thật 15 câu DONES, từ chối 35 câu AI) → điểm phải ∈ {20,30,40,50}, giải mã (a,b); lưu ý bẫy M5: probe 0-điểm không thể bị khóa làm bài private vì LB lấy max.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: |format-ngắn − format-dài| ≤ nhiễu đo ở Ngày 1 → bác giả thuyết "độ dài là nút thắt", P5 thiết kế tự do. Điểm DONES-null ∉ lattice → giả thuyết metric sai, dừng mọi tuning. a ≥ 12 → bỏ đầu tư DONES (trái với khuyến nghị M7-C); a ≤ 9 → ưu tiên tuyệt đối retrieval DONES (3.33đ/câu).
- Kỳ vọng: +0 điểm trực tiếp; loại 3 nhánh thiết kế sai trước khi tốn GPU-giờ | Tự tin: cao | Chi phí: S (4 slot vốn bỏ trống, ~2h code)

### Đề xuất B: Trả lời bằng chọn-câu-trích-dẫn (selection-first) + evidence-last trong cửa sổ 4096 (dựa trên: M3-B, M5-B, M7-B)
- Giả thuyết: với judge nhị phân so tham chiếu concise, pipeline "retrieval → LLM chỉ sinh *chỉ số câu* (≤5 token) → ghép 1–3 câu nguyên văn + 1 dòng chốt" khử triệt để hallucination/mâu thuẫn (nhóm lỗi rubric S2 phạt nặng nhất), decode ~0 → trả ngân sách cho retrieval; và theo M5, bắt buộc pack top-chunk *sát trước câu hỏi* để lọt sliding-window 4096 của 32B.
- Cách làm cụ thể: S3b chạy 3 chế độ trên sample25 + corpus gold (7B local): {generative 02-style, quote-only, quote+1-line} × {chunk-best-đầu, chunk-best-cuối}; metric retrieval-only `in_window_hit` (quote nằm trong 4096 token cuối, thay doc-hit@k — chính xác metric của M5-B) + J1 + p95 token sinh. Chế độ thắng local đem lên LB xác nhận bằng 1 slot (sau chiến dịch A).
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: bác quote-only nếu thua generative > 2/25 trên J1 *và* thua trên 1 probe LB; bác evidence-last nếu best-cuối không thắng best-đầu quá 1 câu/25. Phản chứng đã biết phải kiểm chéo với kết quả Đề xuất A: nếu probe format cho thấy judge ẩn *ưa câu dài* (verbosity bias ngược chiều), quote-only bị loại dù thắng local.
- Kỳ vọng: +3–8 điểm nếu judge ưa concise + giảm mạnh runtime decode | Tự tin: vừa | Chi phí: S–M

### Đề xuất C: Chẩn đoán oracle-headroom trước khi chi 1 giờ nào cho retrieval (dựa trên: M4-A, M3-C, M7-C)
- Giả thuyết: chưa ai biết 57.14 thiếu bao nhiêu điểm *do retrieval*; nếu oracle-chunk (đưa đúng đoạn chứa quote) chỉ lật ≤ 2/25 câu thì mọi đầu tư RRF/BM25/reranker/chunking (kể cả reranker bằng chính 32B của tôi vòng trước) đều vô nghĩa — nút thắt nằm ở generator/format/coverage.
- Cách làm cụ thể: ngay khi S3 có ≥ 20/25 tài liệu gold + quote (chưa cần C200): (1) retrieval-only trên 02-style: đo quote-recall@5 và `in_window_hit` (0 giờ GPU); (2) 7B chạy 25 câu × {chunk thật, chunk oracle} (~1–2h); (3) ma trận lỗi 4 nhóm của S3b. Chỉ khi oracle lật ≥ 4/25 và nhóm "đúng tài liệu sai đoạn" ≥ 50% mới mở hộp retrieval: khi đó ưu tiên thứ tự BM25+RRF tự viết (M7-C) → LLM-rerank pointwise 1-token (M3-C), cả hai 0 byte zip.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: oracle − thật ≤ 2/25 → bác "retrieval là nút thắt", đóng toàn bộ nhánh retrieval cho tới khi Đề xuất A/B xong; oracle cao nhưng generation không lật khi thay context → nút thắt là generator, quay lại Đề xuất B.
- Kỳ vọng: +0 trực tiếp; chặn 1–2 tuần đầu tư sai nhánh (M6/M7-C đều định làm retrieval ngay) | Tự tin: cao | Chi phí: S

## 4. Bình chọn
VOTES: M7-A, M5-B, M4-A

---

# M4 = deepseek (deepseek-v4-pro) — vai: root_cause

## 1. Nhận xét từng thành viên
- M1: điểm mạnh — cộng trọng số LB 15/35 và audit `reference_short` trước khi sinh câu tổng hợp là rất cần. | lỗ hổng — vẫn xoay quanh judge local; giả định có thể đo lật `reference_short ↔ concise` bằng J1 trước khi J1 được neo vào LB, nên rủi ro vẫn là proxy-bọc-proxy.
- M2: điểm mạnh — nộp song song + không biến hiệu chỉnh judge thành dự án riêng là đúng. | lỗ hổng — ba nộp đầu chỉ đổi `max_new_tokens`/ctx nhưng mỗi ngày không có đối chứng cùng điều kiện; khó tách nguyên nhân nếu điểm lệch do nhiễu generation.
- M3: điểm mạnh — dùng LB làm dev harness và đo “khẩu vị” độ dài judge ẩn là mũi thí nghiệm sắc nhất. | lỗ hổng — đốt 3 lượt LB ngay khi chưa đo nhiễu nền; extractive-first có thể hại Phase 2 (chất lượng chuyên gia) và câu cần suy luận.
- M5: điểm mạnh — chỉ ra sliding-window 4096 và “evidence phải nằm trong 4096 token cuối” rất cụ thể, bác được doc-hit@k ảo. | lỗ hổng — cấm tuyệt đối 7B 4-bit để chọn generator hơi cứng; ba nộp đề xuất (max 384, nộp lại zip, đảo chunk) chưa đo được format judge.
- M6: điểm mạnh — cắt S5 khỏi critical path, ra ngưỡng paired net-win rõ ràng, ba nộp cụ thể. | lỗ hổng — kỳ vọng +5–9 điểm LB từ ba chỉnh đơn là quá cao; hệ số 0.42 prefill/0.62 decode cho A100→H100 chưa được đo thực nghiệm, chủ yếu lý thuyết.
- M7: điểm mạnh — probe LB P/prefill/DONES-null dùng lattice điểm để tách thời gian và điểm theo domain rất thông minh; cảnh báo 1h vs 2h và private variance quan trọng. | lỗ hổng — bốn probe tốn 4/39 lượt; “raise lỗi cuối để exfiltrate số trang corpus” có thể phạm luật; citation “arXiv 2512.13961” cho kiến trúc Olmo cần đánh dấu chưa kiểm chứng.
- M8: điểm mạnh — cắt S5, giữ S1/S3/S3b, ba bài nộp one-knob, kịch bản A100 cố định. | lỗ hổng — ngưỡng doc-hit@5 < 60% chưa có cơ sở; C100 quá nhỏ để chọn retrieval; chưa tách được length vs parser trong ba nộp.

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)
Tôi giữ ba giả thuyết gốc của mình (retrieval, Docling mất evidence, generator không concise) nhưng **sửa trọng số thời gian**: trước đây tôi đợi J1 hiệu chỉnh rồi chạy local; nay tôi theo M3/M7/M8: **LB là thí nghiệm rẻ nhất và quyết định nhất**, phải có 1–3 nộp sớm để đo length/nhiễu, còn local chỉ để chọn nhanh trước khi nộp.

Tôi **bỏ** việc bắt đầu bằng `A/B max_new_tokens` thuần local (quá phụ thuộc J1); thay bằng cặp nộp LB ngắn-vs-dài. Tôi **thêm** cửa sổ 4096 vào thí nghiệm oracle-chunk của mình, theo M5, vì nếu quote nằm ngoài cửa sổ thì mọi doc-hit@k không có ý nghĩa. Tôi **giảm** kỳ vọng vào J1: dùng J1 ở chế độ “so sánh cùng reference”, không dùng để chốt generator; tham khảo M6/M8 bỏ bớt S5/J2.

## 3. Đề xuất cuối (tối đa 3)
### Đề xuất A: Thí nghiệm local tách ba nguyên nhân bằng “vị trí evidence + oracle chunk” (dựa trên: M4-A, M5-B)
- Giả thuyết: 57.14 do một trong ba: (H1) retrieval đưa sai chunk/tài liệu; (H2) evidence đưa đúng tài liệu nhưng nằm ngoài 4096 token cuối nên 75% lớp sliding-window không thấy; (H3) generator sinh dài/lệch “concise” nên judge bác.
- Cách làm cụ thể: Trên 15–25 câu sample25 đã có `quote`, chạy 7B 4-bit greedy với đúng prompt 02_rag ở ba biến: (1) chunk-best đầu context như 02_rag; (2) đảo chunk-best xuống cuối, sát câu hỏi; (3) oracle: nhét đúng đoạn chứa `quote` vào 4096 token cuối. Đo `in_window_hit` (quote có nằm trong 4096 token cuối không), `n_new_tokens`, và verdict J1 (cùng `reference_short`). Không dùng 7B để chốt điểm tuyệt đối, chỉ để định hướng trước khi nộp 32B.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu (3) thắng (1) ≥ 4–5/25 câu cùng chiều và lỗi “đúng tài liệu sai đoạn” chiếm ưu thế → H1. Nếu trong (1), ≥ 30% quote ngoài cửa sổ và (2) thắng (1) ≥ 5 câu mà thua ≤ 1 → H2 (lúc đó bỏ doc-hit, chọn theo `in_window_hit`). Nếu cả ba chênh ≤ 2 câu và lỗi chủ yếu “đúng đoạn sinh sai” → H3. Bác bỏ H1/H2 nếu không đạt ngưỡng trên.
- Kỳ vọng: Không trực tiếp + điểm, nhưng chọn đúng nhánh sửa cho LB trong 1 buổi. | Tự tin: vừa | Chi phí: M

### Đề xuất B: Ba lần nộp LB “control / concise / verbose” để đo khẩu vị độ dài judge ẩn (dựa trên: M3-A, M6-A, M8-B)
- Giả thuyết: Judge ẩn có verbosity bias một trong hai chiều; đây là ẩn số lớn nhất chưa đo, và chỉ LB đo được.
- Cách làm cụ thể: Chuẩn bị zip 02_rag với vá an toàn tối thiểu (glob đệ quy, try/except, UTF-8, watchdog) gọi là `base`. Ngày 1 nộp `base` giữ `max_new_tokens=2048`, prompt gốc — đó là control để đo nhiễu so với 57.14 đã biết. Ngày 2 nộp cùng `base`, chỉ sửa `max_new_tokens=512` + prompt answer-first yêu cầu 1–3 câu, số/đơn vị trước. Ngày 3 nộp cùng `base`, chỉ sửa `max_new_tokens=1600` + prompt yêu cầu giải thích có căn cứ. Mọi thứ khác giữ nguyên. Lưu predictions local nếu có (server không trả output), ghi thời gian.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu |điểm concise − verbose| ≤ nhiễu nền (ước từ `base` vs 57.14 hoặc từ chênh runtime/điểm cùng zip) → độ dài không phải nút thắt, bác bỏ H3, dồn chi phí retrieval/parser. Nếu concise thắng ≥ 3.33 điểm → judge ưa ngắn, mọi P4/P5 theo hướng cắt token. Nếu verbose thắng ≥ 3.33 điểm và `base` thấp hơn concise → judge ưa dài, phải ngược lại. Nếu concise bị tụt > 4 điểm so với `base` đồng thời verbose cao hơn `base` → xác nhận ưa dài.
- Kỳ vọng: +3.3 đến +6.7 nếu chọn đúng format; ít nhất loại/giữ H3 chỉ trong 3 ngày. | Tự tin: vừa | Chi phí: S

### Đề xuất C: Đo quote-recall Docling vs PyMuPDF trước khi đổi parser trên LB (dựa trên: M4-B, M1-C)
- Giả thuyết: Docling có thể làm mất bảng/công thức/text khiến retrieval sai dù embedding tăng; nhưng chưa được đo nên đừng đốt một lượt LB để thử parser.
- Cách làm cụ thể: Với 20–25 tài liệu gold lấy được, trích text bằng PyMuPDF và Docling. Với mỗi câu sample25 có `quote`, kiểm fuzzy containment ≥ 0.9 trong từng parser; đếm quote-recall, số trang không text, số bảng/công thức bị mất ở 5 trang soi tay. Nếu cần, lặp trên 5 PDF nhiễu.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Nếu Docling tìm < 70% quote trong khi PyMuPDF > 90% trên cùng bộ → parser là nút thắt, đưa PyMuPDF vào nộp LB sau Đề xuất B. Nếu cả hai ≥ 90% → bác bỏ giả thuyết parser, tập trung embedding/chunk/cửa sổ. Nếu PyMuPDF cũng < 70% → tài liệu sai hoặc bản scan, quay lại S3.
- Kỳ vọng: Không trực tiếp + điểm; quyết định có nên dùng một lượt LB cho parser hay không. | Tự tin: vừa | Chi phí: S–M

## 4. Bình chọn
VOTES: M7-A, M3-A, M5-B

---

# M5 = grok (x-ai/grok-4.7) — vai: red_team

## 1. Nhận xét từng thành viên
- M1: điểm mạnh — audit `reference_short` và điểm quy đổi 0,3·DONES+0,7·AI bắt đúng hai chỗ local có thể đảo 01/02 so với LB. | lỗ hổng — không đụng ngân sách giờ và zip nào được phép thành best public; Wilson trên slice 3–4 câu rộng đến mức không bác được giả thuyết nào.
- M2: điểm mạnh — control trước khi đụng C200/synthetic, và không nhân wall-clock A100 thành runtime H100; McNemar 6–0 là ngưỡng đúng cho n nhỏ. | lỗ hổng — ngày 2 đổi `max_new_tokens` khi chưa có sàn nhiễu cùng zip và chưa biết 2080 s nằm ở parse hay decode.
- M3: điểm mạnh — “khẩu vị” độ dài của judge ẩn chỉ đo được trên LB, J1 không neo được. | lỗ hổng — ba probe format trước sàn nhiễu; zip may mắn thành best public thì private khóa đúng zip đó. Trích RankGPT/GPT-4 không suy ra Olmo pointwise. Đề xuất B là viết lại, không phải check rẻ.
- M4: điểm mạnh — oracle-chunk tách retrieval khỏi generator, falsifiable, không cần C200. | lỗ hổng — vẫn chấm 7B 4-bit bằng J1; “oracle” lấy từ PyMuPDF có thể không phải text Docling mà model nhìn, và chưa kiểm cửa sổ 4096.
- M6: điểm mạnh — bẫy `generation_config` không ép greedy, `glob` phân biệt hoa/thường, fork sau CUDA, shim lọt zip. | lỗ hổng — mỗi bài gộp nhiều nút; +5 đến +9 điểm không có cơ chế quy được. 24/32 lớp SWA là số của 7B, không phải 32B. Công thức 0,90/0,42/0,62 là độ chính xác giả.
- M7: điểm mạnh — probe P và hệ thức \(T \approx P+2(T_{50}-P)\): nếu private cũng 1 h và corpus cùng cỡ thì 02 chỉ an toàn khi \(P \ge 560\) s. Lattice 57,14 hữu ích. | lỗ hổng — cố tình raise để kéo số trang corpus qua “lỗi ngắn” là lấy dữ liệu ẩn qua kênh lỗi, rủi ro DQ; tôi không nhận. Lattice thiếu nghiệm (3, 33). “1 h” (M7) mâu thuẫn “private 2 h” trong brief và entry 30/11 của M8 — chưa phân xử.
- M8: điểm mạnh — cắt S5 khỏi đường tới nộp, harness phải chạy Linux, một nút mỗi bài. | lỗ hổng — ngày 1 vẫn gộp glob đệ quy + try/except + watchdog; +3,3 đến +7 từ độ dài chưa có sàn nhiễu. Đòi số nộp lại 01/02 trong khi brief đã loại việc đó.

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)
Giữ: doc-hit trên quote PyMuPDF không phải thứ 02 nhìn thấy; `reference_short` rút từ gold dài không được làm cổng generator; 7B 4-bit/Ollama Q4 không được chấp nhận hay bác prompt; zip khác nhau mà thắng nhờ nhiễu sẽ bị private chấm lại.

Bỏ: ngày 1 cắt 2048→384 (làm hỏng control — M2-A đúng hơn). Bỏ khẳng định 48/64 lớp: đó là ngoại suy từ 24+8 của 7B. Bỏ ý “nộp lại cùng zip khóa một pipeline khác”: cùng zip chỉ khóa cùng code; hại là điểm public bị thổi và private là một lần quay mới (M7). Đảo chunk không được nộp trước khi đo cửa sổ.

Sửa điểm: **6,5/10 — sửa nhỏ rồi làm**, chưa đủ tốt. v1→v2 đúng hướng (cách ly sample25, synthetic không xếp generator, C500 không chặn). Thừa/sai lúc này: kappa ≥ 0,8 làm cổng, S3b bắt doc-hit trên code không có doc/page, chạy LLM trên câu tổng hợp. Ba mục dưới là chặn, không phải tuỳ chọn.

## 3. Đề xuất cuối (tối đa 3)
### Đề xuất A: Control happy-path rồi probe P, cấm khóa zip (dựa trên: M2-A, M7-A)
- Giả thuyết: 2080 s chưa tách được parse khỏi decode; nếu private bị 1 h (nguồn M7, mâu thuẫn brief — coi là rủi ro cho đến khi đọc lại trang evaluation) và corpus cùng cỡ, pipeline 2048 token fail cả bài. Zip khác 02 mà vô tình thành best public sẽ bị chấm private.
- Cách làm cụ thể: Preflight trên Linux (WSL/docker), không phải Windows: unzip, `main.py` ở gốc, không có `eval/` hay `dev_shim.py`, không load LLM thứ hai, không 4-bit. Ép `do_sample=False`, `temperature=None` trong `generate` (không tin `generation_config`). Diff `chat_template` 7B vs 32B, chỉ file. Ngày 1: 02 nguyên + try/except từng câu chỉ khi exception (fallback cố định); không đổi glob, prompt, thứ tự chunk, `max_new_tokens`. Ngày 2: đúng zip ngày 1. Ngày 3, chỉ sau khi ngày 1 Finished và điểm ≥ 29: sau khi build index, `answer()` trả chuỗi hằng. Không raise, không in số trang/tên file — chỉ lấy thời gian server trả về.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Bác “happy-path khớp BTC” nếu ngày 1 không Finished, hoặc |điểm − 57,14| ≥ 1,43, hoặc |thời gian − 2080| / 2080 > 15%. Sàn nhiễu = |ngày 1 − ngày 2|. Bác “1 câu LB là tín hiệu” nếu sàn ≥ 1,43. Giả định corpus private = public: \(T_{100} \approx P + 2(T_{ngày1}-P)\). Bác “02 an toàn giờ” nếu \(T_{100} > 0{,}8\times\) ngân sách đang tranh chấp (2880 s nếu 1 h, 5760 s nếu 2 h). Zip mới chỉ được phép vượt best nếu điểm ≥ best + sàn nhiễu; probe P cố ý điểm thấp nên không được nộp khi chưa có Finished ≥ 29.
- Kỳ vọng: +0 điểm; tránh mất cả private hoặc khóa zip sai (1 câu DONES = 3,33 điểm LB) | Tự tin: cao | Chi phí: S

### Đề xuất B: in-window hit + oracle, không dùng doc-hit làm cổng (dựa trên: M4-A, M4-B, M1-C)
- Giả thuyết: S3b/S6 sẽ chọn retrieval bằng doc-hit trên quote PyMuPDF, trong khi 02 nhồi chunk điểm cao lên đầu, ctx 10k, và Olmo có `sliding_window` 4096 (đã có trong config 7B; pattern lớp của 32B chưa đọc). Quote có thể vắng trong markdown Docling. Near-dup arXiv/publisher làm hit ảo.
- Cách làm cụ thể: Ngay khi có ≥ 5 PDF gold, không đợi 20/25. (1) Đọc `config.json` 32B: `sliding_window` và danh sách loại lớp — không cần GPU. (2) Cùng cờ Docling của starter, fuzzy ≥ 0,9 quote trong markdown vs PyMuPDF. (3) Pack đúng 02 (best trước, ctx 10k, câu hỏi cuối); ghi quote có nằm trong 4096 token cuối hay không. (4) Trên 10 câu có quote: context retrieved vs context chỉ là đoạn quote; so khớp số/đơn vị trong `reference`, chưa dùng điểm J1. Đếm near-dup Jaccard ≥ 0,9 trên text đã trích (M1-C) trước khi tin hit@k.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Bác giả thuyết cửa sổ/parser nếu config 32B không có SWA, hoặc ≥ 90% quote có trong markdown Docling và ≥ 80% số đó nằm trong 4096 token cuối. Bác “retrieval là nút thắt” nếu oracle không tăng khớp số thêm ≥ 3/10 câu. Nếu ≥ 30% quote ngoài cửa sổ, hoặc near-dup > 10% PDF và hit@10 lệch > 5 điểm % sau dedup: cấm cổng “02 > 01” / doc-hit@k; chỉ chọn theo `in_window_hit`. Ablation 7B không được phép đổi prompt nộp.
- Kỳ vọng: +0 nếu bác; +5 đến +15 balanced acc local trên câu đã đúng tài liệu nếu ≥ 30% quote ngoài cửa sổ và oracle xác nhận | Tự tin: vừa | Chi phí: S

### Đề xuất C: Khóa cổng generator cho đến khi nhãn concise giữ đủ số (dựa trên: M1-A, M2-B)
- Giả thuyết: Bộ test J1 chỉ đo tự nhất quán với `reference_short` do LLM rút từ gold 81–2677 ký tự, không phải tham chiếu ẩn concise. Krumdick (nguồn M1/M2): reference sai hại hơn không reference. Smoke 7B 4-bit chạm trần 256/256 nên verdict local đang chấm bài cụt. Kappa ≥ 0,8 có thể fail vì paradox dù agreement cao (M7) — không được là cổng.
- Cách làm cụ thể: Trước S5 và trước mọi đổi prompt/trần/PoT: người viết `reference_concise` 1–2 câu cho 25 mẫu, giữ mọi số, đơn vị, thực thể; gắn `outside_doc`, `false_premise`, `closed_book` (DeepSeek không corpus, người duyệt). Ma trận lật trên cùng bộ câu trả lời: gold / `reference_short` / `reference_concise`. Bỏ kappa làm cổng; chỉ ghi bảng bất đồng. Generator chỉ được đổi sau đề xuất A (đã có sàn nhiễu) và chỉ khi 7B **bf16**, prompt 02, context vừa 16 GB, có p95 `n_new_tokens` < 0,5×trần. 4-bit và Ollama không được chấp nhận hay bác.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Bác “J1 dùng được để chọn generator” nếu lật short↔concise ≥ 4/25, hoặc thứ tự 01 vs 02 đảo, hoặc J1 bác ≥ 5/25 câu người viết đủ fact và không mâu thuẫn. Bác “7B không bị trần chi phối” nếu p95 bf16 ≥ 0,8×trần. Khi bác: không nộp biến prompt; chỉ nộp biến đã qua A/B. Nếu lật ≤ 2/25 và thứ tự 01/02 không đổi: giữ S2, đề xuất này dừng.
- Kỳ vọng: +0 trực tiếp; tránh lệch hướng ±2–4 điểm local trên n=25 | Tự tin: vừa | Chi phí: S

## 4. Bình chọn
VOTES: M7-A, M4-A, M2-A

---

# M6 = gemini (google/gemini-4-argon) — vai: pragmatist

## 1. Nhận xét từng thành viên
- **M1**: điểm mạnh: Chỉ ra đúng độ lệch trọng số giữa `sample25` (40/60) và Public LB (15/35, nơi 1 câu DONES = 3.33 điểm gấp 2.33× câu AI = 1.43 điểm) cùng quy trình duyệt tay 25 `reference_short` kèm 3 cờ (`outside_doc`, `false_premise`, `closed_book`) ở M1-A, M1-B. | lỗ hổng: Vẫn chấp nhận chờ xây xong toàn bộ C200 và S5 mới nộp bài; M1-C tốn thời gian chạy MinHash dedup trên C200 trước khi có baseline chạy được.
- **M2**: điểm mạnh: Thiết kế chuỗi nộp bài song song tách biệt từng biến (1-knob-at-a-time) ngay sau S1 (M2-A), bỏ cổng bắt buộc `J2 kappa ≥ 0.8` và 15–20 câu người viết (M2-B), tính đúng kiểm định McNemar chính xác ($N=25$ cần 6–0 câu lật mới đạt $p<0.05$). | lỗ hổng: Ngày 1 của M2-A khuyên nộp lại `02_rag` nguyên bản làm đối chứng — việc đã bị mục 4 trong brief loại bỏ và không tận dụng được lượt nộp để sửa lỗi `glob`/trần token.
- **M3**: điểm mạnh: Nêu đúng rủi ro thiên kiến độ dài hai chiều (verbosity bias) của Judge ẩn (dẫn Saito et al. 2023, Zheng et al. 2023) và ý tưởng dùng LB để đo khẩu vị độ dài câu trả lời (M3-A). | lỗ hổng: M3-B (chỉ trích câu nguyên văn) sẽ gãy ở các câu cần tính toán (`ln2`, khối lượng riêng Li) hoặc bác bỏ tiền đề sai; M3-C (dùng 32B rerank pointwise 20 chunk/câu) phát sinh thêm 1000 lượt prefill cho 50 câu (2000 lượt ở private), rất dễ cháy ngân sách 1 giờ cứng.
- **M4**: điểm mạnh: Thí nghiệm `oracle-chunk` (M4-A) và đối chiếu tỷ lệ giữ `quote` vàng giữa Docling vs PyMuPDF trên 20–25 PDF mẫu (M4-B) rất sắc bén để tách lỗi parser khỏi lỗi retrieval/generator. | lỗ hổng: Hoàn toàn đóng khung trong thử nghiệm local, bỏ trống 39 lượt nộp LB, không giải quyết câu hỏi về chuỗi bài nộp sớm, Ollama hay thuê A100.
- **M5**: điểm mạnh: Đào sâu nhất vào code thực tế của starter kit và `config.json` của `Olmo-3.1-32B-Instruct`: phát hiện `02_rag` vứt bỏ tên file/trang (`doc-hit@k` vô dụng nếu không sửa hook), xếp chunk điểm cao nhất lên **đầu** context 10k khiến nó rơi ra ngoài `sliding_window=4096` của 48/64 lớp (M5-B), `use_cache=false` trong config, giới hạn **1 GB là sau giải nén**, và rủi ro bài thắng public do nhiễu ngẫu nhiên sẽ khoá chết bài private (M5-A). | lỗ hổng: M5-A mất 2 ngày đầu chỉ để nộp cùng 1 bài đo nhiễu; M5-C yêu cầu chạy 7B ở `bf16` với prompt `02_rag` (~12k token) — điều mà mục 2 của brief đã chứng minh là tràn 16 GB VRAM local.
- **M7**: điểm mạnh: Phát hiện 3 thông tin sống còn: (1) Hạn chót đăng ký form & ghép đội **01/11/2026**; (2) Luật 1h wall-clock cứng (vượt = terminated/failed), nên nếu `P` (load+parse+index) $< 560\text{ s}$ thì `02_rag` (2080 s/50 câu) sẽ chết ở 100 câu; (3) Khoảng cách vào Top 20 (59.05) chỉ là **+1 câu DONES (+3.33 điểm)**, và bài báo van Elburg (2508.11758v2) chứng minh câu hỏi synthetic chỉ cần 1 chunk nên không được dùng synthetic để chọn `top-k`/chunking (M7-C). | lỗ hổng: M7-A khuyên đốt 3 ngày nộp bài "hi sinh" (trả chuỗi rỗng hoặc bỏ trống 35 câu AI để giải phương trình nghiệm nguyên) và cố tình `raise` exception ở dòng cuối để lấy log qua stderr — vừa lãng phí 3 ngày không có điểm thực, vừa có rủi ro vi phạm quy chế thi.
- **M8**: điểm mạnh: Rất thực dụng khi cắt hẳn S5 (câu hỏi tổng hợp) và J2/kappa khỏi đường găng trước ngày 05/11 (M8-A), bắt buộc chạy S1 trong WSL2/Docker Linux, cache kết quả parse/embedding theo hash corpus, và lên kịch bản thuê A100 2–3h ngay sau S1 (M8-C). | lỗ hổng: Bài nộp Ngày 1 của M8-B chỉ bọc `try/except` cho `02_rag` mà giữ nguyên `ctx=10k` và `max_new_tokens=2048`, chưa giải quyết ngay lỗi rớt cửa sổ trượt 4096 token.

---

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)
Từ góc nhìn **Pragmatist (Thực dụng)**, tôi giữ chấm điểm Plan v2 ở mức **7.0/10 — "sửa nhỏ rồi làm"**, giữ danh sách **STOP doing** (dừng J2/kappa, dừng viết 15–20 câu thủ công, dừng chạy 32B Q4 offload CPU qua Ollama), nhưng **sửa 3 điểm lớn** sau khi tiếp thu bằng chứng vượt trội từ hội đồng:
1. **BỎ hẳn S5 (60–80 câu tổng hợp) khỏi đường găng tuần đầu (tiếp thu M8-A, M7-C, M5-C):** Vòng trước tôi vẫn giữ `synth60` để đo `retrieve()`. Tuy nhiên, M7-C dẫn chứng từ van Elburg (arXiv 2508.11758v2) rằng câu hỏi synthetic thường chỉ cần 1 đoạn ngắn nên Context Precision có $\tau_\mu = 0.08$ so với câu hỏi người. Việc dựng S5 lúc này tốn 1–1.5 ngày vô ích. Tuần đầu chỉ dùng `sample25` + tài liệu gold S3 + nhiễu C50/C200.
2. **SỬA S3b từ đo `doc-hit@k` sang đo `in_window_quote_hit@4096` + đối chiếu Docling vs PyMuPDF (tiếp thu M5-B, M4-B, M1-A/B):** M5 chỉ ra `02_rag` không lưu `(doc, page)` và nhồi chunk điểm cao nhất lên đầu context 10k, làm 75% số lớp (`sliding_window=4096`) không nhìn thấy chunk tốt nhất khi sinh câu trả lời ở cuối. Tôi sửa S3b để đo trực tiếp tỷ lệ `quote` vàng nằm trong **4096 token cuối cùng** của prompt, đồng thời tính điểm local theo đúng công thức LB ($0.3 \cdot \text{acc}_{\text{DONES}} + 0.7 \cdot \text{acc}_{\text{AI}}$).
3. **SỬA chuỗi 3 bài nộp LB đầu tiên để tách biến độc lập (1-knob-at-a-time) và đảm bảo an toàn Private LB (tiếp thu M5-A/B, M7-B, M8-B):** Vòng trước tôi gộp cả đổi PyMuPDF + giảm token + đổi prompt vào Ngày 1, làm mất khả năng quy thuộc nguyên nhân nếu điểm biến động. Tôi tách lại thành 3 bước sạch (mỗi ngày 1 biến), vừa lấy điểm vượt ngưỡng Top 20 (59.05), vừa suy ra được thời gian parse `P` mà không cần nộp bài 0 điểm như M7-A.

---

## 3. Đề xuất cuối (tối đa 3)

### Đề xuất A: Cắt gọt Plan v2 về 48h (Dừng S5/J2, sửa S2/S3b sang `in_window_hit@4096` & trọng số LB) (dựa trên: M5-B, M8-A, M1-A, M1-B, M4-B)
- **Giả thuyết:** Cắt bỏ hoàn toàn S5 (câu hỏi tổng hợp + câu người viết) và J2/kappa khỏi đường găng tuần đầu, đồng thời sửa S3b để đo tỷ lệ `quote` vàng lọt vào cửa sổ trượt 4096 token cuối sẽ rút ngắn thời gian dựng hạ tầng từ 6 ngày xuống **2 ngày (48h)**, tiết kiệm ~15 giờ GPU local và bắt trúng lỗi kiến trúc của `02_rag`.
- **Cách làm cụ thể (3 sửa bắt buộc cho Plan v2 + Ngân sách thời gian):**
  - **VIỆC PHẢI DỪNG NGAY (STOP doing):** (1) Hoãn vô thời hạn S5 (60–80 câu synthetic + 15–20 câu người viết) cho tới sau ngày 01/11; (2) Bỏ J2 (`kappa ≥ 0.8`) và rút 6 bài test J1 xuống còn 3 bài cốt lõi; (3) Tuyệt đối không chạy `olmo-3.1:32b-instruct` Q4 offload CPU; (4) Kiểm tra ngay việc đăng ký form dự thi IEEE CIS trước hạn chót **01/11/2026** (theo M7).
  - **Ngày 1 (6h — S1 + S2 tinh gọn trên WSL2/Linux):**
    1. Chạy S1 trong WSL2/Linux (để khớp môi trường server và đóng gói bằng `build.sh`). Trong `dev_shim.py`, thêm `assert` kiểm tra model đã nạp đúng 4-bit NF4 và gỡ lệnh `.to("cuda")` thừa; thêm kiểm tra bước đóng gói: tuyệt đối không dính `eval/` vào zip và tổng dung lượng **sau giải nén** $\le 1\text{ GB}$.
    2. **S2 (2 giờ):** Người duyệt 25 câu `sample25` để viết `reference_short` dạng **atomic claims** (bắt buộc giữ đủ mọi con số, đơn vị, thực thể của gold), gán 3 cờ (`outside_doc`, `false_premise`, `closed_book`). Hiệu chỉnh J1 (DeepSeek API, `temperature=0`) chỉ bằng 3 cổng: (i) `reference_short`/diễn đạt lại $\to$ Đúng $\ge 24/25$; (ii) Đổi số/đơn vị hoặc thiếu 1 vế trong câu nhiều vế $\to$ Bác $\ge 23/25$; (iii) Lạc đề $\to$ Bác $25/25$.
    3. Trong `evaluate.py`, tính điểm chính theo công thức quy đổi LB: $\text{Score}_{\text{LB}} = 50 \times (\frac{\text{correct}_{\text{DONES}}}{10} + \frac{\text{correct}_{\text{AI}}}{15})$.
  - **Ngày 2 (6h — S3 + S3b + S4 rút gọn C100):**
    1. Tải tài liệu gold cho $\ge 20/25$ câu mẫu (S3) + gom thêm ~75 PDF cùng chủ đề làm tập nhiễu `C100` (gộp S4, cache text/embedding ra đĩa theo `sha256` để không bao giờ parse lại).
    2. **Sửa S3b:** Chạy `Docling` vs `PyMuPDF` trên bộ PDF gold (M4-B) và đo 2 chỉ số retrieval mới: **`quote_containment`** (% `quote` vàng còn nguyên sau khi parse) và **`in_window_quote_hit@4096`** (% `quote` vàng thực sự nằm trong 4096 token cuối cùng của prompt đưa vào Olmo). Chạy thêm thí nghiệm `oracle-chunk` (đưa trực tiếp đoạn chứa `quote` vào 7B 4-bit, theo M4-A) để tách lỗi retrieval khỏi lỗi generator.
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:**
  - Nếu trong S3b, cách xếp chunk mặc định của `02_rag` (best-chunk ở đầu context 10k) làm `in_window_quote_hit@4096` $< 50\%$ trong khi đảo thứ tự (`evidence-last` hoặc cắt context $\le 3800$ token) nâng chỉ số này lên $> 85\%$ và giúp 7B 4-bit thắng $\ge +3$ câu trên `sample25` $\to$ xác nhận lỗi cửa sổ trượt là nút thắt số 1.
  - **Tiêu chí bác bỏ:** Nếu `PyMuPDF` làm rơi $> 15\%$ số `quote` vàng nằm trong bảng biểu so với `Docling` trên tập gold S3 $\to$ bác bỏ việc thay thế hoàn toàn Docling bằng PyMuPDF thô, bắt buộc bật `page.find_tables()` của PyMuPDF trước khi nộp biến thể parser.
- **Kỳ vọng:** Tiết kiệm 3–4 ngày làm việc; có bộ eval chuẩn xác ngay tối Ngày 2 | **Tự tin:** cao | **Chi phí:** S (12 giờ làm việc trong 2 ngày)

---

### Đề xuất B: Chiến dịch 3 bài nộp LB tuần đầu — Tách biến độc lập (1-knob-at-a-time) (dựa trên: M5-A, M5-B, M7-B, M7-C, M8-B, M2-A)
- **Giả thuyết:** Thay vì nộp bài "hi sinh" 0 điểm (M7-A) hay gộp nhiều thay đổi làm mù nguyên nhân (M6-A cũ), việc nộp 3 bài liên tiếp từ Ngày 1 đến Ngày 3 mà mỗi bài chỉ thay đổi **đúng 1 cụm biến số độc lập** (Ngày 1: Cửa sổ 4096 + Trần sinh + An toàn; Ngày 2: Đổi Parser PyMuPDF; Ngày 3: Hybrid BM25+Dense RRF + Title-boost) sẽ vừa đo được chính xác thời gian parse/decode trên server, vừa lật ít nhất 1–2 câu DONES (+3.33 đến +6.67 điểm) để vượt mốc Top 20 (59.05).
- **Cách làm cụ thể (3 bài nộp trong Ngày 1–3, đóng gói qua WSL2 `build.sh`):**
  - **Bài 1 (Ngày 1) — `Sub1_window4k_safe` (Cố định Docling + MiniLM của `02_rag`, chỉ sửa cửa sổ Attention & độ dài sinh):**
    - Giữ nguyên Docling và MiniLM của `02_rag` để làm mốc so sánh parser cho Ngày 2.
    - Sửa 4 lỗi chí mạng ở tầng chuẩn bị prompt & sinh: (1) Quét file bằng `Path(corpus_dir).rglob("*")` lọc `p.suffix.lower() == ".pdf"` và bọc `try/except` từng câu (nếu lỗi thì fallback về closed-book); (2) Thu gọn context từ 10k xuống **3600 token** và **đảo thứ tự đặt chunk điểm cao nhất nằm sát ngay trước câu hỏi** (`evidence-last`, lọt 100% vào cửa sổ trượt 4096 của cả 64 lớp 32B); (3) Đặt tường minh `do_sample=False, use_cache=True, max_new_tokens=384`; (4) Thêm 1 câu vào prompt: *"Answer concisely in 1–3 sentences, stating the exact value/unit first; address all parts of the question and explicitly refute any false premise."*
  - **Bài 2 (Ngày 2) — `Sub2_pymupdf_speed` (Giữ nguyên generator/prompt Bài 1, chỉ thay Docling bằng PyMuPDF):**
    - Thay toàn bộ bước parse Docling bằng `fitz` (PyMuPDF + `find_tables()` chuyển bảng sang markdown).
    - **Phép đo kép:** Hiệu số thời gian $(T_{\text{Sub1}} - T_{\text{Sub2}})$ cho biết chính xác thời gian Docling tiêu tốn trên corpus thật (giải quyết bài toán tìm $P$ của M7-A mà không mất lượt nộp), đồng thời hiệu số điểm $(\text{Score}_{\text{Sub2}} - \text{Score}_{\text{Sub1}})$ cho biết PyMuPDF có làm mất thông tin bảng biểu trên tập test ẩn hay không.
  - **Bài 3 (Ngày 3) — `Sub3_hybrid_rrf_dones` (Giữ parser/generator tốt nhất từ Bài 1–2, chỉ nâng cấp Retrieval):**
    - Chuyển MiniLM lên GPU (`device="cuda"`), chèn header `[DOC: <tên file> | <dòng tiêu đề đầu>]` vào mỗi chunk (giúp bắt dính 13/15 câu AI nêu tên bài báo), và kết hợp **BM25 (viết bằng numpy) + MiniLM qua RRF ($k=60$)** để bắt các thuật ngữ/số liệu hạt nhân của DONES (40 MeV, 125 mA, LIPAc, WPENS).
    - **Quy tắc cổng local $\to$ LB (chống nhiễu $N=25$):** Trước khi nộp Bài 3, kiểm tra trên `sample25`: chỉ nộp nếu `in_window_quote_hit@4096` tăng $\ge +10\%$ và kết quả chấm local thắng Bài 2 ít nhất **$\ge +3$ câu ròng (hoặc $\ge +5.0$ điểm quy đổi LB)** và thua $\le 1$ câu.
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:**
  - Nếu Bài 1 đạt $> 59.05$ điểm và thời gian giảm từ 2080 s xuống $< 1300\text{ s}$ $\to$ xác nhận giả thuyết `sliding_window=4096` + cắt `max_new_tokens=384` vừa tăng điểm vừa cứu bài nộp khỏi giới hạn 1h của Private LB.
  - **Tiêu chí bác bỏ:** Nếu Bài 1 tụt $\ge 2.86$ điểm (tương đương mất 2 câu AI hoặc 1 câu DONES) so với 57.14 $\to$ Judge ẩn thiên vị câu trả lời dài (verbosity bias như M3 cảnh báo); lập tức khôi phục `max_new_tokens=1024` và giữ nguyên thứ tự đảo chunk ở bài kế tiếp. Nếu Bài 2 tụt điểm so với Bài 1 $\to$ giữ lại Docling cho các bài sau.
- **Kỳ vọng:** +3.3 đến +8.0 điểm LB (đạt ~60.5–65.0 điểm, lọt Top 20 ngay ngày thứ 3) và giảm runtime xuống $< 800\text{ s}$ | **Tự tin:** cao | **Chi phí:** S (2 giờ code/ngày)

---

### Đề xuất C: Kiểm tra 0-đồng config 32B ngay hôm nay, Thuê A100 80GB (2.5h ở Ngày 3) & Khoá biên giới Ollama (dựa trên: M8-C, M2-C, M5, M7)
- **Giả thuyết:** (1) Tải và đối chiếu các file cấu hình JSON/Jinja của bản 32B ngay trên máy local (tốn 0 đồng, 10 phút) sẽ chặn trước các lỗi ngầm về `use_cache`, `do_sample` và chat template; (2) Một phiên thuê A100 80GB kéo dài 2.5 giờ ngay sau khi có `C100` và kết quả LB Bài 1–2 là đủ để đo chính xác tốc độ `Olmo-3.1-32B-Instruct` bf16 thật và độ ổn định (variance) mà không cần thuê H100 đắt đỏ.
- **Cách làm cụ thể (Trả lời câu hỏi 3, 4, 7):**
  1. **Kiểm tra 0-đồng ngay trong Ngày 1 (15 phút, không cần GPU 80GB):**
     - Dùng script Python tải riêng 3 file nhẹ (`config.json`, `generation_config.json`, `tokenizer_config.json` / `chat_template.jinja`) của `allenai/Olmo-3.1-32B-Instruct` và `allenai/Olmo-3-7B-Instruct`.
     - Diff trực tiếp: xác nhận `sliding_window=4096`, kiểm tra xem `generation_config.json` của 32B có mặc định `do_sample: true, temperature: 0.6` không (để ghi đè tường minh trong code nộp), và kiểm tra chuỗi `apply_chat_template` khi có/không có system prompt.
  2. **Phiên thuê A100 80GB duy nhất (2.5 giờ vào Ngày 3, chi phí ~$4):**
     - **0:00–0:30:** Chạy container Linux offline (`HF_HUB_OFFLINE=1`), load `Olmo-3.1-32B-Instruct` bf16, đo VRAM đỉnh và tốc độ prefill/decode (tok/s) ở các mức context `{2k, 3.6k, 8k, 10k}`.
     - **0:30–1:30:** Chạy `Sub1` và `Sub3` bằng **32B bf16 thật** trên `sample25` + `C100` **2 lần liên tiếp** để: (a) đếm số câu lật chữ/lật verdict do nhân SDPA không tất định trên GPU (M5-A), và (b) đối chiếu xem các câu 7B 4-bit làm sai ở S3b có được 32B bf16 giải đúng không (đặc biệt các câu tính toán `calc` và `false_premise`).
     - **1:30–2:30:** Kết hợp thời gian thực tế của `Sub1` và `Sub2` trên LB (H100) với số đo trên A100 (áp dụng tỷ lệ băng thông HBM $3.35 / 2.04 \approx 1.64\times$ cho decode `bs=1` và $\sim 2.4\times$ thực tế cho prefill) để lập phương trình dự phóng thời gian chạy 100 câu Private LB. **Không cần thuê H100** trừ khi thời gian dự phóng Private LB vượt quá 80% giới hạn (tức $> 2880\text{ s}$ nếu trần 1h, hoặc $> 5760\text{ s}$ nếu trần 2h).
  3. **Quy chế cứng cho Ollama local:**
     - **CHỈ DÙNG** `olmo-3:7b-instruct` (với `options={"num_ctx": 8192, "temperature": 0}`) để test nhanh luồng code và chạy kiểm tra `closed_book` trong vài phút.
     - **KHÔNG DÙNG** Ollama làm Judge J2, không dùng `32b-instruct` Q4 offload CPU, và không dùng điểm Ollama để chọn prompt nộp LB.
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:**
  - Nếu trên A100 80GB, chạy 32B bf16 greedy 2 lần liên tiếp vẫn làm lật điểm $> 1/25$ câu do SDPA non-deterministic $\to$ bắt buộc bật `torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)` hoặc `torch.use_deterministic_algorithms(True, warn_only=True)` trong `main.py` nếu không làm chậm decode quá 15%, nhằm tránh rủi ro bài thắng public bị tụt điểm ở private (theo M5-A).
  - **Tiêu chí bác bỏ việc dùng 7B 4-bit chọn prompt:** Nếu thứ hạng giữa 2 prompt trên `sample25` khi chạy bằng 7B 4-bit local bị **đảo chiều** so với khi chạy bằng 32B bf16 trên A100 $\to$ từ tuần thứ 2 trở đi, máy local chỉ được dùng để tune retrieval (`in_window_quote_hit@4096`), mọi thay đổi prompt/generator bắt buộc test qua phiên thuê A100 ngắn (1 giờ/tuần) trước khi nộp LB.
- **Kỳ vọng:** Triệt tiêu 100% lỗi cấu hình/OOM/timeout trên 32B thật; tiết kiệm >50% ngân sách thuê cloud | **Tự tin:** cao | **Chi phí:** S (15 phút local + 2.5 giờ A100)

---

## 4. Bình chọn
VOTES: M5-B, M8-A, M7-B

---

# M7 = qwen (qwen/qwen3.8-max-0902) — vai: competition_veteran

## 1. Nhận xét từng thành viên

- M1: điểm mạnh: chỉ ra rủi ro lớn nhất là `reference_short` không giống tham chiếu ẩn “concise”; đề xuất audit nhãn 25 câu và metric theo đúng cấu trúc LB là rất đáng giá. | lỗ hổng: đề xuất 0.3·DONES + 0.7·AI không khớp công thức domain-balanced trong brief (`50·(a/15+b/35)` = trung bình accuracy hai domain); các ngưỡng trên n=25 hơi quá chính xác; ưu tiên làm nhãn trước khi dùng LB vẫn có thể làm chậm nộp bài.

- M2: điểm mạnh: thực dụng, giữ nhịp nộp song song, giảm hiệu chỉnh judge thành công cụ so sánh tương đối, không biến J2/kappa thành dự án. | lỗ hổng: “ngày 1 nộp 02 làm control” dễ trùng với việc đã bị loại là nộp lại nguyên bản; thuê A100 trước khi có probe runtime trên LB là chưa đúng thứ tự; chưa xử lý rủi ro sliding-window 4096.

- M3: điểm mạnh: nhìn đúng LB là thiết bị đo thật, đề xuất 3 probe format/độ dài để học “khẩu vị” judge ẩn; ý tưởng extractive-first đáng chú ý. | lỗ hổng: quote-only/extractive-first là thay đổi quá lớn và có thể fail nếu judge muốn câu trả lời tự nhiên chứ không phải trích dẫn thô; LLM rerank bằng 32B pointwise có nguy cơ đội runtime rất lớn nếu chưa đo prefill.

- M4: điểm mạnh: các thí nghiệm oracle-chunk và so Docling/PyMuPDF bằng quote vàng là cách rẻ để tách retrieval/parser/generator. | lỗ hổng: vẫn nghiêng về chạy local 7B nhiều, dễ chậm; chưa có cơ chế nộp LB sớm để neo tín hiệu; kết quả 7B 4-bit không đủ đại diện cho 32B bf16.

- M5: điểm mạnh: phát hiện rất sắc về sliding-window 4096 và việc phải đo “in-window hit” thay vì doc-hit thô; quy tắc nộp an toàn và cảnh báo private chấm lại bài public tốt nhất là đúng tinh thần competition. | lỗ hổng: dành một ngày chỉ để lặp cùng zip có thể lãng phí slot; “cấm” generator trên 7B quá tuyệt đối có thể làm tê liệt vòng lặp local; một số giả định về zip/build chưa được brief xác nhận đầy đủ.

- M6: điểm mạnh: thúc ép nộp sớm, cắt phần synthetic thừa, đề xuất 3 bài LB liên tiếp và quy tắc net-win trên sample25 rất hợp tinh thần Kaggle. | lỗ hổng: kỳ vọng +5–9 điểm thiếu bằng chứng; thay đổi parser/context/retrieval gần như cùng lúc ở 3 bài đầu sẽ gây khó quy nguyên nhân; công thức quy đổi A100→H100 quá chính xác trong khi chưa có số đo thật.

- M7 (tôi): điểm mạnh: nêu probe đo fixed cost/prefill và ý tưởng DONES-null để tách điểm theo domain; gắn sliding-window với evidence-last. | lỗ hổng: dùng 4 slot probe là quá nhiều; DONES-null khó thực hiện vì `answer()` không nhận domain label; cần merge với cách probe 1-knob của M8/M3.

- M8: điểm mạnh: cân bằng tốt giữa nộp sớm và eval tối giản; cắt S5 khỏi critical path, giữ S3/S3b, đề xuất 3 bài A/B một nút; dùng LB làm holdout thật là đúng. | lỗ hổng: chưa đưa explicit metric “in-window 4096”; ngày 3 chuyển PyMuPDF có thể confound nếu chưa audit quote coverage trước.

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)

- Giữ: LB phải được dùng như harness chính trong 3 ngày đầu; phải có ít nhất một probe đo phần thời gian không sinh token; synthetic chỉ dùng để chọn retrieval/chunking, không dùng để xếp generator; generator chỉ được chốt dựa trên sample25 + LB.

- Bỏ: ý định dùng DONES-null trong 3 bài đầu, vì `answer(question, corpus_dir, model_id)` không có domain label và heuristic domain dễ làm probe sai; bỏ việc dành nguyên một ngày chỉ để lặp cùng zip; bỏ yêu cầu J2/kappa như cổng chặn; bỏ chạy generator trên câu tổng hợp ở critical path.

- Sửa: gộp probe của M7-A với chuỗi 1-knob của M3-A/M8-B thành 3 bài: (1) 02 đã hardening nhưng giữ kiến trúc gốc, (2) probe không sinh token để đo load/index/retrieval, (3) cap `max_new_tokens` để test độ dài. Nhận M5-B làm metric retrieval chính: quote phải nằm trong vùng cửa sổ 4096 có hiệu lực, không chỉ hit tài liệu. Nhận M8-A/M6-B để cắt S5. Nhận M1-A nhưng chỉ làm ở mức time-box: 25 `reference_short` dạng atomic claim, không biến thành dự án nhãn dài ngày.

- Đổi quan điểm so với chính mình: trước đây tôi ưu tiên DONES-null để biết điểm DONES/AI. Nhưng M5-B và M8-A thuyết phục hơn: trong 3 slot đầu, probe runtime và probe độ dài cho quyết định nhanh hơn; phân tách domain có thể làm sau nếu còn slot hoặc nếu local có domain classifier đủ tin.

- Về Ollama/A100: chỉ dùng Ollama cho smoke test/logic hoặc lọc closed-book rẻ, không dùng làm judge hay proxy 32B. A100 chỉ thuê nếu sau probe runtime mà ngân sách thời gian còn <20% hoặc cần xác nhận chat template/32B bf16; không thuê trước khi có số LB.

## 3. Đề xuất cuối (tối đa 3)

### Đề xuất A: 3 bài LB đầu tiên = safe baseline, non-generation probe, concise-cap probe (dựa trên: M7-A, M3-A, M5-A, M8-B)

- Giả thuyết: LB là eval harness miễn phí và duy nhất khớp judge ẩn. Ba bài đầu nên đo: (i) hardening có làm đổi điểm không, (ii) phần không sinh token tốn bao nhiêu, (iii) rút ngắn câu trả lời có lợi hay hại. Giả định: Codabench chấp nhận bài điểm thấp và không trả output; ta có thể buộc model load/index dù câu trả lời là chuỗi ngắn.

- Cách làm cụ thể:
  1. Tạo bản hardening tối thiểu cho `02_rag`:
     - Tìm PDF đệ quy nhưng chỉ dùng nếu không làm số PDF tăng đột biến; lọc đuôi case-insensitive: `p.suffix.lower() == ".pdf"`.
     - `try/except` từng câu; nếu lỗi, trả về chuỗi fallback ngắn kiểu `"Insufficient evidence to answer."`.
     - Ép UTF-8 cho mọi read/write.
     - Ép greedy deterministic: `do_sample=False`, `temperature=0`, `top_p=None`; nếu starter đã greedy thì thêm assert/log.
     - Watchdog chỉ can thiệp khi sắp vượt budget: nếu thời gian còn <10–15%, giảm `max_new_tokens` về 64 hoặc trả fallback.
     - Không đưa `dev_shim.py`, file dev, cache local vào zip.
  2. Ngày 1 — Sub1 `02-safe-original-gen`: giữ prompt/context/`max_new_tokens=2048` như 02, chỉ thêm hardening. Mục tiêu: so với 57.14/2080 s để biết hardening/nondeterminism có đáng kể không.
  3. Ngày 2 — Sub2 `no-gen probe`: giữ load model + parse/index + retrieval/prompt assembly nếu có, nhưng không sinh dài; mỗi câu trả về chuỗi ngắn cố định hoặc sinh tối đa 1 token nếu cần để tránh lỗi pipeline. Mục tiêu: đo `T_non_gen` ≈ load + parse/index + retrieval/prompt overhead.
  4. Ngày 3 — Sub3 `cap-512`: từ Sub1, chỉ đổi `max_new_tokens` 2048 → 512; giữ prompt/context để cô lập yếu tố độ dài. Nếu local sample25 cho thấy 512 cắt mất >10% câu cần thiết, có thể dùng 640, nhưng phải ghi rõ.

- Thí nghiệm kiểm chứng + tiêu chí bác bỏ:
  - Sub1 vs mốc 57.14: nếu lệch <1.43 điểm và runtime ±5%, hardening gần như trung tính; nếu lệch ≥1.43, phải xem hardening hoặc nondeterminism là biến thật, mọi cải tiến sau phải so với cả Sub1 và mốc BTC.
  - Sub2: tính `v_gen ≈ (T_Sub1 − T_Sub2)/50`. Nếu `T_Sub2 + 50*v_gen` hiện tại >3400 s cho public 50 câu hoặc `T_Sub2 + 100*v_gen` >6800 s cho private 100 câu, thiết kế sinh dài hiện tại không an toàn; phải ưu tiên giảm token/parser. Nếu `T_Sub2` rất lớn (>50% T_Sub1), nút thắt không phải generation mà là load/index/retrieval.
  - Sub3: nếu điểm giảm ≥3.33 so với Sub1, cap 512 có hại hoặc judge cần câu dài hơn; bác bỏ cap. Nếu điểm không giảm hoặc tăng và runtime giảm ≥20%, nhận cap làm mặc định mới. Nếu giảm đúng 1.43, chưa kết luận; chỉ xác nhận thêm bằng sample25/LB khi có paired flips thuận.

- Kỳ vọng: +0 trực tiếp từ probe; nếu Sub3 phù hợp: +1.4 đến +3.3 điểm và giảm mạnh runtime | Tự tin: cao | Chi phí: S

### Đề xuất B: Eval harness tối giản — synthetic chỉ cho retrieval, generator chỉ quyết trên sample25/LB (dựa trên: M8-A, M6-B, M1-A, M4-B)

- Giả thuyết: van Elburg 2025 trong brief cho thấy synthetic không xếp hạng generator tốt; vì vậy không nên đốt GPU để chạy generator trên synthetic. Judge local chỉ đủ dùng để so sánh tương đối và phát hiện hồi quy, không phải chân lý tuyệt đối.

- Cách làm cụ thể:
  1. Cắt S5 khỏi critical path. Nếu vẫn tạo synthetic, chỉ dùng cho retrieval-only qua hook `retrieve(question, corpus_dir)`; không chạy LLM sinh câu trả lời trên synthetic. Bỏ 15–20 câu human nếu chưa có owner.
  2. Rút S2 còn 3 test đủ chặn:
     - Paraphrase đúng của `reference_short`/gold ngắn phải được chấm đúng ≥24/25.
     - Đổi số/đơn vị/thực thể chính phải bị bác ≥90%.
     - Trả lời dài lạc đề phải bị bác ≥24/25.
     Bỏ cổng J2/kappa ≥0.8; nếu có J2 thì chỉ ghi bất đồng.
  3. Dựng `reference_short` theo atomic claim trong 1 buổi: mỗi câu có các fact atom: giá trị + đơn vị + thực thể + điều kiện; nếu câu có false premise, ghi rõ expected behavior là bác tiền đề. Không viết lại gold dài theo văn phong.
  4. `evaluate.py` báo:
     - accuracy riêng DONES và AI.
     - balanced score = `50 * (acc_DONES + acc_AI)`, không dùng micro-accuracy.
     - paired flips so với baseline, số câu thắng/thua theo domain.
     - không quyết định trên slice có <8 câu.
  5. Retrieval/chunking chỉ được chọn bằng retrieval-only: `doc-hit@k`, `quote-recall@k`, và `in_window_quote@k` theo Đề xuất C. Trước khi đo, dedup near-duplicate PDF bằng sha256/MinHash để tránh hit ảo.
  6. Generator/prompt chỉ được đưa lên LB nếu qua cổng sample25:
     - Có paired comparison với baseline cùng retrieval.
     - Net flips ≥4 và số câu thua ≤1, hoặc net flips ≥6 nếu judge/local noise >1.
     - Không được làm domain kia giảm >1 câu.

- Thí nghiệm kiểm chứng + tiêu chí bác bỏ:
  - Nếu trên sample25 với corpus gold, `02` không thắng `01` hoặc `doc-hit@5` <60%, dừng tune retrieval mù; phải debug parser/retrieval trước.
  - Nếu J1 fail 3 test sau 2 lần sửa rubric, không dùng J1 để xếp generator; chuyển sang LB làm trọng tài chính.
  - Nếu một thay đổi generator vượt cổng local nhưng lên LB không tăng ≥1.43 điểm so với bài tốt nhất trong Sub1/Sub3, bác thay đổi đó.
  - Nếu retrieval-only tăng mạnh nhưng generation không tạo ≥2 net flips trên sample25, giả thuyết “retrieval là nút thắt” bị bác cho cấu hình hiện tại.

- Kỳ vọng: tiết kiệm 2–3 ngày và nhiều giờ GPU; tránh tối ưu theo proxy sai | Tự tin: cao | Chi phí: S

### Đề xuất C: Retrieval/packing theo sliding-window 4096 + hybrid BM25/dense cho DONES (dựa trên: M5-B, M7-C, M6-A, M4-B)

- Giả thuyết: Olmo có sliding-window 4096 ở đa số lớp, nên evidence nằm quá xa câu hỏi có thể bị giảm tác dụng; câu DONES nhiều acronym/số/bảng nên MiniLM thuần dễ thiếu lexical. Vì vậy cần đo quote trong cửa sổ hiệu dụng và thêm BM25.

- Cách làm cụ thể:
  1. Audit parser trước khi thay Docling:
     - Với 20–25 PDF gold, kiểm tra `quote` xuất hiện trong text PyMuPDF và text Docling bằng fuzzy containment ≥0.9.
     - Nếu PyMuPDF đạt ≥90% và nhanh hơn nhiều, ưu tiên PyMuPDF cho P4.
     - Nếu PyMuPDF <70% trong khi Docling >90%, giữ Docling hoặc chỉ dùng PyMuPDF cho tài liệu không có bảng/công thức.
  2. Chunking:
     - Chunk theo section/paragraph, 400–800 token, overlap ~15%.
     - Prefix mỗi chunk bằng đường dẫn header + title/filename, ví dụ: `IFMIF-DONES / §3.2 Beam diagnostics /`.
     - Với câu AI nêu tên paper, boost title/filename match.
  3. Retrieval hybrid:
     - BM25 tự viết hoặc thư viện nhẹ + dense MiniLM hiện có.
     - Fuse bằng RRF k=60.
     - Không thêm model >1GB.
  4. Packing context:
     - Mục tiêu tổng prompt ≤3800 token nếu có thể; nếu không, luôn đặt chunk điểm cao nhất ngay trước câu hỏi.
     - Không để evidence quan trọng nằm ngoài 4096 token cuối.
     - Với context dài, có thể lặp top-1 chunk ở cuối nhưng phải đo runtime.
  5. Metric retrieval-only chính:
     - `quote-recall@k`: quote vàng có xuất hiện trong các chunk trả về.
     - `in_window_quote@k`: quote vàng có nằm trong vùng prompt sẽ được model thấy hiệu quả, đặc biệt 4096 token cuối.
     - Báo riêng DONES và AI.

- Thí nghiệm kiểm chứng + tiêu chí bác bỏ:
  - Trên DONES gold/C50, hybrid BM25+dense phải tăng `quote-recall@5` ≥10 điểm phần trăm hoặc ≥4 câu so với dense-only, đồng thời `in_window_quote@5` ≥80%. Nếu không, bác hybrid hoặc quay lại sửa chunking/embedder.
  - Nếu retrieval-only thắng nhưng chạy generation trên sample25 không đạt ≥2 net flips so với retrieval cũ trong cùng packing, nút thắt không nằm ở retrieval; chuyển sang format/generator.
  - Nếu PyMuPDF làm mất quote nhiều hơn Docling ≥20 điểm phần trăm, không được thay Docling toàn cục chỉ vì tốc độ.

- Kỳ vọng: +3.3–6.7 điểm nếu retrieval DONES là nút thắt; đồng thời giảm runtime nhờ context ngắn hơn | Tự tin: vừa | Chi phí: M

## 4. Bình chọn

VOTES: M5-B, M8-A, M3-A

---

# M8 = glm (z-ai/glm-5.3) — vai: simplifier

## 1. Nhận xét từng thành viên
- **M1**: Điểm mạnh nhất: đề xuất A soi `reference_short` (proxy-bọc-proxy vì gold dài ≠ tham chiếu ẩn concise) bằng ma trận lật 3 chiều — đúng chỗ rủi ro nhãn, rẻ. | Lỗ hổng: chỉ thêm cổng phân tích, không giải phóng kênh tín hiệu miễn phí đang bỏ trống (LB); đề xuất B đòi "≥8 câu/type" tăng khối lượng synthetic mà chính trích dẫn van Elburg nói không xếp hạng được generator.
- **M2**: Điểm mạnh nhất: A — nộp song song từ S1, mỗi bài 1 knob, đúng với dữ kiện mới "đồng ý nộp song song"; B đúng khi hạ judge xuống công cụ so sánh tương đối. | Lỗ hổng: A đổi knob (max_new 640) trước khi xác minh lại bản build gốc (chưa có điểm tái lập 57.14 → delta không quy được cho knob hay lỗi đóng gói); C thuê A100 "ngay sau S1" hơi sớm.
- **M3**: Điểm mạnh nhất: A — đảo trục dùng LB tất định làm thiết bị đo, đo sàn nhiễu bằng nộp lại bản gốc; ý thức verbosity bias hai chiều. | Lỗ hổng: B (extractive-first) là viết lại toàn bộ pipeline khi chưa có bằng chứng nào rằng generator là nút thắt (thiếu bước oracle); A đốt 3 slot thuần đo trước khi có biến thể an toàn nào.
- **M4**: Điểm mạnh nhất: A oracle-chunk là thí nghiệm triage rẻ–quyết định nhất toàn hội đồng; B so Docling vs PyMuPDF bằng quote vàng cũng rẻ và quyết. | Lỗ hổng: chuỗi phụ thuộc nặng ("sau khi J1 hiệu chỉnh", "sau S3"), mâu thuẫn với thực tế 39 slot LB đang cháy mà chưa có bài; ngưỡng của C vẫn dựa 7B 4-bit.
- **M5**: Điểm mạnh nhất: A quy tắc "biến thể phải thắng cả 2 mốc neo" + đo nhiễu server; B phát hiện metric doc-hit đo sai chỗ (cửa sổ trượt 4096, ctx 10k) — góc nhìn mới thật. | Giả định chưa kiểm chứng: "48/64 lớp SWA của 32B" (brief chỉ xác nhận 24/32 của 7B — cùng hướng nhưng tôi không kiểm được); ngày-1 đã đổi max_new 384 nên mất điểm neo build gốc.
- **M6**: Điểm mạnh nhất: B — danh sách cắt cụ thể nhất (không chạy LLM trên synthetic, judge 2 test, quy tắc net-win ≥ max(2, k_noise+1)), cộng số arity S6 ≈ 460 generation = 4–29h GPU lộ vỡ kế hoạch. | Lỗ hổng: A gộp 2–3 knob/bài nộp (PyMuPDF + max_new 512) → delta không quy được; câu "3600 token kích hoạt đủ 32/32 lớp" sai về kiến trúc (SWA là cố định); hệ số quy đổi 3 chữ số thập phân của C là giả chính xác.
- **M7**: Điểm mạnh nhất: A bộ probe (P, prefill, lattice) phân rã đúng các ẩn số không đo được ở local (P, kích thước corpus, chi phí prefill trên H100 thật); điểm 9 và 6 (kappa paradox) chính xác. | Lỗ hổng: 4 slot probe trước khi có biến thể cải thiện điểm; trộn nguồn đã tra (trang cuộc thi) với nguồn tôi không kiểm được (paper 2512.13961, corpus private=public — đánh dấu chưa kiểm chứng); bộ nghiệm lattice thiếu cặp (3,33).

## 2. Tôi đổi quan điểm thế nào (giữ / bỏ / sửa gì, vì sao)
- **Chấm v2: 7/10 — "sửa nhỏ rồi làm"**, với điều kiện các sửa dưới đây.
- **Giữ** phần tôi cũ: cắt S5 khỏi critical path, S2 gọn còn 3 test. **Sửa**: thay ngưỡng quyết định mơ hồ của tôi ("≥2–3 câu lật") bằng quy tắc của M6-B/M5-A: generator đổi chỉ tin khi net-win ≥ max(2, k_noise+1) trên sample25, rồi xác nhận LB với |Δ| ≥ max(2×nhiễu, 2.86).
- **Bỏ** đề xuất cũ thuê A100 "ngay sau S1": tôi bị M7-th.10 thuyết phục — LB cho sẵn số H100 trên corpus thật, thứ rental không tái tạo được; check template 7B/32B chỉ cần tải file tokenizer vài MB (M7-th.4, chi phí 0). A100 chỉ còn value sau khi có probe.
- **Bỏ** ngày-3 "đổi PyMuPDF" trong chuỗi nộp cũ của tôi: đổi parser là một quyết định kiến trúc, phải qua quote-containment local (M4-B) trước, không đốt slot để thử.
- **Tôi không theo** M3-B (extractive-first) cho tới khi oracle-chunk (M4-A) chứng minh generator là nút thắt — bằng chứng hiện tại chỉ là smoke test 3 câu.
- **Ollama**: giữ quan điểm gần-(e), nhưng chấp nhận 2 dụng cụ hẹp của M6-C/M7-th.11: (c) lọc cờ `closed_book_easy`, (a) smoke raw-mode nếu ép `num_ctx` ≥ prompt thật + temperature 0. Cấm làm judge, cấm proxy 32B.

## 3. Đề xuất cuối (tối đa 3)

### Đề xuất A: Chuỗi 4 slot "đo trước, tối ưu sau" (dựa trên: M7-A, M2-A, M5-A)
- **Giả thuyết:** Ba ẩn số lớn nhất (lỗi đóng gói, cấu trúc corpus, P = load+parse+index) đo được từng cái một bằng slot LB tất định; mỗi bài chỉ 1 thay đổi.
- **Cách làm cụ thể:** Ngày 1: build lại **02_rag nguyên văn** → kỳ vọng ≈57.14/2080 s. Ngày 2: + **chỉ** bản vá an toàn (rglob không phân biệt hoa/thường, try/except từng câu, UTF-8, watchdog từng câu). Ngày 3: **P-probe** — giữ nguyên load/parse/index, `answer()` trả chuỗi hằng (điểm ~0, không bao giờ thành best-public vì 57.14 đã tồn tại → rủi ro private = 0); wall-time = P trên corpus thật → T_private ≈ 2×2080 − P. Ngày 4: 1 knob format (prompt answer-first + max_new_tokens ~512, đã kiểm không thua trên sample25 theo đúng quy tắc của plan).
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Ngày 1 lệch ≥ 2.86 → build/Linux/offline hỏng, dừng mọi biến thể. Ngày 2 giữ ≈57.14 → corpus phẳng; tăng ≥ 3 điểm → corpus lồng nhau (sửa retrieval scope trong P4). Ngày 3: P < 560 s → bản tốt nhất hiện tại **có nguy cơ fail private 100 câu** nếu quy định 1h cứng áp dụng (xung đột M7-th.1 vs brief "2h mềm" — phải đọc lại rule trước ngày 3) → P4 phải cắt ctx/max_new/parser trước khi tối ưu accuracy. Ngày 4 mất ≥ 3 câu AI → judge ưa dài, hoàn tác.
- **Kỳ vọng:** chặn rủi ro private = 0 điểm | **Tự tin: cao** | **Chi phí: S** (4 slot, ~2h code)

### Đề xuất B: Cắt eval + triage bằng oracle-chunk (dựa trên: M6-B, M2-B, M4-A, M7-th.9)
- **Giả thuyết:** S2/S5/S6 hiện có số lượng cổng vượt quá những gì 25 câu + judge proxy hỗ trợ được; synthetic chỉ đủ tin cho retrieval (van Elburg: τ retriever 0.44–0.75, generator ≈0).
- **Cách làm cụ thể:** (i) J1 còn 3 test tự động (paraphrase ≥24/25; đổi số/đơn vị → bác ≥90%; lạc đề → bác 25/25), **bỏ cổng kappa ≥0.8** (M7-th.6: kappa thấp vì cấu trúc biên, không vì judge kém), J2 chỉ là báo cáo; (ii) synthetic **chỉ chạy hook `retrieve()`**: quote-recall@5 + doc-hit@k + thời gian index/retrieve, **không generation LLM**; (iii) LLM chỉ chạy trên sample25: 01 vs 02 + **oracle-chunk** (context thật vs đúng chunk chứa quote) → bảng lỗi 4 nhóm quyết định hướng P4; (iv) cache parse+embedding theo hash corpus; (v) **xóa**: 15–20 câu human (không người phụ trách), C500, bảng tương quan local↔LB (<6 điểm LB không ước lượng được tương quan).
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Oracle − retrieved ≥ 15 điểm % và "đúng tài liệu sai đoạn" ≥ 50% lỗi → ưu tiên retrieval (P4 hybrid BM25+RRF, chọn bằng quote-recall trên tài liệu gold, không tốn LLM). Oracle ≈ retrieved (≤5 điểm) → ưu tiên generator/format. Nếu 02 ≤ 01 trên sample25 → hỏng thượng nguồn, dừng eval.
- **Kỳ vọng:** tiết kiệm ~3 ngày + ~14h GPU; bài thật từ ~24–25/10 | **Tự tin: cao** | **Chi phí: S**

### Đề xuất C: Kiểm parser + cửa sổ 4096 không tốn slot (dựa trên: M4-B, M5-B)
- **Giả thuyết:** Docling có thể làm mất quote (bảng/công thức, `do_table_structure=False` vì thiếu cv2) và ctx 10k đặt evidence ngoài cửa sổ 4096 của ~75% lớp SWA (khớp hướng 24/32 của 7B trong brief; con số 48/64 của M5 chưa kiểm chứng) → doc-hit và chất lượng sinh đều đo lệch.
- **Cách làm cụ thể:** Trên ≥20 tài liệu gold (S3): trích text bằng PyMuPDF và Docling (cùng cờ với 02), đo tỉ lệ quote chứa được (fuzzy ≥0.9) từng bên; với bản sao có instrument provenance của 02 (không phải bài nộp), tính **in_window_hit** = tỉ lệ quote của chunk tốt nhất nằm trong 4096 token cuối ở ctx 10k; ablation 7B: chunk tốt nhất đặt đầu vs cuối.
- **Thí nghiệm kiểm chứng + tiêu chí bác bỏ:** Docling ≥90% chứa **và** ≥80% in-window → giữ Docling, vấn đề chỉ còn tốc độ (đã có số từ P-probe). Docling <70% → P4 đổi PyMuPDF (có dữ kiện, không đoán). Chunk-cuối thắng ≥5 câu → P4 phải đảo thứ tự context.
- **Kỳ vọng:** tránh chọn sai parser/thứ tự = ±1–2 câu | **Tự tin: vừa** | **Chi phí: S** (không LLM, không slot)

## 4. Bình chọn
VOTES: M7-A, M4-A, M6-B

---

_Câu trả lời các vòng trước: thư mục r1/_

_Nguồn tra cứu: 175 link từ 0 lượt search — xem sources.md_
