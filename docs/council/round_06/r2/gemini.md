# M6 = gemini (google/gemini-4-argon) — vai: pragmatist — vòng 2

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
