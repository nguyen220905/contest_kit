# M8 = grok (x-ai/grok-4.7) — vai: red_team — vòng 1

## 1. Chẩn đoán nhanh
Mốc BTC đã có trên public LB: Zero Shot **29.05** (733 s), Simple RAG **57.14** (2080 s). Top hiện **68.1** ở **605 s**; có bài RAG chậm hơn zero-shot và chỉ được **19.52**. Nộp `01_zero_shot`/`02_rag` không dạy thêm gì, chỉ đốt 2 lượt/ngày. Điểm đang “dừng” trước khi chạy vì plan nhắm sai hợp đồng chấm: judge nhị phân, accuracy cân bằng 2 domain, tie-break là thời gian; gold mẫu dài/LaTeX không phải target an toàn, và model retrieval trong plan (BGE-M3, reranker-v2-m3) không nhét vừa zip 1 GB. Chênh ~2 điểm trên LB ≈ 1 câu — không phải tín hiệu phương pháp.

## 2. Đề xuất (tối đa 3, xếp theo kỳ vọng/chi phí)
### Đề xuất A: Khóa hợp đồng câu trả lời (atomic + bác tiền đề sai)
- Giả thuyết: Judge nhị phân so với đáp án ngắn, đúng đơn vị/số; câu trả lời dài kiểu gold mẫu (thêm số thứ hai, nhầm IFMIF với IFMIF-DONES, không bác tiền đề) bị đánh sai dù retrieval đúng. Prompt starter (“concisely… based on your knowledge”) còn cho model ghi đè số trong PDF bằng tri thức tham số.
- Dựa trên bằng chứng nào trong brief: Brief coi gold 80–2700 ký tự là tín hiệu độ dài cho judge; trang chính thức nói reference cố ý ngắn để chấm tự động. Trong `sample_qa.json`, câu 7 hỏi nhiệt độ bẫy nitơ **on-line** nhưng gold nói không có bẫy on-line; câu 9 hỏi thể tích nhà chính IFMIF-DONES nhưng gold dùng khối chữ nhật IFMIF 137×111×40.5 m, trong khi thiết kế DONES công bố là mặt bằng chữ L ~160×75×32 m. LB đã có bài dưới zero-shot — context nhiễu/số thừa có thể làm hại.
- Cách làm cụ thể: Với 25 câu mẫu, viết tay 1 đáp án atomic (1 sự kiện, 1 đơn vị, 1 số; nếu tiền đề sai thì câu đầu là phủ định). Không bắt chước LaTeX/đoạn giải thích của gold. Prompt bắt buộc: chỉ dùng số có trong fragment; nếu câu hỏi giả định sai thì nói rõ không có; cấm đưa phương án số thứ hai. Cấm dùng LLM khác Olmo để **sinh** câu trả lời cuối (luật Codabench: model duy nhất được load qua `model_id`); embedder nhỏ như starter thì BTC đã bundle.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Hai judge local độc lập (không phải model sẽ nộp) chấm 25 cặp (atomic vs bản dài kiểu gold) so với gold mẫu **và** so với checklist sự kiện tự viết. Bác giả thuyết nếu bản dài thắng atomic ≥ 5 câu và không câu nào bị trừ vì số phụ/tiền đề. Nếu atomic thắng hoặc hòa trên ≥ 20/25, dừng tối ưu “trả lời dài như mẫu”.
- Kỳ vọng: tránh trần dưới RAG 57 khi retrieval đã đúng; trên proxy local +5 đến +15 điểm % so với bản dài có số thừa | Tự tin: vừa | Chi phí: S

### Đề xuất B: Lượt nộp đầu = pipeline nhanh có lưới an toàn, không phải baseline BTC
- Giả thuyết: Khoảng cách thật là 57 → 68 và 2080 s → 605 s, tức parse Docling toàn corpus + nhồi ~10k token không phải hướng top. Bundle BGE-M3 (repo ~4.59 GB) hoặc bge-reranker-v2-m3 (~2.27 GB) làm submission **Failed** vì zip giải nén ≤ 1 GB. Một PDF ném exception làm cả run 0 điểm vì không thấy output.
- Dựa trên bằng chứng nào trong brief: Plan mục 3 nộp nguyên 01/02; mục 4/6 tính hybrid + reranker BGE. Brief đã ghi Docling chậm, ctx lớn OOM, wall-clock 1 h tính cả parse. LB xác nhận RAG chính thức ~35 phút / 57.14; top ~10 phút.
- Cách làm cụ thể: Không nộp 01/02. Zip chỉ MiniLM (~90 MB) + index rỗng (index phải build lúc chạy trên `corpus_dir` thật). Parse PyMuPDF đa tiến trình, `try/except` từng PDF và từng câu (lỗi → `""` hoặc fallback zero-shot, luôn `return str`). BM25 (`rank-bm25` có sẵn) + cosine MiniLM, top-5, cắt theo token bằng tokenizer Olmo, chứng cứ đầu+cuối. Ước lượng: đếm PDF local giả lập; nếu parse+embed dự kiến > 25 phút cho corpus cỡ baseline, cắt Docling hoàn toàn. `du -sh` bản unzipped trước khi nộp; > 900 MB thì không nộp.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Local: 30 PDF nhiễu + 5 PDF bảng, đo wall-clock parse+embed trên CPU; nhân với số PDF BTC (~36 phút Docling là cận trên đã biết). Bác “cần BGE/reranker trong zip” nếu `du` > 1 GB. Bác “Docling là bước bắt buộc của lượt 1” nếu pipeline PyMuPDF+BM25+MiniLM xong index giả lập trong < 15 phút và recall top-5 trên 10 câu DONES có số ≥ recall chunk Docling 800 ký tự. Lượt LB duy nhất đáng tiêu: pipeline này. Nếu LB ≤ 57.14 thì lỗi là chất lượng chunk/bảng, không phải tốc độ.
- Kỳ vọng: tránh 0 điểm vì Failed; nếu đứng được phía trên 57 thì +0 đến +11 so với RAG chính thức (phần trên 68 chưa có cơ sở) | Tự tin: cao với “đừng nộp baseline / đừng bundle BGE”; vừa với +điểm | Chi phí: S

### Đề xuất C: Ngưỡng nhiễu trước mọi ablation tốn lượt nộp
- Giả thuyết: Public ≈ 50 câu, accuracy cân bằng 2 domain nên 1 câu domain nhỏ đổi ~2 điểm % (bậc thang LB 59.05 / 61.43 / 63.33). SE domain nếu ~20–25 câu và p≈0.6 là ~10 điểm; điểm cân bằng nhiễu vài điểm, cộng thêm judge ẩn. Ablation < ~1 câu không phân biệt được cải tiến với nhiễu, và bài public cao nhất bị chấm lại trên 100% — overfit 50 câu public bị phạt đúng lúc private.
- Dựa trên bằng chứng nào trong brief: Brief ước 1 câu ≈ 2–4 điểm trong 1 domain và định nộp ablation retrieval/prompt. Terms: private tự động lấy bài public điểm cao nhất, không nộp mới. Trang evaluation: hòa thì thời gian suy luận thấp hơn thắng; judge nhị phân, rubric không công bố.
- Cách làm cụ thể: Trên 25 câu mẫu, chấm mỗi câu 2 lần bằng 2 prompt judge khác nhau (đúng/sai, không thang điểm). Chỉ coi một thay đổi là “đáng nộp” nếu trên tập dev giữ ra (không phải 25 câu đã soi) nó lật ≥ 3 câu **và** không làm chậm quá ngân sách còn lại sau parse. Không nộp self-consistency nhiều mẫu: 50 câu × vài rollout × 32B dễ vỡ 1 h; tie-break cũng phạt bài chậm.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Tỷ lệ hai judge local bất đồng trên 25 câu. Nếu bất đồng ≥ 3/25, mọi Δ LB < 4 điểm bị coi là nhiễu — bác kế hoạch “mỗi ngày một ablation nhỏ”. Nếu bất đồng = 0 và một thay đổi local lật ≥ 3 câu factual có số (không phải câu diễn giải), mới được tốn 1 lượt.
- Kỳ vọng: không cộng điểm trực tiếp; tránh mất ~10 lượt và tránh chọn nhầm bài overfit làm bài private | Tự tin: cao | Chi phí: S

## 3. Điểm đáng ngờ / thông tin còn thiếu trong brief
- Brief nói chưa có điểm LB; public đã có mốc BTC và top 68.1 / 605 s. Cột “Fact Sheet Answers” trên UI Codabench giống header mặc định của nền tảng (thấy ở contest khác), không phải task thứ hai — vẫn nên nhìn UI một lần trước khi nộp.
- Luật “only model” trên Codabench mâu thuẫn với starter (MiniLM + Docling) và với tin Discord trong brief. Mình không mở được Discord nên không xác nhận câu 15/10. An toàn: model phụ chỉ embed/rerank nhỏ đã có trong môi trường; không dùng LLM khác để sinh câu trả lời. Câu nên hỏi BTC: “Does ‘the only model you may load’ forbid any `AutoModelForCausalLM` other than `model_id`, including a draft model for speculative decoding and a small rewriter, or only forbid replacing Olmo as the generator of the returned string?”
- Gold mẫu không phải đáp án ẩn và có ít nhất một số khả năng sai domain (nhà chính). Đừng hiệu chỉnh judge local cho khớp nguyên văn gold.
- Công thức ảnh trên trang evaluation không trích được text; giả định \(A=100\times\frac{1}{2}(\mathrm{acc}_D+\mathrm{acc}_P)\). Nếu sai, bậc ~2 điểm cần đo lại từ cột LB.
- Không biết số PDF, tỷ lệ scan, và bao nhiêu câu public thuộc DONES. Ảnh scan + Docling `no OCR` = chunk rỗng, im lặng, không crash.

## 4. Nguồn đã tra cứu
- Evaluation (judge nhị phân, accuracy cân bằng, tie-break thời gian, top 20 vào Phase 2) — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/evaluation/ — judge ẩn, binary, domain nhỏ tính bằng domain lớn; private = full 100 câu.
- Overview (reference cố ý ngắn) — https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/ — “Reference answers are deliberately concise…”.
- Codabench API competition 17954 (terms + how to participate) — https://www.codabench.org/api/competitions/17954/ — chỉ được load `model_id`; private tự động chấm lại bài public điểm cao nhất; zip 1 GB; 1 h public; không thấy output.
- Leaderboard public (điểm và thời gian) — https://www.codabench.org/competitions/17954/ — Zero Shot 29.05/733 s, Simple RAG 57.14/2080 s, top 68.1/605 s, có bài 19.52.
- sample_qa.json — https://raw.githubusercontent.com/jucls/IEEE-CIS-DSQA-Starter-Kit/main/sample/sample_qa.json — câu 7 tiền đề sai; câu 9 gold dùng kích thước IFMIF 137×111×40.5.
- IFMIF-DONES buildings (Luque et al.) — https://www.researchgate.net/publication/389721709_IFMIF-DONES'_Buildings_and_Plant_Services — nhà chính DONES chữ L ~160×75×32 m, khác khối chữ nhật giai đoạn IFMIF.
- BAAI/bge-m3 — https://huggingface.co/BAAI/bge-m3/tree/main — repo ~4.59 GB, vượt trần zip.
- BGE-Reranker-v2 — https://bge-model.com/bge/bge_reranker_v2.html — bge-reranker-v2-m3 ~2.27 GB.
- Starter README — https://raw.githubusercontent.com/jucls/IEEE-CIS-DSQA-Starter-Kit/main/README.md — 02_rag ~36 phút / 50 câu; 25 câu mẫu không thuộc LB.
