Hội đồng model — round_01 — 2026-10-15 01:58 — 2 vòng

> Các model KHÔNG thấy code, chỉ thấy brief. Đồng thuận là tín hiệu, không phải bằng chứng — kiểm chứng với code/log thật trước khi đưa vào PLAN.

## Thành viên
- M1 = minimax (xiaomi/mimo-v2.6-flash) — vai: data_detective
- M2 = glm (z-ai/glm-5.3) — vai: simplifier
- M3 = gemini (google/gemini-4-argon) — vai: pragmatist
- M4 = openai (openai/gpt-6-luna) — vai: plan_critic
- M5 = qwen (qwen/qwen3.8-max-0902) — vai: competition_veteran
- M6 = kimi (moonshotai/kimi-k3) — vai: alternative
- M7 = deepseek (deepseek-v4-pro) — vai: root_cause
- M8 = grok (x-ai/grok-4.7) — vai: red_team

## Bảng xếp hạng đồng thuận (phiếu vòng 2 cho đề xuất vòng 1)

| Hạng | ID | Đề xuất | Của | Điểm | Người bầu |
|---|---|---|---|---|---|
| 1 | M8-B | Lượt nộp đầu = pipeline nhanh có lưới an toàn, không phải baseline BTC | grok | 3 | M7 |
| 2 | M8-A | Khóa hợp đồng câu trả lời (atomic + bác tiền đề sai) | grok | 2 | M7 |
| 3 | M8-C | Ngưỡng nhiễu trước mọi ablation tốn lượt nộp | grok | 1 | M7 |

## Chi tiết top 3 đề xuất được bầu

**M8-B** (grok)

### Đề xuất B: Lượt nộp đầu = pipeline nhanh có lưới an toàn, không phải baseline BTC
- Giả thuyết: Khoảng cách thật là 57 → 68 và 2080 s → 605 s, tức parse Docling toàn corpus + nhồi ~10k token không phải hướng top. Bundle BGE-M3 (repo ~4.59 GB) hoặc bge-reranker-v2-m3 (~2.27 GB) làm submission **Failed** vì zip giải nén ≤ 1 GB. Một PDF ném exception làm cả run 0 điểm vì không thấy output.
- Dựa trên bằng chứng nào trong brief: Plan mục 3 nộp nguyên 01/02; mục 4/6 tính hybrid + reranker BGE. Brief đã ghi Docling chậm, ctx lớn OOM, wall-clock 1 h tính cả parse. LB xác nhận RAG chính thức ~35 phút / 57.14; top ~10 phút.
- Cách làm cụ thể: Không nộp 01/02. Zip chỉ MiniLM (~90 MB) + index rỗng (index phải build lúc chạy trên `corpus_dir` thật). Parse PyMuPDF đa tiến trình, `try/except` từng PDF và từng câu (lỗi → `""` hoặc fallback zero-shot, luôn `return str`). BM25 (`rank-bm25` có sẵn) + cosine MiniLM, top-5, cắt theo token bằng tokenizer Olmo, chứng cứ đầu+cuối. Ước lượng: đếm PDF local giả lập; nếu parse+embed dự kiến > 25 phút cho corpus cỡ baseline, cắt Docling hoàn toàn. `du -sh` bản unzipped trước khi nộp; > 900 MB thì không nộp.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Local: 30 PDF nhiễu + 5 PDF bảng, đo wall-clock parse+embed trên CPU; nhân với số PDF BTC (~36 phút Docling là cận trên đã biết). Bác “cần BGE/reranker trong zip” nếu `du` > 1 GB. Bác “Docling là bước bắt buộc của lượt 1” nếu pipeline PyMuPDF+BM25+MiniLM xong index giả lập trong < 15 phút và recall top-5 trên 10 câu DONES có số ≥ recall chunk Docling 800 ký tự. Lượt LB duy nhất đáng tiêu: pipeline này. Nếu LB ≤ 57.14 thì lỗi là chất lượng chunk/bảng, không phải tốc độ.
- Kỳ vọng: tránh 0 điểm vì Failed; nếu đứng được phía trên 57 thì +0 đến +11 so với RAG chính thức (phần trên 68 chưa có cơ sở) | Tự tin: cao với “đừng nộp baseline / đừng bundle BGE”; vừa với +điểm | Chi phí: S

**M8-A** (grok)

### Đề xuất A: Khóa hợp đồng câu trả lời (atomic + bác tiền đề sai)
- Giả thuyết: Judge nhị phân so với đáp án ngắn, đúng đơn vị/số; câu trả lời dài kiểu gold mẫu (thêm số thứ hai, nhầm IFMIF với IFMIF-DONES, không bác tiền đề) bị đánh sai dù retrieval đúng. Prompt starter (“concisely… based on your knowledge”) còn cho model ghi đè số trong PDF bằng tri thức tham số.
- Dựa trên bằng chứng nào trong brief: Brief coi gold 80–2700 ký tự là tín hiệu độ dài cho judge; trang chính thức nói reference cố ý ngắn để chấm tự động. Trong `sample_qa.json`, câu 7 hỏi nhiệt độ bẫy nitơ **on-line** nhưng gold nói không có bẫy on-line; câu 9 hỏi thể tích nhà chính IFMIF-DONES nhưng gold dùng khối chữ nhật IFMIF 137×111×40.5 m, trong khi thiết kế DONES công bố là mặt bằng chữ L ~160×75×32 m. LB đã có bài dưới zero-shot — context nhiễu/số thừa có thể làm hại.
- Cách làm cụ thể: Với 25 câu mẫu, viết tay 1 đáp án atomic (1 sự kiện, 1 đơn vị, 1 số; nếu tiền đề sai thì câu đầu là phủ định). Không bắt chước LaTeX/đoạn giải thích của gold. Prompt bắt buộc: chỉ dùng số có trong fragment; nếu câu hỏi giả định sai thì nói rõ không có; cấm đưa phương án số thứ hai. Cấm dùng LLM khác Olmo để **sinh** câu trả lời cuối (luật Codabench: model duy nhất được load qua `model_id`); embedder nhỏ như starter thì BTC đã bundle.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Hai judge local độc lập (không phải model sẽ nộp) chấm 25 cặp (atomic vs bản dài kiểu gold) so với gold mẫu **và** so với checklist sự kiện tự viết. Bác giả thuyết nếu bản dài thắng atomic ≥ 5 câu và không câu nào bị trừ vì số phụ/tiền đề. Nếu atomic thắng hoặc hòa trên ≥ 20/25, dừng tối ưu “trả lời dài như mẫu”.
- Kỳ vọng: tránh trần dưới RAG 57 khi retrieval đã đúng; trên proxy local +5 đến +15 điểm % so với bản dài có số thừa | Tự tin: vừa | Chi phí: S

**M8-C** (grok)

### Đề xuất C: Ngưỡng nhiễu trước mọi ablation tốn lượt nộp
- Giả thuyết: Public ≈ 50 câu, accuracy cân bằng 2 domain nên 1 câu domain nhỏ đổi ~2 điểm % (bậc thang LB 59.05 / 61.43 / 63.33). SE domain nếu ~20–25 câu và p≈0.6 là ~10 điểm; điểm cân bằng nhiễu vài điểm, cộng thêm judge ẩn. Ablation < ~1 câu không phân biệt được cải tiến với nhiễu, và bài public cao nhất bị chấm lại trên 100% — overfit 50 câu public bị phạt đúng lúc private.
- Dựa trên bằng chứng nào trong brief: Brief ước 1 câu ≈ 2–4 điểm trong 1 domain và định nộp ablation retrieval/prompt. Terms: private tự động lấy bài public điểm cao nhất, không nộp mới. Trang evaluation: hòa thì thời gian suy luận thấp hơn thắng; judge nhị phân, rubric không công bố.
- Cách làm cụ thể: Trên 25 câu mẫu, chấm mỗi câu 2 lần bằng 2 prompt judge khác nhau (đúng/sai, không thang điểm). Chỉ coi một thay đổi là “đáng nộp” nếu trên tập dev giữ ra (không phải 25 câu đã soi) nó lật ≥ 3 câu **và** không làm chậm quá ngân sách còn lại sau parse. Không nộp self-consistency nhiều mẫu: 50 câu × vài rollout × 32B dễ vỡ 1 h; tie-break cũng phạt bài chậm.
- Thí nghiệm kiểm chứng + tiêu chí bác bỏ: Tỷ lệ hai judge local bất đồng trên 25 câu. Nếu bất đồng ≥ 3/25, mọi Δ LB < 4 điểm bị coi là nhiễu — bác kế hoạch “mỗi ngày một ablation nhỏ”. Nếu bất đồng = 0 và một thay đổi local lật ≥ 3 câu factual có số (không phải câu diễn giải), mới được tốn 1 lượt.
- Kỳ vọng: không cộng điểm trực tiếp; tránh mất ~10 lượt và tránh chọn nhầm bài overfit làm bài private | Tự tin: cao | Chi phí: S

## Câu trả lời vòng cuối (vòng 2)

---

# M1 = minimax (xiaomi/mimo-v2.6-flash) — vai: data_detective

❌ LỖI: HTTP 403: {"error":{"message":"Key limit exceeded (daily limit). Manage it using https://openrouter.ai/workspaces/default/keys/f96c6842de0aaa1eef19b48ef88399fa21ed9e91f18163c6568260994375e48d $8.76 of $7.00 used; resets 2026-10-15T00:00:00.000Z.","code":403,"metadata":{"limit_source":"openrouter_key_limit","remedy_hint":"Raise or remove this API key's usage limit at https://openrouter.ai/settings/keys, or u

---

# M2 = glm (z-ai/glm-5.3) — vai: simplifier

❌ LỖI: HTTP 403: {"error":{"message":"Key limit exceeded (daily limit). Manage it using https://openrouter.ai/workspaces/default/keys/f96c6842de0aaa1eef19b48ef88399fa21ed9e91f18163c6568260994375e48d $8.76 of $7.00 used; resets 2026-10-15T00:00:00.000Z.","code":403,"metadata":{"limit_source":"openrouter_key_limit","remedy_hint":"Raise or remove this API key's usage limit at https://openrouter.ai/settings/keys, or u

---

# M3 = gemini (google/gemini-4-argon) — vai: pragmatist

❌ LỖI: HTTP 403: {"error":{"message":"Key limit exceeded (daily limit). Manage it using https://openrouter.ai/workspaces/default/keys/f96c6842de0aaa1eef19b48ef88399fa21ed9e91f18163c6568260994375e48d $8.76 of $7.00 used; resets 2026-10-15T00:00:00.000Z.","code":403,"metadata":{"limit_source":"openrouter_key_limit","remedy_hint":"Raise or remove this API key's usage limit at https://openrouter.ai/settings/keys, or u

---

# M4 = openai (openai/gpt-6-luna) — vai: plan_critic

❌ LỖI: HTTP 403: {"error":{"message":"Key limit exceeded (daily limit). Manage it using https://openrouter.ai/workspaces/default/keys/f96c6842de0aaa1eef19b48ef88399fa21ed9e91f18163c6568260994375e48d $8.76 of $7.00 used; resets 2026-10-15T00:00:00.000Z.","code":403,"metadata":{"limit_source":"openrouter_key_limit","remedy_hint":"Raise or remove this API key's usage limit at https://openrouter.ai/settings/keys, or u

---

# M5 = qwen (qwen/qwen3.8-max-0902) — vai: competition_veteran

❌ LỖI: HTTP 403: {"error":{"message":"Key limit exceeded (daily limit). Manage it using https://openrouter.ai/workspaces/default/keys/f96c6842de0aaa1eef19b48ef88399fa21ed9e91f18163c6568260994375e48d $8.76 of $7.00 used; resets 2026-10-15T00:00:00.000Z.","code":403,"metadata":{"limit_source":"openrouter_key_limit","remedy_hint":"Raise or remove this API key's usage limit at https://openrouter.ai/settings/keys, or u

---

# M6 = kimi (moonshotai/kimi-k3) — vai: alternative

❌ LỖI: HTTP 403: {"error":{"message":"Key limit exceeded (daily limit). Manage it using https://openrouter.ai/workspaces/default/keys/f96c6842de0aaa1eef19b48ef88399fa21ed9e91f18163c6568260994375e48d $8.76 of $7.00 used; resets 2026-10-15T00:00:00.000Z.","code":403,"metadata":{"limit_source":"openrouter_key_limit","remedy_hint":"Raise or remove this API key's usage limit at https://openrouter.ai/settings/keys, or u

---

# M7 = deepseek (deepseek-v4-pro) — vai: root_cause

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

---

# M8 = grok (x-ai/grok-4.7) — vai: red_team

❌ LỖI: HTTP 403: {"error":{"message":"Key limit exceeded (daily limit). Manage it using https://openrouter.ai/workspaces/default/keys/f96c6842de0aaa1eef19b48ef88399fa21ed9e91f18163c6568260994375e48d $8.76 of $7.00 used; resets 2026-10-15T00:00:00.000Z.","code":403,"metadata":{"limit_source":"openrouter_key_limit","remedy_hint":"Raise or remove this API key's usage limit at https://openrouter.ai/settings/keys, or u

---

_Câu trả lời các vòng trước: thư mục r1/_

_Nguồn tra cứu: 171 link từ 0 lượt search — xem sources.md_
