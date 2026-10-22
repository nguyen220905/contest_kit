# M7 = qwen (qwen/qwen3.8-max-0902) — vai: competition_veteran — vòng 2

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
