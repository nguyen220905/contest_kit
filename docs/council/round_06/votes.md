# Bảng xếp hạng đồng thuận — round_06

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
