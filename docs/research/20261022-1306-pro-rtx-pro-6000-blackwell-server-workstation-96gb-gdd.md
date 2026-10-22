# RTX PRO 6000 Blackwell Server/Workstation 96GB GDDR7 vs H100 80GB SXM vs A100 40GB SXM4 for single-stream LLM inference with HuggingFace transformers bf16 (no vLLM, sdpa, batch size 1) on a 32B dense model (~64GB weights): memory bandwidth (TB/s), dense BF16 tensor TFLOPS (without sparsity), measured decode tokens/s and prefill throughput if available; does PyTorch 2.8 cu128 support sm_120 well for bf16 sdpa; how does splitting a 32B model across 2x A100 40GB with device_map=auto (pipeline, sequential layers) affect batch-1 decode speed vs single GPU; Docling layout model speed on RTX PRO 6000 / RTX 5090 vs datacenter GPUs. Give numbers with sources.

> Perplexity pro · 2026-10-22 13:06. Kiểm chứng link quan trọng trước khi dựa vào.

## Tóm tắt phát hiện

- Với **single-stream, batch size 1, BF16 dense inference**, yếu tố chi phối decode thường là **băng thông bộ nhớ**, không phải peak tensor TFLOPS. Về thông số lý thuyết, RTX PRO 6000 Workstation có **96 GB GDDR7, 1.792 TB/s**; H100 SXM có **80 GB HBM3, 3.35 TB/s**; A100 40 GB SXM4 có **1.555 TB/s**.[1][2][3]
- Peak **dense BF16 Tensor Core**: RTX PRO 6000 được NVIDIA công bố ở mức AI/FP4 tổng quát, không cung cấp rõ trong trang sản phẩm con số BF16 dense tương đương để so trực tiếp; H100 SXM đạt **989 TFLOPS BF16 dense** hoặc **1,979 TFLOPS với sparsity**; A100 40 GB SXM đạt **312 TFLOPS dense** hoặc **624 TFLOPS với sparsity**.[2][3]
- Với model 32B khoảng **64 GB weights**, RTX PRO 6000 là lựa chọn duy nhất trong ba loại có thể chứa toàn bộ model trên một GPU với dư địa đáng kể cho KV cache; A100 40 GB không đủ, còn H100 80 GB vừa đủ về trọng lượng weights nhưng cần kiểm tra overhead, embedding/output layers và KV cache.
- Chưa tìm thấy benchmark công khai đáng tin cậy đo đúng cấu hình người hỏi: **Transformers thuần, BF16, SDPA, batch 1, single-stream, cùng một model 32B, decode và prefill trên cả ba GPU**. Các số token/s từ llama.cpp, quantization, vLLM/TensorRT-LLM hoặc roofline không nên dùng như số đo trực tiếp cho cấu hình này.[4][5][6]
- **PyTorch 2.8.0 + cu128** thuộc nhóm build hỗ trợ Blackwell/sm_120; PyTorch xác nhận các bản stable 2.7 và 2.8 build với CUDA 12.8+ hỗ trợ Blackwell.[7][8] Tuy nhiên, “nhận diện được sm_120” không đồng nghĩa mọi kernel SDPA, Triton, cuBLAS và fused attention đều đạt hiệu năng tối ưu; cần kiểm tra backend thực tế trên máy.
- `device_map="auto"` của Accelerate là **model parallelism tuần tự**, không phải tensor parallelism: trong một forward batch-1, chỉ một GPU hoạt động tại một thời điểm và activation phải chuyển giữa GPU.[9] Vì vậy 2×A100 40GB chủ yếu giúp model chạy được, không kỳ vọng tốc độ decode gần gấp đôi; với 32B, một phép ước lượng bandwidth-only cho mỗi GPU đọc khoảng 32 GB weights mỗi token cho ra trần lý thuyết khoảng **48.6 tok/s**, trước overhead truyền activation, kernel và KV cache.
- Docling có benchmark chính thức cho **RTX 5090** và **L40S**, nhưng chưa thấy số chính thức cho RTX PRO 6000, H100 hoặc A100. Trong pipeline Standard không OCR, Docling ghi nhận RTX 5090 **7.9 pages/s CPU-only hoặc 1.5 pages/s GPU** trên PDF test; với pipeline VLM GraniteDocling, RTX 5090 đạt **3.8 pages/s** trên PDF và **3.6–4.5 pages/s** trên bộ ViDoRe.[10] Con số “CPU-only” trong bảng này là điều cần kiểm tra lại trước khi diễn giải.

## Kỹ thuật / ý tưởng đáng thử (mỗi ý: là gì · bằng chứng · điều kiện áp dụng · nguồn)

- **Ưu tiên RTX PRO 6000 cho bản BF16 nguyên trạng trên một GPU** · *Là gì:* đặt toàn bộ model 32B, weights BF16 và KV cache trên một GPU 96 GB, dùng Transformers `attn_implementation="sdpa"`, `torch_dtype=torch.bfloat16`, batch 1. *Bằng chứng:* NVIDIA công bố RTX PRO 6000 Workstation có 96 GB GDDR7 ECC và 1.792 TB/s; dung lượng này phù hợp hơn đáng kể với model khoảng 64 GB weights.[1] *Điều kiện áp dụng:* model thực sự có khoảng 32B tham số và weights BF16 khoảng 64 GB; phải chừa bộ nhớ cho CUDA context, KV cache, temporary buffers và allocator fragmentation. *Nguồn:* NVIDIA RTX PRO 6000 specifications.[1]

- **Đo roofline decode trước khi tối ưu kernel** · *Là gì:* dùng xấp xỉ \( \text{tok/s} \leq \text{memory bandwidth}/\text{bytes read per token} \). Nếu mỗi token phải đọc gần như toàn bộ 64 GB weights, trần lý thuyết là khoảng **28 tok/s** trên 1.792 TB/s của RTX PRO 6000, **52 tok/s** trên 3.35 TB/s của H100, và **24 tok/s** trên 1.555 TB/s của A100 nếu toàn bộ 64 GB nằm trên một A100; đây là upper bound lý tưởng, không phải benchmark. *Bằng chứng:* các băng thông được NVIDIA công bố lần lượt là 1.792, 3.35 và 1.555 TB/s.[1][2][3] *Điều kiện áp dụng:* phù hợp để sanity-check decode batch-1; không dùng để dự đoán prefill, vì prefill tận dụng GEMM và tensor cores tốt hơn. *Nguồn:* thông số NVIDIA; công thức là suy luận từ bandwidth và footprint, không phải số đo.[1][2][3]

- **Tách riêng benchmark prefill và decode** · *Là gì:* đo TTFT/prefill với prompt cố định, sau đó đo decode sau khi KV cache đã ổn định; báo cáo prompt tokens/s và generated tokens/s riêng. *Bằng chứng:* decode batch-1 của dense LLM thường chịu giới hạn đọc weights mỗi bước, trong khi prefill có nhiều phép tính song song hơn; các benchmark “prompt processing” và “generation” trên RTX 5090 được công bố tách biệt, nhưng dùng runtime/định dạng khác với cấu hình yêu cầu.[4][5] *Điều kiện áp dụng:* phải cố định model revision, context length, `use_cache=True`, số token sinh, temperature, tokenizer, CUDA graphs và warm-up. *Nguồn:* các benchmark RTX 5090 chỉ nên dùng làm tham khảo phương pháp, không làm số tương đương trực tiếp.[4][5]

- **So sánh H100 và RTX PRO 6000 theo hai trường hợp: model fit một GPU và model không fit** · *Là gì:* chạy 32B BF16 trên RTX PRO 6000; chạy H100 SXM nếu toàn bộ weights và KV cache fit; so sánh thêm phiên bản offload/sharding. *Bằng chứng:* H100 có bandwidth 3.35 TB/s và BF16 dense 989 TFLOPS, vượt xa RTX PRO 6000 về băng thông lý thuyết; RTX PRO 6000 lại có 96 GB capacity so với H100 80 GB.[2] *Điều kiện áp dụng:* H100 có lợi hơn nếu workload thực sự memory-bandwidth-bound và model fit; RTX PRO 6000 có lợi về capacity, tránh sharding hoặc CPU offload. *Nguồn:* NVIDIA H100 datasheet và NVIDIA RTX PRO 6000 specifications.[1][2]

- **Dùng PyTorch 2.8.0+cu128 nhưng xác minh kernel SDPA thực tế** · *Là gì:* kiểm tra `torch.cuda.get_arch_list()`, `torch.cuda.get_device_capability()`, backend SDPA và profiler; chạy cả `flash`, `mem_efficient` và `math` nếu API/model cho phép. *Bằng chứng:* PyTorch cho biết stable 2.7/2.8 build với CUDA 12.8+ hỗ trợ Blackwell/sm_120.[7][8] Release 2.8 cũng liệt kê các thay đổi trong SDPA, nhưng thông tin này không chứng minh mọi hình dạng attention của model đều dùng kernel tối ưu.[11] *Điều kiện áp dụng:* cần NVIDIA driver tương thích, wheel cu128 thật sự, CUDA runtime đúng và phiên bản Transformers không ép fallback. *Nguồn:* PyTorch forum/release information.[7][8][11]

- **Kiểm tra bằng một phép thử SDPA tối thiểu trước khi chạy model 32B** · *Là gì:* chạy attention BF16 với các shape giống model, kiểm tra output, kernel profiler và tốc độ; sau đó benchmark end-to-end. *Bằng chứng:* PyTorch có thể liệt kê kiến trúc CUDA được compile bằng `torch.cuda.get_arch_list()`, và các báo cáo hỗ trợ Blackwell nhấn mạnh cần dùng build CUDA 12.8+.[8] *Điều kiện áp dụng:* đặc biệt cần trên RTX PRO 6000/RTX 5090; không suy ra từ việc `torch.randn(...).cuda()` chạy thành công. *Nguồn:* PyTorch discussion về sm_120.[8]

- **Dùng 2×A100 40GB với `device_map="auto"` chỉ như phương án “fit model”, không phải tăng tốc mặc định** · *Là gì:* chia các block transformer liên tiếp lên hai A100, đo với một GPU không thể chạy vì thiếu VRAM và so với cấu hình offload; không so với một A100 đơn nếu model không fit. *Bằng chứng:* Accelerate nói rõ multi-GPU model parallelism có thể chỉ để một GPU hoạt động tại một thời điểm và GPU sau phải chờ GPU trước truyền output.[9] *Điều kiện áp dụng:* phù hợp khi mục tiêu là chạy model BF16 không quantize; không phù hợp nếu mục tiêu là throughput batch-1 tối đa. *Nguồn:* tài liệu Hugging Face Accelerate.[9]

- **Thử pipeline parallelism thực sự nếu có nhiều request hoặc chấp nhận batching** · *Là gì:* dùng pipeline engine có micro-batches để các stage hoạt động đồng thời, thay vì gọi forward tuần tự từ `device_map`. *Bằng chứng:* Accelerate phân biệt rõ pipeline parallelism với model parallelism tuần tự và nói pipeline có thể sử dụng các GPU cùng lúc.[9] *Điều kiện áp dụng:* hiệu quả hơn khi batch hoặc số request đủ lớn; batch-1 decode với một chuỗi duy nhất vẫn dễ chịu bubble/communication overhead. *Nguồn:* Accelerate Big Model Inference và distributed inference documentation.[9][12]

- **Đo chi phí truyền activation giữa hai A100** · *Là gì:* profile các lần chuyển tensor giữa `cuda:0` và `cuda:1`, đặc biệt giữa từng nhóm layer; kiểm tra topology NVLink/NVSwitch thay vì chỉ nhìn tổng số GPU. *Bằng chứng:* A100 SXM có NVLink được NVIDIA công bố ở mức 600 GB/s trong hệ thống HGX; Accelerate vẫn cảnh báo device-map inference hoạt động tuần tự.[3][9] *Điều kiện áp dụng:* cần biết server có NVLink/NVSwitch hay chỉ PCIe; placement layer không cân bằng có thể làm một GPU rỗi lâu. *Nguồn:* A100 datasheet và Accelerate.[3][9]

- **Benchmark Docling theo page batch size, không chỉ theo GPU model** · *Là gì:* chạy `AcceleratorDevice.CUDA`, thử `layout_batch_size` và `page_batch_size` tăng dần, ghi riêng layout detector, OCR, table extraction và end-to-end pages/s. *Bằng chứng:* tài liệu Docling nói `page_batch_size` cao hơn kích hoạt GPU batch inference, đặc biệt cho layout detection; mặc định layout batch size là 4 trong ví dụ.[10] *Điều kiện áp dụng:* nếu mục tiêu là latency một trang, batch lớn có thể làm latency tăng; nếu mục tiêu throughput tài liệu, batch/concurrency thường quan trọng hơn peak BF16 TFLOPS. *Nguồn:* Docling GPU guide.[10]

- **Dùng benchmark Docling chính thức hiện có làm baseline, nhưng không nội suy tuyến tính sang RTX PRO 6000** · *Là gì:* lấy RTX 5090 làm baseline phần mềm rồi đo lại trên RTX PRO 6000, H100 và A100 bằng đúng commit, PDF, driver, CUDA, batch và pipeline. *Bằng chứng:* Docling công bố RTX 5090, RTX 5070 và L40S, nhưng không công bố RTX PRO 6000/H100/A100 trong bảng kết quả.[10] *Điều kiện áp dụng:* bắt buộc nếu cần số “layout model speed” đáng tin cậy; không dùng chênh lệch bandwidth để quy đổi pages/s vì pipeline còn CPU, I/O, OCR và post-processing. *Nguồn:* Docling GPU guide.[10]

## Bẫy và cảnh báo đã biết

- **Không trộn dense và sparse TFLOPS.** NVIDIA thường hiển thị BF16/FP16 dưới dạng `312 | 624` cho A100 và `989 | 1,979` cho H100, trong đó số cao hơn là có sparsity; yêu cầu hiện tại là **không sparsity**, nên dùng 312 TFLOPS cho A100 40GB SXM và 989 TFLOPS cho H100 SXM.[2][3]
- **Thông số RTX PRO 6000 không đồng nhất giữa các trang/phiên bản.** Trang Workstation NVIDIA hiện ghi 1.792 GB/s và 600 W, còn Server Edition được tìm thấy ghi 1.597 GB/s và khoảng 1 PFLOP BF16; Workstation và Server không nên gộp thành một SKU duy nhất.[1][13]
- **Không gọi số 5090 quantized là BF16 Transformers.** Các kết quả 32B khoảng 43–66 tok/s được tìm thấy cho RTX 5090 dùng GGUF/quantization hoặc runtime khác, không phải model BF16 64 GB chạy bằng Transformers thuần.[4][5][6]
- **Không coi roofline là throughput đo được.** Công thức bandwidth/weights bỏ qua hiệu suất đọc thực tế, cache, fused kernels, layernorm, attention, RoPE, KV-cache traffic, CUDA launch overhead và synchronization.
- **32B “~64 GB weights” chưa phải toàn bộ yêu cầu VRAM.** Cần cộng KV cache theo context length, temporary activations, logits, CUDA context, allocator reserve và các buffer của SDPA; H100 80 GB có thể không còn đủ nếu context dài hoặc model có embedding/lm-head lớn.
- **`device_map="auto"` không phải tensor parallelism.** Nó phân phối module/layer, nhưng không chia một GEMM qua hai GPU; do đó hai A100 không tự động biến thành một GPU có gấp đôi bandwidth hiệu dụng.[9]
- **Sequential placement có thể không dùng đều hai GPU.** Accelerate mô tả `device_map="auto"` sẽ lấp GPU trước rồi mới sang CPU/disk nếu cần; `sequential` còn có thể để GPU sau không dùng hết.[9]
- **Pipeline parallelism không mặc nhiên nhanh hơn batch-1.** Pipeline cần micro-batch để che latency; một single-stream token-by-token vẫn có dependency tuần tự giữa các stage.
- **SDPA “support” không đồng nghĩa FlashAttention tối ưu.** PyTorch 2.8 cu128 hỗ trợ kiến trúc sm_120, nhưng model có thể rơi về memory-efficient hoặc math backend tùy head dimension, mask, dropout, sequence length, dtype và layout.[7][8][11]
- **Cần phân biệt workstation RTX PRO 6000 và server RTX PRO 6000.** Chúng có thể khác bandwidth, power limit, cooling và clock; kết quả Workstation không đại diện trực tiếp cho Server Edition.[1][13]
- **Docling benchmark có điểm bất thường cần xác minh.** Bảng chính thức ghi RTX 5090 Standard Inline (no OCR) là 7.9 pages/s “CPU-only” và 1.5 pages/s GPU, trong khi mục tiêu thường là GPU nhanh hơn; nên kiểm tra commit, cờ accelerator và ý nghĩa cột trước khi dùng làm kết luận.[10]
- **Docling pages/s không phải layout-model images/s.** End-to-end pages/s còn bao gồm PDF parsing, rendering, layout, OCR/table stages và serialization; muốn so GPU phải đo từng stage hoặc ít nhất cố định pipeline và bật/tắt OCR.
- **Không dùng NVLink A100 để suy ra tốc độ `device_map`.** NVLink giúp liên kết phần cứng, nhưng Accelerate naive model parallelism vẫn thực thi các layer theo chuỗi và chịu synchronization/activation transfer.[3][9]
- **Các số từ TensorRT-LLM/vLLM không trả lời đúng cấu hình.** Chúng có thể dùng quantization, continuous batching, CUDA graph, kernel chuyên biệt hoặc pipeline/tensor parallelism; chỉ nên xem là tham khảo về trần hệ thống, không phải phép đo Transformers BF16 batch 1.[6]

## Nguồn (đánh số: tiêu đề — URL)

1. NVIDIA RTX PRO 6000 Blackwell Workstation Edition — https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/
2. NVIDIA H100 Tensor Core GPU Datasheet — https://i.dell.com/sites/csdocuments/App-Merchandizing_Documents/en/us/h100-gpu.pdf?ref=cpcl_nvidia-pdf_cta_itemlink_h100gpudatasheet
3. NVIDIA A100 Tensor Core GPU Datasheet — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
4. RTX 5090 Local LLM Benchmarks: Tokens per Second by Model — https://presenc.ai/research/rtx-5090-local-llm-benchmarks-2026
5. RTX 5090 ML Benchmarks, Tok/s, $/hr — https://www.codesota.com/hardware/rtx-5090
6. TensorRT-LLM Performance Overview — https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-overview.html
7. PyTorch Forum: RTX 5050 / sm_120 Support Timeline — https://discuss.pytorch.org/t/rtx-5050-sm-120-support-timeline-when-in-stable-pytorch/224288/8
8. PyTorch Forum: Request for SM_120 / RTX 50xx CUDA 12.8 Support — https://discuss.pytorch.org/t/request-for-sm-120-rtx-50xx-cuda-12-8-support-in-pytorch/222207/2
9. Hugging Face Accelerate: Big Model Inference — https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling
10. Docling GPU Support and Performance Results — https://github.com/docling-project/docling/blob/main/docs/usage/gpu.md
11. PyTorch 2.8.0 Release — https://github.com/pytorch/pytorch/releases/tag/v2.8.0
12. Hugging Face Accelerate: Distributed Inference — https://github.com/huggingface/accelerate/blob/main/docs/source/usage_guides/distributed_inference.md
13. NVIDIA RTX PRO 6000 Blackwell Server Edition — https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/

## Nguồn trả về từ API
- i.dell.com › sites › csdocumentsNVIDIA H100 Tensor Core GPU Datasheet - Dell — https://i.dell.com/sites/csdocuments/App-Merchandizing_Documents/en/us/h100-gpu.pdf?ref=cpcl_nvidia-pdf_cta_itemlink_h100gpudatasheet
- NVIDIA A100 TENSOR CORE GPU — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf.
- NVIDIA A100 | Tensor Core GPU — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
- NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog — https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- Comparing the NVIDIA RTX PRO 6000 vs. H100 for AI Inference — https://openmetal.io/resources/blog/comparing-the-nvidia-rtx-pro-6000-vs-h100-for-ai-inference/
- RTX PRO 6000 Blackwell GPU Specs, Benchmarks & Cloud Pricing — https://www.computecomparison.com/gpu/rtx-pro-6000-blackwell
- RTX PRO 6000 Blackwell Server Edition — https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
- [PDF] NVIDIA RTX PRO 6000 Blackwell Workstation Edition — https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf
- Includes final GPU / memory clocks and final TFLOPS performance specs. — https://www.techpowerup.com/gpu-specs/docs/nvidia-gh100-architecture.pdf
- www.nvidia.com · content · damNVIDIA A100 | Tensor Core GPU — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf
- LLM Inference Sizing and Performance Guidance — https://blogs.vmware.com/cloud-foundation/2024/09/25/llm-inference-sizing-and-performance-guidance/
- NVIDIA RTX PRO 6000 Blackwell Workstation Edition — https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/
- NVIDIA H100 SXM Specs, Pricing & AI Inference Benchmarks ... — https://inferencex.semianalysis.com/chips/h100
- NVIDIA RTX PRO 6000 Workstation GPU Review: Blackwell Architecture ... — https://www.storagereview.com/review/nvidia-rtx-pro-6000-workstation-gpu-review-blackwell-architecture-and-96-gb-for-pro-workflows?amp
- 96GB Memory Upgrade! NVIDIA RTX PRO™ 6000 ... — https://www.leadtek.com/eng/news/product_news_detail/1811
- The Roofline Model: Why LLM Decode Is Memory-Bound — https://skillsuites.com/roofline-model-llm-inference/
- www.codesota.com · hardware · rtx-5090RTX 5090: ML Benchmarks, Tok/s, $/hr (April 2026) — https://www.codesota.com/hardware/rtx-5090
- NVIDIA RTX 5090 (32 GB VRAM): Which LLMs Can It Run? | CanItRun — https://canitrun.dev/gpus/rtx-5090/
- RTX 5090 Local LLM Benchmarks: Tokens per Second by Model ... — https://presenc.ai/research/rtx-5090-local-llm-benchmarks-2026
- vdf.ai › llm-tokens-per-second-calculatorLLM Tokens per Second Calculator (GPU Inference Speed) — https://vdf.ai/llm-tokens-per-second-calculator/
- Best AI Models for the RTX 5090: What 32GB Unlocks — https://localaimaster.com/gpu/best-ai-models-rtx-5090
- omegaprime669/rtx-5090-benchmarks · Datasets at Hugging Face — https://huggingface.co/datasets/omegaprime669/rtx-5090-benchmarks
- Local LLM on NVIDIA RTX 5090: Best Models, Tokens/sec, Build ... — https://www.local-llm.net/benchmarks/rtx-5090/
- Is RTX PRO 6000 good for AI inference? Performance, cost, and limits — https://www.cloudpe.com/blog/rtx-pro-6000-for-ai-inference/
- Is the GeForce RTX 5090 Good for Running LLMs? - TechReviewer — https://www.techreviewer.com/tech-specs/nvidia-rtx-5090-gpu-for-llms/
- GitHub - chsasank/blackwell: Production LLM deployment specs ... — https://github.com/chsasank/blackwell
- Overview — TensorRT LLM - nvidia.github.io — https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-overview.html
- RTX 5090 Local LLM Benchmarks, Context Scaling & Supported ... — https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-5090/
- README.md · witcheer/rtx-5090-benchmarks at d9c047b0744b1a334243e30878ef36270ee3e5dc — https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/d9c047b0744b1a334243e30878ef36270ee3e5dc/README.md
- Overview — TensorRT LLM — https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/perf-overview.html
- docling/docs/usage/gpu.md at 54cd6d7406ee9431d45e8bcd1b0ae53389cfb56c · docling-project/docling — https://github.com/docling-project/docling/blob/54cd6d7406ee9431d45e8bcd1b0ae53389cfb56c/docs/usage/gpu.md
- docling/docs/usage/gpu.md at a1c5ff2a8c5ab2bf87d7c1eb896b5bc351f51220 · docling-project/docling — https://github.com/docling-project/docling/blob/a1c5ff2a/docs/usage/gpu.md?plain=1
- docling/docs/usage/gpu.md at main · docling-project/docling — https://github.com/docling-project/docling/blob/main/docs/usage/gpu.md
- GPU Performance Optimization & Benchmark #3442 — https://github.com/docling-project/docling/discussions/3442
- Big Model Inference - Hugging Face — https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling
- github.com › huggingface › accelerateaccelerate/docs/source/usage_guides/distributed_inference.md ... — https://github.com/huggingface/accelerate/blob/main/docs/source/usage_guides/distributed_inference.md
- Device Mapping and Inference | huggingface/accelerate | DeepWiki — https://deepwiki.com/huggingface/accelerate/5.2-device-mapping-and-inference
- How 🤗 Accelerate runs very large models thanks to PyTorch — https://huggingface.co/blog/accelerate-large-models
- Hugging Face Accelerate For... — https://rocm.docs.amd.com/en/docs-7.0.0/how-to/rocm-for-ai/fine-tuning/multi-gpu-fine-tuning-and-inference.html
- Fine-tuning and inference using multiple GPUs — https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/fine-tuning/multi-gpu-fine-tuning-and-inference.html
- Fine-tuning and inference using multiple accelerators — https://rocm.docs.amd.com/en/docs-6.1.5/how-to/llm-fine-tuning-optimization/multi-gpu-fine-tuning-and-inference.html
- Fine-tuning and inference using multiple accelerators — https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/multi-gpu-fine-tuning-and-inference.html
- Optimizing LLMs for Speed and Memory · Hugging Face — https://huggingface.co/docs/transformers/v5.0.0rc1/en/llm_tutorial_optimization
- docling.rs/docs/MODELS.md at master · ... — https://github.com/docling-project/docling.rs/blob/master/docs/MODELS.md
- Fine-tuning and inference using multiple accelerators — https://rocm.docs.amd.com/en/docs-6.3.2/how-to/rocm-for-ai/fine-tuning/multi-gpu-fine-tuning-and-inference.html
- RTX 5050 (sm_120) Support Timeline - When in Stable ... — https://discuss.pytorch.org/t/rtx-5050-sm-120-support-timeline-when-in-stable-pytorch/224288/8
- Request: Add CUDA sm_120 (Blackwell) support for ConvNeXtV2 ... — https://discuss.pytorch.org/t/request-add-cuda-sm-120-blackwell-support-for-convnextv2-fused-kernels/224904
- discuss.pytorch.org › t › rtx-5050-sm-120-supportRTX 5050 (sm_120) Support Timeline - When in Stable PyTorch? — https://discuss.pytorch.org/t/rtx-5050-sm-120-support-timeline-when-in-stable-pytorch/224288
- PyTorch Versions and Supported NVIDIA GPU Compute ... — https://journal.qualiteg.com/pytorch_and_supported_gpu_version/
- docs.gpuhub.com · environment · pytorch280PyTorch for Blackwell - Mint Starter Kit — https://docs.gpuhub.com/environment/pytorch280
- CUDA & PyTorch Compatibility | Guides - Clore.ai — https://docs.clore.ai/guides/getting-started/cuda-pytorch-compatibility
- Pytorch support for sm120 - Page 5 - deployment — https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099?page=5
- RTX 5090 (Blackwellアーキテクチャ) におけるPyTorch cuBLAS ... — https://qiita.com/N622/items/26538a143e7ebcbf897a
- Request for SM_120 / RTX 50xx (CUDA 12.8) Support in PyTorch — https://discuss.pytorch.org/t/request-for-sm-120-rtx-50xx-cuda-12-8-support-in-pytorch/222207/2
- Running PyTorch on RTX 5090 and 5080 GPUs - SaladCloud — https://docs.salad.com/container-engine/tutorials/machine-learning/pytorch-rtx5090
- When is all sm_120 supported in nightly? — https://discuss.pytorch.org/t/when-is-all-sm-120-supported-in-nightly/220070
- RTX 5090でPytorchのパフォーマンスが落ちる問題に関して - Zenn — https://zenn.dev/cynagenautes/articles/0566bd145c350c
- Pytorch support for sm120 - deployment — https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099
- NVIDIA GeForce RTX 5090 — https://discuss.pytorch.org/t/nvidia-geforce-rtx-5090/218954
- Release PyTorch 2.8.0 Release · pytorch/pytorch — https://github.com/pytorch/pytorch/releases/tag/v2.8.0
