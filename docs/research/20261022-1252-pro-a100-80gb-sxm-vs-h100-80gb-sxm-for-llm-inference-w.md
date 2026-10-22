# A100 80GB SXM vs H100 80GB SXM for LLM inference with HuggingFace transformers (no vLLM, sdpa attention, bf16, batch size 1) on a 32B dense model: measured ratios for prefill (compute-bound) and decode (memory-bandwidth bound) tokens/s, memory bandwidth 2.0 vs 3.35 TB/s, TFLOPS; also how much faster is Docling PDF layout parsing on H100 vs A100; is A100 PCIe 80GB different from SXM? Give numbers with sources.

> Perplexity pro · 2026-10-22 12:52. Kiểm chứng link quan trọng trước khi dựa vào.

## Tóm tắt phát hiện

Không tìm thấy benchmark công khai, có thể kiểm chứng, khớp chính xác toàn bộ điều kiện **Transformers thuần, không vLLM, SDPA, BF16, batch size 1, mô hình dense 32B** trên **A100 80GB SXM và H100 80GB SXM**. Vì vậy không nên báo một “tỉ lệ đo được” duy nhất như thể đó là kết quả chuẩn; các tỉ lệ đáng tin cậy nhất có thể suy ra từ thông số phần cứng là:

- **Decode, bị giới hạn bởi băng thông HBM:** khoảng \(3.35/2.039 \approx 1.64\times\) nghiêng về H100 SXM.
- **Prefill, bị giới hạn bởi compute BF16 Tensor Core:** khoảng \(1{,}979/624 \approx 3.17\times\) nghiêng về H100 SXM theo peak lý thuyết; throughput thực tế thường thấp hơn và phụ thuộc kernel, kích thước prompt, model implementation và mức sử dụng Tensor Core.[1][2]
- **Docling:** tài liệu chính thức công khai số đo trên A100 cho layout model, nhưng không công bố đối chứng H100, nên hiện chưa có cơ sở để đưa ra một tỉ lệ H100/A100 đáng tin cậy.[3]
- **A100 PCIe 80GB khác SXM:** cùng dung lượng HBM2e và cùng peak BF16 danh nghĩa 312 TFLOPS theo thông số cơ bản, nhưng PCIe có băng thông bộ nhớ khoảng **1.935 TB/s**, còn SXM khoảng **2.039 TB/s**; SXM cũng có nền tảng NVLink tích hợp khác, làm PCIe chậm hơn nhẹ ở decode và có thể khác đáng kể trong workload đa GPU.[2]

## Kỹ thuật / ý tưởng đáng thử (mỗi ý: là gì · bằng chứng · điều kiện áp dụng · nguồn)

- **Dùng tỉ lệ băng thông để ước lượng decode trước khi benchmark.** · Là gì: khi batch 1 và mỗi bước decode chỉ sinh một token, phần lớn thời gian dành cho đọc trọng số và KV cache; do đó tỉ lệ lý tưởng gần tỉ lệ băng thông HBM. H100 SXM có **3.35 TB/s**, A100 SXM có **2.039 TB/s**, tương đương khoảng **1.64×**.[1][2] · Bằng chứng: NVIDIA mô tả H100 SXM có HBM3 80GB và hơn 3 TB/s, trong khi A100 80GB SXM có 2.039 GB/s.[1][2] · Điều kiện áp dụng: batch nhỏ, decode tuần tự, model dense, cùng precision và cùng độ dài context; nếu kernel không tận dụng băng thông hoặc KV cache lớn gây cache miss, tỉ lệ thực tế có thể thấp hơn. · Nguồn: [1][2]

- **Dùng tỉ lệ BF16 Tensor Core để ước lượng prefill.** · Là gì: với prefill dài, GEMM có kích thước lớn và mức song song cao hơn, nên H100 có lợi thế compute rõ rệt. Peak BF16 Tensor Core của H100 SXM là **1,979 TFLOPS**, A100 SXM là **624 TFLOPS**, tương đương khoảng **3.17×**.[1][2] · Bằng chứng: bảng thông số NVIDIA công bố các mức BF16 Tensor Core này; H100 còn có kiến trúc Tensor Core thế hệ mới.[1][6] · Điều kiện áp dụng: prompt đủ dài để GEMM lớn, không bị giới hạn bởi input pipeline, launch overhead, attention hoặc memory bandwidth; với prompt ngắn, tốc độ có thể không đạt tỉ lệ peak. · Nguồn: [1][2][6]

- **Benchmark riêng prefill và decode, không dùng một tokens/s duy nhất.** · Là gì: đo prefill bằng prompt cố định với nhiều độ dài, đo decode bằng warm-up và sinh số token cố định; báo riêng prompt tokens/s, decode tokens/s, latency/token và p50/p95. · Bằng chứng: các nghiên cứu gần đây phân biệt rõ cold prefill, warm prefill và decode; trên H100, một phân tích Qwen3-32B báo compute utilization trung bình khoảng **92.0%** cho cold prefill nhưng chỉ **7.9%** cho decode, cho thấy hai pha vận hành ở các chế độ phần cứng khác nhau.[4] · Điều kiện áp dụng: phải cố định model revision, PyTorch/CUDA, tokenizer, context length, `use_cache`, số warm-up, nhiệt độ GPU và cách tính token. · Nguồn: [4]

- **Ép xác nhận SDPA backend thay vì chỉ đặt `attn_implementation="sdpa"`.** · Là gì: kiểm tra PyTorch thực sự chọn Flash Attention, memory-efficient attention hay math backend; SDPA là dispatcher có thể chọn nhiều backend lúc chạy.[5][7] · Bằng chứng: tài liệu PyTorch nêu SDPA dispatch tới nhiều backend; tài liệu Hugging Face nêu SDPA có thể gọi FlashAttention hoặc memory-efficient kernels bên dưới.[5][7] · Điều kiện áp dụng: giữ nguyên backend giữa A100 và H100 để so sánh công bằng; ghi lại PyTorch, CUDA, driver, dtype và log backend. · Nguồn: [5][7]

- **Đo bằng một harness tối giản thay vì dựa vào benchmark Transformers cũ.** · Là gì: chạy trực tiếp `model.generate` hoặc vòng lặp forward có `past_key_values`, dùng CUDA events, đồng bộ trước/sau vùng đo và loại bỏ thời gian tokenize/CPU. · Bằng chứng: Hugging Face cho biết bộ công cụ benchmark cũ của Transformers đã deprecated và khuyến nghị dùng thư viện benchmark bên ngoài.[8] · Điều kiện áp dụng: đặc biệt quan trọng với batch 1, vì overhead Python và kernel launch có thể làm sai tỉ lệ giữa GPU. · Nguồn: [8]

- **Nếu mục tiêu là Docling throughput, tăng batch theo từng stage.** · Là gì: thử riêng `layout_batch_size`, `ocr_batch_size` và `table_batch_size`, thay vì chỉ đổi GPU. · Bằng chứng: tài liệu triển khai Docling khuyến nghị tăng batch theo stage để tăng throughput GPU; một repo tối ưu Docling cũng xác định ba tham số này là các nút điều chỉnh chính.[9][10] · Điều kiện áp dụng: xử lý nhiều trang độc lập; không phù hợp nếu yêu cầu latency thấp cho một trang duy nhất hoặc VRAM bị giới hạn. · Nguồn: [9][10]

- **Với Docling, tái tạo benchmark theo đúng model và pipeline trước khi suy luận H100 nhanh hơn bao nhiêu.** · Là gì: dùng cùng phiên bản Docling, layout model, OCR, TableFormer, độ phân giải ảnh và batch; đo wall-clock toàn pipeline lẫn riêng layout inference. · Bằng chứng: báo cáo kỹ thuật Docling công bố benchmark trên L4, CPU x86 và M3 Max, trong đó layout model trung bình **44 ms/trang trên L4**, nhưng không công bố số đo A100–H100 trực tiếp.[3] · Điều kiện áp dụng: cần phân biệt “layout parsing” với toàn bộ PDF conversion; OCR, table recognition, rasterization và I/O có thể che lấp lợi ích GPU. · Nguồn: [3]

## Bẫy và cảnh báo đã biết

- **Không gọi 3.17× là tốc độ prefill đo được.** Đây chỉ là tỉ lệ peak BF16 Tensor Core: \(1{,}979/624\). H100 có thể đạt thấp hơn do attention, kích thước GEMM, memory traffic, kernel implementation và overhead.[1][2]

- **Không gọi 1.64× là tốc độ decode chắc chắn.** Đây là trần ước lượng theo HBM bandwidth: \(3.35/2.039\). Decode batch 1 còn chịu ảnh hưởng của kernel launch, độ dài context, KV-cache layout, L2 hit rate và mức song song của các GEMM nhỏ.[1][2][4]

- **Peak BF16 trong bảng NVIDIA có thể gây nhầm lẫn.** A100 được trình bày với BF16 Tensor Core **312 TFLOPS cơ bản / 624 TFLOPS với sparsity**, còn H100 được trình bày **989 TFLOPS / 1,979 TFLOPS với sparsity** trong bảng thông số; so sánh dense với dense phải dùng **312 và 989**, cho tỉ lệ khoảng \(989/312 \approx 3.17×\), không phải trộn một giá trị dense với một giá trị sparsity.[1][2]

- **H100 SXM và H100 PCIe không phải cùng GPU platform.** H100 SXM có 80GB HBM3 và 3.35 TB/s; H100 PCIe có HBM2e và khoảng 2.0 TB/s. Tương tự, A100 PCIe có 1.935 TB/s còn A100 SXM có 2.039 TB/s.[1][2][6]

- **A100 PCIe 80GB không chỉ khác ở đầu nối.** Hai bản đều có 80GB HBM2e, nhưng PCIe là card dual-slot air-cooled hoặc single-slot liquid-cooled, còn SXM là module máy chủ; interconnect, tản nhiệt, công suất và topology hệ thống khác nhau.[2]

- **NVLink không quyết định nhiều trong một GPU, nhưng quan trọng khi chạy nhiều GPU.** A100 SXM có NVLink 600 GB/s theo thông số NVIDIA; H100 SXM có NVLink 900 GB/s. Với tensor parallelism hoặc pipeline phân tách, chênh lệch interconnect có thể làm kết quả khác xa tỉ lệ HBM đơn GPU.[1][2]

- **Docling không có số H100-vs-A100 công khai đáng tin cậy trong các nguồn đã tìm thấy.** Nguồn chính thức nêu **28 ms/image trên một A100** cho `heron-101`, nhưng không nêu thời gian H100; vì vậy không thể suy ra chính xác H100 nhanh hơn bao nhiêu chỉ từ tài liệu đó.[3]

- **“PDF layout parsing” có thể chỉ là một stage, không phải toàn pipeline.** Docling bao gồm layout analysis, table structure recognition, OCR, PDF parsing và các bước xử lý khác; benchmark tổng thể phụ thuộc loại PDF, số bảng, số trang, OCR và I/O.[3][11]

- **Có báo cáo vấn đề GPU utilization thấp ở Docling.** Một issue của dự án mô tả layout model có thể sử dụng A100 chỉ khoảng 0–20% và speedup so với CPU chỉ khoảng 2× trong một cấu hình; đây là báo cáo issue, không phải benchmark chuẩn hóa, nhưng cho thấy cần kiểm tra pipeline và batching thay vì chỉ đổi GPU.[12]

- **Batch size 1 làm phép so sánh nhạy với overhead.** H100 có thể không đạt ưu thế lý thuyết nếu mỗi bước tạo ra quá ít công việc; cần báo cả latency/token và GPU utilization, không chỉ tokens/s.

- **Không trộn benchmark vLLM/TensorRT-LLM với Transformers thuần.** Các engine này dùng kernel, continuous batching, paged KV cache và scheduling khác; số đo của chúng không trả lời trực tiếp cấu hình được hỏi.

## Nguồn (đánh số: tiêu đề — URL)

1. NVIDIA H100 GPU — https://www.nvidia.com/en-us/data-center/h100/  
2. NVIDIA A100 Tensor Core GPU — https://www.nvidia.com/en-us/data-center/a100/  
3. Docling Papers — https://docling.ai/papers/  
4. Dissecting GPU Utilization for LLM Inference on Nvidia Hopper — https://arxiv.org/html/2609.12923v1  
5. Reproducibility — PyTorch documentation — https://docs.pytorch.org/docs/stable/notes/randomness.html  
6. NVIDIA Hopper Architecture In-Depth — https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/  
7. Out-of-the-box acceleration and memory savings with PyTorch 2.0 — https://pytorch.org/blog/out-of-the-box-acceleration/  
8. Hugging Face Transformers Benchmarks — https://huggingface.co/docs/transformers/v4.40.1/en/benchmarks  
9. Docling GPU support — https://docling-project.github.io/docling/usage/gpu/  
10. docling-fast — https://github.com/sch0tten/docling-fast  
11. Docling Technical Report — https://arxiv.org/html/2408.09869v4  
12. Layout Model is slow on all CUDA GPUs — https://github.com/docling-project/docling/issues/1364

## Nguồn trả về từ API
- H100 GPU - NVIDIA — https://www.nvidia.com/en-us/data-center/h100/
- High-Performance Computing — https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA A100 | Tensor Core GPU — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
- NVIDIA A100 TENSOR CORE GPU — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf?_hsenc=p2ANqtz--qxYxdL-xIp3hxNxkKU5B6ACDwOZg5MtaPtxT66STeTrhCadf62lCalftlWGfnf3hNFNpOkj53UYOOLCs8y5haPzdbSw
- Includes final GPU / memory clocks and final TFLOPS performance specs. — https://www.techpowerup.com/gpu-specs/docs/nvidia-gh100-architecture.pdf
- NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog — https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- ACCELERATING A TRITON FUSED KERNEL FOR W4A16 — https://www.arxiv.org/pdf/2402.00025
- LLM Inference Sizing and Performance Guidance — https://blogs.vmware.com/cloud-foundation/2024/09/25/llm-inference-sizing-and-performance-guidance/
- NVIDIA H100 Tensor Core GPU - Deep Learning ... — https://lambda.ai/blog/nvidia-h100-gpu-deep-learning-performance-analysis
- Papers - Docling — https://docling.ai/papers/
- docling.ai › benchmarksBenchmarks — Docling — https://docling.ai/benchmarks/
- H100 vs A100: GPU Benchmarks, Specs & Cloud Pricing Compared ... — https://jarvislabs.ai/blog/h100-vs-a100
- Docling Technical Report — https://arxiv.org/html/2408.09869v4
- LLM Inference: Prefill, Decode, and KV Cache | Neel Mishra — https://neelmishra.github.io/blog/mlops/llm-inference/llm-inference-fundamentals.html
- [PDF] NVIDIA A100 Tensor Core GPU Architecture — https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
- Reproducibility — PyTorch 2.14 documentation — https://docs.pytorch.org/docs/stable/notes/randomness.html
- Audio Spectrogram Transformer · Hugging Face — https://huggingface.co/docs/transformers/v4.45.1/en/model_doc/audio-spectrogram-transformer
- Video Vision Transformer (ViViT) · Hugging Face — https://huggingface.co/docs/transformers/v4.47.0/en/model_doc/vivit
- BEiT · Hugging Face — https://huggingface.co/docs/transformers/v5.20.2/en/model_doc/beit
- Video Vision Transformer (ViViT) · Hugging Face — https://huggingface.co/docs/transformers/v5.13.1/model_doc/vivit
- Out of the box acceleration and memory savings ... — https://pytorch.org/blog/out-of-the-box-acceleration/
- BEiT · Hugging Face — https://huggingface.co/docs/transformers/v5.5.0/en/model_doc/beit
- GPU inference · Hugging Face — https://huggingface.co/docs/transformers/v4.49.0/en/perf_infer_gpu_one
- GPU inference — https://huggingface.co/docs/transformers/v4.37.0/perf_infer_gpu_one
- GPU inference - Hugging Face — https://huggingface.co/docs/transformers/v4.39.0/en/perf_infer_gpu_one
- DeiT · Hugging Face — https://huggingface.co/docs/transformers/v5.8.1/en/model_doc/deit
- ViTMSN · Hugging Face — https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/vit_msn
- GPU inference — https://huggingface.co/docs/transformers/v4.37.0/en/perf_infer_gpu_one
- Pipeline - Hugging Face — https://huggingface.co/docs/transformers/v4.56.2/pipeline_tutorial
- BEiT — https://huggingface.co/docs/transformers/v4.57.1/model_doc/beit
- HBM Is Not All You Need: Efficient Disaggregated LLM Serving across Memory-heterogeneous Accelerators — https://arxiv.org/pdf/2606.29986v1
- Benchmarks - Hugging Face — https://huggingface.co/docs/transformers/v4.23.1/en/benchmarks
- Benchmarks · Hugging Face — https://huggingface.co/docs/transformers/v4.40.1/en/benchmarks
- MemExplorer: Navigating the Heterogeneous Memory Design Space for Agentic Inference NPUs — https://arxiv.org/pdf/2604.16007v2
- github.com · blob · maintransformers/docs/source/en/model_doc/depth_pro.md at main ... -... — https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/depth_pro.md
- ALGOTECH/QwQ-32B-TRT · Hugging Face — https://huggingface.co/ALGOTECH/QwQ-32B-TRT?library=transformers
- GPU Inference: H100 vs A100 vs L4 — https://inferenceengineering.tech/learn/gpu-inference/
- 基准测试 - Hugging Face — https://huggingface.co/docs/transformers/v4.48.0/zh/benchmarks
- Training — https://huggingface.co/blog/ettin-reranker
- Benchmarks — https://huggingface.co/docs/transformers/v4.41.1/ja/benchmarks
- LLMVisor: A Real-Time Latency Attribution Model for Multi- ... — https://arxiv.org/html/2608.08382v1
- [Intel Gaudi] #6. GEMM, Attention, vLLM on Gaudi - Blog — https://blog.squeezebits.com/intel-gaudi-gemm-attention-performance
- Dissecting GPU Utilization for LLM Inference on Nvidia Hopper - arXiv — https://arxiv.org/html/2609.12923v1
- nvidia-dynamo/recipes/README.md at main · LLM360 ... - GitHub — https://github.com/LLM360/nvidia-dynamo/blob/main/recipes/README.md
- Qwen3-32B FP8 | NVIDIA Dynamo Documentation — https://docs.dynamo.nvidia.com/dynamo/v1.4.1/recipes/qwen3-32b-fp8
- Layout Model is slow on all CUDA GPUs · Issue #1364 · docling-project/docling — https://github.com/docling-project/docling/issues/1364
- GitHub - sch0tten/docling-fast: GPU-accelerated batch PDF extraction … — https://github.com/sch0tten/docling-fast
- docling.rs/docs/PDF_CONFORMANCE.md at master ... - GitHub — https://github.com/docling-project/docling.rs/blob/master/docs/PDF_CONFORMANCE.md
- Docling: An Efficient Open-Source Toolkit for AI-driven ... — https://arxiv.org/html/2501.17887v1
- IBM Docling Technical Manual & OCR Guide - Docling.org — https://docling.org/doc/index.html
- Model Catalog - Docling — https://docling-project-docling.mintlify.app/advanced/model-catalog
- github.com · docling-project · doclingGitHub - docling-project/docling.rs: Full Rust port of Python ... — https://github.com/docling-project/docling.rs
- Index - Docling — https://docling-project.github.io/docling/
- Performance characteristics and optimization #306 — https://github.com/docling-project/docling/discussions/306
- docling-project.github.io · docling · usageGPU support - Docling — https://docling-project.github.io/docling/usage/gpu/
- Model Catalog - Docling — https://docling-project.github.io/docling/usage/model_catalog/
- docling-project/docling-models · Hugging Face — https://huggingface.co/docling-project/docling-models
