# NVR — NVIDIA Velocity Runtime

**A Windows-first local LLM inference runner focused on measured performance.**

NVR sits above optimized inference engines rather than reimplementing CUDA kernels. The first backend target is [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA; later adapters can add ExLlamaV3 and TensorRT-LLM.

> **Status: early implementation.** NVR includes NVIDIA diagnostics and an initial llama.cpp CUDA 13 launch path for Qwen GGUF files. Benchmark automation and the desktop UI are not yet implemented.

## Goals

- Detect NVIDIA GPU, driver, VRAM, and runtime environment.
- Launch existing inference backends with validated configurations.
- Benchmark prompt processing and token generation separately.
- Measure TTFT, inter-token latency, throughput, peak VRAM/RAM, and run metadata.
- Search configurations safely and retain reproducible results.
- Offer a CLI first, then a desktop GUI and OpenAI-compatible local API.

## Model targets

| Target | Initial treatment | Status |
|---|---|---|
| Qwen3.8-27B | Local GGUF launch through llama.cpp with a memory-conscious preset | Efficient preset available; requires benchmark validation |
| “inlinking mini 2” | Placeholder pending exact model ID | Disabled until identity is confirmed |
| DeepSeek V4.1 Flash 1017 | Provider/model integration profile | Local weight availability not established |
| GLM 5.3 Flash | Compatibility profile | Requires artifact and architecture verification |

The **20% target** applies only to Qwen3.8-27B and the confirmed Mini 2 model, on the same machine and under matched conditions. It means at least 1.20× llama.cpp generation throughput (output tokens/s) at the same quality, model revision, quantization, context, prompt, and decoding settings. This is a target, **not a claimed result**. Report prompt processing, TTFT, and latency separately; never trade away correctness or silently change quantization to manufacture a win.

## Build

Install Rust stable, then from the repository root:

```powershell
cargo build --release
```

The executable is `target\\release\\nvr.exe`.

## Run Qwen3.8-27B

NVR can discover a CUDA 13 llama.cpp backend under:

`%USERPROFILE%\\.lmstudio\\extensions\\backends\\llama.cpp-*-cuda13-*\\llama-server.exe`

It also locates the LM Studio CUDA vendor DLL directory and adds it to the child process PATH. No system-wide PATH edit is required.

Pass the full path to the actual GGUF file you downloaded:

```powershell
.\\target\\release\\nvr.exe run qwen3_8_27b --model "D:\\Models\\your-qwen-model.gguf"
```

Replace the example path with the real GGUF path. The Qwen3.8-27B memory-conscious defaults are:
- GPU layers: 999 (request all layers on GPU)
- Context: 8192
- Logical batch: 256
- Physical batch / ubatch: 128
- Flash Attention: on
- KV cache K/V: F16
- Host: 127.0.0.1 (local machine only)
- Port: 1234

The smaller batch and micro-batch settings are intended to reduce temporary compute-buffer memory. They may affect prompt-processing throughput; measure both memory and speed before choosing a final preset. They do not guarantee a fixed VRAM or RAM reduction.

Open `http://127.0.0.1:1234` after the server reports that it is listening. Keep the NVR terminal open; press Ctrl+C there to stop the server.

Optional overrides:

```powershell
.\\target\\release\\nvr.exe run qwen3_8_27b --model "D:\\Models\\your-model.gguf" --backend "C:\\path\\to\\llama-server.exe" --port 1234 --ctx 8192 --batch 256 --ubatch 128 --gpu-layers 999
```

Environment overrides:
- `NVR_LLAMA_SERVER`: full path to llama-server.exe (or its directory)
- `NVR_CUDA_VENDOR_DIR`: CUDA vendor DLL directory

NVR does not download model weights. It launches the exact local GGUF path you provide. A successful server start is not itself a benchmark; measure prompt and generation throughput separately under matched settings.

## Repository layout

```
apps/
  cli/                 Rust command-line entry point
crates/
  runtime/             GPU diagnostics and llama.cpp launch orchestration
  benchmark/           Benchmark schema and comparison methodology
  model-catalog/        Explicit model target metadata
docs/
  architecture.md
  benchmarking.md
  model-targets.md
```

## Principles

- No invented performance claims; publish raw results and exact test conditions.
- Keep inference in established optimized backends initially.
- Never kill unrelated processes to free VRAM.
- Localhost-only API by default; require explicit opt-in and authentication for network exposure.
- Treat remote model APIs and local model weights as different execution modes.

See [architecture](docs/architecture.md), [benchmark protocol](docs/benchmarking.md), and [model targets](docs/model-targets.md).
