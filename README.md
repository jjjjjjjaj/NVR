# NVR — NVIDIA Velocity Runtime

**A Windows-first local LLM inference runner focused on measured performance.**

NVR launches optimized inference backends rather than reimplementing CUDA kernels. The initial backend is [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA.

> **Status: early implementation.** NVR includes NVIDIA diagnostics and a llama.cpp CUDA launch path for Qwen GGUF files. Benchmark automation and the desktop UI are not yet implemented.

## Qwen3.8-27B Efficient Profile

NVR includes a dedicated memory-conscious launch profile for **Qwen3.8-27B**. It is intended to reduce temporary inference-buffer use while keeping the selected GGUF file, quantization, and context size unchanged.

| Setting | Default |
|---|---:|
| Context size | 8,192 tokens |
| GPU layers | 999 (request all layers on GPU) |
| Logical batch | 256 |
| Physical batch / micro-batch | 128 |
| Flash Attention | Enabled when supported by the backend |
| KV cache K/V | F16 |
| Bind address | 127.0.0.1 |
| Port | 1234 |

The batch and micro-batch defaults are selected to limit temporary compute-buffer allocation. Actual VRAM and RAM use depends on the GGUF, backend build, context, and runtime conditions. **No fixed memory saving or benchmark result is claimed by this profile.** The 4 GB VRAM and 3.5 GB system-RAM reductions are goals that require a matched baseline measurement before they can be reported as achieved results.

Run the profile with:

```powershell
.\target\release\nvr.exe run qwen3_8_27b --model "D:\Models\your-qwen-model.gguf"
```

Replace the example with the full path to the local GGUF file. Optional overrides:

```powershell
.\target\release\nvr.exe run qwen3_8_27b --model "D:\Models\your-model.gguf" --ctx 8192 --batch 256 --ubatch 128 --gpu-layers 999
```

NVR does not download model weights. It launches the exact local GGUF path supplied.

## Other model targets

| Target | Initial treatment | Status |
|---|---|---|
| “inlinking mini 2” | Pending exact model ID | Disabled until identity is confirmed |
| DeepSeek V4.1 Flash 1017 | Provider/model integration profile | Local weight availability not established |
| GLM 5.3 Flash | Compatibility profile | Requires artifact and architecture verification |

The **20% throughput target** applies only to Qwen3.8-27B and the confirmed Mini 2 model, under matched conditions. It means at least 1.20× llama.cpp generation throughput (output tokens/s) at the same quality, model revision, quantization, context, prompt, and decoding settings. This is a target, **not a claimed result**. Prompt processing, time to first token, and latency must be reported separately.

## Build

Install Rust stable, then from the repository root:

```powershell
cargo build --release
```

The executable is `target\release\nvr.exe`.

## Backend discovery

NVR can discover a CUDA 13 llama.cpp backend under:

`%USERPROFILE%\.lmstudio\extensions\backends\llama.cpp-*-cuda13-*\llama-server.exe`

It also locates the LM Studio CUDA vendor DLL directory and adds it to the child process PATH. No system-wide PATH edit is required.

Open `http://127.0.0.1:1234` after the server reports that it is listening. Keep the NVR terminal open; press Ctrl+C there to stop the server.

Environment overrides:
- `NVR_LLAMA_SERVER`: full path to llama-server.exe (or its directory)
- `NVR_CUDA_VENDOR_DIR`: CUDA vendor DLL directory

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
