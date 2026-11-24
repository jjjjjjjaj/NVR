# NVR — NVIDIA Velocity Runtime

**A Windows-first local LLM inference runner focused on measured performance.**

NVR is intended to sit above optimized inference engines rather than reimplement their CUDA kernels. The first backend target is [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA; later adapters can add ExLlamaV3 and TensorRT-LLM.

> **Status: early scaffold / not production-ready.** This repository currently contains the project architecture, a working GPU diagnostic CLI scaffold, model target definitions, and a reproducible benchmark specification. It does not yet launch models or claim performance gains.

## Goals

- Detect NVIDIA GPU, driver, VRAM, and runtime environment.
- Launch existing inference backends with validated configurations.
- Benchmark prompt processing and token generation separately.
- Measure TTFT, inter-token latency, throughput, peak VRAM/RAM, and run metadata.
- Search configurations safely and retain reproducible results.
- Offer a CLI first, then a desktop GUI and OpenAI-compatible local API.

## Model targets

| Target | Initial treatment | Performance objective |
|---|---|---|
| Qwen3.8-27B | Local model profile; exact upstream artifact/revision must be confirmed before download | User-run comparison against llama.cpp |
| “inlinking mini 2” | Placeholder pending exact model ID; do not silently substitute a similarly named model | User-run comparison against llama.cpp |
| DeepSeek V4.1 Flash 1017 | Provider/model integration profile; determine whether a supported local weight release exists before treating as local inference | API latency/cost benchmark if remote |
| GLM 5.3 Flash | Compatibility profile; verify official artifact, architecture, and memory requirements before attempting local load | Compatibility and performance measurements |

The **20% target** applies only to Qwen3.8-27B and the confirmed Mini 2 model, on the same machine and under matched conditions. It means at least 1.20× llama.cpp generation throughput (output tokens/s) at the same quality, model revision, quantization, context, prompt, and decoding settings. It is a target, **not a claimed result**. Report prompt processing, TTFT, and latency separately; never trade away correctness or silently change quantization to manufacture a win.

## Repository layout

```
apps/
  cli/                 Rust command-line entry point
crates/
  runtime/             GPU diagnostics and runtime orchestration
  benchmark/           Benchmark schema and comparison methodology
  model-catalog/        Explicit model target metadata
docs/
  architecture.md
  benchmarking.md
  model-targets.md
```

## Quick start (scaffold)

Requirements: Rust stable and NVIDIA drivers with `nvidia-smi` available on PATH.

```powershell
cargo run -p nvr -- doctor
```

This first diagnostic checks for `nvidia-smi`; it does not initialize CUDA or prove a backend can load a model.

## Roadmap

1. Reliable NVIDIA diagnostics and environment report.
2. llama.cpp CUDA backend lifecycle and health checks.
3. Standardized benchmark harness with warmups, repeated trials, and JSON output.
4. Configuration search with memory safety margins and rollback.
5. OpenAI-compatible local API and desktop GUI.
6. Additional backend adapters after parity tests.

## Principles

- No invented performance claims; publish raw results and exact test conditions.
- Keep inference in established optimized backends initially.
- Never kill unrelated processes to free VRAM.
- Localhost-only API by default; require explicit opt-in and authentication for network exposure.
- Treat remote model APIs and local model weights as different execution modes.

See [architecture](docs/architecture.md), [benchmark protocol](docs/benchmarking.md), and [model targets](docs/model-targets.md).
