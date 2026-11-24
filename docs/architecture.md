# Architecture

## Product boundary

NVR is an orchestration and measurement layer. The first implementation delegates tensor execution to llama.cpp with its CUDA backend. NVR owns process lifecycle, configuration validation, model metadata, benchmark orchestration, result storage, and user-facing diagnostics. It does not initially implement kernels, quantization, or a new model format.

## Components

- **CLI (`apps/cli`)**: stable commands and machine-readable output.
- **Runtime (`crates/runtime`)**: environment checks, GPU inventory, backend discovery, launch plans, process supervision.
- **Benchmark (`crates/benchmark`)**: run protocol, metrics, aggregation, JSON result format.
- **Model catalog (`crates/model-catalog`)**: explicit model IDs, architecture notes, context limits only when verified, execution mode.
- **Desktop UI (future)**: Tauri 2 + React/TypeScript, calling the same runtime interfaces.
- **Local API (future)**: OpenAI-compatible endpoints, bound to 127.0.0.1 by default.

## Backend adapter contract

Each backend adapter should expose: capability discovery, version/build metadata, model format support, launch argument generation, start/stop, readiness probe, metrics collection where available, and structured error mapping. Keep backend-specific flags inside adapters; never scatter them through UI code.

## Safety and resource control

1. Estimate memory before launch, but label estimates and retain a configurable safety margin.
2. Do not assume all VRAM is available; query current free memory immediately before launch.
3. Never terminate unrelated GPU processes automatically.
4. On timeout or crash, capture logs and exit status; do not report a successful run.
5. Bind API to loopback by default. Network access requires explicit configuration and authentication.

## Proposed milestones

M0 scaffold and diagnostics; M1 llama.cpp discovery and launch; M2 repeatable benchmark; M3 optimizer; M4 API; M5 desktop app; M6 additional backends.
