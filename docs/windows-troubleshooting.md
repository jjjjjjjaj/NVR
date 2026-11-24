# NVR Windows setup and troubleshooting

## Model picker appears every launch

The Qwen launcher now searches `%USERPROFILE%\.lmstudio\models` recursively for a GGUF whose filename identifies Qwen and 27B. It prefers Q4_K_M when multiple matches exist. If the file is in a custom folder, set `NVR_QWEN_MODEL` to its full path. The picker is a fallback when automatic discovery finds nothing.

## Server starts, but chat returns 404 File Not Found

Use the OpenAI-compatible route, not just the server root:

- Base URL: `http://127.0.0.1:1234/v1`
- Model list: `GET /v1/models`
- Chat: `POST /v1/chat/completions`

A PowerShell example is in the repository README. Use the model ID returned by `/v1/models`.

## CUDA DLL or backend executable cannot be found

The launcher expects an LM Studio CUDA 13 llama.cpp backend by default. Set `NVR_LLAMA_SERVER` to the full path of `llama-server.exe` if automatic discovery selects no backend or the wrong one. The runtime prepends the matching LM Studio CUDA vendor directory to the child process PATH if present. It intentionally does not make global PATH changes.

Do not fetch individual DLLs from random download sites. Confirm the exact missing DLL name, backend architecture, CUDA runtime package and NVIDIA driver compatibility.

## Rust build says a dependency is unresolved

Build from the repository root so Cargo resolves the workspace dependencies:

```powershell
cargo build --release --bin nvr-qwen
```

If Cargo reports a missing crate, confirm the relevant crate declares the workspace dependency in its own `Cargo.toml`; declaring a dependency in the workspace root alone does not make it available to every member.

## NVR is slower than LM Studio

Same visible UI settings do not prove identical backend execution. Record each runner's llama.cpp version/build, complete command line, CUDA DLL/backend version, model file hash, prompt, output token limit, sampling, warmup, VRAM peak, GPU clocks/utilization/power/temperature, and system RAM. Alternate at least five matched runs. Do not change quantization or silently alter workload to improve the number.
