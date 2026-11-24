# NVR — NVIDIA Velocity Runtime

**A Windows-first local LLM inference runner focused on measured performance.**

NVR orchestrates established inference backends (initially llama.cpp with CUDA), adds Windows-friendly model discovery and launch diagnostics, and is intended to provide repeatable benchmarks. It does not currently implement its own CUDA kernels.

> **Status: early implementation.** The Qwen Windows launcher is working on the maintainer's machine. Windows runtime behavior and performance must still be validated on each target system.

## Current Qwen3.8-27B launcher

Build from the repository root:

```powershell
cargo build --release --bin nvr-qwen
```

Run:

```powershell
.\target\release\nvr-qwen.exe
```

The launcher searches recursively (up to 8 directory levels) under:

```
%USERPROFILE%\.lmstudio\models
```

It selects a matching Qwen 27B GGUF automatically, preferring a filename containing `Q4_K_M`. If no match is found, it opens a file picker. To choose a specific file without the picker, set:

```powershell
$env:NVR_QWEN_MODEL = "C:\path\to\Qwen3.8-27B-Q4_K_M.gguf"
.\target\release\nvr-qwen.exe
```

The selected model is launched with the current preset:

| Setting | Value |
|---|---:|
| GPU layers | 999 (request all layers) |
| Context | 8192 |
| Logical batch | 512 |
| Physical batch / ubatch | 512 |
| Flash Attention | ON |
| KV cache K/V | F16 |
| Host | 127.0.0.1 |
| Port | 1234 |

Keep the NVR terminal open while using the model. Press Ctrl+C in that terminal to stop it.

## Talking to the model (important: API route)

The server base URL is `http://127.0.0.1:1234`. The base URL by itself is **not** the chat endpoint. For OpenAI-compatible chat completions, use:

`POST http://127.0.0.1:1234/v1/chat/completions`

A 404 `File Not Found` commonly means the client is requesting the wrong route (for example, the root URL or a UI-specific route), not that the model failed to load.

First, inspect the model ID exposed by the server:

```powershell
$models = Invoke-RestMethod -Uri "http://127.0.0.1:1234/v1/models"
$modelId = $models.data[0].id
$modelId
```

Then send a chat request:

```powershell
$body = @{
  model = $modelId
  messages = @(
    @{ role = "user"; content = "Reply with exactly: NVR works." }
  )
  temperature = 0.2
  max_tokens = 64
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://127.0.0.1:1234/v1/chat/completions" `
  -Method Post -ContentType "application/json" -Body $body
```

For an OpenAI-compatible client, configure base URL `http://127.0.0.1:1234/v1` and use the model ID returned by `/v1/models`. Keep the API bound to localhost; do not expose it to a network without adding authentication and explicit access controls.

## Backend and DLL troubleshooting

NVR currently discovers an LM Studio CUDA 13 llama.cpp backend under:

`%USERPROFILE%\.lmstudio\extensions\backends\llama.cpp-*-cuda13-*\llama-server.exe`

It adds the corresponding LM Studio CUDA vendor DLL directory to the child process PATH when that directory exists. It does not modify the system-wide PATH. If discovery fails, install the LM Studio CUDA 13 llama.cpp backend or set `NVR_LLAMA_SERVER` to the full path of `llama-server.exe`. If a CUDA DLL cannot be loaded, verify that the backend architecture matches Windows and the installed NVIDIA driver, and inspect the exact missing DLL reported by Windows; do not download individual DLLs from third-party DLL sites.

## Benchmark status

Latest user-reported test (not yet an apples-to-apples controlled comparison):

| Runner | Generation speed | VRAM |
|---|---:|---|
| NVR | 78.43 tok/s | About 4 GB less than LM Studio |
| LM Studio | About 120 tok/s | About 4 GB more than NVR |

NVR's log also reported 591.31 prompt tokens/s and 1,024 generated tokens. The user reported LM Studio generated approximately 2.6k tokens. Since output length, prompt, warmup, backend build and runtime flags have not all been verified identical, this is an investigation baseline, not a confirmed performance ratio. See [issue #1](https://github.com/jjjjjjjaj/NVR/issues/1).

## Other planned model targets

- Qwen3.8-27B: local GGUF launch (current working target).
- “inlinking mini 2”: exact model identifier still needs confirmation.
- DeepSeek V4.1 Flash 1017 and GLM 5.3 Flash: compatibility/integration research; do not assume local weights are available.

A 20% speedup over a matched llama.cpp baseline is a target, not a measured claim. Any performance comparison must hold model files, quantization, prompt, output length, context, sampling, and hardware conditions constant.

## Repository layout

```
apps/cli/                 Rust CLI and Qwen launcher
crates/runtime/           llama.cpp discovery and process launch
crates/benchmark/         benchmark schema and methodology
crates/model-catalog/     model target metadata
docs/                     architecture and benchmark protocol
```

## Principles

- No invented performance claims; retain raw measurements and exact conditions.
- Use optimized inference backends rather than pretending NVR has independent CUDA kernels.
- Never kill unrelated processes to free VRAM.
- Bind local APIs to 127.0.0.1 by default.
- Treat remote model APIs and local model weights as different execution modes.
