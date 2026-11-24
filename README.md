# NVR — NVIDIA Velocity Runtime

## Build

Install Rust stable, then from the repository root:

```powershell
cargo build --release
```

The executable is `target\\release\\nvr.exe`.

## Run Qwen3.8-27B GGUF through LM Studio's CUDA 13 llama.cpp backend

NVR can discover a CUDA 13 llama.cpp backend under:

`%USERPROFILE%\\.lmstudio\\extensions\\backends\\llama.cpp-*-cuda13-*\\llama-server.exe`

It also locates the matching LM Studio vendor DLL directory and adds it to the child process PATH. No system-wide PATH edit is required.

Pass the full path to the actual GGUF file you downloaded:

```powershell
.\\target\\release\\nvr.exe run qwen3_8_27b --model "D:\\Models\\Qwen3.8-27B-Q4_K_M.gguf"
```

The filename above is an example; replace it with the real file path. Defaults:
- GPU layers: 999 (request all layers on GPU)
- Context: 8192
- Logical batch: 512
- Physical batch / ubatch: 512
- Flash Attention: on
- KV cache K/V: F16
- Host: 127.0.0.1 (local machine only)
- Port: 1234

Open `http://127.0.0.1:1234` after the server reports that it is listening. Keep the NVR terminal open; press Ctrl+C there to stop the server.

### Optional overrides

```powershell
.\\target\\release\\nvr.exe run qwen3_8_27b --model "D:\\Models\\your-model.gguf" --backend "C:\\path\\to\\llama-server.exe" --port 1234 --ctx 8192 --batch 512 --ubatch 512 --gpu-layers 999
```

You can also set:
- `NVR_LLAMA_SERVER`: full path to llama-server.exe (or its directory)
- `NVR_CUDA_VENDOR_DIR`: CUDA vendor DLL directory

NVR does not download model weights. It launches the exact local GGUF path you provide. A successful server start is not itself a benchmark; measure prompt and generation throughput separately under matched settings.
