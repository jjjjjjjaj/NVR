# Benchmark protocol

## Fair comparison

A performance result is valid only when both runners use the same:
- exact model files and cryptographic hashes;
- quantization and tokenizer;
- prompt text, prompt token count, and context length;
- sampling parameters, batch settings where comparable, and output length;
- GPU, driver, power policy, temperature range, and background workload.

If a backend requires a different quantization or model conversion, label it as a separate quality/performance comparison, not an apples-to-apples speed result.

## Procedure

1. Record hardware, OS, driver, backend version/build flags, model hash, and full launch configuration.
2. Run one or more warmups and exclude them from reported samples.
3. Run at least five measured trials for each workload.
4. Use fixed prompts and fixed output token limits; include short and long prompts.
5. Record prompt tokens/s, generation tokens/s, time to first token (TTFT), inter-token latency (ITL), peak VRAM, peak system RAM, GPU utilization, power, and temperature where available.
6. Report median, minimum, maximum, and dispersion (standard deviation or IQR). Preserve raw per-run data.
7. Alternate runner order or repeat the sequence to reduce thermal/order bias.

## Qwen3.8-27B efficient preset

The Qwen3.8-27B preset uses an 8192-token context, logical batch 256, physical batch (ubatch) 128, Flash Attention, F16 KV cache, and a request to offload all layers to the GPU. The reduced batch sizes are intended to lower temporary compute-buffer memory use.

This is a configuration change, not proof of a particular memory saving. Do not publish a fixed VRAM/RAM reduction or a speed improvement until matched trials have been completed. Compare against LM Studio using the same GGUF file, context, batch/ubatch, GPU offload, KV cache, prompt, output length, and backend build wherever possible. If LM Studio does not expose an equivalent setting, document the difference.

## 20% target

For a given workload, the generation-throughput target is:

`NVR median generation tok/s >= 1.20 * llama.cpp median generation tok/s`

This is an acceptance target, not an assertion. Also report TTFT, prompt throughput, ITL, memory, and output correctness. A result that is faster only because it changes quantization, context, output length, or quality does not satisfy the target.

## Result schema (v1)

Each result should include:
- `schema_version`
- `timestamp_utc`
- `runner`, `backend_name`, `backend_version`, `backend_build`
- `hardware`, `driver_version`, `os`
- `model_id`, `model_revision`, `model_files_sha256`, `quantization`
- `workload_id`, `prompt_sha256`, `prompt_tokens`, `output_tokens`, `context_length`, `sampling`
- `metrics`: `prompt_tokens_per_second`, `generation_tokens_per_second`, `ttft_ms`, `itl_ms_median`, `peak_vram_bytes`, `peak_ram_bytes`
- `trial_index`, `warmup`, `exit_status`, `notes`

Do not aggregate failed runs into successful-run throughput statistics. Store raw records so later versions can recompute summaries.
