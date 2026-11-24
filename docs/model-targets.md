# Model targets and verification status

Model labels supplied for the project are not enough to safely download or benchmark weights. Confirm exact upstream repository, revision, license, file format, quantization, and architecture before enabling a one-click download.

| User label | Catalog key | Current status |
|---|---|---|
| Qwen3.8-27B | `qwen3_8_27b` | Exact official model repository/revision must be verified |
| “inlinking mini 2” | `inlinking_mini_2` | Unresolved spelling / model identity; intentionally disabled |
| DeepSeek 4.1 Flash 1017 | `deepseek_v4_1_flash_1017` | Verify whether target is hosted API or downloadable local weights; do not assume local support |
| GLM 5.3 Flash | `glm_5_3_flash` | Verify official artifact, MoE architecture, active/total parameters, context, quantizations, and actual memory needs |

## Rules

- Never silently replace a user-specified model with a similarly named one.
- Keep remote API targets distinct from local-weight targets.
- Do not hardcode unverified parameter counts, context lengths, licenses, or memory requirements.
- A model profile is not proof of backend compatibility. Compatibility requires a real load and correctness test.
- Record exact model revision and file hashes in every benchmark.
