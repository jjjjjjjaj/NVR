use serde::{Deserialize, Serialize};

/// Stable raw-trial record. Keep per-trial values; calculate summaries downstream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTrial {
    pub schema_version: u32,
    pub runner: String,
    pub backend_name: String,
    pub backend_version: Option<String>,
    pub model_id: String,
    pub model_revision: Option<String>,
    pub quantization: Option<String>,
    pub workload_id: String,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub context_length: u64,
    pub generation_tokens_per_second: Option<f64>,
    pub prompt_tokens_per_second: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub itl_ms_median: Option<f64>,
    pub peak_vram_bytes: Option<u64>,
    pub peak_ram_bytes: Option<u64>,
    pub trial_index: u32,
    pub warmup: bool,
    pub exit_status: String,
}
