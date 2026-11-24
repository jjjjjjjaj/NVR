use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTarget {
    pub key: &'static str,
    pub display_name: &'static str,
    pub execution_mode: ExecutionMode,
    pub verification: Verification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode { LocalWeights, RemoteApi, Unresolved }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verification { NeedsUpstreamVerification, NeedsExactName, NeedsCompatibilityTest }

pub const TARGETS: &[ModelTarget] = &[
    ModelTarget { key: "qwen3_8_27b", display_name: "Qwen3.8-27B", execution_mode: ExecutionMode::LocalWeights, verification: Verification::NeedsUpstreamVerification },
    ModelTarget { key: "inlinking_mini_2", display_name: "inlinking mini 2 (unverified)", execution_mode: ExecutionMode::Unresolved, verification: Verification::NeedsExactName },
    ModelTarget { key: "deepseek_v4_1_flash_1017", display_name: "DeepSeek 4.1 Flash 1017", execution_mode: ExecutionMode::Unresolved, verification: Verification::NeedsUpstreamVerification },
    ModelTarget { key: "glm_5_3_flash", display_name: "GLM 5.3 Flash", execution_mode: ExecutionMode::Unresolved, verification: Verification::NeedsCompatibilityTest },
];
