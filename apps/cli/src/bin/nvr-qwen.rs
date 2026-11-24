use anyhow::{Context, Result};
use nvr_runtime::{discover_llama_server, launch_llama_server, LlamaLaunchConfig};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const MODEL_ENV: &str = "NVR_QWEN_MODEL";
const MODEL_ROOT: &str = ".lmstudio/models";
const MAX_SCAN_DEPTH: usize = 8;

fn main() {
    println!("==================================================");
    println!(" NVR - NVIDIA Velocity Runtime | Qwen Launcher");
    println!("==================================================");
    println!();

    if let Err(error) = run() {
        eprintln!();
        eprintln!("[NVR ERROR] {error:#}");
        eprintln!();
        eprintln!("Press Enter to close this window.");
        let mut input = String::new();
        let _ = std::io::stdin().read_line(&mut input);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let model_path = match discover_qwen_model()? {
        Some(path) => {
            println!("Automatically selected Qwen model:");
            path
        }
        None => {
            println!("No Qwen3.8-27B GGUF was found automatically.");
            println!("Opening the model picker as a fallback.");
            match rfd::FileDialog::new()
                .set_title("Select your Qwen3.8-27B GGUF model")
                .add_filter("GGUF model", &["gguf"])
                .pick_file()
            {
                Some(path) => path,
                None => {
                    println!("No model selected. Launcher cancelled.");
                    return Ok(());
                }
            }
        }
    };

    if !model_path.is_file() {
        anyhow::bail!("Model file does not exist: {}", model_path.display());
    }

    println!("Selected model: {}", model_path.display());
    println!("Preset: Qwen3.8-27B | context 8192 | batch 512 | ubatch 512");
    println!("GPU layers: 999 | Flash Attention: ON | KV cache: F16");
    println!();

    let backend = discover_llama_server(None).context(
        "Could not find the LM Studio CUDA 13 llama.cpp backend. Install the CUDA 13 llama.cpp runtime in LM Studio, or set NVR_LLAMA_SERVER to the full path of llama-server.exe.",
    )?;

    let config = LlamaLaunchConfig {
        model_path,
        port: 1234,
        context_size: 8192,
        batch_size: 512,
        ubatch_size: 512,
        gpu_layers: 999,
    };

    println!("Launching model with NVR...");
    println!("Local API base: http://127.0.0.1:1234");
    println!("Chat API: http://127.0.0.1:1234/v1/chat/completions");
    println!("Keep this window open while using the model.");
    println!("Press Ctrl+C here to stop the server.");
    println!();

    let status =
        launch_llama_server(&backend, &config).context("Failed to launch the llama.cpp backend")?;

    if !status.success() {
        anyhow::bail!("llama-server exited with status {status}");
    }

    Ok(())
}

/// Explicit NVR_QWEN_MODEL wins. Otherwise search the standard LM Studio
/// model directory for Qwen 3.8 27B GGUF files. Prefer Q4_K_M deterministically.
fn discover_qwen_model() -> Result<Option<PathBuf>> {
    if let Some(value) = env::var_os(MODEL_ENV) {
        let path = PathBuf::from(value);
        if !path.is_file() {
            anyhow::bail!(
                "{MODEL_ENV} points to a file that does not exist: {}",
                path.display()
            );
        }
        return Ok(Some(path));
    }

    let Some(profile) = env::var_os("USERPROFILE").or_else(|| env::var_os("HOME")) else {
        return Ok(None);
    };
    let root = PathBuf::from(profile).join(MODEL_ROOT);
    if !root.is_dir() {
        return Ok(None);
    }

    let mut candidates = Vec::new();
    collect_qwen_candidates(&root, 0, &mut candidates)?;
    candidates.sort_by(|a, b| {
        let a_q4 = a
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .contains("q4_k_m");
        let b_q4 = b
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .contains("q4_k_m");
        b_q4
            .cmp(&a_q4)
            .then_with(|| a.to_string_lossy().cmp(&b.to_string_lossy()))
    });

    if candidates.len() > 1 {
        println!(
            "Found {} matching Qwen3.8-27B GGUF files; selecting the preferred Q4_K_M file (or first sorted match). Set {MODEL_ENV} to override.",
            candidates.len()
        );
    }
    Ok(candidates.into_iter().next())
}

fn collect_qwen_candidates(dir: &Path, depth: usize, output: &mut Vec<PathBuf>) -> Result<()> {
    if depth > MAX_SCAN_DEPTH {
        return Ok(());
    }
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if depth == 0 => {
            return Err(error)
                .with_context(|| format!("Could not scan model directory {}", dir.display()));
        }
        Err(_) => return Ok(()), // Skip inaccessible nested directories.
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(value) => value,
            Err(_) => continue,
        };
        if file_type.is_dir() {
            collect_qwen_candidates(&path, depth + 1, output)?;
        } else if file_type.is_file() {
            let filename = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if filename.ends_with(".gguf")
                && filename.contains("qwen")
                && (filename.contains("27b")
                    || path
                        .to_string_lossy()
                        .to_ascii_lowercase()
                        .contains("qwen3.8-27b"))
            {
                output.push(path);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_candidate_requires_gguf_and_qwen_name() {
        let root = std::env::temp_dir().join(format!("nvr-qwen-discovery-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("Qwen3.8-27B-Q4_K_M.gguf"), b"test").unwrap();
        fs::write(root.join("other-27B.gguf"), b"test").unwrap();
        fs::write(root.join("Qwen3.8-27B.txt"), b"test").unwrap();

        let mut found = Vec::new();
        collect_qwen_candidates(&root, 0, &mut found).unwrap();
        assert_eq!(found.len(), 1);
        assert!(found[0]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("Qwen3.8-27B"));
        let _ = fs::remove_dir_all(root);
    }
}
