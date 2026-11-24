use anyhow::{Context, Result};
use nvr_runtime::{discover_llama_server, launch_llama_server, LlamaLaunchConfig};
use std::path::PathBuf;

fn main() {
    println!("==================================================");
    println!(" NVR - NVIDIA Velocity Runtime | Qwen Launcher");
    println!("==================================================");
    println!();
    println!("Choose your Qwen3.8-27B GGUF model in the file picker.");

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
    let selected_model: Option<PathBuf> = rfd::FileDialog::new()
        .set_title("Select your Qwen3.8-27B GGUF model")
        .add_filter("GGUF model", &["gguf"])
        .pick_file();

    let model_path = match selected_model {
        Some(path) => path,
        None => {
            println!("No model selected. Launcher cancelled.");
            return Ok(());
        }
    };

    if !model_path.is_file() {
        anyhow::bail!("Selected model file does not exist: {}", model_path.display());
    }

    println!();
    println!("Selected model: {}", model_path.display());
    println!("Preset: Qwen3.8-27B | context 8192 | batch 512 | ubatch 512");
    println!("GPU layers: 999 | Flash Attention: ON | KV cache: F16");
    println!();

    let backend = discover_llama_server(None)
        .context("Could not find the LM Studio CUDA 13 llama.cpp backend. Install the CUDA 13 llama.cpp runtime in LM Studio, or set NVR_LLAMA_SERVER to llama-server.exe.")?;

    let config = LlamaLaunchConfig {
        model_path,
        port: 1234,
        context_size: 8192,
        batch_size: 512,
        ubatch_size: 512,
        gpu_layers: 999,
    };

    println!("Launching model with NVR...");
    println!("Local API: http://127.0.0.1:1234");
    println!("Keep this window open while using the model.");
    println!("Press Ctrl+C here to stop the server.");
    println!();

    let status = launch_llama_server(&backend, &config)
        .context("Failed to launch the llama.cpp backend")?;

    if !status.success() {
        anyhow::bail!("llama-server exited with status {status}");
    }

    Ok(())
}
