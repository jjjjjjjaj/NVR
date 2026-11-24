use anyhow::{bail, Context, Result};
use nvr_runtime::{discover_llama_server, launch_llama_server, LlamaLaunchConfig};
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if let Err(error) = run() {
        eprintln!("NVR error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("doctor") => doctor(),
        Some("version") | Some("--version") | Some("-V") => {
            println!("NVR {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("run") => run_model(args.collect()),
        _ => {
            eprintln!("NVR — NVIDIA Velocity Runtime");
            eprintln!("Usage:");
            eprintln!("  nvr doctor");
            eprintln!("  nvr version");
            eprintln!("  nvr run qwen3_8_27b --model <path-to-GGUF> [options]");
            eprintln!("Options:");
            eprintln!("  --backend <path>       Path to llama-server.exe or its folder");
            eprintln!("  --port <port>          Local server port (default: 1234)");
            eprintln!("  --ctx <tokens>         Context size (default: 8192)");
            eprintln!("  --batch <tokens>       Logical batch size (default: 512)");
            eprintln!("  --ubatch <tokens>      Physical batch size (default: 512)");
            eprintln!("  --gpu-layers <count>   GPU layers (default: 999 = offload all)");
            bail!("Missing or unknown command")
        }
    }
}

fn doctor() -> Result<()> {
    println!("NVR environment diagnostic");
    match nvr_runtime::nvidia_smi_path() {
        Some(path) => {
            println!("nvidia-smi: found at {}", path.display());
            match Command::new(&path)
                .arg("--query-gpu=name,memory.total,driver_version")
                .arg("--format=csv,noheader")
                .output()
            {
                Ok(output) if output.status.success() => {
                    println!("{}", String::from_utf8_lossy(&output.stdout).trim());
                }
                Ok(output) => eprintln!("nvidia-smi exited with status {}", output.status),
                Err(err) => eprintln!("Could not execute nvidia-smi: {err}"),
            }
        }
        None => {
            println!("nvidia-smi: not found on PATH");
            println!("Install/update the NVIDIA driver, then reopen the terminal.");
        }
    }
    println!("Note: this checks driver visibility; run a model to test inference.");
    Ok(())
}

fn run_model(args: Vec<String>) -> Result<()> {
    let mut args = args.into_iter();
    let model_key = args.next().context("Expected model key: qwen3_8_27b")?;
    if model_key != "qwen3_8_27b" {
        bail!("Unknown or not-yet-enabled local model '{model_key}'. Currently enabled: qwen3_8_27b");
    }

    let mut model_path: Option<PathBuf> = None;
    let mut backend_path: Option<PathBuf> = None;
    let mut port: u16 = 1234;
    let mut context_size: u32 = 8192;
    let mut batch_size: u32 = 512;
    let mut ubatch_size: u32 = 512;
    let mut gpu_layers: u32 = 999;

    while let Some(flag) = args.next() {
        let value = args.next().with_context(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => model_path = Some(PathBuf::from(value)),
            "--backend" => backend_path = Some(PathBuf::from(value)),
            "--port" => port = value.parse().context("Invalid --port value")?,
            "--ctx" => context_size = value.parse().context("Invalid --ctx value")?,
            "--batch" => batch_size = value.parse().context("Invalid --batch value")?,
            "--ubatch" => ubatch_size = value.parse().context("Invalid --ubatch value")?,
            "--gpu-layers" => gpu_layers = value.parse().context("Invalid --gpu-layers value")?,
            _ => bail!("Unknown option '{flag}'"),
        }
    }

    let model_path = model_path.context("Missing required --model <path-to-GGUF>")?;
    if context_size == 0 || batch_size == 0 || ubatch_size == 0 {
        bail!("Context, batch, and ubatch sizes must be greater than zero");
    }

    let backend = discover_llama_server(backend_path.as_deref())?;
    let config = LlamaLaunchConfig {
        model_path,
        port,
        context_size,
        batch_size,
        ubatch_size,
        gpu_layers,
    };
    let status = launch_llama_server(&backend, &config)?;
    if !status.success() {
        bail!("llama-server exited with status {status}");
    }
    Ok(())
}
