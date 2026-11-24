use anyhow::{bail, Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};

#[derive(Debug, Clone)]
pub struct BackendPaths {
    pub server_exe: PathBuf,
    pub cuda_vendor_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct LlamaLaunchConfig {
    pub model_path: PathBuf,
    pub port: u16,
    pub context_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub gpu_layers: u32,
}

/// Locate llama-server. An explicit path or NVR_LLAMA_SERVER takes priority;
/// otherwise discover an LM Studio CUDA 13 backend in the current user's profile.
pub fn discover_llama_server(explicit_path: Option<&Path>) -> Result<BackendPaths> {
    let requested = explicit_path
        .map(Path::to_path_buf)
        .or_else(|| env::var_os("NVR_LLAMA_SERVER").map(PathBuf::from));

    let server_exe = if let Some(path) = requested {
        normalize_server_path(path)?
    } else {
        discover_lm_studio_cuda13_backend()?
    };

    let vendor_override = env::var_os("NVR_CUDA_VENDOR_DIR").map(PathBuf::from);
    let default_vendor = server_exe
        .parent()
        .and_then(Path::parent)
        .map(|backends_dir| {
            backends_dir
                .join("vendor")
                .join("win-llama-cuda13-vendor-v1")
        });

    let cuda_vendor_dir = vendor_override
        .filter(|path| path.is_dir())
        .or_else(|| default_vendor.filter(|path| path.is_dir()));

    Ok(BackendPaths {
        server_exe,
        cuda_vendor_dir,
    })
}

fn normalize_server_path(path: PathBuf) -> Result<PathBuf> {
    let server = if path.is_dir() {
        path.join(server_executable_name())
    } else {
        path
    };

    if !server.is_file() {
        bail!(
            "llama-server executable was not found at '{}'. Pass --backend with the full path to llama-server.exe.",
            server.display()
        );
    }
    Ok(server)
}

fn server_executable_name() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

fn discover_lm_studio_cuda13_backend() -> Result<PathBuf> {
    let user_profile = env::var_os("USERPROFILE")
        .or_else(|| env::var_os("HOME"))
        .context("Could not determine the user profile directory")?;

    let backends_dir = PathBuf::from(user_profile)
        .join(".lmstudio")
        .join("extensions")
        .join("backends");

    let entries = fs::read_dir(&backends_dir).with_context(|| {
        format!(
            "Could not read LM Studio backends directory '{}'. Install the LM Studio CUDA 13 llama.cpp runtime, or pass --backend with the full path to llama-server.exe.",
            backends_dir.display()
        )
    })?;

    let mut candidates: Vec<(String, PathBuf)> = entries
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            let path = entry.path();
            if name.starts_with("llama.cpp-")
                && name.contains("cuda13")
                && path.is_dir()
            {
                let server = path.join(server_executable_name());
                if server.is_file() {
                    return Some((name, server));
                }
            }
            None
        })
        .collect();

    // Prefer the lexicographically newest versioned backend directory.
    candidates.sort_by(|left, right| right.0.cmp(&left.0));
    candidates
        .into_iter()
        .next()
        .map(|(_, path)| path)
        .context("No LM Studio CUDA 13 llama-server.exe was found. Pass --backend with its full path.")
}

/// Start llama-server with the NVR Qwen benchmark preset.
/// The server is deliberately bound to loopback, not exposed to the network.
pub fn launch_llama_server(
    backend: &BackendPaths,
    config: &LlamaLaunchConfig,
) -> Result<ExitStatus> {
    if !config.model_path.is_file() {
        bail!(
            "Model file does not exist: '{}'. Use --model with the full path to your downloaded GGUF file.",
            config.model_path.display()
        );
    }
    if !backend.server_exe.is_file() {
        bail!(
            "llama-server executable does not exist: '{}'",
            backend.server_exe.display()
        );
    }

    let mut command = build_llama_server_command(backend, config)?;
    println!("Starting llama.cpp server...");
    println!("Model: {}", config.model_path.display());
    println!("Backend: {}", backend.server_exe.display());
    println!("GPU layers: {}", config.gpu_layers);
    println!("Context: {}", config.context_size);
    println!("Logical batch: {}", config.batch_size);
    println!("Physical batch (ubatch): {}", config.ubatch_size);
    println!("Flash Attention: on");
    println!("KV cache K/V: f16");
    println!("Local endpoint: http://127.0.0.1:{}/", config.port);
    println!("Press Ctrl+C in this window to stop the server.\n");

    command
        .status()
        .with_context(|| format!("Failed to start '{}'", backend.server_exe.display()))
}

fn build_llama_server_command(
    backend: &BackendPaths,
    config: &LlamaLaunchConfig,
) -> Result<Command> {
    let mut command = Command::new(&backend.server_exe);
    command
        .arg("--model")
        .arg(&config.model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--n-gpu-layers")
        .arg(config.gpu_layers.to_string())
        .arg("--ctx-size")
        .arg(config.context_size.to_string())
        .arg("--batch-size")
        .arg(config.batch_size.to_string())
        .arg("--ubatch-size")
        .arg(config.ubatch_size.to_string())
        .arg("--flash-attn")
        .arg("on")
        .arg("--cache-type-k")
        .arg("f16")
        .arg("--cache-type-v")
        .arg("f16");

    if let Some(vendor_dir) = &backend.cuda_vendor_dir {
        let mut search_paths = vec![vendor_dir.clone()];
        if let Some(existing_path) = env::var_os("PATH") {
            search_paths.extend(env::split_paths(&existing_path));
        }
        let joined_path = env::join_paths(search_paths)
            .context("Could not construct PATH with the CUDA vendor directory")?;
        command.env("PATH", joined_path);
    }

    Ok(command)
}

/// Locate nvidia-smi using PATH. This is a diagnostic, not a CUDA capability test.
pub fn nvidia_smi_path() -> Option<PathBuf> {
    let executable = if cfg!(windows) { "nvidia-smi.exe" } else { "nvidia-smi" };
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|dir| dir.join(executable))
        .find(|candidate| candidate.is_file())
}

/// Spawn the backend without blocking. Kept available for future supervised API mode.
#[allow(dead_code)]
fn spawn_backend(command: &mut Command) -> Result<Child> {
    command.spawn().context("Failed to spawn inference backend")
}
