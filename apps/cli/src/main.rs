use std::process::Command;

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("doctor") => doctor(),
        Some("version") | Some("--version") | Some("-V") => {
            println!("NVR {}", env!("CARGO_PKG_VERSION"));
        }
        _ => {
            eprintln!("NVR — NVIDIA Velocity Runtime");
            eprintln!("Usage: nvr <doctor|version>");
            std::process::exit(2);
        }
    }
}

fn doctor() {
    println!("NVR environment diagnostic");
    match nvr_runtime::nvidia_smi_path() {
        Some(path) => {
            println!("nvidia-smi: found at {}", path.display());
            match Command::new(&path).arg("--query-gpu=name,memory.total,driver_version")
                .arg("--format=csv,noheader").output() {
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
    println!("Note: this checks driver visibility only; CUDA backend/model loading is not implemented yet.");
}
