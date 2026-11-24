use std::path::{Path, PathBuf};

/// Locate nvidia-smi using PATH. This is a diagnostic, not a CUDA capability test.
pub fn nvidia_smi_path() -> Option<PathBuf> {
    let executable = if cfg!(windows) { "nvidia-smi.exe" } else { "nvidia-smi" };
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .map(|dir| dir.join(executable))
        .find(|candidate| is_file(candidate))
}

fn is_file(path: &Path) -> bool {
    path.is_file()
}
