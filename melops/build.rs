//! Build script for melops
//! Symlinks ONNX Runtime and Executor Provider DLLs from Conda environment to target dir
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Error type that does not escape string in error messages
#[derive(Debug)]
struct BuildErr(String);

impl std::fmt::Display for BuildErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for BuildErr {}

fn get_target_dir() -> Result<PathBuf> {
    let out_dir = match std::env::var("OUT_DIR") {
        Ok(var) => PathBuf::from(var),
        Err(e) => {
            println!("cargo::error=environment variable `OUT_DIR` should be set during build");
            return Err(e.into());
        }
    };

    match out_dir.ancestors().nth(3) {
        Some(dir) => Ok(dir.to_path_buf()),
        None => {
            let msg = format!(
                "could not determine target directory from OUT_DIR={:?}",
                out_dir.display()
            );
            println!("cargo::error={}", msg);
            Err(BuildErr(msg).into())
        }
    }
}

fn get_conda_prefix() -> Result<PathBuf> {
    match std::env::var("CONDA_PREFIX") {
        Ok(var) => Ok(PathBuf::from(var)),
        Err(e) => {
            println!(
                "cargo::error=environment variable `CONDA_PREFIX` should be set when using conda environments"
            );
            Err(e.into())
        }
    }
}

#[allow(unused)]
fn symlink_lib(ort_dir: &Path, target_dir: &Path, lib_name: &str) -> Result<()> {
    let src = ort_dir.join(lib_name);
    let dst = target_dir.join(lib_name);

    if !src.exists() {
        let msg = format!("could not find library at {:?}", src.display());
        println!("cargo::error={}", msg);
        return Err(BuildErr(msg).into());
    }

    if dst.exists()
        && let Ok(()) = std::fs::remove_file(&dst)
    {
        println!(
            "cargo::warning=remove existing library symlink at {:?}",
            dst.display()
        );
    }

    #[cfg(target_os = "windows")]
    std::os::windows::fs::symlink_file(&src, &dst)?;

    #[cfg(target_os = "linux")]
    std::os::unix::fs::symlink(&src, &dst)?;

    Ok(())
}

#[allow(unused_variables)]
fn main() -> Result<()> {
    let target_dir = get_target_dir()?;

    let conda_prefix = get_conda_prefix()?;
    println!("cargo::rerun-if-env-changed=CONDA_PREFIX");

    #[cfg(target_os = "windows")]
    let ort_dir = conda_prefix.join("Lib\\site-packages\\onnxruntime\\capi");

    #[cfg(target_os = "linux")]
    let ort_dir = conda_prefix.join("lib/python3.13/site-packages/onnxruntime/capi");

    #[cfg(feature = "openvino")]
    {
        #[cfg(target_os = "windows")]
        let libs = [
            "onnxruntime_providers_shared.dll",
            "onnxruntime_providers_openvino.dll",
        ];

        #[cfg(target_os = "linux")]
        let libs = [
            "libonnxruntime_providers_shared.so",
            "libonnxruntime_providers_openvino.so",
        ];

        for lib in &libs {
            symlink_lib(&ort_dir, &target_dir, lib)?;
        }
    }

    Ok(())
}
