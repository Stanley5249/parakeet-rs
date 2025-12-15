// Build script for parakeet-cli
// Symlinks ONNX Runtime and Executor Provider DLLs from Conda enviroment to target dir
#![allow(dead_code)]

use std::env;
use std::path::PathBuf;

#[cfg(target_os = "windows")]
const ORT_DYLIBS: [&str; 2] = [
    "onnxruntime_providers_openvino.dll",
    "onnxruntime_providers_shared.dll",
];

#[cfg(target_os = "linux")]
const ORT_DYLIBS: [&str; 1] = ["libonnxruntime_providers_openvino.so"];

fn main() {
    // println!("cargo:rerun-if-changed=build.rs");

    #[cfg(any(feature = "openvino"))]
    if let Err(e) = symlink_ort_providers() {
        println!(
            "cargo:error=Failed to symlink ONNX Runtime providers: {}",
            e
        );
    }
}

fn get_target_dir() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    out_dir
        .ancestors()
        .nth(3)
        .expect("Invalid OUT_DIR")
        .to_path_buf()
}

fn symlink_ort_providers() -> std::io::Result<()> {
    // ONNX Runtime providers DLLs from conda environment

    let conda_prefix = PathBuf::from(env::var("CONDA_PREFIX").expect("CONDA_PREFIX not set"));

    let ort_dir = conda_prefix
        .join("Lib")
        .join("site-packages")
        .join("onnxruntime")
        .join("capi");

    let target_dir = get_target_dir();

    for dylib in ORT_DYLIBS {
        let src = ort_dir.join(dylib);
        let dst = target_dir.join(dylib);
        symlink_dll(&src, &dst)?;
    }

    println!("cargo:rerun-if-changed={}", ort_dir.display());
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    Ok(())
}

fn symlink_dll(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    std::os::windows::fs::symlink_file(src, dst)?;

    #[cfg(target_os = "linux")]
    std::os::unix::fs::symlink(src, dst)?;

    Ok(())
}
