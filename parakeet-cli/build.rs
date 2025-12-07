// Build script for parakeet-cli
// Symlinks ONNX Runtime and OpenVINO DLLs from pixi environment when openvino feature is enabled

fn main() {
    // println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "openvino")]
    if let Err(e) = setup_openvino_ep() {
        println!("cargo:warning=Failed to setup OpenVINO EP: {}", e);
    }
}

#[cfg(feature = "openvino")]
fn setup_openvino_ep() -> std::io::Result<()> {
    use std::{env, fs, path::PathBuf};

    // // Find workspace root
    let workspace_root = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| env::current_dir().unwrap());

    // // ONNX Runtime DLLs from pixi Python package (onnxruntime-openvino)
    let onnxruntime_capi = workspace_root
        .join(".pixi")
        .join("envs")
        .join("default")
        .join("Lib")
        .join("site-packages")
        .join("onnxruntime")
        .join("capi");

    // // Target directory for symlinks
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let target_dir = out_dir.ancestors().nth(3).expect("Invalid OUT_DIR");
    fs::create_dir_all(target_dir)?;

    // // Required ONNX Runtime DLLs from Python package
    let onnxruntime_dlls = [
        "onnxruntime_providers_openvino.dll",
        #[cfg(target_os = "windows")]
        "onnxruntime_providers_shared.dll",
    ];

    // // Symlink ONNX Runtime DLLs
    for dll_name in &onnxruntime_dlls {
        let src = onnxruntime_capi.join(dll_name);
        let dst = target_dir.join(dll_name);

        if !src.exists() {
            println!("cargo:warning=DLL not found: {}", src.display());
            continue;
        }

        symlink_dll(&src, &dst)?;
    }

    Ok(())
}

#[allow(dead_code)]
fn symlink_dll(_src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    use std::fs;

    if dst.exists() || dst.symlink_metadata().is_ok() {
        let _ = fs::remove_file(dst);
    }

    #[cfg(all(feature = "openvino", target_os = "windows"))]
    std::os::windows::fs::symlink_file(_src, dst)?;

    #[cfg(all(feature = "openvino", target_os = "linux"))]
    std::os::unix::fs::symlink(_src, dst)?;

    Ok(())
}
