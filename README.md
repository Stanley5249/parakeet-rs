# Melops 🦜

Fast audio captioning toolkit with NVIDIA's Parakeet models via ONNX Runtime.

This is a fork of [altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs), restructured as a Cargo workspace with additional tooling.

## Workspace Structure

This project is organized as a Cargo workspace:

- `parakeet-rs/` - Core ASR library crate (no execution provider features)
- `melops/` - CLI binary `mel` with configurable execution provider features
- `melops-dl/` - Minimal yt-dlp wrapper for downloading and organizing audio by metadata

## Quick Start

Assumes you understand `cargo` and `pixi` basics.

```bash
# Generate captions from local audio file
pixi run mel cap audio.wav
# Or run cargo with selected package (-p <package>)
pixi run cargo run -p melops cap audio.wav

# Download and generate captions from URL
pixi run mel dl "https://youtu.be/jNQXAC9IVRw"

# With OpenVINO (-e <environment> sets feature flag)
pixi run -e openvino mel cap audio.wav
# Or run with feature flag (-F <features...>)
pixi run -e openvino cargo run -p melops -F openvino cap audio.wav
```

## ONNX Runtime

ONNX Runtime supports various _Execution Providers_ for hardware acceleration.

Using pykeio's `ort` crates, the `onnxruntime` library is downloaded from their CDN by default and linked automatically, with popular providers enabled. CUDA, TensorRT, DirectML, CoreML, XNNPACK, and WebGPU work out of the box. See [Execution providers | ort](https://ort.pyke.io/perf/execution-providers) for details.

ONNX Runtime uses two strategies to link execution providers: static and shared.

On conda-forge, `onnxruntime-cpp` is built with CPU execution provider only, which limits its usefulness.

On PyPI, several packages are available: `onnxruntime` (CPU), `onnxruntime-gpu` (CUDA and TensorRT), `onnxruntime-directml` (DirectML), `onnxruntime-openvino` (OpenVINO), and more. These distribute `onnxruntime.dll` rather than `onnxruntime.lib`, since Python loads dynamic libraries at runtime. Shared EPs include `onnxruntime_providers_shared.dll` and `onnxruntime_providers_<ep>.dll` in the `capi` folder.

This project provides OpenVINO support via `onnxruntime-openvino` and `openvino` packages from PyPI and conda-forge, managed by pixi. On Windows:

1. OpenVINO is a shared EP, so it works with any onnxruntime build. We use the pykeio CDN version by default. The PyPI package includes `onnxruntime_providers_shared.dll` and `onnxruntime_providers_openvino.dll`, which must be copied near the executable since shared libraries load at runtime. For static EPs, finding prebuilt `onnxruntime.lib` is difficult; you may need to build and link manually. See [Linking | ort](https://ort.pyke.io/setup/linking).

2. OpenVINO itself is distributed on both PyPI and conda-forge. On Windows, the NPU plugin is unavailable on conda-forge, so the PyPI version may be preferred. On Linux, the conda-forge version is recommended. The `openvino.dll` and plugins are in the system path within the conda environment and loaded at runtime. On PyPI, they're in the `libs` folder; add this to your system path.

For other EPs, see [Execution providers | onnxruntime](https://onnxruntime.ai/docs/execution-providers/).

## License

MIT
