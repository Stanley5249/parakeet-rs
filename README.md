# Melops 🦜

Fast speech recognition toolkit with NVIDIA's Parakeet models via ONNX Runtime.

This is a fork of [altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs), restructured as a Cargo workspace with additional tooling.

## Workspace Structure

This project is organized as a Cargo workspace:

- `parakeet-rs/` - Core library crate (no execution provider features)
- `melops-cli/` - CLI binary with configurable execution provider features
- `melops-dl/` - Minimal yt-dlp wrapper for downloading and organizing audio by metadata

## Quick Start

Assumes you understand `cargo` and `pixi` basics.

```bash
# Using pixi task
pixi run melops audio.wav
# Or run cargo with selected package (-p <package>)
pixi run cargo run -p melops-cli

# With GPU acceleration (-e <environment> sets feature flag)
pixi run -e openvino melops audio.wav
# Or run with feature flag (-F <features...>)
pixi run -e openvino cargo run -p melops-cli -F openvino
```

## License

### Code

MIT OR Apache-2.0

### Models

The Parakeet ONNX models (downloaded separately from HuggingFace) are licensed under **CC-BY-4.0** by NVIDIA. This workspace does not distribute the models.
