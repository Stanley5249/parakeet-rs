# parakeet-rs
[![Rust](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/parakeet-rs.svg)](https://crates.io/crates/parakeet-rs)

Fast speech recognition with NVIDIA's Parakeet models via ONNX Runtime.
Note: CoreML doesn't stable with this model - stick w/ CPU (or other GPU EP like CUDA). But its incredible fast in my Mac M3 16gb' CPU compared to Whisper metal! :-)

## Workspace Structure

This project is organized as a Cargo workspace:

- **`parakeet-rs/`** - Core library crate (no execution provider features)
- **`parakeet-cli/`** - CLI binary with configurable execution provider features

## Models

**CTC (English-only)**: Fast & accurate
```rust
use parakeet_rs::Parakeet;

let mut parakeet = Parakeet::from_pretrained(".", None)?;
let result = parakeet.transcribe_file("audio.wav")?;
println!("{}", result.text);

// Or transcribe in-memory audio
// let result = parakeet.transcribe_samples(audio, 16000, 1)?;

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

**TDT (Multilingual)**: 25 languages with auto-detection
```rust
use parakeet_rs::ParakeetTDT;

let mut parakeet = ParakeetTDT::from_pretrained("./tdt", None)?;
let result = parakeet.transcribe_file("audio.wav")?;
println!("{}", result.text);

// Or transcribe in-memory audio
// let result = parakeet.transcribe_samples(audio, 16000, 1)?;

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

**EOU (Streaming)**: Real-time ASR with end-of-utterance detection
```rust
use parakeet_rs::ParakeetEOU;

let mut parakeet = ParakeetEOU::from_pretrained("./eou", None)?;

// Prepare your audio (Vec<f32>, 16kHz mono, normalized)
let audio: Vec<f32> = /* your audio samples */;

// Process in 160ms chunks for streaming
const CHUNK_SIZE: usize = 2560; // 160ms at 16kHz
for chunk in audio.chunks(CHUNK_SIZE) {
    let text = parakeet.transcribe(chunk, false)?;
    print!("{}", text);
}
```

**Sortformer v2 & v2.1 (Speaker Diarization)**: Streaming 4-speaker diarization
```toml
parakeet-rs = { version = "0.2", features = ["sortformer"] }
```
```rust
use parakeet_rs::sortformer::{Sortformer, DiarizationConfig};

let mut sortformer = Sortformer::with_config(
    "diar_streaming_sortformer_4spk-v2.onnx", // or v2.1.onnx
    None,
    DiarizationConfig::callhome(),  // or dihard3(),custom()
)?;
let segments = sortformer.diarize(audio, 16000, 1)?;
for seg in segments {
    println!("Speaker {} [{:.2}s - {:.2}s]", seg.speaker_id, seg.start, seg.end);
}
```
See `examples/diarization.rs` for combining with TDT transcription.

## GPU Acceleration (Direct ONNX Runtime Control)

The library provides direct access to ONNX Runtime's `SessionBuilder`, giving you full control over execution providers and configuration:

```rust
use parakeet_rs::ParakeetTDT;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::execution_providers::{
    OpenVINOExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider,
};

// OpenVINO with GPU - specify device type and optimization level
let builder = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .with_execution_providers([
        OpenVINOExecutionProvider::default()
            .with_device_type("GPU")  // "CPU", "GPU", "GPU.0", "NPU", etc.
            .build(),
        CPUExecutionProvider::default().build(),
    ])?;

let mut model = ParakeetTDT::from_pretrained("./tdt", Some(builder))?;

// CUDA with specific device ID
let builder = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .with_execution_providers([
        CUDAExecutionProvider::default()
            .with_device_id(0)
            .build(),
        CPUExecutionProvider::default().build(),
    ])?;

let mut model = ParakeetTDT::from_pretrained("./tdt", Some(builder))?;
```

### Using the CLI

The CLI binary supports feature flags for different execution providers:

```bash
# Build with OpenVINO support
cargo build -p parakeet-cli --features openvino --release

# Build with CUDA support
cargo build -p parakeet-cli --features cuda --release

# Run
./target/release/parakeet audio.wav
```

Available features for `parakeet-cli`:
- `cpu` (default)
- `cuda`
- `tensorrt`
- `coreml`
- `directml`
- `rocm`
- `openvino`
- `webgpu`

## Setup

**CTC**: Download from [HuggingFace](https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main/onnx): `model.onnx`, `model.onnx_data`, `tokenizer.json`

**TDT**: Download from [HuggingFace](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx): `encoder-model.onnx`, `encoder-model.onnx.data`, `decoder_joint-model.onnx`, `vocab.txt`

**EOU**: Download from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx): `encoder.onnx`, `decoder_joint.onnx`, `tokenizer.json`

**Diarization (Sortformer v2 & v2.1)**: Download from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/tree/main): `diar_streaming_sortformer_4spk-v2.onnx` or `v2.1.onnx`.

Quantized versions available (int8). All files must be in the same directory.

## Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
parakeet-rs = "0.1"

# For CUDA support, add ort with cuda feature
ort = { version = "2.0.0-rc.10", features = ["cuda"] }
```

The library itself has no execution provider features - your application enables them via the `ort` dependency.

## Features

- [CTC: English with punctuation & capitalization](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [TDT: Multilingual (auto lang detection)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [EOU: Streaming ASR with end-of-utterance detection](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)
- [Sortformer v2 & v2.1: Streaming speaker diarization (up to 4 speakers)](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) NOTE: you can also download v2.1 model same way.
- Token-level timestamps (CTC, TDT)

## Notes

- Audio: 16kHz mono WAV (16-bit PCM or 32-bit float)
- CoreML currently unstable with this model - use CPU or other GPU providers

## License

Code: MIT OR Apache-2.0

FYI: The Parakeet ONNX models (downloaded separately from HuggingFace) are licensed under **CC-BY-4.0** by NVIDIA. This library does not distribute the models.
