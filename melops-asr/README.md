# melops-asr

Extensible ASR library with trait-based architecture for automatic speech recognition.

## Features

- **Type-safe audio input**: `AudioBuffer` validates 16kHz mono audio before processing
- **Automatic chunking**: Splits long audio into overlapping chunks (default: 60s with 1s overlap)
- **Overlap deduplication**: Smart merging of overlapping regions to avoid duplicate text
- **Timestamp adjustment**: Continuous timestamps across chunk boundaries
- **Trait-based extensibility**: Support for multiple model architectures (TDT, CDC, EOU, Whisper)

## Quick Start

```rust
use melops_asr::types::AudioBuffer;
use melops_asr::models::parakeet_tdt::ParakeetTdt;
use ort::session::Session;

// Load audio (validates 16kHz, converts stereo to mono)
let audio = AudioBuffer::from_file("audio.wav")?;

// Create pipeline with default settings
let builder = Session::builder()?;
let pipeline = ParakeetTdt::from_pretrained("model_dir", Some(builder))?;

// Transcribe
let result = pipeline.transcribe(&audio)?;
println!("{}", result.text);

// Access timestamps
for token in result.tokens {
    println!("[{:.2}s - {:.2}s] {}", token.start, token.end, token.text);
}
```

## GPU Acceleration

```rust
use ort::execution_providers::{CUDAExecutionProvider, CPUExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;

let builder = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .with_execution_providers([
        CUDAExecutionProvider::default().with_device_id(0).build(),
        CPUExecutionProvider::default().build(),
    ])?;

let pipeline = ParakeetTdt::from_pretrained(model_dir, Some(builder))?;
```

## Chunked Transcription (Long Audio)

For audio longer than 60 seconds, use `transcribe_chunked`:

```rust
use melops_asr::chunk::ChunkConfig;

let audio = AudioBuffer::from_file("long_audio.wav")?;

let chunk_config = ChunkConfig {
    duration_sec: 60.0,  // 60 second chunks
    overlap_sec: 1.0,    // 1 second overlap
};

let result = pipeline.transcribe_chunked(&audio, Some(chunk_config))?;
```

## Architecture

### Trait Hierarchy

The library is built around three core traits:

- **`AudioPreprocessor`**: Converts raw audio to model-specific features (mel spectrograms)
- **`AsrModel`**: Runs inference on features to produce model output (tokens, timestamps)
- **`Detokenizer`**: Converts model output to human-readable transcription

### Type Safety

`AudioBuffer` ensures audio data is properly validated:

```rust
// From file (validates sample rate, converts stereo)
let audio = AudioBuffer::from_file("audio.wav")?;

// From raw samples (validates and converts)
let audio = AudioBuffer::from_wav(samples, sample_rate, channels)?;

// Pre-validated 16kHz mono (no validation)
let audio = AudioBuffer::from_raw_mono(samples);

// Slice by time
let segment = audio.slice_time(10.0, 20.0);
```

## Chunking Strategy

1. **Split**: Audio divided into overlapping chunks

   ```
   Chunk 0: [0s -------- 60s]
   Chunk 1:       [59s -------- 119s]
   Chunk 2:              [118s -------- 178s]
   ```

2. **Transcribe**: Each chunk processed independently

3. **Adjust**: Timestamps offset by chunk start time

4. **Deduplicate**: Overlapping regions merged (skips tokens in overlap zone)

5. **Merge**: Produces continuous token stream with seamless timestamps

## Module Structure

```
melops-asr/src/
├── lib.rs          # Module exports and documentation
├── error.rs        # thiserror-based Error enum
├── types.rs        # AudioBuffer, Token, Transcription
├── chunk.rs        # ChunkConfig, split_audio
├── traits.rs       # AudioPreprocessor, AsrModel, Detokenizer
└── models/
    ├── mod.rs
    └── parakeet_tdt.rs  # ParakeetTdt pipeline
```

## Error Handling

```rust
use melops_asr::error::{Error, Result};

match AudioBuffer::from_wav(samples, 44100, 1) {
    Err(Error::InvalidSampleRate { expected, got }) => {
        eprintln!("Expected {}Hz, got {}Hz", expected, got);
    }
    Err(Error::InvalidChannels(n)) => {
        eprintln!("Unsupported channel count: {}", n);
    }
    Ok(audio) => { /* proceed */ }
}
```

## Dependencies

- `parakeet-rs`: Core ASR model inference
- `hound`: WAV audio I/O
- `ort`: ONNX Runtime bindings
- `thiserror`: Error handling
- `tracing`: Logging

## Future Enhancements

- CDC (Conformer-CTC) model support
- EOU (End-of-Utterance) streaming model support
- Whisper model support
- Parallel chunk processing
- Smarter deduplication (Levenshtein distance, word-level alignment)
