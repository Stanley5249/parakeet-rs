//! melops-asr: Extensible ASR library with trait-based architecture.
//!
//! This crate provides high-level ASR pipelines while maintaining flexibility
//! for different model architectures (TDT, CDC, EOU, Whisper).
//!
//! # Architecture
//!
//! The library is built around three core traits:
//!
//! - [`traits::AudioPreprocessor`]: Converts raw audio to model-specific features
//! - [`traits::AsrModel`]: Runs inference on features to produce model output
//! - [`traits::Detokenizer`]: Converts model output to human-readable transcription
//!
//! # Quick Start
//!
//! ```ignore
//! use melops_asr::types::AudioBuffer;
//! use melops_asr::pipelines::ParakeetTdt;
//! use ort::session::Session;
//!
//! // Load audio
//! let audio = AudioBuffer::from_file("audio.wav")?;
//!
//! // Create pipeline
//! let builder = Session::builder()?;
//! let pipeline = ParakeetTdt::from_pretrained("model_dir", Some(builder))?;
//!
//! // Transcribe
//! let result = pipeline.transcribe(&audio)?;
//! println!("{}", result.text);
//! ```

pub mod chunk;
pub mod detokenizer;
pub mod error;
pub mod models;
pub mod pipelines;
pub mod preprocessor;
pub mod traits;
pub mod types;
