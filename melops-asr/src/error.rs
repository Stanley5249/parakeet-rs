//! Error types for melops-asr

use thiserror::Error;

/// ASR pipeline error variants
#[derive(Debug, Error)]
pub enum Error {
    /// Audio sample rate validation failed
    #[error("invalid sample rate: expected {expected}Hz, got {got}Hz")]
    InvalidSampleRate { expected: u32, got: u32 },

    /// Audio channel validation failed
    #[error("invalid audio channels: expected mono or stereo, got {0} channels")]
    InvalidChannels(u16),

    /// Model inference error
    #[error("model error: {0}")]
    Model(String),

    /// Audio preprocessing error
    #[error("preprocessing error: {0}")]
    Preprocessing(String),

    /// Audio chunking error
    #[error("chunking error: {0}")]
    Chunking(String),

    /// ONNX Runtime error
    #[error(transparent)]
    Ort(#[from] ort::Error),

    /// IO error
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// WAV file error
    #[error(transparent)]
    Hound(#[from] hound::Error),

    /// Upstream parakeet-rs error
    #[error(transparent)]
    ParakeetRs(#[from] parakeet_rs::Error),
}

/// Result type alias for melops-asr operations
pub type Result<T> = std::result::Result<T, Error>;
