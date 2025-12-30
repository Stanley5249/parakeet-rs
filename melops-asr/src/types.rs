//! Core types for melops-asr

/// Token with timestamp information (start and end in seconds)
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// Transcription result with text and timestamped tokens
#[derive(Debug, Clone)]
pub struct Transcription {
    pub text: String,
    pub tokens: Vec<Token>,
}
