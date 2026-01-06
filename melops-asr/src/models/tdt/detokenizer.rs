//! Detokenizer for converting token IDs to text segments with timestamps.

use tokenizers::Tokenizer;

/// Token with timing information from TDT decoder.
#[derive(Clone, Debug)]
pub struct TokenDuration {
    /// Token ID (not blank)
    pub token_id: usize,
    /// Encoder frame index where token was emitted
    pub frame_index: usize,
    /// Duration prediction (number of frames to skip)
    pub duration: usize,
}

impl TokenDuration {
    pub fn new(token_id: usize, frame_index: usize, duration: usize) -> Self {
        Self {
            token_id,
            frame_index,
            duration,
        }
    }
}

/// Detokenizer for TDT models.
pub struct TdtDetokenizer {
    pub tokenizer: Tokenizer,
}

impl TdtDetokenizer {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}
