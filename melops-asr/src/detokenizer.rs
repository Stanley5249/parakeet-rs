//! Detokenizer for converting model output to transcriptions.

use crate::error::{DetokenizationError, Result};
use crate::traits::Detokenizer;
use crate::types::{Token, Transcription};
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

/// SentencePiece-based detokenizer for TDT models.
pub struct SentencePieceDetokenizer {
    pub tokenizer: Tokenizer,
    /// Duration of one encoder frame in seconds.
    /// For TDT: 8 mel frames/encoder frame * (160 samples/mel frame / 16000 Hz) = 0.08s (80ms)
    pub encoder_frame_duration: f32,
}

impl SentencePieceDetokenizer {
    /// Create detokenizer from tokenizer and encoder frame duration.
    ///
    /// # Arguments
    /// - `tokenizer`: HuggingFace tokenizer
    /// - `encoder_frame_duration`: Duration of one encoder frame in seconds
    pub fn new(tokenizer: Tokenizer, encoder_frame_duration: f32) -> Self {
        Self {
            tokenizer,
            encoder_frame_duration,
        }
    }

    /// Create detokenizer for TDT models from tokenizer and audio config.
    ///
    /// # Arguments
    /// - `tokenizer`: HuggingFace tokenizer
    /// - `hop_length`: Audio samples per mel frame (typically 160)
    /// - `sample_rate`: Audio sample rate in Hz (typically 16000)
    ///
    /// # TDT Architecture
    /// TDT encoder applies 8x subsampling to the mel spectrogram.
    /// Encoder frame duration = `(8 * hop_length) / sample_rate`
    ///
    /// Standard config: `(8 * 160) / 16000 = 0.08` seconds (80ms)
    pub fn for_tdt(tokenizer: Tokenizer, hop_length: usize, sample_rate: usize) -> Self {
        const ENCODER_STRIDE: usize = 8; // TDT encoder applies 8x subsampling
        let encoder_frame_duration = (ENCODER_STRIDE * hop_length) as f32 / sample_rate as f32;
        Self::new(tokenizer, encoder_frame_duration)
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Convert encoder frame index to timestamp in seconds.
    #[inline]
    fn frame_to_timestamp(&self, encoder_frame: usize) -> f32 {
        encoder_frame as f32 * self.encoder_frame_duration
    }

    /// Merge two token vectors, handling overlap deduplication.
    fn merge_chunk_tokens(
        mut existing: Vec<Token>,
        new_tokens: Vec<Token>,
        overlap_sec: f32,
    ) -> Vec<Token> {
        if existing.is_empty() {
            return new_tokens;
        }

        if new_tokens.is_empty() {
            return existing;
        }

        // Find the last timestamp in accumulated tokens
        let existing_end = existing.last().map(|t| t.end).unwrap_or(0.0);
        let overlap_start = existing_end - overlap_sec;

        // Skip tokens in the overlap region (before midpoint of overlap)
        let new_start_idx = new_tokens
            .iter()
            .position(|t| t.start >= overlap_start + (overlap_sec * 0.5))
            .unwrap_or(0);

        existing.extend_from_slice(&new_tokens[new_start_idx..]);
        existing
    }
}

impl Detokenizer for SentencePieceDetokenizer {
    type Input = Vec<TokenDuration>;

    fn decode(&self, input: &Self::Input) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();

        for (i, td) in input.iter().enumerate() {
            let token_text = self
                .tokenizer
                .id_to_token(td.token_id as u32)
                .ok_or(DetokenizationError::InvalidTokenId(td.token_id))?;

            let start = self.frame_to_timestamp(td.frame_index);
            let end = if let Some(next) = input.get(i + 1) {
                self.frame_to_timestamp(next.frame_index)
            } else {
                start + self.encoder_frame_duration
            };

            let text = token_text.replace('▁', " ");

            tokens.push(Token { text, start, end });
        }

        Ok(tokens)
    }

    fn merge_tokens<I>(token_chunks: I, overlap_sec: f32) -> Vec<Token>
    where
        I: IntoIterator<Item = Vec<Token>>,
    {
        token_chunks
            .into_iter()
            .fold(Vec::new(), |acc, new_tokens| {
                Self::merge_chunk_tokens(acc, new_tokens, overlap_sec)
            })
    }

    fn build_transcription(tokens: Vec<Token>) -> Transcription {
        let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
        Transcription {
            text: text.trim().to_string(),
            tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_empty_existing() {
        let result = SentencePieceDetokenizer::merge_tokens(
            [
                vec![],
                vec![Token {
                    text: "hello".to_string(),
                    start: 0.0,
                    end: 1.0,
                }],
            ],
            1.0,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "hello");
    }

    #[test]
    fn merge_empty_new() {
        let result = SentencePieceDetokenizer::merge_tokens(
            [
                vec![Token {
                    text: "hello".to_string(),
                    start: 0.0,
                    end: 1.0,
                }],
                vec![],
            ],
            1.0,
        );

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn merge_with_overlap() {
        let result = SentencePieceDetokenizer::merge_tokens(
            [
                vec![
                    Token {
                        text: "hello".to_string(),
                        start: 0.0,
                        end: 1.0,
                    },
                    Token {
                        text: " world".to_string(),
                        start: 1.0,
                        end: 2.0,
                    },
                ],
                vec![
                    Token {
                        text: " world".to_string(),
                        start: 1.5,
                        end: 2.0,
                    },
                    Token {
                        text: " test".to_string(),
                        start: 2.5,
                        end: 3.0,
                    },
                ],
            ],
            1.0,
        );

        // existing_end = 2.0, overlap_start = 1.0
        // threshold = overlap_start + (overlap_sec * 0.5) = 1.0 + 0.5 = 1.5
        // new_tokens[0].start = 1.5 >= 1.5, so it's included
        // Result: existing (2) + new_tokens from index 0 (2) = 4 tokens
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[1].text, " world");
        assert_eq!(result[2].text, " world");
        assert_eq!(result[3].text, " test");
    }
}
