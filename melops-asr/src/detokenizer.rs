//! Detokenizer for converting model output to transcriptions.

use crate::error::{Error, Result};
use crate::traits::Detokenizer;
use crate::types::{Token, Transcription};
use parakeet_rs::Vocabulary;

/// Output from TDT model forward pass.
#[derive(Debug, Clone)]
pub struct TdtOutput {
    /// Decoded token IDs
    pub tokens: Vec<usize>,
    /// Frame indices for each token
    pub frame_indices: Vec<usize>,
    /// Duration of each token in frames
    pub durations: Vec<usize>,
}

/// SentencePiece-based detokenizer for TDT models.
pub struct SentencePieceDetokenizer {
    pub vocabulary: Vocabulary,
    /// Duration of one encoder frame in seconds.
    /// For TDT: 8 mel frames/encoder frame * (160 samples/mel frame / 16000 Hz) = 0.08s (80ms)
    encoder_frame_duration: f32,
}

impl SentencePieceDetokenizer {
    /// Create detokenizer from vocabulary and encoder frame duration.
    ///
    /// # Arguments
    /// - `vocabulary`: SentencePiece vocabulary
    /// - `encoder_frame_duration`: Duration of one encoder frame in seconds
    pub fn new(vocabulary: Vocabulary, encoder_frame_duration: f32) -> Self {
        Self {
            vocabulary,
            encoder_frame_duration,
        }
    }

    /// Create detokenizer for TDT models from vocabulary and audio config.
    ///
    /// # Arguments
    /// - `vocabulary`: SentencePiece vocabulary
    /// - `hop_length`: Audio samples per mel frame (typically 160)
    /// - `sample_rate`: Audio sample rate in Hz (typically 16000)
    ///
    /// # TDT Architecture
    /// TDT encoder applies 8x subsampling to the mel spectrogram.
    /// Encoder frame duration = `(8 * hop_length) / sample_rate`
    ///
    /// Standard config: `(8 * 160) / 16000 = 0.08` seconds (80ms)
    pub fn for_tdt(vocabulary: Vocabulary, hop_length: usize, sample_rate: usize) -> Self {
        const ENCODER_STRIDE: usize = 8; // TDT encoder applies 8x subsampling
        let encoder_frame_duration = (ENCODER_STRIDE * hop_length) as f32 / sample_rate as f32;
        Self::new(vocabulary, encoder_frame_duration)
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.size()
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
    type Input = TdtOutput;

    fn decode(&self, input: &Self::Input) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();

        for (i, &token_id) in input.tokens.iter().enumerate() {
            let token_text = self
                .vocabulary
                .id_to_text(token_id)
                .ok_or(Error::InvalidTokenId(token_id))?;

            // Calculate token timestamp from encoder frame index
            let start = self.frame_to_timestamp(input.frame_indices[i]);
            let end = if let Some(&next_frame) = input.frame_indices.get(i + 1) {
                self.frame_to_timestamp(next_frame)
            } else {
                // Last token: assume 1 encoder frame duration
                start + self.encoder_frame_duration
            };

            // Handle SentencePiece format (▁ prefix for word start)
            let text = token_text.replace('▁', " ");

            // Skip special tokens (but keep <unk>)
            if !(token_text.starts_with('<') && token_text.ends_with('>') && token_text != "<unk>")
            {
                tokens.push(Token { text, start, end });
            }
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
