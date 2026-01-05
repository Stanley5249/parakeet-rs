//! Detokenizer for converting model output to transcriptions.

use crate::error::{DetokenizationError, Result};
use crate::traits::AsrModel;
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

use crate::models::tdt::TdtModel;

impl TdtModel {
    /// Decode model output to timestamped tokens.
    pub fn decode(&self, token_durations: &[TokenDuration]) -> Result<Vec<Token>> {
        let mut tokens = Vec::with_capacity(token_durations.len());

        for td in token_durations {
            let text = self
                .detokenizer
                .tokenizer
                .id_to_token(td.token_id as u32)
                .ok_or(DetokenizationError::InvalidTokenId(td.token_id))?;

            let start = self.frame_to_secs(td.frame_index);
            let end = self.frame_to_secs(td.frame_index + td.duration);

            let text = text.replace('▁', " ");

            tokens.push(Token { text, start, end });
        }

        Ok(tokens)
    }

    /// Merge model outputs with frame-based overlap detection.
    pub fn merge_outputs<I>(chunks: I) -> Vec<TokenDuration>
    where
        I: IntoIterator<Item = Vec<TokenDuration>>,
    {
        chunks.into_iter().fold(Vec::new(), Self::merge_two_outputs)
    }

    fn merge_two_outputs(
        mut chunk1: Vec<TokenDuration>,
        chunk2: Vec<TokenDuration>,
    ) -> Vec<TokenDuration> {
        if chunk2.is_empty() {
            return chunk1;
        }

        // Find where chunk1 ends (in frames)
        let last_token = match chunk1.last() {
            Some(td) => td,
            None => return chunk2,
        };

        let chunk1_end_frame = last_token.frame_index + last_token.duration;

        if let Some(i) = chunk2
            .iter()
            .position(|td| td.frame_index >= chunk1_end_frame)
        {
            chunk1.extend_from_slice(&chunk2[i..]);
        }

        chunk1
    }

    /// Build transcription from tokens.
    pub fn build_transcription(tokens: Vec<Token>) -> Transcription {
        let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
        Transcription { text, tokens }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_empty_chunks() {
        let chunk1 = vec![];
        let chunk2 = vec![TokenDuration {
            token_id: 1,
            frame_index: 0,
            duration: 10,
        }];

        let result = TdtModel::merge_outputs([chunk1, chunk2]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].token_id, 1);
    }

    #[test]
    fn merges_with_frame_overlap() {
        let chunk1 = vec![TokenDuration::new(1, 0, 10), TokenDuration::new(2, 10, 10)];

        let chunk2 = vec![
            TokenDuration::new(2, 15, 5),
            TokenDuration::new(3, 20, 5),
            TokenDuration::new(4, 25, 5),
        ];

        let result = TdtModel::merge_outputs([chunk1, chunk2]);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0].frame_index, 0);
        assert_eq!(result[1].frame_index, 10);
        assert_eq!(result[2].frame_index, 20);
        assert_eq!(result[3].frame_index, 25);
    }

    #[test]
    fn merges_at_boundary() {
        let chunk1 = vec![TokenDuration::new(1, 0, 10)];

        let chunk2 = vec![TokenDuration::new(1, 10, 10), TokenDuration::new(2, 20, 5)];

        let result = TdtModel::merge_outputs([chunk1, chunk2]);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].frame_index, 0);
        assert_eq!(result[1].frame_index, 10);
        assert_eq!(result[2].frame_index, 20);
    }
}
