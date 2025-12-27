//! Detokenizer for converting model output to transcriptions.

use crate::error::Result;
use crate::traits::Detokenizer;
use crate::types::{Token, Transcription};
use parakeet_rs::Vocabulary;
use parakeet_rs::decoder_tdt::ParakeetTDTDecoder;

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
    decoder: ParakeetTDTDecoder,
    vocabulary: Vocabulary,
    hop_length: usize,
    sample_rate: usize,
}

impl SentencePieceDetokenizer {
    /// Create detokenizer from pretrained vocabulary.
    pub fn from_pretrained(vocab: Vocabulary, hop_length: usize, sample_rate: usize) -> Self {
        let decoder = ParakeetTDTDecoder::from_vocab(vocab.clone());
        Self {
            decoder,
            vocabulary: vocab,
            hop_length,
            sample_rate,
        }
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.size()
    }
}

impl Detokenizer for SentencePieceDetokenizer {
    type Input = TdtOutput;

    fn decode(&self, input: &Self::Input) -> Result<Transcription> {
        let result = self.decoder.decode_with_timestamps(
            &input.tokens,
            &input.frame_indices,
            &input.durations,
            self.hop_length,
            self.sample_rate,
        )?;

        Ok(Transcription {
            text: result.text,
            tokens: result
                .tokens
                .into_iter()
                .map(|t| Token {
                    text: t.text,
                    start: t.start,
                    end: t.end,
                })
                .collect(),
        })
    }
}
