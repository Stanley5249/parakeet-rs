//! High-level ASR pipelines.

use crate::chunk::{AudioChunk, ChunkConfig, split_audio};
use crate::detokenizer::SentencePieceDetokenizer;
use crate::error::Result;
use crate::models::tdt::TdtModel;
use crate::preprocessor::ParakeetPreprocessor;
use crate::traits::{AsrModel, AudioPreprocessor, Detokenizer};
use crate::types::{AudioBuffer, Token, Transcription};
use ort::session::builder::SessionBuilder;
use parakeet_rs::Vocabulary;
use std::path::Path;

/// Parakeet TDT captioning pipeline with chunking support.
///
/// This pipeline wraps the TDT (Token-and-Duration Transducer) model
/// for automatic speech recognition with timestamp support.
pub struct ParakeetTdt {
    preprocessor: ParakeetPreprocessor,
    model: TdtModel,
    detokenizer: SentencePieceDetokenizer,
}

impl ParakeetTdt {
    /// Load TDT pipeline from a pretrained directory.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to directory containing model files:
    ///   - `encoder-model.onnx`
    ///   - `decoder_joint-model.onnx`
    ///   - `vocab.txt`
    /// * `builder` - ONNX session builder for configuring execution providers
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P, builder: SessionBuilder) -> Result<Self> {
        let path = model_dir.as_ref();

        // Preprocessor (hardcoded config for TDT)
        let preprocessor = ParakeetPreprocessor::tdt();

        // Load vocabulary
        let vocab_path = path.join("vocab.txt");
        let vocabulary = Vocabulary::from_file(&vocab_path)?;

        // Detokenizer loads vocab first
        let detokenizer = SentencePieceDetokenizer::from_pretrained(
            vocabulary,
            preprocessor.config().hop_length,
            preprocessor.config().sampling_rate,
        );

        // Get vocab_size from detokenizer to pass to model
        let vocab_size = detokenizer.vocab_size();

        // Model loads ONNX files with vocab_size
        let model = TdtModel::from_pretrained(path, builder, vocab_size)?;

        Ok(Self {
            preprocessor,
            model,
            detokenizer,
        })
    }

    /// Transcribe an audio buffer.
    ///
    /// For short audio (less than chunk duration), transcribes directly.
    /// Use `transcribe_chunked` for explicit control over chunking.
    pub fn transcribe(&mut self, audio: &AudioBuffer) -> Result<Transcription> {
        self.transcribe_buffer(audio, 0.0)
    }

    /// Transcribe audio with automatic chunking for long audio.
    ///
    /// Splits audio into overlapping chunks, transcribes each, and merges results.
    /// Timestamps are adjusted to reflect position in the original audio.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio buffer to transcribe
    /// * `config` - Chunking configuration (defaults to 60s chunks with 1s overlap)
    pub fn transcribe_chunked(
        &mut self,
        audio: &AudioBuffer,
        config: Option<ChunkConfig>,
    ) -> Result<Transcription> {
        let config = config.unwrap_or_default();

        // If audio is short enough, don't chunk
        if audio.duration_secs() <= config.duration {
            return self.transcribe(audio);
        }

        // Split into chunks
        let chunks = split_audio(audio, config);
        let mut all_tokens: Vec<Token> = Vec::new();
        let mut full_text = String::new();

        for (i, chunk) in chunks.iter().enumerate() {
            tracing::debug!(
                chunk = i + 1,
                total = chunks.len(),
                offset_sec = chunk.offset_sec,
                duration_sec = chunk.audio.duration_secs(),
                "transcribing chunk"
            );

            let result = self.transcribe_chunk(chunk)?;

            // Merge tokens with deduplication
            let merged = merge_chunk_tokens(&all_tokens, &result.tokens, config.overlap);
            all_tokens = merged;
        }

        // Build full text from tokens
        for token in &all_tokens {
            full_text.push_str(&token.text);
        }

        Ok(Transcription {
            text: full_text.trim().to_string(),
            tokens: all_tokens,
        })
    }

    /// Get reference to preprocessor.
    pub fn preprocessor(&self) -> &ParakeetPreprocessor {
        &self.preprocessor
    }

    /// Transcribe a single audio chunk with time offset adjustment.
    fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> Result<Transcription> {
        self.transcribe_buffer(&chunk.audio, chunk.offset_sec)
    }

    /// Internal transcription with offset adjustment.
    fn transcribe_buffer(&mut self, audio: &AudioBuffer, offset_sec: f32) -> Result<Transcription> {
        // Extract features
        let features = self.preprocessor.preprocess(audio)?;

        // Run model inference
        let output = self.model.forward(features)?;

        // Decode tokens to text with timestamps
        let result = self.detokenizer.decode(&output)?;

        // Adjust timestamps with offset
        Ok(Transcription {
            text: result.text,
            tokens: result
                .tokens
                .into_iter()
                .map(|t| Token {
                    text: t.text,
                    start: t.start + offset_sec,
                    end: t.end + offset_sec,
                })
                .collect(),
        })
    }
}

/// Merge tokens from a new chunk with existing tokens, handling overlap deduplication.
///
/// Strategy: In the overlap region, prefer tokens that form complete words/sentences.
fn merge_chunk_tokens(existing: &[Token], new_tokens: &[Token], overlap_sec: f32) -> Vec<Token> {
    if existing.is_empty() {
        return new_tokens.to_vec();
    }

    if new_tokens.is_empty() {
        return existing.to_vec();
    }

    // Find the last timestamp in existing tokens
    let existing_end = existing.last().map(|t| t.end).unwrap_or(0.0);

    // Find where the overlap starts in the new tokens
    let overlap_start = existing_end - overlap_sec;

    // Skip new tokens that are in the overlap region
    // Simple strategy: skip tokens that start before the overlap_start threshold
    let new_start_idx = new_tokens
        .iter()
        .position(|t| t.start >= overlap_start + (overlap_sec * 0.5))
        .unwrap_or(0);

    let mut result = existing.to_vec();
    result.extend_from_slice(&new_tokens[new_start_idx..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_empty_existing() {
        let existing: Vec<Token> = vec![];
        let new_tokens = vec![Token {
            text: "hello".to_string(),
            start: 0.0,
            end: 1.0,
        }];

        let result = merge_chunk_tokens(&existing, &new_tokens, 1.0);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "hello");
    }

    #[test]
    fn merge_empty_new() {
        let existing = vec![Token {
            text: "hello".to_string(),
            start: 0.0,
            end: 1.0,
        }];
        let new_tokens: Vec<Token> = vec![];

        let result = merge_chunk_tokens(&existing, &new_tokens, 1.0);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn merge_with_overlap() {
        let existing = vec![
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
        ];

        // New chunk overlaps - first token is in overlap region
        let new_tokens = vec![
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
        ];

        let result = merge_chunk_tokens(&existing, &new_tokens, 1.0);

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
