//! Core traits for ASR pipeline components.
//!
//! This module defines the trait hierarchy for building extensible ASR pipelines.
//! Each trait represents a distinct stage in the transcription process:
//!
//! - [`AudioPreprocessor`]: Converts raw audio to model-specific features
//! - [`AsrModel`]: Runs inference on features to produce model output
//! - [`Detokenizer`]: Converts model output to human-readable transcription

use crate::audio::SAMPLE_RATE;
use crate::chunk::ChunkConfig;
use crate::error::Result;
use crate::types::{Token, Transcription};

/// Preprocesses raw audio into model-specific features.
///
/// Implementations handle audio normalization, feature extraction (e.g., mel spectrograms),
/// and any model-specific transformations required before inference.
pub trait AudioPreprocessor {
    /// The feature type produced by this preprocessor.
    type Features;

    /// Extract features from audio samples.
    ///
    /// # Arguments
    ///
    /// * `audio` - 16kHz mono audio samples (f32 slice)
    ///
    /// # Returns
    ///
    /// Model-specific features ready for inference
    fn preprocess(&self, audio: &[f32]) -> Result<Self::Features>;
}

/// ASR model that performs inference on preprocessed features.
///
/// This trait abstracts over different model architectures (TDT, CDC, EOU, Whisper)
/// while providing a uniform interface for the pipeline.
///
/// # Stateless vs Stateful Models
///
/// - **Stateless models** (TDT, CDC): Features are just the audio features
/// - **Stateful models** (EOU): Features include state, e.g., `(AudioFeatures, EouState)`
///
/// The pipeline manages state for stateful models by extracting new state from the output.
pub trait AsrModel {
    /// Input features type (from preprocessor or including state).
    type Features;

    /// Output type from model inference.
    type Output;

    /// Run inference on the given features.
    ///
    /// Note: Takes `&mut self` because ONNX Runtime's Session::run requires it.
    fn forward(&mut self, features: Self::Features) -> Result<Self::Output>;
}

/// Converts model output to timestamped tokens.
///
/// Handles token-to-text conversion, timestamp calculation, token merging, and transcription building.
pub trait Detokenizer {
    /// The model output type this detokenizer can process.
    type Input;

    /// Decode model output to timestamped tokens.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw model output (tokens, frame indices, etc.)
    ///
    /// # Returns
    ///
    /// Vector of timestamped tokens
    fn decode(&self, input: &Self::Input) -> Result<Vec<Token>>;

    /// Merge tokens from multiple chunks, handling overlap deduplication.
    ///
    /// # Arguments
    ///
    /// * `token_chunks` - Iterator of token vectors from consecutive chunks
    /// * `overlap_sec` - Overlap duration in seconds between chunks
    fn merge_tokens<I>(token_chunks: I, overlap_sec: f32) -> Vec<Token>
    where
        I: IntoIterator<Item = Vec<Token>>;

    /// Build transcription from tokens.
    ///
    /// Knows how to join tokens based on the tokenization scheme (e.g., SentencePiece).
    fn build_transcription(tokens: Vec<Token>) -> Transcription;
}

/// A complete ASR pipeline combining preprocessor, model, and detokenizer.
///
/// The pipeline ensures type compatibility between components at compile time
/// through associated type constraints.
pub struct AsrPipeline<P, M, D> {
    /// Audio preprocessor
    pub preprocessor: P,
    /// ASR model
    pub model: M,
    /// Output detokenizer
    pub detokenizer: D,
}

impl<P, M, D> AsrPipeline<P, M, D>
where
    P: AudioPreprocessor,
    M: AsrModel<Features = P::Features>,
    D: Detokenizer<Input = M::Output>,
{
    /// Create a new pipeline from components.
    pub fn new(preprocessor: P, model: M, detokenizer: D) -> Self {
        Self {
            preprocessor,
            model,
            detokenizer,
        }
    }

    /// Internal token generation with offset adjustment.
    fn transcribe_with_offset(&mut self, data: &[f32], offset_sec: f32) -> Result<Vec<Token>> {
        let features = self.preprocessor.preprocess(data)?;
        let output = self.model.forward(features)?;
        let tokens = self.detokenizer.decode(&output)?;

        // Adjust timestamps with offset
        Ok(tokens
            .into_iter()
            .map(|t| Token {
                text: t.text,
                start: t.start + offset_sec,
                end: t.end + offset_sec,
            })
            .collect())
    }

    /// Transcribe audio samples, returning tokens.
    ///
    /// Runs: preprocess → model → detokenize
    ///
    /// Use `Detokenizer::build_transcription()` to convert tokens to transcription.
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<Vec<Token>> {
        self.transcribe_with_offset(audio, 0.0)
    }

    /// Transcribe audio with automatic chunking, returning merged tokens.
    ///
    /// Splits audio into overlapping chunks, transcribes each, and merges results.
    ///
    /// Use `Detokenizer::build_transcription()` to convert tokens to transcription.
    pub fn transcribe_chunked(&mut self, data: &[f32], config: ChunkConfig) -> Result<Vec<Token>> {
        let token_chunks: Result<Vec<_>> = config
            .iter_ranges(data.len())
            .enumerate()
            .map(|(i, (range, offset_sec))| {
                let chunk = &data[range];
                let duration_sec = chunk.len() as f32 / SAMPLE_RATE as f32;

                tracing::debug!(
                    chunk = i + 1,
                    offset_sec,
                    duration_sec,
                    "transcribing chunk"
                );

                self.transcribe_with_offset(chunk, offset_sec)
            })
            .collect();

        Ok(D::merge_tokens(token_chunks?, config.overlap))
    }

    /// Transcribe audio from an iterator stream, returning merged tokens.
    ///
    /// Processes audio in chunks with overlap, reading incrementally from the iterator.
    #[allow(unused_variables)]
    pub fn transcribe_stream(
        &mut self,
        data: impl Iterator<Item = f32>,
        config: ChunkConfig,
    ) -> Result<Vec<Token>> {
        todo!()
    }
}
