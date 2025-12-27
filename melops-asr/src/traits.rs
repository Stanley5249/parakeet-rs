//! Core traits for ASR pipeline components.
//!
//! This module defines the trait hierarchy for building extensible ASR pipelines.
//! Each trait represents a distinct stage in the transcription process:
//!
//! - [`AudioPreprocessor`]: Converts raw audio to model-specific features
//! - [`AsrModel`]: Runs inference on features to produce model output
//! - [`Detokenizer`]: Converts model output to human-readable transcription

use crate::error::Result;
use crate::types::{AudioBuffer, Transcription};

/// Preprocesses raw audio into model-specific features.
///
/// Implementations handle audio normalization, feature extraction (e.g., mel spectrograms),
/// and any model-specific transformations required before inference.
pub trait AudioPreprocessor {
    /// The feature type produced by this preprocessor.
    type Features;

    /// Extract features from an audio buffer.
    ///
    /// # Arguments
    ///
    /// * `audio` - Validated 16kHz mono audio buffer
    ///
    /// # Returns
    ///
    /// Model-specific features ready for inference
    fn preprocess(&self, audio: &AudioBuffer) -> Result<Self::Features>;
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

/// Converts model output to human-readable transcription.
///
/// Handles token-to-text conversion, timestamp calculation, and any
/// post-processing (e.g., punctuation restoration, capitalization).
pub trait Detokenizer {
    /// The model output type this detokenizer can process.
    type Input;

    /// Decode model output to a transcription with timestamps.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw model output (tokens, frame indices, etc.)
    ///
    /// # Returns
    ///
    /// Transcription containing full text and timestamped tokens
    fn decode(&self, input: &Self::Input) -> Result<Transcription>;
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

    /// Transcribe an audio buffer.
    ///
    /// Runs the full pipeline: preprocess → model → detokenize
    pub fn transcribe(&mut self, audio: &AudioBuffer) -> Result<Transcription> {
        let features = self.preprocessor.preprocess(audio)?;
        let output = self.model.forward(features)?;
        self.detokenizer.decode(&output)
    }
}
