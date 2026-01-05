//! Audio preprocessor for ASR models.

use crate::audio::{PreprocessorConfig, SAMPLE_RATE, extract_features};
use crate::error::Result;
use crate::traits::AudioPreprocessor;
use ndarray::Array2;

/// Parakeet audio preprocessor.
///
/// Extracts mel-spectrogram features from audio for ASR inference.
pub struct ParakeetPreprocessor {
    config: PreprocessorConfig,
}

impl ParakeetPreprocessor {
    /// Create a TDT-specific preprocessor (128 mel features).
    pub fn tdt() -> Self {
        Self {
            config: PreprocessorConfig::tdt(),
        }
    }

    /// Get the preprocessor configuration.
    pub fn config(&self) -> &PreprocessorConfig {
        &self.config
    }
}

impl AudioPreprocessor for ParakeetPreprocessor {
    type Features = Array2<f32>;

    fn preprocess(&self, audio: &[f32]) -> Result<Self::Features> {
        extract_features(audio.to_vec(), SAMPLE_RATE, 1, &self.config)
    }
}
