//! Audio preprocessor for Parakeet models.

use crate::audio::SAMPLE_RATE;
use crate::error::Result;
use crate::traits::AudioPreprocessor;
use ndarray::Array2;
use parakeet_rs::{PreprocessorConfig, audio};

/// Parakeet audio preprocessor.
///
/// Extracts mel-spectrogram features from audio for ASR inference.
pub struct ParakeetPreprocessor {
    config: PreprocessorConfig,
}

impl ParakeetPreprocessor {
    /// Create a TDT-specific preprocessor (128 features).
    pub fn tdt() -> Self {
        Self {
            config: PreprocessorConfig {
                feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
                feature_size: 128,
                hop_length: 160,
                n_fft: 512,
                padding_side: "right".to_string(),
                padding_value: 0.0,
                preemphasis: 0.97,
                processor_class: "ParakeetProcessor".to_string(),
                return_attention_mask: true,
                sampling_rate: 16000,
                win_length: 400,
            },
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
        let features = audio::extract_features_raw(
            audio.to_vec(),
            SAMPLE_RATE,
            1, // Already mono
            &self.config,
        )?;
        Ok(features)
    }
}
