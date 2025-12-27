//! Core types for melops-asr

use crate::error::{Error, Result};
use hound::WavReader;
use std::path::Path;

/// Expected sample rate for ASR models (16kHz)
pub const SAMPLE_RATE: u32 = 16000;

/// Validated audio buffer containing 16kHz mono f32 samples.
///
/// This newtype ensures audio data is properly validated before
/// reaching the preprocessing stage, preventing invalid input errors.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    samples: Vec<f32>,
}

impl AudioBuffer {
    /// Load audio from a WAV file.
    ///
    /// Validates sample rate is 16kHz and converts stereo to mono if needed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.map(|s| s as f32 / 32768.0))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        };

        Self::from_wav(samples, spec.sample_rate, spec.channels)
    }

    /// Construct from raw WAV data with validation.
    ///
    /// # Arguments
    ///
    /// * `samples` - Raw audio samples (interleaved if stereo)
    /// * `sample_rate` - Sample rate in Hz (must be 16000)
    /// * `channels` - Number of channels (1 for mono, 2 for stereo)
    ///
    /// # Errors
    ///
    /// Returns error if sample rate is not 16kHz or channels > 2.
    pub fn from_wav(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Result<Self> {
        // Validate sample rate
        if sample_rate != SAMPLE_RATE {
            return Err(Error::InvalidSampleRate {
                expected: SAMPLE_RATE,
                got: sample_rate,
            });
        }

        // Validate channels
        if channels == 0 || channels > 2 {
            return Err(Error::InvalidChannels(channels));
        }

        // Convert stereo to mono if needed
        let samples = if channels == 2 {
            samples
                .chunks(2)
                .map(|chunk| chunk.iter().sum::<f32>() / 2.0)
                .collect()
        } else {
            samples
        };

        Ok(Self { samples })
    }

    /// Construct from pre-validated 16kHz mono samples.
    ///
    /// Use this when you've already validated the audio format.
    /// No validation is performed.
    pub fn from_raw_mono(samples: Vec<f32>) -> Self {
        Self { samples }
    }

    /// Get the audio samples as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.samples
    }

    /// Get the number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / SAMPLE_RATE as f32
    }

    /// Consume the buffer and return the underlying samples.
    pub fn into_inner(self) -> Vec<f32> {
        self.samples
    }

    /// Get a slice of samples for a time range.
    ///
    /// # Arguments
    ///
    /// * `start_sec` - Start time in seconds
    /// * `end_sec` - End time in seconds
    pub fn slice_time(&self, start_sec: f32, end_sec: f32) -> AudioBuffer {
        let start_sample = (start_sec * SAMPLE_RATE as f32) as usize;
        let end_sample = ((end_sec * SAMPLE_RATE as f32) as usize).min(self.samples.len());

        let samples = self.samples[start_sample..end_sample].to_vec();
        Self { samples }
    }
}

impl AsRef<[f32]> for AudioBuffer {
    fn as_ref(&self) -> &[f32] {
        &self.samples
    }
}

/// Token with timestamp information
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// Transcription result with text and timestamped tokens
#[derive(Debug, Clone)]
pub struct Transcription {
    pub text: String,
    pub tokens: Vec<Token>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_sample_rate() {
        let samples = vec![0.0; 1000];
        let result = AudioBuffer::from_wav(samples, 44100, 1);

        match result {
            Err(Error::InvalidSampleRate {
                expected: 16000,
                got: 44100,
            }) => {}
            _ => panic!("expected InvalidSampleRate error"),
        }
    }

    #[test]
    fn validates_channels() {
        let samples = vec![0.0; 1000];
        let result = AudioBuffer::from_wav(samples, 16000, 5);

        match result {
            Err(Error::InvalidChannels(5)) => {}
            _ => panic!("expected InvalidChannels error"),
        }
    }

    #[test]
    fn accepts_mono_16khz() {
        let samples = vec![0.1, 0.2, 0.3];
        let buffer = AudioBuffer::from_wav(samples.clone(), 16000, 1).unwrap();

        assert_eq!(buffer.as_slice(), &samples[..]);
    }

    #[test]
    fn converts_stereo_to_mono() {
        // Stereo samples: [L, R, L, R]
        let samples = vec![0.2, 0.4, 0.6, 0.8];
        let buffer = AudioBuffer::from_wav(samples, 16000, 2).unwrap();

        // Should average to [0.3, 0.7] (with floating point tolerance)
        let result = buffer.as_slice();
        assert!((result[0] - 0.3).abs() < 0.001);
        assert!((result[1] - 0.7).abs() < 0.001);
    }

    #[test]
    fn computes_duration() {
        let samples = vec![0.0; 32000]; // 2 seconds at 16kHz
        let buffer = AudioBuffer::from_raw_mono(samples);

        assert!((buffer.duration_secs() - 2.0).abs() < 0.001);
    }

    #[test]
    fn slices_by_time() {
        let samples: Vec<f32> = (0..48000).map(|i| i as f32).collect(); // 3 seconds
        let buffer = AudioBuffer::from_raw_mono(samples);

        let slice = buffer.slice_time(1.0, 2.0);
        assert_eq!(slice.len(), 16000); // 1 second
        assert!((slice.as_slice()[0] - 16000.0).abs() < 0.001);
    }
}
