//! Audio loading utilities.

use hound::{Result, SampleFormat, WavReader};
use std::path::Path;

/// Expected sample rate for ASR models (16kHz)
pub const SAMPLE_RATE: u32 = 16000;

/// Load audio from a WAV file as mono f32 samples at 16kHz.
///
/// Validates sample rate is 16kHz and converts stereo to mono if needed.
///
/// # Errors
///
/// Returns error if:
/// - File cannot be read
/// - Sample rate is not 16kHz
/// - Channel count is invalid (0 or > 2)
pub fn read_audio_mono(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    // Validate sample rate
    if spec.sample_rate != SAMPLE_RATE {
        return Err(hound::Error::Unsupported);
    }

    // Validate channels
    if spec.channels == 0 || spec.channels > 2 {
        return Err(hound::Error::Unsupported);
    }

    // Read samples based on format
    // TODO: support i24 and i32
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<_>>()?,
        SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / i16::MAX as f32))
            .collect::<Result<_>>()?,
    };

    // Convert stereo to mono if needed
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| chunk.iter().sum::<f32>() / 2.0)
            .collect()
    } else {
        samples
    };

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use hound::WavWriter;

    use super::*;

    /// Helper to create a minimal WAV file for testing
    fn create_test_wav(
        path: &Path,
        sample_rate: u32,
        channels: u16,
        samples: &[f32],
    ) -> Result<()> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec)?;
        for &sample in samples {
            writer.write_sample((sample * 32768.0) as i16)?;
        }
        writer.finalize()?;
        Ok(())
    }

    #[test]
    fn reads_mono_16khz() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mono.wav");

        let test_samples = vec![0.1, 0.2, 0.3];
        create_test_wav(&path, 16000, 1, &test_samples).unwrap();

        let result = read_audio_mono(&path).unwrap();

        // Allow small floating point differences
        for (expected, actual) in test_samples.iter().zip(result.iter()) {
            assert!((expected - actual).abs() < 0.01);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn converts_stereo_to_mono() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_stereo.wav");

        // Stereo samples: [L, R, L, R]
        let test_samples = vec![0.2, 0.4, 0.6, 0.8];
        create_test_wav(&path, 16000, 2, &test_samples).unwrap();

        let result = read_audio_mono(&path).unwrap();

        // Should average to [0.3, 0.7]
        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.3).abs() < 0.01);
        assert!((result[1] - 0.7).abs() < 0.01);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_44khz.wav");

        create_test_wav(&path, 44100, 1, &[0.0, 0.1]).unwrap();

        let result = read_audio_mono(&path);

        assert!(result.is_err(), "expected error for wrong sample rate");

        std::fs::remove_file(path).ok();
    }
}
