//! Audio chunking utilities for processing long audio files.

use crate::types::AudioBuffer;

/// Configuration for audio chunking.
#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Duration of each chunk in seconds
    pub duration: f32,
    /// Overlap between chunks in seconds (for deduplication)
    pub overlap: f32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            duration: 60.0,
            overlap: 1.0,
        }
    }
}

impl ChunkConfig {
    /// Create a new chunk configuration.
    pub fn new(duration_sec: f32, overlap_sec: f32) -> Self {
        Self {
            duration: duration_sec,
            overlap: overlap_sec,
        }
    }

    /// Calculate the step size between chunks (duration - overlap).
    pub fn step_sec(&self) -> f32 {
        (self.duration - self.overlap).max(1.0)
    }
}

/// A chunk of audio with its time offset in the original audio.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// The audio buffer for this chunk
    pub audio: AudioBuffer,
    /// Start time of this chunk in the original audio (seconds)
    pub offset_sec: f32,
}

/// Split audio into overlapping chunks for processing.
///
/// Returns an iterator of AudioChunk with their time offsets.
pub fn split_audio(audio: &AudioBuffer, config: ChunkConfig) -> Vec<AudioChunk> {
    let total_duration = audio.duration_secs();

    // If audio is shorter than chunk duration, return as single chunk
    if total_duration <= config.duration {
        return vec![AudioChunk {
            audio: audio.clone(),
            offset_sec: 0.0,
        }];
    }

    let step = config.step_sec();
    let mut chunks = Vec::new();
    let mut start = 0.0;

    while start < total_duration {
        let end = (start + config.duration).min(total_duration);
        let chunk_audio = audio.slice_time(start, end);

        chunks.push(AudioChunk {
            audio: chunk_audio,
            offset_sec: start,
        });

        start += step;

        // Avoid tiny trailing chunks
        if total_duration - start < config.overlap {
            break;
        }
    }

    chunks
}

/// Calculate the number of chunks that will be produced for a given audio duration.
pub fn estimate_chunk_count(total_duration_sec: f32, config: &ChunkConfig) -> usize {
    if total_duration_sec <= config.duration {
        return 1;
    }

    let step = config.step_sec();
    ((total_duration_sec - config.duration) / step).ceil() as usize + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_audio(duration_sec: f32) -> AudioBuffer {
        let samples = vec![0.0; (duration_sec * 16000.0) as usize];
        AudioBuffer::from_raw_mono(samples)
    }

    #[test]
    fn short_audio_returns_single_chunk() {
        let audio = make_audio(30.0); // 30 seconds
        let config = ChunkConfig::new(60.0, 1.0);

        let chunks = split_audio(&audio, config);

        assert_eq!(chunks.len(), 1);
        assert!((chunks[0].offset_sec - 0.0).abs() < 0.001);
    }

    #[test]
    fn long_audio_splits_with_overlap() {
        let audio = make_audio(150.0); // 2.5 minutes
        let config = ChunkConfig::new(60.0, 1.0);

        let chunks = split_audio(&audio, config);

        // Step is 59 seconds (60 - 1)
        // Chunks at: 0, 59, 118
        assert_eq!(chunks.len(), 3);
        assert!((chunks[0].offset_sec - 0.0).abs() < 0.001);
        assert!((chunks[1].offset_sec - 59.0).abs() < 0.001);
        assert!((chunks[2].offset_sec - 118.0).abs() < 0.001);
    }

    #[test]
    fn chunk_config_default() {
        let config = ChunkConfig::default();

        assert!((config.duration - 60.0).abs() < 0.001);
        assert!((config.overlap - 1.0).abs() < 0.001);
    }

    #[test]
    fn step_calculation() {
        let config = ChunkConfig::new(60.0, 5.0);

        assert!((config.step_sec() - 55.0).abs() < 0.001);
    }

    #[test]
    fn estimate_count_short() {
        let config = ChunkConfig::new(60.0, 1.0);

        assert_eq!(estimate_chunk_count(30.0, &config), 1);
    }

    #[test]
    fn estimate_count_long() {
        let config = ChunkConfig::new(60.0, 1.0);

        // 150 seconds, step 59: chunks at 0, 59, 118 = 3 chunks
        assert_eq!(estimate_chunk_count(150.0, &config), 3);
    }
}
