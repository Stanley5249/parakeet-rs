//! Audio chunking utilities for processing long audio files.

use crate::audio::SAMPLE_RATE;

/// Default chunk duration in seconds (4 minutes)
const DEFAULT_CHUNK_DURATION: f32 = 240.0;

/// Default chunk overlap in seconds
const DEFAULT_CHUNK_OVERLAP: f32 = 1.0;

/// Configuration for audio chunking.
#[derive(clap::Args, Clone, Copy, Debug)]
pub struct ChunkConfig {
    /// Chunk duration in seconds for long audio
    #[arg(long, default_value_t = DEFAULT_CHUNK_DURATION)]
    pub duration: f32,

    /// Chunk overlap in seconds
    #[arg(long, default_value_t = DEFAULT_CHUNK_OVERLAP)]
    pub overlap: f32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            duration: DEFAULT_CHUNK_DURATION,
            overlap: DEFAULT_CHUNK_OVERLAP,
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

    /// Get chunk size in samples.
    pub fn chunk_samples(&self) -> usize {
        (self.duration * SAMPLE_RATE as f32) as usize
    }

    /// Get overlap size in samples.
    pub fn overlap_samples(&self) -> usize {
        (self.overlap * SAMPLE_RATE as f32) as usize
    }

    /// Get step size in samples (chunk - overlap).
    pub fn step_samples(&self) -> usize {
        self.chunk_samples().saturating_sub(self.overlap_samples())
    }

    /// Create an iterator over chunk ranges for a given total size.
    ///
    /// Returns an iterator of `(Range<usize>, f32)` where:
    /// - First element is the range to slice the data
    /// - Second element is the time offset in seconds
    pub fn iter_ranges(&self, len: usize) -> ChunkRangeIter {
        ChunkRangeIter {
            len,
            chunk_size: self.chunk_samples(),
            step_size: self.step_samples(),
            position: 0,
        }
    }
}

/// Iterator over chunk ranges with time offsets.
pub struct ChunkRangeIter {
    len: usize,
    chunk_size: usize,
    step_size: usize,
    position: usize,
}

impl Iterator for ChunkRangeIter {
    type Item = (std::ops::Range<usize>, f32);

    fn next(&mut self) -> Option<Self::Item> {
        // If we haven't started yet and audio is short, return full range
        if self.position == 0 && self.len <= self.chunk_size {
            self.position = self.len; // Mark as consumed
            return Some((0..self.len, 0.0));
        }

        // Check if we've reached the end
        if self.position >= self.len {
            return None;
        }

        let start = self.position;
        let end = (start + self.chunk_size).min(self.len);
        let offset_sec = start as f32 / SAMPLE_RATE as f32;

        self.position += self.step_size;

        Some((start..end, offset_sec))
    }
}

/// Iterator over audio chunks with their time offsets.
///
/// Yields tuples of `(&[f32], f32)` where:
/// - First element is the audio chunk slice
/// - Second element is the start time offset in seconds
pub fn chunk_audio<'a>(
    data: &'a [f32],
    config: &'a ChunkConfig,
) -> impl Iterator<Item = (&'a [f32], f32)> + 'a {
    config
        .iter_ranges(data.len())
        .map(move |(range, offset)| (&data[range], offset))
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

    fn make_audio(duration_sec: f32) -> Vec<f32> {
        vec![0.0; (duration_sec * SAMPLE_RATE as f32) as usize]
    }

    #[test]
    fn short_audio_returns_single_chunk() {
        let audio = make_audio(30.0); // 30 seconds
        let config = ChunkConfig::new(60.0, 1.0);

        let mut iter = chunk_audio(&audio, &config);
        let (chunk, offset_sec) = iter.next().unwrap();

        assert!((offset_sec - 0.0).abs() < 0.001);
        assert_eq!(chunk.len(), audio.len());
        assert!(iter.next().is_none());
    }

    #[test]
    fn long_audio_splits_with_overlap() {
        let audio = make_audio(150.0); // 2.5 minutes
        let config = ChunkConfig::new(60.0, 1.0);

        let mut iter = chunk_audio(&audio, &config);

        // Step is 59 seconds (60 - 1)
        // Chunks at: 0, 59, 118
        let (_, offset_sec) = iter.next().unwrap();
        assert!((offset_sec - 0.0).abs() < 0.001);

        let (_, offset_sec) = iter.next().unwrap();
        assert!((offset_sec - 59.0).abs() < 0.001);

        let (_, offset_sec) = iter.next().unwrap();
        assert!((offset_sec - 118.0).abs() < 0.001);

        assert!(iter.next().is_none());
    }

    #[test]
    fn step_calculation() {
        let config = ChunkConfig::new(60.0, 5.0);

        assert!((config.step_sec() - 55.0).abs() < 0.001);
        assert_eq!(config.step_samples(), 55 * SAMPLE_RATE as usize);
    }

    #[test]
    fn chunk_samples_calculation() {
        let config = ChunkConfig::new(60.0, 1.0);

        assert_eq!(config.chunk_samples(), 60 * SAMPLE_RATE as usize);
        assert_eq!(config.overlap_samples(), SAMPLE_RATE as usize);
        assert_eq!(config.step_samples(), 59 * SAMPLE_RATE as usize);
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
