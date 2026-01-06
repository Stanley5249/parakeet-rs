//! Core traits for ASR pipeline components.

use crate::chunk::ChunkConfig;
use crate::error::Result;
use crate::types::Segment;

/// ASR model that performs inference on preprocessed features.
///
/// This trait abstracts over different model architectures (TDT, CDC, EOU, Whisper)
/// while providing a uniform interface for the pipeline.
pub trait AsrModel {
    /// Output type from model inference.
    ///
    /// Represents a single unit of model output (e.g., a token with timing).
    /// The `forward` method returns `Vec<Self::Output>`, a sequence of these items.
    type Output;

    /// Convert frame index to seconds.
    fn frame_to_secs(&self, frame: usize) -> f32;

    /// Convert seconds to frame index.
    fn secs_to_frame(&self, secs: f32) -> usize;

    /// Run inference on the given audio, returning a sequence of output items.
    ///
    /// Returns `Vec<Self::Output>` where each item represents a decoded unit
    /// (e.g., token with timing information).
    ///
    /// Note: Takes `&mut self` because ONNX Runtime's Session::run requires it.
    fn forward(&mut self, audio: &[f32]) -> Result<Vec<Self::Output>>;

    /// Convert a sequence of model outputs to text segments with timestamps.
    fn to_segments(&self, output: &[Self::Output]) -> Result<Vec<Segment>>;

    /// Apply frame offset to a sequence of model outputs.
    ///
    /// Used for chunked transcription where frame indices need to be offset
    /// to represent absolute positions in the full audio.
    fn offset_outputs(output: &mut [Self::Output], frame_offset: usize);

    /// Merge output sequences from multiple chunks into a single sequence.
    fn merge_outputs(chunks: impl IntoIterator<Item = Vec<Self::Output>>) -> Vec<Self::Output>;

    /// Transcribe audio samples, returning segments.
    fn transcribe(&mut self, audio: &[f32]) -> Result<Vec<Segment>> {
        let output = self.forward(audio)?;
        self.to_segments(&output)
    }

    /// Transcribe audio with automatic chunking, returning merged segments.
    fn transcribe_chunked(&mut self, audio: &[f32], config: ChunkConfig) -> Result<Vec<Segment>> {
        use crate::audio::SAMPLE_RATE;

        let output_chunks: Result<Vec<_>> = config
            .iter_ranges(audio.len())
            .enumerate()
            .map(|(i, (range, offset_sec))| {
                let chunk = &audio[range];
                let duration_sec = chunk.len() as f32 / SAMPLE_RATE as f32;

                tracing::debug!(chunk = i + 1, duration_sec, "transcribing chunk");

                let frame_offset = self.secs_to_frame(offset_sec);
                let mut output = self.forward(chunk)?;
                Self::offset_outputs(&mut output, frame_offset);

                Ok(output)
            })
            .collect();

        let merged_output = Self::merge_outputs(output_chunks?);
        self.to_segments(&merged_output)
    }

    /// Transcribe audio from an iterator stream, returning merged segments.
    #[allow(unused_variables)]
    fn transcribe_stream(
        &mut self,
        audio: impl Iterator<Item = f32>,
        config: ChunkConfig,
    ) -> Result<Vec<Segment>> {
        todo!()
    }
}
