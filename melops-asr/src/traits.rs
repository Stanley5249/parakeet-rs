//! Core traits for ASR pipeline components.

use crate::chunk::ChunkConfig;
use crate::error::Result;
use crate::models::tdt::TdtModel;
use crate::types::Token;

/// ASR model that performs inference on preprocessed features.
///
/// This trait abstracts over different model architectures (TDT, CDC, EOU, Whisper)
/// while providing a uniform interface for the pipeline.
pub trait AsrModel {
    /// Output type from model inference.
    type Output;

    /// Run inference on the given audio.
    ///
    /// Note: Takes `&mut self` because ONNX Runtime's Session::run requires it.
    fn forward(&mut self, audio: &[f32]) -> Result<Self::Output>;

    /// Apply frame offset to model output.
    ///
    /// Used for chunked transcription where frame indices need to be offset
    /// to represent absolute positions in the full audio.
    fn offset_output(output: &mut Self::Output, frame_offset: usize);

    /// Convert frame index to seconds.
    ///
    /// The conversion depends on the model's subsampling factor and mel-spectrogram hop length.
    fn frame_to_secs(&self, frame: usize) -> f32;

    /// Convert seconds to frame index.
    ///
    /// The conversion depends on the model's subsampling factor and mel-spectrogram hop length.
    fn secs_to_frame(&self, secs: f32) -> usize;
}

impl TdtModel {
    /// Transcribe audio samples, returning tokens.
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<Vec<Token>> {
        let token_durations = self.forward(audio)?;
        self.decode(&token_durations)
    }

    /// Transcribe audio with automatic chunking, returning merged tokens.
    pub fn transcribe_chunked(&mut self, audio: &[f32], config: ChunkConfig) -> Result<Vec<Token>> {
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
                Self::offset_output(&mut output, frame_offset);

                Ok(output)
            })
            .collect();

        let merged_output = Self::merge_outputs(output_chunks?);
        self.decode(&merged_output)
    }

    /// Transcribe audio from an iterator stream, returning merged tokens.
    #[allow(unused_variables)]
    pub fn transcribe_stream(
        &mut self,
        audio: impl Iterator<Item = f32>,
        config: ChunkConfig,
    ) -> Result<Vec<Token>> {
        todo!()
    }
}
