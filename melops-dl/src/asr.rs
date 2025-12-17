//! ASR audio presets: 16kHz mono WAV extraction.
//!
//! **Formats:** [`AudioFormat::Pcm16`] (16-bit, standard), [`AudioFormat::Float32`] (32-bit, higher precision)
//!
//! ```no_run
//! use melops_dl::{dl::download, asr::AudioFormat};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! download(&["https://youtube.com/watch?v=example"], AudioFormat::Pcm16.into())?;
//! # Ok(())
//! # }
//! ```
//!
//! **Output:** `downloads/domain/uploader/title.wav` + `title.info.json`

use crate::dl::{DlOptions, OutputTemplates, Paths, PostProcessor, PostProcessorArgs};

/// 16kHz mono WAV format (pcm_s16le or pcm_f32le).
#[derive(Copy, Clone, Debug, Default)]
pub enum AudioFormat {
    /// 16-bit PCM (standard, smaller files)
    #[default]
    Pcm16,
    /// 32-bit float PCM (higher precision, ~2x larger)
    Float32,
}

impl From<AudioFormat> for PostProcessorArgs {
    /// FFmpeg args: `-ar 16000 -ac 1 -c:a pcm_s16le` (Pcm16) or `pcm_f32le` (Float32)
    fn from(format: AudioFormat) -> Self {
        let extract_audio = match format {
            AudioFormat::Pcm16 => vec![
                "-ar".to_string(),
                "16000".to_string(), // 16kHz sample rate
                "-ac".to_string(),
                "1".to_string(), // mono
                "-c:a".to_string(),
                "pcm_s16le".to_string(), // 16-bit PCM
            ],
            AudioFormat::Float32 => vec![
                "-ar".to_string(),
                "16000".to_string(), // 16kHz sample rate
                "-ac".to_string(),
                "1".to_string(), // mono
                "-c:a".to_string(),
                "pcm_f32le".to_string(), // 32-bit float PCM
            ],
        };
        Self { extract_audio }
    }
}

impl From<AudioFormat> for DlOptions {
    /// ASR preset: best audio → 16kHz mono WAV, organized by `domain/uploader/title`, saves `.info.json`
    fn from(format: AudioFormat) -> Self {
        Self {
            format: Some("ba".to_string()),
            paths: Some(Paths::system_default()),
            outtmpl: Some(OutputTemplates::default()),
            postprocessors: Some(vec![PostProcessor {
                key: "FFmpegExtractAudio".to_string(),
                preferredcodec: Some("wav".to_string()),
            }]),
            postprocessor_args: Some(format.into()),
            writeinfojson: Some(true),
            getcomments: Some(true),
            quiet: Some(true),
            no_warnings: Some(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_to_postprocessor_args_pcm16() {
        let format = AudioFormat::Pcm16;
        let args: PostProcessorArgs = format.into();

        assert_eq!(
            args.extract_audio,
            vec!["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]
        );
    }

    #[test]
    fn test_audio_format_to_postprocessor_args_float32() {
        let format = AudioFormat::Float32;
        let args = PostProcessorArgs::from(format);

        assert_eq!(
            args.extract_audio,
            vec!["-ar", "16000", "-ac", "1", "-c:a", "pcm_f32le"]
        );
    }

    #[test]
    #[allow(unused_variables)] // false positive suppression
    fn test_audio_format_to_dl_options() {
        let opts: DlOptions = AudioFormat::Pcm16.into();
        let format = "ba".to_string();

        assert!(matches!(
            opts,
            DlOptions {
                format: Some(format),
                paths: Some(_),
                outtmpl: Some(_),
                postprocessors: Some(_),
                postprocessor_args: Some(_),
                writeinfojson: Some(true),
                getcomments: Some(true),
                quiet: Some(true),
                no_warnings: Some(true),
            }
        ));
    }

    #[test]
    fn test_audio_format_default() {
        let format = AudioFormat::default();
        assert!(matches!(format, AudioFormat::Pcm16));
    }
}
