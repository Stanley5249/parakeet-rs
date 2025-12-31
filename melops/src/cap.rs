//! Cap subcommand - generate captions from audio file to SRT.

use crate::cli::CaptionConfig;
use crate::srt::{self, display_subtitle};
use eyre::{Context, Result};
use hf_hub::api::sync::Api;
use melops_asr::audio::read_audio_mono;
use melops_asr::chunk::ChunkConfig;
use melops_asr::pipelines::ParakeetTdt;
#[allow(unused_imports)]
use ort::execution_providers::*;
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use srtlib::Subtitle;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MODEL_ID: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";

/// CLI arguments for caption generation.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Path to input WAV file
    pub path: PathBuf,

    /// Output SRT path (default: same as input with .srt extension)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    #[command(flatten)]
    pub caption_config: CaptionConfig,
}

/// Resolved configuration for caption generation.
#[derive(Debug)]
pub struct Config {
    pub path: PathBuf,
    pub output: Option<PathBuf>,
    pub preview: bool,
    pub chunk_config: ChunkConfig,
}

impl TryFrom<Args> for Config {
    type Error = eyre::Error;

    fn try_from(args: Args) -> Result<Self> {
        Ok(Self {
            path: args.path,
            output: args.output,
            preview: args.caption_config.preview,
            chunk_config: args.caption_config.chunk_config,
        })
    }
}

pub fn execute(config: Config) -> Result<()> {
    // Resolve output path
    let output = config
        .output
        .unwrap_or_else(|| config.path.with_extension("srt"));

    tracing::info!(
        input = ?config.path.display(),
        output = ?output.display(),
        "generating captions"
    );

    let subtitles = caption_from_wav_file(&config.path, config.chunk_config)?;

    tracing::info!(path = ?output.display(), "write srt file");

    // Write to file
    std::fs::write(&output, display_subtitle(&subtitles))
        .wrap_err_with(|| format!("failed to write srt: {:?}", output.display()))?;

    // Display preview or full output to stdout
    if config.preview {
        print!("{}", srt::preview_subtitles(&subtitles, 2, 2));
    }

    Ok(())
}

/// Perform ASR on WAV file and return captions as subtitles.
fn caption_from_wav_file(wav_path: &Path, chunk_config: ChunkConfig) -> Result<Vec<Subtitle>> {
    let audio = read_audio_mono(wav_path)
        .wrap_err_with(|| format!("failed to load audio: {:?}", wav_path.display()))?;

    tracing::info!("locating model");
    let api = Api::new()?;
    let repo = api.model(MODEL_ID.to_string());

    let s = Instant::now();

    tracing::info!("loading model");

    let builder = build_session()?;
    let mut model = ParakeetTdt::from_repo(&repo, builder)?;

    let d = s.elapsed();
    tracing::info!(duration = %format_secs(d.as_secs_f32()), "model loaded");

    let s = Instant::now();

    let tokens = model
        .transcribe_chunked(&audio, chunk_config)
        .wrap_err("transcription failed")?;

    let d = s.elapsed();
    tracing::info!(duration = %format_secs(d.as_secs_f32()), "inference completed");

    let subtitles = srt::to_subtitles(&tokens);

    Ok(subtitles)
}

/// Build execution config with execution providers configured by Cargo features.
///
/// Configures ONNX Runtime session with execution providers in priority order. The first
/// available provider is used; CPU is always available as fallback.
///
/// # Execution Providers
///
/// Enabled via Cargo features:
/// - `cuda` - NVIDIA CUDA
/// - `tensorrt` - NVIDIA TensorRT
/// - `openvino` - Intel OpenVINO
/// - `directml` - DirectML (Windows)
/// - `coreml` - CoreML (macOS)
///
/// Ensure required hardware, drivers, and runtime dependencies are installed for the
/// desired provider.
fn build_session() -> Result<SessionBuilder> {
    Ok(Session::builder()?.with_execution_providers([
        #[cfg(feature = "cuda")]
        CUDAExecutionProvider::default().build(),
        #[cfg(feature = "tensorrt")]
        TensorRTExecutionProvider::default().build(),
        #[cfg(feature = "openvino")]
        OpenVINOExecutionProvider::default()
            .with_device_type("HETERO:GPU,CPU")
            .with_cache_dir(".cache/ort")
            .with_precision("FP16")
            .build(),
        #[cfg(feature = "directml")]
        DirectMLExecutionProvider::default().build(),
        #[cfg(feature = "coreml")]
        CoreMLExecutionProvider::default().build(),
    ])?)
}

/// Format seconds as a string with two decimal places.
fn format_secs(secs: f32) -> String {
    format!("{:.2}s", secs)
}
