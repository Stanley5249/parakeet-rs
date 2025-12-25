//! Parakeet CLI - Speech-to-text transcription tool

use eyre::{Context, OptionExt, Result};
use hf_hub::api::sync::Api;
use hound::WavReader;
use melops_cli::srt;
#[allow(unused_imports)]
use ort::execution_providers::*;
use ort::session::Session;
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use srtlib::Subtitles;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const MODEL_ID: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";
const MODEL_FILES: &[&str] = &[
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
    "vocab.txt",
];

fn main() -> Result<()> {
    let (non_blocking, _guard) = tracing_appender::non_blocking(std::io::stderr());

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let subtitles = run()?;
    print!("{subtitles}");

    Ok(())
}

fn run() -> Result<Subtitles> {
    let audio_path = parse_args()?;
    log_wav_spec(&audio_path)?;

    let model_dir = fetch_model()?;

    let s = Instant::now();

    let mut model = load_model(&model_dir)?;

    let d = s.elapsed();
    tracing::info!(duration = %format_secs(d.as_secs_f32()), "model loaded");

    let s = Instant::now();

    let result = model.transcribe_file(audio_path, Some(TimestampMode::Sentences))?;

    let d = s.elapsed();
    tracing::info!(duration = %format_secs(d.as_secs_f32()), "inference completed");

    let subtitles = srt::to_subtitles(&result.tokens);

    Ok(subtitles)
}

fn parse_args() -> Result<PathBuf> {
    env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("usage: melops <audio.wav>"))
}

fn log_wav_spec(path: &Path) -> Result<()> {
    let reader = WavReader::open(path)
        .wrap_err_with(|| format!("failed to open audio: {}", path.display()))?;

    let spec = reader.spec();
    let duration = reader.duration() as f32 / spec.sample_rate as f32;

    tracing::debug!(
        path = %path.display(),
        duration = format_secs(duration),
        channels = spec.channels,
        sample_rate = spec.sample_rate,
        bits_per_sample = spec.bits_per_sample,
        fromat = ?spec.sample_format,
        "wav spec"
    );
    Ok(())
}

/// Fetch model files from Hugging Face Hub.
fn fetch_model() -> Result<PathBuf> {
    tracing::info!("locating model...");

    let api = Api::new()?;
    let repo = api.model(MODEL_ID.to_string());

    MODEL_FILES
        .iter()
        .map(|file| repo.get(file))
        .try_fold(None, |_, res| res.map(Some))?
        .ok_or_eyre("no model files specified")?
        .parent()
        .ok_or_eyre("failed to get model directory")
        .map(|path| path.to_path_buf())
}

/// Load Parakeet model with execution providers configured by Cargo features.
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
fn load_model(model_dir: &PathBuf) -> Result<ParakeetTDT> {
    tracing::info!(dir = %model_dir.display(), "loading model");

    let builder = Session::builder()?.with_execution_providers([
        #[cfg(feature = "cuda")]
        CUDAExecutionProvider::default().build(),
        #[cfg(feature = "tensorrt")]
        TensorRTExecutionProvider::default().build(),
        #[cfg(feature = "openvino")]
        OpenVINOExecutionProvider::default()
            .with_device_type("GPU")
            .build(),
        #[cfg(feature = "directml")]
        DirectMLExecutionProvider::default().build(),
        #[cfg(feature = "coreml")]
        CoreMLExecutionProvider::default().build(),
    ])?;

    Ok(ParakeetTDT::from_pretrained(model_dir, Some(builder))?)
}

/// Format seconds as a string with two decimal places.
fn format_secs(secs: f32) -> String {
    format!("{:.2}s", secs)
}
