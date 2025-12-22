//! Parakeet CLI - Speech-to-text transcription tool

use hf_hub::api::sync::Api;
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[cfg(feature = "openvino")]
use ort::execution_providers::OpenVINOExecutionProvider;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(not(any(feature = "openvino", feature = "cuda")))]
use ort::execution_providers::CPUExecutionProvider;

const MODEL_ID: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";
const MODEL_FILES: &[&str] = &[
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
    "vocab.txt",
];

fn main() -> ExitCode {
    tracing_subscriber::fmt::init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> eyre::Result<()> {
    let audio_path = parse_args()?;
    print_audio_info(&audio_path)?;

    let model_dir = fetch_model()?;

    let model_load_start = Instant::now();
    let mut model = load_model(&model_dir)?;
    let model_load_time = model_load_start.elapsed();
    println!("Model loaded in {:.2} s", model_load_time.as_secs_f32());

    let inference_start = Instant::now();
    let result = transcribe(&mut model, &audio_path)?;
    let inference_time = inference_start.elapsed();
    println!(
        "Inference completed in {:.2} s",
        inference_time.as_secs_f32()
    );

    print_result(&result.text, &result.tokens);
    Ok(())
}

fn parse_args() -> eyre::Result<PathBuf> {
    env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("Usage: parakeet <audio.wav>"))
}

fn print_audio_info(audio_path: &PathBuf) -> eyre::Result<()> {
    let reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let duration = reader.duration() as f32 / spec.sample_rate as f32;
    println!("Audio metadata:");
    println!("  Sample rate: {} Hz", spec.sample_rate);
    println!("  Channels: {}", spec.channels);
    println!("  Bits per sample: {}", spec.bits_per_sample);
    println!("  Duration: {:.2} s", duration);
    Ok(())
}

fn fetch_model() -> eyre::Result<PathBuf> {
    println!("Locating model...");
    let api = Api::new()?;
    let repo = api.model(MODEL_ID.to_string());

    let (first, files) = MODEL_FILES
        .split_first()
        .ok_or_else(|| eyre::eyre!("No model files specified"))?;

    let path = repo
        .get(first)?
        .parent()
        .map(|p| p.to_path_buf())
        .ok_or_else(|| eyre::eyre!("Failed to get model directory"))?;

    for file in files {
        repo.get(file)?;
    }

    Ok(path)
}

fn load_model(model_dir: &PathBuf) -> eyre::Result<ParakeetTDT> {
    println!("Loading model from: {}", model_dir.display());

    // Configure execution providers directly with full control
    // The order matters: first available provider will be used
    let builder = Session::builder()?.with_execution_providers(build_execution_providers())?;

    Ok(ParakeetTDT::from_pretrained(model_dir, Some(builder))?)
}

/// Build execution providers based on enabled features.
///
/// This demonstrates the new direct ORT provider configuration:
/// - You have full control over provider settings (device type, device ID, etc.)
/// - Providers are tried in order; first available one is used
/// - Always include CPU as fallback
fn build_execution_providers() -> Vec<ExecutionProviderDispatch> {
    // OpenVINO with configurable device support (when feature is enabled)
    #[cfg(feature = "openvino")]
    {
        // Get device type from environment variable or use default "CPU"
        // Supported: "CPU", "GPU", "GPU.0", "GPU.1", "NPU", "AUTO:GPU,CPU", etc.
        let device_type = env::var("OPENVINO_DEVICE").unwrap_or_else(|_| "CPU".to_string());
        println!("  Configuring OpenVINO provider (device: {})", device_type);

        // Return OpenVINO provider with CPU fallback
        // If OpenVINO EP DLLs are not found at runtime, it will fallback to CPU
        return vec![
            OpenVINOExecutionProvider::default()
                .with_device_type(device_type)
                .build()
                .error_on_failure(),
        ];
    }

    // CUDA support (when feature is enabled)
    #[cfg(feature = "cuda")]
    {
        println!("  Attempting CUDA provider");
        return vec![
            CUDAExecutionProvider::default().with_device_id(0).build(),
            CPUExecutionProvider::default().build(),
        ];
    }

    // Default: CPU provider
    #[cfg(not(any(feature = "openvino", feature = "cuda")))]
    {
        println!("  CPU provider enabled");
        vec![CPUExecutionProvider::default().build().error_on_failure()]
    }
}

fn transcribe(
    model: &mut ParakeetTDT,
    audio_path: &PathBuf,
) -> eyre::Result<parakeet_rs::TranscriptionResult> {
    println!("Transcribing: {}", audio_path.display());
    Ok(model.transcribe_file(audio_path, Some(TimestampMode::Sentences))?)
}

fn print_result(text: &str, tokens: &[parakeet_rs::TimedToken]) {
    println!("\n{text}");
    println!("\nSentences:");
    for t in tokens {
        println!("[{:.2}s - {:.2}s]: {}", t.start, t.end, t.text);
    }
}
