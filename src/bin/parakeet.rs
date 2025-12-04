//! Parakeet CLI - Speech-to-text transcription tool

use hf_hub::api::sync::Api;
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, Transcriber};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const MODEL_ID: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";
const MODEL_FILES: &[&str] = &[
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
    "vocab.txt",
];

fn main() -> ExitCode {
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
    let start = Instant::now();

    let model_dir = fetch_model()?;
    let mut model = load_model(&model_dir)?;
    let result = transcribe(&mut model, &audio_path)?;

    print_result(&result.text, &result.tokens, start.elapsed());
    Ok(())
}

fn parse_args() -> eyre::Result<PathBuf> {
    env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("Usage: parakeet <audio.wav>"))
}

fn fetch_model() -> eyre::Result<PathBuf> {
    println!("Locating model...");
    let api = Api::new()?;
    let repo = api.model(MODEL_ID.to_string());

    for file in MODEL_FILES {
        repo.get(file)?;
    }

    repo.get(MODEL_FILES[0])?
        .parent()
        .map(|p| p.to_path_buf())
        .ok_or_else(|| eyre::eyre!("Failed to get model directory"))
}

fn load_model(model_dir: &PathBuf) -> eyre::Result<ParakeetTDT> {
    println!("Loading model from: {}", model_dir.display());
    let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::OpenVINO);
    Ok(ParakeetTDT::from_pretrained(model_dir, Some(config))?)
}

fn transcribe(
    model: &mut ParakeetTDT,
    audio_path: &PathBuf,
) -> eyre::Result<parakeet_rs::TranscriptionResult> {
    println!("Transcribing: {}", audio_path.display());
    Ok(model.transcribe_file(audio_path, Some(TimestampMode::Sentences))?)
}

fn print_result(text: &str, tokens: &[parakeet_rs::TimedToken], elapsed: std::time::Duration) {
    println!("\n{text}");
    println!("\nSentences:");
    for t in tokens {
        println!("[{:.2}s - {:.2}s]: {}", t.start, t.end, t.text);
    }
    println!("\n✓ Completed in {:.2}s", elapsed.as_secs_f32());
}
