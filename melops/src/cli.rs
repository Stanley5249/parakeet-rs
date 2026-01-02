//! CLI argument definitions using clap.

use crate::cap::CapCommand;
use crate::dl::DlCommand;
use clap::{Args, Parser, Subcommand, ValueEnum};
use eyre::Result;
use melops_asr::chunk::{ChunkConfig, DEFAULT_CHUNK_DURATION, DEFAULT_CHUNK_OVERLAP};

#[derive(Debug, Parser)]
#[command(name = "mel")]
#[command(about = "Audio captioning and download tools")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Generate captions from audio file to SRT subtitles
    Cap(CapCommand),

    /// Download and generate captions from audio URL
    Dl(DlCommand),
}

/// Model source type selection.
#[derive(Debug, Default, Clone, Copy, ValueEnum)]
pub enum ModelSource {
    /// Automatically detect: try cache first, fall back to HuggingFace Hub
    #[default]
    Auto,
    /// Load from local filesystem path
    Path,
    /// Load from HuggingFace cache only (error if not cached)
    Cache,
    /// Download from HuggingFace Hub (may use cache)
    Api,
}

/// CLI arguments for model loading configuration.
#[derive(Args, Debug)]
pub struct ModelArgs {
    /// Model repository ID or local path
    #[arg(long)]
    pub model_id: String,

    /// Model source type (auto, path, cache, api)
    #[arg(long, value_enum, default_value_t = ModelSource::default())]
    pub model_source: ModelSource,
}

/// CLI arguments for chunk configuration.
#[derive(Args, Clone, Copy, Debug)]
pub struct ChunkArgs {
    /// Chunk duration in seconds for long audio
    #[arg(long, default_value_t = DEFAULT_CHUNK_DURATION)]
    pub duration: f32,

    /// Chunk overlap in seconds
    #[arg(long, default_value_t = DEFAULT_CHUNK_OVERLAP)]
    pub overlap: f32,
}

impl From<ChunkArgs> for ChunkConfig {
    fn from(args: ChunkArgs) -> Self {
        ChunkConfig::new(args.duration, args.overlap)
    }
}

impl From<ChunkConfig> for ChunkArgs {
    fn from(config: ChunkConfig) -> Self {
        ChunkArgs {
            duration: config.duration,
            overlap: config.overlap,
        }
    }
}

/// Shared caption generation options.
#[derive(Args, Debug)]
pub struct CaptionArgs {
    /// Show preview of first and last subtitles
    #[arg(long)]
    pub preview: bool,

    #[command(flatten)]
    pub chunk_args: ChunkArgs,
}

/// Execute CLI command - separated for testing.
pub fn run(cli: Cli) -> Result<()> {
    tracing::debug!(?cli, "parsed arguments");

    match cli.command {
        Commands::Cap(args) => crate::cap::execute(args.try_into()?),
        Commands::Dl(args) => crate::dl::execute(args.try_into()?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cap_command() {
        let cli = Cli::parse_from(["mel", "cap", "--model-id", "test_model", "audio.wav"]);

        match &cli.command {
            Commands::Cap(crate::cap::CapCommand {
                path, output: None, ..
            }) if path.to_str() == Some("audio.wav") => {}
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }

    #[test]
    fn parses_cap_with_output() {
        let cli = Cli::parse_from([
            "mel",
            "cap",
            "--model-id",
            "test_model",
            "audio.wav",
            "-o",
            "output.srt",
        ]);

        match &cli.command {
            Commands::Cap(crate::cap::CapCommand {
                path,
                output: Some(output),
                ..
            }) if path.to_str() == Some("audio.wav") && output.to_str() == Some("output.srt") => {}
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }

    #[test]
    fn parses_dl_command() {
        let cli = Cli::parse_from([
            "mel",
            "dl",
            "--model-id",
            "test_model",
            "https://example.com/video",
        ]);

        match &cli.command {
            Commands::Dl(crate::dl::DlCommand {
                url, output: None, ..
            }) if url == "https://example.com/video" => {}
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }

    #[test]
    fn parses_dl_with_output() {
        let cli = Cli::parse_from([
            "mel",
            "dl",
            "--model-id",
            "test_model",
            "https://example.com/video",
            "-o",
            "/tmp/output",
        ]);

        match &cli.command {
            Commands::Dl(crate::dl::DlCommand {
                url,
                output: Some(output),
                ..
            }) if url == "https://example.com/video" && output.to_str() == Some("/tmp/output") => {}
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }
}
