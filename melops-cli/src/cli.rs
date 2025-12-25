//! CLI argument definitions using clap.

use clap::{Parser, Subcommand};
use eyre::Result;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "melops")]
#[command(about = "Speech recognition and audio download tools")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Transcribe audio file to SRT subtitles
    Run {
        /// Path to input WAV file
        path: PathBuf,

        /// Output SRT path (default: same as input with .srt extension)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Download and transcribe audio from URL
    Dl {
        /// URL to download
        url: String,

        /// Output directory (default: system download directory)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

/// Execute CLI command - separated for testing.
pub fn run_cli(cli: Cli) -> Result<()> {
    tracing::debug!(?cli, "parsed arguments");

    match cli.command {
        Commands::Run { path, output } => crate::run::execute(path, output),
        Commands::Dl { url, output } => crate::dl::execute(url, output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_run_command() {
        let cli = Cli::parse_from(["melops", "run", "audio.wav"]);

        assert!(matches!(
            &cli.command,
            Commands::Run { path, output: None }
            if path == "audio.wav"
        ));
    }

    #[test]
    fn parses_run_with_output() {
        let cli = Cli::parse_from(["melops", "run", "audio.wav", "-o", "output.srt"]);

        assert!(matches!(
            &cli.command,
            Commands::Run { path, output }
            if path.to_str() == Some("audio.wav")
            && output.as_deref().is_some_and(|p| p == "output.srt")
        ));
    }

    #[test]
    fn parses_dl_command() {
        let cli = Cli::parse_from(["melops", "dl", "https://example.com/video"]);

        assert!(matches!(
            &cli.command,
            Commands::Dl {
                url,
                output: None,
            } if url == "https://example.com/video"
        ));
    }

    #[test]
    fn parses_dl_with_output() {
        let cli = Cli::parse_from([
            "melops",
            "dl",
            "https://example.com/video",
            "-o",
            "/tmp/output",
        ]);

        assert!(matches!(
            &cli.command,
            Commands::Dl {
                url,
                output,
            } if url == "https://example.com/video" &&
                 output.as_deref().is_some_and(|p| p == "/tmp/output")
        ));
    }
}
