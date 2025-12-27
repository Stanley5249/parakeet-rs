//! CLI argument definitions using clap.

use clap::{Parser, Subcommand};
use eyre::Result;

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
    Cap(crate::cap::Args),

    /// Download and generate captions from audio URL
    Dl(crate::dl::Args),
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
    use melops_asr::chunk::ChunkConfig;

    fn assert_default_chunk_config(config: &ChunkConfig) {
        assert!((config.duration - 360.0).abs() < 0.001);
        assert!((config.overlap - 1.0).abs() < 0.001);
    }

    #[test]
    fn parses_run_command() {
        let cli = Cli::parse_from(["mel", "cap", "audio.wav"]);

        match &cli.command {
            Commands::Cap(crate::cap::Args {
                path,
                output: None,
                chunk_config,
            }) if path.to_str() == Some("audio.wav") => {
                assert_default_chunk_config(chunk_config);
            }
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }

    #[test]
    fn parses_run_with_output() {
        let cli = Cli::parse_from(["mel", "cap", "audio.wav", "-o", "output.srt"]);

        match &cli.command {
            Commands::Cap(crate::cap::Args {
                path,
                output: Some(output),
                chunk_config,
            }) if path.to_str() == Some("audio.wav") && output.to_str() == Some("output.srt") => {
                assert_default_chunk_config(chunk_config);
            }
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }

    #[test]
    fn parses_dl_command() {
        let cli = Cli::parse_from(["mel", "dl", "https://example.com/video"]);

        match &cli.command {
            Commands::Dl(crate::dl::Args {
                url,
                output: None,
                chunk_config,
            }) if url == "https://example.com/video" => {
                assert_default_chunk_config(chunk_config);
            }
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }

    #[test]
    fn parses_dl_with_output() {
        let cli = Cli::parse_from([
            "mel",
            "dl",
            "https://example.com/video",
            "-o",
            "/tmp/output",
        ]);

        match &cli.command {
            Commands::Dl(crate::dl::Args {
                url,
                output: Some(output),
                chunk_config,
            }) if url == "https://example.com/video" && output.to_str() == Some("/tmp/output") => {
                assert_default_chunk_config(chunk_config);
            }
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }
}
