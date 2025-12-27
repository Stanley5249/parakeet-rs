//! Dl subcommand - download and generate captions from audio URL.

use color_eyre::Section;
use eyre::{Context, OptionExt, Result, eyre};
use melops_asr::chunk::ChunkConfig;
use melops_dl::asr::AudioFormat;
use melops_dl::dl::{DownloadOptions, download};
use std::path::PathBuf;

/// CLI arguments for download and caption generation.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// URL to download
    pub url: String,

    /// Output directory (default: system download directory)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    #[command(flatten)]
    pub chunk_config: ChunkConfig,
}

/// Resolved configuration for download and caption generation.
#[derive(Debug)]
pub struct Config {
    pub url: String,
    pub output_dir: Option<PathBuf>,
    pub chunk_config: ChunkConfig,
}

impl TryFrom<Args> for Config {
    type Error = eyre::Error;

    fn try_from(args: Args) -> Result<Self> {
        Ok(Self {
            url: args.url,
            output_dir: args.output,
            chunk_config: args.chunk_config,
        })
    }
}

pub fn execute(config: Config) -> Result<()> {
    tracing::info!(url = config.url, "downloading audio");

    let mut opts: DownloadOptions = AudioFormat::Pcm16.into();

    // Override output directory if provided
    if let Some(home) = config.output_dir.as_deref() {
        opts.paths = Some(opts.paths.expect("paths should be some").with_home(home));
    }

    // Download audio
    let (file_path, _info) = download(&config.url, opts).wrap_err("failed to download audio")?;

    // Get actual downloaded file path from post_hook
    let audio_path = file_path.ok_or_eyre("yt-dlp did not return downloaded file path")?;

    // Verify file exists
    if !audio_path.exists() {
        let e = eyre!(
            "audio downloaded but file not found at expected location: {:?}",
            audio_path.display()
        )
        .suggestion(format!("mel cap {:?}", audio_path.display()));
        return Err(e);
    }

    tracing::info!(
        downloaded = ?audio_path.display(),
        "audio downloaded, starting captioning"
    );

    // Generate SRT path (same directory and name as audio, but .srt extension)
    let srt_path = audio_path.with_extension("srt");

    // Generate captions using cap module's logic
    let cap_config = crate::cap::Config {
        path: audio_path.clone(),
        output: Some(srt_path),
        chunk_config: config.chunk_config,
    };

    crate::cap::execute(cap_config)
        .with_note(|| {
            format!(
                "audio downloaded successfully to: {:?}",
                audio_path.display()
            )
        })
        .with_suggestion(|| format!("mel cap {:?}", audio_path.display()))
}
