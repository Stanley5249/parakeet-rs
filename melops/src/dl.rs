//! Dl subcommand - download and generate captions from audio URL.

use color_eyre::Section;
use eyre::{Context, OptionExt, Result, eyre};
use melops_dl::asr::AudioFormat;
use melops_dl::dl::{DownloadOptions, download};
use std::path::PathBuf;

pub fn execute(url: String, output_dir: Option<PathBuf>) -> Result<()> {
    tracing::info!(url, "downloading audio");

    let mut opts: DownloadOptions = AudioFormat::Pcm16.into();

    // Override output directory if provided
    if let Some(home) = output_dir.as_deref() {
        opts.paths = Some(opts.paths.expect("paths should be some").with_home(home));
    }

    // Download audio
    let (file_path, _info) = download(&url, opts).wrap_err("failed to download audio")?;

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
    crate::cap::execute(&audio_path, Some(srt_path))
        .with_note(|| {
            format!(
                "audio downloaded successfully to: {:?}",
                audio_path.display()
            )
        })
        .with_suggestion(|| format!("mel cap {:?}", audio_path.display()))
}
