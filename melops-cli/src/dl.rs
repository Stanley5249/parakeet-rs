//! Dl subcommand - download and transcribe audio from URL.

use eyre::{Context, Result};
use melops_dl::asr::AudioFormat;
use melops_dl::dl::{DownloadInfo, DownloadOptions, download};
use std::path::PathBuf;

pub fn execute(url: String, output_dir: Option<PathBuf>) -> Result<()> {
    tracing::info!(url, "downloading audio");

    let mut opts: DownloadOptions = AudioFormat::Pcm16.into();

    // Override output directory if provided
    if let Some(home) = output_dir.as_deref() {
        opts.paths = Some(
            opts.paths
                .expect("paths should always be some")
                .with_home(home),
        );
    }

    // Download audio
    let info = download(&url, opts).wrap_err("failed to download audio")?;

    // Resolve downloaded audio path
    let audio_path = resolve_audio_path(&info, output_dir)?;

    tracing::info!(
        downloaded = %audio_path.display(),
        "audio downloaded, starting transcription"
    );

    // Generate SRT path (same directory and name as audio, but .srt extension)
    let mut srt_path = audio_path.clone();
    srt_path.set_extension("srt");

    // Transcribe using run module's logic
    crate::run::execute(audio_path, Some(srt_path))
}

/// Resolve the downloaded audio file path from info dict and output directory.
///
/// Uses the ASR output template pattern: `Extractor/uploader/title/title.wav`
fn resolve_audio_path(info: &DownloadInfo, output_dir: Option<PathBuf>) -> Result<PathBuf> {
    // Determine base directory
    let mut path = output_dir
        .unwrap_or_else(|| dirs::download_dir().expect("failed to find download directory"));

    // Add extractor_key
    if let Some(ref extractor) = info.extractor_key {
        path.push(extractor);
    }

    // Add uploader or uploader_id
    if let Some(uploader) = info.uploader.as_deref() {
        path.push(uploader);
    } else if let Some(uploader_id) = info.uploader_id.as_deref() {
        path.push(uploader_id);
    }

    // Add title directory
    path.push(&info.title);

    // Add filename: title.wav
    path.push(&info.title);
    path.set_extension("wav");

    tracing::debug!(resolved_path = %path.display(), "resolved audio path");

    if !path.exists() {
        tracing::error!(
            path = %path.display(),
            "audio file not found at expected location"
        );
    }

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use melops_dl::dl::DownloadInfo;

    #[test]
    fn constructs_audio_path_from_metadata() {
        let info = DownloadInfo {
            id: "abc123".to_string(),
            title: "Test Video".to_string(),
            extractor_key: Some("Youtube".to_string()),
            uploader: Some("test_uploader".to_string()),
            uploader_id: None,
            duration: Some(60.0),
            webpage_url: None,
            description: None,
            upload_date: None,
            view_count: None,
            like_count: None,
            age_limit: None,
        };

        let base = PathBuf::from("/tmp/downloads");
        let path = resolve_audio_path(&info, Some(base)).unwrap();

        let mut expected = PathBuf::from("/tmp/downloads");
        expected.push("Youtube");
        expected.push("test_uploader");
        expected.push("Test Video");
        expected.push("Test Video.wav");

        assert_eq!(path, expected);
    }

    #[test]
    fn uses_uploader_id_fallback() {
        let info = DownloadInfo {
            id: "abc123".to_string(),
            title: "Test Video".to_string(),
            extractor_key: Some("Youtube".to_string()),
            uploader: None,
            uploader_id: Some("uploader_id_123".to_string()),
            duration: Some(60.0),
            webpage_url: None,
            description: None,
            upload_date: None,
            view_count: None,
            like_count: None,
            age_limit: None,
        };

        let base = PathBuf::from("/tmp/downloads");
        let path = resolve_audio_path(&info, Some(base)).unwrap();

        assert!(path.to_string_lossy().contains("uploader_id_123"));
    }

    #[test]
    fn uses_system_download_dir_when_none() {
        let info = DownloadInfo {
            id: "abc123".to_string(),
            title: "Test".to_string(),
            extractor_key: Some("Youtube".to_string()),
            uploader: Some("user".to_string()),
            uploader_id: None,
            duration: None,
            webpage_url: None,
            description: None,
            upload_date: None,
            view_count: None,
            like_count: None,
            age_limit: None,
        };

        let path = resolve_audio_path(&info, None).unwrap();

        // Should start with system download directory
        let download_dir = dirs::download_dir().expect("failed to get download dir");
        assert!(path.starts_with(&download_dir));
    }
}
