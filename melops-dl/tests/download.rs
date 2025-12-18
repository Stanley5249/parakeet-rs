//! ASR preset download integration tests.
//!
//! Tests: YouTube download, WAV format (16kHz mono 16-bit PCM via hound),
//! path grouping (Extractor/uploader/title).
//!
//! Uses "Me at the zoo" (jNQXAC9IVRw) - predictable metadata.

use eyre::{Context, Result};
use melops_dl::asr::{ASR_OUTPUT_TEMPLATE, AudioFormat};
use melops_dl::dl::{DownloadInfo, DownloadOptions, OutputPaths, OutputTemplates, download};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

const URL: &str = "https://youtu.be/jNQXAC9IVRw";
const EXPECTED_TITLE: &str = "Me at the zoo";
const EXPECTED_UPLOADER: &str = "jawed";
const EXPECTED_EXTRACTOR: &str = "Youtube";

struct TestContext {
    temp_dir: PathBuf,
    info: DownloadInfo,
}

static TEST_CONTEXT: LazyLock<Result<TestContext>> = LazyLock::new(|| {
    let mut temp_dir = std::env::temp_dir();
    temp_dir.push("melops-dl-test");

    // Clean up previous test run
    if temp_dir.exists() {
        std::fs::remove_dir_all(&temp_dir).ok();
    }
    std::fs::create_dir_all(&temp_dir).context("failed to create temp dir")?;

    let paths = OutputPaths::simple(&temp_dir, &temp_dir);
    let mut preset: DownloadOptions = AudioFormat::Pcm16.into();
    preset.paths = Some(paths);
    preset.outtmpl = Some(OutputTemplates::simple(ASR_OUTPUT_TEMPLATE.to_string()));

    let info = download(URL, preset).context("yt-dlp download failed for ASR Pcm16 preset")?;

    Ok(TestContext { temp_dir, info })
});

#[track_caller]
fn get_test_context() -> &'static TestContext {
    TEST_CONTEXT.as_ref().expect("download failed")
}

fn expected_dir(temp_dir: &Path) -> PathBuf {
    let mut path = temp_dir.to_path_buf();
    path.push(EXPECTED_EXTRACTOR);
    path.push(EXPECTED_UPLOADER);
    path.push(EXPECTED_TITLE);
    path
}

fn wav_path(temp_dir: &Path) -> PathBuf {
    let mut path = expected_dir(temp_dir);
    path.push(format!("{EXPECTED_TITLE}.wav"));
    path
}

#[test]
#[ignore = "network I/O"]
fn files_exist() {
    let ctx = get_test_context();

    let wav = wav_path(&ctx.temp_dir);
    assert!(wav.exists(), "WAV file not found: {wav:?}");

    let mut info_json = expected_dir(&ctx.temp_dir);
    info_json.push(format!("{EXPECTED_TITLE}.info.json"));
    assert!(info_json.exists(), "info.json not found: {info_json:?}");
}

#[test]
#[ignore = "network I/O"]
fn path_structure() {
    let ctx = get_test_context();
    let wav = wav_path(&ctx.temp_dir);

    let mut expected = ctx.temp_dir.clone();
    expected.push(EXPECTED_EXTRACTOR);
    expected.push(EXPECTED_UPLOADER);
    expected.push(EXPECTED_TITLE);
    expected.push(format!("{EXPECTED_TITLE}.wav"));

    assert_eq!(wav, expected, "path structure mismatch");
}

#[test]
#[ignore = "network I/O"]
fn wav_format() {
    let ctx = get_test_context();
    let wav = wav_path(&ctx.temp_dir);

    let reader = hound::WavReader::open(&wav).expect("failed to open WAV file");
    let spec = reader.spec();

    assert!(
        matches!(
            spec,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            }
        ),
        "unexpected WAV format: {spec:?}, expected 16kHz mono 16-bit PCM"
    );
}

#[test]
#[ignore = "network I/O"]
fn info_dict_fields() {
    let ctx = get_test_context();
    let info = &ctx.info;

    assert_eq!(info.id, "jNQXAC9IVRw", "video ID mismatch");
    assert_eq!(info.title, EXPECTED_TITLE, "title mismatch");
    assert_eq!(
        info.extractor_key.as_deref(),
        Some(EXPECTED_EXTRACTOR),
        "extractor mismatch"
    );
    assert!(
        info.uploader.as_deref() == Some(EXPECTED_UPLOADER)
            || info.uploader_id.as_deref() == Some(EXPECTED_UPLOADER),
        "uploader mismatch: {:?} / {:?}",
        info.uploader,
        info.uploader_id
    );
    assert!(info.duration.is_some(), "missing duration");

    println!("{info:?}");
}
