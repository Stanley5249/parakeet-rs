//! Integration tests for melops CLI.

use clap::Parser;
use melops_cli::cli::{Cli, run_cli};

const URL: &str = "https://youtu.be/jNQXAC9IVRw";

#[test]
#[ignore = "network I/O and model download required"]
fn dl_downloads_and_transcribes() {
    let temp_dir = std::env::temp_dir().join("melops-cli-test");

    // Clean up previous test run
    if temp_dir.exists() {
        std::fs::remove_dir_all(&temp_dir).ok();
    }
    std::fs::create_dir_all(&temp_dir).expect("failed to create temp dir");

    let cli = Cli::parse_from(["melops", "dl", URL, "-o", temp_dir.to_str().unwrap()]);

    run_cli(cli).expect("failed to download and transcribe");

    // Verify SRT file was created
    // Expected path: temp_dir/Youtube/jawed/Me at the zoo/Me at the zoo.srt
    let mut srt_path = temp_dir.clone();
    srt_path.push("Youtube");
    srt_path.push("jawed");
    srt_path.push("Me at the zoo");
    srt_path.push("Me at the zoo.srt");

    assert!(
        srt_path.exists(),
        "SRT file not found: {}",
        srt_path.display()
    );
}
