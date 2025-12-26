//! Mel CLI - Audio captioning tool

use clap::Parser;
use eyre::Result;
use melops::cli::{Cli, run_cli};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let (non_blocking, _guard) = tracing_appender::non_blocking(std::io::stderr());

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    run_cli(Cli::parse())
}
