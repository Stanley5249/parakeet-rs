//! Mel CLI - Audio captioning tool

use clap::Parser;
use eyre::Result;
use melops::cli::{Cli, run};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    color_eyre::install()?;

    let (non_blocking, _guard) = tracing_appender::non_blocking(std::io::stderr());

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    run(Cli::parse())
}
