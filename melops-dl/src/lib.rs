//! Type-safe Rust bindings to [yt-dlp](https://github.com/yt-dlp/yt-dlp) Python library.
//!
//! ## Modules
//!
//! - [`dl`] - Core yt-dlp API wrappers
//! - [`asr`] - ASR presets for 16kHz mono audio extraction
//!
//! ## Quick Start
//!
//! **ASR preset** (16kHz mono WAV):
//! ```no_run
//! use melops_dl::{dl::download, asr::AudioFormat};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! download(&["https://youtube.com/watch?v=example"], AudioFormat::Pcm16.into())?;
//! # Ok(())
//! # }
//! ```
//!
//! **Custom configuration**:
//! ```no_run
//! use melops_dl::dl::{download, DlOptions, Paths, OutputTemplates, PostProcessor};
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut templates = HashMap::new();
//! templates.insert("default".to_string(), "%(uploader)s/%(title)s.%(ext)s".to_string());
//!
//! let opts = DlOptions {
//!     format: Some("bestaudio".to_string()),
//!     paths: Some(Paths::system_default()),
//!     outtmpl: Some(OutputTemplates(Some(templates))),
//!     postprocessors: Some(vec![PostProcessor {
//!         key: "FFmpegExtractAudio".to_string(),
//!         preferredcodec: Some("mp3".to_string()),
//!     }]),
//!     writeinfojson: Some(true),
//!     quiet: Some(true),
//!     ..Default::default()
//! };
//!
//! download(&["https://youtube.com/watch?v=example"], opts)?;
//! # Ok(())
//! # }
//! ```

pub mod asr;
pub mod dl;
