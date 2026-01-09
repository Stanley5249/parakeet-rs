# melops-dl

Minimal Rust wrapper for [yt-dlp](https://github.com/yt-dlp/yt-dlp) Python library.

## Motivation

YouTube and similar platforms constantly change their APIs to block downloaders. The yt-dlp community maintains compatibility through rapid updates. Rather than reimplementing these workarounds in Rust, this crate provides type-safe bindings to the battle-tested Python implementation.

## Quick Start

```rust
use melops_dl::{dl::download, asr::AudioFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    download(&["https://youtube.com/watch?v=example"], AudioFormat::Pcm16.into())?;
    Ok(())
}
```

Output: `downloads/domain/uploader/title.wav` + `.info.json` metadata

See [yt-dlp documentation](https://github.com/yt-dlp/yt-dlp) for full configuration options.

## License

MIT
