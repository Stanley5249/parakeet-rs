//! yt-dlp Python API wrappers.
//!
//! Type-safe bindings to [yt-dlp](https://github.com/yt-dlp/yt-dlp) `YoutubeDL` parameters.
//!
//! ```no_run
//! use melops_dl::{dl::download, asr::AudioFormat};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let info = download("https://youtube.com/watch?v=example", AudioFormat::Pcm16.into())?;
//! println!("Downloaded: {}", info.title);
//! # Ok(())
//! # }
//! ```

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Filename templates using `%(field)s` syntax. Key `default` required.
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct OutputTemplates(pub Option<HashMap<String, String>>);

impl OutputTemplates {
    /// Create with a single default template.
    pub fn simple(default: String) -> Self {
        Self(Some(HashMap::from([("default".to_string(), default)])))
    }
}

/// Download directories: `home`, `temp`, optional type-specific paths.
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct OutputPaths(pub Option<HashMap<String, String>>);

impl OutputPaths {
    /// Create with home and temp directories.
    pub fn simple(home: &Path, temp: &Path) -> Self {
        Self::default().with_home(home).with_temp(temp)
    }

    /// System download dir for `home`, cache dir for `temp`.
    pub fn system_default() -> Self {
        let home = dirs::download_dir().expect("failed to get download directory");
        let temp = std::env::temp_dir();
        Self::simple(&home, &temp)
    }

    pub fn with_home(self, home: &Path) -> Self {
        self.with_key("home".to_string(), home)
    }

    pub fn with_temp(self, temp: &Path) -> Self {
        self.with_key("temp".to_string(), temp)
    }

    fn with_key(self, key: String, value: &Path) -> Self {
        let mut inner = self.0.unwrap_or_default();
        inner.insert(key, value.to_string_lossy().to_string());
        Self(Some(inner))
    }
}

/// Post-download operation: `key` (e.g., `"FFmpegExtractAudio"`), optional `preferredcodec`.
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct PostProcessor {
    pub key: String,
    pub preferredcodec: Option<String>,
}

/// CLI arguments passed to yt-dlp post-processors.
///
/// The `ffmpeg` field contains arguments passed to FFmpeg during post-processing.
/// Example: `["-ar", "16000", "-ac", "1"]` for 16kHz mono conversion.
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct PostProcessorArgs {
    pub ffmpeg: Vec<String>,
}

/// yt-dlp download configuration passed to `YoutubeDL(params)`.
///
/// # Warning
/// Setting `getcomments` to true can take a very long time!
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct DownloadOptions {
    pub format: Option<String>,
    pub paths: Option<OutputPaths>,
    pub outtmpl: Option<OutputTemplates>,
    pub postprocessors: Option<Vec<PostProcessor>>,
    pub postprocessor_args: Option<PostProcessorArgs>,
    pub writeinfojson: Option<bool>,
    pub getcomments: Option<bool>,
    pub quiet: Option<bool>,
    pub no_warnings: Option<bool>,
}

/// Essential metadata from yt-dlp info dict.
///
/// Extracted via `FromPyObject` from the sanitized info dict returned by `extract_info`.
/// Only includes fields commonly needed for ASR pipelines.
#[derive(Clone, Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub struct DownloadInfo {
    /// Video identifier (required by yt-dlp)
    pub id: String,
    /// Video title (required by yt-dlp)
    pub title: String,
    /// Extractor name (e.g., "Youtube")
    pub extractor_key: Option<String>,
    /// Full name of the video uploader
    pub uploader: Option<String>,
    /// Nickname or ID of the video uploader
    pub uploader_id: Option<String>,
    /// Length of the video in seconds
    pub duration: Option<f64>,
    /// URL to the video webpage
    pub webpage_url: Option<String>,
    /// Full video description
    pub description: Option<String>,
    /// Video upload date in UTC (YYYYMMDD)
    pub upload_date: Option<String>,
    /// How many users have watched the video
    pub view_count: Option<i64>,
    /// Number of positive ratings
    pub like_count: Option<i64>,
    /// Age restriction for the video (0 = no restriction)
    pub age_limit: Option<i64>,
}

/// Download a single URL and return the info dict.
///
/// Uses `extract_info(url, download=True)` to download and get metadata in one request.
pub fn download(url: &str, opts: DownloadOptions) -> Result<DownloadInfo, PyErr> {
    Python::attach(|py| {
        let module = PyModule::from_code(py, c_str!(include_str!("./dl.py")), c"dl.py", c"dl")?;

        let py_params = opts.into_pyobject(py)?;

        let info = module.getattr("download")?.call1((url, py_params))?;

        info.extract()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyAnyMethods;
    use std::ffi::CStr;

    /// Compare Python object with dict/list literal using recursive equality.
    #[track_caller]
    fn assert_py_eq(py: Python, py_obj: &Bound<PyAny>, expected: &'static CStr) {
        let py_expected = py.eval(expected, None, None).unwrap();
        assert!(py_obj.eq(&py_expected).unwrap());
    }

    #[test]
    fn output_templates_default() {
        Python::attach(|py| {
            let templates = OutputTemplates::default();
            let py_obj = templates.into_pyobject(py).unwrap();
            assert!(py_obj.is_none());
        });
    }

    #[test]
    fn output_templates_simple() {
        Python::attach(|py| {
            let templates = OutputTemplates::simple("%(title)s.%(ext)s".to_string());
            let py_obj = templates.into_pyobject(py).unwrap();
            assert_py_eq(py, py_obj.as_any(), c"{'default': '%(title)s.%(ext)s'}");
        });
    }

    #[test]
    fn output_templates_custom() {
        Python::attach(|py| {
            let map = HashMap::from([
                ("default".to_string(), "%(title)s.%(ext)s".to_string()),
                (
                    "infojson".to_string(),
                    "metadata/%(title)s.json".to_string(),
                ),
            ]);

            let templates = OutputTemplates(Some(map));
            let py_obj = templates.into_pyobject(py).unwrap();
            assert_py_eq(
                py,
                py_obj.as_any(),
                c"{'default': '%(title)s.%(ext)s', 'infojson': 'metadata/%(title)s.json'}",
            );
        });
    }

    #[test]
    fn paths_system_default() {
        Python::attach(|py| {
            let paths = OutputPaths::system_default();
            let py_obj = paths.into_pyobject(py).unwrap();

            // Verify structure (can't compare exact paths as they're system-dependent)
            assert!(py_obj.contains("home").unwrap());
            assert!(py_obj.contains("temp").unwrap());
            assert!(py_obj.len().unwrap() == 2);
        });
    }

    #[test]
    fn paths_custom() {
        Python::attach(|py| {
            let map = HashMap::from([
                ("home".to_string(), "/custom/downloads".to_string()),
                ("temp".to_string(), "/custom/temp".to_string()),
                ("infojson".to_string(), "/custom/metadata".to_string()),
            ]);

            let paths = OutputPaths(Some(map));
            let py_obj = paths.into_pyobject(py).unwrap();
            assert_py_eq(
                py,
                py_obj.as_any(),
                c"{'home': '/custom/downloads', 'temp': '/custom/temp', 'infojson': '/custom/metadata'}"
            );
        });
    }

    #[test]
    fn postprocessor() {
        Python::attach(|py| {
            let processor = PostProcessor {
                key: "FFmpegExtractAudio".to_string(),
                preferredcodec: Some("wav".to_string()),
            };
            let py_obj = processor.into_pyobject(py).unwrap();
            assert_py_eq(
                py,
                py_obj.as_any(),
                c"{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}",
            );
        });
    }

    #[test]
    fn postprocessor_args() {
        Python::attach(|py| {
            let args = PostProcessorArgs {
                ffmpeg: vec![
                    "-ar".to_string(),
                    "16000".to_string(),
                    "-ac".to_string(),
                    "1".to_string(),
                ],
            };
            let py_obj = args.into_pyobject(py).unwrap();
            assert_py_eq(
                py,
                py_obj.as_any(),
                c"{'ffmpeg': ['-ar', '16000', '-ac', '1']}",
            );
        });
    }

    #[test]
    fn dl_options_custom() {
        Python::attach(|py| {
            let opts = DownloadOptions {
                format: Some("bestvideo+bestaudio".to_string()),
                quiet: Some(false),
                writeinfojson: Some(false),
                ..Default::default()
            };
            let py_obj = opts.into_pyobject(py).unwrap();
            assert_py_eq(
                py,
                py_obj.as_any(),
                c"{'format': 'bestvideo+bestaudio', 'paths': None, 'outtmpl': None, 'postprocessors': None, 'postprocessor_args': None, 'writeinfojson': False, 'getcomments': None, 'quiet': False, 'no_warnings': None}"
            );
        });
    }

    #[test]
    fn postprocessors_list() {
        Python::attach(|py| {
            let processors = vec![
                PostProcessor {
                    key: "FFmpegExtractAudio".to_string(),
                    preferredcodec: Some("wav".to_string()),
                },
                PostProcessor {
                    key: "FFmpegVideoConvertor".to_string(),
                    preferredcodec: Some("mp4".to_string()),
                },
            ];

            let py_obj = processors.into_pyobject(py).unwrap();
            assert_py_eq(
                py,
                &py_obj,
                c"[{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}, {'key': 'FFmpegVideoConvertor', 'preferredcodec': 'mp4'}]"
            );
        });
    }
}
