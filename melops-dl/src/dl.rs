//! Type-safe yt-dlp Python API wrappers.
//!
//! Bindings to [yt-dlp](https://github.com/yt-dlp/yt-dlp) `YoutubeDL` parameters.
//!
//! ```no_run
//! use melops_dl::{dl::download, asr::AudioFormat};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let (file_path, info) = download("https://youtube.com/watch?v=example", AudioFormat::Pcm16.into())?;
//! println!("Downloaded '{}' to {:?}", info.title, file_path);
//! # Ok(())
//! # }
//! ```

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Filename templates using `%(field)s` syntax.
///
/// Maps output types to template strings. Key `default` is required.
///
/// See: <https://github.com/yt-dlp/yt-dlp#output-template>
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct OutputTemplates(pub Option<HashMap<String, String>>);

impl OutputTemplates {
    /// Creates template with single default pattern.
    ///
    /// ```
    /// # use melops_dl::dl::OutputTemplates;
    /// let templates = OutputTemplates::simple("%(title)s.%(ext)s".to_string());
    /// println!("{:?}", templates);
    /// ```
    pub fn simple(default: String) -> Self {
        Self(Some(HashMap::from([("default".to_string(), default)])))
    }
}

/// Download directories: `home`, `temp`, optional type-specific paths.
///
/// See: <https://github.com/yt-dlp/yt-dlp#filesystem-options>
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct OutputPaths(pub Option<HashMap<String, String>>);

impl OutputPaths {
    /// Creates paths with home and temp directories.
    pub fn simple(home: &Path, temp: &Path) -> Self {
        Self::default().with_home(home).with_temp(temp)
    }

    /// Uses system download dir for `home`, cache dir for `temp`.
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

/// Post-processor specification: `key` (e.g., `FFmpegExtractAudio`), optional `preferredcodec`.
///
/// See: <https://github.com/yt-dlp/yt-dlp#post-processing-options>
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct PostProcessor {
    /// Post-processor name (e.g., `FFmpegExtractAudio`, `FFmpegVideoConvertor`)
    pub key: String,
    /// Preferred output codec (e.g., `wav`, `mp3`, `mp4`)
    pub preferredcodec: Option<String>,
}

/// CLI arguments passed to yt-dlp post-processors.
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct PostProcessorArgs {
    /// FFmpeg arguments (e.g., `["-ar", "16000", "-ac", "1"]` for 16kHz mono)
    ///
    /// ```
    /// # use melops_dl::dl::PostProcessorArgs;
    /// let args = PostProcessorArgs {
    ///     ffmpeg: vec!["-ar".to_string(), "16000".to_string()],
    /// };
    /// println!("Sample rate: 16kHz via {:?}", args.ffmpeg);
    /// ```
    pub ffmpeg: Vec<String>,
}

/// yt-dlp download configuration passed to `YoutubeDL(params)`.
///
/// Maps to Python dict for `YoutubeDL` constructor. Use `cli_to_api.py` to convert CLI flags.
///
/// See: <https://github.com/yt-dlp/yt-dlp#embedding-yt-dlp>
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct DownloadOptions {
    /// Format selection (e.g., `bestaudio`, `bestvideo+bestaudio`)
    pub format: Option<String>,
    /// Download directories (`home`, `temp`, type-specific)
    pub paths: Option<OutputPaths>,
    /// Filename templates for different output types
    pub outtmpl: Option<OutputTemplates>,
    /// Post-processing steps (extraction, conversion)
    pub postprocessors: Option<Vec<PostProcessor>>,
    /// CLI arguments for post-processors
    pub postprocessor_args: Option<PostProcessorArgs>,
    /// Write metadata JSON to disk
    pub writeinfojson: Option<bool>,
    /// Restrict filenames to ASCII characters
    pub restrictfilenames: Option<bool>,
    /// Fetch comments (warning: slow)
    pub getcomments: Option<bool>,
    /// Suppress console output
    pub quiet: Option<bool>,
    /// Suppress warnings
    pub no_warnings: Option<bool>,
    /// Keep video file after post-processing (prevents deletion of original file)
    pub keepvideo: Option<bool>,
}

/// Essential metadata from yt-dlp info dict.
///
/// Subset of fields from `YoutubeDL.sanitize_info()`. Full dict available via JSON.
///
/// See: <https://github.com/yt-dlp/yt-dlp#output-template>
#[derive(Clone, Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub struct DownloadInfo {
    /// Video ID (platform-specific, required)
    pub id: String,
    /// Video title (required)
    pub title: String,
    /// Extractor name (e.g., `Youtube`, `Vimeo`)
    pub extractor_key: Option<String>,
    /// Uploader full name
    pub uploader: Option<String>,
    /// Uploader username or channel ID
    pub uploader_id: Option<String>,
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Video webpage URL
    pub webpage_url: Option<String>,
    /// Video description text
    pub description: Option<String>,
    /// Upload date in UTC (`YYYYMMDD`)
    pub upload_date: Option<String>,
    /// View count
    pub view_count: Option<i64>,
    /// Number of likes
    pub like_count: Option<i64>,
    /// Age restriction (`0` = none)
    pub age_limit: Option<i64>,
}

/// Downloads media from URL using yt-dlp.
///
/// Returns `(file_path, info)` where `file_path` is the final processed file location.
/// `file_path` is `None` if download failed or no file was saved.
///
/// # Errors
///
/// Returns `PyErr` if yt-dlp download fails or Python API call errors.
///
/// # Example
///
/// ```no_run
/// use melops_dl::{dl::download, asr::AudioFormat};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let (file_path, info) = download(
///     "https://youtube.com/watch?v=BaW_jenozKc",
///     AudioFormat::Pcm16.into()
/// )?;
///
/// if let Some(path) = file_path {
///     println!("Downloaded '{}' to: {}", info.title, path.display());
/// }
/// # Ok(())
/// # }
/// ```
pub fn download(
    url: &str,
    opts: DownloadOptions,
) -> Result<(Option<PathBuf>, DownloadInfo), PyErr> {
    Python::attach(|py| {
        let module = PyModule::from_code(py, c_str!(include_str!("./dl.py")), c"dl.py", c"dl")?;

        let py_params = opts.into_pyobject(py)?;

        module
            .getattr("download")?
            .call1((url, py_params))?
            .extract()
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
                c"{'format': 'bestvideo+bestaudio', 'paths': None, 'outtmpl': None, 'postprocessors': None, 'postprocessor_args': None, 'writeinfojson': False, 'restrictfilenames': None, 'getcomments': None, 'quiet': False, 'no_warnings': None, 'keepvideo': None}"
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
