//! yt-dlp Python API wrappers.
//!
//! Type-safe bindings to [yt-dlp](https://github.com/yt-dlp/yt-dlp) YoutubeDL parameters.
//!
//! ```no_run
//! use melops_dl::{dl::download, asr::AudioFormat};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! download(&["https://youtube.com/watch?v=example"], AudioFormat::Pcm16.into())?;
//! # Ok(())
//! # }
//! ```

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;

/// Filename templates using `%(field)s` syntax. Key `default` required.
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct OutputTemplates(pub Option<HashMap<String, String>>);

/// Download directories: `home`, `temp`, optional type-specific paths.
#[derive(Clone, Debug, IntoPyObject)]
pub struct Paths(pub Option<HashMap<String, String>>);

impl Paths {
    /// System download dir for `home`, cache dir for `temp`.
    pub fn system_default() -> Self {
        Self(Some(HashMap::from([
            (
                "home".to_string(),
                dirs::download_dir()
                    .expect("Failed to get download directory")
                    .to_string_lossy()
                    .to_string(),
            ),
            (
                "temp".to_string(),
                dirs::cache_dir()
                    .expect("Failed to get cache directory")
                    .to_string_lossy()
                    .to_string(),
            ),
        ])))
    }
}

/// Post-download operation: `key` (e.g., `"FFmpegExtractAudio"`), optional `preferredcodec`.
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct PostProcessor {
    pub key: String,
    pub preferredcodec: Option<String>,
}

/// FFmpeg CLI arguments for post-processors.
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct PostProcessorArgs {
    #[pyo3(item("ExtractAudio"))]
    pub extract_audio: Vec<String>,
}

/// yt-dlp download configuration passed to `YoutubeDL(params)`.
#[derive(Clone, Debug, Default, IntoPyObject)]
pub struct DlOptions {
    pub format: Option<String>,
    pub paths: Option<Paths>,
    pub outtmpl: Option<OutputTemplates>,
    pub postprocessors: Option<Vec<PostProcessor>>,
    pub postprocessor_args: Option<PostProcessorArgs>,
    pub writeinfojson: Option<bool>,
    pub getcomments: Option<bool>,
    pub quiet: Option<bool>,
    pub no_warnings: Option<bool>,
}

/// Download videos using yt-dlp.
pub fn download(urls: &[&str], opts: DlOptions) -> Result<(), PyErr> {
    Python::attach(|py| {
        let module = PyModule::from_code(py, c_str!(include_str!("./dl.py")), c"dl.py", c"dl")?;

        let py_urls = PyList::new(py, urls)?;
        let py_params = opts.into_pyobject(py)?;

        module.getattr("download")?.call1((py_urls, py_params))?;

        Ok(())
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
    fn test_output_templates_default() {
        Python::attach(|py| {
            let templates = OutputTemplates::default();
            let py_obj = templates.into_pyobject(py).unwrap();
            assert!(py_obj.is_none());
        });
    }

    #[test]
    fn test_output_templates_custom() {
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
    fn test_paths_system_default() {
        Python::attach(|py| {
            let paths = Paths::system_default();
            let py_obj = paths.into_pyobject(py).unwrap();

            // Verify structure (can't compare exact paths as they're system-dependent)
            assert!(py_obj.contains("home").unwrap());
            assert!(py_obj.contains("temp").unwrap());
            assert!(py_obj.len().unwrap() == 2);
        });
    }

    #[test]
    fn test_paths_custom() {
        Python::attach(|py| {
            let map = HashMap::from([
                ("home".to_string(), "/custom/downloads".to_string()),
                ("temp".to_string(), "/custom/temp".to_string()),
                ("infojson".to_string(), "/custom/metadata".to_string()),
            ]);

            let paths = Paths(Some(map));
            let py_obj = paths.into_pyobject(py).unwrap();
            assert_py_eq(
                py,
                py_obj.as_any(),
                c"{'home': '/custom/downloads', 'temp': '/custom/temp', 'infojson': '/custom/metadata'}"
            );
        });
    }

    #[test]
    fn test_postprocessor() {
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
    fn test_postprocessor_args() {
        Python::attach(|py| {
            let args = PostProcessorArgs {
                extract_audio: vec![
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
                c"{'ExtractAudio': ['-ar', '16000', '-ac', '1']}",
            );
        });
    }

    #[test]
    fn test_dl_options_custom() {
        Python::attach(|py| {
            let opts = DlOptions {
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
    fn test_postprocessors_list() {
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
