use crate::audio;
use crate::config::PreprocessorConfig;
use crate::decoder::TranscriptionResult;
use crate::decoder_tdt::ParakeetTDTDecoder;
use crate::error::{Error, Result};
use crate::model_tdt::ParakeetTDTModel;
use crate::timestamps::{process_timestamps, TimestampMode};
use crate::transcriber::Transcriber;
use crate::vocab::Vocabulary;
use ort::session::builder::SessionBuilder;
use std::path::{Path, PathBuf};

/// Parakeet TDT model for multilingual ASR
pub struct ParakeetTDT {
    model: ParakeetTDTModel,
    decoder: ParakeetTDTDecoder,
    preprocessor_config: PreprocessorConfig,
    model_dir: PathBuf,
}

impl ParakeetTDT {
    /// Load Parakeet TDT model from path with optional ORT session builder.
    ///
    /// # Arguments
    /// * `path` - Directory containing encoder-model.onnx, decoder_joint-model.onnx, and vocab.txt
    /// * `builder` - Optional ORT SessionBuilder (defaults to CPU if None)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use parakeet_rs::ParakeetTDT;
    /// use ort::session::Session;
    /// use ort::session::builder::GraphOptimizationLevel;
    /// use ort::execution_providers::{OpenVINOExecutionProvider, CPUExecutionProvider};
    /// use ort::execution_providers::openvino::OpenVINODeviceType;
    ///
    /// // Configure OpenVINO to use GPU with full control
    /// let builder = Session::builder()?
    ///     .with_optimization_level(GraphOptimizationLevel::Level3)?
    ///     .with_intra_threads(4)?
    ///     .with_execution_providers([
    ///         OpenVINOExecutionProvider::default()
    ///             .with_device_type(OpenVINODeviceType::GPU)
    ///             .build(),
    ///         CPUExecutionProvider::default().build(),
    ///     ])?;
    ///
    /// let mut model = ParakeetTDT::from_pretrained("./model", Some(builder))?;
    /// ```
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        builder: Option<SessionBuilder>,
    ) -> Result<Self> {
        let path = path.as_ref();

        if !path.is_dir() {
            return Err(Error::Config(format!(
                "TDT model path must be a directory: {}",
                path.display()
            )));
        }

        let vocab_path = path.join("vocab.txt");
        if !vocab_path.exists() {
            return Err(Error::Config(format!(
                "vocab.txt not found in {}",
                path.display()
            )));
        }

        // TDT-specific preprocessor config (128 features instead of 80)
        let preprocessor_config = PreprocessorConfig {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: 128,
            hop_length: 160,
            n_fft: 512,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: 0.97,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: 16000,
            win_length: 400,
        };

        // Load vocab first to get the actual vocabulary size
        let vocab = Vocabulary::from_file(&vocab_path)?;
        let vocab_size = vocab.size();

        let model = ParakeetTDTModel::from_pretrained(path, builder, vocab_size)?;
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);

        Ok(Self {
            model,
            decoder,
            preprocessor_config,
            model_dir: path.to_path_buf(),
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn preprocessor_config(&self) -> &PreprocessorConfig {
        &self.preprocessor_config
    }
}

impl Transcriber for ParakeetTDT {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        let features =
            audio::extract_features_raw(audio, sample_rate, channels, &self.preprocessor_config)?;
        let (tokens, frame_indices, durations) = self.model.forward(features)?;

        let mut result = self.decoder.decode_with_timestamps(
            &tokens,
            &frame_indices,
            &durations,
            self.preprocessor_config.hop_length,
            self.preprocessor_config.sampling_rate,
        )?;

        // Apply timestamp mode conversion
        let mode = mode.unwrap_or(TimestampMode::Tokens);
        result.tokens = process_timestamps(&result.tokens, mode);

        // Rebuild full text from processed tokens
        result.text = if mode == TimestampMode::Tokens {
            result
                .tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<String>()
                .trim()
                .to_string()
        } else {
            result
                .tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };

        Ok(result)
    }
}
