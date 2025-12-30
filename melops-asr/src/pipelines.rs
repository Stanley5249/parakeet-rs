//! High-level ASR pipelines.

use crate::detokenizer::SentencePieceDetokenizer;
use crate::models::tdt::TdtModel;
use crate::preprocessor::ParakeetPreprocessor;
use crate::traits::AsrPipeline;
use eyre::{OptionExt, Result as EyreResult, WrapErr};
use hf_hub::CacheRepo;
use hf_hub::api::sync::ApiRepo;
use ort::session::builder::SessionBuilder;
use parakeet_rs::Vocabulary;
use std::path::{Path, PathBuf};

pub trait ModelRepo {
    fn resolve(&self, file_name: &str) -> EyreResult<PathBuf>;

    /// Try resolving multiple file names, return first successful match
    fn resolve_any<I, S>(&self, candidates: I) -> EyreResult<PathBuf>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        candidates
            .into_iter()
            .find_map(|name| self.resolve(name.as_ref()).ok())
            .ok_or_eyre("no model found from candidates")
    }
}

impl ModelRepo for &Path {
    fn resolve(&self, file_name: &str) -> EyreResult<PathBuf> {
        self.join(file_name)
            .canonicalize()
            .wrap_err(format!("failed to resolve model: {file_name}"))
    }
}

impl ModelRepo for &CacheRepo {
    fn resolve(&self, file_name: &str) -> EyreResult<PathBuf> {
        self.get(file_name)
            .ok_or_eyre(format!("model not found in cache: {file_name}"))
    }
}

impl ModelRepo for &ApiRepo {
    fn resolve(&self, file_name: &str) -> EyreResult<PathBuf> {
        self.get(file_name)
            .wrap_err(format!("failed to download from api: {file_name}"))
    }
}

/// Parakeet TDT ASR pipeline.
///
/// Wraps the TDT (Token-and-Duration Transducer) model for ASR with timestamps.
/// Uses the generic `AsrPipeline` for chunking and streaming.
pub type ParakeetTdt = AsrPipeline<ParakeetPreprocessor, TdtModel, SentencePieceDetokenizer>;

impl ParakeetTdt {
    /// Load TDT pipeline from a model repository.
    ///
    /// # Arguments
    ///
    /// * `repo` - Model repository (local path, HF cache, or HF API)
    /// * `session_builder` - ONNX session builder for configuring execution providers
    pub fn from_repo<R: ModelRepo>(repo: R, session_builder: SessionBuilder) -> EyreResult<Self> {
        // Resolve model paths with priority fallback
        let encoder_path = repo.resolve_any([
            "encoder-model.onnx",
            "encoder.onnx",
            "encoder-model.int8.onnx",
        ])?;

        let decoder_path = repo.resolve_any([
            "decoder_joint-model.onnx",
            "decoder_joint.onnx",
            "decoder_joint-model.int8.onnx",
        ])?;

        let vocab_path = repo.resolve("vocab.txt")?;

        // Create ONNX sessions
        let encoder_session = session_builder
            .clone()
            .commit_from_file(&encoder_path)
            .wrap_err("failed to load encoder session")?;

        let decoder_session = session_builder
            .commit_from_file(&decoder_path)
            .wrap_err("failed to load decoder session")?;

        // Load preprocessor and detokenizer
        let preprocessor = ParakeetPreprocessor::tdt();
        let vocabulary =
            Vocabulary::from_file(&vocab_path).wrap_err("failed to load vocabulary")?;

        let detokenizer = SentencePieceDetokenizer::for_tdt(
            vocabulary,
            preprocessor.config().hop_length,
            preprocessor.config().sampling_rate,
        );

        // Create model from sessions
        let model = TdtModel::new(encoder_session, decoder_session, detokenizer.vocab_size());

        Ok(AsrPipeline::new(preprocessor, model, detokenizer))
    }
}
