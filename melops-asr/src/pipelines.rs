//! High-level ASR pipelines.

use crate::audio::MelSpectrogram;
use crate::detokenizer::SentencePieceDetokenizer;
use crate::models::tdt::TdtModel;
use crate::traits::AsrPipeline;
use eyre::{ContextCompat, OptionExt, Result, WrapErr, eyre};
use hf_hub::CacheRepo;
use hf_hub::api::sync::ApiRepo;
use ort::session::builder::SessionBuilder;
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Model repository sources.
///
/// Enum containing different types of model repositories.
/// Replaces the previous ModelRepo trait for simpler usage.
#[derive(Debug)]
pub enum ModelRepo {
    /// Local filesystem path
    Path(PathBuf),
    /// HuggingFace cache repository
    Cache(CacheRepo),
    /// HuggingFace API repository
    Api(ApiRepo),
}

impl ModelRepo {
    /// Resolve a file name to its full path in this repository.
    pub fn resolve(&self, file_name: &str) -> Result<PathBuf> {
        match self {
            ModelRepo::Path(path) => path
                .join(file_name)
                .canonicalize()
                .wrap_err(format!("failed to resolve model: {file_name}")),
            ModelRepo::Cache(cache_repo) => cache_repo
                .get(file_name)
                .wrap_err(format!("failed to download from cache: {file_name}")),
            ModelRepo::Api(api_repo) => api_repo
                .get(file_name)
                .wrap_err(format!("failed to download from api: {file_name}")),
        }
    }

    /// Try resolving multiple file names, return first successful match.
    pub fn resolve_any(&self, candidates: &[&str]) -> Result<PathBuf> {
        candidates
            .iter()
            .find_map(|name| self.resolve(name).ok())
            .ok_or_eyre("no model found from candidates")
    }
}

/// Parakeet TDT ASR pipeline.
///
/// Wraps the TDT (Token-and-Duration Transducer) model for ASR with timestamps.
/// Uses the generic `AsrPipeline` for chunking and streaming.
pub type ParakeetTdt = AsrPipeline<TdtModel, SentencePieceDetokenizer>;

impl ParakeetTdt {
    /// Load TDT pipeline from a model repository.
    ///
    /// # Arguments
    ///
    /// * `repo` - Model repository (local path, HF cache, or HF API)
    /// * `session_builder` - ONNX session builder for configuring execution providers
    pub fn from_repo(repo: &ModelRepo, session_builder: SessionBuilder) -> Result<Self> {
        // Resolve model paths with priority fallback
        let encoder_path = repo.resolve_any(&[
            "encoder-model.onnx",
            "encoder.onnx",
            "encoder-model.int8.onnx",
        ])?;

        let decoder_path = repo.resolve_any(&[
            "decoder_joint-model.onnx",
            "decoder_joint.onnx",
            "decoder_joint-model.int8.onnx",
        ])?;

        let tokenizer_path = repo.resolve("tokenizer.json")?;

        // Create ONNX sessions
        let encoder_session = session_builder
            .clone()
            .commit_from_file(&encoder_path)
            .wrap_err("failed to load encoder session")?;

        let decoder_session = session_builder
            .commit_from_file(&decoder_path)
            .wrap_err("failed to load decoder session")?;

        // Boxed dyn error does not implement the Error trait
        // https://deepwiki.com/search/how-to-convert-box-dyn-error-i_e1d11897-6026-4438-9785-1997b61beccf
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| eyre!(e))
            .wrap_err(format!(
                "failed to load tokenizer from {:?}",
                tokenizer_path
            ))?;

        let detokenizer = SentencePieceDetokenizer::for_tdt(
            tokenizer,
            MelSpectrogram::TDT.hop_length,
            MelSpectrogram::TDT.sample_rate,
        );

        // Create model from sessions
        let model = TdtModel::new(encoder_session, decoder_session, detokenizer.vocab_size());

        Ok(AsrPipeline::new(MelSpectrogram::TDT, model, detokenizer))
    }
}
