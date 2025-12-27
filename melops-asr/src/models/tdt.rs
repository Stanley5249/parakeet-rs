//! TDT (Token-and-Duration Transducer) model implementation.

use crate::detokenizer::TdtOutput;
use crate::error::{Error, Result};
use crate::traits::AsrModel;
use ndarray::{Array1, Array2, Array3, s};
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use std::path::{Path, PathBuf};

/// TDT model for ASR inference.
///
/// Implements the Token-and-Duration Transducer architecture
/// with encoder and joint decoder components.
pub struct TdtModel {
    encoder: Session,
    decoder_joint: Session,
    vocab_size: usize,
}

impl TdtModel {
    /// Load TDT model from pretrained directory.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to directory containing ONNX files
    /// * `builder` - Session builder for execution provider config
    /// * `vocab_size` - Vocabulary size from detokenizer
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        builder: SessionBuilder,
        vocab_size: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let encoder_path = find_encoder(model_dir)?;
        let decoder_joint_path = find_decoder_joint(model_dir)?;

        let encoder = builder.clone().commit_from_file(&encoder_path)?;
        let decoder_joint = builder.commit_from_file(&decoder_joint_path)?;

        Ok(Self {
            encoder,
            decoder_joint,
            vocab_size,
        })
    }

    fn run_encoder(&mut self, features: &Array2<f32>) -> Result<(Array3<f32>, i64)> {
        let batch_size = 1;
        let time_steps = features.shape()[0];
        let feature_size = features.shape()[1];

        // TDT encoder expects (batch, features, time) not (batch, time, features)
        let input = features
            .t()
            .to_shape((batch_size, feature_size, time_steps))
            .map_err(|e| Error::Model(format!("Failed to reshape encoder input: {e}")))?
            .to_owned();

        let input_length = Array1::from_vec(vec![time_steps as i64]);

        let input_value = ort::value::Value::from_array(input)?;
        let length_value = ort::value::Value::from_array(input_length)?;

        let outputs = self.encoder.run(ort::inputs!(
            "audio_signal" => input_value,
            "length" => length_value
        ))?;

        let encoder_out = &outputs["outputs"];
        let encoder_lens = &outputs["encoded_lengths"];

        let (shape, data) = encoder_out
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

        let (_, lens_data) = encoder_lens
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder lengths: {e}")))?;

        let shape_dims = shape.as_ref();
        if shape_dims.len() != 3 {
            return Err(Error::Model(format!(
                "Expected 3D encoder output, got shape: {shape_dims:?}"
            )));
        }

        let b = shape_dims[0] as usize;
        let t = shape_dims[1] as usize;
        let d = shape_dims[2] as usize;

        let encoder_array = Array3::from_shape_vec((b, t, d), data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to create encoder array: {e}")))?;

        Ok((encoder_array, lens_data[0]))
    }

    fn greedy_decode(
        &mut self,
        encoder_out: &Array3<f32>,
        _encoder_len: i64,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let encoder_dim = encoder_out.shape()[1];
        let time_steps = encoder_out.shape()[2];
        let vocab_size = self.vocab_size;
        let max_tokens_per_step = 10;
        let blank_id = vocab_size - 1;

        // States: (num_layers=2, batch=1, hidden_dim=640)
        let mut state_h = Array3::<f32>::zeros((2, 1, 640));
        let mut state_c = Array3::<f32>::zeros((2, 1, 640));

        let mut tokens = Vec::new();
        let mut frame_indices = Vec::new();
        let mut durations = Vec::new();

        let mut t = 0;
        let mut emitted_tokens = 0;
        let mut last_emitted_token = blank_id as i32;

        while t < time_steps {
            let frame = encoder_out.slice(s![0, .., t]).to_owned();
            let frame_reshaped = frame
                .to_shape((1, encoder_dim, 1))
                .map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?
                .to_owned();

            let targets = Array2::from_shape_vec((1, 1), vec![last_emitted_token])
                .map_err(|e| Error::Model(format!("Failed to create targets: {e}")))?;

            let outputs = self.decoder_joint.run(ort::inputs!(
                "encoder_outputs" => ort::value::Value::from_array(frame_reshaped)?,
                "targets" => ort::value::Value::from_array(targets)?,
                "target_length" => ort::value::Value::from_array(Array1::from_vec(vec![1i32]))?,
                "input_states_1" => ort::value::Value::from_array(state_h.clone())?,
                "input_states_2" => ort::value::Value::from_array(state_c.clone())?
            ))?;

            let (_, logits_data) = outputs["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

            let vocab_logits: Vec<f32> = logits_data.iter().take(vocab_size).copied().collect();
            let duration_logits: Vec<f32> = logits_data.iter().skip(vocab_size).copied().collect();

            let token_id = vocab_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(blank_id);

            let duration_step = if !duration_logits.is_empty() {
                duration_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            } else {
                0
            };

            if token_id != blank_id {
                if let Ok((h_shape, h_data)) =
                    outputs["output_states_1"].try_extract_tensor::<f32>()
                {
                    let dims = h_shape.as_ref();
                    state_h = Array3::from_shape_vec(
                        (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                        h_data.to_vec(),
                    )
                    .map_err(|e| Error::Model(format!("Failed to update state_h: {e}")))?;
                }
                if let Ok((c_shape, c_data)) =
                    outputs["output_states_2"].try_extract_tensor::<f32>()
                {
                    let dims = c_shape.as_ref();
                    state_c = Array3::from_shape_vec(
                        (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                        c_data.to_vec(),
                    )
                    .map_err(|e| Error::Model(format!("Failed to update state_c: {e}")))?;
                }

                tokens.push(token_id);
                frame_indices.push(t);
                durations.push(duration_step);
                last_emitted_token = token_id as i32;
                emitted_tokens += 1;
            } else {
                if duration_step > 0 && emitted_tokens > 0 {
                    t += duration_step;
                } else {
                    t += 1;
                }
                emitted_tokens = 0;
            }

            if emitted_tokens >= max_tokens_per_step {
                t += 1;
                emitted_tokens = 0;
            }
        }

        Ok((tokens, frame_indices, durations))
    }
}

impl AsrModel for TdtModel {
    type Features = Array2<f32>;
    type Output = TdtOutput;

    fn forward(&mut self, features: Self::Features) -> Result<Self::Output> {
        let (encoder_out, encoder_len) = self.run_encoder(&features)?;
        let (tokens, frame_indices, durations) = self.greedy_decode(&encoder_out, encoder_len)?;
        Ok(TdtOutput {
            tokens,
            frame_indices,
            durations,
        })
    }
}

fn find_encoder(dir: &Path) -> Result<PathBuf> {
    let candidates = [
        "encoder-model.onnx",
        "encoder.onnx",
        "encoder-model.int8.onnx",
    ];
    for candidate in &candidates {
        let path = dir.join(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|s| s.to_str())
                && name.starts_with("encoder")
                && name.ends_with(".onnx")
            {
                return Ok(path);
            }
        }
    }
    Err(Error::Model(format!(
        "No encoder model found in {}",
        dir.display()
    )))
}

fn find_decoder_joint(dir: &Path) -> Result<PathBuf> {
    let candidates = [
        "decoder_joint-model.onnx",
        "decoder_joint-model.int8.onnx",
        "decoder_joint.onnx",
        "decoder-model.onnx",
    ];
    for candidate in &candidates {
        let path = dir.join(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(Error::Model(format!(
        "No decoder_joint model found in {}",
        dir.display()
    )))
}
