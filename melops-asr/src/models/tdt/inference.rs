//! ONNX inference for TDT encoder and decoder-joint.

use crate::error::{ModelError, Result};
use crate::models::tdt::core::TdtModel;
use crate::models::tdt::detokenizer::TokenDuration;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use ort::{inputs, value::Tensor, value::Value};

impl TdtModel {
    pub(super) fn encode(&mut self, audio_signal: Array2<f32>) -> Result<(Array3<f32>, i64)> {
        let audio_length =
            Value::from_array(Array1::from_elem((1,), audio_signal.shape()[0] as i64))?;

        let audio_signal = Value::from_array(audio_signal.reversed_axes().insert_axis(Axis(0)))?;

        let input_value = inputs!(
            "audio_signal" => audio_signal,
            "length" => audio_length,
        );

        let mut outputs = self.encoder.run(input_value)?;

        let encoder_outputs =
            outputs
                .remove("outputs")
                .ok_or_else(|| ModelError::MissingOutput {
                    name: "outputs".to_string(),
                })?;

        let encoded_lengths =
            outputs
                .remove("encoded_lengths")
                .ok_or_else(|| ModelError::MissingOutput {
                    name: "encoded_lengths".to_string(),
                })?;

        let encoder_outputs = encoder_outputs
            .try_extract_array()?
            .to_owned()
            .into_dimensionality::<Ix3>()?;

        let encoded_lengths = encoded_lengths
            .try_extract_array()?
            .to_owned()
            .into_dimensionality::<Ix1>()?;

        Ok((encoder_outputs, encoded_lengths[0]))
    }

    pub(super) fn greedy_decode(
        &mut self,
        encoder_output: Array3<f32>,
        encoded_length: usize,
    ) -> Result<Vec<TokenDuration>> {
        let blank_id = self.detokenizer.vocab_size();
        let max_symbols_per_step = 10;

        let state_h = Array3::<f32>::zeros((2, 1, 640));
        let state_c = Array3::<f32>::zeros((2, 1, 640));

        let mut states_1 = Tensor::from_array(state_h)?.into_dyn();
        let mut states_2 = Tensor::from_array(state_c)?.into_dyn();

        let mut tokens = Vec::new();
        let mut frame_index = 0;

        let target = Array2::from_elem((1, 1), blank_id as i32);
        let mut target = Tensor::from_array(target)?;

        let target_length = Array1::from_elem((1,), 1);
        let target_length = Tensor::from_array(target_length)?;

        while frame_index < encoded_length - 1 {
            let frame = encoder_output
                .slice_axis(Axis(2), (frame_index..frame_index + 1).into())
                .into_owned();
            let frame = Tensor::from_array(frame)?;

            // Label looping: emit multiple tokens per frame if decoder keeps predicting non-blank
            'inner: {
                for _ in 0..max_symbols_per_step {
                    let mut outputs = self.decoder_joint.run(inputs!(
                        "encoder_outputs" => &frame,
                        "targets" => &target,
                        "target_length" => &target_length,
                        "input_states_1" => &states_1,
                        "input_states_2" => &states_2
                    ))?;

                    let logits_view: ArrayViewD<f32> = outputs["outputs"].try_extract_array()?;

                    let logits_flat = logits_view.flatten();

                    // Decoder outputs: [vocab_0..vocab_n, blank, duration_0..duration_4]
                    let text_logits = logits_flat.slice_axis(Axis(0), (0..blank_id + 1).into());
                    let token_id = text_logits.argmax()?;

                    let duration_logits = logits_flat.slice_axis(Axis(0), (blank_id + 1..).into());
                    let duration_idx = duration_logits.argmax()?;

                    let skip = self.durations.get(duration_idx).copied().ok_or_else(|| {
                        ModelError::DurationIndexOutOfBounds {
                            index: duration_idx,
                            max: self.durations.len() - 1,
                        }
                    })?;

                    if token_id != blank_id {
                        // Update LSTM states for next token prediction
                        states_1 = outputs.remove("output_states_1").ok_or_else(|| {
                            ModelError::MissingOutput {
                                name: "output_states_1".to_string(),
                            }
                        })?;
                        states_2 = outputs.remove("output_states_2").ok_or_else(|| {
                            ModelError::MissingOutput {
                                name: "output_states_2".to_string(),
                            }
                        })?;

                        tokens.push(TokenDuration {
                            token_id,
                            frame_index,
                            duration: skip,
                        });

                        target[[0, 0]] = token_id as i32;
                    }

                    tracing::trace!(frame_index, skip);

                    frame_index = encoded_length.min(frame_index + skip);

                    // Duration > 0: advance to next frame
                    if skip != 0 {
                        break 'inner;
                    }
                }

                // Max symbols reached without duration prediction: force frame advance
                frame_index += 1;
            }
        }

        Ok(tokens)
    }
}
