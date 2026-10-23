//! Load and run trained **ONNX** models through [`tract`](https://github.com/sonos/tract) — the
//! opt-in `onnx` feature, the bridge from *hand-authored* [`crate::nn`] graphs to networks trained in
//! PyTorch/TensorFlow and exported to ONNX. `tract` is pure Rust (no C/C++ runtime), but needs `std`,
//! so the whole path is feature-gated and never enters the default `no_std` build.
//!
//! The canonical use is an Earth-observation segmentation CNN — e.g. an onboard cloud mask: run a
//! U-Net over a multispectral tile, [`threshold_mask`] the per-pixel score, and feed the usable
//! pixels to the swath/footprint geometry. [`OnnxModel`] deliberately exposes only plain `f32`
//! slices, never a `tract` type, so the runtime stays swappable.

use tract_onnx::prelude::*;

/// A loaded, optimized ONNX model ready for repeated inference.
pub struct OnnxModel {
    plan: std::sync::Arc<TypedRunnableModel>,
}

/// An ONNX load-or-run failure. Wraps the runtime's message as a `String` so the public surface never
/// leaks `tract` types (the model runtime behind this module can change without a breaking change).
#[derive(Debug, Clone)]
pub struct OnnxError(String);

impl core::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "onnx: {}", self.0)
    }
}

impl std::error::Error for OnnxError {}

impl From<TractError> for OnnxError {
    fn from(e: TractError) -> Self {
        Self(e.to_string())
    }
}

impl OnnxModel {
    /// Parse, type-infer and optimize an ONNX model from its serialized bytes (a `.onnx` protobuf).
    ///
    /// # Errors
    /// Returns [`OnnxError`] if the bytes are not a valid ONNX graph, use an op `tract` cannot lower,
    /// or carry a shape/type that fails inference.
    pub fn load(onnx_bytes: &[u8]) -> Result<Self, OnnxError> {
        let mut reader = onnx_bytes;
        let plan = tract_onnx::onnx()
            .model_for_read(&mut reader)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { plan })
    }

    /// Run the model on a single `channels x height x width` image (implicit batch of 1), returning the
    /// flattened output tensor in row-major order.
    ///
    /// # Errors
    /// Returns [`OnnxError`] if `input.len() != channels * height * width`, or if the model rejects the
    /// supplied shape or does not produce an `f32` output.
    pub fn run_chw(
        &self,
        input: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, OnnxError> {
        let expected = channels
            .checked_mul(height)
            .and_then(|hw| hw.checked_mul(width));
        if expected != Some(input.len()) {
            return Err(OnnxError(format!(
                "input length {} does not match channels*height*width = {channels}*{height}*{width}",
                input.len()
            )));
        }
        let tensor = Tensor::from_shape(&[1, channels, height, width], input)?;
        let outputs = self.plan.run(tvec!(tensor.into()))?;
        let view = outputs[0].to_plain_array_view::<f32>()?;
        Ok(view.iter().copied().collect())
    }
}

/// Threshold per-pixel probabilities (e.g. a sigmoid cloud score) into a boolean mask: `p >= threshold`.
#[must_use]
pub fn threshold_mask(probabilities: &[f32], threshold: f32) -> Vec<bool> {
    probabilities.iter().map(|&p| p >= threshold).collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn read_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    #[test]
    fn onnx_cloud_cnn_matches_onnxruntime() {
        // Fixture: a Conv->ReLU->MaxPool->Conv->Sigmoid CNN (tests/fixtures/gen_cloud_fixture.py); the
        // reference output is onnxruntime's, so this asserts tract reproduces onnxruntime to <1e-4.
        let model = OnnxModel::load(include_bytes!("../tests/fixtures/cloud_tiny.onnx")).unwrap();
        let input = read_f32(include_bytes!("../tests/fixtures/cloud_tiny_input.f32"));
        let expected = read_f32(include_bytes!("../tests/fixtures/cloud_tiny_output.f32"));

        let got = model.run_chw(&input, 3, 8, 8).unwrap();

        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-4, "tract {g} vs onnxruntime {e}");
        }
    }

    #[test]
    fn run_chw_rejects_a_mismatched_input_length() {
        let model = OnnxModel::load(include_bytes!("../tests/fixtures/cloud_tiny.onnx")).unwrap();
        let e = model.run_chw(&[0.0; 10], 3, 8, 8).unwrap_err();
        // `tract`'s own error types never leak, so this rendering is the whole public surface of
        // the failure — nothing else tells a caller which of the four numbers disagreed.
        assert_eq!(
            e.to_string(),
            "onnx: input length 10 does not match channels*height*width = 3*8*8"
        );
    }

    #[test]
    fn threshold_mask_splits_on_the_cut() {
        assert_eq!(
            threshold_mask(&[0.1, 0.5, 0.9], 0.5),
            vec![false, true, true]
        );
    }
}
