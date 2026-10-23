//! A [`Sequential`] model — an ordered stack of [`Layer`]s run end to end, with the tensor shape and
//! scratch buffers managed for you, plus a little-endian weight-blob loader.
//!
//! This is the layer above [`crate::nn`]: instead of sizing and threading buffers by hand, describe
//! the network as a list of layers and call [`Sequential::forward`]. It has **no ML-framework
//! dependency** — you train elsewhere, dump the weights as little-endian `f64`, load them with
//! [`weights_from_le_bytes`], and run inference here (embedded / offline). Tensors flow in **HWC**
//! layout; a [`Layer::Dense`] flattens the current `h·w·c` tensor and thereafter the shape is
//! `1 × 1 × out_dim`.

// `out_h`/`out_w` are the natural tensor-dimension pair; they only look alike.
#![allow(clippy::similar_names)]

use crate::nn::{Conv2d, conv_out_dim, conv2d, dense, max_pool2d, pool_out_dim, relu, softmax};

/// One layer of a [`Sequential`] model, owning its parameters.
#[derive(Debug, Clone)]
pub enum Layer {
    /// 2-D convolution. `weights` is `[out_channels][ky][kx][in_channels]`, `bias` per output channel.
    Conv {
        /// Convolution weights, `[out_channels][ky][kx][in_channels]` row-major.
        weights: Vec<f64>,
        /// Bias per output channel.
        bias: Vec<f64>,
        /// Number of output channels.
        out_channels: usize,
        /// Square kernel side.
        kernel: usize,
        /// Stride.
        stride: usize,
        /// Zero-padding on each side.
        pad: usize,
    },
    /// 2-D max pooling with a square window.
    MaxPool {
        /// Pooling window side.
        pool: usize,
        /// Stride.
        stride: usize,
    },
    /// Fully-connected layer over the flattened tensor. `weights` is `[out_dim][in_dim]`.
    Dense {
        /// Dense weights, `[out_dim][in_dim]` row-major.
        weights: Vec<f64>,
        /// Bias per output unit.
        bias: Vec<f64>,
        /// Number of output units.
        out_dim: usize,
    },
    /// In-place `ReLU`.
    Relu,
    /// In-place softmax over the whole current vector.
    Softmax,
}

/// A stack of [`Layer`]s with a fixed input shape, run in order by [`Self::forward`].
#[derive(Debug, Clone)]
pub struct Sequential {
    input_h: usize,
    input_w: usize,
    input_ch: usize,
    layers: Vec<Layer>,
}

impl Sequential {
    /// A model taking an `input_h × input_w × input_ch` HWC tensor through `layers`.
    #[must_use]
    pub fn new(input_h: usize, input_w: usize, input_ch: usize, layers: Vec<Layer>) -> Self {
        Self {
            input_h,
            input_w,
            input_ch,
            layers,
        }
    }

    /// Run `input` (an `input_h × input_w × input_ch` HWC tensor) through every layer and return the
    /// output vector. The shape is tracked across layers; a [`Layer::Dense`] flattens it to `1×1×n`.
    ///
    /// # Panics
    ///
    /// If `input` is smaller than `input_h · input_w · input_ch`, or a layer's parameters are sized
    /// inconsistently with the running shape (the same contracts as the [`crate::nn`] primitives).
    #[must_use]
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let (mut h, mut w, mut ch) = (self.input_h, self.input_w, self.input_ch);
        let mut buf = input[..h * w * ch].to_vec();

        for layer in &self.layers {
            match layer {
                Layer::Conv {
                    weights,
                    bias,
                    out_channels,
                    kernel,
                    stride,
                    pad,
                } => {
                    let conv = Conv2d {
                        weights,
                        bias,
                        out_channels: *out_channels,
                        kernel: *kernel,
                        stride: *stride,
                        pad: *pad,
                    };
                    let out_h = conv_out_dim(h, *kernel, *stride, *pad);
                    let out_w = conv_out_dim(w, *kernel, *stride, *pad);
                    let mut out = vec![0.0; out_h * out_w * out_channels];
                    let (rows, cols) = conv2d(&buf, h, w, ch, &conv, &mut out);
                    buf = out;
                    h = rows;
                    w = cols;
                    ch = *out_channels;
                }
                Layer::MaxPool { pool, stride } => {
                    let out_h = pool_out_dim(h, *pool, *stride);
                    let out_w = pool_out_dim(w, *pool, *stride);
                    let mut out = vec![0.0; out_h * out_w * ch];
                    max_pool2d(&buf, h, w, ch, *pool, *stride, &mut out);
                    buf = out;
                    h = out_h;
                    w = out_w;
                }
                Layer::Dense {
                    weights,
                    bias,
                    out_dim,
                } => {
                    let in_dim = h * w * ch;
                    let mut out = vec![0.0; *out_dim];
                    dense(&buf, weights, bias, in_dim, *out_dim, &mut out);
                    buf = out;
                    h = 1;
                    w = 1;
                    ch = *out_dim;
                }
                Layer::Relu => relu(&mut buf),
                Layer::Softmax => softmax(&mut buf),
            }
        }
        buf
    }
}

/// Decode a little-endian `f64` weight blob into a `Vec<f64>` — the counterpart to dumping a trained
/// layer's weights with `numpy.ndarray.astype('<f8').tobytes()` or `torch.Tensor.numpy().tobytes()`.
/// Trailing bytes that don't complete an `f64` are ignored.
#[must_use]
pub fn weights_from_le_bytes(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn forward_tracks_shape_through_conv_pool_dense_softmax() {
        // 4×4×1 → conv(2×2 all-ones)→3×3 → relu → maxpool(3,3)→1×1 → dense 1→3 → softmax.
        let input: Vec<f64> = (0..16).map(|i| f64::from(i) - 8.0).collect();
        let model = Sequential::new(
            4,
            4,
            1,
            vec![
                Layer::Conv {
                    weights: vec![1.0, 1.0, 1.0, 1.0],
                    bias: vec![0.0],
                    out_channels: 1,
                    kernel: 2,
                    stride: 1,
                    pad: 0,
                },
                Layer::Relu,
                Layer::MaxPool { pool: 3, stride: 3 },
                Layer::Dense {
                    weights: vec![1.0, -1.0, 0.5],
                    bias: vec![0.0, 0.0, 0.0],
                    out_dim: 3,
                },
                Layer::Softmax,
            ],
        );
        let out = model.forward(&input);
        assert_eq!(out.len(), 3);
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // The pooled response is positive, so the +1 unit dominates the −1 one.
        assert!(out[0] > out[1]);
    }

    #[test]
    fn sequential_matches_manual_primitive_chaining() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3×3×1
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity-ish
        let bias = vec![0.5];

        let model = Sequential::new(
            3,
            3,
            1,
            vec![
                Layer::Conv {
                    weights: weights.clone(),
                    bias: bias.clone(),
                    out_channels: 1,
                    kernel: 2,
                    stride: 1,
                    pad: 0,
                },
                Layer::Relu,
            ],
        );
        let got = model.forward(&input);

        // Manual: conv then relu.
        let conv = Conv2d {
            weights: &weights,
            bias: &bias,
            out_channels: 1,
            kernel: 2,
            stride: 1,
            pad: 0,
        };
        let mut manual = vec![0.0; 4];
        let _ = conv2d(&input, 3, 3, 1, &conv, &mut manual);
        relu(&mut manual);

        assert_eq!(got, manual);
    }

    #[test]
    fn weights_from_le_bytes_round_trips() {
        let values = [1.5_f64, -2.25, 3.125e10, 0.0];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.push(0x7f); // a trailing partial f64 to be ignored
        assert_eq!(weights_from_le_bytes(&bytes), values);
    }

    #[test]
    fn forward_takes_the_whole_multi_channel_input_tensor() {
        // Every other `Sequential` fixture is single-channel, and `/ ch` is the identity at
        // `ch == 1` — so the length of the slice `forward` copies out of `input` was never
        // observable. `Relu` preserves the element count, which makes the kill a length
        // assertion rather than a primitive's "input too small" panic.
        let input: Vec<f64> = (0..12).map(|i| f64::from(i) - 6.0).collect(); // 2×3×2
        let model = Sequential::new(2, 3, 2, vec![Layer::Relu]);
        let out = model.forward(&input);
        assert_eq!(out.len(), 12);
        let want: Vec<f64> = input.iter().map(|v| v.max(0.0)).collect();
        assert_eq!(out, want);
    }

    #[test]
    fn dense_flattens_the_whole_non_square_multi_channel_tensor() {
        // The only other `Layer::Dense` fixture reaches this line behind a 3×3 max-pool that
        // leaves `h = w = ch = 1`, where every rewrite of `h·w·ch` is the identity. The softmax
        // that follows it there also hides a wrong `in_dim`: it normalises whatever it is given,
        // so "the outputs sum to one" holds for the mutant too.
        let input: Vec<f64> = (1..=12).map(f64::from).collect(); // 2×3×2
        let model = Sequential::new(
            2,
            3,
            2,
            vec![Layer::Dense {
                weights: vec![1.0; 12],
                bias: vec![0.0],
                out_dim: 1,
            }],
        );
        assert_eq!(model.forward(&input), vec![78.0]); // 1 + 2 + … + 12
    }
}
