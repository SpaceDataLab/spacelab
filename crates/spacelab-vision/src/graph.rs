//! A [`Graph`] model — a directed acyclic graph of [`Op`] nodes, the generalisation of the linear
//! [`crate::model::Sequential`] to networks with **skip connections**: residual adds (ResNet) and
//! channel concatenations (U-Net).
//!
//! Each node names its input nodes by index and applies one [`crate::nn`] op to them; the runner
//! evaluates the nodes in order (each node's inputs must precede it — a topological order) and caches
//! every node's output tensor so a later node can reach back to it. That reach-back is the skip
//! connection: an [`Op::Add`] takes two tensors and sums them (the ResNet residual), an [`Op::Concat`]
//! stacks them along the channel axis (the U-Net decoder merge). Tensors flow in **HWC** layout, as in
//! [`crate::nn`]; the graph's output is its last node.

// out_h/out_w are the natural tensor-dimension pair; the eval() match is one big dispatch; the doc
// names architectures (ResNet, U-Net); single letters (h, w, c) are tensor-shape notation.
#![allow(
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use crate::nn::{
    BatchNormParams, Conv2d, ConvTranspose2d, add, batch_norm, concat_channels, conv_out_dim,
    conv_transpose2d, conv2d, dense, global_avg_pool, max_pool2d, pad2d, pool_out_dim, relu,
    sigmoid, softmax,
};

/// One operation in a [`Graph`], owning its parameters. Unary ops read `inputs[0]`; [`Op::Add`] and
/// [`Op::Concat`] read `inputs[0]` and `inputs[1]`.
#[derive(Debug, Clone)]
pub enum Op {
    /// The graph input tensor (no inputs).
    Input,
    /// 2-D convolution. `weights` is `[out_channels][ky][kx][in_channels]`.
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
    /// 2-D transposed convolution (learned upsampling). `weights` is `[in_channels][ky][kx][out_channels]`.
    ConvTranspose {
        /// Transposed-convolution weights, `[in_channels][ky][kx][out_channels]` row-major.
        weights: Vec<f64>,
        /// Bias per output channel.
        bias: Vec<f64>,
        /// Number of output channels.
        out_channels: usize,
        /// Square kernel side.
        kernel: usize,
        /// Stride (upsampling factor).
        stride: usize,
        /// Zero-padding removed from each output side.
        pad: usize,
    },
    /// 2-D max pooling.
    MaxPool {
        /// Pooling window side.
        pool: usize,
        /// Stride.
        stride: usize,
    },
    /// Inference batch normalization (per channel).
    BatchNorm {
        /// Scale per channel.
        gamma: Vec<f64>,
        /// Shift per channel.
        beta: Vec<f64>,
        /// Running mean per channel.
        mean: Vec<f64>,
        /// Running variance per channel.
        var: Vec<f64>,
        /// Numerical epsilon.
        eps: f64,
    },
    /// In-place `ReLU`.
    Relu,
    /// In-place logistic sigmoid.
    Sigmoid,
    /// Elementwise residual add of two same-shaped tensors — the ResNet skip.
    Add,
    /// Channel concatenation of two same-spatial tensors — the U-Net skip.
    Concat,
    /// Global average pooling to `1×1×ch`.
    GlobalAvgPool,
    /// Fully-connected layer over the flattened tensor. `weights` is `[out_dim][in_dim]`.
    Dense {
        /// Dense weights, `[out_dim][in_dim]` row-major.
        weights: Vec<f64>,
        /// Bias per output unit.
        bias: Vec<f64>,
        /// Number of output units.
        out_dim: usize,
    },
    /// Softmax over the whole current vector.
    Softmax,
    /// Zero-pad by `pad` on every side.
    Pad {
        /// Padding on each side.
        pad: usize,
    },
}

/// A graph node: an [`Op`] and the indices of the nodes feeding it.
#[derive(Debug, Clone)]
pub struct Node {
    /// The operation.
    pub op: Op,
    /// Indices of the input nodes (must be `< this node's index`).
    pub inputs: Vec<usize>,
}

impl Node {
    /// A node with no inputs (an [`Op::Input`]).
    #[must_use]
    pub fn source(op: Op) -> Self {
        Self {
            op,
            inputs: Vec::new(),
        }
    }

    /// A unary node reading a single input.
    #[must_use]
    pub fn unary(op: Op, input: usize) -> Self {
        Self {
            op,
            inputs: alloc_vec(input),
        }
    }

    /// A binary node (for [`Op::Add`] / [`Op::Concat`]) reading two inputs.
    #[must_use]
    pub fn binary(op: Op, a: usize, b: usize) -> Self {
        Self {
            op,
            inputs: {
                let mut v = alloc_vec(a);
                v.push(b);
                v
            },
        }
    }
}

fn alloc_vec(x: usize) -> Vec<usize> {
    let mut v = Vec::with_capacity(2);
    v.push(x);
    v
}

/// A DAG of [`Node`]s with a fixed input shape, evaluated in node order by [`Self::forward`].
#[derive(Debug, Clone)]
pub struct Graph {
    input_h: usize,
    input_w: usize,
    input_ch: usize,
    nodes: Vec<Node>,
}

/// A node's cached output: the tensor and its `(h, w, ch)` shape.
type Tensor = (Vec<f64>, usize, usize, usize);

impl Graph {
    /// A graph taking an `input_h × input_w × input_ch` HWC tensor through `nodes` (node 0 must be an
    /// [`Op::Input`]; every node's inputs must have smaller indices).
    #[must_use]
    pub fn new(input_h: usize, input_w: usize, input_ch: usize, nodes: Vec<Node>) -> Self {
        Self {
            input_h,
            input_w,
            input_ch,
            nodes,
        }
    }

    /// Evaluate the graph on `input` and return the last node's output tensor.
    ///
    /// # Panics
    ///
    /// If a node references an input index `≥` its own, or a layer's parameters are sized
    /// inconsistently with the running shape (the same contracts as the [`crate::nn`] primitives).
    #[must_use]
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut cache: Vec<Tensor> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let t = self.eval(node, input, &cache);
            cache.push(t);
        }
        cache.pop().map(|t| t.0).unwrap_or_default()
    }

    fn eval(&self, node: &Node, input: &[f64], cache: &[Tensor]) -> Tensor {
        let in0 = || &cache[node.inputs[0]];
        match &node.op {
            Op::Input => {
                let n = self.input_h * self.input_w * self.input_ch;
                (
                    input[..n].to_vec(),
                    self.input_h,
                    self.input_w,
                    self.input_ch,
                )
            }
            Op::Conv {
                weights,
                bias,
                out_channels,
                kernel,
                stride,
                pad,
            } => {
                let (buf, h, w, ch) = in0();
                let conv = Conv2d {
                    weights,
                    bias,
                    out_channels: *out_channels,
                    kernel: *kernel,
                    stride: *stride,
                    pad: *pad,
                };
                let out_h = conv_out_dim(*h, *kernel, *stride, *pad);
                let out_w = conv_out_dim(*w, *kernel, *stride, *pad);
                let mut out = vec![0.0; out_h * out_w * out_channels];
                let (rh, rw) = conv2d(buf, *h, *w, *ch, &conv, &mut out);
                (out, rh, rw, *out_channels)
            }
            Op::ConvTranspose {
                weights,
                bias,
                out_channels,
                kernel,
                stride,
                pad,
            } => {
                let (buf, h, w, ch) = in0();
                let ct = ConvTranspose2d {
                    weights,
                    bias,
                    out_channels: *out_channels,
                    kernel: *kernel,
                    stride: *stride,
                    pad: *pad,
                };
                let out_h = (*h - 1) * stride + kernel - 2 * pad;
                let out_w = (*w - 1) * stride + kernel - 2 * pad;
                let mut out = vec![0.0; out_h * out_w * out_channels];
                let (rh, rw) = conv_transpose2d(buf, *h, *w, *ch, &ct, &mut out);
                (out, rh, rw, *out_channels)
            }
            Op::MaxPool { pool, stride } => {
                let (buf, h, w, ch) = in0();
                let out_h = pool_out_dim(*h, *pool, *stride);
                let out_w = pool_out_dim(*w, *pool, *stride);
                let mut out = vec![0.0; out_h * out_w * ch];
                max_pool2d(buf, *h, *w, *ch, *pool, *stride, &mut out);
                (out, out_h, out_w, *ch)
            }
            Op::BatchNorm {
                gamma,
                beta,
                mean,
                var,
                eps,
            } => {
                let (buf, h, w, ch) = in0();
                let mut out = buf.clone();
                let bn = BatchNormParams {
                    gamma,
                    beta,
                    mean,
                    var,
                    eps: *eps,
                };
                batch_norm(&mut out, h * w, *ch, &bn);
                (out, *h, *w, *ch)
            }
            Op::Relu => {
                let (buf, h, w, ch) = in0();
                let mut out = buf.clone();
                relu(&mut out);
                (out, *h, *w, *ch)
            }
            Op::Sigmoid => {
                let (buf, h, w, ch) = in0();
                let mut out = buf.clone();
                sigmoid(&mut out);
                (out, *h, *w, *ch)
            }
            Op::Add => {
                let (a, h, w, ch) = in0();
                let b = &cache[node.inputs[1]];
                let mut out = a.clone();
                add(&mut out, &b.0);
                (out, *h, *w, *ch)
            }
            Op::Concat => {
                let (a, h, w, ca) = in0();
                let (b, _, _, cb) = &cache[node.inputs[1]];
                let mut out = vec![0.0; h * w * (ca + cb)];
                concat_channels(a, *ca, b, *cb, h * w, &mut out);
                (out, *h, *w, ca + cb)
            }
            Op::GlobalAvgPool => {
                let (buf, h, w, ch) = in0();
                let mut out = vec![0.0; *ch];
                global_avg_pool(buf, h * w, *ch, &mut out);
                (out, 1, 1, *ch)
            }
            Op::Dense {
                weights,
                bias,
                out_dim,
            } => {
                let (buf, h, w, ch) = in0();
                let in_dim = h * w * ch;
                let mut out = vec![0.0; *out_dim];
                dense(buf, weights, bias, in_dim, *out_dim, &mut out);
                (out, 1, 1, *out_dim)
            }
            Op::Softmax => {
                let (buf, h, w, ch) = in0();
                let mut out = buf.clone();
                softmax(&mut out);
                (out, *h, *w, *ch)
            }
            Op::Pad { pad } => {
                let (buf, h, w, ch) = in0();
                let (oh, ow) = (h + 2 * pad, w + 2 * pad);
                let mut out = vec![0.0; oh * ow * ch];
                pad2d(buf, *h, *w, *ch, *pad, &mut out);
                (out, oh, ow, *ch)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::unreadable_literal,
        clippy::identity_op
    )]
    use super::*;
    use crate::nn::{Conv2d, conv2d};

    #[test]
    fn residual_block_graph_matches_manual_chaining() {
        // Input → Conv → BN → Relu → Conv → BN → Add(input) → Relu, all shape-preserving. The Add reaches
        // back to the input node — the residual skip — and the whole graph must equal the same op chain
        // done by hand.
        let (h, w, c) = (4usize, 4usize, 2usize);
        let input: Vec<f64> = (0..h * w * c)
            .map(|i| f64::from(i as u32) * 0.1 - 1.5)
            .collect();
        let mut seed = 3u64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f64) / ((1u64 << 24) as f64) - 0.5
        };
        let cw: Vec<f64> = (0..c * 3 * 3 * c).map(|_| rng()).collect();
        let cb: Vec<f64> = (0..c).map(|_| rng()).collect();
        let bn = |eps| Op::BatchNorm {
            gamma: vec![1.1, 0.9],
            beta: vec![0.05, -0.1],
            mean: vec![0.2, -0.3],
            var: vec![0.8, 1.2],
            eps,
        };
        let conv_op = || Op::Conv {
            weights: cw.clone(),
            bias: cb.clone(),
            out_channels: c,
            kernel: 3,
            stride: 1,
            pad: 1,
        };
        let g = Graph::new(
            h,
            w,
            c,
            vec![
                Node::source(Op::Input),     // 0
                Node::unary(conv_op(), 0),   // 1
                Node::unary(bn(1e-5), 1),    // 2
                Node::unary(Op::Relu, 2),    // 3
                Node::unary(conv_op(), 3),   // 4
                Node::unary(bn(1e-5), 4),    // 5
                Node::binary(Op::Add, 5, 0), // 6: residual skip to the input
                Node::unary(Op::Relu, 6),    // 7
            ],
        );
        let got = g.forward(&input);

        // Manual chain of the same ops.
        let conv = Conv2d {
            weights: &cw,
            bias: &cb,
            out_channels: c,
            kernel: 3,
            stride: 1,
            pad: 1,
        };
        let bnp = BatchNormParams {
            gamma: &[1.1, 0.9],
            beta: &[0.05, -0.1],
            mean: &[0.2, -0.3],
            var: &[0.8, 1.2],
            eps: 1e-5,
        };
        let mut a = vec![0.0; h * w * c];
        let _ = conv2d(&input, h, w, c, &conv, &mut a);
        batch_norm(&mut a, h * w, c, &bnp);
        relu(&mut a);
        let mut b = vec![0.0; h * w * c];
        let _ = conv2d(&a, h, w, c, &conv, &mut b);
        batch_norm(&mut b, h * w, c, &bnp);
        add(&mut b, &input);
        relu(&mut b);
        assert_eq!(got, b);
    }

    #[test]
    fn concat_skip_graph_matches_manual_merge() {
        // A U-Net-style merge: Input → Conv(branch), and Input(identity) → Concat(branch, input).
        let (h, w, c) = (3usize, 3usize, 2usize);
        let input: Vec<f64> = (0..h * w * c).map(|i| f64::from(i as u32)).collect();
        let cw = vec![0.25; 3 * 3 * 3 * c]; // 3→3 channels
        let cb = vec![0.0; 3];
        let g = Graph::new(
            h,
            w,
            c,
            vec![
                Node::source(Op::Input), // 0
                Node::unary(
                    Op::Conv {
                        weights: cw.clone(),
                        bias: cb.clone(),
                        out_channels: 3,
                        kernel: 3,
                        stride: 1,
                        pad: 1,
                    },
                    0,
                ), // 1: 3 channels
                Node::binary(Op::Concat, 1, 0), // 2: concat branch (3) then the input (2) → 5 channels
            ],
        );
        let got = g.forward(&input);

        let conv = Conv2d {
            weights: &cw,
            bias: &cb,
            out_channels: 3,
            kernel: 3,
            stride: 1,
            pad: 1,
        };
        let mut branch = vec![0.0; h * w * 3];
        let _ = conv2d(&input, h, w, c, &conv, &mut branch);
        let mut merged = vec![0.0; h * w * 5];
        concat_channels(&branch, 3, &input, 2, h * w, &mut merged);
        assert_eq!(got, merged);
        assert_eq!(got.len(), h * w * 5);
    }

    use crate::model::weights_from_le_bytes;

    // A sequential cursor over a flat weight blob, matching the order the torch fixture dumps in.
    struct Cursor {
        data: Vec<f64>,
        pos: usize,
    }
    impl Cursor {
        fn new(bytes: &[u8]) -> Self {
            Self {
                data: weights_from_le_bytes(bytes),
                pos: 0,
            }
        }
        fn take(&mut self, n: usize) -> Vec<f64> {
            let s = self.data[self.pos..self.pos + n].to_vec();
            self.pos += n;
            s
        }
    }

    #[test]
    fn resnet_matches_pytorch_end_to_end() {
        // A real residual classifier (stem → residual block → global-average-pool → dense → softmax)
        // built in PyTorch (double precision), its weights and I/O dumped to resnet.bin. Our Graph, with
        // the same weights, must reproduce the softmax output to 1e-9. (fixture: /tmp/torch_nets.py)
        let mut c = Cursor::new(include_bytes!("../tests/fixtures/resnet.bin"));
        let input = c.take(8 * 8 * 3);
        let conv = |c: &mut Cursor, oc, k, ic, pad| Op::Conv {
            weights: c.take(oc * k * k * ic),
            bias: c.take(oc),
            out_channels: oc,
            kernel: k,
            stride: 1,
            pad,
        };
        let bn = |c: &mut Cursor, ch| Op::BatchNorm {
            gamma: c.take(ch),
            beta: c.take(ch),
            mean: c.take(ch),
            var: c.take(ch),
            eps: 1e-5,
        };
        let c1 = conv(&mut c, 4, 3, 3, 1);
        let b1 = bn(&mut c, 4);
        let c2 = conv(&mut c, 4, 3, 4, 1);
        let b2 = bn(&mut c, 4);
        let c3 = conv(&mut c, 4, 3, 4, 1);
        let b3 = bn(&mut c, 4);
        let dense = Op::Dense {
            weights: c.take(3 * 4),
            bias: c.take(3),
            out_dim: 3,
        };
        let expect = c.take(3);
        let g = Graph::new(
            8,
            8,
            3,
            vec![
                Node::source(Op::Input),            // 0
                Node::unary(c1, 0),                 // 1
                Node::unary(b1, 1),                 // 2
                Node::unary(Op::Relu, 2),           // 3  stem
                Node::unary(c2, 3),                 // 4
                Node::unary(b2, 4),                 // 5
                Node::unary(Op::Relu, 5),           // 6
                Node::unary(c3, 6),                 // 7
                Node::unary(b3, 7),                 // 8
                Node::binary(Op::Add, 8, 3),        // 9  residual + stem
                Node::unary(Op::Relu, 9),           // 10
                Node::unary(Op::GlobalAvgPool, 10), // 11
                Node::unary(dense, 11),             // 12
                Node::unary(Op::Softmax, 12),       // 13
            ],
        );
        let out = g.forward(&input);
        assert_eq!(out.len(), 3);
        for (o, e) in out.iter().zip(&expect) {
            assert!((o - e).abs() < 1e-9, "{o} vs torch {e}");
        }
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unet_matches_pytorch_end_to_end() {
        // A real U-Net (encoder → pool → bottleneck → transposed-conv upsample → concat skip → decoder →
        // sigmoid) from PyTorch. Our Graph must reproduce the per-pixel sigmoid mask to 1e-9 — exercising
        // Conv, MaxPool, ConvTranspose, Concat and Sigmoid together. (fixture: /tmp/torch_nets.py)
        let mut c = Cursor::new(include_bytes!("../tests/fixtures/unet.bin"));
        let input = c.take(8 * 8);
        let enc1 = Op::Conv {
            weights: c.take(4 * 3 * 3 * 1),
            bias: c.take(4),
            out_channels: 4,
            kernel: 3,
            stride: 1,
            pad: 1,
        };
        let bott = Op::Conv {
            weights: c.take(8 * 3 * 3 * 4),
            bias: c.take(8),
            out_channels: 8,
            kernel: 3,
            stride: 1,
            pad: 1,
        };
        let up = Op::ConvTranspose {
            weights: c.take(8 * 2 * 2 * 4),
            bias: c.take(4),
            out_channels: 4,
            kernel: 2,
            stride: 2,
            pad: 0,
        };
        let dec1 = Op::Conv {
            weights: c.take(4 * 3 * 3 * 8),
            bias: c.take(4),
            out_channels: 4,
            kernel: 3,
            stride: 1,
            pad: 1,
        };
        let dec2 = Op::Conv {
            weights: c.take(1 * 1 * 1 * 4),
            bias: c.take(1),
            out_channels: 1,
            kernel: 1,
            stride: 1,
            pad: 0,
        };
        let expect = c.take(8 * 8);
        let g = Graph::new(
            8,
            8,
            1,
            vec![
                Node::source(Op::Input),                            // 0
                Node::unary(enc1, 0),                               // 1
                Node::unary(Op::Relu, 1),                           // 2  skip
                Node::unary(Op::MaxPool { pool: 2, stride: 2 }, 2), // 3
                Node::unary(bott, 3),                               // 4
                Node::unary(Op::Relu, 4),                           // 5
                Node::unary(up, 5),                                 // 6
                Node::binary(Op::Concat, 6, 2),                     // 7  up then skip
                Node::unary(dec1, 7),                               // 8
                Node::unary(Op::Relu, 8),                           // 9
                Node::unary(dec2, 9),                               // 10
                Node::unary(Op::Sigmoid, 10),                       // 11
            ],
        );
        let out = g.forward(&input);
        assert_eq!(out.len(), 8 * 8);
        for (o, e) in out.iter().zip(&expect) {
            assert!((o - e).abs() < 1e-9, "{o} vs torch {e}");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // padding copies exact values and writes exact zeros
    fn the_pad_op_borders_the_tensor_with_zeros() {
        // 2×2×1 input, Pad{1} → 4×4×1 with the input centred and a zero border.
        let g = Graph::new(
            2,
            2,
            1,
            vec![Node::source(Op::Input), Node::unary(Op::Pad { pad: 1 }, 0)],
        );
        let out = g.forward(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.len(), 16);
        assert_eq!(&out[0..4], &[0.0, 0.0, 0.0, 0.0]); // top border
        assert_eq!(&out[4..8], &[0.0, 1.0, 2.0, 0.0]); // input row 0
        assert_eq!(&out[8..12], &[0.0, 3.0, 4.0, 0.0]); // input row 1
        assert_eq!(&out[12..16], &[0.0, 0.0, 0.0, 0.0]); // bottom border
    }

    #[test]
    fn conv_transpose_sizes_its_buffer_for_a_non_square_strided_padded_shape() {
        // The U-Net fixture reaches `Op::ConvTranspose` with a square 4×4 tensor at pad 0, and
        // feeds it into `Op::Concat`, which allocates from `rh, rw` and reads only the first
        // `h·w` positions — so an over-sized buffer is laundered before anything asserts on it.
        // Here `h ≠ w`, `stride ≠ 1`, `pad ∉ {0, 1}` and `kernel ≠ stride`, and the node is last,
        // so `forward` returns the buffer itself.
        let input: Vec<f64> = (1..=12).map(f64::from).collect(); // 3×4×1
        let weights: Vec<f64> = (1..=16).map(f64::from).collect(); // [ic=1][ky=4][kx=4][oc=1]
        let bias = vec![0.0];
        let g = Graph::new(
            3,
            4,
            1,
            vec![
                Node::source(Op::Input),
                Node::unary(
                    Op::ConvTranspose {
                        weights: weights.clone(),
                        bias: bias.clone(),
                        out_channels: 1,
                        kernel: 4,
                        stride: 2,
                        pad: 2,
                    },
                    0,
                ),
            ],
        );
        let got = g.forward(&input);
        // (3−1)·2 + 4 − 2·2 = 4 and (4−1)·2 + 4 − 2·2 = 6, worked out by hand rather than by
        // re-running the formula under test.
        assert_eq!(got.len(), 4 * 6);

        let ct = ConvTranspose2d {
            weights: &weights,
            bias: &bias,
            out_channels: 1,
            kernel: 4,
            stride: 2,
            pad: 2,
        };
        let mut manual = vec![0.0; 4 * 6];
        assert_eq!(conv_transpose2d(&input, 3, 4, 1, &ct, &mut manual), (4, 6));
        assert_eq!(got, manual);
    }

    #[test]
    fn dense_flattens_a_non_square_multi_channel_tensor() {
        // The only graph-level `Op::Dense` sits behind a `GlobalAvgPool`, which hands it a 1×1
        // tensor — and `h/w` is `h*w` when both are 1. Here `h ≠ w` and `ch > 1`, so each factor
        // of `h·w·ch` is separately observable.
        let input: Vec<f64> = (1..=12).map(f64::from).collect(); // 2×3×2
        let g = Graph::new(
            2,
            3,
            2,
            vec![
                Node::source(Op::Input),
                Node::unary(
                    Op::Dense {
                        weights: vec![1.0; 12],
                        bias: vec![0.0],
                        out_dim: 1,
                    },
                    0,
                ),
            ],
        );
        assert_eq!(g.forward(&input), vec![78.0]); // 1 + 2 + … + 12
    }

    #[test]
    fn the_pad_op_borders_a_non_square_tensor_at_a_padding_other_than_one() {
        // The 2×2 / pad-1 fixture above asserts every value and still cannot see this line: 2 is
        // the one integer with `h + 2 == h · 2`, and 1 the one padding with `2 · pad == 2 / pad`,
        // so every rewrite of `h + 2·pad` produced the same 4×4. `h ≠ w` and `pad = 2` break both.
        let input: Vec<f64> = (1..=6).map(f64::from).collect(); // 2×3×1
        let g = Graph::new(
            2,
            3,
            1,
            vec![Node::source(Op::Input), Node::unary(Op::Pad { pad: 2 }, 0)],
        );
        let out = g.forward(&input);
        assert_eq!(out.len(), 6 * 7);
        for oy in 0..6 {
            for ox in 0..7 {
                let want = if (2..4).contains(&oy) && (2..5).contains(&ox) {
                    input[(oy - 2) * 3 + (ox - 2)]
                } else {
                    0.0
                };
                assert_eq!(out[oy * 7 + ox], want, "at ({oy},{ox})");
            }
        }
    }
}
