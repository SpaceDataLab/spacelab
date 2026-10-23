//! CNN forward-pass primitives on caller-provided row-major `f64` tensors (HWC layout).
//!
//! Convolution weights are laid out `[out_channel][ky][kx][in_channel]`, dense weights
//! `[out][in]`; biases are per output channel/unit. Each op writes into a caller output slice sized
//! with [`conv_out_dim`] / [`pool_out_dim`].

// Padding offsets cross zero (usize→isize); the single letters (h, w, k, x, y, c) are the notation of
// tensor shapes.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::doc_markdown
)]

/// Output height or width of a convolution: `(size + 2·pad − kernel) / stride + 1`.
#[must_use]
pub fn conv_out_dim(size: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (size + 2 * pad - kernel) / stride + 1
}

/// Output height or width of a pooling window: `(size − pool) / stride + 1`.
#[must_use]
pub fn pool_out_dim(size: usize, pool: usize, stride: usize) -> usize {
    (size - pool) / stride + 1
}

/// A 2-D convolution layer's parameters (square kernel).
pub struct Conv2d<'w> {
    /// Weights, `[out_channel][ky][kx][in_channel]` row-major.
    pub weights: &'w [f64],
    /// Bias per output channel.
    pub bias: &'w [f64],
    /// Number of output channels.
    pub out_channels: usize,
    /// Square kernel side.
    pub kernel: usize,
    /// Stride.
    pub stride: usize,
    /// Zero-padding on each side.
    pub pad: usize,
}

/// 2-D convolution of an `h × w × in_ch` HWC `input` with `conv`, writing the `out_h × out_w ×
/// out_channels` result into `out`. Returns `(out_h, out_w)`.
///
/// # Panics
///
/// If any slice is smaller than its layout requires.
#[must_use]
pub fn conv2d(
    input: &[f64],
    h: usize,
    w: usize,
    in_ch: usize,
    conv: &Conv2d,
    out: &mut [f64],
) -> (usize, usize) {
    let (k, stride, pad, oc_n) = (conv.kernel, conv.stride, conv.pad, conv.out_channels);
    let out_h = conv_out_dim(h, k, stride, pad);
    let out_w = conv_out_dim(w, k, stride, pad);
    assert!(input.len() >= h * w * in_ch, "input too small");
    assert!(out.len() >= out_h * out_w * oc_n, "output too small");
    assert!(
        conv.weights.len() >= oc_n * k * k * in_ch && conv.bias.len() >= oc_n,
        "weights/bias too small"
    );
    for oy in 0..out_h {
        for ox in 0..out_w {
            for oc in 0..oc_n {
                let mut acc = conv.bias[oc];
                for ky in 0..k {
                    let sy = (oy * stride + ky) as isize - pad as isize;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    let sy = sy as usize;
                    for kx in 0..k {
                        let sx = (ox * stride + kx) as isize - pad as isize;
                        if sx < 0 || sx >= w as isize {
                            continue;
                        }
                        let in_base = (sy * w + sx as usize) * in_ch;
                        let wt_base = ((oc * k + ky) * k + kx) * in_ch;
                        acc += input[in_base..in_base + in_ch]
                            .iter()
                            .zip(&conv.weights[wt_base..wt_base + in_ch])
                            .map(|(a, b)| a * b)
                            .sum::<f64>();
                    }
                }
                out[(oy * out_w + ox) * oc_n + oc] = acc;
            }
        }
    }
    (out_h, out_w)
}

/// In-place `ReLU`: `max(0, x)` on every element.
pub fn relu(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = (*v).max(0.0);
    }
}

/// 2-D max pooling of an `h × w × ch` HWC `input`, writing the pooled result into `out`. Returns
/// `(out_h, out_w)`.
///
/// # Panics
///
/// If `out` is smaller than the pooled size.
pub fn max_pool2d(
    input: &[f64],
    h: usize,
    w: usize,
    ch: usize,
    pool: usize,
    stride: usize,
    out: &mut [f64],
) -> (usize, usize) {
    let out_h = pool_out_dim(h, pool, stride);
    let out_w = pool_out_dim(w, pool, stride);
    assert!(out.len() >= out_h * out_w * ch, "output too small");
    for oy in 0..out_h {
        for ox in 0..out_w {
            for c in 0..ch {
                let mut m = f64::NEG_INFINITY;
                for py in 0..pool {
                    for px in 0..pool {
                        m = m.max(input[((oy * stride + py) * w + (ox * stride + px)) * ch + c]);
                    }
                }
                out[(oy * out_w + ox) * ch + c] = m;
            }
        }
    }
    (out_h, out_w)
}

/// Dense (fully-connected) layer: `out = weights · input + bias`, with `weights` laid out
/// `[out_dim][in_dim]`.
///
/// # Panics
///
/// If any slice is smaller than its dimension requires.
pub fn dense(
    input: &[f64],
    weights: &[f64],
    bias: &[f64],
    in_dim: usize,
    out_dim: usize,
    out: &mut [f64],
) {
    assert!(
        input.len() >= in_dim && bias.len() >= out_dim,
        "input/bias too small"
    );
    assert!(
        weights.len() >= out_dim * in_dim && out.len() >= out_dim,
        "weights/out too small"
    );
    for (o, slot) in out.iter_mut().take(out_dim).enumerate() {
        let base = o * in_dim;
        *slot = bias[o]
            + weights[base..base + in_dim]
                .iter()
                .zip(&input[..in_dim])
                .map(|(a, b)| a * b)
                .sum::<f64>();
    }
}

/// In-place numerically-stable softmax (subtracts the max before exponentiating).
pub fn softmax(x: &mut [f64]) {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = libm::exp(*v - max);
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// In-place logistic sigmoid `1/(1+e^{−x})` on every element.
pub fn sigmoid(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = 1.0 / (1.0 + libm::exp(-*v));
    }
}

/// In-place residual add `a += b` (elementwise) — the skip connection of a ResNet block.
///
/// # Panics
///
/// If `b` is shorter than `a`.
pub fn add(a: &mut [f64], b: &[f64]) {
    assert!(b.len() >= a.len(), "residual shorter than input");
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

/// Per-channel inference batch-norm parameters: `y = gamma·(x−mean)/√(var+eps) + beta`.
pub struct BatchNormParams<'p> {
    /// Scale per channel.
    pub gamma: &'p [f64],
    /// Shift per channel.
    pub beta: &'p [f64],
    /// Running mean per channel.
    pub mean: &'p [f64],
    /// Running variance per channel.
    pub var: &'p [f64],
    /// Numerical epsilon.
    pub eps: f64,
}

/// In-place inference batch normalization of an HWC tensor with `pixels = h·w` spatial positions and
/// `ch` channels.
pub fn batch_norm(x: &mut [f64], pixels: usize, ch: usize, bn: &BatchNormParams) {
    for c in 0..ch {
        let scale = bn.gamma[c] / libm::sqrt(bn.var[c] + bn.eps);
        let shift = bn.beta[c] - bn.mean[c] * scale;
        for p in 0..pixels {
            x[p * ch + c] = x[p * ch + c] * scale + shift;
        }
    }
}

/// Global average pooling of an HWC tensor (`pixels = h·w`, `ch` channels): the spatial mean per
/// channel, written to `out` (length `ch`).
pub fn global_avg_pool(input: &[f64], pixels: usize, ch: usize, out: &mut [f64]) {
    let n = pixels as f64;
    for c in 0..ch {
        let mut s = 0.0;
        for p in 0..pixels {
            s += input[p * ch + c];
        }
        out[c] = s / n;
    }
}

/// Concatenate two HWC tensors along the channel axis (both `pixels = h·w`): `a`'s `ca` channels then
/// `b`'s `cb` channels, into `out` (`pixels·(ca+cb)`) — the U-Net skip merge.
pub fn concat_channels(a: &[f64], ca: usize, b: &[f64], cb: usize, pixels: usize, out: &mut [f64]) {
    let cc = ca + cb;
    for p in 0..pixels {
        for c in 0..ca {
            out[p * cc + c] = a[p * ca + c];
        }
        for c in 0..cb {
            out[p * cc + ca + c] = b[p * cb + c];
        }
    }
}

/// Zero-pad an HWC tensor by `pad` on every side, writing the `(h+2·pad) × (w+2·pad) × ch` result into
/// `out` (which is fully overwritten). Returns `(out_h, out_w)`.
pub fn pad2d(
    input: &[f64],
    h: usize,
    w: usize,
    ch: usize,
    pad: usize,
    out: &mut [f64],
) -> (usize, usize) {
    let (oh, ow) = (h + 2 * pad, w + 2 * pad);
    for v in out.iter_mut().take(oh * ow * ch) {
        *v = 0.0;
    }
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * ch;
            let dst = ((y + pad) * ow + (x + pad)) * ch;
            out[dst..dst + ch].copy_from_slice(&input[src..src + ch]);
        }
    }
    (oh, ow)
}

/// A transposed-convolution (up-convolution) layer's parameters.
pub struct ConvTranspose2d<'w> {
    /// Weights, `[in_channel][ky][kx][out_channel]` row-major.
    pub weights: &'w [f64],
    /// Bias per output channel.
    pub bias: &'w [f64],
    /// Number of output channels.
    pub out_channels: usize,
    /// Square kernel side.
    pub kernel: usize,
    /// Stride (upsampling factor).
    pub stride: usize,
    /// Zero-padding removed from each side of the output.
    pub pad: usize,
}

/// 2-D transposed convolution (learned upsampling) of an `h × w × in_ch` HWC `input`, writing the
/// `out_h × out_w × out_channels` result into `out`, where `out_h = (h−1)·stride + kernel − 2·pad`.
/// Each input element scatter-adds a scaled kernel into the output — the U-Net decoder's upsampler.
/// Returns `(out_h, out_w)`.
#[must_use]
pub fn conv_transpose2d(
    input: &[f64],
    h: usize,
    w: usize,
    in_ch: usize,
    ct: &ConvTranspose2d,
    out: &mut [f64],
) -> (usize, usize) {
    let (k, stride, pad, oc_n) = (ct.kernel, ct.stride, ct.pad, ct.out_channels);
    let out_h = (h - 1) * stride + k - 2 * pad;
    let out_w = (w - 1) * stride + k - 2 * pad;
    for p in 0..out_h * out_w {
        for oc in 0..oc_n {
            out[p * oc_n + oc] = ct.bias[oc];
        }
    }
    for iy in 0..h {
        for ix in 0..w {
            for ic in 0..in_ch {
                let v = input[(iy * w + ix) * in_ch + ic];
                for ky in 0..k {
                    let oy = (iy * stride + ky) as isize - pad as isize;
                    if oy < 0 || oy >= out_h as isize {
                        continue;
                    }
                    for kx in 0..k {
                        let ox = (ix * stride + kx) as isize - pad as isize;
                        if ox < 0 || ox >= out_w as isize {
                            continue;
                        }
                        let wt = ((ic * k + ky) * k + kx) * oc_n;
                        let ob = (oy as usize * out_w + ox as usize) * oc_n;
                        for oc in 0..oc_n {
                            out[ob + oc] += v * ct.weights[wt + oc];
                        }
                    }
                }
            }
        }
    }
    (out_h, out_w)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn conv2d_matches_a_hand_computed_window_sum() {
        // 3×3 single-channel input; a 2×2 all-ones kernel sums each window.
        let input = [1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let weights = [1., 1., 1., 1.];
        let bias = [0.0];
        let conv = Conv2d {
            weights: &weights,
            bias: &bias,
            out_channels: 1,
            kernel: 2,
            stride: 1,
            pad: 0,
        };
        let mut out = [0.0; 4];
        let dims = conv2d(&input, 3, 3, 1, &conv, &mut out);
        assert_eq!(dims, (2, 2));
        assert_eq!(out, [12., 16., 24., 28.]);
    }

    #[test]
    fn relu_clamps_negatives() {
        let mut x = [-1.0, 2.0, -3.0, 0.5];
        relu(&mut x);
        assert_eq!(x, [0.0, 2.0, 0.0, 0.5]);
    }

    #[test]
    fn max_pool_takes_the_window_maximum() {
        let input = [1.0, 4.0, 3.0, 2.0]; // 2×2×1
        let mut out = [0.0];
        let dims = max_pool2d(&input, 2, 2, 1, 2, 2, &mut out);
        assert_eq!(dims, (1, 1));
        assert_eq!(out[0], 4.0);
    }

    #[test]
    fn dense_is_matrix_vector_plus_bias() {
        let input = [1.0, 2.0];
        let weights = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3][2]
        let bias = [0.0, 0.0, 10.0];
        let mut out = [0.0; 3];
        dense(&input, &weights, &bias, 2, 3, &mut out);
        assert_eq!(out, [1.0, 2.0, 13.0]);
    }

    #[test]
    fn softmax_is_a_probability_distribution() {
        let mut x = [0.0, 0.0, 0.0];
        softmax(&mut x);
        for &v in &x {
            assert!((v - 1.0 / 3.0).abs() < 1e-12);
        }
        let mut y = [1.0, 2.0, 3.0];
        softmax(&mut y);
        assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(y[2] > y[1] && y[1] > y[0]); // order-preserving
    }

    #[test]
    fn tiny_forward_pass_is_a_valid_classifier() {
        // conv → relu → maxpool → dense → softmax on a 4×4×1 image with mixed signs.
        let input: Vec<f64> = (0..16).map(|i| f64::from(i) - 8.0).collect();
        let weights = [1.0, 1.0, 1.0, 1.0];
        let bias = [0.0];
        let conv = Conv2d {
            weights: &weights,
            bias: &bias,
            out_channels: 1,
            kernel: 2,
            stride: 1,
            pad: 0,
        };
        let mut c = vec![0.0; 3 * 3];
        let (ch, cw) = conv2d(&input, 4, 4, 1, &conv, &mut c);
        relu(&mut c);
        let mut p = vec![0.0; 1];
        max_pool2d(&c, ch, cw, 1, 3, 3, &mut p); // 3×3 → 1×1: the strongest response
        let mut logits = [0.0; 2];
        dense(&p, &[1.0, -1.0], &[0.0, 0.0], 1, 2, &mut logits);
        softmax(&mut logits);
        assert!((logits.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // The pooled response is positive, so the +1 class wins.
        assert!(logits[0] > logits[1]);
    }

    #[test]
    fn sigmoid_is_the_logistic() {
        let mut x = [0.0, 100.0, -100.0, 1.0];
        sigmoid(&mut x);
        assert!((x[0] - 0.5).abs() < 1e-15);
        assert!(x[1] > 0.999_999 && x[2] < 1e-6);
        assert!((x[3] - 1.0 / (1.0 + libm::exp(-1.0))).abs() < 1e-15);
    }

    #[test]
    fn add_is_the_residual() {
        let mut a = [1.0, 2.0, 3.0];
        add(&mut a, &[10.0, 20.0, 30.0]);
        assert_eq!(a, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn batch_norm_standardizes_per_channel() {
        // Two pixels, two channels (HWC): channel 0 = {1, 3}, channel 1 = {5, 7}.
        let mut x = [1.0, 5.0, 3.0, 7.0];
        let bn = BatchNormParams {
            gamma: &[1.0, 1.0],
            beta: &[0.0, 0.0],
            mean: &[2.0, 6.0],
            var: &[1.0, 1.0],
            eps: 0.0,
        };
        batch_norm(&mut x, 2, 2, &bn);
        assert_eq!(x, [-1.0, -1.0, 1.0, 1.0]); // (x − mean)/√var
    }

    #[test]
    fn global_avg_pool_is_the_spatial_mean() {
        // 2×2×2 HWC; channel 0 = {1,2,3,4} (mean 2.5), channel 1 = {10,20,30,40} (mean 25).
        let input = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let mut out = [0.0; 2];
        global_avg_pool(&input, 4, 2, &mut out);
        assert_eq!(out, [2.5, 25.0]);
    }

    #[test]
    fn concat_channels_appends_the_skip() {
        // Two 1×2 tensors: a has 1 channel, b has 2 → 3-channel output, a first.
        let a = [1.0, 2.0]; // pixels 0,1
        let b = [3.0, 4.0, 5.0, 6.0]; // (3,4) then (5,6)
        let mut out = [0.0; 6];
        concat_channels(&a, 1, &b, 2, 2, &mut out);
        assert_eq!(out, [1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn pad2d_frames_the_input_with_zeros() {
        let input = [5.0]; // 1×1×1
        let mut out = [0.0; 9]; // 3×3×1
        let dims = pad2d(&input, 1, 1, 1, 1, &mut out);
        assert_eq!(dims, (3, 3));
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn conv_transpose_upsamples_by_scatter() {
        // 2×2 input, 2×2 kernel, stride 2 (non-overlapping) ⇒ 4×4, each 2×2 block = input·kernel.
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = [1.0, 2.0, 3.0, 4.0]; // [ic=1][ky=2][kx=2][oc=1]
        let bias = [0.0];
        let ct = ConvTranspose2d {
            weights: &weights,
            bias: &bias,
            out_channels: 1,
            kernel: 2,
            stride: 2,
            pad: 0,
        };
        let mut out = [0.0; 16];
        let dims = conv_transpose2d(&input, 2, 2, 1, &ct, &mut out);
        assert_eq!(dims, (4, 4));
        assert_eq!(
            out,
            [
                1., 2., 2., 4., // top-left block ·1, top-right ·2
                3., 4., 6., 8., //
                3., 6., 4., 8., // bottom-left ·3, bottom-right ·4
                9., 12., 12., 16.,
            ]
        );
    }

    #[test]
    fn resnet_block_is_the_identity_when_the_convolutions_are_zero() {
        // conv → bn → relu → conv → bn → (+input) → relu with zero convolutions and identity BN
        // collapses to relu(input): the residual path carries the signal through.
        let input: Vec<f64> = (0..4 * 4 * 2).map(|i| f64::from(i) - 12.0).collect();
        let (h, w, c) = (4usize, 4usize, 2usize);
        let zero_w = vec![0.0; c * 3 * 3 * c];
        let zero_b = vec![0.0; c];
        let conv = Conv2d {
            weights: &zero_w,
            bias: &zero_b,
            out_channels: c,
            kernel: 3,
            stride: 1,
            pad: 1,
        };
        let id_bn = BatchNormParams {
            gamma: &[1.0, 1.0],
            beta: &[0.0, 0.0],
            mean: &[0.0, 0.0],
            var: &[1.0, 1.0],
            eps: 0.0,
        };
        let mut buf = vec![0.0; h * w * c];
        let _ = conv2d(&input, h, w, c, &conv, &mut buf);
        batch_norm(&mut buf, h * w, c, &id_bn);
        relu(&mut buf);
        let mut buf2 = vec![0.0; h * w * c];
        let _ = conv2d(&buf, h, w, c, &conv, &mut buf2);
        batch_norm(&mut buf2, h * w, c, &id_bn);
        add(&mut buf2, &input); // residual
        relu(&mut buf2);

        let mut expect = input.clone();
        relu(&mut expect);
        assert_eq!(buf2, expect);
    }

    #[test]
    fn unet_decoder_step_upsamples_concatenates_and_convolves() {
        // Upsample a 2×2×2 feature map ×2, concat a 4×4×3 skip, then a 3×3 conv → the shapes a U-Net
        // decoder produces, and the concat carries the up-features then the skip.
        let input = vec![1.0; 2 * 2 * 2];
        let ct_w = vec![0.5; 2 * 2 * 2 * 2]; // [ic=2][ky=2][kx=2][oc=2]
        let ct_b = [0.0, 0.0];
        let ct = ConvTranspose2d {
            weights: &ct_w,
            bias: &ct_b,
            out_channels: 2,
            kernel: 2,
            stride: 2,
            pad: 0,
        };
        let mut up = vec![0.0; 4 * 4 * 2];
        let (uh, uw) = conv_transpose2d(&input, 2, 2, 2, &ct, &mut up);
        assert_eq!((uh, uw), (4, 4));

        let skip = vec![9.0; 4 * 4 * 3];
        let mut cat = vec![0.0; 4 * 4 * 5];
        concat_channels(&up, 2, &skip, 3, uh * uw, &mut cat);
        // The first two channels of each pixel are the up-features, the next three the skip.
        assert_eq!(&cat[0..2], &up[0..2]);
        assert_eq!(&cat[2..5], &[9.0, 9.0, 9.0]);

        let conv_w = vec![0.01; 4 * 3 * 3 * 5]; // 3×3, 5→4 channels
        let conv_b = vec![0.0; 4];
        let conv = Conv2d {
            weights: &conv_w,
            bias: &conv_b,
            out_channels: 4,
            kernel: 3,
            stride: 1,
            pad: 1,
        };
        let mut dec = vec![0.0; 4 * 4 * 4];
        let (dh, dw) = conv2d(&cat, uh, uw, 5, &conv, &mut dec);
        assert_eq!((dh, dw), (4, 4)); // conv preserves the upsampled spatial size
    }

    #[test]
    fn conv2d_strides_and_pads() {
        // Every convolution in the suite above runs at stride 1 and pad 0, which makes `oy · stride`
        // the identity and `− pad` a no-op, so neither the output-size formula nor the input
        // coordinates could be observed. A 3×3 all-ones kernel at stride 2 over a padded 4×4 sums
        // exactly the windows written out below.
        assert_eq!(conv_out_dim(4, 3, 2, 1), 2);
        assert_eq!(conv_out_dim(7, 3, 2, 1), 4);
        assert_eq!(conv_out_dim(5, 1, 1, 0), 5);

        let input: Vec<f64> = (1..=16).map(f64::from).collect();
        let weights = [1.0; 9];
        let bias = [0.0];
        let conv = Conv2d {
            weights: &weights,
            bias: &bias,
            out_channels: 1,
            kernel: 3,
            stride: 2,
            pad: 1,
        };
        let mut out = [0.0; 4];
        assert_eq!(conv2d(&input, 4, 4, 1, &conv, &mut out), (2, 2));
        //  (0,0): 1+2+5+6            (0,1): 2+3+4+6+7+8
        //  (1,0): 5+6+9+10+13+14     (1,1): 6+7+8+10+11+12+14+15+16
        assert_eq!(out, [14.0, 30.0, 57.0, 99.0]);
    }

    #[test]
    fn pad2d_frames_a_multi_channel_tensor_and_clears_what_was_there() {
        // The existing fixture is 1×1×1 with pad 1, where the output size, the row stride, the channel
        // stride and the extent of the zero-fill are all indistinguishable from each other. Use a
        // 2×3×3 input with pad 2, and start from a buffer full of a sentinel so that a fill which
        // stops short leaves it visible.
        let input: Vec<f64> = (1..=18).map(f64::from).collect();
        let mut out = vec![-99.0; 6 * 7 * 3];
        assert_eq!(pad2d(&input, 2, 3, 3, 2, &mut out), (6, 7));
        for oy in 0..6 {
            for ox in 0..7 {
                for c in 0..3 {
                    let want = if (2..4).contains(&oy) && (2..5).contains(&ox) {
                        input[((oy - 2) * 3 + (ox - 2)) * 3 + c]
                    } else {
                        0.0
                    };
                    assert_eq!(out[(oy * 7 + ox) * 3 + c], want, "at ({oy},{ox},{c})");
                }
            }
        }
    }

    #[test]
    fn conv_transpose_crops_the_padding_it_promises_to() {
        // The existing fixture has pad 0 and a kernel equal to the stride, so the scatter never
        // overlaps and `− pad` is a no-op — the two bounds tests are never both consulted and the
        // output-size formula is never wrong by an amount anything notices. A 3×3 kernel at stride 2
        // with pad 1 overlaps *and* crops: (2−1)·2 + 3 − 2 = 3.
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights: Vec<f64> = (1..=9).map(f64::from).collect(); // [ic=1][ky=3][kx=3][oc=1]
        let bias = [0.0];
        let ct = ConvTranspose2d {
            weights: &weights,
            bias: &bias,
            out_channels: 1,
            kernel: 3,
            stride: 2,
            pad: 1,
        };
        let mut out = [0.0; 9];
        assert_eq!(conv_transpose2d(&input, 2, 2, 1, &ct, &mut out), (3, 3));
        // Each input v scatters v·w[ky][kx] to (2·iy + ky − 1, 2·ix + kx − 1), dropping what falls
        // outside; the centre cell collects one tap from each of the four inputs, 9 + 14 + 9 + 4.
        assert_eq!(out, [5.0, 14.0, 10.0, 14.0, 36.0, 24.0, 15.0, 34.0, 20.0]);
    }

    #[test]
    fn softmax_survives_arguments_that_would_overflow_the_exponential() {
        // The doc comment claims numerical stability and the existing test runs on 0..2, where
        // subtracting the max is optional. At 1000 it is not: `exp(2002)` is infinite, and the whole
        // distribution collapses to zeros.
        let mut x = [1000.0, 1001.0, 1002.0];
        softmax(&mut x);
        let z = libm::exp(-2.0) + libm::exp(-1.0) + 1.0;
        for (v, want) in x
            .iter()
            .zip([libm::exp(-2.0) / z, libm::exp(-1.0) / z, 1.0 / z])
        {
            assert!((v - want).abs() < 1e-15, "{x:?}");
        }
        assert!((x.iter().sum::<f64>() - 1.0).abs() < 1e-15, "{x:?}");
    }
}
