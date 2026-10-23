//! **deepvision** — the CNN forward-pass primitives, the third Phase-8 line.
//!
//! The building blocks that run a convolutional network over space imagery: [`conv2d`], [`relu`],
//! [`max_pool2d`], [`dense`] and [`softmax`]. Pure Rust, `no_std`, `libm` only — **no ML framework**,
//! so it links anywhere (embedded / offline) and adds no heavy dependency tree. Tensors are
//! caller-provided row-major `f64` slices in **HWC** layout (height, width, channel); a caller sizes
//! the outputs with [`conv_out_dim`] / [`pool_out_dim`].
//!
//! This is the inference kernel. Loading *trained* models exported to ONNX is the opt-in `onnx`
//! module (`--features onnx`, pure-Rust `tract`); it pulls a `std`-only dependency and stays out of
//! the default `no_std` build.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

pub mod nn;

#[cfg(any(feature = "std", test))]
pub mod model;

#[cfg(any(feature = "std", test))]
pub mod graph;

#[cfg(any(feature = "std", test))]
pub mod detect;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use nn::{
    BatchNormParams, Conv2d, ConvTranspose2d, add, batch_norm, concat_channels, conv_out_dim,
    conv_transpose2d, conv2d, dense, global_avg_pool, max_pool2d, pad2d, pool_out_dim, relu,
    sigmoid, softmax,
};

#[cfg(any(feature = "std", test))]
pub use model::{Layer, Sequential, weights_from_le_bytes};

#[cfg(any(feature = "std", test))]
pub use graph::{Graph, Node, Op};

#[cfg(any(feature = "std", test))]
pub use detect::{BBox, decode_box, generate_anchors, nms};

#[cfg(feature = "onnx")]
pub use onnx::{OnnxError, OnnxModel, threshold_mask};
