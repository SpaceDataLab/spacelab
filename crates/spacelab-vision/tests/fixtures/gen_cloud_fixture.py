#!/usr/bin/env python3
"""Generate the ONNX cloud-mask verification fixture for the `onnx` feature test.

A tiny but real segmentation CNN (Conv -> ReLU -> MaxPool -> Conv -> Sigmoid) with seeded weights,
plus its onnxruntime reference output on a seeded input. The Rust test loads the same `.onnx` through
`tract` and asserts a per-element match, so onnxruntime is the oracle and `tract` is the system under
test. Kept tiny (a few KB) so it is a legitimate committed test fixture, not a dataset.

Reproduce (requires `onnx` + `onnxruntime`, not project dependencies):
    python3 gen_cloud_fixture.py
Outputs `cloud_tiny.onnx` and `cloud_tiny_io.json` next to this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
IN_SHAPE = (1, 3, 8, 8)  # N C H W

rng = np.random.default_rng(0)
w1 = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)  # out=4, in=3, 3x3
b1 = rng.standard_normal((4,)).astype(np.float32)
w2 = rng.standard_normal((1, 4, 1, 1)).astype(np.float32)  # out=1, in=4, 1x1
b2 = rng.standard_normal((1,)).astype(np.float32)

nodes = [
    helper.make_node("Conv", ["X", "W1", "B1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["c1"], ["r1"]),
    helper.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
    helper.make_node("Conv", ["p1", "W2", "B2"], ["c2"], kernel_shape=[1, 1]),
    helper.make_node("Sigmoid", ["c2"], ["Y"]),
]
graph = helper.make_graph(
    nodes,
    "cloud_tiny",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(IN_SHAPE))],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])],
    [
        numpy_helper.from_array(w1, "W1"),
        numpy_helper.from_array(b1, "B1"),
        numpy_helper.from_array(w2, "W2"),
        numpy_helper.from_array(b2, "B2"),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
model.ir_version = 9  # onnxruntime/tract-compatible
onnx.checker.check_model(model)
onnx.save(model, HERE / "cloud_tiny.onnx")

x = rng.standard_normal(IN_SHAPE).astype(np.float32)
sess = ort.InferenceSession((HERE / "cloud_tiny.onnx").read_bytes(), providers=["CPUExecutionProvider"])
y = sess.run(["Y"], {"X": x})[0].astype(np.float32)

# Raw little-endian f32 blobs so the Rust test reads them with `include_bytes!` + `f32::from_le_bytes`
# — no JSON/serde dependency in the crate. Shapes are constants in the test (1x3x8x8 -> 1x1x4x4).
(HERE / "cloud_tiny_input.f32").write_bytes(x.reshape(-1).tobytes())
(HERE / "cloud_tiny_output.f32").write_bytes(y.reshape(-1).tobytes())
print(f"wrote cloud_tiny.onnx ({(HERE / 'cloud_tiny.onnx').stat().st_size} B) + input/output .f32")
print(f"output range [{y.min():.6f}, {y.max():.6f}]")
