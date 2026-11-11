#!/usr/bin/env python3
"""`tools/no-step-accuracy.py` against `derivations/sdtm_no_step.mac`'s 40-digit values, and its single frozen step
against `cooling_grid`.

    uv run --no-project --with numpy --with pandas --with scipy python -I tools/test_no_step_accuracy.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "no_step_accuracy", HERE / "no-step-accuracy.py"
)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["no_step_accuracy"] = tool
spec.loader.exec_module(tool)
fit = tool.load("evaluate_sdtm", "evaluate-sdtm.py").fit

# The gate's shipped constants, hours and ap, and its four states (D0, N0, H) with the exact D after 3 h.
SHIPPED = (
    6295024178295275 / 3.6e14,
    11864859040774021 / 3.6e14,
    2477339239626375 / 1e14,
)
QUIET = 7.0 / (1.0 + 7.0 / SHIPPED[2])
STATES = (
    (QUIET, 7.0, 300.0, 4.762337486992401450533057980401915786359e1),
    (100.0, 50.0, 300.0, 9.351877850432057488886992206688563205822e1),
    (200.0, 250.0, 7.0, 3.268600295901665138825566032361688976479e1),
    (25.0, 25.0, 80.0, 2.890174578042530797076779496998899109804e1),
)
# State 1 by the frozen step taken 1, 2 and 4 times across the interval.
FROZEN_1 = {
    1: 4.826744663563632020381657310165554284854e1,
    2: 4.778366668363259473193813576693850872342e1,
    4: 4.766340293714888448323564590456697131921e1,
}


def test_the_exact_step_is_the_gate_s_at_every_state() -> None:
    d0, n0, drive, exact = (np.array(c) for c in zip(*STATES, strict=True))
    alpha, beta = tool.exact_coefficients(n0, drive, *SHIPPED, 3.0)
    assert np.allclose(alpha * d0 + beta, exact, rtol=2e-15, atol=0.0), (
        alpha * d0 + beta - exact
    )


def test_both_grids_reach_the_gate_s_first_state_after_a_quiet_spell() -> None:
    ap = np.concatenate([np.full(2000, 7.0), [300.0]])
    exact = tool.exact_grid(ap, *SHIPPED, 10_800.0, 0)
    assert np.isclose(exact[-2], QUIET, rtol=1e-14) and np.isclose(
        exact[-1], STATES[0][3], rtol=1e-14
    )
    for k, value in FROZEN_1.items():
        frozen = tool.frozen_grid(ap, *SHIPPED, 10_800.0, 0, k)
        assert np.isclose(frozen[-1], value, rtol=1e-14), (k, frozen[-1] - value)


def test_one_frozen_step_is_cooling_grid_bit_for_bit() -> None:
    ap = np.random.default_rng(11).gamma(0.6, 25.0, 3000)
    ours = tool.frozen_grid(ap, *SHIPPED, 10_800.0, 40, 1)
    assert np.array_equal(ours, fit.cooling_grid(ap, *SHIPPED, 10_800.0, 40))


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
