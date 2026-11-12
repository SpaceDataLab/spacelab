#!/usr/bin/env python3
"""The coordinates of `tools/residual-maps.py` are `dipole.rs`'s, whose values `derivations/geomagnetic_dipole.mac` gates.

uv run --project tools --locked python -I tools/test_residual_maps.py
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "residual_maps", HERE / "residual-maps.py"
)
assert spec is not None and spec.loader is not None
maps = importlib.util.module_from_spec(spec)
sys.modules["residual_maps"] = maps
spec.loader.exec_module(maps)

# `the_pole_moment_and_coordinates_are_the_dipole_s_own` in dipole.rs: the pole at 2025.0, and the magnetic
# latitude and longitude of two points, 40 digits from the Maxima gate.
POLE_2025 = (
    1.410040345385084652804351241132443450879,
    -1.269950826609024893067310314044349777553,
)
POINTS = (
    (
        0.75,
        -1.25,
        9.107179692664895196843757699506943027127e-1,
        2.380742697044007398825410601065239014113e-2,
    ),
    (
        -0.5,
        2.0,
        -6.592931672066791639928741206656271586312e-1,
        -2.99898909685567668133232270057321063092,
    ),
)


def test_the_pole_and_coordinates_at_2025_are_the_gated_ones() -> None:
    axis = maps.dipole_axis(np.array([2025.0]))
    assert abs(math.asin(axis[0, 2]) - POLE_2025[0]) < 1e-15
    assert abs(math.atan2(axis[0, 1], axis[0, 0]) - POLE_2025[1]) < 1e-15
    for lat, lon, mlat, mlon in POINTS:
        got = maps.magnetic(axis, np.array([lat]), np.array([lon]))
        assert abs(got[0][0] - mlat) < 1e-15 and abs(got[1][0] - mlon) < 1e-15, (
            lat,
            lon,
            got,
        )


def test_the_coefficients_are_the_table_at_its_epochs_and_linear_between() -> None:
    epochs, values = maps.dipole_table()
    for k in (epochs.size - 6, epochs.size - 2):
        g10, g11, h11 = values[k]
        m = math.sqrt(g10 * g10 + g11 * g11 + h11 * h11)
        axis = maps.dipole_axis(np.array([epochs[k]]))[0]
        assert np.allclose(axis, [-g11 / m, -h11 / m, -g10 / m], rtol=0, atol=1e-15), (
            epochs[k]
        )
    middle = 0.5 * (values[-3] + values[-2])
    g10, g11, h11 = middle
    m = math.sqrt(g10 * g10 + g11 * g11 + h11 * h11)
    axis = maps.dipole_axis(np.array([0.5 * (epochs[-2] + epochs[-1])]))[0]
    assert np.allclose(axis, [-g11 / m, -h11 / m, -g10 / m], rtol=0, atol=1e-15)


def test_the_sun_s_own_meridian_is_magnetic_noon_and_its_antipode_midnight() -> None:
    axis = maps.dipole_axis(np.array([2010.0, 2010.0]))
    lon = np.radians([40.0, 220.0])
    dec = np.radians([5.0, -12.0])
    # A row at the sub-solar point itself reads local solar time 12 h.
    noon = maps.magnetic_local_time(axis, dec, lon, np.array([12.0, 12.0]), dec)
    assert np.allclose(noon, 12.0, atol=1e-12), noon
    midnight = maps.magnetic_local_time(
        axis, -dec, lon + np.pi, np.array([0.0, 0.0]), dec
    )
    assert np.allclose(midnight % 24.0, 0.0, atol=1e-12) or np.allclose(
        midnight, 24.0, atol=1e-12
    ), midnight


def test_a_nan_outside_both_groups_leaves_the_contrast_and_its_interval_alone() -> None:
    rng = np.random.default_rng(7)
    clusters = np.repeat(np.arange(40), 10)
    values = rng.normal(size=clusters.size) + (clusters % 2)
    a = np.tile(np.arange(10) < 4, 40)
    b = np.tile(np.arange(10) >= 6, 40)
    clean = maps.contrast(values, a, b, clusters, 500, np.random.default_rng(1))
    # Every cluster carries a NaN on a row of neither group, as a model does where it has no answer.
    spoiled = values.copy()
    spoiled[np.tile(np.arange(10) == 5, 40)] = np.nan
    seen = maps.contrast(spoiled, a, b, clusters, 500, np.random.default_rng(1))
    assert seen == clean, (seen, clean)
    assert math.isclose(clean[0], values[a].mean() - values[b].mean(), rel_tol=1e-12)
    assert clean[1] < clean[0] < clean[2], clean


def test_the_mean_interval_is_the_mean_and_collapses_when_every_cluster_agrees() -> (
    None
):
    clusters = np.repeat(np.arange(30), 4)
    values = np.repeat(np.linspace(-1.0, 1.0, 30), 4)
    a = np.tile([True, True, False, True], 30)
    m = maps.mean_interval(values, a, clusters, 400, np.random.default_rng(3))
    assert math.isclose(m[0], values[a].mean(), abs_tol=1e-15) and m[1] < m[0] < m[2], m
    flat = maps.mean_interval(
        np.full(120, 0.25), a, clusters, 400, np.random.default_rng(3)
    )
    assert np.allclose(flat, 0.25, rtol=0.0, atol=1e-15), flat


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
