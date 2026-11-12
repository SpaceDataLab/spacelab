#!/usr/bin/env python3
"""The resampler against the laws of `derivations/cluster_design_effect.mac`, on series whose rho is known.

uv run --project tools --locked python -I tools/test_cluster_intervals.py
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "cluster_intervals", HERE / "cluster-intervals.py"
)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["cluster_intervals"] = tool
spec.loader.exec_module(tool)


def test_the_phase_resampler_keeps_storms_whole_and_reports_the_pair() -> None:
    # Two storms: every comparison of one reads +0.1 and of the other -0.1, on model a. Model b is a
    # shifted by 0.3, so its spread equals a's on every draw and only the means move.
    storms = np.array([0, 0, 0, 1, 1, 1])
    log_a = np.array([0.1, 0.1, 0.1, -0.1, -0.1, -0.1])
    draws = tool.phase_draws(log_a, log_a + 0.3, storms, 400)
    assert np.allclose(draws[:, 5], 0.0)
    # Whole storms: a draw is both of them, or the first twice, or the second twice.
    values = set(np.round(np.log(draws[:, 0]), 12))
    assert values == {-0.1, 0.0, 0.1}
    assert np.allclose(np.log(draws[:, 1]) - np.log(draws[:, 0]), 0.3)
    # Each storm with one value: a draw has no spread, unless it holds both storms.
    assert np.allclose(draws[np.abs(np.log(draws[:, 0])) > 1e-9, 3], 0.0)
    assert np.allclose(draws[np.abs(np.log(draws[:, 0])) < 1e-9, 3], 0.1)
    # The distance from 1, and its difference, with |0.1| - |0.4|, 0 - 0.3 and |-0.1| - |0.2|.
    assert set(np.round(draws[:, 2], 12)) == {-0.3, -0.1}


def random_effects(
    rho: float, k: int, m: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, math.sqrt(rho), k)
    e = rng.normal(0.0, math.sqrt(1.0 - rho), (k, m))
    ids = np.repeat(np.arange(k), m)
    return (u[:, None] + e).ravel(), ids


def test_enumeration_is_the_exact_law_for_few_clusters() -> None:
    means = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    sums = means[:, None] * 10.0
    counts = np.full(means.size, 10.0)
    draws = tool.bootstrap_means(sums, counts, 0)
    assert draws.shape[0] == means.size**means.size
    assert math.isclose(draws.mean(), means.mean(), rel_tol=1e-12)
    expected = ((means - means.mean()) ** 2).sum() / means.size**2
    assert math.isclose(draws.var(), expected, rel_tol=1e-12)


def test_monte_carlo_reaches_the_same_law_within_its_own_error() -> None:
    rng = np.random.default_rng(3)
    means = rng.normal(size=40)
    sums = means[:, None] * 5.0
    counts = np.full(means.size, 5.0)
    n = 20000
    draws = tool.bootstrap_means(sums, counts, n)
    expected = ((means - means.mean()) ** 2).sum() / means.size**2
    # The sample variance of n near-normal draws has relative standard error sqrt(2/(n - 1)).
    assert abs(draws.var() / expected - 1.0) < 4.0 * math.sqrt(2.0 / (n - 1))


def test_a_known_rho_is_recovered_within_fishers_error() -> None:
    for rho, k, m in ((0.05, 400, 50), (0.2, 300, 25), (0.5, 200, 10)):
        values, ids = random_effects(rho, k, m, seed=int(rho * 100))
        found = tool.anova_rho(values, ids)
        error = math.sqrt(tool.fisher_rho_variance(rho, m, k))
        assert abs(found - rho) < 4.0 * error, (rho, found, error)


def test_the_cluster_bootstrap_widens_by_the_design_effect() -> None:
    for rho, k, m in ((0.05, 400, 50), (0.2, 300, 25)):
        values, ids = random_effects(rho, k, m, seed=11 + int(rho * 100))
        n = 4000
        by_cluster = tool.bootstrap_means(*tool.cluster_sums(values[:, None], ids), n)
        by_row = tool.bootstrap_means(
            *tool.cluster_sums(values[:, None], np.arange(values.size)), n
        )
        measured = by_cluster.var() / by_row.var()
        expected = 1.0 + (m - 1) * rho
        # The numerator's sampling error, over k clusters, and the two Monte-Carlo errors, added in quadrature.
        error = expected * math.sqrt(2.0 / (k - 1) + 2.0 / (n - 1) + 2.0 / (n - 1))
        assert abs(measured - expected) < 4.0 * error, (measured, expected, error)


def test_blocks_of_an_ar1_series_reach_its_long_run_factor() -> None:
    phi, n, block = 0.9, 200_000, 500
    rng = np.random.default_rng(5)
    noise = rng.normal(size=n)
    x = np.empty(n)
    x[0] = noise[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + math.sqrt(1.0 - phi**2) * noise[i]
    ids = np.arange(n) // block
    draws = 4000
    by_block = tool.bootstrap_means(*tool.cluster_sums(x[:, None], ids), draws)
    by_row = tool.bootstrap_means(*tool.cluster_sums(x[:, None], np.arange(n)), draws)
    closed = (1.0 + phi) / (1.0 - phi) - 2.0 * phi * (1.0 - phi**block) / (
        block * (1.0 - phi) ** 2
    )
    k = n // block
    error = closed * math.sqrt(2.0 / (k - 1) + 4.0 / (draws - 1))
    assert abs(by_block.var() / by_row.var() - closed) < 4.0 * error


def test_pairing_keeps_what_two_models_share() -> None:
    k, m = 60, 40
    rng = np.random.default_rng(9)
    shared = rng.normal(0.0, 5.0, k)[:, None] * np.ones((k, m))
    ids = np.repeat(np.arange(k), m)
    a = (shared + rng.normal(0.0, 1.0, (k, m))).ravel()
    b = (shared + 0.5 + rng.normal(0.0, 1.0, (k, m))).ravel()
    draws = tool.bootstrap_means(*tool.cluster_sums(np.column_stack([a, b]), ids), 4000)
    a_low, a_high = tool.interval(draws[:, 0])
    b_low, b_high = tool.interval(draws[:, 1])
    assert a_low < b_high and b_low < a_high, "the two intervals are meant to overlap"
    low, high = tool.interval(draws[:, 0] - draws[:, 1])
    assert high < 0.0, (low, high)
    assert low < -0.5 < high


def test_the_row_level_is_not_widened_by_cluster_labels() -> None:
    t = np.array(
        ["2010-01-01T00", "2010-01-01T12", "2010-02-15T00", "2011-03-01T00"],
        dtype="datetime64[ns]",
    )
    labels = tool.levels(t, np.array(["A", "A", "B", "A"]), np.array([-1, 0, 0, -1]))
    assert labels["row"].tolist() == [0, 1, 2, 3]
    assert labels["day"].tolist() == [0, 0, 1, 2]
    assert labels["year"].tolist() == [0, 0, 0, 1]
    # A storm spanning two blocks is one cluster, and the rows outside it stay in their blocks.
    assert labels["storm"][1] == labels["storm"][2]
    assert labels["storm"][0] != labels["storm"][1]
    assert labels["storm"][0] != labels["storm"][3]
    assert labels["family"].tolist() == [0, 0, 1, 0]


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
