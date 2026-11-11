#!/usr/bin/env python3
"""The harness refuses a leak: `tools/evaluate-sdtm.py`'s guards against deliberately leaked folds.

    uv run --no-project --with numpy --with pandas --with scipy python -I tools/test_evaluate_sdtm.py

The tests build a small fake corpus and hand each guard the fold it must refuse. They also rebuild the
coupling map with and without an extreme test hour, to show that the hour reaches the leaked map and does
not reach the fold's own.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "evaluate_sdtm", HERE / "evaluate-sdtm.py"
)
assert spec is not None and spec.loader is not None
harness = importlib.util.module_from_spec(spec)
sys.modules["evaluate_sdtm"] = harness
spec.loader.exec_module(harness)

T = np.array(pd.date_range("2010-01-01", periods=240, freq="D"), dtype="datetime64[ns]")
FAMILY = np.array(["CHAMP", "GRACE", "Swarm"] * 80)
ORIGIN = {"name": "o", "kind": "origin", "cut": "2010-05-01", "until": "2010-07-01"}
FROZEN = {**ORIGIN, "name": "z", "kind": "frozen", "until": "2011-01-01"}
STORM = {"name": "s", "kind": "storm", "start": "2010-03-01", "end": "2010-03-11"}
JOINT = {**STORM, "name": "j", "kind": "joint", "family": "GRACE"}
FOLD = {"name": "f", "kind": "family", "family": "Swarm"}


def refused(split, train, test) -> bool:
    try:
        harness.guard(T, FAMILY, split, train, test)
    except harness.LeakError:
        return True
    return False


def test_each_split_passes_its_own_guard() -> None:
    for split in (ORIGIN, FROZEN, STORM, JOINT, FOLD):
        train, test = harness.masks(T, FAMILY, split)
        assert train.size and test.size, split
        assert not refused(split, train, test), split


def test_origin_scores_after_the_cut_and_trains_before_it() -> None:
    train, test = harness.masks(T, FAMILY, ORIGIN)
    assert T[train].max() < T[test].min()
    assert T[test].max() < np.datetime64("2010-07-01")


def test_a_future_row_is_refused() -> None:
    for split in (ORIGIN, FROZEN):
        train, test = harness.masks(T, FAMILY, split)
        assert refused(split, np.append(train, test[0]), test[1:]), split


def test_frozen_scores_every_row_after_its_cut() -> None:
    train, test = harness.masks(T, FAMILY, FROZEN)
    cut = harness.stamp(FROZEN["cut"])
    assert np.array_equal(T[test], T[np.searchsorted(T, cut) :])
    assert T[train].max() < cut


def test_a_row_in_both_sets_is_refused() -> None:
    train, test = harness.masks(T, FAMILY, ORIGIN)
    assert refused(ORIGIN, np.append(train, test[0]), test)


def test_a_row_of_the_storm_is_refused() -> None:
    train, test = harness.masks(T, FAMILY, STORM)
    assert refused(STORM, np.append(train, test[0]), test[1:])


def test_the_held_out_family_is_refused() -> None:
    train, test = harness.masks(T, FAMILY, FOLD)
    assert refused(FOLD, np.append(train, test[0]), test[1:])


def test_joint_holds_out_the_family_and_the_storm_both() -> None:
    train, test = harness.masks(T, FAMILY, JOINT)
    assert (FAMILY[test] == "GRACE").all()
    assert not (FAMILY[train] == "GRACE").any()
    inside = (T[train] >= np.datetime64("2010-03-01")) & (
        T[train] < np.datetime64("2010-03-11")
    )
    assert not inside.any()
    outside_family = np.flatnonzero(FAMILY != "GRACE")
    in_storm = outside_family[
        (T[outside_family] >= np.datetime64("2010-03-01"))
        & (T[outside_family] < np.datetime64("2010-03-11"))
    ]
    assert refused(JOINT, np.append(train, in_storm[0]), test)


def test_a_map_on_a_test_hour_is_refused() -> None:
    late = np.array(
        [np.datetime64("2010-04-01"), np.datetime64("2010-05-02")],
        dtype="datetime64[ns]",
    )
    try:
        harness.guard_map(late, ORIGIN)
    except harness.LeakError:
        return
    raise AssertionError("the guard let a test hour through")


def test_an_extreme_test_hour_moves_a_leaked_map_and_not_the_folds_own() -> None:
    epoch = pd.Timestamp("2010-01-01")
    n = 24 * 180
    rng = np.random.default_rng(1)
    hourly = rng.choice(np.array([0.0, 2.0, 4.0, 7.0, 15.0, 27.0, 48.0, 80.0]), size=n)
    coupling = 100.0 * (hourly + rng.uniform(0.0, 1.0, n))
    hours = np.asarray(
        epoch + pd.to_timedelta(np.arange(n), unit="h"), dtype="datetime64[ns]"
    )
    cut = harness.stamp(ORIGIN["cut"])
    test_hours = hours >= cut
    assert test_hours.any()
    changed = coupling.copy()
    changed[test_hours] *= 50.0

    own_before = harness.fold_drive(coupling, hourly, epoch, ORIGIN)
    own_after = harness.fold_drive(changed, hourly, epoch, ORIGIN)
    # The drive on training hours reads a map that never saw the test hours, so it is unchanged.
    assert np.array_equal(own_before[~test_hours], own_after[~test_hours])

    def leaked(values: np.ndarray) -> np.ndarray:
        return harness.fit.coupling_map(values, hourly, pd.DatetimeIndex(hours))(values)

    assert not np.array_equal(
        leaked(coupling)[~test_hours], leaked(changed)[~test_hours]
    )


def test_heating_alone_is_the_shipped_state_without_cooling_and_the_kernel_without_its_onset() -> (
    None
):
    fit, g = harness.fit, harness.g
    ap = np.random.default_rng(5).gamma(0.6, 25.0, 4000)
    cadence, heating = 10_800.0, 17.4
    alone = fit.cooling_grid(ap, heating, 32.6, np.inf, cadence, 0)
    decay = np.exp(-cadence / 3600.0 / heating)
    y, explicit = 0.0, np.empty(ap.size)
    for i, u in enumerate(ap):
        y = u + (y - u) * decay
        explicit[i] = y
    assert np.allclose(alone, explicit, rtol=1e-12, atol=0.0)
    # `storm_grid` starts in steady state and `cooling_grid` from rest: compare after 300 steps, 50 heating times.
    gap = [
        np.abs(g.storm_grid(ap, onset, heating * 3600.0, cadence) - alone)[300:].max()
        / alone[300:].max()
        for onset in (1.0, 1.0e-3)
    ]
    assert gap[1] < 1e-7 and 500.0 < gap[0] / gap[1] < 2000.0, gap


def test_the_rungs_pack_their_kernel_slots_and_refuse_a_state_they_are_the_limit_of() -> (
    None
):
    fit = harness.fit
    for limit, n_kernel in ((None, 2), ("first-order", 1), ("instant", 0)):
        pk = fit.Packing(
            n_c=7, scaled=("SA", "SB"), free_sigma=True, cooling=False, limit=limit
        )
        assert pk.n_kernel == n_kernel and pk.i_boundary == 7 + n_kernel
        used = [j for j in (pk.i_onset, pk.i_gap) if j >= 0]
        assert used == list(range(7, 7 + n_kernel)), (limit, used)
        assert pk.size == 7 + n_kernel + 1 + 2 + 1
    try:
        fit.CorpusModel({}, np.array([]), (), (), 0.0, None, True, limit="instant")
    except ValueError:
        pass
    else:
        raise AssertionError("a rung with cooling was accepted")


def test_no_scales_is_the_shipped_form_on_one_level() -> None:
    fields = harness.FIELDS
    assert "scales" not in fields["default"]
    assert {k: v for k, v in fields["no_scales"].items() if k != "scales"} == fields[
        "default"
    ]
    assert fields["no_scales"]["scales"] is False
    pk = harness.fit.Packing(n_c=7, scaled=(), free_sigma=True, cooling=True)
    assert pk.i_scale == pk.i_boundary + 1 and pk.size == pk.i_scale + 1


def test_the_composition_rival_takes_three_slots_last_and_moves_no_other() -> None:
    fields, fit = harness.FIELDS, harness.fit
    assert {k: v for k, v in fields["composition"].items() if k != "composition"} == (
        fields["default"]
    )
    common = {
        "n_c": 7,
        "scaled": ("SA",),
        "free_sigma": True,
        "cooling": True,
        "free_excess": True,
    }
    shipped = fit.Packing(**common, free_gain=True, free_knee=True)
    rival = fit.Packing(**common, free_gain=True, free_knee=True, composition=True)
    assert (rival.i_gain, rival.i_knee, rival.i_excess) == (
        shipped.i_gain,
        shipped.i_knee,
        shipped.i_excess,
    )
    assert rival.size == shipped.size + 3 and rival.i_oxygen == shipped.size
    assert shipped.i_oxygen == -1


def test_the_delta_and_linear_rivals_pack_their_slots_and_refuse_each_other() -> None:
    fields, fit = harness.FIELDS, harness.fit
    for name in ("delta", "linear"):
        assert {k: v for k, v in fields[name].items() if k != name} == fields["default"]
    common = {"n_c": 7, "scaled": ("SA",), "free_sigma": True, "cooling": True}
    shipped = fit.Packing(**common, free_gain=True)
    both = fit.Packing(**common, free_gain=True, composition=True, delta=True)
    assert both.i_gain == shipped.i_gain and both.size == shipped.size + 4
    assert (both.i_oxygen, both.i_delta) == (shipped.size, shipped.size + 3)
    linear = fit.Packing(**common, free_gain=True, linear=True)
    assert linear.size == shipped.size and linear.i_cool == shipped.i_cool
    for wrong in ({"delta": True, "linear": True}, {"delta": True, "mix": 0.5}):
        try:
            fit.CorpusModel({}, np.array([]), (), (), 0.0, None, True, **wrong)
        except ValueError:
            continue
        raise AssertionError(f"{wrong} was accepted")


def test_the_delta_cooling_is_the_shipped_one_at_zero_and_settles_where_maxima_says() -> (
    None
):
    fit = harness.fit
    ap = np.random.default_rng(7).gamma(0.6, 25.0, 3000)
    shipped = fit.cooling_grid(ap, 6.0, 20.0, 300.0, 10_800.0, 0)
    assert np.array_equal(
        fit.cooling_grid(ap, 6.0, 20.0, 300.0, 10_800.0, 0, delta=0.0), shipped
    )
    # `sdtm_overcooling.mac` A4: under a held drive h the state settles at h (Hs - delta)/(Hs + h).
    h, hs, delta = 120.0, 250.0, 40.0
    held = fit.cooling_grid(np.full(2000, h), 9.0, 30.0, hs, 10_800.0, 0, delta=delta)
    assert np.isclose(held[-1], h * (hs - delta) / (hs + h), rtol=1e-12, atol=0.0)
    # A storm, then quiet under a still-high N: below zero while delta N > H Hs.
    storm = np.r_[np.full(8, 300.0), np.full(400, 5.0)]
    after = fit.cooling_grid(storm, 9.0, 30.0, hs, 10_800.0, 0, delta=delta)
    assert after.min() < 0.0 < after[-1]


def test_the_linear_form_is_exact_settles_linearly_and_undershoots() -> None:
    fit = harness.fit
    tc, tn, beta, q = 9.0, 30.0, 0.35, 200.0
    held = fit.linear_grid(np.full(3000, q), tc, tn, beta, 3600.0, 0)
    assert np.isclose(held[-1], (1.0 - beta) * q, rtol=1e-12, atol=0.0)
    # `sdtm_overcooling.mac` B4: after the drive stops, D e^(t/tn) -> -beta Q tn/(tn - tc).
    out = fit.linear_grid(
        np.r_[np.full(3000, q), np.zeros(600)], tc, tn, beta, 3600.0, 0
    )
    t = np.arange(1, 601, dtype=float)
    assert out[3000:].min() < 0.0
    assert np.isclose(
        out[-1] * np.exp(t[-1] / tn), -beta * q * tn / (tn - tc), rtol=1e-9
    )
    # Exact for a held drive: three 1 h steps are one 3 h step.
    ap = np.random.default_rng(3).gamma(0.6, 25.0, 500)
    coarse = fit.linear_grid(ap, tc, tn, beta, 10_800.0, 0)
    fine = fit.linear_grid(np.repeat(ap, 3), tc, tn, beta, 3600.0, 0)
    assert np.allclose(fine[2::3], coarse, rtol=1e-12, atol=1e-10)
    # Where the two times meet, the step is the limit of the steps beside it.
    met = fit.linear_grid(ap, 12.0, 12.0, beta, 10_800.0, 0)
    near = fit.linear_grid(ap, 12.0, 12.0 * (1.0 + 1e-7), beta, 10_800.0, 0)
    assert np.allclose(met, near, rtol=1e-5, atol=1e-9)


def test_the_corpus_variant_field_is_the_shipped_form_on_another_corpus() -> None:
    fields = harness.FIELDS
    assert {k: v for k, v in fields["gc_v02"].items() if k != "corpus"} == fields[
        "default"
    ]
    assert "corpus" not in fields["default"]


def test_the_oxygen_factor_moves_ln_rho_by_the_oxygen_mass_fraction() -> None:
    g = harness.g
    # `derivations/sdtm_composition.mac`'s shipped boundary at T_inf 700 and 1000 K, 200 km.
    sigma, radius, base = 3.341439818644104e-5, 6_356_766.0, 120_000.0
    zeta = np.full(2, (200_000.0 - base) * (radius + base) / (radius + 200_000.0))
    t_inf = np.array([700.0, 1000.0])
    w = np.array([0.38866192176198305, 0.2908741220196561])
    d_ln = np.array([0.002326011527589191, 0.0012183443381514763])
    shipped, slope = g.profile(t_inf, sigma, zeta)
    assert np.allclose(slope, d_ln, rtol=1e-12, atol=0.0)
    same = g.profile(t_inf, sigma, zeta, np.zeros(2))
    assert np.array_equal(same[0], shipped) and np.array_equal(same[1], slope)
    c = np.log(1.040 / 0.487542453257362)
    moved, moved_slope = g.profile(t_inf, sigma, zeta, np.full(2, c))
    assert np.allclose(moved - shipped, np.log1p(w * np.expm1(c)), rtol=1e-12, atol=0.0)
    h = 1.0e-3
    central = (
        g.profile(t_inf + h, sigma, zeta, np.full(2, c))[0]
        - g.profile(t_inf - h, sigma, zeta, np.full(2, c))[0]
    ) / (2.0 * h)
    assert np.allclose(moved_slope, central, rtol=1e-8, atol=0.0)


def test_the_cap_sets_of_the_lowest_rungs_are_nested_in_the_shipped_one() -> None:
    g = harness.g
    k = (9 + 1) ** 2
    shipped, harmonics, bates = (
        g.harmonic_mask(g.CAPS[name], k).reshape(g.DRIVERS, k)
        for name in (harness.FORM["caps"], "harmonics", "bates")
    )
    storm = harness.fit.STORM_COLUMNS
    assert not harmonics[storm].any() and np.array_equal(
        np.delete(harmonics, storm, axis=0), np.delete(shipped, storm, axis=0)
    )
    assert (bates <= harmonics).all() and bates.sum() == 4 and bates[:4, 0].all()


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
