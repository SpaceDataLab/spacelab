#!/usr/bin/env python3
"""The blind runner's amended rules (`docs/sdtm1-blind-protocol.md` §9, `plans/18` Ф30.N).

G3: the rule's quadrature against `derivations/group_sequential_obf.gp`'s 30-digit values. S2: the
rule's level under the null; "holds" within its share under the skewed nulls the 2020 cut measured
(§9 A2, Scope); the refusal of the frozen span; no word below the cluster count. Each S2 check is also
run on a planted defect, and must fail on it (`plans/18` rule 16).

uv run --project tools --locked python -I tools/test_score_blind.py
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import tempfile

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent


def load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tool = load("score_blind", HERE / "score-blind.py")

# `derivations/group_sequential_obf.gp`, 30 digits.
PARI_SPEND = (
    0.000518330932018486641630998891268,
    0.0143174725366206260938723510037,
    0.0499923214783315940101841310531,
)
PARI_LEVEL_ROUNDED_DOWN = 0.0500120045484412693393725711423
PARI_LEVEL_TENTH = 0.0554298071494329898970189257448
# The null-level test's replicates a cell: a Monte Carlo half-width of about 0.003 at 0.05.
REPS = 20_000
# §9 A2, Scope (2026-11-25): the skewness of B5's cluster means on the 2020 cut at the upper ends of
# their intervals, the storm level's for the clusters' draws and the day level's for the days'
# (`tools/score-blind.py shape`).
B5_SKEW = (0.59, 0.16)


def test_the_quadrature_agrees_with_the_derivation() -> None:
    for mine, pari in zip(tool.spend(), PARI_SPEND, strict=True):
        assert abs(mine - pari) < 1e-11, (mine, pari)
    equal = [1.0, 2.0, 3.0]
    down = tool.crossing([3.471, 2.454, 2.004], equal)
    assert abs(down - PARI_LEVEL_ROUNDED_DOWN) < 1e-11, down
    tenth = tool.crossing(list(tool.REGISTERED_BOUNDS), [0.1, 0.3, 1.0])
    assert abs(tenth - PARI_LEVEL_TENTH) < 1e-11, tenth


def test_the_bounds_are_the_registered_ones_at_equal_counts() -> None:
    for n in (10, 12, 14):
        assert tool.z_bounds([n, 2 * n, 3 * n]) == list(tool.REGISTERED_BOUNDS)


def test_the_bounds_follow_the_realised_counts() -> None:
    # Fixed bounds at unequal fractions exceed the level; the rule's bounds spend the schedule.
    assert tool.crossing(list(tool.REGISTERED_BOUNDS), [10.0, 30.0, 100.0]) > 0.055
    for counts in ([10, 30, 100], [10, 24, 38], [14, 24, 34], [14, 28, 30]):
        bounds = tool.z_bounds(counts)
        level = tool.crossing(bounds, [float(n) for n in counts])
        assert tool.spend()[-1] - 1e-4 < level <= tool.spend()[-1], (counts, level)
    # A look that takes no decision leaves its spend to the next.
    skipped = tool.z_bounds([None, 24, 38])
    assert skipped[0] is None
    assert math.isclose(
        tool.crossing(skipped[1:], [24.0, 38.0]), tool.spend()[-1], abs_tol=1e-4
    )
    assert tool.crossing(skipped[1:], [24.0, 38.0]) <= tool.spend()[-1]


def test_cr2_is_the_one_sample_t_at_equal_counts() -> None:
    rng = np.random.default_rng(1)
    for k in (2, 5, 10, 40):
        means = rng.normal(size=k)
        w, c, nu = tool.cr2_design(np.full(k, 7.0))
        mean, se = tool.cr2_mean(means * 7.0, np.full(k, 7.0), w, c)
        assert math.isclose(nu, k - 1, rel_tol=1e-12)
        assert math.isclose(se, means.std(ddof=1) / math.sqrt(k), rel_tol=1e-12)
        assert math.isclose(mean, means.mean(), rel_tol=1e-12)
    # Unequal counts: E V = tau^2 sum w^2, the derivation's tr A.
    counts = np.array([1.0, 1.0, 20.0, 20.0, 3.0, 3.0, 27.0])
    w, c, nu = tool.cr2_design(counts)
    s2 = w @ w
    assert math.isclose(c @ (1.0 - 2.0 * w + s2), s2, rel_tol=1e-12)
    assert 1.0 < nu < counts.size - 1


def no_word_below_the_count() -> bool:
    """A look with fewer than ten clusters at a level takes no decision, however far its statistic;
    the last look then reads `not resolved`."""
    far = (-50.0, 1.0, 30.0)
    few = [
        {"day": (*far, 200 * k, 200.0 * k), "storm": (*far, 6 + k, 6.0 + k)}
        for k in (1, 2, 3)
    ]  # 7, 8 and 9 clusters at the storm level
    enough = {"day": (*far, 200, 200.0), "storm": (*far, 10, 10.0)}
    first = tool.sequential(few[:1])[0][0]
    last = tool.sequential(few)[2][0]
    decided = tool.sequential([enough])[0][0]
    return (
        first.startswith("no decision")
        and last == "not resolved"
        and decided == "holds"
    )


def test_no_word_below_the_cluster_count() -> None:
    assert no_word_below_the_count()


def test_no_word_below_the_cluster_count_fails_on_a_planted_defect() -> None:
    kept = tool.MIN_CLUSTERS
    tool.MIN_CLUSTERS = 9
    try:
        assert not no_word_below_the_count()
    finally:
        tool.MIN_CLUSTERS = kept


def refuses_the_frozen_span(refusal) -> bool:
    """`blind` refuses a row a millisecond before `FROZEN_END`, Swarm's 2026-03 among them, and takes
    one at it."""
    sat = np.array(["SA"])
    before = np.array([tool.FROZEN_END - np.timedelta64(1, "ms")])
    at = np.array([tool.FROZEN_END.astype("datetime64[ms]")])
    swarm = np.array([np.datetime64("2026-03-15T12:00", "ms")])
    return (
        refusal(before, sat) is not None
        and refusal(swarm, sat) is not None
        and refusal(at, sat) is None
    )


def test_blind_refuses_the_frozen_span() -> None:
    assert refuses_the_frozen_span(tool.blind_refusal)


def test_blind_refusal_fails_on_a_planted_defect() -> None:
    def a_day_late(t, sat):
        return tool.blind_refusal(t + np.timedelta64(1, "D"), sat)

    assert not refuses_the_frozen_span(a_day_late)


def test_gap_reads_only_its_month() -> None:
    inside = np.array([np.datetime64("2026-03-01T00:00", "ms")])
    end = np.array([np.datetime64("2026-03-31T23:59", "ms")])
    after = np.array([np.datetime64("2026-04-01T00:00", "ms")])
    assert tool.gap_refusal(inside, np.array(["SA"])) is None
    assert tool.gap_refusal(end, np.array(["SC"])) is None
    assert tool.gap_refusal(after, np.array(["SB"])) is not None
    assert tool.gap_refusal(inside, np.array(["GC"])) is not None


def test_a_look_is_complete_when_every_satellite_reaches_its_last_month() -> None:
    days = np.arange("2026-05-01", "2026-11-01", dtype="datetime64[D]")
    t = np.concatenate([days, days[days < np.datetime64("2026-10-01")]])
    sat = np.array(["SA"] * days.size + ["GC"] * (t.size - days.size))
    assert tool.complete_looks(t, sat) == 0
    assert tool.complete_looks(days, np.array(["SA"] * days.size)) == 1
    # GC never reaches October; once SA's rows run 183 days past the look, GC counts as ended.
    later = np.arange("2026-11-01", "2027-05-10", dtype="datetime64[D]")
    t = np.concatenate([t, later])
    sat = np.concatenate([sat, ["SA"] * later.size])
    assert tool.complete_looks(t, sat) == 2


def level_holds(law: str, cells) -> bool:
    """The rule's level under each null of `cells`, read with `law`, has a Monte Carlo interval, 95 %
    jointly over the cells, that reaches down to 0.05."""
    for per_look, shape, storm_sd, day_sd in cells:
        design = tool.null_design(per_look, 11)
        found = tool.null_level(design, shape, storm_sd, day_sd, REPS, 5)[law]
        low, _ = tool.mc_interval(found["holds"] + found["fails"], REPS, len(cells))
        if low > tool.LEVEL:
            return False
    return True


def test_the_rule_holds_its_level_under_the_null() -> None:
    cells = [
        (per_look, shape, 1.4, 0.8)
        for per_look in tool.NULL_SCENARIOS
        for shape in tool.GATED_SHAPES
    ]
    assert level_holds("student", cells)


def test_the_level_test_fails_on_a_planted_defect() -> None:
    # The statistics read against the normal bounds: the defect §9 A2 (b) removes.
    assert not level_holds("normal", [((10, 10, 10), "normal", 1.4, 0.8)])


def draws_have_their_skewness(draw) -> bool:
    """A draw of the skewed null has mean 0, variance 1 and the skewness it is asked for."""
    rng = np.random.default_rng(3)
    for gamma in (-1.5, 0.59, 2.0):
        x = draw(gamma, rng, 1_000_000)
        if abs(x.mean()) > 0.01 or abs(x.var() - 1.0) > 0.02:
            return False
        if abs(tool.skewness(x) - gamma) > 0.05 * max(1.0, abs(gamma)):
            return False
    return True


def test_the_skewed_draws_have_their_skewness() -> None:
    assert draws_have_their_skewness(tool.null_draw)
    x = np.array([0.0, 1.0, 1.0, 5.0, -2.0, 0.5])
    assert math.isclose(tool.skewness(x), stats.skew(x, bias=False), rel_tol=1e-12)


def test_the_skewness_test_fails_on_a_planted_defect() -> None:
    # Gamma(2 / gamma^2) for Gamma(4 / gamma^2): the skewness comes out sqrt(2) gamma.
    def misshaped(gamma, rng, size):
        k = 2.0 / gamma**2
        return (
            math.copysign(1.0, gamma) * (rng.standard_gamma(k, size) - k) / math.sqrt(k)
        )

    assert not draws_have_their_skewness(misshaped)


def holds_within_its_share(shape, factor: float) -> bool:
    """Under the mean-zero null of `shape`, "holds" read at `factor` times the registered bound has a
    Monte Carlo interval, 95 % jointly over `NULL_SCENARIOS`, that reaches down to its share,
    `LEVEL / 2`, in every calendar, at a storm's 1.4 and a day's 0.8."""
    for per_look in tool.NULL_SCENARIOS:
        design = tool.null_design(per_look, 11)
        found = tool.null_level(design, shape, 1.4, 0.8, REPS, 5, holds_factor=factor)
        rule = found["student"]
        assert rule["holds"] == int((rule["margin"] >= factor).sum())
        low, _ = tool.mc_interval(rule["holds"], REPS, len(tool.NULL_SCENARIOS))
        if low > tool.LEVEL / 2:
            return False
    return True


def test_b5_reads_holds_only_past_its_factor() -> None:
    # One look at 30 degrees of freedom, every level's statistic 1.1 times the bound on either side.
    bound = tool.t_bound(tool.REGISTERED_BOUNDS[0], 30.0)
    assert 1.0 < 1.1 < tool.B5_HOLDS_FACTOR

    def look(t: float) -> dict:
        return {
            "day": (t, 1.0, 30.0, 200, 200.0),
            "storm": (t, 1.0, 30.0, 10, 10.0),
        }

    low, high = look(-1.1 * bound), look(1.1 * bound)
    assert tool.sequential([low])[0][0] == "holds"
    assert tool.sequential([low], tool.B5_HOLDS_FACTOR)[0][0] == "continue"
    assert tool.sequential([high], tool.B5_HOLDS_FACTOR)[0][0] == "fails"


def test_b5_holds_its_share_under_its_skewed_null() -> None:
    assert holds_within_its_share(B5_SKEW, tool.B5_HOLDS_FACTOR)


def test_b5_share_fails_on_the_unamended_rule() -> None:
    # The bound of "holds" as registered, before the Scope's amendment of 2026-11-25.
    assert not holds_within_its_share(B5_SKEW, 1.0)




def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, test in tests:
        test()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
