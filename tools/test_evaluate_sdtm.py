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
    for split in (ORIGIN, STORM, JOINT, FOLD):
        train, test = harness.masks(T, FAMILY, split)
        assert train.size and test.size, split
        assert not refused(split, train, test), split


def test_origin_scores_after_the_cut_and_trains_before_it() -> None:
    train, test = harness.masks(T, FAMILY, ORIGIN)
    assert T[train].max() < T[test].min()
    assert T[test].max() < np.datetime64("2010-07-01")


def test_a_future_row_is_refused() -> None:
    train, test = harness.masks(T, FAMILY, ORIGIN)
    assert refused(ORIGIN, np.append(train, test[0]), test[1:])


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


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
