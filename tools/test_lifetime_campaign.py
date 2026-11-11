#!/usr/bin/env python3
"""The campaign's arithmetic against its derivations.

    python3 tools/test_lifetime_campaign.py

Pins each value to what PARI/GP printed from `derivations/tle_kozai_recovery.gp` and
`derivations/split_conformal_coverage.gp`, and checks the parsers on hand-made lines. Standard library
only, like the campaign.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import math
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "lifetime_campaign", Path(__file__).with_name("lifetime-campaign.py")
)
assert spec is not None and spec.loader is not None
campaign = importlib.util.module_from_spec(spec)
sys.modules["lifetime_campaign"] = campaign
spec.loader.exec_module(campaign)


GP_A = 6795375.473460956


def test_recovery_matches_the_second_engine() -> None:
    got = campaign.mean_motion_to_semi_major_axis(15.5, 0.001, math.radians(51.6))
    assert abs(got - GP_A) < 1e-7, got


def test_bands_match_the_second_engine() -> None:
    for n, m, lo, hi in (
        (20, 9, 6, 9),
        (30, 19, 14, 19),
        (50, 29, 22, 29),
        (100, 49, 40, 48),
        (100, 99, 82, 95),
    ):
        assert campaign.beta_binomial_band(m, n, 0.1) == (lo, hi), (n, m)


def test_tle_epochs_and_decay_dates() -> None:
    utc = dt.UTC
    assert (
        campaign.tle_epoch("90001.50000000")
        == dt.datetime(1990, 1, 1, 12, tzinfo=utc).timestamp()
    )
    assert (
        campaign.tle_epoch("05032.00000000")
        == dt.datetime(2005, 2, 1, tzinfo=utc).timestamp()
    )
    assert (
        campaign.decay_time("1994 Mar  7")
        == dt.datetime(1994, 3, 7, 12, tzinfo=utc).timestamp()
    )
    assert (
        campaign.decay_time("1994 Mar  7 0559?")
        == dt.datetime(1994, 3, 7, 5, 59, tzinfo=utc).timestamp()
    )
    assert campaign.decay_time("-") is None
    assert campaign.decay_time("1994 Mar") is None


def test_element_lines_parse_and_a_bad_line_is_skipped() -> None:
    text = (
        "1 00028U 60002 1  60092.49305556 0.00000000 +00000-0 +00000-0 0    17\n"
        "2 00028  48.3000 256.6100 0473631 241.3700 180.4578 15.50819974     4\n"
        "3 00028     1 UNK       SGP4 TEME UTC  Earth                          misc/bbels\n"
        "1 00028U 60002 1  60093.49305556 0.00000000 +00000-0 +00000-0 0    17\n"
        "2 00028  48.3000 256.6100 04x3631 241.3700 180.4578 15.50819974     4\n"
    )
    sets = campaign.read_elements(text)
    assert len(sets) == 1
    epoch, state = sets[0]
    assert epoch == campaign.tle_epoch("60092.49305556")
    assert abs(state[1] - 0.0473631) < 1e-12
    assert abs(state[2] - math.radians(48.3)) < 1e-12


def test_quantile_is_nearest_rank_and_censored_sorts_last() -> None:
    values = [1.0, 2.0, 3.0, 4.0, math.inf]
    assert campaign.quantile(values, 0.5) == 3.0
    assert campaign.quantile(values, 0.05) == 1.0
    assert campaign.quantile(values, 0.95) == math.inf


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
