#!/usr/bin/env python3
"""Independent verification of parse_icgem_time_variable / TimeVariableCoefficient::evaluate against
the real EIGEN-6S2 time-variable gravity model, and the source of the baked test constants.

The Rust parser reads the ICGEM `gfc` temporal format (gfct / trnd / acos / asin) as a piecewise-linear
secular part plus periodic terms. This script re-implements that formula independently, in Python,
straight from the ICGEM definition:

    G(t) = G(t0_seg) + trnd_seg * (t - t0_seg)
           + sum_k [ acos_k * cos(2*pi*(t - t0_k)/T_k) + asin_k * sin(2*pi*(t - t0_k)/T_k) ]

where the secular part is the segment [t0_seg, t1_seg) that brackets t (the low-degree zonals of
EIGEN-6S / ITSG-Grace are given as a *sequence* of annual segments), and each periodic term carries its
own reference epoch t0_k and period T_k. Epochs are YYYYMMDD.dd; the day->year-fraction conversion uses
the real calendar (leap-aware), matching pyshtools' shio.icgem reader and the Rust `decimal_year`.

Source model (not vendored — large external download):
    EIGEN-6S2, Foerste et al. 2016, GFZ Potsdam / GRGS Toulouse, doi:10.5880/icgem.2016.008
    http://icgem.gfz-potsdam.de/  ->  getmodel/gfc/<hash>/EIGEN-6S2.gfc   (degree/order 260, ~6 MB)

A verbatim (2,0)/(3,0) excerpt of that file is kept at
    crates/spacelab-propagation/tests/fixtures/eigen6s2_lowdeg_timevariable.gfc
and the parser reproduces the full file on it exactly.

Usage:
    python derivations/icgem_timevariable_verify.py [EIGEN-6S2.gfc]
With no argument it uses the vendored excerpt (which reproduces the full model on (2,0)/(3,0)).
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta


def year_fraction(yyyymmdd: str | float) -> float:
    """YYYYMMDD[.dd] -> decimal year, leap-aware real calendar (matches Rust `decimal_year`)."""
    s = str(yyyymmdd)
    if "." in s:
        s, frac = s.split(".")
        residual = float("0." + frac)
    else:
        residual = 0.0
    d = datetime.strptime(s, "%Y%m%d") + timedelta(days=residual)
    y0 = datetime(d.year, 1, 1)
    y1 = datetime(d.year + 1, 1, 1)
    return d.year + (d - y0) / (y1 - y0)


def load(path: str) -> dict[tuple[int, int], dict]:
    """Parse the time-variable lines into {(n,m): {'seg': [[t0,t1,c,trend]], 'per': [[t0,T,acos,asin]]}}."""
    coeffs: dict[tuple[int, int], dict] = {}

    def num(x: str) -> float:
        return float(x.replace("D", "E").replace("d", "e"))

    with open(path) as fh:
        text = fh.read()
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5 or p[0] not in ("gfct", "trnd", "dot", "acos", "asin"):
            continue
        n, m = int(p[1]), int(p[2])
        d = coeffs.setdefault((n, m), {"seg": [], "per": []})
        t0 = year_fraction(p[7]) if len(p) > 7 else None
        t1 = year_fraction(p[8]) if len(p) > 8 else None
        if p[0] == "gfct":
            d["seg"].append([t0, t1, num(p[3]), 0.0])
        elif p[0] in ("trnd", "dot"):
            for s in d["seg"]:
                if s[0] is not None and abs(s[0] - t0) < 1e-6:
                    s[3] = num(p[3])
        else:  # acos / asin
            T = num(p[-1])
            entry = next((e for e in d["per"] if abs(e[1] - T) < 1e-9 and e[0] == t0), None)
            if entry is None:
                entry = [t0, T, 0.0, 0.0]
                d["per"].append(entry)
            entry[2 if p[0] == "acos" else 3] = num(p[3])
    return coeffs


def evaluate(coeffs: dict, n: int, m: int, epoch_yyyymmdd: int) -> tuple[float, float]:
    """(C_nm(t), decimal_year(t)) at the given YYYYMMDD epoch."""
    t = year_fraction(str(epoch_yyyymmdd))
    d = coeffs[(n, m)]
    seg = next((s for s in d["seg"] if s[0] <= t < s[1]), None)
    if seg is None:  # extrapolate by the nearest end segment
        below = [s for s in d["seg"] if s[0] <= t]
        seg = below[-1] if below else d["seg"][0]
    c = seg[2] + seg[3] * (t - seg[0])
    for t0, T, acos, asin in d["per"]:
        c += acos * math.cos(2 * math.pi * (t - t0) / T) + asin * math.sin(2 * math.pi * (t - t0) / T)
    return c, t


# Epochs falling in three different (2,0) piecewise segments, plus a (3,0) case.
CASES = [(2, 0, 19950615), (2, 0, 20050701), (2, 0, 20120901), (3, 0, 20080315)]


def main() -> None:
    default = os.path.join(
        os.path.dirname(__file__),
        "..",
        "crates",
        "spacelab-propagation",
        "tests",
        "fixtures",
        "eigen6s2_lowdeg_timevariable.gfc",
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default
    coeffs = load(path)
    print(f"# {path}")
    print("# these are the constants baked into time_variable_gravity.rs / smoke.py:")
    for n, m, ep in CASES:
        c, t = evaluate(coeffs, n, m, ep)
        print(f"({n}, {m}, {t!r}, {c!r}),")


if __name__ == "__main__":
    main()
