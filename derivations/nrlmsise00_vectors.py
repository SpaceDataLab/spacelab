#!/usr/bin/env python
"""Emit full-precision NRLMSISE-00 reference vectors from the reference C implementation.

The published verification set (Picone et al., the 17 cases the model's own `nrlmsise-00_test.c`
prints) is quoted to seven significant digits, which caps any assertion against it at ~1e-5
relative. That is four orders coarser than the agreement two IEEE-754 implementations of the same
algorithm actually reach, and the gap is where wrong arithmetic hides: a term of `globe7` worth
1e-6 of the answer is invisible at 1e-5 and glaring at 1e-12.

This script asks the same C — Dominik Brodowski's public-domain port, the source
`derivations/nrlmsise00_tables.py` already reads the coefficient tables from, wrapped by the
`nrlmsise00` PyPI package — for `%.17e` output over a sweep built to move every term:

* **Altitude is stratified by regime**, not drawn uniformly. The model is three models stitched
  together — `densm`/`glob7s` below 72.5 km, the mesospheric blend to 120 km, `gts7`'s diffusive
  equilibrium above — and a uniform draw over 0..1000 km would put nine cases in ten in the
  thermosphere and leave the spline machinery cold.
* **Everything else is a Halton sequence** on the first eight primes. Deterministic, reproducible,
  and — unlike the published set, which varies one coordinate at a time off a single base point —
  it moves the inputs *together*, which is the only way a cross term is distinguishable from the
  main effects it multiplies.
* **The switch array is swept, including switches turned _off_.** The documented standard set is
  one path; `sw[9] = -1` takes the 3-hourly `ap_array` instead of the daily scalar; a `2` means
  "main effect off, cross terms on". A **`0`** is the one that matters most for verification: the
  internal `swc[i]` weights multiply whole groups of terms, and while every `swc[i]` is `1` the
  multiplication is indistinguishable from a division. Every switch is therefore driven to `0` in
  some case, and half of those cases also carry `sw[9] = -1`, so "off" co-occurs with the
  `ap_array` branch as well as with the daily-`ap` one.

Both entry points are emitted. `gtd7d` differs from `gtd7` only in `d[5]`, where it adds the
anomalous-oxygen mass — the script asserts exactly that rather than taking it on trust, so the Rust
side can carry one extra number per case instead of eleven.

    uv run --no-project --with nrlmsise00 python derivations/nrlmsise00_vectors.py
"""

from __future__ import annotations

import itertools
import math

from nrlmsise00 import core

ZN3_TOP = 72.5  # below: `densm` + `glob7s`; the lower-atmosphere branch
MIXED_TOP = (
    120.0  # below: the mesospheric spline blend; above: `gts7` diffusive equilibrium
)
REGIMES = ((0.0, ZN3_TOP), (ZN3_TOP, MIXED_TOP), (MIXED_TOP, 400.0), (400.0, 1000.0))

N_CASES = 128
# One further case per switch, with that switch on and every other one off — plus a final one that
# takes the 3-hourly `ap_array` branch in the same silence. The mixed arms above never reach this
# corner: a term guarded by `sw[i] != 0` and weighted by some `swc[j]` needs `i` on and `j` off *at
# the same time*, and drawing the switches independently makes that combination rare enough that
# most of the guarded groups are only ever evaluated with every weight equal to one.
N_ISOLATED = 25
# And one per *pair* of switches, with that pair off and all twenty-one others on. One-hot covers
# "group `i` on with every weight off"; this covers "group `i` on with weights `j` and `k` off",
# which is what a guard over several switches needs. `globe7`'s local-time block is the example that
# forced it: `if !(sw[7] == 0 && sw[8] == 0 && sw[14] == 0)` computes the diurnal trigonometry, and
# inverting any one of those three tests is invisible unless exactly one of the three is on *and*
# something downstream still multiplies the result in — measured on `18704` evaluations, the pair
# cases separate all three, one-hot and the mixed arms separate none. Leaving two out subsumes
# leaving one out, so that shorter family is not generated separately.
N_PAIRS = 23 * 22 // 2
# And every *triple* drawn from the six switches that leaving out in pairs already showed to
# interact — (5, 15), (10, 16) and (19, 21), the three pairs whose profile contradicts itself and
# so is answered by the temperature bound rather than by the fit. Pairwise coverage cannot separate
# a guard written over three switches at once, and strength-3 over all twenty-three would be 1771
# further cases with no evidence that any of them matters; this is strength 3 over the switches
# that have earned it.
CONTRADICTORY = (5, 10, 15, 16, 19, 21)
N_TRIPLES = 20  # C(6, 3)
# Every altitude at which the model changes its mind: the three spline node tables, the turbopause,
# the fully-mixed floor, the lower-thermosphere gate and the eight `altl` chemistry ceilings. A
# stratified draw lands *near* these and never *on* them, so which way each `>` falls at its own
# boundary is a question no other case asks — and it is a question with a real answer, since the
# branches on either side compute different things.
# The first thermospheric node is the trap: the array is written `{120, 110, 100, 90, 72.5}` and
# then its first entry is overwritten with the fitted `pdl[1][15]`, so the altitude the code
# actually compares against is 123.435 km and a case at 120 km tests a boundary that is not there.
ZA = 1.23435e02
NODE_ALTITUDES = (
    0.0, 10.0, 15.0, 20.0, 32.5, 45.0, 55.0, 62.5, 72.5, 90.0,
    100.0, 110.0, 120.0, ZA, 160.0, 200.0, 240.0, 250.0, 300.0, 320.0, 450.0,
)  # fmt: skip
N_NODES = len(NODE_ALTITUDES)
# `switches[0]` converts the output to cgs, and it is the one switch the port fixes at 0, so it is
# reached only where a case turns it on. One case is not enough: below `zmix` the mixing weight is
# zero and the thermospheric numbers the conversion scales never reach the answer, so a scaling
# error there is invisible. These four are above `zmix` and one per regime.
UNITS_ALTITUDES = (68.0, 95.0, 250.0, 700.0)
N_UNITS = len(UNITS_ALTITUDES)
# The 3-hourly `ap_array` branch with one switch off at a time. The mixed arms reach that branch
# often, but only ever with every `swc` equal to one — a term inside it that is weighted by some
# `swc[j]` is then multiplied by `1`, and multiplying by one and dividing by it are the same
# operation. This is one-hot's argument again, for the half of the model one-hot cannot reach:
# `sw[9] = -1` has to stay on for the branch to be entered at all.
AP_ISOLATED = tuple(i for i in range(1, 24) if i != 9)
N_AP_ISOLATED = len(AP_ISOLATED)
# And `2` — the third state a switch can take, "main effect off, cross terms on", which sets
# `sw[i] = 0` while leaving `swc[i] = 1`. It is the only setting under which the two arrays differ,
# so while no case uses it every `sw[i]` and `swc[i]` are interchangeable.
N_STATE2 = 23
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19)

# `switches[0]` selects the output units and the port fixes it at 0; the rest are the model's own.
DEFAULT_SWITCHES = [0] + [1] * 23


def halton(index: int, base: int) -> float:
    """The `index`-th term of the van der Corput sequence in `base`, in [0, 1)."""
    out, f, i = 0.0, 1.0 / base, index
    while i > 0:
        i, digit = divmod(i, base)
        out += digit * f
        f /= base
    return out


def switch_pairs() -> list[tuple[int, int]]:
    """Every unordered pair of the model's own switches, in a fixed order."""
    return [(j, k) for j in range(1, 24) for k in range(j + 1, 24)]


def switch_triples() -> list[tuple[int, ...]]:
    """Every unordered triple of `CONTRADICTORY`, in a fixed order."""
    return list(itertools.combinations(CONTRADICTORY, 3))


def tail_index(case: int) -> int:
    """Position of `case` within the four appended families, or -1 if it precedes them."""
    return case - (N_CASES + N_ISOLATED + N_PAIRS + N_TRIPLES)


def flags_for(case: int) -> list[int]:
    """The switch array. Eight of the nine arms are paths no published case takes."""
    tail = tail_index(case)
    if tail >= 0:
        s = list(DEFAULT_SWITCHES)
        if tail < N_NODES:
            return s
        tail -= N_NODES
        if tail < N_UNITS:
            s[0] = 1
            return s
        tail -= N_UNITS
        if tail < N_AP_ISOLATED:
            s[9] = -1
            s[AP_ISOLATED[tail]] = 0
            return s
        s[1 + tail - N_AP_ISOLATED] = 2
        return s
    if case >= N_CASES + N_ISOLATED + N_PAIRS:
        s = list(DEFAULT_SWITCHES)
        for i in switch_triples()[case - N_CASES - N_ISOLATED - N_PAIRS]:
            s[i] = 0
        return s
    if case >= N_CASES + N_ISOLATED:
        j, k = switch_pairs()[case - N_CASES - N_ISOLATED]
        s = list(DEFAULT_SWITCHES)
        s[j] = 0
        s[k] = 0
        return s
    if case >= N_CASES:
        j = case - N_CASES
        s = [0] * 24
        s[9 if j == 24 else j] = -1 if j == 24 else 1
        return s
    s = list(DEFAULT_SWITCHES)
    arm = case % 4
    if arm == 1:  # the 3-hourly ap_array path
        s[9] = -1
    elif (
        arm == 2
    ):  # main effects off, cross terms on, across the four largest variations
        for i in (3, 5, 7, 11):
            s[i] = 2
    elif arm == 3:
        # Every switch drawn independently over {0, 1, 2}, offset per index so that each one is
        # driven to 0 in roughly a third of these cases rather than all of them together.
        k = case // 4
        for i in range(1, 24):
            s[i] = (k + i) % 3
        # …and half of those carry the ap_array branch, so an `swc[i] = 0` reaches the terms that
        # only the 3-hourly path evaluates. The other half keeps its drawn value, which covers the
        # daily-`ap` branch with the magnetic switch itself off.
        if k % 2 == 0:
            s[9] = -1
    return s


def case_inputs(case: int) -> dict:
    """One sweep point. Halton on eight dimensions, with altitude stratified over the four regimes."""
    h = [halton(case + 1, p) for p in PRIMES]
    lo, hi = REGIMES[case % len(REGIMES)]
    # `lst` is an independent input to the C, not derived from `sec`/`g_long`. Most cases keep it
    # physically consistent; every fourth breaks the relation, which is the only way the terms that
    # read `lst` separate from the terms that read `sec`.
    sec = 86400.0 * h[1]
    g_long = -180.0 + 360.0 * h[4]
    lst = sec / 3600.0 + g_long / 15.0 if case % 4 else 24.0 * h[1]
    alt = lo + (hi - lo) * h[2]
    tail = tail_index(case)
    if 0 <= tail < N_NODES:
        alt = NODE_ALTITUDES[tail]
    elif N_NODES <= tail < N_NODES + N_UNITS:
        alt = UNITS_ALTITUDES[tail - N_NODES]
    flags = flags_for(case)
    if alt < ZN3_TOP:
        # The one switch combination this reference cannot answer for. `gtd7` sizes its turbopause
        # blend by dividing by the fully-mixed N2 density, which `gts7` writes only when
        # `switches[15]` is on — and the C keeps that density in a file-scope `static`. With mixing
        # off below `72.5 km` it therefore returns a number left over from the *previous* call, so
        # its output is a function of the call order rather than of the inputs. Never ask it.
        flags[15] = 1
    return {
        "year": 0,
        "doy": 1 + int(365 * h[0]),
        "sec": sec,
        "alt": alt,
        "g_lat": -90.0 + 180.0 * h[3],
        "g_long": g_long,
        "lst": lst,
        "f107A": 65.0 + 235.0 * h[5],  # solar minimum to a strong maximum
        "f107": 65.0 + 235.0 * h[6],
        "ap": 0.0 + 300.0 * h[7],  # quiet to a severe storm
        "ap_a": [100.0 * halton(case + 1, 23) + 5.0 * k for k in range(7)],
        "flags": flags,
    }


def rust_case(case: int, gtd7: list[float], gtd7d5: float) -> str:
    args = case_inputs(case)
    switches = ", ".join(str(s) for s in args["flags"])
    ap_a = ", ".join(f"{v:.17e}" for v in args["ap_a"])
    values = ", ".join(f"{v:.17e}" for v in [*gtd7, gtd7d5])
    return (
        f"    // case {case}: alt {args['alt']:.3f} km, lat {args['g_lat']:+.3f}, "
        f"lon {args['g_long']:+.3f}, doy {args['doy']}, ap {args['ap']:.3f}\n"
        f"    Case {{ doy: {args['doy']}, sec: {args['sec']:.17e}, alt: {args['alt']:.17e},\n"
        f"        g_lat: {args['g_lat']:.17e}, g_long: {args['g_long']:.17e}, lst: {args['lst']:.17e},\n"
        f"        f107a: {args['f107A']:.17e}, f107: {args['f107']:.17e}, ap: {args['ap']:.17e},\n"
        f"        ap_array: [{ap_a}],\n"
        f"        switches: [{switches}], out: [{values}] }},\n"
    )


def main() -> None:
    total = (
        N_CASES
        + N_ISOLATED
        + N_PAIRS
        + N_TRIPLES
        + N_NODES
        + N_UNITS
        + N_AP_ISOLATED
        + N_STATE2
    )
    rows = []
    for case in range(total):
        args = case_inputs(case)
        d, t = core.gtd7(**args)
        dd, td = core.gtd7d(**args)
        # The two entry points share everything but the total-mass slot; assert it rather than assume.
        assert [x for i, x in enumerate(d) if i != 5] == [
            x for i, x in enumerate(dd) if i != 5
        ], case
        assert t == td, case
        # The reference has corners where it returns NaN or a negative density, and we deliberately
        # return a bounded number there instead — so a case that lands in one is not an oracle, it is
        # a disagreement baked into a table. Refuse to emit it rather than discover it downstream.
        for name, values in (("gtd7", [*d, *t]), ("gtd7d", [dd[5]])):
            for i, v in enumerate(values):
                if not math.isfinite(v) or v < 0.0:
                    raise SystemExit(
                        f"case {case}: {name} slot {i} is {v}; {args['flags']}"
                    )
        rows.append(rust_case(case, [*d, *t], dd[5]))
    print(f"// {total} cases from the reference C (Brodowski's nrlmsise-00), %.17e.")
    print("".join(rows), end="")


if __name__ == "__main__":
    main()
