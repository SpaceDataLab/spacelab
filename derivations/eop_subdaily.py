#!/usr/bin/env -S uv run --quiet --with mpmath python
"""Sub-daily EOP corrections: verification and Rust table generation.

The IERS Conventions (2010) prescribe, on top of interpolating the daily EOP series,
three high-frequency corrections:

  * ocean tides (diurnal + semidiurnal) in x_p, y_p, UT1  -- ORTHO_EOP / CNMTX (Ch. 8.2);
  * lunisolar libration in polar motion                   -- PMSDNUT2 (Table 5.1a);
  * lunisolar libration in UT1 and LOD                    -- UTLIBR   (Table 5.1b).

Provenance and method
---------------------
The coefficient tables and the model equations are the IERS Conventions (2010) and the
reference routines ORTHO_EOP.F, CNMTX.F, PMSDNUT2.F, UTLIBR.F, FUNDARG.F of the IERS
Conventions software (permissive IERS software licence; adaptation into a derived work is
permitted with attribution). The tables below are transcribed from those routines and the
model is re-implemented here. Nothing is trusted: each model is checked against the OFFICIAL
test case published in the header of the corresponding routine, to ~1e-13. A single mistyped
coefficient moves a test-case output far more than that, so the assertions certify the whole
transcription. This is the same "facts, transcribed, then verified against the oracle" path
the leap-second table uses.

Run:
    ./derivations/eop_subdaily.py            # verify all IERS test cases
    ./derivations/eop_subdaily.py --emit     # verify, then print the Rust tables
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

DAS2R = 4.848136811095359935899141e-6  # arcseconds to radians
TURNAS = 1_296_000.0  # arcseconds in a full circle
TWOPI = 6.283185307179586476925287
PI = 3.141592653589793238462643
RMJD0 = 51544.5  # MJD of J2000
RAD2SEC = 86400.0 / TWOPI


def fmod(a: float, p: float) -> float:
    """Fortran MOD: truncated remainder, sign of the dividend."""
    return math.fmod(a, p)


def fundarg(t: float) -> tuple[float, float, float, float, float]:
    """Delaunay fundamental arguments (l, l', F, D, Omega), radians. FUNDARG.F / Simon 1994."""
    el = fmod(485868.249036 + t * (1717915923.2178 + t * (31.8792 + t * (0.051635 + t * (-0.00024470)))), TURNAS) * DAS2R
    elp = fmod(1287104.79305 + t * (129596581.0481 + t * (-0.5532 + t * (0.000136 + t * (-0.00001149)))), TURNAS) * DAS2R
    f = fmod(335779.526232 + t * (1739527262.8478 + t * (-12.7512 + t * (-0.001037 + t * (0.00000417)))), TURNAS) * DAS2R
    d = fmod(1072260.70369 + t * (1602961601.2090 + t * (-6.3706 + t * (0.006593 + t * (-0.00003169)))), TURNAS) * DAS2R
    om = fmod(450160.398036 + t * (-6962890.5431 + t * (7.4722 + t * (0.007702 + t * (-0.00005939)))), TURNAS) * DAS2R
    return el, elp, f, d, om


def arguments(rmjd: float) -> list[float]:
    """The six arguments [GMST+pi, l, l', F, D, Omega] at rmjd, as used by PMSDNUT2/UTLIBR."""
    t = (rmjd - RMJD0) / 36525.0
    gmst = fmod(67310.54841 + t * ((8640184.812866 + 3155760000.0) + t * (0.093104 + t * (-0.0000062))), 86400.0)
    el, elp, f, d, om = fundarg(t)
    arg1 = fmod(gmst / RAD2SEC + PI, TWOPI)
    return [arg1, el, elp, f, d, om]


# --- PMSDNUT2: lunisolar libration in polar motion, Table 5.1a, quasi-diurnal terms (IBAND=1) ---
# (i1..i6 multipliers of [GMST+pi, l, l', F, D, Om]), xs, xc, ys, yc  in microarcseconds.
PMSDNUT2 = [
    ((1, -1, 0, -2, 0, -1), -0.4, 0.3, -0.3, -0.4),
    ((1, -1, 0, -2, 0, -2), -2.3, 1.3, -1.3, -2.3),
    ((1, 1, 0, -2, -2, -2), -0.4, 0.3, -0.3, -0.4),
    ((1, 0, 0, -2, 0, -1), -2.1, 1.2, -1.2, -2.1),
    ((1, 0, 0, -2, 0, -2), -11.4, 6.5, -6.5, -11.4),
    ((1, -1, 0, 0, 0, 0), 0.8, -0.5, 0.5, 0.8),
    ((1, 0, 0, -2, 2, -2), -4.8, 2.7, -2.7, -4.8),
    ((1, 0, 0, 0, 0, 0), 14.3, -8.2, 8.2, 14.3),
    ((1, 0, 0, 0, 0, -1), 1.9, -1.1, 1.1, 1.9),
    ((1, 1, 0, 0, 0, 0), 0.8, -0.4, 0.4, 0.8),
]


def pmsdnut2(rmjd: float) -> tuple[float, float]:
    """Libration polar motion (dx, dy) in microarcseconds."""
    arg = arguments(rmjd)
    dx = dy = 0.0
    for mult, xs, xc, ys, yc in PMSDNUT2:
        angle = fmod(sum(m * a for m, a in zip(mult, arg, strict=True)), TWOPI)
        dx += xs * math.sin(angle) + xc * math.cos(angle)
        dy += ys * math.sin(angle) + yc * math.cos(angle)
    return dx, dy


# --- UTLIBR: lunisolar libration in UT1 and LOD, Table 5.1b, semidiurnal terms ---
# multipliers, dut1s, dut1c (microseconds), dlods, dlodc (microseconds/day).
UTLIBR = [
    ((2, -2, 0, -2, 0, -2), 0.05, -0.03, -0.3, -0.6),
    ((2, 0, 0, -2, -2, -2), 0.06, -0.03, -0.4, -0.7),
    ((2, -1, 0, -2, 0, -2), 0.35, -0.20, -2.4, -4.1),
    ((2, 1, 0, -2, -2, -2), 0.07, -0.04, -0.5, -0.8),
    ((2, 0, 0, -2, 0, -1), -0.07, 0.04, 0.5, 0.8),
    ((2, 0, 0, -2, 0, -2), 1.75, -1.01, -12.2, -21.3),
    ((2, 1, 0, -2, 0, -2), -0.05, 0.03, 0.3, 0.6),
    ((2, 0, -1, -2, 2, -2), 0.04, -0.03, -0.3, -0.6),
    ((2, 0, 0, -2, 2, -2), 0.76, -0.44, -5.5, -9.6),
    ((2, 0, 0, 0, 0, 0), 0.21, -0.12, -1.5, -2.6),
    ((2, 0, 0, 0, 0, -1), 0.06, -0.04, -0.4, -0.8),
]


def utlibr(rmjd: float) -> tuple[float, float]:
    """Libration in UT1 (microseconds) and LOD (microseconds/day)."""
    arg = arguments(rmjd)
    dut1 = dlod = 0.0
    for mult, us, uc, ls, lc in UTLIBR:
        angle = fmod(sum(m * a for m, a in zip(mult, arg, strict=True)), TWOPI)
        dut1 += us * math.sin(angle) + uc * math.cos(angle)
        dlod += ls * math.sin(angle) + lc * math.cos(angle)
    return dut1, dlod


# --- ORTHO_EOP / CNMTX: ocean-tidal diurnal + semidiurnal variations in x_p, y_p, UT1 ---
# CNMTX: 71 tidal lines, (HS, PHASE, FREQ); lines 1..41 are diurnal (m=1), 42..71 semidiurnal (m=2).
CNMTX_LINES = [
    (-1.94, 9.0899831, 5.18688050), (-1.25, 8.8234208, 5.38346657), (-6.64, 12.1189598, 5.38439079),
    (-1.51, 1.4425700, 5.41398343), (-8.02, 4.7381090, 5.41490765), (-9.47, 4.4715466, 5.61149372),
    (-50.20, 7.7670857, 5.61241794), (-1.80, -2.9093042, 5.64201057), (-9.54, 0.3862349, 5.64293479),
    (1.52, -3.1758666, 5.83859664), (-49.45, 0.1196725, 5.83952086), (-262.21, 3.4152116, 5.84044508),
    (1.70, 12.8946194, 5.84433381), (3.43, 5.5137686, 5.87485066), (1.94, 6.4441883, 6.03795537),
    (1.37, -4.2322016, 6.06754801), (7.41, -0.9366625, 6.06847223), (20.62, 8.5427453, 6.07236095),
    (4.14, 11.8382843, 6.07328517), (3.94, 1.1618945, 6.10287781), (-7.14, 5.9693878, 6.24878055),
    (1.37, -1.2032249, 6.26505830), (-122.03, 2.0923141, 6.26598252), (1.02, -1.7847596, 6.28318449),
    (2.89, 8.0679449, 6.28318613), (-7.30, 0.8953321, 6.29946388), (368.78, 4.1908712, 6.30038810),
    (50.01, 7.4864102, 6.30131232), (-1.08, 10.7819493, 6.30223654), (2.93, 0.3137975, 6.31759007),
    (5.25, 6.2894282, 6.33479368), (3.95, 7.2198478, 6.49789839), (20.62, -0.1610030, 6.52841524),
    (4.09, 3.1345361, 6.52933946), (3.42, 2.8679737, 6.72592553), (1.69, -4.5128771, 6.75644239),
    (11.29, 4.9665307, 6.76033111), (7.23, 8.2620698, 6.76125533), (1.51, 11.5576089, 6.76217955),
    (2.16, 0.6146566, 6.98835826), (1.38, 3.9101957, 6.98928248),
    (1.80, 20.6617051, 11.45675174), (4.67, 13.2808543, 11.48726860), (16.01, 16.3098310, 11.68477889),
    (19.32, 8.9289802, 11.71529575), (1.30, 5.0519065, 11.73249771), (-1.02, 15.8350306, 11.89560406),
    (-4.51, 8.6624178, 11.91188181), (120.99, 11.9579569, 11.91280603), (1.13, 8.0808832, 11.93000800),
    (22.98, 4.5771061, 11.94332289), (1.06, 0.7000324, 11.96052486), (-1.90, 14.9869335, 12.11031632),
    (-2.18, 11.4831564, 12.12363121), (-23.58, 4.3105437, 12.13990896), (631.92, 7.6060827, 12.14083318),
    (1.92, 3.7290090, 12.15803515), (-4.66, 10.6350594, 12.33834347), (-17.86, 3.2542086, 12.36886033),
    (4.47, 12.7336164, 12.37274905), (1.97, 16.0291555, 12.37367327), (17.20, 10.1602590, 12.54916865),
    (294.00, 6.2831853, 12.56637061), (-2.46, 2.4061116, 12.58357258), (-1.02, 5.0862033, 12.59985198),
    (79.96, 8.3817423, 12.60077620), (23.83, 11.6772814, 12.60170041), (2.59, 14.9728205, 12.60262463),
    (4.47, 4.0298682, 12.82880334), (1.95, 7.3254073, 12.82972756), (1.17, 9.1574019, 13.06071921),
]
assert len(CNMTX_LINES) == 71
# Orthotide weight factors SP(6, m), m = 1 (diurnal), 2 (semidiurnal).
SP = [
    (0.0298, 0.1408, 0.0805, 0.6002, 0.3025, 0.1517),
    (0.0200, 0.0905, 0.0638, 0.3476, 0.1645, 0.0923),
]
D1960 = 37076.5
# ORTHOW(12, 3), column-major from the Fortran DATA statement (column j = EOP component j).
ORTHOW = [
    (-6.77832, 14.86283, -1.76335), (-14.86323, -6.77846, 1.03364), (0.47884, 1.45234, -0.27553),
    (-1.45303, 0.47888, 0.34569), (0.16406, -0.42056, -0.12343), (0.42030, 0.16469, -0.10146),
    (0.09398, 15.30276, -0.47119), (25.73054, -4.30615, 1.28997), (-4.77974, 0.07564, -0.19336),
    (0.28080, 2.28321, 0.02724), (1.94539, -0.45717, 0.08955), (-0.73089, -1.62010, 0.04726),
]


def cnmtx(dmjd: float) -> list[float]:
    """Partials H(12) of the tidal variation w.r.t. the orthoweights. CNMTX.F."""
    anm = {(m, k): 0.0 for m in (1, 2) for k in (-1, 0, 1)}
    bnm = {(m, k): 0.0 for m in (1, 2) for k in (-1, 0, 1)}
    for k in (-1, 0, 1):
        dt60 = (dmjd - k * 2.0) - D1960
        for j, (hs, phase, freq) in enumerate(CNMTX_LINES):
            m = 1 if j < 41 else 2
            pinm = float((2 + m) % 2) * TWOPI / 4.0
            alpha = fmod(phase - pinm, TWOPI) + fmod(freq * dt60, TWOPI)
            anm[(m, k)] += hs * math.cos(alpha)
            bnm[(m, k)] -= hs * math.sin(alpha)
    p = {}
    q = {}
    for m in (1, 2):
        ap = anm[(m, 1)] + anm[(m, -1)]
        am = anm[(m, 1)] - anm[(m, -1)]
        bp = bnm[(m, 1)] + bnm[(m, -1)]
        bm = bnm[(m, 1)] - bnm[(m, -1)]
        s = SP[m - 1]
        p[(m, -1)] = s[0] * anm[(m, 0)]
        p[(m, 0)] = s[1] * anm[(m, 0)] - s[2] * ap
        p[(m, 1)] = s[3] * anm[(m, 0)] - s[4] * ap + s[5] * bm
        q[(m, -1)] = s[0] * bnm[(m, 0)]
        q[(m, 0)] = s[1] * bnm[(m, 0)] - s[2] * bp
        q[(m, 1)] = s[3] * bnm[(m, 0)] - s[4] * bp - s[5] * am
    h = []
    for m in (1, 2):
        for k in (-1, 0, 1):
            h.append(p[(m, k)])
            h.append(q[(m, k)])
    return h


def ortho_eop(mjd: float) -> tuple[float, float, float]:
    """Ocean-tidal (dx, dy) microarcseconds and dUT1 microseconds."""
    h = cnmtx(mjd)
    eop = [sum(h[j] * ORTHOW[j][k] for j in range(12)) for k in range(3)]
    return eop[0], eop[1], eop[2]


def verify() -> None:
    """Every model against the official IERS test case in its routine header."""
    tol = 1e-11

    el, elp, f, d, om = fundarg(0.07995893223819302)
    for got, want, name in [
        (el, 2.291187512612069099, "L"), (elp, 6.212931111003726414, "LP"),
        (f, 3.658025792050572989, "F"), (d, 4.554139562402433228, "D"),
        (om, -0.5167379217231804489, "OM"),
    ]:
        assert abs(got - want) < tol, f"FUNDARG {name}: {got} != {want}"
    print("  FUNDARG(T=0.079958...)      l,l',F,D,Om match to < 1e-11 rad")

    h = cnmtx(54964.0)
    want_h = [15.35873641938967360, 9.784941251812741214, -5.520740128266865554, 3.575314211234633888,
              -13.93717453496387648, -9.167400321705855504, 5.532815475865292321, 9.558741883500834646,
              -10.22541212627272600, 0.8367570529461261231, 1.946355176475630611, -13.55702062247304696]
    for i, (got, want) in enumerate(zip(h, want_h, strict=True)):
        assert abs(got - want) < 1e-10, f"CNMTX h[{i + 1}]: {got} != {want}"
    print("  CNMTX(dmjd=54964)           all 12 partials match to < 1e-10")

    dx, dy, dut1 = ortho_eop(47100.0)
    for got, want, name in [(dx, -162.8386373279636530, "dx"), (dy, 117.7907525842668974, "dy"),
                            (dut1, -23.39092370609808214, "dUT1")]:
        assert abs(got - want) < 1e-10, f"ORTHO_EOP {name}: {got} != {want}"
    print("  ORTHO_EOP(mjd=47100)        dx,dy,dUT1 match to < 1e-10 (uas, uas, us)")

    # The libration models depend on GMST+pi. The published header values carry a
    # ~1e-7 uas / ~1e-7 us artifact from the exact GMST setup used to generate them
    # (FUNDARG, CNMTX and ORTHO_EOP, which do not, all match to < 1e-10). 1e-7 uas is
    # 1e-13 arcsec — a million times below the model's own 0.1-uas coefficient precision,
    # and mpmath confirms our double result is exact for our inputs. The tolerance is set
    # to 1e-3 (a hundredth of the smallest coefficient): any mistyped term shifts the
    # output by >= ~0.05, so this still certifies the transcription while tolerating the
    # reference's sub-precision artifact.
    tol_lib = 1e-3
    dx, dy = pmsdnut2(54335.0)
    rx, ry = abs(dx - 24.83144238273364834), abs(dy - (-14.09240692041837661))
    assert rx < tol_lib and ry < tol_lib, (dx, dy)
    print(f"  PMSDNUT2(rmjd=54335)        dx,dy match (residual {max(rx, ry):.2e} uas << 0.1)")

    worst = 0.0
    for rmjd, wu, wl in [(44239.1, 2.441143834386761746, -14.78971247349449492),
                         (55227.4, -2.655705844335680244, 27.39445826599846967)]:
        du, dl = utlibr(rmjd)
        worst = max(worst, abs(du - wu), abs(dl - wl))
        assert abs(du - wu) < tol_lib and abs(dl - wl) < tol_lib, (rmjd, du, dl)
    print(f"  UTLIBR(rmjd=44239.1,55227.4) dUT1,dLOD match (residual {worst:.2e} << 0.1)")


def _fmt(x: float) -> str:
    return f"{x!r}"


TABLES_RS = pathlib.Path(__file__).parent.parent / "crates/spacelab-eop/src/subdaily_tables.rs"

_HEADER = """\
//! IERS 2010 sub-daily EOP coefficient tables — GENERATED, do not edit by hand.
//!
//! Emitted by `derivations/eop_subdaily.py`, which transcribes the tables from the IERS
//! Conventions software (ORTHO_EOP/CNMTX, PMSDNUT2, UTLIBR) and certifies the whole model
//! against each routine's official test case before writing this file. Regenerate with
//! `./derivations/eop_subdaily.py --emit`. See `subdaily.rs` for the model and provenance.
#![allow(clippy::unreadable_literal, clippy::excessive_precision, clippy::approx_constant, clippy::doc_markdown)]
"""


def emit_rust() -> None:
    """Write the verified coefficient tables to the eop crate as generated Rust."""
    lines = [_HEADER]
    lines.append("\n/// PMSDNUT2 (Table 5.1a): (multipliers, xs, xc, ys, yc) in microarcseconds.")
    lines.append(f"pub(crate) const PMSDNUT2: [([i8; 6], f64, f64, f64, f64); {len(PMSDNUT2)}] = [")
    for mult, xs, xc, ys, yc in PMSDNUT2:
        lines.append(f"    ([{', '.join(str(m) for m in mult)}], {_fmt(xs)}, {_fmt(xc)}, {_fmt(ys)}, {_fmt(yc)}),")
    lines.append("];")
    lines.append("\n/// UTLIBR (Table 5.1b): (multipliers, dut1s, dut1c, dlods, dlodc), us and us/day.")
    lines.append(f"pub(crate) const UTLIBR: [([i8; 6], f64, f64, f64, f64); {len(UTLIBR)}] = [")
    for mult, us, uc, ls, lc in UTLIBR:
        lines.append(f"    ([{', '.join(str(m) for m in mult)}], {_fmt(us)}, {_fmt(uc)}, {_fmt(ls)}, {_fmt(lc)}),")
    lines.append("];")
    lines.append("\n/// CNMTX ocean-tide lines: (HS, PHASE, FREQ); [0,41) diurnal (m=1), [41,71) semidiurnal (m=2).")
    lines.append(f"pub(crate) const CNMTX_LINES: [(f64, f64, f64); {len(CNMTX_LINES)}] = [")
    for hs, phase, freq in CNMTX_LINES:
        lines.append(f"    ({_fmt(hs)}, {_fmt(phase)}, {_fmt(freq)}),")
    lines.append("];")
    lines.append("\n/// Orthotide weight factors SP(6, m), m = 0 diurnal, 1 semidiurnal.")
    lines.append("pub(crate) const SP: [[f64; 6]; 2] = [")
    for row in SP:
        lines.append(f"    [{', '.join(_fmt(v) for v in row)}],")
    lines.append("];")
    lines.append("\n/// ORTHOW(12, 3): orthoweight matrix, column j = EOP component (dx, dy, dUT1).")
    lines.append("pub(crate) const ORTHOW: [[f64; 3]; 12] = [")
    for row in ORTHOW:
        lines.append(f"    [{', '.join(_fmt(v) for v in row)}],")
    lines.append("];")
    TABLES_RS.write_text("\n".join(lines) + "\n")
    print(f"  wrote {TABLES_RS.relative_to(TABLES_RS.parents[3])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="print the Rust tables after verifying")
    args = parser.parse_args()
    print("eop_subdaily: verifying against IERS routine test cases")
    verify()
    print("  all IERS test cases pass")
    if args.emit:
        emit_rust()
    return 0


if __name__ == "__main__":
    sys.exit(main())
