#!/usr/bin/env python
"""Generate the Fairhead & Bretagnon TDB-TT coefficient table.

TDB-TT is a published scientific model, not an implementation: the periodic series
is Fairhead & Bretagnon (1990, A&A 229, 240) with the fundamental arguments of
Simon et al. (1994, A&A 282, 663), as adopted for IAU 2006 resolution B3 and
described in IERS Conventions 2010 (TN36) section 10.1. A machine-readable copy of
the 787 amplitudes ships in the SOFA/ERFA `dtdb.c` source; this script extracts
them and emits a Rust `const` module.

Provenance / licensing: the numbers are the published model coefficients (facts),
not any implementation's original expression. They are cross-checked here against
pyerfa (ERFA, BSD-3-Clause) over a dense grid and against the SOFA `t_dtdb`
self-test vector before emission. spacelab never links SOFA or ERFA.

The self-test vector is topocentric, so this script implements the topocentric
terms too even though `spacelab-fa` exposes only the geocentric conversion: they
are what proves the extracted table is complete and in the right order. Their size
is reported below and quoted in the Rust doc.

Run:
    timeout 300 uv run --no-project --with pyerfa --with numpy python \
        derivations/tdb_tables.py

Writes crates/spacelab-fa/src/tdb_tables.rs and prints the validation residuals.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import erfa
import numpy as np

SOFA_SRC = Path(
    "/home/ilgrad/Documents/projects/space_work/repos/sofa_c-20231011/sofa/20231011/c/src"
)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/spacelab-fa/src/tdb_tables.rs"
FIXTURE = ROOT / "crates/spacelab-fa/tests/fixtures/dtdb_pyerfa.csv"

DJ00 = 2451545.0
DJM = 365250.0  # days per Julian millennium (sofam.h DJM)
D2PI = 6.283185307179586476925287
DD2R = 1.745329251994329576923691e-2

# SOFA t_dtdb golden vector (t_sofa_c.c): topocentric, so it exercises every block.
GOLDEN_ARGS = (2448939.5, 0.123, 0.76543, 5.0123, 5525.242, 3190.0)
GOLDEN_DTDB = -0.1280368005936998991e-2

# Group boundaries in `fairhd`, read off the five evaluation loops in dtdb.c.
GROUP_END = (474, 679, 764, 784, 787)


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", text)


def array_body(text: str, decl_regex: str) -> str:
    """Return the text inside the outermost {...} of a `<decl> = { ... };` array."""
    m = re.search(decl_regex + r"\s*=\s*\{", text)
    if not m:
        raise ValueError(f"declaration not found: {decl_regex}")
    start = m.end() - 1
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    raise ValueError(f"unbalanced braces: {decl_regex}")


def rows_of_three(body: str) -> list[tuple[str, str, str]]:
    """Split `{a, b, c},` groups, keeping the literals as written."""
    out: list[tuple[str, str, str]] = []
    for group in re.findall(r"\{([^{}]*)\}", body):
        fields = [f.strip() for f in group.split(",") if f.strip()]
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"expected amplitude, frequency, phase: {group!r}")
        out.append((fields[0], fields[1], fields[2]))
    return out


SRC = (SOFA_SRC / "dtdb.c").read_text()
CLEAN = strip_comments(SRC)

fairhd = rows_of_three(array_body(CLEAN, r"fairhd\s*\[\s*787\s*\]\s*\[\s*3\s*\]"))
if len(fairhd) != 787:
    raise ValueError(f"expected 787 terms, parsed {len(fairhd)}")

# The JPL planetary-mass adjustment is written inline rather than tabulated.
wj_src = CLEAN[CLEAN.index("wj =") :]
wj_src = wj_src[: wj_src.index(";")]
jpl = re.findall(
    r"(-?\d+\.\d+)e-6\s*\*\s*sin\(\s*(-?\d+\.\d+)\s*\*\s*t\s*\+\s*(-?\d+\.\d+)\s*\)",
    wj_src,
)
if len(jpl) != 4:
    raise ValueError(f"expected 4 JPL-mass sine terms, parsed {len(jpl)}")
jpl_t2 = re.search(r"(-?\d+\.\d+)e-6\s*\*\s*t\s*\*\s*t", wj_src)
if not jpl_t2:
    raise ValueError("JPL-mass t^2 term not found")

# The topocentric block, used only to validate against the golden vector.
wt_src = CLEAN[CLEAN.index("wt =") :]
wt_src = wt_src[: wt_src.index(";")]
topo = re.findall(
    r"([-+])\s*(\d+\.\d+)e-10\s*\*\s*([uv])\s*\*\s*(sin|cos)\(([^)]*)\)", wt_src
)
if len(topo) != 10:
    raise ValueError(f"expected 10 topocentric terms, parsed {len(topo)}")


# --------------------------------------------------------------------------- #
# Reference evaluation, in Python, from the parsed literals
# --------------------------------------------------------------------------- #
amp = np.array([float(a) for a, _, _ in fairhd])
freq = np.array([float(f) for _, f, _ in fairhd])
phase = np.array([float(p) for _, _, p in fairhd])
jpl_amp = np.array([float(a) for a, _, _ in jpl])
jpl_freq = np.array([float(f) for _, f, _ in jpl])
jpl_phase = np.array([float(p) for _, _, p in jpl])
JPL_T2 = float(jpl_t2.group(1)) * 1e-6


def fairhead(t: float) -> float:
    """The periodic series plus the JPL planetary-mass adjustment, in seconds."""
    w = [0.0] * 5
    lo = 0
    for g, hi in enumerate(GROUP_END):
        w[g] = float(np.sum(amp[lo:hi] * np.sin(freq[lo:hi] * t + phase[lo:hi])))
        lo = hi
    wf = t * (t * (t * (t * w[4] + w[3]) + w[2]) + w[1]) + w[0]
    wj = (
        float(np.sum(jpl_amp * 1e-6 * np.sin(jpl_freq * t + jpl_phase)))
        + JPL_T2 * t * t
    )
    return wf + wj


def topocentric(t: float, ut: float, elong: float, u: float, v: float) -> float:
    """Moyer (1981) and Murray (1983) diurnal terms, in seconds, from `u`, `v` in km."""
    tsol = math.fmod(ut, 1.0) * D2PI + elong
    w = t / 3600.0
    args = {
        "tsol": tsol,
        "elsun": math.fmod(280.46645683 + 1296027711.03429 * w, 360.0) * DD2R,
        "emsun": math.fmod(357.52910918 + 1295965810.481 * w, 360.0) * DD2R,
        "d": math.fmod(297.85019547 + 16029616012.090 * w, 360.0) * DD2R,
        "elj": math.fmod(34.35151874 + 109306899.89453 * w, 360.0) * DD2R,
        "els": math.fmod(50.07744430 + 44046398.47038 * w, 360.0) * DD2R,
    }
    total = 0.0
    for sign, magnitude, radius, fn, expr in topo:
        angle = eval(expr.replace("2.0 *", "2.0*"), {"__builtins__": {}}, args)
        f = math.sin(angle) if fn == "sin" else math.cos(angle)
        total += (
            (1.0 if sign == "+" else -1.0)
            * float(magnitude)
            * 1e-10
            * (u if radius == "u" else v)
            * f
        )
    return total


def dtdb(
    date1: float, date2: float, ut: float, elong: float, u: float, v: float
) -> float:
    t = ((date1 - DJ00) + date2) / DJM
    return topocentric(t, ut, elong, u, v) + fairhead(t)


# --------------------------------------------------------------------------- #
# Validate
# --------------------------------------------------------------------------- #
mine_golden = dtdb(*GOLDEN_ARGS)
erfa_golden = float(erfa.dtdb(*GOLDEN_ARGS))
print(f"golden: mine={mine_golden:.19e}")
print(f"        sofa={GOLDEN_DTDB:.19e}  d={abs(mine_golden - GOLDEN_DTDB):.2e}")
print(f"        erfa={erfa_golden:.19e}  d={abs(mine_golden - erfa_golden):.2e}")
assert abs(mine_golden - GOLDEN_DTDB) < 1e-15, "golden vector mismatch"
assert abs(mine_golden - erfa_golden) < 1e-18, "erfa mismatch on the golden vector"

# A dense geocentric sweep: 1900-2100, every 30 days. This is the case the Rust
# side implements, so it is the one that has to agree to the last bit.
jds = np.arange(2415020.5, 2488070.5, 30.0)
worst_geo = 0.0
worst_at = 0.0
for jd in jds:
    mine = fairhead((jd - DJ00) / DJM)
    ref = float(erfa.dtdb(jd, 0.0, 0.0, 0.0, 0.0, 0.0))
    if abs(mine - ref) > worst_geo:
        worst_geo, worst_at = abs(mine - ref), jd
print(
    f"geocentric sweep: {len(jds)} epochs 1900-2100, worst |d| = {worst_geo:.2e} s at JD {worst_at}"
)
assert worst_geo < 1e-18, "geocentric series does not reproduce ERFA"

# What the geocentric-only choice costs a ground observer, and what the
# two-term approximation other libraries use costs everybody.
equatorial = max(
    abs(topocentric((jd - DJ00) / DJM, 0.5, lon, 6378.137, 0.0))
    for jd in jds[::7]
    for lon in (0.0, 1.5, 3.0, 4.5)
)
worst_two_term = 0.0
for jd in jds:
    t = (jd - DJ00) / DJM
    g = (357.53 + 35999.05 * t * 10.0) * DD2R
    approx = 0.001658 * math.sin(g) + 0.000014 * math.sin(2.0 * g)
    worst_two_term = max(worst_two_term, abs(approx - fairhead(t)))
print(f"topocentric term at the equator: up to {equatorial * 1e6:.2f} us")
print(f"two-term approximation: worst {worst_two_term * 1e6:.3f} us over the same span")

print("VALIDATION PASSED — table reproduces ERFA and the SOFA t_dtdb vector")

# --------------------------------------------------------------------------- #
# Emit Rust
# --------------------------------------------------------------------------- #
rows = "\n".join(f"    [{a}, {f}, {p}]," for a, f, p in fairhd)
jpl_rows = "\n".join(f"    [{a}e-6, {f}, {p}]," for a, f, p in jpl)

lines = [
    "// @generated by derivations/tdb_tables.py — DO NOT EDIT BY HAND.",
    "//",
    "// Fairhead & Bretagnon (1990, A&A 229, 240) TDB-TT periodic series with the",
    "// fundamental arguments of Simon et al. (1994, A&A 282, 663), as adopted for IAU",
    "// 2006 resolution B3 and described in IERS Conventions 2010 (TN36) section 10.1.",
    "// Coefficients are the published model constants. Validated against pyerfa (ERFA,",
    f"// BSD-3-Clause) over {len(jds)} epochs from 1900 to 2100 to < 1e-18 s, and against",
    "// the SOFA t_dtdb self-test vector to < 1e-15 s. spacelab does not link SOFA or",
    "// ERFA.",
    "",
    "// Verbatim published numeric data: keep the digits, no digit grouping.",
    "#![allow(clippy::excessive_precision, clippy::unreadable_literal,",
    "    clippy::large_const_arrays, clippy::approx_constant)]",
    "#![cfg_attr(rustfmt, rustfmt::skip)]",
    "",
    "/// One past the last index of each power-of-`t` group in [`FAIRHEAD`].",
    "///",
    "/// The published series is one flat table whose blocks multiply `t^0` through",
    "/// `t^4`; the split is carried here rather than in five arrays so the table stays",
    "/// byte-comparable with the source it was extracted from.",
    f"pub(crate) const GROUP_END: [usize; 5] = {list(GROUP_END)};".replace(
        "[", "["
    ).replace("]", "]"),
    "",
    "/// Amplitude (seconds), frequency (rad per Julian millennium) and phase (rad).",
    f"pub(crate) const FAIRHEAD: [[f64; 3]; {len(fairhd)}] = [",
    rows,
    "];",
    "",
    "/// Periodic part of the adjustment to JPL planetary masses, same layout.",
    f"pub(crate) const JPL_MASS: [[f64; 3]; {len(jpl)}] = [",
    jpl_rows,
    "];",
    "",
    "/// Secular part of the same adjustment: the coefficient of `t^2`, in seconds.",
    f"pub(crate) const JPL_MASS_T2: f64 = {jpl_t2.group(1)}e-6;",
    "",
]
OUT.write_text("\n".join(lines))
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")

# --------------------------------------------------------------------------- #
# Emit the oracle fixture the Rust test reads
# --------------------------------------------------------------------------- #
# The epoch is stored as the integer nanosecond count `Epoch` is built from, and the
# oracle is called on the same `(day, fraction)` split the Rust side derives from it.
# Anything looser measures the two epochs drifting apart rather than the two series:
# a Julian Date near 2.46e6 resolves to 40 us in an f64, and `d(TDB-TT)/dt` is
# 3.3e-10, so a whole-JD fixture argument shows up as a 1e-16 s disagreement that has
# nothing to do with the coefficients.
NS_PER_DAY = 86400 * 10**9
rng = np.random.default_rng(20260728)
sample = np.concatenate(
    [
        np.arange(-36525.0, 36526.0, 365.25),  # a century each side, annually
        np.arange(-10.0, 10.25, 0.25),  # a fortnight at six-hour spacing
        rng.uniform(-36525.0, 36525.0, 160),  # and unaligned epochs
    ]
)
fixture = []
for ns in sorted({round(float(d) * NS_PER_DAY) for d in sample}):
    day, rem = divmod(ns, NS_PER_DAY)
    ref = float(erfa.dtdb(DJ00, day + rem / NS_PER_DAY, 0.0, 0.0, 0.0, 0.0))
    fixture.append(f"{ns},{ref:.17e}")
FIXTURE.write_text("\n".join(fixture) + "\n")
print(f"wrote {FIXTURE} ({len(fixture)} epochs)")
