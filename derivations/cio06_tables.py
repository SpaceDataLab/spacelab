#!/usr/bin/env python
"""Generate the IAU 2006/2000A CIP X,Y and CIO locator s coefficient tables.

The CIP X,Y series and the CIO locator s series are IERS-published scientific
constants (IERS Conventions 2010, Technical Note 36, tables 5.2a/5.2b for X,Y and
5.2c/5.2d for s; original models: Capitaine, Wallace & Chapront 2003 A&A 412, 567
for the precession, Mathews, Herring & Buffett 2002 for the MHB2000 nutation). A
machine-readable copy of these same numbers ships in the SOFA/ERFA `xy06.c` and
`s06.c` sources; this script extracts them and emits a Rust `const` module.

Provenance / licensing: the numbers are the IERS/IAU model coefficients (facts),
not any implementation's original expression. They are cross-checked here against
pyerfa (ERFA, BSD-3-Clause) and the SOFA self-test vectors before emission, and the
generated Rust attributes the model to IERS TN36. spacelab never links SOFA/ERFA.

Run:
    timeout 180 uv run --no-project --with pyerfa --with numpy python derivations/cio06_tables.py

Writes crates/spacelab-fa/src/cio_tables.rs and prints the validation residuals.
"""

from __future__ import annotations

import re
from pathlib import Path

import erfa

SOFA_SRC = Path("/home/ilgrad/Documents/projects/space_work/repos/sofa_c-20231011/sofa/20231011/c/src")
OUT = Path(__file__).resolve().parent.parent / "crates/spacelab-fa/src/cio_tables.rs"

DAS2R = 4.848136811095359935899141e-6  # arcsec -> rad (sofam.h)
DJ00 = 2451545.0
DJC = 36525.0

# SOFA t_xy06 / t_s06 golden vectors (t_sofa_c.c) at TT JD 2400000.5 + 53736.0.
GOLDEN_JD = (2400000.5, 53736.0)
GOLDEN_XY = (0.5791308486706010975e-3, 0.4020579816732958141e-4)
GOLDEN_S_XY = (0.5791308486706011000e-3, 0.4020579816732961219e-4)
GOLDEN_S = -0.1220032213076463117e-7


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", text)


def body_after(text: str, decl_regex: str) -> str:
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


def top_level_groups(body: str) -> list[str]:
    """Split a brace body into the content of each top-level {...} group."""
    groups: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in body:
        if ch == "{":
            depth += 1
            if depth == 1:
                cur = []
                continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                groups.append("".join(cur))
                continue
        if depth >= 1:
            cur.append(ch)
    return groups


INT = re.compile(r"-?\d+")
FLOAT = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?")


def ints(s: str) -> list[int]:
    return [int(x) for x in INT.findall(s)]


def float_tokens(s: str) -> list[str]:
    return FLOAT.findall(s)


# --------------------------------------------------------------------------- #
# Parse xy06.c
# --------------------------------------------------------------------------- #
xy_src = strip_comments((SOFA_SRC / "xy06.c").read_text())

xyp_groups = top_level_groups(body_after(xy_src, r"static const double xyp\[2\]\[MAXPT\+1\]"))
assert len(xyp_groups) == 2, len(xyp_groups)
xyp = [float_tokens(g) for g in xyp_groups]
assert all(len(row) == 6 for row in xyp), [len(r) for r in xyp]

mfals = [ints(g) for g in top_level_groups(body_after(xy_src, r"static const int mfals\[\]\[5\]"))]
mfapl = [ints(g) for g in top_level_groups(body_after(xy_src, r"static const int mfapl\[\]\[14\]"))]
assert all(len(r) == 5 for r in mfals) and len(mfals) == 653, (len(mfals),)
assert all(len(r) == 14 for r in mfapl) and len(mfapl) == 656, (len(mfapl),)

nc = ints(body_after(xy_src, r"static const int nc\[\]"))
a_tok = float_tokens(body_after(xy_src, r"static const double a\[\]"))
jaxy = ints(body_after(xy_src, r"static const int jaxy\[\]"))
jasc = ints(body_after(xy_src, r"static const int jasc\[\]"))
japt = ints(body_after(xy_src, r"static const int japt\[\]"))
assert len(nc) == 1309, len(nc)
assert len(a_tok) == 4755, len(a_tok)
assert len(jaxy) == len(jasc) == len(japt) == 20

NFLS = 653
a_val = [float(t) for t in a_tok]
nc_pad = [*nc, len(a_tok) + 1]  # sentinel: last block runs to NA

# Flatten per frequency: freq p (0..1308) owns a[nc[p]-1 .. nc[p+1]-2] (0-based).
xy_mult: list[list[int]] = []
xy_offset: list[int] = [0]
xy_code: list[int] = []
xy_amp_tok: list[str] = []
for p in range(1309):
    mult = list(mfals[p]) + [0] * 9 if p < NFLS else list(mfapl[p - NFLS])
    assert len(mult) == 14
    xy_mult.append(mult)
    lo, hi = nc_pad[p] - 1, nc_pad[p + 1] - 1  # 0-based [lo, hi)
    for j, idx in enumerate(range(lo, hi)):
        code = jaxy[j] | (jasc[j] << 1) | (japt[j] << 2)
        xy_code.append(code)
        xy_amp_tok.append(a_tok[idx])
    xy_offset.append(len(xy_code))
assert xy_offset[-1] == 4755, xy_offset[-1]

# --------------------------------------------------------------------------- #
# Parse s06.c
# --------------------------------------------------------------------------- #
s_src = strip_comments((SOFA_SRC / "s06.c").read_text())
s_poly = float_tokens(body_after(s_src, r"static const double sp\[\]"))
assert len(s_poly) == 6, len(s_poly)

s_mult: list[list[int]] = []
s_pt: list[int] = []
s_sin: list[str] = []
s_cos: list[str] = []
for power, arr in enumerate(["s0", "s1", "s2", "s3", "s4"]):
    for g in top_level_groups(body_after(s_src, rf"static const TERM {arr}\[\]")):
        inner = re.search(r"\{([^{}]*)\}", g)
        assert inner is not None
        mult = ints(inner.group(1))
        assert len(mult) == 8, (arr, mult)
        rest = g[inner.end() :]
        coeffs = float_tokens(rest)
        assert len(coeffs) == 2, (arr, coeffs)
        s_mult.append(mult)
        s_pt.append(power)
        s_sin.append(coeffs[0])
        s_cos.append(coeffs[1])
assert len(s_mult) == 66, len(s_mult)

max_mult = max(abs(m) for row in xy_mult for m in row)
max_s_mult = max(abs(m) for row in s_mult for m in row)
mult_ty = "i8" if max_mult <= 127 else "i16"
print(f"parsed: xy freqs=1309 amps=4755, s terms=66; max|xy mult|={max_mult} ({mult_ty}), max|s mult|={max_s_mult}")

# --------------------------------------------------------------------------- #
# Validate in Python against pyerfa (ERFA) and SOFA golden vectors
# --------------------------------------------------------------------------- #
def eval_xy(t: float) -> tuple[float, float]:
    fa = [
        erfa.fal03(t), erfa.falp03(t), erfa.faf03(t), erfa.fad03(t), erfa.faom03(t),
        erfa.fame03(t), erfa.fave03(t), erfa.fae03(t), erfa.fama03(t), erfa.faju03(t),
        erfa.fasa03(t), erfa.faur03(t), erfa.fane03(t), erfa.fapa03(t),
    ]
    pt = [t**k for k in range(6)]
    xpr = sum(float(xyp[0][j]) * pt[j] for j in range(6))
    ypr = sum(float(xyp[1][j]) * pt[j] for j in range(6))
    xacc = [0.0, 0.0]
    import math
    for p in range(1309):
        arg = sum(xy_mult[p][k] * fa[k] for k in range(14) if xy_mult[p][k])
        sc = (math.sin(arg), math.cos(arg))
        for idx in range(xy_offset[p], xy_offset[p + 1]):
            code = xy_code[idx]
            xy, ssel, pw = code & 1, (code >> 1) & 1, code >> 2
            xacc[xy] += a_val[idx] * sc[ssel] * pt[pw]
    x = DAS2R * (xpr + xacc[0] / 1e6)
    y = DAS2R * (ypr + xacc[1] / 1e6)
    return x, y


def eval_s(t: float, x: float, y: float) -> float:
    import math
    fa = [
        erfa.fal03(t), erfa.falp03(t), erfa.faf03(t), erfa.fad03(t), erfa.faom03(t),
        erfa.fave03(t), erfa.fae03(t), erfa.fapa03(t),
    ]
    w = [float(v) for v in s_poly]
    for i in range(66):
        arg = sum(s_mult[i][k] * fa[k] for k in range(8) if s_mult[i][k])
        w[s_pt[i]] += float(s_sin[i]) * math.sin(arg) + float(s_cos[i]) * math.cos(arg)
    poly = (w[0] + (w[1] + (w[2] + (w[3] + (w[4] + w[5] * t) * t) * t) * t) * t) * DAS2R
    return poly - x * y / 2.0


t = ((GOLDEN_JD[0] - DJ00) + GOLDEN_JD[1]) / DJC
mx, my = eval_xy(t)
ex, ey = erfa.xy06(*GOLDEN_JD)
ms = eval_s(t, *GOLDEN_S_XY)
es = erfa.s06(GOLDEN_JD[0], GOLDEN_JD[1], *GOLDEN_S_XY)

print(f"X: mine={mx:.19e} erfa={float(ex):.19e} golden={GOLDEN_XY[0]:.19e}")
print(f"   d(erfa)={abs(mx-float(ex)):.2e}  d(golden)={abs(mx-GOLDEN_XY[0]):.2e}")
print(f"Y: mine={my:.19e} erfa={float(ey):.19e} golden={GOLDEN_XY[1]:.19e}")
print(f"   d(erfa)={abs(my-float(ey)):.2e}  d(golden)={abs(my-GOLDEN_XY[1]):.2e}")
print(f"s: mine={ms:.19e} erfa={float(es):.19e} golden={GOLDEN_S:.19e}")
print(f"   d(erfa)={abs(ms-float(es)):.2e}  d(golden)={abs(ms-GOLDEN_S):.2e}")

TOL = 5e-18
assert abs(mx - float(ex)) < TOL and abs(mx - GOLDEN_XY[0]) < TOL, "X mismatch"
assert abs(my - float(ey)) < TOL and abs(my - GOLDEN_XY[1]) < TOL, "Y mismatch"
assert abs(ms - float(es)) < TOL and abs(ms - GOLDEN_S) < TOL, "s mismatch"
print("VALIDATION PASSED — tables reproduce ERFA + SOFA golden vectors to < 5e-18")

# --------------------------------------------------------------------------- #
# Emit Rust
# --------------------------------------------------------------------------- #
def fmt_rows(rows: list[list], per_line: int, fmt) -> str:
    out = []
    for r in rows:
        out.append("    [" + ", ".join(fmt(v) for v in r) + "],")
    return "\n".join(out)


def fmt_flat(vals: list, per_line: int, fmt) -> str:
    out = []
    for i in range(0, len(vals), per_line):
        out.append("    " + ", ".join(fmt(v) for v in vals[i : i + per_line]) + ",")
    return "\n".join(out)


lines = [
    "// @generated by derivations/cio06_tables.py — DO NOT EDIT BY HAND.",
    "//",
    "// IAU 2006/2000A CIP X, Y series and CIO locator s series. Coefficients are the",
    "// IERS-published model constants (IERS Conventions 2010, Technical Note 36,",
    "// tables 5.2a/5.2b for X,Y and 5.2c/5.2d for s; Capitaine, Wallace & Chapront",
    "// 2003 A&A 412, 567; Mathews, Herring & Buffett 2002 MHB2000). Validated against",
    "// pyerfa (ERFA, BSD-3-Clause) and the SOFA t_xy06 / t_s06 self-test vectors to",
    "// < 5e-18 rad. spacelab does not link SOFA or ERFA.",
    "",
    "// Verbatim IERS/SOFA numeric data: keep the published digits, no digit grouping.",
    "#![allow(clippy::excessive_precision, clippy::unreadable_literal,",
    "    clippy::large_const_arrays, clippy::approx_constant)]",
    "#![cfg_attr(rustfmt, rustfmt::skip)]",
    "",
    "/// Polynomial part of X, Y in arcseconds; row 0 = X, row 1 = Y; columns t^0..t^5.",
    "pub(crate) const XYP: [[f64; 6]; 2] = [",
    fmt_rows([list(xyp[0]), list(xyp[1])], 6, str),
    "];",
    "",
    "/// Fundamental-argument multipliers per frequency (14 args: l, l', F, D, Ω,",
    "/// Mercury..Neptune longitudes, pA). Luni-solar frequencies use only args 0..5.",
    f"pub(crate) const XY_MULT: [[{mult_ty}; 14]; 1309] = [",
    fmt_rows(xy_mult, 14, str),
    "];",
    "",
    "/// Start offset into `XY_CODE` / `XY_AMP` for each frequency; last entry = 4755.",
    "pub(crate) const XY_OFFSET: [u16; 1310] = [",
    fmt_flat(xy_offset, 20, str),
    "];",
    "",
    "/// Packed amplitude descriptor: bit0 = X(0)/Y(1), bit1 = sin(0)/cos(1),",
    "/// bits2.. = power of t (0..4).",
    "pub(crate) const XY_CODE: [u8; 4755] = [",
    fmt_flat(xy_code, 30, str),
    "];",
    "",
    "/// Amplitudes in microarcseconds, indexed by `XY_OFFSET`.",
    "pub(crate) const XY_AMP: [f64; 4755] = [",
    fmt_flat(xy_amp_tok, 8, str),
    "];",
    "",
    "/// CIO locator s polynomial in arcseconds, t^0..t^5.",
    "pub(crate) const S_POLY: [f64; 6] = [",
    "    " + ", ".join(str(v) for v in s_poly) + ",",
    "];",
    "",
    "/// CIO locator s series multipliers (8 args: l, l', F, D, Ω, Venus, Earth, pA).",
    f"pub(crate) const S_MULT: [[{'i8' if max_s_mult <= 127 else 'i16'}; 8]; 66] = [",
    fmt_rows(s_mult, 8, str),
    "];",
    "",
    "/// Power of t (0..4) for each s series term.",
    "pub(crate) const S_PT: [u8; 66] = [",
    fmt_flat(s_pt, 22, str),
    "];",
    "",
    "/// Sine coefficients (arcseconds) for each s series term.",
    "pub(crate) const S_SIN: [f64; 66] = [",
    fmt_flat(s_sin, 6, str),
    "];",
    "",
    "/// Cosine coefficients (arcseconds) for each s series term.",
    "pub(crate) const S_COS: [f64; 66] = [",
    fmt_flat(s_cos, 6, str),
    "];",
    "",
]
OUT.write_text("\n".join(lines))
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")
