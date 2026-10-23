#!/usr/bin/env python
"""Generate the IAU 2000A nutation (Δψ, Δε) coefficient tables.

The IAU 2000A nutation is the MHB2000 model (Mathews, Herring & Buffett 2002,
J.Geophys.Res. 107, B4), a luni-solar series plus a planetary series, adopted by
the IAU as the nutation part of the IAU 2000A precession-nutation and republished
in the IERS Conventions (2010, Technical Note 36, tables 5.3a/5.3b). A
machine-readable copy of these same numbers ships in the SOFA/ERFA `nut00a.c`
source; this script extracts the two static tables and emits a Rust `const`
module.

Provenance / licensing: the numbers are the IERS/IAU (MHB2000) model coefficients
(facts), not any implementation's original expression. They are cross-checked here
against pyerfa (ERFA, BSD-3-Clause) at a spread of epochs and against the SOFA
self-test vectors before emission, and the generated Rust attributes the model to
IERS. spacelab never links SOFA or ERFA.

The luni-solar and planetary blocks use *slightly different* Delaunay-type
arguments, faithfully reproduced from `nut00a.c` (see the MHB2000 note in that
file): the luni-solar l' and D and the planetary l, F, D, Ω and Neptune longitude
are MHB2000 polynomials that differ from the IERS-2003 `fundamental.rs` versions
in their last digits. Those seven small argument polynomials are transcribed here,
validated as part of the full-series comparison below (a wrong digit fails the
< 1e-18 rad check), and emitted alongside the tables.

Run:
    timeout 180 uv run --no-project --with pyerfa --with numpy python derivations/nut00a_tables.py

Writes crates/spacelab-fa/src/nut_tables.rs and prints the validation residuals.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import erfa

SOFA_SRC = Path("/home/ilgrad/Documents/projects/space_work/repos/sofa_c-20231011/sofa/20231011/c/src")
OUT = Path(__file__).resolve().parent.parent / "crates/spacelab-fa/src/nut_tables.rs"

DAS2R = 4.848136811095359935899141e-6  # arcsec -> rad (sofam.h)
U2R = DAS2R / 1e7  # 0.1 microarcsecond -> rad
TURNAS = 1_296_000.0  # arcsec in a full turn (sofam.h)
D2PI = 2.0 * math.pi
DJ00 = 2451545.0
DJC = 36525.0

# SOFA t_nut00a / t_nut06a golden vectors (t_sofa_c.c) at TT JD 2400000.5 + 53736.0.
GOLDEN_JD = (2400000.5, 53736.0)
GOLDEN_NUT00A = (-0.9630909107115518431e-5, 0.4063239174001678710e-4)  # dpsi, deps
GOLDEN_NUT06A = (-0.9630912025820308797e-5, 0.4063238496887249798e-4)  # dpsi, deps

# MHB2000 argument polynomials, transcribed verbatim from nut00a.c and kept as the
# source digit strings for provenance. Validated by the full-series comparison.
# Luni-solar l' and D: arcsec polynomials in t (t^0..t^4); reduce fmod(.,TURNAS)*DAS2R.
LS_ELP_TOK = ["1287104.79305", "129596581.0481", "-0.5532", "0.000136", "-0.00001149"]
LS_D_TOK = ["1072260.70369", "1602961601.2090", "-6.3706", "0.006593", "-0.00003169"]
# Planetary fast arguments: radians, linear a + b*t; reduce fmod(.,2*pi). [a, b]
PL_AL_TOK = ["2.35555598", "8328.6914269554"]
PL_AF_TOK = ["1.627905234", "8433.466158131"]
PL_AD_TOK = ["5.198466741", "7771.3771468121"]
PL_AOM_TOK = ["2.18243920", "-33.757045"]
PL_ANE_TOK = ["5.321159000", "3.8127774000"]


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


def row_tokens(group: str) -> list[str]:
    return [tok.strip() for tok in group.split(",")]


# --------------------------------------------------------------------------- #
# Parse nut00a.c
# --------------------------------------------------------------------------- #
src = strip_comments((SOFA_SRC / "nut00a.c").read_text())

xls_groups = top_level_groups(body_after(src, r"\}\s*xls\[\]"))
xpl_groups = top_level_groups(body_after(src, r"\}\s*xpl\[\]"))
NLS = len(xls_groups)
NPL = len(xpl_groups)
assert NLS == 678, NLS
assert NPL == 687, NPL

# Luni-solar: 5 integer multipliers (nl,nlp,nf,nd,nom) + 6 double coefficients
# (sp,spt,cp, ce,cet,se). The doubles are integer-valued but declared `double`;
# keep the exact source tokens.
xls_mult: list[list[int]] = []
xls_coef_tok: list[list[str]] = []
for g in xls_groups:
    t = row_tokens(g)
    assert len(t) == 11, (len(t), g)
    xls_mult.append([int(x) for x in t[:5]])
    xls_coef_tok.append(t[5:])

# Planetary: 13 integer multipliers + 4 integer coefficients (sp,cp, se,ce).
xpl_mult: list[list[int]] = []
xpl_coef: list[list[int]] = []
for g in xpl_groups:
    t = row_tokens(g)
    assert len(t) == 17, (len(t), g)
    xpl_mult.append([int(x) for x in t[:13]])
    xpl_coef.append([int(x) for x in t[13:]])

max_xls_mult = max(abs(m) for row in xls_mult for m in row)
max_xpl_mult = max(abs(m) for row in xpl_mult for m in row)
max_xpl_coef = max(abs(c) for row in xpl_coef for c in row)
xls_mult_ty = "i8" if max_xls_mult <= 127 else "i16"
xpl_mult_ty = "i8" if max_xpl_mult <= 127 else "i16"
xpl_coef_ty = "i16" if max_xpl_coef <= 32767 else "i32"
print(f"parsed: NLS={NLS} luni-solar, NPL={NPL} planetary terms")
print(
    f"  max|xls mult|={max_xls_mult} ({xls_mult_ty}), max|xpl mult|={max_xpl_mult} "
    f"({xpl_mult_ty}), max|xpl coef|={max_xpl_coef} ({xpl_coef_ty})"
)


# --------------------------------------------------------------------------- #
# Evaluate the series exactly as nut00a.c does (for validation)
# --------------------------------------------------------------------------- #
def _poly4_das2r(tok: list[str], t: float) -> float:
    c = [float(x) for x in tok]
    a = c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])))
    return math.fmod(a, TURNAS) * DAS2R


def _linear_tau(tok: list[str], t: float) -> float:
    c = [float(x) for x in tok]
    return math.fmod(c[0] + c[1] * t, D2PI)


def eval_nut00a(jd1: float, jd2: float) -> tuple[float, float]:
    t = ((jd1 - DJ00) + jd2) / DJC

    # Luni-solar arguments: l, F, Ω from IERS 2003 (== fundamental.rs); l', D MHB2000.
    el = erfa.fal03(t)
    elp = _poly4_das2r(LS_ELP_TOK, t)
    f = erfa.faf03(t)
    d = _poly4_das2r(LS_D_TOK, t)
    om = erfa.faom03(t)
    ls = (el, elp, f, d, om)

    dp = 0.0
    de = 0.0
    for i in range(NLS - 1, -1, -1):  # reverse order, exactly as nut00a.c
        m = xls_mult[i]
        arg = math.fmod(m[0] * ls[0] + m[1] * ls[1] + m[2] * ls[2] + m[3] * ls[3] + m[4] * ls[4], D2PI)
        sarg = math.sin(arg)
        carg = math.cos(arg)
        c = [float(x) for x in xls_coef_tok[i]]
        dp += (c[0] + c[1] * t) * sarg + c[2] * carg
        de += (c[3] + c[4] * t) * carg + c[5] * sarg
    dpsils = dp * U2R
    depsls = de * U2R

    # Planetary arguments: l, F, D, Ω, Neptune MHB2000; Me..Ur and pA from IERS 2003.
    al = _linear_tau(PL_AL_TOK, t)
    af = _linear_tau(PL_AF_TOK, t)
    ad = _linear_tau(PL_AD_TOK, t)
    aom = _linear_tau(PL_AOM_TOK, t)
    apa = erfa.fapa03(t)
    alme = erfa.fame03(t)
    alve = erfa.fave03(t)
    alea = erfa.fae03(t)
    alma = erfa.fama03(t)
    alju = erfa.faju03(t)
    alsa = erfa.fasa03(t)
    alur = erfa.faur03(t)
    alne = _linear_tau(PL_ANE_TOK, t)
    pl = (al, af, ad, aom, alme, alve, alea, alma, alju, alsa, alur, alne, apa)

    dp = 0.0
    de = 0.0
    for i in range(NPL - 1, -1, -1):  # reverse order, exactly as nut00a.c
        m = xpl_mult[i]
        arg = math.fmod(sum(m[k] * pl[k] for k in range(13)), D2PI)
        sarg = math.sin(arg)
        carg = math.cos(arg)
        c = xpl_coef[i]
        dp += c[0] * sarg + c[1] * carg
        de += c[2] * sarg + c[3] * carg
    dpsipl = dp * U2R
    depspl = de * U2R

    return dpsils + dpsipl, depsls + depspl


def eval_nut06a(jd1: float, jd2: float) -> tuple[float, float]:
    t = ((jd1 - DJ00) + jd2) / DJC
    fj2 = -2.7774e-6 * t
    dp, de = eval_nut00a(jd1, jd2)
    return dp + dp * (0.4697e-6 + fj2), de + de * fj2


# --------------------------------------------------------------------------- #
# Validate vs pyerfa (ERFA) across epochs, and vs the SOFA golden vectors
# --------------------------------------------------------------------------- #
TOL = 1e-18
worst = 0.0
worst_epoch = 0.0
for t in (-3.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.06, 0.1, 0.5, 1.0, 2.0, 3.0):
    jd2 = t * DJC
    m00 = eval_nut00a(DJ00, jd2)
    e00 = erfa.nut00a(DJ00, jd2)
    m06 = eval_nut06a(DJ00, jd2)
    e06 = erfa.nut06a(DJ00, jd2)
    for a, b in ((m00[0], e00[0]), (m00[1], e00[1]), (m06[0], e06[0]), (m06[1], e06[1])):
        d = abs(a - float(b))
        if d > worst:
            worst, worst_epoch = d, t
print(f"vs pyerfa (nut00a + nut06a), 12 epochs t∈[-3,3]: worst |Δ| = {worst:.3e} rad at t={worst_epoch}")

m00 = eval_nut00a(*GOLDEN_JD)
m06 = eval_nut06a(*GOLDEN_JD)
e00 = erfa.nut00a(*GOLDEN_JD)
e06 = erfa.nut06a(*GOLDEN_JD)
print("SOFA epoch (JD 2400000.5 + 53736.0):")
print(f"  nut00a dpsi: mine={m00[0]:.19e} erfa={float(e00[0]):.19e} golden={GOLDEN_NUT00A[0]:.19e}")
print(f"         d(erfa)={abs(m00[0]-float(e00[0])):.2e}  d(golden)={abs(m00[0]-GOLDEN_NUT00A[0]):.2e}")
print(f"  nut00a deps: mine={m00[1]:.19e} erfa={float(e00[1]):.19e} golden={GOLDEN_NUT00A[1]:.19e}")
print(f"         d(erfa)={abs(m00[1]-float(e00[1])):.2e}  d(golden)={abs(m00[1]-GOLDEN_NUT00A[1]):.2e}")
print(f"  nut06a dpsi: mine={m06[0]:.19e} erfa={float(e06[0]):.19e} golden={GOLDEN_NUT06A[0]:.19e}")
print(f"         d(erfa)={abs(m06[0]-float(e06[0])):.2e}  d(golden)={abs(m06[0]-GOLDEN_NUT06A[0]):.2e}")
print(f"  nut06a deps: mine={m06[1]:.19e} erfa={float(e06[1]):.19e} golden={GOLDEN_NUT06A[1]:.19e}")
print(f"         d(erfa)={abs(m06[1]-float(e06[1])):.2e}  d(golden)={abs(m06[1]-GOLDEN_NUT06A[1]):.2e}")

assert worst < TOL, f"series vs pyerfa exceeds {TOL}: {worst}"
for mine, gold in ((m00, GOLDEN_NUT00A), (m06, GOLDEN_NUT06A)):
    assert abs(mine[0] - gold[0]) < 1e-13 and abs(mine[1] - gold[1]) < 1e-13
print(f"VALIDATION PASSED — series reproduces ERFA to < {TOL} rad and the SOFA golden vectors to < 1e-13")

# Reference for the spacelab-fa CIO completeness test: the classical (angles) route
# xys06a builds X, Y from the FULL IAU 2000A nutation. Printed here so cio.rs can
# hardcode it and assert cip_xy (the X,Y series) agrees to ~µas — proving the series
# carries the full 2000A nutation rather than a truncation.
xa, ya, sa = erfa.xys06a(*GOLDEN_JD)
print(f"erfa.xys06a @ SOFA epoch: X={float(xa):.19e} Y={float(ya):.19e} s={float(sa):.19e}")


# --------------------------------------------------------------------------- #
# Emit Rust
# --------------------------------------------------------------------------- #
def fmt_rows(rows: list[list], fmt) -> str:
    return "\n".join("    [" + ", ".join(fmt(v) for v in r) + "]," for r in rows)


def fmt_pair(tok: list[str]) -> str:
    return "[" + ", ".join(tok) + "]"


lines = [
    "// @generated by derivations/nut00a_tables.py — DO NOT EDIT BY HAND.",
    "//",
    "// IAU 2000A nutation (MHB2000) luni-solar and planetary series in longitude and",
    "// obliquity (Δψ, Δε). Coefficients are the IERS/IAU-published model constants",
    "// (Mathews, Herring & Buffett 2002 MHB2000; IERS Conventions 2010, Technical",
    "// Note 36, tables 5.3a/5.3b). Validated against pyerfa (ERFA, BSD-3-Clause) over",
    "// 12 epochs and the SOFA t_nut00a / t_nut06a self-test vectors to < 1e-18 rad.",
    "// spacelab does not link SOFA or ERFA.",
    "//",
    "// The seven MHB2000 argument polynomials below are transcribed from nut00a.c and",
    "// validated as part of that < 1e-18 rad series check; they differ in their last",
    "// digits from the IERS-2003 arguments in `fundamental.rs`, exactly as MHB2000",
    "// specifies for the two blocks.",
    "",
    "// Verbatim IERS/SOFA numeric data: keep the published digits, no digit grouping.",
    "#![allow(clippy::excessive_precision, clippy::unreadable_literal,",
    "    clippy::large_const_arrays, clippy::approx_constant)]",
    "#![cfg_attr(rustfmt, rustfmt::skip)]",
    "",
    "/// Luni-solar term fundamental-argument multipliers (l, l', F, D, Ω).",
    f"pub(crate) const XLS_MULT: [[{xls_mult_ty}; 5]; {NLS}] = [",
    fmt_rows(xls_mult, str),
    "];",
    "",
    "/// Luni-solar coefficients in units of 0.1 µas: for Δψ the (sin, t·sin, cos)",
    "/// triple, for Δε the (cos, t·cos, sin) triple — order [sp, spt, cp, ce, cet, se].",
    f"pub(crate) const XLS_COEF: [[f64; 6]; {NLS}] = [",
    fmt_rows(xls_coef_tok, str),
    "];",
    "",
    "/// Planetary term fundamental-argument multipliers",
    "/// (l, F, D, Ω, Me, Ve, Ea, Ma, Ju, Sa, Ur, Ne, pA).",
    f"pub(crate) const XPL_MULT: [[{xpl_mult_ty}; 13]; {NPL}] = [",
    fmt_rows(xpl_mult, str),
    "];",
    "",
    "/// Planetary coefficients in units of 0.1 µas — order [Δψ sin, Δψ cos, Δε sin,",
    "/// Δε cos] i.e. [sp, cp, se, ce].",
    f"pub(crate) const XPL_COEF: [[{xpl_coef_ty}; 4]; {NPL}] = [",
    fmt_rows(xpl_coef, str),
    "];",
    "",
    "// --- MHB2000 arguments (see module header) -------------------------------- //",
    "",
    "/// Luni-solar l' (mean anomaly of the Sun), MHB2000: arcsec polynomial in t,",
    "/// powers t⁰..t⁴; reduce `fmod(·, 1 296 000)` then scale by arcsec→rad.",
    f"pub(crate) const LS_ELP: [f64; 5] = {fmt_pair(LS_ELP_TOK)};",
    "",
    "/// Luni-solar D (mean elongation of the Moon from the Sun), MHB2000; same form.",
    f"pub(crate) const LS_D: [f64; 5] = {fmt_pair(LS_D_TOK)};",
    "",
    "/// Planetary l (mean anomaly of the Moon), MHB2000: radians, `a + b·t`,",
    "/// reduce `fmod(·, 2π)`. Layout `[a, b]`.",
    f"pub(crate) const PL_AL: [f64; 2] = {fmt_pair(PL_AL_TOK)};",
    "",
    "/// Planetary F (Moon latitude argument), MHB2000; `[a, b]`, radians.",
    f"pub(crate) const PL_AF: [f64; 2] = {fmt_pair(PL_AF_TOK)};",
    "",
    "/// Planetary D (Moon elongation), MHB2000; `[a, b]`, radians.",
    f"pub(crate) const PL_AD: [f64; 2] = {fmt_pair(PL_AD_TOK)};",
    "",
    "/// Planetary Ω (Moon node), MHB2000; `[a, b]`, radians.",
    f"pub(crate) const PL_AOM: [f64; 2] = {fmt_pair(PL_AOM_TOK)};",
    "",
    "/// Neptune mean longitude, MHB2000; `[a, b]`, radians.",
    f"pub(crate) const PL_ANE: [f64; 2] = {fmt_pair(PL_ANE_TOK)};",
    "",
]
OUT.write_text("\n".join(lines))
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")
