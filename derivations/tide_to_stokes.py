#!/usr/bin/env python3
"""Turn a cotidal atlas into ocean-tide geopotential coefficients (IERS 2010 Eq. 6.21).

`ocean_tide_fes2004.py` and `ocean_tide_fes2014.py` transcribe coefficient files somebody else
computed. This script computes them: it reads a tide atlas as it is actually distributed -- a grid of
amplitude and Greenwich phase lag per constituent -- and performs the spherical-harmonic analysis and
the loading conversion itself. That closes the dependency on a third party publishing a `Cnm-Snm`
file for every atlas we want to use, which is the reason no modern open atlas is in the tree: GOT
ships elevation grids and nothing else.

The mathematics is not derived here. `derivations/ocean_tide_stokes.mac` proves the three
convention-bearing steps symbolically -- the analysis integral, the (A+, A-) -> (Cbar, Sbar) map, and
the Doodson-Warburg rotation -- and this script implements what that file prints:

    A_nm(+/-) = 1/(4 pi (2 - delta_m0)) * integral over the sphere of H Pbar_nm exp(-/+ i m lambda)
    S_nm(+/-) + i C_nm(+/-) = i K_n exp(i chi_f) A_nm(+/-)
    K_n = 4 pi G rho_w (1 + kprime_n) / (g_e (2n + 1))

with H = Z exp(-i psi) the complex tide (IERS Eq. 6.17: xi = Z cos(theta_f - psi)) and chi_f the
Doodson-Warburg bias of Table 6.6, which depends on the tidal species and on the *sign* of the
constituent's Cartwright-Edden amplitude. That sign is read from the IERS Conventions Centre's own
`ADMINT.F`, the same file `derivations/hardisp_tables.py` reads, rather than recalled.

DEGREE 6, AND WHY NOT MORE

IERS 2010 section 6.3.1 publishes the load deformation coefficients kprime_n for n = 2..6 and no
further, so degree 6 is where the conventional constant set stops. The analysis itself is exact to
degree 719 on a 0.125-degree grid; the cap is the physics input, not the quadrature, and lifting it
needs a primary source for kprime_n at higher degree (Farrell 1972) that this repository does not
have. Emitting a degree-20 table on a guessed kprime_n would be worse than not emitting one.

QUADRATURE

The atlas grid is equiangular, which is the one case with an exact quadrature: Driscoll & Healy
(1994) weights in latitude are exact for band limits below nlat/2, and the longitude integral on a
uniform periodic grid is a discrete Fourier transform, exact for band limits below nlon/2. So the
analysis introduces no truncation error of its own at degree 6 -- `--self-test` measures that claim
rather than asserting it, by analysing fields whose coefficients are known exactly.

Land is zero, not missing. The integral in Eq. 6.21 is over the ocean; a fill value is the statement
that there is no ocean there, which is the value zero for the mass that redistributes.

TWO ATLASES

GOT5.5 (R. D. Ray, NASA GSFC) and EOT20 (DGFI-TUM) ship the same quantity on the same 1/8-degree
grid and differ in everything a reader could get wrong: the variable names, the spelling of the
units, the file stems, the fill value, and whether the 0-degree meridian is repeated at 360.
`ATLASES` states each atlas's conventions, and `read_grid` checks a file against them instead of
guessing from it. EOT20's repeat is not a copy: in every one of its 17 grids it carries the land
fill on 232 rows where 0 degrees is ocean, all of them poleward of 66 degrees, where FES2014b fills
EOT20. The reader keeps 0 degrees and drops the repeat, after checking that it differs nowhere else.

h5py has no wheel for Python 3.15 yet, hence the pinned interpreter.

    timeout 600 uv run --no-project --python 3.14 --with numpy --with h5py python \
        derivations/tide_to_stokes.py --self-test
    timeout 1800 uv run --no-project --python 3.14 --with numpy --with h5py python \
        derivations/tide_to_stokes.py --atlas got5.5 ~/.cache/spacelab/tides/GOT5.5 \
            crates/spacelab-propagation/src/ocean_tide_got.rs
    timeout 1800 uv run --no-project --python 3.14 --with numpy --with h5py python \
        derivations/tide_to_stokes.py --atlas eot20 ~/.cache/spacelab/tides/EOT20/ocean_tides_x \
            crates/spacelab-propagation/src/ocean_tide_eot20.rs
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys
import urllib.request

import numpy as np

# ── IERS Conventions 2010 constants ────────────────────────────────────────────────────────────────
G_GRAV = 6.67428e-11  # Table 1.1, m^3 kg^-1 s^-2
RHO_W = 1025.0  # section 6.3.1, seawater density, kg m^-3
G_E = 9.7803278  # Table 1.1, mean equatorial gravity, m s^-2
K_PRIME = {2: -0.3075, 3: -0.195, 4: -0.132, 5: -0.1032, 6: -0.0892}  # section 6.3.1
MAX_DEGREE = 6
SCALE_LITERAL = "1.0e-11"
SCALE = 1.0e-11

ADMINT = "https://iers-conventions.obspm.fr/content/chapter7/software/hardisp/ADMINT.F"

# Constituent name -> Doodson number. Cross-checked against the Doodson/Darwin pairs the shipped
# FES2004 table carries in its own comments, so the two tables key the same wave the same way.
DOODSON = {
    # long period
    "sa": 56554, "ssa": 57555, "mm": 65455, "mf": 75555, "mt": 85455,
    # diurnal
    "2q1": 125755, "sig1": 127555, "q1": 135655, "rho1": 137455, "o1": 145555,
    "tau1": 147555, "m1": 155655, "chi1": 157455, "pi1": 162556, "p1": 163555,
    "s1": 164556, "k1": 165555, "psi1": 166554, "phi1": 167555, "theta1": 173655,
    "j1": 175455, "so1": 183555, "oo1": 185555,
    # semi-diurnal
    "eps2": 227655, "2n2": 235755, "mu2": 237555, "n2": 245655, "nu2": 247455,
    "m2": 255555, "lambda2": 263655, "l2": 265455, "t2": 272556, "s2": 273555,
    "r2": 274554, "k2": 275555, "eta2": 285455,
    # quarter-diurnal
    "m4": 455555,
}
"""Constituent file stem -> Doodson number.

Not recalled: the major waves are the Doodson/Darwin pairs the shipped FES2004 table carries in its
own comments, and the minor ones are the numbers `derivations/ocean_tide_admittance.py` obtained from
`pyfes` and emitted into `ocean_tide_admittance.rs`. The two sets overlap on J1, L2, Mu2, Nu2, OO1,
R2 and T2 and agree there, which is the only reason to trust either.

M4 is listed so that it is *recognised* and then skipped for a stated reason rather than skipped for
want of an entry; `chi_bias` refuses it, because Table 6.6 has no row for a shallow-water compound
tide. MS4, GOT's other quarter-diurnal, is absent for the same reason one step earlier. GOT's
third-degree supplements (l2', n2', m1', m3) are deliberately absent too: their conversion is not
Eq. 6.21, and the exact stem match keeps them from being taken for their second-degree namesakes.
"""


# ── Cartwright-Edden amplitude signs, from the IERS Conventions Centre's ADMINT.F ──────────────────
def admint_source(cache: pathlib.Path) -> str:
    """`ADMINT.F`, fetched once and cached outside the repository."""
    if cache.exists():
        return cache.read_text(encoding="latin-1")
    with urllib.request.urlopen(ADMINT, timeout=60) as response:  # pinned https constant
        text = response.read().decode("latin-1")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="latin-1")
    return text


def data_block(source: str, name: str) -> str:
    """The text between `DATA <name>/` and the closing slash, continuation markers removed."""
    start = source.index(f"DATA {name}/") + len(f"DATA {name}/")
    end = source.index("/", start)
    return "\n".join(
        line[6:] if line.startswith("     .") else line
        for line in source[start:end].splitlines()
    )


def cartwright_signs(source: str) -> dict[int, float]:
    """{Doodson number: Cartwright-Edden amplitude, metres, signed}."""
    amps = [float(v) for v in re.findall(r"-?\.\d+", data_block(source, "TAMP"))]
    mults = [int(v) for v in re.findall(r"-?\d+", data_block(source, "IDD"))]
    if len(mults) != 6 * len(amps):
        raise SystemExit(f"ADMINT.F: {len(amps)} amplitudes but {len(mults)} multipliers")
    out: dict[int, float] = {}
    for i, amp in enumerate(amps):
        c = mults[6 * i : 6 * i + 6]
        doodson = c[0] * 100000 + sum((c[k] + 5) * 10 ** (5 - k) for k in range(1, 6))
        # The catalogue lists a constituent once; a repeat would mean the decode is wrong.
        if doodson in out and out[doodson] != amp:
            raise SystemExit(f"ADMINT.F: Doodson {doodson} carries two amplitudes")
        out[doodson] = amp
    return out


def chi_bias(doodson: int, amplitude: float) -> float | None:
    """The Doodson-Warburg phase bias chi_f, IERS 2010 Table 6.6, or `None` if the table has no row.

    Table 6.6 covers the long-period, diurnal and semi-diurnal species and stops there, so a
    shallow-water compound tide such as M4 (species 4) has no bias to apply and no equilibrium
    amplitude whose sign could supply one. That is a gap in the conventions, not in this script, and
    inventing a value for it would put an unattributable number in the table -- so the constituent is
    skipped and said to be skipped.
    """
    species = doodson // 100000
    positive = amplitude > 0.0
    if species == 0:
        return np.pi if positive else 0.0
    if species == 1:
        return np.pi / 2 if positive else -np.pi / 2
    if species == 2:
        return 0.0 if positive else np.pi
    return None


# Pbar_nm at sin(phi) = 3/10, from Rodrigues' formula in `derivations/ocean_tide_stokes.mac` at 25
# digits. Orthonormality does not pin a sign -- flip any single Pbar_nm and it still holds, and the
# synthetic-analysis test below builds its field from the same recursion it analyses, so a sign error
# cancels there too. Only these values, built by a different formula in a different program, catch it.
PBAR_AT_0_3 = {
    (0, 0): 1.0,
    (1, 0): 0.5196152422706631880582339,
    (1, 1): 1.652271164185830606181743,
    (2, 0): -0.8161648117874232391893484,
    (2, 1): 1.108377192114669959749068,
    (2, 2): 1.762207422524374682756566,
    (3, 0): -1.011999876482205900866868,
    (3, 1): -0.8500569833840552597710391,
    (3, 2): 1.398708779553485179309672,
    (3, 3): 1.815729364470377231774158,
    (4, 0): 0.2188125,
    (4, 1): -1.608612948707674676599456,
    (4, 2): -0.5646630660181343930857266,
    (4, 3): 1.634156428023339508596742,
    (4, 4): 1.837164625644297016818752,
    (5, 0): 1.145516598997887721136373,
    (5, 1): -0.1376995030798495655922013,
    (5, 2): -1.693232157655514762518859,
    (5, 3): -0.2860494185819886437617035,
    (5, 4): 1.827955722412761971657218,
    (5, 5): 1.838082978426292034846937,
    (6, 0): 0.4657693953565777503724266,
    (6, 1): 1.517434340560176223073416,
    (6, 2): -0.5240566027804097237725102,
    (6, 3): -1.70938835669766824567432,
    (6, 4): -0.01481169473800073797618661,
    (6, 5): 1.988190728182069668485762,
    (6, 6): 1.82501678282047919072281,
}


# ── Spherical-harmonic machinery ───────────────────────────────────────────────────────────────────
def legendre(max_degree: int, sin_lat: np.ndarray) -> np.ndarray:
    """`Pbar[n, m, :]`, 4pi-normalised, no Condon-Shortley, by forward-column recursion.

    Same convention as `tools/gen_shbasis_oracle.mac`, which builds it from Rodrigues' formula
    instead; `--self-test` measures the two against each other through orthonormality.
    """
    t = sin_lat
    u = np.sqrt(np.maximum(0.0, 1.0 - t * t))
    p = np.zeros((max_degree + 1, max_degree + 1, t.size))
    p[0, 0] = 1.0
    for m in range(1, max_degree + 1):
        p[m, m] = (np.sqrt(3.0) * u if m == 1 else u * np.sqrt((2 * m + 1) / (2 * m)) * p[m - 1, m - 1])
    for m in range(max_degree + 1):
        for n in range(m + 1, max_degree + 1):
            a = np.sqrt((2 * n - 1) * (2 * n + 1) / ((n - m) * (n + m)))
            p[n, m] = a * t * p[n - 1, m]
            if n - 2 >= m:
                b = np.sqrt((2 * n + 1) * (n + m - 1) * (n - m - 1) / ((n - m) * (n + m) * (2 * n - 3)))
                p[n, m] -= b * p[n - 2, m]
    return p


def dh_weights(nlat: int) -> np.ndarray:
    """Driscoll & Healy (1994) latitude weights for `nlat` equiangular colatitudes from 0.

    `sum(w) == 2` is the integral of sin(theta) over [0, pi]; `--self-test` checks it.
    """
    j = np.arange(nlat)
    theta = np.pi * j / nlat
    k = np.arange(nlat // 2)
    terms = np.sin(np.outer(theta, 2 * k + 1)) / (2 * k + 1)
    return (4.0 / nlat) * np.sin(theta) * terms.sum(axis=1)


def analyse(field: np.ndarray, sin_lat: np.ndarray, max_degree: int) -> np.ndarray:
    """`A[n, m, sign]` with sign 0 = prograde (+), 1 = retrograde (-), for a complex field.

    `field` is `[colatitude, longitude]` with colatitude ascending from the north pole and the south
    pole row already dropped -- the sampling Driscoll & Healy weights are defined on.
    """
    nlat, nlon = field.shape
    w = dh_weights(nlat)
    p = legendre(max_degree, sin_lat)
    # (1/nlon) * sum over longitude of f exp(-i m lambda): the forward DFT, exact for m < nlon/2.
    ft = np.fft.fft(field, axis=1) / nlon
    out = np.zeros((max_degree + 1, max_degree + 1, 2), dtype=complex)
    for n in range(max_degree + 1):
        for m in range(n + 1):
            # exp(-i m lambda) is bin +m; exp(+i m lambda) is bin -m.
            for sign, bin_index in ((0, m), (1, -m)):
                if m == 0 and sign == 1:
                    continue  # degenerate at m = 0; the whole term goes in the prograde slot
                integral = 0.5 * np.sum(w * p[n, m] * ft[:, bin_index])
                out[n, m, sign] = integral / (1 if m == 0 else 2)
    return out


# ── The atlas ──────────────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class Atlas:
    """What one atlas's files are called and what they state about themselves."""

    short: str  # the name the generated docs use for the model
    static: str  # the Rust table's name
    latitude: str  # netCDF variable names
    longitude: str
    units: dict[str, str]  # amplitude, phase, latitude, longitude
    title_attr: str  # the global attribute that names the product
    title_mark: str  # text that attribute must contain
    suffix: str  # file stem = constituent name + suffix
    module_doc: str  # the generated module's `//!` block
    spelled: str  # model names the `doc_markdown` allowance covers
    wave_doc: str  # the table's `///` block, `{n}` = the wave count


ATLASES = {
    "got5.5": Atlas(
        short="GOT",
        static="GOT_WAVES",
        latitude="latitude",
        longitude="longitude",
        units={"amplitude": "cm", "phase": "degrees",
               "latitude": "degrees north", "longitude": "degrees east"},
        title_attr="Title",
        title_mark="GOT5.5 ocean tidal elevations",
        suffix="",
        module_doc="""//! GOT5.5 ocean-tide geopotential coefficients (IERS Conventions 2010, Eq. 6.21) --
//! machine-computed by `derivations/tide_to_stokes.py` from the GSFC cotidal grids.
//!
//! [`crate::ocean_tide_fes2004`] and [`crate::ocean_tide_fes2014`] transcribe a coefficient file
//! somebody else computed. These values are this repository's own spherical-harmonic analysis of
//! published *elevation* grids, which is what an atlas actually ships; the three conventions that
//! carry a sign are proved in `derivations/ocean_tide_stokes.mac` before the generator applies them.
//! `tests/ocean_tide_atlas.rs` measures the result against both FES tables.
//!
//! Degree 6, because IERS 2010 section 6.3.1 publishes the load deformation coefficients `k'n` no
//! further; the analysis itself is exact past degree 700 on this grid, so the cap is the physics
//! input and not the quadrature. GOT's two quarter-diurnal compounds (M4, MS4) are absent because
//! Table 6.6 has no phase bias for their species, and its four third-degree supplements (GOT5.6)
//! because their conversion is not Eq. 6.21.
//!
//! The 10^-11 unit is [`SCALE`], applied at use. Regenerate, never hand-edit.""",
        spelled="GOT5.5, FES2014b, MS4",
        wave_doc="""/// The {n} GOT5.5 waves -- the explicitly solved diurnal and semi-diurnal constituents
/// plus the ones GOT publishes as inferred -- to degree = order = 6, sorted by Doodson number so
/// the accumulation order matches the FES tables.""",
    ),
    "eot20": Atlas(
        short="EOT20",
        static="EOT20_WAVES",
        latitude="lat",
        longitude="lon",
        units={"amplitude": "cm", "phase": "degrees",
               "latitude": "degrees_north", "longitude": "degrees_east"},
        title_attr="title",
        # The load-tide grids share the layout and say "load tide component" here.
        title_mark="EOT20, ocean tide component",
        suffix="_ocean_eot20",
        module_doc="""//! EOT20 ocean-tide geopotential coefficients (IERS Conventions 2010, Eq. 6.21) --
//! machine-computed by `derivations/tide_to_stokes.py` from the DGFI-TUM cotidal grids.
//!
//! EOT20 is an empirical atlas: a residual tidal analysis of eleven altimetry missions against
//! FES2014b between 66°S and 66°N, with FES2014b itself filling the higher latitudes. Like
//! [`crate::ocean_tide_got`], it ships elevation grids and no coefficients, so these values are
//! this repository's own spherical-harmonic analysis of them. `tests/ocean_tide_atlas.rs` measures
//! the result against both FES tables, and FES2014b is EOT20's reference model, not an independent
//! atlas.
//!
//! Degree 6, because IERS 2010 section 6.3.1 publishes the load deformation coefficients `k'n` no
//! further. EOT20's one compound tide, M4, is absent because Table 6.6 has no phase bias for its
//! species.
//!
//! Hart-Davis, M., Piccioni, G., Dettmering, D., Schwatke, C., Passaro, M., Seitz, F. (2021):
//! EOT20 - A global Empirical Ocean Tide model from multi-mission satellite altimetry. SEANOE,
//! doi:10.17882/79489, CC BY 4.0. Article: ESSD 13, 3869-3884, doi:10.5194/essd-13-3869-2021.
//!
//! The 10^-11 unit is [`SCALE`], applied at use. Regenerate, never hand-edit.""",
        spelled="EOT20, FES2014b, DGFI-TUM",
        wave_doc="""/// The {n} EOT20 waves to degree = order = 6, sorted by Doodson number so the accumulation order
/// matches the FES tables.""",
    ),
}
"""The atlases `read_grid` knows, by `--atlas` name."""


def read_grid(path: pathlib.Path, atlas: Atlas) -> tuple[np.ndarray, np.ndarray]:
    """`(complex tide in metres [colatitude, longitude], sin(latitude))` from one atlas netCDF file."""
    import h5py

    with h5py.File(path, "r") as fh:
        title = fh.attrs.get(atlas.title_attr, b"")
        title = title.decode() if isinstance(title, bytes) else str(title)
        lat = np.asarray(fh[atlas.latitude], dtype=float)
        lon = np.asarray(fh[atlas.longitude], dtype=float)
        amp = np.asarray(fh["amplitude"], dtype=float)
        pha = np.asarray(fh["phase"], dtype=float)
        fill = float(fh["amplitude"].attrs["_FillValue"][0])
        # The units are read, not assumed: the centimetre and the degree below are conversions, and
        # a file that changed either would otherwise be scaled silently by the wrong factor.
        variables = {"amplitude": "amplitude", "phase": "phase",
                     "latitude": atlas.latitude, "longitude": atlas.longitude}
        units = {k: fh[v].attrs["units"].decode() for k, v in variables.items()}
    if atlas.title_mark not in title:
        raise SystemExit(f"{path.name}: {atlas.title_attr} is {title!r}, expected {atlas.title_mark!r} in it")
    if units != atlas.units:
        raise SystemExit(f"{path.name}: units are {units}, expected {atlas.units}")
    # Equality, not a threshold: GOT's fill is 9.97e36 and EOT20's is 0, and a land cell carries it
    # exactly in both.
    land = (amp == fill) | ~np.isfinite(amp) | ~np.isfinite(pha)
    amp = np.where(land, 0.0, amp) / 100.0  # the file is in centimetres
    pha = np.where(land, 0.0, pha)
    h = amp * np.exp(-1j * np.deg2rad(pha))  # IERS Eq. 6.17: xi = Z cos(theta - psi)
    # `analyse` takes column j to be longitude 360 j / nlon: a periodic grid that starts at 0.
    nlon = round(360.0 / (lon[1] - lon[0]))
    if lon[0] != 0.0 or np.max(np.abs(lon - 360.0 * np.arange(lon.size) / nlon)) > 1e-9:
        raise SystemExit(f"{path.name}: longitudes are not a uniform grid from 0")
    if lon.size == nlon + 1:  # the 0-degree meridian repeated at 360
        differs = h[:, -1] != h[:, 0]
        if np.any(differs & ~land[:, -1]):
            raise SystemExit(f"{path.name}: the 360-degree column is ocean and differs from 0 degrees")
        if np.any(differs):
            rows = lat[differs]
            print(
                f"  {path.name}: 360-degree repeat dropped; it carries the land fill on {rows.size} rows"
                f" ({rows[rows < 0].min():+.3f} to {rows[rows < 0].max():+.3f},"
                f" {rows[rows > 0].min():+.3f} to {rows[rows > 0].max():+.3f} deg)",
                file=sys.stderr,
            )
        h = h[:, :-1]
    elif lon.size != nlon:
        raise SystemExit(f"{path.name}: {lon.size} longitudes for a {360.0 / nlon} degree step")
    if lat[0] < lat[-1]:  # the file runs south to north; the quadrature runs from the north pole
        lat, h = lat[::-1], h[::-1]
    if abs(lat[-1] + 90.0) < 1e-6:  # drop the duplicate south pole: DH samples [0, pi)
        lat, h = lat[:-1], h[:-1]
    return h, np.sin(np.deg2rad(lat))


def coefficients(
    path: pathlib.Path, chi: float, atlas: Atlas
) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """One constituent's `{(n, m): (C+, S+, C-, S-)}` in units of `SCALE`."""
    h, sin_lat = read_grid(path, atlas)
    a = analyse(h, sin_lat, MAX_DEGREE)
    rot = np.exp(1j * chi)
    out: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for n in range(2, MAX_DEGREE + 1):
        k = 4 * np.pi * G_GRAV * RHO_W * (1 + K_PRIME[n]) / (G_E * (2 * n + 1)) / SCALE
        for m in range(n + 1):
            packed = [1j * k * rot * a[n, m, s] for s in (0, 1)]
            out[(n, m)] = (packed[0].imag, packed[0].real, packed[1].imag, packed[1].real)
    return out


# ── Self-test: the claims this script makes about its own quadrature ───────────────────────────────
def self_test(nlat: int = 1440, nlon: int = 2880) -> int:
    rng = np.random.default_rng(20260905)
    lat = 90.0 - 180.0 * np.arange(nlat) / nlat
    sin_lat = np.sin(np.deg2rad(lat))
    lon = 360.0 * np.arange(nlon) / nlon
    lam = np.deg2rad(lon)
    failures = 0

    w = dh_weights(nlat)
    err = abs(w.sum() - 2.0)
    print(f"  sum of Driscoll-Healy weights - 2      {err:.3e}")
    failures += err > 1e-12

    ref = legendre(MAX_DEGREE, np.array([0.3]))
    worst = max(abs(ref[n, m, 0] - v) for (n, m), v in PBAR_AT_0_3.items())
    print(f"  worst Pbar vs the Maxima oracle         {worst:.3e}")
    failures += worst > 1e-14

    p = legendre(MAX_DEGREE, sin_lat)
    worst = 0.0
    for n in range(MAX_DEGREE + 1):
        for m in range(n + 1):
            for n2 in range(MAX_DEGREE + 1):
                if n2 < m:
                    continue
                got = 0.5 * np.sum(w * p[n, m] * p[n2, m]) * (1 if m == 0 else 0.5)
                worst = max(worst, abs(got - (1.0 if n == n2 else 0.0)))
    print(f"  worst Pbar orthonormality residual     {worst:.3e}")
    failures += worst > 1e-12

    worst = 0.0
    for n in range(2, MAX_DEGREE + 1):
        for m in range(n + 1):
            ap, am = rng.normal(size=2) + 1j * rng.normal(size=2)
            if m == 0:
                am = 0j
            field = p[n, m][:, None] * (ap * np.exp(1j * m * lam)[None, :] + am * np.exp(-1j * m * lam)[None, :])
            a = analyse(field, sin_lat, MAX_DEGREE)
            worst = max(worst, abs(a[n, m, 0] - ap), abs(a[n, m, 1] - am))
            leak = max(
                abs(a[n2, m2, s])
                for n2 in range(MAX_DEGREE + 1)
                for m2 in range(n2 + 1)
                for s in (0, 1)
                if (n2, m2) != (n, m)
            )
            worst = max(worst, leak)
    print(f"  worst synthetic analysis residual      {worst:.3e}")
    failures += worst > 1e-11

    print("self-test FAILED" if failures else "self-test passed")
    return 1 if failures else 0


# ── Emission ───────────────────────────────────────────────────────────────────────────────────────
def emit(waves: dict[int, tuple[str, dict[tuple[int, int], tuple[float, float, float, float]]]], atlas: Atlas) -> str:
    body = []
    for doodson in sorted(waves):
        name, cells = waves[doodson]
        body.append(f"    // {name} (Doodson {doodson})")
        body.append(f"    OceanTideWave {{ doodson: {doodson}, coeffs: &[")
        for (n, m), v in sorted(cells.items()):
            body.append(f"        ({n}, {m}, {v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f}, {v[3]:.5f}),")
        body.append("    ] },")
    rows = "\n".join(body)
    return f"""{atlas.module_doc}

// Computed coefficients carry many significant digits and several share leading ones, and the docs
// spell the model names ({atlas.spelled}) the way the conventions do.
#![allow(
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::unreadable_literal
)]

use crate::ocean_tide_fes2004::OceanTideWave;

/// {atlas.short} coefficient scale: values are in units of 10^-11, matching the FES tables.
pub const SCALE: f64 = {SCALE_LITERAL};

{atlas.wave_doc.format(n=len(waves))}
#[rustfmt::skip]
pub static {atlas.static}: &[OceanTideWave] = &[
{rows}
];
"""


# The tide height and the load tide are different physical quantities on identically shaped grids,
# and `extrapolated` is the tide height pushed a degree into land for altimeter users. Only these two
# directories hold what Eq. 6.21 takes, so they are named rather than globbed for: a recursive sweep
# that picked up `load_tides` would produce a plausible table of the wrong thing.
SUBDIRS = ("ocean_tides", "inferred_tides")


def grids(root: pathlib.Path) -> list[pathlib.Path]:
    """Every constituent grid under an atlas root, or under a single directory of them."""
    found = [p for sub in SUBDIRS for p in sorted((root / sub).glob("*.nc"))]
    return found or sorted(root.glob("*.nc"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas", choices=sorted(ATLASES), help="which atlas the files are")
    ap.add_argument("root", nargs="?", help="atlas root, or a directory of its ocean-tide netCDF files")
    ap.add_argument("out", nargs="?", help="Rust module to write")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--admint-cache",
        default=str(pathlib.Path.home() / ".cache/spacelab/tides/ADMINT.F"),
        help="where the fetched ADMINT.F is kept (outside the repository)",
    )
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.atlas or not args.root or not args.out:
        ap.error("--atlas, root and out are required unless --self-test is given")
    atlas = ATLASES[args.atlas]

    signs = cartwright_signs(admint_source(pathlib.Path(args.admint_cache)))
    waves: dict[int, tuple[str, dict[tuple[int, int], tuple[float, float, float, float]]]] = {}
    for path in grids(pathlib.Path(args.root)):
        if path.name.startswith("._"):  # macOS resource forks travel in the tar
            continue
        # The stem is matched exactly, apostrophe and all: GOT marks the third-degree constituents
        # with a prime (l2', n2', m1'), and folding those onto the second-degree names would silently
        # overwrite a major wave with a supplement whose phase convention is a different one.
        name = path.stem.lower().removesuffix(atlas.suffix)
        if name not in DOODSON:
            print(f"  skipping {path.name}: no Doodson number recorded for '{name}'", file=sys.stderr)
            continue
        doodson = DOODSON[name]
        if doodson not in signs:
            # A compound tide such as M4 has no astronomical amplitude, so no sign, so no bias.
            print(f"  skipping {path.name}: Doodson {doodson} has no Cartwright-Edden amplitude", file=sys.stderr)
            continue
        chi = chi_bias(doodson, signs[doodson])
        if chi is None:
            print(f"  skipping {path.name}: species {doodson // 100000} has no Table 6.6 bias", file=sys.stderr)
            continue
        waves[doodson] = (name, coefficients(path, chi, atlas))
        print(f"  {name:8s} Doodson {doodson}  H = {signs[doodson]:+.6f} m  chi = {np.rad2deg(chi):+7.1f} deg")

    if not waves:
        raise SystemExit(f"no usable grids under {args.root}")
    pathlib.Path(args.out).write_text(emit(waves, atlas), encoding="utf-8")
    print(f"{len(waves)} waves -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
