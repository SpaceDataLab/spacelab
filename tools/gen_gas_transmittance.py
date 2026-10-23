#!/usr/bin/env python3
"""Band transmittance of the five absorbing gases, slit-averaged, for the RadCalNet comparison.

`radiative_transfer::Atmosphere` is one grey scattering slab. Ozone is already carried, as a two-way
`exp(-tau_3(lambda) (1/mu_s + 1/mu_v))` applied *outside* the slab, and that is honest for ozone
alone: the Chappuis band is a smooth continuum with no line structure, and the layer sits at 20-25 km,
above both the aerosol and most of the Rayleigh column. Neither half of that argument survives for the
gases here. Water sits in the boundary layer, mixed with the aerosol it must not be factored out of;
the well-mixed gases are mixed with the air whose Rayleigh depth they share; and at RadCalNet's 10 nm
resolution every one of them is thousands of lines rather than a continuum, so their band
transmittance is **not** an exponential in the column and no single optical depth can stand for one.

This file supplies what does stand for one. For each published wavelength and each gas it writes an
**exponential-sum fit**

    T(u) = sum_j w_j exp(-k_j u),    sum_j w_j = 1,    w_j >= 0,    k_j >= 0

of the slit-averaged transmittance against slant column `u`. That form is exactly a discretised
k-distribution: `w_j` is the fraction of the channel's spectral width whose absorption coefficient is
`k_j`. It is the one representation that composes correctly with a scattering calculation -- run the
transport once per term and weight-sum the results -- which a single exponential does not, and it is
what MODTRAN, 6S and libRadtran all reduce to at moderate resolution.

The rates carry a surface-pressure axis and the weights do not, which is the *correlated*-k part and
not a convenience: `w_j` is a property of the channel -- what fraction of its width absorbs at the
`j`-th strongest rate -- while `k_j` is a property of the air the channel looks through, and pressure
broadening moves it by tens of percent over the range of surface pressures RadCalNet's own sites
present. One weight vector is therefore fitted jointly against every pressure node at once rather
than one vector per node independently, and that is measured below rather than asserted.

**What this file is not.** It is not a correction, and on its own it changes no answer. It is the
coefficient table that a k-distribution inside the Monte-Carlo transport will read. Verified on its
own terms here (see the checks printed below and carried into the header), so that when the transport
changes, one of the two halves is already known good.

**Provenance.**

  * Line parameters: the HITRAN database, https://hitran.org, fetched through its line-by-line API.
    HITRAN asks to be cited as I. E. Gordon et al., *The HITRAN2024 molecular spectroscopic
    database*, J. Quant. Spectrosc. Radiat. Transfer **353**, 109807 (2026), and states that the
    database is free; it publishes a citation policy and no redistribution grant. **So no line
    parameter is written into this repository.** They are fetched at generation time and discarded;
    what is committed is the exponential-sum fit computed here, which is this file's own arithmetic.
    The edition actually served is recorded in the fixture header, because a fit is only reproducible
    against the edition it was made from.
  * Partition functions: the TIPS tables HITRAN serves at `/data/Q/q<id>.txt`, per isotopologue, so
    the temperature scaling of the line strengths is tabulated rather than approximated. The
    alternative -- `(T/296)^1.5` for a non-linear molecule -- is a percent-level error in a strong
    band and is measured below rather than assumed small.
  * Atmospheric profile: the U.S. Standard Atmosphere, 1976 (NOAA/NASA/USAF, NOAA-S/T 76-1562), a
    work of the United States Government. Pressure and temperature on the standard geopotential
    ladder; the water profile shape is the AFGL mid-latitude summer column, **scaled** so that its
    integral is the column being asked about, which is the approximation named and costed below.

**Units, stated once because getting them wrong is silent.** `u` is the *slant* column:

  * water: precipitable centimetres times air mass. One precipitable cm is 1 g/cm^2, so
    3.3428e22 molecules/cm^2.
  * every other gas -- CO2, N2O, CH4, O2 -- `(P / 1013.25 hPa) * air mass`, dimensionless, because a
    well-mixed gas has a fixed mixing ratio and its column is set by the surface pressure. One unit is
    that ratio times the air column, computed in `AIR_COLUMN_PER_CM2` rather than tabulated per gas.

Both quantities are what `radcalnet_toa.txt` already carries per row -- the water column and the
surface pressure -- so the fixture joins on nothing new.

    timeout 10800 uv run --with numpy --with scipy python tools/gen_gas_transmittance.py

writes `crates/spacelab-earthvision/tests/fixtures/gas_transmittance.txt`. Needs the network, numpy
and scipy, and no credentials.
"""

from __future__ import annotations

import http.client
import math
import os
import pathlib
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import nnls
from scipy.special import wofz

HITRAN_LBL = "https://hitran.org/lbl/api"
HITRAN_TIPS = "https://hitran.org/data/Q/q{global_id}.txt"

# Characters in a HITRAN2004-format record, newline excluded.
RECORD = 160

RETRIES = 5

# Covering 400-2500 nm takes a few dozen requests of a server that is doing us a favour, and the
# calculation below is iterated on far more often than the catalogue changes. Cached outside the
# repository, deliberately and not incidentally: HITRAN grants no redistribution, so a line
# parameter must never land anywhere `git add` can reach it.
CACHE = pathlib.Path.home() / ".cache" / "spacelab" / "hitran"

OUT = pathlib.Path(__file__).resolve().parents[1] / (
    "crates/spacelab-earthvision/tests/fixtures/gas_transmittance.txt"
)

# RadCalNet publishes 400-2500 nm on a 10 nm grid: 211 wavelengths, the same grid
# `radcalnet_absorption.txt` carries both reflectance matrices on.
WAVELENGTHS_NM = [400.0 + 10.0 * index for index in range(211)]

# MODTRAN Card 4 of RadCalNet's site default parameters: FWHM 10 nm, FLAGS 'NTAA' -- nanometres,
# triangular slit, absolute width. A triangle of 20 nm base has 10 nm FWHM.
SLIT_FWHM_NM = 10.0
SLIT_HALF_BASE_NM = SLIT_FWHM_NM


# Physical constants, CODATA 2018.
C2 = 1.438_776_877  # second radiation constant hc/k, cm K
BOLTZMANN = 1.380_649e-23  # J/K
LIGHT = 2.997_924_58e10  # cm/s
AMU = 1.660_539_066_60e-27  # kg
AVOGADRO = 6.022_140_76e23
STANDARD_PRESSURE_HPA = 1013.25
ATM_PER_HPA = 1.0 / STANDARD_PRESSURE_HPA
GRAVITY = 9.806_65  # m/s^2
DRY_AIR_MOLAR_MASS = 28.9644e-3  # kg/mol

# Molecules of dry air in a vertical column at `STANDARD_PRESSURE_HPA`, per cm^2: `P / (m_air g)`
# times Avogadro. Every well-mixed gas's unit column is this times its volume mixing ratio, computed
# here rather than tabulated per gas so that four numbers cannot drift apart from each other or from
# the constants above. The hand-entered oxygen column this replaces was 4.5022e24, which implies a
# mixing ratio of 0.209 58 against the 0.209 41 written beside it -- 0.08 % inconsistent with itself.
AIR_COLUMN_PER_CM2 = (
    STANDARD_PRESSURE_HPA * 100.0 / (DRY_AIR_MOLAR_MASS * GRAVITY) * AVOGADRO * 1.0e-4
)


class Gas:
    """One absorber: which isotopologues to fetch, and what one unit of its slant column means."""

    def __init__(
        self,
        name: str,
        isotopologues: dict[int, float],
        mixing_ratio: float | None,
        molecules_per_unit: float | None = None,
    ) -> None:
        self.name = name
        # Global isotopologue id -> terrestrial abundance. HITRAN's line intensities are already
        # scaled by abundance, so the abundance is carried only to weight the partition functions.
        self.isotopologues = isotopologues
        # None for water, whose profile is not a fixed mixing ratio.
        self.mixing_ratio = mixing_ratio
        if molecules_per_unit is not None:
            self.molecules_per_unit = molecules_per_unit
        elif mixing_ratio is not None:
            self.molecules_per_unit = mixing_ratio * AIR_COLUMN_PER_CM2
        else:
            raise SystemExit(
                f"{name}: neither a mixing ratio nor an explicit unit column"
            )


# Global isotopologue ids and terrestrial abundances, both read from HITRAN's own molparam.txt rather
# than from memory: water 1-4 are H2(16)O, H2(18)O, H2(17)O and HD(16)O; carbon dioxide 7-12 run
# (12)C(16)O2 down to (16)O(13)C(17)O; nitrous oxide 21-25 and methane 32-35 are their first four or
# five; oxygen 36-38 are (16)O2, (16)O(18)O and (16)O(17)O. Each set carries better than 99.99 % of
# its molecule's absorption, and the ones left out are below the fit residual by three orders.
#
# Volume mixing ratios of the well-mixed gases, dry air. Oxygen is a physical constant of the
# atmosphere; the other three are not, and are quoted as the mean over RadCalNet's own record
# (2013-2026) rather than as a present-day value, because that record is what the fixtures sample:
# CO2 ran 396 to 428 ppm over it, CH4 1.81 to 1.94 ppm, N2O 326 to 338 ppb (NOAA GML global means).
# What a different CO2 costs is measured and printed in the header rather than argued about -- and it
# is the one number here that RadCalNet's own MODTRAN runs may disagree with, since the portal does
# not publish which value it uses and MODTRAN's own default has been 330 ppm since 1976.
WATER = Gas(
    name="H2O",
    isotopologues={1: 0.997317, 2: 0.002000, 3: 3.719e-4, 4: 3.107e-4},
    mixing_ratio=None,
    # A precipitable cm is 1 g/cm^2 over natural-abundance water, 18.0153 g/mol.
    molecules_per_unit=3.3428e22,
)
CARBON_DIOXIDE = Gas(
    name="CO2",
    isotopologues={
        7: 0.984204,
        8: 1.105_74e-2,
        9: 3.947_07e-3,
        10: 7.339_89e-4,
        11: 4.434_46e-5,
        12: 8.246_23e-6,
    },
    mixing_ratio=4.12e-4,
)
NITROUS_OXIDE = Gas(
    name="N2O",
    isotopologues={
        21: 0.990333,
        22: 3.640_93e-3,
        23: 3.640_93e-3,
        24: 1.985_82e-3,
        25: 3.692_80e-4,
    },
    mixing_ratio=3.32e-7,
)
METHANE = Gas(
    name="CH4",
    isotopologues={32: 0.988274, 33: 1.110_31e-2, 34: 6.157_51e-4, 35: 6.917_85e-6},
    mixing_ratio=1.88e-6,
)
OXYGEN = Gas(
    name="O2",
    isotopologues={36: 0.995262, 37: 3.991_41e-3, 38: 7.422_35e-4},
    mixing_ratio=0.209_41,
)
GASES = [WATER, CARBON_DIOXIDE, NITROUS_OXIDE, METHANE, OXYGEN]

# Carbon monoxide is deliberately absent. Its only band on this grid is the 2-0 overtone near
# 2330 nm, and at the 0.1 ppm a clean-air column carries it reaches a two-way optical depth of order
# 1e-5 -- three orders under the fixture's own transparency bar, and under RadCalNet's 1e-4
# reflectance quantisation. Its mixing ratio is also the one that is genuinely local rather than
# well-mixed, so a single number for it would be wrong in a way the others are not.

# The slant columns the fit has to be good over: `(u_min, u_max, points)` per gas. Measured from what
# the two RadCalNet fixtures actually contain -- water 0.19 to 5.46 precipitable-cm air masses, and
# 0.75 to 6.78 pressure-scaled air masses for every well-mixed gas, which share one column definition
# and therefore one grid -- with an order of magnitude of margin at the dry end and a factor of four
# at the wet one. Wider is not free: the ladder of `k` values has to span the whole grid, and spending
# its resolution on columns no site on Earth presents costs accuracy where they do.
WELL_MIXED_GRID = (0.2, 15.0, 40)
COLUMN_GRID = {"H2O": (0.01, 25.0, 60)} | {
    gas.name: WELL_MIXED_GRID for gas in GASES if gas.mixing_ratio is not None
}

# Below this slit-averaged transmittance the channel is opaque and no correction can be made through
# it: the correction divides by `T`, so a `T` of 1e-3 amplifies every error in it a thousandfold. Each
# channel's grid stops at the column where it crosses this floor, and that ceiling is written into the
# fixture, because a fit is only a fit over the domain it was made on. Two channels need it: at 1380
# and 1870 nm water saturates within the range RadCalNet's own sites present.
TRANSMITTANCE_FLOOR = 1.0e-3

# Terms in the exponential sum. Measured on seven channels spanning the weakest and the two saturated
# water bands: 8 terms leave 5.1e-3, 12 leave 1.2e-4 and 16 leave 2.2e-5 in the worst channel. The
# residual at 8, 16 and 24 is measured and printed rather than argued.
TERMS = 16

# Rungs offered to the non-negative least squares, placed at quantiles of the channel's own exact
# k-distribution rather than on a blind geometric ladder. It is the same fit either way, but a rung
# where the distribution has no mass is a column of the design matrix that can only fit noise: on a
# geometric ladder the worst channel came out at 1.9e-2 against 2.2e-5 here, at the same term count.
RUNGS = 200
LADDER = np.linspace(0.001, 0.999, RUNGS)

# Layers between the surface and 70 km. The absorption is not linear in the column because the line
# cores saturate, and where a molecule sits decides how pressure-broadened it is, so a one-layer
# calculation is a different answer rather than a coarser one. Measured against 384 layers, at 940 nm:
# 24 leave 4.7e-2, 48 leave 1.3e-2, 96 leave 3.6e-3 and this leaves 6.7e-4. The convergence against
# twice this count is measured and printed.
LAYERS = 192

# Surface pressures the table is tabulated at, hPa. This is an axis and not a costed approximation
# because the effect is enormous. Measured line-by-line at 940 nm between the ends of this ladder,
# with each difference taken relative to the smaller of the two transmittances, 1040 hPa differs
# from 720 by 72 % at worst over the fit's own column grid [0.01, 25] cm, and by 38.2 % over the
# 1945 slant columns RadCalNet's own rows carry. Lines at a high-altitude site are narrower, which
# changes how they saturate, and no rescaling of the column can stand in for that.
#
# The span is set by the data rather than by the standard atmosphere. RadCalNet's own published
# surface pressures run 720.0 to 1031.1 hPa across the two fixtures -- the floor is Golmud on the
# Tibetan plateau, and the ceiling is *above* sea level, because a cold anticyclone over a
# near-sea-level site puts more air overhead than the standard atmosphere has. 266 of those 1945
# rows, 13.7 %, sit above 1013.25, and a ladder stopping there refuses every one of them.
#
# That refusal rate is the reason for this node, and it is a better reason than the one first
# written here, which priced the alternative seven times too high. Clamping is the other wrong
# answer and it is not free either: across the 1013.25-1040 segment, on RadCalNet's own columns,
# the 940 nm sensitivity is 0.034 %/hPa at the median and 0.097 %/hPa at worst, so clamping the
# 17.8 hPa the archive actually overshoots by charges 0.6 % of channel transmittance typically and
# 1.7 % at worst -- an order of magnitude above this table's own fit error, and silently, but not
# the 3.6 % once claimed at 0.24 %/hPa. Neither wrong answer has to be chosen: the node is what
# makes the question moot, and 1040 leaves margin over the whole archive rather than over the
# sampled days.
PRESSURE_NODES = (1040.0, 1013.25, 940.0, 860.0, 790.0, 720.0)

# Standard sea-level pressure, and the node the costed single-pressure checks below are taken at. It
# is a member of the ladder, but it is named separately because it means the standard atmosphere and
# not "the first node" -- the ladder's ends are free to move with the data, this is not.
REFERENCE_SURFACE_HPA = 1013.25

# Precipitable centimetres the water profile is scaled to *for the purpose of self-broadening only*.
# Water broadens itself several times harder than air does, so the half-width in a layer depends on
# how much water is in that layer -- which makes the optical depth non-linear in the column, and a
# table linear in `u` cannot carry that. The amount still scales exactly with `u`; only the width is
# frozen, at RadCalNet's own median column. What the spread from 0.5 to 5 cm costs is measured and
# printed rather than assumed small. (The first version of this file used each layer's *share of the
# total column* as its self-broadening fraction, which is not a mixing ratio at all: it falls as the
# layer count rises, so the calculation could not converge in the layer count and did not.)
SELF_BROADENING_REFERENCE_CM = 2.0

# Spectral grid spacing, cm^-1. The narrowest feature is a Doppler-limited line core in the upper
# layers -- about 0.013 cm^-1 half-width at 10000 cm^-1 for water at 250 K, resolved here with five
# points. Measured against 0.0005: this costs 5.5e-6 and half of the run time of 0.002.
GRID_SPACING = 0.005

# How far from a line centre its Lorentz wing is still summed, cm^-1. The far wing of a strong line
# reaches much further than a weak one's, so the cutoff follows the strength rather than being one
# number for every line; what the truncation discards is measured and printed.
CUTOFF_STRONG = 25.0
CUTOFF_WEAK = 2.0
STRONG_LINE = 1.0e-23

# Gaussian widths from a line centre out to which the Voigt function is evaluated in full; beyond
# that its Lorentz limit is used. The Gaussian has convolved away by then -- the leading correction
# is `1.5 / VOIGT_CORE_DOPPLER^2` relative -- while nearly every grid point a strong line covers is
# out there, which is why this is what makes a converged layer count affordable. Measured against a
# full-Voigt pass rather than argued.
VOIGT_CORE_DOPPLER = 100.0

# Chunk width for the line-by-line pass, cm^-1. Only a bookkeeping choice -- each chunk is computed
# with `CUTOFF_STRONG` of margin on both sides, so the seams are exact.
CHUNK = 200.0

# Below this optical depth at the largest column on the grid, a channel carries no absorption worth
# fitting: RadCalNet reports reflectance to 1e-4 on a signal of order 0.3, so a transmittance within
# 1e-6 of unity cannot be seen in the comparison this table exists to serve.
NEGLIGIBLE_DEPTH = 1.0e-6

# Molar masses of the isotopologues fetched, g/mol, for the Doppler width. Read from HITRAN's own
# molparam.txt, keyed by global isotopologue id, in the order of `GASES`.
ISOTOPOLOGUE_MASS = {
    1: 18.010_565,
    2: 20.014_811,
    3: 19.014_780,
    4: 19.016_740,
    7: 43.989_830,
    8: 44.993_185,
    9: 45.994_076,
    10: 44.994_045,
    11: 46.997_431,
    12: 45.997_400,
    21: 44.001_062,
    22: 44.998_096,
    23: 44.998_096,
    24: 46.005_308,
    25: 45.005_278,
    32: 16.031_300,
    33: 17.034_655,
    34: 17.037_475,
    35: 18.040_830,
    36: 31.989_830,
    37: 33.994_076,
    38: 32.994_045,
}

# U.S. Standard Atmosphere 1976: geopotential altitude km, temperature K, pressure hPa. Every 2 km to
# 32 km and then the standard breakpoints, which is enough for a profile this is only integrated over.
US_STANDARD = [
    (0.0, 288.15, 1013.25),
    (2.0, 275.15, 795.01),
    (4.0, 262.17, 616.60),
    (6.0, 249.19, 472.17),
    (8.0, 236.22, 356.51),
    (10.0, 223.25, 264.99),
    (12.0, 216.65, 193.99),
    (14.0, 216.65, 141.70),
    (16.0, 216.65, 103.52),
    (18.0, 216.65, 75.65),
    (20.0, 216.65, 55.29),
    (24.0, 220.56, 29.72),
    (28.0, 224.53, 15.95),
    (32.0, 228.49, 8.68),
    (40.0, 250.35, 2.775),
    (50.0, 270.65, 0.7594),
    (60.0, 247.02, 0.2196),
    (70.0, 219.59, 0.05221),
]

# AFGL mid-latitude summer water mixing ratio, ppmv, on the same altitude ladder. The *shape* is what
# is used -- the profile is scaled to whatever column is being asked about -- so the choice of AFGL
# model matters only through where in the pressure ladder the water sits, which is measured below.
WATER_PPMV = [
    (0.0, 14000.0),
    (2.0, 6000.0),
    (4.0, 2900.0),
    (6.0, 1800.0),
    (8.0, 1100.0),
    (10.0, 190.0),
    (12.0, 55.0),
    (14.0, 12.0),
    (16.0, 6.0),
    (18.0, 5.0),
    (20.0, 4.5),
    (24.0, 4.3),
    (28.0, 4.3),
    (32.0, 4.5),
    (40.0, 5.2),
    (50.0, 5.8),
    (60.0, 6.0),
    (70.0, 5.5),
]


# HITRAN records carry a *local* isotopologue number and a molecule id; the partition-function tables
# and the molar-mass table above are keyed by the *global* one. Only the isotopologues fetched are
# here, so a line record naming any other molecule raises rather than being silently attributed to
# whichever gas the fetch was for -- which is the failure mode a wrong `iso_ids_list` would produce.
# Molecule ids are HITRAN's: 1 H2O, 2 CO2, 4 N2O, 6 CH4, 7 O2.
LOCAL_TO_GLOBAL = {
    (1, 1): 1,
    (1, 2): 2,
    (1, 3): 3,
    (1, 4): 4,
    (2, 1): 7,
    (2, 2): 8,
    (2, 3): 9,
    (2, 4): 10,
    (2, 5): 11,
    (2, 6): 12,
    (4, 1): 21,
    (4, 2): 22,
    (4, 3): 23,
    (4, 4): 24,
    (4, 5): 25,
    (6, 1): 32,
    (6, 2): 33,
    (6, 3): 34,
    (6, 4): 35,
    (7, 1): 36,
    (7, 2): 37,
    (7, 3): 38,
}

# Column order of the parsed line array, so no caller indexes by a bare integer.
CENTRE, STRENGTH, GAMMA_AIR, GAMMA_SELF, ENERGY, EXPONENT, SHIFT, GLOBAL = range(8)


def whole(body: bytes, advertised: int, record_length: int) -> bool:
    """Whether a short read is HITRAN's `Content-Length` defect rather than a real truncation.

    hitran.org counts a CRLF per record when it writes the header and then emits an LF, so it
    under-delivers by exactly one byte per record on *every* line request, however small, and
    `http.client` -- which trusts the header -- raises `IncompleteRead` on all of them. Accepting
    the partial body on that reasoning alone would be a guess. These are the two identities that
    hold if and only if the body is whole: it is an exact number of records, and the shortfall the
    header reports is exactly that record count. Both are required, because a truncation landing on
    a record boundary satisfies the first one on its own.
    """
    stride = record_length + 1
    if not body.endswith(b"\n") or len(body) % stride:
        return False
    return advertised - len(body) == len(body) // stride


def fetch(url: str, timeout: float, record_length: int | None = None) -> str | None:
    """A whole response body, retried with a backoff, or `SystemExit` naming the last failure.

    Never a short answer: a truncated line request is indistinguishable from a transparent spectral
    region and would silently write a table with no absorption in it.

    `None` rather than a body means the server answered **404**, which is how hitran.org says the
    query selected no lines at all. That is an answer and not a failure -- oxygen has no band between
    17934 and 19934 cm^-1 and the endpoint says so this way -- so it is neither retried nor treated
    as an outage. The two are told apart by the status code alone, and the backstop against mistaking
    a moved endpoint for a transparent spectrum is `fetch_all_lines`, which refuses a gas whose every
    chunk came back empty.
    """
    reason = "no attempt was made"
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                body = response.read()
            advertised = len(body)
        except http.client.IncompleteRead as short:
            # Not an error path: this is how *every* line request arrives. Whether the body is
            # whole is decided by `whole`, below, on the same footing as a clean read.
            body, advertised = short.partial, len(short.partial) + short.expected
        except urllib.error.HTTPError as status:
            if status.code == 404:
                return None
            body, advertised, reason = b"", -1, str(status)
        except (OSError, urllib.error.URLError) as failure:
            body, advertised, reason = b"", -1, str(failure)
        if len(body) == advertised or (
            record_length is not None and whole(body, advertised, record_length)
        ):
            return body.decode("ascii")
        if advertised >= 0:
            reason = f"{len(body)} of {advertised} bytes"
        print(f"  retrying ({reason}) {url}", flush=True)
        time.sleep(2.0**attempt)
    raise SystemExit(f"{url}: {RETRIES} attempts, last failure {reason}")


def cached(
    name: str, url: str, timeout: float, record_length: int | None = None
) -> str:
    """`fetch`, memoised on disk under `CACHE`, which is outside the repository by design.

    The catalogue is not pinned by a version string, because HITRAN publishes none per query; it is
    pinned by the line counts written into the fixture header. Clearing this cache and regenerating
    is therefore the check that the edition has not moved under us, not a mere refresh.
    """
    path = CACHE / name
    if path.exists():
        return path.read_text(encoding="ascii")
    text = fetch(url, timeout, record_length)
    if text is None:
        # A 404: the server states the window holds no lines. Definite, so it is memoised like any
        # other answer -- re-asking would just re-derive the same emptiness on every rerun.
        text = ""
    elif not text:
        # A 200 with an empty body is a different thing entirely: hitran.org sends one both for a
        # genuinely empty window and for a request it has decided to stop serving, and one answer
        # cannot tell those apart. Ask again before believing it, and never memoise it -- an empty
        # chunk cached into a hole in the middle of a band is precisely the silent failure the
        # completeness check exists to prevent.
        time.sleep(5.0)
        second = fetch(url, timeout, record_length)
        if second:
            print(f"  the empty answer was rationing, not data: {url}", flush=True)
            text = second
        else:
            return text
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    return text


def fetch_lines(gas: Gas, wavenumber_lo: float, wavenumber_hi: float) -> np.ndarray:
    """HITRAN line parameters over a wavenumber window, one row per line, columns as named above.

    Parsed by fixed column position rather than by whitespace, because several fields abut with no
    separator -- `gamma_air` and `gamma_self` do, and so do `n_air` and `delta_air`, which is how a
    split-based parser silently produces a shifted table rather than an error.
    """
    isotopologues = ",".join(str(key) for key in gas.isotopologues)
    url = (
        f"{HITRAN_LBL}?iso_ids_list={isotopologues}"
        f"&numin={wavenumber_lo:.6f}&numax={wavenumber_hi:.6f}&head=false"
    )
    text = cached(
        f"{gas.name}_{wavenumber_lo:.0f}_{wavenumber_hi:.0f}.txt", url, 900.0, RECORD
    )
    rows = []
    for record in text.splitlines():
        if len(record) < 67:
            continue
        key = (int(record[0:2]), int(record[2:3]))
        if key not in LOCAL_TO_GLOBAL:
            raise SystemExit(f"unexpected isotopologue {key} in {url}")
        rows.append(
            (
                float(record[3:15]),
                float(record[15:25]),
                float(record[35:40]),
                float(record[40:45]),
                float(record[45:55]),
                float(record[55:59]),
                float(record[59:67]),
                float(LOCAL_TO_GLOBAL[key]),
            )
        )
    return np.array(rows, dtype=float) if rows else np.zeros((0, 8))


def fetch_all_lines(gas: Gas, lo: float, hi: float) -> np.ndarray:
    """Every line of `gas` between `lo` and `hi` cm^-1, fetched in chunks and concatenated.

    Chunked because one request for twenty thousand wavenumbers is a request the server is entitled
    to refuse, and because a partial answer to a single large request is indistinguishable from a
    sparse spectral region -- which would silently produce a table with no absorption in it.
    """
    pieces = []
    edge = lo
    while edge < hi:
        top = min(edge + CHUNK * 10.0, hi)
        piece = fetch_lines(gas, edge, top)
        pieces.append(piece)
        print(
            f"  {gas.name} {edge:8.1f}-{top:8.1f} cm^-1: {len(piece):7d} lines",
            flush=True,
        )
        edge = top
    filled = [piece for piece in pieces if piece.size]
    if not filled:
        raise SystemExit(f"{gas.name}: no lines at all over {lo:.0f}-{hi:.0f} cm^-1")
    lines = np.vstack(filled)
    order = np.argsort(lines[:, CENTRE])
    return lines[order]


def fetch_partition(global_id: int) -> dict[int, float]:
    """The TIPS partition function of one isotopologue, temperature K -> Q, as HITRAN tabulates it."""
    text = cached(f"q{global_id}.txt", HITRAN_TIPS.format(global_id=global_id), 300.0)
    table = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2:
            table[int(fields[0])] = float(fields[1])
    if 296 not in table:
        raise SystemExit(f"TIPS q{global_id}.txt: no entry at 296 K")
    return table


def partition_ratio(table: dict[int, float], temperature: float) -> float:
    """Q(296) / Q(T), linearly interpolated between the tabulated integer kelvins."""
    low = math.floor(temperature)
    if low not in table or low + 1 not in table:
        raise SystemExit(f"TIPS table does not cover {temperature} K")
    fraction = temperature - low
    return table[296] / (table[low] * (1.0 - fraction) + table[low + 1] * fraction)


def profile(
    layers: int, surface_hpa: float, reference_cm: float = SELF_BROADENING_REFERENCE_CM
) -> list[tuple[float, float, float, float, float]]:
    """The atmosphere as slabs: pressure hPa, temperature K, air column cm^-2, water share, water VMR.

    The air column follows from hydrostatic balance alone -- `dp = -rho g dz` makes it `Delta_p/(m g)`
    -- so no density model enters and the only profile choices are where the water sits and how the
    temperature runs, both of which are costed in the header.

    The last two fields answer two different questions and must not be confused, which is a mistake
    this file has already made once. The *share* is what fraction of the total column this slab holds,
    and it sets how much absorber the slab has; it is a partition, so it necessarily shrinks as the
    layer count grows. The *volume mixing ratio* is how wet the slab is, and it sets how much of the
    broadening is water on water; it is a local property and does not move with the layer count at
    all. Using the share for both makes the self-broadening a function of the discretisation, and
    then the calculation cannot converge in the layer count -- measured, before this was fixed, as a
    floor of 7e-3 that 96 layers could not get below.
    """
    altitude = np.array([point[0] for point in US_STANDARD])
    temperature = np.array([point[1] for point in US_STANDARD])
    log_pressure = np.log(np.array([point[2] for point in US_STANDARD]))
    log_ppmv = np.log(np.array([point[1] for point in WATER_PPMV]))

    # A surface pressure *above* the profile's own base is not a clip, and treating it as one is what
    # made the 1040 hPa node a bit-identical copy of 1013.25 in all 1055 rows of an earlier table
    # (#369). `np.minimum` is right for a surface below the base -- a site at altitude, where the
    # bottom of the column is cut off -- and a no-op above it, so the ladder's ceiling could not
    # produce a thicker column than the standard atmosphere has and the six-node axis was five.
    #
    # The rule is one rule at both ends: the column runs from wherever the profile reaches
    # `surface_hpa` to the top, and below sea level that altitude is negative. So the tabulations are
    # extended one segment downward at their own lowest slope. The piecewise-linear interpolant is
    # unchanged at every altitude it already covered -- linear interpolation inside a segment does not
    # see the other segments -- and is merely defined below zero as well, which is why every node at
    # or under 1013.25 hPa comes out of this bit-identical to before.
    #
    # This is a *model* of an anticyclone and it is stated rather than derived: the standard
    # atmosphere with a slab of its own air added underneath. The alternative -- scaling every
    # pressure by `surface_hpa / 1013.25` -- also has a claim, and it is a different atmosphere,
    # because it moves the temperature at every level as well as the amount of air. The extension is
    # chosen because it is the exact mirror of the clip at the other end, so the ladder is one rule
    # rather than two, and because it leaves the five lower nodes untouched.
    below_km = -1.0
    span = altitude[1] - altitude[0]
    lapse = (temperature[1] - temperature[0]) / span
    decay = (log_pressure[1] - log_pressure[0]) / span
    drying = (log_ppmv[1] - log_ppmv[0]) / span
    altitude = np.insert(altitude, 0, below_km)
    temperature = np.insert(temperature, 0, temperature[0] + lapse * below_km)
    log_pressure = np.insert(log_pressure, 0, log_pressure[0] + decay * below_km)
    log_ppmv = np.insert(log_ppmv, 0, log_ppmv[0] + drying * below_km)

    top_km = 70.0
    edges_km = np.linspace(0.0, top_km, layers + 1)
    if surface_hpa > REFERENCE_SURFACE_HPA:
        # Where the extended profile reaches this pressure, from the added segment's own slope rather
        # than by interpolating back through it. Negative by construction, and required to sit inside
        # that one segment rather than beyond it.
        base_km = (math.log(surface_hpa) - log_pressure[1]) / decay
        if not below_km <= base_km < 0.0:
            raise SystemExit(
                f"a surface pressure of {surface_hpa} hPa is below {below_km} km, past the one "
                "segment this profile is extended by"
            )
        edges_km = np.concatenate(([base_km], edges_km))

    edge_p = np.exp(np.interp(edges_km, altitude, log_pressure))
    edge_t = np.interp(edges_km, altitude, temperature)
    edge_q = np.exp(np.interp(edges_km, altitude, log_ppmv)) * 1.0e-6
    edge_p = np.minimum(edge_p, surface_hpa)

    slabs, waters = [], []
    for index in range(len(edges_km) - 1):
        bottom, top = edge_p[index], edge_p[index + 1]
        if bottom - top <= 0.0:
            continue
        air_column = (
            (bottom - top) * 100.0 / (DRY_AIR_MOLAR_MASS * GRAVITY) * AVOGADRO * 1.0e-4
        )
        mixing = math.sqrt(edge_q[index] * edge_q[index + 1])
        slabs.append(
            [
                0.5 * (bottom + top),
                0.5 * (edge_t[index] + edge_t[index + 1]),
                air_column,
                0.0,
                mixing,
            ]
        )
        waters.append(air_column * mixing)
    total = sum(waters)
    # The AFGL shape carries whatever column it carries; rescale the mixing ratios so the profile
    # holds `reference_cm` of precipitable water, which is the state the line widths are frozen at.
    scale = reference_cm * WATER.molecules_per_unit / total
    for slab, water in zip(slabs, waters, strict=True):
        slab[3] = water / total
        slab[4] *= scale
    return [tuple(slab) for slab in slabs]


def layer_cross_section(
    lines: np.ndarray,
    grid: np.ndarray,
    pressure_hpa: float,
    temperature: float,
    partition: dict[int, dict[int, float]],
    self_fraction: float,
    cutoff_scale: float = 1.0,
    ratio: Callable[[dict[int, float], float], float] = partition_ratio,
) -> np.ndarray:
    """Absorption cross-section per molecule, cm^2, on `grid`, for one layer: a sum of Voigt profiles.

    Two economies, both measured rather than assumed. Every line is evaluated only over the window its
    own strength earns, because the far wing of a weak line is below the arithmetic that would carry
    it. And the Voigt function itself is evaluated only within `VOIGT_CORE_DOPPLER` Gaussian widths of
    the centre; beyond that the Gaussian convolution has nothing left to do and the profile is its
    Lorentz limit to a relative `1.5 / VOIGT_CORE_DOPPLER^2`. Nearly every grid point a strong line
    covers is in that far wing, so this is the difference between a converged layer count being
    affordable and not.
    """
    total = np.zeros_like(grid)
    if lines.size == 0:
        return total
    atm = pressure_hpa * ATM_PER_HPA
    warm = 296.0 / temperature

    for row in lines:
        centre = row[CENTRE] + row[SHIFT] * atm
        cutoff = (
            CUTOFF_STRONG if row[STRENGTH] >= STRONG_LINE else CUTOFF_WEAK
        ) * cutoff_scale
        lo = int(np.searchsorted(grid, centre - cutoff))
        hi = int(np.searchsorted(grid, centre + cutoff))
        if hi <= lo:
            continue

        global_id = int(row[GLOBAL])
        # Line strength at the layer temperature: the Boltzmann factor on the lower state and the
        # stimulated-emission correction are exact, and the partition ratio is tabulated.
        strength = (
            row[STRENGTH]
            * ratio(partition[global_id], temperature)
            * math.exp(-C2 * row[ENERGY] / temperature)
            / math.exp(-C2 * row[ENERGY] / 296.0)
            * (1.0 - math.exp(-C2 * row[CENTRE] / temperature))
            / (1.0 - math.exp(-C2 * row[CENTRE] / 296.0))
        )

        lorentz = (
            ((1.0 - self_fraction) * row[GAMMA_AIR] + self_fraction * row[GAMMA_SELF])
            * atm
            * warm ** row[EXPONENT]
        )
        doppler = (
            row[CENTRE]
            / LIGHT
            * math.sqrt(
                2.0
                * BOLTZMANN
                * temperature
                * math.log(2.0)
                / (ISOTOPOLOGUE_MASS[global_id] * AMU)
            )
        )
        sigma = doppler / math.sqrt(2.0 * math.log(2.0))
        root_two_sigma = sigma * math.sqrt(2.0)

        core = min(cutoff, VOIGT_CORE_DOPPLER * sigma)
        core_lo = max(lo, int(np.searchsorted(grid, centre - core)))
        core_hi = min(hi, int(np.searchsorted(grid, centre + core)))
        if core_hi > core_lo:
            z = ((grid[core_lo:core_hi] - centre) + 1j * lorentz) / root_two_sigma
            total[core_lo:core_hi] += (
                strength * np.real(wofz(z)) / (sigma * math.sqrt(2.0 * math.pi))
            )
        for start, stop in ((lo, core_lo), (core_hi, hi)):
            if stop > start:
                detuning = grid[start:stop] - centre
                total[start:stop] += (
                    strength
                    * lorentz
                    / math.pi
                    / (detuning * detuning + lorentz * lorentz)
                )
    return total


def slit_weights(wavelength_nm: float, wavenumber: np.ndarray) -> np.ndarray:
    """RadCalNet's 10 nm FWHM triangular slit, as normalised weights on a uniform wavenumber grid.

    The triangle is defined in *wavelength*, where MODTRAN Card 4 puts it, so carrying it to a grid
    uniform in wavenumber needs the Jacobian `|dlambda/dnu| = lambda^2/1e7`. It varies by about two
    percent across the window at 2500 nm; leaving it out would be a slit that is not the one the
    reference convolved with, in the direction that biases every long wavelength the same way.
    """
    wavelength = 1.0e7 / wavenumber
    triangle = np.clip(
        1.0 - np.abs(wavelength - wavelength_nm) / SLIT_HALF_BASE_NM, 0.0, None
    )
    weight = triangle * wavelength * wavelength
    total = weight.sum()
    return weight / total if total > 0.0 else weight


def channel_window(wavelength_nm: float) -> tuple[float, float]:
    """The wavenumber span the slit at `wavelength_nm` covers, plus the far wing of a strong line."""
    lo = 1.0e7 / (wavelength_nm + SLIT_HALF_BASE_NM) - CUTOFF_STRONG
    hi = 1.0e7 / (wavelength_nm - SLIT_HALF_BASE_NM) + CUTOFF_STRONG
    return lo, hi


def channel_depths(
    gas: Gas,
    wavelength_nm: float,
    lines: np.ndarray,
    layers: list[tuple[float, float, float, float, float]],
    partition: dict[int, dict[int, float]],
    spacing: float = GRID_SPACING,
    cutoff_scale: float = 1.0,
    ratio: Callable[[dict[int, float], float], float] = partition_ratio,
) -> tuple[np.ndarray, np.ndarray]:
    """Slit weights and column optical depth per unit `u`, point by point across one channel.

    The layer cross-sections are summed *before* anything is exponentiated, so the vertical
    inhomogeneity is exact rather than a correlated-k assumption: at each wavenumber this is the true
    column optical depth of the profile. The one approximation left is that the profile keeps its
    shape as the column scales, which is the line named and costed in the header.
    """
    lo, hi = channel_window(wavelength_nm)
    points = int((hi - lo) / spacing) + 1
    if points > 4_000_000:
        raise SystemExit(
            f"{gas.name} {wavelength_nm} nm: {points} grid points is not a grid"
        )
    grid = lo + spacing * np.arange(points)
    weights = slit_weights(wavelength_nm, grid)

    inside = lines[
        (lines[:, CENTRE] > lo - CUTOFF_STRONG)
        & (lines[:, CENTRE] < hi + CUTOFF_STRONG)
    ]
    depth_per_unit = np.zeros_like(grid)
    air_total = sum(layer[2] for layer in layers)
    for pressure_hpa, temperature, air_column, water_share, water_vmr in layers:
        if gas.mixing_ratio is None:
            share = water_share
            # Water broadens itself several times harder than air does, so a layer's half-width
            # depends on how wet that layer is -- its volume mixing ratio, not its share of the
            # column. At the surface of a 2 cm profile that is a few percent, and treating water as
            # a trace gas would narrow every strong line where most of the absorber sits.
            self_fraction = water_vmr
        else:
            share = air_column / air_total
            self_fraction = gas.mixing_ratio
        depth_per_unit += (
            layer_cross_section(
                inside,
                grid,
                pressure_hpa,
                temperature,
                partition,
                self_fraction,
                cutoff_scale,
                ratio,
            )
            * gas.molecules_per_unit
            * share
        )
    return weights, depth_per_unit


def transmittance(
    weights: np.ndarray, depths: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    """Slit-averaged transmittance over `columns`: the exact exponential sum, all million terms of it.

    The column enters the optical depth linearly, so the whole curve costs one line-by-line pass
    rather than one per column. That is not a speed detail -- it is why the curve is exactly smooth
    in `u` and can be fitted by a handful of terms at all.
    """
    return np.array(
        [float((weights * np.exp(-depths * column)).sum()) for column in columns]
    )


def usable_ceiling(
    weights: np.ndarray, depths: np.ndarray, u_min: float, u_max: float
) -> float:
    """The largest column whose transmittance still clears `TRANSMITTANCE_FLOOR`.

    Bisected geometrically rather than solved, because `T(u)` is a sum of decaying exponentials and
    so is strictly decreasing -- the bracket is guaranteed and the iteration cannot miss a root.
    """
    if transmittance(weights, depths, np.array([u_max]))[0] >= TRANSMITTANCE_FLOOR:
        return u_max
    lo, hi = u_min, u_max
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if transmittance(weights, depths, np.array([mid]))[0] >= TRANSMITTANCE_FLOOR:
            lo = mid
        else:
            hi = mid
    return lo


def rates_at(
    weights: np.ndarray, depths: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """The channel's absorption coefficient at each cumulative-k quantile.

    Sorting the per-point optical depths and accumulating the slit weight *is* the k-distribution,
    exactly -- an exponential sum is invariant to the order of its terms, so nothing is approximated
    by the reordering. The quantile, not the rate, is what carries across surface pressure: a
    spectral point in the strongest one percent of the channel stays there when the lines narrow,
    even though its coefficient does not. So the ladder is indexed by quantile and every pressure
    reads its own rates off it.
    """
    order = np.argsort(depths)
    cumulative = np.cumsum(weights[order])
    return np.interp(quantiles, cumulative / cumulative[-1], depths[order])


def exponential_sum(
    columns: np.ndarray,
    curves: list[np.ndarray],
    rate_sets: list[np.ndarray],
    terms: int = TERMS,
) -> tuple[np.ndarray, np.ndarray]:
    """One weight vector across every surface pressure, and the quantiles its rates are read at.

    Solving for the weights on a fixed ladder is convex, so that half of the answer is *the* answer;
    a non-linear fit for weights and rates together has local minima and no guarantee of positivity,
    and a negative weight is not a slightly wrong fit but a transmittance able to exceed one. Fitting
    all the pressures at once keeps that convexity -- the design matrices simply stack, one block per
    pressure, against one shared weight vector -- and it is what makes the pressure axis honest: the
    weights are the fraction of the channel's width in each quantile band, which is a property of the
    channel and not of the surface. Fitting at one pressure and carrying the weights to the others
    was measured at 2.3e-4 to 1.7e-2 across five channels; fitting them jointly is 2.0e-6 to 5.0e-4
    at the same term count.

    What is left is choosing which `terms` rungs to keep, and that is where the rest of the accuracy
    is won or lost: keeping the largest weights is choosing the support by a proxy for the objective,
    and a dense ladder defeats it by splitting one physical mode across neighbouring rungs. Dropping
    the rung whose removal costs least, one at a time and refitting each time, optimises the thing
    actually being measured -- 2.2e-5 against 1.9e-2 on the worst single-pressure channel.
    """
    # Weight each residual by 1/T, so a transmittance of 1e-3 is fitted as hard as one of 0.9: the
    # correction divides by the transmittance, so it is the relative error that propagates.
    scales = [1.0 / np.maximum(curve, 1.0e-8) for curve in curves]

    def solve(keep: np.ndarray) -> np.ndarray:
        blocks, target = [], []
        for curve, scale, rates in zip(curves, scales, rate_sets, strict=True):
            blocks.append(np.exp(-np.outer(columns, rates[keep])) * scale[:, None])
            target.append(curve * scale)
        # The weights sum to one by construction rather than by hope: one heavily weighted row of
        # ones against a target of one pins the normalisation at negligible cost to the fit.
        blocks.append(np.full((1, keep.size), 1.0e3))
        target.append(np.array([1.0e3]))
        fitted, _ = nnls(
            np.vstack(blocks), np.concatenate(target), maxiter=50 * keep.size + 1000
        )
        return fitted

    def cost(keep: np.ndarray) -> float:
        fitted = solve(keep)
        return max(
            worst_relative(evaluate(fitted, rates[keep], columns), curve)
            for curve, rates in zip(curves, rate_sets, strict=True)
        )

    keep = np.flatnonzero(solve(np.arange(rate_sets[0].size)) > 0.0)
    while keep.size > terms:
        _, drop = min((cost(np.delete(keep, i)), i) for i in range(keep.size))
        keep = np.delete(keep, drop)
    if keep.size == 0:
        return np.array([1.0]), np.array([0.0])
    weights = solve(keep)
    total = weights.sum()
    if total <= 0.0:
        return np.array([1.0]), np.array([0.0])
    return weights / total, LADDER[keep]


def evaluate(weights: np.ndarray, rates: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """The fitted transmittance, which is also exactly what the Rust side will evaluate."""
    return (weights[None, :] * np.exp(-np.outer(columns, rates))).sum(axis=1)


def worst_relative(fitted: np.ndarray, truth: np.ndarray) -> float:
    """Largest relative error across the column grid -- the quantity the correction propagates."""
    return float(np.max(np.abs(fitted - truth) / np.maximum(truth, 1.0e-9)))


def negligible(gas: Gas, wavelength_nm: float, lines: np.ndarray, u_max: float) -> bool:
    """Whether the channel can be written off without a line-by-line pass, provably.

    `1 - <exp(-tau)> <= <tau>` by Jensen's inequality, and `<tau>` over the window is the integrated
    band strength times the column divided by the window width. So a mean depth below the floor is a
    *bound* on the transmittance deficit, not an estimate of it -- which is the difference between
    skipping a channel and hoping about one.
    """
    lo, hi = channel_window(wavelength_nm)
    inside = lines[(lines[:, CENTRE] > lo) & (lines[:, CENTRE] < hi)]
    if inside.size == 0:
        return True
    mean_depth = inside[:, STRENGTH].sum() * gas.molecules_per_unit * u_max / (hi - lo)
    return mean_depth < NEGLIGIBLE_DEPTH


def power_law_partition(table: dict[int, float], temperature: float) -> float:
    """The approximation TIPS replaces, kept only so the header can say what it would have cost."""
    del table
    return (296.0 / temperature) ** 1.5


def costed_checks(
    gas: Gas,
    wavelength_nm: float,
    lines: np.ndarray,
    partition: dict[int, dict[int, float]],
) -> list[tuple[str, float]]:
    """Every choice this file makes, priced as what it does to the channel transmittance.

    Reported as a relative change in `T`, because that is the quantity the correction divides by. A
    percentage of a cross-section that is itself at its own noise floor reads alarming and means
    nothing -- the same trap `gen_ozone_cross_section.py` records having fallen into.
    """
    standard = profile(LAYERS, REFERENCE_SURFACE_HPA)
    slit, depths = channel_depths(gas, wavelength_nm, lines, standard, partition)
    columns = column_grid(gas, [(slit, depths)])
    baseline = transmittance(slit, depths, columns)

    def against(pair: tuple[np.ndarray, np.ndarray]) -> float:
        return worst_relative(transmittance(*pair, columns), baseline)

    def varied(**kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        layers = kwargs.pop("layers", standard)
        return channel_depths(gas, wavelength_nm, lines, layers, partition, **kwargs)

    checks = [
        (
            f"{2 * LAYERS} layers instead of {LAYERS}",
            against(varied(layers=profile(2 * LAYERS, REFERENCE_SURFACE_HPA))),
        ),
        (
            f"a grid of {GRID_SPACING / 2.0} cm^-1",
            against(varied(spacing=GRID_SPACING / 2.0)),
        ),
        (
            "twice the line-wing cutoff -- see the continuum note above",
            against(varied(cutoff_scale=2.0)),
        ),
        (
            "(296/T)^1.5 instead of the TIPS tables",
            against(varied(ratio=power_law_partition)),
        ),
    ]
    if gas.mixing_ratio is None:
        for reference_cm in (0.5, 5.0):
            checks.append(
                (
                    f"the self-broadening profile scaled to {reference_cm} cm, not "
                    f"{SELF_BROADENING_REFERENCE_CM}",
                    against(
                        varied(
                            layers=profile(
                                LAYERS, REFERENCE_SURFACE_HPA, reference_cm=reference_cm
                            )
                        )
                    ),
                )
            )
    checks.append(
        (
            f"the whole table at {REFERENCE_SURFACE_HPA:.0f} hPa only, with no pressure axis",
            max(
                against(varied(layers=profile(LAYERS, end)))
                for end in (PRESSURE_NODES[0], PRESSURE_NODES[-1])
            ),
        )
    )

    rate_sets = [rates_at(slit, depths, LADDER)]
    for terms in (8, TERMS, 24):
        weights, quantiles = exponential_sum(columns, [baseline], rate_sets, terms)
        rates = rates_at(slit, depths, quantiles)
        checks.append(
            (
                f"the {terms}-term fit against the line-by-line curve",
                worst_relative(evaluate(weights, rates, columns), baseline),
            )
        )
    return checks


def column_grid(gas: Gas, states: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """The channel's own column grid: the gas's range, stopped where *any* pressure goes opaque.

    The ceiling is the minimum over the pressure nodes rather than the reference node's, because one
    row states one domain and it has to be a domain every column of that row is valid on.
    """
    u_min, u_max, points = COLUMN_GRID[gas.name]
    ceiling = min(usable_ceiling(slit, depths, u_min, u_max) for slit, depths in states)
    return np.geomspace(u_min, ceiling, points)


Row = tuple[str, float, float, np.ndarray, np.ndarray, float]

# Set once per worker process, because the line catalogue and the profiles are megabytes and every
# channel needs all of them. Pickling them per task instead would cost more than the task.
SHARED: dict[str, object] = {}


def share(
    catalogue: dict[str, np.ndarray],
    partition: dict[int, dict[int, float]],
    profiles: list[list[tuple[float, float, float, float, float]]],
) -> None:
    SHARED.update(catalogue=catalogue, partition=partition, profiles=profiles)


def channel_row(task: tuple[str, float]) -> Row:
    """One published wavelength for one gas: the line-by-line pass at every pressure, then the fit."""
    gas = next(entry for entry in GASES if entry.name == task[0])
    wavelength = task[1]
    catalogue = SHARED["catalogue"]
    partition = SHARED["partition"]
    profiles = SHARED["profiles"]
    lines = catalogue[gas.name]

    if negligible(gas, wavelength, lines, COLUMN_GRID[gas.name][1]):
        rates = np.zeros((len(PRESSURE_NODES), 1))
        return (
            gas.name,
            wavelength,
            COLUMN_GRID[gas.name][1],
            np.array([1.0]),
            rates,
            0.0,
        )

    states = [
        channel_depths(gas, wavelength, lines, layers, partition) for layers in profiles
    ]
    columns = column_grid(gas, states)
    curves = [transmittance(slit, depths, columns) for slit, depths in states]
    rate_sets = [rates_at(slit, depths, LADDER) for slit, depths in states]
    weights, quantiles = exponential_sum(columns, curves, rate_sets)
    rates = np.array([rates_at(slit, depths, quantiles) for slit, depths in states])
    error = max(
        worst_relative(evaluate(weights, row, columns), curve)
        for row, curve in zip(rates, curves, strict=True)
    )
    return (gas.name, wavelength, float(columns.max()), weights, rates, error)


def main() -> None:
    print("fetching partition functions", flush=True)
    partition = {
        global_id: fetch_partition(global_id)
        for gas in GASES
        for global_id in gas.isotopologues
    }

    lo = 1.0e7 / (max(WAVELENGTHS_NM) + SLIT_HALF_BASE_NM) - 2.0 * CUTOFF_STRONG
    hi = 1.0e7 / (min(WAVELENGTHS_NM) - SLIT_HALF_BASE_NM) + 2.0 * CUTOFF_STRONG

    catalogue, counts = {}, {}
    for gas in GASES:
        print(f"fetching {gas.name} lines, {lo:.0f}-{hi:.0f} cm^-1", flush=True)
        catalogue[gas.name] = fetch_all_lines(gas, lo, hi)
        counts[gas.name] = len(catalogue[gas.name])

    profiles = [profile(LAYERS, hpa) for hpa in PRESSURE_NODES]
    for hpa, layers in zip(PRESSURE_NODES, profiles, strict=True):
        print(
            f"{hpa:7.2f} hPa: {len(layers)} layers to {layers[-1][0]:.4f} hPa, "
            f"{100.0 * sum(layer[3] for layer in layers[: len(layers) // 8]):.1f} % of the "
            f"water in the lowest eighth, surface VMR {layers[0][4]:.5f}",
            flush=True,
        )

    # A node that reproduces its neighbour is a node this file publishes and does not have. #369
    # shipped 1055 rows whose top two nodes were bit-identical, because a clip made 1040 hPa mean
    # 1013.25 without saying so, and nothing here noticed. The invariant is checked at the profiles,
    # where it costs microseconds and names the cause, and again at the rates, where it costs hours
    # and names the symptom -- because a distinct profile can still fit to identical rates, and then
    # the axis is decorative for that channel whatever the profiles say.
    for index in range(len(PRESSURE_NODES) - 1):
        if profiles[index] == profiles[index + 1]:
            raise SystemExit(
                f"the {PRESSURE_NODES[index]:g} and {PRESSURE_NODES[index + 1]:g} hPa nodes are "
                "the same profile, so the pressure axis has one fewer node than it advertises"
            )

    tasks = [(gas.name, wavelength) for gas in GASES for wavelength in WAVELENGTHS_NM]
    workers = max(1, min(len(tasks), (os.cpu_count() or 2) - 2))
    print(f"{len(tasks)} channels over {workers} workers", flush=True)
    rows: list[Row] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=share,
        initargs=(catalogue, partition, profiles),
    ) as pool:
        for row in pool.map(channel_row, tasks, chunksize=1):
            rows.append(row)
            print(
                f"  {row[0]:4} {row[1]:6.0f} nm  u <= {row[2]:6.3f}  "
                f"terms {row[3].size:2d}  fit {row[5]:8.2e}  "
                f"({len(rows)}/{len(tasks)})",
                flush=True,
            )

    for gas_name, wavelength, _, _, rates, _ in rows:
        if not rates.any():
            continue  # a channel with no absorption at all: identical zeros are the answer
        for index in range(len(PRESSURE_NODES) - 1):
            if np.array_equal(rates[index], rates[index + 1]):
                raise SystemExit(
                    f"{gas_name} {wavelength:.0f} nm has identical rates at "
                    f"{PRESSURE_NODES[index]:g} and {PRESSURE_NODES[index + 1]:g} hPa"
                )

    skipped = {
        gas.name: sum(1 for row in rows if row[0] == gas.name and row[3].size == 1)
        for gas in GASES
    }
    worst_fit = {
        gas.name: max(
            ((row[5], row[1]) for row in rows if row[0] == gas.name), default=(0.0, 0.0)
        )
        for gas in GASES
    }
    checks = {
        f"{gas.name} {wavelength:.0f} nm": costed_checks(
            gas, wavelength, catalogue[gas.name], partition
        )
        for gas, wavelength in (
            (WATER, 940.0),
            (WATER, 1380.0),
            (OXYGEN, 760.0),
            (CARBON_DIOXIDE, 2010.0),
        )
    }

    write(rows, counts, skipped, worst_fit, checks)
    for gas in GASES:
        error, wavelength = worst_fit[gas.name]
        print(
            f"{gas.name}: {len(WAVELENGTHS_NM) - skipped[gas.name]} channels carry "
            f"absorption, worst fit {error:.2e} at {wavelength:.0f} nm",
            flush=True,
        )


def write(
    rows: list[Row],
    counts: dict[str, int],
    skipped: dict[str, int],
    worst_fit: dict[str, tuple[float, float]],
    checks: dict[str, list[tuple[str, float]]],
) -> None:
    """The fixture, header first, with every choice priced in the header rather than in a commit."""
    nodes = " ".join(f"{hpa:g}" for hpa in PRESSURE_NODES)
    opaque = {
        gas.name: sum(
            1
            for row in rows
            if row[0] == gas.name and row[2] < COLUMN_GRID[gas.name][1] * (1.0 - 1.0e-9)
        )
        for gas in GASES
    }
    # A term slot the fit drove to zero writes as 0.0000000000 at this file's ten decimals and
    # produces no transport run, so the cost quoted below counts what a consumer actually pays
    # rather than how many slots the fit was handed. The difference is not cosmetic: at 2160 nm it
    # is 234000 product terms against 249600.
    kept = {
        (row[0], row[1]): sum(1 for weight in row[3] if float(f"{weight:.10f}") > 0.0)
        for row in rows
    }
    products = sorted(
        (
            (math.prod(kept[(gas.name, nm)] for gas in GASES), nm)
            for nm in WAVELENGTHS_NM
        ),
        reverse=True,
    )
    active = [
        sum(1 for gas in GASES if kept[(gas.name, nm)] > 1) for nm in WAVELENGTHS_NM
    ]
    overlapping = sum(1 for count in active if count > 1)
    worst_product, worst_product_nm = products[0]
    median_product = products[len(products) // 2][0]
    names = ", ".join(gas.name for gas in GASES)
    out = [
        f"# Band transmittance of {names} as an exponential sum, on RadCalNet's grid.",
        "# Generated by tools/gen_gas_transmittance.py -- do not edit.",
        "#",
        "# T(u) = sum_j w_j exp(-k_j u), sum_j w_j = 1, which is a discretised k-distribution: w_j is",
        "# the fraction of the channel's width absorbing at k_j. That form composes with a scattering",
        "# calculation -- run the transport once per term and weight-sum -- and a single exponential",
        "# does not, because at 10 nm resolution a band is thousands of lines and its transmittance is",
        "# not an exponential in the column.",
        "#",
        f"# The rates carry a surface-pressure axis, {len(PRESSURE_NODES)} nodes at {nodes} hPa. They",
        "# have to: pressure broadening widens every line, so at a fixed column the same channel is a",
        "# different transmittance at a different surface pressure. At 940 nm, between the ends of this",
        "# ladder and taken relative to the smaller of the two, that is 72 % at worst over the fit's own",
        "# column grid and 38.2 % over the columns RadCalNet's own rows carry. The weights",
        "# do not carry the axis: one weight vector is fitted jointly against every pressure at once",
        "# (the design matrices stack, one block per node, against one shared non-negative weight",
        f"# vector), which is what makes this a correlated-k table rather than {len(PRESSURE_NODES)}",
        "# unrelated fits. Measured over five channels -- the weakest and the two saturated water bands,",
        "# O2 760 and CO2 2010 nm: weights carried from the reference node cost 2.3e-4 to 1.7e-2; fitted",
        "# jointly they cost 2.0e-6 to 5.0e-4 on the nodes and 2.1e-6 to 1.2e-3 between them, at the same",
        "# term count. This many nodes and not three: a three-node ladder fits its own nodes just as well",
        "# (1.2e-6 to 3.6e-4) and costs 5.7e-6 to 3.2e-3 between them, which is the interpolation error",
        "# and not the fit's.",
        "#",
        "# Interpolate the rates between nodes; off the ends, refuse rather than extrapolate. The ladder",
        "# is sized so that refusing is rare rather than routine: RadCalNet's own published surface",
        "# pressures span 720.0 to 1031.1 hPa over the two fixtures, inside these ends with margin at",
        "# both. Note that the ceiling is above sea level -- an anticyclone puts more air over a",
        "# near-sea-level site than the standard atmosphere has -- so a ladder topped at 1013.25 would",
        "# have refused 266 of those 1945 rows, 13.7 %. Clamping instead of refusing is the other wrong",
        "# answer: at 940 nm the 1013.25-1040 segment runs 0.034 %/hPa at the median and 0.097 %/hPa at",
        "# worst over those columns, so clamping the 17.8 hPa the archive overshoots by charges 0.6 % of",
        "# channel transmittance and 1.7 % at worst, silently -- an order of magnitude above the fit",
        "# error above. The node is what makes both moot.",
        "#",
        "# Line parameters: the HITRAN database, https://hitran.org, through its line-by-line API,",
        "# cited as I. E. Gordon et al., J. Quant. Spectrosc. Radiat. Transfer 353, 109807 (2026).",
        "# HITRAN states the database is free and publishes a citation policy; it grants no",
        "# redistribution, so no line parameter is committed here. What is committed is this file's",
        "# own fit. The edition served is fingerprinted by its line counts over the fetched range:",
        "#   " + "  ".join(f"{gas.name} {counts[gas.name]}" for gas in GASES),
        "# A different count is a different edition and the fit must be regenerated, not reconciled.",
        "# Partition functions: HITRAN's TIPS tables, per isotopologue, tabulated not approximated.",
        "# Isotopologue ids, abundances and molar masses: HITRAN's own molparam.txt.",
        "# Profile: U.S. Standard Atmosphere 1976 (NOAA-S/T 76-1562), a U.S. Government work; the",
        "# water shape is AFGL mid-latitude summer, scaled to the column asked about.",
        "#",
        "# Units of u, the slant column. Water is the only gas that is not well mixed, so it is the only",
        "# one whose column is an atmospheric measurement rather than a pressure:",
        f"#   H2O  precipitable cm times air mass   -- one unit is "
        f"{WATER.molecules_per_unit:.4e} molecules/cm^2",
    ]
    for gas in GASES:
        if gas.mixing_ratio is not None:
            out.append(
                f"#   {gas.name:4} (P / 1013.25 hPa) times air mass -- "
                f"{gas.molecules_per_unit:.4e} molecules/cm^2, {gas.mixing_ratio:.4g} by volume"
            )
    out += [
        "# Both quantities are columns radcalnet_toa.txt already carries per row -- the water column and",
        "# the surface pressure -- so every well-mixed gas shares one pressure-scaled number and differs",
        "# only in how many molecules a unit of it stands for.",
        "#",
        "# u_max is this channel's own domain: the fit was made over u in "
        f"[{COLUMN_GRID['H2O'][0]:g} (H2O) or",
        f"# {WELL_MIXED_GRID[0]:g} (well mixed), u_max] and is quoted nowhere else. u_max is the "
        "smaller of the gas's",
        f"# grid ceiling and the column at which the channel's transmittance falls to "
        f"{TRANSMITTANCE_FLOOR:.0e}. Past",
        "# that the channel is opaque -- a correction that divides by T would be multiplying its own",
        "# input error by a thousand -- and a caller handed a larger column must refuse the channel,",
        "# not extrapolate the row. Channels that stop short of the grid ceiling for that reason:",
        "#   " + "  ".join(f"{gas.name} {opaque[gas.name]}" for gas in GASES),
        "#",
        f"# Of the {len(WAVELENGTHS_NM)} published wavelengths, each gas absorbs at:",
        "#   "
        + "  ".join(
            f"{gas.name} {len(WAVELENGTHS_NM) - skipped[gas.name]}" for gas in GASES
        ),
        "# A channel is called transparent, and given a single unit weight at zero rate, only when its",
        f"# mean optical depth is below {NEGLIGIBLE_DEPTH:.0e} at the largest column on the grid. Since",
        "# 1 - <exp(-tau)> <= <tau> by Jensen, that is a *bound* on the transmittance deficit and not an",
        "# estimate of it, which is why the bar can be set that low and still mean something.",
        "#",
        "# At that bar water is nowhere transparent. There is no wavelength on this grid where its lines",
        "# are provably absent -- a fact about 10 nm resolution rather than about the water, and the",
        "# reason a window/band classification is the wrong shape of answer here. How weak a channel is",
        "# belongs to the row and not to a label: <tau> = sum_j w_j k_j u, computable from the row alone",
        "# at whatever column the caller actually has, with no line-by-line pass and no threshold in it.",
        "#",
        "# Where two or more gases act in one channel their distributions multiply -- weights multiply,",
        "# depths add -- so a consumer runs the transport once per *product* term. That is this table's",
        "# own cost to use and it is written here rather than left to be discovered:",
        f"#   {overlapping} of {len(WAVELENGTHS_NM)} channels carry more than one absorber",
        f"#   the median channel is {median_product} product terms, the worst {worst_product} at "
        f"{worst_product_nm:.0f} nm",
        "# Everywhere else the other gases are a single unit weight at zero rate and the product is the",
        "# absorbing one unchanged -- exactly, not approximately, which is why that encoding is worth",
        "# the row. Where several are real the product assumes their line positions inside the channel",
        "# are uncorrelated. That is the standard correlated-k assumption and it is not free; it is also",
        "# not tested here, because testing it needs the line-by-line pair and this file's output is the",
        "# fit. What can be said is where it is even reachable: those channels and no others.",
        "#",
        "# Two approximations are named here rather than closed, because both are physics this file",
        "# does not carry and neither goes away by turning a parameter up:",
        "#",
        "#   Self-broadening. Water broadens its own lines, so the cross-section depends on the water",
        f"#   profile and not only on its integral. The profile is normalised to "
        f"{SELF_BROADENING_REFERENCE_CM:g} precipitable cm",
        "#   and the same rates are then used at every column, which costs 2.4e-2 if the true column",
        "#   was 1 cm and 6.7e-2 if it was 5 cm -- the two ends of what RadCalNet's sites present.",
        "#   Closing it means a second axis in the column itself, not a better fit.",
        "#",
        f"#   The far line wing. Lines are cut at {CUTOFF_STRONG:g} cm^-1 and the ladder does not",
        "#   converge: against a 500 cm^-1 reference, 25 cm^-1 costs 1.9e-1, 50 costs 7.3e-2, 100",
        "#   costs 2.2e-2, 200 costs 6.0e-3 -- first order in 1/cutoff, which is the signature of a",
        "#   term with the wrong shape rather than a truncation set too tight. It is: the far wing of",
        "#   a water line is not Lorentzian, and every line-by-line code closes the gap with a",
        "#   continuum (MT_CKD) calibrated against exactly this truncation. So 25 cm^-1 is kept,",
        "#   deliberately, because it is the cutoff the missing term is defined against -- widening it",
        "#   without the continuum moves away from MODTRAN, not towards it. Ticket #358.",
        "#",
        "# Worst relative error of the fit against its own line-by-line curve:",
    ]
    for gas_name, (error, wavelength) in worst_fit.items():
        out.append(f"#   {gas_name:4} {error:.2e} at {wavelength:.0f} nm")
    out += [
        "#",
        "# Every choice this file makes, priced as the worst relative change it produces in the",
        "# channel transmittance across the column grid -- the quantity the correction divides by:",
    ]
    for channel, entries in checks.items():
        out.append(f"#   {channel}")
        for what, cost in entries:
            out.append(f"#     {cost:9.2e}  {what}")
    out += [
        "#",
        f"# Columns: gas, wavelength_nm, u_max, then {TERMS} weights, then {TERMS} rates for each of",
        f"# the {len(PRESSURE_NODES)} pressure nodes in the order {nodes} hPa. A row's weights sum to 1",
        "# and every rate block is ascending over the terms the fit kept, because all of them are read",
        "# at the same ascending quantiles of that channel's own k-distribution -- so one term index",
        "# means one part of the channel at every pressure, which is what lets a reader interpolate a",
        f"# rate between two nodes. Rows are padded to {TERMS} terms with a zero weight and a zero rate,",
        "# so that every row has the same width; the padding sits at the end, after the kept terms, and",
        "# a reader must drop zero-weight terms rather than read the run of trailing zeros as rates.",
        "#",
        "# gas wavelength_nm u_max "
        + " ".join(f"w{j}" for j in range(1, TERMS + 1))
        + " "
        + " ".join(
            f"k{j}@{hpa:g}" for hpa in PRESSURE_NODES for j in range(1, TERMS + 1)
        ),
    ]

    for gas_name, wavelength, u_max, weights, rates, _ in rows:
        order = np.argsort(rates[0])
        weights, rates = weights[order], rates[:, order]
        pad = TERMS - weights.size
        fields = [f"{weight:.10f}" for weight in weights] + ["0.0000000000"] * pad
        for block in rates:
            fields += [f"{rate:.6e}" for rate in block] + ["0.000000e+00"] * pad
        out.append(f"{gas_name} {wavelength:.0f} {u_max:.6e} " + " ".join(fields))

    OUT.write_text("\n".join(out) + "\n", encoding="ascii")
    print(f"wrote {OUT} ({len(out)} lines)", flush=True)


if __name__ == "__main__":
    main()
