"""RadCalNet: a measured surface, the atmosphere measured over it, and MODTRAN's answer.

RadCalNet instruments a handful of stable targets and publishes, every half hour in cloud-free
conditions, the surface reflectance measured on the ground, the atmosphere measured over it, and the
top-of-atmosphere reflectance MODTRAN 5.3.3 produces from the two. That makes it the rung above
`gen_aeronet_oracle.py`: there the atmosphere was measured and the surface was chosen, here both are
measured and the scene is one that happened.

Its Quick Start Guide names this use:

  "These additional data could be used for various purposes, e.g.: a) to alternatively simulate
   top-of-atmosphere reflectance over the RadCalNet site with (radiative transfer) tools differing
   from the MODTRAN model used for the RadCalNet processing"

**Terms.** RadCalNetDataPolicy_v1.1.pdf makes the data "freely and publically available" and asks
that users acknowledge RadCalNet and the specific sites. There is no redistribution prohibition, so
the excerpt is committed, with the policy quoted in full in the fixture header. Credentials come from
`RADCALNET_USER` / `RADCALNET_PASS` and are in no file here. Magellium's `radcalnet_api_client.py`
carries "All rights reserved" with no licence grant; its endpoints are facts and were read as such,
and no code was taken from it.

**What MODTRAN is told**, from the site default parameters in the format specification, because a
comparison against a model whose settings are unknown is not a comparison: `MODEL: 2`, `IEMSCT: 2`,
`SURREF: LAMBER`, `NSURF: 1`, `DIS: t`, `NSTR: 8`, `CO2MX: 390`, `IHAZE: 2` (rural), `NSSALB: 0`,
`H1: 100.000`, and -- the one that matters most here -- Card 3A1 `IPH: 2` with Card 3A2 `G: 0.500`,
a Henyey-Greenstein aerosol phase function of asymmetry 0.5. Card 4 reports it through a 10 nm FWHM
triangular slit on a 10 nm grid from 400 to 2500 nm. So the scatterer in the Rust test is quoted
from their deck, not chosen; `NSSALB: 0` hands the single-scattering albedo to MODTRAN's internal
rural model and publishes nothing, which leaves exactly one free parameter and the test says so.

Four things a naive version gets wrong, each of which cost a debugging round here:

  1. **Fill values.** Specification section 1.6: 9999 no data, 9998 not processed to TOA, 9997
     anomalous atmosphere, 9996 anomalous surface, 9995 cloudy, and an input file's fill is copied
     into the output file. Read as a number, 9999 is a reflectance of 9999. Every one is dropped
     here and the count is printed.
  2. **The uncertainty is flag-encoded.** Section 3.5: a value is reported *negative* when an input
     sits outside the uncertainty look-up table's range, and has **a 9 added to the left of the
     decimal point** inside predetermined absorption regions where the uncertainty can exceed 10 %.
     So `9.0155` is 0.0155 with a caveat and `-0.0155` is 0.0155 out of range. Taking either at face
     value puts a 900 % uncertainty in the denominator of a gate.
  3. **The `.input` files have no blank line** between the atmospheric block and the reflectance
     block. A parser that ends the atmospheric block at a blank line swallows all 211 reflectance
     rows and hands back the *uncertainty* table instead -- a plausible-looking number that is not a
     reflectance. Nothing here counts lines: blocks are delimited by whether the first field parses
     as an integer.
  4. **Gas bands are not where reading the spectrum by eye says they are.** The conventional SWIR
     "windows" at 1240, 1550 and 2130 nm carry 1.1 %, 2.8 % and 4.7 % of water absorption over the
     range of columns in this fixture -- measured by
     `radcalnets_own_data_says_which_windows_are_windows` from these files alone, with no
     radiative-transfer model in the loop. That is why `radcalnet_absorption.txt` exists: it carries
     the whole 400-2500 nm grid so the comparable intervals are read off data rather than picked.

**Every published key is classified, and the generator will not run past one that is not.** `parse`
returns whole blocks and the readers take what they need, which left "we use everything RadCalNet
gives us" as an assertion nobody checked -- `CONSUMED` and `WAIVED` make it a list with a reason
against each omission, and `audit` raises on any key in neither. Writing that list turned up two
invariants worth holding rather than waiving, both now checked on every downloaded day: the file's
own `Year` / `DOY(U)` against the date in its filename, which is where every row's epoch comes from
and what the drift regression reads; and the `.input`'s atmospheric block against the `.output`'s,
because the surface reflectance comes from one file and the top-of-atmosphere reflectance from the
other and they are comparable only if both describe the same atmosphere at the same times.

    RADCALNET_USER=... RADCALNET_PASS=... uv run python tools/gen_radcalnet_oracle.py

writes `crates/spacelab-earthvision/tests/fixtures/radcalnet_toa.txt` (selected wavelengths, the
Monte-Carlo comparison) and `crates/spacelab-earthvision/tests/fixtures/radcalnet_absorption.txt`
(the whole grid, the model-free classification). Stdlib only.
"""

from __future__ import annotations

import base64
import math
import os
import pathlib
import statistics
import textwrap
import urllib.error
import urllib.request

API = "https://www.radcalnet.org/api/json/"

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / (
    "crates/spacelab-earthvision/tests/fixtures"
)
OUT = FIXTURES / "radcalnet_toa.txt"
OUT_ABSORPTION = FIXTURES / "radcalnet_absorption.txt"
OUT_HOLDOUT = FIXTURES / "radcalnet_holdout.txt"

DOWNLOADED = "2026-08-20"

# Specification section 1.6. A fill value in an input file is copied to the output file.
FILL = {"9999", "9998", "9997", "9996", "9995"}

# Every key the two files publish, and what becomes of it.
#
# `parse` hands back whole blocks and the readers below take what they need, which is convenient and
# leaves "we use everything RadCalNet gives us" as an assertion nobody checks. These two literals
# make it a list. A key in neither is a measurement nobody has decided either to use or to waive, and
# a renamed one would otherwise be read past in silence, so `audit` refuses to go on when it meets
# one -- the same discipline as the hold-out's own depth check further down.
#
# The waivers carry their reason because "we do not read it" and "it cannot enter the answer" are
# different statements and only the second is a defence.
CONSUMED = {
    "Site": "the fixture's site column",
    "UTC": "the fixture's time column",
    "Zen": "solar zenith -- mu_s, and the air mass every gate is regressed on",
    "P": "surface pressure, which scales the Rayleigh depth",
    "WV": "columnar water: read into the fixture, and bounded rather than charged, because the slab has no gaseous absorption to take a derivative of",
    "O3": "columnar ozone, applied outside the scattering layer with a published cross-section",
    "AOD": "aerosol optical depth at 550 nm",
    "Ang": "the Angstrom exponent that carries AOD to every other wavelength",
    "Type": "R/L -- which retrieval produced the aerosol, the column that stops IHAZE: 2 being quoted from a deck the fixture cannot show",
    "Year": "cross-checked against the filename rather than written out; the epoch the drift regression reads comes from the filename, so the file's own opinion of it is an invariant worth holding",
    "DOY(U)": "the same cross-check, on the day of year",
}

WAIVED = {
    "Lat": "the geometry is published directly as Zen, so nothing here recomputes it from a position",
    "Lon": "the same, and the two local-time columns it would explain are themselves waived below",
    "Alt": "the slab reads the *measured* surface pressure, which is the altitude's effect on the answer and more besides",
    "T": "a grey scattering slab has no temperature: Bodhaine's depth is a function of wavelength and of the column number density P/(m g), so T has no path into the answer and perturbing it would return a zero describing the model rather than the scene",
    "DOY(L)": "local solar day, redundant with UTC and the date",
    "Local": "local solar time, redundant with UTC",
    "Azi": "solar azimuth, degenerate at nadir: only mu_s enters a plane-parallel slab and the relative azimuth has nothing to be relative to",
    "esd": "Earth-Sun distance, which cancels in a *reflectance* comparison -- both sides have the solar irradiance divided out, and esd enters only through it",
}

# Six instruments at six targets, chosen after scanning ~22 days per instrument across each
# archive: every site the portal actually serves except the two empty directories (`ERUS`,
# `Australia`), the eight-file pilot `GABU` -- whose files identify themselves as `GONA01` in their
# own header and would double-count Gobabeb -- and the two sparse Baotou instruments.
#
# Three days each, and the roles are the point. A **sweep** day is one where the atmosphere barely
# moves while the sun does, so several times from it separate the geometry error from the aerosol
# error; the other two are that site's own extremes. Between them the fixture spans surface pressure
# 720-1029 hPa, solar zenith 2.6-81.5 deg, Angstrom 0.08-1.8 and columnar water 0.24-3.37 cm, which
# no single site comes close to on its own.
SCENES = [
    # site  instrument  year  doy  role     why this day
    (
        "RVUS",
        "RVUS00",
        2021,
        324,
        "sweep",
    ),  # AOD fixed at 0.021-0.025, zenith 58-77 deg
    (
        "RVUS",
        "RVUS00",
        2020,
        303,
        "clean",
    ),  # AOD 0.013, the thinnest column in the fixture
    ("RVUS", "RVUS00", 2017, 99, "loaded"),  # AOD 0.105-0.128, this site's dustiest
    (
        "GONA",
        "GONA01",
        2022,
        12,
        "sweep",
    ),  # zenith down to 2.6 deg -- the sun nearly overhead
    ("GONA", "GONA01", 2020, 174, "clean"),  # AOD 0.013
    (
        "GONA",
        "GONA01",
        2020,
        269,
        "loaded",
    ),  # AOD 0.245-0.288 at Angstrom 1.81, fine dust
    (
        "LCFR",
        "LCFR01",
        2023,
        340,
        "sweep",
    ),  # zenith 66-81.5 deg, AOD fixed -- the long air masses
    ("LCFR", "LCFR01", 2016, 45, "clean"),  # AOD 0.009
    ("LCFR", "LCFR01", 2021, 208, "humid"),  # 3.37 cm of water, the wettest column
    ("BTCN", "BTCN02", 2018, 303, "sweep"),  # zenith 54-72 deg over grey gravel
    ("BTCN", "BTCN02", 2022, 289, "clean"),  # AOD 0.042
    ("BTCN", "BTCN02", 2024, 114, "ozone"),  # 396 DU, near the top of the ozone range
    (
        "BSCN",
        "BSCN00",
        2023,
        272,
        "sweep",
    ),  # Baotou sand, a second target 2 km from BTCN
    ("BSCN", "BSCN00", 2022, 266, "clean"),  # AOD 0.024
    ("BSCN", "BSCN00", 2018, 135, "loaded"),  # AOD 0.248-0.298 at 856 hPa
    (
        "GSCN",
        "GSCN01",
        2023,
        210,
        "sweep",
    ),  # Golmud at 2824 m: 720-741 hPa, a third of the Rayleigh
    ("GSCN", "GSCN01", 2024, 231, "clean"),  # AOD 0.034
    ("GSCN", "GSCN01", 2023, 303, "coarse"),  # Angstrom 0.077 and zenith to 80.2 deg
]

# Times taken from a sweep day. Five is enough to see a trend in air mass and few enough that the
# Monte-Carlo tests stay near the AERONET test's cost in the workspace suite.
SWEEP_TIMES = 5

# Days per site in the hold-out, which is a different fixture built by a different rule. `SCENES`
# above was chosen after scanning the archive -- the roles say so out loud -- and a set chosen that
# way can only ever report how well the choosing went. The hold-out samples each site's sorted
# archive on the date instead, so nothing about a day's weather enters its selection.
# 20 rather than a handful because this fixture also carries the epoch and driver regressions, whose
# standard errors fall as 1/sqrt(n): at five per site two of the six per-site epoch slopes sat close
# enough to their own error bar that the sample size, not the reference, decided the verdict. The
# shallowest archive here is Golmud's 177 days, so 20 is available at every site without the number
# becoming per-site.
HOLDOUT_PER_SITE = 20
# Stride through the hold-out's season targets. Coprime to `HOLDOUT_PER_SITE`, so the targets are a
# permutation of the year rather than a ramp; 13 measured the flattest of the coprime strides.
SEASON_STRIDE = 13
# A guard against a floating-point tie cycling the descent, not a bound it reaches: every archive
# settles in under ten sweeps.
BALANCE_SWEEPS = 64

# The wavelengths of `ozone_cross_section.txt`, so the two fixtures join on one column. Nine are gas
# windows and five are diagnostics: 760 and 940 nm sit inside the O2-A and water bands on purpose,
# and 1240, 1550 and 2130 nm are the conventional SWIR windows this fixture shows are not windows.
WAVELENGTHS_NM = [
    410,
    440,
    490,
    550,
    600,
    660,
    680,
    760,
    870,
    940,
    1030,
    1240,
    1550,
    2130,
]

# The whole published grid, for the model-free absorption classification.
ABSORPTION_NM = list(range(400, 2501, 10))

# What the two spectral tables write where RadCalNet published no reflectance, in its own fill
# convention. Nothing is filtered here beyond that: whether a reflectance is bright enough to divide
# by is the consumer's judgement, and `radcalnet_measured_radiance.rs` makes it explicitly.
MISSING = "  9999.0"


def credentials() -> tuple[str, str]:
    user = os.environ.get("RADCALNET_USER")
    password = os.environ.get("RADCALNET_PASS")
    if not user or not password:
        raise SystemExit(
            "RADCALNET_USER and RADCALNET_PASS must be set. RadCalNet data needs a portal "
            "account (https://www.radcalnet.org), and a credential does not belong in a file "
            "this repository holds."
        )
    return user, password


def get(path: str, auth: tuple[str, str]) -> str:
    """One authenticated GET against the RadCalNet JSON API."""
    token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode("ascii")
    request = urllib.request.Request(
        API + path, headers={"Authorization": f"Basic {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read().decode("ascii")
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise SystemExit(
                f"{API + path}: {error.code}. RADCALNET_USER / RADCALNET_PASS were rejected."
            ) from error
        raise


def listing(site: str, auth: tuple[str, str]) -> list[str]:
    """The site's data-directory file names, from the API's JSON directory listing."""
    names = []
    for line in get(f"{site}/data/", auth).splitlines():
        if '"name"' in line:
            names.append(line.split('"name":"', 1)[1].split('"', 1)[0])
    if not names:
        raise SystemExit(f"{site}: empty data listing")
    return names


def parse(
    text: str,
) -> tuple[dict[str, str], list[dict[str, list[str]]], list[dict[int, list[str]]]]:
    """Header, the key-led blocks, and the numeric blocks, per the format specification.

    Nothing counts lines. `.output` files separate their blocks with three blank lines and `.input`
    files with none, and the atmospheric block has fifteen rows in one and twelve in the other, so
    the only reliable delimiter is whether a line's first field parses as an integer wavelength.

    Returns the key-led blocks in order -- `[0]` the atmosphere, `[1]` its uncertainties -- and the
    numeric blocks in order -- `[0]` the reflectance, `[-1]` its uncertainties.
    """
    lines = text.splitlines()
    head = {}
    for line in lines[:4]:
        key, value = line.split("\t", 1)
        head[key.rstrip(":")] = value

    keyed: list[dict[str, list[str]]] = []
    numeric: list[dict[int, list[str]]] = []
    current_keyed: dict[str, list[str]] = {}
    current_numeric: dict[int, list[str]] = {}
    for line in lines[4:]:
        fields = line.split("\t")
        # `.input` rows end in trailing tabs -- one on most rows, two on some uncertainty rows --
        # and `.output` rows end in none, so a naive split gives the two files different widths for
        # the same thirteen half-hourly columns. Nothing ever indexed the extra fields, because the
        # column numbers come from the shorter file, but `float("")` was one selection rule away and
        # comparing the two files' atmospheres is a comparison of widths until they are gone. A line
        # left with nothing is a blank one, which is how `.output` separates its blocks.
        while fields and fields[-1] == "":
            fields.pop()
        if not fields:
            continue
        if fields[0].isdigit():
            if current_keyed:
                keyed.append(current_keyed)
                current_keyed = {}
            current_numeric[int(fields[0])] = fields[1:]
        elif fields[0].endswith(":"):
            if current_numeric:
                numeric.append(current_numeric)
                current_numeric = {}
            current_keyed[fields[0].rstrip(":")] = fields[1:]
    if current_keyed:
        keyed.append(current_keyed)
    if current_numeric:
        numeric.append(current_numeric)
    return head, keyed, numeric


def audit(
    name: str,
    head: dict[str, str],
    keyed: list[dict[str, list[str]]],
    numeric: list[dict[int, list[str]]],
) -> None:
    """Refuse to read a file carrying a key nobody has classified.

    Called on every file downloaded rather than on the first, because a key that appears at one site
    and not another is exactly the case a single sample would miss.
    """
    unknown = sorted(
        {key for block in (head, *keyed) for key in block} - set(CONSUMED) - set(WAIVED)
    )
    if unknown:
        raise SystemExit(
            f"{name} publishes {unknown}, which is in neither CONSUMED nor WAIVED. Classify it "
            "before this fixture is written: an unclassified key is a measurement nobody has "
            "decided to use or to waive, and a renamed one is a silent loss."
        )
    # `parse` returns the reflectance block and its uncertainties, and the readers take [0] and [-1].
    # A third block would make [-1] quietly mean something else.
    if len(numeric) != 2:
        raise SystemExit(
            f"{name} has {len(numeric)} numeric blocks, not the reflectance and its uncertainty "
            "that every reader here assumes"
        )


def same_atmosphere(
    output: dict[str, list[str]], source: dict[str, list[str]], name: str
) -> None:
    """The `.input`'s atmosphere against the `.output`'s, on every key the two share.

    The BOA reflectance comes from one file and the TOA reflectance from the other, and they are only
    comparable if both describe the same atmosphere at the same times. RadCalNet generates the second
    from the first, so they agree -- but "so they agree" is the assumption, and it is one column
    mismatch away from comparing a surface to a top of atmosphere that never sat above it.
    """
    for key in sorted(set(output) & set(source)):
        if output[key] != source[key]:
            raise SystemExit(
                f"{name}: the .input and .output disagree about {key}, so the surface rows and the "
                "top-of-atmosphere rows do not describe the same atmosphere"
            )


def uncertainty(raw: str) -> tuple[float, str]:
    """Decode a reflectance uncertainty and its flags, per specification section 3.5.

    Negative means an input was outside the uncertainty look-up table's range; a 9 to the left of the
    decimal point means a predetermined absorption region where the uncertainty can exceed 10 %.
    """
    value = float(raw)
    flags = []
    if value < 0.0:
        flags.append("lut")
        value = -value
    if value >= 9.0:
        flags.append("band")
        value -= 9.0
    return value, "+".join(flags) if flags else "ok"


def bodhaine_1999(wavelength_um: float) -> float:
    """Rayleigh optical depth of the whole atmosphere at 1013.25 hPa (Bodhaine et al. 1999).

    The same expression `tools/gen_pyspectral_oracle.py` uses, for the same reason: choosing a
    Rayleigh optical-depth model inside the Rust test would put a third implementation's error into
    the residual, where it would be read as ours.
    """
    w2 = wavelength_um * wavelength_um
    numerator = 1.0455996 - 341.29061 / w2 - 0.90230850 * w2
    denominator = 1.0 + 0.0027059889 / w2 - 85.968563 * w2
    return 0.0021520 * numerator / denominator


def clean_columns(
    atmosphere: dict[str, list[str]],
    toa: dict[int, list[str]],
    boa: dict[int, list[str]],
) -> tuple[list[int], int]:
    """Columns where the atmosphere and at least eight of `WAVELENGTHS_NM` survived processing.

    A fill value is the expected way a column goes missing. The other way is a *negative* published
    quantity, and all six of these are physically non-negative: an optical depth, an exponent, an
    angle from the zenith, a pressure, a column of water, a column of ozone. RadCalNet published
    `O3 = -280.0` at Railroad Valley on 2023-305, which an earlier version of this generator copied
    straight through -- and downstream `exp(-tau_3 (1/mu_s + 1/mu_v))` turned it into a transmittance
    *above* one, so the model added light, while the driver regression read the point at ten standard
    deviations from the site's own median and let it produce the whole ozone coefficient.

    The specification's negative-means-out-of-range convention (section 3.5) is documented for the
    *uncertainty* fields. Whether it reaches a value in the atmospheric block is not established
    here, so this does not take the absolute value and call it 280 DU -- that would be inventing a
    reading. An input that cannot be interpreted is dropped, exactly as a fill value is, and counted
    so the number is visible rather than silent.
    """
    columns = []
    unreadable = 0
    for column in range(len(atmosphere["UTC"])):
        published = ("AOD", "Ang", "Zen", "P", "WV", "O3")
        if any(atmosphere[key][column] in FILL for key in published):
            continue
        if any(float(atmosphere[key][column]) < 0.0 for key in published):
            unreadable += 1
            continue
        usable = sum(
            1
            for w in WAVELENGTHS_NM
            if w in toa and toa[w][column] not in FILL and boa[w][column] not in FILL
        )
        if usable >= 8:
            columns.append(column)
    return columns, unreadable


def choose(columns: list[int], role: str, zeniths: list[float]) -> list[int]:
    """Which of a day's clean columns enter the fixture, by a rule rather than by hand.

    A sweep day contributes `SWEEP_TIMES` columns spaced evenly through the clean ones, which is the
    widest range of air mass that day offers. Any other day contributes the single column nearest its
    own median solar zenith -- a representative rather than an extreme, because that day was already
    chosen for an extreme in something else and stacking two would confound them.
    """
    if not columns:
        return []
    if role != "sweep":
        median = statistics.median(zeniths)
        pairs = zip(columns, zeniths, strict=True)
        return [min(pairs, key=lambda pair: abs(pair[1] - median))[0]]
    if len(columns) <= SWEEP_TIMES:
        return list(columns)
    last = len(columns) - 1
    picked = {columns[round(i * last / (SWEEP_TIMES - 1))] for i in range(SWEEP_TIMES)}
    return sorted(picked)


def comparison_rows(
    head: dict[str, str],
    year: int,
    doy: int,
    blocks: tuple[dict[str, list[str]], ...],
    tables: tuple[dict[int, list[str]], ...],
    column: int,
) -> tuple[list[str], int, int]:
    """One column of one day as gate-fixture rows, with what was dropped and what carries a flag.

    Shared by the chosen fixture and the hold-out so the two are the same measurement of the same
    thing. A hold-out written by a second copy of this formatting would be testing the copy.
    """
    atmosphere, atmosphere_unc = blocks
    toa, toa_unc, boa, boa_unc = tables
    rows: list[str] = []
    dropped = 0
    flagged = 0
    time = atmosphere["UTC"][column]
    pressure = float(atmosphere["P"][column])
    zenith = float(atmosphere["Zen"][column])
    for wavelength in WAVELENGTHS_NM:
        cells = [table[wavelength][column] for table in (toa, toa_unc, boa, boa_unc)]
        if any(cell in FILL for cell in cells):
            dropped += 1
            continue
        toa_value, toa_flag = uncertainty(cells[1])
        boa_value, boa_flag = uncertainty(cells[3])
        flag = toa_flag if boa_flag == "ok" else f"{toa_flag}/{boa_flag}"
        if flag != "ok":
            flagged += 1
        rows.append(
            f"{head['Site'].strip():8s} {year}-{doy:03d} {time:>6s} "
            f"{zenith:9.4f} {wavelength:6d} "
            f"{float(cells[2]):8.4f} {boa_value:9.4f} "
            f"{float(cells[0]):8.4f} {toa_value:9.4f} "
            f"{float(atmosphere['AOD'][column]):8.4f} "
            f"{float(atmosphere['Ang'][column]):9.4f} "
            f"{bodhaine_1999(wavelength / 1000.0) * pressure / 1013.25:13.8f} "
            f"{pressure:9.2f} {float(atmosphere['WV'][column]):8.4f} "
            f"{float(atmosphere['O3'][column]):7.1f} "
            f"{float(atmosphere_unc['AOD'][column]):9.4f} "
            f"{float(atmosphere_unc['Ang'][column]):9.4f} "
            f"{float(atmosphere_unc['O3'][column]):8.2f} "
            f"{float(atmosphere_unc['P'][column]):8.2f} "
            f"{float(atmosphere_unc['WV'][column]):8.3f} {flag:>8s} "
            f"{atmosphere['Type'][column]:>4s}"
        )
    return rows, dropped, flagged


def day(
    site: str,
    instrument: str,
    year: int,
    doy: int,
    names: list[str],
    auth: tuple[str, str],
) -> tuple[
    dict[str, str], tuple[dict[str, list[str]], ...], tuple[dict[int, list[str]], ...]
]:
    """Both files of one site-day, parsed: the `.output` for TOA and the `.input` for BOA."""
    same_day = [n for n in names if n.startswith(f"{instrument}_{year}_{doy:03d}_")]
    output = next(n for n in same_day if n.endswith(".output"))
    source = next(n for n in same_day if n.endswith(".input"))
    head, keyed, numeric = parse(get(f"{site}/data/{output}", auth))
    boa_head, boa_keyed, boa_numeric = parse(get(f"{site}/data/{source}", auth))

    audit(output, head, keyed, numeric)
    audit(source, boa_head, boa_keyed, boa_numeric)
    same_atmosphere(keyed[0], boa_keyed[0], f"{instrument}_{year}_{doy:03d}")
    same_atmosphere(keyed[1], boa_keyed[1], f"{instrument}_{year}_{doy:03d}")
    # The date of every row comes from the filename, and the drift regression reads it as an epoch.
    # The file states its own opinion twice; disagreement would put a whole day at the wrong year.
    for key, expected in (("Year", year), ("DOY(U)", doy)):
        stated = {int(value) for value in keyed[0][key]}
        if stated != {expected}:
            raise SystemExit(
                f"{output} is named for {year}-{doy:03d} but its own {key} column says {stated}"
            )

    return (
        head,
        (keyed[0], keyed[1]),
        (numeric[0], numeric[-1], boa_numeric[0], boa_numeric[-1]),
    )


def archive(names: list[str], site: str, instrument: str) -> list[tuple[int, int]]:
    """Every day this instrument published both files for, minus the days `SCENES` already took."""
    used = {(y, d) for s, _, y, d, _ in SCENES if s == site}
    kinds: dict[tuple[int, int], set[str]] = {}
    for name in names:
        if not name.startswith(f"{instrument}_"):
            continue
        parts = name.split("_")
        kinds.setdefault((int(parts[1]), int(parts[2])), set()).add(
            name.rsplit(".", 1)[-1]
        )
    return sorted(
        d
        for d, suffixes in kinds.items()
        if {"input", "output"} <= suffixes and d not in used
    )


def epoch_year(day: tuple[int, int]) -> float:
    """The day as a decimal year, which is the drift regression's own abscissa."""
    return day[0] + (day[1] - 1) / 365.25


def imbalance(picked: list[tuple[int, int]]) -> float:
    """`|sum (e - e_bar)(doy - doy_bar)|` -- the balance equation this sample has to satisfy.

    Zero here is zero slope and zero `t` in the day-of-year-on-epoch regression, because all three
    share that numerator. The covariance is the objective rather than the `t` statistic so that the
    residual variance, which is not a property of the design, stays out of it.
    """
    epochs = [epoch_year(d) for d in picked]
    mean_epoch = sum(epochs) / len(epochs)
    mean_doy = sum(d[1] for d in picked) / len(picked)
    return abs(
        sum(
            (e - mean_epoch) * (d[1] - mean_doy)
            for e, d in zip(epochs, picked, strict=True)
        )
    )


def season_balanced(days: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """`HOLDOUT_PER_SITE` days spread through the archive whose season does not trend with epoch.

    An evenly spaced *index* is uniform in archive position, and archive position is uniform in
    season only if a site delivers at an even rate through the year. These do not -- cloud,
    maintenance and campaign schedules are seasonal, and the rate changes across the record as
    instruments are added -- so the index rule walks the sample through the calendar. #354 measured
    La Crau's doing it at 15.1 days per year of record, `t` 8.8, and a sample that drifts through
    the calendar is indistinguishable in the drift regression from a reference that is moving.

    The epoch spread the index rule gave is kept exactly: one day from each of `HOLDOUT_PER_SITE`
    equal-count strata through the sorted archive. Season is then balanced across those strata in
    two steps.

    **A seed that spreads season.** Stratum `k` takes the day at the `(SEASON_STRIDE * k mod n + .5)
    / n` quantile of its own day-of-year order. `SEASON_STRIDE` is coprime to `HOLDOUT_PER_SITE`, so
    the targets are a permutation of the whole year rather than a ramp: they correlate with `k` at
    `t = -0.35` where the index rule's own season target is a perfect ramp.

    **Then the balance equation, exactly.** Coordinate descent over the same candidates, accepting a
    move only when it strictly decreases [`imbalance`]. That is balanced sampling on an auxiliary
    variable, and the auxiliary variable is the *date* -- no property of the scene, its weather or
    its measured atmosphere can enter, which is the constraint a hold-out exists to keep. It
    terminates because a strictly decreasing non-negative objective over a finite candidate set
    cannot cycle, and the sweep cap is a guard against a floating-point tie rather than a bound
    anything reaches.

    Measured on the six archives: the achieved day-of-year slope is 0.00 d/yr at every site, against
    the seed's own -13.1 at the smallest, and the season spread the annual harmonic needs is
    untouched -- day-of-year standard deviation moves 103.0 -> 104.1 at the largest archive and
    71.0 -> 71.5 at the smallest.
    """
    total = len(days)
    strata = [
        days[total * k // HOLDOUT_PER_SITE : total * (k + 1) // HOLDOUT_PER_SITE]
        for k in range(HOLDOUT_PER_SITE)
    ]
    by_season = [sorted(block, key=lambda d: (d[1], d[0])) for block in strata]
    picked = [
        block[
            min(
                len(block) - 1,
                int(
                    (SEASON_STRIDE * k % HOLDOUT_PER_SITE + 0.5)
                    / HOLDOUT_PER_SITE
                    * len(block)
                ),
            )
        ]
        for k, block in enumerate(by_season)
    ]

    for _ in range(BALANCE_SWEEPS):
        moved = False
        for k, block in enumerate(by_season):
            best, score = picked[k], imbalance(picked)
            for candidate in block:
                trial = [*picked[:k], candidate, *picked[k + 1 :]]
                if imbalance(trial) < score:
                    best, score = candidate, imbalance(trial)
            if best != picked[k]:
                picked[k] = best
                moved = True
        if not moved:
            break
    else:
        raise SystemExit(
            f"season balance did not settle in {BALANCE_SWEEPS} sweeps over {total} days"
        )
    return sorted(picked)


def holdout(
    auth: tuple[str, str],
) -> tuple[list[str], int, int, int, list[tuple[int, int]], int]:
    """The hold-out fixture's rows, chosen by date through each archive rather than by inspection.

    [`season_balanced`] picks the days: `HOLDOUT_PER_SITE` epoch strata, one day each, chosen so the
    sample's season does not trend with epoch. A picked day can turn out to be entirely cloud-filled,
    and the repair is stated rather than exercised as judgement: walk forward through the sorted
    archive to the next day nobody has taken. Within a day the column is the one nearest that day's
    median solar zenith, which is `choose`'s own non-sweep rule -- a representative rather than an
    extreme, because an extreme picked out of a day is a choice and this fixture exists to have none.
    """
    rows: list[str] = []
    dropped = 0
    flagged = 0
    negative = 0
    used: list[tuple[int, int]] = []
    skipped = 0
    for site, instrument in dict.fromkeys((s, i) for s, i, *_ in SCENES):
        names = listing(site, auth)
        days = archive(names, site, instrument)
        if len(days) < HOLDOUT_PER_SITE:
            raise SystemExit(
                f"{site}: {len(days)} days outside SCENES, too few to hold {HOLDOUT_PER_SITE} out"
            )
        picked = dict.fromkeys(season_balanced(days))
        consumed: set[tuple[int, int]] = set()
        for start in picked:
            for index in range(days.index(start), len(days)):
                if days[index] in consumed:
                    continue
                consumed.add(days[index])
                year, doy = days[index]
                head, blocks, tables = day(site, instrument, year, doy, names, auth)
                columns, unreadable = clean_columns(blocks[0], tables[0], tables[2])
                negative += unreadable
                if not columns:
                    skipped += 1
                    continue
                zeniths = [float(blocks[0]["Zen"][c]) for c in columns]
                column = choose(columns, "holdout", zeniths)[0]
                more, lost, flags = comparison_rows(
                    head, year, doy, blocks, tables, column
                )
                rows += more
                dropped += lost
                flagged += flags
                used.append(days[index])
                break
    return rows, dropped, flagged, negative, used, skipped


def main() -> None:
    auth = credentials()
    rows: list[str] = []
    scenes: list[str] = []
    surface: dict[int, list[str]] = {w: [] for w in ABSORPTION_NM}
    top: dict[int, list[str]] = {w: [] for w in ABSORPTION_NM}
    # RadCalNet's own "predetermined absorption region" mask, collected per wavelength across every
    # scene so that its claimed uniformity is checked rather than assumed.
    band_mask: dict[int, set[bool]] = {w: set() for w in ABSORPTION_NM}
    dropped = 0
    flagged = 0
    negative = 0

    for site, instrument, year, doy, role in SCENES:
        names = listing(site, auth)
        head, blocks, tables = day(site, instrument, year, doy, names, auth)
        atmosphere = blocks[0]
        toa, toa_unc, boa = tables[0], tables[1], tables[2]

        columns, unreadable = clean_columns(atmosphere, toa, boa)
        negative += unreadable
        zeniths = [float(atmosphere["Zen"][c]) for c in columns]
        for column in choose(columns, role, zeniths):
            time = atmosphere["UTC"][column]
            pressure = float(atmosphere["P"][column])
            zenith = float(atmosphere["Zen"][column])
            air_mass = 1.0 / math.cos(math.radians(zenith)) + 1.0
            more, lost, flags = comparison_rows(head, year, doy, blocks, tables, column)
            rows += more
            dropped += lost
            flagged += flags

            scenes.append(
                f"scene {head['Site'].strip():8s} {year}-{doy:03d} {time:>6s} {role:>7s} "
                f"{air_mass:8.4f} {pressure:9.2f} "
                f"{float(atmosphere['AOD'][column]):8.4f} "
                f"{float(atmosphere['Ang'][column]):9.4f} "
                f"{float(atmosphere['WV'][column]):8.4f} "
                f"{float(atmosphere['O3'][column]):7.1f}"
            )
            for wavelength in ABSORPTION_NM:
                for table, into in ((boa, surface), (toa, top)):
                    cell = table[wavelength][column] if wavelength in table else MISSING
                    into[wavelength].append(
                        MISSING if cell in FILL else f"{float(cell):8.4f}"
                    )
                raw = toa_unc[wavelength][column] if wavelength in toa_unc else MISSING
                if raw not in FILL:
                    band_mask[wavelength].add(uncertainty(raw)[1].endswith("band"))

    # Section 3.5 calls these regions "predetermined", which says the mask is a property of the
    # wavelength and not of the retrieval. Checked here rather than trusted: a wavelength that is
    # flagged in one scene and not another would make the single column below a lie.
    split = sorted(w for w, seen in band_mask.items() if len(seen) > 1)
    if split:
        raise SystemExit(
            f"RadCalNet's absorption-region flag is not uniform across scenes at {split}, so it "
            "is not the predetermined spectral mask specification section 3.5 describes"
        )

    OUT.write_text("\n".join(comparison_header() + rows) + "\n")
    OUT_ABSORPTION.write_text(
        "\n".join(
            absorption_header()
            + scenes
            + [
                "#",
                "# band_nm radcalnet_absorption_region -- their flag, not ours; see the header",
            ]
            + [
                f"band {w:6d} {int(any(band_mask[w])):d}"
                for w in ABSORPTION_NM
                if band_mask[w]
            ]
            + [
                "#",
                "# boa_nm tau_rayleigh_1013 surface-reflectance-per-scene, in the order above",
            ]
            + [
                f"boa {w:6d} {bodhaine_1999(w / 1000.0):12.8f} " + " ".join(surface[w])
                for w in ABSORPTION_NM
            ]
            + [
                "#",
                "# toa_nm tau_rayleigh_1013 top-of-atmosphere-reflectance-per-scene, same order",
            ]
            + [
                f"toa {w:6d} {bodhaine_1999(w / 1000.0):12.8f} " + " ".join(top[w])
                for w in ABSORPTION_NM
            ]
        )
        + "\n"
    )

    instruments = {row.split()[0] for row in rows}
    print(
        f"{OUT}: {len(rows)} rows, {len(instruments)} instruments, "
        f"{len(scenes)} scenes, {dropped} dropped on fill values, "
        f"{negative} columns dropped on a negative published quantity"
    )
    print(f"  {flagged} rows carry a RadCalNet uncertainty flag")
    reachable = sum(
        1
        for w in ABSORPTION_NM
        if sum(
            up != MISSING and down != MISSING
            for up, down in zip(top[w], surface[w], strict=True)
        )
        >= 10
    )
    print(
        f"{OUT_ABSORPTION}: {len(ABSORPTION_NM)} wavelengths x {len(scenes)} scenes, "
        f"{reachable} wavelengths with at least 10 usable scenes"
    )

    held, held_dropped, held_flagged, held_negative, days_used, days_skipped = holdout(
        auth
    )
    overlap = sorted(
        {(row.split()[0], row.split()[1]) for row in held}
        & {(row.split()[0], row.split()[1]) for row in rows}
    )
    if overlap:
        raise SystemExit(
            f"the hold-out shares {overlap} with the fitted fixture, so it is not a hold-out"
        )
    OUT_HOLDOUT.write_text("\n".join(holdout_header(days_used) + held) + "\n")
    print(
        f"{OUT_HOLDOUT}: {len(held)} rows over {len(days_used)} days, "
        f"{held_dropped} dropped on fill values, {held_flagged} flagged, "
        f"{held_negative} columns dropped on a negative published quantity, "
        f"{days_skipped} chosen days passed over as fully cloud-filled"
    )


def policy() -> list[str]:
    """The RadCalNet data policy and the acknowledgement it asks for, verbatim."""
    return [
        "# Downloaded from the RadCalNet portal's JSON API, https://www.radcalnet.org/api/json/,",
        f"# on {DOWNLOADED}. The portal's data policy (RadCalNetDataPolicy_v1.1.pdf), in full:",
        "#",
        '#   "The RadCalNet data distributed through the RadCalNet portal are freely and publically',
        "#   available. Users of RadCalNet data must acknowledge the contribution of RadCalNet and",
        "#   if appropriate any specific sites in any presentation or publication. The citation for",
        '#   a publication should read something like: \'Calibration data for this "work" came from',
        "#   the CEOS WGCV RadCalNet service (https://www.radcalnet.org/) and in particular from",
        "#   site X and Y.' Users of RadCalNet data should note that data from RadCalNet are offered",
        "#   on a best efforts basis with responsibility for data quality and interpretation",
        "#   residing with the individual site owners, but which has undergone community based peer",
        "#   review of methods, uncertainty and equivalence. It is only valid for the conditions",
        "#   stated i.e. at nadir and times indicated. Individual site owners may be able to help",
        "#   users to interpret or extend the range of applicability (view angles etc) and should be",
        '#   contacted directly."',
        "#",
        "# Acknowledgement, in the form the policy asks for: calibration data for this work came",
        "# from the CEOS WGCV RadCalNet service (https://www.radcalnet.org/) and in particular from",
        "# sites RVUS (Railroad Valley Playa, United States), GONA (Gobabeb, Namibia), LCFR (La Crau,",
        "# France), BTCN and BSCN (Baotou, China) and GSCN (Golmud, China).",
        "#",
        "# Cite: Bouvet et al., 'RadCalNet: A Radiometric Calibration Network for Earth Observing",
        "# Imagers', Remote Sens. 11, 2401 (2019), doi:10.3390/rs11202401. Format per",
        "# QA4EO-WGCV-RadCalNet-R2 v1.0, 26 November 2019.",
    ]


def field_table() -> list[str]:
    """Every key the download publishes and what became of it, wrapped for a fixture header."""
    lines = [
        "# WHAT IS READ AND WHAT IS NOT. Between the header, the atmospheric block and its",
        f"# uncertainty twin the two files publish {len(CONSUMED) + len(WAIVED)} distinct keys:"
        f" {len(CONSUMED)} are read and {len(WAIVED)} are waived.",
        "# Each waiver states why the key cannot enter the answer rather than only that nothing",
        "# reads it -- the two are different claims and only the first is a defence. The generator",
        "# refuses to write this file if it meets a key in neither list, because an unclassified",
        "# key is a measurement nobody has decided to use or to waive, and a renamed one would",
        "# otherwise be read past in silence.",
        "#",
    ]
    for verdict, table in (("read", CONSUMED), ("waived", WAIVED)):
        for key, why in table.items():
            wrapped = textwrap.wrap(why, width=76)
            lines.append(f"#   {verdict:>6s} {key:<7s} {wrapped[0]}")
            lines += [f"#{'':>18s}{more}" for more in wrapped[1:]]
    return lines


def targets() -> list[str]:
    """What each target is, and what it does and does not represent."""
    return [
        "# TARGETS AND WHAT THEY REPRESENT. RVUS is a 1 km x 1 km area of Railroad Valley playa at",
        "# 1435 m; GONA a 30 m disk of Namib gravel plain at 510 m; LCFR a 30 m disk of La Crau",
        "# pebbles near sea level; BTCN the grey gravel square of the Baotou site, about 48 m across,",
        "# at 1270 m; BSCN the Baotou sand target, a second instrument 2 km away; GSCN the Golmud",
        "# site on the Tibetan plateau at 2824 m, where the surface pressure is a third lower than at",
        "# any other site and the Rayleigh depth with it. The 30 m targets are small enough that an",
        "# adjacency effect is a real part of their error budget. BTCN, BSCN and GSCN publish no TOA",
        "# reflectance beyond about 1 um, so their rows stop there and the SWIR is RVUS, GONA and",
        "# LCFR only.",
    ]


def comparison_header() -> list[str]:
    return [
        "# RadCalNet: an in-situ surface reflectance, the atmosphere measured over it, and the",
        "# top-of-atmosphere reflectance MODTRAN 5.3.3 produces from the two.",
        "# Generated by tools/gen_radcalnet_oracle.py -- do not edit.",
        "#",
        *policy(),
        "#",
        "# WHAT IS MEASURED AND WHAT IS NOT. boa, aod550, angstrom, pressure_hpa, water_cm and",
        "# ozone_du are measurements, and so are the five *_unc columns -- RadCalNet publishes an",
        "# uncertainty for every atmospheric input, which is what lets the Rust test charge its own",
        "# side of the comparison for input error instead of only the reference's side. toa is",
        "# MODTRAN 5.3.3 driven by them: DISORT at 8 streams (DIS: t, NSTR: 8), a Lambertian surface",
        "# (SURREF: LAMBER, NSURF: 1), rural aerosol (IHAZE: 2) with a Henyey-Greenstein phase",
        "# function of asymmetry 0.5 (IPH: 2, G: 0.500), reported through a 10 nm FWHM triangular",
        "# slit (V1: 400 V2: 2500 DV: 10 FWHM: 10 FLAGS: NTAA). It is not a satellite measurement",
        "# and must not be quoted as one.",
        "#",
        "# tau_rayleigh is Bodhaine et al. (1999) at 1013.25 hPa scaled by the measured surface",
        "# pressure, computed by the generator so that choosing a Rayleigh model is not a third",
        "# implementation whose error lands in the residual.",
        "#",
        "# VIEW GEOMETRY. Nadir only -- RadCalNet simulates no other, and the data policy says the",
        "# data are valid only at nadir. Nothing here speaks for an off-nadir view or for a BRDF.",
        "#",
        *targets(),
        "#",
        "# HOW THE SCENES WERE CHOSEN. Three days per instrument, after scanning about 22 days each",
        "# across the whole archive. One is a *sweep* day, where the atmosphere barely moves while",
        "# the sun does, and contributes five times so that geometry and aerosol errors separate;",
        "# the other two are that site's own extremes in aerosol load, water, ozone or Angstrom",
        "# exponent and contribute the single column nearest their own median zenith. A fixture drawn",
        "# from the middle of every distribution would make the residuals look better than they are.",
        "#",
        "# UNCERTAINTY FLAGS (specification section 3.5), decoded rather than carried raw: a",
        "# negative value means an input was outside the uncertainty look-up table's range ('lut'),",
        "# and a 9 added left of the decimal point marks a predetermined absorption region where the",
        "# uncertainty can exceed 10 % ('band'). The magnitudes below are decoded; the flag column",
        "# is 'toa/boa' when the two differ.",
        "#",
        "# AEROSOL TYPE. The last column is RadCalNet's own per-retrieval Type. It is what says",
        "# which aerosol model produced a given column, and the section of docs/verification.md",
        "# that calls IHAZE: 2 (rural) *quoted rather than chosen* rests on it being R. It is",
        "# carried so that a future regeneration cannot change it silently.",
        "#",
        *field_table(),
        "#",
        "# site date time_utc solar_zenith_deg wavelength_nm boa boa_unc toa toa_unc",
        "#   aod550 angstrom tau_rayleigh pressure_hpa water_cm ozone_du",
        "#   aod550_unc angstrom_unc ozone_du_unc pressure_hpa_unc water_cm_unc flag type",
    ]


def holdout_header(days: list[tuple[int, int]]) -> list[str]:
    return [
        "# RadCalNet again, on days nobody looked at: the hold-out for the one parameter",
        "# `radcalnet_toa.txt` had to fit.",
        "# Generated by tools/gen_radcalnet_oracle.py -- do not edit.",
        "#",
        "# WHY THIS FILE EXISTS. The aerosol single-scattering albedo is the one input RadCalNet",
        "# does not publish (NSSALB: 0 hands it to MODTRAN's internal rural model), so it is fitted",
        "# -- and it was fitted on the same comparisons every residual is then reported on. One free",
        "# parameter against 313 points is defensible and is said out loud, but it is not a",
        "# prediction. This file makes it one.",
        "#",
        "# HOW THE DAYS WERE CHOSEN, which matters more here than how many there are. Not by scanning:",
        f"# {HOLDOUT_PER_SITE} equal-count strata through each instrument's sorted archive, after removing the days",
        "# `radcalnet_toa.txt` already uses, and one day out of each.",
        "#",
        "# Which day is the part that had to change. Taking the stratum's first is an evenly spaced",
        "# *index*, and an index is uniform in archive position, not in season -- these sites do not",
        "# deliver at an even rate through the year, so the sample walks through the calendar. It was",
        "# measured doing it, at 15.1 days per year of record at La Crau, t 8.8, and a sample drifting",
        "# through the calendar is indistinguishable in the drift regression from a reference that is",
        "# moving. So the day is chosen to satisfy the balance equation sum (e - e_bar)(doy - doy_bar)",
        "# = 0 instead, seeded by a season target that is a permutation of the year rather than a ramp.",
        f"# Achieved over the chosen days: 0.00 days per year of record at every one of the {len({s for s, *_ in SCENES})}",
        "# sites, against 15.1 at La Crau before, with the season spread the annual harmonic needs",
        "# left intact.",
        "#",
        "# Nothing about a day's weather, aerosol load or water column enters the selection -- only its",
        "# date -- because that is exactly what the fitted fixture did and what a hold-out must not",
        "# repeat. A column whose published atmosphere carries a negative quantity -- an optical",
        "# depth, an angle, a pressure or a gas column cannot be one -- is dropped with the fill",
        "# values rather than guessed at, and the count is printed when this file is written.",
        "# Where a chosen day turns out to be entirely cloud-filled the generator walks forward",
        "# to the next untaken day in the same listing, which is a stated repair rather than a",
        "# judgement. Within a day the column is the one nearest that day's own median solar zenith.",
        "# The generator refuses to write this file if any (site, date) it holds also appears in the",
        "# fitted fixture.",
        "#",
        f"# Days held out ({len(days)}):",
        *(
            "#   " + " ".join(f"{y}-{d:03d}" for y, d in days[start : start + 10])
            for start in range(0, len(days), 10)
        ),
        "#",
        *policy(),
        "#",
        *targets(),
        "#",
        "# Columns are `radcalnet_toa.txt`'s exactly, written by the same code, so the two are the",
        "# same measurement of the same thing and the comparison between them is not a comparison of",
        "# two formatters.",
        "#",
        *field_table(),
        "#",
        "# site date time_utc solar_zenith_deg wavelength_nm boa boa_unc toa toa_unc",
        "#   aod550 angstrom tau_rayleigh pressure_hpa water_cm ozone_du",
        "#   aod550_unc angstrom_unc ozone_du_unc pressure_hpa_unc water_cm_unc flag type",
    ]


def absorption_header() -> list[str]:
    return [
        "# RadCalNet's surface and top-of-atmosphere reflectance across the whole published grid,",
        "# 400-2500 nm at 10 nm, on 42 scenes.",
        "# Generated by tools/gen_radcalnet_oracle.py -- do not edit.",
        "#",
        "# WHAT THIS IS FOR. `radiative_transfer::Atmosphere` is one scattering slab and carries no",
        "# gaseous absorption, so it may only be compared to MODTRAN where no gas absorbs. Deciding",
        "# where that is by reading a spectrum for notches is not good enough -- it put three",
        "# wavelengths carrying 2-5 % of water absorption into an earlier version of this fixture's",
        "# window list. This file lets the Rust test decide it from measurement instead, by",
        "# regressing -ln(rho_TOA/rho_BOA) on the two-way water column across the scenes at every",
        "# wavelength. No radiative-transfer model of ours enters that, so nothing it finds can be",
        "# an error of ours.",
        "#",
        "# Both reflectances are here rather than their ratio, because the ratio alone cannot say",
        "# where it is readable. That two-way-attenuation form neglects the path reflectance, whose",
        "# size relative to the signal is tau_scat / (4 mu_s rho_BOA) -- it needs rho_BOA, it runs",
        "# from 0.004 to 21 across this fixture, and the regression is only meaningful at the small",
        "# end. `radcalnet_measured_radiance.rs` bounds it explicitly and says what the bound costs.",
        "#",
        *policy(),
        "#",
        *targets(),
        "#",
        "# A reflectance is written 9999.0 exactly where RadCalNet published a fill value; nothing",
        "# else is filtered out here.",
        "#",
        "# tau_rayleigh_1013 is Bodhaine et al. (1999) at 1013.25 hPa; scale it by each scene's own",
        "# pressure. The scattering optical depth at a wavelength is that plus",
        "# aod550 * (lambda/550)^-angstrom, and mu_s is 1/(air_mass - 1) at nadir view.",
        "#",
        "# RADCALNET'S OWN OPINION, in the `band` rows. Specification section 3.5 marks",
        "# 'predetermined absorption regions' by adding 9 to the left of the decimal point of the",
        "# uncertainty. That mask is a property of the wavelength -- the generator verifies it is",
        "# identical in every scene rather than assuming it -- so it is one flag per wavelength,",
        "# emitted only where RadCalNet published an uncertainty at all.",
        "#",
        "# It is a *different criterion* from the regression above, not a second opinion on the",
        "# same one, and the two thresholds are worth keeping apart: theirs marks where their",
        "# retrieved uncertainty exceeds 10 %, which happens only in the near-opaque bands, while",
        "# ours marks where water absorption is measurable at all. So their mask is the narrower.",
        "# The test's business is that it is a *subset*: nothing RadCalNet calls an absorption",
        "# region may be a window of ours. A violation is a finding about one of the two methods.",
        "#",
        "# scene site date time_utc role air_mass pressure_hpa aod550 angstrom water_cm ozone_du",
    ]


if __name__ == "__main__":
    main()
