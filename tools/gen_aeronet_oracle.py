#!/usr/bin/env python3
"""Measured aerosol atmospheres from AERONET, as the input side of the earthvision correction.

`atmospheric_correction.rs` has always been evaluated as a *solver*: against DISORT
(`gen_disort_oracle.py`) and against libRadtran through pyspectral (`gen_pyspectral_oracle.py`),
on optical depths somebody chose -- 0.1, 1e-4, 0.7, 1.2 in the module's own fixtures, and a US
Standard profile in `docs/verification.md` Section 5.1. Nothing in that chain has ever been an
atmosphere anyone measured. This script is that half: AERONET publishes the aerosol optical depth,
the single-scattering albedo and the asymmetry parameter that `Atmosphere`'s three fields *are*,
retrieved together from the same sky-radiance inversion, so the triple is self-consistent rather
than three numbers assembled by hand.

**Provenance, since data that appeared by an unversioned HTTP GET is not an oracle.**

  * Source: the AERONET Version 3 web service, `print_web_data_inv_v3` (inversion products) at
    `aeronet.gsfc.nasa.gov`, NASA Goddard Space Flight Center. The exact query string for every
    request is written into the fixture header, so the fetch is reproducible from the file alone.
  * Level: **2.0 only** (`ALM20=1`), the quality-assured level -- automatically cloud-screened *and*
    carrying pre-field and post-field calibration. Level 1.5 is cloud-screened but not quality
    assured, and the difference between them is exactly the kind of thing an oracle must not absorb
    silently.
  * Algorithm: Version 3 almucantar inversion (Dubovik and King retrieval), products `AOD`
    (`AOD_Extinction-Total`), `SSA` (`Single_Scattering_Albedo`) and `ASY`
    (`Asymmetry_Factor-Total`), at the four inversion wavelengths 440, 675, 870 and 1020 nm.
  * Terms: AERONET data are public and free, and the AERONET Data Use Policy asks that the site PI
    be contacted and informed of the intended use before publication, and that the PI and the
    network be acknowledged. The `Contact: PI=...` line of each downloaded file is copied verbatim
    into the fixture header, which is where that obligation is recorded. PI email addresses are
    published by NASA in the file at the recorded URL and are deliberately **not** copied here --
    the attribution is the name, and republishing an address in a source repository is a separate
    decision nobody asked for.

**The domain this leg can cover, which is narrower than it first looks.** Version 3 promotes an
almucantar SSA to Level 2.0 only when the aerosol load is high enough for the sky radiance to
constrain it -- in practice `AOD(440) >= 0.4`. Below that the retrieval exists at Level 1.5 and no
higher. So a quality-assured (tau, omega, g) triple is available for *loaded* atmospheres and not
for clean ones, and every record here is one. That is a restriction of the measurement, not of the
model: it means this fixture spans the correction's hard end, and says nothing about a pristine
maritime column.

**What AERONET does not measure, stated before the numbers rather than after.** It does not measure
ozone. Version 3 takes the ozone column from an ancillary source and removes its absorption while
producing the aerosol optical depth, so what this file carries is an *aerosol* tau with the
molecular and gaseous terms already taken out. That matters here specifically: the pyspectral leg
measured a US Standard ozone column running **16.5 % high at 600 nm** (Chappuis band peak) when read
as a pure Rayleigh slab. Feeding the slab a measured aerosol tau and nothing else keeps ozone out of
*both* sides of the comparison -- ours and the reference solver's -- so it cancels rather than
hiding in the residual. `Ozone(Dobson)` and `Precipitable_Water(cm)` are carried in the fixture as
recorded context, from the coincident Version 3 direct-sun record: the slab has no gaseous
absorption at all, so neither is an input, and both are here so the domain of the residual is
documented and the ozone source is named rather than assumed.

**Why no Rayleigh term is added.** A physically complete column at 440 nm carries a molecular
optical depth comparable to a modest aerosol load, and adding it would be more realistic. It would
also put a *third* implementation -- a Rayleigh optical-depth model and a barometric reduction to
station pressure -- inside a residual that is supposed to measure our radiative transfer, which is
the trap `gen_pyspectral_oracle.py` names in its own header. The molecular half already has its own
oracle in that leg. This one is the aerosol half, and every number in it was measured.

**Site selection, and what each one is for.** Five sites spanning the aerosol types that behave
differently in a correction, each over a season when that type dominates:

  * `Mongu_Inn` (Zambia), Aug-Sep 2016 -- southern African savanna burning; the most absorbing
    aerosol in the set.
  * `Alta_Floresta` (Brazil), Sep 2007 -- Amazon biomass burning smoke.
  * `Solar_Village` (Saudi Arabia), Mar-Apr 2008 -- Arabian mineral dust.
  * `Banizoumbou` (Niger), Mar-Apr 2006 -- Sahelian dust off the Bodele depression.
  * `GSFC` (Maryland, USA), Jul 2002 -- North American industrial haze; the nearly conservative end.

Within each site the kept retrievals are `KEEP_PER_SITE` evenly spaced quantiles of `AOD(440)` among
those with all four wavelengths valid, ends included -- a deterministic rule that spans the site's
own range and does not let a hand-picked case in.

    uv run python tools/gen_aeronet_oracle.py

writes `crates/spacelab-earthvision/tests/fixtures/aeronet_atmosphere.txt`. Standard library only;
this reaches the network, so it is a generator and never part of a test run.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import urllib.parse
import urllib.request
from dataclasses import dataclass

INV_URL = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3"
SUN_URL = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
OUT = pathlib.Path(__file__).resolve().parents[1] / (
    "crates/spacelab-earthvision/tests/fixtures/aeronet_atmosphere.txt"
)

WAVELENGTHS_NM = (440, 675, 870, 1020)

# The header rows of a Version 3 web-service file, before the column line.
HEADER_ROWS = 5
COLUMN_ROW = 5
INV_COLUMN_ROW = 6  # the inversion files carry one extra title line

# AERONET writes an absent value as -999; anything at or below this is missing.
ABSENT = -900.0

# Retrievals kept per site. Six rather than three because the residual against DISORT turned out to
# be sampling noise (ticket #327), and the worst of a noise distribution is only as meaningful as the
# number of draws it is the worst of.
KEEP_PER_SITE = 6


@dataclass(frozen=True)
class Site:
    """One site over one season, and why it is in the set."""

    name: str
    start: dt.date
    end: dt.date
    regime: str


SITES = (
    Site("Mongu_Inn", dt.date(2016, 8, 1), dt.date(2016, 9, 30), "savanna-burning"),
    Site("Alta_Floresta", dt.date(2007, 9, 1), dt.date(2007, 9, 30), "amazon-smoke"),
    Site("Solar_Village", dt.date(2008, 3, 1), dt.date(2008, 4, 30), "arabian-dust"),
    Site("Banizoumbou", dt.date(2006, 3, 1), dt.date(2006, 4, 30), "sahelian-dust"),
    Site("GSFC", dt.date(2002, 7, 1), dt.date(2002, 7, 31), "industrial-haze"),
)


def query(site: Site, product: str | None) -> tuple[str, str]:
    """The full URL for one request, and the base it was built on."""
    fields: dict[str, object] = {
        "site": site.name,
        "year": site.start.year,
        "month": site.start.month,
        "day": site.start.day,
        "year2": site.end.year,
        "month2": site.end.month,
        "day2": site.end.day,
        "AVG": 10,  # every retrieval, not a daily average
        "if_no_html": 1,
    }
    if product is None:
        base, level = SUN_URL, {"AOD20": 1}
    else:
        base, level = INV_URL, {"ALM20": 1, "product": product}
    fields.update(level)
    return f"{base}?{urllib.parse.urlencode(fields)}", base


def fetch(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=300) as response:
        text = response.read().decode("utf-8", "replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise RuntimeError(f"no data returned by {url}")
    return lines


def table(lines: list[str], column_row: int) -> tuple[list[str], list[list[str]]]:
    return lines[column_row].split(","), [
        line.split(",") for line in lines[column_row + 1 :]
    ]


def contact(lines: list[str]) -> str:
    """The `Contact: PI=...` line, with the email addresses dropped."""
    for line in lines[: HEADER_ROWS + 2]:
        if "Contact: PI=" in line:
            names = line.split("Contact: PI=", 1)[1].split(";", 1)[0]
            return names.strip().replace("_", " ")
    return "unrecorded"


def value(row: dict[str, str], column: str) -> float:
    return float(row[column])


@dataclass(frozen=True)
class Retrieval:
    """One almucantar inversion, at the four wavelengths it retrieves."""

    site: str
    regime: str
    date: str
    time: str
    solar_zenith_deg: float
    aod: tuple[float, ...]
    ssa: tuple[float, ...]
    asymmetry: tuple[float, ...]
    water_cm: float
    ozone_du: float


def collect(site: Site) -> tuple[list[Retrieval], str, list[str]]:
    """Every complete Level 2.0 retrieval at `site`, with its PI contact and the URLs used."""
    urls: list[str] = []
    products: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    pi = "unrecorded"
    for product in ("AOD", "SSA", "ASY"):
        url, _ = query(site, product)
        urls.append(url)
        lines = fetch(url)
        pi = contact(lines)
        columns, rows = table(lines, INV_COLUMN_ROW)
        products[product] = {
            (row[1], row[2]): dict(zip(columns, row, strict=True)) for row in rows
        }

    url, _ = query(site, None)
    urls.append(url)
    sun_lines = fetch(url)
    sun_columns, sun_rows = table(sun_lines, COLUMN_ROW)
    sun = [dict(zip(sun_columns, row, strict=True)) for row in sun_rows]
    by_date: dict[str, list[dict[str, str]]] = {}
    for record in sun:
        by_date.setdefault(record["Date(dd:mm:yyyy)"], []).append(record)

    retrievals: list[Retrieval] = []
    for key, aod_row in products["AOD"].items():
        ssa_row = products["SSA"].get(key)
        asy_row = products["ASY"].get(key)
        if ssa_row is None or asy_row is None:
            continue
        aod = tuple(
            value(aod_row, f"AOD_Extinction-Total[{nm}nm]") for nm in WAVELENGTHS_NM
        )
        ssa = tuple(
            value(ssa_row, f"Single_Scattering_Albedo[{nm}nm]") for nm in WAVELENGTHS_NM
        )
        asy = tuple(
            value(asy_row, f"Asymmetry_Factor-Total[{nm}nm]") for nm in WAVELENGTHS_NM
        )
        if min(aod + ssa + asy) <= ABSENT:
            continue
        water, ozone = coincident(by_date.get(key[0], []), key[1])
        retrievals.append(
            Retrieval(
                site=site.name,
                regime=site.regime,
                date=key[0],
                time=key[1],
                solar_zenith_deg=value(
                    ssa_row, "Solar_Zenith_Angle_for_Measurement_Start(Degrees)"
                ),
                aod=aod,
                ssa=ssa,
                asymmetry=asy,
                water_cm=water,
                ozone_du=ozone,
            )
        )
    return retrievals, pi, urls


def seconds(clock: str) -> int:
    hour, minute, second = (int(part) for part in clock.split(":"))
    return 3600 * hour + 60 * minute + second


def coincident(day: list[dict[str, str]], clock: str) -> tuple[float, float]:
    """Precipitable water and ozone from the direct-sun record nearest in time on the same day.

    Both are context rather than input -- the slab has no gaseous absorption -- so the nearest
    record on the day is close enough, and AERONET's ozone is a daily ancillary column anyway.
    """
    if not day:
        return float("nan"), float("nan")
    target = seconds(clock)
    nearest = min(
        day, key=lambda record: abs(seconds(record["Time(hh:mm:ss)"]) - target)
    )
    return (
        value(nearest, "Precipitable_Water(cm)"),
        value(nearest, "Ozone(Dobson)"),
    )


def spanning(retrievals: list[Retrieval], keep: int = KEEP_PER_SITE) -> list[Retrieval]:
    """`keep` evenly spaced quantiles of `AOD(440)` -- the site's own range, no picking.

    The ends are always the site's minimum and maximum, so the span the fixture claims is the span
    the site had; the interior points divide the sorted order evenly. At `keep = 3` this is the
    minimum, the median and the maximum.
    """
    ordered = sorted(retrievals, key=lambda r: (r.aod[0], r.date, r.time))
    if len(ordered) <= keep:
        return ordered
    last = len(ordered) - 1
    picked = sorted({round(index * last / (keep - 1)) for index in range(keep)})
    return [ordered[index] for index in picked]


def main() -> None:
    stamp = dt.datetime.now(tz=dt.UTC).date().isoformat()
    header: list[str] = [
        "# Measured aerosol atmospheres from AERONET Version 3, Level 2.0 almucantar inversions.",
        "# Generated by tools/gen_aeronet_oracle.py -- do not edit.",
        f"# Downloaded {stamp} from aeronet.gsfc.nasa.gov (NASA Goddard Space Flight Center).",
        "#",
        "# AERONET Data Use Policy: contact the site PI about the intended use before publishing,",
        "# and acknowledge the PI and the network. The PI of each site is named below; the email",
        "# addresses are in the file header at the URL recorded with it.",
        "#",
        "# Level 2.0 only. Version 3 promotes an almucantar SSA to Level 2.0 only where the load",
        "# constrains it -- in practice AOD(440) >= 0.4 -- so every record here is a loaded column",
        "# and this fixture says nothing about a pristine one.",
        "#",
        "# aod is AERONET's AOD_Extinction-Total: an *aerosol* optical depth, with the molecular and",
        "# gaseous terms already removed by the Version 3 processing. No Rayleigh term is added here;",
        "# the molecular half has its own oracle in the pyspectral leg. water_cm and ozone_du come",
        "# from the coincident direct-sun record and are context, not input -- the slab carries no",
        "# gaseous absorption, so ozone is absent from both sides of any comparison built on this.",
        "#",
    ]
    rows: list[str] = []
    for site in SITES:
        retrievals, pi, urls = collect(site)
        kept = spanning(retrievals)
        header.append(
            f"# {site.name}: {site.regime}, {site.start.isoformat()}..{site.end.isoformat()}, "
            f"PI {pi}; {len(retrievals)} complete Level 2.0 retrievals, {len(kept)} kept."
        )
        for url in urls:
            header.append(f"#   {url}")
        for retrieval in kept:
            for index, nm in enumerate(WAVELENGTHS_NM):
                rows.append(
                    f"{retrieval.site:16s} {retrieval.regime:16s} {retrieval.date} "
                    f"{retrieval.time} {retrieval.solar_zenith_deg:10.6f} {nm:6d} "
                    f"{retrieval.aod[index]:9.6f} {retrieval.ssa[index]:9.6f} "
                    f"{retrieval.asymmetry[index]:9.6f} {retrieval.water_cm:8.4f} "
                    f"{retrieval.ozone_du:8.3f}"
                )

    header += [
        "#",
        "# site  regime  date(dd:mm:yyyy)  time_utc  solar_zenith_deg  wavelength_nm"
        "  aod  ssa  asymmetry  water_cm  ozone_du",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
