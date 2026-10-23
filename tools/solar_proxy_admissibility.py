#!/usr/bin/env python3
"""Which solar drivers SDTM-1 may be fitted on, and what each one is worth over F10.7.

`plans/16` Ф15.B admits a driver only if three things hold at once: **commercial use is permitted**,
the series **spans the corpus** (2000-2025, CHAMP through GRACE-FO) or is bridged by a documented
regression, and the **operational latency is at most a day**. None of the three is a matter of
opinion, so none of them is asserted here -- this script measures all three and prints the table that
goes into `docs/verification.md`.

## The terms layer is machine-read, not interpreted

Every candidate's licence comes from the provider's own metadata rather than from a web page someone
read once:

  * the GOES-R EUVS daily files carry a `license` global attribute -- for the science-quality
    products it reads *"These data may be redistributed and used without restriction."*, which is the
    whole of NOAA's grant and needs no interpretation;
  * LISIRD publishes a machine-readable catalogue (`https://lasp.colorado.edu/lemr/lisird/lisird`) in
    which each dataset either names a licence (`lds:license lds:cc0`, `lds:cc_by_4.0`, ...) or does
    not. **Not naming one is the finding**, and it is what separates the two Mg II series and the
    Lyman-alpha composite from the GOES products that carry CC0. The server content-negotiates and
    answers a bare request with N-Triples rather than the Turtle a browser gets, so the fetch asks for
    N-Triples explicitly and the parser reads exactly that one grammar.

A dataset with no stated licence is not admitted. It can still be *read* -- the standing MT_CKD and
HITRAN rule (`CONTRIBUTING.md`, "Oracles under non-commercial terms"): the output is a dated number
about our own model and nothing sold consumes the data. Nothing fetched here is written into the
repository; the cache is `~/.cache/spacelab/space-weather`.

## What "worth over F10.7" means, and why it is the number that decides

A proxy that F10.7 predicts adds nothing but a citation. So each EUV series is regressed on exactly
the form SDTM-1's exospheric temperature uses -- `[1, f, f^2, d]` with `f = (F10.7a - 150)/100` and
`d = (F10.7 - F10.7a)/100` -- and what is reported is the **residual**: the part of the EUV variation
F10.7 cannot reach. That single number reads two ways, which is why it is the only regression here:

  * forwards, it is the **bridge error** -- the accuracy with which the proxy can be reconstructed
    from F10.7 over the years before the instrument existed;
  * backwards, it is the **information the proxy carries** beyond the driver already admitted.

Both are converted into physical units so they can be compared with anything: into sfu by the fitted
local sensitivity `dp/dF10.7a`, and into kelvin of exospheric temperature by Jacchia's (1971)
night-time relation `T_inf = 379 + 3.24 F10.7a + 1.3 (F10.7 - F10.7a)`, i.e. **3.24 K per sfu**. That
conversion is a published slope of the same family as SDTM-1's own linear term, not a fit of ours, and
it is used only to put a residual on an axis a reader can judge.

The residual is compared against the **inter-satellite difference** on days two GOES spacecraft both
measured. A residual below that floor is instrument scatter dressed up as information; above it, the
proxy is measuring something F10.7 is not.

## Run

    uv run --no-project --with netCDF4 --with numpy python tools/solar_proxy_admissibility.py --fetch

`--fetch` downloads what is missing into the cache (about 25 MB, most of it SDO/EVE); without it the
script reads the cache and fails with the URL of whatever is absent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys
import urllib.request

import numpy as np

CACHE = pathlib.Path.home() / ".cache" / "spacelab" / "space-weather"

GOES_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
EUVS = {
    "g16": f"{GOES_BASE}/goes16/l2/data/euvs-l2-avg1d_science/sci_euvs-l2-avg1d_g16_s20170207_e20250406_v1-0-7.nc",
    "g18": f"{GOES_BASE}/goes18/l2/data/euvs-l2-avg1d_science/sci_euvs-l2-avg1d_g18_s20220909_e20260910_v1-0-7.nc",
    "g19": f"{GOES_BASE}/goes19/l2/data/euvs-l2-avg1d_science/sci_euvs-l2-avg1d_g19_s20240920_e20260910_v1-0-7.nc",
}
OTHER = {
    "SW-All.csv": "https://celestrak.org/SpaceData/SW-All.csv",
    "lisird_lemr.nt": "https://lasp.colorado.edu/lemr/lisird/lisird",
    "MgII_composite.dat": "https://www.iup.uni-bremen.de/gome/solar/MgII_composite.dat",
    "composite_lyman_alpha.csv": "https://lasp.colorado.edu/lisird/latis/dap/composite_lyman_alpha.csv",
    "latest_EVE_L3_merged_1nm.ncdf": "https://lasp.colorado.edu/eve/data_access/eve_data/products/merged/latest_EVE_L3_merged_1nm.ncdf",
}

# LISIRD identifiers for every candidate `plans/16` Ф15.B lists, plus the F10.7 sources for contrast.
CANDIDATES = [
    "noaa_goes16_euvs_1d",
    "noaa_goes18_euvs_1d",
    "noaa_goes19_euvs_1d",
    "noaa_goes16_euvs_mgii_1d",
    "noaa_goes18_euvs_mgii_1d",
    "sdo_eve_lines_l3",
    "sdo_eve_ssi_1nm_l3",
    "soho_sem_P1D",
    "composite_lyman_alpha",
    "bremen_composite_mgii",
    "composite_mg_index",
    "sorce_mg_index_P1D",
    "penticton_radio_flux",
    "noaa_radio_flux",
    "lasp_filtered_drao_f107",
    "cls_radio_flux_f30",
]

# Jacchia (1971), night-time global exospheric temperature: T_inf = 379 + 3.24 F10.7a + 1.3 dF10.7.
KELVIN_PER_SFU = 3.24
F107_REFERENCE = 150.0
F107_SCALE = 100.0

# The three GOES-R EUVS daily columns that stand in for JB2008's S10, Y10 and M10.
EUVS_PROXIES = ("irr_304", "irr_1216", "MgII_standard")


ACCEPT = {"lisird_lemr.nt": "application/n-triples"}


def fetch(name: str, url: str, allow: bool) -> pathlib.Path:
    path = CACHE / name
    if path.exists():
        return path
    if not allow:
        sys.exit(f"{path} is absent; re-run with --fetch to download it from {url}")
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"fetching {url}", file=sys.stderr)
    headers = {"Accept": ACCEPT[name]} if name in ACCEPT else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        path.write_bytes(response.read())
    return path


NONE_STATED = "-- none stated --"
TRIPLE = re.compile(
    r'^<([^>]+)>\s+<([^>]+)>\s+(?:<([^>]+)>|"((?:[^"\\]|\\.)*)")\s*\.\s*$'
)


def lisird_licences(path: pathlib.Path) -> dict[str, str]:
    """Per-dataset licence out of LISIRD's catalogue. Absence is a value, not a gap."""
    datasets: set[str] = set()
    identifier: dict[str, str] = {}
    licence_of: dict[str, str] = {}
    licence_name: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = TRIPLE.match(line)
        if not match:
            continue
        subject, predicate, resource, literal = match.groups()
        local = predicate.rsplit("#", 1)[-1]
        if local == "type" and resource and resource.endswith("#Dataset"):
            datasets.add(subject)
        elif local == "identifier" and literal is not None:
            identifier[subject] = literal
        elif local == "license" and resource:
            licence_of[subject] = resource
        elif local == "name" and literal is not None:
            licence_name[subject] = literal

    def named(subject: str) -> str:
        if subject not in licence_of:
            return NONE_STATED
        licence = licence_of[subject]
        return licence_name.get(licence, licence.rsplit("#", 1)[-1])

    return {identifier.get(s, s.rsplit("#", 1)[-1]): named(s) for s in datasets}


def read_euvs(path: pathlib.Path) -> tuple[str, dict[str, dict[dt.date, float]]]:
    """GOES-R EUVS daily science file: its licence, and one `date -> value` map per proxy."""
    import netCDF4  # imported here so --help works without the optional dependency

    data = netCDF4.Dataset(path)
    epoch = dt.datetime(2000, 1, 1, 12, tzinfo=dt.UTC)
    seconds = np.asarray(data.variables["time"][:], dtype=float)
    day = [(epoch + dt.timedelta(seconds=s)).date() for s in seconds]
    out = {}
    for name in EUVS_PROXIES:
        value = np.asarray(data.variables[name][:], dtype=float)
        flag = np.asarray(
            data.variables[("MgII" if name.startswith("MgII") else name) + "_flag"][:]
        )
        out[name] = {
            d: v
            for d, v, ok in zip(day, value, flag == 0, strict=True)
            if ok and v > -9000.0
        }
    return str(data.getncattr("license")).strip(), out


def read_eve(path: pathlib.Path) -> tuple[str, dict[str, dict[dt.date, float]]]:
    """SDO/EVE L3 merged: the ESP diodes that survived the 2014 MEGS-A loss, and the line that did not.

    The `He II 30.4` *line* needs MEGS-A and stops on 2014-05-26; ESP channel 9 is a 27.16-33.8 nm
    band containing the same line and runs to the present, as does the MEGS-P Lyman-alpha diode. That
    distinction is the whole reason to read this file rather than trust the mission dates.
    """
    import netCDF4

    data = netCDF4.Dataset(path)
    stamp = np.asarray(data.variables["yyyydoy"][:], dtype=float)
    day = [
        dt.date(int(v) // 1000, 1, 1) + dt.timedelta(days=int(v) % 1000 - 1)
        for v in stamp
    ]
    diode = {
        str(n).strip(): i for i, n in enumerate(data.variables["diodemeta_name"][:])
    }
    line = {
        str(n).strip(): i for i, n in enumerate(data.variables["linesmeta_name"][:])
    }
    diodes = np.asarray(data.variables["diode_irradiance"][:], dtype=float)
    lines = np.asarray(data.variables["line_irradiance"][:], dtype=float)
    picked = {
        "esp_27_34": diodes[diode["Channel 9 (27.16-33.8nm)"]],
        "megsp_1216": diodes[diode["Lyman-alpha (121-122nm)"]],
        "megsa_304_line": lines[line["He II"]],
    }
    out = {
        name: {
            d: v for d, v in zip(day, value, strict=True) if np.isfinite(v) and v > 0.0
        }
        for name, value in picked.items()
    }
    return str(data.getncattr("license")).strip(), out


def read_celestrak(path: pathlib.Path) -> dict[dt.date, tuple[float, float]]:
    """`date -> (F10.7 observed, F10.7 81-day centred mean)`, observations only."""
    rows = {}
    for line in path.read_text(errors="replace").splitlines()[1:]:
        field = line.split(",")
        if len(field) < 28 or field[26] != "OBS":
            continue
        try:
            day = dt.date.fromisoformat(field[0])
            rows[day] = (float(field[24]), float(field[27]))
        except ValueError:
            continue
    return rows


def regress(proxy: np.ndarray, f: np.ndarray, d: np.ndarray) -> dict[str, float]:
    """Least squares of the proxy on SDTM-1's own solar driver form, and the residual it leaves."""
    design = np.column_stack([np.ones_like(f), f, f * f, d])
    coefficients, *_ = np.linalg.lstsq(design, proxy, rcond=None)
    residual = proxy - design @ coefficients
    rms = float(np.sqrt(np.mean(residual**2)))
    variance = float(np.var(proxy))
    # Local sensitivity at the sample mean, in proxy units per sfu: d/dF10.7a of a1 f + a2 f^2.
    mean_f = float(np.mean(f))
    slope = abs(coefficients[1] + 2.0 * coefficients[2] * mean_f) / F107_SCALE
    return {
        "n": float(proxy.size),
        "r2": 1.0 - rms**2 / variance if variance > 0 else float("nan"),
        "rms": rms,
        "rms_relative": rms / float(np.mean(proxy)),
        "sfu": rms / slope if slope > 0 else float("nan"),
        "kelvin": KELVIN_PER_SFU * rms / slope if slope > 0 else float("nan"),
    }


# The shipped SDTM-1 profile, re-implemented here so the kelvin-to-density conversion below does not
# depend on the Rust: `crates/spacelab-propagation/src/sdtm/mod.rs`, the closed form
# `derivations/sdtm_bates.mac` proves. At T_inf = 1058 K this gives 3.5114e-12 kg/m^3 at 400 km, which
# is the absolute anchor that module's own test asserts.
BOUNDARY_ALTITUDE = 120_000.0
BOUNDARY_TEMPERATURE = 380.0
REFERENCE_RADIUS = 6_356_766.0
BOUNDARY_GRAVITY = (
    9.80665 * (REFERENCE_RADIUS / (REFERENCE_RADIUS + BOUNDARY_ALTITUDE)) ** 2
)
BOLTZMANN = 1.380649e-23
ATOMIC_MASS = 1.66053906892e-27
BOUNDARY_SPECIES = (  # N2, O2, O, Ar, He: Jacchia (1971) 120 km values, helium's alpha = -0.38
    (28.0134, 4.0e17, 0.0),
    (31.9988, 7.5e16, 0.0),
    (15.9994, 7.6e16, 0.0),
    (39.948, 1.8e15, 0.0),
    (4.0026, 3.4e13, -0.38),
)
SIGMA = 2.0e-5


def sdtm_density(t_inf: float, altitude: float) -> float:
    geopotential = (
        (altitude - BOUNDARY_ALTITUDE)
        * (REFERENCE_RADIUS + BOUNDARY_ALTITUDE)
        / (REFERENCE_RADIUS + altitude)
    )
    temperature = t_inf - (t_inf - BOUNDARY_TEMPERATURE) * np.exp(-SIGMA * geopotential)
    integral = (SIGMA * geopotential + np.log(temperature / BOUNDARY_TEMPERATURE)) / (
        SIGMA * t_inf
    )
    return float(
        sum(
            mass
            * ATOMIC_MASS
            * number
            * (BOUNDARY_TEMPERATURE / temperature) ** (1.0 + alpha)
            * np.exp(-mass * ATOMIC_MASS * BOUNDARY_GRAVITY * integral / BOLTZMANN)
            for mass, number, alpha in BOUNDARY_SPECIES
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true", help="download missing files into the cache"
    )
    argument = parser.parse_args()

    paths = {name: fetch(name, url, argument.fetch) for name, url in OTHER.items()}
    stated: dict[str, str] = {}
    series: dict[str, dict[dt.date, float]] = {}
    for satellite, url in EUVS.items():
        licence, proxies = read_euvs(
            fetch(pathlib.PurePosixPath(url).name, url, argument.fetch)
        )
        stated[f"GOES EUVS {satellite}"] = licence
        series.update({f"{satellite} {name}": value for name, value in proxies.items()})
    licence, proxies = read_eve(paths["latest_EVE_L3_merged_1nm.ncdf"])
    stated["SDO/EVE L3 merged"] = licence
    series.update({f"eve {name}": value for name, value in proxies.items()})

    flux = read_celestrak(paths["SW-All.csv"])
    licences = lisird_licences(paths["lisird_lemr.nt"])
    today = dt.datetime.now(dt.UTC).date()

    print("## Terms, as each provider states them\n")
    print("`license` global attribute of the data file itself:")
    for source, licence in stated.items():
        print(f"  {source:<20} {licence!r}")
    print("\nLISIRD catalogue, `lds:license` per dataset:")
    width = max(len(name) for name in CANDIDATES)
    for name in CANDIDATES:
        print(f"  {name:<{width}}  {licences.get(name, '-- not in the catalogue --')}")

    print("\n## Coverage and latency\n")
    fmt = "  {:<22} {:>10} {:>12} {:>7} {:>9}"
    print(fmt.format("series", "first", "last", "days", "latency"))
    for name, value in series.items():
        day = sorted(value)
        print(
            fmt.format(
                name,
                str(day[0]),
                str(day[-1]),
                str(len(day)),
                f"{(today - day[-1]).days} d",
            )
        )
    observed = sorted(flux)
    print(
        fmt.format(
            "CelesTrak F10.7 OBS",
            str(observed[0]),
            str(observed[-1]),
            str(len(observed)),
            f"{(today - observed[-1]).days} d",
        )
    )
    for name in ("MgII_composite.dat", "composite_lyman_alpha.csv"):
        rows = [
            line
            for line in paths[name].read_text(errors="replace").splitlines()
            if line.strip() and not line.startswith(";") and not line[0].isalpha()
        ]
        if name.startswith("MgII"):
            span = f"{rows[0].split()[0]} .. {rows[-1].split()[0]} (decimal year)"
        else:
            epoch = dt.date(1947, 1, 1)
            span = " .. ".join(
                str(epoch + dt.timedelta(days=float(row.split(",")[0])))
                for row in (rows[0], rows[-1])
            )
        print(
            f"  {name:<22} {span}  {len(rows)} rows  (no stated licence: oracle only)"
        )

    print("\n## What each EUV proxy is worth over F10.7\n")
    print(
        "Regression on [1, f, f^2, d]; the residual is both the bridge error before the instrument"
    )
    print(
        f"existed and the information the proxy adds. Kelvin at {KELVIN_PER_SFU} K/sfu (Jacchia 1971).\n"
    )
    head = "  {:<22} {:>6} {:>7} {:>11} {:>9} {:>7} {:>7}"
    print(head.format("proxy", "days", "R^2", "rms", "rms/mean", "sfu", "K"))
    for name, value in series.items():
        day = [d for d in sorted(value) if d in flux]
        if len(day) < 100:
            continue
        proxy = np.array([value[d] for d in day])
        pair = np.array([flux[d] for d in day])
        out = regress(
            proxy,
            (pair[:, 1] - F107_REFERENCE) / F107_SCALE,
            (pair[:, 0] - pair[:, 1]) / F107_SCALE,
        )
        print(
            head.format(
                name,
                f"{out['n']:.0f}",
                f"{out['r2']:.4f}",
                f"{out['rms']:.4e}",
                f"{out['rms_relative'] * 100:.2f}%",
                f"{out['sfu']:.2f}",
                f"{out['kelvin']:.1f}",
            )
        )

    print(
        "\n## The instrument floor: the same quantity from two spacecraft on the same day\n"
    )
    pairs = [
        (f"{left} {name}", f"{right} {name}")
        for left, right in (("g16", "g18"), ("g18", "g19"))
        for name in EUVS_PROXIES
    ]
    pairs += [("eve esp_27_34", "g16 irr_304"), ("eve megsp_1216", "g16 irr_1216")]
    for left, right in pairs:
        a, b = series.get(left, {}), series.get(right, {})
        common = sorted(set(a) & set(b))
        if len(common) < 100:
            continue
        difference = np.array([a[d] - b[d] for d in common])
        rms = float(np.sqrt(np.mean(difference**2)))
        mean = float(np.mean([a[d] for d in common]))
        scale = (
            " (different bands, so the spread is calibration plus bandwidth)"
            if left.startswith("eve")
            else ""
        )
        print(
            f"  {left:<15} - {right:<14} n {len(common):>5}  bias {np.mean(difference):+.3e}"
            f"  rms {rms:.3e}  ({rms / mean * 100:.2f}% of the mean){scale}"
        )

    print("\n## And what those kelvin are worth as density\n")
    print(
        "The SDTM-1 profile at T_inf = 1058 K (F10.7a 141, the module's own reference field), moved"
    )
    print(
        "by the residuals above. This is an upper bound on what admitting an EUV driver could explain,"
    )
    print(
        "not a prediction that it will: it assumes the whole residual maps to T_inf at Jacchia's slope.\n"
    )
    reference = 1058.0
    for altitude in (300_000.0, 400_000.0, 550_000.0):
        base = sdtm_density(reference, altitude)
        moved = "   ".join(
            f"{change:5.1f} K -> {100.0 * (sdtm_density(reference + change, altitude) / base - 1.0):+6.2f}%"
            for change in (24.6, 34.8, 52.0)
        )
        print(f"  {altitude / 1000.0:5.0f} km  rho {base:.4e} kg/m^3   {moved}")


if __name__ == "__main__":
    main()
