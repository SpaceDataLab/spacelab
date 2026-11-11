#!/usr/bin/env python3
"""SDTM-1's held-out residuals in geomagnetic coordinates, by storm phase: `plans/17` Ф23.O.

An auroral heating term (`Q_auroral`) would put energy in at high magnetic latitude during a storm's main phase.
Before any such term, this tool asks whether the residuals show the pattern it would remove. The residuals are
the family folds of `tools/evaluate-sdtm.py` (each row scored by the fit that never saw its mission family), at
the harness's 2 h rows. The coordinates are the centred dipole of `spacelab-geomagnetism/src/dipole.rs`, gated
in `derivations/geomagnetic_dipole.mac`, with its three degree-1 coefficients interpolated linearly between the
IGRF-14 epochs (IGRF's own convention) instead of extrapolated from 2025.

The test, fixed before the residuals are read: in each storm phase and each family, the mean residual at
`|MLat| >= 60` minus that at `|MLat| < 40`, with an interval resampling whole storms (Ф30.C). **Go** for
`Q_auroral` only if, in the main and recovery phase, that contrast is negative (the model too thin at high
latitude) and its interval excludes zero on at least two families. The same contrast in geographic latitude is
printed beside it, and the map by magnetic latitude and magnetic local time is descriptive only.

    tools/residual-maps.py [--field default] [--draws 4000]

Run under `uv run --no-project --with numpy --with pandas --with scipy python -I`.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
IGRF = ROOT / "crates" / "spacelab-gnss" / "tests" / "fixtures" / "igrf14coeffs.txt"
HIGH, LOW = 60.0, 40.0
PHASE_NAMES = ("pre-storm", "onset", "main and recovery", "post-storm")
MLT_SECTORS = (
    ("night", 21.0, 3.0),
    ("dawn", 3.0, 9.0),
    ("day", 9.0, 15.0),
    ("dusk", 15.0, 21.0),
)
MLAT_BANDS = ((0.0, 40.0), (40.0, 60.0), (60.0, 90.01))


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dipole_table(path: pathlib.Path = IGRF) -> tuple[np.ndarray, np.ndarray]:
    """IGRF-14's epochs and its `(g10, g11, h11)` at each, with the secular variation as the last row's slope."""
    epochs: list[float] = []
    rows: dict[tuple[str, str, str], list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = line.split()
        if cells[:3] == ["g/h", "n", "m"]:
            epochs = [float(c) for c in cells[3:-1]]
        elif cells[:3] in (["g", "1", "0"], ["g", "1", "1"], ["h", "1", "1"]):
            rows[(cells[0], cells[1], cells[2])] = [float(c) for c in cells[3:]]
    columns = [rows[("g", "1", "0")], rows[("g", "1", "1")], rows[("h", "1", "1")]]
    values = np.array([c[:-1] for c in columns]).T
    slope = np.array([c[-1] for c in columns])
    return np.array(epochs), np.vstack([values, values[-1] + 5.0 * slope])


def dipole_axis(year: np.ndarray, table=None) -> np.ndarray:
    """Unit vector toward the north geomagnetic pole, Earth-fixed, rows x 3: `dipole.rs`'s `axis`."""
    epochs, values = table if table is not None else dipole_table()
    knots = np.append(epochs, epochs[-1] + 5.0)
    g10, g11, h11 = (np.interp(year, knots, values[:, k]) for k in range(3))
    moment = np.sqrt(g10 * g10 + g11 * g11 + h11 * h11)
    return np.column_stack([-g11, -h11, -g10]) / moment[:, None]


def unit(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    c = np.cos(latitude)
    return np.column_stack(
        [c * np.cos(longitude), c * np.sin(longitude), np.sin(latitude)]
    )


def magnetic(axis: np.ndarray, latitude: np.ndarray, longitude: np.ndarray):
    """Magnetic latitude and longitude, radians: `dipole.rs`'s `magnetic_latitude` and `magnetic_longitude`."""
    r = unit(latitude, longitude)
    y = np.column_stack([-axis[:, 1], axis[:, 0], np.zeros(len(axis))])
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    x = np.cross(y, axis)
    mlat = np.arcsin(np.clip(np.sum(r * axis, axis=1), -1.0, 1.0))
    mlon = np.arctan2(np.sum(r * y, axis=1), np.sum(r * x, axis=1))
    return mlat, mlon


def magnetic_local_time(axis, latitude, longitude, lst_hours, declination):
    """Hours in `[0, 24)`: noon on the Sun's magnetic meridian. The sub-solar longitude is where the row's own
    local solar time would read 12 h, so the coordinate cannot disagree with the time the product is filed under."""
    sub_lon = longitude + (12.0 - lst_hours) * (np.pi / 12.0)
    _, mlon = magnetic(axis, latitude, longitude)
    _, mlon_sun = magnetic(axis, declination, sub_lon)
    hours = 12.0 + (mlon - mlon_sun) * (12.0 / np.pi)
    return hours - 24.0 * np.floor(hours / 24.0)


def declination(day_of_year: np.ndarray) -> np.ndarray:
    """The Sun's declination to about a degree (Cooper 1969); it moves the magnetic local time far less than a bin."""
    return np.radians(23.44) * np.sin(2.0 * np.pi * (284.0 + day_of_year) / 365.0)


def contrast(values, a, b, clusters, draws, rng) -> tuple[float, float, float]:
    """Mean over `a` minus mean over `b`, with an interval resampling the clusters that hold a row of either
    group, both groups on each draw. Rows outside both groups never enter a sum: a model's NaN on a row it
    could not answer would otherwise void every draw that picks its cluster."""
    inside = a | b
    values, a, b = values[inside], a[inside], b[inside]
    ids = np.unique(clusters[inside], return_inverse=True)[1]
    k = ids.max() + 1
    sums = np.column_stack(
        [
            np.bincount(ids, weights=w, minlength=k)
            for w in (values * a, a, values * b, b)
        ]
    )
    out = np.empty(draws)
    for i in range(draws):
        s = sums[rng.integers(0, k, k)].sum(axis=0)
        out[i] = s[0] / s[1] - s[2] / s[3] if s[1] > 0 and s[3] > 0 else np.nan
    low, high = np.nanpercentile(out, [2.5, 97.5])
    return values[a].mean() - values[b].mean(), float(low), float(high)


def storm_phases(rows, g, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Each time's phase (an index into `PHASE_NAMES`) and storm, -1 outside every storm, on the protocol's
    first-peak `t0`; and the storms. A time in two storms' phases goes to the first."""
    ap, epoch = rows.attrs["ap"], rows.attrs["ap_epoch"]
    start = (rows.t.min().normalize() - epoch).days * 8
    found = g.storm_onsets(ap[start:], epoch + np.timedelta64(start // 8, "D"))
    phase = np.full(t.size, -1)
    storm = np.full(t.size, -1)
    for n, s in enumerate(found):
        hours = (t - np.datetime64(s["t0"])) / np.timedelta64(1, "h")
        edges = (
            g.PHASES[0][0],
            g.PHASES[1][0],
            0.0,
            s["phase3_end"],
            s["phase3_end"] + 12.0,
        )
        for k in range(4):
            here = (phase < 0) & (hours >= edges[k]) & (hours < edges[k + 1])
            phase[here], storm[here] = k, n
    return phase, storm, found


def mean_interval(values, a, clusters, draws, rng) -> tuple[float, float, float]:
    """Mean over `a`, with an interval resampling the clusters that hold a row of `a`."""
    ids = np.unique(clusters[a], return_inverse=True)[1]
    k = ids.max() + 1
    total = np.bincount(ids, weights=values[a], minlength=k)
    count = np.bincount(ids, minlength=k)
    picks = rng.integers(0, k, (draws, k))
    low, high = np.percentile(
        total[picks].sum(axis=1) / count[picks].sum(axis=1), [2.5, 97.5]
    )
    return values[a].mean(), float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--field", default="default")
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    intervals = load("cluster_intervals", "cluster-intervals.py")
    fit, g = harness.fit, harness.g
    rows = fit.load_corpus(harness.STEP, phase=harness.PHASE)
    parts = [
        np.load(harness.OUT / args.field / f"{s['name']}.npz")
        for s in harness.read_splits()
        if s["kind"] == "family"
    ]
    test = np.concatenate([p["test"] for p in parts])
    series = {
        "SDTM-1": np.concatenate([p["residual"] for p in parts]),
        "NRLMSISE-00": np.concatenate([p["msis"] for p in parts]),
    }
    r = rows.iloc[test]
    done = np.isfinite(series["NRLMSISE-00"])
    t = r.t.to_numpy()
    family = r.family.to_numpy()
    year = r.t.dt.year.to_numpy() + (r.t.dt.day_of_year.to_numpy() - 0.5) / 365.25
    latitude, longitude = (
        np.radians(r.latitude.to_numpy()),
        np.radians(r.longitude.to_numpy()),
    )
    axis = dipole_axis(year)
    mlat = np.degrees(magnetic(axis, latitude, longitude)[0])
    mlt = magnetic_local_time(
        axis,
        latitude,
        longitude,
        r.lst.to_numpy(),
        declination(r.t.dt.day_of_year.to_numpy()),
    )
    phase, storm, found = storm_phases(rows, g, t)
    print(
        f"{args.field}, family folds: {done.sum()} rows, {len(found)} storms; "
        f"rows in a phase: "
        + ", ".join(f"{PHASE_NAMES[k]} {(done & (phase == k)).sum()}" for k in range(4))
    )
    rng = np.random.default_rng(intervals.SEED)
    print(
        f"\ncontrast: mean ln(model/measured) at |lat| >= {HIGH:g} minus at |lat| < {LOW:g}, x 100, storms resampled"
    )
    print("phase | family | rows high / low | model | magnetic | geographic")
    go = 0
    for k in range(4):
        for f in sorted(set(family)):
            here = done & (phase == k) & (family == f)
            if not here.any():
                continue
            high_m, low_m = here & (np.abs(mlat) >= HIGH), here & (np.abs(mlat) < LOW)
            geo = np.abs(r.latitude.to_numpy())
            high_g, low_g = here & (geo >= HIGH), here & (geo < LOW)
            if not (high_m.any() and low_m.any()):
                continue
            for name, values in series.items():
                m = contrast(100.0 * values, high_m, low_m, storm, args.draws, rng)
                gg = contrast(100.0 * values, high_g, low_g, storm, args.draws, rng)
                mark = " *" if m[2] < 0.0 or m[1] > 0.0 else ""
                if k == 2 and name == "SDTM-1" and m[2] < 0.0:
                    go += 1
                print(
                    f"{PHASE_NAMES[k]} | {f} | {high_m.sum()} / {low_m.sum()} | {name} | "
                    f"{m[0]:+.1f} [{m[1]:+.1f}, {m[2]:+.1f}]{mark} | {gg[0]:+.1f} [{gg[1]:+.1f}, {gg[2]:+.1f}]"
                )
    verdict = "GO" if go >= 2 else "NO-GO"
    print(
        f"\nmain and recovery: SDTM-1 too thin at high magnetic latitude, interval below zero, on {go} families: {verdict}"
    )
    print(
        "\nmap, all families, SDTM-1: mean ln(model/measured) x 100, interval resampling storms, (rows)"
    )
    print("phase | |MLat| | " + " | ".join(name for name, _, _ in MLT_SECTORS))
    for k in range(4):
        for lo, hi in MLAT_BANDS:
            band = done & (phase == k) & (np.abs(mlat) >= lo) & (np.abs(mlat) < hi)
            cells = []
            for _, a, b in MLT_SECTORS:
                sector = (mlt >= a) & (mlt < b) if a < b else (mlt >= a) | (mlt < b)
                cell = band & sector
                if cell.any():
                    m = mean_interval(
                        100.0 * series["SDTM-1"], cell, storm, args.draws, rng
                    )
                    cells.append(
                        f"{m[0]:+.1f} [{m[1]:+.1f}, {m[2]:+.1f}] ({cell.sum()})"
                    )
                else:
                    cells.append("—")
            print(
                f"{PHASE_NAMES[k]} | [{lo:g}, {min(hi, 90):g}) | " + " | ".join(cells)
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
