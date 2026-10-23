#!/usr/bin/env python
"""Regenerate the README figures, and the captions that quote their numbers, from the library.

Every figure checks spacelab against something that did not come from here, through the Python
bindings a user would call:

* SGP4/SDP4 against python-sgp4, Vallado's reference. Both descend from Vallado, so agreement pins
  the wrapper -- units, epochs, frames -- not the model; the `spacelab-sgp4` crate docs say why.
* The GCRF -> ITRF rotation of the IAU 2006/2000A CIO chain against ERFA, the SOFA lineage, over two
  centuries, as the angle of the residual rotation. Against ERFA's identical route (`xy06` series)
  and against its matrix route (`c2t06a`), whose gap is the one between the two formulations.
* The access-window engine against brute force: a fixed-step sampler converges on the engine's
  coverage and pass count as its step shrinks.
* The engine's coverage map, whose poleward edge must sit at the analytic reach of the sensor.

Nothing here is a timing, so nothing depends on the machine being quiet.

    just figures            # build the wheel, rewrite the PNGs and the README blocks
    just figures --check    # exit 1 if a quoted number has moved
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import erfa
import matplotlib as mpl
import numpy as np
import spacedatalab as sl
from sgp4.api import Satrec

mpl.use("Agg")
import matplotlib.pyplot as plt  # after `use`: the backend is chosen before pyplot loads

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "docs" / "figures"
# The PyPI page carries the same block, copied from this README by SpaceDataLab/spacedatalab.
READMES = (ROOT / "README.md",)
# Absolute, so the PyPI page, which resolves no relative link, shows them too.
URL = "https://raw.githubusercontent.com/SpaceDataLab/spacelab/main/docs/figures"
BEGIN = "<!-- BEGIN GENERATED FIGURES — tools/gen-figures.py -->"
END = "<!-- END GENERATED FIGURES -->"

RE = 6_378_137.0
MU = 3.986_004_418e14
WE = (
    7.292_115_146_706_979e-5  # rad/s, the rate the coverage engine rotates the Earth at
)
DAY_NS = 86_400 * 10**9
J2000_JD = 2_451_545.0
ARCSEC = math.pi / (180.0 * 3600.0)
ORBIT = (RE + 550e3, math.radians(53.0))  # the LEO the coverage figures fly
MASK_DEG = 5.0

ISS = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
GPS = (
    "1 24876U 97035A   08264.51782528 -.00000029  00000-0  00000+0 0  9990",
    "2 24876  55.6410  10.7530 0053050  55.1450 305.3620  2.00563190 79325",
)

BLUE, TEAL, RED = "#1f6feb", "#0e9f6e", "#d1495b"
plt.rcParams.update(
    {
        "figure.figsize": (6.4, 4.0),
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def sig(x: float) -> str:
    """Two significant figures, positional where that stays short."""
    if x == 0.0:
        return "0"
    if 1e-3 <= abs(x) < 1e4:
        return f"{float(f'{x:.2g}'):g}"
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa} × 10^{int(exponent)}"


def length(metres: float) -> str:
    """A length in the unit that keeps it between 1 and 1000."""
    for unit, scale in (
        ("km", 1e3),
        ("m", 1.0),
        ("mm", 1e-3),
        ("µm", 1e-6),
        ("nm", 1e-9),
    ):
        if metres >= scale:
            return f"{sig(metres / scale)} {unit}"
    return f"{sig(metres / 1e-12)} pm"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, metadata={"Software": None})
    plt.close(fig)


def sgp4_reference() -> str:
    fig, ax = plt.subplots()
    worst, identical, total = {}, 0, 0
    for label, (l1, l2), step, colour in (
        ("ISS, SGP4", ISS, 1.0, BLUE),
        ("GPS, SDP4", GPS, 5.0, TEAL),
    ):
        ref = Satrec.twoline2rv(l1, l2)
        minutes = np.arange(0.0, 1440.0 + step, step)
        errors = []
        for m in minutes:
            code, r_km, _ = ref.sgp4(ref.jdsatepoch, ref.jdsatepochF + m / 1440.0)
            if code != 0:
                raise RuntimeError(f"python-sgp4 error {code} at {m} min for {label}")
            r, _ = sl.sgp4_propagate(l1, l2, float(m))
            errors.append(math.dist(r, [c * 1e3 for c in r_km]))
        errors = np.array(errors)
        worst[label] = float(errors.max())
        identical += int((errors == 0.0).sum())
        total += errors.size
        shown = (
            errors > 0.0
        )  # a log axis has no zero; the bit-identical samples are counted instead
        ax.semilogy(
            minutes[shown] / 60.0, errors[shown], ".", ms=2.5, color=colour, label=label
        )
    ax.set(
        xlabel="hours since epoch",
        ylabel="|Δr| (m)",
        title="SGP4 against Vallado's reference (python-sgp4)",
    )
    ax.legend(loc="lower right", markerscale=4)
    save(fig, "sgp4-reference.png")
    return (
        f"SGP4 against python-sgp4 over 24 h: at most **{length(worst['ISS, SGP4'])}** (ISS) "
        f"and **{length(worst['GPS, SDP4'])}** (GPS); {round(100 * identical / total)} % of the "
        f"samples agree to the bit. Same lineage, so this pins units, epochs and frames."
    )


def two_part(ns: int) -> tuple[float, float]:
    """Nanoseconds since J2000 as a two-part Julian date, exact to the picosecond."""
    days, rest = divmod(ns, DAY_NS)
    return J2000_JD + days, rest / DAY_NS


def residual_angle(ours: np.ndarray, theirs: np.ndarray) -> float:
    """The angle of the small rotation `ours · theirsᵀ`, from its antisymmetric part."""
    d = ours @ theirs.T
    return 0.5 * math.hypot(d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1])


def frames_erfa() -> str:
    xp, yp = 0.1 * ARCSEC, 0.35 * ARCSEC
    years = np.arange(1900.0, 2100.0 + 1e-9, 1.0 / 12.0)
    same, matrix = [], []
    for year in years:
        ut1 = round((year - 2000.0) * 365.25 * DAY_NS)
        tt = (
            ut1 + 69_184_000_000
        )  # any fixed TT - UT1 serves: both sides receive the same pair
        ours = (
            np.array(
                [
                    sl.gcrf_to_itrf(
                        list(e * 1e7), [0.0] * 3, ut1, tt, xp, yp, 0.0, 0.0, 0.0
                    )[0]
                    for e in np.eye(3)
                ]
            ).T
            / 1e7
        )
        x, y = erfa.xy06(*two_part(tt))
        rc2i = erfa.c2ixys(x, y, erfa.s06(*two_part(tt), x, y))
        rpom = erfa.pom00(xp, yp, erfa.sp00(*two_part(tt)))
        series = erfa.c2tcio(rc2i, erfa.era00(*two_part(ut1)), rpom)
        same.append(residual_angle(ours, series))
        matrix.append(
            residual_angle(ours, erfa.c2t06a(*two_part(tt), *two_part(ut1), xp, yp))
        )
    same_uas = np.array(same) / ARCSEC * 1e6
    matrix_uas = np.array(matrix) / ARCSEC * 1e6
    fig, ax = plt.subplots()
    ax.semilogy(
        years, matrix_uas, color=RED, lw=0.7, label="ERFA matrix route (c2t06a)"
    )
    shown = same_uas > 0.0
    ax.semilogy(
        years[shown],
        same_uas[shown],
        color=BLUE,
        lw=0.7,
        label="ERFA identical route (xy06)",
    )
    ax.set(
        xlabel="year (TT)",
        ylabel="residual rotation (µas)",
        title="GCRF → ITRF, IAU 2006/2000A, against ERFA",
    )
    ax.legend(loc="lower right")
    save(fig, "frames-erfa.png")
    worst = float(same_uas.max())
    gap = float(matrix_uas.max())
    return (
        f"The CIO chain against ERFA's identical route, 1900–2100: at most **{sig(worst)} µas**, "
        f"{length(worst * 1e-6 * ARCSEC * RE)} on the ground. ERFA's matrix route differs by "
        f"up to {sig(gap)} µas: two formulations of one model, not an error."
    )


def naive_coverage(centres, zeniths, dt):
    """Fixed-step coverage per cell: the fraction of samples in view, and the passes counted."""
    a, inc = ORBIT
    t = np.arange(0.0, 86_400.0, dt)
    u = math.sqrt(MU / a**3) * t
    ex, ey, ez = (
        a * np.cos(u),
        a * np.sin(u) * math.cos(inc),
        a * np.sin(u) * math.sin(inc),
    )
    c, s = np.cos(WE * t), np.sin(WE * t)
    sat = np.stack([c * ex + s * ey, -s * ex + c * ey, ez], axis=1)
    fraction, passes = [], []
    for centre, zenith in zip(
        centres, zeniths, strict=True
    ):  # per cell: bounded memory at 1 s
        los = sat - centre
        covered = los @ zenith >= math.sin(math.radians(MASK_DEG)) * np.linalg.norm(
            los, axis=1
        )
        fraction.append(covered.mean())
        passes.append(int((covered[1:] & ~covered[:-1]).sum() + covered[0]))
    return np.array(fraction), np.array(passes)


def access_windows() -> str:
    cells = sl.grid_cells(
        [float(x) for x in range(-80, 81, 20)], [30.0 * j for j in range(12)]
    )
    centres = np.array([c for c, _ in cells])
    zeniths = np.array([z for _, z in cells])
    m = np.array(
        sl.coverage_metrics(
            *ORBIT, 0.0, 0.0, sl.Sensor.elevation(MASK_DEG), cells, 86_400.0, tol=1e-3
        )
    )
    covered, passes = m[:, 1].sum(), int(m[:, 0].sum())
    steps = np.array([600.0, 300.0, 120.0, 60.0, 30.0, 10.0, 5.0, 2.0, 1.0])
    error, missed = [], []
    for dt in steps:
        fraction, found = naive_coverage(centres, zeniths, dt)
        error.append(100.0 * abs(fraction.sum() - covered) / covered)
        missed.append(passes - int(found.sum()))
    fig, ax = plt.subplots()
    ax.loglog(steps, error, "o-", color=BLUE, label="coverage difference (%)")
    ax.set(
        xlabel="sampling step (s)",
        ylabel="|coverage difference| (%)",
        title="Brute-force sampling converges on the event engine",
    )
    ax.invert_xaxis()
    right = ax.twinx()
    right.semilogx(steps, missed, "s--", color=RED, label="passes missed")
    right.set_ylabel("passes missed")
    right.grid(False)
    right.spines["right"].set_visible(True)
    lines = ax.get_lines() + right.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
    save(fig, "access-windows.png")
    return (
        f"{len(cells)} cells, 24 h: at a 1 s step brute force finds **{passes - missed[-1]} of "
        f"{passes}** passes and agrees on coverage to **{sig(error[-1])} %**; at 600 s it misses "
        f"{missed[0]}. The engine needs no step: it brackets every edge and refines it to 1 ms."
    )


def coverage_map() -> str:
    lats, lons = np.arange(-88.75, 89.0, 2.5), np.arange(-180.0, 180.0, 2.5)
    cells = sl.grid_cells([float(x) for x in lats], [float(x) for x in lons])
    a, inc = ORBIT
    m = np.array(
        sl.coverage_metrics(
            a, inc, 0.0, 0.0, sl.Sensor.elevation(MASK_DEG), cells, 86_400.0, tol=1.0
        )
    )
    fraction = m[:, 1].reshape(len(lats), len(lons)) * 100.0
    # Spherical reach: the Earth-central angle a sensor at this mask sees from this radius.
    mask = math.radians(MASK_DEG)
    reach = math.degrees(inc + math.acos(RE * math.cos(mask) / a) - mask)
    edge = float(np.abs(lats[fraction.max(axis=1) > 0.0]).max())
    fig, ax = plt.subplots()
    image = ax.pcolormesh(lons, lats, fraction, cmap="viridis", shading="nearest")
    for sign in (1, -1):
        ax.axhline(sign * reach, color="white", ls="--", lw=1.0)
    fig.colorbar(image, ax=ax, label="time in view (%)")
    ax.set(
        xlabel="longitude (°)",
        ylabel="latitude (°)",
        title="One satellite, 550 km at 53°, 24 h, 5° mask",
    )
    ax.grid(False)
    save(fig, "coverage-map.png")
    return (
        f"The engine's coverage of {len(cells)} cells ends at **{edge:.2f}°** latitude, the last "
        f"cell inside the analytic reach of the 5° mask, **{reach:.1f}°** (dashed)."
    )


def block(captions: list[str]) -> str:
    alts = (
        "SGP4 against python-sgp4",
        "GCRF to ITRF against ERFA",
        "Brute-force sampling against the event engine",
        "Coverage map with the analytic reach",
    )
    names = ("sgp4-reference", "frames-erfa", "access-windows", "coverage-map")
    img = [f"![{alt}]({URL}/{name}.png)" for alt, name in zip(alts, names, strict=True)]
    return "\n".join(
        [
            BEGIN,
            "",
            f"| {img[0]} | {img[1]} |",
            "|:--:|:--:|",
            f"| {captions[0]} | {captions[1]} |",
            f"| {img[2]} | {img[3]} |",
            f"| {captions[2]} | {captions[3]} |",
            "",
            END,
        ]
    )


def main() -> int:
    check = sys.argv[1:] == ["--check"]
    if not check:
        FIGURES.mkdir(parents=True, exist_ok=True)
    captions = [
        f() for f in (sgp4_reference, frames_erfa, access_windows, coverage_map)
    ]
    new = block(captions)
    stale = []
    for readme in READMES:
        text = readme.read_text(encoding="utf-8")
        found = re.search(re.escape(BEGIN) + ".*?" + re.escape(END), text, flags=re.S)
        if found is None:
            raise SystemExit(f"{readme}: the generated block's markers are missing")
        if found.group(0) != new:
            stale.append(readme)
            if not check:
                readme.write_text(text.replace(found.group(0), new), encoding="utf-8")
    if check and stale:
        print("stale:", *(str(p.relative_to(ROOT)) for p in stale))
        return 1
    print("\n".join(captions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
