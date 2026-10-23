#!/usr/bin/env -S uv run --quiet --with numpy --with scikit-learn python
"""Phase-6 ML gate: is coverage-vector clustering anything more than k-means on orbital elements?

The plan's null hypothesis (plans/03-roadmap.md §Phase 6, edit #4): a coverage vector may be a
deterministic function of the orbital elements (a, e, i, RAAN/LTAN). If so, clustering coverage adds
*nothing* over clustering elements, and the "coverage-pattern" ML layer has no novelty. The only source
of information beyond the elements is the **sensor model** (FOV / off-nadir, sun geometry, day/night,
local time). This experiment tests both claims cheaply, before any Rust is written.

Two parts:
  A. Regime recovery. Four well-separated regimes (Starlink-LEO, GNSS-MEO, SSO-EO, GEO-comms). Cluster
     on elements and on purely-geometric coverage; measure agreement with the ground-truth regime and
     with each other. High ARI(coverage, elements) => geometry alone carries no new information.
  B. Sensor information. One LEO family (elements nearly identical), each satellite given one of two
     sensors: wide-FOV all-time (radar/comms) vs narrow-FOV day-only (optical). Elements cannot see the
     sensor; sensor-aware coverage can. If ARI(coverage+sensor, sensor-label) >> ARI(elements,
     sensor-label), the sensor model is a real, separable information source — the layer is alive.

Fidelity is deliberately light (circular orbits, J2 nodal precession, latitude-band coverage with an
elevation-mask footprint, approximate day/night). It is enough to settle the *structural* question,
not to quote operational numbers.

Run:  timeout 300 uv run --no-project --with numpy --with scikit-learn python experiments/null_hypothesis_coverage.py
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)

MU = 398_600.4418  # km³/s²
RE = 6378.137  # km
J2 = 1.082_63e-3
WE = 7.292_115_9e-5  # rad/s, Earth rotation
OBLIQ = np.radians(23.44)

BANDS = np.radians(np.arange(-85.0, 86.0, 10.0))  # 18 latitude-band centres


def subpoints(a, inc, raan, phase, days=2.0, dt=60.0):
    """Sub-satellite latitude/longitude and a day/night flag over the span (circular orbit + J2)."""
    n = np.sqrt(MU / a**3)
    raan_dot = -1.5 * n * J2 * (RE / a) ** 2 * np.cos(inc)
    t = np.arange(0.0, days * 86400.0, dt)
    u = phase + n * t  # argument of latitude
    om = raan + raan_dot * t
    lat = np.arcsin(np.sin(inc) * np.sin(u))
    gmst = WE * t
    lon = np.mod(om + np.arctan2(np.cos(inc) * np.sin(u), np.cos(u)) - gmst + np.pi, 2 * np.pi) - np.pi
    # approximate subsolar point (fixed epoch near equinox), for the day/night gate
    doy = 80.0
    lam_ecl = 2 * np.pi * (doy + t / 86400.0) / 365.25
    ra_sun = np.arctan2(np.cos(OBLIQ) * np.sin(lam_ecl), np.cos(lam_ecl))
    dec_sun = np.arcsin(np.sin(OBLIQ) * np.sin(lam_ecl))
    subsolar_lon = np.mod(ra_sun - gmst + np.pi, 2 * np.pi) - np.pi
    hour_angle = lon - subsolar_lon
    solar_elev = np.arcsin(
        np.sin(lat) * np.sin(dec_sun) + np.cos(lat) * np.cos(dec_sun) * np.cos(hour_angle)
    )
    return lat, lon, solar_elev > 0.0


def coverage_vector(a, inc, raan, phase, elev_mask_deg, day_only):
    """Fraction of time each latitude band is within the elevation-masked footprint (sensor-gated)."""
    lat, _lon, sunlit = subpoints(a, inc, raan, phase)
    eps = np.radians(elev_mask_deg)
    # Earth-central half-angle of the access footprint at this altitude and elevation mask.
    lam = np.arccos((RE / a) * np.cos(eps)) - eps
    lam_deg = np.degrees(lam)
    lat_deg = np.degrees(lat)
    gate = sunlit if day_only else np.ones_like(sunlit, dtype=bool)
    cover = np.zeros(len(BANDS))
    for b, bc in enumerate(np.degrees(BANDS)):
        hit = (np.abs(bc - lat_deg) < lam_deg) & gate
        cover[b] = hit.mean()
    return cover


def jitter(base, spread):
    return base + RNG.uniform(-spread, spread)


def part_a():
    """Four regimes: element clustering vs geometric-coverage clustering."""
    regimes = {
        "Starlink-LEO": {"a": 6928.0, "inc": 53.0},
        "GNSS-MEO": {"a": 26_560.0, "inc": 55.0},
        "SSO-EO": {"a": 7078.0, "inc": 98.6},
        "GEO-comms": {"a": 42_164.0, "inc": 0.1},
    }
    per = 25
    elems, covs, truth = [], [], []
    for label, (_name, base) in enumerate(regimes.items()):
        for _ in range(per):
            a = jitter(base["a"], base["a"] * 0.003)
            inc = np.radians(jitter(base["inc"], 1.0))
            raan = RNG.uniform(0, 2 * np.pi)
            phase = RNG.uniform(0, 2 * np.pi)
            elems.append([a, np.degrees(inc)])
            covs.append(coverage_vector(a, inc, raan, phase, elev_mask_deg=5.0, day_only=False))
            truth.append(label)
    elems = StandardScaler().fit_transform(np.array(elems))
    covs = StandardScaler().fit_transform(np.array(covs))
    truth = np.array(truth)
    ke = KMeans(4, n_init=10, random_state=0).fit_predict(elems)
    kc = KMeans(4, n_init=10, random_state=0).fit_predict(covs)
    print("PART A — four regimes (element vs geometric-coverage clustering)")
    print(f"  ARI(elements, truth)          = {ari(truth, ke):.3f}   AMI = {ami(truth, ke):.3f}")
    print(f"  ARI(coverage, truth)          = {ari(truth, kc):.3f}   AMI = {ami(truth, kc):.3f}")
    print(f"  ARI(coverage, elements)       = {ari(ke, kc):.3f}   <- null test: ~1 => geometry adds nothing")
    return ari(ke, kc)


def part_b():
    """One LEO family; two sensor types. Can the sensor be recovered from coverage but not elements?"""
    per = 30
    elems, covs, sensor = [], [], []
    for _ in range(per):
        a = jitter(6928.0, 20.0)  # nearly identical elements across the family
        inc = np.radians(jitter(53.0, 0.5))
        raan = RNG.uniform(0, 2 * np.pi)
        phase = RNG.uniform(0, 2 * np.pi)
        s = RNG.integers(0, 2)  # 0 = wide all-time, 1 = narrow day-only
        elems.append([a, np.degrees(inc), raan])
        if s == 0:
            covs.append(coverage_vector(a, inc, raan, phase, elev_mask_deg=5.0, day_only=False))
        else:
            covs.append(coverage_vector(a, inc, raan, phase, elev_mask_deg=45.0, day_only=True))
        sensor.append(int(s))
    elems = StandardScaler().fit_transform(np.array(elems))
    covs = StandardScaler().fit_transform(np.array(covs))
    sensor = np.array(sensor)
    ke = KMeans(2, n_init=10, random_state=0).fit_predict(elems)
    kc = KMeans(2, n_init=10, random_state=0).fit_predict(covs)
    a_el = ari(sensor, ke)
    a_cov = ari(sensor, kc)
    print("\nPART B — one family, two sensors (does the sensor add information beyond elements?)")
    print(f"  ARI(elements, sensor)         = {a_el:.3f}   <- elements are blind to the sensor (expect ~0)")
    print(f"  ARI(coverage+sensor, sensor)  = {a_cov:.3f}   <- sensor-aware coverage separates them")
    return a_el, a_cov


def main():
    print("=" * 84)
    print("Phase-6 ML null-hypothesis experiment: coverage clustering vs orbital elements")
    print("=" * 84)
    null = part_a()
    a_el, a_cov = part_b()
    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    geometry_redundant = null > 0.85
    sensor_informative = a_cov - a_el > 0.3
    if geometry_redundant:
        print(f"  - Geometric coverage ~ orbital elements (ARI {null:.2f}): no novelty from geometry alone.")
    else:
        print(f"  - Geometric coverage already differs from elements (ARI {null:.2f}).")
    if sensor_informative:
        print(f"  - Sensor model IS a separable information source ({a_cov:.2f} vs {a_el:.2f}).")
        print("  => ML layer is alive, but its value lives in the SENSOR-aware features, not raw geometry.")
    else:
        print(f"  - Sensor model did not separate ({a_cov:.2f} vs {a_el:.2f}).")
        print("  => On this setup the layer shows no information beyond elements. Rethink the features.")


if __name__ == "__main__":
    main()
