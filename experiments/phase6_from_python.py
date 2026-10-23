#!/usr/bin/env python
"""Phase-6 gate computed entirely from Python via the spacelab bridge — no Rust example, no CSV.

Where experiments/phase6_real_engine.py reads features a Rust example wrote to CSV, this computes them
in-process through the PyO3 bridge: `spacelab.coverage_metrics` runs the production access-window
engine per satellite, and the 18 latitude-band features (coverage fraction + max-gap ratio) are
assembled in NumPy. The null-hypothesis conclusion is then reproduced with the bridge as the only link
to the Rust core.

Build the wheel first, then run with it on the path:
  cd crates/spacelab-python && maturin build --release -o /tmp/w
  uv run --no-project --with /tmp/w/*.whl --with numpy --with scikit-learn --with betula-cluster \
      python experiments/phase6_from_python.py
"""

from __future__ import annotations

import math

import numpy as np
import spacelab as sl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)
RE = 6_378_137.0
LATS = [-80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0]
LONS = [30.0 * j for j in range(12)]
CELLS = sl.grid_cells(LATS, LONS)
HORIZON = 43_200.0


def features(a, inc_deg, raan_deg, phase_deg, sensor):
    """18 band features (9 mean coverage fractions + 9 mean max-gap ratios), from the real engine."""
    m = sl.coverage_metrics(
        a, math.radians(inc_deg), math.radians(raan_deg), math.radians(phase_deg),
        sensor, CELLS, HORIZON, tol=2.0, threads=8,
    )
    m = np.asarray(m).reshape(len(LATS), len(LONS), 4)  # [band, lon, (passes,cov,maxgap,meangap)]
    cov = m[:, :, 1].mean(axis=1)
    gap = np.minimum(m[:, :, 2] / HORIZON, 1.0).mean(axis=1)
    return np.concatenate([cov, gap])


def scale(X):
    return StandardScaler().fit_transform(X)


def cluster(X, k):
    return KMeans(k, n_init=10, random_state=0).fit_predict(scale(X))


def main():
    wide = sl.Sensor.elevation(5.0)
    sensors = [("wide", wide), ("sar", sl.Sensor.sar(20.0, 46.0)), ("agile", sl.Sensor.agile(20.0))]
    regimes = [("Starlink-LEO", RE + 550e3, 53.0), ("GNSS-MEO", RE + 20_180e3, 55.0),
               ("SSO-EO", RE + 700e3, 98.0), ("GEO-comms", 42_164e3, 0.1)]

    elem_a, cov_a, reg = [], [], []
    for label, (_name, a0, i0) in enumerate(regimes):
        for _ in range(8):
            a = a0 * (1 + RNG.uniform(-0.003, 0.003))
            inc = i0 + RNG.uniform(-1, 1)
            raan, phase = RNG.uniform(0, 360), RNG.uniform(0, 360)
            elem_a.append([a, inc, raan])
            cov_a.append(features(a, inc, raan, phase, wide))
            reg.append(label)

    elem_b, cov_b, sen = [], [], []
    for k in range(45):
        a = (RE + 550e3) * (1 + RNG.uniform(-0.001, 0.001))
        inc = 53.0 + RNG.uniform(-0.5, 0.5)
        raan, phase = RNG.uniform(0, 360), RNG.uniform(0, 360)
        s = k % 3
        elem_b.append([a, inc, raan])
        cov_b.append(features(a, inc, raan, phase, sensors[s][1]))
        sen.append(s)

    elem_a, cov_a, reg = np.array(elem_a), np.array(cov_a), np.array(reg)
    elem_b, cov_b, sen = np.array(elem_b), np.array(cov_b), np.array(sen)

    print("=" * 80)
    print("Phase-6 gate via the PyO3 bridge — features computed in Python, engine in Rust")
    print("=" * 80)
    print("\nPART A — regimes (coverage vs elements)")
    ke, kc = cluster(elem_a, 4), cluster(cov_a, 4)
    print(f"  ARI(coverage, truth)={ari(reg, kc):.3f}  ARI(elements, truth)={ari(reg, ke):.3f}  "
          f"ARI(coverage, elements)={ari(ke, kc):.3f}")
    print("\nPART B — one family, three sensors")
    ke, kc = cluster(elem_b, 3), cluster(cov_b, 3)
    a_el, a_cov = ari(sen, ke), ari(sen, kc)
    print(f"  ARI(coverage+sensor, sensor)={a_cov:.3f}   ARI(elements, sensor)={a_el:.3f}")
    print("\n" + "=" * 80)
    verdict = "confirmed" if a_cov - a_el > 0.3 else "NOT confirmed"
    print(f"  Phase-6 finding {verdict} natively from Python (no CSV): sensor recoverable from")
    print(f"  coverage ({a_cov:.2f}), invisible to elements ({a_el:.2f}).")


if __name__ == "__main__":
    main()
