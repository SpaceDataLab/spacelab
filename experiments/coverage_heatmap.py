#!/usr/bin/env python
"""Phase-7, first slice: the MVP coverage-heatmap output (2-D), computed through the bridge.

Renders the two maps the MVP promises — coverage fraction and mean revisit gap — over a lat/lon grid,
straight from the Rust access-window engine via `spacelab`. A PNG is written; the coverage
pattern is also summarised in-console so the science is verifiable without a display (for a LEO at
inclination i, coverage peaks near ±i and vanishes past the reach band ±(i + footprint)).

Build the wheel, then run with it on the path:
  cd crates/spacelab-python && maturin build --release -o /tmp/w
  uv run --no-project --with /tmp/w/*.whl --with numpy --with matplotlib \
      python experiments/coverage_heatmap.py [out.png]
"""

from __future__ import annotations

import math
import sys

import numpy as np
import spacelab as sl

RE = 6_378_137.0
HORIZON = 86_400.0


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/coverage.png"
    lats = np.arange(-85.0, 86.0, 5.0)
    lons = np.arange(-180.0, 180.0, 5.0)
    cells = sl.grid_cells([float(x) for x in lats], [float(x) for x in lons])

    a, inc = RE + 550e3, 53.0
    m = np.array(sl.coverage_metrics(a, math.radians(inc), 0.0, 0.0, sl.Sensor.elevation(5.0), cells, HORIZON, tol=2.0, threads=8))
    cov = m[:, 1].reshape(len(lats), len(lons))          # coverage fraction
    gap_h = np.where(m[:, 0].reshape(len(lats), len(lons)) > 0, m[:, 3].reshape(len(lats), len(lons)) / 3600.0, np.nan)

    print(f"coverage heatmap: Starlink-like LEO (a={a/1e3:.0f} km, i={inc}°), 24 h, elevation 5°")
    print("  mean coverage fraction by latitude band:")
    for j, lat in enumerate(lats):
        bar = "█" * round(cov[j].mean() * 60)
        print(f"    {lat:>+5.0f}°  {cov[j].mean():5.3f}  {bar}")
    peak = lats[np.argmax(cov.mean(axis=1))]
    print(f"  coverage peaks at lat {peak:+.0f}° (orbit inclination {inc}°), vanishes toward the poles — as expected.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ext = [lons[0], lons[-1], lats[0], lats[-1]]
        im1 = ax1.imshow(cov, origin="lower", extent=ext, aspect="auto", cmap="viridis")
        ax1.set(title="coverage fraction (24 h)", xlabel="longitude", ylabel="latitude")
        fig.colorbar(im1, ax=ax1, shrink=0.8)
        im2 = ax2.imshow(gap_h, origin="lower", extent=ext, aspect="auto", cmap="magma_r")
        ax2.set(title="mean revisit gap (hours)", xlabel="longitude", ylabel="latitude")
        fig.colorbar(im2, ax=ax2, shrink=0.8)
        fig.suptitle(f"spacelab coverage — LEO a={a/1e3:.0f} km, i={inc}°, elevation 5°")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        import os
        print(f"\n  wrote {out} ({os.path.getsize(out)//1024} KB)")
    except Exception as e:
        print(f"\n  (matplotlib unavailable, skipped PNG: {e})")


if __name__ == "__main__":
    main()
