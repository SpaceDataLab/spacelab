#!/usr/bin/env python
"""NRLMSISE-00 thermospheric drag in spacelab — the modern LEO density model, now a drag force.

Two parts:
  1. Cross-check spacelab's raw NRLMSISE-00 density against the reference `nrlmsise00` Python package
     (Brodowski's C, the same model) at a few altitudes — end-to-end validation of the port + units.
  2. Propagate a 350 km LEO for a day under point-mass gravity + NRLMSISE-00 drag and report the
     semi-major-axis decay (which, with a point-mass central force, comes entirely from drag).

    uv run --no-project --with /tmp/w/*.whl --with nrlmsise00 --with numpy python \
        experiments/nrlmsise00_drag_demo.py
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import spacedatalab as sl
from nrlmsise00 import msise_model

MU = 3.986_004_418e14


def sma(r: np.ndarray, v: np.ndarray) -> float:
    return 1.0 / (2.0 / float(np.linalg.norm(r)) - float(v @ v) / MU)


def main() -> None:
    # 1) density cross-check vs the reference package (below ~500 km gtd7 and gtd7d agree)
    doy, sec = 1, 43_200.0  # 2024-01-01, 12:00 UT
    print("NRLMSISE-00 density: spacelab vs reference nrlmsise00 package")
    for alt in (250.0, 400.0):
        rho_sl = sl.nrlmsise00_density(doy, sec, alt, 0.0, 0.0, 150.0, 150.0, 4.0)
        dens, _ = msise_model(datetime(2024, 1, 1, 12, 0, 0), alt, 0.0, 0.0, 150, 150, 4)
        rho_ref = dens[5] * 1000.0
        rel = abs(rho_sl - rho_ref) / rho_ref
        print(f"  {alt:5.0f} km:  spacelab {rho_sl:.4e}  ref {rho_ref:.4e} kg/m^3   rel {rel:.2e}")
        assert rel < 0.02, f"density mismatch at {alt} km: {rel:.2e}"

    # 2) orbital decay from NRLMSISE-00 drag over a day
    alt0 = 350e3
    r0 = [6.378_137e6 + alt0, 0.0, 0.0]
    vc = (MU / np.linalg.norm(r0)) ** 0.5
    inc = np.radians(51.6)
    v0 = [0.0, vc * np.cos(inc), vc * np.sin(inc)]
    day = 86_400.0
    naive = round(8765.5 * 86_400.0 * 1e9)  # 2024-01-01 UTC
    tt_ns = naive + round(69.184 * 1e9)
    ut1_ns = naive + round(0.008757 * 1e9)

    eph = sl.propagate_drag_nrlmsise00_dense(
        r0, v0, day, tt_ns, ut1_ns, doy, sec, 0.02, f107=150.0, f107a=150.0, ap=4.0, tol=1e-10
    )
    r_end, v_end = eph.eval(day)
    decay = sma(np.array(r0), np.array(v0)) - sma(np.array(r_end), np.array(v_end))
    print(
        f"NRLMSISE-00 drag decays the {alt0 / 1e3:.0f} km orbit's semi-major axis by "
        f"{decay:.1f} m over a day (C_dA/m = 0.02 m^2/kg, F10.7 = 150)"
    )
    assert decay > 10.0, "drag at 350 km should lower the SMA by tens of metres or more per day"


if __name__ == "__main__":
    main()
