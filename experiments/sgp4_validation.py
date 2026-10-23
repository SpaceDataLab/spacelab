#!/usr/bin/env python
"""Cross-validate the Rust SGP4 against python-sgp4 (Vallado's reference implementation).

The plan's MVP acceptance criteria name python-sgp4 and skyfield as the reference stack; this closes
the SGP4 half as a runnable, committed artifact. Both implement the same standard SGP4/SDP4 over the
same WGS-72 constants and return TEME state, so a faithful port must agree to well under a metre over a
full day — a km-scale gap would mean a constants or branch (near-earth vs deep-space) mismatch, not
mere rounding. This is the propagator every coverage and access-window result is built on, so its
agreement with the reference is load-bearing.

Build the wheel first, then run with it on the path:
  cd crates/spacelab-python && maturin build --release -o /tmp/w
  uv run --no-project --with /tmp/w/*.whl --with sgp4 python experiments/sgp4_validation.py
"""

from __future__ import annotations

import spacedatalab as sl
from sgp4.api import Satrec

# Near-earth (SGP4 branch); the reference case already used by the smoke test.
ISS = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
# Deep-space (SDP4 branch, period > 225 min): a GPS-class ~12 h orbit exercises the resonance/lunar-
# solar path, which is a completely different code branch from the ISS case.
GPS = (
    "1 24876U 97035A   08264.51782528 -.00000029  00000-0  00000+0 0  9990",
    "2 24876  55.6410  10.7530 0053050  55.1450 305.3620  2.00563190 79325",
)

KM = 1000.0


def norm(v):
    return sum(c * c for c in v) ** 0.5


def compare(name: str, tle: tuple[str, str], step_min: float, span_min: float):
    l1, l2 = tle
    ref = Satrec.twoline2rv(l1, l2)
    jd0, fr0 = ref.jdsatepoch, ref.jdsatepochF

    max_pos = 0.0
    max_vel = 0.0
    n = int(span_min / step_min)
    for k in range(n + 1):
        minutes = k * step_min
        err, r_km, v_km = ref.sgp4(jd0, fr0 + minutes / 1440.0)
        if err != 0:
            raise RuntimeError(f"python-sgp4 error {err} at t={minutes} min for {name}")
        r_ref = [c * KM for c in r_km]  # m, TEME
        v_ref = [c * KM for c in v_km]  # m/s, TEME
        r_sl, v_sl = sl.sgp4_propagate(l1, l2, float(minutes))
        max_pos = max(max_pos, norm([r_sl[i] - r_ref[i] for i in range(3)]))
        max_vel = max(max_vel, norm([v_sl[i] - v_ref[i] for i in range(3)]))

    print(f"{name:11s}: max |Δr| = {max_pos:.3e} m   max |Δv| = {max_vel:.3e} m/s   over {span_min:.0f} min")
    return max_pos, max_vel


def main() -> None:
    # 24 h at 1-min cadence for the near-earth case; a coarser but longer sweep for deep space.
    iss_pos, iss_vel = compare("ISS/SGP4", ISS, step_min=1.0, span_min=1440.0)
    gps_pos, gps_vel = compare("GPS/SDP4", GPS, step_min=5.0, span_min=1440.0)

    # A faithful port agrees with the reference to sub-metre / sub-mm/s; leave headroom over pure
    # numerical noise but far below anything a real constants/branch bug would produce.
    assert iss_pos < 1.0, f"ISS position disagrees with python-sgp4 by {iss_pos:.3e} m"
    assert iss_vel < 1e-3, f"ISS velocity disagrees by {iss_vel:.3e} m/s"
    assert gps_pos < 1.0, f"GPS position disagrees with python-sgp4 by {gps_pos:.3e} m"
    assert gps_vel < 1e-3, f"GPS velocity disagrees by {gps_vel:.3e} m/s"
    print("Rust SGP4/SDP4 matches the python-sgp4 reference to sub-metre over a full day")


if __name__ == "__main__":
    main()
