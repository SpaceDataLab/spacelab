"""Compare a NEPTUNE selenocentric run against the NAIF reconstruction of the same arc.

Read by tools/neptune-oracle/check.sh. The arbiter is SPICE itself, through spiceypy, so that
nothing in spacelab takes part in judging the build of the oracle that will later judge spacelab.

Two frame statements worth keeping straight, since neither is an error and both are small:

  * NEPTUNE calls its output frame MCRF and defines it as "axes aligned with GCRF, origin at the
    Moon's centre of mass"; SPICE is asked for J2000. GCRF and J2000 differ by the frame bias,
    about 23 mas, which at LRO's 1800 km orbit radius is 0.2 m. It is inside every residual below
    and is not worth removing.
  * The reconstruction is the operational product, not truth. It carries its own orbit-determination
    error, which for LRO is at the tens-of-metres level radially.
"""

import argparse
import sys

import numpy as np
import spiceypy as spice

LRO = "-85"
MOON = "301"


def read_mcs(path):
    """Return (utc_strings, states) from a NEPTUNE .mcs file, positions and rates in km, km/s."""
    epochs, states = [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(
                    f"expected 8 fields, got {len(fields)}: {line.rstrip()}"
                )
            epochs.append(fields[0])
            states.append([float(value) for value in fields[2:]])
    if not epochs:
        raise ValueError(f"no data rows in {path}")
    return epochs, np.asarray(states)


def utc_to_et(stamp):
    """NEPTUNE writes 2025-10-01T00:00:00.000000Z; SPICE wants no delimiter and no zone letter."""
    return spice.utc2et(stamp.rstrip("Z").replace("T", " "))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcs", required=True)
    parser.add_argument("--spk", required=True)
    parser.add_argument("--lsk", required=True)
    parser.add_argument("--cap-km", type=float, default=5.0)
    args = parser.parse_args()

    spice.furnsh(args.lsk)
    spice.furnsh(args.spk)

    epochs, states = read_mcs(args.mcs)
    residuals = np.empty(len(epochs))
    for index, stamp in enumerate(epochs):
        truth, _ = spice.spkezr(LRO, utc_to_et(stamp), "J2000", "NONE", MOON)
        residuals[index] = np.linalg.norm(states[index, :3] - truth[:3])

    rms = float(np.sqrt(np.mean(residuals**2)))
    worst = float(residuals.max())
    hours = (utc_to_et(epochs[-1]) - utc_to_et(epochs[0])) / 3600.0
    print(
        f"span            {epochs[0]} .. {epochs[-1]}  ({hours:.2f} h, {len(epochs)} samples)"
    )
    print(f"first residual  {residuals[0] * 1e3:.3f} m")
    print(f"rms residual    {rms:.4f} km")
    print(f"worst residual  {worst:.4f} km  at {epochs[int(residuals.argmax())]}")

    # The growth curve, not just its endpoint: a build broken in the force model and a build broken
    # in the initial state both end up at the same worst residual, and they separate here.
    start = utc_to_et(epochs[0])
    elapsed = np.array([utc_to_et(stamp) - start for stamp in epochs])
    print("elapsed        residual")
    for minutes in (10.0, 30.0, 60.0, 360.0, 1440.0, 4320.0, 10080.0):
        seconds = minutes * 60.0
        if seconds > elapsed[-1]:
            continue
        index = int(np.abs(elapsed - seconds).argmin())
        print(f"{minutes:8.0f} min  {residuals[index] * 1e3:12.1f} m")

    if worst > args.cap_km:
        print(
            f"FAIL: worst {worst:.4f} km exceeds the {args.cap_km:.4f} km cap",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
