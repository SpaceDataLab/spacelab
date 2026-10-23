#!/usr/bin/env python3
"""Machine-generate the IGRF-14 coefficient tables for `spacelab-gnss`.

Downloads (or reads) the IAGA `igrf14coeffs.txt` and emits
`crates/spacelab-geomagnetism/src/igrf14_coeffs.rs` — the epoch-2025.0 Schmidt
semi-normalised Gauss coefficients (nT) and the 2025-2030 secular variation
(nT/yr), indexed `[n][m]`. Never hand-transcribe the coefficients; regenerate
this on an IGRF generation bump.

    curl -sSL https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt -o igrf14coeffs.txt
    python derivations/igrf14_tables.py igrf14coeffs.txt
"""

import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "igrf14coeffs.txt"
OUT = "crates/spacelab-geomagnetism/src/igrf14_coeffs.rs"


def main() -> None:
    with open(SRC) as fh:
        lines = fh.read().splitlines()
    header = next(line for line in lines if line.split()[:2] == ["g/h", "n"])
    toks = header.split()
    i2025, isv = toks.index("2025.0"), toks.index("2025-30")

    g = [[0.0] * 14 for _ in range(14)]
    h = [[0.0] * 14 for _ in range(14)]
    gs = [[0.0] * 14 for _ in range(14)]
    hs = [[0.0] * 14 for _ in range(14)]
    for line in lines:
        f = line.split()
        if not f or f[0] not in ("g", "h"):
            continue
        n, m = int(f[1]), int(f[2])
        value, sv = float(f[i2025]), (float(f[isv]) if len(f) > isv else 0.0)
        (g if f[0] == "g" else h)[n][m] = value
        (gs if f[0] == "g" else hs)[n][m] = sv

    def emit(name, mat):
        rows = ",\n    ".join(
            "[" + ", ".join(repr(x) for x in r) + "]" for r in mat
        )
        return (
            f"#[rustfmt::skip]\npub(crate) const {name}: [[f64; 14]; 14] = [\n"
            f"    {rows},\n];\n"
        )

    out = (
        "//! IGRF-14 (epoch 2025.0) Schmidt semi-normalised Gauss coefficients (nT) and 2025-2030 secular\n"
        "//! variation (nT/yr), machine-generated from the IAGA `igrf14coeffs.txt`. Indexed `[n][m]`,\n"
        "//! `n, m = 0..=13`. Regenerate with `derivations/igrf14_tables.py` on an IGRF bump.\n\n"
        "#![allow(clippy::unreadable_literal, clippy::excessive_precision)]\n\n"
        + "\n".join(
            emit(n, mat)
            for n, mat in [("G_2025", g), ("H_2025", h), ("G_SV", gs), ("H_SV", hs)]
        )
    )
    with open(OUT, "w") as fh:
        fh.write(out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
