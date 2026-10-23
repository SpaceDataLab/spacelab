#!/usr/bin/env python3
"""Extract the 342-constituent tidal catalogue that HARDISP's ADMINT carries.

The catalogue is the Cartwright-Tayler-Edden harmonic development: for each constituent, its
six Doodson multipliers and its Cartwright-Edden equilibrium amplitude. Transcribing 342 rows of
seven numbers by hand is not a thing a person does correctly, so it is machine-read out of the
IERS Conventions Centre's own `ADMINT.F` and emitted as Rust.

`ADMINT.F` is fetched rather than vendored, on the same terms as `tools/iers-oracle/run.sh`: it
belongs to the Conventions Centre. What this repository commits is this reader and the table it
produces.

Run: uv run python derivations/hardisp_tables.py
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

SOURCE = "https://iers-conventions.obspm.fr/content/chapter7/software/hardisp/ADMINT.F"
OUT = (
    Path(__file__).resolve().parent.parent / "crates/spacelab-slr/src/loading_tables.rs"
)
COUNT = 342


def fortran_source() -> str:
    # The URL is a pinned https constant, not caller input.
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        return response.read().decode("latin-1")


def data_block(source: str, name: str) -> str:
    """The text between `DATA <name>/` and the closing slash, continuation markers removed."""
    start = source.index(f"DATA {name}/") + len(f"DATA {name}/")
    end = source.index("/", start)
    body = source[start:end]
    return "\n".join(
        line[6:] if line.startswith("     .") else line for line in body.splitlines()
    )


def grouped(value: float) -> str:
    """Six decimals with a digit separator, which is what the workspace lints require."""
    whole, decimals = f"{value:.6f}".split(".")
    return f"{whole}.{decimals[:3]}_{decimals[3:]}".rjust(13)


def main() -> int:
    source = fortran_source()

    amplitudes = [float(v) for v in re.findall(r"-?\.\d+", data_block(source, "TAMP"))]
    doodson = [int(v) for v in re.findall(r"-?\d+", data_block(source, "IDD"))]

    if len(amplitudes) != COUNT:
        print(f"expected {COUNT} amplitudes, read {len(amplitudes)}", file=sys.stderr)
        return 1
    if len(doodson) != 6 * COUNT:
        print(
            f"expected {6 * COUNT} Doodson multipliers, read {len(doodson)}",
            file=sys.stderr,
        )
        return 1

    rows = [
        f"    ({doodson[6 * i]:2}, {doodson[6 * i + 1]:2}, {doodson[6 * i + 2]:2}, "
        f"{doodson[6 * i + 3]:2}, {doodson[6 * i + 4]:2}, {doodson[6 * i + 5]:2}, "
        f"{grouped(amplitudes[i])}),"
        for i in range(COUNT)
    ]

    OUT.write_text(
        f"""//! The Cartwright-Tayler-Edden harmonic development, {COUNT} constituents.
//!
//! Machine-read out of the IERS Conventions Centre's `ADMINT.F` by
//! `derivations/hardisp_tables.py`. Regenerate, never hand-edit: transcribing {COUNT} rows of six
//! integer multipliers and an amplitude by hand is not an operation that succeeds.

/// One constituent: the six Doodson multipliers `(tau, s, h, p, N', p_s)` and its Cartwright-Edden
/// equilibrium amplitude in metres. The amplitude's sign is part of the development; the admittance
/// is formed against its magnitude and the sign is carried into the phase.
pub(crate) type Constituent = (i8, i8, i8, i8, i8, i8, f64);

/// Ordered as `ADMINT` carries them — by descending `|amplitude|` within the whole catalogue, not
/// by frequency. The order is immaterial to the result but is kept so a reader can diff the two.
#[rustfmt::skip]
pub(crate) static CARTWRIGHT_EDDEN: [Constituent; {COUNT}] = [
{chr(10).join(rows)}
];
""",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({COUNT} constituents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
