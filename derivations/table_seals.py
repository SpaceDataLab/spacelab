#!/usr/bin/env python
"""Seal the machine-generated coefficient tables: FNV-1a over the doubles they hold.

Most of the tables listed below are transcribed from published models -- XGM2019e, FES2004,
FES2014b, NRLMSISE-00, and the Niell/GMF/VMF3 mapping-function coefficients -- and are verified by
provenance, not by anything an accuracy test can assert. `GOT_WAVES` is the exception and is sealed
for a different reason: it is *computed* by `derivations/tide_to_stokes.py`, so the seal is what says
the committed table is still the one that generator produced from the atlas it names. A single flipped sign at degree 85 of the gravity field moves the
acceleration orders of magnitude below the agreement the field oracles already demand, so every
existing test still passes over the corrupted table. Mutation testing makes that concrete: 8619 of
the crate's mutants are `delete -` on a table entry, and nothing in the suite can see any of them.

What a test can see is that the bytes are not the ones the generator wrote. This script computes
that seal by parsing the emitted Rust and reading every literal with Python's own float parser, so
the constant the Rust test asserts comes from a different parse of the same text than rustc's. The
Rust side folds the table it actually compiled; agreement means the two parses landed on the same
doubles in the same order.

Re-run and update `crates/spacelab-propagation/tests/generated_tables.rs` and the seal tests in
`crates/spacelab-propagation/src/nrlmsise00/mod.rs` and `crates/spacelab-gnss/src/models.rs`
whenever a table is legitimately regenerated:

    uv run --no-project python derivations/table_seals.py
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK = 0xFFFFFFFFFFFFFFFF

# (path under `crates/`, table names in the order the Rust test folds them).
TABLES: list[tuple[str, tuple[str, ...]]] = [
    ("spacelab-propagation/src/geopotential_coeffs.rs", ("MODEL_COEFFS",)),
    ("spacelab-propagation/src/ocean_tide_fes2004.rs", ("FES2004_WAVES",)),
    ("spacelab-propagation/src/ocean_tide_fes2014_table.in", ("FES2014B_WAVES",)),
    ("spacelab-propagation/src/ocean_tide_got.rs", ("GOT_WAVES",)),
    (
        "spacelab-propagation/src/nrlmsise00/tables.rs",
        ("PT", "PD", "PS", "PDL", "PTM", "PDM", "PTL", "PMA", "SAM", "PAVGM"),
    ),
    (
        "spacelab-gnss/src/models.rs",
        (
            "NIELL_AH_AVG",
            "NIELL_BH_AVG",
            "NIELL_CH_AVG",
            "NIELL_AH_AMP",
            "NIELL_BH_AMP",
            "NIELL_CH_AMP",
            "NIELL_AW",
            "NIELL_BW",
            "NIELL_CW",
            "GMF_AH_MEAN",
            "GMF_BH_MEAN",
            "GMF_AH_AMPL",
            "GMF_BH_AMPL",
            "GMF_AW_MEAN",
            "GMF_BW_MEAN",
            "GMF_AW_AMPL",
            "GMF_BW_AMPL",
            "BH_A",
            "BW_A",
            "CH_A",
            "CW_A",
            "BH_B",
            "BW_B",
            "CH_B",
            "CW_B",
        ),
    ),
    ("spacelab-gnss/src/igrf14_coeffs.rs", ("G_2025", "H_2025", "G_SV", "H_SV")),
    (
        "spacelab-fa/src/nut_tables.rs",
        (
            "XLS_MULT",
            "XLS_COEF",
            "XPL_MULT",
            "XPL_COEF",
            "LS_ELP",
            "LS_D",
            "PL_AL",
            "PL_AF",
            "PL_AD",
            "PL_AOM",
            "PL_ANE",
        ),
    ),
    (
        "spacelab-fa/src/cio_tables.rs",
        (
            "XYP",
            "XY_MULT",
            "XY_OFFSET",
            "XY_CODE",
            "XY_AMP",
            "S_POLY",
            "S_MULT",
            "S_PT",
            "S_SIN",
            "S_COS",
        ),
    ),
    (
        "spacelab-propagation/src/tide_step2_tables.rs",
        ("DIURNAL", "LONG_PERIOD", "SEMIDIURNAL"),
    ),
]

NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def body(text: str, name: str) -> str:
    """The initializer of `pub static|const <name>`, from the `=` to the `;` that closes the item.

    Cutting at the `=` is what keeps the type annotation out: `[f64; 150]` and `f64` both carry
    digits that would otherwise be folded in as data. No initializer here contains a `;`.

    `pub`, and any `pub(...)` restriction on it, is optional: a table private to its module or
    crate is sealed by a unit test inside it.
    """
    decl = rf"^(?:pub(?:\([^)]*\))? )?(?:static|const) {name}\b[^=]*="
    start = re.search(decl, text, re.MULTILINE)
    if start is None:
        raise SystemExit(f"{name}: no declaration found")
    rest = text[start.end() :]
    return rest[: rest.index(";")]


def values(text: str, name: str) -> list[float]:
    """Every literal in the table, in file order, with `//` comments removed first."""
    stripped = "\n".join(line.split("//")[0] for line in body(text, name).splitlines())
    return [float(tok) for tok in NUMBER.findall(stripped)]


def seal(vals: list[float]) -> int:
    h = FNV_OFFSET
    for v in vals:
        for b in struct.pack("<d", v):
            h = ((h ^ b) * FNV_PRIME) & MASK
    return h


def main() -> None:
    for filename, names in TABLES:
        path = Path("crates") / filename
        if not path.exists():
            raise SystemExit(f"{path}: not found -- run from the repository root")
        text = path.read_text()
        folded: list[float] = []
        for name in names:
            vals = values(text, name)
            if not vals:
                raise SystemExit(f"{name}: parsed no values")
            folded.extend(vals)
        print(f"{filename:<48} {len(folded):>7} values  0x{seal(folded):016x}")


if __name__ == "__main__":
    main()
