#!/usr/bin/env python
"""Re-derive the machine-generated coefficient tables from their PRIMARY SOURCE files.

`table_seals.py` seals a table against a second parse of the *emitted Rust*. That catches a hand
edit and nothing else: it cannot see a generator that mis-read the source, a source file the issuing
agency has since revised, or a table regenerated with different truncation options. Every one of
those produces a table that is internally consistent and wrong, which is the failure mode neither
mutation testing nor an accuracy oracle can reach -- both of them assume the coefficients are the
ones the model publishes.

This script closes that loop by going back to the file the agency publishes:

    primary file -> the extraction the generator performs -> FNV-1a -> the constant the Rust asserts

The extractions below are written from the file formats, not imported from the generators, so a
disagreement is a real finding rather than a shared bug. Agreement means the compiled table is what
the primary source says today, under two independent parses of two different artefacts.

Primary files are not vendored -- XGM2019e is 1.1 GB in full and FES2014b 10.8 MB, and neither is
ours to redistribute -- so `tools/fetch-specs.sh` fetches them into `data/specs/` as digest-named
snapshots. Point this script at that directory:

    uv run --no-project python derivations/spec_seals.py data/specs
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK = 0xFFFFFFFFFFFFFFFF

# The seals the Rust tests assert, and where they live. Kept here rather than scraped so that a
# disagreement names the file to look at.
EXPECTED = {
    "XGM2019e": (0x0EFBB3111EF0B35D, "spacelab-propagation/tests/generated_tables.rs"),
    "FES2014b": (0x4A82951C121044F4, "spacelab-propagation/tests/generated_tables.rs"),
    "IGRF-14": (0xAB0848C895262035, "spacelab-gnss/src/igrf.rs"),
}

# `data/specs` holds digest-named snapshots; match on the stem the fetch script assigns.
PATTERNS = {
    "XGM2019e": "xgm2019e.*.gfc",
    "FES2014b": "fes2014b.*.txt",
    "IGRF-14": "igrf14coeffs.*.txt",
}


def seal(values) -> int:
    h = FNV_OFFSET
    for v in values:
        for b in struct.pack("<d", v):
            h = ((h ^ b) * FNV_PRIME) & MASK
    return h


def gfc_float(tok: str) -> float:
    """ICGEM writes Fortran `D` exponents in some models and `E` in others."""
    return float(tok.replace("d", "e").replace("D", "e"))


def xgm2019e(path: Path, degree: int = 120):
    """`(n, m, C_nm, S_nm)` for 2 <= n <= degree, in file order -- `geopotential_coeffs.rs`.

    Degrees 0 and 1 are dropped: C00 = 1 is the constructor's, and C10 = C11 = 0 holds because the
    model is mass-centred. The header's normalisation is asserted rather than assumed, since a
    4pi-normalised file would parse cleanly and be wrong by a factor per degree.
    """
    norm = None
    with path.open(encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("norm"):
                norm = line.split()[1]
            elif line.startswith("end_of_head"):
                break
        else:
            raise SystemExit(f"{path}: no end_of_head -- not an ICGEM gfc file")
        if norm != "fully_normalized":
            raise SystemExit(f"{path}: normalisation is {norm!r}, not fully_normalized")
        for line in fh:
            toks = line.split()
            if not toks or toks[0] != "gfc":
                continue
            n, m = int(toks[1]), int(toks[2])
            if n > degree:
                return
            if n >= 2:
                yield from (float(n), float(m), gfc_float(toks[3]), gfc_float(toks[4]))


def fes2014b(path: Path, start_degree: int = 2):
    """Doodson number then `(n, m, C+, S+, C-, S-)` per row -- `ocean_tide_fes2014_table.in`.

    Waves ascending by Doodson number, cells within a wave ascending by (n, m): the order
    `ocean_tide_fes2014.py` emits and `generated_tables.rs` folds. The model is legitimately sparse
    at high degree, so an absent (n, m) is absence, never a gap to fill.
    """
    waves: dict[int, dict[tuple[int, int], tuple[float, ...]]] = {}
    with path.open() as fh:
        for lineno, line in enumerate(fh):
            if lineno < 4:  # HEADER_LINES
                continue
            toks = line.split()
            if len(toks) < 8:
                continue
            try:
                doodson = int(toks[0].replace(".", "").replace(",", ""))
                n, m = int(toks[2]), int(toks[3])
                cells = tuple(float(t) for t in toks[4:8])
            except ValueError:
                continue
            if n < start_degree:
                continue
            waves.setdefault(doodson, {})[(n, m)] = cells
    for doodson in sorted(waves):
        yield float(doodson)
        for n, m in sorted(waves[doodson]):
            yield from (float(n), float(m), *waves[doodson][(n, m)])


def igrf14(path: Path, epoch: str = "2025.0", sv: str = "2025-30", nmax: int = 13):
    """`G[n][m]`, `H[n][m]`, then the two secular-variation blocks -- `igrf14_coeffs.rs`.

    Each block is the full `[0..=13][0..=13]` square row-major, so the structural zeros above the
    diagonal are folded in too; that is what the Rust test flattens.
    """
    lines = path.read_text().splitlines()
    header = next(line for line in lines if line.split()[:2] == ["g/h", "n"])
    toks = header.split()
    i_epoch, i_sv = toks.index(epoch), toks.index(sv)
    blocks = [[[0.0] * (nmax + 1) for _ in range(nmax + 1)] for _ in range(4)]
    for line in lines:
        toks = line.split()
        if not toks or toks[0] not in ("g", "h"):
            continue
        n, m = int(toks[1]), int(toks[2])
        if n > nmax:
            continue
        # The header's `g/h n m` is three tokens and so is a row's `g 1 0`, so the epoch columns
        # line up directly. A row that stops short of the SV column contributes a zero rate, as the
        # generator has it -- the file does this for the degrees IGRF does not extrapolate.
        value = float(toks[i_epoch])
        rate = float(toks[i_sv]) if len(toks) > i_sv else 0.0
        blocks[0 if toks[0] == "g" else 1][n][m] = value
        blocks[2 if toks[0] == "g" else 3][n][m] = rate
    for block in blocks:
        for row in block:
            yield from row


EXTRACT = {"XGM2019e": xgm2019e, "FES2014b": fes2014b, "IGRF-14": igrf14}


def main() -> int:
    spec_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/specs")
    if not spec_dir.is_dir():
        print(f"{spec_dir} does not exist -- run tools/fetch-specs.sh first")
        return 2

    rc = 0
    for name, pattern in PATTERNS.items():
        want, where = EXPECTED[name]
        snapshots = sorted(spec_dir.glob(pattern))
        if not snapshots:
            print(f"{name:<10} SKIP   no {pattern} in {spec_dir}")
            rc = max(rc, 1)
            continue
        if len(snapshots) > 1:
            print(
                f"{name:<10} SKIP   {len(snapshots)} snapshots match {pattern}; keep one"
            )
            rc = max(rc, 1)
            continue
        got = seal(EXTRACT[name](snapshots[0]))
        ok = got == want
        rc = max(rc, 0 if ok else 2)
        print(
            f"{name:<10} {'ok  ' if ok else 'WRONG'}   0x{got:016x}"
            f"{'' if ok else f'  != 0x{want:016x} in {where}'}"
            f"   <- {snapshots[0].name}"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
