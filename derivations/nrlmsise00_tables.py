#!/usr/bin/env python
"""Machine-extract the NRLMSISE-00 coefficient tables from the NRL FORTRAN to Rust.

The source is `nrlmsise00_sub.for`, `BLOCK DATA GTD7BK` -- the model's own distribution by
M. Picone, A. E. Hedin and D. Drob at the U.S. Naval Research Laboratory, published through NASA
GSFC's modelweb. A work of the United States Government is not subject to domestic copyright
(17 U.S.C. section 105), which is what makes these numbers safe to ship where a third party's
transcription of them would not be. This script used to read Brodowski's C `nrlmsise-00_data.c`
instead; that file carries no licence grant at all, and it is a transcription of this one.

Values are emitted verbatim -- the exact decimal digits of the FORTRAN, only lowercasing the
exponent `E` -- so a diff against `nrlmsise00_sub.for` is byte-trivial and nothing is rounded. Never
hand-transcribe these; regenerate.

    uv run --no-project python derivations/nrlmsise00_tables.py <path-to-nrlmsise00_sub.for> <out.rs>

The one thing to understand before editing: FORTRAN `COMMON` is *storage*, and NRLMSISE-00 declares
the same storage two different ways. The model routines see

    COMMON/PARM7/PT(150),PD(150,9),PS(150),PDL(25,2),PTL(100,4),PMA(100,10),SAM(100)

while `BLOCK DATA GTD7BK` fills it as 64 chunks of 50 named PT1..PT3, PA1..PA3, ... PAA1, PAA2.
Both are 3200 reals over the same bytes, so the `DATA` statements are read here as one flat stream in
source order and then sliced by the *model's* declaration. Column-major order is what makes each
slice contiguous: `PD(150,9)` varies its first index fastest, so `PD(:,j)` is 150 consecutive values
and becomes row `j` of the Rust `[[f64; 150]; 9]`. `LOWER7` and `MAVG7` are separate blocks and are
read on their own.
"""

from __future__ import annotations

import re
import sys

_NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

# `PARM7`, `LOWER7` and `MAVG7` as the *model* routines declare them, in the order the emitted Rust
# file lists them. Name, Rust dimensions (outer first), and the FORTRAN element count.
PARM7 = [
    ("PT", (150,), 150),
    ("PD", (9, 150), 1350),
    ("PS", (150,), 150),
    ("PDL", (2, 25), 50),
    ("PTL", (4, 100), 400),
    ("PMA", (10, 100), 1000),
    ("SAM", (100,), 100),
]
# Declared wider than the data that fills them, exactly as the Rust has carried since the C-derived
# generator: the extra slots are zero and no code reads them. Kept so this change is a provenance
# change and not a layout change.
PADDED = {"PTM": 50}
# `SAM` is unread by the model routines and is emitted anyway, because it ships with the
# published coefficient set and `mod.rs` keeps a compile-time reference to it saying so.
EMIT = ("PT", "PD", "PS", "PDL", "PTM", "PDM", "PTL", "PMA", "SAM", "PAVGM")


def data_chunks(text: str) -> dict[str, list[str]]:
    """Every `DATA <name>/ ... /` statement in the file, as name -> values.

    FORTRAN comment lines (`C` in column 1) sit between the chunks and would otherwise contribute
    digits -- `C         O2 DENSITY` and `C         N2 DENSITY` each carry a `2`.
    """
    body = "\n".join(line for line in text.splitlines() if not line[:1].upper().startswith("C"))
    out: dict[str, list[str]] = {}
    for m in re.finditer(r"DATA (\w+)/(.*?)/", body, re.DOTALL):
        out[m.group(1)] = [n.group(0).replace("E", "e") for n in _NUM.finditer(m.group(2))]
    return out


def parm7_order(text: str) -> list[str]:
    """The 64 chunk names of `COMMON/PARM7/` as `BLOCK DATA GTD7BK` declares them, in order.

    Read from the declaration rather than hardcoded, because the order *is* the storage layout: get
    it wrong and every table after the first is silently shifted. The `PTM`/`PDM`/`PAVGM` statements
    are interleaved among these in the source and belong to other blocks, which is why the stream is
    assembled by name instead of by position.
    """
    at = text.index("BLOCK DATA GTD7BK")
    decl = text[text.index("COMMON/PARM7/", at) : text.index("COMMON/LOWER7/", at)]
    return re.findall(r"(\w+)\(50\)", decl)


def emit_1d(name: str, n: int, values: list[str]) -> str:
    padded = values + ["0.0"] * (n - len(values))
    return f"#[rustfmt::skip]\npub static {name}: [f64; {n}] = [{', '.join(padded)}];\n"


def emit_2d(name: str, rows: int, cols: int, values: list[str]) -> str:
    body = "\n".join(
        f"    [{', '.join(values[r * cols : (r + 1) * cols])}]," for r in range(rows)
    )
    return f"#[rustfmt::skip]\npub static {name}: [[f64; {cols}]; {rows}] = [\n{body}\n];\n"


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    with open(src) as fh:
        text = fh.read()

    chunks = data_chunks(text)
    order = parm7_order(text)
    if len(order) != 64:
        raise SystemExit(f"PARM7: found {len(order)} chunk names in the declaration, expected 64")
    flat = [v for name in order for v in chunks[name]]
    total = sum(n for _, _, n in PARM7)
    if len(flat) != total:
        raise SystemExit(f"PARM7: read {len(flat)} values, the declaration holds {total}")
    tables: dict[str, tuple[tuple[int, ...], list[str]]] = {}
    at = 0
    for name, dims, n in PARM7:
        tables[name] = (dims, flat[at : at + n])
        at += n

    for name, dims in (("PTM", (10,)), ("PDM", (8, 10)), ("PAVGM", (10,))):
        values = chunks[name]
        want = dims[0] * (dims[1] if len(dims) > 1 else 1)
        if len(values) != want:
            raise SystemExit(f"{name}: read {len(values)} values, declared {want}")
        tables[name] = (dims, values)

    blocks = [
        "//! NRLMSISE-00 coefficient tables — machine-generated by `derivations/nrlmsise00_tables.py`\n"
        "//! from `BLOCK DATA GTD7BK` of the NRL FORTRAN `nrlmsise00_sub.for` (Picone, Hedin & Drob,\n"
        "//! Naval Research Laboratory; a US Government work, not subject to domestic copyright).\n"
        "//! Do not edit by hand; regenerate from the source. Values are the FORTRAN's own decimal\n"
        "//! digits with the exponent lowercased; `PTM` is declared wider than the data that fills it\n"
        "//! and the surplus slots are zero, which no code reads.\n"
    ]
    for name in EMIT:
        dims, values = tables[name]
        if len(dims) == 1:
            blocks.append(emit_1d(name, PADDED.get(name, dims[0]), values))
        else:
            blocks.append(emit_2d(name, dims[0], dims[1], values))
    with open(out, "w") as fh:
        fh.write("\n".join(blocks))
    for name in EMIT:
        print(f"  {name:6s} {tables[name][0]}")
    print(f"wrote {len(EMIT)} tables to {out}")


if __name__ == "__main__":
    main()
