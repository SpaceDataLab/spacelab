"""Write the embedded geopotential in NEPTUNE's own EGM2008 file format.

NEPTUNE ships EGM2008 (tide-free) and this workspace embeds XGM2019e (zero-tide). Comparing the two
propagators on their own fields would measure the two gravity solutions -- and the zero-tide/tide-free
offset in C20 alone is 9e-6 of the term that dominates a LEO arc. So the same move as the lunar case:
*our* coefficients are pushed into NEPTUNE under the filename its source fixes
(`src/gravity.f90`, `dataFileName(MODEL_EGM2008) = "egm2008.dat"`), and the residual is then two
implementations of one field.

The coefficients are read out of the generated Rust table rather than re-fetched from ICGEM, so what
NEPTUNE integrates is bit-for-bit what `GeoPotential::embedded` integrates -- a re-fetch could drift
from the committed table without anything noticing.

The row layout is NEPTUNE's, not ICGEM's: its reader is a *formatted* read,
`(a1,2(i5),2(e25.15e2),2(e20.10e2))`, whose fields are offset one column from the layout the file it
ships actually uses. That works only because every numeric field is right-justified with a leading
blank, which the widths below preserve; `--check` re-reads the result through the Fortran field
boundaries to prove it.

Run: python3 tools/neptune-oracle/make_gravity_overlay.py [degree] [--check]
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = ROOT / "crates/spacelab-propagation/src/geopotential_coeffs.rs"
DEST = ROOT / "tools/neptune-oracle/overlays/egm2008-xgm2019e/egm2008.dat"

ROW = re.compile(r"^\s*\((\d+),\s*(\d+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)\),\s*$")
CONST = re.compile(r"^pub const (MODEL_GM|MODEL_RE|MODEL_NAME): [^=]+= (.+);$")

# Fortran field boundaries of `(a1,2(i5),2(e25.15e2),2(e20.10e2))`, as 0-based half-open slices.
FIELDS = {"n": (1, 6), "m": (6, 11), "c": (11, 36), "s": (36, 61)}


def read_table() -> tuple[dict[str, str], list[tuple[int, int, float, float]]]:
    meta: dict[str, str] = {}
    coefficients = []
    for line in TABLE.read_text().splitlines():
        constant = CONST.match(line)
        if constant:
            meta[constant.group(1)] = constant.group(2).strip('"')
            continue
        row = ROW.match(line)
        if row:
            coefficients.append(
                (
                    int(row.group(1)),
                    int(row.group(2)),
                    float(row.group(3)),
                    float(row.group(4)),
                )
            )
    return meta, coefficients


def render(
    meta: dict[str, str], coefficients: list[tuple[int, int, float, float]], degree: int
) -> str:
    # "radius" is the second keyword NEPTUNE's header scan looks for, and it takes the *first* line
    # containing it -- so the word must not appear anywhere above its own line.
    head = [
        f"{meta['MODEL_NAME']} in NEPTUNE's EGM2008 file layout.",
        "",
        "product_type                  gravity_field",
        f"modelname                     {meta['MODEL_NAME']}",
        f"earth_gravity_constant        {float(meta['MODEL_GM']):.10E}",
        f"radius                        {float(meta['MODEL_RE']):.10E}",
        f"max_degree                    {degree}",
        "norm                          fully_normalized",
        "tide_system                   zero_tide",
        "",
        "    n    m            Cnm                     Snm                   sigmaCnm          sigmaSnm",
        "end_of_head " + "=" * 89,
    ]
    rows = [
        f"{n:5d}{m:5d}{c:25.15E}{s:25.15E}{0.0:20.10E}{0.0:20.10E}"
        for n, m, c, s in coefficients
        if n <= degree
    ]
    return "\n".join(head + rows) + "\n"


def check(
    text: str, coefficients: list[tuple[int, int, float, float]], degree: int
) -> int:
    """Re-read the emitted rows through NEPTUNE's field boundaries, not through `split()`."""
    want = [(n, m, c, s) for n, m, c, s in coefficients if n <= degree]
    body = text.split("end_of_head")[1].splitlines()[1:]
    if len(body) != len(want):
        print(f"FAIL: {len(body)} rows emitted, {len(want)} expected")
        return 1
    for line, (n, m, c, s) in zip(body, want, strict=True):
        got = {
            key: line[start:stop].replace("D", "E")
            for key, (start, stop) in FIELDS.items()
        }
        if (int(got["n"]), int(got["m"])) != (n, m):
            print(
                f"FAIL: degree/order read back as {got['n']!r},{got['m']!r}, wanted {n},{m}"
            )
            return 1
        if float(got["c"]) != c or float(got["s"]) != s:
            print(f"FAIL: ({n},{m}) read back as {got['c']!r},{got['s']!r}")
            return 1
    print(f"check ok: {len(want)} rows round-trip through the Fortran field boundaries")
    return 0


def main() -> int:
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    degree = int(arguments[0]) if arguments else 70
    meta, coefficients = read_table()
    text = render(meta, coefficients, degree)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(text)
    print(f"wrote {DEST}  ({meta['MODEL_NAME']} to degree {degree})")
    if "--check" in sys.argv[1:]:
        return check(text, coefficients, degree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
