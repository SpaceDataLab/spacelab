#!/usr/bin/env python3
"""Two reprocessings of one satellite's densities on the same instants (`plans/17` Ф30.G, F72).

TU Delft's archive holds some satellite-months at more than one version (`v02`, `v02b`, `v02c`);
`tools/build-density-corpus.py` keeps the newest. For every month of `--sat` that holds both `--old` and
`--new`, this prints the median of new/old over the rows both versions keep, and the median ratio and
difference in each tercile of the old density: a multiplicative change shows one ratio in every
tercile, an additive one the same difference. Rows are kept as the corpus keeps them: flag 0 and a
positive density.

    uv run --project tools --locked python -I tools/density-versions.py [--sat GC] [--old v02] [--new v02c]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import pathlib
import sys
import zipfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "build_density_corpus", HERE / "build-density-corpus.py"
)
assert spec is not None and spec.loader is not None
corpus = importlib.util.module_from_spec(spec)
sys.modules["build_density_corpus"] = corpus
spec.loader.exec_module(corpus)


def kept(archive: zipfile.ZipFile, name: str) -> dict[str, float]:
    """The month's kept rows, instant to density."""
    out = {}
    for line in corpus.payload(archive, name).decode().splitlines():
        cells = line.split()
        if not cells or line.startswith("#"):
            continue
        flagged = len(cells) > 10 and float(cells[10]) != 0.0
        rho = float(cells[8])
        if not flagged and rho > 0.0:
            out[cells[0] + " " + cells[1]] = rho
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", type=pathlib.Path, default=corpus.ARCHIVE)
    parser.add_argument("--sat", default="GC")
    parser.add_argument("--old", default="v02")
    parser.add_argument("--new", default="v02c")
    args = parser.parse_args()
    archive = zipfile.ZipFile(args.archive)
    months: dict[str, dict[str, str]] = {}
    for name in archive.namelist():
        hit = corpus.MEMBER.match(name)
        if hit and hit["sat"] == args.sat and hit["version"] in (args.old, args.new):
            # A version may sit twice, as `.zip` and `.txt`: the builder checks they are the same bytes.
            months.setdefault(f"{hit['year']}-{hit['month']}", {}).setdefault(
                hit["version"], name
            )
    both = sorted(m for m, v in months.items() if len(v) == 2)
    print(
        f"{args.sat}: {len(both)} of {len(months)} months hold {args.old} and {args.new}"
    )
    print(
        f"month | rows | median {args.new}/{args.old} | by tercile of {args.old}: "
        "median density, ratio, difference kg/m3"
    )
    medians = []
    for month in both:
        old, new = (kept(archive, months[month][v]) for v in (args.old, args.new))
        common = sorted(set(old) & set(new))
        a = np.array([old[k] for k in common])
        b = np.array([new[k] for k in common])
        medians.append(float(np.median(b / a)))
        edges = np.quantile(a, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        cells = []
        for low, high in itertools.pairwise(edges):
            m = (a >= low) & (a <= high)
            cells.append(
                f"{np.median(a[m]):.2e} {np.median(b[m] / a[m]):.4f} {np.median(b[m] - a[m]):+.2e}"
            )
        print(f"{month} | {a.size} | {medians[-1]:.4f} | " + " | ".join(cells))
    if medians:
        print(
            f"median of the monthly medians {np.median(medians):.4f}, "
            f"from {min(medians):.4f} to {max(medians):.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
