#!/usr/bin/env python3
"""The density corpus with one satellite's densities from another reprocessing, rows unchanged (`plans/17` Ф30.G-2).

`tools/build-density-corpus.py` keeps each satellite-month's newest version. This copies its corpus to
`--out` and, for `--sat`, replaces each row's density by `--version`'s at the same instant where that
version holds the month and keeps the row; every other row, and every other satellite, is the corpus's
own. The rows are the same rows, so a field fitted on the variant is scored on the same test rows as the
shipped one and the folds stay paired. The other satellites' files are linked, not copied.

    uv run --project tools --locked python -I tools/corpus-variant.py --sat GC --version v02 --out corpus-gc-v02
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
SATELLITES = ("CH", "GA", "GB", "GC", "GO", "SA", "SB", "SC")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", type=pathlib.Path, default=corpus.ARCHIVE)
    parser.add_argument("--sat", required=True, choices=SATELLITES)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--out", required=True, help="a directory beside the corpus, by name"
    )
    args = parser.parse_args()
    out = corpus.OUT.parent / args.out
    if out.resolve() == corpus.OUT.resolve():
        sys.exit("--out must not be the corpus itself")
    out.mkdir(parents=True, exist_ok=True)
    for sat in SATELLITES:
        if sat != args.sat:
            link = out / f"corpus_{sat}.npz"
            link.unlink(missing_ok=True)
            link.symlink_to(corpus.OUT / f"corpus_{sat}.npz")
    base = dict(np.load(corpus.OUT / f"corpus_{args.sat}.npz"))
    archive = zipfile.ZipFile(args.archive)
    gps = corpus.gps_offsets()
    rho = base["rho"].copy()
    replaced = np.zeros(rho.size, dtype=bool)
    months = []
    for name in sorted(archive.namelist()):
        hit = corpus.MEMBER.match(name)
        if not (hit and hit["sat"] == args.sat and hit["version"] == args.version):
            continue
        if name.endswith(".txt") and name[:-4] + ".zip" in archive.namelist():
            continue
        data, _ = corpus.parse(corpus.payload(archive, name), args.sat, gps)
        at = np.searchsorted(base["t_ms"], data["t_ms"])
        inside = at < base["t_ms"].size
        same = np.zeros(at.size, dtype=bool)
        same[inside] = base["t_ms"][at[inside]] == data["t_ms"][inside]
        rho[at[same]] = data["rho"][same]
        replaced[at[same]] = True
        months.append(
            {
                "member": name,
                "rows": int(data["t_ms"].size),
                "matched": int(same.sum()),
            }
        )
    np.savez_compressed(out / f"corpus_{args.sat}.npz", **{**base, "rho": rho})
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "base": str(corpus.OUT.name),
                "sat": args.sat,
                "version": args.version,
                "rows": int(rho.size),
                "replaced": int(replaced.sum()),
                "months": months,
            },
            indent=1,
        )
        + "\n"
    )
    ratio = rho[replaced] / base["rho"][replaced]
    print(
        f"{args.sat}: {replaced.sum()} of {rho.size} rows from {args.version} in {len(months)} months; "
        f"median {args.version}/corpus {np.median(ratio):.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
