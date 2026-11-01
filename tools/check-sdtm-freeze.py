#!/usr/bin/env python3
"""The SDTM-1 baseline is frozen: a change to its field, its form or its data does not pass silently.

`plans/17` D-30 and Ф30.A. Every number SDTM-1 quotes so far was read on the folds that chose it
(`docs/sdtm-selection-ledger.md`), so a test can only be run against a field that has stopped moving.
`docs/sdtm1-baseline.json` names that field: the code of the coefficient tables and the evaluator, with
comment lines ignored so that a sentence can still be corrected, and the corpus and driver files it was
fitted on, which are git-ignored and are checked where they exist.

    python3 tools/check-sdtm-freeze.py          # the code, and any data file present
    python3 tools/check-sdtm-freeze.py --data   # and every data file must be present
    python3 tools/check-sdtm-freeze.py --write  # a new baseline: a decision, entered in the ledger

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

BASELINE = "docs/sdtm1-baseline.json"
CODE = (
    "crates/spacelab-propagation/src/sdtm/coefficients.rs",
    "crates/spacelab-propagation/src/sdtm/solar_wind_coefficients.rs",
    "crates/spacelab-propagation/src/sdtm/mod.rs",
    "crates/spacelab-propagation/src/sdtm/atmosphere.rs",
)
DATA = (
    "data/validation/tudelft-v02/corpus/manifest.json",
    "data/validation/tudelft-v02/corpus/corpus_CH.npz",
    "data/validation/tudelft-v02/corpus/corpus_GA.npz",
    "data/validation/tudelft-v02/corpus/corpus_GB.npz",
    "data/validation/tudelft-v02/corpus/corpus_GC.npz",
    "data/validation/tudelft-v02/corpus/corpus_GO.npz",
    "data/validation/tudelft-v02/corpus/corpus_SA.npz",
    "data/validation/tudelft-v02/corpus/corpus_SB.npz",
    "data/validation/tudelft-v02/corpus/corpus_SC.npz",
    "data/validation/density/SW-All.csv",
    "data/validation/density/Hp60_ap60_complete_series.txt",
    "data/validation/density/omni2_sw_hourly.txt",
)
EMIT = {
    "default": "tools/fit-sdtm-corpus.py --fit-sigma --cooling --excess-saturation fit "
    "--storm-gain input-power --caps storm-log1-lin --emit",
    "solar_wind": "tools/fit-sdtm-corpus.py --fit-sigma --cooling --ap60 --solar-wind fit "
    "--excess-saturation fit --storm-gain input-power --caps storm-log1-lin --emit",
}


def code_digest(path: pathlib.Path) -> str:
    """sha256 of the lines that are not whole-line comments."""
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("//")
    ]
    return hashlib.sha256("\n".join(kept).encode()).hexdigest()


def file_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def check(root: pathlib.Path, need_data: bool) -> list[str]:
    baseline = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    problems = []
    for name, expected in baseline["code"].items():
        path = root / name
        if not path.exists():
            problems.append(f"{name}: gone")
        elif code_digest(path) != expected:
            problems.append(f"{name}: changed since the baseline")
    for name, expected in baseline["data"].items():
        path = root / name
        if not path.exists():
            if need_data:
                problems.append(f"{name}: absent")
            continue
        if path.stat().st_size != expected["size"]:
            problems.append(f"{name}: a different size from the baseline")
        elif file_digest(path) != expected["sha256"]:
            problems.append(f"{name}: changed since the baseline")
    return problems


def write(root: pathlib.Path) -> None:
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", CODE[0], CODE[1]],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    baseline = {
        "last_emit_commit": commit,
        "emit": EMIT,
        "code": {name: code_digest(root / name) for name in CODE},
        "data": {
            name: {
                "size": (root / name).stat().st_size,
                "sha256": file_digest(root / name),
            }
            for name in DATA
        },
    }
    (root / BASELINE).write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--root", type=pathlib.Path, default=pathlib.Path(__file__).parent.parent
    )
    args = parser.parse_args()
    if args.write:
        write(args.root)
        return 0
    problems = check(args.root, args.data)
    for problem in problems:
        print(problem)
    print(f"SDTM-1 baseline: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
