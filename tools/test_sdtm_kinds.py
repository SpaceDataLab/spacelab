#!/usr/bin/env python3
"""The two field kinds of `tools/evaluate-sdtm.py`, spheres and lifetimes, held to what they reproduce
(`plans/18` Ф30.O).

* **The regression row**, our output against our own earlier output (`plans/18` §6): the baseline, written
  by `lifetime_campaign --write-field` and read back as any exported field is, reproduces to the printed
  digits the `C_d` tables of `docs/verification.md` §10.39 and §10.43, §10.43's arms, and §10.37's paired
  row on the calibration part. The printed numbers are read from the document itself.
* **S2, a field exported twice gives the same scores**: the baseline compiled in, as the fit's `--emit`
  wrote it, and the baseline as a field file give the worker's output bit for bit, on ODERACS E's lifetimes
  and on the calibration part at 7 days.

Each check also runs on a planted defect and must fail on it (`plans/18` rule 16): a field whose boundary
scale is read as the logarithm the fit holds it as does not reproduce §10.39's Starshine 3; §10.37's row,
recomputed from the errors its run recorded, fails with the observed and persistence errors swapped; a field
file written to six digits does not give the same scores. Swapping the heating and nitric-oxide times is no
planted defect for the printed digits: they set how fast the storm response reaches its steady state, which
does not depend on them. The swap moved ODERACS E's lifetimes by up to 1.1 %, which S2 sees, and Starshine 3's
`C_d` not at its two printed decimals.

    uv run --project tools --locked python -I tools/test_sdtm_kinds.py          # S2, the defects: minutes
    uv run --project tools --locked python -I tools/test_sdtm_kinds.py --full   # and the whole row: hours
    uv run --project tools --locked python -I tools/test_sdtm_kinds.py --only spheres   # one part of it

Needs `data/validation/reentry/` as `tools/lifetime-campaign.py` and `fetch-reentry-elements.py` leave it,
and `cargo build -p spacelab-propagation --release --example lifetime_campaign` and
`cargo build -p spacelab --release --example cpf_mean_state`.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DOC = HERE.parent / "docs" / "verification.md"
spec = importlib.util.spec_from_file_location(
    "evaluate_sdtm", HERE / "evaluate-sdtm.py"
)
assert spec is not None and spec.loader is not None
harness = importlib.util.module_from_spec(spec)
sys.modules["evaluate_sdtm"] = harness
spec.loader.exec_module(harness)
lc = harness.lifetime_campaign()

STARSHINE_3 = 26929
ODERACS_E = 22994
SPHERE_HEADER = (
    "| sphere | lead, d | true, d | `C_d` SDTM-1 needs | `C_d` NRLMSISE-00 needs |"
)
ARM_HEADER = "| arm | years | spheres | SDTM-1 needs | NRLMSISE-00 needs |"


def section(number: str) -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.index(f"\n### {number} ")
    return text[start : text.index("\n### ", start + 1)]


def table(text: str, header: str) -> list[list[str]]:
    """The rows of the table whose header line starts with `header`, as cells."""
    lines = text[text.index(header) :].splitlines()
    rows = []
    for line in lines[2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def number(text: str) -> str:
    return text.replace(" ", "")


# The digits the tools print: `C_d` to three decimals, days to two. The document took them from there and
# rounded half up where it prints fewer, so 2.1647 is 2.165 and then 2.17, not 2.16.
CD_DIGITS = 3
DAY_DIGITS = 2


def shown(value: float, digits: int) -> Decimal:
    """`value` as a tool prints it."""
    return Decimal(f"{value:.{digits}f}")


def at(value: Decimal, printed: str) -> str:
    """`value` rounded half up to as many decimals as `printed` has."""
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    return f"{value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)}"


def printed_spheres(
    numbers: set[int] | None = None,
) -> list[tuple[str, int, str, str, str, str]]:
    """`(what, number, lead, printed true days, SDTM-1's printed C_d, NRLMSISE-00's)` of §10.39 and §10.43."""
    by_name = {name: n for n, name, *_ in lc.SPHERES}
    out = []
    for where in ("10.39", "10.43"):
        for name, leads, truths, sdtm, msis in table(section(where), SPHERE_HEADER):
            columns = [c.split(" / ") for c in (leads, truths, sdtm, msis)]
            if by_name[name] not in (numbers or {by_name[name]}):
                continue
            for lead, truth, a, b in zip(*columns, strict=True):
                out.append(
                    (f"§{where} {name}", by_name[name], int(number(lead)), truth, a, b)
                )
    return out


def sphere_mismatches(
    rows: list[dict], models: tuple[str, ...], numbers=None
) -> list[str]:
    found = {(r["number"], r["lead"]): r for r in rows}
    bad = []
    for what, n, lead, truth, *needs in printed_spheres(numbers):
        r = found.get((n, lead))
        if r is None:
            bad.append(f"{what} at {lead} d: not reproduced")
            continue
        pairs = [("true d", r["true_d"], DAY_DIGITS, number(truth))] + [
            (f"C_d {m}", r[m], CD_DIGITS, printed)
            for m, printed in zip(("sdtm", "msis"), needs, strict=True)
            if m in models
        ]
        for label, value, digits, printed in pairs:
            mine = "outside" if value is None else at(shown(value, digits), printed)
            if mine != printed:
                bad.append(
                    f"{what} at {lead} d, {label}: printed {printed}, reproduced {mine}"
                )
    return bad


def arm_mismatches(rows: list[dict]) -> list[str]:
    """§10.43's arms: median (range), or the range alone for one sphere; and its medians at leads of
    90 days or less."""
    bad = []
    # An arm's printed name begins with the harness's name before its years: "Starshine 3 (§10.39)".
    keys = {arm.split(",")[0]: numbers for arm, numbers in harness.SPHERE_ARMS.items()}
    for name, _years, _count, sdtm, msis, *_ in table(section("10.43"), ARM_HEADER):
        numbers = next(n for short, n in keys.items() if name.startswith(short))
        for model, printed in (("sdtm", sdtm), ("msis", msis)):
            values = [
                shown(r[model], CD_DIGITS)
                for r in rows
                if r["number"] in numbers and r[model] is not None
            ]
            parts = re.findall(r"\d\.\d+", printed)
            mine = [statistics.median(values), min(values), max(values)][
                3 - len(parts) :
            ]
            for value, part in zip(mine, parts, strict=True):
                if at(value, part) != part:
                    bad.append(
                        f"§10.43 arm {name}, {model}: printed {printed}, reproduced {at(value, part)}"
                    )
    sentence = re.search(
        r"SDTM-1 (\d\.\d+) to (\d\.\d+), NRLMSISE-00 (\d\.\d+) to (\d\.\d+)",
        section("10.43"),
    )
    assert sentence is not None, "§10.43's sentence on the two minima"
    short = {
        (arm, model): statistics.median(
            [
                shown(r[model], CD_DIGITS)
                for r in rows
                if r["number"] in keys[arm] and r["lead"] <= 90
            ]
        )
        for arm in ("ODERACS", "ANDE")
        for model in ("sdtm", "msis")
    }
    for (arm, model), printed in zip(
        (("ODERACS", "sdtm"), ("ANDE", "sdtm"), ("ODERACS", "msis"), ("ANDE", "msis")),
        sentence.groups(),
        strict=True,
    ):
        if at(short[arm, model], printed) != printed:
            bad.append(
                f"§10.43 {arm} at <= 90 d, {model}: printed {printed}, reproduced {at(short[arm, model], printed)}"
            )
    return bad


def calibration_mismatches(by: dict) -> list[str]:
    printed = next(
        row
        for row in table(section("10.37"), "| part | lead | objects |")
        if row[0] == "calibration"
    )
    line = next(
        row for row in lc.paired_lines(by, ("cal",)) if row.startswith("cal | all |")
    )
    mine = [cell.strip() for cell in line.split("|")][2:]
    return [
        f"§10.37 calibration row: printed {' | '.join(printed[2:])}, reproduced {' | '.join(mine)}"
    ] * (mine != printed[2:])


def field_file(directory: pathlib.Path, name: str, edit=None) -> pathlib.Path:
    """The shipped field as `--write-field` writes it, through `edit` if given."""
    text = subprocess.run(
        [str(lc.WORKER), "--write-field"], capture_output=True, text=True, check=True
    ).stdout
    path = directory / name
    path.write_text(text if edit is None else edit(text))
    return path


def boundary_as_log(text: str) -> str:
    """The boundary scale exponentiated once more, as a reader that took the exported scale for its log."""
    return re.sub(
        r"^boundary_scale (\S+)$",
        lambda m: f"boundary_scale {np.exp(float(m.group(1))):e}",
        text,
        flags=re.MULTILINE,
    )


def swap_observed(by: dict) -> dict:
    """SDTM-1's observed and persistence errors exchanged, every other cell kept."""
    return {
        key: {
            **v,
            "sdtm": {
                **v["sdtm"],
                "ln_observed": v["sdtm"]["ln_persistence"],
                "ln_persistence": v["sdtm"]["ln_observed"],
            },
        }
        if "sdtm" in v
        else v
        for key, v in by.items()
    }


def six_digits(text: str) -> str:
    return re.sub(r"-?\d\.\d+e-?\d+", lambda m: f"{float(m.group()):.5e}", text)


def check(name: str, bad: list[str], want_bad: bool) -> bool:
    ok = bool(bad) == want_bad
    verdict = "ok" if ok else "FAILED"
    print(f"{verdict}: {name}" + ("" if not bad else f" ({len(bad)} mismatches)"))
    for line in bad[:12]:
        print(f"    {line}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--full", action="store_true", help="and the whole regression row"
    )
    parser.add_argument(
        "--only",
        choices=("quick", "spheres", "lifetimes"),
        help="one part: S2 and the planted defects, or one half of the regression row",
    )
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    parts = (
        {args.only}
        if args.only
        else {"quick", "spheres", "lifetimes"}
        if args.full
        else {"quick"}
    )
    results = []
    with tempfile.TemporaryDirectory() as scratch:
        directory = pathlib.Path(scratch)
        baseline = field_file(directory, "baseline.field")
        boundary = field_file(directory, "boundary.field", boundary_as_log)
        rounded = field_file(directory, "rounded.field", six_digits)
        chosen = [s for s in lc.SPHERES if s[0] == ODERACS_E]
        if "quick" in parts:
            results += quick(args.processes, chosen, baseline, boundary, rounded)
        if "spheres" in parts:
            rows = harness.sphere_rows(
                lc,
                list(lc.SPHERES),
                ("sdtm", "msis"),
                args.processes,
                ("--field", str(baseline)),
            )
            results.append(
                check(
                    "regression, §10.39 and §10.43, every sphere and lead",
                    sphere_mismatches(rows, ("sdtm", "msis")),
                    False,
                )
            )
            results.append(
                check("regression, §10.43's arms", arm_mismatches(rows), False)
            )
        if "lifetimes" in parts:
            by = lc.calibration_scores(
                ("sdtm",), args.processes, ("--field", str(baseline))
            )
            recorded = lc.recorded_calibration()
            for key, v in by.items():
                if "msis" in recorded[key]:
                    v["msis"] = recorded[key]["msis"]
            results.append(
                check(
                    "regression, §10.37's calibration row",
                    calibration_mismatches(by),
                    False,
                )
            )
    print(f"{sum(results)} of {len(results)} checks hold")
    return 0 if all(results) else 1


def quick(processes: int, chosen, baseline, boundary, rounded) -> list[bool]:
    """S2 and the planted defects: minutes."""
    results = []
    # S2: compiled in, and as a file; the file at six digits must not pass.
    runs = {
        label: lc.sphere_needs(chosen, ("sdtm",), processes, extra)[1]
        for label, extra in (
            ("compiled", ()),
            ("file", ("--field", str(baseline))),
            ("rounded", ("--field", str(rounded))),
        )
    }
    lives = {
        label: lc.calibration_scores(("sdtm",), processes, extra, (7,))
        for label, extra in (("compiled", ()), ("file", ("--field", str(baseline))))
    }
    raw = ("b", "seconds_observed", "seconds_persistence")

    def differ(a: dict, b: dict) -> list[str]:
        return [
            f"{key}: {a.get(key)} against {b.get(key)}"
            for key in a.keys() | b.keys()
            if a.get(key) != b.get(key)
        ]

    results.append(
        check(
            "S2 spheres, the field compiled in and as a file",
            differ(runs["compiled"], runs["file"]),
            False,
        )
    )
    results.append(
        check(
            "S2 lifetimes at 7 d, the field compiled in and as a file",
            differ(
                {
                    k: tuple(v["sdtm"][f] for f in raw)
                    for k, v in lives["compiled"].items()
                    if "sdtm" in v
                },
                {
                    k: tuple(v["sdtm"][f] for f in raw)
                    for k, v in lives["file"].items()
                    if "sdtm" in v
                },
            ),
            False,
        )
    )
    results.append(
        check(
            "S2 planted: the file at six digits differs",
            differ(runs["compiled"], runs["rounded"]),
            True,
        )
    )

    # The regression row on one sphere, then its planted defect.
    three = {STARSHINE_3}
    for label, path, want_bad in (
        ("the baseline as a file", baseline, False),
        ("planted: the boundary scale read as its log", boundary, True),
    ):
        rows = harness.sphere_rows(
            lc,
            [s for s in lc.SPHERES if s[0] == STARSHINE_3],
            ("sdtm",),
            processes,
            ("--field", str(path)),
        )
        results.append(
            check(
                f"regression, §10.39's Starshine 3, {label}",
                sphere_mismatches(rows, ("sdtm",), three),
                want_bad,
            )
        )

    # §10.37's row from the errors its run recorded: the pairing and the bootstrap alone.
    recorded = lc.recorded_calibration()
    for label, by, want_bad in (
        ("from the recorded errors", recorded, False),
        ("planted: observed and persistence swapped", swap_observed(recorded), True),
    ):
        results.append(
            check(
                f"regression, §10.37's calibration row {label}",
                calibration_mismatches(by),
                want_bad,
            )
        )
    return results


if __name__ == "__main__":
    sys.exit(main())
