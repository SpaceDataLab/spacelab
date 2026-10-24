#!/usr/bin/env python3
"""Every model module names the check that makes its formulas a measurement, or is in the ledger.

`docs/verification.md` publishes what was measured. This script asks the complementary question of
each module under `crates/*/src`: does its *own* documentation name the thing that would catch a
wrong formula — a `derivations/*.mac` identity, a `spec_seals` / `table_seals` seal, a verification
row (`verification.md` §4.13), or an oracle test by path? A module that names none is not wrong; it
is unverified by anything a reader can re-run from the module itself, and the ledger in
`docs/verification.md` (§12)
records each such module with a category, so that the number can only go down deliberately.

Three failure modes, all mechanical:

  * a module that names no check and is not in the ledger — the debt grew silently;
  * a ledger entry whose module now names a check — the ledger is stale and the count is wrong;
  * a ledger entry whose file no longer exists.

The bridges (`spacelab-capi`, `spacelab-python`) are skipped: their claim is parity with the crates,
measured in §6 and §7, not a formula of their own; so is `spacelab-python-harness`, which only loads
the second. `spacelab-gpu` is skipped because its lanes are
hardware-gated and its verdicts sit in §2.

    uv run python tools/check-derivations.py            # the census, by category
    uv run python tools/check-derivations.py --check    # fail on the three cases above
    uv run python tools/check-derivations.py --ledger   # a ledger block for every uncovered module
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER_DOC = ROOT / "docs" / "verification.md"
MARKER = "## 12 · Formula provenance census"
SKIP_CRATES = {"spacelab-capi", "spacelab-python", "spacelab-python-harness", "spacelab-gpu"}
CATEGORIES = {"physics", "parser", "generated", "infra"}

# What counts as naming a re-runnable check. A verification row counts when the file is named with
# it: a bare `§4.13` is also how a module cites a textbook or a standard, and on 2026-10-24 28 modules
# were passing on Curtis, Vallado, Danielson and CCSDS section numbers alone. A `tests/...rs` path is
# an oracle test; the rest are the derivation and seal scripts by file name.
NAMED = re.compile(
    r"derivations/[\w./-]+\.(?:mac|py|m|gp|input|sh)"
    r"|\b(?:spec_seals|table_seals)\b"
    r"|verify\.sh"
    r"|verification\.md"
    r"|tests/[\w/]+\.rs"
)

# Guesses for `--ledger` only; the committed ledger is edited by hand.
PARSER_NAMES = {
    "finals", "rinex", "rinex_obs", "sp3", "sinex", "sinex_bias", "rtcm", "ntrip", "crd", "blq",
    "dhf", "omm", "tle", "cdm", "wind_data", "density_data", "space_weather", "vmf_grid", "output",
    "onnx", "ccir", "modip",
}
GENERATED_NAMES = {"cio_tables", "nut_tables", "tdb_tables", "rkn1210_consts", "coefficients", "leap_seconds"}
INFRA_NAMES = {
    "lib", "dual", "hyperdual", "scalar", "calendar", "linalg", "time", "combine", "detectors",
    "parallel", "latency", "converter", "bridge", "math", "types", "driver", "tolerance", "forces",
    "satellite", "graph", "model", "nn", "mod",
}


def module_doc(path: pathlib.Path) -> str:
    """The `//!` block at the top of the file, as one string; stops at the first item."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//!"):
            lines.append(stripped[3:].strip())
        elif stripped.startswith("//") or not stripped:
            continue
        else:
            break
    return " ".join(lines)


def modules() -> list[str]:
    out = []
    for path in sorted(ROOT.glob("crates/*/src/**/*.rs")):
        if path.parts[len(ROOT.parts) + 1] in SKIP_CRATES:
            continue
        out.append(str(path.relative_to(ROOT)))
    return out


def uncovered() -> list[str]:
    return [m for m in modules() if not NAMED.search(module_doc(ROOT / m))]


def ledger() -> dict[str, tuple[str, str]]:
    text = LEDGER_DOC.read_text(encoding="utf-8")
    if MARKER not in text:
        raise SystemExit(f"{LEDGER_DOC.name} has no '{MARKER}' section")
    body = text.split(MARKER, 1)[1].split("```")[1]
    out: dict[str, tuple[str, str]] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("text"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3 or parts[1] not in CATEGORIES:
            raise SystemExit(f"malformed ledger line: {line}")
        out[parts[0]] = (parts[1], parts[2])
    return out


def guess(path: str) -> str:
    stem = pathlib.PurePosixPath(path).stem
    if stem in PARSER_NAMES:
        return "parser"
    if stem in GENERATED_NAMES:
        return "generated"
    if stem in INFRA_NAMES:
        return "infra"
    return "physics"


def main() -> int:
    mods = modules()
    missing = uncovered()
    if "--ledger" in sys.argv:
        for m in missing:
            print(f"{m} | {guess(m)} | no re-runnable check named in the module doc")
        return 0

    book = ledger() if MARKER in LEDGER_DOC.read_text(encoding="utf-8") else {}
    by_cat = {c: sorted(m for m, (cat, _) in book.items() if cat == c) for c in CATEGORIES}
    print(f"{len(mods)} modules; {len(mods) - len(missing)} name a re-runnable check in their own doc")
    for c in ("physics", "parser", "generated", "infra"):
        print(f"  ledger {c:9s} {len(by_cat[c])}")
    if "--check" not in sys.argv:
        return 0

    new = [m for m in missing if m not in book]
    stale = sorted(m for m in book if m not in missing)
    for m in new:
        print(f"names no check and is not in the ledger: {m}")
    for m in stale:
        why = "file is gone" if not (ROOT / m).exists() else "now names a check -- remove it from the ledger"
        print(f"stale ledger entry: {m} ({why})")
    if new or stale:
        print(
            "\nThe ledger in docs/verification.md §12 is the list of modules whose formulas nothing "
            "named in the module re-runs. Name the derivation, seal, row or oracle test in the "
            "module doc, or add the module to the ledger with a category -- but not silently."
        )
        return 1
    print(f"ledger consistent: {len(book)} entries, {len(by_cat['physics'])} of them physics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
