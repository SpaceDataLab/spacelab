#!/usr/bin/env python3
"""A silent change to the frozen field, form or data fails `tools/check-sdtm-freeze.py`.

    python3 tools/test_check_sdtm_freeze.py

Builds a throwaway tree holding a baseline, injects each change, and runs the check itself. Standard
library only, like the check.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_sdtm_freeze", Path(__file__).with_name("check-sdtm-freeze.py")
)
assert spec is not None and spec.loader is not None
freeze = importlib.util.module_from_spec(spec)
sys.modules["check_sdtm_freeze"] = freeze
spec.loader.exec_module(freeze)

COEFFICIENTS = "crates/spacelab-propagation/src/sdtm/coefficients.rs"
DRIVER = "data/validation/density/SW-All.csv"


def build(root: Path) -> None:
    (root / "docs").mkdir()
    (root / COEFFICIENTS).parent.mkdir(parents=True)
    (root / DRIVER).parent.mkdir(parents=True)
    (root / COEFFICIENTS).write_text(
        "// fitted\nconst A: f64 = 1.5;\n", encoding="utf-8"
    )
    (root / DRIVER).write_text("2003-10-29,150\n", encoding="utf-8")
    baseline = {
        "code": {COEFFICIENTS: freeze.code_digest(root / COEFFICIENTS)},
        "data": {
            DRIVER: {
                "size": (root / DRIVER).stat().st_size,
                "sha256": freeze.file_digest(root / DRIVER),
            }
        },
    }
    (root / freeze.BASELINE).write_text(json.dumps(baseline), encoding="utf-8")


def problems_after(change) -> list[str]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        build(root)
        change(root)
        return freeze.check(root, need_data=False)


def edit(root: Path, name: str, old: str, new: str) -> None:
    text = (root / name).read_text(encoding="utf-8")
    assert old in text, old
    (root / name).write_text(text.replace(old, new), encoding="utf-8")


def test_an_untouched_tree_passes() -> None:
    assert problems_after(lambda _root: None) == []


def test_one_coefficient_fails() -> None:
    found = problems_after(lambda root: edit(root, COEFFICIENTS, "1.5", "1.50001"))
    assert len(found) == 1 and "changed" in found[0], found


def test_one_comment_passes() -> None:
    assert (
        problems_after(
            lambda root: edit(root, COEFFICIENTS, "// fitted", "// refitted")
        )
        == []
    )


def test_one_driver_row_fails_at_the_same_size() -> None:
    found = problems_after(lambda root: edit(root, DRIVER, "150", "151"))
    assert len(found) == 1 and "changed" in found[0], found


def test_a_longer_driver_file_fails() -> None:
    found = problems_after(lambda root: edit(root, DRIVER, "150", "150.5"))
    assert len(found) == 1 and "size" in found[0], found


def test_absent_data_passes_unless_it_is_required() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        build(root)
        (root / DRIVER).unlink()
        assert freeze.check(root, need_data=False) == []
        assert len(freeze.check(root, need_data=True)) == 1


def test_a_missing_coefficient_file_fails() -> None:
    found = problems_after(lambda root: (root / COEFFICIENTS).unlink())
    assert len(found) == 1 and "gone" in found[0], found


def test_the_command_exits_1_on_a_changed_coefficient() -> None:
    """Rule 16 of `plans/18`: the exit code `just gates` reads, green and then red."""

    def gate(root: Path) -> subprocess.CompletedProcess[str]:
        script = Path(__file__).with_name("check-sdtm-freeze.py")
        return subprocess.run(
            [sys.executable, "-I", str(script), "--root", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        build(root)
        clean = gate(root)
        edit(root, COEFFICIENTS, "1.5", "1.50001")
        planted = gate(root)
    assert clean.returncode == 0, clean.stdout
    assert planted.returncode == 1 and "changed" in planted.stdout, planted.stdout


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
