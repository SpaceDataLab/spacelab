#!/usr/bin/env python3
"""`tools/check-comparators.py --check` exits 1 on a planted violation of each kind it is for.

    uv run --no-project python -I tools/test_check_comparators.py

Rule 16 of `plans/18`: a checker is shown red on a planted defect before it is trusted. Each test
builds a throwaway repository that passes, plants one violation, and runs a copy of the checker
inside it, so the exit code under test is the one `just gates` reads.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable

TOOLS = pathlib.Path(__file__).resolve().parent
CHECKER = "check-comparators.py"
TESTS = "crates/x/tests/agree.rs"
LEDGER = "docs/verification.md"
TREE = {
    LEDGER: "\n".join(
        [
            "### 12.1 · Comparator-decided tests",
            "",
            "```text",
            f"{TESTS}::matches_orekit | name:orekit",
            "```",
            "",
        ]
    ),
    TESTS: "\n".join(
        [
            "#[test]",
            "fn matches_orekit() {}",
            "",
            "/// The defining equation holds at every tabulated point (G2).",
            "#[test]",
            "fn reads_the_table() {",
            '    let _ = include_str!("fixtures/sofa_values.txt");',
            "}",
            "",
        ]
    ),
    "crates/x/tests/fixtures/sofa_values.txt": "1.0\n",
}


def run(plant: Callable[[pathlib.Path], None]) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for name, text in TREE.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text, encoding="utf-8")
        (root / "tools").mkdir()
        shutil.copy(TOOLS / CHECKER, root / "tools" / CHECKER)
        plant(root)
        # The checker reads the test files and fixtures through `git ls-files`.
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        done = subprocess.run(
            [sys.executable, "-I", str(root / "tools" / CHECKER), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        return done.returncode, done.stdout + done.stderr


def edit(root: pathlib.Path, name: str, old: str, new: str) -> None:
    text = (root / name).read_text(encoding="utf-8")
    assert old in text, old
    (root / name).write_text(text.replace(old, new, 1), encoding="utf-8")


def test_the_untouched_tree_passes() -> None:
    code, out = run(lambda _root: None)
    assert code == 0, out


def test_a_gate_tag_removed_fails() -> None:
    # Without its tag, a test that reads another code's fixture is that code's to decide.
    code, out = run(lambda root: edit(root, TESTS, " (G2).", "."))
    want = f"comparator-decided and not in the ledger: {TESTS}::reads_the_table"
    assert code == 1 and want in out, out


def test_a_new_test_named_after_another_code_fails() -> None:
    code, out = run(
        lambda root: edit(
            root,
            TESTS,
            "fn matches_orekit() {}",
            "fn matches_orekit() {}\n\n#[test]\nfn sofa_agrees() {}",
        )
    )
    want = f"comparator-decided and not in the ledger: {TESTS}::sofa_agrees"
    assert code == 1 and want in out, out


def test_a_ledger_entry_whose_test_is_gone_fails() -> None:
    code, out = run(lambda root: edit(root, TESTS, "matches_orekit", "matches"))
    want = f"stale ledger entry: {TESTS}::matches_orekit (test is gone)"
    assert code == 1 and want in out, out


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
