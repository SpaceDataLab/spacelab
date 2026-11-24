#!/usr/bin/env python3
"""`tools/check-derivations.py --check` exits 1 on a planted violation of each kind it is for.

    uv run --no-project python -I tools/test_check_derivations.py

Rule 16 of `plans/18`: a checker is shown red on a planted defect before it is trusted. Each test
builds a throwaway tree that passes, plants one violation, and runs a copy of the checker inside it,
so the exit code under test is the one `just gates` reads.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable

TOOLS = pathlib.Path(__file__).resolve().parent
CHECKER = "check-derivations.py"
MODEL = "crates/x/src/model.rs"
PARSER = "crates/x/src/parse.rs"
GATE = "crates/x/tests/model_gate.rs"
LEDGER = "docs/verification.md"
TREE = {
    LEDGER: "\n".join(
        [
            "## 12 · Formula provenance census",
            "",
            "```text",
            f"{PARSER} | parser | no re-runnable check named in the module doc",
            "```",
            "",
        ]
    ),
    "crates/x/src/lib.rs": "//! A scratch crate; its identity is derivations/x.mac.\n",
    MODEL: "//! The model; its gate is tests/model_gate.rs.\n\npub fn f() {}\n",
    PARSER: "//! A parser.\n\npub fn parse() {}\n",
    GATE: "/// The defining equation holds (G2).\n#[test]\nfn defining_equation() {}\n",
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
    code, out = run(lambda root: edit(root, GATE, " (G2).", "."))
    want = f"names no check and is not in the ledger: {MODEL}"
    assert code == 1 and want in out, out


def test_a_new_module_that_names_no_check_fails() -> None:
    def plant(root: pathlib.Path) -> None:
        (root / "crates/x/src/drag.rs").write_text("//! Drag.\n", encoding="utf-8")

    code, out = run(plant)
    want = "names no check and is not in the ledger: crates/x/src/drag.rs"
    assert code == 1 and want in out, out


def test_a_ledger_entry_that_now_names_a_check_fails() -> None:
    code, out = run(
        lambda root: edit(root, PARSER, "A parser.", "A parser; derivations/p.mac.")
    )
    want = f"stale ledger entry: {PARSER} (now names a check"
    assert code == 1 and want in out, out


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
