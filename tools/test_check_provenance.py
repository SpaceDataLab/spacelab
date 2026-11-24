#!/usr/bin/env python3
"""`tools/check-provenance.py --check` exits 1 on a planted violation of each kind it is for.

    uv run --no-project python -I tools/test_check_provenance.py

Rule 16 of `plans/18`: a checker is shown red on a planted defect before it is trusted. Each test
builds a throwaway tree that passes, plants one violation, and runs a copy of the checker inside it,
so the exit code under test is the one `just gates` reads. Standard library only, like the checker.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable

TOOLS = pathlib.Path(__file__).resolve().parent
CHECKER = "check-provenance.py"
LIB = "crates/x/src/lib.rs"
PORT = "crates/x/src/port.rs"
REGISTER = "docs/provenance.md"
TREE = {
    REGISTER: "\n".join(
        [
            "## Not for public release",
            "",
            "```text",
            "exclude: crates/x/src/secret.rs",
            f"depends: {REGISTER}",
            "```",
            "",
            "## The covered set",
            "",
            "```text",
            PORT,
            "```",
            "",
        ]
    ),
    LIB: "//! A scratch crate.\n\npub mod port;\n",
    PORT: "//! A port of the reference implementation's `bar`.\n\npub fn bar() {}\n",
    "crates/x/src/secret.rs": "//! Restricted.\n\npub fn hidden() {}\n",
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


def test_a_register_line_removed_fails() -> None:
    code, out = run(lambda root: edit(root, REGISTER, f"{PORT}\n", ""))
    assert code == 1 and f"not in docs/provenance.md: {PORT}" in out, out


def test_a_register_line_whose_module_stopped_claiming_a_port_fails() -> None:
    code, out = run(lambda root: edit(root, PORT, "A port of the reference", "Our own"))
    assert code == 1 and f"no longer claims a port: {PORT}" in out, out


def test_a_new_file_naming_a_restricted_module_fails() -> None:
    code, out = run(lambda root: edit(root, LIB, "pub mod port;", "// see secret.rs"))
    assert (
        code == 1 and f"newly depends on a licence-restricted module: {LIB}" in out
    ), out


def test_a_recorded_dependency_that_is_gone_fails() -> None:
    code, out = run(
        lambda root: edit(
            root,
            REGISTER,
            f"depends: {REGISTER}",
            f"depends: {REGISTER}\ndepends: {LIB}",
        )
    )
    assert code == 1 and f"no longer does: {LIB}" in out, out


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
