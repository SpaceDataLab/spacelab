#!/usr/bin/env python3
"""The ledger's `--rewrite` and `--since` decisions, on throwaway git repositories.

uv run --no-project python -I tools/test_mutants_ledger.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Callable

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "mutants_ledger", HERE / "mutants-ledger.py"
)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["mutants_ledger"] = tool
spec.loader.exec_module(tool)

CROSS = [
    "fn north() {",
    "    let n = [",
    "        u[1] * e[2] - u[2] * e[1],",
    "        u[2] * e[0] - u[0] * e[2],",
    "    ];",
    "}",
]


def in_repo(test: Callable[[pathlib.Path], None]) -> Callable[[], None]:
    """Run `test` with the tool pointed at a fresh repository holding a ledger and `src/w.rs`."""

    def run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / ".cargo").mkdir()
            (root / "src").mkdir()
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            tool.__dict__.update(ROOT=root, LEDGER=root / ".cargo" / "mutants.toml")
            tool.lines_at.cache_clear()
            test(root)

    return run


def commit(root: pathlib.Path, ledger: list[str], source: list[str]) -> None:
    write(root, ledger, source)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"],
        cwd=root,
        check=True,
    )
    tool.lines_at.cache_clear()


def write(root: pathlib.Path, ledger: list[str], source: list[str]) -> None:
    body = "".join(f'    "{e.replace(chr(92), chr(92) * 2)}",\n' for e in ledger)
    (root / ".cargo" / "mutants.toml").write_text(f"exclude_re = [\n{body}]\n")
    (root / "src" / "w.rs").write_text("\n".join(source) + "\n")
    tool.lines_at.cache_clear()


def test_the_column_is_measured_past_the_indent() -> None:
    entry = r"w\.rs:1:5: replace - with \+ in f"
    assert tool.named(["a - b"], entry) == tool.named(
        ["    a - b"], entry.replace(":5:", ":9:")
    )
    assert tool.named(["a - b"], r"w\.rs:2:5: replace - with \+ in f") is None


@in_repo
def test_a_shift_onto_a_twin_operator_is_reported(root: pathlib.Path) -> None:
    # The y component's `-` is the excluded mutant; one inserted line puts the x component's `-`,
    # same column and same shape, under the unchanged entry.
    entry = r"w\.rs:4:21: replace - with \+ in north"
    commit(root, [entry], CROSS)
    write(root, [entry], ["// one more line", *CROSS])
    every = [
        "src/w.rs:4:21: replace - with + in north",
        "src/w.rs:5:21: replace - with + in north",
    ]
    assert tool.transfers("HEAD", [entry], every) == 1


@in_repo
def test_a_reanchor_that_reuses_a_string_is_not_a_transfer(root: pathlib.Path) -> None:
    # Both components excluded; after the shift the y entry becomes the x entry's old string.
    x, y = (
        r"w\.rs:3:21: replace - with \+ in north",
        r"w\.rs:4:21: replace - with \+ in north",
    )
    commit(root, [x, y], CROSS)
    moved = [y, r"w\.rs:5:21: replace - with \+ in north"]
    write(root, moved, ["// one more line", *CROSS])
    every = [
        "src/w.rs:4:21: replace - with + in north",
        "src/w.rs:5:21: replace - with + in north",
    ]
    assert tool.transfers("HEAD", moved, every) == 0


@in_repo
def test_a_stale_anchor_follows_an_unchanged_function(root: pathlib.Path) -> None:
    entry = r"w\.rs:4:21: replace - with \+ in north"
    commit(root, [entry], CROSS)
    write(root, [entry], ["// one more line", *CROSS])
    mutant = "src/w.rs:5:21: replace - with + in north"
    tool.__dict__.update(function_spans=lambda path: {mutant: (2, 7)})
    new, why = tool.relocate(entry, [mutant], set())
    assert new == r"w\.rs:5:21: replace - with \+ in north", why


@in_repo
def test_a_stale_anchor_does_not_follow_a_changed_function(root: pathlib.Path) -> None:
    entry = r"w\.rs:4:21: replace - with \+ in north"
    commit(root, [entry], CROSS)
    changed = ["// one more line", *CROSS[:4], "        0.0,", *CROSS[5:]]
    write(root, [entry], changed)
    mutant = "src/w.rs:5:21: replace - with + in north"
    tool.__dict__.update(function_spans=lambda path: {mutant: (2, 7)})
    new, why = tool.relocate(entry, [mutant], set())
    assert new is None and "changed" in why, why


@in_repo
def test_a_stale_anchor_does_not_take_a_claimed_mutant(root: pathlib.Path) -> None:
    entry = r"w\.rs:4:21: replace - with \+ in north"
    commit(root, [entry], CROSS)
    write(root, [entry], ["// one more line", *CROSS])
    mutant = "src/w.rs:5:21: replace - with + in north"
    tool.__dict__.update(function_spans=lambda path: {mutant: (2, 7)})
    new, _ = tool.relocate(entry, [mutant], {mutant})
    assert new is None


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
