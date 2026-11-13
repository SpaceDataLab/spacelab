#!/usr/bin/env python3
"""The comparator suite's discovery, row format and table, on hand-made inputs.

uv run --no-project python -I tools/test_compare.py
"""

from __future__ import annotations

import datetime
import importlib.util
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compare", HERE / "compare.py")
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["compare"] = tool
spec.loader.exec_module(tool)


def raises(call, *args) -> str:
    try:
        call(*args)
    except ValueError as e:
        return str(e)
    raise AssertionError(f"{call.__name__}{args} did not raise")


def test_a_row_reads_its_eight_fields_and_refuses_any_other_count() -> None:
    row = tool.parse_row(
        "comparison | Orekit | 13.1.7 | density | relative | 16 | 3.0e-15 | 1.0e-15"
    )
    assert row == tool.Row(
        "Orekit", "13.1.7", "density", "relative", 16, 3.0e-15, 1.0e-15
    )
    assert "8 fields" in raises(
        tool.parse_row, "comparison | Orekit | 13.1.7 | density | 16 | 3e-15 | 1e-15"
    )
    assert "8 fields" in raises(
        tool.parse_row, "Orekit | 13.1.7 | density | relative | 16 | 3e-15 | 1e-15 | x"
    )


def test_the_target_follows_the_file_kind_and_the_module_its_path() -> None:
    assert tool.target("crates/spacelab-fa/tests/delta_at_oracle.rs") == (
        "spacelab-fa",
        ["--test", "delta_at_oracle"],
        "",
    )
    assert tool.target("crates/p/src/harris_priester/mod.rs") == (
        "p",
        ["--lib"],
        "harris_priester",
    )
    assert tool.target("crates/p/src/dsst/srp.rs") == ("p", ["--lib"], "dsst::srp")
    assert tool.target("crates/p/src/lib.rs") == ("p", ["--lib"], "")
    assert "integration test" in raises(tool.target, "crates/p/tests/common/mod.rs")
    assert "not under" in raises(tool.target, "crates/p/examples/x.rs")


def test_discovery_finds_each_parked_fn_and_no_plain_ignore() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "a.rs").write_text(
            "#[test]\n"
            '#[ignore = "comparator: Orekit, not a verdict"]\n'
            "fn first() {}\n"
            "#[test]\n"
            '#[ignore = "slow"]\n'
            "fn slow() {}\n"
            "    #[test]\n"
            '    #[ignore = "comparator: NRL"]\n'
            '    #[expect(clippy::x, reason = "y")]\n'
            "    fn second() {}\n"
        )
        (root / "b.rs").write_text(
            '#[ignore = "comparator: dangling"]\nconst X: u8 = 0;\n'
        )
        assert tool.parked_tests(root, ["a.rs"]) == [
            tool.Parked("a.rs", "first"),
            tool.Parked("a.rs", "second"),
        ]
        assert "no fn after it" in raises(tool.parked_tests, root, ["b.rs"])


def test_the_listed_name_must_be_unique_inside_the_module() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        path = root / "crates/p/src/dsst/srp.rs"
        path.parent.mkdir(parents=True)
        path.write_text("mod tests {\n}\nmod more {\n}\n")
        test = tool.Parked("crates/p/src/dsst/srp.rs", "rates")
        listed = [
            "dsst::srp::tests::rates",
            "dsst::drag::tests::rates",
            "dsst::srp::tests::rates_two",
        ]
        assert (
            tool.full_name(root, test, listed, "dsst::srp") == "dsst::srp::tests::rates"
        )
        # Only the file's own inline modules may stand between its module and the name.
        assert "0 ignored tests match" in raises(
            tool.full_name, root, test, listed, "dsst"
        )
        assert "0 ignored tests match" in raises(
            tool.full_name, root, test, listed, "dsst::gaussian"
        )
        assert "2 ignored tests match" in raises(
            tool.full_name, root, test, [*listed, "dsst::srp::more::rates"], "dsst::srp"
        )


def test_the_table_lists_every_row_and_marks_a_restricted_one() -> None:
    loud = tool.Parked("crates/p/src/a.rs", "loud")
    silent = tool.Parked("crates/p/tests/b.rs", "silent")
    hidden = tool.Parked("crates/q/src/c.rs", "hidden")
    row = tool.Row("Orekit", "13.1.7", "density", "relative", 16, 3.0e-15, 1.0e-15)
    text = tool.table(
        [(loud, [row, row]), (silent, []), (hidden, [row])],
        datetime.date(2026, 11, 12),
        ["crates/q/"],
    )
    assert "Generated 2026-11-12." in text
    lines = [line for line in text.splitlines() if line.startswith("| Orekit")]
    shown = "| Orekit | 13.1.7 | density | relative | 16 | 3.000e-15 | 1.000e-15 |"
    assert lines == [
        f"{shown} `crates/p/src/a.rs::loud` |",
        f"{shown} `crates/p/src/a.rs::loud` |",
        f"{shown} `crates/q/src/c.rs::hidden` | {tool.PRIVATE}",
    ]
    assert text.rstrip().endswith("- `crates/p/tests/b.rs::silent`")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
