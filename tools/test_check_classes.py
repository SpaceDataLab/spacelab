#!/usr/bin/env python3
"""The class check refuses each planted defect, and passes the same text once it is fixed.

uv run --no-project python -I tools/test_check_classes.py

The register here is made up, and so is its `number-private` source, RIVAL-2031, and every number
planted beside it: this file ships, and it prints no number computed from a real source.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "check_classes", HERE / "check-classes.py"
)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["check_classes"] = tool
spec.loader.exec_module(tool)
view = tool.view

MODEL = "RIVAL-2031"
PRIVATE = f" <!-- {view.LINE} -->"
COLUMN = f" <!-- {view.COLUMN} -->"
# (key, verdict, class, Class cell, names)
RIVAL = (
    "rival",
    "restricted",
    "number-private",
    "`number-private` (D-99)",
    "RIVAL-?2031",
)
OPEN = ("open-data", "ship+attr", "ship", "`ship`", "")


def register(*rows: tuple[str, str, str, str, str]) -> str:
    prose = [
        "| Key | What it is | Terms, as read | Verdict | Class |",
        "|---|---|---|---|---|",
    ]
    block = []
    for key, verdict, kind, cell, names in rows:
        prose.append(f"| `{key}` | a source | its terms | **{verdict}** | {cell} |")
        line = f"{key} | {verdict} | read 2026-11-24 | hosts - | paths - | class {kind}"
        block.append(line + (f" | names {names}" if names else ""))
    return (
        f"{tool.DATA}\n\n"
        + "\n".join(prose)
        + "\n\n```text\n"
        + "\n".join(block)
        + "\n```\n"
    )


def problems(files: dict[str, str], reviewed=None, rows=(RIVAL, OPEN)) -> list[str]:
    tree = {rel: view.public_lines(text, rel) for rel, text in files.items()}
    tree.setdefault(tool.REGISTER, view.public_lines(register(*rows)))
    return tool.problems_of(tree, reviewed or [])


def test_a_planted_score_is_refused_and_passes_once_marked() -> None:
    score = f"Pooled, SDTM-1 reads 12.3 % against {MODEL}'s 45.6 %.\n"
    found = problems({"docs/x.md": score})
    assert len(found) == 1 and "(named)" in found[0] and "rival" in found[0], found
    assert problems({"docs/x.md": score.rstrip("\n") + PRIVATE + "\n"}) == []


def test_the_register_decides_what_is_refused() -> None:
    score = f"{MODEL} reads 45.6 %.\n"
    public = ("rival", "restricted", "number-public", "`number-public`", "")
    assert len(problems({"docs/x.md": score})) == 1
    assert problems({"docs/x.md": score}, rows=(public, OPEN)) == []


def test_a_planted_ratio_is_refused_in_a_comment_and_in_code() -> None:
    ratio = f"//! {MODEL} reads 1.234 and 5.678 on the two satellites.\n"
    found = problems({"tests/x.rs": ratio})
    assert len(found) == 1 and "tests/x.rs:1:" in found[0], found
    assert problems({"tests/x.rs": ratio.rstrip("\n") + PRIVATE + "\n"}) == []
    code = f'    assert!(mape < 45.6, "{MODEL}");\n'
    assert len(problems({"tests/x.rs": code})) == 1


def test_a_planted_table_column_is_refused_and_dropped_once_marked() -> None:
    table = (
        f"| bin | NRLMSISE-00 | {MODEL} |\n|---|---|---|\n| all | 51.7 % | 45.6 % |\n"
    )
    found = problems({"docs/x.md": table})
    assert len(found) == 1 and "(column)" in found[0], found
    marked = table.replace(f"| {MODEL} |", f"| {MODEL}{COLUMN} |")
    assert problems({"docs/x.md": marked}) == []
    assert view.public_lines(marked) == [
        "| bin | NRLMSISE-00 |",
        "|---|---|",
        "| all | 51.7 % |",
    ]


def test_a_marked_column_goes_from_a_comment_table_and_keeps_an_escaped_pipe() -> None:
    table = (
        f"/// | bin | `\\|ln\\|` | {MODEL}{COLUMN} |\n/// |---|---|---|\n"
        f"/// | all | 0.12 | 45.6 % |{PRIVATE}\n/// | one | 0.34 | 7.8 % |\n///\n/// | a | b |\n"
    )
    assert view.public_lines(table) == [
        "/// | bin | `\\|ln\\|` |",
        "/// |---|---|",
        None,
        "/// | one | 0.34 |",
        "///",
        "/// | a | b |",
    ]


def test_a_column_marker_outside_a_table_is_refused() -> None:
    try:
        view.public_lines(f"{MODEL} reads{COLUMN}\n", "docs/x.md")
    except SystemExit as e:
        assert "docs/x.md:1:" in str(e) and "outside a table row" in str(e), e
    else:
        raise AssertionError("a column marker outside a table passed")


def test_a_number_on_another_line_of_its_paragraph_is_refused() -> None:
    text = f"It is the most accurate model,\n{MODEL}, in the campaign\n(45.6 % against 25.5 %).\n"
    found = problems({"docs/x.md": text})
    assert len(found) == 1 and "docs/x.md:3:" in found[0], found
    assert "(paragraph)" in found[0], found
    split = f"{MODEL} is excluded.\n\nNRLMSISE-00 reads 51.7 %.\n"
    assert problems({"docs/x.md": split}) == []
    block = f"// {MODEL} reads\n// 45.6 %.\nfn f() {{}}\n"
    assert len(problems({"tests/x.rs": block})) == 1
    apart = f"// {MODEL} is excluded.\nfn f() {{}}\n// 45.6 %.\n"
    assert problems({"tests/x.rs": apart}) == []


def test_a_name_with_no_number_passes_and_its_own_digits_are_not_one() -> None:
    plain = (
        f"{MODEL}: §10.2, Ф14.Z, D-32, 2026-11-24, `rival/mod.rs`, Bruinsma & Boniface "
        f"(2021), over 24 storms; RIVAL2031 and rival-2031 too.\n"
    )
    assert problems({"docs/x.md": plain}) == []


def test_a_private_block_is_not_public() -> None:
    hidden = f"<!-- {view.BEGIN} -->\n{MODEL} reads 45.6 %.\n<!-- {view.END} -->\n"
    assert problems({"docs/x.md": hidden}) == []


def test_a_reviewed_hit_passes_and_a_stale_entry_is_refused() -> None:
    line = f"Zhu et al. find that {MODEL} gains 1–2 %.\n"
    reviewed = [("docs/x.md", "Zhu et al. find", "a published number, cited")]
    assert len(problems({"docs/x.md": line})) == 1
    assert problems({"docs/x.md": line}, reviewed) == []
    found = problems({"docs/x.md": "nothing here\n"}, reviewed)
    assert len(found) == 1 and "REVIEWED fragment matches no hit" in found[0], found


def test_every_row_has_a_class_its_verdict_allows_and_its_prose_says() -> None:
    def refused(row: tuple[str, str, str, str, str], why: str) -> None:
        found = problems({}, rows=(RIVAL, row))
        assert len(found) == 1 and why in found[0], (row, found)

    refused(("x", "restricted", "public", "`public`", ""), "is not one of")
    refused(("x", "ship", "-", "not read: no account", ""), "no class, under")
    refused(("x", "restricted", "-", "open: the user's call", ""), "no class, under")
    refused(("x", "restricted", "ship", "`ship`", ""), "class 'ship' under")
    refused(("x", "ship+attr", "internal", "`internal`", ""), "class 'internal' under")
    refused(("x", "restricted", "internal", "`ship`", ""), "does not say `internal`")
    refused(("x", "user-decision", "-", "`ship`", ""), "says no reason")
    refused(("x", "restricted", "number-private", "`number-private`", ""), "no names")
    refused(
        ("x", "restricted", "number-public", "`number-public`", "X1"), "names under"
    )
    refused(("x", "restricted", "number-private", "`number-private`", "(x"), "compile")
    waits = ("x", "user-decision", "-", "not read: the terms bind per account", "")
    assert problems({}, rows=(RIVAL, waits)) == []
    try:
        tool.class_problems(register(RIVAL).replace(" | class number-private", ""))
    except SystemExit as e:
        assert "malformed data-register row" in str(e), e
    else:
        raise AssertionError("a row with no class field passed")


def test_the_cut_s_refusal_reads_the_written_tree() -> None:
    plants = {
        "docs/x.md": f"{MODEL} reads 45.6 %.\n",
        "crates/x/src/lib.rs": f"//! {MODEL} reads 45.6 %.\npub fn f() {{}}\n",
        "crates/x/tests/t.rs": f'#[test]\nfn t() {{\n    assert!(1.0 < 45.6, "{MODEL}");\n}}\n',
    }
    with tempfile.TemporaryDirectory() as tmp:
        tree = pathlib.Path(tmp)
        (tree / "docs").mkdir()
        (tree / tool.REGISTER).write_text(register(RIVAL, OPEN), encoding="utf-8")
        (tree / "data.bin").write_bytes(f"{MODEL} 45.6\0".encode())
        assert tool.refused_in(tree) == []
        for rel, text in plants.items():
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_text(text, encoding="utf-8")
        found = tool.refused_in(tree)
    assert sorted(p.split(":")[0] for p in found) == sorted(plants), found


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
