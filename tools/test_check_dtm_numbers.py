#!/usr/bin/env python3
"""The DTM-number check refuses each planted number, and passes the same text once it is marked.

uv run --no-project python -I tools/test_check_dtm_numbers.py

Every number planted here is made up: this file ships, and it prints no number computed from the model.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "check_dtm_numbers", HERE / "check-dtm-numbers.py"
)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["check_dtm_numbers"] = tool
spec.loader.exec_module(tool)
view = tool.view

# Spelled in parts: the public cut refuses a line of code that names a restricted module.
MODEL = "DTM" + "2020"
ROW = f"| `{MODEL.lower()}` | code | class `number-private` (D-32) | **restricted** |"
PRIVATE = f" <!-- {view.LINE} -->"
COLUMN = f" <!-- {view.COLUMN} -->"


def problems(files: dict[str, str], reviewed=None) -> list[str]:
    tree = {rel: view.public_lines(text, rel) for rel, text in files.items()}
    tree.setdefault(tool.REGISTER, view.public_lines(ROW))
    return tool.problems_of(tree, reviewed or [])


def test_a_planted_score_is_refused_and_passes_once_marked() -> None:
    score = f"Pooled, SDTM-1 reads 12.3 % against {MODEL}'s 45.6 %.\n"
    found = problems({"docs/x.md": score})
    assert len(found) == 1 and "(named)" in found[0], found
    assert problems({"docs/x.md": score.rstrip("\n") + PRIVATE + "\n"}) == []


def test_a_planted_ratio_is_refused_in_a_comment() -> None:
    ratio = f"//! {MODEL} reads 1.234 and 5.678 on the two satellites.\n"
    found = problems({"tests/x.rs": ratio})
    assert len(found) == 1 and "tests/x.rs:1:" in found[0], found
    assert problems({"tests/x.rs": ratio.rstrip("\n") + PRIVATE + "\n"}) == []


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


def test_a_name_with_no_number_passes() -> None:
    plain = (
        f"{MODEL}: §10.2, Ф14.Z, D-32, 2026-11-24, `{MODEL.lower()}/mod.rs`, Bruinsma & Boniface "
        f"(2021), over 24 storms.\n"
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


def test_the_register_row_must_record_the_class() -> None:
    bare = f"| `{MODEL.lower()}` | code | §1 | **restricted** |"
    found = problems({tool.REGISTER: bare})
    assert len(found) == len(tool.ROW_RECORDS), found
    assert len(problems({tool.REGISTER: "no row here"})) == 1


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
