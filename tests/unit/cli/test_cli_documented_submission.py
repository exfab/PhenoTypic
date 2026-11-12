"""The public submission instructions describe the supported CLI contract."""

from __future__ import annotations

from pathlib import Path
import re
import shlex

import pytest


_DOCS = Path(__file__).resolve().parents[3] / "docs" / "source"
_COMMAND = re.compile(
    r"(?m)^[ \t]*(?:\$[ \t]*)?(?:uv[ \t]+run[ \t]+)?"
    r"(?:python(?:\d+(?:\.\d+)?)?[ \t]+-m[ \t]+phenotypic|phenotypic)"
    r"(?=[ \t]|$)[^\n]*"
)
_OPTION_HEADING = re.compile(r"(?m)^[ \t]*`{1,2}--wait`{1,2}[ \t]*$")


def _obsolete_wait_references(text: str) -> list[str]:
    """Find executable old commands and option declarations, excluding prose."""
    joined = re.sub(r"\\\r?\n[ \t]*", " ", text)
    commands = [match.group().strip() for match in _COMMAND.finditer(joined)]
    found = [
        command
        for command in commands
        if any(
            argument.split("=", 1)[0] == "--wait"
            for argument in shlex.split(command, comments=True)
        )
    ]
    found.extend(match.group().strip() for match in _OPTION_HEADING.finditer(text))
    return found


@pytest.mark.parametrize(
    "text",
    [
        "python -m phenotypic --slurm time=30 --wait",
        "uv run python -m phenotypic \\\n  --mode migrate \\\n  --wait",
        "    phenotypic --input /plates --wait",
        "``--wait``\n   Wait for every job to finish.",
    ],
)
def test_the_guard_recognizes_old_executable_instructions(text: str) -> None:
    assert _obsolete_wait_references(text)


def test_the_guard_allows_explanation_of_the_removed_option() -> None:
    text = (
        "Version 0.21.0 removed `--wait`; SLURM runs return after submission.\n"
        "```bash\nuv run python -m phenotypic --slurm time=30\n```\n"
    )
    assert _obsolete_wait_references(text) == []


def test_public_docs_do_not_advertise_removed_wait_option() -> None:
    paths = sorted([*_DOCS.rglob("*.md"), *_DOCS.rglob("*.rst")])
    assert paths
    obsolete = [
        f"{path.relative_to(_DOCS)}: {reference}"
        for path in paths
        for reference in _obsolete_wait_references(path.read_text(encoding="utf-8"))
    ]
    assert obsolete == []


def test_full_run_output_tree_does_not_promise_an_automatic_report() -> None:
    text = (_DOCS / "tutorials/pages/cli_modes.md").read_text(encoding="utf-8")
    trees = [
        block
        for block in re.findall(r"```text\n(.*?)```", text, flags=re.DOTALL)
        if "deliverables/" in block
    ]
    assert trees
    assert all("processing_report.html" not in tree for tree in trees)
