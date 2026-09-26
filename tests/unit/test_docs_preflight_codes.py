"""The run-preflight code table in the CLI tutorial matches the code set.

Spec ``2026-09-24-cli-preflight`` (review R37): a hand-written table drifts
silently. This fails when a ``FindingCode`` has no row, or a row names a code
that no longer exists.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from phenotypic._cli._cli_preflight import HINTS, FindingCode

PAGE = (
    Path(__file__).resolve().parents[2]
    / "docs" / "source" / "tutorials" / "pages" / "cli_batch_processing.md"
)


def _table_codes() -> list[str]:
    text = PAGE.read_text(encoding="utf-8")
    section = text.split("## Run Preflight Checks", 1)[1]
    return re.findall(r"^\| `(PF-[A-Z0-9-]+)` \|", section, flags=re.MULTILINE)


def test_every_finding_code_has_exactly_one_row() -> None:
    rows = _table_codes()

    assert len(rows) == len(set(rows)), "a code has two rows"
    assert set(rows) == set(get_args(FindingCode))


def test_every_finding_code_has_a_hint() -> None:
    assert set(HINTS) == set(get_args(FindingCode))
