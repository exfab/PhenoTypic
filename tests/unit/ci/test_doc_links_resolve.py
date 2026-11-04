"""Every ``:doc:`` role in the source names a page that exists under docs/source."""

from __future__ import annotations

import re
from pathlib import Path

# Match any target; absoluteness is asserted, not assumed.
_DOC = re.compile(r":doc:`(?:[^`<]*<)?([^`>]+)>?`")
_REPO = Path(__file__).resolve().parents[3]  # anchor on this file, not the cwd


def _targets():
    for path in (_REPO / "src/phenotypic").rglob("*.py"):
        for m in _DOC.finditer(path.read_text(encoding="utf-8")):
            yield path, m.group(1)


def _exists(target: str) -> bool:
    base = _REPO / "docs/source" / target.lstrip("/")
    return (
        any(base.with_suffix(s).exists() for s in (".md", ".rst", ".ipynb"))
        or (base / "index.md").exists()
        or (base / "index.rst").exists()
    )


def test_parser_reads_both_forms() -> None:
    assert _DOC.search(":doc:`/how_to/pages/tuning`").group(1) == "/how_to/pages/tuning"
    assert (
        _DOC.search(":doc:`Tuning </how_to/pages/tuning>`").group(1)
        == "/how_to/pages/tuning"
    )


def test_every_doc_link_in_src_is_absolute_and_resolves() -> None:
    found = list(_targets())
    assert found, "no :doc: links found; the scan is not reaching the source"
    relative = [f"{p}: {t}" for p, t in found if not t.startswith("/")]
    assert relative == []
    broken = [f"{p}: {t}" for p, t in found if not _exists(t)]
    assert broken == []
