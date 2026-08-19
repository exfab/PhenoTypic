"""The vendored NGFF 0.5 schemas must be present, parseable, and unmodified.

Spec §7 forbids a conformance check that skips on a missing fixture, so the
absence of these files is a hard failure here rather than a skip downstream.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ngff" / "0.5"
SCHEMA_NAMES = ("image.schema", "label.schema", "ome.schema", "_version.schema")


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_is_present_and_parses(name: str) -> None:
    path = SCHEMA_DIR / name
    assert path.is_file(), f"vendored NGFF schema missing: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_matches_recorded_digest(name: str) -> None:
    """SOURCE.md pins each file's sha256; a mismatch means someone edited it."""
    recorded = dict(
        re.findall(
            r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|",
            (SCHEMA_DIR / "SOURCE.md").read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )
    actual = hashlib.sha256((SCHEMA_DIR / name).read_bytes()).hexdigest()
    assert recorded.get(name) == actual, (
        f"{name} does not match the digest recorded in SOURCE.md; the vendored "
        "upstream copy must stay byte-identical."
    )


def test_every_schema_is_rooted_at_the_attributes_object() -> None:
    """All three are ``{"required": ["ome"], "properties": {"ome": …}}``.

    This is what the conformance harness must validate against: the whole
    ``attributes`` mapping, NOT ``attributes["ome"]``. Passing the inner block
    fails with "'ome' is a required property" on every store.
    """
    for name in ("image.schema", "label.schema", "ome.schema"):
        payload = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
        assert payload["required"] == ["ome"], name
        assert list(payload["properties"]) == ["ome"], name
        assert payload["description"] == "The zarr.json attributes key", name


def test_ome_schema_requires_series() -> None:
    """Stricter than the prose — §7 calls this out explicitly."""
    payload = json.loads((SCHEMA_DIR / "ome.schema").read_text(encoding="utf-8"))
    assert payload["properties"]["ome"]["required"] == ["series", "version"]


def test_label_schema_requires_image_label() -> None:
    payload = json.loads((SCHEMA_DIR / "label.schema").read_text(encoding="utf-8"))
    assert payload["properties"]["ome"]["required"] == ["image-label", "version"]


def test_image_label_does_not_require_exhaustive_colors() -> None:
    """Pins the fact that re-graded P1: `colors` is OPTIONAL.

    `$defs/image-label` has no `required` list at all, so nothing obliges one
    entry per unique label value. The spec's §2.3 "MUST" is a PhenoTypic
    invention, not an NGFF rule.
    """
    payload = json.loads((SCHEMA_DIR / "label.schema").read_text(encoding="utf-8"))
    image_label = payload["$defs"]["image-label"]
    assert "required" not in image_label
    assert "colors" in image_label["properties"]


def test_every_remote_ref_is_vendored() -> None:
    """A remote $ref raises Unresolvable, which is not a ValidationError."""
    import re

    ids = {
        json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))["$id"]
        for name in SCHEMA_NAMES
    }
    for name in SCHEMA_NAMES:
        raw = (SCHEMA_DIR / name).read_text(encoding="utf-8")
        for ref in re.findall(r'"\$ref"\s*:\s*"(https?://[^"]+)"', raw):
            assert ref in ids, f"{name} references un-vendored {ref}"
