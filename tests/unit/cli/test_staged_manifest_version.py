"""The staged manifest's version moves with its entry shape.

``StagedManifestEntry.reference_digest`` is a key a version-3 reader does not
know: it would construct ``StagedManifestEntry(**entry)`` and die with an
unexpected-keyword ``TypeError``. Writing version 4 makes that reader stop at
its own version check instead, with the "Unsupported staged manifest version"
message, while this reader still loads versions 2 and 3 (entries without the
key load with ``reference_digest=None``).
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from phenotypic._cli._cli_staged_orchestration import (
    StagedManifestEntry,
    load_staged_manifest,
    write_staged_manifest,
)


def _entry(**overrides) -> StagedManifestEntry:
    fields = {
        "dataset": "plate1",
        "image_name": "t01.tiff",
        "stem": "t01",
        "input_path": "/in/plate1/t01.tiff",
        "work_id": "w",
        "relative_image_path": "plate1/t01.tiff",
        "attempt_id": "a",
        "reference_digest": "d" * 64,
    }
    fields.update(overrides)
    return StagedManifestEntry(**fields)


def test_a_manifest_carrying_the_digest_key_is_written_as_version_4(tmp_path):
    path = write_staged_manifest(tmp_path / "manifest.json", [_entry()])

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == 4
    assert raw["images"][0]["reference_digest"] == "d" * 64


def test_a_version_3_reader_refuses_what_this_writer_writes(tmp_path):
    """The check a version-3 reader runs, verbatim, rejects the new file."""
    path = write_staged_manifest(tmp_path / "manifest.json", [_entry()])

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw.get("version") not in (2, 3)


def test_a_version_4_manifest_round_trips(tmp_path):
    entries = [_entry(), _entry(image_name="t02.tiff", stem="t02")]
    path = write_staged_manifest(tmp_path / "manifest.json", entries)

    assert load_staged_manifest(path) == entries


@pytest.mark.parametrize("version", [2, 3])
def test_older_manifests_still_load_without_the_digest(tmp_path, version):
    path = tmp_path / "manifest.json"
    entry = asdict(_entry())
    del entry["reference_digest"]
    path.write_text(
        json.dumps({"version": version, "images": [entry]}), encoding="utf-8"
    )

    (loaded,) = load_staged_manifest(path)
    assert loaded.reference_digest is None


@pytest.mark.parametrize("version", [1, 5, None])
def test_an_unknown_version_is_refused_cleanly(tmp_path, version):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"version": version, "images": []}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unsupported staged manifest version"):
        load_staged_manifest(path)
