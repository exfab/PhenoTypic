"""The staged manifest's version moves with its entry shape.

``StagedManifestEntry.reference_digest`` is a key a version-3 reader does not
know: it would construct ``StagedManifestEntry(**entry)`` and die with an
unexpected-keyword ``TypeError``. Writing version 4 makes that reader stop at
its own version check instead, with the "Unsupported staged manifest version"
message, while this reader still loads versions 2 and 3 (entries without the
key load with ``reference_digest=None``).

Version 5 (#282 P6 T1) adds the top-level ``"republish"`` flag, measure
mode's one republish carrier; version 4 and below read as forward.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from phenotypic._cli._cli_staged_orchestration import (
    StagedManifestEntry,
    load_staged_manifest,
    staged_manifest_republishes,
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


def test_a_manifest_carrying_the_digest_key_is_written_as_version_5(tmp_path):
    path = write_staged_manifest(tmp_path / "manifest.json", [_entry()])

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == 5 and raw["republish"] is False
    assert raw["images"][0]["reference_digest"] == "d" * 64


def test_a_version_3_reader_refuses_what_this_writer_writes(tmp_path):
    """The check a version-3 reader runs, verbatim, rejects the new file."""
    path = write_staged_manifest(tmp_path / "manifest.json", [_entry()])

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw.get("version") not in (2, 3)


def test_a_version_5_manifest_round_trips(tmp_path):
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


@pytest.mark.parametrize("version", [1, 6, None])
def test_an_unknown_version_is_refused_cleanly(tmp_path, version):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"version": version, "images": []}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unsupported staged manifest version"):
        load_staged_manifest(path)


def _write_v4_manifest(path, entries):
    """A manifest as the version-4 writer wrote it: no ``"republish"`` key."""
    path.write_text(
        json.dumps({"version": 4, "images": [asdict(entry) for entry in entries]}),
        encoding="utf-8",
    )
    return path


def test_v5_records_republish(tmp_path):
    path = write_staged_manifest(tmp_path / "m.json", [_entry()], republish=True)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 5
    assert staged_manifest_republishes(path) and len(load_staged_manifest(path)) == 1


def test_a_forward_v5_manifest_does_not_republish(tmp_path):
    assert not staged_manifest_republishes(write_staged_manifest(tmp_path / "m.json", [_entry()]))


def test_a_v4_manifest_reads_as_forward(tmp_path):
    path = _write_v4_manifest(tmp_path / "m.json", [_entry()])
    assert not staged_manifest_republishes(path)
    assert load_staged_manifest(path) == [_entry()]


def test_a_v4_manifest_carrying_a_republish_key_still_reads_as_forward(tmp_path):
    """The flag means nothing below version 5: only the version decides."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"version": 4, "republish": True, "images": []}), encoding="utf-8")
    assert not staged_manifest_republishes(path)


def test_republishes_refuses_an_unknown_version(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"version": 6, "republish": True, "images": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported staged manifest version"):
        staged_manifest_republishes(path)
