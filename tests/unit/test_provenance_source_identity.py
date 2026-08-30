"""A published store must not carry the cluster path of its pipeline file."""

from __future__ import annotations

from pathlib import Path

from phenotypic._core._provenance import (
    initialize_cli_provenance,
    pipeline_source_identity,
)
from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate


def _pipeline_file(tmp_path: Path) -> Path:
    nested = tmp_path / "config" / "deep"
    nested.mkdir(parents=True)
    path = nested / "preprocess_pipeline.json.pht-pipe"
    path.write_text('{"name": "acr_preprocess"}', encoding="utf-8")
    return path


def test_default_records_the_resolved_absolute_path(tmp_path: Path) -> None:
    """Bundle stores keep the absolute path; they never leave the run dir."""
    path = _pipeline_file(tmp_path)
    identity = pipeline_source_identity(path)
    assert identity["source_path"] == str(path.resolve())


def test_basename_only_records_just_the_filename(tmp_path: Path) -> None:
    path = _pipeline_file(tmp_path)
    identity = pipeline_source_identity(path, basename_only=True)
    assert identity["source_path"] == "preprocess_pipeline.json.pht-pipe"
    assert "/" not in identity["source_path"]


def test_the_digest_is_identical_either_way(tmp_path: Path) -> None:
    """sha256 is the identity; basename_only must not weaken it."""
    path = _pipeline_file(tmp_path)
    assert (
        pipeline_source_identity(path)["sha256"]
        == pipeline_source_identity(path, basename_only=True)["sha256"]
    )


def test_initialize_cli_provenance_threads_the_flag(tmp_path: Path) -> None:
    path = _pipeline_file(tmp_path)
    img = Image(load_synth_yeast_plate())
    initialize_cli_provenance(img, path, basename_only=True)
    journal = img._metadata.provenance_journal
    assert journal["pipeline"]["source_path"] == "preprocess_pipeline.json.pht-pipe"
