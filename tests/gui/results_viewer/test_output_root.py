"""Unit tests for :mod:`phenotypic.gui.results_viewer._output_root`.

Validates ``OutputRoot.discover`` against tmp-path fixtures that mimic
the on-disk layout produced by ``python -m phenotypic`` and exercises
the small read helpers (``overlay_path``, ``has_overlay``,
``image_pairs``, ``column_value_sets``, ``pipeline_summary``).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.tools_ import (
    master_measurements_parquet_path,
    measurements_parquet_path,
)

from tests._output_layout import write_pipeline_json


def _write_master_parquet(root: Path, df: pl.DataFrame) -> None:
    """Write ``master_measurements.parquet`` under ``root/deliverables/``."""
    target = master_measurements_parquet_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target)


def _make_minimal_output(
    root: Path,
    dataset: str = "d1",
    *,
    with_overlays: bool = True,
    write_master: bool = True,
) -> pl.DataFrame:
    """Build a minimal CLI-style output directory under ``root``.

    Args:
        root: Existing tmp dir to populate.
        dataset: Single dataset name to create under ``results/``.
        with_overlays: If ``True``, touch overlay PNG files for the
            stems used in the master frame.
        write_master: If ``True``, write the master parquet.

    Returns:
        The DataFrame written to ``master_measurements.parquet`` (so
        tests can compare against expected unique sets).
    """

    (root / "results" / dataset / "overlays").mkdir(parents=True)
    (root / "results" / dataset / "hdf").mkdir(parents=True)
    (root / "results" / dataset / "measurements").mkdir(parents=True)

    df = pl.DataFrame(
        {
            "Metadata_Dataset": [dataset, dataset],
            "Metadata_ImageFile": ["a", "b"],
            "Metadata_Strain": ["s1", "s2"],
            "Size_Area": [100.0, 200.0],
        }
    )
    if write_master:
        _write_master_parquet(root, df)
    if with_overlays:
        for stem in ("a", "b"):
            (root / "results" / dataset / "overlays" / f"{stem}.png").touch()
    return df


def test_discover_succeeds_on_well_formed_root(tmp_path: Path) -> None:
    """A complete output dir yields a populated ``OutputRoot``."""

    df = _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)

    assert out.root == tmp_path.resolve()
    assert out.master_df.height == df.height
    assert "Metadata_Strain" in out.column_value_sets
    assert out.cache_dir == tmp_path.resolve() / ".viewer_cache" / "dzi"


def test_discover_prefers_post_applied_mirror_over_master(tmp_path: Path) -> None:
    """When ``measurements.parquet`` exists, viewer reads it (post-applied)."""
    _make_minimal_output(tmp_path)

    # Seed a post-applied mirror that differs from master (extra "post_tag"
    # column simulates what _seed_measurements writes after post runs).
    mirror_df = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1", "d1"],
            "Metadata_ImageFile": ["a", "b"],
            "Metadata_Strain": ["s1", "s2"],
            "Size_Area": [100.0, 200.0],
            "post_tag": ["tagged", "tagged"],
        }
    )
    mirror_path = measurements_parquet_path(tmp_path)
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    mirror_df.write_parquet(mirror_path)

    out = OutputRoot.discover(tmp_path)
    assert "post_tag" in out.master_df.columns
    assert out.master_df["post_tag"].to_list() == ["tagged", "tagged"]


def test_discover_falls_back_to_master_when_mirror_absent(tmp_path: Path) -> None:
    """Mid-run / legacy outputs without ``measurements.parquet`` use master."""
    df = _make_minimal_output(tmp_path)
    # No measurements.parquet — only master.
    assert not measurements_parquet_path(tmp_path).exists()

    out = OutputRoot.discover(tmp_path)
    # Display frame falls back to the clean master.
    assert out.master_df.height == df.height
    assert "post_tag" not in out.master_df.columns


def test_discover_missing_master_raises(tmp_path: Path) -> None:
    """No ``master_measurements.parquet`` raises ``FileNotFoundError``."""

    _make_minimal_output(tmp_path, write_master=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        OutputRoot.discover(tmp_path)
    msg = str(excinfo.value)
    assert "master_measurements.parquet" in msg
    assert "python -m phenotypic" in msg


def test_discover_missing_results_dir_raises(tmp_path: Path) -> None:
    """A directory without ``results/`` raises ``FileNotFoundError`` with layout hint."""

    df = pl.DataFrame(
        {"Metadata_Dataset": ["d1"], "Metadata_ImageFile": ["a"]}
    )
    _write_master_parquet(tmp_path, df)

    with pytest.raises(FileNotFoundError) as excinfo:
        OutputRoot.discover(tmp_path)
    msg = str(excinfo.value)
    assert "results" in msg
    assert "overlays" in msg


def test_discover_results_with_no_datasets_raises(tmp_path: Path) -> None:
    """An empty ``results/`` directory raises ``FileNotFoundError``."""

    (tmp_path / "results").mkdir()
    df = pl.DataFrame(
        {"Metadata_Dataset": ["d1"], "Metadata_ImageFile": ["a"]}
    )
    _write_master_parquet(tmp_path, df)

    with pytest.raises(FileNotFoundError) as excinfo:
        OutputRoot.discover(tmp_path)
    assert "dataset" in str(excinfo.value).lower()


def test_discover_results_with_no_overlays_succeeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A dataset dir without ``overlays/`` is allowed; the picker disables those entries."""

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    df = pl.DataFrame(
        {"Metadata_Dataset": ["d1"], "Metadata_ImageFile": ["a"]}
    )
    _write_master_parquet(tmp_path, df)

    with caplog.at_level("WARNING"):
        out = OutputRoot.discover(tmp_path)
    assert out.master_df.height == 1
    assert any("overlays" in rec.message.lower() for rec in caplog.records)
    assert out.has_overlay("d1", "a") is False


def test_discover_missing_imagefile_column_raises(tmp_path: Path) -> None:
    """Missing both ``Metadata_ImageFile`` and ``Metadata_ImageName`` raises."""

    (tmp_path / "results" / "d1" / "overlays").mkdir(parents=True)
    _write_master_parquet(
        tmp_path, pl.DataFrame({"Metadata_Dataset": ["d1"], "Other": ["x"]})
    )
    with pytest.raises(ValueError) as excinfo:
        OutputRoot.discover(tmp_path)
    assert "Metadata_ImageFile" in str(excinfo.value)


def test_discover_aliases_imagename_when_imagefile_absent(tmp_path: Path) -> None:
    """``Metadata_ImageName`` is aliased as ``Metadata_ImageFile`` when the latter is absent."""

    (tmp_path / "results" / "d1" / "overlays").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements" / "a.parquet").touch()
    _write_master_parquet(
        tmp_path,
        pl.DataFrame({"Metadata_Dataset": ["d1"], "Metadata_ImageName": ["a"]}),
    )

    out = OutputRoot.discover(tmp_path)
    assert "Metadata_ImageFile" in out.master_df.columns
    assert out.master_df["Metadata_ImageFile"].to_list() == ["a"]


def test_discover_backfills_dataset_from_filesystem(tmp_path: Path) -> None:
    """When master lacks ``Metadata_Dataset``, it is recovered from the per-image parquets."""

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "overlays").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements" / "a.parquet").touch()
    (tmp_path / "results" / "d2" / "measurements").mkdir(parents=True)
    (tmp_path / "results" / "d2" / "overlays").mkdir(parents=True)
    (tmp_path / "results" / "d2" / "measurements" / "b.parquet").touch()
    _write_master_parquet(
        tmp_path,
        pl.DataFrame({"Metadata_ImageFile": ["a", "b"], "Size_Area": [100.0, 200.0]}),
    )

    out = OutputRoot.discover(tmp_path)
    assert "Metadata_Dataset" in out.master_df.columns
    pairs = out.image_pairs(out.master_df)
    assert pairs == [("d1", "a"), ("d2", "b")]


def test_column_value_sets_are_sorted_unique_str(tmp_path: Path) -> None:
    """``column_value_sets`` casts to str, dedupes, sorts, drops nulls."""

    _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)

    cvs = out.column_value_sets
    assert cvs["Metadata_Strain"] == ["s1", "s2"]
    # Numeric column rendered as string.
    assert cvs["Size_Area"] == sorted({"100.0", "200.0"})
    # Every column on the master frame is represented.
    for column in out.master_df.columns:
        assert column in cvs


def test_column_value_sets_outer_mapping_is_immutable(tmp_path: Path) -> None:
    """The mapping itself rejects ``__setitem__`` (``MappingProxyType``)."""

    _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)
    with pytest.raises(TypeError):
        out.column_value_sets["new_column"] = ["x"]  # type: ignore[index]


def test_overlay_path_returns_expected_absolute_path(tmp_path: Path) -> None:
    """``overlay_path`` resolves to ``<root>/results/<ds>/overlays/<stem>.png``."""

    _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)
    expected = (
        tmp_path.resolve()
        / "results"
        / "d1"
        / "overlays"
        / "a.png"
    )
    assert out.overlay_path("d1", "a") == expected


def test_has_overlay_distinguishes_present_and_absent(tmp_path: Path) -> None:
    """``has_overlay`` is True only for files that exist."""

    _make_minimal_output(tmp_path, with_overlays=False)
    # Touch only "a"; leave "b" absent.
    (tmp_path / "results" / "d1" / "overlays" / "a.png").touch()
    out = OutputRoot.discover(tmp_path)

    assert out.has_overlay("d1", "a") is True
    assert out.has_overlay("d1", "b") is False


def test_image_pairs_returns_sorted_unique_tuples(tmp_path: Path) -> None:
    """``image_pairs`` deduplicates and sorts the (dataset, stem) tuples."""

    _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)

    # Feed a frame with shuffled order and a duplicate row.
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1", "d1", "d1"],
            "Metadata_ImageFile": ["b", "a", "a"],
        }
    )
    pairs = out.image_pairs(df)
    assert pairs == [("d1", "a"), ("d1", "b")]


def test_pipeline_summary_reads_name_from_pipeline_json(tmp_path: Path) -> None:
    """A valid ``pipeline.json`` with a ``name`` populates ``pipeline_summary``."""

    _make_minimal_output(tmp_path)
    write_pipeline_json(tmp_path, json.dumps({"name": "test_pipeline"}))
    out = OutputRoot.discover(tmp_path)
    assert out.pipeline_summary == "test_pipeline"


def test_pipeline_summary_is_none_when_missing_or_malformed(
    tmp_path: Path,
) -> None:
    """Missing or malformed ``pipeline.json`` yields ``pipeline_summary=None``."""

    # Case 1: missing → None.
    _make_minimal_output(tmp_path)
    assert OutputRoot.discover(tmp_path).pipeline_summary is None

    # Case 2: malformed JSON → None (does not raise).
    write_pipeline_json(tmp_path, "{not valid json")
    assert OutputRoot.discover(tmp_path).pipeline_summary is None

    # Case 3: parsed JSON dict with no ``name`` or ``class_name`` field → None.
    # Regression: previously returned the literal string ``"pipeline.json"``
    # via the new PIPELINE_JSON constant during the io_constants extraction
    # refactor (the agent substituted the constant where the original code
    # likely had ``return None``). Caught by opus review of PR #78.
    write_pipeline_json(tmp_path, json.dumps({"version": "1.0"}))
    assert OutputRoot.discover(tmp_path).pipeline_summary is None


def test_cache_dir_is_created_on_discover(tmp_path: Path) -> None:
    """``cache_dir`` exists as a real directory after ``discover``."""

    _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)
    assert out.cache_dir.is_dir()
