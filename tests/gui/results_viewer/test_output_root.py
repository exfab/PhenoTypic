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
        df.write_parquet(root / "master_measurements.parquet")
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
    df.write_parquet(tmp_path / "master_measurements.parquet")

    with pytest.raises(FileNotFoundError) as excinfo:
        OutputRoot.discover(tmp_path)
    msg = str(excinfo.value)
    assert "results" in msg
    assert "overlays" in msg


def test_discover_results_with_no_dataset_overlays_raises(tmp_path: Path) -> None:
    """A results dir with no ``<ds>/overlays`` subdir raises ``FileNotFoundError``."""

    (tmp_path / "results").mkdir()
    df = pl.DataFrame(
        {"Metadata_Dataset": ["d1"], "Metadata_ImageFile": ["a"]}
    )
    df.write_parquet(tmp_path / "master_measurements.parquet")

    with pytest.raises(FileNotFoundError) as excinfo:
        OutputRoot.discover(tmp_path)
    assert "overlays" in str(excinfo.value)


@pytest.mark.parametrize("missing", ["Metadata_Dataset", "Metadata_ImageFile"])
def test_discover_missing_required_column_raises(
    tmp_path: Path, missing: str
) -> None:
    """Missing either required column raises ``ValueError`` listing actual columns."""

    (tmp_path / "results" / "d1" / "overlays").mkdir(parents=True)
    columns = {
        "Metadata_Dataset": ["d1"],
        "Metadata_ImageFile": ["a"],
        "Metadata_Strain": ["s1"],
    }
    columns.pop(missing)
    pl.DataFrame(columns).write_parquet(
        tmp_path / "master_measurements.parquet"
    )

    with pytest.raises(ValueError) as excinfo:
        OutputRoot.discover(tmp_path)
    msg = str(excinfo.value)
    assert missing in msg
    # Message should also list the actual columns present.
    for present in columns:
        assert present in msg


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
    (tmp_path / "pipeline.json").write_text(
        json.dumps({"name": "test_pipeline"}), encoding="utf-8"
    )
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
    (tmp_path / "pipeline.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    assert OutputRoot.discover(tmp_path).pipeline_summary is None


def test_cache_dir_is_created_on_discover(tmp_path: Path) -> None:
    """``cache_dir`` exists as a real directory after ``discover``."""

    _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)
    assert out.cache_dir.is_dir()
