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

from phenotypic.gui.results_viewer import _output_root
from phenotypic.gui.results_viewer._output_root import (
    OutputRoot,
    OutputSnapshotChangedError,
    _all_parse_as_float,
)
from phenotypic.sdk_ import (
    master_measurements_parquet_path,
    measurements_parquet_path,
)

from tests._output_layout import write_pipeline_json
from phenotypic.schema import METADATA


def _write_master_parquet(root: Path, df: pl.DataFrame) -> None:
    """Write ``master_measurements.parquet`` under ``root/deliverables/``."""
    target = master_measurements_parquet_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target)


def _tree_bytes(root: Path) -> tuple[tuple[str, ...], dict[str, bytes]]:
    """Capture relative directories and exact file bytes."""
    directories = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    )
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    return directories, files


def _make_minimal_output(
    root: Path,
    dataset: str = "d1",
    *,
    with_overlays: bool = True,
    write_master: bool = True,
) -> pl.DataFrame:
    """Build a minimal CLI-style output directory under ``root``.

    Discovery still enumerates datasets from ``results/`` (per-image
    hdf/measurements live there); overlay PNGs now live under
    ``deliverables/overlays/<dataset>/``.

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

    # discovery still enumerates datasets from results/
    (root / "results" / dataset / "hdf").mkdir(parents=True)
    (root / "results" / dataset / "measurements").mkdir(parents=True)

    df = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": [dataset, dataset],
            str(METADATA.IMAGE_NAME): ["a", "b"],
            "MetadataGenetic_Strain": ["s1", "s2"],
            "Size_Area": [100.0, 200.0],
        }
    )
    if write_master:
        _write_master_parquet(root, df)
    if with_overlays:
        overlays = root / "deliverables" / "overlays" / dataset
        overlays.mkdir(parents=True, exist_ok=True)
        for stem in ("a", "b"):
            (overlays / f"{stem}.png").touch()
    return df


def test_discover_succeeds_on_well_formed_root(tmp_path: Path) -> None:
    """A complete output dir yields a populated ``OutputRoot``."""

    df = _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)

    assert out.root == tmp_path.resolve()
    assert out.master_df.height == df.height
    assert "MetadataGenetic_Strain" in out.column_value_sets
    assert not out.cache_dir.is_relative_to(tmp_path.resolve())
    assert out.cache_dir.name == "dzi"
    assert out.source_fingerprint.startswith("sha256:")


def test_external_cache_path_is_pure_and_owned_by_sandbox(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    sandbox = tmp_path / "sandbox"
    _make_minimal_output(source)

    out = OutputRoot.discover(
        source,
        sandbox_root=sandbox,
    )

    assert out.cache_dir.is_relative_to(
        sandbox / ".phenotypic-gui" / "viewer_cache"
    )
    assert not out.cache_dir.exists()
    assert not (source / ".viewer_cache").exists()


def test_discover_prefers_post_applied_mirror_over_master(
    tmp_path: Path,
) -> None:
    """When ``measurements.parquet`` exists, viewer reads it (post-applied)."""
    _make_minimal_output(tmp_path)

    # Seed a post-applied mirror that differs from master (extra "post_tag"
    # column simulates what _seed_measurements writes after post runs).
    mirror_df = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["d1", "d1"],
            str(METADATA.IMAGE_NAME): ["a", "b"],
            "MetadataGenetic_Strain": ["s1", "s2"],
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


def test_discover_falls_back_to_master_when_mirror_absent(
    tmp_path: Path,
) -> None:
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


def test_discover_without_results_dir_boots_standalone(tmp_path: Path) -> None:
    """A deliverables-only bundle (no ``results/``) now discovers successfully.

    Task 4: ``BundleLayout``-backed discovery boots from a deliverables bundle
    alone — datasets are recovered from the master frame's ``Metadata_Dataset``
    column. ``results/``-backed capabilities (``has_results``/``hdf_path``)
    simply report unavailable.
    """

    df = pl.DataFrame(
        {"MetadataExperiment_Dataset": ["d1"], str(METADATA.IMAGE_NAME): ["a"]}
    )
    _write_master_parquet(tmp_path, df)

    out = OutputRoot.discover(tmp_path)
    assert out.has_results is False
    assert out.hdf_path("d1", "a") is None
    assert "d1" in out.master_df["MetadataExperiment_Dataset"].to_list()


def test_discover_dataset_from_master_with_empty_results(
    tmp_path: Path,
) -> None:
    """Datasets are data-driven: a master's ``Metadata_Dataset`` wins over an empty ``results/``."""

    (tmp_path / "results").mkdir()
    df = pl.DataFrame(
        {"MetadataExperiment_Dataset": ["d1"], str(METADATA.IMAGE_NAME): ["a"]}
    )
    _write_master_parquet(tmp_path, df)

    out = OutputRoot.discover(tmp_path)
    assert out.image_pairs(out.master_df) == [("d1", "a")]


def test_discover_results_with_no_overlays_succeeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A dataset dir without ``overlays/`` is allowed; the picker disables those entries."""

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    df = pl.DataFrame(
        {"MetadataExperiment_Dataset": ["d1"], str(METADATA.IMAGE_NAME): ["a"]}
    )
    _write_master_parquet(tmp_path, df)

    with caplog.at_level("WARNING"):
        out = OutputRoot.discover(tmp_path)
    assert out.master_df.height == 1
    assert any("overlays" in rec.message.lower() for rec in caplog.records)
    assert out.has_overlay("d1", "a") is False


def test_discover_missing_imagefile_column_raises(tmp_path: Path) -> None:
    """Missing the ``Metadata_ImageName`` image-stem column raises."""

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    _write_master_parquet(
        tmp_path, pl.DataFrame({"MetadataExperiment_Dataset": ["d1"], "Other": ["x"]})
    )
    with pytest.raises(ValueError) as excinfo:
        OutputRoot.discover(tmp_path)
    assert str(METADATA.IMAGE_NAME) in str(excinfo.value)


def test_discover_aliases_imagename_when_imagefile_absent(
    tmp_path: Path,
) -> None:
    """Legacy ``Metadata_ImageName`` alone satisfies the image-stem requirement.

    A pre-flip master carries the legacy ``Metadata_ImageName`` column and lacks
    the canonical ``MetadataImage_ImageName``; discovery must alias it forward.
    """

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements" / "a.parquet").touch()
    _write_master_parquet(
        tmp_path,
        pl.DataFrame(
            {"MetadataExperiment_Dataset": ["d1"], "Metadata_ImageName": ["a"]}
        ),
    )

    out = OutputRoot.discover(tmp_path)
    assert str(METADATA.IMAGE_NAME) in out.master_df.columns
    assert out.master_df[str(METADATA.IMAGE_NAME)].to_list() == ["a"]


def test_discover_backfills_dataset_from_filesystem(tmp_path: Path) -> None:
    """When master lacks ``Metadata_Dataset``, it is recovered from the per-image parquets."""

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements" / "a.parquet").touch()
    (tmp_path / "results" / "d2" / "measurements").mkdir(parents=True)
    (tmp_path / "results" / "d2" / "measurements" / "b.parquet").touch()
    _write_master_parquet(
        tmp_path,
        pl.DataFrame(
            {str(METADATA.IMAGE_NAME): ["a", "b"], "Size_Area": [100.0, 200.0]}
        ),
    )

    out = OutputRoot.discover(tmp_path)
    assert "MetadataExperiment_Dataset" in out.master_df.columns
    pairs = out.image_pairs(out.master_df)
    assert pairs == [("d1", "a"), ("d2", "b")]


def test_legacy_backfill_parquets_are_part_of_snapshot_revision(
    tmp_path: Path,
) -> None:
    """Every per-image parquet consulted for backfill invalidates the snapshot."""
    measurements = tmp_path / "results" / "d1" / "measurements"
    measurements.mkdir(parents=True)
    legacy_parquet = measurements / "a.parquet"
    legacy_parquet.write_bytes(b"first")
    _write_master_parquet(
        tmp_path,
        pl.DataFrame({str(METADATA.IMAGE_NAME): ["a"], "Size_Area": [100.0]}),
    )

    output = OutputRoot.discover(tmp_path)
    legacy_parquet.write_bytes(b"second")

    assert output.snapshot_is_current() is False
    refreshed = OutputRoot.discover(tmp_path)
    assert refreshed.source_fingerprint != output.source_fingerprint


def test_discover_retries_complete_read_after_snapshot_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre/post mismatch retries from the first source read."""
    _make_minimal_output(tmp_path)
    overlay = tmp_path / "deliverables" / "overlays" / "d1" / "a.png"
    real_fingerprint = _output_root.paths_fingerprint
    calls = 0

    def _mutate_after_first_fingerprint(paths, *, root=None):
        nonlocal calls
        result = real_fingerprint(paths, root=root)
        calls += 1
        if calls == 1:
            overlay.write_bytes(b"new-revision")
        return result

    monkeypatch.setattr(
        _output_root,
        "paths_fingerprint",
        _mutate_after_first_fingerprint,
    )

    output = OutputRoot.discover(tmp_path)

    assert calls == 4
    assert output.snapshot_is_current() is True


def test_discover_refuses_continuously_changing_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two unstable pre/post reads fail instead of binding mixed generations."""
    _make_minimal_output(tmp_path)
    overlay = tmp_path / "deliverables" / "overlays" / "d1" / "a.png"
    real_fingerprint = _output_root.paths_fingerprint
    revision = 0

    def _mutate_after_every_fingerprint(paths, *, root=None):
        nonlocal revision
        result = real_fingerprint(paths, root=root)
        revision += 1
        overlay.write_bytes(f"revision-{revision}".encode())
        return result

    monkeypatch.setattr(
        _output_root,
        "paths_fingerprint",
        _mutate_after_every_fingerprint,
    )

    with pytest.raises(OutputSnapshotChangedError):
        OutputRoot.discover(tmp_path)


def test_column_value_sets_are_sorted_unique_str(tmp_path: Path) -> None:
    """``column_value_sets`` casts to str, dedupes, sorts, drops nulls."""

    _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)

    cvs = out.column_value_sets
    assert cvs["MetadataGenetic_Strain"] == ["s1", "s2"]
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
    """``overlay_path`` resolves to ``<root>/deliverables/overlays/<ds>/<stem>.png``."""

    _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)
    expected = (
        tmp_path.resolve() / "deliverables" / "overlays" / "d1" / "a.png"
    )
    assert out.overlay_path("d1", "a") == expected


def test_has_overlay_distinguishes_present_and_absent(tmp_path: Path) -> None:
    """``has_overlay`` is True only for files that exist."""

    _make_minimal_output(tmp_path, with_overlays=False)
    # Touch only "a"; leave "b" absent.
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True, exist_ok=True)
    (overlays / "a.png").touch()
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
            "MetadataExperiment_Dataset": ["d1", "d1", "d1"],
            str(METADATA.IMAGE_NAME): ["b", "a", "a"],
        }
    )
    pairs = out.image_pairs(df)
    assert pairs == [("d1", "a"), ("d1", "b")]


def test_pipeline_summary_reads_name_from_pipeline_json(
    tmp_path: Path,
) -> None:
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


def test_cache_dir_is_not_created_on_discover(tmp_path: Path) -> None:
    """Discovery computes the external path without writing it."""

    _make_minimal_output(tmp_path)
    out = OutputRoot.discover(tmp_path)
    assert not out.cache_dir.exists()


def test_discover_leaves_legacy_qc_and_viewer_sidecar_byte_identical(
    tmp_path: Path,
) -> None:
    """Discovery never moves legacy topology or folds a viewer sidecar."""
    source = tmp_path / "run"
    _make_minimal_output(source)
    legacy_qc = source / "qc"
    legacy_qc.mkdir()
    (legacy_qc / "legacy.parquet").write_bytes(b"legacy-qc")
    sidecar = source / ".viewer_cache" / "qc_recipe.json"
    sidecar.parent.mkdir()
    sidecar.write_text('{"version": 1, "checks": []}', encoding="utf-8")
    before = _tree_bytes(source)

    OutputRoot.discover(source, sandbox_root=tmp_path)

    assert _tree_bytes(source) == before


def test_all_parse_as_float_true_for_numeric_strings() -> None:
    assert _all_parse_as_float(["2", "10", "1.5"]) is True


def test_all_parse_as_float_false_for_mixed_or_empty() -> None:
    assert _all_parse_as_float(["2", "x", "10"]) is False
    assert _all_parse_as_float([]) is False


def test_column_value_sets_sorts_numeric_columns_numerically(tmp_path) -> None:
    """An all-numeric metadata column sorts 2 < 10, not lexically '10' < '2'."""
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True)
    df = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["d1"] * 3,
            str(METADATA.IMAGE_NAME): ["a", "b", "c"],
            "MetadataCulture_Time": ["10", "2", "1"],
        }
    )
    _write_master_parquet(tmp_path, df)
    for stem in ("a", "b", "c"):
        (overlays / f"{stem}.png").touch()

    out = OutputRoot.discover(tmp_path)
    assert out.column_value_sets["MetadataCulture_Time"] == ["1", "2", "10"]


def test_column_value_sets_keeps_lexical_for_text_columns(tmp_path) -> None:
    df = _make_minimal_output(tmp_path)  # has Metadata_Strain = s1, s2
    out = OutputRoot.discover(tmp_path)
    assert out.column_value_sets["MetadataGenetic_Strain"] == sorted(
        df.get_column("MetadataGenetic_Strain").to_list()
    )


def test_is_numeric_column_true_for_float_measurement(tmp_path) -> None:
    _make_minimal_output(tmp_path)  # Size_Area is Float64
    out = OutputRoot.discover(tmp_path)
    assert out.is_numeric_column("Size_Area") is True


def test_is_numeric_column_true_for_numeric_string_metadata(tmp_path) -> None:
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True)
    df = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["d1", "d1"],
            str(METADATA.IMAGE_NAME): ["a", "b"],
            "MetadataCulture_Time": ["6", "24"],
        }
    )
    _write_master_parquet(tmp_path, df)
    for stem in ("a", "b"):
        (overlays / f"{stem}.png").touch()
    out = OutputRoot.discover(tmp_path)
    assert out.is_numeric_column("MetadataCulture_Time") is True


def test_is_numeric_column_false_for_text_and_missing(tmp_path) -> None:
    _make_minimal_output(tmp_path)  # Metadata_Strain = s1, s2
    out = OutputRoot.discover(tmp_path)
    assert out.is_numeric_column("MetadataGenetic_Strain") is False
    assert out.is_numeric_column("NoSuchColumn") is False
