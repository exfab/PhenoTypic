"""A tree written before the TSV cutover still opens, and finalize sweeps its stale CSVs (spec B6)."""

from __future__ import annotations

import polars as pl
import pytest

from phenotypic.sdk_ import (
    BundleLayout,
    measurements_table_path,
    remove_stale_csv_sibling,
)
from tests._tiny_run import run_cli

#: CSVs the CLI itself writes and never sweeps. Every other CSV in a finalized
#: tree is a stale generated one.
KEPT = {"measurements.csv", "metadata.csv", "error_analysis.csv"}


def _generated_csvs(deliverables):
    return {p.name for p in deliverables.rglob("*.csv")} - KEPT


def test_bundle_layout_finds_a_pre_cutover_mirror(old_tree_with_measurements_csv):
    layout = BundleLayout.detect(old_tree_with_measurements_csv)
    assert layout.measurements_table_path().name == "measurements.csv"
    assert (
        measurements_table_path(old_tree_with_measurements_csv).name
        == "measurements.csv"
    )
    frame = pl.read_csv(layout.measurements_table_path())
    assert frame.height == pl.read_parquet(layout.mirror_parquet).height > 0


def test_a_new_tree_resolves_to_the_tsv(finished_tiny_run):
    assert measurements_table_path(finished_tiny_run).name == "measurements.tsv"
    layout = BundleLayout.detect(finished_tiny_run)
    assert layout.measurements_table_path() == layout.mirror_tsv


def test_an_empty_folder_resolves_to_the_name_finalize_writes(tmp_path):
    assert measurements_table_path(tmp_path).name == "measurements.tsv"


def test_the_old_tree_holds_generated_csvs_to_sweep(old_tree_with_measurements_csv):
    d = old_tree_with_measurements_csv / "deliverables"
    assert list((d / "measurements_by_feature").glob("*.csv"))
    assert list((d / "measurements_by_category").glob("*.csv"))


def test_refinalize_old_tree_replaces_generated_csv(old_tree_with_measurements_csv):
    d = old_tree_with_measurements_csv / "deliverables"
    metadata_before = (d / "metadata.csv").read_bytes()
    result = run_cli(None, old_tree_with_measurements_csv, mode="recompile")
    assert result.exit_code == 0, result.output
    assert (d / "measurements.tsv").is_file() and (d / "measurements.csv").is_file()
    for split in ("measurements_by_feature", "measurements_by_category"):
        assert not list((d / split).glob("*.csv"))
        assert list((d / split).glob("*.tsv"))
    assert _generated_csvs(d) == set()
    assert (d / "metadata.csv").read_bytes() == metadata_before


def test_finalize_into_an_old_folder_keeps_the_csvs_that_stay(
    old_tree_with_measurements_csv,
):
    d = old_tree_with_measurements_csv / "deliverables"
    error_csv = d / "error_analysis.csv"
    error_csv.write_bytes(b"a,b\n1,2\n")
    metadata_before = (d / "metadata.csv").read_bytes()
    result = run_cli(None, old_tree_with_measurements_csv, mode="recompile")
    assert result.exit_code == 0, result.output
    assert error_csv.read_bytes() == b"a,b\n1,2\n"
    assert (d / "metadata.csv").read_bytes() == metadata_before
    assert pl.read_csv(d / "measurements.csv").equals(
        pl.read_csv(d / "measurements.tsv", separator="\t")
    )


@pytest.mark.parametrize("stem", ["measurements", "error_analysis", "metadata"])
def test_the_sweep_refuses_the_csvs_that_stay(tmp_path, stem):
    (tmp_path / f"{stem}.tsv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / f"{stem}.csv").write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        remove_stale_csv_sibling(tmp_path / f"{stem}.tsv")
    assert (tmp_path / f"{stem}.csv").is_file()


def test_the_sweep_needs_the_tsv_to_exist(tmp_path):
    (tmp_path / "Model.csv").write_text("a\n1\n", encoding="utf-8")
    assert remove_stale_csv_sibling(tmp_path / "Model.tsv") is False
    assert (tmp_path / "Model.csv").is_file()


def test_the_sweep_removes_a_named_analysis_csv(tmp_path):
    (tmp_path / "LogGrowthModel.tsv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "LogGrowthModel.csv").write_text("a\n1\n", encoding="utf-8")
    assert remove_stale_csv_sibling(tmp_path / "LogGrowthModel.tsv") is True
    assert not (tmp_path / "LogGrowthModel.csv").exists()


def test_the_canonical_metadata_view_sweeps_its_csv_twin_and_keeps_the_snapshot(
    tmp_path,
):
    from phenotypic.sdk_._hdf_to_zarr import emit_canonical_metadata_view

    d = tmp_path / "deliverables"
    d.mkdir()
    snapshot = b"Metadata_ImageName,Metadata_Strain\nplate_1,S288C\n"
    (d / "metadata.csv").write_bytes(snapshot)
    (d / "metadata.canonical.csv").write_bytes(b"old\n")
    view = emit_canonical_metadata_view(tmp_path)
    assert view is not None and view.name == "metadata.canonical.tsv"
    assert view.is_file()
    assert not (d / "metadata.canonical.csv").exists()
    assert (d / "metadata.csv").read_bytes() == snapshot
