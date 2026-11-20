"""Finalize publishes TSV deliverables, and keeps the CSV mirror the GUI curates (spec B2, B9)."""

from __future__ import annotations

import json

import polars as pl

from phenotypic.sdk_ import (
    BundleLayout,
    aggregate_publication_marker_path,
    measurements_tsv_path,
)
from phenotypic.sdk_._run_state import aggregate_proof_refusal

#: The only CSVs a finalized run may hold in ``deliverables/`` until #283.
CSV_THE_CLI_MAY_LEAVE = {"measurements.csv", "metadata.csv", "error_analysis.csv"}


def test_finalize_writes_tsv_deliverables(finished_tiny_run):
    d = finished_tiny_run / "deliverables"
    assert (d / "measurements.tsv").is_file() and (d / "measurements.csv").is_file()
    assert {p.name for p in d.rglob("*.csv")} <= CSV_THE_CLI_MAY_LEAVE
    assert {p.suffix for p in (d / "measurements_by_feature").iterdir()} == {".tsv", ".parquet"}
    assert (d / "measurements.tsv").read_bytes().split(b"\n", 1)[0].count(b"\t") > 0


def test_the_tsv_and_csv_mirrors_hold_the_same_frame(finished_tiny_run):
    d = finished_tiny_run / "deliverables"
    tsv = pl.read_csv(d / "measurements.tsv", separator="\t")
    csv = pl.read_csv(d / "measurements.csv")
    assert tsv.equals(csv)
    assert tsv.height == pl.read_parquet(d / "measurements.parquet").height > 0


def test_bundle_layout_names_both_mirror_files(finished_tiny_run):
    layout = BundleLayout.detect(finished_tiny_run)
    assert layout.mirror_csv.name == "measurements.csv"
    assert layout.mirror_tsv == measurements_tsv_path(finished_tiny_run)
    assert layout.mirror_tsv.is_file()


def test_the_aggregate_proof_records_the_tsv_mirror(finished_tiny_run):
    marker = json.loads(
        aggregate_publication_marker_path(finished_tiny_run).read_text(encoding="utf-8")
    )
    outputs = marker["required_outputs"]
    assert outputs["measurements_tsv"]["path"] == "deliverables/measurements.tsv"
    assert {"measurements_csv", "measurements_parquet", "master_parquet"} <= set(outputs)


def test_a_curation_does_not_stale_the_proof(finished_tiny_run):
    d = finished_tiny_run / "deliverables"
    for name in ("measurements.csv", "measurements.tsv"):
        (d / name).write_bytes((d / name).read_bytes() + b"\n")   # what a curation save changes
    assert aggregate_proof_refusal(finished_tiny_run) is None


def test_the_master_is_still_enforced(finished_tiny_run):
    master = finished_tiny_run / "deliverables" / "master_measurements.parquet"
    master.write_bytes(master.read_bytes() + b"\0")
    assert aggregate_proof_refusal(finished_tiny_run) is not None
