"""``finalize_run`` publishes ``metrics.json`` and ``schema.json`` (#281 spec C4, C12).

Both are deliverables: written by the one FINAL writer, declared in the
aggregate proof, byte-stable across identical runs, and never read back.
"""

from __future__ import annotations

import json

import polars as pl

import phenotypic
from phenotypic.sdk_ import (
    RunExitCode,
    aggregate_publication_marker_path,
    metrics_json_path,
    readme_md_path,
    schema_json_path,
)
from phenotypic.sdk_._run_state import aggregate_proof_refusal
from tests._tiny_run import run_cli


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_finalize_writes_both_and_the_proof_covers_them(finished_tiny_run):
    d = finished_tiny_run / "deliverables"
    assert (d / "metrics.json").is_file() and (d / "schema.json").is_file()
    (d / "metrics.json").write_text("{}\n")
    assert aggregate_proof_refusal(finished_tiny_run) is not None


def test_a_tampered_schema_also_stales_the_proof(finished_tiny_run):
    assert aggregate_proof_refusal(finished_tiny_run) is None
    path = schema_json_path(finished_tiny_run)
    path.write_bytes(path.read_bytes() + b" ")
    assert aggregate_proof_refusal(finished_tiny_run) is not None


def test_the_proof_names_both_files(finished_tiny_run):
    outputs = _json(aggregate_publication_marker_path(finished_tiny_run))["required_outputs"]
    assert outputs["metrics_json"]["path"] == "deliverables/metrics.json"
    assert outputs["schema_json"]["path"] == "deliverables/schema.json"


def test_metrics_counts_a_clean_run(finished_tiny_run):
    metrics = _json(metrics_json_path(finished_tiny_run))
    master = pl.read_parquet(finished_tiny_run / "deliverables" / "master_measurements.parquet")
    assert metrics["images"] == {"accepted": 2, "succeeded": 2, "failed": 0}
    assert metrics["objects"] == {"measured": master.height} and master.height > 0
    assert metrics["metadata"] == {"phantoms": 0}
    assert metrics["phenotypic_version"] == phenotypic.__version__
    assert metrics["schema_version"] == 1
    assert "embeddings" not in metrics


def test_metrics_pipeline_sha_is_null_while_v1_tables_name_no_pipeline(finished_tiny_run):
    # Spec C4: the sha every aggregated table shares, never the run config's.
    # A v1 table records no producing pipeline (spec D10), so nothing is shared
    # until P3 T4 feeds the certified run folders' digests.
    assert _json(metrics_json_path(finished_tiny_run))["pipeline_sha256"] is None


def test_metrics_counts_a_terminal_failure(tiny_inputs_one_bad, tmp_path):
    out = tmp_path / "out"
    result = run_cli(tiny_inputs_one_bad, out)
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output
    assert _json(metrics_json_path(out))["images"] == {
        "accepted": 3,
        "succeeded": 2,
        "failed": 1,
    }


def test_metrics_counts_metadata_phantoms(tiny_inputs, tmp_path, metadata_bytes):
    metadata = tmp_path / "layout.csv"
    metadata.write_bytes(metadata_bytes[".csv"] + b"plate_9,W303\n")
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, extra=["--metadata", str(metadata)])
    assert result.exit_code == 0, result.output
    assert _json(metrics_json_path(out))["metadata"] == {"phantoms": 1}


def test_schema_lists_every_published_table(finished_tiny_run):
    d = finished_tiny_run / "deliverables"
    tables = _json(schema_json_path(finished_tiny_run))["tables"]
    expected = {"master_measurements", "measurements"}
    expected |= {f"measurements_by_feature/{p.stem}" for p in (d / "measurements_by_feature").glob("*.parquet")}
    expected |= {f"measurements_by_category/{p.stem}" for p in (d / "measurements_by_category").glob("*.parquet")}
    assert "measurements_by_feature/MeasureSize" in expected
    assert set(tables) == expected
    mirror = pl.read_parquet(d / "measurements.parquet")
    assert [c["name"] for c in tables["measurements"]] == mirror.columns
    area = next(c for c in tables["master_measurements"] if c["name"] == "Size_Area")
    assert area == {
        "name": "Size_Area",
        "dtype": "Float64",
        "unit": "px2",
        "producer": "MeasureSize",
        "member": "SIZE.AREA",
        "desc": area["desc"],
    } and area["desc"]


def test_identical_runs_give_identical_metrics_and_schema(tiny_inputs, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert run_cli(tiny_inputs, a).exit_code == 0
    assert run_cli(tiny_inputs, b).exit_code == 0
    for path in (metrics_json_path, schema_json_path):
        assert path(a).read_bytes() == path(b).read_bytes()


def test_a_recompile_rewrites_them_unchanged(finished_tiny_run):
    before = {p: p(finished_tiny_run).read_bytes() for p in (metrics_json_path, schema_json_path)}
    result = run_cli(None, finished_tiny_run, mode="recompile")
    assert result.exit_code == 0, result.output
    assert {p: p(finished_tiny_run).read_bytes() for p in before} == before
    assert aggregate_proof_refusal(finished_tiny_run) is None


def test_the_readme_documents_both_files(finished_tiny_run):
    readme = readme_md_path(finished_tiny_run).read_text(encoding="utf-8")
    assert "metrics.json" in readme and "schema.json" in readme
