"""``finalize_run`` publishes ``metrics.json`` and ``schema.json`` (#281 spec C4, C12).

Both are deliverables: written by the one FINAL writer, declared in the
aggregate proof, byte-stable across identical runs, and never read back.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from tests._tiny_run import (
    break_store_for_measure,
    repair_store_for_measure,
    run_cli,
    run_slurm_measure_worker,
)


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


# -- #281 P2 C11d (review I5): a --mode measure failure counts ----------------


def _measure_rows(out):
    from phenotypic._cli._cli_failure_tracker import (
        MEASURE_FAILURE_STAGE,
        read_terminal_failures,
    )

    return [
        r
        for r in read_terminal_failures(out)
        if r.failed_stage == MEASURE_FAILURE_STAGE
    ]


def _images(out):
    return _json(metrics_json_path(out))["images"]


def test_a_local_measure_failure_exits_3_and_counts_in_metrics(
    finished_tiny_run,
):
    break_store_for_measure(finished_tiny_run)
    result = run_cli(None, finished_tiny_run, mode="measure")
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, (
        result.output
    )
    # plate_1's old table is still aggregated, but it did not succeed this run.
    assert _images(finished_tiny_run) == {
        "accepted": 2,
        "succeeded": 1,
        "failed": 1,
    }


def test_a_local_measure_failure_is_journalled_under_its_measure_work_id(
    finished_tiny_run,
):
    from phenotypic._cli._cli_finalize_run import _accepted_work_ids

    store, _ = break_store_for_measure(finished_tiny_run)
    assert run_cli(None, finished_tiny_run, mode="measure").exit_code == 3
    rows = _measure_rows(finished_tiny_run)
    assert [(r.dataset, Path(r.relative_image_path).name) for r in rows] == [
        ("ds", store.name)
    ]
    # Keyed outside the forward inventory, so the forward run's state ignores it.
    assert rows[0].work_id not in _accepted_work_ids(finished_tiny_run)


def test_a_clean_measure_keeps_the_forward_counts(finished_tiny_run):
    result = run_cli(None, finished_tiny_run, mode="measure")
    assert result.exit_code == 0, result.output
    assert _images(finished_tiny_run) == {
        "accepted": 2,
        "succeeded": 2,
        "failed": 0,
    }
    assert _measure_rows(finished_tiny_run) == []


def test_a_measure_record_does_not_touch_a_following_forward_run(
    finished_tiny_run, tiny_inputs
):
    from phenotypic.sdk_ import resolve_run_state

    store, original = break_store_for_measure(finished_tiny_run)
    assert run_cli(None, finished_tiny_run, mode="measure").exit_code == 3
    assert _measure_rows(finished_tiny_run)
    repair_store_for_measure(store, original)
    # A plain forward continuation: forward work ids differ from the measure
    # row's, so nothing is skipped as a recorded failure and the run is clean.
    result = run_cli(tiny_inputs, finished_tiny_run)
    assert result.exit_code == RunExitCode.COMPLETE, result.output
    assert "recorded failure" not in result.output
    state = resolve_run_state(finished_tiny_run, depth="deep")
    assert state.diagnostics.failed == 0


def test_a_measure_record_neither_blocks_a_later_measure_nor_takes_retry_failures(
    finished_tiny_run,
):
    store, original = break_store_for_measure(finished_tiny_run)
    assert run_cli(None, finished_tiny_run, mode="measure").exit_code == 3
    rows = _measure_rows(finished_tiny_run)
    # --retry-failures is refused in measure mode before anything is written.
    refused = run_cli(
        None, finished_tiny_run, mode="measure", extra=["--retry-failures"]
    )
    assert refused.exit_code == 2, refused.output
    assert "--retry-failures" in refused.output
    assert _measure_rows(finished_tiny_run) == rows
    # Measure mode never skips a recorded failure, so once the store is
    # repaired a plain re-measure re-measures it and the run is clean again.
    repair_store_for_measure(store, original)
    result = run_cli(None, finished_tiny_run, mode="measure")
    assert result.exit_code == 0, result.output
    assert _images(finished_tiny_run) == {
        "accepted": 2,
        "succeeded": 2,
        "failed": 0,
    }


def test_a_slurm_measure_worker_journals_its_failure(
    finished_tiny_run, monkeypatch
):
    store, _ = break_store_for_measure(finished_tiny_run)
    result = run_slurm_measure_worker(finished_tiny_run, store, monkeypatch)
    assert result.exit_code == 1, result.output
    rows = _measure_rows(finished_tiny_run)
    assert [(r.dataset, r.lifecycle_epoch) for r in rows] == [
        ("ds", "slurm-unfenced")
    ]


def test_a_slurm_measure_finalizer_counts_the_rows_of_its_generation(
    finished_tiny_run, monkeypatch
):
    # The rows come from the real SLURM measure worker, one under an earlier
    # generation and one under this one. The finalizer passes its generation
    # as `execution_epoch` and counts only its own.
    from phenotypic._cli._cli_output_manager import aggregate_measurements
    from phenotypic._cli._cli_slurm_lifecycle import (
        deactivate_generation,
        initialize_slurm_lifecycle,
    )

    for stem, generation in (("plate_2", "gen-old"), ("plate_1", "gen-now")):
        initialize_slurm_lifecycle(
            finished_tiny_run, generation=generation, mode="measure"
        )
        store, _ = break_store_for_measure(finished_tiny_run, stem)
        result = run_slurm_measure_worker(
            finished_tiny_run, store, monkeypatch, generation=generation
        )
        assert result.exit_code == 1, result.output
        if generation == "gen-old":
            deactivate_generation(finished_tiny_run, generation)
    assert {r.lifecycle_epoch for r in _measure_rows(finished_tiny_run)} == {
        "gen-old",
        "gen-now",
    }
    assert aggregate_measurements(
        output_dir=finished_tiny_run,
        dataset_names=["ds"],
        execution_epoch="gen-now",
    )
    assert _images(finished_tiny_run) == {
        "accepted": 2,
        "succeeded": 1,
        "failed": 1,
    }
