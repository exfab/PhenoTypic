"""Startup publishes the reference plan; work-ids carry the per-image digest."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner

from phenotypic import ImagePipeline
from phenotypic._cli import _cli_failure_tracker as ft
from phenotypic._cli import (
    _cli_preflight,
    _cli_process_only,
    _cli_process_single,
    _cli_reference,
    _cli_staged_strategy,
    _cli_staged_workers,
)
from phenotypic._cli._cli_process_single import _worker_work_identity
from phenotypic._cli._cli_types import Dataset
from phenotypic._core._reference_context import ReferenceTableError
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import SubtractBlank
from phenotypic.measure import MeasureSize, MeasureSymZones
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_._io_constants import (
    image_record_path,
    metadata_csv_deliverable_path,
    reference_manifest_path,
    reference_metadata_snapshot_path,
    terminal_failures_jsonl_path,
)
from tests.unit.cli._preflight_support import make_config, make_context


def _ids(**overrides):
    base = dict(
        dataset="plate1", relative_image_path="plate1/t01.tif", input_sha256="a",
        pipeline_fingerprint="b", processing_config_digest="c", mode="full",
    )
    base.update(overrides)
    return ft.compute_work_id(**base)


def test_work_id_unchanged_without_a_reference_digest():
    assert _ids() == _ids(reference_digest=None)


def test_work_id_changes_with_the_reference_digest():
    assert _ids(reference_digest="x") != _ids(reference_digest="y")
    assert _ids(reference_digest="x") != _ids()


def test_every_worker_core_enters_the_reference_context():
    for owner, name in (
        (_cli_process_single, "process_single_image_core"),
        (_cli_process_only, "process_single_apply_only_core"),
        (_cli_staged_workers, "stage1_preprocess_core"),
        (_cli_staged_workers, "stage2_detect_core"),
        (_cli_staged_workers, "stage3_merge_measure_core"),
        (_cli_staged_strategy.StagedGpuStrategy, "_export_objmap_layer"),
    ):
        assert "worker_reference_context(" in inspect.getsource(getattr(owner, name)), name


def test_a_stage2_prefix_failure_is_a_per_image_failure(tmp_path, monkeypatch):
    """Review B1: the prefix runs inside Stage 2's try, so it is classified."""

    class _Boom(RuntimeError):
        pass

    def explode(image, prefix):
        raise _Boom("no reference context")

    monkeypatch.setattr(_cli_staged_workers, "_apply_stage2_prefix", explode)
    monkeypatch.setattr(
        _cli_staged_workers.Image, "load_zarr", classmethod(lambda cls, path: object())
    )
    with pytest.raises(ft.PerImageScientificError) as caught:
        _cli_staged_workers.stage2_detect_core(
            object(), tmp_path, "plate1", "t01", "slot", stage2_prefix=[object()]
        )
    assert isinstance(caught.value.cause, _Boom)


# ---- the mode-scoped operation walk --------------------------------------


def _measurer_only_pipeline() -> ImagePipeline:
    """A reference op that only a measurer runs (full mode only, spec D10)."""
    private = ImagePipeline(ops={"sb": SubtractBlank(), "det": OtsuDetector()})
    return ImagePipeline(
        ops={"det": OtsuDetector()},
        meas={"zones": MeasureSymZones(center_detector=private)},
    )


@pytest.mark.parametrize("mode", ["full", "process", "measure"])
def test_operations_in_scope_is_the_mode_walk(mode):
    pipeline = ImagePipeline(
        ops={"sb": SubtractBlank(), "det": OtsuDetector()},
        meas={"zones": MeasureSymZones(
            center_detector=ImagePipeline(ops={"sb": SubtractBlank()})
        )},
    )
    context = make_context(pipeline, mode)
    assert _cli_preflight.operations_in_scope(context) == (
        _cli_preflight.operations_run_in_mode(pipeline, mode)
    )


def test_startup_plans_only_the_operations_the_mode_runs(run):
    """A process run never needs a column only a measurer's private op reads."""
    base, table, dataset = run
    pipeline = ImagePipeline(
        ops={"sb": SubtractBlank(), "det": OtsuDetector()},
        meas={"zones": MeasureSymZones(center_detector=ImagePipeline(
            ops={"sb": SubtractBlank(blank_column="Metadata_OtherBlank")}
        ))},
    )
    config = _config(
        base, _pipeline_file(base, pipeline), metadata_csv=table, process_only_layer="gray"
    )
    _cli_reference.publish_reference_inputs(config, [dataset], base / "out")
    manifest = _cli_reference.read_reference_manifest(base / "out")
    assert set(manifest["datasets"]["plate1"]["digests"]) == {"t01", "t02"}


def test_reference_operations_follow_the_mode():
    pipeline = _measurer_only_pipeline()
    assert _cli_reference.reference_operations_in_scope(pipeline, "process") == []
    # Measure mode runs the measurers, so their private detector is in scope
    # (MODE_SLOTS); publish_reference_inputs still never plans for measure.
    for mode in ("full", "measure"):
        (found,) = _cli_reference.reference_operations_in_scope(pipeline, mode)
        assert isinstance(found, SubtractBlank), mode


# ---- startup publication --------------------------------------------------


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "images" / "plate1"
    root.mkdir(parents=True)
    for stem, level in (("blank", 20), ("t01", 60), ("t02", 90)):
        tifffile.imwrite(root / f"{stem}.tif", np.full((8, 8), level, dtype=np.uint8))
    table = tmp_path / "layout.csv"
    pd.DataFrame({"ImageName": ["t01", "t02"], "BlankImage": ["blank", "blank"]}).to_csv(
        table, index=False
    )
    dataset = Dataset(
        name="plate1",
        images=[root / "t01.tif", root / "t02.tif"],
        input_dir=root,
        output_dir=tmp_path / "out" / "plate1",
    )
    return tmp_path, table, dataset


def _pipeline_file(base: Path, pipeline: ImagePipeline, name: str = "pipe.json") -> Path:
    path = base / name
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


def _reference_pipeline() -> ImagePipeline:
    return ImagePipeline(ops={"sb": SubtractBlank(), "det": OtsuDetector()})


def _config(base: Path, pipeline: Path, **overrides):
    values = dict(
        pipeline_json=pipeline,
        input_path=base / "images",
        output_dir=base / "out",
        image_type="Image",
    )
    values.update(overrides)
    return make_config(**values)


def test_full_mode_publishes_a_manifest_with_per_image_digests(run):
    base, table, dataset = run
    config = _config(base, _pipeline_file(base, _reference_pipeline()), metadata_csv=table)
    _cli_reference.publish_reference_inputs(config, [dataset], base / "out")
    manifest = _cli_reference.read_reference_manifest(base / "out")
    assert manifest is not None
    assert set(manifest["datasets"]["plate1"]["digests"]) == {"t01", "t02"}
    assert manifest["datasets"]["plate1"]["images"]["blank"].endswith("plate1/blank.tif")


def test_process_mode_snapshots_the_table_and_plans_against_the_snapshot(run):
    base, table, dataset = run
    out = base / "out"
    config = _config(
        base, _pipeline_file(base, _reference_pipeline()),
        metadata_csv=table, process_only_layer="detect_mat",
    )
    _cli_reference.publish_reference_inputs(config, [dataset], out)
    snapshot = reference_metadata_snapshot_path(out)
    assert snapshot.read_bytes() == table.read_bytes()
    manifest = _cli_reference.read_reference_manifest(out)
    assert manifest["table"] == str(snapshot.resolve())
    # The user's path stays the configured one: the run identity and the
    # processing state digest it exactly as they did before this change.
    assert config.metadata_csv == table

    # A continuation without --metadata plans from the snapshot.
    again = _config(
        base, config.pipeline_json, process_only_layer="detect_mat"
    )
    _cli_reference.publish_reference_inputs(again, [dataset], out)
    assert _cli_reference.read_reference_manifest(out)["datasets"] == manifest["datasets"]


def test_a_reference_pipeline_without_any_table_is_refused(run):
    base, _, dataset = run
    config = _config(base, _pipeline_file(base, _reference_pipeline()))
    with pytest.raises(ReferenceTableError, match="--metadata"):
        _cli_reference.publish_reference_inputs(config, [dataset], base / "out")
    assert not reference_manifest_path(base / "out").exists()


def test_a_stale_manifest_is_removed_when_the_pipeline_reads_no_references(run):
    base, table, dataset = run
    out = base / "out"
    plain = _config(base, _pipeline_file(base, ImagePipeline(ops={"det": OtsuDetector()}), "plain.json"))
    untouched = ft.work_id_for_image(plain, "plate1", dataset.images[0])
    reference = _config(base, _pipeline_file(base, _reference_pipeline()), metadata_csv=table)
    _cli_reference.publish_reference_inputs(reference, [dataset], out)
    assert reference_manifest_path(out).is_file()

    _cli_reference.publish_reference_inputs(plain, [dataset], out)
    assert not reference_manifest_path(out).exists()
    # A plain pipeline's work-ids are exactly what they were before the feature.
    assert ft.work_id_for_image(plain, "plate1", dataset.images[0]) == untouched


def test_measure_mode_never_touches_the_manifest(run):
    """Measure may run beside live forward workers that read the manifest."""
    base, table, dataset = run
    out = base / "out"
    forward = _config(base, _pipeline_file(base, _reference_pipeline()), metadata_csv=table)
    _cli_reference.publish_reference_inputs(forward, [dataset], out)
    before = reference_manifest_path(out).read_bytes()

    plain = _pipeline_file(base, ImagePipeline(meas={"size": MeasureSize()}), "measure.json")
    _cli_reference.publish_reference_inputs(
        _config(base, plain, measure_only=True), [dataset], out
    )
    assert reference_manifest_path(out).read_bytes() == before


def test_process_mode_does_not_plan_an_operation_only_a_measurer_runs(run):
    """Cluster C note 1: a reference op inside a measurer is out of process scope."""
    base, _, dataset = run
    out = base / "out"
    config = _config(
        base, _pipeline_file(base, _measurer_only_pipeline()), process_only_layer="gray"
    )
    # No table at all: planning the measurer's op would raise.
    _cli_reference.publish_reference_inputs(config, [dataset], out)
    assert not reference_manifest_path(out).exists()


# ---- the work-id readers --------------------------------------------------


def _worker_identity(config, dataset, image, *, output_dir):
    return _worker_work_identity(
        pipeline=config.pipeline_json,
        image=image,
        input_root=config.input_path,
        dataset_name=dataset,
        image_type=config.image_type,
        nrows=config.nrows,
        ncols=config.ncols,
        bit_depth=config.bit_depth,
        detect_mode=config.detect_mode,
        layer=config.process_only_layer,
        ext=config.ext,
        process_format=config.process_format,
        include_dataset_column=config.include_dataset_column,
        overlay_alpha=config.overlay_alpha,
        save_overlays=config.save_overlays,
        drop_originals=config.drop_originals,
        mode="full",
        reference_digest=ft.image_reference_digest(output_dir, dataset, image, "full"),
    )


def test_selection_and_worker_agree_on_the_reference_work_id(run, tmp_path):
    base, table, dataset = run
    out = base / "out"
    config = _config(base, _pipeline_file(base, _reference_pipeline()), metadata_csv=table)
    image = dataset.images[0]
    without = ft.work_id_for_image(config, "plate1", image)
    assert without == _worker_identity(config, "plate1", image, output_dir=out)

    _cli_reference.publish_reference_inputs(config, [dataset], out)
    with_digest = ft.work_id_for_image(config, "plate1", image)
    assert with_digest != without
    assert with_digest == _worker_identity(config, "plate1", image, output_dir=out)


def test_editing_one_images_blank_changes_only_its_work_id(run):
    base, table, dataset = run
    out = base / "out"
    tifffile.imwrite(dataset.input_dir / "blank2.tif", np.full((8, 8), 25, dtype=np.uint8))
    config = _config(base, _pipeline_file(base, _reference_pipeline()), metadata_csv=table)
    _cli_reference.publish_reference_inputs(config, [dataset], out)
    t01, t02 = (ft.work_id_for_image(config, "plate1", p)[0] for p in dataset.images)

    edited = base / "edited.csv"
    pd.DataFrame({"ImageName": ["t01", "t02"], "BlankImage": ["blank2", "blank"]}).to_csv(
        edited, index=False
    )
    config.metadata_csv = edited
    _cli_reference.publish_reference_inputs(config, [dataset], out)
    assert ft.work_id_for_image(config, "plate1", dataset.images[0])[0] != t01
    assert ft.work_id_for_image(config, "plate1", dataset.images[1])[0] == t02


def test_measure_work_ids_ignore_a_forward_runs_manifest(run):
    base, table, dataset = run
    out = base / "out"
    measure = _config(
        base, _pipeline_file(base, _reference_pipeline()), measure_only=True
    )
    before = ft.work_id_for_image(measure, "plate1", dataset.images[0])
    forward = _config(base, measure.pipeline_json, metadata_csv=table)
    _cli_reference.publish_reference_inputs(forward, [dataset], out)
    assert ft.work_id_for_image(measure, "plate1", dataset.images[0]) == before


# ---- the CLI's un-skippable reference checks (read-only half) -------------


PREVIOUS = "previous-run.txt"


@pytest.fixture
def cli_run(run):
    base, table, _ = run
    previous = base / "out"
    previous.mkdir()
    (previous / PREVIOUS).write_text("keep me", encoding="utf-8")
    return base, table, _pipeline_file(base, _reference_pipeline())


def _invoke(*args: str):
    return CliRunner().invoke(phenotypic_cli, [*args])


def _common(base: Path, pipeline: Path, *mode: str) -> list[str]:
    return [
        "--pipeline", str(pipeline), "--input", str(base / "images"),
        "--output", str(base / "out"), *mode,
    ]


@pytest.mark.parametrize("mode", [(), ("--mode", "process", "--layer", "gray")])
def test_overwrite_without_metadata_is_refused_before_anything_is_deleted(cli_run, mode):
    base, _, pipeline = cli_run
    result = _invoke(*_common(base, pipeline, *mode), "--overwrite", "--skip-validation")
    assert result.exit_code == 2, result.output
    assert "--overwrite deletes the run's reference metadata snapshot" in result.output
    assert (base / "out" / PREVIOUS).read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("mode", [(), ("--mode", "process", "--layer", "gray")])
def test_an_unreadable_reference_table_is_refused_even_when_validation_is_skipped(
    cli_run, mode
):
    base, _, pipeline = cli_run
    bad = base / "bad.csv"
    bad.write_text("Strain\nWT\n", encoding="utf-8")
    result = _invoke(
        *_common(base, pipeline, *mode), "--metadata", str(bad),
        "--overwrite", "--skip-validation",
    )
    assert result.exit_code == 2, result.output
    assert "--metadata:" in result.output
    assert (base / "out" / PREVIOUS).read_text(encoding="utf-8") == "keep me"


def test_a_process_mode_table_inside_the_output_is_refused_for_a_reference_pipeline(cli_run):
    base, table, pipeline = cli_run
    inside = base / "out" / "layout.csv"
    inside.write_bytes(table.read_bytes())
    result = _invoke(
        *_common(base, pipeline, "--mode", "process", "--layer", "gray"),
        "--metadata", str(inside), "--overwrite", "--dry-run",
    )
    assert result.exit_code != 0, result.output
    assert "lies inside --output" in result.output
    assert inside.exists()


def test_process_mode_reports_metadata_ignored_only_without_reference_columns(run):
    base, table, _ = run
    plain = _pipeline_file(base, ImagePipeline(ops={"det": OtsuDetector()}), "plain.json")
    reference = _pipeline_file(base, _reference_pipeline())
    process = ("--mode", "process", "--layer", "gray")

    ignored = _invoke(*_common(base, plain, *process), "--metadata", str(table), "--dry-run")
    assert ignored.exit_code == 0, ignored.output
    assert "--metadata is ignored in --mode process" in ignored.output

    used = _invoke(*_common(base, reference, *process), "--metadata", str(table), "--dry-run")
    assert used.exit_code == 0, used.output
    assert "--metadata is ignored" not in used.output


def _tree(root: Path) -> dict[str, str]:
    """Every file under *root* and its bytes' digest; empty when *root* is absent."""
    if not root.exists():
        return {}
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize(
    ("mode", "snapshot"),
    [((), metadata_csv_deliverable_path),
     (("--mode", "process", "--layer", "gray"), reference_metadata_snapshot_path)],
)
def test_an_invalid_fallback_snapshot_is_refused_even_when_validation_is_skipped(
    cli_run, mode, snapshot
):
    """Review F3: without --metadata the early check reads the snapshot the run would use."""
    base, _, pipeline = cli_run
    path = snapshot(base / "out")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Strain\nWT\n", encoding="utf-8")
    before = _tree(base / "out")
    result = _invoke(*_common(base, pipeline, *mode), "--skip-validation", "--restart")
    assert result.exit_code == 2, result.output
    assert str(path) in result.output
    assert _tree(base / "out") == before


@pytest.mark.parametrize("mode", [(), ("--mode", "process", "--layer", "gray")])
def test_no_table_at_all_is_refused_before_anything_is_written(run, mode):
    """Review F3: refused in the read-only half, so nothing is created under --output."""
    base, _, _ = run
    pipeline = _pipeline_file(base, _reference_pipeline())
    result = _invoke(*_common(base, pipeline, *mode), "--skip-validation", "--restart")
    assert result.exit_code == 2, result.output
    assert "PF-REF-NO-TABLE" in result.output
    assert not (base / "out").exists()


def test_measure_mode_refuses_a_pipeline_that_reads_reference_metadata(cli_run):
    """User decision: re-measuring a reference pipeline is refused up front."""
    base, _, _ = cli_run
    pipeline = _pipeline_file(base, _measurer_only_pipeline(), "measure.json")
    before = _tree(base / "out")
    result = _invoke(
        "--mode", "measure", "--pipeline", str(pipeline),
        "--output", str(base / "out"), "--skip-validation",
    )
    assert result.exit_code == 2, result.output
    assert "center_detector" in result.output
    assert "--mode full" in result.output
    assert _tree(base / "out") == before



# ---- one plan per image: identity and context agree (review F1) -----------


def _replan(out: Path, dataset: str, stem: str) -> None:
    """A concurrent invocation re-planned *stem*: its manifest digest changed."""
    path = reference_manifest_path(out)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["datasets"][dataset]["digests"][stem] = "replanned-by-another-invocation"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest), encoding="utf-8")
    tmp.replace(path)


def test_a_pinned_context_refuses_an_image_planned_differently(run):
    base, table, dataset = run
    out = base / "out"
    config = _config(base, _pipeline_file(base, _reference_pipeline()), metadata_csv=table)
    _cli_reference.publish_reference_inputs(config, [dataset], out)
    pins = {
        stem: _cli_reference.ReferencePin(
            stem, _cli_reference.reference_digest_for(out, "plate1", stem)
        )
        for stem in ("t01", "t02")
    }
    with _cli_reference.worker_reference_context(out, "plate1", pin=pins["t01"]) as active:
        assert active is not None
    _replan(out, "plate1", "t01")
    with pytest.raises(_cli_reference.ReferencePlanStaleError, match="t01"):
        with _cli_reference.worker_reference_context(out, "plate1", pin=pins["t01"]):
            pass
    with _cli_reference.worker_reference_context(out, "plate1", pin=pins["t02"]) as active:
        assert active is not None


def _rgb(level: int, colony: bool) -> np.ndarray:
    image = np.full((32, 32, 3), level, dtype=np.uint8)
    if colony:
        image[10:18, 10:18] = 230
    return image


@pytest.fixture
def worker_run(tmp_path, monkeypatch):
    """One reference frame ready for the ordinary SLURM worker, startup already done."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    root = tmp_path / "images" / "plate1"
    root.mkdir(parents=True)
    tifffile.imwrite(root / "blank.tiff", _rgb(20, False))
    tifffile.imwrite(root / "t01.tiff", _rgb(20, True))
    table = tmp_path / "layout.csv"
    pd.DataFrame({"ImageName": ["t01"], "BlankImage": ["blank"]}).to_csv(table, index=False)
    pipeline = _pipeline_file(
        tmp_path,
        ImagePipeline(
            ops={"sb": SubtractBlank(), "det": OtsuDetector()}, meas={"size": MeasureSize()}
        ),
    )
    dataset = Dataset(
        name="plate1", images=[root / "t01.tiff"], input_dir=root,
        output_dir=tmp_path / "out" / "plate1",
    )
    return tmp_path, table, pipeline, dataset


_WORKER_MODES = {
    "full": (dict(), ["--mode", "full"]),
    "process": (
        dict(process_only_layer="detect_mat"),
        ["--mode", "process", "--layer", "detect_mat"],
    ),
}


def _start_worker(base, table, pipeline, dataset, mode):
    """Startup + selection as the submitter does them; returns the worker's argv."""
    overrides, mode_args = _WORKER_MODES[mode]
    out = base / "out"
    config = _config(
        base, pipeline, metadata_csv=table, save_overlays=False, **overrides
    )
    _cli_reference.publish_reference_inputs(config, [dataset], out)
    image = dataset.images[0]
    expected, _ = ft.work_id_for_image(config, "plate1", image)
    argv = [
        "--pipeline", str(pipeline), "--image", str(image), "--output-dir", str(out),
        "--dataset-name", "plate1", "--image-type", "Image",
        "--input-root", str(base / "images"), "--no-save-overlays", *mode_args,
        "--expected-work-id", expected,
        "--expected-input-sha256", ft.file_sha256(image),
        "--expected-pipeline-sha256", ft.file_sha256(pipeline),
    ]
    return out, expected, argv


@pytest.mark.parametrize("mode", ["full", "process"])
def test_the_slurm_worker_publishes_under_the_reference_work_id(worker_run, mode):
    """Review F6 (M4): the worker's identity check carries the manifest digest."""
    base, table, pipeline, dataset = worker_run
    out, expected, argv = _start_worker(base, table, pipeline, dataset, mode)
    result = CliRunner().invoke(_cli_process_single.main, argv)
    assert result.exit_code == 0, result.output
    record = json.loads(image_record_path(out, "plate1", "t01").read_text(encoding="utf-8"))
    assert record["work_id"] == expected


@pytest.mark.parametrize("mode", ["full", "process"])
def test_the_slurm_worker_refuses_a_plan_replaced_after_its_identity_check(
    worker_run, mode, monkeypatch
):
    """Review F1: a re-plan between identity and apply is refused, never certified."""
    base, table, pipeline, dataset = worker_run
    out, _, argv = _start_worker(base, table, pipeline, dataset, mode)
    identity = _cli_process_single._worker_work_identity

    def identity_then_replan(**kwargs):
        computed = identity(**kwargs)
        _replan(out, "plate1", "t01")
        return computed

    monkeypatch.setattr(_cli_process_single, "_worker_work_identity", identity_then_replan)
    result = CliRunner().invoke(_cli_process_single.main, argv)
    assert result.exit_code == 1, result.output
    assert "ReferencePlanStaleError" in result.output
    assert not image_record_path(out, "plate1", "t01").exists()
    # Not the image's fault: no terminal record, so the next run retries it (F2).
    failures = terminal_failures_jsonl_path(out)
    assert not failures.exists() or not failures.read_text(encoding="utf-8").strip()
