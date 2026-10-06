"""Startup publishes the reference plan; work-ids carry the per-image digest."""

from __future__ import annotations

import inspect
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
    reference_manifest_path,
    reference_metadata_snapshot_path,
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
        output_dir=output_dir,
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
