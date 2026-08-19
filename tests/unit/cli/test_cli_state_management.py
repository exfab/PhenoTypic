"""Tests for processing-state IO routing through the ``.phenotypic`` cache."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from phenotypic._cli._cli_state_management import (
    get_remaining_images_for_datasets,
    load_processing_state,
    save_processing_state,
    update_state_from_events,
    validate_resume_compatibility,
)
from phenotypic._cli._cli_types import (
    Dataset,
    DatasetState,
    ExecutionConfig,
    ProcessingState,
)
from phenotypic._cli._cli_staged_resume import pipeline_content_digest
from phenotypic._cli._cli_update_state import append_completion_event
from phenotypic.sdk_ import event_log_path, processing_state_path


def _make_state(out: Path) -> ProcessingState:
    now = datetime(2026, 6, 3, 12, 0, 0)
    return ProcessingState(
        version="2",
        pipeline_path=Path("p.json"),
        input_path=Path("in"),
        output_dir=out,
        timestamp=now,
        execution_mode="local",
        last_updated=now,
        datasets={},
        config={},
    )


def test_save_writes_under_phenotypic(tmp_path: Path) -> None:
    save_processing_state(_make_state(tmp_path), tmp_path)
    assert processing_state_path(tmp_path).is_file()
    assert not (tmp_path / "processing_state.json").exists()


def test_load_migrates_and_reads_legacy_run(tmp_path: Path) -> None:
    # Simulate a pre-migration run: state at the output root.
    save_processing_state(_make_state(tmp_path), tmp_path)
    new = processing_state_path(tmp_path)
    legacy = tmp_path / "processing_state.json"
    (tmp_path / ".phenotypic").rename(tmp_path / "_tmp")  # extract
    (tmp_path / "_tmp" / "processing_state.json").rename(legacy)
    (tmp_path / "_tmp" / ".processing_state.json.lock").unlink()
    (tmp_path / "_tmp").rmdir()
    assert legacy.is_file() and not new.exists()
    loaded = load_processing_state(tmp_path)
    assert loaded is not None
    # migrate-on-load moved it into .phenotypic
    assert new.is_file()


def test_resume_rejects_process_only_layer_change(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.config = {
        "image_type": "GridImage",
        "nrows": None,
        "ncols": None,
        "process_only_layer": "detect_mat",
    }
    config = ExecutionConfig(
        pipeline_json=Path("p.json"),
        input_path=Path("in"),
        output_dir=tmp_path,
        image_type="GridImage",
        nrows=None,
        ncols=None,
        bit_depth=None,
        n_jobs=1,
        slurm_args={},
        force_local=True,
        wait=False,
        ext=".tiff",
        overlay_alpha=0.3,
        include_dataset_column=True,
        dry_run=False,
        sample=None,
        resume=True,
        retry_failures=False,
        skip_validation=True,
        process_only_layer="rgb",
    )

    compatible, error = validate_resume_compatibility(state, config)

    assert compatible is False
    assert error is not None
    assert "Process-only layer mismatch" in error


def test_event_refresh_preserves_original_image_inventory(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.config["processing_generation"] = "current"
    state.datasets["plate"] = DatasetState(
        initial_images={"a.tif", "b.tif"}
    )
    append_completion_event(
        event_log_path(tmp_path),
        "plate",
        "a.tif",
        "completed",
        generation="previous",
    )
    append_completion_event(
        event_log_path(tmp_path),
        "plate",
        "a.tif",
        "completed",
        stage="stage1",
        generation="current",
    )

    refreshed = update_state_from_events(state, tmp_path)

    assert refreshed.datasets["plate"].initial_images == {"a.tif", "b.tif"}
    assert refreshed.datasets["plate"].completed == set()
    assert refreshed.datasets["plate"].in_progress == {"a.tif"}
    save_processing_state(refreshed, tmp_path)
    loaded = load_processing_state(tmp_path)
    assert loaded is not None
    assert loaded.datasets["plate"].initial_images == {"a.tif", "b.tif"}


def test_cpu_resume_still_requires_retry_failures_for_failed_images(
    tmp_path: Path,
) -> None:
    image = tmp_path / "failed.tif"
    image.touch()
    dataset = Dataset("plate", [image], tmp_path, tmp_path / "out")
    state = _make_state(tmp_path)
    state.datasets["plate"] = DatasetState(failed={"failed.tif"})

    assert get_remaining_images_for_datasets(state, [dataset]) == []
    retry = get_remaining_images_for_datasets(
        state, [dataset], retry_failures=True
    )
    assert retry[0].images == [image]


def test_resume_rejects_changed_pipeline_contents(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text('{"operations": []}', encoding="utf-8")
    state = _make_state(tmp_path)
    state.pipeline_path = pipeline
    state.input_path = tmp_path / "images"
    state.config = {
        "image_type": "Image",
        "pipeline_sha256": pipeline_content_digest(pipeline),
        "process_only_layer": None,
    }
    config = SimpleNamespace(
        pipeline_json=pipeline,
        input_path=state.input_path,
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        detect_mode="gray",
        process_only_layer=None,
    )
    pipeline.write_text('{"operations": [{"changed": true}]}', encoding="utf-8")

    compatible, error = validate_resume_compatibility(state, config)

    assert compatible is False
    assert error == "Pipeline contents changed since the original run"


def test_digest_resume_accepts_same_pipeline_from_new_path(tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    replacement = tmp_path / "replacement.json"
    original.write_text('{"operations": []}', encoding="utf-8")
    replacement.write_bytes(original.read_bytes())
    state = _make_state(tmp_path)
    state.pipeline_path = original
    state.input_path = tmp_path / "images"
    state.config = {
        "image_type": "Image",
        "pipeline_sha256": pipeline_content_digest(original),
        "process_only_layer": None,
    }
    config = SimpleNamespace(
        pipeline_json=replacement,
        input_path=state.input_path,
        # Read unconditionally rather than through the tolerant
        # "skip when the saved state lacks the key" loop, because absent has
        # to mean "no manifest" for a state that predates --image-manifest.
        image_manifest=None,
        image_type="Image",
        nrows=None,
        ncols=None,
        process_only_layer=None,
    )

    assert validate_resume_compatibility(state, config) == (True, None)


def test_resume_rejects_changed_artifact_shaping_setting(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text('{"operations": []}', encoding="utf-8")
    state = _make_state(tmp_path)
    state.pipeline_path = pipeline
    state.input_path = tmp_path / "images"
    state.config = {
        "image_type": "Image",
        "pipeline_sha256": pipeline_content_digest(pipeline),
        "process_only_layer": None,
        "include_dataset_column": True,
    }
    config = SimpleNamespace(
        pipeline_json=pipeline,
        input_path=state.input_path,
        image_manifest=None,
        image_type="Image",
        nrows=None,
        ncols=None,
        process_only_layer=None,
        include_dataset_column=False,
    )

    compatible, error = validate_resume_compatibility(state, config)

    assert compatible is False
    assert error == (
        "include_dataset_column mismatch: saved=True, current=False"
    )
