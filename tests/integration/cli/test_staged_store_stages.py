"""Stage 1/2/3 against a real store. The post-refined objmap test is the point."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import phenotypic
import pytest

from phenotypic import Image
from phenotypic._cli._cli_stage2_token import (
    read_stage2_token,
    stage2_token_exists,
)
from phenotypic.schema import OBJECT
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_.ngff_ import valid_staged_store

#: The measurement column is ``Object_Label``, not ``ObjectLabel`` --
#: ``schema/_object.py`` with ``category() == "Object"``. Resolve it through the
#: schema rather than spelling it, so a rename cannot silently turn the most
#: load-bearing test in this plan into a KeyError.


def _journal(store):
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    return root["attributes"]["phenotypic"]["provenance"]


def _application(journal):
    return journal["applications"][-1]


def test_stage1_publishes_a_store_with_a_zeros_objmap(staged_run) -> None:
    """valid_staged_store requires objmap; Stage 1 must emit it, zeros and all."""
    staged_run.run_stage1()
    store = zarr_store_path(staged_run.output_dir, "ds", "img")
    assert valid_staged_store(store) is True
    assert (Image.load_layer_zarr(store, "objmap") == 0).all()


def test_stage1_store_conforms(staged_run) -> None:
    from tests._ngff_conformance import assert_store_conforms

    staged_run.run_stage1()
    assert_store_conforms(zarr_store_path(staged_run.output_dir, "ds", "img"))


def test_stage2_never_touches_the_store(staged_run) -> None:
    """Only the FINAL store needs interop, so Stage 2 leaves it alone.

    Pins the user ruling that dissolved FLOW-5: with nothing written here, no
    reader -- cached tile route, uncached crop route, or third-party -- can
    ever observe raw pre-drop_frame_background labels.
    """
    staged_run.run_stage1()
    store = zarr_store_path(staged_run.output_dir, "ds", "img")
    before = (store / "zarr.json").read_bytes()
    zeros = Image.load_layer_zarr(store, "objmap")

    staged_run.run_stage2()

    assert (store / "zarr.json").read_bytes() == before
    np.testing.assert_array_equal(Image.load_layer_zarr(store, "objmap"), zeros)
    assert not zeros.any(), "Stage 1 writes zeros; Stage 3 publishes the real objmap"


def test_stage2_drops_a_token_and_retains_the_raw_array(staged_run) -> None:
    from phenotypic._cli._cli_stage2_token import load_stage2_raw, stage2_raw_path

    staged_run.run_stage1()
    staged_run.run_stage2()
    assert stage2_token_exists(staged_run.output_dir, "ds", "img") is True
    assert stage2_raw_path(staged_run.output_dir, "ds", "img").is_file()
    assert load_stage2_raw(staged_run.output_dir, "ds", "img").any()


def test_staged_provenance_original_and_retry_are_durable(
    staged_run_with_provenance,
) -> None:
    run = staged_run_with_provenance
    run.run_stage1()
    store = run.store()
    staged = _journal(store)
    assert staged["status"] == "staged"
    assert _application(staged)["retry_base_length"] == 1
    assert _application(staged)["pipeline"] == {
        "source_path": run.pipeline_path.name,
        "sha256": hashlib.sha256(run.pipeline_path.read_bytes()).hexdigest(),
    }
    assert [entry["operation_name"] for entry in _application(staged)["operations"]] == [
        "CropImage"
    ]
    assert [entry["pipeline_step_path"] for entry in _application(staged)["operations"]] == [
        ["pre-crop"]
    ]
    staged_image = Image.load_zarr(store)
    assert staged_image._original is not None
    original = staged_image._original.copy()
    assert original.shape[:2] != staged_image.shape[:2]
    ome = json.loads((store / "OME" / "zarr.json").read_text(encoding="utf-8"))
    assert ome["attributes"]["ome"]["series"][-1] == "original"

    before_stage2 = (store / "zarr.json").read_bytes()
    run.run_stage2()
    token = read_stage2_token(run.output_dir, "ds", "img")
    compute_duration = token["detector_duration_seconds"]
    assert compute_duration >= 0
    assert (store / "zarr.json").read_bytes() == before_stage2
    assert _journal(store) == staged

    run.run_stage3()
    completed = _journal(store)
    expected_names = ["CropImage", "_FixedBlobDetector", "SmallObjectRemover"]
    expected_paths = [["pre-crop"], ["gpu-detect"], ["post-filter"]]
    assert completed["status"] == "complete"
    assert _application(completed)["retry_base_length"] == 1
    assert [entry["operation_name"] for entry in _application(completed)["operations"]] == (
        expected_names
    )
    assert [entry["pipeline_step_path"] for entry in _application(completed)["operations"]] == (
        expected_paths
    )
    assert [entry["sequence"] for entry in _application(completed)["operations"]] == [1, 2, 3]
    assert _application(completed)["operations"][1]["duration_seconds"] >= compute_duration
    np.testing.assert_array_equal(Image.load_zarr(store)._original, original)

    run.simulate_timeout_after_promote()
    run.run_stage3()
    retried = _journal(store)
    assert retried["status"] == "complete"
    assert _application(retried)["retry_base_length"] == 1
    assert [entry["operation_name"] for entry in _application(retried)["operations"]] == (
        expected_names
    )
    assert [entry["pipeline_step_path"] for entry in _application(retried)["operations"]] == (
        expected_paths
    )
    assert [entry["sequence"] for entry in _application(retried)["operations"]] == [1, 2, 3]
    np.testing.assert_array_equal(Image.load_zarr(store)._original, original)


def test_stage1_hard_interruption_is_retried_from_the_decoded_checkpoint(
    staged_run_with_provenance, monkeypatch
) -> None:
    from phenotypic._cli._cli_staged_resume import classify_staged_image

    class _HardStop(BaseException):
        pass

    run = staged_run_with_provenance
    run.work_id = "provenance-retry-work-id"
    operation = run.plan.pre_pipeline.get_ops()["pre-crop"]
    operation_type = type(operation)
    real_operate = operation_type._operate

    def _stop(*args, **kwargs):
        del args, kwargs
        raise _HardStop()

    monkeypatch.setattr(operation_type, "_operate", _stop)
    with pytest.raises(_HardStop):
        run.run_stage1()

    interrupted = _journal(run.store())
    interrupted_version = _application(interrupted)["phenotypic_version"]
    assert interrupted["status"] == "in_progress"
    assert _application(interrupted)["input_filename"] == run.image_path.name
    assert _application(interrupted)["operations"] == []
    assert classify_staged_image(
        output_dir=run.output_dir,
        dataset="ds",
        image=run.image_path,
        input_root=run.image_path.parent,
        process_only_layer=None,
        markers_required=True,
    ) == "stage1"

    monkeypatch.setattr(operation_type, "_operate", real_operate)
    monkeypatch.setattr(phenotypic, "__version__", "retry-build-sentinel")
    run.run_stage1()
    retried = _journal(run.store())
    assert retried["status"] == "staged"
    assert len(retried["applications"]) == 1
    assert _application(retried)["phenotypic_version"] == interrupted_version
    assert _application(retried)["retry_base_length"] == 1
    assert [entry["operation_name"] for entry in _application(retried)["operations"]] == [
        "CropImage"
    ]


def test_staged_drop_originals_uses_journal_only_checkpoint_and_omits_series(
    staged_run_with_provenance, monkeypatch
) -> None:
    class _HardStop(BaseException):
        pass

    run = staged_run_with_provenance
    run.work_id = "provenance-retry-work-id"
    operation = run.plan.pre_pipeline.get_ops()["pre-crop"]
    operation_type = type(operation)
    real_operate = operation_type._operate

    def _stop(*args, **kwargs):
        del args, kwargs
        raise _HardStop()

    monkeypatch.setattr(operation_type, "_operate", _stop)
    with pytest.raises(_HardStop):
        run.run_stage1(drop_originals=True)

    store = run.store()
    interrupted = _journal(store)
    assert interrupted["status"] == "in_progress"
    assert _application(interrupted)["operations"] == []
    assert not (store / "OME").exists()
    assert not (store / "original").exists()

    monkeypatch.setattr(operation_type, "_operate", real_operate)
    run.run_stage1(drop_originals=True)
    staged = _journal(store)
    assert staged["status"] == "staged"
    assert _application(staged)["retry_base_length"] == 1
    assert Image.load_zarr(store)._original is None
    ome = json.loads((store / "OME" / "zarr.json").read_text(encoding="utf-8"))
    assert "original" not in ome["attributes"]["ome"]["series"]

    run.run_stage2()
    run.run_stage3()
    completed = _journal(store)
    assert completed["status"] == "complete"
    assert [entry["operation_name"] for entry in _application(completed)["operations"]] == [
        "CropImage",
        "_FixedBlobDetector",
        "SmallObjectRemover",
    ]
    assert Image.load_zarr(store)._original is None
    final_ome = json.loads(
        (store / "OME" / "zarr.json").read_text(encoding="utf-8")
    )
    assert "original" not in final_ome["attributes"]["ome"]["series"]


def test_stage3_publishes_the_post_refined_objmap(staged_run_with_size_filter) -> None:
    """The round-trip test is blind to this: it never goes through the stages.

    Post-ops mutate the objmap. Without Stage 3's re-promote the stored label
    image holds raw detector output that disagrees with the parquet.
    """
    from phenotypic._cli._cli_stage2_token import load_stage2_raw

    run = staged_run_with_size_filter  # post-op removes exactly one colony
    run.run_stage1()
    run.run_stage2()
    # From the RAW ARRAY, not the store. Stage 2 does not write into the store
    # (Task 3.3), so at this point the store's objmap is still Stage 1's zeros --
    # sourcing raw_labels from it would make the set empty and the final
    # `published < raw_labels` assertion vacuously False for any real result.
    # Ledger FLOW-14.
    raw_labels = set(np.unique(load_stage2_raw(run.output_dir, "ds", "img"))) - {0}
    assert raw_labels, "fixture must produce detections before post-ops run"
    run.run_stage3()
    published = set(
        np.unique(
            Image.load_layer_zarr(
                zarr_store_path(run.output_dir, "ds", "img"), "objmap"
            )
        )
    ) - {0}
    parquet_labels = set(run.read_measurements()[str(OBJECT.LABEL)].to_list())
    assert published == parquet_labels
    assert published < raw_labels, "the size filter should have removed a colony"


def test_stage3_consumes_the_token_and_the_raw_array(staged_run) -> None:
    from phenotypic._cli._cli_stage2_token import stage2_raw_path

    staged_run.run_stage1()
    staged_run.run_stage2()
    staged_run.run_stage3()
    assert stage2_token_exists(staged_run.output_dir, "ds", "img") is False
    assert not stage2_raw_path(staged_run.output_dir, "ds", "img").exists()


def test_stage3_is_idempotent_under_retry(staged_run_with_border_colony) -> None:
    """The D1 guard. A timeout between the promote and the completion marker
    leaves the classifier reading "stage3", so Stage 3 runs a second time.
    Replaying from the retained raw array must produce an identical result.

    Replaying from the STORE instead re-runs _write_object_output on
    already-refined labels, and drop_frame_background then zeroes whichever
    real colony touches the frame most -- silently, once per retry.
    """
    run = staged_run_with_border_colony  # a colony provably touches the frame
    run.run_stage1()
    run.run_stage2()
    run.run_stage3()
    store = zarr_store_path(run.output_dir, "ds", "img")
    once = Image.load_layer_zarr(store, "objmap").copy()
    measurements_once = run.read_measurements()
    assert set(np.unique(once)) - {0}, "the first pass must publish real labels"

    run.simulate_timeout_after_promote()  # removes the marker, keeps token + raw
    run.run_stage3()

    np.testing.assert_array_equal(Image.load_layer_zarr(store, "objmap"), once)
    assert set(run.read_measurements()[str(OBJECT.LABEL)].to_list()) == set(
        measurements_once[str(OBJECT.LABEL)].to_list()
    )


def test_stage3_replays_from_the_raw_array_not_the_store(
    staged_run, monkeypatch
) -> None:
    """Pins the input source, so a later 'simplification' cannot swap it back.

    Substituting the loader with one that returns a distinct array makes the
    source observable in the PUBLISHED objmap: if Stage 3 read the store
    instead, the substitution would have no effect and the published labels
    would be Stage 1's zeros. Asserting on the value rather than only on a
    call count keeps the guard alive across an import-style refactor.
    """
    from phenotypic._cli import _cli_staged_workers

    staged_run.run_stage1()
    staged_run.run_stage2()
    sentinel = np.zeros((600, 800), dtype=np.uint16)
    sentinel[200:230, 300:330] = 7
    calls: list[tuple] = []

    def _substitute(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(_cli_staged_workers, "load_stage2_raw", _substitute)
    staged_run.run_stage3()

    published = Image.load_layer_zarr(
        zarr_store_path(staged_run.output_dir, "ds", "img"), "objmap"
    )
    assert len(calls) == 1
    assert set(np.unique(published)) - {0} == {7}


def test_stage3_leaves_the_token_alone_on_the_work_id_path(
    staged_run_with_work_id,
) -> None:
    """Preserves today's guard: with a work_id, markers are published by the
    SLURM worker, not here. Making this unconditional double-deletes."""
    from phenotypic._cli._cli_stage2_token import stage2_raw_path
    from phenotypic._cli._cli_staged_resume import stage3_completion_exists

    run = staged_run_with_work_id
    run.run_stage1()
    run.run_stage2()
    run.run_stage3()
    assert stage2_token_exists(run.output_dir, "ds", "img") is True
    assert stage2_raw_path(run.output_dir, "ds", "img").is_file()
    assert stage3_completion_exists(run.output_dir, "ds", "img") is False


def test_stage3_republished_store_still_conforms(staged_run) -> None:
    from tests._ngff_conformance import assert_store_conforms

    staged_run.run_stage1()
    staged_run.run_stage2()
    staged_run.run_stage3()
    assert_store_conforms(zarr_store_path(staged_run.output_dir, "ds", "img"))


def test_stage3_raises_when_publication_fails(staged_run, monkeypatch) -> None:
    staged_run.run_stage1()
    staged_run.run_stage2()
    monkeypatch.setattr(
        "phenotypic._cli._cli_output_manager.OutputManager.save_image_store",
        lambda *a, **k: None,
    )
    with pytest.raises(RuntimeError, match="Stage 3"):
        staged_run.run_stage3()
    assert _journal(staged_run.store())["status"] == "failed"
