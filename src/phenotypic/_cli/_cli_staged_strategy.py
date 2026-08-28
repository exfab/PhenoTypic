"""Local staged GPU execution strategy (Spec 1 §6-§9).

Runs Stage 1 (preprocess -> OME-Zarr store) and Stage 3 (replay -> measure ->
re-promote) with joblib; Stage 2 keeps the detector model resident and streams
the staged stores to a retained raw ``.npy`` plus a consumable token under
``.phenotypic/progress/``. Stage 2 never writes into the store. Content-defined
resume uses a valid store, that Stage-2 pair, and an atomic Stage 3 publication
marker. Legacy parquet-only runs remain compatible.
"""

from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from joblib import Parallel, delayed

from phenotypic import ImagePipeline
from phenotypic.sdk_ import event_log_path, progress_dir, zarr_store_path
from phenotypic.sdk_._io_constants import GUI_RECORD_GENERATION_ENV_VAR

from ._cli_execution_strategies import (
    ExecutionStrategy,
    _publish_local_image_success,
    _record_local_terminal_failure,
)
from ._cli_pipeline_split import split_pipeline_at_gpu
from ._cli_completion import valid_image_success
from ._cli_failure_tracker import PerImageScientificError, work_id_for_image
from ._cli_stage2_token import (
    delete_stage2_raw,
    delete_stage2_token,
    stage2_result_replayable,
)
from ._cli_staged_resume import (
    clear_downstream_artifacts_for_stage1,
    stage3_completion_exists,
    staged_store_matches_work_id,
    valid_stage1_store,
    write_stage3_completion_marker,
)
from ._cli_staged_workers import (
    _image_class,
    emit_missing_prereq,
    ensure_staged_overlay,
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
    stage_event,
)
from ._stages import STAGE_GPU_DETECT, STAGE_MEASURE, STAGE_PREPROCESS
from ._cli_types import Dataset, DatasetResults, ExecutionResults

logger = logging.getLogger(__name__)


class StagedGpuStrategy(ExecutionStrategy):
    """Three-stage local GPU detection: preprocess -> detect -> measure."""

    def execute(
        self, datasets: List[Dataset], output_dir: Path
    ) -> ExecutionResults:
        start = datetime.now()
        cfg = self.config
        plan = split_pipeline_at_gpu(
            ImagePipeline.from_json(cfg.pipeline_json)
        )
        event_log = event_log_path(output_dir)
        tasks = [(ds, img) for ds in datasets for img in ds.images]

        read_kwargs: Dict[str, Any] = {}
        if cfg.bit_depth:
            read_kwargs["bit_depth"] = cfg.bit_depth
        if cfg.detect_mode != "gray":
            read_kwargs["detect_mode"] = cfg.detect_mode

        def _measurement_table_path(ds_name: str, stem: str) -> Path:
            from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

            return (
                zarr_store_path(output_dir, ds_name, stem)
                / MEASUREMENT_TABLE_RELATIVE_PATH
            )

        def _terminal_output_exists(ds_name: str, img: Path) -> bool:
            """Return whether the image has its durable terminal artifact."""
            work_id, _ = work_id_for_image(cfg, ds_name, img)
            if valid_image_success(
                output_dir,
                dataset=ds_name,
                image_stem=img.stem,
                work_id=work_id,
            ):
                return True
            store = zarr_store_path(output_dir, ds_name, img.stem)
            if (
                cfg.resume
                and stage3_completion_exists(output_dir, ds_name, img.stem)
                and staged_store_matches_work_id(store, work_id)
                and _measurement_table_path(ds_name, img.stem).is_file()
            ):
                ensure_staged_overlay(
                    output_dir,
                    ds_name,
                    img.stem,
                    self.output_manager,
                    cfg.image_type,
                )
                _publish_local_image_success(
                    cfg,
                    self.output_manager,
                    output_dir,
                    ds_name,
                    img,
                    uuid4().hex,
                )
                return True
            if cfg.staged_stage3_markers:
                return False
            if cfg.process_only_layer == "objmap":
                from ._cli_process_only import process_only_output_path

                return process_only_output_path(
                    output_dir, img, cfg.input_path, "objmap", fmt="tiff"
                ).is_file()
            terminal = stage3_completion_exists(
                output_dir, ds_name, img.stem
            ) or bool(
                cfg.resume
                and not cfg.staged_stage3_markers
                and _measurement_table_path(ds_name, img.stem).is_file()
            )
            if terminal:
                ensure_staged_overlay(
                    output_dir,
                    ds_name,
                    img.stem,
                    self.output_manager,
                    cfg.image_type,
                )
                return True
            return False

        # ---- Stage 1: CPU preprocess -> staged store (parallel, resumable) --
        def _stage1(ds: Dataset, img: Path) -> None:
            store = zarr_store_path(output_dir, ds.name, img.stem)
            work_id, _ = work_id_for_image(cfg, ds.name, img)
            if cfg.resume and staged_store_matches_work_id(store, work_id):
                return
            if cfg.resume:
                clear_downstream_artifacts_for_stage1(
                    output_dir, ds.name, img.stem
                )
            attempt_id = uuid4().hex
            try:  # isolate one bad image from the batch (failed event logged)
                with stage_event(
                    event_log, ds.name, img.name, STAGE_PREPROCESS
                ):
                    stage1_preprocess_core(
                        plan,
                        img,
                        ds.name,
                        img.stem,
                        output_dir,
                        self.output_manager,
                        cfg.image_type,
                        read_kwargs,
                        work_id=work_id,
                        pipeline_path=cfg.pipeline_json,
                        pipeline_identity=getattr(
                            cfg, "pipeline_identity", None
                        ),
                        drop_originals=cfg.drop_originals,
                    )
            except Exception as exc:
                _record_local_terminal_failure(
                    cfg,
                    output_dir,
                    ds.name,
                    img,
                    exc,
                    traceback.format_exc(),
                    attempt_id,
                )

        Parallel(n_jobs=cfg.n_jobs)(
            delayed(_stage1)(ds, img) for ds, img in tasks
        )

        # ---- Stage 2: resident-model GPU detect -> raw + token (serial) ----
        stage2_pending = [
            (ds, img)
            for ds, img in tasks
            # BOTH halves: a token whose raw array is gone is not a Stage-2
            # result, and re-running Stage 2 is the only thing that recovers it.
            if not stage2_result_replayable(output_dir, ds.name, img.stem)
            and not _terminal_output_exists(ds.name, img)
        ]
        if stage2_pending:
            plan.gpu_detector._ensure_model_loaded()  # load ONCE
        for ds, img in stage2_pending:
            store = zarr_store_path(output_dir, ds.name, img.stem)
            if not valid_stage1_store(store):
                # Stage 1 failed/absent for this image (S6): skip + record. A
                # cascade (stage1 failed -> stage2/stage3 prereq missing)
                # deliberately records a failed event per stage so the per-stage
                # view shows where each image is blocked; overall totals still
                # count the image exactly once (via Stage 3's return value).
                emit_missing_prereq(
                    event_log,
                    ds.name,
                    img.name,
                    STAGE_GPU_DETECT,
                    "staged store",
                )
                continue
            attempt_id = uuid4().hex
            try:
                with stage_event(
                    event_log, ds.name, img.name, STAGE_GPU_DETECT
                ):
                    stage2_detect_core(
                        plan.gpu_detector,
                        output_dir,
                        ds.name,
                        img.stem,
                        cfg.image_type,
                    )
            except Exception as exc:
                _record_local_terminal_failure(
                    cfg,
                    output_dir,
                    ds.name,
                    img,
                    exc,
                    traceback.format_exc(),
                    attempt_id,
                )

        # ---- Stage 3: CPU merge + measure (parallel, resumable) ------------
        results: Dict[str, Dict[str, int]] = {
            ds.name: {"total": len(ds.images), "completed": 0, "failed": 0}
            for ds in datasets
        }

        def _stage3(ds: Dataset, img: Path) -> tuple[str, bool]:
            if cfg.resume and _terminal_output_exists(ds.name, img):
                return ds.name, True
            if not stage2_result_replayable(output_dir, ds.name, img.stem):
                # Stage 2 failed/absent for this image (S6): skip + record.
                emit_missing_prereq(
                    event_log,
                    ds.name,
                    img.name,
                    STAGE_MEASURE,
                    "Stage 2 result",
                )
                return ds.name, False
            attempt_id = uuid4().hex
            work_id, _ = work_id_for_image(cfg, ds.name, img)
            try:
                with stage_event(event_log, ds.name, img.name, STAGE_MEASURE):
                    stage3_merge_measure_core(
                        plan,
                        output_dir,
                        ds.name,
                        img.stem,
                        self.output_manager,
                        cfg.image_type,
                        image_name=img.name,
                        work_id=work_id,
                    )
                    _publish_local_image_success(
                        cfg,
                        self.output_manager,
                        output_dir,
                        ds.name,
                        img,
                        attempt_id,
                    )
                    write_stage3_completion_marker(
                        output_dir, ds.name, img.name, img.stem
                    )
                    # Token FIRST: the reachable intermediate state must be
                    # "no token, orphan raw" (inert), never "token present,
                    # raw missing" (Stage 3 replays into FileNotFoundError).
                    delete_stage2_token(output_dir, ds.name, img.stem)
                    delete_stage2_raw(output_dir, ds.name, img.stem)
                return ds.name, True
            except Exception as exc:
                _record_local_terminal_failure(
                    cfg,
                    output_dir,
                    ds.name,
                    img,
                    exc,
                    traceback.format_exc(),
                    attempt_id,
                )
                return ds.name, False

        if cfg.process_only_layer == "objmap":
            # process-mode: export the objmap layer (mirrored), no measurement.
            self._export_objmap_layer(
                plan, tasks, output_dir, event_log, results
            )
        else:
            for ds_name, ok in Parallel(n_jobs=cfg.n_jobs)(
                delayed(_stage3)(ds, img) for ds, img in tasks
            ):
                results[ds_name]["completed" if ok else "failed"] += 1

        ds_results = {
            name: DatasetResults(
                name=name,
                total=d["total"],
                completed=d["completed"],
                failed=d["failed"],
                failures=[],
            )
            for name, d in results.items()
        }
        try:
            from ._dashboard._manifest_builder import build_manifest

            datasets_inventory = cfg.full_dataset_inventory or {
                dataset.name: [image.name for image in dataset.images]
                for dataset in datasets
            }
            build_manifest(
                output_dir=output_dir,
                progress_dir=progress_dir(output_dir),
                datasets={
                    name: len(images)
                    for name, images in datasets_inventory.items()
                },
                execution_mode="local",
                start_time=start.isoformat(timespec="milliseconds"),
                input_path=cfg.input_path.stem,
                gui_record_generation=os.environ.get(
                    GUI_RECORD_GENERATION_ENV_VAR
                ),
                dataset_inventory=datasets_inventory,
                processing_generation=cfg.processing_generation,
            )
        except Exception:
            # Match the ordinary local strategy: process results remain
            # available, but the GUI completion publisher will fail closed
            # unless this exact-generation manifest was atomically replaced.
            logger.debug(
                "Failed to generate staged-local progress manifest",
                exc_info=True,
            )
        return ExecutionResults(
            datasets=ds_results,
            total_images=len(tasks),
            total_completed=sum(r.completed for r in ds_results.values()),
            total_failed=sum(r.failed for r in ds_results.values()),
            execution_mode="local",
            start_time=start,
            end_time=datetime.now(),
        )

    def _export_objmap_layer(
        self,
        plan,
        tasks: List[tuple[Dataset, Path]],
        output_dir: Path,
        event_log: Path,
        results: Dict[str, Dict[str, int]],
    ) -> None:
        """``--mode process --layer objmap``: replay Stage 2's raw result and
        write the objmap layer (mirrored) — no measurement (Spec 1 §6). Runs
        after Stages 1-2; consumes the token and the raw array after export.

        The merge reads :func:`load_stage2_raw`, **not the store** (ledger
        **FLOW-16**). Stage 2 never writes into the store, so the store's
        objmap here is still Stage 1's zeros; a store read would export an
        all-zeros PNG for every image, silently.

        Nothing is restored or re-promoted afterwards. ``_write_object_output``
        mutates only the in-memory image, so the residue left on disk is Stage
        1's zeros — exactly what the HDF path left. A store write placed after
        ``_publish_local_image_success`` would rewrite ``zarr.json`` and
        invalidate the descriptor the marker just recorded (ledger
        **FLOW-30**/**FLOW-6**).
        """
        from ._cli_process_only import (
            process_only_output_path,
            write_process_only_layer,
        )
        from ._cli_stage2_token import load_stage2_raw

        cfg = self.config
        image_cls = _image_class(cfg.image_type)
        for ds, img in tasks:
            out_path = process_only_output_path(
                output_dir, img, cfg.input_path, "objmap", fmt="tiff"
            )
            work_id, _ = work_id_for_image(cfg, ds.name, img)
            if cfg.resume and valid_image_success(
                output_dir,
                dataset=ds.name,
                image_stem=img.stem,
                work_id=work_id,
            ):
                results[ds.name]["completed"] += 1
                continue
            if not stage2_result_replayable(output_dir, ds.name, img.stem):
                emit_missing_prereq(
                    event_log,
                    ds.name,
                    img.name,
                    STAGE_MEASURE,
                    "Stage 2 result",
                )
                results[ds.name]["failed"] += 1
                continue
            attempt_id = uuid4().hex
            try:
                with stage_event(event_log, ds.name, img.name, STAGE_MEASURE):
                    store = zarr_store_path(output_dir, ds.name, img.stem)
                    image = image_cls.load_zarr(store)
                    raw = load_stage2_raw(output_dir, ds.name, img.stem)
                    try:
                        plan.gpu_detector._write_object_output(image, raw)
                    except MemoryError:
                        raise
                    except Exception as exc:
                        raise PerImageScientificError(
                            STAGE_MEASURE, exc
                        ) from exc
                    write_process_only_layer(image, "objmap", out_path)
                    _publish_local_image_success(
                        cfg,
                        self.output_manager,
                        output_dir,
                        ds.name,
                        img,
                        attempt_id,
                    )
                    # Ordering (ledger FLOW-6): publish, then token, then raw.
                    delete_stage2_token(output_dir, ds.name, img.stem)
                    delete_stage2_raw(output_dir, ds.name, img.stem)
                results[ds.name]["completed"] += 1
            except Exception as exc:
                _record_local_terminal_failure(
                    cfg,
                    output_dir,
                    ds.name,
                    img,
                    exc,
                    traceback.format_exc(),
                    attempt_id,
                )
                results[ds.name]["failed"] += 1
