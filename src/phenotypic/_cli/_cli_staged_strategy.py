"""Local staged GPU execution strategy (Spec 1 §6-§9).

Runs Stage 1 (preprocess -> HDF) and Stage 3 (merge -> measure) with joblib;
Stage 2 keeps the detector model resident and streams the staged HDFs to
sidecars. Content-defined resume: an image's work is done when its measurement
parquet exists; Stage 2 is skipped when the sidecar OR the parquet already
exists (Stage 3 deletes the sidecar, so "parquet exists" is the durable marker).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from joblib import Parallel, delayed

from phenotypic import ImagePipeline
from phenotypic.sdk_ import dataset_hdf_dir, event_log_path

from ._cli_execution_strategies import ExecutionStrategy
from ._cli_pipeline_split import split_pipeline_at_gpu
from ._cli_sidecar import sidecar_exists
from ._cli_staged_workers import (
    _image_class,
    emit_missing_prereq,
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
    stage_event,
)
from ._stages import STAGE_GPU_DETECT, STAGE_MEASURE, STAGE_PREPROCESS
from ._cli_types import Dataset, DatasetResults, ExecutionResults


class StagedGpuStrategy(ExecutionStrategy):
    """Three-stage local GPU detection: preprocess -> detect -> measure."""

    def execute(
        self, datasets: List[Dataset], output_dir: Path
    ) -> ExecutionResults:
        start = datetime.now()
        cfg = self.config
        plan = split_pipeline_at_gpu(ImagePipeline.from_json(cfg.pipeline_json))
        event_log = event_log_path(output_dir)
        tasks = [(ds, img) for ds in datasets for img in ds.images]

        read_kwargs: Dict[str, Any] = {}
        if cfg.bit_depth:
            read_kwargs["bit_depth"] = cfg.bit_depth
        if cfg.detect_mode != "gray":
            read_kwargs["detect_mode"] = cfg.detect_mode

        def _parquet_path(ds_name: str, stem: str) -> Path:
            return self.output_manager.get_output_path(
                ds_name, "measurements", stem
            )

        def _terminal_output_exists(ds_name: str, img: Path) -> bool:
            """Durable "fully done" marker for Stage 2 resume: the run's terminal
            artifact survives a completed run even though Stage 3 deletes the
            sidecar. Full mode -> measurement parquet; objmap process-mode ->
            the exported objmap layer file (objmap mode writes no parquet).
            """
            if cfg.process_only_layer == "objmap":
                from ._cli_process_only import process_only_output_path

                return process_only_output_path(
                    output_dir, img, cfg.input_path, "objmap"
                ).is_file()
            return _parquet_path(ds_name, img.stem).is_file()

        # ---- Stage 1: CPU preprocess -> staged HDF (parallel, resumable) ----
        def _stage1(ds: Dataset, img: Path) -> None:
            hdf = dataset_hdf_dir(output_dir, ds.name) / f"{img.stem}.h5"
            if cfg.resume and hdf.is_file():
                return
            try:  # isolate one bad image from the batch (failed event logged)
                with stage_event(event_log, ds.name, img.name, STAGE_PREPROCESS):
                    stage1_preprocess_core(
                        plan, img, ds.name, img.stem, output_dir,
                        self.output_manager, cfg.image_type, read_kwargs,
                    )
            except Exception:
                pass

        Parallel(n_jobs=cfg.n_jobs)(
            delayed(_stage1)(ds, img) for ds, img in tasks
        )

        # ---- Stage 2: resident-model GPU detect -> sidecar (serial) --------
        plan.gpu_detector._ensure_model_loaded()  # load ONCE
        for ds, img in tasks:
            hdf = dataset_hdf_dir(output_dir, ds.name) / f"{img.stem}.h5"
            if cfg.resume and (
                sidecar_exists(output_dir, ds.name, img.stem)
                or _terminal_output_exists(ds.name, img)
            ):
                continue
            if not hdf.is_file():
                # Stage 1 failed/absent for this image (S6): skip + record. A
                # cascade (stage1 failed -> stage2/stage3 prereq missing)
                # deliberately records a failed event per stage so the per-stage
                # view shows where each image is blocked; overall totals still
                # count the image exactly once (via Stage 3's return value).
                emit_missing_prereq(
                    event_log, ds.name, img.name, STAGE_GPU_DETECT, "staged HDF"
                )
                continue
            try:
                with stage_event(event_log, ds.name, img.name, STAGE_GPU_DETECT):
                    stage2_detect_core(
                        plan.gpu_detector, output_dir, ds.name, img.stem,
                        cfg.image_type,
                    )
            except Exception:
                pass

        # ---- Stage 3: CPU merge + measure (parallel, resumable) ------------
        results: Dict[str, Dict[str, int]] = {
            ds.name: {"total": len(ds.images), "completed": 0, "failed": 0}
            for ds in datasets
        }

        def _stage3(ds: Dataset, img: Path) -> tuple[str, bool]:
            parquet = _parquet_path(ds.name, img.stem)
            if cfg.resume and parquet.is_file():
                return ds.name, True
            if not sidecar_exists(output_dir, ds.name, img.stem):
                # Stage 2 failed/absent for this image (S6): skip + record.
                emit_missing_prereq(
                    event_log, ds.name, img.name, STAGE_MEASURE, "objmap sidecar"
                )
                return ds.name, False
            try:
                with stage_event(event_log, ds.name, img.name, STAGE_MEASURE):
                    stage3_merge_measure_core(
                        plan, output_dir, ds.name, img.stem, self.output_manager,
                        cfg.image_type,
                    )
                return ds.name, True
            except Exception:
                return ds.name, False

        if cfg.process_only_layer == "objmap":
            # process-mode: export the objmap layer (mirrored), no measurement.
            self._export_objmap_layer(plan, tasks, output_dir, event_log, results)
        else:
            for ds_name, ok in Parallel(n_jobs=cfg.n_jobs)(
                delayed(_stage3)(ds, img) for ds, img in tasks
            ):
                results[ds_name]["completed" if ok else "failed"] += 1

        ds_results = {
            name: DatasetResults(
                name=name, total=d["total"], completed=d["completed"],
                failed=d["failed"], failures=[],
            )
            for name, d in results.items()
        }
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
        """``--mode process --layer objmap``: merge the sidecar and write the
        objmap layer (mirrored) — no measurement (Spec 1 §6). Runs after Stages
        1-2; deletes the sidecar after export.
        """
        from ._cli_process_only import (
            process_only_output_path,
            write_process_only_layer,
        )
        from ._cli_sidecar import delete_sidecar, load_sidecar

        cfg = self.config
        image_cls = _image_class(cfg.image_type)
        for ds, img in tasks:
            out_path = process_only_output_path(
                output_dir, img, cfg.input_path, "objmap"
            )
            if cfg.resume and out_path.is_file():
                results[ds.name]["completed"] += 1
                continue
            if not sidecar_exists(output_dir, ds.name, img.stem):
                emit_missing_prereq(
                    event_log, ds.name, img.name, STAGE_MEASURE, "objmap sidecar"
                )
                results[ds.name]["failed"] += 1
                continue
            try:
                with stage_event(event_log, ds.name, img.name, STAGE_MEASURE):
                    hdf = dataset_hdf_dir(output_dir, ds.name) / f"{img.stem}.h5"
                    image = image_cls.load_hdf5(hdf)
                    plan.gpu_detector._write_object_output(
                        image, load_sidecar(output_dir, ds.name, img.stem)
                    )
                    write_process_only_layer(image, "objmap", out_path)
                    delete_sidecar(output_dir, ds.name, img.stem)
                results[ds.name]["completed"] += 1
            except Exception:
                results[ds.name]["failed"] += 1
