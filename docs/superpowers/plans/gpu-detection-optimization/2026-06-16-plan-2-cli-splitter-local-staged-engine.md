# CLI Splitter + Local Staged Engine + Sidecar — Implementation Plan (Spec 1, Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** When a CLI pipeline contains a `GpuDetector`, run detection **locally** as three content-defined stages — CPU preprocess → resident-model GPU detect (sidecar write) → CPU merge+measure — chained from a single command, with content-defined resume and stage-tagged progress. (SLURM chaining is Plan 3.)

**Architecture:** A CLI-side splitter slices the ordered `pipeline.get_ops()` at the first `GpuDetector` into a *pre* sub-`ImagePipeline`, the detector, and a *post* sub-`ImagePipeline` (carrying `meas`/`post`/`filters`/`model`/`qc`/grid presets). A new `StagedGpuStrategy` runs: **Stage 1** applies the pre-pipeline per image and writes the normal `results/<ds>/hdf/<stem>.h5`; **Stage 2** keeps the detector's model resident and streams the staged HDFs through `preprocess`/`infer_batch`, writing each result to a `.npy` **sidecar** (HDF opened read-only); **Stage 3** loads HDF+sidecar, applies it via the accessor, runs post-ops + `measure()`, re-saves the HDF atomically, and **deletes the sidecar**. Resume is content-defined (HDF exists → sidecar exists → measurement parquet exists). `ImagePipeline` is unchanged.

**Tech Stack:** Python, pydantic, numpy, h5py, joblib (Stage 1/3 parallelism), torch `DataLoader` (Stage 2 I/O overlap; optional — falls back to a plain loop), pytest. `uv` runner.

**Source of truth:** Spec 1 §3 (split), §5 (staging/sidecar, D13), §6 (flow), §9 (resume + stage tracking, D10), §8 (output routes). **Depends on Plan 1** (the `GpuDetector` interface: `input_layer`, `output_kind`, `preprocess`, `infer_batch`, `_ensure_model_loaded`).

**Grounded reuse points (verified):**
- `ExecutionConfig` (`_cli/_cli_types.py`): fields incl. `pipeline_json`, `input_path`, `output_dir`, `image_type`, `nrows`, `ncols`, `n_jobs`, `process_only_layer`, `measure_only`, `is_slurm_mode()`.
- `Dataset` (`_cli/_cli_types.py`): `.name`, `.images: List[Path]`.
- `OutputManager` (`_cli/_cli_output_manager.py`): `save_image_hdf(image, dataset_name, image_stem)` (writes `results/<ds>/hdf/<stem>.h5` atomically via temp+rename), `get_output_path(dataset_name, "hdf", image_stem)`, `save_measurements(df, dataset_name, image_stem)`.
- `dataset_hdf_dir(output_dir, dataset)`, `results_dir(output_dir)`, `event_log_path(output_dir)`, `progress_dir(output_dir)` (`tools_/_io_constants.py`).
- `Image.load_hdf5(path)` / `GridImage.load_hdf5(path)` (`_core/_image_parts/_image_io_handler.py`).
- `process_single_hdf_measure_core` (`_cli/_cli_process_single.py`) — the measure-only reload path to mirror in Stage 3.
- `pipeline_requires_gpu(pipeline_path)` (`_cli/_cli_validation.py`).
- `create_execution_strategy(config, output_manager)` (`_cli/_cli_execution_strategies.py`) — the factory to extend; called at `phenotypicCLI.py:1365`.
- `append_event` / `append_completion_event` (`_cli/_cli_update_state.py`).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/phenotypic/_cli/_cli_pipeline_split.py` | `StagePlan` + `split_pipeline_at_gpu` + guards | Create |
| `src/phenotypic/_cli/_cli_sidecar.py` | objmap sidecar path / atomic write / load / delete / exists | Create |
| `src/phenotypic/_cli/_cli_staged_strategy.py` | `StagedGpuStrategy` (Stage 1/2/3 local orchestration + stage-tagged events + resume) | Create |
| `src/phenotypic/_cli/_cli_staged_workers.py` | `stage1_preprocess_core`, `stage2_detect_core`, `stage3_merge_measure_core` | Create |
| `src/phenotypic/_cli/_cli_execution_strategies.py` | route GPU local runs to `StagedGpuStrategy` in `create_execution_strategy` | Modify |
| `tests/_fakes/__init__.py`, `tests/_fakes/fake_gpu_detector.py` | shared CPU `FakeGpuDetector` (promoted from Plan 1) | Create |
| `tests/unit/cli/test_cli_pipeline_split.py` | splitter + guards | Create |
| `tests/unit/cli/test_cli_sidecar.py` | sidecar helpers | Create |
| `tests/integration/cli/test_staged_gpu_local.py` | end-to-end local 3-stage run + resume + process-objmap | Create |

**Note for the engineer:** `tests/unit/cli/` and `tests/integration/cli/` already exist. Promote Plan 1's `_FakeGpuDetector` into `tests/_fakes/fake_gpu_detector.py` (Task 0) so both unit and integration suites share it.

---

### Task 0: Promote the shared `FakeGpuDetector` test helper

**Files:**
- Create: `tests/_fakes/__init__.py` (empty), `tests/_fakes/fake_gpu_detector.py`
- Modify: `tests/unit/abc_/test_gpu_detector_interface.py` (import from the shared helper)

- [ ] **Step 1: Create the shared helper**

`tests/_fakes/fake_gpu_detector.py`:

```python
"""CPU-only GpuDetector for tests (no torch). Shared by unit + integration."""

import numpy as np
from pydantic import PrivateAttr
from skimage.measure import label as _sk_label

from phenotypic.abc_ import GpuDetector


class FakeGpuDetector(GpuDetector):
    """Thresholds the (stacked) input; labels it (instance) or returns the
    binary mask (semantic). ``output_kind``/``input_layer``/``supports_batching``
    are overrideable per test."""

    threshold: float = 0.5
    _loaded: bool = PrivateAttr(default=False)

    def _ensure_model_loaded(self) -> None:
        self._loaded = True

    def _infer_one(self, sample):
        gray = sample.mean(axis=-1) if sample.ndim == 3 else sample
        peak = gray.max()
        mask = gray > (self.threshold * peak) if peak > 0 else gray > 0
        if self.output_kind == "instance":
            return _sk_label(mask).astype(np.uint16)
        return mask
```

- [ ] **Step 2: Re-point Plan 1's interface test at the shared helper**

In `tests/unit/abc_/test_gpu_detector_interface.py`, delete the inline `_FakeGpuDetector` class and replace its references with:

```python
from tests._fakes.fake_gpu_detector import FakeGpuDetector as _FakeGpuDetector
```

- [ ] **Step 3: Run to verify**

Run: `uv run pytest tests/unit/abc_/test_gpu_detector_interface.py -v`
Expected: PASS (same tests, helper now shared).

- [ ] **Step 4: Commit**

```bash
git add tests/_fakes tests/unit/abc_/test_gpu_detector_interface.py
git commit -m "test(gpu): promote FakeGpuDetector to shared tests/_fakes helper"
```

---

### Task 1: `StagePlan` + `split_pipeline_at_gpu` + guards

**Files:**
- Create: `src/phenotypic/_cli/_cli_pipeline_split.py`
- Test: `tests/unit/cli/test_cli_pipeline_split.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/cli/test_cli_pipeline_split.py`:

```python
import pytest

from phenotypic import ImagePipeline
from phenotypic.enhance import BlurGauss
from phenotypic.detect import OtsuDetector
from phenotypic.refine import SmallObjectRemover
from phenotypic.measure import MeasureSize
from phenotypic._cli._cli_pipeline_split import (
    split_pipeline_at_gpu,
    StagePlan,
)
from tests._fakes.fake_gpu_detector import FakeGpuDetector


def test_splits_at_first_gpu_detector():
    pipe = ImagePipeline(
        ops=[BlurGauss(), FakeGpuDetector(), SmallObjectRemover()],
        meas=[MeasureSize()],
    )
    plan = split_pipeline_at_gpu(pipe)
    assert isinstance(plan, StagePlan)
    assert list(plan.pre_pipeline.get_ops().keys()) == ["BlurGauss"]
    assert isinstance(plan.gpu_detector, FakeGpuDetector)
    assert list(plan.post_pipeline.get_ops().keys()) == ["SmallObjectRemover"]
    # post pipeline carries the measurements
    assert "MeasureSize" in plan.post_pipeline.get_meas()


def test_rejects_more_than_one_gpu_detector():
    pipe = ImagePipeline(ops=[FakeGpuDetector(), FakeGpuDetector()])
    with pytest.raises(ValueError, match="more than one GpuDetector"):
        split_pipeline_at_gpu(pipe)


def test_rejects_no_gpu_detector():
    pipe = ImagePipeline(ops=[BlurGauss(), OtsuDetector()])
    with pytest.raises(ValueError, match="no GpuDetector"):
        split_pipeline_at_gpu(pipe)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_cli_pipeline_split.py -v`
Expected: FAIL — `ModuleNotFoundError: ... _cli_pipeline_split`.

- [ ] **Step 3: Implement the splitter**

`src/phenotypic/_cli/_cli_pipeline_split.py`:

```python
"""Split a GpuDetector pipeline at the detector boundary (CLI orchestration).

This is a CLI concern, NOT an ImagePipeline change: ImagePipeline stays a plain
ordered container. The splitter reads the public ordered ``pipeline.get_ops()``
and builds throwaway sub-pipelines the staged strategy runs per stage.
See Spec 1 §3.
"""

from __future__ import annotations

from dataclasses import dataclass

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector


@dataclass
class StagePlan:
    """Result of splitting a pipeline at its (single) GpuDetector."""

    pre_pipeline: ImagePipeline      # ops before the detector (Stage 1)
    gpu_detector: GpuDetector        # the detector itself (Stage 2)
    post_pipeline: ImagePipeline     # ops after + meas/post/filters/model/qc (Stage 3)


def split_pipeline_at_gpu(pipeline: ImagePipeline) -> StagePlan:
    """Partition *pipeline* at the first GpuDetector into pre/detector/post.

    Raises:
        ValueError: if the pipeline contains zero or more than one GpuDetector.
    """
    ops = pipeline.get_ops()  # ordered dict
    gpu_keys = [k for k, op in ops.items() if isinstance(op, GpuDetector)]
    if len(gpu_keys) == 0:
        raise ValueError("staged execution requires a GpuDetector; found none")
    if len(gpu_keys) > 1:
        raise ValueError(
            "staged execution does not support more than one GpuDetector "
            f"per pipeline (found {len(gpu_keys)}: {gpu_keys})"
        )

    gpu_key = gpu_keys[0]
    keys = list(ops.keys())
    cut = keys.index(gpu_key)
    pre_ops = {k: ops[k] for k in keys[:cut]}
    post_ops = {k: ops[k] for k in keys[cut + 1:]}

    pre_pipeline = ImagePipeline(ops=pre_ops, nrows=pipeline.nrows, ncols=pipeline.ncols)
    post_pipeline = ImagePipeline(
        ops=post_ops,
        meas=pipeline.get_meas(),
        post=pipeline.get_post(),
        filters=pipeline.get_filters(),
        model=pipeline.get_model(),
        qc=pipeline.get_qc(),
        nrows=pipeline.nrows,
        ncols=pipeline.ncols,
    )
    return StagePlan(
        pre_pipeline=pre_pipeline,
        gpu_detector=ops[gpu_key],
        post_pipeline=post_pipeline,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_cli_pipeline_split.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_pipeline_split.py tests/unit/cli/test_cli_pipeline_split.py
git commit -m "feat(staged): CLI-side pipeline splitter at the GpuDetector boundary"
```

---

### Task 2: objmap sidecar helpers (atomic)

**Files:**
- Create: `src/phenotypic/_cli/_cli_sidecar.py`
- Test: `tests/unit/cli/test_cli_sidecar.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/cli/test_cli_sidecar.py`:

```python
import numpy as np

from phenotypic._cli._cli_sidecar import (
    sidecar_path,
    write_sidecar,
    load_sidecar,
    sidecar_exists,
    delete_sidecar,
)


def test_path_layout(tmp_path):
    p = sidecar_path(tmp_path, "ds1", "img42")
    assert p == tmp_path / "results" / "ds1" / "objmap" / "img42.npy"


def test_write_load_exists_delete(tmp_path):
    arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert not sidecar_exists(tmp_path, "ds1", "img42")
    write_sidecar(tmp_path, "ds1", "img42", arr)
    assert sidecar_exists(tmp_path, "ds1", "img42")
    np.testing.assert_array_equal(load_sidecar(tmp_path, "ds1", "img42"), arr)
    delete_sidecar(tmp_path, "ds1", "img42")
    assert not sidecar_exists(tmp_path, "ds1", "img42")


def test_write_is_atomic_no_partial_file(tmp_path):
    # the temp file must not remain after a successful write
    write_sidecar(tmp_path, "ds1", "img42", np.zeros((2, 2), np.uint16))
    objmap_dir = tmp_path / "results" / "ds1" / "objmap"
    assert [p.name for p in objmap_dir.iterdir()] == ["img42.npy"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_cli_sidecar.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the sidecar helpers**

`src/phenotypic/_cli/_cli_sidecar.py`:

```python
"""Per-image objmap sidecar for the staged GPU engine (Spec 1 §5, D13).

Stage 2 writes the GPU result here (HDF opened read-only); Stage 3 merges it
into the final HDF and deletes it. Writes are atomic (temp + os.replace).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from phenotypic.tools_ import results_dir

_OBJMAP_LAYER = "objmap"


def sidecar_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """``<output>/results/<dataset>/objmap/<stem>.npy``."""
    return results_dir(output_dir) / dataset / _OBJMAP_LAYER / f"{image_stem}.npy"


def write_sidecar(
    output_dir: Path, dataset: str, image_stem: str, array: np.ndarray
) -> Path:
    """Atomically write *array* to the sidecar (temp + os.replace)."""
    final = sidecar_path(output_dir, dataset, image_stem)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".npy.tmp")
    np.save(tmp, array)
    os.replace(tmp, final)  # atomic on POSIX same-filesystem
    return final


def load_sidecar(output_dir: Path, dataset: str, image_stem: str) -> np.ndarray:
    return np.load(sidecar_path(output_dir, dataset, image_stem))


def sidecar_exists(output_dir: Path, dataset: str, image_stem: str) -> bool:
    return sidecar_path(output_dir, dataset, image_stem).is_file()


def delete_sidecar(output_dir: Path, dataset: str, image_stem: str) -> None:
    p = sidecar_path(output_dir, dataset, image_stem)
    p.unlink(missing_ok=True)
```

(`np.save` appends `.npy`; saving to `final.with_suffix(".npy.tmp")` writes `<stem>.npy.tmp.npy` — fix by passing a base without suffix. Use `tmp = final.parent / (final.name + ".tmp")` and `np.save(tmp, array)` then `os.replace(tmp.with_suffix(tmp.suffix), final)` is fragile. **Simplest robust form below** — use it instead of the snippet above for `write_sidecar`:)

```python
def write_sidecar(output_dir, dataset, image_stem, array):
    final = sidecar_path(output_dir, dataset, image_stem)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".{image_stem}.npy.tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, final)
    return final
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_cli_sidecar.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_sidecar.py tests/unit/cli/test_cli_sidecar.py
git commit -m "feat(staged): atomic per-image objmap sidecar helpers"
```

---

### Task 3: Stage workers — preprocess / detect / merge+measure

**Files:**
- Create: `src/phenotypic/_cli/_cli_staged_workers.py`
- Test: covered by the integration test in Task 5 (these cores are exercised end-to-end). Add focused unit coverage for `stage3_merge_measure_core` here.

- [ ] **Step 1: Write the failing test**

Add to `tests/integration/cli/test_staged_gpu_local.py` (Create):

```python
import numpy as np

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic._cli._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
)
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_sidecar import sidecar_exists
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic.tools_ import dataset_hdf_dir
from tests._fakes.fake_gpu_detector import FakeGpuDetector


def _write_image(tmp_path):
    img = load_synth_yeast_plate()
    p = tmp_path / "img.tiff"
    img.rgb.imsave(filepath=p)
    return p


def test_three_stage_cores_end_to_end(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(output_kind="instance", threshold=0.3)],
        meas=[MeasureSize()],
    )
    pipe_path = out / "pipeline.json"
    pipe_path.parent.mkdir(parents=True)
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipe_path))
    om = OutputManager.from_config(out, save_overlays=False)

    # Stage 1: preprocess -> HDF
    stage1_preprocess_core(plan, image_path, "ds", "img", out, om, image_type="Image")
    assert (dataset_hdf_dir(out, "ds") / "img.h5").is_file()

    # Stage 2: resident detector -> sidecar
    plan.gpu_detector._ensure_model_loaded()
    stage2_detect_core(plan.gpu_detector, out, "ds", "img")
    assert sidecar_exists(out, "ds", "img")

    # Stage 3: merge + measure -> parquet, re-save HDF, delete sidecar
    stage3_merge_measure_core(plan, out, "ds", "img", om, image_type="Image")
    assert (out / "results" / "ds" / "measurements" / "img.parquet").is_file()
    assert not sidecar_exists(out, "ds", "img")  # mandatory cleanup
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/cli/test_staged_gpu_local.py::test_three_stage_cores_end_to_end -v`
Expected: FAIL — `_cli_staged_workers` missing. (If the `OutputManager.from_config` / `imsave` signatures differ, read `_cli_output_manager.py:1001` and `accessors/_multichannel_accessor.py` to confirm, then adjust the test setup — these are existing APIs.)

- [ ] **Step 3: Implement the stage workers**

`src/phenotypic/_cli/_cli_staged_workers.py`:

```python
"""Per-image stage workers for the local staged GPU engine (Spec 1 §5-§6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from phenotypic import GridImage, Image
from phenotypic.abc_ import GpuDetector
from phenotypic.tools_ import dataset_hdf_dir
from phenotypic.tools_.typing_ import ImageTypeName
from ._cli_output_manager import OutputManager
from ._cli_pipeline_split import StagePlan
from ._cli_sidecar import write_sidecar, load_sidecar, delete_sidecar


def stage1_preprocess_core(
    plan: StagePlan,
    image_path: Path,
    dataset_name: str,
    image_stem: str,
    output_dir: Path,
    output_manager: OutputManager,
    image_type: ImageTypeName,
    read_kwargs: Dict[str, Any] | None = None,
) -> None:
    """Read raw image, apply the pre-detector ops, save the staged HDF."""
    read_kwargs = dict(read_kwargs or {})
    image_cls = GridImage if image_type == "GridImage" else Image
    detect_mode = read_kwargs.pop("detect_mode", "gray")
    image = image_cls.imread(image_path, **read_kwargs)
    if detect_mode != "gray":
        image.set_detect_mode(detect_mode)
    plan.pre_pipeline.apply(image, inplace=True)
    output_manager.save_image_hdf(image, dataset_name, image_stem)


def stage2_detect_core(
    detector: GpuDetector,
    output_dir: Path,
    dataset_name: str,
    image_stem: str,
    image_type: ImageTypeName = "Image",
) -> None:
    """Load the input layer (HDF read-only), run inference, write the sidecar.

    The detector's model must already be resident (caller invokes
    ``_ensure_model_loaded()`` once before streaming a shard).
    """
    image_cls = GridImage if image_type == "GridImage" else Image
    hdf = dataset_hdf_dir(output_dir, dataset_name) / f"{image_stem}.h5"
    image = image_cls.load_hdf5(hdf)  # read-only use; never re-saved here
    array = getattr(image, detector.input_layer)[:]
    sample = detector.preprocess(array)
    batch = detector.collate([sample])
    result = detector.infer_batch(batch)[0]
    write_sidecar(output_dir, dataset_name, image_stem, result)


def stage3_merge_measure_core(
    plan: StagePlan,
    output_dir: Path,
    dataset_name: str,
    image_stem: str,
    output_manager: OutputManager,
    image_type: ImageTypeName,
) -> None:
    """Merge the sidecar, apply post-ops + measure, re-save HDF, delete sidecar."""
    image_cls = GridImage if image_type == "GridImage" else Image
    hdf = dataset_hdf_dir(output_dir, dataset_name) / f"{image_stem}.h5"
    image = image_cls.load_hdf5(hdf)

    result = load_sidecar(output_dir, dataset_name, image_stem)
    plan.gpu_detector._write_object_output(image, result)

    # post-detector ops (refiners incl. watershed) + measurement
    plan.post_pipeline.apply(image, inplace=True)
    measurements = plan.post_pipeline.measure(image, apply_post=False)

    output_manager.save_measurements(measurements, dataset_name, image_stem)
    output_manager.save_image_hdf(image, dataset_name, image_stem)  # atomic re-save
    delete_sidecar(output_dir, dataset_name, image_stem)  # mandatory cleanup
```

(`plan.post_pipeline.apply` re-runs only the post-detector ops on the merged image; if `post_pipeline` has no ops it is a no-op. `measure(..., apply_post=False)` mirrors `process_single_image_core` so per-image parquets stay clean.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/cli/test_staged_gpu_local.py::test_three_stage_cores_end_to_end -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_workers.py tests/integration/cli/test_staged_gpu_local.py
git commit -m "feat(staged): per-image stage workers (preprocess / detect / merge+measure)"
```

---

### Task 4: `StagedGpuStrategy` — local 3-stage orchestration + stage events + resume

**Files:**
- Create: `src/phenotypic/_cli/_cli_staged_strategy.py`
- Test: `tests/integration/cli/test_staged_gpu_local.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from datetime import datetime
from phenotypic._cli._cli_types import Dataset, ExecutionConfig
from phenotypic._cli._cli_staged_strategy import StagedGpuStrategy


def _config(out, pipe_path):
    return ExecutionConfig(
        pipeline_json=pipe_path, input_path=out, output_dir=out,
        image_type="Image", nrows=None, ncols=None, bit_depth=None,
        n_jobs=1, slurm_args={}, force_local=True, wait=False, ext=".png",
        overlay_alpha=0.5, include_dataset_column=False, dry_run=False,
        sample=None, resume=False, retry_failures=False, skip_validation=True,
        save_overlays=False,
    )


def test_staged_strategy_runs_all_stages(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(output_kind="instance", threshold=0.3)],
        meas=[MeasureSize()],
    )
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, save_overlays=False)
    om.create_dataset_directories([Dataset("ds", [image_path], tmp_path, out)])

    strat = StagedGpuStrategy(_config(out, pipe_path), om)
    results = strat.execute([Dataset("ds", [image_path], tmp_path, out)], out)

    assert results.total_completed == 1
    assert (out / "results" / "ds" / "measurements" / "img.parquet").is_file()
    assert not sidecar_exists(out, "ds", "img")


def test_staged_strategy_resumes_skipping_done_stages(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    pipe = ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()])
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, save_overlays=False)
    om.create_dataset_directories([Dataset("ds", [image_path], tmp_path, out)])
    ds = [Dataset("ds", [image_path], tmp_path, out)]

    StagedGpuStrategy(_config(out, pipe_path), om).execute(ds, out)
    parquet = out / "results" / "ds" / "measurements" / "img.parquet"
    mtime = parquet.stat().st_mtime_ns

    # second run: Stage 3 already done -> parquet untouched
    StagedGpuStrategy(_config(out, pipe_path), om).execute(ds, out)
    assert parquet.stat().st_mtime_ns == mtime
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/cli/test_staged_gpu_local.py -k staged_strategy -v`
Expected: FAIL — `_cli_staged_strategy` missing. (Confirm `OutputManager.create_dataset_directories` exists at `_cli_output_manager.py:1014`; adjust the call if its name differs.)

- [ ] **Step 3: Implement the staged strategy**

`src/phenotypic/_cli/_cli_staged_strategy.py`:

```python
"""Local staged GPU execution strategy (Spec 1 §6-§9).

Runs Stage 1 (preprocess -> HDF) and Stage 3 (merge -> measure) with joblib;
Stage 2 keeps the detector model resident and streams the staged HDFs to
sidecars. Content-defined resume: HDF exists -> sidecar exists -> parquet exists.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from joblib import Parallel, delayed

from phenotypic import ImagePipeline
from phenotypic.tools_ import dataset_hdf_dir, event_log_path
from ._cli_execution_strategies import ExecutionStrategy
from ._cli_pipeline_split import split_pipeline_at_gpu
from ._cli_sidecar import sidecar_exists
from ._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
)
from ._cli_types import Dataset, DatasetResults, ExecutionResults
from ._cli_update_state import append_event, append_completion_event


class StagedGpuStrategy(ExecutionStrategy):
    """Three-stage local GPU detection: preprocess -> detect -> measure."""

    def execute(self, datasets: List[Dataset], output_dir: Path) -> ExecutionResults:
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

        # ---- Stage 1: CPU preprocess -> staged HDF (parallel, resumable) ----
        def _stage1(ds: Dataset, img: Path) -> None:
            hdf = dataset_hdf_dir(output_dir, ds.name) / f"{img.stem}.h5"
            if cfg.resume and hdf.is_file():
                return
            append_event(event_log, ds.name, img.name, "stage1_started")
            stage1_preprocess_core(
                plan, img, ds.name, img.stem, output_dir, self.output_manager,
                cfg.image_type, read_kwargs,
            )
            append_completion_event(event_log, ds.name, img.name, "stage1_completed")

        Parallel(n_jobs=cfg.n_jobs)(delayed(_stage1)(ds, img) for ds, img in tasks)

        # ---- Stage 2: resident-model GPU detect -> sidecar (serial) --------
        plan.gpu_detector._ensure_model_loaded()  # load ONCE
        for ds, img in tasks:
            if cfg.resume and sidecar_exists(output_dir, ds.name, img.stem):
                continue
            append_event(event_log, ds.name, img.name, "stage2_started")
            stage2_detect_core(
                plan.gpu_detector, output_dir, ds.name, img.stem, cfg.image_type
            )
            append_completion_event(event_log, ds.name, img.name, "stage2_completed")

        # ---- Stage 3: CPU merge + measure (parallel, resumable) ------------
        results: Dict[str, Dict[str, int]] = {
            ds.name: {"total": len(ds.images), "completed": 0, "failed": 0}
            for ds in datasets
        }

        def _stage3(ds: Dataset, img: Path) -> tuple[str, bool]:
            parquet = (
                self.output_manager.get_output_path(ds.name, "measurements", img.stem)
            )
            if cfg.resume and parquet.is_file():
                return ds.name, True
            append_event(event_log, ds.name, img.name, "stage3_started")
            stage3_merge_measure_core(
                plan, output_dir, ds.name, img.stem, self.output_manager,
                cfg.image_type,
            )
            append_completion_event(event_log, ds.name, img.name, "stage3_completed")
            return ds.name, True

        for ds_name, ok in Parallel(n_jobs=cfg.n_jobs)(
            delayed(_stage3)(ds, img) for ds, img in tasks
        ):
            results[ds_name]["completed" if ok else "failed"] += 1

        ds_results = {
            name: DatasetResults(name=name, total=d["total"],
                                 completed=d["completed"], failed=d["failed"],
                                 failures=[])
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
```

(Stage 2 is serial because the resident model is single-GPU here; Plan 3 adds shard-workers + `workers_per_gpu`. The `get_output_path(ds, "measurements", stem)` returns the parquet path — confirm its extension is `.parquet` at `_cli_output_manager.py:1078`; if it returns the dir, use `dataset_measurements_dir(...)/f"{stem}.parquet"`.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/cli/test_staged_gpu_local.py -k staged_strategy -v`
Expected: PASS (both run + resume tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_strategy.py tests/integration/cli/test_staged_gpu_local.py
git commit -m "feat(staged): local StagedGpuStrategy (3-stage + stage events + resume)"
```

---

### Task 5: Route GPU local runs to `StagedGpuStrategy`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py` (the `create_execution_strategy` factory)
- Test: `tests/unit/cli/test_cli_pipeline_split.py` (add a routing test) — or a new `tests/unit/cli/test_staged_routing.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/cli/test_staged_routing.py`:

```python
from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic._cli._cli_execution_strategies import (
    create_execution_strategy, LocalParallelStrategy,
)
from phenotypic._cli._cli_staged_strategy import StagedGpuStrategy
from phenotypic._cli._cli_output_manager import OutputManager
from tests._fakes.fake_gpu_detector import FakeGpuDetector
# reuse the _config helper pattern from the integration test (inline a minimal one here)


def _write_pipe(tmp_path, pipe):
    p = tmp_path / "pipe.json"
    p.write_text(pipe.to_json(), encoding="utf-8")
    return p


def test_gpu_local_routes_to_staged(tmp_path, monkeypatch):
    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path)  # slurm_args={}, measure_only=False
    om = OutputManager.from_config(tmp_path, save_overlays=False)
    strat = create_execution_strategy(cfg, om)
    assert isinstance(strat, StagedGpuStrategy)


def test_cpu_local_routes_to_local_parallel(tmp_path):
    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[OtsuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, save_overlays=False)
    assert isinstance(create_execution_strategy(cfg, om), LocalParallelStrategy)
```

(Add a `_minimal_local_config` helper mirroring `_config` from Task 4 with `measure_only=False`, `slurm_args={}`.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_staged_routing.py -v`
Expected: FAIL — factory still returns `LocalParallelStrategy` for the GPU pipeline.

- [ ] **Step 3: Extend the factory**

In `_cli_execution_strategies.py`, update `create_execution_strategy`:

```python
def create_execution_strategy(config, output_manager):
    from ._cli_validation import pipeline_requires_gpu
    from ._cli_staged_strategy import StagedGpuStrategy

    if config.is_slurm_mode():
        return AutonomousSLURMStrategy(config, output_manager)  # Plan 3 makes this staged
    # Local: staged path when the pipeline needs a GPU and we're doing a forward run.
    if (
        not config.measure_only
        and pipeline_requires_gpu(config.pipeline_json)
    ):
        return StagedGpuStrategy(config, output_manager)
    return LocalParallelStrategy(config, output_manager)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_staged_routing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_execution_strategies.py tests/unit/cli/test_staged_routing.py
git commit -m "feat(staged): route local GPU pipelines through StagedGpuStrategy"
```

---

### Task 6: `--mode process --layer objmap` routes through Stages 1–2

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py` (honor `process_only_layer`)
- Test: `tests/integration/cli/test_staged_gpu_local.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from phenotypic._cli._cli_process_only import process_only_output_path


def test_process_objmap_runs_stages_1_2_then_exports(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    pipe = ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)])
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, save_overlays=False)
    om.create_dataset_directories([Dataset("ds", [image_path], tmp_path, out)])

    cfg = _config(out, pipe_path)
    cfg.process_only_layer = "objmap"
    StagedGpuStrategy(cfg, om).execute([Dataset("ds", [image_path], tmp_path, out)], out)

    # objmap layer exported (mirrored), no measurement parquet
    expected = process_only_output_path(out, image_path, out, "objmap")
    assert expected.is_file()
    assert not (out / "results" / "ds" / "measurements" / "img.parquet").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/cli/test_staged_gpu_local.py -k process_objmap -v`
Expected: FAIL — Stage 3 still runs measurement.

- [ ] **Step 3: Branch the strategy on `process_only_layer`**

In `StagedGpuStrategy.execute`, after Stage 2, replace the Stage-3 block with:

```python
        if cfg.process_only_layer == "objmap":
            # process-mode: export the objmap layer (mirrored), stop (Spec 1 §6).
            from ._cli_sidecar import load_sidecar, delete_sidecar
            from ._cli_process_only import process_only_output_path, write_process_only_layer
            from phenotypic import GridImage, Image
            image_cls = GridImage if cfg.image_type == "GridImage" else Image
            for ds, img in tasks:
                hdf = dataset_hdf_dir(output_dir, ds.name) / f"{img.stem}.h5"
                image = image_cls.load_hdf5(hdf)
                plan.gpu_detector._write_object_output(
                    image, load_sidecar(output_dir, ds.name, img.stem)
                )
                out_path = process_only_output_path(output_dir, img, cfg.input_path, "objmap")
                write_process_only_layer(image, "objmap", out_path)
                delete_sidecar(output_dir, ds.name, img.stem)
                results[ds.name]["completed"] += 1
            # build ExecutionResults as below and return early
            ...
        else:
            # Stage 3 measure (existing block)
            ...
```

(Refactor the Stage-3 measure block and the export branch to share the `ExecutionResults` construction. The `process_only_output_path` / `write_process_only_layer` are existing functions in `_cli_process_only.py`.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/cli/test_staged_gpu_local.py -k process_objmap -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_strategy.py tests/integration/cli/test_staged_gpu_local.py
git commit -m "feat(staged): --mode process --layer objmap routes through stages 1-2"
```

---

### Task 7: Stage-tracking aggregation + regression

**Files:**
- Modify: `src/phenotypic/_cli/_cli_update_state.py` (extend `aggregate_state_from_events` to bucket per-stage) — only if the dashboard needs per-stage counts now; otherwise defer to Plan 3 and just verify events are emitted.
- Test: `tests/integration/cli/test_staged_gpu_local.py`

- [ ] **Step 1: Write the failing test (events emitted per stage)**

Append:

```python
from phenotypic.tools_ import event_log_path


def test_stage_tagged_events_emitted(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    pipe = ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()])
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, save_overlays=False)
    om.create_dataset_directories([Dataset("ds", [image_path], tmp_path, out)])
    StagedGpuStrategy(_config(out, pipe_path), om).execute(
        [Dataset("ds", [image_path], tmp_path, out)], out
    )
    log = event_log_path(out).read_text(encoding="utf-8")
    for marker in ("stage1_completed", "stage2_completed", "stage3_completed"):
        assert marker in log
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/integration/cli/test_staged_gpu_local.py -k stage_tagged -v`
Expected: PASS (events are already emitted by Task 4). If `append_completion_event` rejects the custom status string, read `_cli_update_state.py:105` and relax the status validation (or store the stage in a structured field).

- [ ] **Step 3: Full regression**

Run:
```bash
uv run pytest tests/unit/cli tests/integration/cli/test_staged_gpu_local.py tests/unit/abc_/test_gpu_detector_interface.py -v
uv run mypy src/phenotypic/_cli/_cli_pipeline_split.py src/phenotypic/_cli/_cli_sidecar.py src/phenotypic/_cli/_cli_staged_workers.py src/phenotypic/_cli/_cli_staged_strategy.py
uv run ruff check --fix src/phenotypic/_cli tests/unit/cli tests/integration/cli
```
Expected: PASS / clean.

- [ ] **Step 4: Smoke the broader CLI to catch regressions**

Run: `uv run pytest tests/smoke tests/integration/cli -q -m "not slow"`
Expected: PASS / SKIP — non-GPU CLI runs (`LocalParallelStrategy`) unaffected.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test(staged): stage-tagged events + green regression for local staged engine" --allow-empty
```

---

### Task 8: Update CLAUDE.md + how-to for local staged GPU detection

**Files:**
- Modify: `CLAUDE.md` (CLI section + Gotchas)
- Modify: `docs/source/how_to/pages/gpu_detection_setup.md`
- Test: `tests/unit/test_docs_staged_cli.py` (Create)

- [ ] **Step 1: Write the failing content check**

`tests/unit/test_docs_staged_cli.py`:

```python
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_claude_md_documents_local_staged_gpu():
    txt = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert "GpuDetector" in txt
    assert "stage" in txt.lower() and "sidecar" in txt.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_docs_staged_cli.py -v`
Expected: FAIL — content not present.

- [ ] **Step 3: Update the docs**

In `CLAUDE.md` (Quick Start → CLI section), add:

> When a pipeline contains a `GpuDetector`, `python -m phenotypic` runs detection as
> three internal stages — CPU preprocess → resident-model GPU detect → CPU measure —
> reusing the per-image HDF. Stage 2 writes a per-image `.npy` objmap **sidecar** (HDF
> read-only); Stage 3 merges it into the final HDF and deletes the sidecar. The output
> folder is identical to a single-pass run.

Add a **Gotchas** entry:

> - **GPU pipelines stage internally:** a `GpuDetector` in a CLI run triggers the staged
>   engine (preprocess → GPU → measure), not per-image processing. Notebook
>   `op.apply(image)` is unchanged.

In `docs/source/how_to/pages/gpu_detection_setup.md`, add a **"Local staged GPU
detection"** section describing the three stages, the sidecar, and that
`--mode process --layer objmap` exports objmaps after Stages 1–2.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_docs_staged_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/source/how_to/pages/gpu_detection_setup.md tests/unit/test_docs_staged_cli.py
git commit -m "docs(staged): document local staged GPU detection (CLAUDE.md + how-to)"
```

---

## Self-Review

**Spec coverage:** CLI-side split (§3) → Task 1; sidecar D13 (§5) → Task 2; 3-stage workers (§5-§6) → Task 3; local orchestration + stage events + resume (§6-§9) → Task 4; routing default for GPU runs (D2) → Task 5; `--mode process --layer objmap` (§6) → Task 6; stage tracking (D10) → Task 7. ✓
**`ImagePipeline` unchanged:** only `get_ops`/`get_meas`/… read accessors used; no new pipeline methods. ✓
**Signatures to confirm during execution (existing APIs, flagged inline):** `OutputManager.from_config` / `create_dataset_directories` / `get_output_path` extension; `append_completion_event` status validation. These are reuse points, not new logic.
**Out of scope (Plan 3):** SLURM 3-link `afterany` chaining + per-stage partitions + shard-workers (`workers_per_gpu`/`gpu_batch_size`); licensing scaffolding.

---

## Execution Handoff

Plan 2 drafted. Plan 3 (SLURM + licensing) follows.
