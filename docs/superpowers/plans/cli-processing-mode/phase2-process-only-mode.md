# Phase 2 — `--process-only` CLI Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--process-only {rgb|gray|detect_mat|objmap}`, a run mode that executes `pipeline.apply()` only and writes a single image layer per input image (TIFF at the image's bit depth for `rgb`/`gray`/`detect_mat`; 16-bit raw-label PNG for `objmap`), mirroring the input tree, with full local + SLURM + resume reuse and no measurement/analysis output suite.

**Architecture:** A sibling per-image worker (`process_single_apply_only_core`) runs `pipeline.apply()` then writes one layer via a worker-side bit-depth quantizer; the existing `LocalParallelStrategy`/`AutonomousSLURMStrategy` branch on a new `ExecutionConfig.process_only_layer` for both the per-image callable and the finalize step (manifest-only, no dashboard/aggregation). Machine-state lives in `.phenotypic/` (Phase 1). A new classifier signal makes a process-only run discoverable in the run console without a dashboard/results affordance.

**Tech Stack:** Python 3.12, `uv`, `click`, `tifffile`, `scikit-image` (`img_as_ubyte`/`img_as_uint`), `pytest`, Dash GUI.

**Spec:** `docs/superpowers/specs/2026-06-03-cli-process-only-and-phenotypic-cache-design.md` (§5.3–5.7; decisions D2, D3, D5, D8, D9, D10, D12, D13).

**Depends on:** Phase 1 (`.phenotypic` migration) must be merged first.

**Refinement over the spec discovered while planning (apply this):** the GUI classifier's `is_cli_output` keys on `results/` + `deliverables/master_measurements.parquet`, which a process-only run lacks — so D13 ("visible in run console") requires a **new classifier capability** (`is_process_only_output`, signalled by `.phenotypic/progress/manifest.json`), not "no extra classifier work" as §5.2 first assumed. Task 8 adds it.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/phenotypic/tools_/typing_.py` | `ProcessOnlyLayer` Literal alias | Modify |
| `src/phenotypic/tools_/_io_constants.py` | `phenotypic_cache_pipeline_json_path` helper | Modify |
| `src/phenotypic/_cli/_cli_types.py` | `ExecutionConfig.process_only_layer` field | Modify |
| `src/phenotypic/_cli/_cli_process_only.py` | **New** — output-path mapper, layer writer (bit-depth quantization), `process_single_apply_only_core` | Create |
| `src/phenotypic/_cli/_cli_process_single.py` | Worker-CLI `--process-only` / `--input-root` options | Modify |
| `src/phenotypic/_cli/_cli_execution_strategies.py` | Local/SLURM dispatch (worker + finalize) | Modify |
| `src/phenotypic/_cli/_cli_slurm_array_scripts.py` | Thread `--process-only`/`--input-root`; manifest-only finalize | Modify |
| `src/phenotypic/phenotypicCLI.py` | Top-level `--process-only` option + validation + `ExecutionConfig` wiring + dry-run | Modify |
| `src/phenotypic/gui/shell/_classifier.py` | `is_process_only_output` capability | Modify |
| `src/phenotypic/gui/FEATURES.md` | CI-gated row | Modify |
| `tests/unit/cli/`, `tests/integration/cli/` | Unit + integration tests | Create/Modify |
| `CLAUDE.md`, CLI docstring, `README.md` | Docs | Modify |

---

## Task 1: `ProcessOnlyLayer` Literal alias

**Files:**
- Modify: `src/phenotypic/tools_/typing_.py`
- Modify: `src/phenotypic/tools_/__init__.py`
- Test: `tests/unit/tools_/test_typing_aliases.py` (create if absent)

- [ ] **Step 1: Write the failing test**

```python
from typing import get_args
from phenotypic.tools_.typing_ import ProcessOnlyLayer


def test_process_only_layer_values():
    assert set(get_args(ProcessOnlyLayer)) == {"rgb", "gray", "detect_mat", "objmap"}


def test_process_only_layers_are_image_accessors():
    from phenotypic.data import load_synth_yeast_plate
    img = load_synth_yeast_plate()
    for layer in get_args(ProcessOnlyLayer):
        assert hasattr(img, layer), f"{layer} is not an Image accessor"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tools_/test_typing_aliases.py -v`
Expected: FAIL — `ImportError: cannot import name 'ProcessOnlyLayer'`.

- [ ] **Step 3: Implement**

In `src/phenotypic/tools_/typing_.py`, after `DetectMode`:

```python
#: Image layer a process-only CLI run (``--process-only``) exports. A closed
#: subset of the layers exposed as Image accessors; ``rgb``/``gray``/
#: ``detect_mat`` save as TIFF, ``objmap`` as a raw-label PNG.
ProcessOnlyLayer = Literal["rgb", "gray", "detect_mat", "objmap"]
```

Export `ProcessOnlyLayer` from `src/phenotypic/tools_/__init__.py` (import block + `__all__`).

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tools_/test_typing_aliases.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tools_/typing_.py src/phenotypic/tools_/__init__.py tests/unit/tools_/test_typing_aliases.py
git commit -m "feat(typing): add ProcessOnlyLayer literal alias"
```

---

## Task 2: `ExecutionConfig.process_only_layer` + pipeline.json cache helper

**Files:**
- Modify: `src/phenotypic/_cli/_cli_types.py`
- Modify: `src/phenotypic/tools_/_io_constants.py`, `tools_/__init__.py`
- Test: `tests/unit/tools_/test_io_constants.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/tools_/test_io_constants.py`:

```python
def test_phenotypic_cache_pipeline_json_path(tmp_path):
    from phenotypic.tools_ import phenotypic_cache_pipeline_json_path, PIPELINE_JSON
    assert phenotypic_cache_pipeline_json_path(tmp_path) == tmp_path / ".phenotypic" / PIPELINE_JSON
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k pipeline_json_path -v`
Expected: FAIL — symbol missing.

- [ ] **Step 3: Implement**

In `_io_constants.py` (near `phenotypic_cache_dir`):

```python
def phenotypic_cache_pipeline_json_path(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/pipeline.json`` — the process-only run's
    reproducibility copy. Distinct from :func:`pipeline_json_path`, which roots
    under ``deliverables/`` (process-only writes no deliverables)."""
    return phenotypic_cache_dir(output_dir) / PIPELINE_JSON
```

Export it from `tools_/__init__.py`.

In `_cli_types.py`, add to `ExecutionConfig` (after `measure_only`, with a default so construction sites don't all need updating):

```python
    # Process-only mode: run pipeline.apply() and export a single image layer
    # (no measurement / analysis output). None = normal forward/measure run.
    process_only_layer: Optional[ProcessOnlyLayer] = None
```

Add the import at the top of `_cli_types.py`:
```python
from phenotypic.tools_.typing_ import ExecutionMode, ImageTypeName, ProcessOnlyLayer
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k pipeline_json_path -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_types.py src/phenotypic/tools_/_io_constants.py src/phenotypic/tools_/__init__.py tests/unit/tools_/test_io_constants.py
git commit -m "feat(cli): ExecutionConfig.process_only_layer + .phenotypic pipeline.json helper"
```

---

## Task 3: Process-only output-path mapper + layer writer

**Files:**
- Create: `src/phenotypic/_cli/_cli_process_only.py`
- Test: `tests/unit/cli/test_cli_process_only.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_cli_process_only.py`:

```python
from pathlib import Path

import numpy as np
import tifffile

from phenotypic._cli._cli_process_only import process_only_output_path, write_process_only_layer
from phenotypic.data import load_synth_yeast_plate


def test_output_path_mirrors_one_level(tmp_path):
    out = tmp_path / "out"
    root = tmp_path / "in"
    img = root / "day1" / "plateA.tif"
    assert process_only_output_path(out, img, root, "detect_mat") == out / "day1" / "plateA_detect_mat.tiff"
    assert process_only_output_path(out, img, root, "objmap") == out / "day1" / "plateA_objmap.png"


def test_output_path_flat_and_single_file(tmp_path):
    out = tmp_path / "out"
    root = tmp_path / "in"
    assert process_only_output_path(out, root / "a.tif", root, "rgb") == out / "a_rgb.tiff"
    # single-file input: input_root is the file's parent
    f = tmp_path / "solo.tif"
    assert process_only_output_path(out, f, f.parent, "gray") == tmp_path / "out" / "solo_gray.tiff"


def test_write_rgb_is_uint8_for_8bit_source(tmp_path):
    img = load_synth_yeast_plate()              # 8-bit source; rgb uint8
    p = tmp_path / "rgb.tiff"
    write_process_only_layer(img, "rgb", p)
    arr = tifffile.imread(p)
    assert arr.dtype == np.uint8 and arr.ndim == 3


def test_write_detect_mat_float_quantized_to_source_depth(tmp_path):
    img = load_synth_yeast_plate()              # detect_mat is float64 in [0,1]
    p = tmp_path / "dm.tiff"
    write_process_only_layer(img, "detect_mat", p)
    arr = tifffile.imread(p)
    assert arr.dtype == np.uint8                # 8-bit source -> uint8, not float


def test_write_objmap_is_uint16_png(tmp_path):
    import cv2
    img = load_synth_yeast_plate()
    p = tmp_path / "om.png"
    write_process_only_layer(img, "objmap", p)
    arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    assert arr.dtype == np.uint16              # raw labels, 16-bit regardless of source


def test_objmap_without_objects_warns_and_writes_empty(tmp_path, recwarn):
    img = load_synth_yeast_plate()
    img.reset()                                 # no detection -> empty objmap
    p = tmp_path / "om.png"
    write_process_only_layer(img, "objmap", p)
    assert p.is_file()
    assert any("object map" in str(w.message).lower() for w in recwarn.list)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_cli_process_only.py -v`
Expected: FAIL — module `_cli_process_only` does not exist.

- [ ] **Step 3: Implement the module**

Create `src/phenotypic/_cli/_cli_process_only.py`:

```python
"""Process-only CLI mode: run pipeline.apply() and export a single layer.

Used when the user wants PhenoTypic preprocessing/detection output without the
full measurement/analysis suite. See
docs/superpowers/specs/2026-06-03-cli-process-only-and-phenotypic-cache-design.md.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import skimage as ski

from phenotypic import GridImage, Image, ImagePipeline
from phenotypic.tools_.typing_ import ImageTypeName, ProcessOnlyLayer

logger = logging.getLogger(__name__)


def process_only_output_path(
    output_dir: Path, image_path: Path, input_root: Path, layer: ProcessOnlyLayer
) -> Path:
    """Mirror ``image_path`` (relative to ``input_root``) under ``output_dir``.

    Names the file ``<stem>_<layer>.<ext>`` (``.png`` for objmap, else
    ``.tiff``). Bounded by the 1-level dataset scanner (D12).
    """
    ext = ".png" if layer == "objmap" else ".tiff"
    try:
        rel = image_path.relative_to(input_root)
    except ValueError:
        rel = Path(image_path.name)
    return output_dir / rel.parent / f"{rel.stem}_{layer}{ext}"


def write_process_only_layer(image: Any, layer: ProcessOnlyLayer, out_path: Path) -> None:
    """Write one image layer. TIFF at the image's bit depth for intensity
    layers (float layers quantized via skimage); 16-bit raw-label PNG for
    objmap (D10)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accessor = getattr(image, layer)

    if layer == "objmap":
        if accessor.isempty():
            warnings.warn(
                f"pipeline produced no objects; writing empty object map to {out_path}"
            )
        accessor.imsave(filepath=out_path)  # raw labels (use_label2rgb=False) -> 16-bit PNG
        return

    arr = accessor[:]
    target = image.bit_depth or 8
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = ski.util.img_as_ubyte(arr) if target == 8 else ski.util.img_as_uint(arr)

    import tifffile

    photometric = "rgb" if arr.ndim == 3 and arr.shape[2] >= 3 else "minisblack"
    tifffile.imwrite(out_path, arr, photometric=photometric)


def process_single_apply_only_core(
    pipeline_path: Path,
    image_path: Path,
    input_root: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    layer: ProcessOnlyLayer,
    read_kwargs: Dict[str, Any],
    cli_nrows: Optional[int] = None,
    cli_ncols: Optional[int] = None,
) -> bool:
    """Apply the pipeline to one image and export ``layer``. No measurement.

    Raises on failure (caller logs/handles), mirroring
    :func:`process_single_image_core`.
    """
    pipeline = ImagePipeline.from_json(pipeline_path)
    image_cls = GridImage if image_type == "GridImage" else Image

    read_kwargs = dict(read_kwargs)
    if image_type == "GridImage":
        from ._cli_utils import resolve_grid_shape

        nrows, ncols = resolve_grid_shape(
            cli_nrows=cli_nrows, cli_ncols=cli_ncols,
            pipeline_nrows=pipeline.nrows, pipeline_ncols=pipeline.ncols,
        )
        read_kwargs["nrows"] = nrows
        read_kwargs["ncols"] = ncols

    detect_mode = read_kwargs.pop("detect_mode", "gray")
    image = image_cls.imread(image_path, **read_kwargs)
    if detect_mode != "gray":
        image.set_detect_mode(detect_mode)

    pipeline.apply(image, inplace=True)

    out_path = process_only_output_path(output_dir, image_path, input_root, layer)
    write_process_only_layer(image, layer, out_path)
    return True
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_cli_process_only.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_process_only.py tests/unit/cli/test_cli_process_only.py
git commit -m "feat(cli): process-only output-path mapper + bit-depth-aware layer writer + apply-only worker"
```

---

## Task 4: Worker-CLI `--process-only` / `--input-root` options

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_single.py` (options block + `main` dispatch)
- Test: `tests/unit/cli/test_cli_process_single_options.py` (create)

- [ ] **Step 1: Write the failing test**

```python
from click.testing import CliRunner
from phenotypic._cli._cli_process_single import main


def test_process_only_option_parses(tmp_path, monkeypatch):
    called = {}

    def fake_core(**kwargs):
        called.update(kwargs)
        return True

    monkeypatch.setattr(
        "phenotypic._cli._cli_process_single.process_single_apply_only_core", fake_core
    )
    pipe = tmp_path / "p.json"; pipe.write_text("{}", encoding="utf-8")
    img = tmp_path / "in" / "a.tif"; img.parent.mkdir(parents=True); img.write_bytes(b"x")
    res = CliRunner().invoke(main, [
        "--pipeline", str(pipe), "--image", str(img), "--output-dir", str(tmp_path / "out"),
        "--dataset-name", "in", "--process-only", "detect_mat",
        "--input-root", str(tmp_path / "in"),
    ])
    assert res.exit_code == 0, res.output
    assert called["layer"] == "detect_mat"
    assert str(called["input_root"]) == str(tmp_path / "in")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_cli_process_single_options.py -v`
Expected: FAIL — no `--process-only` option.

- [ ] **Step 3: Implement**

In `_cli_process_single.py`, add two options to `main` (after `--save-inspect`):

```python
@click.option(
    "--process-only",
    "process_only_layer",
    type=click.Choice(["rgb", "gray", "detect_mat", "objmap"]),
    default=None,
    help="Apply-only mode: run pipeline.apply() and save this single layer "
    "(no measurement). Requires --input-root for mirrored output paths.",
)
@click.option(
    "--input-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Root of the input tree, used to compute the mirrored output path "
    "in --process-only mode.",
)
```

Add `process_only_layer: Optional[str]` and `input_root: Optional[Path]` to the `main(...)` signature. Add an import: `from ._cli_process_only import process_single_apply_only_core`. At the **start** of the try-body (before the `measure_only` branch), add:

```python
        if process_only_layer is not None:
            if input_root is None:
                raise click.UsageError("--process-only requires --input-root")
            read_kwargs: Dict[str, Any] = {}
            if bit_depth is not None:
                read_kwargs["bit_depth"] = bit_depth
            if detect_mode != "gray":
                read_kwargs["detect_mode"] = detect_mode
            if event_log is not None:
                append_event(
                    event_log=event_log, dataset=dataset_name, image=image.name,
                    status="started",
                    slurm_job_id=os.environ.get(EnvVar.SLURM_JOB_ID, ""),
                    slurm_array_task_id=os.environ.get(EnvVar.SLURM_ARRAY_TASK_ID, ""),
                )
            click.echo(f"Processing (apply-only, {process_only_layer}) {image.name}...")
            process_single_apply_only_core(
                pipeline_path=pipeline, image_path=image, input_root=input_root,
                output_dir=output_dir, image_type=image_type,  # type: ignore[arg-type]
                layer=process_only_layer,  # type: ignore[arg-type]
                read_kwargs=read_kwargs, cli_nrows=nrows, cli_ncols=ncols,
            )
            if event_log is not None:
                append_completion_event(
                    event_log=event_log, dataset=dataset_name, image=image.name,
                    status="completed", error_msg="",
                )
            click.echo(f"✓ Successfully processed {image.name}")
            sys.exit(0)
```

(The existing `except Exception` block already records failures via `append_failure` into `progress_dir = progress_dir(output_dir)` — confirm that line was converted in Phase 1 Task 3.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_cli_process_single_options.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_process_single.py tests/unit/cli/test_cli_process_single_options.py
git commit -m "feat(cli): worker --process-only/--input-root options"
```

---

## Task 5: Local strategy dispatch (worker + manifest-only finalize)

**Files:**
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py` (`LocalParallelStrategy.execute` + new `_process_single_local_apply_only`)
- Test: `tests/unit/cli/test_local_strategy_process_only.py` (create)

- [ ] **Step 1: Write the failing test**

```python
from datetime import datetime
from pathlib import Path

from phenotypic._cli._cli_execution_strategies import LocalParallelStrategy
from phenotypic.tools_ import manifest_json_path, deliverables_dir


def test_local_process_only_writes_layers_and_manifest_no_deliverables(
    tmp_path, synth_one_level_input, simple_pipeline_json, make_exec_config, make_output_manager
):
    out = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json, input_path=synth_one_level_input,
        output_dir=out, n_jobs=1, force_local=True, process_only_layer="detect_mat",
    )
    strat = LocalParallelStrategy(config, make_output_manager(out))
    # datasets discovered by the CLI scanner; build them here for the unit test
    from phenotypic._cli._cli_directory_scanner import scan_directory_structure, organize_by_dataset
    datasets = organize_by_dataset(scan_directory_structure(synth_one_level_input), out)
    strat.execute(datasets, out)

    # mirrored layer files exist
    tiffs = list(out.rglob("*_detect_mat.tiff"))
    assert tiffs, "no mirrored detect_mat tiffs written"
    # progress manifest written (run console visibility), but no deliverables/dashboard
    assert manifest_json_path(out).is_file()
    assert not deliverables_dir(out).exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_local_strategy_process_only.py -v`
Expected: FAIL — strategy has no process-only branch (writes nothing or errors).

> If `make_exec_config` / `make_output_manager` / `synth_one_level_input` fixtures don't exist, add them to `tests/unit/cli/conftest.py`: `make_exec_config` builds an `ExecutionConfig` with sensible defaults overridable by kwargs; `synth_one_level_input` writes `load_synth_yeast_plate()` to `<tmp>/in/day1/plateA.tif`.

- [ ] **Step 3: Implement**

In `LocalParallelStrategy.execute`, replace the `worker = …` selection (≈line 189) with:

```python
        if self.config.process_only_layer:
            worker = self._process_single_local_apply_only
        elif measure_only:
            worker = self._process_single_local_measure
        else:
            worker = self._process_single_local
```

Replace the finalize block (the `regenerate_dashboard_artifacts(...)` call, ≈line 204-212) with a branch:

```python
        try:
            datasets_totals = {ds.name: len(ds.images) for ds in datasets}
            start_iso = start_time.isoformat(timespec="milliseconds")
            if self.config.process_only_layer:
                # Manifest only — no dashboard HTML, no aggregation (D13).
                from ._dashboard._manifest_builder import build_manifest
                from phenotypic.tools_ import progress_dir as _progress_dir
                build_manifest(
                    output_dir=output_dir,
                    progress_dir=_progress_dir(output_dir),
                    datasets=datasets_totals,
                    execution_mode="local",
                    start_time=start_iso,
                    input_path=self.config.input_path.stem,
                )
            else:
                local_job_meta: dict = {
                    JobMetadataKey.START_TIME: start_iso,
                    JobMetadataKey.INPUT_PATH: self.config.input_path.stem,
                    JobMetadataKey.EXECUTION_MODE: "local",
                }
                regenerate_dashboard_artifacts(output_dir, local_job_meta, datasets_totals)
        except Exception:
            logger.debug("Failed to generate progress artifacts", exc_info=True)
```

Add the new worker method (mirror `_process_single_local`, but call the apply-only core with `input_root=self.config.input_path`):

```python
    def _process_single_local_apply_only(
        self,
        dataset: Dataset,
        image_path: Path,
        output_dir: Path,
        event_log: Path,
    ) -> tuple[str, str, bool, str]:
        """Apply-only (process-only) per-image worker."""
        from ._cli_process_only import process_single_apply_only_core

        append_event(event_log, dataset.name, image_path.name, "started")
        try:
            read_kwargs: Dict[str, Any] = {}
            if self.config.bit_depth:
                read_kwargs["bit_depth"] = self.config.bit_depth
            if self.config.detect_mode != "gray":
                read_kwargs["detect_mode"] = self.config.detect_mode

            process_single_apply_only_core(
                pipeline_path=self.config.pipeline_json,
                image_path=image_path,
                input_root=self.config.input_path,
                output_dir=output_dir,
                image_type=self.config.image_type,
                layer=self.config.process_only_layer,  # type: ignore[arg-type]
                read_kwargs=read_kwargs,
                cli_nrows=self.config.nrows,
                cli_ncols=self.config.ncols,
            )
            append_completion_event(event_log, dataset.name, image_path.name, "completed")
            return (dataset.name, image_path.name, True, "")
        except Exception as e:
            import traceback
            error_msg = str(e)
            tb = traceback.format_exc()
            logger.error("Apply-only failed for %s/%s:\n%s", dataset.name, image_path.name, tb)
            append_completion_event(
                event_log, dataset.name, image_path.name, "failed",
                _truncate_error_message(error_msg),
            )
            try:
                from phenotypic.tools_ import progress_dir as _progress_dir
                append_failure(
                    _progress_dir(output_dir), dataset=dataset.name, image=image_path.name,
                    error_type=type(e).__name__, error_message=error_msg, traceback=tb,
                )
            except Exception:
                logger.warning("Failed to write failure record", exc_info=True)
            return (dataset.name, image_path.name, False, error_msg)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_local_strategy_process_only.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_execution_strategies.py tests/unit/cli/test_local_strategy_process_only.py tests/unit/cli/conftest.py
git commit -m "feat(cli): local strategy process-only dispatch + manifest-only finalize"
```

---

## Task 6: SLURM threading + manifest-only finalize task

**Files:**
- Modify: `src/phenotypic/_cli/_cli_slurm_array_scripts.py` (and `AutonomousSLURMStrategy` in `_cli_execution_strategies.py`)
- Test: `tests/unit/cli/test_slurm_process_only_scripts.py` (create)

> **Read first:** `_cli_slurm_array_scripts.py::generate_all_array_job_scripts` and `AutonomousSLURMStrategy.execute` to see how per-image worker commands and the finalize chain are emitted. The edit threads two new flags into the per-image command and replaces the aggregation finalize with a `build_manifest` call.

- [ ] **Step 1: Write the failing test**

```python
def test_array_script_threads_process_only_and_omits_aggregation(
    tmp_path, simple_pipeline_json, synth_one_level_input, make_exec_config
):
    from phenotypic._cli._cli_slurm_array_scripts import generate_all_array_job_scripts
    from phenotypic._cli._cli_directory_scanner import scan_directory_structure, organize_by_dataset

    out = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json, input_path=synth_one_level_input,
        output_dir=out, process_only_layer="rgb",
        slurm_args={"slurm_partition": "compute"},
    )
    datasets = organize_by_dataset(scan_directory_structure(synth_one_level_input), out)
    scripts = generate_all_array_job_scripts(config, datasets, out)  # adjust to real signature
    blob = "\n".join(p.read_text() for p in scripts) if hasattr(scripts[0], "read_text") else str(scripts)
    assert "--process-only rgb" in blob
    assert "--input-root" in blob
    assert "aggregate_measurements" not in blob   # no measurement aggregation in process-only
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_slurm_process_only_scripts.py -v`
Expected: FAIL — flags not threaded.

- [ ] **Step 3: Implement**

In `_cli_slurm_array_scripts.py`, where the per-image `python -m phenotypic._cli._cli_process_single ...` command is assembled, append (when `config.process_only_layer`):
```python
        f" --process-only {config.process_only_layer}"
        f" --input-root {shlex.quote(str(config.input_path))}"
```
In `AutonomousSLURMStrategy.execute`, when `config.process_only_layer` is set, submit only the image array and a single dependent finalize task that runs `build_manifest` (no `aggregate_measurements`, no dashboard). Reuse the existing finalize-task submission seam but point it at a manifest-only entry (a thin `python -c "from phenotypic._cli._dashboard._manifest_builder import build_manifest; ..."` or a small helper). Guard the existing aggregation/sentinel/checkpoint chain behind `if not config.process_only_layer:`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_slurm_process_only_scripts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_slurm_array_scripts.py src/phenotypic/_cli/_cli_execution_strategies.py tests/unit/cli/test_slurm_process_only_scripts.py
git commit -m "feat(cli): SLURM process-only — thread flags, manifest-only finalize"
```

---

## Task 7: Top-level CLI option, validation, wiring, dry-run

**Files:**
- Modify: `src/phenotypic/phenotypicCLI.py`
- Test: `tests/unit/cli/test_cli_v2.py` (extend) + `tests/integration/cli/`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/cli/test_cli_v2.py`:

```python
from click.testing import CliRunner
from phenotypic.phenotypicCLI import phenotypic_cli


def test_process_only_rejects_measure(tmp_path, simple_pipeline_json, synth_one_level_input):
    r = CliRunner().invoke(phenotypic_cli, [
        "--pipeline", str(simple_pipeline_json), "--input", str(synth_one_level_input),
        "--output-dir", str(tmp_path / "o"), "--process-only", "rgb", "--measure",
    ])
    assert r.exit_code != 0
    assert "process-only" in r.output.lower() and "measure" in r.output.lower()


def test_process_only_warns_ignored_flags(tmp_path, simple_pipeline_json, synth_one_level_input):
    r = CliRunner().invoke(phenotypic_cli, [
        "--pipeline", str(simple_pipeline_json), "--input", str(synth_one_level_input),
        "--output-dir", str(tmp_path / "o2"), "--process-only", "rgb",
        "--no-qc", "--force-local", "--n-jobs", "1", "--dry-run",
    ])
    assert r.exit_code == 0, r.output
    assert "ignored" in r.output.lower()


def test_process_only_dry_run_lists_plan(tmp_path, simple_pipeline_json, synth_one_level_input):
    r = CliRunner().invoke(phenotypic_cli, [
        "--pipeline", str(simple_pipeline_json), "--input", str(synth_one_level_input),
        "--output-dir", str(tmp_path / "o3"), "--process-only", "detect_mat",
        "--dry-run", "--force-local",
    ])
    assert r.exit_code == 0, r.output
    assert "process-only" in r.output.lower()
    assert ".phenotypic" in r.output
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_cli_v2.py -k process_only -v`
Expected: FAIL — no `--process-only` option.

- [ ] **Step 3: Implement**

Add the option to `phenotypic_cli` (after `--no-qc`):

```python
@click.option(
    "--process-only",
    "process_only_layer",
    type=click.Choice(["rgb", "gray", "detect_mat", "objmap"]),
    default=None,
    help="Apply-only mode: run pipeline.apply() and export this single layer "
    "(16-bit-capable TIFF for rgb/gray/detect_mat at the image's bit depth, "
    "raw-label PNG for objmap), mirroring the input tree. Skips measurement, "
    "deliverables, QC, dashboard.",
)
```

Add `process_only_layer: Optional[str]` to the function signature. In the validation block (alongside the `--measure` guards), add:

```python
        if process_only_layer is not None:
            for bad, name in ((measure_only, "--measure"), (recompile is not None, "--recompile")):
                if bad:
                    raise click.UsageError(
                        f"--process-only cannot be combined with {name} "
                        "(conflicting run modes)."
                    )
            if pipeline_json is None or input_path is None:
                raise click.UsageError("--process-only requires --pipeline and --input.")
            for val, name in ((metadata_csv, "--metadata"), (no_qc, "--no-qc"),
                              (no_dataset_column, "--no-dataset-column")):
                if val:
                    click.echo(
                        f"Warning: {name} is ignored in --process-only mode "
                        "(no measurement/aggregation output).", err=True,
                    )
```

Where `ExecutionConfig(...)` is constructed for the forward run, pass `process_only_layer=process_only_layer`. In the dry-run branch, when `process_only_layer` is set, print a plan including: `mode: process-only (<layer>)`, per-dataset image counts, a few sample mirrored output paths (via `process_only_output_path`), execution mode, and the `.phenotypic` cache dir (`phenotypic_cache_dir(output_dir)`).

> Locate the existing `ExecutionConfig(` construction and the dry-run handler in `phenotypicCLI.py`; thread the field and the plan print there. The forward dispatch already routes through the execution strategies (Task 5/6), so no further dispatch wiring is needed once the field is set.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_cli_v2.py -k process_only -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/phenotypicCLI.py tests/unit/cli/test_cli_v2.py
git commit -m "feat(cli): top-level --process-only option, validation, dry-run plan"
```

---

## Task 8: GUI classifier — make process-only runs discoverable

**Files:**
- Modify: `src/phenotypic/gui/shell/_classifier.py` (`Capabilities`, `_classify_dir`)
- Modify: `src/phenotypic/gui/shell/_runs_registry.py` (treat `is_process_only_output` as discoverable)
- Modify: `src/phenotypic/gui/FEATURES.md`
- Test: `tests/unit/gui/test_classifier_process_only.py` (create)

- [ ] **Step 1: Write the failing test**

```python
from phenotypic.gui.shell._classifier import classify
from phenotypic.tools_ import manifest_json_path


def test_process_only_run_is_discoverable(tmp_path):
    # Process-only run: mirrored layer + .phenotypic/progress/manifest.json, no results/deliverables.
    (tmp_path / "day1").mkdir()
    (tmp_path / "day1" / "plateA_detect_mat.tiff").write_bytes(b"II*\x00")
    mp = manifest_json_path(tmp_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text('{"is_complete": true}', encoding="utf-8")
    caps = classify(tmp_path)
    assert caps.is_process_only_output is True
    assert caps.is_cli_output is False          # not a full forward run
    assert caps.has_dashboard is False


def test_forward_run_not_flagged_process_only(tmp_path):
    (tmp_path / "results").mkdir()
    deliv = tmp_path / "deliverables"; deliv.mkdir()
    (deliv / "master_measurements.parquet").write_bytes(b"x")
    caps = classify(tmp_path)
    assert caps.is_cli_output is True
    assert caps.is_process_only_output is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/gui/test_classifier_process_only.py -v`
Expected: FAIL — `Capabilities` has no `is_process_only_output`.

- [ ] **Step 3: Implement**

In `_classifier.py`, add `is_process_only_output: bool` to the `Capabilities` dataclass (default `False` in the `_EMPTY`/`_BAD_PERMS` singletons and the `_classify_dir` return). In `_classify_dir`, after computing `has_deliverables_dir`, detect the process-only signal:

```python
    is_process_only_output = False
    if not (is_cli_output_results and is_cli_output_master):
        # No full-run artifacts: a process-only run is identified by its
        # .phenotypic/progress/manifest.json (machine-state under the hidden dir).
        from phenotypic.tools_ import resolve_manifest_json_path
        is_process_only_output = resolve_manifest_json_path(path).is_file()
```

Add `is_process_only_output=is_process_only_output` to the returned `Capabilities`.

In `_runs_registry._discover_output_dirs`, change the yield predicate to:
```python
                if caps.is_cli_output or caps.is_process_only_output:
                    yield child
```

In `FEATURES.md`, add a row: "Process-only run appears in Recent Runs (progress only, no dashboard)" with `Test ref` = `tests/unit/gui/test_classifier_process_only.py::test_process_only_run_is_discoverable`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/gui/test_classifier_process_only.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/shell/_classifier.py src/phenotypic/gui/shell/_runs_registry.py src/phenotypic/gui/FEATURES.md tests/unit/gui/test_classifier_process_only.py
git commit -m "feat(gui): discover process-only runs via .phenotypic manifest (progress-only)"
```

---

## Task 9: Integration test + docs

**Files:**
- Test: `tests/integration/cli/test_process_only_e2e.py` (create)
- Modify: `CLAUDE.md` (CLI section), `phenotypicCLI.py` docstring, `README.md`

- [ ] **Step 1: Write the failing integration test**

```python
from click.testing import CliRunner

import numpy as np
import tifffile

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.tools_ import manifest_json_path, deliverables_dir, results_dir, phenotypic_cache_dir


def test_process_only_end_to_end(tmp_path, synth_one_level_input, simple_pipeline_json):
    out = tmp_path / "out"
    r = CliRunner().invoke(phenotypic_cli, [
        "--pipeline", str(simple_pipeline_json), "--input", str(synth_one_level_input),
        "--output-dir", str(out), "--process-only", "detect_mat",
        "--force-local", "--n-jobs", "1",
    ])
    assert r.exit_code == 0, r.output
    tiffs = list(out.rglob("*_detect_mat.tiff"))
    assert tiffs, "no mirrored tiffs"
    assert tifffile.imread(tiffs[0]).dtype == np.uint8       # 8-bit source
    assert manifest_json_path(out).is_file()                  # run-console visibility
    assert phenotypic_cache_dir(out).is_dir()
    assert not deliverables_dir(out).exists()                 # no analysis suite
    assert not results_dir(out).exists()
```

- [ ] **Step 2: Run to verify it fails / then passes**

Run: `uv run pytest tests/integration/cli/test_process_only_e2e.py -v`
Expected: PASS once Tasks 1–7 are in. If it fails, the failure points at the missing wiring (worker dispatch or finalize).

- [ ] **Step 3: Docs**

- `phenotypic_cli` docstring: add a `--process-only` paragraph with an example invocation.
- `CLAUDE.md` "CLI" section: add a bullet for `--process-only {rgb|gray|detect_mat|objmap}` (apply-only export; mirrors input tree; TIFF at image bit depth / objmap PNG; state in `.phenotypic/`).
- `README.md`: one line under the CLI usage block.

- [ ] **Step 4: Re-run the integration test + full suites**

Run: `uv run pytest tests/unit/cli tests/unit/gui tests/integration/cli -q`
Expected: PASS.
Run: `uv run mypy src/phenotypic` and `uv run ruff check --fix`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/cli/test_process_only_e2e.py CLAUDE.md src/phenotypic/phenotypicCLI.py README.md
git commit -m "test+docs(cli): process-only end-to-end + CLI docs"
```

---

## Self-Review (completed)

- **Spec coverage:** §5.3 surface → Task 7. §5.4 worker+finalize+input_root → Tasks 3,5,6. §5.5 mirror path (D2/D8/D12) → Task 3. §5.6 bit depth (D10) → Task 3. §5.7 dry-run → Task 7. D3 single layer → Choice type. D9 objmap-empty → Task 3. D13 GUI visibility → Task 8 (corrected: needs a classifier signal, not free). SLURM (D5/full reuse) → Task 6.
- **Placeholder scan:** code shown for every implementation step. The two SLURM/CLI-wiring steps that require reading an existing function first (Task 6 finalize seam, Task 7 `ExecutionConfig(`/dry-run locations) name the exact function and the exact additions — no "handle X" hand-waving.
- **Type consistency:** `ProcessOnlyLayer` (Task 1) used in `ExecutionConfig.process_only_layer` (Task 2), worker `layer` param (Task 3), and both CLI `click.Choice` lists (Tasks 4,7) — same four values. `process_single_apply_only_core` signature is identical at definition (Task 3) and both call sites (Tasks 4,5). `write_process_only_layer`/`process_only_output_path` names match across Tasks 3,5,7. `is_process_only_output` defined and consumed consistently in Task 8.
