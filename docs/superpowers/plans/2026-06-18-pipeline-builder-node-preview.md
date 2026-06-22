# Pipeline Builder — Per-Node Zoomable Layer Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Material "image" icon button to each image-producing builder node card that opens a blocking modal with a zoomable OpenSeadragon viewer and an available-layer toggle, backed by a per-session/per-scope disk HDF cache that computes previews faithfully at full resolution and threads each nested sub-pipeline's real input.

**Architecture:** A new `full_layers` flag on `ImagePipeline.apply_with_intermediates` writes a full v2 HDF snapshot per node to a per-scope temp directory. A new `_preview_cache.py` owns the disk cache, chained-fingerprint staleness, and a recursive `compute_scope` that threads each nested scope's input from its parent's cache. A new `_preview_tiles.py` stages a requested `(scope, node, channel)` HDF layer → PNG → DZI tiles via the existing tiler, served by a new Flask blueprint. The modal (mirroring the existing point picker) hosts an OSD viewer driven by clientside callbacks.

**Tech Stack:** Python 3.12, `uv`, Dash + dash-bootstrap-components, OpenSeadragon 5.x (vendored/CDN), h5py, Pillow, the project's DZI tiler.

## Global Constraints

- **Package manager:** `uv` only. Run tests with `uv run pytest ...`; never bare `python`/`pip`.
- **Lint/type:** `uv run ruff check --fix` and `uv run mypy src/phenotypic` must pass for touched files.
- **Operations are keyword-only pydantic models.** Construct ops with kwargs (`OtsuDetector(ignore_zeros=True)`).
- **Doctests must be runnable** via `load_synth_yeast_plate()`. Imported function-locally: `from phenotypic.data import load_synth_yeast_plate`.
- **GUI is CI-gated:** any PR touching `src/phenotypic/gui/` MUST update `src/phenotypic/gui/FEATURES.md` (the `features-md-gate` job), and `Test ref` on `✅ shipping` rows must resolve.
- **GUI design tokens:** chrome colors from `_design.py` (`COLOR_*`); never inline hex. Data colors `OI_*` only for data series.
- **Dash gotcha:** an `allow_duplicate` single-Output callback that can return one component must wrap its return in a 1-tuple or Dash 500s.
- **Cache root:** `Path(tempfile.gettempdir()) / "phenotypic" / "pipeline-preview"` (browse-tab precedent: `tempfile.gettempdir()/phenotypic/browse`). Wipe on launch + `atexit`.
- **block_ids are globally unique 32-char UUID4 hex.** `scope_path` is a `list[str]` of container block_ids (the breadcrumb); `[]` = root.
- **DZI tiler contract:** `tile(png_path: Path, output_dir: Path, tile_size=254, overlap=1) -> Path` writes `<output_dir>/<png_stem>.dzi` + `<png_stem>_files/<level>/<col>_<row>.png`; idempotent on mtime.

---

## File Structure

**Modify:**
- `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py` — add `full_layers` to `apply_with_intermediates`.
- `src/phenotypic/_core/_image_parts/_image_io_handler.py` — add `load_layer_hdf5` classmethod.
- `src/phenotypic/gui/builder/_ids.py` — preview ids + `action="preview"` usage.
- `src/phenotypic/gui/builder/_linear_layout.py` — preview SVG button in `_block_card`.
- `src/phenotypic/gui/builder/_layout.py` — `build_node_preview_modal()` + mount.
- `src/phenotypic/gui/builder/_callbacks.py` — preview callbacks (open / compute+stage / toggle / clientside).
- `src/phenotypic/gui/builder/_app.py` — register preview route + callbacks.
- `src/phenotypic/gui/FEATURES.md` — new rows.

**Create:**
- `src/phenotypic/gui/builder/_preview_cache.py` — disk cache, fingerprint, manifest, recursive `compute_scope` (input threading).
- `src/phenotypic/gui/builder/_preview_tiles.py` — HDF→PNG staging + DZI tile blueprint.
- `src/phenotypic/gui/builder/assets/preview.js` — `window.__phenotypicNodePreview` OSD glue.
- `tests/unit/core/test_full_layers_intermediates.py`
- `tests/unit/core/test_load_layer_hdf5.py`
- `tests/gui/builder/test_preview_cache.py`
- `tests/gui/builder/test_preview_compute_scope.py`
- `tests/gui/builder/test_preview_tile_blueprint.py`
- `tests/gui/builder/test_preview_button.py`
- `tests/gui/builder/test_node_preview_modal.py`

---

## Task 1: Core `full_layers` flag on `apply_with_intermediates`

**Files:**
- Modify: `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py:884-960`
- Test: `tests/unit/core/test_full_layers_intermediates.py`

**Interfaces:**
- Produces: `ImagePipeline.apply_with_intermediates(image, inplace=False, reset=None, output_dir=None, *, full_layers=False) -> IntermediateResult`. When `output_dir` set and `full_layers=True`, writes `base_00.h5` and each non-read-only op's `{i:02d}_{key}.h5` as a **full v2 snapshot** via `save2hdf5` (so `f["layers"][...]`, `f.attrs["schema_version"]==2`, GridImage `/grid/`). Read-only ops (`MeasureFeatures`, `GridFinder`) write no file. Default `full_layers=False` preserves the existing v1-flat delta behavior.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_full_layers_intermediates.py`:

```python
"""full_layers=True writes complete v2 HDF snapshots per node."""
import h5py
from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector


def test_full_layers_writes_v2_snapshots(tmp_path):
    from phenotypic.data import load_synth_yeast_plate

    image = load_synth_yeast_plate()
    pipeline = ImagePipeline(ops=[GaussianBlur(sigma=1), OtsuDetector()])
    out_dir = tmp_path / "full"

    result = pipeline.apply_with_intermediates(
        image, output_dir=out_dir, full_layers=True
    )

    base = out_dir / "base_00.h5"
    assert base.exists()
    with h5py.File(base, "r") as f:
        assert int(f.attrs["schema_version"]) == 2
        assert "layers" in f
        for layer in ("gray", "detect_mat", "objmap"):
            assert layer in f["layers"]

    enhancer_file = out_dir / "00_GaussianBlur.h5"
    assert enhancer_file.exists()
    with h5py.File(enhancer_file, "r") as f:
        assert "layers" in f
        # full snapshot keeps ALL layers, not just the modified detect_mat
        assert "gray" in f["layers"]
        assert "detect_mat" in f["layers"]
        assert "objmap" in f["layers"]

    detector_file = out_dir / "01_OtsuDetector.h5"
    assert detector_file.exists()
    with h5py.File(detector_file, "r") as f:
        assert "objmap" in f["layers"]
    assert result.image is not None


def test_full_layers_false_keeps_delta_behavior(tmp_path):
    from phenotypic.data import load_synth_yeast_plate

    image = load_synth_yeast_plate()
    pipeline = ImagePipeline(ops=[GaussianBlur(sigma=1)])
    out_dir = tmp_path / "delta"

    pipeline.apply_with_intermediates(image, output_dir=out_dir)  # default

    with h5py.File(out_dir / "00_GaussianBlur.h5", "r") as f:
        # legacy flat layout, only the modified layer
        assert "detect_mat" in f
        assert "layers" not in f
        assert "rgb" not in f
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/core/test_full_layers_intermediates.py -v`
Expected: FAIL — `apply_with_intermediates() got an unexpected keyword argument 'full_layers'`.

- [ ] **Step 3: Implement the flag**

In `_image_pipeline_core.py`, change the signature (lines 884-890) to add a keyword-only param:

```python
    def apply_with_intermediates(
        self,
        image: Image,
        inplace: bool = False,
        reset: Optional[bool] = None,
        output_dir: Optional[Union[str, Path]] = None,
        *,
        full_layers: bool = False,
    ) -> IntermediateResult:
```

Add to the docstring `Args:` block (after the `output_dir` entry, before `Returns:`):

```
            full_layers: When ``True`` and *output_dir* is set, each
                non-read-only operation's full image state is written as a
                complete schema-v2 snapshot via ``save2hdf5`` (all layers +
                class/grid metadata) instead of the delta layout. Used by the
                builder's node-preview cache so any node's HDF reconstructs a
                faithful ``Image``/``GridImage``. Defaults to ``False``.
```

Replace the base-00 write (lines 925-930) so full mode uses `save2hdf5`:

```python
        if output_dir is not None:
            # Save initial base (pre-pipeline state)
            if full_layers:
                img.copy().save2hdf5(output_dir / "base_00.h5")
            else:
                _all_layers = ("rgb", "gray", "detect_mat", "objmap")
                img.copy().save_intermediate_layers(
                    output_dir / "base_00.h5", layers=_all_layers,
                )
```

In the `_capture` closure (lines 938-955), branch on `full_layers` before the delta ladder:

```python
        def _capture(
            i: int,
            key: str,
            current: Image,
            operation: Union[ImageOperation, "ImagePipelineCore"],
        ) -> None:
            if output_dir is not None:
                layers = _layers_modified_by(operation)

                if layers is None:
                    # Read-only op (MeasureFeatures, GridFinder): no file
                    intermediates[key] = None
                elif full_layers:
                    # Faithful full v2 snapshot (all layers + class/grid attrs)
                    current.copy().save2hdf5(output_dir / f"{i:02d}_{key}.h5")
                    intermediates[key] = None
                elif len(layers) == 4:
                    # Corrector: emit a new base with all layers
                    current.copy().save_intermediate_layers(
                        output_dir / f"base_{i:02d}.h5", layers=layers,
                    )
                    intermediates[key] = None
                else:
                    # Delta: save only modified layers
                    current.copy().save_intermediate_layers(
                        output_dir / f"{i:02d}_{key}.h5", layers=layers,
                    )
                    intermediates[key] = None
            else:
                intermediates[key] = current.copy()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/core/test_full_layers_intermediates.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Regression-check the existing delta test + lint**

Run: `uv run pytest tests/unit/core/test_delta_intermediates.py -v && uv run ruff check --fix src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py && uv run mypy src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py`
Expected: existing delta tests still PASS; ruff/mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py tests/unit/core/test_full_layers_intermediates.py
git commit -m "feat(core): full_layers snapshot mode for apply_with_intermediates"
```

---

## Task 2: `load_layer_hdf5` single-layer reader

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py` (insert after `load_hdf5`, ~line 1042)
- Test: `tests/unit/core/test_load_layer_hdf5.py`

**Interfaces:**
- Produces: `Image.load_layer_hdf5(filename, layer: str) -> np.ndarray` — reads one layer (`"rgb"|"gray"|"detect_mat"|"objmap"`) from a v2 (`f["layers"][layer]`) or legacy-flat (`f[layer]`) HDF, read-only, without reconstructing the whole `Image`. Raises `KeyError` if the layer is absent (e.g. `rgb` on a gray-only image).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_load_layer_hdf5.py`:

```python
"""load_layer_hdf5 reads a single layer from v2 and legacy-flat HDFs."""
import numpy as np
import pytest
from phenotypic import Image


def _make_rgb(h=32, w=48):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_load_layer_from_v2(tmp_path):
    img = Image(arr=_make_rgb())
    path = tmp_path / "v2.h5"
    img.save2hdf5(path)

    rgb = Image.load_layer_hdf5(path, "rgb")
    gray = Image.load_layer_hdf5(path, "gray")
    assert rgb.shape == (32, 48, 3)
    assert gray.shape == (32, 48)


def test_load_layer_from_legacy_flat(tmp_path):
    img = Image(arr=_make_rgb())
    path = tmp_path / "flat.h5"
    img.save_intermediate_layers(path, layers=("rgb", "gray"))

    rgb = Image.load_layer_hdf5(path, "rgb")
    assert rgb.shape == (32, 48, 3)


def test_missing_layer_raises(tmp_path):
    img = Image(arr=np.zeros((16, 16), dtype=np.uint8))  # gray-only, no rgb
    path = tmp_path / "gray_only.h5"
    img.save2hdf5(path)
    with pytest.raises(KeyError):
        Image.load_layer_hdf5(path, "rgb")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/core/test_load_layer_hdf5.py -v`
Expected: FAIL — `AttributeError: type object 'Image' has no attribute 'load_layer_hdf5'`.

- [ ] **Step 3: Implement the classmethod**

In `_image_io_handler.py`, immediately after `load_hdf5` (after line 1042, before `save2pickle`), add:

```python
    @classmethod
    def load_layer_hdf5(cls, filename, layer: str) -> np.ndarray:
        """Read a single image layer from an intermediate HDF5 file.

        Reads only the requested dataset without reconstructing a full
        :class:`Image`. Handles both the schema-v2 grouped layout
        (``/layers/<layer>``) and the legacy flat layout (``/<layer>``).

        Args:
            filename: Path to the HDF5 file.
            layer: One of ``"rgb"``, ``"gray"``, ``"detect_mat"``, ``"objmap"``.

        Returns:
            The layer array.

        Raises:
            KeyError: If *layer* is not present in the file.
        """
        with h5py.File(filename, "r") as f:
            schema_version = int(f.attrs.get("schema_version", 1))
            if schema_version >= _SCHEMA_VERSION and "layers" in f:
                group = f["layers"]
            else:
                group = f
            if layer not in group:
                raise KeyError(
                    f"Layer {layer!r} not found in {filename}"
                )
            return group[layer][()]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/core/test_load_layer_hdf5.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint/type-check**

Run: `uv run ruff check --fix src/phenotypic/_core/_image_parts/_image_io_handler.py && uv run mypy src/phenotypic/_core/_image_parts/_image_io_handler.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_core/_image_parts/_image_io_handler.py tests/unit/core/test_load_layer_hdf5.py
git commit -m "feat(core): load_layer_hdf5 single-layer reader"
```

---

## Task 3: Preview disk cache primitives (`_preview_cache.py`)

**Files:**
- Create: `src/phenotypic/gui/builder/_preview_cache.py`
- Test: `tests/gui/builder/test_preview_cache.py`

**Interfaces:**
- Produces:
  - `preview_cache_root() -> Path` — `tempfile.gettempdir()/phenotypic/pipeline-preview`.
  - `init_cache() -> None` / `wipe_cache() -> None` — launch + atexit lifecycle (idempotent).
  - `scope_hash(scope_path: list[str]) -> str` — `sha1("/".join(scope_path))` hexdigest (root `[]` → hash of `""`).
  - `scope_dir(session_id: str, scope_path: list[str]) -> Path` — `root/<session_id>/<scope_hash>/`, created.
  - `wipe_scope(session_id, scope_path) -> None`.
  - `read_manifest(session_id, scope_path) -> Optional[dict]` / `write_manifest(session_id, scope_path, manifest: dict) -> None` — `<scope_dir>/manifest.json`.
  - Manifest schema: `{"fingerprint": str, "scope_key": str, "nodes": {block_id: {"hdf": str, "layers": list[str], "shape": [int,int], "num_objects": int}}, "error": str|None}`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/builder/test_preview_cache.py`:

```python
"""Disk cache primitives: scope dirs, manifest round-trip, lifecycle."""
from phenotypic.gui.builder import _preview_cache as pc


def test_scope_hash_root_vs_nested():
    assert pc.scope_hash([]) == pc.scope_hash([])
    assert pc.scope_hash(["a" * 32]) != pc.scope_hash([])


def test_scope_dir_isolated_per_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    root_dir = pc.scope_dir("sess1", [])
    nested_dir = pc.scope_dir("sess1", ["c" * 32])
    assert root_dir.is_dir()
    assert nested_dir.is_dir()
    assert root_dir != nested_dir


def test_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    manifest = {
        "fingerprint": "abc",
        "scope_key": "",
        "nodes": {"blk": {"hdf": "base_00.h5", "layers": ["rgb"],
                          "shape": [10, 10], "num_objects": 0}},
        "error": None,
    }
    pc.write_manifest("sess1", [], manifest)
    assert pc.read_manifest("sess1", []) == manifest
    assert pc.read_manifest("sess1", ["zzz"]) is None  # missing scope


def test_wipe_scope_removes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    d = pc.scope_dir("sess1", [])
    (d / "base_00.h5").write_bytes(b"x")
    pc.wipe_scope("sess1", [])
    assert not d.exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/gui/builder/test_preview_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.gui.builder._preview_cache'`.

- [ ] **Step 3: Implement the cache primitives**

Create `src/phenotypic/gui/builder/_preview_cache.py`:

```python
"""Disk-backed preview cache for the builder node-preview modal.

One directory per (session, scope). Each scope dir holds full-resolution
per-node HDF snapshots (written by ``apply_with_intermediates(...,
full_layers=True)``), a ``manifest.json`` mapping block_id -> file/layers,
and (lazily) staged PNGs + DZI tile pyramids. The cache lives under the
system temp dir and is wiped on launch + ``atexit``.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

__all__ = [
    "preview_cache_root",
    "init_cache",
    "wipe_cache",
    "scope_hash",
    "scope_dir",
    "wipe_scope",
    "read_manifest",
    "write_manifest",
]

_CACHE_SUBPATH = ("phenotypic", "pipeline-preview")
_atexit_registered = False


def preview_cache_root() -> Path:
    """Cache root (recomputed each call so ``$TMPDIR`` changes are honoured)."""
    return Path(tempfile.gettempdir()).joinpath(*_CACHE_SUBPATH)


def wipe_cache() -> None:
    """Best-effort recursive delete of the cache root. Never raises."""
    shutil.rmtree(preview_cache_root(), ignore_errors=True)


def init_cache() -> None:
    """Wipe stale previews on launch and register an atexit cleanup (idempotent)."""
    global _atexit_registered
    wipe_cache()
    preview_cache_root().mkdir(parents=True, exist_ok=True)
    if not _atexit_registered:
        atexit.register(wipe_cache)
        _atexit_registered = True


def scope_hash(scope_path: list[str]) -> str:
    """Stable hash of a scope_path (list of container block_ids)."""
    return hashlib.sha1("/".join(scope_path).encode("utf-8")).hexdigest()


def scope_dir(session_id: str, scope_path: list[str]) -> Path:
    """Per-(session, scope) directory, created if missing."""
    d = preview_cache_root() / session_id / scope_hash(scope_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def wipe_scope(session_id: str, scope_path: list[str]) -> None:
    """Remove a single scope's cache dir (best-effort)."""
    shutil.rmtree(
        preview_cache_root() / session_id / scope_hash(scope_path),
        ignore_errors=True,
    )


def _manifest_path(session_id: str, scope_path: list[str]) -> Path:
    return scope_dir(session_id, scope_path) / "manifest.json"


def read_manifest(session_id: str, scope_path: list[str]) -> Optional[dict]:
    """Return the scope manifest dict, or None if absent/unreadable."""
    path = _manifest_path(session_id, scope_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(session_id: str, scope_path: list[str], manifest: dict) -> None:
    """Write the scope manifest atomically."""
    path = _manifest_path(session_id, scope_path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest))
    tmp.replace(path)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/gui/builder/test_preview_cache.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint/type-check**

Run: `uv run ruff check --fix src/phenotypic/gui/builder/_preview_cache.py && uv run mypy src/phenotypic/gui/builder/_preview_cache.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/builder/_preview_cache.py tests/gui/builder/test_preview_cache.py
git commit -m "feat(gui): preview disk cache primitives"
```

---

## Task 4: Recursive `compute_scope` with input threading

**Files:**
- Modify: `src/phenotypic/gui/builder/_preview_cache.py`
- Test: `tests/gui/builder/test_preview_compute_scope.py`

**Interfaces:**
- Consumes: `scope_at_path` (`_linear_model.py`), `to_pipeline_dag` + `_find_input_block` + `_topological_image_order` (`_conversion_dag.py`), `_load_preview_image`/`_pipeline_uses_grid` (`_callbacks.py`), `apply_with_intermediates(..., full_layers=True)` (Task 1), `load_hdf5`/`GridImage.load_hdf5` (core), `_DagBuilderState`/`_DagBuilderScope`/`PIPELINE_CLASS_NAME` (`_state.py`).
- Produces: `compute_scope(session_id, state, scope_path, image_path, nrows, ncols) -> dict` — ensures the parent chain is fresh, threads the real input, runs the full-res `full_layers` pipeline for the scope when stale, writes + returns the scope manifest. `_predecessor_block_id(scope, container_id) -> Optional[str]`. `_load_image_auto(path) -> Image|GridImage`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/builder/test_preview_compute_scope.py`:

```python
"""compute_scope: full-res cache, threaded nested input, chained staleness."""
import h5py
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._state import (
    BlockNode, Edge, _DagBuilderState, _DagBuilderScope, _new_block_id,
    state_to_json,
)


def _image_edge(src, tgt):
    return Edge(edge_id=_new_block_id(), source_block_id=src,
                source_port="out", target_block_id=tgt, target_port="in",
                kind="image")


def _linear_root_state(op_blocks):
    """Build a root scope: InputImage -> op_blocks[0] -> ... chained."""
    scope = _DagBuilderScope()  # __post_init__ seeds InputImage at index 0
    input_block = scope.blocks[0]
    scope.blocks.extend(op_blocks)
    prev = input_block.block_id
    for b in op_blocks:
        scope.edges.append(_image_edge(prev, b.block_id))
        prev = b.block_id
    return _DagBuilderState(root=scope)


def test_root_scope_caches_all_nodes(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    blur = BlockNode(block_id=_new_block_id(), class_name="GaussianBlur",
                     params={"sigma": 1})
    state = _linear_root_state([blur])

    manifest = pc.compute_scope("sess1", state, [], image_path=None,
                                nrows=None, ncols=None)

    assert manifest["error"] is None
    # input node + the blur op both have entries
    assert blur.block_id in manifest["nodes"]
    blur_hdf = pc.scope_dir("sess1", [])/ manifest["nodes"][blur.block_id]["hdf"]
    assert blur_hdf.exists()
    with h5py.File(blur_hdf, "r") as f:
        assert "layers" in f


def test_fingerprint_stable_then_invalidates_on_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    blur = BlockNode(block_id=_new_block_id(), class_name="GaussianBlur",
                     params={"sigma": 1})
    state = _linear_root_state([blur])

    fp1 = pc.compute_scope("s", state, [], None, None, None)["fingerprint"]
    fp2 = pc.compute_scope("s", state, [], None, None, None)["fingerprint"]
    assert fp1 == fp2  # no change -> same fingerprint

    blur.params["sigma"] = 5  # edit a param
    state2 = _linear_root_state([blur])
    fp3 = pc.compute_scope("s", state2, [], None, None, None)["fingerprint"]
    assert fp3 != fp1


def test_nested_scope_threads_parent_output(tmp_path, monkeypatch):
    """An inner node sees the parent enhancer's detect_mat, not the raw sample."""
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")

    # Parent scope: InputImage -> GaussianBlur(parent) -> sub-pipeline container
    inner_scope = _DagBuilderScope()
    inner_input = inner_scope.blocks[0]
    inner_op = BlockNode(block_id=_new_block_id(), class_name="GaussianBlur",
                         params={"sigma": 1})
    inner_scope.blocks.append(inner_op)
    inner_scope.edges.append(_image_edge(inner_input.block_id, inner_op.block_id))

    container = BlockNode(block_id=_new_block_id(), class_name="ImagePipeline",
                          params={}, nested=inner_scope)
    parent_blur = BlockNode(block_id=_new_block_id(), class_name="GaussianBlur",
                            params={"sigma": 7})
    state = _linear_root_state([parent_blur, container])

    scope_path = [container.block_id]
    manifest = pc.compute_scope("s", state, scope_path, None, None, None)
    assert manifest["error"] is None
    assert inner_op.block_id in manifest["nodes"]

    # parent + inner scope dirs both exist (no clobbering)
    assert pc.read_manifest("s", []) is not None
    assert pc.read_manifest("s", scope_path) is not None
    # chained fingerprint: inner fp folds in parent fp
    parent_fp = pc.read_manifest("s", [])["fingerprint"]
    assert parent_fp in manifest["fingerprint_inputs"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/gui/builder/test_preview_compute_scope.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'compute_scope'`.

- [ ] **Step 3: Implement `compute_scope` + helpers**

Append to `src/phenotypic/gui/builder/_preview_cache.py` (and extend `__all__` with `"compute_scope"`):

```python
import copy as _copy


def _scope_signature(scope) -> str:
    """Canonical compute-signature of a scope (ignores UI-only state)."""
    blocks = sorted(
        (
            {"id": b.block_id, "cls": b.class_name, "params": b.params}
            for b in scope.blocks
        ),
        key=lambda d: d["id"],
    )
    edges = sorted(
        (
            {"s": e.source_block_id, "t": e.target_block_id,
             "p": e.target_port, "k": e.kind}
            for e in scope.edges
        ),
        key=lambda d: (d["s"], d["t"], d["p"]),
    )
    return json.dumps({"blocks": blocks, "edges": edges}, sort_keys=True)


def _source_identity(image_path, nrows, ncols) -> str:
    from phenotypic.gui.builder._directory_browser import SYNTHETIC_SENTINEL

    key = image_path or SYNTHETIC_SENTINEL
    return f"{key}|{nrows}|{ncols}"


def _promote_scope_state(scope):
    """Build a temp DAG state whose root IS *scope* (whole scope, no prefix)."""
    from phenotypic.gui.builder._state import _DagBuilderState

    return _DagBuilderState(root=_copy.deepcopy(scope))


def _predecessor_block_id(scope, container_id: str):
    """block_id feeding the container's image input (None if it's the source)."""
    for edge in scope.edges:
        if edge.kind == "image" and edge.target_block_id == container_id:
            return edge.source_block_id
    return None


def _load_image_auto(path: Path):
    """Load an HDF snapshot as the class it was saved as (Image/GridImage)."""
    import h5py

    from phenotypic import GridImage, Image

    with h5py.File(path, "r") as f:
        saved = f.attrs.get("phenotypic_class")
        if isinstance(saved, bytes):
            saved = saved.decode("utf-8", errors="replace")
    if saved == "GridImage":
        return GridImage.load_hdf5(path)
    return Image.load_hdf5(path)


def _build_manifest(fingerprint, fingerprint_inputs, scope, pipeline, result,
                    sdir) -> dict:
    """Map the scope's input + op blocks to their HDF files + layer metadata."""
    from phenotypic.gui.builder._conversion_dag import (
        _find_input_block, _topological_image_order,
    )
    from phenotypic.gui.builder._state import PIPELINE_CLASS_NAME, stage_of

    import h5py

    input_block = _find_input_block(scope)
    order = _topological_image_order(scope, input_block)
    non_input = [b for b in order if b.block_id != input_block.block_id]
    ops_blocks = [
        b for b in non_input
        if b.class_name == PIPELINE_CLASS_NAME or stage_of(b.class_name) == "ops"
    ]

    nodes: dict = {}

    def _describe(block_id, filename):
        path = sdir / filename
        if not path.exists():
            return
        layers, shape, num_objects = [], [0, 0], 0
        with h5py.File(path, "r") as f:
            grp = f["layers"] if "layers" in f else f
            for layer in ("rgb", "gray", "detect_mat", "objmap"):
                if layer in grp:
                    layers.append(layer)
            if "gray" in grp:
                shape = list(grp["gray"].shape[:2])
            if "objmap" in grp:
                import numpy as np
                num_objects = int(np.max(grp["objmap"][()])) if grp["objmap"].size else 0
        nodes[block_id] = {
            "hdf": filename, "layers": layers, "shape": shape,
            "num_objects": num_objects,
        }

    _describe(input_block.block_id, "base_00.h5")
    # Invariant: pipeline.get_ops() insertion order == _topological_image_order
    # over the same scope == _run_operations' {i:02d}_{key}.h5 naming. The three
    # must stay in lockstep; do not reorder one without the others.
    for i, (op_key, block) in enumerate(zip(pipeline.get_ops().keys(), ops_blocks)):
        _describe(block.block_id, f"{i:02d}_{op_key}.h5")

    return {
        "fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
        "scope_key": "/".join([]),  # filled by caller via manifest mutation
        "nodes": nodes,
        "error": None,
    }


def compute_scope(session_id, state, scope_path, image_path, nrows, ncols) -> dict:
    """Ensure a scope's full-res preview cache is fresh; return its manifest.

    Recursive: a nested scope's input is threaded from its parent's cache
    (the container's main-flow predecessor HDF). Fingerprints chain so any
    upstream edit invalidates this scope and its descendants.
    """
    from phenotypic.abc_ import GridOperation
    from phenotypic.gui.builder._conversion_dag import to_pipeline_dag
    from phenotypic.gui.builder._linear_model import scope_at_path

    scope = scope_at_path(state.root, list(scope_path))
    if scope is None:
        raise ValueError("compute_scope: stale scope_path")

    promoted = _promote_scope_state(scope)
    pipeline = to_pipeline_dag(promoted)
    sig = _scope_signature(scope)

    if not scope_path:
        input_identity = _source_identity(image_path, nrows, ncols)
        parent_fp = ""
    else:
        parent_manifest = compute_scope(
            session_id, state, list(scope_path[:-1]), image_path, nrows, ncols,
        )
        parent_fp = parent_manifest["fingerprint"]
        input_identity = parent_fp

    fingerprint_inputs = [sig, input_identity]
    fingerprint = hashlib.sha1("\x00".join(fingerprint_inputs).encode()).hexdigest()

    cached = read_manifest(session_id, list(scope_path))
    if cached is not None and cached.get("fingerprint") == fingerprint \
            and cached.get("error") is None:
        return cached

    wipe_scope(session_id, list(scope_path))
    sdir = scope_dir(session_id, list(scope_path))

    try:
        if not scope_path:
            from phenotypic.gui.builder._callbacks import (
                _load_preview_image, _pipeline_uses_grid,
            )
            uses_grid = _pipeline_uses_grid(pipeline, GridOperation)
            image = _load_preview_image(image_path, uses_grid, nrows, ncols)
        else:
            parent_scope = scope_at_path(state.root, list(scope_path[:-1]))
            container_id = scope_path[-1]
            pred_id = _predecessor_block_id(parent_scope, container_id)
            parent_dir = scope_dir(session_id, list(scope_path[:-1]))
            if pred_id is None or pred_id not in parent_manifest["nodes"]:
                pred_file = "base_00.h5"
            else:
                pred_file = parent_manifest["nodes"][pred_id]["hdf"]
            image = _load_image_auto(parent_dir / pred_file)

        result = pipeline.apply_with_intermediates(
            image, output_dir=sdir, full_layers=True,
        )
        manifest = _build_manifest(
            fingerprint, fingerprint_inputs, scope, pipeline, result, sdir,
        )
        manifest["scope_key"] = "/".join(scope_path)
    except Exception as exc:  # noqa: BLE001
        manifest = {
            "fingerprint": fingerprint, "fingerprint_inputs": fingerprint_inputs,
            "scope_key": "/".join(scope_path), "nodes": {},
            "error": f"{type(exc).__name__}: {exc}",
        }

    write_manifest(session_id, list(scope_path), manifest)
    return manifest
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/gui/builder/test_preview_compute_scope.py -v`
Expected: PASS (3 passed). If `to_pipeline_dag` needs the operation registry, the import of `_conversion_dag` triggers `get_registry()` lazily; the synth ops (`GaussianBlur`) are core-registered, so no app context is required.

- [ ] **Step 5: Lint/type-check**

Run: `uv run ruff check --fix src/phenotypic/gui/builder/_preview_cache.py && uv run mypy src/phenotypic/gui/builder/_preview_cache.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/builder/_preview_cache.py tests/gui/builder/test_preview_compute_scope.py
git commit -m "feat(gui): recursive compute_scope with threaded nested input"
```

---

## Task 5: HDF→PNG staging + DZI tile blueprint (`_preview_tiles.py`)

**Files:**
- Create: `src/phenotypic/gui/builder/_preview_tiles.py`
- Test: `tests/gui/builder/test_preview_tile_blueprint.py`

**Interfaces:**
- Consumes: `load_layer_hdf5` (Task 2), `_preview_cache.scope_dir`/`read_manifest`, `_image_renderer._normalize_to_uint8`/`_label_map_to_rgb`/`to_overlay_rgb_array`, `_dzi_tiler.tile`, `is_safe_path_component`.
- Produces:
  - `stage_channel_png(scope_dir: Path, block_id: str, channel: str, hdf_path: Path) -> Path` — read layer → uint8/colorized PNG at `<scope_dir>/tiles_src/<block>__<channel>.png` (idempotent). `channel ∈ {"rgb","gray","detect_mat","objmap","overlay"}`.
  - `preview_dzi_url(url_prefix, session_id, scope_hash, block_id, channel) -> str`.
  - `register_node_preview_routes(app) -> None` — Flask blueprint at `/preview-tiles` with `<session_id>/<scope_hash>/<block_id>/<channel>.dzi` + `..._files/<level>/<filename>`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/builder/test_preview_tile_blueprint.py`:

```python
"""Preview tile blueprint: stage HDF layer -> DZI; reject bad components."""
import numpy as np
from phenotypic import Image
from phenotypic.gui.builder._app import create_app
from phenotypic.gui.builder import _preview_cache as pc


def _seed_scope_hdf(session_id, block_id):
    # NOTE: caller must monkeypatch pc.preview_cache_root AND create the app
    # FIRST — create_app runs init_preview_cache() which wipes the cache root.
    # Seeding after app creation keeps the fixture alive.
    sdir = pc.scope_dir(session_id, [])
    img = Image(arr=np.zeros((48, 64, 3), dtype=np.uint8))
    hdf = sdir / "base_00.h5"
    img.save2hdf5(hdf)
    manifest = {"fingerprint": "fp", "scope_key": "", "error": None,
                "nodes": {block_id: {"hdf": "base_00.h5",
                                     "layers": ["rgb", "gray", "detect_mat", "objmap"],
                                     "shape": [48, 64], "num_objects": 0}}}
    pc.write_manifest(session_id, [], manifest)
    return pc.scope_hash([])


def test_preview_dzi_served(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    app = create_app(image_root=tmp_path)  # wipes the (empty) tmp cache root
    sid = "previewsess0001"
    blk = "b" * 32
    shash = _seed_scope_hdf(sid, blk)  # seed AFTER app creation survives the wipe
    client = app.server.test_client()
    resp = client.get(f"/preview-tiles/{sid}/{shash}/{blk}/gray.dzi")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "<Image" in body and "deepzoom" in body.lower()


def test_preview_rejects_bad_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    app = create_app(image_root=tmp_path)
    sid = "previewsess0001"
    blk = "b" * 32
    shash = _seed_scope_hdf(sid, blk)
    client = app.server.test_client()
    resp = client.get(f"/preview-tiles/{sid}/{shash}/{blk}/bogus.dzi")
    assert resp.status_code == 404
```

> **Fix (plan review #1):** `create_app` runs `init_preview_cache()` which wipes the cache root, so the app must be constructed **before** seeding the cache, with the `preview_cache_root` monkeypatch applied first. Same ordering applies in Task 9's integration test.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/gui/builder/test_preview_tile_blueprint.py -v`
Expected: FAIL — 404 for `gray.dzi` (route not registered yet).

- [ ] **Step 3: Implement staging + blueprint**

Create `src/phenotypic/gui/builder/_preview_tiles.py`:

```python
"""HDF-layer -> PNG staging + DZI tile blueprint for the node-preview modal.

The renderer (`_image_renderer`) and the DZI tiler stay unchanged: for a
requested (scope, node, channel) we read the layer from the node's HDF,
project it to an 8-bit PNG, and hand the PNG path to ``_dzi_tiler.tile``.
With pyvips installed the tiler streams; resident RAM stays near zero.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import dash
import numpy as np
from flask import Blueprint, Response, send_from_directory
from PIL import Image as PILImage
from werkzeug.utils import secure_filename

from phenotypic import Image
from phenotypic.gui._shared.tiles import is_safe_path_component
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._image_renderer import (
    _label_map_to_rgb, _normalize_to_uint8,
)
from phenotypic.gui.results_viewer import _dzi_tiler
from phenotypic.gui.results_viewer._tile_routes import _TILE_NAME_RE, _json_error

logger = logging.getLogger(__name__)

PREVIEW_TILES_PREFIX = "/preview-tiles"
_VALID_CHANNELS = ("rgb", "gray", "detect_mat", "objmap", "overlay")
_HASH_RE = re.compile(r"^[0-9a-f]{40}$")

__all__ = [
    "PREVIEW_TILES_PREFIX",
    "stage_channel_png",
    "preview_dzi_url",
    "register_node_preview_routes",
]


def _src_png_path(scope_dir: Path, block_id: str, channel: str) -> Path:
    return scope_dir / "tiles_src" / f"{block_id}__{channel}.png"


def _channel_to_rgb_uint8(hdf_path: Path, channel: str) -> np.ndarray:
    if channel == "overlay":
        detect = Image.load_layer_hdf5(hdf_path, "detect_mat")
        objmap = Image.load_layer_hdf5(hdf_path, "objmap")
        base = _normalize_to_uint8(detect)
        base = base[..., :3] if base.ndim == 3 else np.stack([base] * 3, -1)
        try:
            from skimage.color import label2rgb
            rgb = label2rgb(objmap, image=base, bg_label=0, alpha=0.4,
                            image_alpha=1.0, kind="overlay")
            return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        except Exception:  # noqa: BLE001
            return _label_map_to_rgb(objmap)
    arr = Image.load_layer_hdf5(hdf_path, channel)
    if channel == "objmap":
        return _label_map_to_rgb(arr)
    u8 = _normalize_to_uint8(arr)
    if u8.ndim == 2:
        return np.stack([u8] * 3, axis=-1)
    return u8[..., :3]


def stage_channel_png(scope_dir: Path, block_id: str, channel: str,
                      hdf_path: Path) -> Path:
    """Render a channel from a node HDF to a cached PNG (idempotent)."""
    png_path = _src_png_path(scope_dir, block_id, channel)
    if png_path.exists() and png_path.stat().st_mtime >= hdf_path.stat().st_mtime:
        return png_path
    rgb = _channel_to_rgb_uint8(hdf_path, channel)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(rgb, mode="RGB").save(png_path, format="PNG")
    return png_path


def preview_dzi_url(url_prefix: str, session_id: str, scope_hash: str,
                    block_id: str, channel: str) -> str:
    base = url_prefix if url_prefix.endswith("/") else f"{url_prefix}/"
    return f"{base}preview-tiles/{session_id}/{scope_hash}/{block_id}/{channel}.dzi"


def _validate(session_id, scope_hash, block_id, channel) -> Optional[Response]:
    if (
        is_safe_path_component(session_id)
        and bool(_HASH_RE.match(scope_hash))
        and is_safe_path_component(block_id)
        and channel in _VALID_CHANNELS
    ):
        return None
    return _json_error("invalid preview tile request", 404)


def register_node_preview_routes(app: dash.Dash) -> None:
    """Register the preview DZI blueprint on the Flask server."""
    bp = Blueprint("builder_node_preview", __name__, url_prefix=PREVIEW_TILES_PREFIX)

    @bp.route("/<session_id>/<scope_hash>/<block_id>/<channel>.dzi")
    def manifest(session_id, scope_hash, block_id, channel) -> Response:
        err = _validate(session_id, scope_hash, block_id, channel)
        if err is not None:
            return err
        sdir = pc.preview_cache_root() / session_id / scope_hash
        manifest_path = sdir / "manifest.json"
        if not manifest_path.exists():
            return _json_error("scope not cached", 404)
        import json
        nodes = json.loads(manifest_path.read_text()).get("nodes", {})
        node = nodes.get(block_id)
        if node is None:
            return _json_error("node not cached", 404)
        hdf_path = sdir / node["hdf"]
        if not hdf_path.exists():
            return _json_error("node hdf missing", 404)
        try:
            png_path = stage_channel_png(sdir, block_id, channel, hdf_path)
            _dzi_tiler.tile(png_path, sdir / "dzi")
        except Exception:  # noqa: BLE001
            logger.exception("preview tile generation failed")
            return _json_error("tile generation failed", 500)
        return send_from_directory(
            sdir / "dzi", f"{block_id}__{channel}.dzi",
            mimetype="application/xml",
        )

    @bp.route("/<session_id>/<scope_hash>/<block_id>/<channel>_files/<int:level>/<filename>")
    def tile_endpoint(session_id, scope_hash, block_id, channel, level,
                      filename) -> Response:
        err = _validate(session_id, scope_hash, block_id, channel)
        if err is not None:
            return err
        secured = secure_filename(filename)
        if secured != filename or not _TILE_NAME_RE.match(filename):
            return _json_error("invalid tile filename", 404)
        tile_dir = (
            pc.preview_cache_root() / session_id / scope_hash / "dzi"
            / f"{block_id}__{channel}_files" / str(level)
        )
        if not tile_dir.is_dir():
            return _json_error("tile cache missing", 404)
        return send_from_directory(tile_dir, filename, mimetype="image/png")

    app.server.register_blueprint(bp)
    logger.debug("Registered node-preview tile routes under %s", PREVIEW_TILES_PREFIX)
```

Wire the route into `src/phenotypic/gui/builder/_app.py`. Add to the point-picker import block (~line 33):

```python
from phenotypic.gui.builder._preview_tiles import register_node_preview_routes
from phenotypic.gui.builder._preview_cache import init_cache as init_preview_cache
```

After `register_point_picker_routes(app, image_root)` (~line 146):

```python
    register_node_preview_routes(app)
    init_preview_cache()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/gui/builder/test_preview_tile_blueprint.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint/type-check**

Run: `uv run ruff check --fix src/phenotypic/gui/builder/_preview_tiles.py src/phenotypic/gui/builder/_app.py && uv run mypy src/phenotypic/gui/builder/_preview_tiles.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/builder/_preview_tiles.py src/phenotypic/gui/builder/_app.py tests/gui/builder/test_preview_tile_blueprint.py
git commit -m "feat(gui): node-preview HDF->DZI staging blueprint"
```

---

## Task 6: Node-card preview button + `preview` action id

**Files:**
- Modify: `src/phenotypic/gui/builder/_ids.py` (no new id needed; reuse `linear_node_action_id`)
- Modify: `src/phenotypic/gui/builder/_linear_layout.py:252-291, 582-626`
- Test: `tests/gui/builder/test_preview_button.py`

**Interfaces:**
- Produces: `_preview_button(*, scope_path, block_id, surface="map") -> html.Button` rendered into the `linear-node-header` of image-producing nodes only (registry category ∈ {Enhancer, Detector, Refiner, Corrector} or `class_name == PIPELINE_CLASS_NAME`). Id = `linear_node_action_id(action="preview", scope_path=..., block_id=..., surface=...)`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/builder/test_preview_button.py`:

```python
"""Image-producing node cards carry a preview action button; measure nodes don't."""
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._linear_layout import _preview_button


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for c in children:
        yield from _walk(c)


def test_preview_button_has_preview_action_id():
    btn = _preview_button(scope_path=[], block_id="b" * 32)
    assert btn.id == ids.linear_node_action_id(
        action="preview", scope_path=[], block_id="b" * 32
    )


def test_preview_action_id_shape():
    pid = ids.linear_node_action_id(action="preview", scope_path=[], block_id="b" * 32)
    assert pid["type"] == ids.LINEAR_NODE_ACTION
    assert pid["action"] == "preview"
    assert pid["block_id"] == "b" * 32
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/gui/builder/test_preview_button.py -v`
Expected: FAIL — `ImportError: cannot import name '_preview_button'`.

- [ ] **Step 3: Implement the button + splice into the card**

In `_linear_layout.py`, add near `_help_button` (after line 291):

```python
def _preview_button(
    *,
    scope_path: Iterable[str],
    block_id: Optional[str],
    surface: str = "map",
) -> Any:
    """Render the node-output preview (zoomable modal) trigger button."""

    # Material Design "image" glyph as an inline SVG (no icon font dependency).
    icon = html.Img(
        src=(
            "data:image/svg+xml;utf8,"
            "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' "
            "viewBox='0 0 24 24' fill='currentColor'><path d='M21 19V5c0-1.1-.9-2-2-2"
            "H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01"
            "L14.5 12l4.5 6H5l3.5-4.5z'/></svg>"
        ),
        style={"width": "16px", "height": "16px"},
    )
    return html.Button(
        icon,
        id=ids.linear_node_action_id(
            action="preview",
            scope_path=list(scope_path),
            block_id=block_id,
            surface=surface,
        ),
        type="button",
        n_clicks=0,
        title="Preview node output",
        className="linear-preview-button",
        **{"aria-label": "Preview node output"},  # type: ignore[arg-type]
    )
```

Add an image-producing predicate near the top of `_linear_layout.py` (after imports):

```python
_IMAGE_PRODUCING_CATEGORIES = {"Enhancer", "Detector", "Refiner", "Corrector"}


def _is_image_producing(registry: Any, class_name: str) -> bool:
    from phenotypic.gui.builder._state import PIPELINE_CLASS_NAME

    if class_name == PIPELINE_CLASS_NAME:
        return True
    try:
        info = registry.get(class_name)
    except Exception:  # noqa: BLE001
        return False
    return getattr(info, "category", None) in _IMAGE_PRODUCING_CATEGORIES
```

In `_block_card`, splice the preview button into the `linear-node-header` `html.Div` (at `_linear_layout.py:591-602`, which already holds badges + `_help_button(...)`). Change its `children` list to append the preview button for image-producing nodes:

```python
        html.Div(
            [
                html.Div(badges, className="linear-node-badges"),
                *_help_button(
                    action_scope="node",
                    scope_path=scope_path,
                    block_id=block.block_id,
                    title=_op_doc(registry, block.class_name),
                ),
                *(
                    [_preview_button(scope_path=scope_path, block_id=block.block_id)]
                    if _is_image_producing(registry, block.class_name)
                    else []
                ),
            ],
            className="linear-node-header",
        ),
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/gui/builder/test_preview_button.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint/type-check**

Run: `uv run ruff check --fix src/phenotypic/gui/builder/_linear_layout.py && uv run mypy src/phenotypic/gui/builder/_linear_layout.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/builder/_linear_layout.py tests/gui/builder/test_preview_button.py
git commit -m "feat(gui): preview SVG button on image-producing node cards"
```

---

## Task 7: Preview modal, ids, and OSD client JS

**Files:**
- Modify: `src/phenotypic/gui/builder/_ids.py` (add preview ids)
- Modify: `src/phenotypic/gui/builder/_layout.py:60, 1995-2045, 4511-4524`
- Create: `src/phenotypic/gui/builder/assets/preview.js`
- Test: `tests/gui/builder/test_node_preview_modal.py`

**Interfaces:**
- Produces: ids `MODAL_NODE_PREVIEW`, `MODAL_NODE_PREVIEW_TITLE`, `PREVIEW_OSD_DIV`, `PREVIEW_LAYER_RADIO`, `PREVIEW_CAPTION`, `PREVIEW_DZI_URL_STORE`, `STORE_PREVIEW_TARGET`, `PREVIEW_OSD_MOUNT_TRIGGER`; `build_node_preview_modal() -> dbc.Modal`; JS `window.__phenotypicNodePreview.{mountViewer(divId, dziUrl), disposeViewer()}`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/builder/test_node_preview_modal.py`:

```python
"""Preview modal mounts in the layout with the expected sub-components."""
from pathlib import Path
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._layout import build_node_preview_modal


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for c in children:
        yield from _walk(c)


def test_modal_has_blocking_props_and_children():
    modal = build_node_preview_modal()
    assert modal.id == ids.MODAL_NODE_PREVIEW
    assert modal.backdrop == "static"
    assert modal.is_open is False
    found = {getattr(n, "id", None) for n in _walk(modal)}
    assert ids.PREVIEW_OSD_DIV in found
    assert ids.PREVIEW_LAYER_RADIO in found
    assert ids.PREVIEW_CAPTION in found
    assert ids.PREVIEW_DZI_URL_STORE in found


def test_preview_js_asset_exists():
    js = Path("src/phenotypic/gui/builder/assets/preview.js")
    assert js.exists()
    text = js.read_text()
    assert "__phenotypicNodePreview" in text
    assert "mountViewer" in text and "disposeViewer" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/gui/builder/test_node_preview_modal.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_node_preview_modal'`.

- [ ] **Step 3a: Add ids**

Append to `src/phenotypic/gui/builder/_ids.py` (near the other `MODAL_*`/`PICKER_*` ids, ~line 829):

```python
MODAL_NODE_PREVIEW = "modal-node-preview"
MODAL_NODE_PREVIEW_TITLE = "modal-node-preview-title"
PREVIEW_OSD_DIV = "preview-osd"
PREVIEW_LAYER_RADIO = "preview-layer-radio"
PREVIEW_CAPTION = "preview-caption"
PREVIEW_DZI_URL_STORE = "preview-dzi-url-store"
STORE_PREVIEW_TARGET = "store-preview-target"
PREVIEW_OSD_MOUNT_TRIGGER = "preview-osd-mount-trigger"
```

- [ ] **Step 3b: Add the modal builder**

In `src/phenotypic/gui/builder/_layout.py`, **first add `COLOR_IMAGE_STAGE_DARK` to the `_design` import block** (lines 32-39, which currently imports `COLOR_BLUE, COLOR_BORDER, COLOR_GOLD, COLOR_MUTED, COLOR_NAVY, COLOR_SURFACE, COLOR_WHITE`) — it is NOT imported there yet (only `_point_picker.py` imports it), and the modal references it. Then add `build_node_preview_modal` near `build_confirm_delete_modal` (after line 2045):

```python
def build_node_preview_modal() -> dbc.Modal:
    """Blocking modal hosting a zoomable OSD viewer with a layer toggle."""
    return dbc.Modal(
        id=ids.MODAL_NODE_PREVIEW,
        is_open=False,
        size="xl",
        backdrop="static",
        keyboard=False,
        scrollable=False,
        children=[
            dbc.ModalHeader(dbc.ModalTitle("Preview", id=ids.MODAL_NODE_PREVIEW_TITLE)),
            dbc.ModalBody(
                [
                    dbc.RadioItems(
                        id=ids.PREVIEW_LAYER_RADIO,
                        options=[],
                        value=None,
                        inline=True,
                        className="mb-2",
                    ),
                    dcc.Loading(
                        html.Div(
                            id=ids.PREVIEW_OSD_DIV,
                            className="node-preview-osd",
                            style={
                                "height": "70vh",
                                "width": "100%",
                                "background": COLOR_IMAGE_STAGE_DARK,
                            },
                            **{"data-testid": "node-preview-osd-canvas"},  # type: ignore[arg-type]
                        ),
                    ),
                    html.Small(id=ids.PREVIEW_CAPTION, className="text-muted d-block mt-2"),
                    dcc.Store(id=ids.PREVIEW_DZI_URL_STORE, data=None),
                    dcc.Store(id=ids.STORE_PREVIEW_TARGET, data=None),
                    dcc.Store(id=ids.PREVIEW_OSD_MOUNT_TRIGGER, data=None),
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="btn-preview-close", color="secondary",
                           outline=True, n_clicks=0)
            ),
        ],
    )
```

Mount it in the `modals = html.Div([...])` block (line 4511) by adding `build_node_preview_modal(),` after `build_point_picker_modal(),`.

- [ ] **Step 3c: Add the OSD client JS**

Create `src/phenotypic/gui/builder/assets/preview.js` (mirrors `point_picker.js` sections A+B, minus click capture):

```javascript
// Node-preview OSD glue. Exposes window.__phenotypicNodePreview.
(function () {
    "use strict";
    const ns = window.__phenotypicNodePreview =
        window.__phenotypicNodePreview || {};

    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix : "/";

    function siblingPrefix(prefix, mountName) {
        let base = prefix.endsWith("/") ? prefix : prefix + "/";
        if (base.endsWith("/builder/")) {
            base = base.slice(0, -"builder/".length);
        }
        return base + mountName + "/";
    }
    const resultsPrefix = siblingPrefix(appPrefix, "results");

    let viewer = null;

    function loadOSD(cb) {
        if (window.OpenSeadragon) { cb(); return; }
        const cdn = document.createElement("script");
        cdn.src = "https://cdn.jsdelivr.net/npm/openseadragon@5/build/openseadragon/openseadragon.min.js";
        cdn.onload = cb;
        cdn.onerror = function () {
            const v = document.createElement("script");
            v.src = resultsPrefix + "assets/openseadragon/openseadragon.min.js";
            v.onload = cb;
            document.head.appendChild(v);
        };
        document.head.appendChild(cdn);
    }

    ns.mountViewer = function (divId, dziUrl) {
        loadOSD(function () {
            const el = document.getElementById(divId);
            if (!el || !dziUrl) { return; }
            if (viewer && viewer._phenotypicDziUrl === dziUrl) { return; }
            if (viewer) { viewer.destroy(); viewer = null; }
            viewer = window.OpenSeadragon({
                element: el,
                prefixUrl: resultsPrefix + "assets/openseadragon/images/",
                tileSources: dziUrl,
                showNavigator: false,
                immediateRender: false,
            });
            viewer._phenotypicDziUrl = dziUrl;
        });
    };

    ns.disposeViewer = function () {
        if (viewer) { viewer.destroy(); viewer = null; }
    };
})();
```

Add a scoped `src/phenotypic/gui/builder/assets/.gitattributes` line `openseadragon/* binary` only if a vendored OSD copy is added under builder assets (the results_viewer copy is reused via `resultsPrefix`, so usually none is needed here).

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/gui/builder/test_node_preview_modal.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint/type-check**

Run: `uv run ruff check --fix src/phenotypic/gui/builder/_ids.py src/phenotypic/gui/builder/_layout.py && uv run mypy src/phenotypic/gui/builder/_layout.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/builder/_ids.py src/phenotypic/gui/builder/_layout.py src/phenotypic/gui/builder/assets/preview.js tests/gui/builder/test_node_preview_modal.py
git commit -m "feat(gui): node-preview modal, ids, and OSD client glue"
```

---

## Task 8: Preview callbacks (open / compute+stage / toggle / clientside)

**Files:**
- Modify: `src/phenotypic/gui/builder/_callbacks.py` (inside `register_callbacks`, near the picker clientside block ~6469-6526)
- Test: `tests/gui/builder/test_preview_callbacks.py`

**Interfaces:**
- Consumes: `compute_scope` (Task 4), `stage_channel_png`/`preview_dzi_url` (Task 5), `scope_dir`/`scope_hash` (Task 3), `render_node_preview` channel rule, `state_from_json`, ids from Task 7.
- Produces (callback behaviors): preview-action click → open modal + write `STORE_PREVIEW_TARGET` `{block_id, scope_path}`; target change → `compute_scope`, populate `PREVIEW_LAYER_RADIO` options/value (available layers; objmap/overlay only when `num_objects>0`), `MODAL_NODE_PREVIEW_TITLE`, `PREVIEW_CAPTION`, `PREVIEW_DZI_URL_STORE`; radio change → re-point `PREVIEW_DZI_URL_STORE` + caption; clientside `PREVIEW_DZI_URL_STORE`→ `mountViewer`, `MODAL_NODE_PREVIEW.is_open`(False)→`disposeViewer`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/builder/test_preview_callbacks.py`. This drives the open + compute callbacks directly as plain functions via a thin registration capture. The simplest robust test exercises the compute helper that the callback delegates to, so factor the body into a module-level helper `build_preview_payload`:

```python
"""Preview compute callback delegate: builds layer options + DZI url for a node."""
import numpy as np
from phenotypic import Image
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._preview_callbacks import build_preview_payload
from phenotypic.gui.builder._state import (
    BlockNode, Edge, _DagBuilderState, _DagBuilderScope, _new_block_id, state_to_json,
)


def _image_edge(src, tgt):
    return Edge(edge_id=_new_block_id(), source_block_id=src, source_port="out",
                target_block_id=tgt, target_port="in", kind="image")


def test_build_payload_lists_available_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    scope = _DagBuilderScope()
    inp = scope.blocks[0]
    det = BlockNode(block_id=_new_block_id(), class_name="OtsuDetector", params={})
    scope.blocks.append(det)
    scope.edges.append(_image_edge(inp.block_id, det.block_id))
    state = _DagBuilderState(root=scope)

    payload = build_preview_payload(
        session_id="sess-preview-01",
        state_data=state_to_json(state),
        block_id=det.block_id,
        scope_path=[],
        image_path=None, nrows=None, ncols=None, url_prefix="/",
    )
    assert payload["error"] is None
    layer_values = {opt["value"] for opt in payload["options"]}
    assert {"rgb", "gray", "detect_mat", "objmap", "overlay"} & layer_values
    assert payload["dzi_url"].endswith(".dzi")
    assert payload["value"] in layer_values
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/gui/builder/test_preview_callbacks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.gui.builder._preview_callbacks'`.

- [ ] **Step 3a: Implement the delegate module**

Create `src/phenotypic/gui/builder/_preview_callbacks.py`:

```python
"""Pure helpers for the node-preview modal callbacks (unit-testable)."""
from __future__ import annotations

from typing import Any, Optional

from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._preview_tiles import preview_dzi_url

_LAYER_LABELS = {
    "rgb": "RGB", "gray": "Gray", "detect_mat": "Detect",
    "objmap": "Objmap", "overlay": "Overlay",
}
_LAYER_ORDER = ("rgb", "gray", "detect_mat", "objmap", "overlay")


def _default_channel(class_name: str, available: list[str]) -> str:
    from phenotypic.gui.builder._image_renderer import _registry_info_for  # type: ignore

    # Mirror render_node_preview: Enhancer->detect_mat, Detector/Refiner->overlay.
    cat = None
    try:
        info = _registry_info_for(class_name)
        cat = getattr(info, "category", None)
    except Exception:  # noqa: BLE001
        cat = None
    if cat == "Enhancer" and "detect_mat" in available:
        return "detect_mat"
    if cat in {"Detector", "Refiner"} and "overlay" in available:
        return "overlay"
    return available[0] if available else "rgb"


def build_preview_payload(
    *, session_id: str, state_data: Any, block_id: str, scope_path: list[str],
    image_path: Optional[str], nrows: Any, ncols: Any, url_prefix: str,
) -> dict:
    """Compute the scope, then build layer options + DZI url for one node."""
    from phenotypic.gui.builder._state import state_from_json

    state = state_from_json(state_data)
    manifest = pc.compute_scope(session_id, state, list(scope_path),
                                image_path, nrows, ncols)
    if manifest.get("error"):
        return {"error": manifest["error"], "options": [], "value": None,
                "dzi_url": None, "title": "Preview", "caption": manifest["error"]}

    node = manifest["nodes"].get(block_id)
    if node is None:
        return {"error": "node not previewable", "options": [], "value": None,
                "dzi_url": None, "title": "Preview", "caption": "Node not previewable"}

    available = [c for c in ("rgb", "gray", "detect_mat") if c in node["layers"]]
    if node.get("num_objects", 0) > 0 and "objmap" in node["layers"]:
        available += ["objmap", "overlay"]
    available = [c for c in _LAYER_ORDER if c in available]

    # class_name for the default-channel rule.
    block = _find_block(state.root, block_id)
    default = _default_channel(getattr(block, "class_name", ""), available)
    shash = pc.scope_hash(list(scope_path))
    h, w = node.get("shape", [0, 0])
    options = [{"label": _LAYER_LABELS[c], "value": c} for c in available]
    return {
        "error": None,
        "options": options,
        "value": default,
        "dzi_url": preview_dzi_url(url_prefix, session_id, shash, block_id, default),
        "title": getattr(block, "label", None) or getattr(block, "class_name", "Preview"),
        "caption": f"{w}×{h} · {default}",
    }


def _find_block(scope, block_id):
    for b in scope.blocks:
        if b.block_id == block_id:
            return b
        if b.nested is not None:
            found = _find_block(b.nested, block_id)
            if found is not None:
                return found
    return None
```

> Note: if `_registry_info_for` is not importable as a private symbol, replace `_default_channel`'s lookup with `from phenotypic.gui.builder._state import stage_of` and branch on `stage_of(class_name)` returning `"ops"` for all — then default to `available[0]`. Verify the symbol exists before relying on it; the test only asserts `value in layer_values`, so any available channel is acceptable.

- [ ] **Step 3b: Register the callbacks**

Inside `register_callbacks(app)` in `_callbacks.py`, near the picker clientside block (~6469), add the server callbacks and 2 clientside callbacks. Import at the top of `_callbacks.py` (with the other builder imports):

```python
from flask import current_app
from phenotypic.gui._config import CFG_URL_PREFIX
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._preview_callbacks import build_preview_payload
from phenotypic.gui.builder._preview_tiles import preview_dzi_url
```

> **Fix (plan review #3 & #4):**
> - **scope_path decoding:** the node-action id stores `scope_path` as a `"/"`-joined string (via `_linear_scope_id_value`, `_ids.py:322`), so `list(scope_raw)` would split a nested path into single characters. Decode with the existing helpers `_decode_linear_scope_path` (`_callbacks.py:901`) and `_decode_linear_optional` (`_callbacks.py:909`) — both already defined in this module.
> - **url prefix:** read `current_app.config.get(CFG_URL_PREFIX, "/")` (the Flask server-config key set in `_app.py:142`, matching `_dzi_url`), NOT `app.config` (Dash's config).

Server callbacks:

```python
    @app.callback(
        Output(ids.MODAL_NODE_PREVIEW, "is_open", allow_duplicate=True),
        Output(ids.STORE_PREVIEW_TARGET, "data"),
        Input({"type": ids.LINEAR_NODE_ACTION, "surface": ALL, "action": "preview",
               "scope_path": ALL, "block_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def open_node_preview(_clicks):
        if not isinstance(ctx.triggered_id, dict) or not ctx.triggered \
                or not ctx.triggered[0].get("value"):
            return no_update, no_update
        tid = ctx.triggered_id
        scope_path = _decode_linear_scope_path(tid.get("scope_path"))
        block_id = _decode_linear_optional(tid.get("block_id"))
        return True, {"block_id": block_id, "scope_path": scope_path}

    @app.callback(
        Output(ids.PREVIEW_LAYER_RADIO, "options"),
        Output(ids.PREVIEW_LAYER_RADIO, "value"),
        Output(ids.MODAL_NODE_PREVIEW_TITLE, "children"),
        Output(ids.PREVIEW_CAPTION, "children"),
        Output(ids.PREVIEW_DZI_URL_STORE, "data"),
        Input(ids.STORE_PREVIEW_TARGET, "data"),
        State(ids.STORE_SESSION_ID, "data"),
        State(ids.STORE_BUILDER_STATE, "data"),
        State(ids.STORE_IMAGE_PATH, "data"),
        State(ids.INPUT_NROWS, "value"),
        State(ids.INPUT_NCOLS, "value"),
        prevent_initial_call=True,
    )
    def compute_node_preview(target, session_id, state_data, image_path,
                             nrows, ncols):
        if not target or not state_data:
            return no_update, no_update, no_update, no_update, no_update
        if not session_id:
            session_id = uuid.uuid4().hex
        url_prefix = current_app.config.get(CFG_URL_PREFIX, "/")
        try:
            payload = build_preview_payload(
                session_id=session_id, state_data=state_data,
                block_id=target["block_id"], scope_path=target["scope_path"],
                image_path=image_path, nrows=nrows, ncols=ncols,
                url_prefix=url_prefix,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Node preview failed")
            return [], None, "Preview", _format_exception(exc), None
        return (payload["options"], payload["value"], payload["title"],
                payload["caption"], payload["dzi_url"])

    @app.callback(
        Output(ids.PREVIEW_DZI_URL_STORE, "data", allow_duplicate=True),
        Output(ids.PREVIEW_CAPTION, "children", allow_duplicate=True),
        Input(ids.PREVIEW_LAYER_RADIO, "value"),
        State(ids.STORE_PREVIEW_TARGET, "data"),
        State(ids.STORE_SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def switch_preview_layer(channel, target, session_id):
        if not channel or not target or not session_id:
            return no_update, no_update
        scope_path = target["scope_path"]
        shash = pc.scope_hash(scope_path)
        url_prefix = current_app.config.get(CFG_URL_PREFIX, "/")
        url = preview_dzi_url(url_prefix, session_id, shash, target["block_id"], channel)
        # Keep the W×H prefix consistent with compute_node_preview's caption.
        manifest = pc.read_manifest(session_id, scope_path) or {}
        node = manifest.get("nodes", {}).get(target["block_id"], {})
        h, w = node.get("shape", [0, 0])
        return url, f"{w}×{h} · {channel}"
```

Call `pc.scope_hash(...)` directly (defined in Task 3). Use `INPUT_NROWS`/`INPUT_NCOLS` ids — the plan reviewer confirmed these are the exact States `run_preview` uses for grid dims (`_callbacks.py:5539-5543`).

Clientside callbacks (mirror the picker block at 6469-6526):

```python
    app.clientside_callback(
        """
        function(dziUrl) {
            const ns = window.__phenotypicNodePreview;
            if (!ns || !ns.mountViewer) { return window.dash_clientside.no_update; }
            if (!dziUrl) { if (ns.disposeViewer) ns.disposeViewer(); return window.dash_clientside.no_update; }
            requestAnimationFrame(function () { ns.mountViewer("preview-osd", dziUrl); });
            return Date.now();
        }
        """,
        Output(ids.PREVIEW_OSD_MOUNT_TRIGGER, "data", allow_duplicate=True),
        Input(ids.PREVIEW_DZI_URL_STORE, "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(isOpen) {
            const ns = window.__phenotypicNodePreview;
            if (!ns || !ns.disposeViewer) { return window.dash_clientside.no_update; }
            if (!isOpen) { ns.disposeViewer(); }
            return Date.now();
        }
        """,
        Output(ids.PREVIEW_OSD_MOUNT_TRIGGER, "data", allow_duplicate=True),
        Input(ids.MODAL_NODE_PREVIEW, "is_open"),
        prevent_initial_call=True,
    )
```

Wire the Close button: add a callback `Output(MODAL_NODE_PREVIEW, "is_open", allow_duplicate=True)` from `Input("btn-preview-close", "n_clicks")` returning `False` (prevent_initial_call=True).

> **Fix (plan review, concern A):** Do NOT add a fan-in guard. The existing `LINEAR_NODE_ACTION` dispatcher (`_callbacks.py:3783-3849`) already has a terminal `else: return _NOOP_FAN_IN` for unknown actions, so `action == "preview"` no-ops safely with the correct arity. Two Dash callbacks may share an overlapping pattern-matching `Input` (the fan-in's `action: ALL` and the new `action: "preview"`) without a duplicate-callback error. The dedicated `open_node_preview` callback owns the preview action; no change to the fan-in is needed.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/gui/builder/test_preview_callbacks.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Smoke-test app construction + lint**

Run: `uv run python -c "from phenotypic.gui.builder._app import create_app; create_app()" && uv run ruff check --fix src/phenotypic/gui/builder/_preview_callbacks.py src/phenotypic/gui/builder/_callbacks.py && uv run mypy src/phenotypic/gui/builder/_preview_callbacks.py`
Expected: app builds with no duplicate-output/registration errors; lint clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/builder/_preview_callbacks.py src/phenotypic/gui/builder/_callbacks.py tests/gui/builder/test_preview_callbacks.py
git commit -m "feat(gui): node-preview open/compute/toggle/clientside callbacks"
```

---

## Task 9: Nested integration test, FEATURES.md, full regression

**Files:**
- Test: `tests/gui/builder/test_preview_nested_integration.py`
- Modify: `src/phenotypic/gui/FEATURES.md`

**Interfaces:**
- Consumes: everything above (end-to-end through the Flask test client + `compute_scope`).

- [ ] **Step 1: Write the nested faithfulness + coexistence test**

Create `tests/gui/builder/test_preview_nested_integration.py`:

```python
"""Nested previews: faithful threaded input + scope coexistence + route serves."""
import numpy as np
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._app import create_app
from phenotypic.gui.builder._state import (
    BlockNode, Edge, _DagBuilderState, _DagBuilderScope, _new_block_id, state_to_json,
)


def _img_edge(src, tgt):
    return Edge(edge_id=_new_block_id(), source_block_id=src, source_port="out",
                target_block_id=tgt, target_port="in", kind="image")


def _nested_state():
    inner = _DagBuilderScope()
    inner_in = inner.blocks[0]
    inner_op = BlockNode(block_id=_new_block_id(), class_name="OtsuDetector", params={})
    inner.blocks.append(inner_op)
    inner.edges.append(_img_edge(inner_in.block_id, inner_op.block_id))
    container = BlockNode(block_id=_new_block_id(), class_name="ImagePipeline",
                          params={}, nested=inner)
    parent_blur = BlockNode(block_id=_new_block_id(), class_name="GaussianBlur",
                            params={"sigma": 5})
    scope = _DagBuilderScope()
    inp = scope.blocks[0]
    scope.blocks.extend([parent_blur, container])
    scope.edges.append(_img_edge(inp.block_id, parent_blur.block_id))
    scope.edges.append(_img_edge(parent_blur.block_id, container.block_id))
    return _DagBuilderState(root=scope), container, inner_op


def test_nested_scopes_coexist_and_serve(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    # create_app runs init_preview_cache() which wipes the cache root, so build
    # the app FIRST, then compute_scope writes into the (surviving) cache.
    app = create_app(image_root=tmp_path)
    state, container, inner_op = _nested_state()
    sid = "nestedsess0001"
    scope_path = [container.block_id]

    manifest = pc.compute_scope(sid, state, scope_path, None, None, None)
    assert manifest["error"] is None
    assert inner_op.block_id in manifest["nodes"]
    # parent + inner dirs coexist
    assert pc.read_manifest(sid, []) is not None
    assert pc.read_manifest(sid, scope_path) is not None

    # the inner detector's objmap tile serves through the blueprint
    client = app.server.test_client()
    shash = pc.scope_hash(scope_path)
    resp = client.get(
        f"/preview-tiles/{sid}/{shash}/{inner_op.block_id}/detect_mat.dzi"
    )
    assert resp.status_code == 200
    assert "deepzoom" in resp.get_data(as_text=True).lower()


def test_parent_edit_invalidates_inner(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    state, container, inner_op = _nested_state()
    sid = "nestedsess0002"
    scope_path = [container.block_id]
    fp1 = pc.compute_scope(sid, state, scope_path, None, None, None)["fingerprint"]

    # edit the PARENT enhancer; inner fingerprint must change (chaining)
    for b in state.root.blocks:
        if b.class_name == "GaussianBlur":
            b.params["sigma"] = 1
    fp2 = pc.compute_scope(sid, state, scope_path, None, None, None)["fingerprint"]
    assert fp1 != fp2
```

- [ ] **Step 2: Run the test to verify it fails, then passes**

Run: `uv run pytest tests/gui/builder/test_preview_nested_integration.py -v`
Expected: PASS if Tasks 1-5 are correct (this test is the integration gate; if it fails, fix the implicated task, do not edit the test to pass).

- [ ] **Step 3: Update FEATURES.md**

Add a `## Node preview` section to `src/phenotypic/gui/FEATURES.md` (after the Builder point picker section), using the 6-column format `| Feature | Element | Expected behaviour | Status | Test layer | Test ref |`:

```
## Node preview

| Feature              | Element                                              | Expected behaviour                                                                                  | Status     | Test layer  | Test ref                                                                                  |
|----------------------|------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------|-------------|-------------------------------------------------------------------------------------------|
| Preview button       | `linear-node-action` `action="preview"` SVG button   | Image-producing node cards show a Material image-icon button; measurement nodes do not               | ✅ shipping | integration | tests/gui/builder/test_preview_button.py::test_preview_button_has_preview_action_id        |
| Preview modal        | `modal-node-preview`                                 | Blocking (`backdrop=static`) OSD viewer + layer radio + caption mount in the layout                  | ✅ shipping | integration | tests/gui/builder/test_node_preview_modal.py::test_modal_has_blocking_props_and_children   |
| Preview tile route   | `/preview-tiles/<sid>/<scope>/<block>/<channel>.dzi` | Stages an HDF layer to DZI; 200 valid manifest; rejects bad channel/scope/sid                        | ✅ shipping | integration | tests/gui/builder/test_preview_tile_blueprint.py::test_preview_dzi_served                  |
| Preview disk cache   | `_preview_cache.compute_scope`                       | Per-scope full-res HDF cache; chained fingerprint; nested scopes thread parent output + coexist      | ✅ shipping | integration | tests/gui/builder/test_preview_nested_integration.py::test_nested_scopes_coexist_and_serve |
| Preview JS surface   | `assets/preview.js`                                  | Exposes `mountViewer`/`disposeViewer` under `window.__phenotypicNodePreview`                          | ✅ shipping | integration | tests/gui/builder/test_node_preview_modal.py::test_preview_js_asset_exists                 |
```

- [ ] **Step 4: Full builder regression + gates**

Run:
```bash
uv run pytest tests/gui/builder -q
uv run pytest tests/unit/core/test_full_layers_intermediates.py tests/unit/core/test_load_layer_hdf5.py tests/unit/core/test_delta_intermediates.py tests/unit/core/test_image_hdf_roundtrip.py -q
uv run ruff check src/phenotypic/gui/builder src/phenotypic/_core
uv run python scripts/check_workflows_md.py
```
Expected: all green; `check_workflows_md.py` passes (no new WORKFLOWS row needed). If the `features-md-gate` runs only in CI, confirm `FEATURES.md` `Test ref`s resolve by running each referenced test id.

- [ ] **Step 5: Capture GUI screenshots (chrome changed)**

Run: `uv run python scripts/capture_gui_tutorial_screenshots.py`
Then `git add` the regenerated PNGs (commit them ALL — do not cherry-pick the collateral churn, per CLAUDE.md).

- [ ] **Step 6: Commit**

```bash
git add tests/gui/builder/test_preview_nested_integration.py src/phenotypic/gui/FEATURES.md docs/source/tutorials scripts
git commit -m "test(gui): nested preview integration + FEATURES.md rows"
```

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** Task 1 = `full_layers` core flag (spec §1); Task 2 = `load_layer_hdf5` (spec §1); Tasks 3-4 = cache + chained fingerprint + threaded input (spec §2, §8); Task 5 = staging + tile route (spec §3); Task 6 = SVG button (spec §4); Task 7 = modal + JS (spec §5); Task 8 = callbacks (spec §6); Task 9 = nested integration + FEATURES.md (spec §7, §8, Testing).
- **GridImage:** verified the grid handler round-trips `/grid/` through `save2hdf5`/`load_hdf5`; `_load_image_auto` (Task 4) selects the class by stored `phenotypic_class`, so no sidecar is needed.
- **Symbols confirmed by plan review:** `_registry_info_for` (`_image_renderer.py:382`), `INPUT_NROWS`/`INPUT_NCOLS` (`_ids.py:849,852`, the exact States `run_preview` uses), `_decode_linear_scope_path`/`_decode_linear_optional` (`_callbacks.py:901,909`), `_NOOP_FAN_IN` terminal-else in the dispatcher, and `CFG_URL_PREFIX` (`phenotypic.gui._config`) all exist as used.
- **Fan-in guard:** NOT needed — the dispatcher already has a terminal `else: return _NOOP_FAN_IN` for unknown actions (plan review concern A). The dedicated `open_node_preview` callback owns the preview action; overlapping pattern Inputs across two callbacks are legal in Dash.
- **Cache wipe ordering:** `create_app` runs `init_preview_cache()` (wipes the cache root), so route/integration tests must construct the app BEFORE seeding/`compute_scope` (Tasks 5 & 9, fixed).
- **FEATURES.md gate:** the `features-md-gate` is PR-level (CLAUDE.md), so the per-task local commits in Tasks 5-8 that touch `gui/` without editing FEATURES.md are fine; the PR is satisfied by Task 9's rows. If a local pre-commit hook ever blocks a gui-touching commit for FEATURES.md, add a `🔭 planned` placeholder row for that feature then (planned rows skip the Test-ref resolve check).
- **No new WORKFLOWS.md row** (single button + modal) — only FEATURES.md.
