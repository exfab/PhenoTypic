# Phase 4 — GUI read paths: pyramid tiles and the staleness traps

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §4.2, §4.4.

**Depends on:** Phase 2.
**Runs in parallel with:** Phases 3 and 5.

This is where the pyramid pays for itself: today the GUI decodes an **entire** layer to
render one whole-plate tile. It is also where the change is most likely to fail silently —
a store directory's `st_mtime_ns` does **not** change when a nested chunk is rewritten, so
every mtime-based staleness check becomes a stale-tile bug rather than an error.

---

### Task 4.1: Store discovery in `OutputRoot`

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md` — **required** (ledger **GEN-36**).
  `features-md-gate` fails any PR that touches `src/phenotypic/gui/` without also diffing
  `FEATURES.md` (`.github/workflows/gui-checks.yml:92-106`); it is a diff gate, not a
  judgement about whether the chrome changed. Record the pyramid-level tile read against the
  existing viewer rows.
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py`
  (line 53 import, `hdf_path` line 494, `has_*` line 630, `rglob("*.h5")` line 888,
  the consistency-report path build at lines 1146–1152)
- Test: `tests/unit/gui/results_viewer/test_output_root_stores.py` (create)
- **Test: four existing files this task breaks** (README test inventory, Phase 4 row). None
  of the twelve files that row assigns to this phase appeared in any Task 4.x `Files:` list
  before the missing-owner review of 2026-08-19; these four are the ones whose subject is
  `OutputRoot` discovery:
  - `tests/gui/results_viewer/test_output_root.py` — `:213` asserts
    `out.hdf_path("d1", "a") is None`; `:202`'s docstring names `hdf_path` as a
    `results/`-backed capability. Both become `store_path`.
  - `tests/unit/gui/results_viewer/test_output_root.py` — `:49` and `:62` call `hdf_path`;
    `:56` seeds `results/plate1/hdf/img001.h5`. Seed a store directory and assert
    `store_path` instead.
  - `tests/gui/results_viewer/test_output_discovery_contracts.py` — `:64` and `:404` write
    `results/<ds>/hdf/a.h5` as the per-image artifact the discovery and fingerprint contracts
    read. These feed `_processing_snapshot_paths`, so they are what proves the D5 fix: after
    the port they must seed a store and the fingerprint must still change when a chunk is
    rewritten.
  - `tests/gui/results_viewer/test_mutation_guard.py` — `:106` and `:254` seed and then mutate
    the same `hdf/a.h5`. `:254`'s "source changed" mutation must become a store republish, or
    the guard tests a file the viewer no longer reads.
- Read (do not edit): `tests/_output_layout.py` — supplies `write_master` /
  `write_complete_manifest`, the repo's seeding helpers for a discoverable output root

**Interfaces:**
- Consumes: `BundleLayout.store_path` (Task 2.1), `BundleLayout.detect`.
- Produces: `OutputRoot.store_path(dataset, stem) -> Path | None`, replacing `hdf_path`.

**Constraints specific to this task:**

- `store_path` uses `is_dir()`, not `is_file()`.

> **REFUTED IN EXECUTION (2026-08-20).** The two paragraphs below name the wrong
> function. `_processing_snapshot_paths` has **no production caller** —
> `grep -rn "_processing_snapshot_paths" src/ tests/` returns only its own definition,
> and `_cancellable_paths_fingerprint` is reached only from
> `_consumed_state_fingerprint`, which uses a *different* helper
> (`_consumed_state_snapshot_paths`). `source_fingerprint` is
> `ProcessingInventory.fingerprint`, built by `_scan_processing_inventory`
> (`_processing_inventory.py:207-215`), which does an unbounded
> `results_root.rglob("*")` and **already descends into every store** — confirmed by
> observing `results/ds/zarr/a.ome.zarr/gray/0` as an inventory entry. The FLOW-11 claim
> that `_processing_inventory.py` consumes `_processing_snapshot_paths` is wrong for the
> same reason; that module builds its own candidate set.
>
> Consequences: the D5 *property* is real and worth guarding, but its live site is the
> inventory scan, which no Phase 4 task owns. Measured cost: a `save2zarr` of the 600x800
> synth plate produces **38 entries per store** (24 files, 14 dirs) against **1** for an
> `.h5`, so the exhaustive scan goes from ~10k to ~400k stat calls at 10k images — the
> plan's own figure, produced by the function it did not name. Bounding
> `_scan_processing_inventory` to `results/<ds>/zarr/*.ome.zarr/zarr.json` is
> semantically equivalent for any PhenoTypic-written store (nothing writes into a
> promoted store; the promote rewrites the root last) and ~40x cheaper, but it changes
> user-visible staleness detection and was escalated rather than taken.
>
> `_processing_snapshot_paths` was ported anyway — the exit criterion forbids `.h5`
> anywhere under `src/phenotypic/gui/` — and its test docstring records that it is dead.
> It is otherwise untouched; **deleting it belongs to Phase 6's removal list.**
>
> **RESOLVED (user ruling, 2026-08-20).** `_scan_processing_inventory`'s results walk is
> now bounded: `_walk_results_without_descending_into_stores` records each store as
> exactly two entries — the store directory and its root `zarr.json` — and never enters
> it, while the rest of `results/` (dataset dirs, `measurements/*.parquet`) keeps its
> exhaustive walk. The overlays walk is unchanged, and `ProcessingInventoryAssurance`
> keeps its two values: `"exhaustive"` simply stops descending into stores, and no knob
> was added.
>
> What it detects: **every** write PhenoTypic makes, because the commit protocol writes
> the root last and promotes by rename, and nothing opens a promoted store for writing.
> What it deliberately does not: out-of-contract external modification — a hand-edited
> chunk, or a store rsynced mid-flight. This matches Task 3.8, where per-image completion
> markers already fingerprint a store by its root alone.
>
> Measured on a 1200x1600 store (38 entries): the inventory for a 3-image run falls from
> ~122 entries to **12**, a 10x reduction, and the factor grows with plate size — a
> 4000x3000 store holds 58 entries against 1 for the `.h5` it replaced.
>
> The guard is `test_discovery_never_lists_a_directory_inside_a_store`, which counts
> `os.scandir` rather than inspecting the entry list. That distinction is load-bearing:
> a recursive walk that filters its results back down to the store roots produces an
> **identical** inventory while doing all the work the bound exists to avoid, and passes
> every result-set assertion. It was run as a mutation and only the scandir count caught
> it.

- **Line 886–889 is a correctness problem, not just a cost problem.**
  `_processing_snapshot_paths` does
  `paths.extend(path for path in layout.results_dir.rglob("*.h5") if path.is_file())`, and
  those paths feed `_cancellable_paths_fingerprint`, whose directory branch (`:832-834`)
  emits **one sentinel byte and does not recurse**. If the port yields store *directories*,
  `snapshot.processing_fingerprint` — and therefore `OutputRoot.source_fingerprint`
  (`:512`) — **stops changing when per-image results change**, so a viewer open across a run
  never notices new or rewritten images. The port must enumerate each store's
  **`zarr.json`**, which also happens to solve the cost problem (a naive
  `rglob("*.ome.zarr")` recurses into every store — ~400k stat calls at 10k images).
  OPEN-QUESTIONS **D5**.

  **`_processing_snapshot_paths` has a SECOND consumer this task must also satisfy**
  (ledger **FLOW-11**): `_processing_inventory.py` records `st_size` / `st_mtime_ns` /
  `st_ctime_ns` per path (`:249-255`, `:371-379`) and `_inventory_is_current` compares all
  three (`:426-431`). Enumerating each store's `zarr.json` happens to be right for it too —
  but its directory special-case (`:389-395`) means a store-*directory* port would silently
  freeze the inventory under `read_only_bounded`, the same D5 failure shape in a second
  place. Assert the inventory goes non-current after a store republish.

- **Line 1146–1152 is a staleness fingerprint, not a report label.** An earlier draft of
  this task described it as "an `("hdf", hdf_path)` pair for the output-consistency report…
  update whatever label the report renders". That is wrong. `_image_source_token`
  (`:1138-1178`) hashes, per source path:

  ```python
  f"{stat.st_dev}\0{stat.st_ino}\0{stat.st_size}\0{stat.st_mtime_ns}\0{stat.st_ctime_ns}\n"
  ```

  **None of those five fields moves** when a chunk inside a store is rewritten. The token
  drives `bound_image_source_token` (`:649`) and `_capture_image_source_tokens` (`:405`,
  `:1093`) — i.e. whether the viewer's binding to an image's pixel source is still valid.
  Ported as a relabel it goes silently blind. It must key on **`store / "zarr.json"`**.
  OPEN-QUESTIONS **D4**.

- **The rule underneath D4 and D5:** anywhere the old code took a fingerprint or a stat of
  the `.h5`, the store equivalent is the **root `zarr.json`** — never the store directory.
  `paths_fingerprint` "handles directories" only in the sense that it does not raise; it
  ignores their contents. The spec's §4.2 table is misleading on exactly this point and
  should be corrected.

- [ ] **Step 1: Write the failing test**

> **Corrected (wrong-symbol sweep).** Two defects in an earlier draft of this block:
>
> 1. **`_seed` wrote `deliverables/master_measurements.parquet` as `b""`.**
>    `OutputRoot.discover` reads it with `pl.read_parquet` (`_output_root.py:340`), which
>    raises `ComputeError: parquet: File out of specification` on empty bytes — so **every**
>    test here errored before reaching `store_path`. The repo's own helper for this is
>    `tests._output_layout.write_master` (used, with `write_complete_manifest`, by
>    `tests/gui/results_viewer/test_output_root.py`).
> 2. **`OutputRoot.discover` takes a required keyword-only `cache_root`** (`:334`) whose
>    directory must not sit inside the selected output. A bare `discover(tmp_path)` raises
>    `TypeError`. Route every call through the `_discover` helper below, mirroring
>    `tests/gui/results_viewer/test_output_root.py:34-39`.

```python
"""OutputRoot discovers store directories without walking into them."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import zarr_store_path

from tests._output_layout import write_complete_manifest, write_master


def _discover(root: Path) -> OutputRoot:
    """Discover with a test-owned cache OUTSIDE the selected output."""
    source = Path(root).resolve()
    return OutputRoot.discover(
        source, cache_root=source.parent / ".test-phenotypic-viewer-cache"
    )


def _seed(root: Path, stems: list[str]) -> None:
    """Seed a minimal store-backed output: real master, then one store per stem."""
    write_master(
        root,
        pl.DataFrame(
            {
                "Metadata_Dataset": ["ds"] * len(stems),
                str(IMAGE.IMAGE_NAME): list(stems),
                "Size_Area": [100.0] * len(stems),
            }
        ),
    )
    write_complete_manifest(root, total_images=len(stems))
    (root / "results" / "ds" / "measurements").mkdir(parents=True, exist_ok=True)
    for stem in stems:
        store = zarr_store_path(root, "ds", stem)
        (store / "gray" / "0").mkdir(parents=True)
        (store / "zarr.json").write_text("{}", encoding="utf-8")


def test_store_path_resolves_a_directory(tmp_path: Path) -> None:
    _seed(tmp_path, ["a"])
    root = _discover(tmp_path)
    assert root.store_path("ds", "a") == zarr_store_path(tmp_path, "ds", "a")


def test_store_path_is_none_when_absent(tmp_path: Path) -> None:
    _seed(tmp_path, ["a"])
    assert _discover(tmp_path).store_path("ds", "missing") is None


def test_discovery_does_not_walk_into_stores(tmp_path: Path, monkeypatch) -> None:
    """A recursive scan costs 400k stat calls at 10k images."""
    _seed(tmp_path, ["a", "b"])
    visited: list[str] = []
    real_iterdir = Path.iterdir

    def _counting(self):
        visited.append(str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _counting)
    _discover(tmp_path)
    assert not any("/gray/" in seen for seen in visited), visited


def test_discovery_finds_every_store(tmp_path: Path) -> None:
    _seed(tmp_path, ["a", "b", "c"])
    root = _discover(tmp_path)
    assert all(root.store_path("ds", stem) is not None for stem in "abc")
```

- [ ] **Step 2: Run it to verify it fails.** Expected: `AttributeError: … 'store_path'`.

Add two tests for the fingerprint sites:

> **Corrected (wrong-symbol sweep).** An earlier draft called
> `_image_source_token([store / "zarr.json"])`. The real signature takes no path list:
> `_image_source_token(layout, dataset, stem, *, has_overlay) -> str`
> (`_output_root.py:1138-1142`) — it *derives* the source paths internally, which is exactly
> why the port has to change them there. Drive it with a real
> `BundleLayout.detect(...)` (the only public constructor, `_io_constants.py:1999`), before
> and after the port.

```python
def test_processing_fingerprint_changes_when_a_store_changes(tmp_path: Path) -> None:
    """Enumerating directories would freeze this permanently (D5)."""
    _seed(tmp_path, ["a"])
    before = _discover(tmp_path).source_fingerprint
    (zarr_store_path(tmp_path, "ds", "a") / "zarr.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    assert _discover(tmp_path).source_fingerprint != before


def test_image_source_token_changes_when_a_store_changes(tmp_path: Path) -> None:
    """It is a staleness fingerprint, not a report label (D4)."""
    from phenotypic.gui.results_viewer._output_root import _image_source_token
    from phenotypic.sdk_ import BundleLayout

    _seed(tmp_path, ["a"])
    layout = BundleLayout.detect(tmp_path)
    before = _image_source_token(layout, "ds", "a", has_overlay=False)
    (zarr_store_path(tmp_path, "ds", "a") / "zarr.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    assert _image_source_token(layout, "ds", "a", has_overlay=False) != before


def test_the_bound_token_tracks_the_store_too(tmp_path: Path) -> None:
    """The public surface the token actually reaches the viewer through."""
    _seed(tmp_path, ["a"])
    before = _discover(tmp_path).bound_image_source_token("ds", "a")
    (zarr_store_path(tmp_path, "ds", "a") / "zarr.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    assert _discover(tmp_path).bound_image_source_token("ds", "a") != before
```

- [ ] **Step 3: Port the four sites.** Replace `hdf_path` with `store_path` (delegating to
`BundleLayout.store_path`); replace `rglob("*.h5")` with a bounded
`(results / dataset / DIR_ZARR).glob(f"*{STORE_SUFFIX}")` per dataset directory that yields
**each store's `zarr.json`**, not the store directory; and point `_image_source_token`'s
source path at `store / "zarr.json"` likewise.

- [ ] **Step 4: Run** `uv run pytest tests/unit/gui/results_viewer tests/gui/results_viewer -q`. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_output_root.py tests/unit/gui/results_viewer
git commit -m "refactor(gui): discover store directories without walking into them

rglob('*.h5') becomes a bounded per-dataset glob for *.ome.zarr. A naive
rglob port recurses into every store, which is ~400k stat calls at 10k
images -- pathological on exactly the runs the viewer is for. The
output-consistency report's hdf path pair is ported too; it is a real call
site the spec's affected-module table did not list."
```

---

### Task 4.2: Pyramid-level tile reads

**Files:**
- Modify: `src/phenotypic/gui/_shared/tiles.py`
  (`_load_hdf_layer_rgb` line 291, `_hdf_layer_array_to_rgb` line 333, `crop_hdf_rgb`
  line 349, `_crop_hdf_layer_window` line 396, the caller at lines 509–518, `__all__` at
  line 1155)
- Test: `tests/unit/gui/shared/test_tiles_zarr.py` (create)
- **Test: three existing files this task breaks** (README test inventory, Phase 4 row):
  - `tests/gui/_shared/test_tiles.py` — the direct suite for the functions being renamed.
    `:135`, `:165`, `:201` build inputs with `Image.save2hdf5(str(h5))` → `save2zarr`; the two
    stub `OutputRoot`s at `:236` and `:285` define `hdf_path(self, ds, stem)` and must define
    `store_path`, returning a directory (`:246` / `:294` currently
    `write_bytes(b"")` an empty `x.h5`, which must become a store built by `save2zarr` —
    an empty file is not a substitutable stand-in for a store).
  - `tests/gui/results_viewer/colony_view/test_cropper.py` — `:107` writes `plate.h5` as the
    cropper's input; port to a store and the `crop_store_rgb` signature.
  - `tests/gui/results_viewer/colony_view/test_grid.py` — `:261` writes a zero-byte
    `hdf_dir/<stem>.h5` purely as an existence marker for axis selection. Port to
    `zarr_store_path(...).mkdir()` plus a real store, for the same reason as above.

**Interfaces:**
- Produces:
  ```python
  def _load_zarr_layer_rgb(store: str, content_token: str, layer: LayerName, target_px: int) -> Image
  def crop_store_rgb(store_path, layer, window, content_token, ...) -> Image
  def select_pyramid_level(store_path: Path, layer: str, target_px: int) -> int
  ```

**Constraints specific to this task:**
- `select_pyramid_level` returns the **smallest level whose longest edge still covers**
  `target_px` — i.e. the coarsest level that does not under-sample the request. Reading a
  level finer than needed is the current behaviour and wastes the whole point of the change;
  reading one coarser produces a visibly soft tile.
- Level metadata comes from `phenotypic.pyramid.levels` plus the per-level array shapes.
  **Never infer the level count from directory listing** — a `.part` sweep or a partially
  written store would give a wrong answer.
- **Read amplification is real, and the docstring must say so.** Slicing a 64×64 colony
  crop from a sharded level 0 costs a shard-index read plus one full `1024×1024` inner
  chunk. It is cheap, but it is not "the same as h5py", which an earlier draft implied.
- **Resize the LRU, or key it on the resolved level** (ledger **FLOW-10**).
  `_load_zarr_layer_rgb` gains a fourth key element (`target_px`) while the cache stays at
  `_HDF_LAYER_CACHE_SIZE = 4` (`gui/_shared/tiles.py:287`). Across several distinct target
  sizes × four layers it thrashes on exactly the path the pyramid was introduced to
  accelerate, so the measured win may not appear at all. Either size it to
  level-count × layer-count, or key on the **resolved level** so distinct `target_px` values
  that select the same level share one entry. The second is better: it bounds the key space
  by the data rather than by request variety.

- [ ] **Step 1: Write the failing test**

```python
"""Tile reads select a pyramid level instead of decoding the whole layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.gui._shared.tiles import select_pyramid_level
from phenotypic.data import load_synth_yeast_plate


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")


def test_full_size_request_selects_level_zero(store: Path) -> None:
    level0 = Image.load_layer_zarr(store, "gray", level=0)
    assert select_pyramid_level(store, "gray", max(level0.shape)) == 0


def test_small_request_selects_a_coarse_level(store: Path) -> None:
    assert select_pyramid_level(store, "gray", 64) > 0


def test_selected_level_still_covers_the_request(store: Path) -> None:
    """Coarser than the request would render visibly soft."""
    for target in (64, 128, 256, 512, 1024):
        level = select_pyramid_level(store, "gray", target)
        shape = Image.load_layer_zarr(store, "gray", level=level).shape
        assert max(shape) >= target or level == 0


def test_selection_never_reads_finer_than_necessary(store: Path) -> None:
    level = select_pyramid_level(store, "gray", 256)
    if level > 0:
        finer = Image.load_layer_zarr(store, "gray", level=level - 1).shape
        assert max(finer) > 256


def test_level_count_comes_from_metadata_not_directory_listing(
    store: Path, monkeypatch
) -> None:
    """A .part sweep or a partial write would make a listing lie."""
    import shutil

    shutil.rmtree(store / "gray" / "1")
    with pytest.raises(Exception):
        select_pyramid_level(store, "gray", 64)


def test_single_level_store_always_selects_zero(tmp_path: Path) -> None:
    """The levels=1 path: builder node previews."""
    flat = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "f.ome.zarr", layers=("gray",)
    )
    assert select_pyramid_level(flat, "gray", 32) == 0


def test_crop_matches_the_full_resolution_slice(store: Path) -> None:
    from phenotypic.gui._shared.tiles import crop_store_rgb

    full = Image.load_layer_zarr(store, "gray", level=0)
    crop = np.asarray(crop_store_rgb(store, "gray", (10, 10, 74, 74), "tok"))
    np.testing.assert_array_equal(crop[..., 0], full[10:74, 10:74])
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.** Rename the four functions, add `select_pyramid_level` reading
`phenotypic.pyramid.levels` and each level's shape, and update `__all__` at line 1155.

- [ ] **Step 4: Run** `uv run pytest tests/unit/gui tests/gui/_shared -q`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/tiles.py tests/unit/gui
git commit -m "perf(gui): read a pyramid level instead of decoding the whole layer

select_pyramid_level picks the coarsest level that still covers the request,
from phenotypic.pyramid.levels rather than a directory listing (a .part
sweep or a partial write would make a listing lie). Read amplification is
documented honestly: a 64x64 crop from a sharded level 0 costs a shard-index
read plus one full 1024x1024 inner chunk -- cheap, but not free."
```

---

### Task 4.3: The staleness traps

> **Fixtures this task must create** (ledger **GEN-8**) — Phase 3 Task 3.3 specifies its
> four in a table; these were used but never defined.
>
> | Fixture | File | Must expose |
> |---|---|---|
> | `live_viewer` | `tests/e2e/gui/conftest.py` | `get_tile(ds, stem, layer)`, `republish_with_objmap(ds, stem, value)` (goes through `save2zarr`) — against a running tile route |
> | `builder_preview` | `tests/unit/gui/builder/conftest.py` | `png_bytes(block_id, channel)`, `rewrite_node_store(block_id)` |
>
> `live_viewer` is the larger of the two; model it on the existing real-loaded-viewer pattern
> rather than inventing a new harness.

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_tile_routes.py`
  (`_ensure_hdf_layer_source_png` lines 462–477: `file_fingerprint` at 473,
  `stat().st_mtime_ns` compare at 466/469, `os.utime` at 477)
- Modify: `src/phenotypic/gui/builder/_preview_tiles.py` (line 76 mtime compare)
- Modify: `src/phenotypic/gui/_shared/tiles.py` (line 518 mtime-keyed crop path)
- Test: `tests/unit/gui/results_viewer/test_tile_cache_invalidation.py` (create)
- **Test: two existing files this task breaks** (README test inventory, Phase 4 row) — both
  are staleness suites, which is why they land here rather than on Task 4.2 even though
  Task 4.2 renames the functions they call:
  - `tests/gui/results_viewer/test_tile_routes.py` — the module docstring (`:14`) and layout
    comment (`:137`) describe a `results/d1/hdf/img001.h5` fixture; `:171`
    `img.save2hdf5(...)` builds it; `:329` reads `output_root.hdf_path(dataset, stem)`;
    `:362` and `:476` build further `.h5` inputs, `:476` through `h5py.File(..., "w")`. Port
    the fixture to `save2zarr` and the lookup to `store_path`, and re-key the route's
    freshness assertion on `zarr.json` rather than the artifact's own `st_mtime_ns` — that
    assertion is the one that currently passes for the wrong reason and would keep passing
    against a store while serving a stale tile.
  - `tests/gui/results_viewer/colony_view/test_crop_routes.py` — `:60-62` build the input
    with `h5py.File`; `:79` passes it into the app fixture; `:124`, `:145`, `:173` unpack it;
    and `:147-154` is a **content-change-under-the-same-path** test
    (`replacement.replace(hdf_path)`), which is exactly the shape that goes silently stale on
    a store directory. That test must be ported, not deleted.

**Constraints specific to this task:**

**Three live staleness sites, plus one that only needs a port.** An earlier draft of this
task counted four traps and included `_shared/tiles.py:518` among them; that is wrong, and
it also omitted the two sites that genuinely go content-blind (now in Task 4.1).

| Site | Problem |
|---|---|
| `_tile_routes.py:473` | `file_fingerprint()` opens its argument as a file → `IsADirectoryError` on a store. Must key on `paths_fingerprint([store / "zarr.json"])`. |
| `_tile_routes.py:466,469,477` | `stat().st_mtime_ns` compare and `os.utime` against the store |
| `_preview_tiles.py:76` | same mtime compare |
| `_shared/tiles.py:518` | **not a staleness site.** `crop_hdf_rgb` opens with `del mtime_ns` (`:386`) and its docstring says the parameter is "accepted for caller/API compatibility; crop reads are windowed and not full-layer cached" (`:375-376`). The `os.stat(h5).st_mtime_ns` feeds a discarded parameter, and `os.stat` works fine on a directory. Port it to the store path; do **not** add a staleness fix. OPEN-QUESTIONS **D15**. |

- **A store directory's `st_mtime_ns` does not change when a nested chunk is rewritten**
  (verified). Every staleness check must move to the **root `zarr.json`**, which the
  promote writes last on every publish — so its mtime and contents change exactly when the
  store's contents do.
- **`paths_fingerprint` does not "handle directories" in the sense the spec implies.** It
  emits a single sentinel byte for a directory and does not recurse
  (`_io_constants.py:215-217`), so `paths_fingerprint([store])` is a constant function of
  the path and would freeze the tile cache **permanently**. Always pass
  `[store / "zarr.json"]`.
- `os.utime` at line 477 must be applied against the root `zarr.json`'s mtime, not the
  directory's.
- ⚠️ **The content token must include the root's mtime, not only its bytes.** A Stage-3
  re-promote whose metadata did not change produces a **byte-identical** root `zarr.json`.
  The PNG is regenerated (its mtime is now older) — but `_load_zarr_layer_rgb` is
  `@functools.lru_cache`d on `(path, token, layer)` (`gui/_shared/tiles.py:290-292`), and an
  unchanged token means the "regenerated" PNG is written from the **old decoded array**. So
  the token is
  `paths_fingerprint([root_json]) + str(root_stat.st_mtime_ns)`, which changes on every
  promote whether or not the metadata did. Recorded as OPEN-QUESTIONS **B7/P17**.
- **Nothing writes into a promoted store, so there is no mid-run staleness to reason about.**
  Stage 2 keeps its raw output under `.phenotypic/progress/` (user ruling — only the *final*
  store needs interop), so between Stage 1 and Stage 3 the store holds a zeros objmap and the
  GUI correctly shows nothing. Root-keying then means the cache invalidates on **promotes**,
  which is exactly when the contents actually change.

  This **retires** the D6 reasoning rather than relying on it. D6 argued that the cached tile
  route happens not to notice an in-place write — which left the **uncached crop route**
  (`gui/_shared/tiles.py:349-392`; `del mtime_ns`, *"crop reads are windowed and not
  full-layer cached"*) fully exposed, serving raw pre-`drop_frame_background` labels to the
  colony view for hours on SLURM. That was **FLOW-5**. With no in-place write anywhere,
  neither route can observe intermediate labels, and spec §3.5's claim that the in-store
  write buys "the GUI can render a real objmap mid-run" is withdrawn along with the write.

- [ ] **Step 1: Write the failing test**

```python
"""Cache invalidation across the mtime/fingerprint traps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.sdk_ import file_fingerprint, paths_fingerprint
from phenotypic.data import load_synth_yeast_plate


def test_file_fingerprint_raises_on_a_store_directory(tmp_path: Path) -> None:
    """Pins the exact reason the tile route must switch helpers."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    with pytest.raises(IsADirectoryError):
        file_fingerprint(store)


def test_paths_fingerprint_keys_on_the_root_json(tmp_path: Path) -> None:
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    assert paths_fingerprint([store / "zarr.json"]).startswith("sha256:")


def test_store_directory_mtime_does_not_change_when_a_chunk_is_rewritten(
    tmp_path: Path,
) -> None:
    """The verified fact the whole task exists for.

    Demonstrated by writing a chunk file directly, since nothing in the design
    opens a promoted store for writing any more (the Stage-2 in-place write was
    removed). The fact still governs: a nested chunk rewrite leaves the store
    directory's own mtime untouched, which is why every staleness check keys on
    the root zarr.json rather than on the directory.
    """
    import os

    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    before = os.stat(store).st_mtime_ns
    chunk = next(p for p in (store / "gray" / "0").rglob("*") if p.is_file())
    chunk.write_bytes(chunk.read_bytes())
    assert os.stat(store).st_mtime_ns == before


def test_root_zarr_json_changes_on_every_publish(tmp_path: Path) -> None:
    image = Image(load_synth_yeast_plate())
    store = image.save2zarr(tmp_path / "p.ome.zarr")
    before = paths_fingerprint([store / "zarr.json"])
    image._metadata.public["Metadata_Strain"] = "BY4742"
    image.save2zarr(tmp_path / "p.ome.zarr")
    assert paths_fingerprint([store / "zarr.json"]) != before


def test_served_tile_changes_after_a_promote(live_viewer) -> None:
    """End-to-end: republish under a live cache and assert the tile changes.

    Keyed on a PROMOTE, not an in-place write: the promote is what rewrites the
    root zarr.json, and it is what publishes a store consumers should see.
    """
    first = live_viewer.get_tile("ds", "img", layer="objmap")
    live_viewer.republish_with_objmap("ds", "img", value=7)  # goes through save2zarr
    second = live_viewer.get_tile("ds", "img", layer="objmap")
    assert first != second


def test_nothing_writes_into_a_promoted_store() -> None:
    """There is no in-place store write anywhere in the design (user ruling).

    Stage 2 writes its raw output under .phenotypic/progress/, never into the
    store, so the GUI cannot observe intermediate labels through EITHER the
    cached tile route or the uncached crop route. This pins the absence, so a
    later 'optimization' cannot reintroduce the FLOW-5 exposure.

    Lives in `tests/unit/sdk_/test_ngff_promote.py`, NOT in the GUI test module:
    it asserts a property of `phenotypic.sdk_.ngff_` and needs no fixture, so
    there is nothing tying it to the viewer suite (ledger SIMP-24 -- an earlier
    draft recorded the relocation in the docstring without performing it).
    An earlier draft requested the heavyweight `live_viewer` and never touched
    it, and paired the hasattr check with `"r+" not in inspect.getsource(ngff_)`
    -- a substring grep that any docstring, comment, or regex containing `r+`
    flips, and that `mode="a"` or `mode="w"` walks straight past. Ledger
    SIMP-15 / GEN-32(b). Belongs beside the other ngff_ unit tests, not here.
    """
    from phenotypic.sdk_ import ngff_

    assert not hasattr(ngff_, "write_objmap_in_place")


def test_builder_preview_invalidates_on_store_change(builder_preview, tmp_path) -> None:
    first = builder_preview.png_bytes("block-1", "gray")
    builder_preview.rewrite_node_store("block-1")
    assert builder_preview.png_bytes("block-1", "gray") != first
```

- [ ] **Step 2: Run to verify failure.** `test_store_directory_mtime_does_not_change…` and
`test_file_fingerprint_raises…` should PASS immediately — they pin facts, not new
behaviour. The two cache-invalidation tests should FAIL.

- [ ] **Step 3: Port all four sites.** In `_ensure_hdf_layer_source_png` → rename to
`_ensure_store_layer_source_png` and key on `store / "zarr.json"`:

```python
def _ensure_store_layer_source_png(
    store: Path, layer: LayerName, source_png: Path, target_px: int
) -> None:
    """Materialise a source PNG for one store layer, invalidating on the root.

    A store directory's ``st_mtime_ns`` does **not** change when a nested chunk
    is rewritten, so staleness is keyed on the root ``zarr.json`` -- which the
    promote writes last on every publish, and therefore changes exactly when
    the store's contents do.
    """
    root_json = store / "zarr.json"
    root_stat = os.stat(root_json)
    if (
        source_png.exists()
        and source_png.stat().st_mtime_ns >= root_stat.st_mtime_ns
    ):
        return
    # Bytes AND mtime: a re-promote with unchanged metadata is byte-identical,
    # and a stale LRU key would serve the previously decoded array.
    content_token = (
        paths_fingerprint([root_json]).removeprefix("sha256:")[:16]
        + f"-{root_stat.st_mtime_ns}"
    )
    _load_zarr_layer_rgb(str(store), content_token, layer, target_px).save(source_png)
    os.utime(source_png, ns=(root_stat.st_mtime_ns, root_stat.st_mtime_ns))
```

Apply the same root-keying at `_preview_tiles.py:76` and `_shared/tiles.py:518`.

- [ ] **Step 4: Run** `uv run pytest tests/unit/gui tests/gui tests/e2e/gui -q`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui tests/unit/gui
git commit -m "fix(gui): key tile staleness on the root zarr.json, not the store dir

Three live sites, two failure modes. file_fingerprint opens its argument as
a file and raises IsADirectoryError on a store -- so the root zarr.json is
used instead. NOT the store directory: paths_fingerprint reduces a
directory to a constant, since it emits one sentinel byte and does not
recurse into contents, which would freeze the cache permanently. The spec's
'paths_fingerprint handles directories' wording is misleading and has been
corrected there too.

Separately, a store directory's st_mtime_ns does NOT change when a nested
chunk is rewritten -- verified by test -- so every mtime compare and every
os.utime moves to that same root file, plus its mtime, since a re-promote
with unchanged metadata is byte-identical while the decoded-array LRU key
must still move.

tiles.py:518 is NOT a staleness site: crop_hdf_rgb opens with del mtime_ns
and its docstring says the parameter is accepted for API compatibility
only. It needs the zarr port, not a staleness fix."
```

---

### Task 4.4: Builder preview tiles

**Files:**
- Modify: `src/phenotypic/gui/builder/_preview_tiles.py`
  (`_channel_to_rgb_uint8` lines 50–63, `stage_channel_png` line 73, the manifest read at
  lines 124–128)
- Modify: `src/phenotypic/gui/builder/_preview_cache.py`
  (`_load_image_auto`, the h5py class probe, lines 158–170; the `_describe` closure inside
  `_build_manifest`, lines 192–210)
- Test: `tests/unit/gui/builder/test_preview_tiles_zarr.py` (create)
- **Test: three existing files** (README test inventory, Phase 4 row) — **shared with Phase 2
  Task 2.4**, which is what actually breaks them. Task 2.4 renames the manifest node key and
  replaces `save_intermediate_layers`, so it owns the manifest half and must leave these three
  green; this task owns only the rendering half. Keep the split explicit or the two tasks will
  each assume the other did it:
  - `tests/gui/builder/test_preview_cache.py` — `:25` hard-codes
    `{"blk": {"hdf": "base_00.h5", ...}}` and `:37` writes `base_00.h5`. **Task 2.4's half.**
  - `tests/gui/builder/test_preview_tile_blueprint.py` — `:16-19` `save2hdf5` + `"hdf"` key;
    `:29-38` calls `save_intermediate_layers` and documents the
    `Image.load_layer_hdf5(hdf, "objmap")` `KeyError` it depends on; `:85-103` passes a bogus
    `node.h5` into `stage_channel_png`. The first two blocks are **Task 2.4's half**;
    `:85-103` is **this task's** — `stage_channel_png` takes a store here, so the "not a
    readable artifact" case must become a malformed store directory.
  - `tests/gui/builder/test_preview_compute_scope.py` — `:100` asserts on
    `pc.scope_dir(...) / "base_00.h5"`, the file `apply_with_intermediates` writes.
    **Task 2.4's half** (it changes all five pipeline call sites).

**Constraints specific to this task:**
- `_preview_cache.py:158-170` (`_load_image_auto`) opens the node artifact with `h5py` to
  read `phenotypic_class` and dispatch `GridImage.load_hdf5` vs `Image.load_hdf5`. Replace
  the whole probe with `load_image_from_store` (Task 2.1), which does exactly this against
  `attributes.phenotypic.image_class`.
- `_preview_cache.py:192-210` reads layer names and shape from the HDF to build the
  manifest node description. Read them from `phenotypic.series` and the level-0 array
  shapes instead — do **not** open a full `Image` for a manifest entry.
- **Lift `_describe` out of `_build_manifest` while porting it** (ledger: wrong-symbol
  sweep). Today it is a **closure** defined at `:192` inside `_build_manifest` (`:173`),
  closing over `sdir` and `nodes`, so nothing outside `compute_scope` can reach it —
  `_preview_cache._describe` is not a module attribute and never was. Port it as a
  module-level `_describe_store_node(store_path: Path) -> dict | None` returning the node
  dict (or `None` when the store is absent), and keep a two-line closure inside
  `_build_manifest` that resolves `sdir / filename` and assigns into `nodes`. The "must not
  open a full `Image`" invariant is the point of this task's port; left inside a closure,
  nothing can assert it.
- Task 2.4 already renamed the manifest key to `"store"` and **introduced**
  `MANIFEST_VERSION` — it did not "bump" it, as an earlier draft of both tasks said.
  `grep -rn "MANIFEST_VERSION" src/phenotypic/gui/` returns nothing today; the builder
  manifest carries no version field at all, and the only `_MANIFEST_VERSION` in the tree is
  the unrelated staged-orchestration constant at `_cli_staged_orchestration.py:47`
  (missing-owner review, 2026-08-19). **Task 2.4 performs the introduction; this task only
  consumes the renamed key** — do not add the constant here, and do not assume it already
  existed. This task consumes the rename at `_preview_tiles.py:124`.

- [ ] **Step 1: Write the failing test**

```python
def test_channel_png_renders_from_a_node_store(tmp_path) -> None:
    from phenotypic import Image
    from phenotypic.gui.builder._preview_tiles import stage_channel_png
    from phenotypic.data import load_synth_yeast_plate

    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr", layers=("gray", "detect_mat", "objmap")
    )
    png = stage_channel_png(tmp_path, "block-1", "gray", store)
    assert png.is_file() and png.stat().st_size > 0


def test_manifest_describe_does_not_load_a_full_image(tmp_path, monkeypatch) -> None:
    """Reads store metadata only; a manifest entry must not cost a full decode."""
    from phenotypic import Image
    from phenotypic.gui.builder import _preview_cache
    from phenotypic.data import load_synth_yeast_plate

    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr", layers=("gray",)
    )
    monkeypatch.setattr(
        Image, "load_zarr", lambda *a, **k: pytest.fail("manifest must not load an Image")
    )
    node = _preview_cache._describe_store_node(store)
    assert node["layers"] == ["gray"]


def test_describe_is_reached_through_compute_scope(tmp_path, monkeypatch) -> None:
    """The lifted helper must still be what the manifest is built from.

    Harness copied from tests/gui/builder/test_preview_compute_scope.py:28-42,
    the only real entry point into _build_manifest.
    """
    from phenotypic.gui.builder import _preview_cache as pc
    from phenotypic.gui.builder._state import (
        BlockNode, Edge, _DagBuilderScope, _DagBuilderState, _new_block_id,
    )

    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    blur = BlockNode(block_id=_new_block_id(), class_name="BlurGauss",
                     params={"sigma": 1})
    scope = _DagBuilderScope()  # __post_init__ seeds InputImage at index 0
    scope.blocks.append(blur)
    scope.edges.append(Edge(
        edge_id=_new_block_id(), source_block_id=scope.blocks[0].block_id,
        source_port="out", target_block_id=blur.block_id, target_port="in",
        kind="image",
    ))
    state = _DagBuilderState(root=scope)

    manifest = pc.compute_scope("sess1", state, [], None, None, None)

    assert manifest["error"] is None
    node = manifest["nodes"][blur.block_id]
    assert node["store"].endswith(".ome.zarr")  # renamed from "hdf" in Task 2.4
    assert "gray" in node["layers"]


def test_class_dispatch_uses_image_class_not_h5py(tmp_path) -> None:
    from phenotypic import GridImage
    from phenotypic.sdk_ import load_image_from_store
    from phenotypic.data import load_synth_yeast_plate

    store = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12).save2zarr(
        tmp_path / "g.ome.zarr"
    )
    assert type(load_image_from_store(store)).__name__ == "GridImage"
```

- [ ] **Step 2: Run to verify failure. Step 3: port. Step 4: run**
`uv run pytest tests/unit/gui/builder tests/gui/builder tests/e2e/gui -q`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/builder tests/unit/gui/builder
git commit -m "refactor(gui): render builder previews from node stores

The h5py class probe becomes load_image_from_store, which dispatches on
attributes.phenotypic.image_class. Manifest descriptions read layer names
and shapes from phenotypic.series and the level-0 arrays rather than
opening a full Image for a cache entry."
```

---

## Phase 4 exit criteria

- [ ] `uv run pytest tests/unit/gui tests/gui -q` is green. **`tests/gui` is not optional
      here** — it is in `testpaths` (`pyproject.toml:218`, **not** `:200`) and holds twelve
      of this phase's files (see the README's test inventory). Omitting it defers the
      breakage to Phase 7. `tests/e2e/gui` is **not** in `testpaths` and is gated on
      `PLAYWRIGHT=1`, so it has never run in CI; it collects cleanly (211 tests) and
      contains no `.h5`/`hdf` reference, so this phase changes nothing in it.
- [ ] `grep -rn "file_fingerprint" src/phenotypic/gui/` returns nothing pointed at a store.
      (It still appears in `_compatibility.py`, `_qc_tab/`, and `analysis/_recipe_state.py`,
      all against real single files — pipeline JSON, the QC DuckDB, recipe files.)
- [ ] `grep -rn "\.h5\|load_hdf5\|hdf_path\|_load_hdf_layer_rgb\|crop_hdf_rgb" src/phenotypic/gui/`
      returns no CODE. Three prose mentions survive in `builder/_preview_cache.py`
      (`:37`, `:201`, `:304`) and are a **documented exception** (lead ruling,
      2026-08-20): they are the recorded reason
      `MANIFEST_VERSION` exists at all — a manifest written before the `.h5` → `.ome.zarr`
      move must MISS and rebuild rather than be read back through a `"hdf"` key. Deleting
      the explanation to satisfy a grep would leave the constant unexplained.
- [ ] The staleness sites plus the two Task 4.1 fingerprints all key on the root
      `zarr.json`, verified by
      `grep -rnE 'zarr\.json|STORE_ROOT_JSON' src/phenotypic/gui/ --include=*.py | wc -l`
      being at least 5. **The literal was hoisted to `ngff_.STORE_ROOT_JSON` in `d1dbaeb6`,
      so a grep for the bare string alone under-counts.** Prefer the constant in new code;
      do not re-introduce the literal to satisfy a grep.
- [ ] A coarse-level read is measurably fewer bytes than level 0 — asserted on
      `ndarray.nbytes` in `test_a_coarse_level_really_is_fewer_bytes`, not on the level
      index, which is only a proxy.

> **Not achieved, by design (report to the phase gate).** "A **whole-plate tile request**
> reads fewer bytes than level 0" is **not** met, and should not be. The only caller of
> `_load_zarr_layer_rgb` is the DZI manifest route, whose source PNG is what
> `_dzi_tiler.tile` builds the deep-zoom pyramid from — so capping it caps the viewer's
> maximum zoom, a user-visible regression rather than an optimisation. That route therefore
> asks for the level-0 longest edge and selects level 0. `select_pyramid_level` is correct,
> tested, and ready for the first caller that genuinely wants a small render; today there
> is none.
