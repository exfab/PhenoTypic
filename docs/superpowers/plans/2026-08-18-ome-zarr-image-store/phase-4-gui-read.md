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

**Interfaces:**
- Consumes: `BundleLayout.store_path` (Task 2.1).
- Produces: `OutputRoot.store_path(dataset, stem) -> Path | None`, replacing `hdf_path`.

**Constraints specific to this task:**

- `store_path` uses `is_dir()`, not `is_file()`.

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

```python
"""OutputRoot discovers store directories without walking into them."""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.sdk_ import zarr_store_path


def _seed(root: Path, stems: list[str]) -> None:
    (root / "deliverables").mkdir(parents=True, exist_ok=True)
    (root / "deliverables" / "master_measurements.parquet").write_bytes(b"")
    for stem in stems:
        store = zarr_store_path(root, "ds", stem)
        (store / "gray" / "0").mkdir(parents=True)
        (store / "zarr.json").write_text("{}", encoding="utf-8")


def test_store_path_resolves_a_directory(tmp_path: Path) -> None:
    _seed(tmp_path, ["a"])
    root = OutputRoot.discover(tmp_path)
    assert root.store_path("ds", "a") == zarr_store_path(tmp_path, "ds", "a")


def test_store_path_is_none_when_absent(tmp_path: Path) -> None:
    _seed(tmp_path, ["a"])
    assert OutputRoot.discover(tmp_path).store_path("ds", "missing") is None


def test_discovery_does_not_walk_into_stores(tmp_path: Path, monkeypatch) -> None:
    """A recursive scan costs 400k stat calls at 10k images."""
    _seed(tmp_path, ["a", "b"])
    visited: list[str] = []
    real_iterdir = Path.iterdir

    def _counting(self):
        visited.append(str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _counting)
    OutputRoot.discover(tmp_path)
    assert not any("/gray/" in seen for seen in visited), visited


def test_discovery_finds_every_store(tmp_path: Path) -> None:
    _seed(tmp_path, ["a", "b", "c"])
    root = OutputRoot.discover(tmp_path)
    assert all(root.store_path("ds", stem) is not None for stem in "abc")
```

- [ ] **Step 2: Run it to verify it fails.** Expected: `AttributeError: … 'store_path'`.

Add two tests for the fingerprint sites:

```python
def test_processing_fingerprint_changes_when_a_store_changes(tmp_path: Path) -> None:
    """Enumerating directories would freeze this permanently (D5)."""
    _seed(tmp_path, ["a"])
    root = OutputRoot.discover(tmp_path)
    before = root.source_fingerprint
    (zarr_store_path(tmp_path, "ds", "a") / "zarr.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    assert OutputRoot.discover(tmp_path).source_fingerprint != before


def test_image_source_token_changes_when_a_store_changes(tmp_path: Path) -> None:
    """It is a staleness fingerprint, not a report label (D4)."""
    from phenotypic.gui.results_viewer._output_root import _image_source_token

    _seed(tmp_path, ["a"])
    store = zarr_store_path(tmp_path, "ds", "a")
    before = _image_source_token([store / "zarr.json"])
    (store / "zarr.json").write_text('{"changed": true}', encoding="utf-8")
    assert _image_source_token([store / "zarr.json"]) != before
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
  (the h5py class probe at lines 160–170, `_describe` at lines 181–208)
- Test: `tests/unit/gui/builder/test_preview_tiles_zarr.py` (create)

**Constraints specific to this task:**
- `_preview_cache.py:160-170` opens the node artifact with `h5py` to read
  `phenotypic_class` and dispatch `GridImage.load_hdf5` vs `Image.load_hdf5`. Replace the
  whole probe with `load_image_from_store` (Task 2.1), which does exactly this against
  `attributes.phenotypic.image_class`.
- `_preview_cache.py:197-208` reads layer names and shape from the HDF to build the
  manifest node description. Read them from `phenotypic.series` and the level-0 array
  shapes instead — do **not** open a full `Image` for a manifest entry.
- Task 2.4 already renamed the manifest key to `"store"` and bumped `MANIFEST_VERSION`;
  this task consumes that rename at `_preview_tiles.py:124`.

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
    from phenotypic import Image
    from phenotypic.gui.builder import _preview_cache
    from phenotypic.data import load_synth_yeast_plate

    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr", layers=("gray",)
    )
    monkeypatch.setattr(
        Image, "load_zarr", lambda *a, **k: pytest.fail("manifest must not load an Image")
    )
    node = _preview_cache._describe("block-1", store.name, base_dir=tmp_path)
    assert node["layers"] == ["gray"]


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

- [ ] `uv run pytest tests/unit/gui tests/gui tests/e2e/gui -q` is green. **`tests/gui`
      is not optional here** — it is in `testpaths` (`pyproject.toml:200`) and holds twelve
      of this phase's files (see the README's test inventory). Omitting it defers the
      breakage to Phase 7.
- [ ] `grep -rn "file_fingerprint" src/phenotypic/gui/` returns nothing pointed at a store.
- [ ] `grep -rn "\.h5\|load_hdf5\|hdf_path\|_load_hdf_layer_rgb\|crop_hdf_rgb" src/phenotypic/gui/` returns nothing.
- [ ] The three live staleness sites plus the two Task 4.1 fingerprints all key on `zarr.json`, verified by
      `grep -rn 'zarr.json' src/phenotypic/gui/ | wc -l` being at least 5.
- [ ] A whole-plate tile request measurably reads fewer bytes than level 0 — assert in
      `test_small_request_selects_a_coarse_level`.
