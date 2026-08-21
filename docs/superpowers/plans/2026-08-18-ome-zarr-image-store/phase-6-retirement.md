# Phase 6 — Retirement: the HDF write path, the dead DataFrame layer, docs

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §5.4, §9, and the supersession notes in the spec header.

**Depends on:** Phases 3, 4, **and 5**. Phase 5 is built on the legacy HDF readers, so
nothing here may run until migration exists and is green.

This is a **public-API removal, not an internal cleanup**. `HDF` is re-exported in
`sdk_/__init__.py:240` and in `__all__`, and `phenotypic.sdk_` is published via a
`:recursive:` autosummary with `undoc-members`, so every public name here appears in the
rendered docs. `save2hdf5` / `load_hdf5` / `load_layer_hdf5` additionally appear in
`docs/source/api_reference/core/*.rst` and in runnable doctests.

---

### Task 6.1: Delete the dead HDF DataFrame layer

**Files:**
- Modify: `src/phenotypic/sdk_/hdf_.py` (1,984 lines → roughly 520)
- Test: `tests/unit/sdk_/test_hdf_open_recovery.py` (must keep passing unchanged)

**What is removed** — verified as having **no** call sites in `src/` or `tests/`, and
corroborated by commits `66734e8e9`, `3e8b58aa0`, `da9eb6dd8` removing the last consumers:

`preallocate_series_layout` (1107), `save_series_new` (1276), `save_series_update` (1367),
`save_series_append` (1506), `load_series` (1544), `_convert_categorical_columns` (1600),
`preallocate_frame_layout` (1622), `save_frame_new` (1727), `save_frame_update` (1797),
`save_frame_append` (1880), `load_frame` (1909), the fixed-length-string codecs
(`_get_string_dtype` 620, `_pad_or_truncate_string` 637, `_apply_fixed_length_to_strings`
658, `_trim_trailing_whitespace` 686, `_decode_fixed_length_strings` 698,
`_encode_values_for_hdf5` 727, `_encode_index_for_hdf5` 853, `_decode_values_from_hdf5`
965, `_decode_index_from_hdf5` 1009, `_create_resizable_dataset` 1085), plus the three
additional dead statics `assert_swmr_on` (525), `get_uncompressed_sizes_for_group` (540),
and `close_handle` (1967).

**What is kept — the keeper list.** Deleting past it breaks live tests:

| Keeper | Why |
|---|---|
| `_open_hdf_with_recovery` (34) | Migration read path; also the retry-with-backoff shape reused by `promote_store` |
| `_clear_hdf_consistency_flags` (113) | Called by the above |
| `safe_writer` (252), `swmr_writer` (277) | **Live callers** at `tests/unit/sdk_/test_hdf_open_recovery.py:104` and `:141`. A literal "delete the remainder" breaks that file. |
| `strict_writer` (314), `swmr_reader` (335) | **No call sites** — verified by grep over `src/` and `tests/`. Kept anyway: they are three-line properties on the same public `HDF` class as the two writers above, and removing half a symmetric writer/reader set is a worse public surface than keeping it. Recorded so the justification is honest rather than borrowed from the two that do have callers. |
| `reader` (338), `get_group` (341) and the group accessors | Migration read path |
| `save_array2hdf5` (493) | Used by `tests/fixtures/legacy_hdf/_generate.py` (Task 5.1), which must remain able to rebuild the golden fixtures after the production writer is gone |

**Constraints specific to this task:**
- The spec's earlier figure of 1,700 lines was **wrong**; the correct enumeration is
  ~1,346 lines, reaching ~1,463 with the three additional dead statics. Reaching 1,700 would
  require deleting the keeper list itself. Do not chase a line count.
- Re-verify emptiness immediately before deleting each symbol:
  `grep -rn "<symbol>" src/ tests/ docs/`. A Phase 3/4/5 change may have added a caller.

- [ ] **Step 1: Write the guard test**

Create `tests/unit/sdk_/test_hdf_surface.py`:

```python
"""Pins the HDF keeper list so a future cleanup cannot delete past it."""

from __future__ import annotations

import pytest

from phenotypic.sdk_.hdf_ import HDF, _clear_hdf_consistency_flags, _open_hdf_with_recovery

KEEPERS = [
    "safe_writer",
    "swmr_writer",
    "strict_writer",
    "swmr_reader",
    "reader",
    "get_group",
    "save_array2hdf5",
]

REMOVED = [
    "preallocate_series_layout",
    "save_series_new",
    "save_series_update",
    "save_series_append",
    "load_series",
    "preallocate_frame_layout",
    "save_frame_new",
    "save_frame_update",
    "save_frame_append",
    "load_frame",
    "assert_swmr_on",
    "get_uncompressed_sizes_for_group",
    "close_handle",
]


@pytest.mark.parametrize("name", KEEPERS)
def test_keeper_survives(name: str) -> None:
    assert hasattr(HDF, name), (
        f"{name} has live callers; deleting it breaks test_hdf_open_recovery.py "
        "or the legacy-fixture generator."
    )


@pytest.mark.parametrize("name", REMOVED)
def test_dead_dataframe_layer_is_gone(name: str) -> None:
    assert not hasattr(HDF, name)


def test_recovery_helpers_survive() -> None:
    assert callable(_open_hdf_with_recovery)
    assert callable(_clear_hdf_consistency_flags)
```

- [ ] **Step 2: Run to confirm the REMOVED half fails.**

```bash
uv run pytest tests/unit/sdk_/test_hdf_surface.py -v
```

- [ ] **Step 3: Delete, re-verifying each symbol's emptiness first.**

```bash
for sym in preallocate_series_layout save_series_new save_series_update \
           save_series_append load_series preallocate_frame_layout save_frame_new \
           save_frame_update save_frame_append load_frame assert_swmr_on \
           get_uncompressed_sizes_for_group close_handle; do
  echo "--- $sym"; grep -rn "$sym" src/ tests/ docs/ | grep -v test_hdf_surface.py
done
```

Expected: only `hdf_.py`'s own definitions. Then delete them.

- [ ] **Step 4: Run the suite**

```bash
uv run pytest tests/unit/sdk_/test_hdf_surface.py tests/unit/sdk_/test_hdf_open_recovery.py -v
uv run pytest tests/unit -q
wc -l src/phenotypic/sdk_/hdf_.py
```

Expected: green, and roughly 520 lines remaining.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/hdf_.py tests/unit/sdk_/test_hdf_surface.py
git commit -m "refactor(sdk): delete the dead HDF DataFrame layer

~1,463 lines with no call sites in src/ or tests/, corroborated by
66734e8e9, 3e8b58aa0, and da9eb6dd8 removing the last consumers. The
keeper list is pinned by test: safe_writer and swmr_writer have live
callers in test_hdf_open_recovery.py, and save_array2hdf5 is what the
legacy-fixture generator uses to rebuild the migration goldens after the
production writer is gone. An earlier estimate of 1,700 lines was wrong;
reaching it would mean deleting the keepers."
```

---

### Task 6.2: Remove the public HDF image API

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`
  (delete `_get_hdf5_group` 693, `_save_array2hdf5` 715, `_save_image2hdfgroup` 751,
  `_save_hdf5_metadata` 846, `save2hdf5` 871, `load_hdf5` 1155, `load_layer_hdf5` 1194;
  the doctests at lines 274 and 895; `save_intermediate_layers` was already replaced in
  Task 2.4)
- Modify: `src/phenotypic/_core/_image_parts/_grid_image_handler.py`
  (`_save_image2hdfgroup` 464)
- Modify: `src/phenotypic/_cli/_cli_output_manager.py` — delete `save_image_hdf` (1633) and
  the already-deprecated `save_image_layers` (1697). `save_image_hdf` **calls
  `image.save2hdf5`** (`:1662`, documented at `:1643`), so leaving it behind makes this
  task's own deletion a `AttributeError` waiting in the CLI, and it is what trips this
  phase's exit grep. Task 3.1 promised exactly this — *"`save_image_hdf` is **kept** in this
  phase and removed in Phase 6"* — but no Phase 6 task named the file until the missing-owner
  review of 2026-08-19. Its three callers were all repointed at `save_image_store` in Phase 3
  (`_cli_staged_workers.py:125` and `:225` in Task 3.3, `_cli_process_single.py:183` in
  Task 3.6), so this is a pure deletion. Re-verify with
  `grep -rn "save_image_hdf\|save_image_layers" src/ tests/` before removing.
- Modify: `docs/source/api_reference/core/image_methods.rst` (`:182` the `save2hdf5`
  `automethod`, `:189` the "Load an Image from HDF5 file created with ``save2hdf5``" prose),
  `docs/source/api_reference/core/grid_image_methods.rst` (`:250`, `:264`)
- **Rewrite: `docs/source/how_to/pages/hdf5_storage.md` (38 lines) → the migration page.**
  The whole page is about `save2hdf5` (`:13`) and `apply_with_intermediates`' HDF output
  (`:18-30`); it is reachable from `how_to/index.rst:47`, so it renders in the built docs and
  would otherwise document a removed API. Retitle it "Store Results in OME-Zarr", rewrite the
  two code blocks against `save2zarr` / `save_intermediate_zarr`, replace the "Why HDF5"
  section with the store's equivalents (pyramid levels, partial reads, napari/Vizarr
  interoperability), and **carry the release note here**: one section naming every removed
  public symbol (`save2hdf5`, `load_hdf5`, `load_layer_hdf5`, `save_intermediate_layers`,
  `GridImage.save2hdf5`) and the `uv run python -m phenotypic --mode migrate --output <run>`
  command that converts an existing tree. Rename the file to `zarr_storage.md` and update
  `docs/source/how_to/index.rst:47` in the same commit.

  > **Corrected (missing-owner review, 2026-08-19).** An earlier draft required
  > `Create: docs/source/release_notes/<next-version>.md`. Verified against the worktree:
  > **there is no release-notes infrastructure of any kind** — no `docs/source/release_notes/`
  > directory, no `CHANGELOG.md`/`HISTORY`/`NEWS` at the repo root, and no release-notes entry
  > in `docs/source/index.rst`'s toctree. A new file under an un-toctree'd directory is an
  > orphan, and this phase's own exit criterion runs `sphinx-build -W`, which promotes the
  > orphan warning to an error. Inventing a release-notes tree (directory + `index.rst` +
  > toctree entry + a versioning convention the project has never had) is a larger change than
  > this task, and it is not what the removal needs: what the removal needs is for the page a
  > user lands on to tell them the API is gone and how to convert their data. The `how_to`
  > page above is that page, it is already in the toctree, and it is otherwise unowned.
  > **Assumption stated:** if the project later adopts release notes, this section moves there
  > verbatim.
- Test: `tests/unit/core/test_image_hdf_roundtrip.py` (rewrite as a removal guard),
  `tests/unit/core/test_load_layer_hdf5.py` (delete)
- **Test: `tests/unit/core/test_image_dtype_conversion.py`** — `:590-592` round-trips through
  `save2hdf5` → `Image.load_hdf5`. Port it to `save2zarr` / `load_zarr`; the assertion it makes
  (float32 layers survive a round trip) is about dtype, not about HDF, so it must survive.
- **Test: `tests/unit/test_fixtures.py`** — delete the `temp_hdf5_file` fixture
  (`:134-146`). `grep -rn "temp_hdf5_file" tests/` returns only its own definition; it is dead
  with the API it was written for.

  > These four test files were on the README inventory's **Phase 2** row until the
  > missing-owner review of 2026-08-19, which moved them here — Phase 2 adds `save2zarr`
  > beside `save2hdf5` and removes nothing, so a port done there would be deleted by this
  > task. See the ownership note under the README's test inventory.

**Constraints specific to this task:**
- **`_load_from_hdf5_group` (971), `_load_v2_grouped` (984), and `_load_legacy_flat_group`
  (1079) are KEPT as private readers.** `--mode migrate` calls them. They are removed from
  no public surface because they were never on one.
- `load_hdf5` itself is removed from the public API, but migration needs a reader entry
  point: keep it as `_load_hdf5_for_migration`, called from `sdk_/_hdf_to_zarr.py` and
  `_io_constants.load_image_from_hdf`. Rename in this task and update **all three** call
  sites — the two in `src/`, plus **`phase-5-migrate.md`'s
  `test_converted_equals_a_freshly_written_store`**, which calls
  `Image.load_hdf5(FIXTURES / "v2_grouped" / "img.h5")` directly. An earlier draft counted
  two and would have left that test raising `AttributeError` at a phase whose exit criterion
  is a green full suite. Recorded as OPEN-QUESTIONS **B11**.
- Re-run `grep -rn "load_hdf5" src/ tests/` after the rename and confirm the only hits are
  the private name.
- The doctests at lines 274 and 895 are **runnable** and will fail collection if left
  pointing at removed methods. Rewrite them against `save2zarr` / `load_zarr`.
- A migration/release note is required — this removes names from a published autosummary.
  It lives in `docs/source/how_to/pages/zarr_storage.md` (this task's rewrite of
  `hdf5_storage.md`), **not** in a new `docs/source/release_notes/` tree; see the `Files:`
  correction above for why.

- [ ] **Step 1: Rewrite the round-trip test as a removal guard**

```python
"""The public HDF image API is gone; the private migration readers remain."""

from __future__ import annotations

import pytest

from phenotypic import GridImage, Image


@pytest.mark.parametrize(
    "name", ["save2hdf5", "load_hdf5", "load_layer_hdf5", "save_intermediate_layers"]
)
@pytest.mark.parametrize("cls", [Image, GridImage])
def test_public_hdf_api_is_removed(cls, name: str) -> None:
    assert not hasattr(cls, name)


@pytest.mark.parametrize(
    "name", ["_load_v2_grouped", "_load_legacy_flat_group", "_load_hdf5_for_migration"]
)
def test_private_migration_readers_survive(name: str) -> None:
    """--mode migrate is built on these; Phase 5 breaks without them."""
    assert hasattr(Image, name)


def test_migration_can_still_read_every_golden_fixture(tmp_path) -> None:
    from pathlib import Path

    from phenotypic.sdk_._hdf_to_zarr import migrate_hdf_to_zarr
    from phenotypic.sdk_.ngff_ import valid_staged_store

    fixtures = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_hdf"
    for layout in (
        "v1_flat",
        "v2_grouped",
        "v2_enh_gray",
        "v2_grid",
        "v2_image_type",
        "v2_work_id",
    ):
        store = migrate_hdf_to_zarr(
            fixtures / layout / "img.h5", tmp_path / f"{layout}.ome.zarr"
        )
        assert valid_staged_store(store) is True
```

- [ ] **Step 2: Run to verify it fails. Step 3: delete the write path and rename the reader.**

- [ ] **Step 4: Fix the docs**

In both `.rst` files, replace `save2hdf5` / `load_hdf5` / `load_layer_hdf5` with
`save2zarr` / `load_zarr` / `load_layer_zarr`. Add a release-note entry:

```markdown
### Removed

- `Image.save2hdf5`, `Image.load_hdf5`, `Image.load_layer_hdf5`, and
  `Image.save_intermediate_layers` (and their `GridImage` counterparts). Per-image
  storage is now an OME-Zarr (NGFF 0.5 / Zarr v3) store; use `save2zarr`,
  `load_zarr`, and `load_layer_zarr`.
- The DataFrame half of `phenotypic.sdk_.HDF` (`save_series_*`, `load_series`,
  `save_frame_*`, `load_frame`, `preallocate_*`, and their fixed-length-string
  codecs). These had no remaining call sites.

### Migration

Existing `.h5` output directories are converted with:

    uv run python -m phenotypic --mode migrate --output <previous-output-dir>

A run whose output contains only `.h5` results now fails with a pointer to this
command rather than converting as a side effect.

### Requires

- Python 3.11 or 3.12. Python 3.10 is no longer supported.
```

- [ ] **Step 5: Run the doc build and the doctests**

```bash
uv run pytest --doctest-modules src/phenotypic/_core -q
uv run sphinx-build -W -b html docs/source docs/_build/html
```

Expected: both green. `-W` turns the autosummary's dangling-reference warnings into errors,
which is how a missed `.rst` reference surfaces.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/_core docs tests/unit/core
git commit -m "refactor!: remove the public HDF image API

save2hdf5 / load_hdf5 / load_layer_hdf5 / save_intermediate_layers are
removed from Image and GridImage. This is a public-API removal, not an
internal cleanup: HDF is re-exported from sdk_ and phenotypic.sdk_ is
published via a :recursive: autosummary with undoc-members, so a release
note and .rst updates ship with it. The private v1-flat and v2-grouped
readers are kept -- --mode migrate is built on them -- and the runnable
doctests are rewritten against save2zarr/load_zarr."
```

---

> **Also delete, found by C12 during Phase 3: a dead SLURM script-generation subtree.**
> `_cli_slurm_scripts.py::generate_all_image_scripts` has **no callers anywhere** in
> `src/` or `tests/`, and it is the **only** caller of
> `generate_image_processing_script` (`_cli_slurm_scripts.py:207`, inside it). So the
> pair is an unreachable subtree, not one dead function.
>
> Verify before deleting — the two must go together, and only together:
>
> ```bash
> grep -rn 'generate_all_image_scripts\|generate_image_processing_script' src/ tests/ --include='*.py'
> ```
>
> `generate_slurm_directives`, from the same module, is **live** — `_cli_sentinel_scripts.py:22,73`
> and `_cli_slurm_scripts.py:84`. Do not take the module with the subtree.
>
> One test references the pair deliberately:
> `tests/unit/cli/test_cli_durable_writes_transport.py:152` documents the unreachability in
> its docstring and covers the `--durable-writes` emission so the claim is not untested.
> It goes with the deletion.

### Task 6.3: Remove the HDF path constants

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (`DIR_HDF` 656, `dataset_hdf_dir` 1447,
  `HdfAttr` 1886, `load_image_from_hdf` 1936, `BundleLayout.hdf_path` 2063)
- Modify: `src/phenotypic/sdk_/__init__.py`

**Constraints specific to this task:**
- `load_image_from_hdf` and `dataset_hdf_dir` are still needed by `_hdf_to_zarr.py`. Move
  both **into** `sdk_/_hdf_to_zarr.py` as private helpers rather than deleting them, and
  drop them from `_io_constants.py` and `__all__`. That keeps the legacy layout knowledge
  in the one module that is allowed to know it.
- `HdfAttr` is dead once `load_image_from_hdf` moves; the moved copy carries its own
  `_PHENOTYPIC_CLASS` constant.
- `BundleLayout.hdf_path` is deleted outright — nothing reads legacy trees through
  `BundleLayout`.

> **Settled at the Phase 6 gate (2026-08-20) — do not re-open.** "Legacy layout knowledge
> lives only in `_hdf_to_zarr`" is **not fully reached, and cannot be.** `_DIR_HDF` ends up
> defined **twice** — `_io_constants.py:661` and `_hdf_to_zarr.py:53` — because
> `_io_constants` needs it for `datasets_needing_migration` (the predicate every writing
> mode refuses an unconverted tree on) while `_hdf_to_zarr` **imports from**
> `_io_constants`. Importing back would be circular, and the only alternatives are a third
> module for one three-character string literal, or a fourth hand-spelled `"hdf"`. Both
> copies are module-private, both are one line, and the duplication is the cheapest of the
> available options. Verified, not assumed: `_hdf_to_zarr.py:36` is
> `from ._io_constants import (...)`.

- [ ] **Step 1: Write the guard**

```python
@pytest.mark.parametrize(
    "name", ["DIR_HDF", "dataset_hdf_dir", "HdfAttr", "load_image_from_hdf"]
)
def test_hdf_path_constants_are_gone(name: str) -> None:
    import phenotypic.sdk_ as sdk

    assert not hasattr(sdk, name)


def test_bundle_layout_has_no_hdf_path() -> None:
    from phenotypic.sdk_ import BundleLayout

    assert not hasattr(BundleLayout, "hdf_path")


def test_migration_still_resolves_legacy_directories(tmp_path) -> None:
    from phenotypic.sdk_._hdf_to_zarr import _dataset_hdf_dir

    assert _dataset_hdf_dir(tmp_path, "ds").name == "hdf"
```

- [ ] **Step 2–4: Run to verify failure, move the helpers, re-run**
`uv run pytest tests/unit -q`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_ tests/unit/sdk_
git commit -m "refactor(sdk): retire the HDF path constants

DIR_HDF, dataset_hdf_dir, HdfAttr, load_image_from_hdf, and
BundleLayout.hdf_path leave the shared layout module. The two that
migration still needs move into _hdf_to_zarr.py as private helpers, so
knowledge of the legacy tree layout lives only in the module allowed to
have it."
```

---

> **Task 6.3a was written and then folded away (ledger SIMP-13).** Round 2 added a task
> whose entire deliverable was one unchecked box — *"record the decision and its reasoning in
> the module docstring"* — with no code change required by either branch, no test, and an
> explicit refusal to choose. That is a comment, not a unit of work in an eight-phase DAG.
>
> It is also already decided, twice over. **Retain** is the only branch compatible with
> `--mode migrate` continuing to work, since Task 5.3's pass 1 leans on the module for legacy
> trees; and deleting the `"hdf"` arm "once migration is complete for all known trees" is
> explicitly future work outside this change. The decision and the `TargetKind` reasoning now
> ride on **Task 6.4**, which already owns the docstring and supersession edits.

---

### Task 6.4: Documentation, CLAUDE.md, and supersessions

**Files:**
- Modify: `src/phenotypic/sdk_/_metadata_migration.py` — **module docstring only**, recording
  the `TargetKind` decision below (ledger **GEN-39**: the 6.3a fold left its deliverable
  naming a file no task declared, and the Phase 6 exit criterion checks it).

**`_metadata_migration.py`'s `"hdf"` `TargetKind`: retain it, and correct the reason.**
Record in the module docstring:

- **Retain the `"hdf"` arm.** It is reachable from `rollback_metadata_migration` and from
  the standalone-bundle path, and those are not going away. Keep `csv`/`parquet`/`json`/`frame`
  unconditionally — **`--mode migrate` pass 1** is built on them.

  > **Numbering corrected (ledger MIG-30 / FLOW-36).** Two lines in this section still used
  > the pre-inversion numbering after MIG-15 swapped the passes, so the section contradicted
  > itself within twenty lines — and this is the documentation-of-record task, so the wrong
  > numbering is what would have landed in the module docstring. Pass **1** is the non-image
  > metadata pass; pass **2** is the image conversion, which touches this module not at all.
- **Do NOT add a `"store"` `TargetKind`.** Nothing needs one: header canonicalization is a
  property of the **read** path (`_normalize_stored_metadata_items`, inside both legacy
  loaders), so by the time `save2zarr` runs the metadata is already canonical. That is the
  same fact that made Task 5.5 unnecessary.

> **Correction to ledger FLOW-8 (raised as FLOW-32 `CONFLICT with FLOW-8`, confirmed
> independently as MIG-21).** FLOW-8 justified retention with *"once stores replace HDFs that
> target set is **empty**, so those branches are unreachable rather than incorrect."* **That
> premise is false for the default path.** `keep_source=True` is the default (Task 5.1) and
> `--delete-sources` is opt-in, so after an ordinary in-place migration `results/<ds>/hdf/*.h5`
> still exists — and `_discover_bundle_targets` (`_metadata_migration.py:797-810`) walks
> `dataset_root/"hdf"` and appends every one of them. The set is not empty; without a
> filter, **pass 1** would rewrite headers into retained files nothing will ever read again
> — plus a full `shutil.copy2` of each. See ledger **MIG-25** / **FLOW-35**: that is why
> pass 1 now excludes `.h5` targets outright.
>
> This is not a correctness break — it is doubled migration cost and receipts binding
> artifacts nothing consumes. The conclusion (**retain**) is unchanged, and it is now
> *better* supported: the arm is retained because it is **reachable and load-bearing for
> legacy trees**, not because it is harmlessly dead. Task 5.3's pass 1 carries the matching
> skip (an `.h5` whose stem already has a valid store is dead weight by definition), which is
> what makes the cost claim in Task 5.4 true rather than aspirational.

**Files:**
- Modify: `CLAUDE.md` (the `--mode` list, the output-layout section, the Gotchas entries
  naming `.h5` and `Image.load_hdf5`)
- Modify: `src/phenotypic/_cli/CLAUDE.md` (file inventory, master-vs-mirror section). The
  **staged-GPU sidecar description moved to Phase 3 Task 3.5** (missing-owner review,
  2026-08-19): the sidecar concept dies there, Phases 4 and 5 run in parallel with Phase 3 and
  their executors read this file, and Phase 3's own exit grep
  (`grep -rn "sidecar" src/phenotypic/_cli/`) covers it. Root `CLAUDE.md`'s staged-GPU
  paragraph and `docs/source/how_to/pages/gpu_detection_setup.md` moved with it, along with
  `tests/unit/test_docs_staged_cli.py`, which pins all three. What remains here is only the
  HDF-retirement half of those files.
- Modify: `src/phenotypic/_core/CLAUDE.md`, `src/phenotypic/gui/CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-08-17-flat-metadata-namespace/design.md`
  (record decisions #1 and #7 as superseded)
- Modify: `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md`
  (**Status:** Draft → Implemented, with the plan linked)

**Constraints specific to this task:**
- The flat-metadata spec's decisions #1 and #7 must be **annotated in place as superseded**,
  with a link back to this design — not silently edited. The reasoning has to survive.
  Decision #3 is explicitly **not** superseded.
- `_cli/CLAUDE.md` currently describes the `.npy` objmap sidecar as the Stage-2 signal in
  several places, and the root `CLAUDE.md` repeats it. Both must move to the **retained raw
  `.npy` plus the consumable token — Stage 2 does NOT write into the store** (ledger
  **GEN-38**: the ninth GEN-22 site and the worst-placed, since it instructs writing the
  withdrawn claim into two project `CLAUDE.md` files, where it would outlive the plan), or the next agent will reintroduce the sidecar.
- The staged-GPU `.npy` sidecar sentence in the root `CLAUDE.md`'s "SLURM array auxiliary
  work" section explicitly carves the sidecar out of the no-parallel-sidecar-jobs rule. That
  carve-out is now obsolete; replace it rather than deleting the surrounding rule, which
  still stands.

- [ ] **Step 1: Update the two specs**

In `2026-08-17-flat-metadata-namespace/design.md`, append to decisions #1 and #7:

```markdown
> **Superseded (2026-08-18)** by
> [2026-08-18-ome-zarr-image-store](../2026-08-18-ome-zarr-image-store/design.md).
> Metadata-schema migration moves out of `--mode recompile` and into `--mode migrate`.
> `recompile` stops rewriting legacy headers but keeps reading them — **decision #3
> (permanent stored-data compatibility) is untouched**, so no existing output directory
> breaks.
```

**Decision #7 needs NO annotation — do not add one.** An earlier draft of this task told you
to narrow it so `--mode migrate` could rewrite `deliverables/metadata.csv`. That rewrite was
**withdrawn** (user ruling; ledger **FLOW-4**), because `metadata_sha256` is recomputed from
the file on every run, so any rewrite forced a full re-finalization no matter what migration
wrote into state. Migration now emits `deliverables/metadata.canonical.csv` as a derived
view and never touches the snapshot.

So decision #7 ("The startup metadata snapshot is immutable provenance… never rewritten")
stands **exactly as written**, and this change records nothing against it. Only decision #1
gets the supersession note above.

Verify no stale narrowing survived anywhere:

```bash
grep -rn "metadata.original.csv\|metadata_original_sha256" docs/ src/
```

Expected: matches only inside the withdrawal notes that explain why they do not exist.

- [ ] **Step 2: Update the four CLAUDE.md files.** Verify no stale reference remains:

```bash
grep -rn "\.h5\|save2hdf5\|load_hdf5\|load_layer_hdf5\|save_intermediate_layers\|npy.*sidecar\|objmap sidecar" CLAUDE.md AGENTS.md src/phenotypic/*/CLAUDE.md docs/source
```

Expected: only migration-context mentions.

- [ ] **Step 3: Build the docs**

```bash
uv run sphinx-build -W -b html docs/source docs/_build/html
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md src/phenotypic/*/CLAUDE.md docs
git commit -m "docs: record the OME-Zarr layout and the two supersessions

The flat-metadata spec's decisions #1 and #7 are annotated in place rather
than edited away, so the reasoning survives; decision #3 is explicitly not
superseded. Every CLAUDE.md that described the .npy objmap sidecar as the
Stage-2 signal now describes the retained raw .npy plus the consumable
token -- Stage 2 does not write into the store -- including the carve-out in
the SLURM auxiliary-work rule, which is obsolete as written while the
surrounding rule still stands."
```

---

## Phase 6 exit criteria

- [ ] `uv run pytest tests -q` is green (whole suite, not just unit).
- [ ] `sphinx-build -W` **introduces no new warnings versus the phase base** (`81f16fda`).

      > **Amended (USER RULING, 2026-08-20).** The criterion used to read
      > *"`uv run sphinx-build -W -b html docs/source docs/_build/html` succeeds."*
      > **It was never achievable** — not at this phase, not at the phase base, not
      > anywhere in the OME-Zarr change. Measured at the Phase 6 gate, both sides fail
      > and both exit **1**:
      >
      > | tree | result |
      > |---|---|
      > | Phase 6 HEAD (clean single-pass build) | `build finished with problems, 24565 warnings (with warnings treated as errors).` |
      > | Phase base `81f16fda` | `build finished with problems, 24585 warnings (with warnings treated as errors).` |
      >
      > **This amendment is not a weakening made to reach green.** A "succeeds"
      > criterion that the base already fails by 24,585 warnings tests nothing about
      > the phase; the no-regression form tests exactly what the phase controls.
      > Phase 6 passes it at **net −20**.
      >
      > **Root cause of the pre-existing debt.** **23,934 of the base's 24,585 (97%)
      > are a single bug**: `more than one target found for cross-reference 'None':
      > phenotypic.abc_.FootprintMixin.None, phenotypic.abc_.ImageOperation.None,
      > phenotypic.abc_.ObjectDetector.None, phenotypic.sdk_.FootprintMixin.None
      > [ref.python]`. Four classes each register a `.None` attribute target, so
      > **every** autodoc'd docstring whose `Returns:` is `None` trips it — 2,312
      > emitted from `enhance/__init__.py`, 1,650 from `refine/`, 1,479 from `detect/`,
      > 1,264 from `abc_/`, 870 from `tune/`. The remaining ~650 are ordinary RST
      > problems: 372 `Bullet list ends without a blank line`, 51 `Inline strong
      > start-string without end-string`, 51 the emphasis equivalent, 14
      > `autosummary.import_cycle`, 8 autodoc import failures, 8 `Block quote ends
      > without a blank line`, 4 `Title underline too short`.
      >
      > **Phase 6's delta: 11 warnings gone, 1 new.** The 11 gone are exactly the
      > removed symbols' docstrings — `HDF.assert_swmr_on`, `close_handle`,
      > `preallocate_series_layout`, `preallocate_frame_layout`,
      > `save_series_{new,update,append}`, `save_frame_{new,update,append}`, and
      > `BundleLayout.hdf_path`. The **one new** warning is
      > `ImageIOHandler.save2zarr`'s docstring hitting **that same pre-existing
      > `.None` bug**, because `api_reference/core/image_methods.rst` now carries
      > `.. automethod:: Image.save2zarr` where it carried `Image.save2hdf5`. Net on
      > that bug alone: **−10**. There is no new *class* of warning.
      >
      > **Method, so the next person can re-measure.** Build the base in a **detached
      > scratch worktree** (`git worktree add --detach <scratch> 81f16fda`) with
      > `PYTHONPATH=<scratch>/src` — the venv's editable install is a plain-path
      > `.pth`, so `PYTHONPATH` wins and autodoc documents the base source; confirm
      > with `python -c "import phenotypic; print(phenotypic.__file__)"` before
      > trusting the run. Never check anything out in the live worktree. Then compare
      > **normalized warning sets**: grep `WARNING` from both logs, strip each tree's
      > path prefix, collapse `:<line>:` to `:L:`, `sort -u`, and `comm`. Two traps:
      > a build killed mid-write and **resumed** reports order-dependent `duplicate
      > object description` warnings that a clean build does not (a resumed run here
      > showed 3 spurious "new" warnings, not 1) — always compare **clean single-pass
      > against clean single-pass**. And the full build takes **~80 min** per side on
      > a contended node, so budget the `timeout` accordingly.

      > **The 24,585-warning debt is separate, scoped work — not part of this change.**
      > It is worth opening on its own, because the expensive half is already done:
      > **97% of it is one root cause**, the ambiguous `.None` cross-reference target,
      > and it is a `conf.py`-level fix (disambiguate or suppress the four `.None`
      > registrations), not 23,934 individual edits. Clearing it would take the build
      > from 24,585 warnings to roughly 650 — at which point the ordinary RST
      > problems become individually fixable and a genuine `-W` gate becomes
      > reachable for the first time. None of that is chasable inside the OME-Zarr
      > change, and absorbing it here would have buried a repo-wide docs fix inside a
      > storage-format refactor.
- [ ] No live reference to the removed public names survives:

      ```bash
      grep -rn "save2hdf5\|load_layer_hdf5\|save_intermediate_layers" src/ docs/source \
        | grep -v 'docs/source/how_to/pages/zarr_storage.md'
      ```

      returns nothing.

      > **Corrected (missing-owner review, 2026-08-19).** The bare grep is self-defeating in
      > two ways. First, Task 6.2's own required deliverable — the section naming every removed
      > symbol, so a user who lands on the migration page learns what went away — **must**
      > contain the strings `save2hdf5`, `load_layer_hdf5`, and `save_intermediate_layers`, and
      > it lives under `docs/source`. A gate that a task's own required output trips is a gate
      > the implementer deletes the output to satisfy. Second, the earlier draft placed that
      > deliverable at `docs/source/release_notes/<next-version>.md`, a directory that does not
      > exist and is in no toctree, so `sphinx-build -W` — the criterion two lines above —
      > would have failed it as an orphan. The single exclusion above resolves both: the one
      > page allowed to name the removed API is the page whose job is to name it.
- [ ] The exclusion above is not hiding anything else:
      `grep -c "save2hdf5\|load_layer_hdf5\|save_intermediate_layers" docs/source/how_to/pages/zarr_storage.md`
      is non-zero (the migration section is actually present) and that file is the **only**
      path the unfiltered grep reports.
- [ ] The forward write path holds no `.h5` reference:
      `grep -rn "\.h5" src/phenotypic --include='*.py' | grep -vE '_hdf_to_zarr|hdf_\.py|_metadata_migration\.py|_image_io_handler\.py'`
      returns **no WRITE site**. **The module allow-list is required, not a weakening** —
      verified that `hdf_.py:192` (`EXT = {".h5", ".hdf5", ".hdf", ".he5"}`, on the kept
      `HDF` class), `_metadata_migration.py:112` (`_HDF_SUFFIXES`), and the retained
      private readers in `_image_io_handler.py` all survive this phase **by design**.
      A bare grep would force the implementer to delete a keeper or silently soften the
      gate (ledger GEN-10).

      > **Corrected in execution (2026-08-20), following the Phase 4 ruling.** The
      > earlier wording said the filtered grep "returns nothing." It never could, and
      > that was never what the criterion is about. Measured at the Phase 6 gate the
      > grep reports **40 lines across 12 modules**, and **not one of them is a write**:
      > 12 are comment-only lines and 17 more are docstring prose (29 explanatory lines
      > in total), 8 are user-facing strings, 2 are path *naming* in
      > `_cli_output_manager`, and exactly **one** touches the filesystem —
      > `_io_constants.py:1500`, `hdf_dir.glob("*.h5")` inside
      > `datasets_needing_migration`, which is the **detect** predicate every writing
      > mode refuses an unconverted tree on. Removing it would delete the guard, not
      > the write path.
      >
      > This is the same shape Phase 4 hit at `builder/_preview_cache.py` and the same
      > ruling applies: **do not delete an explanation, or a message a user needs, to
      > satisfy a grep for a code pattern.** The criterion is extended to its own stated
      > intent — *code on a write path* — rather than the prose being deleted or the
      > allow-list being widened until the count reached zero.
      >
      > **Documented exceptions** (lead ruling, 2026-08-20), each reviewed at the gate:
      >
      > - `_io_constants.py:1500` — `hdf_dir.glob("*.h5")` in
      >   `datasets_needing_migration`. A read/detect predicate, shared by the CLI
      >   refusal and the GUI consistency report. **Load-bearing; must not be removed
      >   while legacy trees exist.**
      > - `_cli_output_manager.py:1477` (`extensions={"hdf": ".h5"}`) and `:1549`
      >   (`ext = self.extensions.get("hdf", ".h5")`) — `get_output_path` *naming* a
      >   legacy path for the only two readers that resolve one
      >   (`image_data_artifact`'s `"hdf"` completion-marker fallback and
      >   `_migrate_legacy_success_evidence`). Its own docstring at `:1451` records
      >   that nothing has written an `.h5` since Phase 3 Task 3.6.
      > - `phenotypicCLI.py:293` (the `--mode full/measure/recompile/process` refusal),
      >   `:1023/:1025/:1033` (`--mode` / `--delete-sources` help), `:1289/:1291`
      >   (the `--help` mode table), `_cli_migrate.py:185` (pass-2 progress), and
      >   `_output_consistency.py:318` (the GUI migration prompt) — **user-facing
      >   strings**. A migration prompt that cannot name the format the user has on
      >   disk is not a migration prompt.
      > - The remaining **29** lines are comments and docstrings explaining *why* the
      >   HDF era ended: `_preview_cache.py:37/:201/:304` (why `MANIFEST_VERSION`
      >   exists — the Phase 4 exception, carried forward unchanged),
      >   `_processing_inventory.py:303` and `ngff_.py:1120` (the measured
      >   entries-per-store figures the store-bounded walk rests on),
      >   `_cli_completion.py:29/:118/:133/:135/:146`, `_cli_migrate.py:8/:20/:143`,
      >   `_io_constants.py:1466/:1473/:1475/:1476`,
      >   `_cli_output_manager.py:1405/:1451/:1453/:1512`,
      >   `_cli_process_single.py:632`, `typing_.py:120`, `_run.py:57/:59`,
      >   `_output_consistency.py:72`, `phenotypicCLI.py:253/:284/:2991`.
      >
      > Mechanized form of the extended criterion — the filtered grep, minus
      > comment-only lines, must report only the sites enumerated above:
      >
      > ```bash
      > grep -rn "\.h5" src/phenotypic --include='*.py' \
      >   | grep -vE '_hdf_to_zarr|hdf_\.py|_metadata_migration\.py|_image_io_handler\.py' \
      >   | grep -vE ':[0-9]+:[[:space:]]*#'
      > ```
- [ ] `wc -l src/phenotypic/sdk_/hdf_.py` is roughly 520.
- [ ] `docs/source/how_to/pages/zarr_storage.md` exists, is reachable from
      `docs/source/how_to/index.rst`, and names every removed public symbol plus the
      `--mode migrate` command. `docs/source/how_to/pages/hdf5_storage.md` no longer exists
      and no toctree references it.
- [ ] `_metadata_migration.py`'s module docstring records the `"hdf"`-arm retention decision,
      the no-`"store"`-`TargetKind` reasoning, and the FLOW-32 correction (ledger **SIMP-13**
      — this is Task 6.3a's deliverable, folded into Task 6.4 rather than shipped as its own
      task).
