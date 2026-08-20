# Phase 3 — CLI write path and the staged-GPU engine

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §3.2–§3.7, §4.3, §4.4.

**Depends on:** Phase 2.
**Runs in parallel with:** Phases 4 and 5.

This is the safety-critical phase. The staged engine's resume classifier is what decides
whether a finished image is reprocessed or skipped, and all three defects the spec's
independent review caught lived here. **Task 3.4's differential test is the gate** — it is
the test that would have caught all three, and it must be written before the classifier is
touched.

---

### Task 3.1: `save_image_store` on `OutputManager`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_output_manager.py`
  (add beside `save_image_hdf`, line 1633; `save_image_layers` at line 1688 is already
  deprecated and is **not** ported)
- Test: `tests/unit/cli/test_cli_output_manager.py` (extend)
- Test (regression only, run — do not edit): `tests/integration/cli/test_staged_gpu_local.py`
  (where `save_image_hdf` is actually covered; the rename at `:742` is listed below)

**Interfaces:**
- Consumes: `Image.save2zarr`, `zarr_store_path`, `ngff_.durable_writes_enabled`.
- Produces:
  ```python
  def save_image_store(
      self,
      image: "Image",
      dataset_name: str,
      image_stem: str,
      *,
      work_id: str | None = None,
      durable: bool | None = None,
  ) -> Optional[Path]
  ```

**Constraints specific to this task:**
- The old signature's `root_attributes: Mapping[str, str] | None` is **replaced by an
  explicit `work_id` argument**. Today the CLI patches `phenotypic_work_id` in post-write
  via `h5py.File(tmp, "r+")` (line 1666); under the new ordering invariant the root
  `zarr.json` is written last, so a post-hoc patch is impossible by construction.
- **`save_image_hdf` has three callers, not two.** Verified:
  `_cli_staged_workers.py:125` and `:225` pass
  `root_attributes={"phenotypic_work_id": work_id}` (at `:129` and `:229`) and become
  `work_id=work_id`; **`_cli_process_single.py:183` passes none** and becomes a bare
  `save_image_store(image, dataset_name, image_stem)`. The spec's §4.4 lists
  `_cli_process_single.py` only as a "loader swap", so its writer swap is under-specified
  there. There is also a name-monkeypatch at
  `tests/integration/cli/test_staged_gpu_local.py:742` that must be renamed with it.
- Failure semantics are preserved exactly: log a warning and return `None`, never raise.
  The staged workers turn `None` into a `RuntimeError` themselves (lines 133 and 231), and
  that layering must not change.
- On failure, clean up the `.part` directory rather than the file — `tmp_path.unlink()` at
  line 1676 becomes `shutil.rmtree(part, ignore_errors=True)`.
- `save_image_hdf` is **kept** in this phase and removed in Phase 6, so a half-migrated
  tree never has two writers fighting.

- [ ] **Step 1: Write the failing test**

> **Corrected (wrong-symbol sweep).** An earlier draft wrote
> `manager = _make_manager(tmp_path)  # existing helper in this module`. There is no such
> helper: `tests/unit/cli/test_cli_output_manager.py` tests **module-level** functions
> (`aggregate_measurements`, `split_master_by_feature`, …) and never names `OutputManager`
> at all — `grep -n OutputManager` on it returns nothing. The tests would have failed with
> `NameError`, not the `AttributeError` Step 2 predicts. The construction that the real
> `save_image_hdf` tests use is `OutputManager.from_config(out, ".tiff", save_overlays=False)`
> (`tests/integration/cli/test_staged_gpu_local.py:114`); define the helper below, and add
> `from phenotypic._cli._cli_output_manager import OutputManager` to the file's existing
> import block (the file already imports six other names from that module).

Append to `tests/unit/cli/test_cli_output_manager.py`:

```python
def _make_manager(tmp_path: Path) -> OutputManager:
    """Same construction the real save_image_hdf tests use.

    See tests/integration/cli/test_staged_gpu_local.py:114.
    """
    return OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)


def test_save_image_store_writes_under_results_dataset_zarr(tmp_path) -> None:
    from phenotypic import Image
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.data import load_synth_yeast_plate

    manager = _make_manager(tmp_path)
    saved = manager.save_image_store(
        Image(load_synth_yeast_plate()), "ds", "img"
    )
    assert saved == zarr_store_path(tmp_path, "ds", "img")
    assert saved.is_dir()


def test_save_image_store_writes_work_id_at_write_time(tmp_path) -> None:
    """The root zarr.json is written last, so a post-hoc patch is impossible."""
    from phenotypic import Image
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes
    from phenotypic.data import load_synth_yeast_plate

    manager = _make_manager(tmp_path)
    saved = manager.save_image_store(
        Image(load_synth_yeast_plate()), "ds", "img", work_id="w-7"
    )
    assert read_phenotypic_attributes(saved)[PhenotypicAttr.WORK_ID] == "w-7"


def test_save_image_store_returns_none_and_logs_on_failure(tmp_path, monkeypatch, caplog) -> None:
    """Preserves save_image_hdf's contract: the workers raise, not the manager."""
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    manager = _make_manager(tmp_path)
    monkeypatch.setattr(
        Image, "save2zarr", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    assert manager.save_image_store(Image(load_synth_yeast_plate()), "ds", "img") is None
    assert any("Failed to save" in record.message for record in caplog.records)


def test_save_image_store_cleans_up_the_part_directory_on_failure(tmp_path, monkeypatch) -> None:
    from phenotypic import Image
    from phenotypic.sdk_ import dataset_zarr_dir
    from phenotypic.data import load_synth_yeast_plate

    manager = _make_manager(tmp_path)
    monkeypatch.setattr(
        Image, "save2zarr", lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
    )
    manager.save_image_store(Image(load_synth_yeast_plate()), "ds", "img")
    leftovers = list(dataset_zarr_dir(tmp_path, "ds").glob("*.part"))
    assert leftovers == []


def test_save_image_store_result_passes_valid_staged_store(tmp_path) -> None:
    from phenotypic import Image
    from phenotypic.sdk_.ngff_ import valid_staged_store
    from phenotypic.data import load_synth_yeast_plate

    manager = _make_manager(tmp_path)
    saved = manager.save_image_store(Image(load_synth_yeast_plate()), "ds", "img")
    assert valid_staged_store(saved) is True
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/cli/test_cli_output_manager.py -k save_image_store -v
```

Expected: `AttributeError: 'OutputManager' object has no attribute 'save_image_store'`.

- [ ] **Step 3: Implement**

```python
    def save_image_store(
        self,
        image: "Image",
        dataset_name: str,
        image_stem: str,
        *,
        work_id: str | None = None,
          durable: bool | None = None,
    ) -> Optional[Path]:
        """Save a processed image as an OME-Zarr store under ``results/<ds>/zarr/``.

        Atomicity comes from :func:`phenotypic.sdk_.ngff_.promote_store`: the
        image is built into a uuid-suffixed ``.part`` sibling and promoted by
        directory rename.

        ``work_id`` is a first-class argument rather than the old
        ``root_attributes`` mapping. The store's root ``zarr.json`` is written
        last so an interrupted write reads as absent, which makes the previous
        post-write patch (``h5py.File(tmp, "r+")``) impossible by construction.

        Args:
            image: Image object with processing results.
            dataset_name: Dataset name.
            image_stem: Image filename without extension.
            work_id: CLI work id, written into ``attributes.phenotypic``.
            durable: ``fsync`` before promoting; ``None`` auto-detects SLURM.

        Returns:
            Path where the store was promoted, or ``None`` if saving failed.
            Callers that require publication (the staged workers) turn ``None``
            into a ``RuntimeError`` themselves; that layering is deliberate.
        """
        from phenotypic.sdk_ import zarr_store_path
        from phenotypic.sdk_.ngff_ import discard_parts_for

        # OutputManager's root attribute is ``base_dir``; there is no
        # ``self.output_dir`` (verified: _cli_output_manager.py:1416).
        final_path = zarr_store_path(self.base_dir, dataset_name, image_stem)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            saved = image.save2zarr(
                final_path,
                work_id=work_id,
                durable=durable,
            )
            logger.info("Saved OME-Zarr store for %s/%s", dataset_name, image_stem)
            return saved
        except Exception as e:
            # One owner for the .part naming convention: re-encoding it here
            # would duplicate matching logic sweep_orphan_parts already has,
            # outside the module that defines the suffix (ledger SIMP-6).
            discard_parts_for(final_path)
            logger.warning(
                "Failed to save OME-Zarr store for %s/%s: %s: %s",
                dataset_name,
                image_stem,
                type(e).__name__,
                e,
            )
            return None
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/cli/test_cli_output_manager.py -v
```

Expected: all PASS. Then run `uv run pytest tests/integration/cli/test_staged_gpu_local.py -q`
— that is where `save_image_hdf` is actually exercised, and it must stay green because this
task does not touch it.

> **Corrected (wrong-symbol sweep).** An earlier draft said "including the pre-existing
> `save_image_hdf` tests" in `test_cli_output_manager.py`. There are none there;
> `save_image_hdf` is covered by `tests/integration/cli/test_staged_gpu_local.py`, hence the
> second command.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_output_manager.py tests/unit/cli/test_cli_output_manager.py
git commit -m "feat(cli): add save_image_store

work_id becomes an explicit argument instead of a root_attributes mapping
patched in post-write: the store's root zarr.json is written last, so
patching it afterwards would violate the ordering invariant that makes an
interrupted write read as absent. Failure still returns None and logs --
the staged workers are what turn that into a RuntimeError."
```

---

### Task 3.2: The consumable Stage-2 token

**Files:**
- Create: `src/phenotypic/_cli/_cli_stage2_token.py`
- Test: `tests/unit/cli/test_cli_stage2_token.py` (create)
- (`src/phenotypic/_cli/_cli_sidecar.py` and `tests/unit/cli/test_cli_sidecar.py` are
  deleted in Task 3.5, once every caller has moved.)

**Interfaces:**
- Consumes: `progress_dir`, `atomic_write_with_writer`.
- Produces:
  ```python
  def stage2_token_path(output_dir: Path, dataset: str, image_stem: str) -> Path
  def write_stage2_token(output_dir, dataset, image_stem, *, objmap_shape: tuple[int, int]) -> Path
  def stage2_token_exists(output_dir, dataset, image_stem) -> bool
  def read_stage2_token(output_dir, dataset, image_stem) -> dict
  def delete_stage2_token(output_dir, dataset, image_stem) -> None

  def stage2_raw_path(output_dir: Path, dataset: str, image_stem: str) -> Path
  def write_stage2_raw(output_dir, dataset, image_stem, array: np.ndarray) -> Path
  def load_stage2_raw(output_dir, dataset, image_stem) -> np.ndarray
  def delete_stage2_raw(output_dir, dataset, image_stem) -> None
  ```

**The raw array is retained, deliberately (OPEN-QUESTIONS D1).**

`<output>/.phenotypic/progress/stage2_raw/<dataset>/<stem>.npy` holds Stage 2's **raw**
detector output — pre-`_write_object_output`, pre-`drop_frame_background`, pre-relabel — and
Stage 3 consumes it. This is not a leftover of the old sidecar; it is the property the old
sidecar provided and that nothing else does.

Without it, Stage 3's input is the store's own objmap, which Stage 3 then re-promotes over —
so the raw output is destroyed the moment Stage 3 first succeeds. The retry window is real:
`save_image_store` lands at `_cli_staged_workers.py:225` but the completion marker is not
written until `:251`, with `save_overlay` and `PlotCoordinator.emit_image` in between. A
timeout there leaves the classifier reading `"stage3"`, and the second pass runs
`_write_object_output` on already-refined labels. `drop_frame_background`
(`_objmap_accessor.py:498-509`) zeroes the label owning the plurality of border pixels
**after excluding the already-zeroed background**, so the plurality falls to whichever real
colony touches the frame most — and that colony is silently deleted, once per retry.

The store's objmap is **not** written by Stage 2 (see Task 3.3 — only the final store needs
third-party interop); the raw `.npy`
is what makes Stage 3 replayable.

**Constraints specific to this task:**
- Path is `<output>/.phenotypic/progress/stage2_done/<dataset>/<stem>.json`, i.e.
  `progress_dir(output_dir) / "stage2_done" / dataset / f"{stem}.json"` — the same shape as
  `stage3_completion_marker_path` (`_cli_staged_resume.py:113`), which uses
  `"stage3_complete"`.
- Written atomically (temp + rename) via `atomic_write_with_writer`, exactly as
  `write_sidecar` does today.
- The token carries the objmap's level-0 shape **and nothing else**. It is **consumable**:
  `delete_stage2_token` mirrors `delete_sidecar`, and the resume planner's `"complete"`
  branch tests its **absence**.
- **NGFF metadata never carries resume state.** In particular `ome.labels` is not a
  substitute: a durable labels list makes the `"complete"` conjunct permanently false, so
  every finished image is reprocessed forever and `migrate_legacy_stage3_markers` is
  silently disabled. `zarr.Group.members()` also enumerates children by store listing and
  returns a partially written `objmap`, so the labels list is not even the only discovery
  path.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/cli/test_cli_stage2_token.py`:

```python
"""The Stage-2 token replaces the .npy sidecar. It must be consumable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic._cli._cli_stage2_token import (
    delete_stage2_token,
    read_stage2_token,
    stage2_token_exists,
    stage2_token_path,
    write_stage2_token,
)
from phenotypic.sdk_ import progress_dir


def test_token_lives_under_progress_not_in_the_store(tmp_path: Path) -> None:
    """Resume state lives where the rest of it already lives."""
    path = stage2_token_path(tmp_path, "ds", "img")
    assert path == progress_dir(tmp_path) / "stage2_done" / "ds" / "img.json"
    assert ".ome.zarr" not in str(path)


def test_write_then_exists_then_delete(tmp_path: Path) -> None:
    assert stage2_token_exists(tmp_path, "ds", "img") is False
    write_stage2_token(tmp_path, "ds", "img", objmap_shape=(64, 48))
    assert stage2_token_exists(tmp_path, "ds", "img") is True
    delete_stage2_token(tmp_path, "ds", "img")
    assert stage2_token_exists(tmp_path, "ds", "img") is False


def test_delete_is_idempotent(tmp_path: Path) -> None:
    delete_stage2_token(tmp_path, "ds", "img")
    delete_stage2_token(tmp_path, "ds", "img")


def test_token_carries_the_objmap_shape_and_no_work_id(tmp_path: Path) -> None:
    """No `work_id` field (ledger FLOW-20).

    `stage2_detect_core` has no work_id parameter, so the field could only ever
    be None -- and a field that can only hold None gets misread as meaningful.
    The work-id check that matters reads `attributes.phenotypic.work_id` off the
    STORE (`staged_store_matches_work_id`), not the token.
    """
    write_stage2_token(tmp_path, "ds", "img", objmap_shape=(64, 48))
    payload = read_stage2_token(tmp_path, "ds", "img")
    assert tuple(payload["objmap_shape"]) == (64, 48)
    assert "work_id" not in payload


def test_token_is_written_atomically(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []
    import phenotypic._cli._cli_stage2_token as module

    real = module.atomic_write_with_writer
    monkeypatch.setattr(
        module,
        "atomic_write_with_writer",
        lambda final, writer: (seen.append(str(final)), real(final, writer))[1],
    )
    write_stage2_token(tmp_path, "ds", "img", objmap_shape=(2, 2))
    assert seen == [str(stage2_token_path(tmp_path, "ds", "img"))]


def test_token_is_valid_json(tmp_path: Path) -> None:
    write_stage2_token(tmp_path, "ds", "img", objmap_shape=(2, 2))
    json.loads(stage2_token_path(tmp_path, "ds", "img").read_text(encoding="utf-8"))


def test_read_missing_token_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_stage2_token(tmp_path, "ds", "img")


# --- the retained raw array (D1) -------------------------------------------


def test_raw_array_lives_beside_the_token(tmp_path: Path) -> None:
    from phenotypic._cli._cli_stage2_token import stage2_raw_path

    assert stage2_raw_path(tmp_path, "ds", "img") == (
        progress_dir(tmp_path) / "stage2_raw" / "ds" / "img.npy"
    )


def test_raw_array_round_trips_exactly(tmp_path: Path) -> None:
    """Stage 3 replays from this, so it must be bit-exact."""
    import numpy as np

    from phenotypic._cli._cli_stage2_token import load_stage2_raw, write_stage2_raw

    array = np.arange(64, dtype=np.uint16).reshape(8, 8)
    write_stage2_raw(tmp_path, "ds", "img", array)
    np.testing.assert_array_equal(load_stage2_raw(tmp_path, "ds", "img"), array)
    assert load_stage2_raw(tmp_path, "ds", "img").dtype == array.dtype


def test_raw_array_is_written_atomically(tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    import phenotypic._cli._cli_stage2_token as module

    seen: list[str] = []
    real = module.atomic_write_with_writer
    monkeypatch.setattr(
        module,
        "atomic_write_with_writer",
        lambda final, writer: (seen.append(str(final)), real(final, writer))[1],
    )
    module.write_stage2_raw(tmp_path, "ds", "img", np.zeros((2, 2), dtype=np.uint16))
    assert seen == [str(module.stage2_raw_path(tmp_path, "ds", "img"))]


def test_raw_delete_is_idempotent(tmp_path: Path) -> None:
    from phenotypic._cli._cli_stage2_token import delete_stage2_raw

    delete_stage2_raw(tmp_path, "ds", "img")
    delete_stage2_raw(tmp_path, "ds", "img")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/cli/test_cli_stage2_token.py -v
```

Expected: `ModuleNotFoundError: No module named 'phenotypic._cli._cli_stage2_token'`.

- [ ] **Step 3: Write the module**

```python
"""Consumable Stage-2 completion token for the staged GPU engine.

Replaces the ``.npy`` objmap sidecar. Stage 2 retains its **raw** detector
output under ``stage2_raw/`` and drops this token; Stage 3 replays the raw
array and consumes both, exactly as it used to consume the sidecar. Stage 2
does **not** write into the promoted store -- only the final store needs
third-party interop, and an in-store write would be visible to the uncached
crop route as raw pre-``drop_frame_background`` labels.

The token is deliberately **not** NGFF metadata. Using ``ome.labels`` as the
"Stage 2 done" signal is not an exact replacement for ``sidecar_exists()`` and
would break resume in two ways:

* The sidecar is consumable -- ``delete_sidecar`` ran at the end of Stage 3 and
  the resume planner's ``"complete"`` branch tests its **absence**. A durable
  labels list makes that conjunct permanently false, so ``"complete"`` never
  fires and every finished image is reprocessed. It also silently disables
  ``migrate_legacy_stage3_markers``.
* The labels list is not the only discovery path: ``zarr.Group.members()``
  enumerates children by store listing and returns a partially written
  ``objmap``, which reads as a mix of real labels and ``fill_value``. NGFF only
  says label images SHOULD be listed; it grants no exclusivity.

Consequently, NGFF metadata never carries resume state. Resume state lives in
``.phenotypic/progress/``, where the rest of it already lives.
"""

from __future__ import annotations

import json
from pathlib import Path

from phenotypic.sdk_ import atomic_write_with_writer, progress_dir

_STAGE2_DIR = "stage2_done"


def stage2_token_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """``<output>/.phenotypic/progress/stage2_done/<dataset>/<stem>.json``."""
    return progress_dir(output_dir) / _STAGE2_DIR / dataset / f"{image_stem}.json"


def write_stage2_token(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    *,
    objmap_shape: tuple[int, int],
) -> Path:
    """Atomically record that Stage 2 published this image's label array.

    Carries the objmap shape only. An earlier draft also carried ``work_id``,
    which ``stage2_detect_core`` has no parameter for and which could therefore
    only ever be ``None`` (ledger **FLOW-20**). The work-id conjunct that
    matters is read off the store by ``staged_store_matches_work_id``.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Image stem.
        objmap_shape: Level-0 ``(y, x)`` extent of the written objmap.

    Returns:
        The token path.
    """
    final = stage2_token_path(output_dir, dataset, image_stem)
    payload = {
        "objmap_shape": [int(objmap_shape[0]), int(objmap_shape[1])],
    }

    def _write(path: str) -> None:
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    atomic_write_with_writer(final, _write)
    return final


def stage2_token_exists(output_dir: Path, dataset: str, image_stem: str) -> bool:
    """Return whether Stage 2 has published and Stage 3 has not yet consumed."""
    return stage2_token_path(output_dir, dataset, image_stem).is_file()


def read_stage2_token(output_dir: Path, dataset: str, image_stem: str) -> dict:
    """Read the token payload.

    Raises:
        FileNotFoundError: If the token does not exist.
    """
    return json.loads(
        stage2_token_path(output_dir, dataset, image_stem).read_text(encoding="utf-8")
    )


def delete_stage2_token(output_dir: Path, dataset: str, image_stem: str) -> None:
    """Consume the token. Idempotent, mirroring ``delete_sidecar``."""
    stage2_token_path(output_dir, dataset, image_stem).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The retained raw detector output
# ---------------------------------------------------------------------------

_STAGE2_RAW_DIR = "stage2_raw"


def stage2_raw_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """``<output>/.phenotypic/progress/stage2_raw/<dataset>/<stem>.npy``."""
    return progress_dir(output_dir) / _STAGE2_RAW_DIR / dataset / f"{image_stem}.npy"


def write_stage2_raw(
    output_dir: Path, dataset: str, image_stem: str, array: "np.ndarray"
) -> Path:
    """Atomically retain Stage 2's **raw** detector output for Stage 3 to replay.

    This is what makes Stage 3 idempotent under retry. Stage 3 re-promotes the
    store over its own objmap, so the store cannot serve as its own input a
    second time: on a replay ``_write_object_output`` would run again on
    already-refined labels, and ``drop_frame_background`` would zero whichever
    real colony touches the frame most -- silently, once per retry.

    Written before the token, so a crash between them leaves no token and
    Stage 2 simply re-runs.
    """
    final = stage2_raw_path(output_dir, dataset, image_stem)

    def _write(path: str) -> None:
        import numpy as np

        with open(path, "wb") as handle:
            np.save(handle, array)

    atomic_write_with_writer(final, _write)
    return final


def load_stage2_raw(output_dir: Path, dataset: str, image_stem: str) -> "np.ndarray":
    """Load the retained raw detector output.

    Raises:
        FileNotFoundError: If Stage 2 did not retain one.
    """
    import numpy as np

    return np.load(stage2_raw_path(output_dir, dataset, image_stem))


def delete_stage2_raw(output_dir: Path, dataset: str, image_stem: str) -> None:
    """Consume the raw array. Idempotent; always paired with the token."""
    stage2_raw_path(output_dir, dataset, image_stem).unlink(missing_ok=True)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/cli/test_cli_stage2_token.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_stage2_token.py tests/unit/cli/test_cli_stage2_token.py
git commit -m "feat(cli): add the consumable Stage-2 token

Replaces the .npy sidecar without moving resume state into NGFF metadata.
A durable ome.labels list would make the resume planner's 'complete'
conjunct permanently false -- every finished image reprocessed forever --
and would silently disable migrate_legacy_stage3_markers. The token lives
under .phenotypic/progress/ beside the Stage-3 marker it pairs with."
```

---

### Task 3.3: Stage 1, Stage 2, and Stage 3 workers

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py`
  (`stage1_preprocess_core` line 99, `stage2_detect_core` line 139,
  `ensure_staged_overlay` line 168, `stage3_merge_measure_core` line 193)
- Test: `tests/integration/cli/test_staged_store_stages.py` (create)
- **Test: `tests/integration/cli/test_staged_gpu_local.py` — two *named* edits, but expect
  the file to be broadly red** (Task 3.5 owns the wholesale port):

  > **Measured (C9): after this task the file runs `20 failed, 2 passed` of 22.** The two
  > edits below are the ones this task must make deliberately; the rest fail because they
  > still assert on `dataset_hdf_dir(out, ...)/"<stem>.h5"` or on sidecars — e.g.
  > `assert staged_hdf.is_file()`, and `FileNotFoundError` on `stage2_raw/ds/img.npy` from
  > the SLURM shard worker. That is **expected sequencing**, not a regression: Task 3.5 ports
  > the file. Do not try to fix it here, and do not read Task 3.3's step-4 command as a gate
  > on this file.
  1. `:740-744` monkeypatches the **instance attribute by name**:
     `monkeypatch.setattr(om, "save_image_hdf", lambda *a, **k: None)`. `save_image_hdf` is
     deliberately **kept** until Phase 6 (Task 3.1), so the patch still *succeeds* — it just
     patches a method Stage 3 no longer calls, the injected failure never occurs, and the
     test fails on a `DID NOT RAISE` instead of an `AttributeError`. Rename it to
     `save_image_store`.
  2. `:746` is `pytest.raises(RuntimeError, match="Stage 3 HDF publication failed")`. Step 3
     below changes both worker messages from `HDF` to `store`
     (`_cli_staged_workers.py:135` and `:235`), so the match string becomes
     `"Stage 3 store publication failed"`. Prefer matching on `"Stage 3"` alone, as this
     task's own `test_stage3_raises_when_publication_fails` already does.

**Interfaces:**
- Consumes: `save_image_store` (3.1), the Stage-2 token (3.2), `valid_staged_store` (1.6).
- Produces: the same four function signatures, with `hdf` locals replaced by `store`.

**Constraints specific to this task:**
- **Stage 1** writes a complete store including a **zeros `objmap`** with its `ome.labels`
  list and `image-label` block. That is what lets `valid_staged_store` mirror
  `valid_staged_hdf` exactly, and it is what today's HDF writer already does.
- **Stage 2 does not touch the store.** It reads the input layer, runs inference, writes the
  **raw** result to `.phenotypic/progress/stage2_raw/`, and drops the token. The store keeps
  Stage 1's zeros objmap until Stage 3 promotes the post-refined one.

  This is a **user ruling**, and it reverses spec §3.4's in-place write. The rationale: only
  the *final* store needs third-party interop, so there is nothing for a mid-run in-store
  objmap to buy — while the cost is real, because the **uncached crop route**
  (`gui/_shared/tiles.py:349-392`, `del mtime_ns`, *"crop reads are windowed and not
  full-layer cached"*) would serve those raw, pre-`drop_frame_background`, pre-relabel labels
  to the colony view for the whole Stage-2 → Stage-3 window — hours, on SLURM.

  It is also exact parity with the HDF path today, where the detector output lived in the
  sidecar and never in the `.h5`. Dissolves ledger **FLOW-5**, **FLOW-12**, **D11**, and the
  **B10** cross-phase dependency.
- **Stage 3 re-promotes the entire store.** This is not optional. Post-ops (refiners, size
  filters) mutate the objmap, and this re-save is what publishes the **post-refined**
  segmentation. Removing it would leave the label image holding raw detector output that
  disagrees with the parquet and with a single-pass run, violating the
  byte-identical-to-single-pass contract in `_cli/CLAUDE.md`.
- **Preserve the existing `work_id is None` guard verbatim.** Today
  `write_stage3_completion_marker` and `delete_sidecar` run only when `work_id is None`
  (`_cli_staged_workers.py:250-258`); the work-id path publishes markers elsewhere, in
  `_cli_staged_slurm_worker.py:409`. Port the token deletion into the **same** guard.
  Making it unconditional here would double-delete against the SLURM worker and change
  resume classification — this is exactly the kind of silent divergence Task 3.4's
  differential test exists to catch.
- There is **no** pyramid-level question for Stage 2 any more: it writes no objmap levels at
  all. Stage 3's promote rebuilds every level from the post-refined array through the normal
  `save2zarr` path, so a stale coarse level under a fresh level 0 is unreachable by
  construction rather than prevented by a rule.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/cli/test_staged_store_stages.py`:

```python
"""Stage 1/2/3 against a real store. The post-refined objmap test is the point."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic._cli._cli_stage2_token import stage2_token_exists
from phenotypic.schema import OBJECT
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_.ngff_ import valid_staged_store

#: The measurement column is ``Object_Label``, not ``ObjectLabel`` --
#: ``schema/_object.py:7,22`` with ``category() == "Object"``. Resolve it
#: through the schema rather than spelling it, so a rename cannot silently
#: turn the most load-bearing test in this plan into a KeyError.


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
    raw_labels = set(
        np.unique(load_stage2_raw(run.output_dir, "ds", "img"))
    ) - {0}
    assert raw_labels, "fixture must produce detections before post-ops run"
    run.run_stage3()
    published = set(
        np.unique(
            Image.load_layer_zarr(
                zarr_store_path(run.output_dir, "ds", "img"), "objmap"
            )
        )
    ) - {0}
    parquet_labels = set(run.read_measurements()[str(OBJECT.LABEL)].tolist())
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

    run.simulate_timeout_after_promote()  # removes the marker, keeps token + raw
    run.run_stage3()

    np.testing.assert_array_equal(Image.load_layer_zarr(store, "objmap"), once)
    assert set(run.read_measurements()[str(OBJECT.LABEL)]) == set(
        measurements_once[str(OBJECT.LABEL)]
    )


def test_stage3_replays_from_the_raw_array_not_the_store(staged_run, monkeypatch) -> None:
    """Pins the input source, so a later 'simplification' cannot swap it back."""
    from phenotypic._cli import _cli_stage2_token

    staged_run.run_stage1()
    staged_run.run_stage2()
    reads: list[str] = []
    # Patch the WORKER's binding, not `_cli_stage2_token`'s (C9). Step 3's code
    # block imports `load_stage2_raw` by name, so the worker holds its own
    # reference and a patch on the defining module is invisible to it -- the
    # test would fail on `reads == []` no matter what Stage 3 did.
    #
    # And assert the PUBLISHED PIXELS, not a call count: the substitute returns
    # a sentinel label, so this survives an import-style refactor and still
    # catches a swap back to the store, where substituting the raw loader would
    # have no observable effect at all.
    sentinel = np.full(shape, 7, dtype=np.uint16)
    monkeypatch.setattr(
        _cli_staged_workers, "load_stage2_raw", lambda *a: sentinel
    )
    staged_run.run_stage3()
    assert reads == ["raw"]


def test_stage3_leaves_the_token_alone_on_the_work_id_path(staged_run_with_work_id) -> None:
    """Preserves today's guard: with a work_id, markers are published by the
    SLURM worker, not here. Making this unconditional double-deletes."""
    run = staged_run_with_work_id
    run.run_stage1()
    run.run_stage2()
    run.run_stage3()
    assert stage2_token_exists(run.output_dir, "ds", "img") is True


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
```

Add four fixtures to `tests/integration/cli/conftest.py`. Each builds a one-image dataset
from `load_synth_yeast_plate()` and a `StagePlan` with a trivial CPU stand-in for the
`GpuDetector` (the stages take the detector as an argument, so no GPU is needed), and
exposes `run_stage1/2/3`, `output_dir`, `store(dataset, stem)`, and `read_measurements()`.

| Fixture | What it adds |
|---|---|
| `staged_run` | The baseline. |
| `staged_run_with_size_filter` | A post-op that provably removes exactly one colony, so `test_stage3_publishes_the_post_refined_objmap` has something to detect. |
| `staged_run_with_work_id` | A non-`None` `work_id`, exercising the guarded tail. |
| `staged_run_with_border_colony` | A detector stand-in whose output has **a real colony touching the frame** plus a background blob, so `drop_frame_background` has a second victim available on a replay. Also exposes `simulate_timeout_after_promote()`, which removes the Stage-3 completion marker while leaving the token and the raw array — reproducing the exact `_cli_staged_workers.py:225`-to-`:251` window. Without a border-touching colony the D1 idempotency test is **vacuous**: `drop_frame_background` returns early at `_objmap_accessor.py:503` when no non-zero label reaches the border, so a second pass would be a harmless no-op and the test would pass even with the defect present. Assert in the fixture that a border colony exists. |

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/integration/cli/test_staged_store_stages.py -v
```

Expected: FAIL — the stages still write HDF.

- [ ] **Step 3: Port the three stage cores**

`stage1_preprocess_core` (line 125):

```python
    saved_store = output_manager.save_image_store(
        image, dataset_name, image_stem, work_id=work_id
    )
    if saved_store is None or not valid_staged_store(saved_store):
        raise RuntimeError(
            f"Stage 1 store publication failed for {dataset_name}/{image_stem}"
        )
```

`stage2_detect_core` (lines 152–166) — load the input layer from the store, write the raw
detector output to the `.npy`, then drop the token. **It does not write into the store**; the
code block below is the authority, and any prose to the contrary is stale (ledger FLOW-19,
GEN-22):

```python
    store = zarr_store_path(output_dir, dataset_name, image_stem)
    image = image_cls.load_zarr(store)  # read-only use; never re-promoted here
    array = getattr(image, detector.input_layer)[:]
    try:
        sample = detector._preprocess(array)
        batch = detector._collate([sample])
        result = detector._infer_batch(batch)[0]
    except MemoryError:
        raise
    except Exception as exc:
        raise PerImageScientificError(STAGE_GPU_DETECT, exc) from exc
    _check_active(active_check)
    # Stage 2 does NOT write into the store. Only the final store needs
    # third-party interop, and an in-store write here would be visible to the
    # uncached crop route as raw pre-drop_frame_background labels. The raw
    # array precedes the token, so a crash before the token leaves no
    # "Stage 2 done" signal and Stage 2 simply recomputes.
    write_stage2_raw(output_dir, dataset_name, image_stem, result)
    write_stage2_token(
        output_dir,
        dataset_name,
        image_stem,
        objmap_shape=(int(result.shape[0]), int(result.shape[1])),
    )
```

> No `work_id=` here: the token carries `objmap_shape` only. **Task 3.2 already defines it
> that way** — see its writer and `test_token_carries_the_objmap_shape_and_no_work_id`
> (ledger FLOW-20).

`ensure_staged_overlay` (line 184): `dataset_hdf_dir(...)/f"{stem}.h5"` →
`zarr_store_path(output_dir, dataset_name, image_stem)`; `load_hdf5` → `load_zarr`.

`stage3_merge_measure_core` (lines 205–258): replace the `load_sidecar` merge with a
store read, keep everything after it, and port the guarded tail **verbatim in shape**:

```python
    store = zarr_store_path(output_dir, dataset_name, image_stem)
    image = image_cls.load_zarr(store)
    image.name = image_stem

    # Replay from Stage 2's RETAINED RAW output, never from the store's own
    # objmap. Stage 3 re-promotes over that objmap, so using it as input makes
    # a retried Stage 3 re-run _write_object_output on already-refined labels
    # -- and drop_frame_background then deletes a real colony. See D1.
    #
    # NOTE (ledger FLOW-21): this restores idempotency for the OBJMAP only. The
    # image loaded here is the already-post-processed store, so a post-op that
    # touches detect_mat or gray is applied twice on a retry. Pre-existing --
    # the HDF path re-saved the same way -- and out of scope here, but
    # test_stage3_is_idempotent_under_retry asserts only the objmap and the
    # label set, so it does not see it.
    result = load_stage2_raw(output_dir, dataset_name, image_stem)
    try:
        plan.gpu_detector._write_object_output(image, result)
        plan.post_pipeline.apply(image, inplace=True)
        measurements = plan.post_pipeline.measure(image, apply_post=False)
    except MemoryError:
        raise
    except Exception as exc:
        raise PerImageScientificError(STAGE_MEASURE, exc) from exc

    _check_active(active_check)
    output_manager.save_measurements(measurements, dataset_name, image_stem)
    _check_active(active_check)
    # Re-promote: post-ops mutate the objmap, and this is what publishes the
    # POST-REFINED segmentation. Without it the stored label image disagrees
    # with the parquet and with a single-pass run.
    saved_store = output_manager.save_image_store(
        image, dataset_name, image_stem, work_id=work_id
    )
    if saved_store is None or not valid_staged_store(saved_store):
        raise RuntimeError(
            f"Stage 3 store publication failed for {dataset_name}/{image_stem}"
        )
    ...
    if work_id is None:
        _check_active(active_check)
        write_stage3_completion_marker(
            output_dir, dataset_name, image_name or image_stem, image_stem
        )
        _check_active(active_check)
        # Consume both. The completion marker is already written above, so a
        # crash between these two deletes classifies "complete" either way and
        # the survivor is inert garbage -- but delete the token FIRST, so the
        # only reachable intermediate state is "no token, orphan raw" (Stage 2
        # would recompute and overwrite it) rather than "token present, raw
        # missing" (Stage 3 would replay into a FileNotFoundError).
        delete_stage2_token(output_dir, dataset_name, image_stem)
        delete_stage2_raw(output_dir, dataset_name, image_stem)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/integration/cli/test_staged_store_stages.py -v
```

Expected: all PASS, in particular `test_stage3_publishes_the_post_refined_objmap`.

- [ ] **Step 5: Prove the post-refined test has teeth**

Temporarily delete the `save_image_store` re-promote from `stage3_merge_measure_core` and
re-run:

```bash
uv run pytest tests/integration/cli/test_staged_store_stages.py::test_stage3_publishes_the_post_refined_objmap -v
```

Expected: FAIL. Restore the re-promote and confirm PASS. Record the observed failure
message in the commit body — a test that cannot be shown to fail is not a guard.

- [ ] **Step 5a: Prove the D1 idempotency test has teeth**

Seed the store **once, in the test**, then make Stage 3 read it:

```python
# 1. In the test, between run_stage2() and the FIRST run_stage3():
store = zarr_store_path(run.output_dir, "ds", "img")
seeded = Image.load_zarr(store)
seeded.objmap[:] = load_stage2_raw(run.output_dir, "ds", "img")
seeded.save2zarr(store)

# 2. In stage3_merge_measure_core, replacing `result = load_stage2_raw(...)`:
    result = image.objmap[:]
```

> **There is no `Image.write_layer_zarr` (C9).** An earlier draft of this seed called one;
> `grep -rn "write_layer_zarr" src/ tests/` returns nothing. The handler exposes
> `save2zarr` / `save_intermediate_zarr` / `load_zarr` / `load_layer_zarr` — reading a layer
> is a one-shot helper, writing one is not. Load, assign, re-promote, as above. What matters
> for the proof is unchanged: the seed happens **once, in the test, upstream of the retry**.

and re-run:

```bash
uv run pytest tests/integration/cli/test_staged_store_stages.py::test_stage3_is_idempotent_under_retry -v
```

Expected: FAIL, with the second pass's objmap missing the border-touching colony that the
first pass kept. Restore the single `load_stage2_raw` call, remove the seed, and confirm
PASS.

If it **passes**, do not adjust the fixture — check the mutation first. Both previous drafts
of this proof produced a passing mutant, and in neither case was the fixture at fault.

> **Two earlier drafts of this proof were wrong, in opposite directions (ledger FLOW-15,
> then FLOW-33). Do not simplify it back.**
>
> *Draft 1* said to substitute `result = image.objmap[:]` alone. Under the round-2 design
> Stage 2 no longer writes the store, so pass 1 loads Stage 1's **zeros** →
> `_write_object_output(image, zeros)` → `drop_frame_background` early-returns
> (`_core/_image_parts/accessors/_objmap_accessor.py:504`) → the published objmap is zeros
> and the measurements empty. Pass 2 loads the same zeros. **The mutated code passes**, and
> the draft told the executor a pass meant the fixture was wrong.
>
> *Draft 2* moved the seed **into** `stage3_merge_measure_core`
> (`image.objmap[:] = load_stage2_raw(...)`, then read it back). That is worse: it is a
> **no-op**. `_write_object_output` opens with `image.objmap[:] = result.astype(np.uint16)`
> (`abc_/_gpu_detector.py:243`), discarding the prior objmap before reading it, and
> `ObjectMap.__setitem__`'s full-slice fast path (`_objmap_accessor.py:203-216`) round-trips
> a `uint16` array losslessly — so seed-then-read is byte-identical to the correct code. And
> the seed would re-run on the retry, re-supplying clean raw labels to the very pass that is
> supposed to see refined ones.
>
> **The seed must happen once, upstream of the retry.** That is why it lives in the test.
> Pass 1 then reads raw labels, refines them, and re-promotes; pass 2 reads the **refined**
> store, `drop_frame_background` fires a second time, and the border-touching colony
> disappears. That is the state D1 actually described.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_workers.py src/phenotypic/sdk_/ngff_.py tests/integration/cli
git commit -m "feat(cli): port the three staged workers to the OME-Zarr store

Stage 1 emits a zeros objmap so valid_staged_store mirrors valid_staged_hdf
exactly. Stage 2 does NOT write into the store: only the final store needs
third-party interop, and an in-store write here would be visible to the
uncached crop route as raw pre-drop_frame_background labels. It writes its
RAW output to .phenotypic/progress/stage2_raw/ and drops the token last --
raw before token, so a crash between them leaves no "done" signal and
Stage 2 simply recomputes.

Stage 3 replays from that raw array, not from the store. The store is what
Stage 3 re-promotes over, so using it as input makes a retried Stage 3
re-run _write_object_output on already-refined labels, and
drop_frame_background then zeroes whichever real colony touches the frame
most -- silently, once per retry. Verified by swapping the input back and
watching test_stage3_is_idempotent_under_retry fail.

Stage 3 still re-promotes the whole store, because post-ops mutate the
objmap and that re-save is what publishes the post-refined segmentation;
verified by deleting it and watching
test_stage3_publishes_the_post_refined_objmap fail. The work_id is None
guard around the marker and the two deletions is preserved verbatim."
```

---

### Task 3.4: Resume classifier and the differential parity test

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`
  (`valid_staged_hdf` line 69, `staged_hdf_matches_work_id` line 99,
  `classify_staged_image` line 167, `migrate_legacy_stage3_markers` line 287,
  `clear_downstream_artifacts_for_stage1` line 314, `reconcile_stage3_publications` line 322)
- Test: `tests/unit/cli/test_staged_resume_parity.py` (create),
  `tests/unit/cli/test_staged_resume.py` (extend)

**Interfaces:**
- Consumes: `valid_staged_store` (1.6), `stage2_token_exists` (3.2), `zarr_store_path` (2.1).
- Produces:
  ```python
  def staged_store_matches_work_id(path: Path, work_id: str) -> bool
  ```
  and a `classify_staged_image` with an unchanged signature and unchanged return values.

**Constraints specific to this task:**
- `classify_staged_image`'s **signature and return values do not change**. Only the
  artifact probes underneath it change: `hdf` → store, `valid_staged_hdf` →
  `valid_staged_store`, `staged_hdf_matches_work_id` → `staged_store_matches_work_id`,
  `sidecar_exists` → `stage2_token_exists`.
- Every branch is preserved **in order**, including the `process_only_layer == "objmap"`
  early return (line 190), both `stage3_completion_exists` branches (lines 205 and 211),
  and the `not markers_required and parquet and not sidecar → "complete"` branch (line
  221). The last one is the branch a durable labels list would have broken.
- `migrate_legacy_stage3_markers` keeps working. It depends on the token's **absence**
  marking completion; if it stops firing, resume state is wrong for every legacy tree.
- This task's differential test is the phase gate. Write it **first**, run it against the
  unmodified HDF classifier to confirm it passes, and only then port.

- [ ] **Step 1: Write the differential parity test**

Create `tests/unit/cli/test_staged_resume_parity.py`:

```python
"""Differential resume parity: the zarr classifier must agree with the HDF one.

This is the test that would have caught all three resume defects the spec's
independent review found. It enumerates every combination
classify_staged_image currently distinguishes and asserts the two artifact
worlds produce the same stage, rather than asserting a hand-written table that
could itself encode the bug.

tests/unit/cli/test_staged_resume.py already parameterizes markers_required at
:57, :86, :108, :128, :146, :165; this mirrors that shape across all four axes.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from phenotypic._cli._cli_staged_resume import classify_staged_image

PROCESS_ONLY_LAYERS = [None, "objmap", "gray"]
MARKERS_REQUIRED = [True, False]
WORK_IDS = [None, "w-1"]
#: Which durable artifacts exist, as
#: (image_state, stage2_signal, parquet, stage3_marker, image_success_marker).
#:
#: The FIFTH axis is load-bearing. classify_staged_image's first branch
#: (_cli_staged_resume.py:182) consults valid_image_success, which reads the
#: per-image completion marker. Without this axis that branch is never
#: exercised -- valid_image_success returns False in both worlds -- and the
#: parity test passes while production breaks. See Task 3.8 / OPEN-QUESTIONS D2.
ARTIFACTS = list(itertools.product([False, True], repeat=5))

CASES = [
    pytest.param(layer, markers, work_id, artifacts, id=f"{layer}-{markers}-{work_id}-{artifacts}")
    for layer, markers, work_id, artifacts in itertools.product(
        PROCESS_ONLY_LAYERS, MARKERS_REQUIRED, WORK_IDS, ARTIFACTS
    )
]


@pytest.mark.parametrize(("layer", "markers", "work_id", "artifacts"), CASES)
def test_zarr_classifier_matches_the_hdf_classifier(
    layer, markers, work_id, artifacts, hdf_world, zarr_world
):
    """hdf_world / zarr_world build the same artifact set in the two formats."""
    hdf_root = hdf_world(artifacts, work_id=work_id)
    zarr_root = zarr_world(artifacts, work_id=work_id)
    common = dict(
        dataset="ds",
        image=Path("img.tif"),
        input_root=Path("/in"),
        process_only_layer=layer,
        markers_required=markers,
        expected_work_id=work_id,
    )
    assert classify_staged_image(output_dir=zarr_root, **common) == (
        hdf_world.classify(output_dir=hdf_root, **common)
    )
```

`hdf_world` and `zarr_world` are fixtures in `tests/unit/cli/conftest.py`. `hdf_world`
pins the **pre-port** HDF classifier: copy `classify_staged_image` and its four probe
functions into `tests/_legacy_staged_resume.py` **before** touching the source, and have
`hdf_world.classify` call that frozen copy. Freezing it is the point — a differential test
against a classifier that moves with the code proves nothing.

- [ ] **Step 2: Run it against the unmodified source to confirm it passes**

```bash
uv run pytest tests/unit/cli/test_staged_resume_parity.py -q
```

Expected: PASS (both sides are still effectively the HDF classifier via `zarr_world`
building HDF artifacts). If it fails here, the fixtures are wrong — fix them before
porting. **Do not proceed until this is green.**

- [ ] **Step 3: Port the classifier**

Replace `valid_staged_hdf` with a re-export and port the work-id probe:

```python
from phenotypic.sdk_.ngff_ import valid_staged_store  # noqa: F401 -- public re-export


def staged_store_matches_work_id(path: Path, work_id: str) -> bool:
    """Return whether a valid staged store is bound to ``work_id``.

    Replaces ``staged_hdf_matches_work_id``. The work id lives in
    ``attributes.phenotypic.work_id``, written at store-build time.
    """
    if not valid_staged_store(path):
        return False
    try:
        from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

        block = read_phenotypic_attributes(path)
        return block.get(PhenotypicAttr.WORK_ID) == work_id
    except (OSError, KeyError, ValueError, TypeError):
        return False
```

In `classify_staged_image`, change only the probes (lines 196–221):

```python
    store = zarr_store_path(output_dir, dataset, stem)
    if expected_work_id is not None:
        store_valid = staged_store_matches_work_id(store, expected_work_id)
    else:
        store_valid = valid_staged_store(store)
    if not store_valid:
        return "stage1"
    ...
    stage2_done = stage2_token_exists(output_dir, dataset, stem)
    ...
    if (
        process_only_layer is None
        and not markers_required
        and parquet.is_file()
        and not stage2_done
    ):
        return "complete"

    # An explicit branch, NOT `stage2_done and raw.is_file()` (ledger FLOW-40).
    # The token is only a flag; Stage 3's real INPUT is the raw .npy. Without
    # this, a token-present/raw-missing image classifies "stage3" forever: the
    # worker reports a missing prereq rather than a scientific failure -- an
    # improvement -- but nothing ever routes it back to Stage 2, so it cannot
    # recover.
    #
    # It must NOT be folded into `stage2_done`, because `not stage2_done` is a
    # conjunct of the "complete" branch above: ANDing the raw in would flip a
    # token-present/raw-missing image that has a parquet all the way to
    # "complete".
    if stage2_done and not stage2_raw_path(output_dir, dataset, stem).is_file():
        return "stage2"

    return "stage3" if stage2_done else "stage2"
```

Update `migrate_legacy_stage3_markers`, `clear_downstream_artifacts_for_stage1`, and
`reconcile_stage3_publications` to use the store path and the token.

**Both artifact-clearing sites in this file must clear the token AND the raw `.npy`** —
`clear_downstream_artifacts_for_stage1` (`:318`) and `reconcile_stage3_publications`
(`:364`). Token first at each, for the reason given in Task 3.3: deleting the raw and
leaving the token makes the next Stage 3 replay into a `FileNotFoundError`, while the
reverse merely orphans a `.npy`. Task 3.5 counts these among the six `delete_sidecar`
sites, but **this task owns the file** (ledger **M6**).

> **`clear_downstream_artifacts_for_stage1` deletes nothing extra.** An earlier draft of
> this task said it must `rmtree` the store because "an `unlink` there raises
> `IsADirectoryError`". That rests on a misreading — **verified**: the function
> (`_cli_staged_resume.py:314-319`) deletes only the `.npy` sidecar and the `.json` Stage-3
> marker. It never unlinks an image artifact, so no `IsADirectoryError` is possible.
>
> **But this site must now clear BOTH the token and the raw `.npy`** (ledger FLOW-18 /
> GEN-35). An earlier draft said "both deletions become plain `.json` unlinks under the new
> design" — that drops the raw array. It is the **sixth** `delete_sidecar` site (Task 3.5
> counts them), and it is the one this task owns. Token first, then raw, as everywhere else:
> if Stage 1 subsequently fails, a raw `.npy` left behind here is orphaned permanently.
>
> Adding an `rmtree(store)` would **introduce** behaviour that does not exist today: at its
> two call sites (`_cli_staged_strategy.py:145`, `_cli_staged_slurm_worker.py:141`, both
> immediately before Stage 1) it opens a window where the image is absent, whereas today the
> previous HDF survives until Stage 1's atomic replace — and it removes the only fallback if
> Stage 1 then fails. Stage 1's promote already replaces the store atomically. Recorded as
> OPEN-QUESTIONS **D13**.

- [ ] **Step 4: Point `zarr_world` at the real artifacts and re-run**

Switch the `zarr_world` fixture to build stores + tokens, and re-run:

```bash
uv run pytest tests/unit/cli/test_staged_resume_parity.py tests/unit/cli/test_staged_resume.py -q
```

Expected: all PASS. A failure names the exact `(layer, markers, work_id, artifacts)`
combination that diverged.

- [ ] **Step 5: Prove the parity test catches the three known defects**

Apply each defect in turn, confirm the parity test fails, then revert:

1. Make `valid_staged_store` require `objmap` to be non-zeros → Stage 1 stores classify
   `"stage1"` forever.
2. Replace `stage2_token_exists` with a durable `ome.labels` probe → the
   `not markers_required` `"complete"` branch never fires.
3. Delete Stage 3's re-promote (Task 3.3) → parity holds but
   `test_stage3_publishes_the_post_refined_objmap` fails; note in the commit that the
   third defect is caught by that test, not this one.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_resume.py tests/unit/cli tests/_legacy_staged_resume.py
git commit -m "feat(cli): port the staged resume classifier to the store

Signature and return values are unchanged; only the artifact probes move.
A differential test enumerates every (process_only_layer, markers_required,
expected_work_id, artifacts) combination the classifier distinguishes and
asserts the zarr world agrees with a frozen copy of the HDF classifier --
freezing it is what makes the comparison mean anything. Verified to fail
under both resume defects the spec's review found. The artifact axis
includes the per-image completion marker, without which branch 1 is never
exercised and the test passes while production breaks (see Task 3.8)."
```

---

### Task 3.5: Staged strategy, controller, SLURM worker, orchestration; delete the sidecar

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py` (lines 22, 33, 37–38, 89, 93, 140,
  142, 171–182, 219–246, 328–336)
- Modify: `src/phenotypic/_cli/_cli_staged_slurm_worker.py` (lines 19, 28, 45–46, 128–134,
  196, 223–227, 276, 316–317, 332–333, 354–360, 382–383, 409)
- Modify: `src/phenotypic/_cli/_cli_staged_controller.py`, `_cli_staged_orchestration.py`
- Delete: `src/phenotypic/_cli/_cli_sidecar.py`, `tests/unit/cli/test_cli_sidecar.py`
- Test: `tests/unit/cli/test_staged_routing.py` (extend),
  `tests/unit/cli/test_staged_controller.py` (extend)
- **Test: `tests/integration/cli/test_staged_gpu_local.py` (969 lines) — port it here**, in
  the same commit that deletes `_cli_sidecar.py`. It is the only file this phase touches that
  becomes an **`ImportError` at collection**, not a failing assertion: line 19 is
  `from phenotypic._cli._cli_sidecar import sidecar_exists, write_sidecar`, and three of this
  phase's exit criteria run it (`tests/integration/cli`, and Task 3.3's and 3.4's re-runs).
  The port is mechanical and has exactly four shapes:
  - line 19 → `from phenotypic._cli._cli_stage2_token import stage2_token_exists, write_stage2_raw`
    (Task 3.2's names);
  - the fourteen `sidecar_exists(out, "ds", <stem>)` assertions (`:131, :140, :164, :224,
    :249, :344, :371, :758, :793, :898, :949, :965, :966`) → `stage2_token_exists(...)`. Where
    the assertion is proving Stage 3 *can still replay*, pair it with
    `stage2_raw_path(...).is_file()`; where it is proving cleanup happened
    (`:140` "mandatory cleanup", `:371`, `:793`), assert **both** are gone — the token-and-raw
    pairing this task introduces is otherwise untested end to end.
  - `write_sidecar(out, "ds", "done", np.zeros((2, 2)))` at `:779` → `write_stage2_raw(...)`
    plus `publish_stage2_token(...)`, or the test's premise (a Stage-2 result awaiting
    Stage 3) no longer holds.
  - the six `dataset_hdf_dir(out, ...) / "<stem>.h5"` builds (`:47` import, `:121`, `:399`,
    `:460`, `:714`) → `zarr_store_path(out, ds, stem)`, with `.is_file()` becoming
    `.is_dir()`. `:714`'s `h5py.File(..., "r+")` corruption injection becomes a write into the
    store's `zarr.json`.
- **Docs: `src/phenotypic/_cli/CLAUDE.md` (the staged-GPU sidecar prose, 15 mentions),
  root `CLAUDE.md` (8), and `docs/source/how_to/pages/gpu_detection_setup.md` (10)** — the
  sidecar concept dies in this task, and these three are the agent- and user-facing
  descriptions of it. Rewrite them against the Task 3.2 vocabulary: the Stage-2 **signal** is
  now a retained raw `.npy` under `.phenotypic/progress/stage2_raw/` plus a **consumable
  token**, and Stage 2 does not write into the store. This is not deferrable to Phase 6 —
  Phases 4 and 5 run in parallel with this one and their executors read `_cli/CLAUDE.md`, and
  this phase's own exit grep covers it.
- **Test: `tests/unit/test_docs_staged_cli.py`** — it pins the prose above:
  `test_claude_md_documents_local_staged_gpu` asserts `"sidecar" in txt.lower()` (line 11) and
  `test_how_to_documents_local_staged_gpu` asserts the same of `gpu_detection_setup.md`
  (line 18). Both go red the moment the word is removed. Re-point both at the new vocabulary
  (`"stage2_raw"` / `"token"`), keeping the `"stage"` and `GpuDetector` assertions and the
  `--gpu-*` flag assertions untouched.

**Constraints specific to this task:**
- `_cli_staged_strategy.py:328` is `--mode process --layer objmap`: it merges the Stage-2
  result and exports, then deletes the token. The merge reads **`load_stage2_raw`**, and the
  **token deletion must stay**, or a subsequent full run misclassifies.

  > **Corrected (ledger FLOW-16).** An earlier draft read *"With the objmap now in the store,
  > the merge is a store read."* Under the round-2 design Stage 2 never writes the store, so
  > the store's objmap at this point is Stage 1's zeros — an executor following that
  > instruction literally makes `--mode process --layer objmap` export an **all-zeros PNG for
  > every image**, silently. The existing source shape is already right: `:363` does
  > `load_sidecar(...)` and `:365` `_write_object_output(image, sidecar)`; the port swaps
  > `load_sidecar` for `load_stage2_raw` and changes nothing else. The task's own test checks
  > only that the token is consumed, so it would not have caught this — assert the exported
  > objmap is non-zero.
- ⚠️ **…and that path must not leave raw detector output published forever.**
  `_cli_staged_strategy.py:360-382` applies `_write_object_output`, writes the exported
  layer, deletes the signal, and **never re-saves the image**. Today the residue is Stage 1's
  zeros inside a non-user-facing `.h5`. Under this design the residue is Stage 2's **raw**
  objmap — pre-`drop_frame_background`, pre-relabel, possibly one giant background label
  covering the plate — sitting in a first-class NGFF label image that napari and Vizarr will
  render. Either re-promote the store after the export or restore the zeros objmap.
  **D11 dissolved when Stage 2 stopped writing into the store**, and it dissolved
  *completely*: `_write_object_output` here mutates only the in-memory `image`, the store is
  never re-promoted on this path, and the residue left behind is Stage 1's zeros — exactly
  what the HDF path leaves today. **There is nothing to restore and nothing to re-promote.**

  > **Corrected (ledger FLOW-30).** An earlier draft opened *"Either re-promote the store
  > after the export or restore the zeros objmap"*, conceded five lines later that D11 had
  > dissolved, and then still prescribed `restore-or-re-promote → publish → delete token →
  > delete raw`. The executor could not tell which half was operative. The first half is
  > withdrawn.

  What remains is the **ordering** (ledger **FLOW-6**): `_publish_local_image_success` runs at
  `_cli_staged_strategy.py:350-359` and the signal delete at `:382`, so the sequence is

  > `_publish_local_image_success` → delete token → delete raw

  Any store write placed **after** the marker publish would rewrite `zarr.json` and invalidate
  the store descriptor the instant it was written — the same production break Task 3.8 exists
  to prevent. That hazard is why no store write belongs on this path at all. Assert
  `valid_image_success` is `True` immediately after the export.
- ⚠️ **The run-start sweep must not delete a live writer's `.part`.** The uuid identifies the
  *attempt*, not whether its process is alive, and the staged SLURM engine explicitly
  assumes stale workers can still be running — that is what `assert_active_epoch`
  (`_cli_staged_slurm_worker.py:346-348`, `_cli_staged_orchestration.py:679`) exists for. A
  recovery controller sweeping while a prior-epoch task is mid-write into its `.part` would
  `rmtree` under it. Gate the sweep on age (mtime older than this run's epoch start) or on a
  lifecycle epoch recorded inside the `.part`. Recorded as OPEN-QUESTIONS **D14**.
- **Delete `clear_stage2_sidecars`** (`_cli_staged_orchestration.py:661-674`, called from
  `phenotypicCLI.py:1590` on `--restart`). It globs `results/*/objmap/*.npy` and becomes a
  permanent no-op. Not a correctness hole — `clear_machine_state` on the same path wipes
  `.phenotypic/`, where the new token lives — but leaving a no-op named after a deleted
  concept is how the next reader concludes sidecars still exist.
- `_cli_staged_slurm_worker.py:409` deletes the token on the work-id path. That is the
  counterpart to Task 3.3's preserved `work_id is None` guard; both must remain.
- **`classify_staged_image`'s token-present/raw-missing branch is Task 3.4's**, not this
  task's — `_cli_staged_resume.py` is in 3.4's `Files:`, not ours, and 3.4 executes first.
  The probe fix below is the half that belongs here; the classifier half must already be in
  place for it to route anywhere.

  > **Relocated (ledger C4).** An earlier draft put the classifier instruction in *this*
  > task while Task 3.4 shipped the unpatched `return "stage3" if stage2_done else "stage2"`
  > — so an executor would build the defect in 3.4, and only later read that it should not
  > have. The two tasks are in different execution clusters, which makes that unrecoverable
  > rather than merely awkward. This is the same wrong-task class as FLOW-38/FLOW-39,
  > recurring on the very finding that named it.
- **Stage 3's prereq probe must test BOTH the token and the raw array, at ALL FIVE sites**
  (ledger **FLOW-17**, extended by **M7**). The same `sidecar_exists`-only gate appears at
  `_cli_staged_strategy.py:175`, `:219` and `:353`, `_cli_staged_slurm_worker.py:196`, and
  `_cli_staged_controller.py:81`. An earlier draft named only the SLURM worker; fixing one
  leaves the other four routing a token-present/raw-missing image into Stage 3, where
  `load_stage2_raw` raises inside `stage_event` and is reported as a terminal *scientific*
  failure. Combined with the classifier branch (Task 3.4), such an image is otherwise
  permanently unreachable.
  `_cli_staged_slurm_worker.py:352-360` gates Stage 3 on `sidecar_exists`, and Task 3.4 maps
  that to `stage2_token_exists`. But the token is now only a *flag*; Stage 3's actual **input**
  is the `.npy`. A token-present/raw-missing state (a partial cleanup, a truncated copy) raises
  an uncaught `FileNotFoundError` from `load_stage2_raw` inside `stage_event`, which reports it
  as a terminal **scientific** failure instead of calling `emit_missing_prereq`. Probe
  `stage2_token_exists(...) and stage2_raw_path(...).is_file()`.
- **Every site that deleted the sidecar must now delete BOTH the token and the raw array.**
  There are **six**, all verified present: `_cli_staged_workers.py:258` (guarded),
  `_cli_staged_strategy.py:246` (local Stage 3, unconditional) and `:382`
  (`_export_objmap_layer`), `_cli_staged_slurm_worker.py:409`,
  `_cli_staged_resume.py:364` (`reconcile_stage3_publications` — **Task 3.4 owns both
  `_cli_staged_resume.py` sites and carries the instruction**, ledger M6), and
  `_cli_staged_resume.py:318` inside `clear_downstream_artifacts_for_stage1`.

  > **Miscount corrected (ledger FLOW-18).** An earlier draft said "five". The sixth site,
  > `_cli_staged_resume.py:318`, belongs to **Task 3.4**, which owns that file — the
  > instruction to clear both there lives in Task 3.4's `clear_downstream_artifacts_for_stage1`
  > blockquote, not here (ledger GEN-35 / FLOW-39). Counted here so the number is right.

  Deleting the token and
  leaving the raw array orphans a `.npy` per image; deleting the raw array and leaving the
  token makes the next Stage 3 replay into a `FileNotFoundError`. Delete the **token
  first** at every site, for the reason given in Task 3.3.
- The run start must **log the resolved durability mode** (`ngff_.describe_durability`) —
  a required mitigation, not a nicety.
- The run start must **sweep orphaned `.part` / `.trash` directories** and log the count.
- ⚠️ **Both belong on every execution path, not only the staged-GPU one.** Spec §3.7 and
  §3.2 are unqualified, and a plain `--mode full` CPU run writes stores through the same
  promote. Wiring them into `_cli_staged_strategy`'s setup alone leaves the common case with
  no durability log and no sweep. Put both in the shared run-setup that
  `_cli_execution_strategies.create_execution_strategy` dispatches through, so every strategy
  inherits them. Recorded as OPEN-QUESTIONS **G6/P21**.
- ⚠️ **The sweep runs from the controller, once, before any worker is submitted — never
  from a worker's own start-up.** A uuid identifies the attempt, not whether its process is
  alive; under a SLURM array the tasks share one output root and start at different times, so
  a per-worker sweep would `rmtree` the `.part` directories its siblings are actively
  filling. `ngff_.sweep_orphan_parts` additionally refuses anything younger than
  `SWEEP_MIN_AGE_SECONDS`, but that age guard is a backstop, not a licence to call it from a
  worker. Recorded as OPEN-QUESTIONS **B6/P16**.
- Delete `_cli_sidecar.py` only after `grep -rn "_cli_sidecar\|sidecar_exists\|write_sidecar\|load_sidecar\|delete_sidecar" src/ tests/` is empty.

- [ ] **Step 1: Write the failing tests**

> **Corrected (wrong-symbol sweep).** An earlier draft imported `main` from
> `phenotypic.phenotypicCLI` and took a `cli_runner` fixture. Neither exists. The click
> command is **`phenotypic_cli`** (`phenotypicCLI.py:1146`; both `__main__.py` and
> `tests/unit/cli/test_cli_mode_contract.py:9` import that name), and there is no
> `cli_runner` fixture anywhere in `tests/` — every existing CLI test constructs
> `CliRunner()` inline at the call site. Both corrections are applied throughout this plan.

Append to `tests/unit/cli/test_staged_routing.py`:

```python
def test_run_start_logs_the_resolved_durability_mode(staged_strategy, caplog) -> None:
    """The same command carries different guarantees in different places."""
    staged_strategy.prepare()
    assert any("durable writes:" in record.message for record in caplog.records)


def test_controller_sweeps_stale_orphaned_part_directories(staged_strategy, tmp_path) -> None:
    import os
    import time

    from phenotypic.sdk_ import dataset_zarr_dir

    orphan = dataset_zarr_dir(tmp_path, "ds") / ".img.ome.zarr.deadbeef.part"
    orphan.mkdir(parents=True)
    old = time.time() - 24 * 60 * 60
    os.utime(orphan, (old, old))
    staged_strategy.prepare()
    assert not orphan.exists()


def test_the_sweep_spares_a_recent_part(staged_strategy, tmp_path) -> None:
    """A uuid says nothing about liveness; under a SLURM array a sibling task
    may be mid-write into exactly this directory."""
    from phenotypic.sdk_ import dataset_zarr_dir

    live = dataset_zarr_dir(tmp_path, "ds") / ".img.ome.zarr.cafef00d.part"
    live.mkdir(parents=True)
    staged_strategy.prepare()
    assert live.is_dir()


def test_workers_never_sweep(staged_run, tmp_path) -> None:
    """Only the controller sweeps, and only before submitting anything."""
    from phenotypic.sdk_ import dataset_zarr_dir

    live = dataset_zarr_dir(tmp_path, "ds") / ".other.ome.zarr.deadbeef.part"
    live.mkdir(parents=True)
    staged_run.run_stage1()
    assert live.is_dir()


def test_a_plain_full_run_also_logs_durability_and_sweeps(tiny_run, caplog) -> None:
    """Spec §3.7 and §3.2 are unqualified; the CPU path uses the same promote."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    CliRunner().invoke(phenotypic_cli, tiny_run.args())  # --mode full, no GpuDetector
    assert any("durable writes:" in record.message for record in caplog.records)


def test_sidecar_module_is_gone() -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phenotypic._cli._cli_sidecar")


def test_process_only_objmap_still_consumes_the_token(staged_run) -> None:
    from phenotypic._cli._cli_stage2_token import stage2_token_exists

    staged_run.run_stage1()
    staged_run.run_stage2()
    staged_run.export_objmap_layer()
    assert stage2_token_exists(staged_run.output_dir, "ds", "img") is False
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/cli/test_staged_routing.py -v
```

- [ ] **Step 3: Port the four modules and delete the sidecar**

Mechanical, one import block and one path expression at a time. After each file:

```bash
uv run pytest tests/unit/cli -q
```

Then:

```bash
grep -rn "_cli_sidecar\|sidecar_exists\|write_sidecar\|load_sidecar\|delete_sidecar" src/ tests/
git rm src/phenotypic/_cli/_cli_sidecar.py tests/unit/cli/test_cli_sidecar.py
```

- [ ] **Step 4: Run the CLI suite**

```bash
uv run pytest tests/unit/cli tests/integration/cli -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli tests/integration/cli
git commit -m "refactor(cli): route the staged engine through the store; drop the sidecar

The run start now logs the resolved durability mode and sweeps orphaned
.part/.trash directories by uuid. --mode process --layer objmap still
consumes the Stage-2 token after export, and the SLURM worker still deletes
it on the work-id path; both are load-bearing for resume classification.
_cli_sidecar.py is deleted only after every caller moved."
```

---

### Task 3.6: Directory scanning, recompile scripts, single-pass, tune

> ⚠️ **Every `.stem` on a path that is now a store must become `store_stem`** (Phase 2
> Task 2.1, ledger **C5**). `Path("img.ome.zarr").stem` is `"img.ome"` — `Path.stem` strips
> only the final suffix, and `.ome.zarr` is two. This task is where store directories start
> reaching those consumers, so it is where the bug would land.
>
> The five sites: `_cli_process_single.py:244`, `:250`; `_cli_execution_strategies.py:165`,
> `:168`, `:173`, `:184`; `_cli_staged_resume.py:178`. Nothing raises if you miss one — the
> run writes `img.ome.parquet`, publishes a marker keyed `"img.ome"`, and then looks for
> `img.ome.ome.zarr`, finds nothing, and reprocesses every image on every run.

> **Corrected by execution (C11, 2026-08-20).** Five of this task's claims were wrong; the
> site list was both over- and under-counted. Each below was checked with `grep` plus a
> read of the enclosing function, and the correction is what shipped.
>
> 1. **Three of the seven named `store_stem` sites are NOT store paths and must keep
>    `Path.stem`.** `_cli_execution_strategies.py:164,168,174,185` are inside
>    `_publish_local_image_success`, whose `image_path` comes from `dataset.images` on the
>    **forward** local path — an input `.tif`. `_cli_staged_resume.py:165` is
>    `classify_staged_image`, whose `image` is likewise an input image from
>    `build_staged_resume_plan`. Measure mode never reaches either: it dispatches to
>    `_process_single_local_measure`, which publishes no marker. `store_stem` **raises** on a
>    non-store path, so following the plan here would have turned every local forward run
>    into a hard `ValueError`.
> 2. **Six store-path sites the plan does not name.** `phenotypicCLI.py`
>    `_regenerate_missing_overlays` (the overlay-present probe, the overlay write, and the
>    failure log line), `_recompile_dataset_image_names`, `_discover_recompile_dataset_names`,
>    and `_cli_recompile_worker.py:_run_overlay_task`. All six consume `scan_store_outputs`
>    or `results/<ds>/zarr/` directly. Net: **8 real sites, not 5** — 2 of the plan's 5 kept.
> 3. **Both loaders needed porting, which the plan does not mention.**
>    `process_single_hdf_measure_core` called `image_cls.load_hdf5`, and its two callers each
>    hand-rolled an `h5py.File(...)` probe of `phenotypic_class`. All three are replaced by
>    `load_image_from_store` (Phase 2), which owns that dispatch. The function is renamed
>    `process_single_store_measure_core`; it has no test or plan references.
> 4. **`_cli_process_single.py:640` had to move to `image_data_artifact` in this task, not
>    Task 3.8.** Once `process_single_image_core` writes a store, that hard-coded
>    `"hdf"` artifact names a file nothing writes, and `publish_image_success` resolves
>    `strict=True` — so every standalone worker (i.e. every SLURM array task) would die with
>    `FileNotFoundError` *after* completing its work. Same minimal closure C10 applied to its
>    own four call sites; it consumes none of Task 3.8's `kind`-dispatch design.
> 5. **`_discover_recompile_dataset_names` / `_recompile_dataset_image_names` keep their
>    `DIR_HDF` branch** as a legacy last resort until Phase 6. Replacing it outright made an
>    unconverted bundle undiscoverable to the recompile run that authorizes its metadata
>    migration (`test_hdf_only_migration_uses_slurm_without_recompile_publication`).
>
> Two further corrections outside the `.stem` list:
>
> - **`tests/unit/cli/test_directory_scanner.py` does not exist.** The only scanner test file
>   is `test_cli_directory_scanner_dotfiles.py`, which covers `scan_directory_structure`, not
>   the output scan. The file was created, not extended.
> - **`tests/integration/cli/test_cli_hdf_output.py` belongs to this task.** The plan's
>   Task 3.6 file list omits it; [`README.md`](README.md)'s existing-test inventory assigns it
>   to Phase 3 ("becomes `test_cli_store_output.py` wholesale"), and Task 3.6 is what breaks
>   it. Ported here. Likewise `test_cli_v2.py`, and `test_cli_recompile{,_slurm}.py` /
>   `test_cli_recompile_metadata_migration_slurm.py` — which that inventory assigns to
>   **Phase 5**, but which this task breaks first.
> - **`OutputManager.create_structure` provisioned `hdf/`.** Left alone it puts an empty
>   `hdf/` in every output tree, contradicting the `zarr/` layout this task's README
>   generator documents — and the forward-layout whitelist test asserts on exactly that set.
>   Changed to `DIR_ZARR`.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_directory_scanner.py` (`scan_hdf_outputs` line 173,
  glob at line 217)
- Modify: `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py` (lines 217–231)
- Modify: `src/phenotypic/_cli/_cli_process_single.py`, `_cli_execution_strategies.py`
- Modify: `src/phenotypic/tune/_tune_cli/_run.py`
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py` (lines 113, 131)
- Test: `tests/unit/cli/test_directory_scanner.py` (extend)

**Constraints specific to this task:**
- `scan_hdf_outputs` becomes `scan_store_outputs`; the glob `hdf_dir.glob("*.h5")` becomes
  `zarr_dir.glob(f"*{STORE_SUFFIX}")` and must be **non-recursive** and match
  **directories**. A store contains files, so a recursive glob or an `is_file()` filter
  finds nothing.
- The AppleDouble guard at line 218 (`not p.name.startswith(".")`) must be **kept**: it now
  also excludes the `.part` and `.trash` directories, which is exactly right.
- `_cli_readme_generator.py` documents the layout to users: `hdf/` → `zarr/`, `.h5` →
  `.ome.zarr`, `Image.load_hdf5` → `Image.load_zarr`, and add a line saying the store is
  readable by napari, QuPath, and Vizarr without a PhenoTypic install — that is a headline
  user-facing benefit of this change.

- [x] **Step 1: Write the failing test**

```python
def test_scan_finds_store_directories_not_files(tmp_path) -> None:
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs
    from phenotypic.sdk_ import zarr_store_path

    for stem in ("a", "b"):
        store = zarr_store_path(tmp_path, "ds", stem)
        store.mkdir(parents=True)
        (store / "zarr.json").write_text("{}", encoding="utf-8")
    datasets = scan_store_outputs(tmp_path)
    assert [p.name for p in datasets[0].images] == ["a.ome.zarr", "b.ome.zarr"]


def test_scan_is_non_recursive(tmp_path) -> None:
    """A recursive scan walks INTO every store: 400k stat calls at 10k images."""
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs
    from phenotypic.sdk_ import zarr_store_path

    store = zarr_store_path(tmp_path, "ds", "a")
    (store / "gray" / "0").mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    (store / "gray" / "0" / "nested.ome.zarr").mkdir()
    assert len(scan_store_outputs(tmp_path)[0].images) == 1


def test_scan_skips_part_and_trash_directories(tmp_path) -> None:
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs
    from phenotypic.sdk_ import dataset_zarr_dir, zarr_store_path

    store = zarr_store_path(tmp_path, "ds", "a")
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    (dataset_zarr_dir(tmp_path, "ds") / ".a.ome.zarr.deadbeef.part").mkdir()
    assert len(scan_store_outputs(tmp_path)[0].images) == 1
```

- [x] **Step 2: Run to verify failure, then port each file**

```bash
uv run pytest tests/unit/cli/test_directory_scanner.py -v
```

- [x] **Step 3: Run the full CLI + tune suites**

```bash
uv run pytest tests/unit/cli tests/unit/tune tests/integration/cli -q
```

- [x] **Step 4: Commit**

```bash
git add src/phenotypic/_cli src/phenotypic/tune tests/unit/cli
git commit -m "refactor(cli): scan for store directories, not .h5 files

The glob is non-recursive and matches directories: a store contains files,
so a recursive scan walks into all ~40 of them (400k stat calls at 10k
images) and an is_file() filter finds nothing at all. The AppleDouble
dotfile guard is kept and now also excludes .part/.trash. The README
generator documents the new layout and that the output is readable by
napari, QuPath, and Vizarr without a PhenoTypic install."
```

---

### Task 3.7: The `--durable-writes` CLI option

**Files:**
- Modify: `src/phenotypic/phenotypicCLI.py` (option block beside `--mode` at line 942; the
  module docstring's option documentation)
- Modify: `src/phenotypic/_cli/_cli_output_manager.py` — carry the resolved tri-state on the
  `OutputManager` (a `durable: bool | None` field set by `from_config`, defaulting to
  `None` = auto-detect) and pass it into `save_image_store`'s `durable=` argument (Task 3.1)
- Modify: `src/phenotypic/_cli/_cli_process_single.py` (`:183`, the bare
  `save_image_store(image, dataset_name, image_stem)`)
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py` (`:125` Stage 1 and `:225` Stage 3 —
  the only other two `save_image_store` call sites)
- Modify: `src/phenotypic/_cli/_cli_staged_slurm_worker.py` — a `--durable-writes` /
  `--no-durable-writes` argparse pair beside `:427-429`, threaded into the two
  `OutputManager.from_config` calls at `:145` and `:282`, and emitted by whatever submits the
  worker (`_cli_staged_slurm.py`)
- Modify: `src/phenotypic/phenotypicCLI.py` — the four `OutputManager.from_config` sites
  (`:360`, `:1934`, `:2160`, `:2525`)
- Test: `tests/unit/cli/test_cli_store_options.py` (create)

> **Corrected (missing-owner review, 2026-08-19).** An earlier draft named
> `src/phenotypic/_cli/_cli_staged_strategy.py` as the module to thread the value through.
> `grep -n 'save_image' src/phenotypic/_cli/_cli_staged_strategy.py` returns **nothing** — the
> strategy only *calls the stage cores*; the writes live in `_cli_staged_workers.py:125,225`.
> As written, the flag would have reached the single-pass CPU path and **never** the staged or
> SLURM path, which is precisely where durability matters (Task 3.5's own constraint says both
> the durability log and the sweep "belong on every execution path"). The three
> `save_image_store` call sites are the same three Task 3.1 enumerates; keep the two lists in
> step.

**The SLURM worker is a fresh process, so an explicit flag must be transported, not
re-derived.** `ngff_.durable_writes_enabled` auto-detects SLURM, so an *unset* flag resolves
correctly in a worker on its own. But `--no-durable-writes` — the case a user reaches for
precisely because they are on a fast local scratch inside a job — is a value that exists only
in the submitting process. If it is not passed down the argparse surface at
`_cli_staged_slurm_worker.py:418-429`, every staged SLURM worker silently re-enables fsync
and the flag appears to do nothing on the one execution path where it costs the most. Assert
this in the test: a worker invoked with `--no-durable-writes` builds an `OutputManager` whose
resolved mode is non-durable **while `SLURM_JOB_ID` is set**.

**Interfaces:**
- Consumes: `ngff_.durable_writes_enabled`, `ngff_.describe_durability`,
  `OutputManager.save_image_store`.
- Produces: one new top-level CLI option and its config plumbing.

**Why this task exists:** spec §3.7 requires `--durable-writes` / `--no-durable-writes`, but
the flag appears in no section that enumerates CLI options, so it had no owning task and
would have shipped unimplemented. Recorded as OPEN-QUESTIONS **P12**.

**`--pyramid-levels` is descoped.** Spec §1.3 also introduces `--pyramid-levels auto|N`;
that lever is **not** implemented. The pyramid depth is a pure function of the level-0 shape
(`ngff_.pyramid_level_count`), which dissolves OPEN-QUESTIONS **P3** — with no user lever,
two stores in one tree cannot disagree, so `valid_staged_store` needs no level check and a
resumed run cannot produce mixed geometry. A single-level store is still reachable
internally, via the private `levels=` argument used by `save_intermediate_zarr`. The lever
can be added later as its own change; the spec's §1.3 should record it as deferred.

**Constraints specific to this task:**
- `--durable-writes` / `--no-durable-writes` is a **tri-state**: unset means auto-detect. A
  plain `click.option(..., is_flag=True)` collapses that to two states and silently loses
  the SLURM detection. Use a paired `--durable-writes/--no-durable-writes` option with
  `default=None`.
- The resolved durability mode is logged at run start (already required by Task 3.5); this
  task is what gives that log line something other than the auto-detection to report.
- The option applies to `--mode full`, `--mode process`, and `--mode measure`, and is
  **rejected** on `--mode recompile` and `--mode migrate`, which do not write image stores
  from a pipeline. Reuse the existing per-mode rejection pattern at lines 1231–1244.

- [ ] **Step 1: Write the failing test**

```python
"""The durability flag must be genuinely tri-state."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli


def test_durable_writes_is_tri_state(tiny_run, monkeypatch, caplog) -> None:
    """Unset must mean auto-detect, not 'off'. A plain is_flag loses that."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    CliRunner().invoke(phenotypic_cli, tiny_run.args())
    assert any("durable writes: off (local)" in r.message for r in caplog.records)

    caplog.clear()
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    CliRunner().invoke(phenotypic_cli, tiny_run.args())
    assert any("durable writes: on (SLURM)" in r.message for r in caplog.records)

    caplog.clear()
    CliRunner().invoke(phenotypic_cli, [*tiny_run.args(), "--no-durable-writes"])
    assert any(
        "durable writes: off (--no-durable-writes)" in r.message for r in caplog.records
    )

    caplog.clear()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    CliRunner().invoke(phenotypic_cli, [*tiny_run.args(), "--durable-writes"])
    assert any(
        "durable writes: on (--durable-writes)" in r.message for r in caplog.records
    )


def test_durable_writes_is_rejected_on_recompile_and_migrate(tiny_run) -> None:
    for mode in ("recompile", "migrate"):
        result = CliRunner().invoke(
            phenotypic_cli,
            ["--mode", mode, "--output", str(tiny_run.output_dir), "--durable-writes"],
        )
        assert result.exit_code != 0
        assert "--durable-writes" in result.output


def test_no_pyramid_levels_option_exists() -> None:
    """Descoped: the pyramid depth is a pure function of shape (P3)."""
    result = CliRunner().invoke(phenotypic_cli, ["--help"])
    assert "--pyramid-levels" not in result.output


def test_pyramid_depth_is_derived_not_configured(tiny_run) -> None:
    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

    CliRunner().invoke(phenotypic_cli, tiny_run.args())
    store = tiny_run.store("ds", "img")
    shape = tiny_run.image_shape
    assert read_phenotypic_attributes(store)[PhenotypicAttr.PYRAMID]["levels"] == (
        ngff_.pyramid_level_count(*shape)
    )
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/cli/test_cli_store_options.py -v
```

Expected: `Error: No such option: --durable-writes`.

- [ ] **Step 3: Add the option**

```python
@click.option(
    "--durable-writes/--no-durable-writes",
    "durable_writes",
    default=None,
    help=(
        "fsync each image store before promoting it. Unset auto-detects: on "
        "under SLURM, off locally. The resolved mode is logged at run start."
    ),
)
```

Reject it on the two modes that do not write stores from a pipeline, mirroring the existing
guards:

```python
    if durable_writes is not None and cli_mode in {"recompile", "migrate"}:
        raise click.UsageError(
            f"--durable-writes is not accepted with --mode {cli_mode}; that mode "
            "does not write image stores from a pipeline."
        )
```

Thread the value through the run config to every `save_image_store` call in
`_cli_staged_strategy.py` and `_cli_process_single.py`.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/cli/test_cli_store_options.py tests/unit/cli -v
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/phenotypicCLI.py src/phenotypic/_cli tests/unit/cli/test_cli_store_options.py
git commit -m "feat(cli): add --durable-writes

A spec §3.7 requirement that no section enumerated as a CLI flag, so it had
no owning task. It is tri-state: unset means auto-detect, and a plain
is_flag would collapse that to 'off' and silently lose the SLURM detection.

--pyramid-levels (spec §1.3) is deliberately NOT added. The pyramid depth
is a pure function of the level-0 shape, which means two stores in one tree
can never disagree -- so valid_staged_store needs no level check and a
resumed run cannot produce mixed geometry. The lever can land later as its
own change."
```

---

### Task 3.8: Per-image completion markers must describe a store, not a file

**Files:**
- Modify: `src/phenotypic/_cli/_cli_completion.py` (`SUCCESS_MARKER_VERSION` line 26,
  `_sha256` lines 29–34, `publish_image_success` line 36, `valid_image_success` lines 117–130,
  `refresh_success_markers_after_metadata_migration` lines 136–155)
- Modify the ONE remaining `"hdf"` artifact declaration: `phenotypicCLI.py:405`

> **Four of the five are already done, and the helper to extend already exists (C10).**
> Task 3.5 could not be green without them: Task 3.3 stopped Stage 1 writing `.h5` while
> the marker still declared one, and `publish_image_success` does
> `artifact.resolve(strict=True)`, so every staged full-mode image died at Stage 3 with
> `FileNotFoundError` on a file nothing writes any more. C10 closed it with
>
> ```python
> def image_data_artifact(output_dir, output_manager, dataset, image_stem) -> tuple[str, Path]:
>     """Return the ``(key, path)`` of the per-image data artifact to certify."""
> ```
>
> in `_cli_completion.py`, returning `("store", <store>/zarr.json)` when a store exists and
> `("hdf", ....h5)` otherwise. It is wired at `_cli_staged_slurm_worker.py:338` and `:392`,
> `_cli_process_single.py:623`, and `_cli_execution_strategies.py:160`.
>
> **C11 found a further site the plan did name but C10 had not reached** --
> `_cli_process_single.py`'s *second* publisher, the standalone
> `phenotypic-process-single` SLURM worker. It is routed now. Nothing in the repo
> covered that publish path: it survived C11's mutation pass until a test spying
> `publish_image_success` through the real CLI was added. Two publishers in one file
> is why "five sites" kept coming up short -- count publishers, not files.
>
> **Extend that helper; do not add a parallel path and do not re-plumb the four callers.**
> `zarr.json` is a regular file, so the existing `{"size", "sha256"}` descriptor and
> `valid_image_success` work unchanged — no `SUCCESS_MARKER_VERSION` bump is required by
> the artifact change alone, and none of this task's *fingerprint* design is consumed.
>
> **The fifth site is a live break, and it is the reason this task still has teeth.**
> `phenotypicCLI.py:405` sits in `_migrate_legacy_success_evidence`, which mints success
> markers for runs that have completion evidence but no marker — and its evidence test at
> `:380` includes `stage3_completion_exists`, a **store-era** signal. So the path fires on a
> staged run interrupted between its Stage-3 marker and its success marker, declares an
> `.h5` that the store-era run never wrote, and `resolve(strict=True)` raises. Route it
> through `image_data_artifact` like the others; it needs `output_manager`, which is already
> in scope there.
>
> **Whether the `"hdf"` fallback branch survives at all is Task 3.6's answer, not this
> task's assumption** — `_publish_local_image_success` is shared with the non-staged
> `LocalParallelStrategy`. Read C11's report before deciding to delete it.
- Test: `tests/unit/cli/test_cli_completion_store.py` (create)

**Why this task exists — this is a silent production break, not a refactor.**

`grep -rn 'publish_image_success|valid_image_success|_cli_completion|SUCCESS_MARKER_VERSION'`
over the spec and the entire plan directory returns **nothing**. The surface was uncosted
until an independent data-flow review found it. Both halves fail on a directory:

```python
def _sha256(path: Path) -> str:          # _cli_completion.py:29
    with path.open("rb") as handle:      # IsADirectoryError on a store -- UNCAUGHT
```

```python
if (not artifact.is_file()               # _cli_completion.py:126 -- False for a store
        or artifact.stat().st_size != descriptor.get("size")
        or _sha256(artifact) != descriptor.get("sha256")):
    return False
```

So `publish_image_success` **kills the publishing worker**, and `valid_image_success` makes
`classify_staged_image`'s first branch (`_cli_staged_resume.py:182`) return `"stage3"` for
every already-finished image on the work-id path, forever. Recorded as OPEN-QUESTIONS **D2**.

**Interfaces:**
- Produces: a `kind`-tagged artifact descriptor, and `SUCCESS_MARKER_VERSION = 2`.

**Constraints specific to this task:**
- ⚠️ **`image_data_artifact` already exists — extend it, do not add a parallel path** (C10).
  Task 3.3 stopped Stage 1 writing `.h5`, but the completion marker still declared
  `"hdf": results/<ds>/hdf/<stem>.h5`, and `publish_image_success` does
  `artifact.resolve(strict=True)` — so **every staged full-mode image failed at Stage 3**
  with `FileNotFoundError` on a file nothing writes any more. That is a live production
  break three clusters wide, between Task 3.3 and this task, and Task 3.5 could not be green
  while it stood.

  C10 closed it minimally, in `_cli_completion.py`:
  `image_data_artifact(output_dir, output_manager, dataset, stem) -> (key, path)` returns
  `("store", <store>/zarr.json)` when a store exists and `("hdf", ….h5)` otherwise. Four call
  sites go through it — `_cli_staged_slurm_worker.py:338,392` and
  `_cli_execution_strategies.py:163`.

  It deliberately consumes **none** of this task's design: `zarr.json` is a regular file, so
  the existing `{"size","sha256"}` descriptor and `valid_image_success` work unchanged — no
  `SUCCESS_MARKER_VERSION` bump, no `kind` dispatch. And it is **already the fingerprint this
  task prescribes** for a store descriptor (root `zarr.json`, content-only, relocatable), so
  your work is to change what it returns and add the `kind` tag, not to re-plumb the callers.

  The `"hdf"` fallback must survive until **Task 3.6**: `_publish_local_image_success` is
  shared with the non-staged `LocalParallelStrategy`, which still writes `.h5` via
  `_cli_process_single.py:183`.
- A store descriptor is
  `{"path": <relative>, "kind": "store", "sha256": file_fingerprint(store / "zarr.json")}`.

  Two separate requirements, both load-bearing:

  1. **Key on the root `zarr.json`, not the directory.** A directory fingerprint emits one
     sentinel byte and does not recurse (`_io_constants.py:215-217`), so it is a constant
     function of the path and would validate a store whose contents changed. Same trap as
     OPEN-QUESTIONS **D4/D5**.
  2. **Use content-only `file_fingerprint`, NOT `paths_fingerprint`** (ledger **FLOW-3**).
     `paths_fingerprint` folds the **absolute resolved path** into the digest before the
     contents (`_io_constants.py:196-211`: `name = resolved.as_posix()` when no `root=` is
     given, then `digest.update(encoded_name)`). Every existing file descriptor is
     deliberately **relocatable** — relative path plus content-only `_sha256`
     (`_cli_completion.py:71-75`) — so a path-sensitive store descriptor would make stores
     the one artifact class that breaks when a directory moves.

     That is a regression, and on this system a likely one: `Path.resolve()` follows
     symlinks, so the same tree reached via `/rhome/...` versus `/bigdata/...`, or through
     an automount that differs between a login node and a compute node, hashes differently.
     A SLURM job could invalidate markers a login-node run just wrote. Archiving a run from
     `/scratch` to `/bigdata`, or copying one to share it, does the same.

     The failure is **invisible**: `valid_image_success` catches broadly and returns `False`,
     so the symptom is a silent full re-finalization and every image reprocessing, with no
     message naming the cause. `zarr.json` is a regular file, so `file_fingerprint` applies
     directly and is simpler than the alternative.
- File descriptors keep their existing `{"size", "sha256"}` shape and gain
  `"kind": "file"`. `valid_image_success` dispatches on `kind`, defaulting to `"file"` when
  absent so a marker written by an older version still parses.
- **`SUCCESS_MARKER_VERSION` must be bumped to `2`.** After migration a v1 marker still
  describes the retained `.h5` — **which is exactly the problem.** With `keep_source=True`
  (the default, Task 5.1) that file is still present, still the recorded size, still the
  recorded sha256. So without the bump a v1 marker **validates**, against a stale artifact,
  while the store it should be describing goes entirely unverified. The image reads as
  `complete` on the strength of a file the forward path no longer opens.

  > **This rationale was inverted in three earlier drafts and is corrected here (ledger
  > FLOW-2(a), re-raised as FLOW-23).** Those drafts read *"A v1 marker describes an `.h5`
  > that no longer exists; without the bump those markers are read and fail validation,
  > silently reprocessing every image."* That is the opposite failure — and it describes the
  > `--delete-sources` path, not the default one. The bump is not protecting against
  > over-reprocessing; it is protecting against a **false `complete`**. Task 5.6 states the
  > same correction at its own site; the two now agree.

- `refresh_success_markers_after_metadata_migration` (`:136-155`) exists because rewriting a
  per-image HDF invalidates the marker's `sha256`. **It still needs `kind` dispatch**, because
  `_cli_completion.py`'s per-descriptor loop is shared with `valid_image_success` and will meet
  store descriptors there. Recorded as OPEN-QUESTIONS **D10**.

  > **Two earlier justifications for this bullet were both dead (ledger FLOW-24, then
  > FLOW-36).** The first cited *"Header-only **store** migration (Phase 5 Task 5.5)"* — Task
  > 5.5 was **cut**. The second cited *"`--mode migrate` pass 2 does the same thing to the
  > per-image measurement parquets"* — the parquet rewrite is **pass 1** after the MIG-15
  > inversion, and it now runs *before any store exists*, so no marker holds a store
  > descriptor at the moment the bridge would run. Task 5.3's own blockquote says the bridge
  > is not used by migrate at all.
  >
  > The bridge is retained for the **recompile** path it already serves, and the `kind`
  > dispatch is required by the shared descriptor loop, not by migrate. Do not re-justify it
  > by a migrate pass.

- **The `kind` dispatch must cover the whole comparison block, not just the `is_file()`
  guard** (ledger **FLOW-31**). Reading `_cli_completion.py:242-289` in order: `:262` is the
  `if not artifact.is_file(): raise` everyone notices — then `:265-266` **unconditionally**
  compute `current_sha = _sha256(artifact)` and `current_size = artifact.stat().st_size`, and
  `_sha256` opens its argument as a file (`:29-34`), so it raises `IsADirectoryError` on a
  store. Past that, `:271-278` raises `"Uncertified artifact change"` when the recorded sha
  differs with no receipt, and `:286-289` compares `descriptor.get("size")` — which a store
  descriptor (`{"path", "kind", "sha256"}`) does not carry, so `None != current_size` raises.
  Fixing only the `is_file()` guard leaves migration aborting with `IsADirectoryError` on the
  first converted image. Dispatch on `kind` at the top of the loop body and take a separate
  branch for `"store"`.

- [ ] **Step 1: Write the failing test**

```python
"""Per-image completion markers over a store directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic._cli._cli_completion import (
    SUCCESS_MARKER_VERSION,
    publish_image_success,
    valid_image_success,
)
from phenotypic.sdk_ import zarr_store_path


def test_marker_version_is_bumped() -> None:
    """A v1 marker describes the RETAINED .h5, which still validates.

    keep_source=True is the default, so without the bump a v1 marker passes
    against a stale artifact while the store goes unverified -- a false
    `complete`, not a spurious reprocess. See FLOW-23.
    """
    assert SUCCESS_MARKER_VERSION >= 2


def test_publishing_a_store_artifact_does_not_raise(published_store) -> None:
    """_sha256 opens its argument as a file; on a directory that is fatal."""
    assert published_store.marker.is_file()


def test_a_published_store_validates(published_store) -> None:
    assert valid_image_success(
        published_store.output_dir,
        dataset="ds",
        image_stem="img",
        work_id="w-1",
    ) is True


def test_a_rewritten_store_invalidates_the_marker(published_store) -> None:
    """Keying on the directory instead of zarr.json would miss this."""
    root = published_store.store / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"]["work_id"] = "different"
    root.write_text(json.dumps(payload), encoding="utf-8")
    assert valid_image_success(
        published_store.output_dir, dataset="ds", image_stem="img", work_id="w-1"
    ) is False


def test_a_relocated_output_tree_still_validates(published_store, tmp_path) -> None:
    """Store descriptors must be relocatable, like every file descriptor.

    paths_fingerprint would fold the absolute path into the digest, so moving
    the tree -- or reaching it through a different symlink/automount, which on
    this cluster is routine -- would silently invalidate every marker and
    trigger a full re-finalization with no message saying why (FLOW-3).
    """
    import shutil

    moved = tmp_path / "relocated"
    shutil.copytree(published_store.output_dir, moved)
    assert valid_image_success(
        moved, dataset="ds", image_stem="img", work_id="w-1"
    ) is True


def test_a_deleted_store_invalidates_the_marker(published_store) -> None:
    import shutil

    shutil.rmtree(published_store.store)
    assert valid_image_success(
        published_store.output_dir, dataset="ds", image_stem="img", work_id="w-1"
    ) is False


def test_a_file_descriptor_without_kind_still_validates(legacy_file_marker) -> None:
    """Defaulting kind to 'file' keeps older markers parseable."""
    assert valid_image_success(**legacy_file_marker) is True


#: Modules where a ``"hdf":`` key is correct and must survive this phase.
#: Everything outside this set is a per-image artifact declaration that Step 3
#: ports to ``"store":``.
_KEEPS_AN_HDF_KEY = {
    # `TargetKind == "hdf"` -- legacy-tree metadata migration. Retained by
    # decision D10 (reachable for legacy trees, not dead); Task 6.4 records the
    # reasoning in this module's docstring. 7 lines.
    "sdk_/_metadata_migration.py",
    # OutputManager's legacy `save_layers` / `extensions` dict keys, their
    # docstring, and the `layer == "hdf"` extension dispatch -- the HDF writer
    # itself is kept until Phase 6 (Task 3.1). 4 lines; Phase 6 Task 6.3 removes
    # them and this allowlist entry with them.
    "_cli/_cli_output_manager.py",
}


def test_every_hdf_artifact_declaration_is_ported() -> None:
    """The five sites that declare the per-image image-state artifact.

    Scoped, not a bare zero-hit sweep: ``"hdf":`` appears **17** times under
    ``src/phenotypic`` and 11 of them are correct. The five this task ports are
    phenotypicCLI.py:400, _cli_staged_slurm_worker.py:332 and :382,
    _cli_process_single.py:640, and _cli_execution_strategies.py:167. The
    twelfth, ``gui/builder/_preview_cache.py:208``, is **not** allowlisted --
    Phase 2 Task 2.4 already renamed it to ``"store"``, and this phase depends on
    Phase 2, so a hit there means Task 2.4 regressed.
    """
    import re
    from pathlib import Path as _Path

    src = _Path(__file__).resolve().parents[3] / "src" / "phenotypic"
    hits = [
        f"{p.relative_to(src)}:{n}"
        for p in src.rglob("*.py")
        if str(p.relative_to(src)) not in _KEEPS_AN_HDF_KEY
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r'"hdf"\s*:', line)
    ]
    assert hits == [], hits


def test_the_allowlist_itself_is_not_stale() -> None:
    """An allowlist that stops matching anything is a silent no-op."""
    from pathlib import Path as _Path

    src = _Path(__file__).resolve().parents[3] / "src" / "phenotypic"
    for rel in _KEEPS_AN_HDF_KEY:
        assert (src / rel).is_file(), rel
```

> **Corrected (missing-owner review, 2026-08-19).** An earlier draft asserted
> `hits == []` over **all** of `src/phenotypic`. Verified against the worktree:
> `grep -rnE '"hdf"\s*:' src/phenotypic` returns **17** lines, not 5. Eleven of the other
> twelve are load-bearing and survive this phase — seven `TargetKind` comparisons in
> `sdk_/_metadata_migration.py` (`:1796, :1821, :1885, :1987, :2064, :2428, :2436`) and four
> in `_cli/_cli_output_manager.py` (`:1406` docstring, `:1457`/`:1458` `from_config`,
> `:1524` extension dispatch). The twelfth, `gui/builder/_preview_cache.py:208`, is ported by
> **Phase 2 Task 2.4**, not here. As written the test could not pass without deleting
> migration logic the plan explicitly retains, so the implementer's only move would have been
> to weaken the gate.

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/cli/test_cli_completion_store.py -v
```

Expected: `test_publishing_a_store_artifact_does_not_raise` fails with `IsADirectoryError`,
and `test_marker_version_is_bumped` fails at `1 >= 2`.

- [ ] **Step 3: Implement the `kind` dispatch, bump the version, port the five sites.**

- [ ] **Step 4: Re-run the differential parity test**

```bash
uv run pytest tests/unit/cli/test_cli_completion_store.py tests/unit/cli/test_staged_resume_parity.py -q
```

Expected: green — and the parity test now actually exercises branch 1, because Task 3.4's
fifth artifact axis makes `valid_image_success` return `True` in some combinations.

- [ ] **Step 5: Prove the fifth axis matters**

Temporarily revert `ARTIFACTS` to `repeat=4` and re-introduce the `is_file()` check. The
parity test should PASS despite the broken classifier — demonstrating the blind spot. Restore
both and confirm the parity test now FAILS under the same defect.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli tests/unit/cli/test_cli_completion_store.py
git commit -m "fix(cli): describe a store, not a file, in per-image completion markers

_sha256 opened its argument as a file (IsADirectoryError kills the
publishing worker) and valid_image_success required is_file() (so every
finished image reclassified stage3 forever on the work-id path). Artifact
descriptors gain a kind tag; store descriptors fingerprint the root
zarr.json's CONTENTS via file_fingerprint -- not the directory (which
fingerprints to a constant, since paths_fingerprint emits one sentinel byte
and does not recurse) and not via paths_fingerprint, which folds the
absolute resolved path into the digest and would make stores the one
artifact class that breaks when a tree is moved or reached through a
different symlink. SUCCESS_MARKER_VERSION
is bumped so v1 markers describing a vanished .h5 are not read and failed.
This surface appeared in neither the spec nor the plan until a data-flow
review found it, and the resume parity test could not see it -- hence the
fifth artifact axis in Task 3.4."
```

---

## Phase 3 exit criteria

- [ ] `uv run pytest tests/unit/cli tests/integration/cli -q` is green.
- [ ] `uv run pytest tests/unit/cli/test_staged_resume_parity.py -q` is green, and has been
      demonstrated to fail under both injected resume defects.
- [ ] `test_stage3_publishes_the_post_refined_objmap` has been demonstrated to fail when
      Stage 3's re-promote is removed.
- [ ] `grep -rn "sidecar" src/phenotypic/_cli/ --include='*.py' | grep -vi appledouble`
      returns nothing.

      > **Corrected (missing-owner review, 2026-08-19).** The bare form of this grep cannot
      > pass. Verified: it hits four unrelated **AppleDouble** comments
      > (`_measurement_sources.py:156`, `_cli_chunk_writer.py:297`,
      > `_cli_directory_scanner.py:22`, `_dashboard/_generator.py:911`) that describe macOS
      > `._<name>` files and have nothing to do with the Stage-2 sidecar, plus the fifteen
      > prose mentions in `src/phenotypic/_cli/CLAUDE.md` — a file Task 3.5 now owns for
      > exactly this reason. Same failure shape as Phase 6's exit grep: the implementer's only
      > move against the bare form is to delete a keeper or silently soften the gate.
- [ ] `grep -rn "sidecar" src/phenotypic/_cli/CLAUDE.md` returns only the AppleDouble
      dotfile rule, if that file mentions it — the staged-GPU sidecar prose is retired in
      Task 3.5, not deferred to Phase 6.
- [ ] `uv run pytest tests/unit/test_docs_staged_cli.py -q` is green — it asserts
      `"sidecar" in CLAUDE.md.lower()` and in `gpu_detection_setup.md`, both of which Task 3.5
      rewrites.
- [ ] `grep -rn "dataset_hdf_dir\|\.h5" src/phenotypic/_cli/` returns only migration-path
      references (Phase 5) — nothing in the forward run path.
- [ ] A run start log line contains `durable writes:`.
