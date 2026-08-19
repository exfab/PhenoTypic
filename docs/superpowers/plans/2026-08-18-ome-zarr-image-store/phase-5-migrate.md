# Phase 5 — `--mode migrate`

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §5, plus the supersession notes in the spec header.

**Fixtures this phase defines** (ledger **GEN-29** — every one of these was used across
Tasks 5.2/5.3/5.6/5.7 without a task creating it; Phase 3 Task 3.3 and Phase 4 Task 4.3
already carry tables in this form). All live in `tests/unit/sdk_/conftest.py` (not the repo-root `conftest.py`, which would
make six migration fixtures global to the whole suite), and `LegacyRun` is a frozen dataclass
defined beside them in `tests/unit/sdk_/_migration_fixtures.py`, imported explicitly by every
test that annotates it — `from __future__ import annotations` hides an unimported name from
the runtime but not from mypy (ledger **GEN-41**):

| Fixture | Type | What it is |
|---|---|---|
| `legacy_run` | `Path` | An output root with one dataset `ds`, one image `img`, `results/ds/hdf/img.h5`, `results/ds/measurements/img.parquet`, and a `deliverables/metadata.csv`. No completion markers. |
| `finished_legacy_run` | `LegacyRun` dataclass | `legacy_run` plus authorized markers and a valid aggregate. Attributes: `.path`, `.work_id`, `.stems`, `.full_run_args()`. **Not a `Path`** — annotate it as `LegacyRun`; two round-2 tests annotated it `: Path` while calling `.path`/`.work_id`/`.stems` on it. |
| `markerless_legacy_run` | `Path` | A pre-markers archive: `success_markers_required` falsey, no aggregate. Exercises MIG-23. |
| `half_migrated_run` | `Path` | `legacy_run` with **one of two** images converted — the state Task 5.7's predicate exists to catch. Both images in the *same* dataset, which is what makes the predicate per-image. |
| `migrated_run` | `Path` | `legacy_run` fully converted, sources retained. |
| `published_store` | dataclass | **Defined in Phase 3 Task 3.8**, which is where it is first used (`.output_dir`, `.marker`, `.store`). Listed here only as a cross-reference — Phase 5 depends on Task 3.8, so the fixture cannot live in Phase 5 (ledger **GEN-41**). |

**Depends on:** Phase 2, **plus two narrow edges into Phase 3**:

| Edge | Consumers | Symbol |
|---|---|---|
| Phase 3 **Task 3.4** | Task **5.1** | `staged_store_matches_work_id` (`_cli_staged_resume`) |
| Phase 3 **Task 3.8** | Tasks 5.2, 5.6, 5.7 | the `kind`-tagged completion-marker descriptors |

Only Tasks **5.3** and **5.4** are free of both and can run in parallel with Phase 3 as
before. Recorded as ledger **MIG-5**: an earlier draft declared the whole phase independent,
so an agent executing it per the README DAG would find those symbols absent.

> **The Task 5.1 → Task 3.4 edge is new in round 2 and was not declared (ledger GEN-23).**
> The FLOW-1 fix added `test_work_id_survives_conversion` to Task 5.1, which imports
> `staged_store_matches_work_id` from `phenotypic._cli._cli_staged_resume` — defined in Phase
> 3 Task 3.4. That is MIG-5's exact failure shape, reintroduced by the fix for a different
> concern: an agent executing Phase 5 per the README DAG hits `ImportError` at Task 5.1
> Step 2. Declared here rather than worked around; the test *should* go through the real
> consumer, since threading the attribute is only useful if that consumer reads it.

**Runs in parallel with:** Phase 4, and with Phase 3 only up to **Task 3.4** (for Task 5.1)
and **Task 3.8** (for Tasks 5.2/5.6/5.7) — see the edge table above. Tasks 5.3 and 5.4 have
no Phase 3 edge at all (ledger **GEN-40**).
**Must land before Phase 6** — migration is built on the legacy HDF readers that Phase 6
retires from the public surface.

`--mode migrate` converts an existing output tree **in place**: per-image `.h5` →
`.ome.zarr`, and legacy per-topic metadata headers → canonical flat `Metadata_<Label>` in
every non-image target. **`deliverables/metadata.csv` is not touched at all** — it is
immutable input provenance (user ruling, ledger FLOW-4); a canonical *view* is emitted beside
it as `metadata.canonical.csv`, and no `metadata.original.csv` is ever created.

---

### Task 5.1: `migrate_hdf_to_zarr` and `migrate_run_hdf_to_zarr`

**Files:**
- Create: `src/phenotypic/sdk_/_hdf_to_zarr.py`
- Test: `tests/unit/sdk_/test_hdf_to_zarr.py` (create)
- Fixtures: `tests/fixtures/legacy_hdf/{v1_flat,v2_grouped,v2_enh_gray,v2_grid,v2_image_type,v2_work_id}/` (create)

**Interfaces:**
- Consumes: `Image._load_v2_grouped` (line 984), `Image._load_legacy_flat_group` (line 1079),
  `Image.save2zarr`, `load_image_from_hdf`.
- Produces:
  ```python
  def migrate_hdf_to_zarr(src: Path, dst: Path | None = None, *, keep_source: bool = True) -> Path
  def migrate_run_hdf_to_zarr(output_dir: Path, *, keep_source: bool = True, njobs: int = 1, dry_run: bool = False) -> MigrationReport
  @dataclass(frozen=True)
  class MigrationReport:
      converted: int
      skipped: int
      failed: tuple[tuple[Path, str], ...]
      headers_migrated: int          # pass 1; see Task 5.3
      header_failures: tuple[tuple[Path, str], ...]
  def _conversion_is_faithful(src: Path, store: Path) -> bool   # --delete-sources gate, Task 5.3
  ```

**Constraints specific to this task:**
- Reuse the **existing** v1-flat and v2-grouped loaders. Do not write a third HDF reader.
- **The legacy `enh_gray` layer maps to `detect_mat`.** It is the pre-rename name still
  handled at `_cli_staged_resume.py:82` and must not be dropped silently. A fixture with an
  `enh_gray` layer is mandatory, not optional.
- ⚠️ **`_load_v2_grouped` cannot currently read that fixture.** Verified: the v2 loader does
  a bare `layers["detect_mat"]` (`_image_io_handler.py:1035-1036`) with **no fallback**;
  only the v1-flat loader has one (`:1100-1108`, `# Backward compat: try 'detect_mat'
  first, fall back to 'enh_gray'`). Meanwhile `valid_staged_hdf`
  (`_cli_staged_resume.py:81-83`) accepts `enh_gray` at `schema_version >= 2`, so the code
  believes such files exist in the wild. **Step 3a below adds the fallback**, which is a
  change to a legacy reader that must land here — before Phase 6 retires it. Recorded as
  OPEN-QUESTIONS **D8**.
- **Thread `phenotypic_work_id` through the conversion** (ledger **FLOW-1**, Critical).
  Verified: `staged_hdf_matches_work_id` reads the HDF **root attribute**
  (`_cli_staged_resume.py:99-110`), which the CLI writes as a post-write patch
  (`_cli_output_manager.py:1665-1670`) and which lives in **no image field** — so
  `load_image_from_hdf` does not carry it and `save2zarr` would never see it. Dropping it
  makes every image classify `"stage1"` on the markers-required path: full reprocessing from
  original inputs a migrated archive may no longer have.

  Read it off the source root and pass it explicitly:
  `save2zarr(dst, work_id=<source root attr or None>)`. This is **upstream of** the marker
  work in Task 5.6 — republishing markers does not help if the store itself fails the
  work-id conjunct.
- **Restore the metadata sections verbatim after loading** (ledger **MIG-2**).
  `_load_v2_grouped` drops `Metadata_ImageType` (`_image_io_handler.py:1069-1073` skips any
  key the constructor already set — verified by execution: `GridSection` in, `Image` out),
  and Phase 6 **keeps** that loader as the migration reader. Either add the verbatim restore
  there, beside Step 3a's `enh_gray` fallback, or re-apply the stored sections in
  `migrate_hdf_to_zarr` after the loader returns. Both are legacy-reader changes that must
  land in this phase, before Phase 6.
- Header canonicalization happens **in the same pass**: a converted store is canonical by
  construction. There is no separate header pass for anything that goes through conversion.
- Sources are **retained by default** (`keep_source=True`). Deletion is opt-in.
- Migration is **resumable and restartable**: a store that already exists and passes
  `valid_staged_store` is skipped, so re-running after an interruption is the recovery
  procedure. There is no `--resume` flag.
- Conversion goes through the §3.2 promote, so an interrupted conversion leaves no valid
  root and is simply redone.

- [ ] **Step 1: Build the golden fixtures**

Write a one-off generator under `tests/fixtures/legacy_hdf/_generate.py` and commit both it
and the `.h5` files it produces, so the fixtures can be rebuilt after Phase 6 removes the
production writer.

**Six fixtures, not three** (ledger **MIG-3**). The original three could not have caught
either Critical this review found, because none of them carried the field that breaks:

| Fixture | What it pins |
|---|---|
| `v1_flat` | the legacy flat layout |
| `v2_grouped` | the current grouped layout |
| `v2_enh_gray` | the pre-rename `enh_gray` layer (needs Step 3a's loader fallback) |
| **`v2_grid`** | a `GridImage` — `nrows`/`ncols`/`grid_finder` survival through conversion. `load_image_from_hdf` dispatches on `phenotypic_class`, and nothing currently tests that path |
| **`v2_image_type`** | a **non-default `Metadata_ImageType`** (e.g. `GridSection`). This is the fixture that would have caught **MIG-2** |
| **`v2_work_id`** | a root `phenotypic_work_id` attribute. This is the fixture that would have caught **FLOW-1** |

- [ ] **Step 1a: Pin the generator's fidelity to the real writer — this window closes at Phase 6**

The generator must be hand-rolled on `HDF.save_array2hdf5`, because `save2hdf5` and
`_save_image2hdfgroup` are deleted in Phase 6. That makes every fixture a hand-modelled
*approximation* of what production actually wrote, and after Phase 6 no one can check.

Phase 5 runs **before** Phase 6, so the real writer still exists right now. Use it:

```python
def test_the_generator_matches_the_real_writer(tmp_path: Path) -> None:
    """One-time fidelity check, only possible before Phase 6 deletes save2hdf5.

    After that, the goldens are unfalsifiable against production forever.
    """
    import h5py

    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate
    from tests.fixtures.legacy_hdf._generate import write_v2_grouped

    real, generated = tmp_path / "real.h5", tmp_path / "generated.h5"
    image = Image(load_synth_yeast_plate())
    image.save2hdf5(real)
    write_v2_grouped(generated, image)

    def _shape(path):
        out = {}
        with h5py.File(path, "r") as fh:
            fh.visititems(
                lambda name, obj: out.__setitem__(
                    name, (getattr(obj, "shape", None), str(getattr(obj, "dtype", "")))
                )
            )
            return out, dict(fh.attrs)

    assert _shape(generated) == _shape(real)
```

Mark it `@pytest.mark.skipif` on the *absence* of `save2hdf5` with a message pointing here,
so after Phase 6 it self-documents rather than silently vanishing.

- [ ] **Step 2: Write the failing test**

```python
"""Legacy HDF -> store conversion must equal a freshly written store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.sdk_._hdf_to_zarr import migrate_hdf_to_zarr, migrate_run_hdf_to_zarr
from phenotypic.sdk_.ngff_ import (
    PhenotypicAttr,
    read_phenotypic_attributes,
    valid_staged_store,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_hdf"


@pytest.mark.parametrize(
    "layout",
    ["v1_flat", "v2_grouped", "v2_enh_gray", "v2_grid", "v2_image_type", "v2_work_id"],
)
def test_conversion_produces_a_valid_store(layout: str, tmp_path: Path) -> None:
    store = migrate_hdf_to_zarr(FIXTURES / layout / "img.h5", tmp_path / "img.ome.zarr")
    assert valid_staged_store(store) is True


@pytest.mark.parametrize("layout", ["v1_flat", "v2_grouped", "v2_enh_gray"])
def test_converted_store_conforms(layout: str, tmp_path: Path) -> None:
    from tests._ngff_conformance import assert_store_conforms

    assert_store_conforms(
        migrate_hdf_to_zarr(FIXTURES / layout / "img.h5", tmp_path / "img.ome.zarr")
    )


def test_enh_gray_maps_to_detect_mat(tmp_path: Path) -> None:
    """The pre-rename layer name; dropping it silently loses the detection matrix."""
    store = migrate_hdf_to_zarr(
        FIXTURES / "v2_enh_gray" / "img.h5", tmp_path / "img.ome.zarr"
    )
    block = read_phenotypic_attributes(store)
    assert "detect_mat" in block[PhenotypicAttr.SERIES]
    assert "enh_gray" not in block[PhenotypicAttr.SERIES]
    assert Image.load_layer_zarr(store, "detect_mat").any()


def test_converted_store_matches_the_fixture_s_AUTHORED_metadata(tmp_path: Path) -> None:
    """Assert against what the fixture was BUILT with, not against another load.

    The previous version of this test compared `migrate_hdf_to_zarr(...)` to
    `Image.load_hdf5(...).save2zarr(...)` -- both sides through the SAME
    `_load_v2_grouped`, so any loader-level fidelity loss compared equal and the
    test certified the bug (ledger MIG-2).
    """
    from tests.fixtures.legacy_hdf._generate import V2_IMAGE_TYPE_AUTHORED

    store = migrate_hdf_to_zarr(
        FIXTURES / "v2_image_type" / "img.h5", tmp_path / "converted.ome.zarr"
    )
    protected = read_phenotypic_attributes(store)[PhenotypicAttr.METADATA]["protected"]
    assert protected["Metadata_ImageType"] == V2_IMAGE_TYPE_AUTHORED


def test_grid_state_survives_conversion(tmp_path: Path) -> None:
    from phenotypic import GridImage

    store = migrate_hdf_to_zarr(
        FIXTURES / "v2_grid" / "img.h5", tmp_path / "grid.ome.zarr"
    )
    back = GridImage.load_zarr(store)
    assert (back.nrows, back.ncols) == (16, 24)
    assert back.grid_finder is not None


def test_work_id_survives_conversion(tmp_path: Path) -> None:
    """FLOW-1: without this, every migrated image reclassifies "stage1"."""
    from phenotypic._cli._cli_staged_resume import staged_store_matches_work_id

    store = migrate_hdf_to_zarr(
        FIXTURES / "v2_work_id" / "img.h5", tmp_path / "wid.ome.zarr"
    )
    assert staged_store_matches_work_id(store, "w-fixture") is True


def test_converted_equals_a_freshly_written_store(tmp_path: Path) -> None:
    """Structural equivalence only -- see the test above for content fidelity."""
    converted = migrate_hdf_to_zarr(
        FIXTURES / "v2_grouped" / "img.h5", tmp_path / "converted.ome.zarr"
    )
    fresh = Image._load_hdf5_for_migration(
        FIXTURES / "v2_grouped" / "img.h5"
    ).save2zarr(tmp_path / "fresh.ome.zarr")
    for layer in ("gray", "detect_mat", "objmap"):
        np.testing.assert_array_equal(
            Image.load_layer_zarr(converted, layer),
            Image.load_layer_zarr(fresh, layer),
        )
    a = read_phenotypic_attributes(converted)
    b = read_phenotypic_attributes(fresh)
    for key in (PhenotypicAttr.SERIES, PhenotypicAttr.LABELS, PhenotypicAttr.METADATA):
        assert a[key] == b[key]


def test_legacy_headers_are_canonicalized_in_the_same_pass(tmp_path: Path) -> None:
    store = migrate_hdf_to_zarr(
        FIXTURES / "v1_flat" / "img.h5", tmp_path / "img.ome.zarr"
    )
    block = read_phenotypic_attributes(store)
    for section in ("protected", "public"):
        for key in block[PhenotypicAttr.METADATA][section]:
            assert key.startswith("Metadata_"), key


def test_source_is_retained_by_default(tmp_path: Path) -> None:
    src = tmp_path / "img.h5"
    src.write_bytes((FIXTURES / "v2_grouped" / "img.h5").read_bytes())
    migrate_hdf_to_zarr(src, tmp_path / "img.ome.zarr")
    assert src.is_file()


def test_source_deletion_is_opt_in(tmp_path: Path) -> None:
    src = tmp_path / "img.h5"
    src.write_bytes((FIXTURES / "v2_grouped" / "img.h5").read_bytes())
    migrate_hdf_to_zarr(src, tmp_path / "img.ome.zarr", keep_source=False)
    assert not src.exists()


def test_run_migration_is_idempotent(legacy_run: Path) -> None:
    """Re-running after an interruption is the recovery procedure."""
    first = migrate_run_hdf_to_zarr(legacy_run)
    second = migrate_run_hdf_to_zarr(legacy_run)
    assert first.converted > 0
    assert second.converted == 0
    assert second.skipped == first.converted


def test_dry_run_writes_nothing(legacy_run: Path) -> None:
    from phenotypic.sdk_ import dataset_zarr_dir

    report = migrate_run_hdf_to_zarr(legacy_run, dry_run=True)
    assert report.converted > 0
    assert not dataset_zarr_dir(legacy_run, "ds").exists()


def test_a_failed_conversion_is_reported_not_raised(legacy_run: Path) -> None:
    corrupt = next((legacy_run / "results").rglob("*.h5"))
    corrupt.write_bytes(b"not an hdf")
    report = migrate_run_hdf_to_zarr(legacy_run)
    assert len(report.failed) == 1
    assert report.failed[0][0] == corrupt
```

- [ ] **Step 3: Run to verify failure.** Expected: `ModuleNotFoundError`, plus a `KeyError:
'detect_mat'` on the `v2_enh_gray` fixture once the module exists — that second failure is
what Step 3a fixes.

- [ ] **Step 3a: Give `_load_v2_grouped` the `enh_gray` fallback**

In `_image_io_handler.py`, replace the bare lookup at line 1035 with the same
backward-compat shape the v1-flat loader already uses at `:1100-1108`:

```python
        # Detection matrix + mode. Backward compat: 'enh_gray' is the
        # pre-rename name, still accepted by valid_staged_hdf at
        # _cli_staged_resume.py:82, so schema-2 files carrying it exist.
        if "detect_mat" in layers:
            detect_mat_ds = layers["detect_mat"]
            detect_matrix_data = detect_mat_ds[()]
            detect_mode = detect_mat_ds.attrs.get("detect_mode", "gray")
            if isinstance(detect_mode, bytes):
                detect_mode = detect_mode.decode("utf-8", errors="replace")
        else:
            detect_matrix_data = layers["enh_gray"][()]
            detect_mode = "gray"
```

Add a matching test in `tests/unit/core/` asserting a v2-grouped HDF carrying `enh_gray`
loads with a populated `detect_mat`. Phase 6 Task 6.2 keeps `_load_v2_grouped` as a private
migration reader, so this fallback survives the retirement.

- [ ] **Step 4: Implement.** `migrate_hdf_to_zarr` opens the source through
`load_image_from_hdf` (which already dispatches `Image`/`GridImage` on `phenotypic_class`),
renames `enh_gray` → `detect_mat` if present, canonicalizes headers via the existing
metadata-migration helpers, and calls `save2zarr`. `migrate_run_hdf_to_zarr` walks
`results/*/hdf/*.h5`, skips any stem whose store already passes `valid_staged_store`, and
fans out over `njobs` processes.

- [ ] **Step 5: Run** `uv run pytest tests/unit/sdk_/test_hdf_to_zarr.py -v`. Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_hdf_to_zarr.py tests/unit/sdk_/test_hdf_to_zarr.py tests/fixtures/legacy_hdf
git commit -m "feat(sdk): convert legacy per-image HDFs to OME-Zarr stores

Reuses the existing v1-flat and v2-grouped loaders rather than adding a
third reader. The legacy enh_gray layer maps to detect_mat -- it is the
pre-rename name still handled in the staged resume path, and a fixture
carrying it is part of the suite. Header canonicalization happens in the
same pass, so a converted store is canonical by construction. Conversion
goes through the promote, so an interrupted run is simply re-run."
```

---

### Task 5.2: Emit a canonical metadata view — do **not** rewrite the snapshot

**Files:**
- Modify: `src/phenotypic/sdk_/_hdf_to_zarr.py`
- Test: `tests/unit/sdk_/test_metadata_canonical_view.py` (create)

> **This task was cut down in round 1 of the refinery (user ruling).** It previously rewrote
> `deliverables/metadata.csv` with canonical headers after copying the untouched bytes to
> `deliverables/metadata.original.csv`, and narrowed flat-metadata decision #7 to permit it.
> **All of that is withdrawn.** Recorded as ledger **FLOW-4** / **PRE-D9**.

**Why the rewrite is gone — the first reason is decisive:**

1. **It could not have worked.** `metadata_sha256` is **recomputed from the file on every
   run** (`phenotypicCLI.py:1338` for recompile, `:2135` for full — verified), not read back
   from state. So the moment migration rewrote the file, the next run computed a new digest,
   `expected_finalization` (`_cli_completion.py:391-399`) diverged from the published
   `finalization_input_digest` (`:541-547`), and the entire tree re-finalized — regardless of
   what migration wrote into `state.config`. An earlier draft of this task offered two
   options for the digest and had them **backwards**: leaving the state value stale is
   *self-consistent* (both sides read `state.config`, neither reads the file), so it is
   *updating* it that breaks the aggregate. Neither option was workable.
2. **It bought nothing.** The read path already canonicalizes legacy headers in memory
   (`_cli/_metadata_join.py:86-104`), and `_snapshot_metadata_csv`'s own docstring
   (`phenotypicCLI.py:246-251`) states the snapshot is normalized only on read.
3. **It created a revert hazard.** `_snapshot_metadata_csv:270-280` overwrites the
   destination whenever the passed `--metadata` bytes differ, so re-passing the original CSV
   after a migration would silently undo it.

**Consequences of the cut, all good:** flat-metadata decision #7 stands **unnarrowed** (its
supersession is withdrawn from the spec); `metadata.original.csv` and
`metadata_original_sha256` do not exist; the digest question closes; and the two tests that
could not both hold are gone.

**What remains — optional and additive.** If a canonical view is wanted for downstream
tooling, emit it as a **new file**, never in place:

**Constraints specific to this task:**
- Write `deliverables/metadata.canonical.csv`. **Never** touch `deliverables/metadata.csv`.
- It is a derived view, so it carries no provenance role and needs no digest in state.
- Skip silently when there is no `metadata.csv` to derive from.
- Written beside the snapshot, in the same `deliverables/` directory. Migration is in
  place, so there is no second location to reconcile.

- [ ] **Step 1: Write the failing test**

```python
"""Migration derives a canonical view; it never rewrites the snapshot."""

from __future__ import annotations

import csv
from pathlib import Path


def test_the_snapshot_is_never_rewritten(legacy_run: Path) -> None:
    """flat-metadata decision #7, unnarrowed."""
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    source = legacy_run / "deliverables" / "metadata.csv"
    before = source.read_bytes()
    migrate_run_hdf_to_zarr(legacy_run)
    assert source.read_bytes() == before


def test_no_original_copy_is_created(legacy_run: Path) -> None:
    """metadata.original.csv was an artifact of the withdrawn rewrite."""
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(legacy_run)
    assert not (legacy_run / "deliverables" / "metadata.original.csv").exists()


def test_a_canonical_view_is_emitted_beside_it(legacy_run: Path) -> None:
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(legacy_run)
    view = legacy_run / "deliverables" / "metadata.canonical.csv"
    assert view.is_file()
    with view.open(encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert all(column.startswith("Metadata_") for column in header if column), header


def test_the_aggregate_publication_survives_migration(finished_legacy_run) -> None:
    """The test that could not pass under the rewrite -- now it can."""
    from phenotypic._cli._cli_completion import aggregate_publication_is_valid
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(finished_legacy_run.path)
    assert aggregate_publication_is_valid(finished_legacy_run.path) is True


def test_the_view_sits_beside_the_snapshot(legacy_run: Path) -> None:
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(legacy_run)
    deliverables = legacy_run / "deliverables"
    assert (deliverables / "metadata.canonical.csv").is_file()
    assert (deliverables / "metadata.csv").is_file()  # snapshot untouched
```

- [ ] **Step 2: Run to verify failure.** Expected: the canonical view is absent. The three
negative assertions should pass immediately — they pin behaviour the cut *removes*, so they
are regression guards against the rewrite creeping back.

- [ ] **Step 3: Implement** the derived view. Reuse the same in-memory canonicalization the
read path uses; do not add a second mapping.

- [ ] **Step 4: Re-run**, including `test_the_aggregate_publication_survives_migration`,
which depends on Task 5.6 for the marker half.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_hdf_to_zarr.py tests/unit/sdk_/test_metadata_canonical_view.py
git commit -m "feat(migrate): derive a canonical metadata view, never rewrite the snapshot

The planned in-place rewrite of deliverables/metadata.csv could not have
worked: metadata_sha256 is recomputed from the file on every run, not read
from state, so any rewrite made the next run's expected_finalization
diverge from the published digest and re-finalize the whole tree --
whatever migration wrote into state. It also bought nothing, since the read
path already canonicalizes in memory, and it created a revert hazard via
_snapshot_metadata_csv.

deliverables/metadata.canonical.csv is emitted as a derived view instead.
flat-metadata decision #7 therefore stands unnarrowed and its supersession
is withdrawn from the spec."
```

---

### Task 5.3: `--mode migrate` CLI wiring

**Files:**
- Create: `src/phenotypic/_cli/_cli_migrate.py`
- **Modify: `src/phenotypic/sdk_/typing_.py` — add `"migrate"` to `CliMode` (line 121).**
  It is `Literal["full", "measure", "recompile", "process"]`, and `phenotypicCLI.py:1214`
  does `cli_mode = cast(CliMode, mode)`. Without the new member, step 1 below —
  `migrate_only = cli_mode == "migrate"` — is a **`comparison-overlap` error under
  `uv run mypy src/phenotypic`**: mypy narrows `cli_mode` to the four-member Literal and
  proves the comparison is always `False`. Update the `#:` doc comment above it in the same
  edit; that comment enumerates the modes and is the file's only description of them.
  (Missing-owner review, 2026-08-19: the file was in no task's `Files:` list.)
- Modify: `src/phenotypic/phenotypicCLI.py` (`--mode` choices line 943; the mode-validation
  block at lines 1217–1244; the module docstring's mode list at lines 71–80 and 1183)
- Modify: `src/phenotypic/sdk_/_metadata_migration.py` — add the `kinds` filter parameter
  to `preflight_metadata_schema` and `migrate_metadata_bundle`, threaded into
  `_discover_bundle_targets`. **Additive and default-off** (ledger **MIG-26**); no existing
  caller changes behaviour.
- Modify: `src/phenotypic/sdk_/_hdf_to_zarr.py` — `_conversion_is_faithful` (below) and the
  two new `MigrationReport` fields extend **Task 5.1's** module (ledger **GEN-42**).
- Test: `tests/unit/cli/test_cli_migrate_mode.py` (create)

**Interfaces:**
- Produces: `--mode migrate --output <output-dir> [--njobs N] [--dry-run] [--delete-sources]`

  > **Copy mode was specified and then removed (2026-08-19, user ruling).** An intermediate
  > draft added `--input <src> --output <dst>`, writing a converted tree at a new path and
  > leaving the source untouched. **Withdrawn** — spec §5.1 carries the full reasoning; in
  > short, in-place migration only has to convert images, whereas copy mode had to reproduce
  > the *entire* output contract at a new path, and its copy set omitted
  > `results/<ds>/measurements/*.parquet` — marker-bound artifacts whose absence would have
  > made the destination reprocess every image. The safety it reached for is already supplied
  > by the default `keep_source=True`.
  >
  > The interface therefore reverts to **`--output` naming the tree to convert**, exactly like
  > `recompile`. That incidentally settles the `--input`/`--output` inconsistency three
  > reviewers flagged in this task's own tests (ledger **MIG-17** / **FLOW-29** / **GEN-25**),
  > since those tests already assumed this form.

  `--delete-sources` survives the removal, and is now the *only* reclaim path: it is what
  makes `keep_source=False` reachable at all — both migrate functions take the parameter and
  Task 5.1 tests both branches, but no CLI surface exposed it, so every migration permanently
  doubled the tree's footprint (ledger **MIG-9**). Its precondition is stated below and is
  deliberately stronger than `valid_staged_store`.

**Constraints specific to this task:**
- **`migrate` reuses `recompile`'s `--input`/`--pipeline` validation, but NOT its
  `--dry-run` rejection** (ledger **FLOW-34**). Read `phenotypicCLI.py:1216-1242`: there are
  no tuples — `measure_only = cli_mode == "measure"` (`:1216`) and
  `recompile_only = cli_mode == "recompile"` (`:1217`) are booleans, and the block reads

  ```python
  if measure_only or recompile_only:
      if input_path is not None:
          raise click.UsageError(...)                    # :1230-1233  <- reuse this
      if dry_run:
          raise click.UsageError(
              f"--dry-run cannot be combined with --mode {cli_mode}."   # :1235-1237
          )                                              #             <- migrate MUST be exempt
  ```

  `--dry-run` is a **required** part of migrate: spec §5.1's interface line, this task's own
  constraint, `migrate_run_hdf_to_zarr(dry_run=...)`, `test_dry_run_reports_without_writing`,
  and Phase 5 exit criterion 2 all depend on it. Three explicit edits:

  1. introduce `migrate_only = cli_mode == "migrate"`;
  2. add it to the `--input` guard (`:1230`) and to the
     `recompile_only and pipeline_json is not None` guard (`:1238-1242`);
  3. **exempt it from the `--dry-run` guard** (`:1235`).

  > An earlier draft said *"`migrate` is added to the same tuples `recompile` already appears
  > in. No guard needs a new exemption."* Both halves are wrong — there are no tuples, and
  > folding migrate into that condition rejects `--dry-run`, failing four tests and an exit
  > criterion at implementation.
- **Migration is in place.** `results/<ds>/zarr/` appears beside `results/<ds>/hdf/`;
  `measurements/`, `overlays/`, `deliverables/`, and machine state stay exactly where they
  are. Resumption is the same property stated in Task 5.1 — re-scan the tree and skip any
  image whose store already passes `valid_staged_store`.
- **`--delete-sources` is the only irreversible step in this plan, and its precondition must
  be stronger than structural validity** (ledger **MIG-20**). `valid_staged_store` proves a
  store is well-formed, not that the conversion preserved content — and **both** Criticals
  this review found (MIG-2's dropped `Metadata_ImageType`, FLOW-1's dropped
  `phenotypic_work_id`) produce structurally valid stores. Unlinking the `.h5` on that
  evidence loses the original permanently, with no receipt and no rollback.

  Gate each unlink on a positive re-read comparison against its source **plus** a passing
  `valid_image_success` for that image after republication. Unlink only when both hold.

  ⚠️ **The comparison must be VALUE-level, not key-level** (ledger **MIG-28**). An earlier
  draft specified "layer names, shapes and dtypes, the metadata key set, and
  `phenotypic_work_id`" — that catches FLOW-1 (a genuinely absent attribute) and **misses
  MIG-2**, the other Critical it exists to catch. MIG-2 is not a dropped key:
  `_load_v2_grouped`'s restore loop is

  ```python
  for mapped, value in decoded.items():
      if mapped in target and target[mapped] is not None:
          continue                      # <- the stored value is SKIPPED, not dropped
      target[mapped] = value
  ```
  (`_core/_image_parts/_image_io_handler.py:1073-1076`)

  and the `Image` constructor has already set `Metadata_ImageType` to a non-`None` default. So
  the key is **present**, carrying `"Image"` where the file said `"GridSection"`, and a key-set
  comparison sees two identical key sets and returns `True`. The `.h5` is then unlinked and the
  correct value is gone permanently. Shapes and dtypes are equally blind to **content**: a
  conversion that wrote a correctly-shaped zero `detect_mat` passes.

  Specify instead: `protected`/`public`/`imported` compared as **full mappings** after
  normalization; a per-layer content digest (sha256 of `np.ascontiguousarray(...).tobytes()`)
  for `rgb`/`gray`/`detect_mat`/`objmap`; `phenotypic_work_id`; and grid state
  (`nrows`/`ncols`) when the source is a `GridImage`. That is one extra read of a file about to
  be deleted — cheap insurance on the only irreversible step in the plan.

  **`_conversion_is_faithful` is a real symbol, not just a monkeypatch target** (ledger
  **MIG-29**). Signature `_conversion_is_faithful(src: Path, store: Path) -> bool`, in
  `sdk_/_hdf_to_zarr.py` (Task 5.1's module — listed in this task's Files as an extension).
  Called **per image**, immediately before that image's unlink. Policy, which the one-image
  test does not pin: unlink per image, accumulate refusals into `MigrationReport.failed`, and
  exit non-zero at the end naming every source left in place. A single unfaithful image does
  not block the other 99 from reclaiming space, but the run is not reported as clean.

  Add a test that a store whose metadata **value** was mutated (not whose key was removed) is
  rejected — that is the MIG-2 shape.
- **`--mode migrate` is explicitly TWO-PASS** (user ruling; ledger **MIG-7b**). State both
  in the mode's help text and in the run summary:

  | Pass | Covers |
  |---|---|
  | **1** | `migrate_metadata_bundle` over every **non-image** target — `csv`, `parquet`, `json`, `frame`: per-dataset `measurements/*.parquet` and the root pipeline JSON (`_metadata_migration.py:44`, `:797-845`) |
  | **2** | per-image `results/*/hdf/*.h5` → `results/*/zarr/*.ome.zarr`, then the marker republication of Task 5.6, then the aggregate republish **last** |

  > **The order is non-image FIRST, and it is load-bearing (ledger MIG-15, FLOW TRACE-4).**
  > An earlier draft ran images first. But the non-image pass rewrites
  > `results/<ds>/measurements/*.parquet` — **marker-bound artifacts**: every per-image
  > completion marker carries that parquet's `size` and `sha256`. Rewriting them *after*
  > Task 5.6 republished the markers invalidates every marker it just wrote, silently
  > reintroducing the exact MIG-1 failure Task 5.6 exists to prevent — on the default path.
  >
  > The repair mechanism does exist (`refresh_success_markers_after_metadata_migration`, wired
  > for recompile at `_cli_recompile_metadata_migration.py:72-82`), but relying on it here is
  > strictly worse: Task 5.4 removes **both** of its production callers, so it would have to be
  > carried across along with its receipt binding, while Task 5.6 is concurrently modifying it.
  > Canonicalizing first and publishing markers over the **final** bytes needs no bridge, no
  > receipts, and no ordering hazard.
  >
  > **The pass-1 entry point is `migrate_metadata_bundle`** (`_metadata_migration.py:2291`,
  > re-exported at `sdk_/__init__.py:259-260`) — **not `migrate_metadata_schema`, which an
  > earlier draft named and which is not a symbol in this codebase.**

  **Pass 1's call sequence, stated exactly** (ledger **MIG-27**). "Lift the preflight from
  `migrate_metadata_schema_for_recompile`" is *not* sufficient — the load-bearing line in that
  function is the one above the preflight:

  ```python
  # Construct the layout DIRECTLY. Never pass a Path.
  layout = BundleLayout(deliverables_base=deliverables_dir(root), output_root=root)
  report = preflight_metadata_schema(layout)          # writes nothing -> this IS --dry-run
  result = migrate_metadata_bundle(layout, expected_plan_fingerprint=report.plan_fingerprint)
  ```

  Passing a `Path` routes through `_resolve_bundle` → `BundleLayout.detect`
  (`_io_constants.py:2000-2044`), which **raises `FileNotFoundError` unless
  `deliverables/master_measurements.parquet` exists**. A pre-aggregate or interrupted legacy
  run is precisely the migration subject, and the plan's own `legacy_run` fixture has no
  `master_measurements.parquet` — so **every Task 5.3 test would fail at pass 1**, with an
  error message ("Point the viewer at a `python -m phenotypic` output dir") that names nothing
  relevant. `migrate_metadata_schema_for_recompile` constructs the layout directly for exactly
  this reason, and says so in its docstring: *"a recoverable run may have per-image HDF
  authority even when an earlier aggregate is absent."*

  Treat any `report.status` outside `{compatible, applied}` as an abort **before** pass 2,
  surfaced through `header_failures`. `preflight_metadata_schema` writing nothing is also what
  makes pass-1 `--dry-run` free — state that, it is not incidental.

  ⚠️ **The `.h5` exclusion needs a parameter that does not exist yet** (ledger **MIG-26**).
  `migrate_metadata_bundle(source, *, expected_plan_fingerprint)` takes no filter,
  `preflight_metadata_schema(source)` takes one argument, `_discover_bundle_targets(layout)` is
  private and takes only the layout, and `BundleLayout` is a frozen two-field dataclass. So a
  caller **cannot** express "non-image targets only" today. Thread an explicit
  `kinds: frozenset[TargetKind] | None = None` through `preflight_metadata_schema` and
  `migrate_metadata_bundle` into `_discover_bundle_targets`, defaulting to `None` = today's
  behaviour so the `rollback_metadata_migration` and standalone-bundle call sites are
  untouched. Add a unit test asserting pass 1's target set contains no `.h5`.

  Without this the executor meets a constraint they are not authorized to satisfy, and both
  likely resolutions are bad: silently drop the exclusion (MIG-25 lands in full), or edit
  `_metadata_migration.py` off-plan.

  Three constraints on pass 1, all consequences of running it first:

  - ⚠️ **Pass 1 MUST NOT touch `.h5` targets at all** — unconditionally, not conditionally on
    a store existing (ledger **MIG-25** / **FLOW-35**, raised independently by two reviewers).

    `_discover_bundle_targets` walks `dataset_root/"hdf"` with `rglob("*")` and appends every
    `.h5` (`_metadata_migration.py:797-812`), and the apply path for one is `_migrate_hdf_copy`
    (`:1365-1369`), which does **`shutil.copy2(source, temp)`** — a full byte copy — before
    rewriting attrs and publishing by rename. Every pre-flat-metadata `.h5` is `migratable`
    even when its headers are already canonical, because `_inspect_hdf` sets
    `needs_metadata_marker` on a missing marker alone (`:604-606`) and `_target_status`
    returns `"migratable"` if *either* signal is set (`:229-239`).

    So on a first migration — the only one that matters — pass 1 byte-copies and rewrites
    **every `.h5` in the archive**, single-process, with no SLURM path and requiring free
    space for a full second copy. Three things break at once:

    1. **The rollback story that justified removing copy mode is destroyed.** Spec §5.1 tells
       the user *"the `.h5` files survive the conversion, so if the stores are wrong the
       originals are still there."* After pass 1 they are not the originals — every one has
       been rewritten, before a single store exists.
    2. **It invalidates the pre-existing per-image markers**, which bind the `.h5`'s `size`
       and `sha256`. That is MIG-1's failure mode in the `.h5` direction, and Task 5.6's
       republication only rescues images that convert successfully.
    3. **It falsifies Task 5.4's cost claim** — *"the SLURM fan-out existed only because
       copying large HDFs is slow"* — which is the stated reason for deleting 879 lines. The
       fan-out existed for exactly this rewrite.

    > **An earlier draft made the skip conditional** (*"skip an `.h5` whose stem already has a
    > valid store"*, ledger MIG-21) and then observed *"on a first migration nothing is
    > skipped and the cost claim holds"* — which is the cost claim **failing**, drawn as the
    > opposite conclusion. Under the pre-inversion order the skip fired for every converted
    > image; the inversion structurally disabled it.

    **The unconditional exclusion is correct, not merely cheaper**, and Task 6.4 supplies the
    reason: header canonicalization is a property of the **read** path
    (`_normalize_stored_metadata_items`, inside both legacy loaders), so `save2zarr` writes
    canonical metadata whether or not the source `.h5` header was ever rewritten. Rewriting it
    first is dead work in **every** case — the same fact that made Task 5.5 unnecessary. This
    also reconciles the pass table, which already says pass 1 covers *"every **non-image**
    target"*, with what the code would otherwise do.

  - **`--dry-run` suppresses both passes** and reports what each would touch. **An
    interruption between passes is safe to re-run**, but *not* for the reason an earlier draft
    gave — "pass 1 is idempotent by content" is false, since a parquet rewrite is not
    byte-idempotent and a re-applied rewrite changes every sha256. The two real mechanisms,
    both verified in `migrate_metadata_bundle` (`:2291-2332`), are: (a)
    `if requested_receipt.is_file(): return _apply_receipt(...)` short-circuits a re-run onto
    the existing receipt, and (b) `if report.status == "compatible": return
    _compatible_result(report)` makes an already-canonical bundle a no-op that rewrites
    nothing. **Name both**, because an executor who "optimizes" past the receipt check on the
    strength of the wrong reason breaks marker validity for the whole tree.

  - **`MigrationReport` gains `headers_migrated: int` and
    `header_failures: tuple[tuple[Path, str], ...]`** so a pass-1 failure has somewhere to
    appear — today it has `converted`/`skipped`/`failed` only. Declare them on the dataclass
    in **Task 5.1**, which owns `_hdf_to_zarr.py`, not here (ledger GEN-42).

  Without pass 1 those non-image targets lose their migration path entirely, because Task 5.4
  stops `recompile` from rewriting anything — a regression against flat-metadata decision #1
  that the supersession note would not have acknowledged.
- **Local-only, parallel via `--njobs`.** No SLURM controller, no array, no chunking, no
  `MaxArraySize` accounting. Migration is one-time, resumable, and restartable, so it does
  not justify another scheduler surface. A test asserts no `sbatch` is invoked.
- **A run whose output contains only `.h5` results fails with a pointer to this mode**
  rather than auto-migrating. Format conversion rewrites the entire results tree; that is
  typed deliberately, not triggered as a side effect of an unrelated `--mode full`.

- [ ] **Step 1: Write the failing test**

> **Corrected (wrong-symbol sweep).** Three defects in an earlier draft of this block:
>
> 1. **`main` is not a symbol in `phenotypic.phenotypicCLI`.** The click command is
>    **`phenotypic_cli`** (`phenotypicCLI.py:1146`), the name `__main__.py` and
>    `tests/unit/cli/test_cli_mode_contract.py:9` both import.
> 2. **There is no `cli_runner` fixture** — `grep -rn "def cli_runner" tests/` is empty.
>    Every existing CLI test constructs `CliRunner()` inline
>    (`tests/unit/cli/test_cli_mode_contract.py:13` and 15 other files). This plan follows
>    that, rather than introducing a fixture the repo has managed without.
> 3. **`--pipeline p.json` / `--input imgs` never reached the mode guard under test.**
>    Both options are declared `click.Path(exists=True)` (`phenotypicCLI.py:914` and
>    `:926`), so click exits **2 during parsing** and the asserted `"migrate"` string never
>    appears in the output. The test passed for the wrong reason — or rather, asserted
>    something the run never produced. It must hand click paths that exist.

```python
"""``--mode migrate`` CLI contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli


def test_migrate_is_an_accepted_mode() -> None:
    result = CliRunner().invoke(phenotypic_cli, ["--mode", "migrate", "--help"])
    assert result.exit_code == 0


def test_migrate_rejects_pipeline_and_input(tmp_path: Path, legacy_run) -> None:
    """Same validation as recompile: the tree is named by --output alone.

    Both flags are ``click.Path(exists=True)``, so the arguments must exist on
    disk or click exits 2 while parsing and the mode guard never runs.
    """
    pipeline = tmp_path / "p.json"
    pipeline.write_text("{}", encoding="utf-8")
    images = tmp_path / "imgs"
    images.mkdir()

    for flag, value in (("--pipeline", pipeline), ("--input", images)):
        result = CliRunner().invoke(
            phenotypic_cli,
            ["--mode", "migrate", "--output", str(legacy_run), flag, str(value)],
        )
        assert result.exit_code != 0
        assert result.exit_code != 2, "must fail in the mode guard, not click parsing"
        assert "migrate" in result.output


def test_migration_is_in_place(legacy_run) -> None:
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_.ngff_ import valid_staged_store

    assert CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)]
    ).exit_code == 0
    assert valid_staged_store(zarr_store_path(legacy_run, "ds", "img"))


def test_sources_are_retained_unless_delete_sources_is_passed(legacy_run) -> None:
    """MIG-9: --delete-sources is the only path to keep_source=False."""
    hdf = legacy_run / "results" / "ds" / "hdf"

    assert CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)]
    ).exit_code == 0
    assert list(hdf.glob("*.h5")), "retained by default"

    assert CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--delete-sources"],
    ).exit_code == 0
    assert not list(hdf.glob("*.h5"))


def test_delete_sources_refuses_when_the_re_read_diverges(
    legacy_run, monkeypatch
) -> None:
    """MIG-20: a lossy conversion can still be structurally valid, so the
    precondition for the one irreversible step must re-read and compare."""
    from phenotypic.sdk_ import _hdf_to_zarr

    monkeypatch.setattr(_hdf_to_zarr, "_conversion_is_faithful", lambda *a, **k: False)
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--delete-sources"],
    )
    assert result.exit_code != 0
    assert list((legacy_run / "results" / "ds" / "hdf").glob("*.h5")), "nothing unlinked"


def test_migrate_converts_a_legacy_tree(legacy_run) -> None:
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_.ngff_ import valid_staged_store

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)]
    )
    assert result.exit_code == 0
    assert valid_staged_store(zarr_store_path(legacy_run, "ds", "img"))


def test_migrate_never_submits_a_slurm_job(legacy_run, monkeypatch) -> None:
    """One-time, resumable work does not justify another scheduler surface."""
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("migrate must not shell out")
    )
    assert CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)]
    ).exit_code == 0


def test_a_legacy_only_output_fails_with_a_pointer(legacy_format_run) -> None:
    """Conversion rewrites the whole results tree; it must be typed deliberately."""
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "recompile", "--output", str(legacy_format_run)],
    )
    assert result.exit_code != 0
    assert "--mode migrate" in result.output


def test_dry_run_reports_without_writing(legacy_run) -> None:
    from phenotypic.sdk_ import dataset_zarr_dir

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--dry-run"],
    )
    assert result.exit_code == 0
    assert not dataset_zarr_dir(legacy_run, "ds").exists()
```

- [ ] **Step 2–4: Run to verify failure, implement, re-run.**

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_migrate.py src/phenotypic/phenotypicCLI.py tests/unit/cli/test_cli_migrate_mode.py
git commit -m "feat(cli): add --mode migrate

Joins the existing mode choice list and reuses recompile's argument
validation. Local-only with --njobs: migration is one-time, resumable, and
restartable, so it does not justify another SLURM controller/array surface
with its own chunking and MaxArraySize accounting. A legacy-only output
root fails with a pointer to this mode instead of auto-migrating, because
format conversion rewrites the whole results tree."
```

---

### Task 5.4: Move the metadata-schema migration under `migrate` and drop its SLURM fan-out

**Files:**
- Modify: `src/phenotypic/_cli/_cli_recompile_metadata_migration.py` (98 lines)
- Delete: `src/phenotypic/_cli/_cli_recompile_metadata_migration_slurm.py` (345 lines),
  `_cli_recompile_metadata_migration_worker.py` (534 lines) — **879** lines deleted. The
  spec's "~950" counts `_cli_recompile_metadata_migration.py` (98) too, but that file is
  *modified*, not deleted
- Modify: `src/phenotypic/phenotypicCLI.py` — **`:196` (the module-scope
  `import _cli_recompile_metadata_migration_slurm`)** plus recompile's migration hook.
  Missing `:196` makes the entire CLI unimportable, since the module is deleted here
  (ledger **FLOW-9**). Loud and immediate, but it costs a cycle.
- **Delete: `tests/unit/cli/test_cli_recompile_metadata_migration_slurm.py` (2,121 lines).**
  It is the deleted modules' own suite and the only remaining importer: **26 import sites** —
  14 `from phenotypic._cli._cli_recompile_metadata_migration_slurm import …` and 12
  `from phenotypic._cli._cli_recompile_metadata_migration_worker import …` (the first at
  `:189`, the last at `:1412`). Every one becomes a `ModuleNotFoundError` at collection the
  moment Step 3 deletes the modules, so this cannot be deferred to a later phase. It is a
  **delete, not a port**: its entire subject is the SLURM fan-out, and this task's own premise
  is that the fan-out has no remaining justification. Before deleting, read it once for
  assertions about *migration semantics* rather than about scheduling — those move into
  `tests/unit/cli/test_recompile_no_longer_migrates.py` or Task 5.1's suite.
- **Modify: `tests/unit/schema/test_no_metadata_literals.py`** — remove the allowlist entry
  keyed on the deleted file's path at `:173`
  (`"tests/unit/cli/test_cli_recompile_metadata_migration_slurm.py": {"MetadataSample_Strain",
  "MetadataGenetic_Strain"}`). Leave the neighbouring `test_cli_recompile.py` entry alone —
  that file survives. **The gate does check** — `test_legacy_metadata_allowlist_entries_are_not_stale`
  (`:293-308`) walks `_LEGACY_ALLOWED` and appends `"<rel>: file no longer exists"` for any
  key whose path is missing, then asserts the list is empty. So deleting the test file without
  pruning this entry turns one deletion into a **second** red test in a different package,
  and `uv run pytest tests/unit/cli -q` (this task's Step 2–4 command) will not show it —
  it lives under `tests/unit/schema/`.
- Test: `tests/unit/cli/test_recompile_no_longer_migrates.py` (create)

> **Corrected (missing-owner review, 2026-08-19).** An earlier draft deleted two modules
> (879 lines) without naming a single importer of either. Both entries above were in no task's
> `Files:` list, so no agent was authorized to touch them — and `test_the_slurm_fanout_modules_are_gone`,
> this task's own test, would have passed while the file next to it failed to collect.

**Constraints specific to this task:**
- This **supersedes flat-metadata decision #1** ("Every recompile migrates automatically…
  not restricted to a special command"). `recompile` **stops rewriting** legacy headers but
  **keeps reading** them — its decision #3 (permanent stored-data compatibility) is
  untouched, so no existing output directory breaks. Recompile simply no longer mutates one
  as a side effect.
- The SLURM fan-out existed only because copying large HDFs is slow. Conversion through the
  promote is not, and migration is local-only (Task 5.3), so the fan-out has no remaining
  justification. Delete it; do not port it.
- Record the supersession in the flat-metadata spec in Task 6.4.

- [ ] **Step 1: Write the failing test**

> **Two distinct fixtures, or these tests contradict each other.** An earlier draft had
> `test_a_legacy_only_output_fails_with_a_pointer` (Task 5.3) asserting
> `--mode recompile --output <legacy_run>` exits **non-zero**, while
> `test_recompile_still_reads_legacy_headers` asserted the same command exits **zero** — and
> its body referenced an undefined `legacy_run_v2`. The intended distinction is real but the
> fixtures did not encode it. Recorded as OPEN-QUESTIONS **D16**. Define both in
> `tests/unit/cli/conftest.py`:
>
> - **`legacy_format_run`** — output tree whose `results/*/hdf/*.h5` exist and whose
>   `results/*/zarr/` does not. `recompile` must **fail** with a pointer to `--mode migrate`.
> - **`legacy_headers_run`** — output tree already converted to stores, but whose metadata
>   headers are still legacy per-topic names. `recompile` must **succeed**, read those
>   headers, and **not** rewrite them.

```python
from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli


def test_recompile_still_reads_legacy_headers(legacy_headers_run) -> None:
    """Decision #3 is untouched: no existing output directory breaks."""
    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "recompile", "--output", str(legacy_headers_run)]
    )
    assert result.exit_code == 0


def test_recompile_does_not_rewrite_headers(legacy_headers_run) -> None:
    before = _read_headers(legacy_headers_run)
    CliRunner().invoke(
        phenotypic_cli, ["--mode", "recompile", "--output", str(legacy_headers_run)]
    )
    assert _read_headers(legacy_headers_run) == before


def test_the_slurm_fanout_modules_are_gone() -> None:
    import importlib

    for name in (
        "phenotypic._cli._cli_recompile_metadata_migration_slurm",
        "phenotypic._cli._cli_recompile_metadata_migration_worker",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_migrate_performs_the_header_migration(legacy_headers_run) -> None:
    before = _read_headers(legacy_headers_run)
    CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_headers_run)]
    )
    after = _read_headers(legacy_headers_run)
    assert after != before
    assert all(h.startswith("Metadata_") for h in after)
```

- [ ] **Step 2–4: Run to verify failure, implement, re-run**
      `uv run pytest tests/unit/cli tests/unit/schema/test_no_metadata_literals.py -q`.
      The schema path is not optional: the allowlist entry pruned above lives there, and
      `tests/unit/cli` alone reports green while it is red.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "refactor(cli): move metadata-schema migration into --mode migrate

Supersedes flat-metadata decision #1. Recompile stops REWRITING legacy
headers but keeps READING them, so decision #3 (permanent stored-data
compatibility) is untouched and no existing output directory breaks --
recompile simply no longer mutates one as a side effect. The ~950-line
SLURM fan-out existed only because copying large HDFs is slow; conversion
through the promote is not, and migration is local-only, so it is deleted
rather than ported."
```

---

### Task 5.5: ~~Header-only migration via hard-link promote~~ — **CUT**

**Cut in round 1 of the refinery** (ledger SIMP-1, corroborated by the migration
specialist). Two reviewers reached this independently.

**It had no production caller.** Verified by grep over all nine plan documents: only its own
definition and its own five tests. Task 5.3's `--mode migrate` driver never invoked it.

**And it had no reachable input.** Legacy per-topic headers are canonicalized **on ingest,
in memory** — `ensure_metadata_prefix` (`sdk_/_metadata_helpers.py:292-305`) resolves the
historical per-topic spellings to `member.value`, and `_remap_legacy_metadata_key`
(`_core/_image_parts/_image_io_handler.py:92-106`) routes every stored key through it. An
in-memory `Image` therefore cannot hold a known legacy header, so `save2zarr` cannot write
one, and there was no store for this task to operate on.

> Note the reasoning that does **not** work, since it is tempting: "a store can only exist
> because this version wrote it at `metadata_schema_version = 2`". That marker was a
> hard-coded constant written over metadata the same function documents as "verbatim and
> unvalidated", so it evidenced nothing about the headers — which is separately why the
> marker itself is now dropped (ledger MIG-4, user ruling).

**What the cut removes:** the hard-link promote, the `os.link`-failure copy fallback, a
`refresh_success_markers_after_metadata_migration` store bridge, a conformance re-check, the
"hard-linked `.part` shares bytes with the live store" hazard, and the receipt/rollback gap
the migration specialist raised as MIG-6 — which dissolves entirely with the task.

**Ripples applied with the cut:**

- Spec §5.3 is annotated as withdrawn (done — see the spec's inline callout).
- Task 3.8's constraint justifying the store-descriptor bridge is **re-anchored on full
  migration** rather than on this task, so it does not get cut alongside it. Full
  `--mode migrate` writes brand-new stores and still needs it (Task 5.6 below).
- Task 5.4's `legacy_headers_run` fixture describes an unreachable state; it is re-based on
  a `.h5`-backed tree in that task.
- OPEN-QUESTIONS **D16** (the two-fixture split) is reworded, not deleted.
- Before deleting the hard-link promote primitive, confirm nothing in Phases 1 or 3 depends
  on it. (It does not: `promote_store` is the only promote, and it never hard-links.)

**Condition attached to the cut — do not skip this.** With Task 5.5 gone, the product has
**zero** header-migration path for stores. That is safe only while the ingest normalization
above holds, so it must be **pinned by test rather than inferred**. Add to
`tests/unit/sdk_/test_hdf_to_zarr.py`:

```python
def test_a_store_written_from_legacy_headers_comes_out_canonical(tmp_path: Path) -> None:
    """The invariant that makes header-only store migration unnecessary.

    If ingest ever stops canonicalizing, this fails and the cut of Task 5.5 has
    to be revisited -- rather than silently shipping stores with legacy headers
    and no migration path.
    """
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

    image = Image(load_synth_yeast_plate())
    image._metadata.public["MetadataPlate_Strain"] = "BY4741"  # legacy per-topic spelling
    store = image.save2zarr(tmp_path / "legacy.ome.zarr")

    public = read_phenotypic_attributes(store)[PhenotypicAttr.METADATA]["public"]
    assert "MetadataPlate_Strain" not in public
    assert public["Metadata_Strain"] == "BY4741"
```

---

### Task 5.6: Migration re-publishes run state

**Files:**
- Modify: `src/phenotypic/sdk_/_hdf_to_zarr.py`
- Modify: `src/phenotypic/_cli/_cli_completion.py`
  (`refresh_success_markers_after_metadata_migration` — the `is_file()` `raise` and the
  version skip, **not** the docstring lines Task 3.8 originally cited)
- Test: `tests/unit/sdk_/test_migration_republishes_state.py` (create),
  `tests/integration/cli/test_migrate_end_to_end.py` (create)

**Why this task exists — it is a Critical the spec and the plan both missed.**

`--mode migrate` converts images but never re-publishes the per-image completion markers
that gate them, and three mechanisms make that fatal. All three verified in source:

1. **The version gate is strict equality.** `valid_image_success` rejects on
   `marker.get("version") != SUCCESS_MARKER_VERSION`, and Task 3.8 bumps that constant to 2.
   After migration, **every** finished image in **every** legacy tree is unknown-to-complete.
2. **Without the bump it is worse, not better.** A v1 marker keeps validating against the
   retained `.h5` (`keep_source=True`) — asserting completeness for an artifact the forward
   path no longer reads. So the bump is right; but Task 3.8's stated rationale ("without the
   bump those markers are read and fail validation") is **inverted**, and neither branch is
   actually resolved by the bump alone. Fix that rationale in Task 3.8.
3. **The one bridge that exists cannot handle a store.**
   `refresh_success_markers_after_metadata_migration` **hard-`raise`s**
   `RuntimeError("Success marker artifact is missing")` when `not artifact.is_file()`, and
   skips markers whose version differs before any descriptor is read.

**Consequence if unaddressed:** Task 5.2's own `test_the_aggregate_publication_survives_migration`
**cannot pass** — `aggregate_publication_is_valid` compares `source_set_digest`, computed
from `valid_image_success`, which is `False` for every image — and the first post-migration
run re-processes and re-finalizes the entire tree.

**Depends on:** Phase 3 Task 3.8 (the `kind`-tagged descriptors). See the Phase 5 header note.

**Constraints specific to this task:**
- **Republication is keyed on MARKER state, not on conversion state** (ledger **FLOW-22**).
  Task 5.1 skips an image whose store already passes `valid_staged_store` — so trace an
  interruption: migration promotes image X's store, then the process dies before rewriting X's
  marker. On resume X is **skipped**, and if republication rides on "was converted this run"
  its marker is never rewritten. It stays v1 forever, and on the local path — the very gap
  FLOW-2(b) named — X is reprocessed from source inputs a migrated archive may no longer have.
  Republish for every image whose store is valid **and that already has a marker** which does
  not yet describe it. Republication **rewrites; it never creates** — that one clause is what
  keeps this rule from contradicting the markerless-tree no-op below (ledger **FLOW-37**): a
  *missing* marker also "does not describe the store", so the looser wording fired on every
  image of a pre-markers archive, where `publish_image_success` has no `work_id`,
  `attempt_id`, or `lifecycle_epoch` to be given (it does **not** short-circuit on
  `success_markers_required`, unlike its three siblings). The operation is idempotent, so
  running it over skipped images costs a marker read.
- **Republication REPLACES the artifact set; it does not add to it** (ledger **MIG-22**). The
  post-condition is `artifacts == {"measurements": …, "zarr": …}` — assert the whole mapping,
  not just that `artifacts["zarr"]["kind"] == "store"`. If it merely adds, the stale `"hdf"`
  descriptor still validates under the default `keep_source=True` and hides the defect
  entirely. Conversely `"measurements"` must be **preserved verbatim under that literal key**:
  `_current_success_work_ids` (`_cli_completion.py:475`) indexes it by name.
- **One test must drive both passes through the CLI** (ledger **MIG-15(c)** /
  **MIG-24**). Every other test in this task calls `migrate_run_hdf_to_zarr` directly, which
  is **pass 2 only** — pass 1 lives in the CLI driver, so the interaction that MIG-15 is
  about (pass 1 rewriting the parquets that pass 2's markers fingerprint) is exercised by
  nothing. In `tests/integration/cli/test_migrate_end_to_end.py`:

  ```python
  def test_a_full_migrate_leaves_the_run_valid_and_idle(
      finished_legacy_run: LegacyRun
  ) -> None:
      """Both passes, in order, through the real entry point.

      This is the test MIG-15 predicts will fail against a plan that runs the
      image pass first: the marker republication would fingerprint parquets that
      the non-image pass then rewrites.
      """
      from click.testing import CliRunner

      from phenotypic._cli._cli_completion import (
          aggregate_publication_is_valid,
          valid_image_success,
      )
      from phenotypic.phenotypicCLI import phenotypic_cli

      tree = finished_legacy_run.path
      result = CliRunner().invoke(
          phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
      )
      assert result.exit_code == 0

      for stem in finished_legacy_run.stems:
          assert valid_image_success(
              tree, dataset="ds", image_stem=stem,
              work_id=finished_legacy_run.work_id,
          )
      assert aggregate_publication_is_valid(tree) is True

      # And the migrated tree does no work on the next full run.
      second = CliRunner().invoke(phenotypic_cli, finished_legacy_run.full_run_args())
      assert second.exit_code == 0
      assert "0 images" in second.output or "complete" in second.output
  ```
- **A legacy tree with no markers is a documented no-op, not an exception** (ledger
  **MIG-23**). `refresh_success_markers_after_metadata_migration` and
  `_current_success_work_ids` short-circuit when `state.config["success_markers_required"]` is
  falsey (`_cli_completion.py:165-172`, `:380-383`), but `publish_aggregate_snapshot`
  **raises** `RuntimeError` when state is missing or no markers are authorized (`:504-512`) and
  resolves the four deliverables paths with `strict=True`. A pre-markers archive is a likely
  migration subject; aborting there would leave the stores written and the run reported as
  failed. Guard the aggregate republish and report zero.
- Re-publish each converted image's marker at version 2 with a `kind: "store"` descriptor,
  **preserving `work_id`, `attempt_id`, and `lifecycle_epoch`** — those identify the run that
  produced the result, and rewriting them would falsely re-attribute it.
- Then re-publish the aggregate marker so `finalization_input_digest` and `source_set_digest`
  both describe the migrated tree.
- Extend the `is_file()` `raise` to dispatch on `kind`, mirroring `valid_image_success`.
- **Stage-3 completion markers need no work** — verified: `migrate_legacy_stage3_markers`
  (`_cli_staged_resume.py:287-311`) regenerates them from **parquet presence** (`:295-303`),
  not from the image artifact, so they self-heal on the next run.

- [ ] **Step 1: Write the failing test**

```python
"""Migration must leave the run's published state valid, not just its pixels."""

from __future__ import annotations

from pathlib import Path


def test_every_image_still_validates_after_migration(finished_legacy_run: LegacyRun) -> None:
    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    work_id = finished_legacy_run.work_id
    migrate_run_hdf_to_zarr(finished_legacy_run.path)
    for stem in finished_legacy_run.stems:
        assert valid_image_success(
            finished_legacy_run.path, dataset="ds", image_stem=stem, work_id=work_id
        ) is True


def test_the_aggregate_publication_survives_migration(finished_legacy_run: LegacyRun) -> None:
    """The test Task 5.2 already has, which cannot pass without this task."""
    from phenotypic._cli._cli_completion import aggregate_publication_is_valid
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(finished_legacy_run.path)
    assert aggregate_publication_is_valid(finished_legacy_run.path) is True


def test_work_id_and_epoch_are_preserved(finished_legacy_run: LegacyRun) -> None:
    """Rewriting them would falsely re-attribute the result to the migration."""
    import json

    from phenotypic._cli._cli_completion import image_completion_marker_path
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    stem = finished_legacy_run.stems[0]
    before = json.loads(
        image_completion_marker_path(finished_legacy_run.path, "ds", stem).read_text()
    )
    migrate_run_hdf_to_zarr(finished_legacy_run.path)
    after = json.loads(
        image_completion_marker_path(finished_legacy_run.path, "ds", stem).read_text()
    )
    for key in ("work_id", "attempt_id", "lifecycle_epoch"):
        assert after[key] == before[key], key
    assert after["version"] == 2
    assert after["artifacts"]["zarr"]["kind"] == "store"


def test_a_migrated_run_does_no_work_on_the_next_full_run(
    finished_legacy_run: LegacyRun,
) -> None:
    """The end-to-end consequence: migration must not cause reprocessing."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(finished_legacy_run.path)
    result = CliRunner().invoke(phenotypic_cli, finished_legacy_run.full_run_args())
    assert result.exit_code == 0
    assert "0 images" in result.output or "complete" in result.output.lower()
```

`finished_legacy_run` is a new fixture: a **completed** legacy run produced by the real
staged pipeline (so its markers, `work_id`, and aggregate are genuine), exposing `path`,
`work_id`, `stems`, and `full_run_args()`.

- [ ] **Step 2: Run to verify it fails.** Expected: `valid_image_success` returns `False` for
every image, and `aggregate_publication_is_valid` is `False`.

- [ ] **Step 3: Implement** the marker rewrite in `migrate_run_hdf_to_zarr`, plus the `kind`
dispatch in `refresh_success_markers_after_metadata_migration`.

- [ ] **Step 4: Re-run**, including Task 5.2's `test_migration_keeps_the_published_aggregate_valid`,
which should now pass for the first time.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_hdf_to_zarr.py src/phenotypic/_cli/_cli_completion.py tests/unit/sdk_/test_migration_republishes_state.py
git commit -m "fix(migrate): re-publish run state, not only pixels

--mode migrate converted images but left every per-image completion marker
describing a vanished .h5 at a version the loader rejects on strict
equality -- so after migration every finished image read as
unknown-to-complete and the next run re-processed and re-finalized the
whole tree. Task 5.2's own aggregate-validity test could not pass.

Markers are re-published at version 2 with a kind:'store' descriptor,
preserving work_id/attempt_id/lifecycle_epoch, and the refresh bridge now
dispatches on kind instead of hard-raising on a non-file artifact.
Stage-3 markers need no work: they regenerate from parquet presence."
```

---

### Task 5.7: One migration predicate, applied everywhere

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (the predicate)
- Modify: `src/phenotypic/phenotypicCLI.py` (mode validation, replacing the "only `.h5`" check)
- Modify: `src/phenotypic/gui/results_viewer/_output_consistency.py` (one new reason)
- Test: `tests/unit/sdk_/test_migration_predicate.py`, plus a GUI case

**Why this task exists** (ledger MIG-8, user-approved): the spec's guard was "output contains
**only** `.h5` results fails with a pointer", tested through `--mode recompile` alone. But
migration is explicitly resumable, so a **half-migrated** tree is the expected state after any
interruption — and it is neither "only `.h5`" nor fully converted, so it passed the guard.
`--mode full` would then silently reprocess every unconverted image from source. After Phase 4
the GUI has no detection at all: it discovers the tree, lists every image, and resolves
unconverted ones to `None`, rendering silently empty.

**The wiring is small because the GUI surface already exists** — this is an added case in an
existing classifier, not a new component:

```python
def datasets_needing_migration(output_dir: Path) -> list[str]:
    """Datasets holding at least one `.h5` result without a VALID store.

    One predicate, so the CLI and the GUI cannot disagree about what
    "needs migrating" means.

    Per-IMAGE, not per-dataset: the half-migrated tree this exists to catch
    has converted and unconverted images in the SAME dataset, so a
    dataset-level "has .h5 and has no zarr/ dir" test misses it entirely.

    Validity, not existence: `valid_staged_store`, not `path.exists()`. A
    store written at an older `store_schema_version` is present but the
    loader refuses it (MIG-4 gates by value), so an existence test reads
    that tree as clean while every image fails to open.
    """
```

**Constraints specific to this task:**
- Apply it to **every mode that consumes results**, not `recompile` alone — and **`migrate`
  itself is exempt** (ledger **MIG-19**). It is the remedy; guarding it with its own predicate
  makes the tree unmigratable.
- **Split the consumer treatment, keep the one predicate** (ledger **MIG-19**). A mode that
  writes or reprocesses (`full`, `measure`, `recompile`, `process`) **refuses**, because after
  Phase 6 the forward path genuinely cannot read those images. The **viewer is informational,
  not danger**: a half-migrated tree's deliverables, measurements, and dashboards are all
  still readable, and the images that are missing are precisely the ones it would otherwise
  render empty. Same predicate, same reason text, different severity — the reason string
  carries the remedy (`--mode migrate`), so a user reading the banner knows what to run.
- The GUI path is one new reason on `OutputConsistencyReport`
  (`gui/results_viewer/_output_consistency.py`), which `gui/_snapshot_status.py:74-85`
  **already renders** as a danger banner with the reason text. `inspect_output_consistency`
  is already called from `OutputRoot.discover` and three other sites. No new component, no new
  callback, no new layout.
- The message names the fix: `this output needs --mode migrate`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_half_migrated_tree_is_detected(half_migrated_run: Path) -> None:
    """The expected state after any interruption -- migration is resumable."""
    from phenotypic.sdk_ import datasets_needing_migration

    assert datasets_needing_migration(half_migrated_run) == ["ds"]


def test_a_fully_migrated_tree_is_clean(migrated_run: Path) -> None:
    from phenotypic.sdk_ import datasets_needing_migration

    assert datasets_needing_migration(migrated_run) == []


def test_full_mode_refuses_a_half_migrated_tree(half_migrated_run, tmp_path) -> None:
    """Without this, --mode full silently reprocesses from source.

    ``--pipeline`` and ``--input`` are ``click.Path(exists=True)``
    (``phenotypicCLI.py:914``, ``:926``), so both must exist on disk or click
    exits 2 while parsing and the migration guard never runs.
    """
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    pipeline = tmp_path / "p.json"
    pipeline.write_text("{}", encoding="utf-8")
    images = tmp_path / "imgs"
    images.mkdir()

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "full", "--output", str(half_migrated_run),
         "--pipeline", str(pipeline), "--input", str(images)],
    )
    assert result.exit_code != 0
    assert result.exit_code != 2, "must fail in the migration guard, not click parsing"
    assert "--mode migrate" in result.output


def test_the_viewer_surfaces_it(half_migrated_run: Path) -> None:
    """Reported through the EXISTING consistency surface, not a new one."""
    from phenotypic.gui.results_viewer._output_consistency import (
        inspect_output_consistency,
    )
    from phenotypic.sdk_ import BundleLayout

    report = inspect_output_consistency(BundleLayout(
        deliverables_base=deliverables_dir(half_migrated_run),
        output_root=half_migrated_run,
    ))
    assert any("--mode migrate" in reason for reason in report.reasons)
```

- [ ] **Step 2-4:** run to verify failure, implement, re-run including `tests/gui`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_ src/phenotypic/phenotypicCLI.py src/phenotypic/gui tests
git commit -m "feat: one migration predicate, applied to every consumer

The old guard fired only when a tree was ENTIRELY .h5, tested through
recompile alone -- so a half-migrated tree, which is the expected state
after any interruption of a resumable migration, passed it and --mode full
silently reprocessed every unconverted image from source. The GUI had no
detection at all and rendered unconverted images as empty.

datasets_needing_migration() is now the single definition, applied to every
mode that consumes results. The viewer reports it as a reason on the
existing OutputConsistencyReport, which _snapshot_status already renders --
no new component."
```

---

## Phase 5 exit criteria

- [ ] Green:
      ```bash
      uv run pytest \
        tests/unit/sdk_/test_hdf_to_zarr.py \
        tests/unit/sdk_/test_metadata_canonical_view.py \
        tests/unit/sdk_/test_migration_republishes_state.py \
        tests/unit/sdk_/test_migration_predicate.py \
        tests/unit/cli/test_cli_migrate_mode.py \
        tests/unit/cli/test_recompile_no_longer_migrates.py \
        tests/integration/cli/test_migrate_end_to_end.py -q
      ```
      > **Regenerated from the current task list (ledger GEN-25).** The previous block named
      > `test_metadata_csv_migration.py` (renamed by Task 5.2) and
      > `test_header_only_migration.py` (whose task, 5.5, was cut), and covered neither Task
      > 5.6 nor Task 5.7.
- [ ] `uv run python -m phenotypic --mode migrate --output <a real legacy run> --dry-run`
      reports a non-zero conversion count for each pass and writes nothing.
- [ ] Running migrate twice on the same tree converts zero on the second run **and migrates
      zero headers**. The second half holds because an already-canonical bundle short-circuits
      to `_compatible_result` (`_metadata_migration.py:2318-2319`), **not** because of any
      skip — an earlier draft credited the MIG-21 skip, which would have made the criterion
      pass either way and so gate nothing (ledger **MIG-31**).
- [ ] `grep -rn "sbatch" src/phenotypic/_cli/_cli_migrate.py src/phenotypic/sdk_/_hdf_to_zarr.py` returns nothing.
- [ ] The six golden fixtures — `v1_flat`, `v2_grouped`, `v2_enh_gray`, `v2_grid`,
      `v2_image_type`, `v2_work_id` — are committed with their generator, and Task 5.1
      Step 1a's generator-fidelity check passes **for both writer paths** (`Image` via
      `write_v2_grouped` and `GridImage` via `write_v2_grid`).

      > **Names corrected (ledger SIMP-20).** The previous wording named `v2_rich` "plus the
      > two negative cases" — artifacts from a fixture-merge proposal (SIMP-14) that was
      > considered and **declined** in the same round. `v2_rich` appeared exactly once in the
      > whole plan, in this criterion. The count was right and every name was wrong, which is
      > the worst combination for a checklist.
- [ ] After a full migrate, `aggregate_publication_is_valid(<tree>)` is `True` and
      `--mode full` on the same tree does no work.
- [ ] `grep -rn "migrate_metadata_schema\b" docs/ src/` returns nothing — the symbol does not
      exist (ledger **MIG-15**).
