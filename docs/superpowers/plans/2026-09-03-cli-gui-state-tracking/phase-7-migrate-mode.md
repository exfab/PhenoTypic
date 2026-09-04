# Phase 7 — `--mode migrate` conversion, and refusal everywhere else

**Depends on:** P1–P6. **Blocks:** nothing.

**Spec:** D1 (clean break), §11.1 (legacy paths move *into* migrate), §15.1 (residual risk).

**Goal:** every existing tree can be converted, and every mode that is not `migrate`
refuses an unconverted one with a pointer to the command that fixes it.

> **§15.1: "The migrate step is the riskiest part of this design."** It rewrites machine
> state across the whole tree and, unlike the rest of the change, **cannot be rolled back by
> reverting code.** It needs the receipt/rollback discipline the existing metadata migration
> has, plus its own dry-run mode.

That risk is materially lower than the spec assessed, because
[D-A](OPEN-QUESTIONS.md#d-a-per-store-metadata-is-written-at-promote-time-not-backfilled)
removed the store rewrite: migrate converts `.phenotypic/` — markers, state, identity — and
**does not touch a single byte inside any `results/**/*.ome.zarr`**. A converted tree's
stores are the stores it already had.

---

## What migrate converts

| From | To | Task |
|---|---|---|
| `image_complete/<ds>/<stem>.json` (v2 marker) | `images/<ds>/<stem>.json` record, `stages={"measured": …}` | 2 |
| `stage2_done/<ds>/<stem>.json` | `stages.stage2` in the same record | 2 |
| `stage3_complete/<ds>/<stem>.json` | `stages.stage3` in the same record | 2 |
| `processing_generation: <uuid4>` | content-derived generation; `restart_epoch: 0` | 3 |
| `processing_state.datasets.{completed,failed,started}` | **deleted from the file** (§4.2) | 3 |
| `slurm_generation` / `lifecycle_epoch` | `scheduler_epoch` | 3 |
| joined embedded tables | **left alone** — see Task 4 | 4 |
| `master_measurements.csv` | deleted; master is parquet-only (D8) | 4 |
| legacy `.h5` per-image files | unchanged — the existing OME-Zarr migration already owns this | — |

**What migrate does NOT do:** re-mint `work_id` (D-C keeps the digest unchanged, so every
existing `work_id` stays valid), rewrite `deliverables/metadata.csv` (project `CLAUDE.md`,
spec D9/FLOW-4 — there is **no exception, including migrate**), or write into any store.

---

## Task 1: Detection and refusal

**Files:**
- Create: `src/phenotypic/_cli/_cli_schema_gate.py`
- Modify: `src/phenotypic/phenotypicCLI.py`
- Test: `tests/unit/cli/test_schema_gate.py` *(new)*

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("mode", ["full", "measure", "recompile", "process"])
def test_every_writing_mode_refuses_an_unconverted_tree(tmp_path, mode):
    """D1: clean break. New code reads only the consolidated schema, so a mode that
    silently half-read a legacy tree would produce a run whose proofs certify
    nothing."""
    import pytest

    _build_legacy_tree(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _invoke_cli(mode=mode, output=tmp_path)
    assert "--mode migrate" in str(exc.value), (
        "the refusal must name the command that fixes it; a refusal the user cannot "
        "act on is the bug class this whole change exists to remove"
    )


def test_a_converted_tree_is_accepted(tmp_path):
    _build_legacy_tree(tmp_path)
    _invoke_cli(mode="migrate", output=tmp_path)
    _invoke_cli(mode="full", output=tmp_path)      # must not raise


def test_a_fresh_output_directory_is_not_an_unconverted_tree(tmp_path):
    """An empty directory has no schema to be wrong about. Refusing one would make
    every new run start with an error."""
    _invoke_cli(mode="full", output=tmp_path / "brand-new")


def test_the_gui_reports_rather_than_refuses(tmp_path):
    """§4.3: a half-migrated tree is an ADVISORY, not a gate. The GUI is a reader;
    refusing to display a legacy tree would be a regression from today, where it
    displays one."""
    from phenotypic.sdk_ import resolve_run_state

    _build_legacy_tree(tmp_path)
    state = resolve_run_state(tmp_path, depth="deep")
    assert any("migrate" in advisory for advisory in state.advisories)
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

```python
STATE_SCHEMA_VERSION = 3


def requires_conversion(output_dir: Path) -> str | None:
    """Return why *output_dir* needs `--mode migrate`, or ``None``.

    Detection is by **presence of the old shape**, never by absence of the new one:
    an empty directory and a half-written new run both lack ``images/``, and only one
    of them is a legacy tree. The signals, in order of certainty:

    1. ``.phenotypic/progress/image_complete/`` exists
    2. ``.phenotypic/progress/stage2_done/`` or ``stage3_complete/`` exists
    3. ``processing_state.json`` carries ``datasets.completed`` (deleted by §4.2)
    4. ``processing_state.json`` has no ``restart_epoch`` **and** has ``work_ids``

    Returns:
        A message naming the specific evidence and the command, or ``None``.
    """
```

Wire it into `phenotypicCLI.py` for `full`, `measure`, `recompile` and `process`. **Not**
for the GUI or any reader — spec §4.3 makes a half-migrated tree an advisory.

- [ ] **Step 4: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_schema_gate.py -v
git add src/phenotypic/_cli/_cli_schema_gate.py src/phenotypic/phenotypicCLI.py \
        tests/unit/cli/test_schema_gate.py
git commit -m "feat(cli): refuse an unconverted tree in every writing mode

D1. Detection is by presence of the old shape, not absence of the new -- an empty
directory is not a legacy tree. Readers get an advisory, not a refusal (§4.3)."
```

---

## Task 2: Convert the per-image markers

**Files:**
- Modify: `src/phenotypic/_cli/_cli_migrate.py`
- Create: `src/phenotypic/_cli/_cli_migrate_state.py`
- Test: `tests/unit/cli/test_migrate_state.py` *(new)*

- [ ] **Step 1: Write the failing tests**

```python
def test_three_markers_become_one_record(tmp_path):
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers
    from phenotypic.sdk_ import image_record_path

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a",
                          image_complete=True, stage2_done=True, stage3_complete=True)
    convert_per_image_markers(tmp_path)

    record = json.loads(image_record_path(tmp_path, "plate", "a").read_text())
    assert set(record["stages"]) == {"stage2", "stage3", "measured"}
    assert record["work_id"] == "w"


def test_artifact_descriptors_survive_conversion_byte_for_byte(tmp_path):
    """The descriptors are the content proof. Re-deriving them during migration
    would certify whatever is on disk NOW, including a corrupted artifact -- which
    turns migrate from a format change into a laundering step."""
    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    before = json.loads(_legacy_marker_path(tmp_path, "plate", "a").read_text())["artifacts"]

    from phenotypic._cli._cli_migrate_state import convert_per_image_markers

    convert_per_image_markers(tmp_path)
    after = json.loads(image_record_path(tmp_path, "plate", "a").read_text())["artifacts"]
    assert after == before


def test_conversion_is_idempotent(tmp_path):
    """Re-running after an interruption is the recovery procedure -- the CLI's
    documented migrate contract."""
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    convert_per_image_markers(tmp_path)
    first = image_record_path(tmp_path, "plate", "a").read_bytes()
    convert_per_image_markers(tmp_path)
    assert image_record_path(tmp_path, "plate", "a").read_bytes() == first


def test_a_stage2_token_with_no_image_complete_still_converts(tmp_path):
    """Stage 2 finished and Stage 3 never ran -- a real interrupted-run state, and
    the one a naive 'iterate image_complete/' conversion drops on the floor."""
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers
    from phenotypic.sdk_ import image_record_path

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a",
                          image_complete=False, stage2_done=True)
    convert_per_image_markers(tmp_path)
    record = json.loads(image_record_path(tmp_path, "plate", "a").read_text())
    assert set(record["stages"]) == {"stage2"}


def test_the_legacy_trees_are_removed_only_after_every_record_is_written(tmp_path):
    """Marker-last, applied to the migration itself. A conversion that deletes as it
    goes and then dies leaves a tree that is neither shape."""
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    _plant_legacy_markers(tmp_path, dataset="plate", stem="b", image_complete=True)
    with _fail_after_n_records(1):
        with pytest.raises(RuntimeError):
            convert_per_image_markers(tmp_path)
    assert _legacy_marker_path(tmp_path, "plate", "a").exists(), (
        "the legacy tree was removed before conversion completed"
    )
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

Enumerate the union of all three legacy trees, not just `image_complete/` — an image with
a stage-2 token and no completion marker is a real interrupted state. Write every record
first, then remove the three legacy trees. **Copy `artifacts` verbatim**; never re-derive.

- [ ] **Step 4: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_migrate_state.py -v
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "feat(cli): migrate converts the three marker trees into one record

Spec §6.1, §11.1. Artifact descriptors are copied verbatim -- re-deriving them would
certify whatever is on disk now, turning migrate into a laundering step."
```

---

## Task 3: Convert `processing_state.json`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_migrate_state.py`
- Test: `tests/unit/cli/test_migrate_state.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_uuid_generation_becomes_content_derived(tmp_path):
    from phenotypic._cli._cli_migrate_state import convert_processing_state

    _plant_legacy_state(tmp_path, processing_generation="deadbeef" * 4)
    convert_processing_state(tmp_path)
    config = _state_config(tmp_path)
    assert config["restart_epoch"] == 0
    assert config["processing_generation"] != "deadbeef" * 4
    assert config["processing_generation"] == _expected_generation(config)


def test_the_derived_dataset_counts_are_removed(tmp_path):
    """§4.2: processing_state.datasets.{completed,failed,started} is DELETED from
    the file -- it was already re-aggregated from the event log on every load
    (_cli_state_management.py:121), a cache of a cache."""
    from phenotypic._cli._cli_migrate_state import convert_processing_state

    _plant_legacy_state(tmp_path)
    convert_processing_state(tmp_path)
    for dataset in _state(tmp_path)["datasets"].values():
        assert not {"completed", "failed", "started"} & set(dataset)


def test_work_ids_are_untouched(tmp_path):
    """D-C keeps processing_configuration_digest unchanged, so every existing
    work_id stays valid. Re-minting them would invalidate every marker migrate just
    converted, and a tree half-migrated across that boundary is unrecoverable
    without the original config."""
    from phenotypic._cli._cli_migrate_state import convert_processing_state

    _plant_legacy_state(tmp_path)
    before = _state_config(tmp_path)["work_ids"]
    convert_processing_state(tmp_path)
    assert _state_config(tmp_path)["work_ids"] == before


def test_the_metadata_snapshot_is_byte_unchanged_by_a_full_migrate(tmp_path):
    """Project CLAUDE.md, spec D9/FLOW-4. There is NO exception, including migrate.

    An earlier draft of this rule carved one out -- migrate would rewrite
    deliverables/metadata.csv with canonical headers after copying the original to
    metadata.original.csv. That was WITHDRAWN and never implemented: a snapshot that
    is sometimes rewritten is not provenance, and 'the original is recoverable over
    there' is a weaker guarantee than 'the bytes you supplied are still the bytes on
    disk'. metadata.original.csv does not exist and must not be created."""
    _plant_legacy_tree(tmp_path, metadata=b"Well,Strain\r\nA1,BY4741\r\n")
    before = (tmp_path / "deliverables" / "metadata.csv").read_bytes()
    _invoke_cli(mode="migrate", output=tmp_path)
    assert (tmp_path / "deliverables" / "metadata.csv").read_bytes() == before
    assert (tmp_path / "deliverables" / "metadata.canonical.csv").is_file()
    assert not (tmp_path / "deliverables" / "metadata.original.csv").exists()
```

- [ ] **Step 2: Run to verify failure, implement, re-run.**

- [ ] **Step 3: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "feat(cli): migrate converts processing_state.json to the v3 schema

Spec §4.2, §5.1. work_ids are untouched (D-C); metadata.csv is byte-identical, with
the canonical view emitted alongside it."
```

---

## Task 4: The embedded-table question, and the master's schema stamp

**Files:**
- Modify: `src/phenotypic/_cli/_cli_migrate_state.py`
- Test: `tests/unit/cli/test_migrate_state.py`

Existing trees have embedded tables that are **already metadata-joined** — the shape P4
inverted. Two options, and the plan takes the cheaper one:

**Decision: migrate leaves embedded tables alone and stamps the master.**

Rewriting every store's embedded table would reintroduce exactly the post-proof store
mutation D-A removed — the hardlink re-promote, the receipts, all of it — for trees that
already have a correct master. Instead, `finalize_run`'s step 1 projects each embedded
table onto its recorded `phenotypic.measurement_columns` before concatenating. That
projection is free, it is the same boundary P4 uses, and the column list is already written
into every existing store.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_legacy_joined_embedded_table_projects_to_measurements_only(tmp_path):
    """The projection uses the store's own recorded measurement_columns, so a
    pre-inversion store aggregates to the same master a post-inversion one does --
    with no store rewrite."""
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import master_measurements_parquet_path

    _build_legacy_tree_with_joined_embedded_tables(tmp_path)
    _invoke_cli(mode="migrate", output=tmp_path)
    finalize_run(tmp_path, dataset_names=["plate"])

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    assert "Metadata_Strain" not in master.columns
    assert "Metadata_ImageFile" in master.columns


def test_migrate_writes_no_byte_into_any_store(tmp_path):
    """D-A, applied to migrate. This is what made §15.1's 'riskiest part' claim
    smaller: migrate converts .phenotypic/ and nothing else."""
    _build_legacy_tree_with_joined_embedded_tables(tmp_path)
    stores = sorted((tmp_path / "results").rglob("*.ome.zarr"))
    before = {
        p: p.stat().st_mtime_ns
        for store in stores for p in sorted(store.rglob("*")) if p.is_file()
    }
    _invoke_cli(mode="migrate", output=tmp_path)
    after = {
        p: p.stat().st_mtime_ns
        for store in stores for p in sorted(store.rglob("*")) if p.is_file()
    }
    assert before == after


def test_the_master_carries_a_schema_version_an_old_reader_fails_on(tmp_path):
    """§7.3: 'anything filtering master on a user-metadata column would return EMPTY
    rather than error. This is the one genuinely dangerous failure mode in §7, and
    it is why the migrate step must tag the master with a schema version so an old
    reader fails loudly.'"""
    import pyarrow.parquet as pq

    from phenotypic.sdk_ import master_measurements_parquet_path

    _build_legacy_tree(tmp_path)
    _invoke_cli(mode="migrate", output=tmp_path)
    schema = pq.read_schema(master_measurements_parquet_path(tmp_path))
    assert schema.metadata[b"phenotypic.master_schema_version"] == b"2"
```

- [ ] **Step 2: Implement, run, commit**

```bash
git add -A src/phenotypic tests/unit/cli
git commit -m "feat(cli): project legacy embedded tables instead of rewriting them

D-A applied to migrate: not one byte is written into any store. finalize_run
projects each table onto its own recorded measurement_columns, so a pre-inversion
store aggregates identically. The master is stamped master_schema_version=2 so a
reader that filters on a user-metadata column fails loudly instead of silently
returning nothing (§7.3)."
```

---

## Task 5: Dry-run and rollback — the phase gate

**Files:**
- Modify: `src/phenotypic/_cli/_cli_migrate_state.py`, `phenotypicCLI.py`
- Test: `tests/unit/cli/test_migrate_state.py`, `tests/integration/`

§15.1 requires "the receipt/rollback discipline the existing metadata migration has, plus
its own dry-run mode".

- [ ] **Step 1: Write the failing tests**

```python
def test_dry_run_reports_and_writes_nothing(tmp_path):
    _build_legacy_tree(tmp_path)
    before = _tree_fingerprint(tmp_path)
    report = _invoke_cli(mode="migrate", output=tmp_path, dry_run=True)
    assert _tree_fingerprint(tmp_path) == before
    assert "2 records" in report and "processing_state.json" in report


def test_an_interrupted_migration_is_resumable_not_corrupt(tmp_path):
    """Re-running after an interruption is the recovery procedure. The tree between
    attempts must be readable by SOMETHING -- either the old shape or the new one,
    never neither."""
    _build_legacy_tree(tmp_path)
    with _fail_midway():
        with pytest.raises(RuntimeError):
            _invoke_cli(mode="migrate", output=tmp_path)
    assert _tree_is_readable_by_one_schema(tmp_path)
    _invoke_cli(mode="migrate", output=tmp_path)
    _invoke_cli(mode="full", output=tmp_path)


def test_the_pre_existing_metadata_receipt_path_still_raises_on_uncertified_drift(tmp_path):
    """INV-IMMUTABLE's one exception, kept scoped. D-A cut §6.4's GENERALISATION,
    not refresh_success_markers_after_metadata_migration itself -- it serves a real
    historical case and it keeps its RuntimeError."""
    import pytest

    from phenotypic._cli._cli_completion import (
        refresh_success_markers_after_metadata_migration,
    )

    _build_tree_with_uncertified_artifact_drift(tmp_path)
    with pytest.raises(RuntimeError):
        refresh_success_markers_after_metadata_migration(tmp_path)
```

- [ ] **Step 2: Implement, run.**

- [ ] **Step 3: Phase gate — a real tree**

Run `--mode migrate` against a **real legacy output tree on GPFS**, not a fixture, via the
**`slurm-job`** skill. Then:

```bash
uv run python -m phenotypic --mode full --output <migrated> ...   # must resume, not restart
uv run phenotypic-gui --root <migrated>                            # must bind and display
```

Confirm the migrated run resumes without reprocessing a single image — if it reprocesses,
`work_id` changed somewhere and D-C was violated.

- [ ] **Step 4: Full suite and docs**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix <every path this phase touched>
```

Full `tests/unit` + `tests/gui` as a Slurm job, compared against the recorded baseline of
four known pre-existing failures.

Update:
- `CLAUDE.md`'s `--mode migrate` bullet — it now also converts run-state schema v2 → v3
- `docs/source/how_to/pages/migrate_ome_zarr.md`
- `src/phenotypic/_cli/CLAUDE.md`'s master-vs-mirror rules, if P4 left anything

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cli): migrate gains a dry run and resumable conversion

Spec §15.1. Verified end to end on a real legacy tree on GPFS: migrate, then a full
run that resumes without reprocessing an image, then a GUI bind. The pre-existing
metadata receipt path keeps its RuntimeError for uncertified artifact drift --
INV-IMMUTABLE's one exception, still scoped to migrate."
```

---

## Closing the change

- [ ] **Final: confirm the headline claims, with numbers**

Do not write any of these from the spec. Measure each:

```bash
git diff --stat main...HEAD | tail -1                  # net lines
grep -c '' src/phenotypic/gui/results_viewer/_output_consistency.py 2>/dev/null || echo "deleted"
```

- **9 sources → 3 authorities** — `grep -rn 'resolve_run_state' src/ | wc -l` versus the
  count of files that previously read a completion source.
- **14 tokens → 6** — list the six and grep that nothing else reaches disk as identity.
- **~1,400 lines deleted** — the real `git diff --stat`, not the estimate.
- **The completion predicate's cost** — re-run spike S-5 against a migrated tree and quote
  the before/after.

State the measured numbers in the PR body. A claim in a PR that came from the spec rather
than from the tree is the same class of error this change spent seven phases removing.
