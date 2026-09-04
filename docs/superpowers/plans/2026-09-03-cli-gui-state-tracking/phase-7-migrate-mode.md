# Phase 7 — `--mode migrate` conversion, and refusal everywhere else

**Depends on:** P1–P6. *(Task 1's `requires_conversion` is BUILT IN P1 — see CAN-11. It is specified here in full and referenced from there.)* **Blocks:** nothing.

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
| **pre-markers tree** (`success_markers_required` absent, `version="2.0.0"`) — **the v0.17.3 floor** | `_migrate_legacy_success_evidence` **ported into migrate**, sequenced after the HDF→Zarr conversion and before the migrator's own publisher. It is the only producer of the *content-derived* `work_id` a later resume re-derives. See Task 2b. | 2b |
| `processing_generation: <uuid4>` **and** the migrator's inventory-derived one | content-derived generation; `restart_epoch: 0` | 3 |
| `processing_state.datasets.{completed,failed,started}` | **deleted from the file** (§4.2) | 3 |
| `slurm_generation` / `lifecycle_epoch` | `scheduler_epoch` | 3 |
| joined embedded tables | **left alone**, projected at read — see Task 4 | 4 |
| `master_measurements.csv` | deleted; master is parquet-only (D8) | **4, Step 0** |
| `deliverables/metadata.canonical.csv` | **emitted** alongside the untouched snapshot | **3, Step 4** |
| legacy `.h5` per-image files | unchanged — the existing OME-Zarr migration already owns this | — |
| anything below **v0.17.3** | **refused** with a version string and a pointer | 1 |

> **Two rows in the first draft's table had no implementing step (CAN-32):** the CSV
> deletion was assigned to Task 4, which had no step for it, and Task 3 asserted
> `metadata.canonical.csv` exists while no task built it. Both now name a step.

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

- [ ] **Step 3b: Add the v0.17.3 floor as a third outcome (U-1)**

`requires_conversion` currently has two outcomes — convert, or don't. **Migration has no
lower bound at all**, which is an open-ended compatibility obligation nobody scoped. U-1
bounds it: **v0.17.3**, verified to predate both the marker schema (`379acee4`, 2026-08-17)
and OME-Zarr, writing `version="2.0.0"` with no `success_markers_required`.

> **`state.version` cannot express this floor (MIG-14) — see the gated question below.**
> Verified: `version="2.0.0"` is the value at v0.17.3 **and** still the value immediately
> before `379acee4` introduced `"3.0.0"`. It is a *state-schema* version, not a *package*
> version, and the two are not in bijection. The earlier draft's test planted
> `version="1.0.0"`, a value **no tree has ever carried**.
>
> What the on-disk state *can* express is the **pre-markers shape**: schema `2.0.0` with
> **no `work_ids` key at all** — the concept did not exist at v0.17.3. That is a reliable
> signal and it is what the code below uses.

**`requires_conversion` reads the raw JSON itself and never calls `load_processing_state`
(MIG-14b).** Two reasons, both disqualifying:

- `load_processing_state` calls `migrate_legacy_machine_state(output_dir)` at `:109` — **a
  write**. A refusal gate that mutates the tree before refusing it is worse than the silent
  path it replaces.
- It reads `state_dict[ProcessingStateKey.VERSION]` unguarded (`:167`), so an absent version
  raises `KeyError`, and `json.loads` at `:115` raises on a truncated file. **A gate that
  crashes on a malformed tree is worse than no gate.** Map absent, unparseable and malformed
  to explicit verdicts, never to an exception — INV-VERDICT's degrade half applies to the
  gate as much as to the reader.

```python
def requires_conversion(output_dir: Path) -> ConversionVerdict | None:
    """Return why *output_dir* needs `--mode migrate`, or ``None``.

    Three outcomes, not two:
      - ``None``                     -- already current
      - ``ConversionVerdict.CONVERT`` -- convertible; the message names the evidence
      - ``ConversionVerdict.BELOW_FLOOR`` -- older than v0.17.3, the supported floor
        (U-1). Refuse with the version string found and a pointer, rather than
        attempting a conversion whose inputs this build has never seen.
    """
```

```python
def test_a_malformed_state_file_yields_a_verdict_not_an_exception(tmp_path):
    """MIG-14b. A refusal gate that raises on a malformed tree is worse than the
    silent path it replaces, and one that mutates the tree while deciding is worse
    still -- which is why this reads raw JSON rather than load_processing_state."""
    import pytest

    for payload in ("{truncated", "null", "[]", '{"config": {}}'):
        _write_raw_state(tmp_path, payload)
        verdict = requires_conversion(tmp_path)       # must not raise
        assert verdict is not None
        assert not _tree_was_mutated(tmp_path), (
            "requires_conversion wrote to the tree while deciding whether to "
            "refuse it -- load_processing_state's migrate_legacy_machine_state "
            "side effect leaked in"
        )


def test_the_pre_markers_shape_is_detected_by_absent_work_ids(tmp_path):
    """MIG-14a. state.version cannot separate the floor: "2.0.0" is the value both
    at v0.17.3 and immediately before the marker commit. The reliable signal is the
    absent `work_ids` key -- the concept did not exist at v0.17.3."""
    _plant_legacy_state(tmp_path, version="2.0.0", work_ids=None)
    assert requires_conversion(tmp_path) is ConversionVerdict.CONVERT
```

- [ ] **Step 3c: Classify the three shapes that had no stated behaviour (CAN-32)**

Each gets a row in `requires_conversion`'s docstring and a test:

| Shape | Correct classification | Why it is not obvious |
|---|---|---|
| **Bundle-only** — `deliverables/` with no `.phenotypic/` | `None` (nothing to convert) | `BundleLayout.detect` explicitly supports it (`_io_constants.py:2468-2482`), and it trips none of the four signals. Its master stays unstamped, which under P4's stamp-at-finalize is the **correct** pre-v2 signal. |
| **modern `--mode process` tree** | `None` after P3 converts its records | Process **does** call `publish_image_success` (`_cli_process_single.py:789,943`), so signal 1 fires — but it has **no master at all**, so any master-touching step must no-op rather than raise. |
| **pre-markers `--mode process` tree** | **needs a process arm in migrate** — see below | `_cli_process_only.py` ships in v0.17.3, so this is *inside* the supported range. It has no `image_complete/`, so signal 1 does **not** fire; it trips signal 3 and classifies `CONVERT`. But migrate's discovery enumerates `.h5` and `results/<ds>/zarr/*.ome.zarr` (`_cli_migrate.py:611,1377`) — **a process-only run wrote neither.** Its outputs are process layers under the mirrored input tree. |

> **MIG-11: the pre-markers process tree is a hard regression as first written, and it is
> the same failure as CAN-7 in a shape the resolution did not enumerate.** Classified
> `CONVERT`, converted to nothing, so Task 2b Step 3's conditional deletion correctly
> declines to delete `datasets.*` — and the tree is then **refused forever**, since the next
> run re-classifies `CONVERT` and converts nothing again. Today `--mode process` converts it
> in place.
>
> The ported helper already has the arm: `phenotypicCLI.py:602-612` branches on
> `config.process_only_layer is not None` and publishes a `process_output` artifact resolved
> through `process_only_output_path`. **Porting it (Task 2b) is what fixes this too** — the
> two resolve together, because both need the same identity material. Do not add a second
> process arm to the migrator; route through the ported helper.
>
> Add a test that a pre-markers process tree converts and the next `--mode process` run
> processes zero images.
| **Interrupted migrate** | `CONVERT`, and the re-run completes it | Already correct; keep the test that says so. |

- [ ] **Step 4: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_schema_gate.py -v
git add src/phenotypic/_cli/_cli_schema_gate.py src/phenotypic/phenotypicCLI.py \
        tests/unit/cli/test_schema_gate.py
git commit -m "feat(cli): refuse an unconverted tree, and bound migration at v0.17.3

D1 + U-1. Detection is by presence of the old shape, not absence of the new -- an
empty directory is not a legacy tree. A third outcome refuses anything below the
floor by version string rather than attempting a conversion this build has never
seen inputs for. Readers get an advisory, not a refusal (§4.3).

NOTE: this task ships in P1, not P7 (CAN-11) -- the clean break lands in P3 and the
gate must precede it, or a legacy tree silently produces an empty master."
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
first, then **rename** the three legacy trees (Step 5). **Copy `artifacts` verbatim**;
never re-derive.

**Merge, do not overwrite (CAN-13).** When `images/<ds>/<stem>.json` already exists, union
the `stages` maps and keep the later `completed_at` rather than replacing. The
both-shapes-present case is real — see Step 6's coexistence window — and
`test_conversion_is_idempotent` cannot catch it, because after the first pass the legacy
trees are gone and the second call is a no-op.

```python
def test_conversion_merges_into_an_existing_record(tmp_path):
    """CAN-13. An old-build worker writing image_complete/ after a partial migrate
    must not clobber a newer record's stages."""
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers
    from phenotypic._cli._cli_image_record import read_image_record, record_stage

    record_stage(tmp_path, "plate", "a", "stage3", {"at": "new"})
    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    convert_per_image_markers(tmp_path)

    stages = read_image_record(tmp_path, "plate", "a")["stages"]
    assert {"stage3", "measured"} <= set(stages), "the newer record's stage3 was lost"
```

---

## Task 2b: Port the legacy promoter into migrate (CAN-7, U-1)

> ### Reversed in round 2 — read this before anything else
>
> **An earlier version of this task deleted `_migrate_legacy_success_evidence`**, on the
> reasoning that the HDF→Zarr migrator already mints identity and publishes records, so the
> pre-markers path worked end to end without it. **That was wrong**, and the migration
> specialist's `CONFLICT with CAN-7` was upheld on evidence:
>
> - `git show v0.17.3:src/phenotypic/_cli/_cli_state_management.py | grep -c work_ids`
>   returns **0**. The floor tree has no `work_ids` key at all — the concept did not exist.
> - So `_configured_work_id` (`_cli_migrate_image.py:125`) **always** falls through to
>   `_migration_work_id` = `sha256("migration:<ds>/<stem>")` — a **synthetic** id.
> - `_migrate_legacy_success_evidence` mints via `work_id_for_image(config, dataset.name,
>   image)` (`phenotypicCLI.py:590-592`) — the **content-derived** id, computed by the same
>   function a later resume uses.
>
> Two producers, two different id schemes, and **only the one I proposed deleting mints the
> id resume re-derives.** Delete it and a migrated pre-markers tree gets synthetic ids that
> never match, so the next `--mode full` reprocesses every image — the exact outcome
> CAN-7 exists to prevent. Task 2b's own `images_processed == 0` assertion was unachievable
> as written; that it was unachievable is what makes it a good assertion.
>
> **The orchestrator made this call from its own verification and reported it to the user as
> a simplification.** It was neither. MIG-1's original direction — port the helper into
> migrate — stands.

**What the task actually is: move the promoter, do not delete it.**

A v0.17.3 tree — the supported floor — has **no `image_complete/`, no `stage2_done/`, no
`stage3_complete/`, and no OME-Zarr stores**. Task 2 enumerates three empty trees and
converts nothing, while Task 3 deletes `datasets.{completed,failed,started}` — which for
that shape is the *only* record of what finished. Migrate would report success over a tree
with zero records, and the next `--mode full` would reprocess every image from source.

**The correct framing (and the first reading of this was wrong):** the two conversions are
**not** chained producer→consumer. The HDF→Zarr migrator is *itself* a producer of the
record schema — it calls `publish_image_success` at `_cli_migrate.py:1413` and
`_cli_migrate_image.py:567`. **They are alternative producers of one shape, and P3 revised
only one of them.**

Verified against source:

- `_configured_work_id` (`_cli_migrate_image.py:125`) falls back to `_migration_work_id`
  = `sha256("migration:<ds>/<stem>")` (`:120-122`) when the state carries no `work_ids`.
- `_existing_marker_identity` (`:142`) supplies defaults when no marker exists.
- `_migration_marker_artifacts` derives descriptors from **the store the migrator just
  wrote** — which is not the laundering Task 2 forbids. Task 2's rule is *never re-derive
  descriptors for an artifact migrate did not create*; certifying one it just created is
  what a publisher does.

So the pre-markers path already works end to end. What is required:

- [ ] **Step 1: Move `_migrate_legacy_success_evidence` into `_cli_migrate_state`**

Spec §11.1 assigns "every `_legacy_*` helper moves **into** migrate" to P7, and P6's
deletion ledger row 10 defers it here explicitly. **This is that helper**, and the
destination is migrate — not deletion.

Move, do not rewrite: `phenotypicCLI.py:560` (`_migrate_legacy_success_evidence`) and
`:544` (`_requires_legacy_success_migration`) become part of `_cli_migrate_state`. Delete
only the **dispatch** from the `--mode full` resume path (`:2375-2378`) and its user-facing
echo (`:2380-2383`), because after P1's `requires_conversion` gate a legacy tree never
reaches that code. The promotion itself now happens inside `--mode migrate`.

**Order matters and must be stated:** the promoter reads `ds_state.completed` and resolves
each image's data artifact through `image_data_artifact`, which returns the **store** when
one exists and the `.h5` otherwise. So it must run **after** the HDF→Zarr conversion, or it
certifies files that are about to be replaced. That is the one real ordering constraint in
this phase.

**Reconcile the two producers.** The migrator also publishes records
(`_cli_migrate_image.py:567`) using `_configured_work_id`, which on this shape yields the
synthetic id. After the promoter has populated `state.config["work_ids"]` with
content-derived ids, `_configured_work_id`'s lookup succeeds and the synthetic fallback
stops firing. **Sequence them so the promoter runs first**, or the two producers disagree
about the same image's identity — which is the defect this task exists to prevent, arriving
from the other direction.

Three tests exercise the helper (`tests/unit/cli/test_cli_state_management.py:316`,
`test_cli_completion_store.py:606`, `test_embedded_measurement_migration.py:312`) plus a
maintained pre-markers fixture (`tests/unit/sdk_/_migration_fixtures.py:440-447`). **Keep
all four**; retarget the tests at the new call path rather than deleting them.

- [ ] **Step 2: Prove it, with the real migrator**

```python
def test_a_pre_markers_tree_converts_end_to_end(tmp_path):
    """CAN-7 / U-1. v0.17.3 is the floor and it predates markers AND stores, so
    this is the shape that must work, not an edge case.

    Built through the REAL HDF migrator. A hand-planted fixture cannot catch this
    class of drift -- that is exactly how the gap survived the first draft.
    """
    from phenotypic.sdk_ import resolve_run_state

    tree = _build_v0_17_3_tree(tmp_path)          # _migration_fixtures.py:440-447
    assert _state_config(tree).get("success_markers_required") is None

    _invoke_cli(mode="migrate", output=tree)

    state = resolve_run_state(tree, depth="deep")
    assert state.completion == "complete"
    assert len(state.images) == _image_count(tree), "records were not produced"

    reprocessed = _invoke_cli(mode="full", output=tree)
    assert reprocessed.images_processed == 0, (
        "a migrated pre-markers tree reprocessed from source"
    )


def test_migrated_records_carry_the_work_id_resume_re_derives(tmp_path):
    """CAN-7's decisive assertion, and the one that caught the wrong resolution.

    v0.17.3 has no `work_ids` key at all, so `_configured_work_id` always falls
    back to the SYNTHETIC sha256("migration:<ds>/<stem>"). Resume re-derives via
    `work_id_for_image` -- content-derived. If the record carries the synthetic id,
    nothing matches and every image reprocesses, while every other assertion in
    this file still passes.

    Assert the two agree, rather than asserting a downstream symptom.
    """
    from phenotypic._cli._cli_failure_tracker import work_id_for_image
    from phenotypic._cli._cli_image_record import read_image_record

    tree = _build_v0_17_3_tree(tmp_path)
    _invoke_cli(mode="migrate", output=tree)

    for dataset, image in _images_of(tree):
        expected, _ = work_id_for_image(_config_for(tree), dataset, image)
        record = read_image_record(tree, dataset, image.stem)
        assert record["work_id"] == expected, (
            f"{dataset}/{image.stem}: record carries {record['work_id'][:12]}…, "
            f"resume will look for {expected[:12]}…  — the synthetic migration id "
            "leaked into the record"
        )
        assert not record["work_id"].startswith(_synthetic_prefix(dataset, image.stem))
```

- [ ] **Step 3: Make Task 3's deletion conditional**

Never delete `datasets.{completed,failed,started}` in the same pass that failed to consume
them. Delete only after the tree has records for that dataset — cheap, and it converts a
silent data loss into a loud refusal if a future shape slips through.

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

**Decision: migrate leaves embedded tables alone; `finalize_run` projects them at read.**
The master **stamp moved to P4** (CAN-9) — migrate leaves a legacy master *unstamped*,
which is the honest pre-v2 signal, and `read_master_measurements` refuses it.

Rewriting every store's embedded table would reintroduce exactly the post-proof store
mutation D-A removed, for trees that already have a correct master. Instead,
`finalize_run`'s step 1 projects each embedded table onto its recorded
`phenotypic.measurement_columns` before concatenating — the same boundary P4 uses, with the
column list already in every existing store.

### The projection is necessary but not sufficient (CAN-10)

The first draft claimed projection makes *"a pre-inversion store aggregate to the same
master a post-inversion one does."* **True for the column set; false for row count and
dtype**, and its only test checked column *names*, which passes under both defects.

**(a) Duplicate-key fan-out.** `prepare_embedded_measurement_table` right-joins with
metadata as the **left** frame and `maintain_order="right"`
(`_embedded_measurement_tables.py:88-93`), logging *"preserving duplicate-key fan-out"* at
`:81-86` — deliberately. A legacy store whose metadata CSV has *k* rows per key holds each
measurement row *k* times. Projection preserves that, and P4's global join then fans it out
**again** → *k²* rows in the mirror. A P4-era store carries no fan-out at all; it moved to
`finalize_run` step 3.

**(b) Join-key dtype drift.** `_restore_join_key_dtypes` (`:22-39`) logs a warning and
leaves a column as the string-safe matching type on failure (`:31-38`). So a legacy store
can carry a join key as `str` where the baseline was `int64`; concatenating it with a fresh
store either raises or silently upcasts, and after the global join the affected rows' user
metadata is null.

**Fix both in the projection**, since it is already the one place legacy shape is
normalized:

1. **Row-collapse on the store's own `target.column`** — the descriptor already names it
   (`_measurement_tables.py:459-465`) — for stores whose recorded `join_status` is
   `joined`. **Prove the dedup key safe**: two genuinely distinct objects must never
   collapse. If it cannot be proved for some store shape, that shape gets an advisory and is
   excluded, not silently deduped.
2. **Normalize join-key dtypes at concat**, against the fresh-store baseline — or refuse the
   store with an advisory. Never upcast silently.

- [ ] **Step 0: Delete `master_measurements.csv` (CAN-32)**

The conversion table assigned this here and the first draft had no step for it. Delete the
file if present; `master_measurements_csv_path()` and `MASTER_MEASUREMENTS_CSV` are already
gone in P4, so this is the on-disk half.

- [ ] **Step 0b: Handle a store with no measurement descriptor (CAN-32)**

`read_embedded_measurement_descriptor`'s docstring (`_measurement_tables.py:340-346`) says
an absent descriptor is *"a normal state, not a fault: a `--mode process` run never
measures, and a store written before embedded tables has none."* But
`embedded_measurement_columns` (`:382`) raises `KeyError` on it, so the projection has a
**reachable unhandled path**. Skip such a store and raise an advisory, per INV-VERDICT's
degrade-toward-`incomplete` rule.

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


def test_migrate_leaves_a_legacy_master_unstamped(tmp_path):
    """CAN-9 / U-3. The first draft stamped the master during migrate while
    explicitly NOT re-running finalization -- so the stamped file was still the
    legacy metadata-joined master, and the stamp certified a shape it would not
    have until the next finalize_run. Its own two tests contradicted each other.

    P4 mints the stamp where finalize_run writes the master, so stamp and shape
    come from one code path. Migrate leaves the legacy master unstamped, which
    read_master_measurements correctly refuses as pre-v2.
    """
    import pyarrow.parquet as pq
    import pytest

    from phenotypic.sdk_ import master_measurements_parquet_path
    from phenotypic.sdk_._master_io import read_master_measurements

    _build_legacy_tree(tmp_path)
    _invoke_cli(mode="migrate", output=tmp_path)

    schema = pq.read_schema(master_measurements_parquet_path(tmp_path))
    assert b"phenotypic.master_schema_version" not in (schema.metadata or {})
    with pytest.raises(ValueError, match="migrate|schema"):
        read_master_measurements(tmp_path)


def test_a_legacy_fanout_store_and_a_fresh_store_produce_equal_masters(tmp_path):
    """CAN-10. Projection fixes the column set; it does not undo the k-times row
    duplication a legacy joined table carries, and P4's global join then squares
    it. Assert ROW COUNTS, which is what the first draft's column-membership test
    could not see."""
    import polars as pl

    from phenotypic.sdk_ import master_measurements_parquet_path, measurements_parquet_path

    legacy = _build_legacy_tree_with_fanout(tmp_path / "legacy", rows_per_key=3)
    fresh = _build_equivalent_fresh_tree(tmp_path / "fresh", rows_per_key=3)
    _invoke_cli(mode="migrate", output=legacy)
    _finalize(legacy)
    _finalize(fresh)

    for path_of in (master_measurements_parquet_path, measurements_parquet_path):
        a = pl.read_parquet(path_of(legacy))
        b = pl.read_parquet(path_of(fresh))
        assert a.height == b.height, f"{path_of.__name__}: {a.height} != {b.height}"
        assert a.equals(b)


def test_a_legacy_store_with_a_drifted_join_key_dtype_does_not_null_its_metadata(tmp_path):
    """CAN-10(b). _restore_join_key_dtypes leaves the string-safe type on failure
    (_embedded_measurement_tables.py:31-38), so a legacy store can carry a key as
    str where the baseline was int64. Silent upcasting at concat nulls the metadata
    for exactly those rows after the global join."""
    import polars as pl

    from phenotypic.sdk_ import measurements_parquet_path

    tree = _build_legacy_tree_with_string_join_key(tmp_path)
    _invoke_cli(mode="migrate", output=tree)
    _finalize(tree)

    mirror = pl.read_parquet(measurements_parquet_path(tree))
    measured = mirror.filter(pl.col("QC_MetadataOnly").fill_null(False).not_())
    assert measured.height > 0
    assert measured["Metadata_Strain"].null_count() == 0
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
    """INV-PROVEN's certified-transition exception, kept scoped. D-A cut §6.4's GENERALISATION,
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

- [ ] **Step 1b: Rename the legacy trees; do not delete them (CAN-12)**

§15.1 requires *"the receipt/rollback discipline the existing metadata migration has, plus
its own dry-run mode."* The first draft delivered the dry run and **resumability** — a
different property. Resumability guarantees a re-run finishes; it says nothing about
recovering the previous state. Task 2 removed the three legacy trees outright, with no copy
and no receipt, where the existing metadata migration leaves receipts
(`_cli_completion.py:340-350`).

The consequence: after a successful migrate, a user who **reverts the code** — the ordinary
first response to a regression in a change this size — has a tree the old build reads as
entirely unprocessed. For 6,000 images and no backup that is a full reprocess, and nothing
in the plan, the CLI help, or the `CLAUDE.md` bullet said migrate is one-way.

**Rename `image_complete/`, `stage2_done/` and `stage3_complete/` to
`.phenotypic/legacy-v2/`** — same filesystem, a directory rename, no byte copied — and have
`requires_conversion` ignore that path. `migrate --revert` then costs a rename back.

This puts a directory on disk that outlives migration, so be explicit about what it is:
**retained for revert, read by nothing.** It is not tracked state — nothing derives from it,
nothing must be kept in sync with it, and no verdict consults it. `_cli/CLAUDE.md`'s
register (Task 6) lists it under its own heading, *not* in the tracked-state table, and
says when it can be deleted.

```python
def test_migrate_is_revertible(tmp_path):
    """CAN-12 / §15.1."""
    tree = _build_legacy_tree(tmp_path)
    before = _tree_fingerprint(tree)
    _invoke_cli(mode="migrate", output=tree)
    _invoke_cli(mode="migrate", output=tree, revert=True)
    assert _tree_fingerprint(tree) == before


def test_the_retained_legacy_tree_is_invisible_to_detection(tmp_path):
    """It must not make an already-converted tree look unconverted."""
    from phenotypic._cli._cli_schema_gate import requires_conversion

    tree = _build_legacy_tree(tmp_path)
    _invoke_cli(mode="migrate", output=tree)
    assert (tree / ".phenotypic" / "legacy-v2").is_dir()
    assert requires_conversion(tree) is None
```

- [ ] **Step 1c: State the coexistence rule (CAN-13)**

Nothing in the spec or plan mentions it. A SLURM array launched from the **old build** holds
the old schema for its entire lifetime — up to 30 d on `batch`/`intel`/`epyc` — and P2's
`restart_epoch` fence cannot reach it, because an old-build worker never calls the new
publisher at all: it writes the three legacy trees directly. So a tree migrated while such
an array is live **re-acquires the old shape** and is then refused by every writing mode,
including the array's own dependent finalizer.

Put the operational rule in the phase doc, in `--mode migrate`'s `--help`, and **in the
refusal message itself**: drain or `scancel` in-flight arrays before migrating. Step 3's
merge-not-overwrite conversion is the second half — with the legacy trees renamed rather
than deleted, a late old-build write lands in a directory nothing reads, which is the safer
failure.

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
INV-PROVEN's certified-transition exception, still scoped to migrate."
```

---

## Task 6: The tracked-state register, in `_cli/CLAUDE.md`

**Files:**
- Modify: `src/phenotypic/_cli/CLAUDE.md`

**This lands last on purpose:** the CLI's state contract is not final until migrate exists,
and a register written earlier would describe a shape no tree has. It is **not optional** —
the whole change is a claim about which state is tracked and which is derived, and that
claim has to live where the next reader looks.

- [ ] **Step 1: Correct what the file already gets wrong**

`_cli/CLAUDE.md:251-254` currently states, of the store fingerprint: *"the root is written
**last** by the promote protocol and nothing writes into the store after publication, so a
valid root implies a complete store."*

**The second clause is false and was false before this change** (CAN-3):
`replace_embedded_measurement_table` (`sdk_/_measurement_tables.py:242`) is reached on the
`--mode measure` forward path, and its in-place branch (`:284-290`) rewrites the embedded
table with **no root rewrite at all** — so a valid root does *not* imply unchanged
contents. Rewrite the paragraph to INV-PROVEN's actual statement: an artifact carrying a
content proof changes only where the proof changes with it, and here are the paths where
that holds and the one where it did not until P4 repaired it.

Also correct `## Per-image completion markers` (`:235`): `SUCCESS_MARKER_VERSION` is now
`RECORD_VERSION`, the three marker trees are one record, and the `_migrate_legacy_success_evidence`
paragraph (`:257-261`) describes a function P7 deletes.

- [ ] **Step 2: Add the register — three tables, and the split IS the content**

**(a) Tracked state — written down, and irreducibly so.** Exactly these, and the "why it
cannot be derived" column is the load-bearing one:

| # | State | File | Writer | Why it cannot be derived |
|---|---|---|---|---|
| 1 | Accepted inventory | `processing_state.json` → `config.work_ids` | `create_initial_state`, resume | A directory listing is a different question from "what did this run accept". |
| 2 | Terminal failures | `.phenotypic/terminal_failures.jsonl` | `append_terminal_failure` | A failure leaves no artifact; absence of output is indistinguishable from not-yet-started. |
| 3 | Liveness & ownership | `slurm_lifecycle.json`, `slurm_jobs.jsonl`, `gui_launch_owner.json` | CLI submitter / **GUI** | External-system and process facts; a crashed worker leaves no trace. |
| 4 | `restart_epoch` | `.phenotypic/restart_epoch.json` | `bump_restart_epoch` | A content-derived generation cannot distinguish "deliberately fresh attempt" from "same config again". **Preserved by `clear_machine_state`** — a counter that resets on the operation it fences is not a fence. |

**Four. If a fifth appears, that is a design regression** — say so in the file, and name
the organising principle it violates: *move state that is tracked to state that is checked.*

**(b) Content proofs — not tracked state.** Per-image record, aggregate proof, run proof
are digest manifests over artifacts that already exist. Give the publication order and say
it is never reordered: store root `zarr.json` last → per-image record after artifacts →
aggregate proof after outputs → run proof after aggregate.

**(c) Derived, and how.** One row per fact, naming the deriving function — this is the
table that stops a future contributor writing a counter:

| Fact | Derived from | By |
|---|---|---|
| "is this run done?" | 1 + 2 + 3 + the proofs | `resolve_run_state(output_dir, depth=...)` |
| `processing_generation` | `sha256(pipeline_sha256 ‖ scientific_config_digest ‖ restart_epoch)` | `mint_run_identity` |
| `work_id` | content: schema version, dataset, input-relative path, input sha256, pipeline fingerprint, per-image config digest, mode | `work_id_for_image` |
| `inventory_digest` / `source_set_digest` / `scientific_config_digest` / `finalization_input_digest` | `config` fields + the verified set | `run_identity`, `finalization_input_object` |
| per-dataset completed/failed counts | the per-image records | `RunState.diagnostics` — **and nothing branches on them** |
| the master | the marker-authorized embedded tables, and **nothing else** | `finalize_run` step 1 (INV-INPUTS) |

State explicitly what was **deleted** and must not come back:
`processing_state.datasets.{completed,failed,started}` (a cache of a cache — already
re-aggregated from the event log on every load), `manifest.json` as evidence,
`publication_id`, and the event log as a completion source.

- [ ] **Step 3: Record the read/write asymmetry and the migration floor**

Two rules a future contributor will otherwise breach:

- **Readers live in `sdk_/_run_state.py`; writers stay in `phenotypic._cli`.** INV-LAYER's
  AST test enforces it. Name the test so a reader can find out why their import failed.
- **Migration floor is v0.17.3** (U-1). Below it, migrate refuses with a version string and
  a pointer. Say that v0.17.3 predates both the marker schema and OME-Zarr, so the floor is
  the pre-markers shape, and that the HDF→Zarr migrator is *itself* a producer of the record
  schema (CAN-7) — not a stage that runs before one.

- [ ] **Step 4: Update the two contract statements this change invalidates**

`## Output layout & deliverables` (`:298`): master is parquet-only, un-joined, intrinsic
identity only; the mirror carries the join and the phantoms; `finalize_run` is the single
path for `full`/`measure`/`recompile`. Keep the existing "feed analysis and dashboards from
the mirror, not the master" rule — it is now doing more work than before, and say why.

`## Legacy-tree migration` (`:175`): the state-schema conversion, the v0.17.3 floor, the
legacy-tree rename, and the revert path.

- [ ] **Step 5: Verify every claim, then commit**

Every path, function name and line reference gets a `grep` before the commit. Root
`AGENTS.md` is a symlink to the project `CLAUDE.md`, so check whether anything you wrote
duplicates a rule that belongs there instead.

```bash
git add src/phenotypic/_cli/CLAUDE.md
git commit -m "docs(cli): record the tracked-state register and how everything else derives

Four tracked states, each with why it cannot be derived; the content proofs that
are not tracked state; and one row per derived fact naming the function that
derives it. Corrects the pre-existing claim at :251-254 that nothing writes into a
store after publication -- --mode measure's in-place branch always could."
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
