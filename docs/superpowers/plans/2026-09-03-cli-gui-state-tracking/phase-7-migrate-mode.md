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
| `stage3_complete/<ds>/<stem>.json` | `stages.stage3` in the same record | 2 |
| **pre-markers tree** (`success_markers_required` absent, `version="2.0.0"`) — **the v0.17.3 floor** | `_migrate_legacy_success_evidence` **ported into migrate**, sequenced after the HDF→Zarr conversion and before the migrator's own publisher. It is the only producer of the *content-derived* `work_id` a later resume re-derives. See Task 2b. | 2b |
| `processing_generation: <uuid4>` **and** the migrator's inventory-derived one | content-derived generation; `restart_epoch: 0` | 3 |
| `processing_state.datasets.{completed,failed,started}` | **deleted from the file** (§4.2) | 3 |
| ~~`slurm_generation` / `lifecycle_epoch`~~ → ~~`scheduler_epoch`~~ | **ROW WITHDRAWN.** Migrate neither. §5.1's collapse was withdrawn (`design.md:323-345`, user-ruled) — both are on-disk keys with live readers, and rewriting them here would be the collapse as a *tree migration*, which is strictly worse than the rename that was already rejected. | — |
| joined embedded tables | **left alone**, projected at read — see Task 4 | 4 |
| `master_measurements.csv` | deleted; master is parquet-only (D8) | **4, Step 0** |
| `deliverables/metadata.canonical.csv` | **emitted** alongside the untouched snapshot | **3, Step 4** |
| legacy `.h5` per-image files | unchanged — the existing OME-Zarr migration already owns this | — |
| ~~anything below v0.17.3~~ | **No version floor (U-6).** `state.version` cannot express one; detection is by shape, and a pre-markers tree is supported however old. | 1 |

> **Two rows in the first draft's table had no implementing step (CAN-32):** the CSV
> deletion was assigned to Task 4, which had no step for it, and Task 3 asserted
> `metadata.canonical.csv` exists while no task built it. Both now name a step.

**What migrate does NOT do:** re-mint `work_id` (D-C keeps the digest unchanged, so every
existing `work_id` stays valid), rewrite `deliverables/metadata.csv` (project `CLAUDE.md`,
spec D9/FLOW-4 — there is **no exception, including migrate**), or write into any store.

---

## Task 1: Detection and refusal

> **`_invoke_cli` is defined in P1 Task 3b, not here** — see that task for the helper and,
> more importantly, for the rule that goes with it: assert on `result.exit_code` and
> `result.output`, never `pytest.raises(SystemExit, match=...)`. `str(SystemExit(2))` is
> `"2"`, so a `match=` on the refusal message cannot fail for its own reason. **The three
> tests in Step 1 below are written in the broken idiom and must be converted.**

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

`STATE_SCHEMA_VERSION = 3`, and the function's full contract — all five detection signals,
the no-raise rule, and why it reads raw JSON — is written **once**, in Step 3b below. Build
it from there.

Wire it into `phenotypicCLI.py` for `full`, `measure`, `recompile` and `process`. **Not**
for the GUI or any reader — spec §4.3 makes a half-migrated tree an advisory.

- [ ] **Step 3b: Detect the pre-markers shape (U-1, as amended by U-6)**

> **U-6 (round 2, user).** U-1 named **v0.17.3** as the floor, on my assurance that
> `state.version` could detect it. **That assurance was wrong** — `"2.0.0"` is the value at
> v0.17.3 *and* immediately before `379acee4` introduced `"3.0.0"`, so it spans both sides.
> It is a *state-schema* version, not a *package* version.
>
> **Ruling: key on the pre-markers shape, with no sub-floor.** Schema `2.0.0` with **no
> `work_ids` key** — the concept did not exist at v0.17.3. This supports every pre-markers
> tree, including ones older than v0.17.3, and that costs nothing: they are the **same
> shape**, and the ported promoter (Task 2b) handles them identically. Root-level machine
> state, from before `.phenotypic/` existed (2026-06-17, one day before the v0.17.3 tag), is
> already converted by `migrate_legacy_machine_state`, which **ships today** — refusing it
> would be removing support, not bounding scope.
>
> So there is **no `BELOW_FLOOR` verdict.** `requires_conversion` keeps two outcomes plus
> honest failure on a tree it cannot classify. Delete that enum member and its test.

`requires_conversion` therefore detects **shape**, never a version number:

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

    Two outcomes plus honest failure (U-6):
      - ``None``                      -- already current
      - ``ConversionVerdict.CONVERT`` -- convertible; the message names the evidence
      - it never raises: absent, unparseable and malformed state all map to a
        verdict, because a refusal gate that crashes on a malformed tree is worse
        than the silent path it replaces (MIG-14b)

    **There is no version floor.** U-1 named v0.17.3, but ``state.version`` cannot
    express it -- ``"2.0.0"`` is the value both at v0.17.3 and immediately before
    ``"3.0.0"`` was introduced. Detection is by SHAPE:

    1. ``.phenotypic/progress/image_complete/`` exists
    2. ``stage3_complete/`` exists. **NOT ``stage2_done/``** (U-9): that tree is
       current, not legacy, so firing on it would classify every modern GPU run
       CONVERT and strand it -- an INV-DISCHARGEABLE violation.
    3. ``processing_state.json`` carries ``datasets.completed`` (deleted by §4.2)
    4. ``processing_state.json`` has ``work_ids`` and no ``restart_epoch``
    5. a **present, parseable, object-shaped** ``processing_state.json`` that carries
       ``version`` (or ``datasets``) and has **no ``work_ids`` key** -- the
       pre-markers shape

    An **absent** state file returns ``None``, mirroring ``load_processing_state``'s
    own ``return None`` (``_cli_state_management.py:111-112``). An **unparseable**
    one gets a distinct verdict that names the file and does **not** point at
    ``--mode migrate``, which cannot repair it.

    Reads the raw JSON directly. Never ``load_processing_state``, which writes via
    ``migrate_legacy_machine_state`` (``:109``) and raises on an absent version
    (``:167``) -- a gate must not mutate the tree it is deciding about.
    """
```

> ### INV-DISCHARGEABLE — no verdict may be emitted that migrate cannot discharge
>
> **The missing invariant behind MIG-11 and MIG-20, stated once.**
>
> `_refuse_unmigrated_output` fires for `full`, `measure`, `recompile` and `process` at
> `phenotypicCLI.py:1661-1662`, **before `--restart` is handled**. So a tree classified
> `CONVERT` that migrate cannot actually convert is **refused by every writing mode
> forever**, and the only escape is `--overwrite`, which deletes the outputs. A gate that
> can strand a user's tree is worse than no gate.
>
> An earlier draft diagnosed exactly this for the pre-markers process tree (MIG-11) and then
> wrote a signal — and a test — that reproduced it for three more shapes. Signal 5 as first
> written fires on:
> 1. a **fresh output directory** (no state file), which
>    `test_a_fresh_output_directory_is_not_an_unconverted_tree` requires to be `None`;
> 2. a **bundle-only tree**, which has no `processing_state.json` at all — and whose Step-3c
>    row still says "trips none of the **four** signals", a stale count now that there are
>    five;
> 3. an **unreadable state file** — so a *modern* tree whose state is truncated classifies
>    `CONVERT` and is permanently refused.
>
> **The test that closes this class, not just these instances:** for every shape the gate can
> return `CONVERT` for, one successful `--mode migrate` must make `requires_conversion`
> return `None`.
>
> ```python
> @pytest.mark.parametrize("shape", _EVERY_CONVERTIBLE_SHAPE)
> def test_every_convert_verdict_is_dischargeable_by_one_migrate(tmp_path, shape):
>     """INV-DISCHARGEABLE. A CONVERT that migrate cannot discharge strands the tree
>     behind a refusal in every writing mode, escapable only by --overwrite.
>
>     This is the test that closes MIG-11, MIG-20, and the next shape nobody
>     enumerated -- which is why it is parametrized over the shape list rather than
>     written per shape."""
>     tree = _build(shape, tmp_path)
>     assert requires_conversion(tree) is ConversionVerdict.CONVERT
>     _invoke_cli(mode="migrate", output=tree)
>     assert requires_conversion(tree) is None, (
>         f"{shape}: migrate ran successfully and the gate still refuses the tree"
>     )
> ```
>
> `_EVERY_CONVERTIBLE_SHAPE` is the same list Step 3c classifies. **Adding a shape there
> without adding it here is the bug this invariant exists to catch.**

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
| **Bundle-only** — `deliverables/` with no `.phenotypic/` | `None` (nothing to convert) | `BundleLayout.detect` explicitly supports it (`_io_constants.py:2468-2482`). It trips none of the **five** signals, because signal 5 requires a *present* state file — see INV-DISCHARGEABLE. Its master stays unstamped, which under P4's stamp-at-finalize is the **correct** pre-v2 signal. |
| **Unreadable `processing_state.json`** | a distinct verdict, **not `CONVERT`** | Migrate cannot repair a truncated state file, so pointing at it would strand the tree (INV-DISCHARGEABLE). Name the file; do not name `--mode migrate`. |
| **modern `--mode process` tree** | `None` after P3 converts its records | Process **does** call `publish_image_success` (`_cli_process_single.py:789,943`), so signal 1 fires — but it has **no master at all**, so any master-touching step must no-op rather than raise. |
| **pre-markers `--mode process` tree** | **needs a process arm in migrate** — see below | `_cli_process_only.py` ships in v0.17.3, so this is *inside* the supported range. It has no `image_complete/`, so signal 1 does **not** fire; it trips signal 3 and classifies `CONVERT`. But migrate's discovery enumerates `.h5` and `results/<ds>/zarr/*.ome.zarr` (`_cli_migrate.py:611,1377`) — **a process-only run wrote neither.** Its outputs are process layers under the mirrored input tree. |
| **Interrupted migrate** | `CONVERT`, and the re-run completes it | Already correct; keep the test that says so. |

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

- [ ] **Step 4: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_schema_gate.py -v
git add src/phenotypic/_cli/_cli_schema_gate.py src/phenotypic/phenotypicCLI.py \
        tests/unit/cli/test_schema_gate.py
git commit -m "feat(cli): refuse an unconverted tree, detecting by shape not version

D1 + U-1 as amended by U-6. Detection is by presence of the old shape, not absence
of the new -- an empty directory is not a legacy tree, and state.version cannot
express a package-version floor ("2.0.0" spans both sides of v0.17.3). The
pre-markers signal is an absent work_ids key. Reads raw JSON, never
load_processing_state, which would write to the tree while deciding whether to
refuse it. Readers get an advisory, not a refusal (§4.3).

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
                          image_complete=True, stage3_complete=True)
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


def test_a_stage3_marker_with_no_image_complete_still_converts(tmp_path):
    """Stage 2 finished and Stage 3 never ran -- a real interrupted-run state, and
    the one a naive 'iterate image_complete/' conversion drops on the floor."""
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers
    from phenotypic.sdk_ import image_record_path

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a",
                          image_complete=False, stage3_complete=True)
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
first, then **rename** the legacy trees aside — the rename primitive, its collision rule and
the `--revert` path are **Task 5 Step 1b**, not this task's to invent. **Copy `artifacts`
verbatim**; never re-derive.

**Merge, do not overwrite (CAN-13).** When `images/<ds>/<stem>.json` already exists, union
the `stages` maps and keep the later `completed_at` rather than replacing. The
both-shapes-present case is real — **Task 5 Step 1c** is the coexistence window that
produces it: an old-build SLURM array holds the old schema for its whole lifetime, up to
30 d, and writes the legacy trees directly. `test_conversion_is_idempotent` cannot catch
this, because after the first pass the legacy trees are renamed aside and the second call
is a no-op.

```python
def test_conversion_merges_into_an_existing_record(tmp_path):
    """CAN-13. An old-build worker writing image_complete/ after a partial migrate
    must not clobber a newer record's stages."""
    from phenotypic._cli._cli_image_record import record_stage
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers
    from phenotypic.sdk_._image_record import read_image_record

    record_stage(tmp_path, "plate", "a", "stage3", {"at": "new"})
    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    convert_per_image_markers(tmp_path)

    stages = read_image_record(tmp_path, "plate", "a")["stages"]
    assert {"stage3", "measured"} <= set(stages), "the newer record's stage3 was lost"
```

- [ ] **Step 4: Run the tests.** Expected: PASS (6 passed).

```bash
uv run pytest tests/unit/cli/test_migrate_state.py -q
```

- [ ] **Step 5: Prove the two load-bearing assertions can fail**

Both guard against a conversion that looks right and is not, so neither is worth having
unless it has been seen red:

| Test | Break it by | Must fail with |
|---|---|---|
| `test_artifact_descriptors_survive_conversion_byte_for_byte` | re-deriving descriptors from the store instead of copying them | a descriptor mismatch, not a `KeyError` |
| `test_the_legacy_trees_are_removed_only_after_every_record_is_written` | moving the rename inside the per-image loop | marker `a` gone after the injected failure |

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli/test_migrate_state.py
git commit -m "feat(cli): convert the three per-image legacy trees into one record

Spec §11.1. Enumerates the union of image_complete/, stage2_done/ and
stage3_complete/ -- an image with a stage-2 token and no completion marker is a
real interrupted state that an image_complete/-only walk drops.

Descriptors are copied verbatim, never re-derived: re-deriving would certify
whatever is on disk now, including a corrupted artifact, which turns migrate from
a format change into a laundering step.

Merge, do not overwrite (CAN-13). An old-build array holds the old schema for its
whole lifetime and can write a legacy tree after a partial migrate; clobbering
would lose the newer record's stages. The rename-aside and revert path are Task 5."
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
>
> ### Round 4 (U-10): the conclusion holds, the REASON changed — and the task got simpler
>
> **The verdict above is unchanged: port the promoter.** But read its argument again with
> U-10 in hand and the load-bearing clause is gone. It said the helper must be ported
> *because it mints the content-derived id resume re-derives*. Under U-10 **migrate mints no
> matching id at all** — it marks the record `provenance: "migrated"` and
> `valid_image_success` accepts it on artifact validity, with no `work_id` comparison. The
> synthetic-vs-content-derived distinction the reversal turned on no longer decides anything.
>
> **So why is the helper still ported?** For the half the reversal did not emphasise, which
> was always the real one — **MIG-1's**: on a pre-3.0.0 tree, `state.datasets.<ds>.completed`
> is the *only* record of what finished. Migrate needs the helper to know **which images to
> publish records for**. It is being ported for its enumeration, not its identity minting.
>
> **And this dissolves MIG-18.** That finding said porting was hard because migrate builds no
> `ExecutionConfig`, and `work_id_for_image` needs one. Migrate no longer calls it. Drop the
> identity-minting half of the port along with the reconstruction table's ⚠️ rows; keep the
> `datasets.completed` walk and the `process_only_layer` arm at `phenotypicCLI.py:602-612`.
> **The port gets smaller, not larger** — the first genuinely simplifying consequence in this
> phase, and the one to check first if the task starts growing during implementation.
>
> **Task 2b's `images_processed == 0` assertion is now achievable**, and by a different
> mechanism than the reversal assumed: not because the ids match, but because the marking
> means no id is compared.

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

> ### U-7 (round 3, user): migration logic lives **only** in `--mode migrate`
>
> An earlier draft of this task proposed leaving `_migrate_legacy_success_evidence` on the
> `--mode full` resume path, because that path has the `ExecutionConfig` the identity needs.
> **Overruled.** A legacy conversion that fires silently inside a normal run — with no flag,
> no opt-in, and a one-line notice — is precisely the hidden state transition this change
> exists to remove. It moves.
>
> **What that costs, stated plainly, because it is not free (MIG-18).** `work_id_for_image`
> hashes seven fields. On a v0.17.3 tree migrate can recover some, approximate others, and
> cannot reproduce the rest at all — **which is why U-10 below stops trying**:
>
> | Field | Available to migrate on a v0.17.3 tree? |
> |---|---|
> | `schema_version`, `dataset`, `relative_image_path`, `mode` | ✅ constant, or read off the tree |
> | `input_sha256` | ⚠️ **only if the sources still exist.** `state.input_path` records where they *were* (so migrate needs no `--input`, and root `CLAUDE.md`'s rule stands), but the field is `file_sha256` of the original image and is **never persisted** — it is a hash input, not a stored value. Sources moved, archived or deleted make it unobtainable, and archived runs are exactly the ones being migrated. |
> | `pipeline_fingerprint` | ⚠️ `<output>/pipeline.json` exists (`_persist_pipeline_to_output_dir`, `_cli_output_manager.py:325-357,606`) but is a `to_json()` **re-serialization**; `work_id` hashes the user's original file. Substituting the re-serialized digest produces a confidently wrong id. |
> | `processing_configuration_digest` — `image_type`, `nrows`, `ncols`, `bit_depth`, `ext` | ✅ in `processing_state.json:config`, which holds **seven** keys at v0.17.3 (`n_jobs` and `slurm_args` are the other two and are correctly not digest inputs) |
> | `processing_configuration_digest` — `save_overlays`, `include_dataset_column` | ⚠️ *derivable by inference* — does `results/<ds>/overlays/` hold files; does the master carry `Metadata_Dataset`. Inference, not record. |
> | `processing_configuration_digest` — **`detect_mode`, `process_only_layer`, `process_format`, `overlay_alpha`, `drop_originals`** | ❌ **never written to disk at that vintage** |
>
> **Read the bottom four rows together and the conclusion is forced.** Even granting every
> ⚠️ its best case, five of the twelve digest fields are simply absent, and two of the seven
> top-level fields depend on files outside the tree. There is no reconstruction here — only
> degrees of fabrication. That is the finding U-8 was written before, and U-10 after.
>
> ### U-8 WITHDRAWN, U-10 (round 4, user) replaces it
>
> **U-8 said:** migrate omits the unrecoverable fields from the digest, and a forward run
> reading that state omits them too, so both hash the same reduced payload and the `work_id`s
> match.
>
> **There is no such forward run, and it cannot be added cheaply.** Three independent facts,
> each verified in source:
>
> 1. `work_id_for_image` (`_cli_failure_tracker.py:329-350`) **recomputes** the identity from
>    a live `ExecutionConfig` every call. It never reads `state.config`, so what migrate
>    writes there cannot influence what a forward run looks for.
> 2. `processing_configuration_digest_from_values` (`:200-214`) takes all twelve fields as
>    **required** keyword parameters and writes them into the payload unconditionally. There
>    is no "absent" to express.
> 3. A **second** producer, `_worker_work_identity` (`_cli_process_single.py:122-171`),
>    derives the same id from argv, and the two are cross-checked with a hard `RuntimeError`
>    at `:723-729`. Both would need the same new notion of "unasserted".
>
> And two of the seven `work_id` fields are irreproducible regardless of any of that:
> `input_sha256` is `file_sha256` of the **original input image**, never persisted anywhere —
> it is a hash *input*, not a stored field — and `pipeline_fingerprint` hashes the user's
> original pipeline file, where the tree holds only a `to_json()` re-serialization.
>
> Making U-8 work therefore means threading "not asserted" through both producers,
> `ExecutionConfig` and the cross-check — an invasive change to the identity path — for the
> benefit of trees that never had an identity fence at all.
>
> ### U-10: mark the record; do not fabricate the identity
>
> **Migrate publishes per-image records carrying `provenance: "migrated"`.** For a record so
> marked:
>
> 1. **`valid_image_success` accepts it on artifact validity alone** — the artifacts exist
>    and their content proofs verify — with **no `work_id` comparison**.
> 2. **`resolve_run_state` emits one advisory** naming the affected images: the configuration
>    fence is unavailable for them.
> 3. **Every unmarked record is fenced exactly as before.** This is the half a careless
>    implementation breaks; `test_an_unmarked_record_is_still_fenced_on_work_id` is what
>    keeps it honest, because relaxing the comparison generally would strip the fence from
>    every modern tree — a far larger hole than the one U-10 opens deliberately.
>
> **What migrate writes into `work_id` for such a record.** Not a synthetic id dressed up as
> a real one. Either omit the key entirely, or carry the synthetic
> `sha256("migration:<ds>/<stem>")` **beside** the marking, where its provenance is
> unambiguous and no reader compares it. Decide in Task 2b and state which; what must not
> happen is a value in `work_id` that a reader could mistake for a content-derived identity.
>
> **The cost, stated once:** those images lose the configuration fence. Re-run the tree later
> under a different pipeline or `detect_mode` and it is **not** reprocessed.
> `test_a_migrated_tree_is_not_reprocessed_under_ANY_forward_config` asserts exactly that,
> including the case where reprocessing is arguably the better answer — the test documents
> the cost rather than hiding it.
>
> **Why this is the right trade and not the cheap one:**
>
> - It **removes no guarantee**. `git show v0.17.3:.../_cli_state_management.py` has no
>   `work_ids` key at all, and its config block holds seven entries of which five are digest
>   inputs. The fence never existed on these trees. U-10 declines to fabricate one.
> - It is **self-limiting**. Once an image is reprocessed by a modern run it acquires a real
>   `work_id` and the marking is gone — `test_the_marking_does_not_survive_reprocessing` is
>   the guard. The weakening cannot reach any image a v2 run has produced.
> - It is **visible**. An advisory naming the unavailable fence beats a fabricated id that
>   silently asserts a fence that is not there.
>
> **This applies only to trees migrated from the pre-markers shape.** A forward run always
> asserts its own identity; there is no path by which a modern run acquires the marking.
>
> **And this is what discharges MIG-21.** The pre-markers `--mode process` tree classifies
> `CONVERT`; before U-10, migrate converted nothing, so `_refuse_unmigrated_output`
> (`phenotypicCLI.py:1661-1662`, which runs **before** `--restart` is handled) refused the
> tree in every writing mode permanently — escapable only by `--overwrite`, which deletes the
> outputs. `--mode process` handles that tree in place today, so it was a **regression, not a
> gap**. With U-10 migrate emits marked records, `requires_conversion` returns `None`
> afterwards, and INV-DISCHARGEABLE's parametrized test covers the shape.

- [ ] **Step 1: Move `_migrate_legacy_success_evidence` into `_cli_migrate_state`**

Spec §11.1 assigns "every `_legacy_*` helper moves **into** migrate" to P7, and P6's
deletion ledger row 10 defers it here explicitly. **This is that helper**, and the
destination is migrate — not deletion.

`phenotypicCLI.py:560` (`_migrate_legacy_success_evidence`) and `:544`
(`_requires_legacy_success_migration`) become part of `_cli_migrate_state`. **Delete the
`--mode full` dispatch entirely** — `:2375-2378` and its user-facing echo at `:2380-2383`.
Per **U-7**, no migration logic remains on a normal run path; after P1's
`requires_conversion` gate, a legacy tree never reaches that code anyway.

**This is a rewrite, not a move**, and the plan should stop calling it one. The helper's
signature is `(state, config: ExecutionConfig, datasets: Sequence[Dataset], output_dir)`,
and migrate has neither `config` nor `datasets`. What moves is the *logic*: read
`ds_state.completed`, derive each image's identity, seed `work_ids`. What must be **built**
is the construction of its inputs from the tree, per the U-7 table above — `state.input_path`
for the source images, `<output>/pipeline.json` for the pipeline, the config block for the
four recoverable science fields, `results/<ds>/overlays/` and the master for the two
derivable ones, and migrate flags for the three that are neither.

**The ordering constraint an earlier draft stated is unsatisfiable, and the real seam is
one step earlier (MIG-19).** That draft required the promoter to run *after* the HDF→Zarr
conversion (so `image_data_artifact` resolves the store) **and** *before* the migrator's
publisher (so `_configured_work_id` hits). Those are the **same call**:
`migrate_image_task` (`_cli_migrate_image.py:434`) converts at `:463` and publishes at
`:567`, in one per-image function, run per task — optionally under `joblib.Parallel` — by
`_execute_migration_tasks` (`_cli_migrate.py:852-880`). There is no instant at which
conversion is complete for all images and publication has not begun.

**The seam that works:** run the promoter's **seeding half** before
`_ensure_migration_processing_state` (`_cli_migrate.py:567`), which mints the synthetic ids
at `:632` and writes them at `:662`/`:695` — but **only for stems not already present**
(`:628-634`, compared through `source_image_stem`). Seeded content-derived ids therefore
survive it, and `_configured_work_id` (`:125-140`, same `source_image_stem` comparison)
then hits for every image.

That puts the promoter *before* conversion, so its own `publish_image_success` half would
certify `.h5` artifacts about to be replaced — **harmless**, because
`image_completion_marker_path` is keyed by dataset/stem and `_valid_migration_marker`
(`_cli_migrate_image.py:207`) forces a republish against the store.

> **In migrate, the promoter's job is work-id seeding, not publication.** Split it
> accordingly: seed `state.config["work_ids"]` before `_ensure_migration_processing_state`;
> let `migrate_image_task` do the publishing it already does.
>
> Add an assertion that `_ensure_migration_processing_state` did **not** overwrite a seeded
> id — a regression there is silent. And note in a comment that the promoter keys `work_ids`
> by `image.name` while both readers compare via `source_image_stem`: that mismatch is *why*
> the seam holds, and a future edit that "tidies" it breaks the seam.

Three tests exercise the helper (`tests/unit/cli/test_cli_state_management.py:316`,
`test_cli_completion_store.py:606`, `test_embedded_measurement_migration.py:312`). **Keep
all three**; retarget them at the new call path rather than deleting them.

> **A NEW fixture is required — the existing one is not the floor shape (MIG-15).**
> `make_markerless` (`tests/unit/sdk_/_migration_fixtures.py:437-449`) calls
> `strip_completion_evidence` on a **modern** `build_completed_run`: it sets
> `success_markers_required = False` — **not absent** — and **retains content-derived
> `work_ids`**. So Task 2b Step 2's `assert _state_config(tree).get(
> "success_markers_required") is None` fails against it, and under U-6 it is not the
> pre-markers shape at all, since signal 5 keys on *absent* `work_ids`.
>
> Worse, a test built on it exercises `_configured_work_id`'s **hit** path — the branch
> where the lookup succeeds — which is precisely the blind spot MIG-10 found. It would
> pass while the defect it was written for is live.
>
> The new fixture must produce: `version="2.0.0"`, **no `work_ids` key**, **no**
> `success_markers_required` key, `datasets.<ds>.completed` populated, per-image `.h5`,
> no store, no `image_complete/`. That is the v0.17.3 shape; nothing in `tests/` builds
> it today.

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


def test_a_migrated_record_is_accepted_without_a_work_id_comparison(tmp_path):
    """U-10, and the assertion the whole ruling rests on.

    CAN-7's original form asserted that a migrated record carries the work_id a
    FORWARD RUN re-derives. Round 4 disproved that this is achievable: work_id
    hashes input_sha256 (file_sha256 of the ORIGINAL image, never persisted --
    only fed into the digest) and pipeline_fingerprint (the user's original file,
    where the tree holds only a to_json() re-serialization), and seven of the
    twelve processing_configuration_digest fields were never recorded at v0.17.3.

    So the record does NOT carry a matching work_id, and must not pretend to. It
    carries provenance="migrated", and valid_image_success accepts it on artifact
    validity alone.
    """
    from phenotypic.sdk_._image_record import read_image_record
    from phenotypic.sdk_ import valid_image_success

    tree = _build_v0_17_3_tree(tmp_path, detect_mode="rgb", overlay_alpha=0.7)
    _invoke_cli(mode="migrate", output=tree)

    for dataset, image in _images_of(tree):
        record = read_image_record(tree, dataset, image.stem)
        assert record["provenance"] == "migrated"
        assert valid_image_success(tree, dataset, image.stem) is True


def test_a_migrated_tree_is_not_reprocessed_under_ANY_forward_config(tmp_path):
    """The outcome CAN-7 exists to protect, now achieved by marking rather than by
    a fabricated identity. Two forward configs disagreeing on every science flag
    must BOTH reuse the migrated images -- because neither compares work_id.

    This is also the honest statement of what U-10 costs: the second config SHOULD
    arguably reprocess, and does not. That is the configuration fence being
    unavailable, which is what the advisory says.
    """
    for detect_mode, alpha in (("rgb", 0.7), ("gray", 0.2)):
        tree = _build_v0_17_3_tree(tmp_path / f"{detect_mode}", detect_mode="rgb",
                                   overlay_alpha=0.7)
        _invoke_cli(mode="migrate", output=tree)
        reprocessed = _invoke_cli(mode="full", output=tree,
                                  detect_mode=detect_mode, overlay_alpha=alpha)
        assert reprocessed.images_processed == 0, (
            "a migrated pre-markers tree reprocessed from source"
        )


def test_the_unavailable_fence_is_surfaced_as_an_advisory(tmp_path):
    """U-10's cost must be VISIBLE where the state is read. An advisory that says
    the fence is unavailable is the whole reason this is better than writing a
    fabricated work_id, which would silently assert a fence that does not exist."""
    from phenotypic.sdk_ import resolve_run_state

    tree = _build_v0_17_3_tree(tmp_path)
    _invoke_cli(mode="migrate", output=tree)

    advisories = resolve_run_state(tree, depth="deep").advisories
    assert any("migrated" in a and "configuration" in a for a in advisories)


def test_the_marking_does_not_survive_reprocessing(tmp_path):
    """U-10 is self-limiting, and this is the test that keeps it so. Once an image
    IS reprocessed by a modern run it acquires a real work_id, and the weakened
    fence must not persist. If this ever fails, the weakening has become permanent
    and spreads to images a v2 run has touched -- which is a different ruling from
    the one given."""
    from phenotypic.sdk_._image_record import read_image_record

    tree = _build_v0_17_3_tree(tmp_path)
    _invoke_cli(mode="migrate", output=tree)
    _invoke_cli(mode="full", output=tree, restart=True)     # forces real reprocessing

    for dataset, image in _images_of(tree):
        record = read_image_record(tree, dataset, image.stem)
        assert record.get("provenance") != "migrated", (
            "a reprocessed image kept the migrated marking; the unavailable fence "
            "is now permanent for an image a modern run produced"
        )


def test_an_unmarked_record_is_still_fenced_on_work_id(tmp_path):
    """The other half, and the one a careless implementation breaks: U-10 relaxes
    the check ONLY for marked records. If valid_image_success stops comparing
    work_id generally, every modern tree silently loses its configuration fence --
    a far larger hole than the one U-10 opened deliberately."""
    from phenotypic.sdk_ import valid_image_success

    tree = _build_modern_tree(tmp_path)                      # real work_ids, no marking
    dataset, image = next(iter(_images_of(tree)))
    _corrupt_record_work_id(tree, dataset, image.stem)

    assert valid_image_success(tree, dataset, image.stem) is False
```

- [ ] **Step 3: Make Task 3's deletion conditional, and give `datasets.failed` a home**

Never delete `datasets.{completed,failed,started}` in the same pass that failed to consume
them. Delete only after the tree has records for that dataset — cheap, and it converts a
silent data loss into a loud refusal if a future shape slips through.

**`datasets.failed` has no destination, and the conditional deletion does not cover it
(MIG-16).** Records are success-only, so conditioning on "records exist" discards the
failure set the moment any image succeeds. §4.1 makes terminal failures one of the **three
written authorities** precisely because *"a failure leaves no artifact"* — so deleting them
un-migrated is deleting an authority.

Convert them: each `datasets.<ds>.failed` entry becomes a `terminal_failures.jsonl` record
via `append_terminal_failure` (`_cli_failure_tracker.py:353`), which is where §4.1 says that
fact lives. A floor tree has no `failed_stage` or exception to record — write the migration
as the stage and a message naming the tree's vintage, rather than inventing detail the tree
does not contain.

**Condition the deletion of `failed` on that conversion having run**, not on records
existing.

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
store aggregates identically, subject to the fan-out and dtype corrections in CAN-10.

The master schema stamp is NOT minted here (CAN-9). finalize_run mints it where it
writes the master, so stamp and shape come from one code path; migrate leaves a
legacy master unstamped, which read_master_measurements correctly refuses as pre-v2
(U-3)."
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

**Rename `image_complete/` and `stage3_complete/` to
`.phenotypic/legacy-v2/`** — same filesystem, a directory rename, no byte copied — and have
`requires_conversion` ignore that path. `migrate --revert` then costs a rename back.

This puts a directory on disk that outlives migration, so be explicit about what it is:
**retained for revert, read by nothing.** It is not tracked state — nothing derives from it,
nothing must be kept in sync with it, and no verdict consults it. `_cli/CLAUDE.md`'s
register (Task 6) lists it under its own heading, *not* in the tracked-state table, and
says when it can be deleted.

**Three things the rename needs that the first draft left undefined:**

1. **`clear_machine_state` must preserve it (MIG-12).** That function rmtree's **every**
   child of `.phenotypic/` except `TERMINAL_FAILURES_JSONL` (`sdk_/_io_constants.py:1105-1116`),
   and `legacy-v2/` is such a child — so `--restart` would silently destroy the revert path.
   **P2 Task 1 already solves this exact coupling for `restart_epoch.json`, with a test.**
   Add `legacy-v2/` to the same `_PRESERVED_ON_RESTART` set and extend that test rather than
   writing a second mechanism.

   **Read `_PRESERVED_ON_RESTART`'s docstring before adding to it.** That docstring is the
   single home of the membership rule (P2 Task 1); this task does not restate it. Applying
   it: `legacy-v2/` qualifies — a restart is not a revert.

   The one thing not derivable from the set or its docstring, and therefore the only thing
   worth saying here: **`verification_cache.json`'s exclusion is enforced by
   `test_clear_machine_state_deletes_the_persisted_cache` in
   `tests/unit/sdk_/test_verification_cache_disk.py`** — a different suite from anything this
   task runs, so a wrong addition here goes red somewhere you are not looking.

2. **`--revert` must exist (MIG-13).** `test_migrate_is_revertible` invokes it; no flag, no
   rename primitive and no collision rule were ever specified, so the phase gate cannot be
   implemented as written. Define: `--mode migrate --revert` renames `.phenotypic/legacy-v2/`
   back over the current trees, refusing if `images/` has records the legacy trees do not
   cover.

3. **A second migrate must not collide on a non-empty `legacy-v2/` (MIG-13).**
   `os.replace` raises on a non-empty target directory and `shutil.move` **nests** inside it —
   and Step 1c *expects* a late old-build worker to recreate `image_complete/`, which is
   exactly how the second migrate arises. Follow `promote_store`'s existing move-aside
   discipline (`sdk_/ngff_.py:1737-1790`): a uuid-suffixed trash path, then replace, then
   reclaim. Do not invent a third rename protocol.

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

**And document `provenance` where the fence is described (U-10).** The register's job is to
say which state is tracked and how it is derived; a field that changes *how a record is
verified* and is absent from the guide is the exact failure this task exists to prevent.
State three things: `"migrated"` is written only by `--mode migrate`; a record so marked is
accepted on artifact validity with no `work_id` comparison; and **absent means `"forward"`**,
so the strict reading is the default and a writer that forgets the field produces a fenced
record rather than an accepted one. Say plainly that this is a deliberately weakened fence
for trees that never had one, that it clears on reprocessing, and that `resolve_run_state`
advises when it is in effect.

**And the same false claim in source, not only in the guide (flow-r4 Min1).**
`image_data_artifact`'s own docstring (`_cli_completion.py:132-135`) makes the claim CAN-3
disproved — the one the paragraph above corrects. A module guide fixed while the docstring
beside the function still asserts the opposite leaves the more authoritative copy wrong:
readers trust the code. Correct both in this step.

- [ ] **Step 2: Add the register — four headings, and the split IS the content**

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
| `processing_generation` | `sha256(pipeline_sha256 ‖ per_image_config_digest ‖ restart_epoch)` | `mint_run_identity` |
| `work_id` | content: schema version, dataset, input-relative path, input sha256, pipeline fingerprint, per-image config digest, mode | `work_id_for_image` |
| `inventory_digest` / `source_set_digest` / `scientific_config_digest` / `finalization_input_digest` | `config` fields + the verified set | `run_identity`, `finalization_input_object` |
| per-dataset completed/failed counts | the per-image records | `RunState.diagnostics` — **and nothing branches on them** |
| the master | the marker-authorized embedded tables, and **nothing else** | `finalize_run` step 1 (INV-INPUTS) |

State explicitly what was **deleted** and must not come back:
`processing_state.datasets.{completed,failed,started}` (a cache of a cache — already
re-aggregated from the event log on every load), `manifest.json` as evidence,
`publication_id`, and the event log as a completion source.

**(d) Neither tracked nor derived — retained, and read by nothing.** Two artifacts survive
on disk without belonging to any table above, and both need saying *because* they look like
counter-examples to (a)'s "four, and a fifth is a regression".

| Artifact | Written by | What consults it | When it may be deleted |
|---|---|---|---|
| `.phenotypic/legacy-v2/` | `--mode migrate`, by renaming `image_complete/` + `stage3_complete/` | Nothing. It exists only so `migrate --revert` is a rename back. | Once the migrated tree has been reprocessed, or the operator accepts migration is final. Deleting it costs the revert path and nothing else. |
| `.phenotypic/verification_cache.json` | `persist_states` after a deep pass | Only `resolve_run_state(depth="shallow")`, and only to **skip re-hashing** an artifact whose `(size, mtime_ns)` is unchanged. | Any time. A missing cache costs one deep pass. `clear_machine_state` deletes it — unlike `restart_epoch.json`, it is **not** preserved across `--restart`. |

**The rule that keeps both out of (a): nothing branches on them and no verdict is derived
from them.** Delete either one and every answer this system gives is identical, only slower
(the cache) or one option poorer (the revert). That is the whole test for whether a future
artifact belongs here or is a fifth tracked state.

**Say the cache's weaker guarantee here too, not only in its module docstring (U-11).**
In-process, "this entry lets a previously deep-verified result stand" meant *by this
process, minutes ago*. On disk it means *by some process* — possibly an older build with
different verification rules, possibly another user. It stays safe because every doubt
falls through to `deep`: entry absent, stat tuple moved, recorded identity ≠ current, file
missing, unreadable, or unparseable. A reader who needs to know why that list is exhaustive
should be sent to §9.1, not left to infer it.

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
