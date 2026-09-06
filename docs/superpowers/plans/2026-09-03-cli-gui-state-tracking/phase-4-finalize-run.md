# Phase 4 — Embedded-table inversion and `finalize_run`

**Depends on:** P3. **Blocks:** P5–P7. *(P0 no longer gates this phase — S-4 was cut by CAN-25 and its one real question lives in Task 1.)*

**Spec:** §7 (measurement and metadata data flow), D8 — as amended by
[D-A](OPEN-QUESTIONS.md#d-a-per-store-metadata-is-written-at-promote-time-not-backfilled).

**Goal:** embedded per-image tables carry **measurements only**; each store's user metadata
is written as `tables/metadata/pht-metadata.parquet` **in the same `.part` as the
measurements**, before the root `zarr.json`; the metadata join moves to finalization; and
`finalize_run` becomes the one aggregation + join + publish path for `full`, `measure` and
`recompile`.

**Local path only.** SLURM and `--njobs` fan-out is P5.

### What D-A changes from spec §7

`finalize_run` is **six steps, not seven** — step 6 ("backfill `pht-metadata.parquet` per
store — certified re-promote") is cut. The metadata table is written at promote time
instead, so **no NEW path** writes into a promoted store (**INV-PROVEN**, first obligation).

§7.4's late-metadata guarantee narrows correspondingly, and the narrowing must be
documented where users read it, not only here:

> A `metadata.csv` edit changes `metadata_sha256`, invalidating
> `finalization_input_digest`, so the next invocation re-runs `finalize_run` — re-joining
> the mirror. **Stores keep the metadata snapshot they were built against**; each store's
> **`attributes.phenotypic.metadata_table.snapshot_sha256`** records which one, and
> `resolve_run_state` raises an advisory when they diverge (P1 Task 5).

> **The key is `metadata_table`, not `metadata` — and this header used to say the wrong
> one.** `phenotypic.metadata` is already taken by the `{protected, public, imported}`
> image-metadata sections (`sdk_/ngff_.py:569-580`). P1 shipped the correct spelling and
> the reader for it: `_METADATA_TABLE_ATTR = "metadata_table"` (`sdk_/_run_state.py:377`,
> with the reasoning at `:364-376`) and `_store_metadata_snapshot`
> (`sdk_/_run_state.py:641-667`). Task 2 Step 3 always said `metadata_table`; this summary
> was the stale half, which is the "accurate correction appended, document left
> self-contradictory" defect in the document's own normative header. *(D-A carries the same
> stale spelling at `OPEN-QUESTIONS.md:95,99` — amended in the same commit as this plan.)*

### State-artifact budget: this phase adds nothing

EXECUTION.md's HARD STOP holds at **4 tracked / 3 content proofs / 2 neither**. Checked
against the repaired plan, every candidate is a *relocation* or a *deletion*:

| Change | Effect on the count |
|---|---|
| `attributes.phenotypic.metadata_table.snapshot_sha256` on the store root | **Relocation.** The digest leaves the measurements Parquet (which no longer describes a join) and lands in the root. One home before, one home after. Read only by an advisory, and *"an advisory is never a gate"* — so not tracked state by test 1 either. |
| `tables/metadata/pht-metadata.parquet` | A **data table** inside the store, not run state. Nothing branches on its presence to decide a verdict. |
| `source_set_digest` / `source_image_count` in the run proof | **Duplication into the run proof, net 0** — and classified as a **cross-check, not a second authority** (user ruling). See the row note below. |
| `sdk_/_master_io.py` | A module. No file, key, or field. |
| ~~`phenotypic.master_schema_version`~~ | **Cut** by user ruling — it would have been a new key something branches on, i.e. a fifth tracked artifact by test 1. |

> ### Row note — `source_set_digest` is duplicated, not relocated, and that is correct
>
> **⚠ CORRECTED.** An earlier version of this row said *"**Relocation** out of the aggregate
> proof … Net −1 field."* **The fields do not leave the aggregate proof, and cannot**:
> `current_aggregate_is_current` reads both off it at `_cli_completion.py:786-788`, and Task
> 3 Step 7 itself says they are published *"into **both** proofs"*. The arithmetic is
> **+2 in the run proof, −1 (`publication_id`) in each of two proofs → net 0**, and
> `source_set_digest` acquires a second on-disk home. A row asserting a −1 that does not
> happen is exactly the kind of number this change has been punished for.
>
> **Ruled (user): this is not a rule-3 violation.** The aggregate↔run binding **is a
> comparison of two independently computed values** — that is what makes it a check at all.
> If they shared one home there would be nothing to compare. That is categorically different
> from mirroring one fact into two places, for the same reason a checksum stored beside its
> data is not a second home for the data. **Classified as a cross-check, not an authority;
> net 0.**
>
> This is the distinction that separates it from the metadata digest one row above, where
> the two copies would have been the *same* fact asserted twice with no comparison between
> them — and which is therefore a relocation to a single home.

**If any repair to this plan would raise the count, stop and ask.** Two places where that
could happen are called out at their sites: Step 6b's ambiguity test (if v1-no-metadata and
v2 prove behaviourally distinguishable, the answer is the stamp — which is a stop), and the
cost-inversion measurement in Task 2 Step 3 (if it comes back bad, the answer is *not* to
re-mirror the digest).

### ⛔ STANDING RULE for every test in this phase: establish, then assert

> **Every assertion of a negative, or of an equality, must be preceded by an assertion that
> the fixture produced the thing whose absence or equality is being claimed.**

`assert x not in frame` is satisfied by an empty frame. `assert a == b` is satisfied by two
empty things. `pytest.raises`-free "must not raise" is satisfied by a fixture that never
built the state the raise was about. **In every one of those cases the test passes while
testing nothing**, and the failure is invisible because green is the expected colour.

This is not a hypothetical for this plan — it is its most-repeated defect, found at three
levels in one test alone:

1. the legacy-arm test's fixture did not reach the arm (the arm's *existence* was assumed);
2. its replacement did not either, for a different reason (`authorized_measurement_sources`
   never returned `None`);
3. and even reaching the arm, the poison would have been skipped by a recovery predicate, so
   the property was never exercised (`_aggregate_needs_image_name_recovery`).

Each layer looked green. **The plan already knew the discipline and applied it twice** —
`test_metadata_added_after_the_stores_still_joins_every_measured_row` carries *"the
`measured.height > 0` guard matters: without it the assertion is vacuously true on an
all-phantom frame"* — out of roughly eleven places that need it. The rule above is that
observation generalized so the next author does not have to rediscover it eleven times.

**The form to use**, wherever a fixture's product is what makes an assertion meaningful:

```python
assert master.height > 0, "fixture produced no rows; the assertion below is vacuous"
assert "GHOST.tif" not in master["Metadata_ImageFile"].to_list()
```

The guard's message says *why it is there*, so a later reader does not delete it as
redundant. **Reviewers: a bare negative or equality assertion with no preceding guard is a
finding, not a style note.**

---

## File Structure

| File | Responsibility |
|---|---|
| **Modify** `src/phenotypic/_cli/_embedded_measurement_tables.py:42` | `prepare_embedded_measurement_table` returns the **unjoined** baseline plus a separate metadata projection. |
| **Modify** `src/phenotypic/sdk_/_measurement_tables.py` | `write_embedded_measurement_table` (`:86-107`) and `build_measurement_table_descriptor` (`:109-129`) gain metadata-table analogues. **And repair `replace_embedded_measurement_table`'s in-place branch (`:285-291`)** — see below, the first draft's requirement was not sufficient (CAN-3, flow-r2 C5). |
| **Modify** `src/phenotypic/_core/_image_parts/_image_io_handler.py` | **Where the `.part` is actually written, and the row the first draft was missing (B4).** The table write and the root assembly are here, not in `sdk_`: `:1377-1388` writes the table, builds the descriptor, and folds it into the root written last. The payload parameter is typed `PreparedEmbeddedMeasurementTable` at **three** signatures — `:1076` in `save2zarr` (`:1069`), `:1139` in `_save_store` (`:1129`), `:1224` in `_write_store_part` (`:1213`) — and threaded at `:1126` and `:1204`. Task 1's type change reaches all five. |
| **Modify** `src/phenotypic/sdk_/CLAUDE.md:134-138` | **The THIRD home of the claim §0 falsified, and no phase named it** (found during P1 execution). |
| **Modify** `tests/unit/sdk_/test_ngff_promote.py:66` | `test_nothing_writes_into_a_promoted_store` — the name over-claims what the test proves. |
| **Modify** `src/phenotypic/_cli/_cli_output_manager.py:1871-1968` | **The FORWARD producer, and the one the first draft missed (B4).** `OutputManager.save_image_store` calls `prepare_embedded_measurement_table` at `:1936` and threads the result as `save_kwargs["measurement_table"]` at `:1948`. Missing *this* one un-inverts every image a normal `full` run writes — a strictly larger hole than the `--mode measure` one the plan already named. |
| **Modify** `src/phenotypic/_cli/_cli_output_manager.py:1970-2002` | `replace_image_store_measurements` feeds the **joined** producer at `:1993-1996`. Bring it onto `prepare_image_tables`, or `--mode measure` silently un-inverts every image it touches. |
| **Modify** the five `PreparedEmbeddedMeasurementTable(...)` fixture constructors | The type change's test-side blast radius, measured: `tests/_output_layout.py:173`, `tests/unit/gui/results_viewer/conftest.py:105`, `.../test_measurement_source.py:41`, `.../test_measurement_routes.py:54`, `tests/unit/cli/test_embedded_measurement_aggregation.py:134`. |
| **Create** `src/phenotypic/sdk_/_master_io.py` | The **one home** for the v1/v2 master discrimination and its retirement condition (Task 3 Step 6). **No schema stamp is minted** — user ruling, see that step. |
| **Modify** `tests/unit/sdk_/test_run_state_layering.py:60-68` | INV-LAYER's watched set is seven modules listed by `__file__` (M6). A new `sdk_` module that `sdk_/__init__.py` re-exports is unwatched until it is added to `_MODULES`; the comment at `:52-59` is the argument for why. Add `_master_io` in the same commit that creates it. |
| **Modify** `src/phenotypic/_cli/_embedded_measurement_tables.py:106-131` | `embedded_measurement_table_matches` is **reclaim authority**, not provenance (M1) — six migrator call sites (`_cli_migrate.py:1374,1380`; `_cli_migrate_image.py:293,329,838,857`). See below. |
| **Modify** `src/phenotypic/sdk_/_measurement_tables.py:132-231` | `_valid_embedded_measurement_contract`. **What changes is not what the first draft said** — see the provenance-triple block below. The measurements table's own triple stays *valid unchanged*; what the contract does not yet cover is the new `tables/metadata/` group and descriptor (M2). |
| **Modify** `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py:186-202` | The **third** `_consistent_embedded_join_keys` call site (M2), at script-generation time, serialising `"metadata_join_keys"` into the finalizer task (`:200`). Retiring the function from `finalize_run` alone leaves the abort firing **at submission**, before any worker runs. Delete the task-schema field in the same commit. |
| **Modify** `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py:557,569` | **The marker site the 28 strict xfails name, absent from the first draft's repoint table (B3).** `marker_path = image_completion_marker_path(...)` and `marker.get("version") != SUCCESS_MARKER_VERSION`. Its two `RuntimeError("Cannot restore marker authority: …")` arms (`:430`, `:493`) are where the handled-domain-error xfail instances land. |

> **A third copy of the false claim survives this phase unless it is fixed here.**
> The README's *"the change is not done until the two module guides say what is tracked"*
> names `_cli/CLAUDE.md` (P7 Task 6) and `gui/CLAUDE.md` (P6 Task 8). There is a **third**,
> in the module guide for the package that actually owns `promote_store`:
>
> `sdk_/CLAUDE.md:134-138` — *"**Nothing writes into a promoted store**, and the root
> `zarr.json` is written **last** — the completion marker and the viewer's staleness scan
> both key on that root alone, so violating either makes both report stale data as fresh
> with nothing failing."*
>
> That is spec §6.2's ⚠ claim verbatim. After P7 Task 6 corrects `_cli/CLAUDE.md:251-254`,
> this becomes the **surviving** copy — and the most authoritative-looking one, since it sits
> beside the code it describes. Correct it in the same commit that repairs the branch, so the
> guide and the behaviour change together.
>
> **And the guard it rests on does not cover what its name says.**
> `tests/unit/sdk_/test_ngff_promote.py:66` `test_nothing_writes_into_a_promoted_store` proves
> only that `promote_store` is a **rename rather than a merge-in-place** — it builds a part,
> promotes, and asserts the directory and chunk inodes changed. It never exercises
> `replace_embedded_measurement_table`'s in-place branch, which is the path that actually
> writes into a promoted store today. Its docstring goes further and names the three
> subsystems that "rest on" the invariant — the completion markers, the viewer's staleness
> scan, and `valid_staged_store` — which are precisely the three §0 says are unprotected.
>
> **Rename it to what it proves** (`test_promote_store_replaces_rather_than_merges`) and move
> the three-subsystem claim into the new test that covers the repaired branch. A test whose
> name asserts more than its body is how a reader gets from *"there is a guard called that"*
> to *"the invariant holds"* — which is the exact inference CAN-3 showed was false.

> **The in-place branch cannot be fixed by "refresh the root with the table" alone (C5).**
> `build_measurement_table_descriptor` (`:109-129`) returns exactly `{schema_version, type,
> format, path, measurement_columns, target}` — **no digest, no join keys, no join status** —
> and `measurement_columns` is the *pre-join baseline* tuple. So a metadata edit that changes
> values but not the measurement schema leaves `current == descriptor` **true**, the in-place
> branch fires, and the Parquet is rewritten inside a promoted store with no `.part` and no
> root rewrite.
>
> **After P4's inversion this gets worse, not better:** the descriptor becomes a pure
> function of the measurement schema and the objmap target, so *every* metadata-driven
> re-measure takes that branch.
>
> Two changes, both required:
> 1. **Widen what `current == descriptor` compares, so it tests "has anything the store
>    certifies changed" rather than "did the column list change".** The first draft said
>    "put `metadata_snapshot_sha256` into the descriptor". **Under the user's ruling that is
>    no longer the right shape** (see Task 2 Step 3): after the inversion the *measurements*
>    table carries no join at all, so its recorded digest is `""` on every store and putting
>    `""` in its descriptor discriminates nothing. The digest lives in the store root's
>    `metadata_table` block instead. So the staleness comparison must widen to that block
>    **and** to the new `tables.metadata` descriptor — i.e. `replace_…` compares the whole
>    of what the store certifies, not one table's column list.
> 2. **Refresh the root whenever the payload changes**, not only on a descriptor change.
>
> **(2) alone is sufficient for correctness; (1) is what keeps the fast path meaningful.**
> If only (2) ships, every `--mode measure` re-promotes and the in-place branch becomes
> dead code — which is a legitimate outcome, and cheaper to reason about. State which of
> the two you shipped in the commit message, because a reader cannot tell from the code.
>
> **State the cost honestly.** (2) means the copytree/hardlink re-promote runs on **every**
> `--mode measure`, not just on a descriptor change. That is precisely spike S-1's cost —
> which D-A cut, on grounds the ledger already records as false (CAN-3): the mechanism
> already ships. So the cost is now *larger* than when S-1 was dropped and is still
> unmeasured. If `--mode measure` on a large tree becomes slow, this is the reason, and
> measuring it is a follow-up, not a blocker for P4.

> **M1 — the shape change reaches the migrator, and in two opposite directions.**
> `embedded_measurement_table_matches` builds its expected table from
> `prepare_embedded_measurement_table(...)` **including `parquet_metadata()`** — join status,
> join keys, snapshot digest, measurement columns — and asserts
> `actual.equals(expected, check_metadata=True)`. Its docstring calls row count *"reclaim
> authority rather than an incidental property"*, so this is a correctness gate, not a
> convenience check.
>
> - **`_cli_migrate_image.py:284-300`** re-writes the store when `not exact` (the comparison
>   is at `:293`, the rewrite at `:294-300`). A shape change makes `exact` False on **every
>   pre-P4 store**, so P4 would create a **fourth** post-proof store-write path — which
>   INV-PROVEN's own wording forbids ("Do not write a fourth"). Gate the comparison on the
>   store's schema, so a pre-inversion store is compared against pre-inversion expectations.
>   **That gate is a compatibility branch and must carry its retirement condition at the
>   site**, in the form EXECUTION.md now requires (user, 2026-09-06).
> - **`_cli_migrate.py:1369-1384`** (`:1374`, and the re-check inside the publication commit
>   at `:1380`) and **`_cli_migrate_image.py:827-870`** (`:838`, `:857`) under
>   `--delete-sources`: a mismatch keeps the sources, which is the safe direction — but
>   `--delete-sources` then becomes **permanently impossible** on any pre-P4 store. Say so
>   in P7's docs rather than letting a user discover it.
> - The sixth site is `_cli_migrate_image.py:329` (`if not embedded_measurement_table_matches(...)`),
>   the post-write validation. Six call sites, measured at `869e9dee`; the first draft's
>   `_cli_migrate.py:1331,1337` / `_cli_migrate_image.py:278,314,777,796` were all stale.
| **Create** `src/phenotypic/_cli/_cli_finalize_run.py` | `finalize_run(output_dir, …)` — the one path. ~260 lines. |
| **Modify** `src/phenotypic/_cli/_cli_output_manager.py:1351` | `_aggregate_measurements_unlocked` delegates aggregation to `finalize_run`. |
| **Modify** `src/phenotypic/_cli/_cli_recompile_worker.py:805` | `_run_post_master_steps` becomes a `finalize_run` call. Its master write is separate and earlier (`:771-782`), and **re-raises** on a CSV failure at `:778` — see Task 4 Step 4. |
| **Modify** `src/phenotypic/_cli/_cli_completion.py:1011` | Aggregate proof's `required_outputs` drops the `master_csv` descriptor (D8); the assembled list is at `:1047`. |
| **Delete** (D8) | See Task 4 Step 4 for the **measured** dependency set. The four symbols the first draft named are the tip of it, and two of the dependents are in the GUI. |
| **Modify** `src/phenotypic/_cli/_cli_recompile_tables.py` (333 lines) | **Hard-`isinstance`-checks `PreparedEmbeddedMeasurementTable` at `:102`** — the exact type Task 1 replaces — raising `TypeError` on the new one. Its `DIR_IMAGE_COMPLETE` read at `:183` is **deliberate and stays** (B2). |
| **Modify** `src/phenotypic/_cli/_cli_recompile_recovery.py` (860 lines) | Five executable marker-format sites: `:409`, `:499`, `:659`, `:731` (`image_completion_marker_path`) and `:804` (`SUCCESS_MARKER_VERSION`). |
| **Test** `tests/unit/cli/test_finalize_run.py` *(new)* | INV-INPUTS, the six steps, the three entry points. |

> **These two modules are a three-round blind spot, and the largest one found (gen-r3).**
> 1,193 lines that read the marker format and type-check the producer Task 1 inverts —
> appearing in **zero plan documents, zero ledger entries, and zero reviewer reports across
> three rounds**. Recompile is one of the three entry points §7.4 routes through
> `finalize_run` and P4 Task 4 parametrizes a byte-identical master over, so its internals
> were in scope from the first draft; nobody opened them.
>
> `_cli_recompile_tables.py:102` fails **closed** — `TypeError("Recompile table preparation
> returned an invalid payload")` — so this surfaces as a crash rather than corruption, which
> is the good direction. But it means `--mode recompile` is broken from the moment Task 1
> lands until this is fixed, and no test in the plan covers it.
>
> ### The marker-site census, with the criterion that makes it reproducible
>
> An earlier draft said *"Measured: 33 marker-format reads across the two files"* with no
> definition of "marker-format read", so the number could neither be reproduced nor used to
> check the sweep for completeness — the shape of P3's 384-that-was-really-1152. Replaced
> with a criterion and its output, both re-derivable in one command:
>
> ```bash
> grep -rn "image_completion_marker_path\|SUCCESS_MARKER_VERSION\|DIR_IMAGE_COMPLETE" \
>   src/phenotypic/_cli/_cli_recompile_tables.py \
>   src/phenotypic/_cli/_cli_recompile_recovery.py \
>   src/phenotypic/_cli/_cli_recompile_slurm_scripts.py
> ```
>
> **15 textual occurrences at `869e9dee`**, of which 4 are imports (`_cli_recompile_tables.py:12`,
> `_cli_recompile_recovery.py:26,39`, `_cli_recompile_slurm_scripts.py:15,36` — five import
> lines across three files) and 2 are prose (`_cli_recompile_tables.py:133`,
> `_cli_recompile_recovery.py:57`). That leaves **eight executable sites**:
>
> | Site | Token | Fate |
> |---|---|---|
> | `_cli_recompile_recovery.py:409` | `image_completion_marker_path` | repoint |
> | `_cli_recompile_recovery.py:499` | `image_completion_marker_path` | repoint |
> | `_cli_recompile_recovery.py:659` | `image_completion_marker_path` | repoint |
> | `_cli_recompile_recovery.py:731` | `image_completion_marker_path` | repoint |
> | `_cli_recompile_recovery.py:804` | `SUCCESS_MARKER_VERSION` | → `RECORD_VERSION` |
> | `_cli_recompile_slurm_scripts.py:557` | `image_completion_marker_path` | repoint |
> | `_cli_recompile_slurm_scripts.py:569` | `SUCCESS_MARKER_VERSION` | → `RECORD_VERSION` |
> | `_cli_recompile_tables.py:183` | `DIR_IMAGE_COMPLETE` | **leave alone** — P3's deliberate legacy arm (B2) |
>
> **Seven sites to repoint, one to leave alone.** The census is not the whole edit: the
> *fields* each site reads off the payload still have to be checked against the record
> schema. The record is a superset of the marker, so a field-by-field port is mechanical —
> but "mechanical" is a claim about the fields, and this census does not establish it.
> Enumerate them before editing, the way P3 Step 3b enumerates the staged-engine sites.

### What recompile actually needs from a record — and why the collapsed form is NOT simpler

> ### ⛔ CORRECTED: `publish_image_record` merges `stages` and nothing else (B1)
>
> An earlier draft of this section claimed:
>
> > **P3's merge rule removes the need for the round trip entirely** — `publish_image_record`
> > merges rather than replaces (CAN-6), so recompile publishes the **new `artifacts` only**
> > and the merge preserves identity and `stages` untouched. … ~40 lines → one call.
>
> **That is false against the shipped function, and the merge rule's own docstring says so
> in the sentence the claim was drawn from** (`_cli_image_record.py:99-101`):
>
> > **`stages` is a contribution, not a replacement (CAN-6 rule 1).** It is unioned with
> > whatever is on disk.
>
> The scoping is exact — `stages`, and only `stages`. `work_id`, `dataset`, `image_stem`,
> `relative_image_path`, `mode`, `attempt_id` and `lifecycle_epoch` are **required
> keyword-only parameters** (`:80-95`), and at `:158-174` the record is rebuilt from the
> arguments with `_existing_stages` consulted for `stages` alone:
>
> ```python
> merged = _existing_stages(output_dir, dataset, image_stem)     # :158
> merged.update({str(key): dict(value) for key, value in stages.items()})
> record = {..., "work_id": work_id, ..., "stages": merged, "artifacts": descriptors, ...}
> ```
>
> **Following the withdrawn claim fails silently, not loudly.** Either `TypeError` at the
> call site (loud, fine), or — if an implementer supplies placeholders to satisfy the
> signature — the record's `work_id` is overwritten, `valid_image_success` rejects every
> recompiled image, `authorized_measurement_sources` returns `{}`, and `finalize_run` writes
> an **empty master and raises nothing**. That is verbatim the CAN-22 failure the comment at
> `_cli_completion.py:955-964` exists to prevent: *"`{}` is a VALID schema-3 result meaning
> 'no successful measurements yet', so P4's `finalize_run` would write an empty master and
> raise nothing. A successful-looking run that discarded every measurement."*
>
> **So the seven-field read-back stays**, and the "~40 lines → one call" estimate is
> withdrawn with the claim that justified it. What may still go is `_marker_artifacts`'
> duplicated containment check — `publish_image_record` already does
> `resolve(strict=True)` + `relative_to(output_root)` at `:152-156`. **But not the rest of
> `_marker_artifacts`:** it re-resolves **every** artifact role on the payload
> (`_cli_recompile_tables.py:41-57`), and `_republish_table_marker` (`:60-84`) hands the
> whole map back so all of them are re-fingerprinted. Publishing "the new `artifacts` only"
> would shrink the certified set to `measurements` and silently drop the `image` and
> `overlay` roles from `valid_image_success`' coverage.

Recompile uses the per-image payload for exactly two things:

1. **An identity round-trip after rewriting a table.** `_republish_table_marker`
   (`_cli_recompile_tables.py:60-84`) reads seven fields — `work_id`, `dataset`,
   `relative_image_path`, `image_stem`, `mode`, `attempt_id`, `lifecycle_epoch` — and hands
   every one straight back to `publish_image_success` with freshly-resolved artifacts. It
   must re-publish because rewriting the embedded table invalidates the payload's artifact
   digests. **P3 already repointed the file it reads**: `_replace_and_republish_table`
   passes `record_path = image_record_path(...)` at `:139`, with the comment at `:126-138`
   naming that fix and the `recompile_store_lock_path` half of it. The seven fields are read
   off the **record** today; nothing here is P4's work.
2. **Discovery fallback.** `_standalone_marker_sources` (`:150-200`) discovers "valid
   embedded authority when no processing state is present".

| Site | State at `869e9dee` | P4's change |
|---|---|---|
| `_republish_table_marker` (`:60-84`) | **Already reads the record** (P3). | **None required.** Optionally drop `_marker_artifacts`' duplicated `relative_to` check (`:41-57`), keeping its every-role resolution. |
| `_standalone_marker_sources` (`:150-200`) | **Already scans BOTH shapes, each on its own predicate** — `(progress / DIR_IMAGE_RECORDS, record_rejection), (progress / DIR_IMAGE_COMPLETE, marker_rejection)` at `:181-183`. Its docstring (`:151-176`) calls itself "the seventh site of the defect fixed in `authorized_measurement_sources`" and states why the legacy arm stays. | **None. Do not touch it.** ⛔ The first draft said *"glob `DIR_IMAGE_RECORDS` instead of `DIR_IMAGE_COMPLETE`"* — that **reverts P3's fix** and re-breaks legacy trees for the whole P4→P7 window, because the schema gate that would otherwise refuse a legacy tree is disarmed until P7. |
| `:102` | `isinstance(prepared, PreparedEmbeddedMeasurementTable)` | → `PreparedImageTables`. **This is the crash**, and the only live one in this file: it fails closed with `TypeError("Recompile table preparation returned an invalid payload")`, so `--mode recompile` breaks on the first run after Task 1 lands. |
| `_cli_recompile_recovery.py:39` (import), `:804` (use) | `SUCCESS_MARKER_VERSION` | → `RECORD_VERSION`. |
| `_cli_recompile_recovery.py:26` (import), `:409`, `:499`, `:659`, `:731` | `image_completion_marker_path` | → `image_record_path`. **Four call sites, not five** — the fifth the first draft counted (`:57`) is a docstring mention inside `recompile_store_lock_path`. |
| `_cli_recompile_slurm_scripts.py:36` (import), `:557` | `image_completion_marker_path` | → `image_record_path`. **The site the 28 strict xfails name by number**, absent from the first draft's table entirely. |
| `_cli_recompile_slurm_scripts.py:15` (import), `:569` | `SUCCESS_MARKER_VERSION` | → `RECORD_VERSION`. Path and version **move together** — those functions return `None`/`False` on a version mismatch, so a path-only repoint disables overlay and table-authority repair silently. That is the xfail's own reason string. |
| **Test** `tests/unit/cli/test_embedded_table_inversion.py` *(new)* | — | Intrinsic/user metadata boundary; curation re-keying. |

### ⚠ RULED (user, 2026-09-06): the repointed recovery sites read BOTH shapes

`_standalone_marker_sources` reads both, deliberately, because the schema gate stays
disarmed until P7 (`phase-7-migrate-mode.md:1279`) and a legacy tree can still reach
`--mode recompile` in that window. **The same is true of every site in the table above, so
they get the same treatment: both shapes, each on its own predicate — P3's precedent.**

The record-only alternative was rejected on the same ground as B7's legacy arm: it would
make legacy trees lose overlay and table-authority repair for the whole P4→P7 window, and
silently — those functions return `None`/`False` on a version mismatch rather than raising.
Consistency with the shape P3 already established matters more here than saving a branch.

**Each legacy arm carries the retirement condition** EXECUTION.md requires (user,
2026-09-06), with the same trigger as B7's: *delete when the schema gate is armed and
refuses legacy trees before they reach recompile (P7 Task 5 Step 1d)*. One trigger for
every legacy arm this phase adds means they retire together rather than one at a time.

**Delete the xfail tripwires in the same commit as the repoint.** They are strict, so they
go red — as a *failure* — the moment the repoint lands, and left in place they record the
debt as paid while the work is still pending (drift-register Entry 32's shape):

- `_RECOMPILE_READS_THE_LEGACY_MARKER_UNTIL_P4`: **2 definitions**
  (`tests/unit/cli/test_cli_recompile.py:75`, `tests/unit/cli/test_cli_recompile_slurm.py:51`)
  and **28 decorations** (3 and 27 occurrences respectively, minus the two definitions).
- One further inline instance with its own reason string, carrying stale line numbers:
  `tests/unit/cli/test_embedded_measurement_recompile.py:31-40`.

**Budget for the instances whose post-repoint outcome is not knowable from the current run.**
An `xfail(strict=True)` that fails for a *handled domain error* rather than the defect tells
you nothing about what it does once the defect is gone. Two named cases:

- The `RuntimeError("Cannot restore marker authority: …")` arms at
  `_cli_recompile_slurm_scripts.py:430` and `:493`.
- `tests/unit/cli/test_cli_recompile_slurm.py:2905` — `assert not overlay.exists()` sits
  *after* the `pytest.raises(SlurmGenerationInactiveError)` block (`:2892-2903`). While the
  test fails with `DID NOT RAISE`, **line 2905 has never executed**. When the repoint makes
  the raise happen, that assertion runs for the first time and its outcome is unknown. This
  is drift Entry 34's mechanism; budget for it rather than discovering it at the gate.

**Test that `--mode recompile` still round-trips**, since nothing in the plan covered it
before: recompile a two-image tree and assert each record's identity fields and `stages` are
byte-identical to before, with only `artifacts` digests changed.

---

## Interfaces

**Produces:**

```python
# phenotypic._cli._cli_finalize_run

def finalize_run(
    output_dir: Path,
    *,
    dataset_names: Sequence[str],
    include_dataset_column: bool = True,          # H4 -- see below; NOT optional to omit
    pipeline: "ImagePipeline | None" = None,
    metadata_csv: Path | None = None,
    no_qc: bool = False,
    study_config: dict | None = None,
    shard_paths: Sequence[Path] | None = None,   # P5 supplies these; None = local concat
    commit_guard: "CommitGuard | None" = None,
) -> Path | None:
    """The one aggregation + join + publish path (spec §7.4)."""
```

> **`include_dataset_column` is not droppable (H4).** The first draft's signature omitted
> it, while Task 4 Step 3 says `_aggregate_measurements_unlocked` "delegates its body" — and
> it is a live parameter *of that body*, threaded end to end:
>
> | Site | Role |
> |---|---|
> | `_cli_output_manager.py:1354` | declared on `_aggregate_measurements_unlocked` |
> | `_cli_output_manager.py:1452` | consumed by `aggregate_parquet_files` |
> | `_cli_output_manager.py:1548`, `:1566` | supplied by `aggregate_measurements` |
> | `_cli_output_manager.py:2032` | supplied by `OutputManager.aggregate_master_csv` from `self.include_dataset_column` |
> | `_cli_recompile_slurm_scripts.py:149`, `:200` | serialised into the recompile SLURM finalizer task |
> | `_cli_recompile_worker.py:368` | read back off the task dict |
>
> Dropping it changes behaviour for every `include_dataset_column=False` run and breaks the
> recompile task schema at both ends.

```python
# phenotypic._cli._embedded_measurement_tables

@dataclass(frozen=True)
class PreparedImageTables:
    measurements: pd.DataFrame          # intrinsic identity only, NO user metadata
    metadata: pd.DataFrame | None       # user metadata rows + join keys, or None
    measurement_columns: tuple[str, ...]
    # The join provenance below describes the METADATA table, not the
    # measurements table. See the block immediately after this listing.
    join_status: Literal["joined", "not_requested", "no_common_keys"]
    join_keys: tuple[str, ...]
    metadata_snapshot_sha256: str

def prepare_image_tables(
    measurements: pd.DataFrame, metadata_csv: Path | None
) -> PreparedImageTables: ...
```

> ### Which provenance triple lands on which file (H7 — answered by the user's ruling)
>
> `join_status`, `join_keys` and `metadata_snapshot_sha256` are read by three gates, so
> leaving their post-inversion values unstated leaves three behaviours unspecified:
> `_valid_embedded_measurement_contract` (`sdk_/_measurement_tables.py:132-231`, gating
> `replace_embedded_measurement_table`'s in-place branch at `:285-291`);
> `embedded_measurement_table_matches` (`_cli/_embedded_measurement_tables.py:106-131`,
> which compares with `check_metadata=True`, so every migrator decision depends on the exact
> bytes); and `_consistent_embedded_join_keys` (being retired, but live at
> `_cli_recompile_slurm_scripts.py:186-202` until its task-schema field goes).
>
> **After the inversion the measurements table carries no join at all**, so by the
> contract's own rule (`sdk_/_measurement_tables.py:215-218`) its triple is:
>
> | File | `phenotypic.join.status` | `phenotypic.join.keys` | `phenotypic.metadata.snapshot_sha256` |
> |---|---|---|---|
> | `tables/measurements/table.parquet` | `"not_requested"` | `[]` | `""` |
> | `tables/metadata/pht-metadata.parquet` | the real status | the real keys | the real digest |
> | store root `attributes.phenotypic.metadata_table` | `join_kind` | `join_keys` | `snapshot_sha256` |
>
> **`_valid_embedded_measurement_contract` therefore needs no change for the measurements
> table** — `not_requested` with empty keys and empty digest is exactly the shape it already
> accepts (`:216-218`). The File Structure row that said it "rejects `join_status ==
> 'not_requested'` with a non-empty digest — but the inverted producer must record the
> snapshot digest on an *unjoined* table" described a conflict that the ruling dissolves:
> the digest does not go on that file. What the contract *does* need is coverage of the new
> `tables/metadata/` group document and descriptor (M2, Task 2 Step 3).
>
> The `join_kind`/`join_left`/`join_right` triple on the measurements table
> (`right`/`metadata`/`measurements`) is required unconditionally by `:201-206` and is not
> conditioned on `join_status`, so it rides along unchanged. **Note this makes those three
> constants on that file** — a reader can no longer learn anything from them there. That is
> the price of not changing the contract's required-key set in this phase; record it, do not
> quietly rely on it.

**Consumes:** P3's `publish_image_record`; `phenotypic.sdk_.promote_store`,
`MEASUREMENT_TABLE_RELATIVE_PATH`.

---

## Task 1: Split the embedded table into measurements and metadata

**Files:**
- Modify: `src/phenotypic/_cli/_embedded_measurement_tables.py:42`
- Test: `tests/unit/cli/test_embedded_table_inversion.py`

**This is subtraction, not invention.** `prepare_embedded_measurement_table` already
computes `measurement_columns` from the baseline **before** joining
(`_embedded_measurement_tables.py:55`) and writes it as
`phenotypic.measurement_columns`. "Embedded table without user metadata" is exactly that
existing projection.

- [ ] **Step 1: Write the failing tests**

```python
def test_intrinsic_identity_stays_in_the_measurement_table(tmp_path):
    """Spec §7.1: a concatenated row that cannot say which image it came from is
    unusable. Metadata_ImageFile, Metadata_Dataset and the object label stay."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(_measurements_with_metadata(), _metadata_csv(tmp_path))
    assert "Metadata_ImageFile" in prepared.measurements.columns
    assert "Metadata_Dataset" in prepared.measurements.columns


def test_user_metadata_leaves_the_measurement_table(tmp_path):
    """§7.3's contract change. Metadata_Strain came from --metadata, not from the
    image, so it belongs in pht-metadata.parquet."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(_measurements_with_metadata(), _metadata_csv(tmp_path))
    assert "Metadata_Strain" not in prepared.measurements.columns
    assert "Metadata_Strain" in prepared.metadata.columns


def test_the_measurement_table_equals_the_pre_join_baseline_exactly(tmp_path):
    """The boundary already has a name: measurement_columns, computed from the
    baseline BEFORE joining (_embedded_measurement_tables.py:55). This asserts the
    new split IS that projection rather than a re-derivation of it."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    baseline = _measurements_with_metadata()
    prepared = prepare_image_tables(baseline, _metadata_csv(tmp_path))
    assert tuple(prepared.measurements.columns) == prepared.measurement_columns


def test_no_metadata_table_when_the_join_was_not_requested(tmp_path):
    """§7.2: absence is the honest signal."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(_measurements_with_metadata(), None)
    assert prepared.metadata is None
    assert prepared.join_status == "not_requested"


def test_no_metadata_table_when_no_columns_are_in_common(tmp_path):
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(
        _measurements_with_metadata(), _unrelated_metadata_csv(tmp_path)
    )
    assert prepared.metadata is None
    assert prepared.join_status == "no_common_keys"


def test_duplicate_metadata_keys_preserve_fan_out(tmp_path):
    """The behaviour prepare_embedded_measurement_table already warns about, and
    the one S-4 spiked. Losing it silently changes row counts in the mirror."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(
        _measurements_with_metadata(), _metadata_csv_with_duplicate_keys(tmp_path)
    )
    assert len(prepared.metadata) == 3
```

- [ ] **Step 2: Run to verify failure**

Run the same command as the step that follows. **Expected: every test in the file red, and
the failure text must name the symbol under test as missing.** A collection ERROR from a
different cause — a fixture typo, a bad import elsewhere in the file — is also red, and is
not evidence this step passed. Read the reason, not the colour.

- [ ] **Step 3: Implement**

`prepare_image_tables` keeps `prepare_embedded_measurement_table`'s normalization and its
`prepare_metadata_join_keys` call, and then **stops before the right join**
(`_embedded_measurement_tables.py:90-96`). `measurements` is the baseline; `metadata` is
the semi-join of the metadata frame onto that image's distinct join keys.

**The `not_requested` and `no_common_keys` early returns move with it**
(`:56-63` and `:75-81`): both already return the baseline unchanged, so under the inversion
they return `metadata=None` and the same triple they record today. Only the third return
(`:97-103`) changes shape.

*(S-4 was cut — CAN-25 showed it computed `M ⋉ K_i` against `(M ⋉ K_all) ⋉ K_i` where
`K_i ⊆ K_all`, a set-theoretic identity whose FAIL branch was unreachable. Its one real
question — metadata-only rows appearing as phantoms in the mirror and in no store's metadata
table, and a fan-out key matching two images not being double-counted — is covered by this
task's own tests below, which is where it always belonged.)*

Keep `prepare_embedded_measurement_table` as a thin wrapper for one release **only if** a
caller outside this change needs it; grep first, and delete it if not.

- [ ] **Step 4: Run the tests**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/cli/test_embedded_table_inversion.py -q
```

**Expected: `6 passed`.** The count is the floor, not decoration — a single-path invocation
that collects nothing exits **5 (`NO_TESTS_COLLECTED`)** and prints `no tests ran`, which
under `-q` is easy to read past. Six is the number of `def test_` in Step 1; if the run
reports fewer, tests are missing, not passing.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_embedded_measurement_tables.py \
        tests/unit/cli/test_embedded_table_inversion.py
git commit -m "feat(cli): split the embedded table into measurements and metadata

Spec §7.1-7.2. Subtraction, not invention: measurement_columns already recorded
this boundary, computed from the baseline before the join."
```

---

## Task 2: Write both tables at promote time

**Files:**
- Modify: `src/phenotypic/sdk_/_measurement_tables.py`
- Test: `tests/unit/cli/test_embedded_table_inversion.py`, `tests/unit/sdk_/`

Implements **D-A** and **INV-PROVEN**.

- [ ] **Step 1: Write the failing tests**

```python
def test_both_tables_land_in_the_same_part_before_the_root(tmp_path):
    """D-A / INV-PROVEN. The root zarr.json is written last and is the record's
    content anchor (_store_artifact_matches, _cli_completion.py:110-123), so
    anything written after it is a mutation of a proven artifact. Writing metadata
    in the same .part is what makes the backfill unnecessary."""
    store = _build_store_with_metadata(tmp_path)
    assert (store / "tables" / "measurements" / "table.parquet").is_file()
    assert (store / "tables" / "metadata" / "pht-metadata.parquet").is_file()
    root = json.loads((store / "zarr.json").read_text())
    assert "metadata" in root["attributes"]["phenotypic"]["tables"]
    # M2: a Zarr v3 hierarchy needs a group document for the new group too.
    assert (store / "tables" / "metadata" / "zarr.json").is_file()


def test_the_store_records_the_metadata_snapshot_it_was_built_against(tmp_path):
    """D-A: stores keep the metadata they were built with, and say which one. That
    is what lets resolve_run_state DERIVE the divergence advisory instead of
    tracking a backfill stage.

    The key is `metadata_table`, NOT `metadata` -- P1 shipped the reader against
    that spelling (sdk_/_run_state.py:377, read at :641-667) because
    `phenotypic.metadata` is taken by the image-metadata sections.
    """
    store = _build_store_with_metadata(tmp_path)
    root = json.loads((store / "zarr.json").read_text())
    assert root["attributes"]["phenotypic"]["metadata_table"]["snapshot_sha256"]


def test_a_metadata_free_run_records_no_metadata_table_block(tmp_path):
    """H2. `resolve_run_state`'s divergence advisory fires when a store's recorded
    snapshot is neither None nor the run's current `metadata_sha256`
    (sdk_/_run_state.py:1295-1300). On a run with no --metadata, `metadata_sha256`
    is None and the producer records the digest as "" -- so writing the block
    unconditionally makes `"" not in (None, None)` true and reports EVERY store on
    EVERY metadata-free run as diverged.

    The code two blocks above that filter carries the reason this matters:
    "An advisory that is always on teaches people to ignore the one that will
    matter" (:1280-1281).
    """
    store = _build_store_without_metadata(tmp_path)
    root = json.loads((store / "zarr.json").read_text())
    assert "metadata_table" not in root["attributes"]["phenotypic"]
    assert "metadata" not in root["attributes"]["phenotypic"]["tables"]


def test_a_no_common_keys_run_still_records_its_snapshot(tmp_path):
    """The other half of H2, and the reason the rule is not simply "omit whenever
    there is no metadata table".

    `no_common_keys` means a metadata.csv WAS supplied and matched nothing. There is
    no pht-metadata.parquet to write, but the store was still built against a
    specific snapshot, and a later edit to that snapshot is exactly the divergence
    the advisory exists to surface. So: omit the block only for `not_requested`.
    """
    store = _build_store_with_unrelated_metadata(tmp_path)
    root = json.loads((store / "zarr.json").read_text())
    assert root["attributes"]["phenotypic"]["metadata_table"]["snapshot_sha256"]
    assert "metadata" not in root["attributes"]["phenotypic"]["tables"]


def test_finalize_run_writes_no_byte_into_a_proven_store(tmp_path):
    """INV-PROVEN, first obligation: no NEW path writes into a promoted store.

    Publish a record, snapshot every mtime under the store, run finalize_run, and
    assert not one file moved. This is the test that would have caught the
    backfill if it had shipped."""
    from phenotypic._cli._cli_finalize_run import finalize_run

    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a")
    before = {p: p.stat().st_mtime_ns for p in sorted(store.rglob("*")) if p.is_file()}

    # STANDING RULE. This is INV-PROVEN's ONLY gate, and `before == after` is
    # satisfied by `{} == {}` -- which is what a store path that does not exist,
    # or a publish that silently did nothing, produces. Establish the snapshot is
    # a real one before trusting its stability.
    assert before, "fixture published no store files; before == after is vacuous"
    assert (store / STORE_ROOT_JSON) in before, "the root zarr.json is not in the snapshot"

    finalize_run(tmp_path, dataset_names=["plate"])
    after = {p: p.stat().st_mtime_ns for p in sorted(store.rglob("*")) if p.is_file()}
    assert before == after, "finalize_run mutated a store that carries a content proof"


def test_measure_mode_refreshes_the_table_and_the_root_together(tmp_path):
    """INV-PROVEN, second obligation -- and the reason the invariant is stated the
    way it is (CAN-3).

    The stronger claim ("nothing ever writes into a proven store") is FALSE and was
    false before this change. --mode measure re-measures from stores and calls
    replace_embedded_measurement_table (sdk_/_measurement_tables.py:242), whose
    IN-PLACE branch (:285-291) fires when the descriptor is unchanged: it rewrites
    tables/measurements/table.parquet directly in the promoted store, with no
    .part, no copytree, and NO ROOT REWRITE.

    Two things break as a result, and both are silent:
      1. the record's store digest still matches, so the proof certifies content
         that changed underneath it;
      2. `snapshot_sha256` lives in the root, so D-A's divergence advisory reads a
         value this branch never refreshes -- it reports stale metadata as current.
    """
    from phenotypic.sdk_ import STORE_ROOT_JSON

    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a", metadata=True)
    root_before = (store / STORE_ROOT_JSON).read_bytes()

    _run_measure_mode(tmp_path, metadata=_edited_metadata_csv(tmp_path))

    root_after = (store / STORE_ROOT_JSON).read_bytes()

    # `root_after != root_before` PASSES FOR THE WRONG REASON and its message
    # overclaims: a re-promote rewrites the journal's `applied_at_utc` and
    # `duration_seconds` (omitted only under reproducible_provenance), so the root
    # differs whether or not the table or the snapshot was refreshed. Kept as a
    # cheap smoke check with an honest message; THE SECOND ASSERTION IS THE
    # LOAD-BEARING ONE and must not be weakened or reordered away.
    assert root_after != root_before, "the root was not rewritten at all"

    assert _snapshot_sha256(store) == _sha256_of(_edited_metadata_csv(tmp_path)), (
        "the embedded table was rewritten without refreshing the root's recorded "
        "snapshot, so the per-image proof still certifies the old digest and the "
        "divergence advisory reads a stale value -- INV-PROVEN's second obligation"
    )


def test_measure_mode_writes_the_metadata_table_not_a_joined_one(tmp_path):
    """INV-PROVEN, second obligation, other half.

    replace_image_store_measurements feeds prepare_embedded_measurement_table --
    the JOINED producer -- at _cli_output_manager.py:1993-1996. P4 Task 1 replaced
    that producer with prepare_image_tables everywhere else. If this call site is
    missed, --mode measure on an inverted tree writes joined tables and no
    pht-metadata.parquet, silently un-inverting every image it touches.

    There are TWO producers, not one (B4). The forward one --
    OutputManager.save_image_store at :1936, threading save_kwargs["measurement_table"]
    at :1948 -- is the larger hole, because it is on the path every `full` run takes.
    Parametrize this test over both modes rather than testing `measure` alone.
    """
    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a", metadata=True)
    _run_measure_mode(tmp_path, metadata=_metadata_csv(tmp_path))

    measurements = _read_embedded_measurements(store)
    assert "Metadata_Strain" not in measurements.columns
    assert (store / "tables" / "metadata" / "pht-metadata.parquet").is_file()
```

- [ ] **Step 2: Run to verify failure**

Run the same command as the step that follows. **Expected: every test in the file red, and
the failure text must name the symbol under test as missing.** A collection ERROR from a
different cause — a fixture typo, a bad import elsewhere in the file — is also red, and is
not evidence this step passed. Read the reason, not the colour.

- [ ] **Step 3: Implement**

**The `.part` writer is in `_core`, not `sdk_` (B4).** `_core/_image_parts/_image_io_handler.py:1377-1388`
writes the measurement table, builds its descriptor, and folds it into the root written
last; `sdk_/_measurement_tables.py` owns only the two pieces it calls,
`write_embedded_measurement_table` (`:86-107`) and `build_measurement_table_descriptor`
(`:109-129`). Both halves change, and the parameter's **type** changes at three signatures
(`:1076`, `:1139`, `:1224`) plus two threading sites (`:1126`, `:1204`).

Extend the writer to emit `tables/metadata/pht-metadata.parquet` when
`prepared.metadata is not None`, in the same `.part` and before the root. Extend the root's
`attributes.phenotypic.tables` with a `metadata` descriptor, and write the
`tables/metadata/zarr.json` group document alongside it (M2) — `write_embedded_measurement_table`
already writes group documents for `tables/` and `tables/measurements/` (`:95-104`), and
`_valid_embedded_measurement_contract` checks them **by exact equality** (`:168-181`), so a
missing one is a contract failure rather than a cosmetic omission. Give the new descriptor
its own schema-version constant, the analogue of `MEASUREMENT_TABLE_SCHEMA_VERSION`.

**The snapshot digest goes in a NEW root key, and the name matters (flow-r2 C5).** An
earlier draft said to add `attributes.phenotypic.metadata = {"snapshot_sha256": …}`. That
key is **already taken**: `PhenotypicAttr.METADATA` holds `{protected, public, imported}`
image-metadata sections (`sdk_/ngff_.py:569-580`), carrying things like bit depth
(`:1130-1138`). Writing a snapshot digest there would collide with per-image metadata.

Use `attributes.phenotypic.metadata_table`:

```json
"metadata_table": {"snapshot_sha256": "…", "join_keys": [...], "join_kind": "…"}
```

### ⚠ RULED (user, 2026-09-06): the digest gets ONE home — the store root. It is not mirrored.

An earlier draft of this step ended *"Mirroring it into the root at promote time … keeps the
Parquet copy as the authority the Parquet itself carries. **Write both; never derive one
from the other at read time.**"* **That sentence is struck.** It is literally what
EXECUTION.md's HARD STOP rule 3 rejects — *"Does it add a second home for a value that
already exists? → that is the defect this change removes, not a fix."*

**The resolution is not an exception to rule 3; the premise changed.** Today the digest is
one leg of a self-describing provenance triple on the measurements Parquet —
`join_status`, `join_keys`, `snapshot_sha256` — whose coherence two readers check
(`sdk_/_measurement_tables.py:216-227`, `_cli_output_manager.py:940-965`). **After this
phase's inversion the measurements table carries no join at all**, so by that contract's own
rule its triple becomes `not_requested` / `[]` / `""`. **The digest stops being true of that
file.** It therefore *relocates* to `attributes.phenotypic.metadata_table.snapshot_sha256`
on the store root — where P1 already shipped the reader (`sdk_/_run_state.py:641-667`) — and
the join provenance belongs to the new **metadata** table's own Parquet KV. One home per
fact, every consumer reading it there. No register row for a second home is needed, because
there is no second home.

**Why the root and not the metadata Parquet.** D-A's divergence advisory reads the digest
from `sdk_` on the deep path, and P1 Task 5 describes that as "one attribute read per store
… from a value the store already carries". Read from a Parquet it is not an attribute read —
it is **opening a Parquet footer per store**, a different cost and a new dependency on the
INV-LAYER plain-JSON path, and §9.2's numbers do not include it. The store root is already
being read by that path.

> #### ⛔ REQUIRED BEFORE THE DESIGN IS COMMITTED: measure the cost inversion (user ruling)
>
> The mixed-authority refusal `_consistent_embedded_join_keys`
> (`_cli_output_manager.py:914-966`) **currently gets the digest for free**: it is already
> calling `pq.read_schema(path)` per store (`:940-941`) for `join_status` and `join_keys`,
> and pulls the digest out of the same schema object (`:958`). Moving the digest to the root means
> that reader opens **N store roots in addition to** N Parquet footers — a *cost inversion
> for that reader*, in exchange for the advisory's saving.
>
> **Measure it before committing the design; do not assume it.** The comparison is: (a) N
> `pq.read_schema` calls, as today; (b) N `pq.read_schema` + N root-`zarr.json` reads. Use a
> store-count in the range the run actually sees. If (b) is materially worse, the answer is
> **not** to re-mirror the digest — it is that `_consistent_embedded_join_keys` is being
> retired by Task 3 anyway (CAN-2), so measure whether *anything else* still needs the
> per-store triple after that retirement. Report the numbers; every number in this plan must
> be one someone measured.
>
> *(This measurement is a gate on the design, not on the phase: if it comes back bad, stop
> and ask rather than reinstating the mirror.)*

**Omit the block entirely when `join_status == "not_requested"` (H2).** The advisory fires
when a store's recorded snapshot is neither `None` nor the run's current `metadata_sha256`
(`sdk_/_run_state.py:1295-1300`). A metadata-free run has `metadata_sha256 = None` and a
producer digest of `""`, so an unconditional block makes `"" not in (None, None)` true and
reports **every store of every metadata-free run** as diverged. `_store_metadata_snapshot`
returns `None` when the key is absent (`sdk_/_run_state.py:664-667`), which is the correct
reading for a store that never had a snapshot. Write the block for `joined` **and** for
`no_common_keys` — the latter had a snapshot even though it produced no table.

> **That is a WRITE rule, and on its own it leaves the same advisory always-on through the
> measure path (NEW-2).** "Omit when `not_requested`" says nothing about a block that is
> *already there*. A store built with `--metadata` and then re-measured **without** it goes
> through `replace_embedded_measurement_table`'s root refresh; if that refresh only adds and
> updates keys, the stale `metadata_table.snapshot_sha256` survives, the run's
> `metadata_sha256` is now `None`, `_store_metadata_snapshot` returns the stale digest, and
> `sha not in (None, current_metadata)` fires the advisory on every such store — H2's exact
> failure arriving through the measure path instead of the promote path.
>
> **State the rule as a total function of `join_status`, not as an append:**
>
> | `join_status` | `metadata_table` block | `tables.metadata` descriptor |
> |---|---|---|
> | `joined` | written | written |
> | `no_common_keys` | written (digest only) | **absent** |
> | `not_requested` | **removed if present, else absent** | **removed if present, else absent** |
>
> The third row is the one an append-only implementation gets wrong, and it is invisible
> until someone re-measures without `--metadata`.

```python
def test_re_measuring_without_metadata_clears_the_stores_metadata_block(tmp_path):
    """NEW-2. The removal half of H2's rule.

    Omitting the block on a metadata-free BUILD is not enough: a store that LOSES
    its metadata must lose the block too, or `_store_metadata_snapshot` keeps
    returning a digest that no longer describes anything and the divergence
    advisory fires on every such store forever.
    """
    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a", metadata=True)
    assert _root(store)["attributes"]["phenotypic"]["metadata_table"]["snapshot_sha256"]

    _run_measure_mode(tmp_path, metadata=None)

    phenotypic = _root(store)["attributes"]["phenotypic"]
    assert "metadata_table" not in phenotypic
    assert "metadata" not in phenotypic["tables"]
    assert not (store / "tables" / "metadata" / "pht-metadata.parquet").exists()
```

The metadata table's own Parquet KV keys carry the join, self-describing from the file
itself (§7.2): `phenotypic.join.keys`, `phenotypic.join.kind`,
`phenotypic.metadata.snapshot_sha256`. That is the property that makes the store useful to a
third party at all, and Task 5 is the test of it.

**Order (M1 — corrected).** The first draft stated the write order as *"chunks → both tables
→ `OME/zarr.json` → root `zarr.json`"* and called it load-bearing. **The actual order is
arrays → OME group + `OME/zarr.json` (`_image_io_handler.py:1368-1375`) → tables
(`:1377-1388`) → root last (`:1390+`).** Root-last is preserved either way, so this is not a
correctness defect — but a sequence a plan calls load-bearing must be stated correctly, or
the next reader "fixes" the code to match the plan. **The obligation is exactly one thing:
the root `zarr.json` is written last, and `promote_store` follows it.** Both tables must
land before the root; where they sit relative to the OME group does not matter.

- [ ] **Step 4: Run the tests plus the NGFF conformance suite**

```bash
# Task 2's own tests -- run FIRST and alone, so their count is unambiguous.
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/cli/test_embedded_table_inversion.py -q

# Then the suites the new table can break.
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/sdk_ tests/unit/core/test_ngff_conformance.py \
  tests/unit/cli/test_embedded_measurement_tables.py -q
```

**Expected: Task 2 adds `8` tests in total** — the four store-shape tests, the two H2
tests (`..._records_no_metadata_table_block`, `..._no_common_keys_run_still_records_its_snapshot`),
NEW-2's removal test, and `test_measure_mode_writes_the_metadata_table_not_a_joined_one`.
**The first command's `passed` count plus whatever lands under `tests/unit/sdk_/` must sum
to 8, and neither location may collect zero.** The second command reports 0 failures and a
non-zero collected count for each of its three paths.

Task 2's **Files** list names two test locations (`tests/unit/cli/test_embedded_table_inversion.py`
and `tests/unit/sdk_/`), so a floor tied to one file is only half the gate — which is why
the total is stated against the task rather than against a path.

The store gains a table, so its NGFF conformance must be re-checked — a non-conforming
store is one `napari` cannot open, which is half of why it is OME-Zarr.

> ### ⛔ Why this step is split, and why a multi-path command needs per-path counts
>
> **This is the step the `tests/_ngff_conformance.py` defect was found in, and the first
> repair fixed the path without fixing the mechanism.** With three paths in one invocation,
> a path that collects **zero** is invisible: the other two collect, the total is non-zero,
> and the exit code is 0. The command reports PASS having never run the thing it names.
>
> Two things close it, and both are cheap:
>
> 1. **Task 2's own tests run alone**, so their floor is checkable against a single number.
>    An earlier draft's command did not include
>    `tests/unit/cli/test_embedded_table_inversion.py` at all — the file Task 2's own
>    **Files** list names — so Task 2's gate would not have run Task 2's tests.
> 2. **Verify collection before trusting a multi-path run**, once, when the paths change:
>
>    ```bash
>    for p in tests/unit/sdk_ tests/unit/core/test_ngff_conformance.py \
>             tests/unit/cli/test_embedded_measurement_tables.py; do
>      printf '%s: ' "$p"
>      QT_QPA_PLATFORM=offscreen uv run pytest "$p" --collect-only -q 2>&1 | tail -1
>    done
>    ```
>
> **The general rule for this plan: every verification command states an expected output
> that a zero-collection run cannot satisfy.** `PASS` alone never qualifies.

> **`tests/_ngff_conformance.py` is not a test file.** An earlier draft passed it to pytest
> directly. It is a **helper module** — leading underscore, no `test_*` functions, exporting
> `assert_store_conforms` and `assert_ome_xml_valid` — and `pyproject.toml` sets no custom
> `python_files`, so naming it on the command line collects **zero tests** and the step
> would have reported PASS having checked nothing. The conformance suite that exercises it
> is `tests/unit/core/test_ngff_conformance.py`; ten further modules import the helper, so
> the `tests/unit` gate in Task 4 Step 5 is what actually covers them all.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/sdk_ src/phenotypic/_cli tests/unit
git commit -m "feat(sdk): write pht-metadata.parquet in the store's original promote

D-A. No post-proof store mutation exists on any forward path, so §6.3's hardlink
re-promote and §6.4's receipt generalisation are both unnecessary. INV-PROVEN is
pinned by a property test over every file's mtime across a finalize_run."
```

---

## Task 3: `finalize_run` — the one path

**Files:**
- Create: `src/phenotypic/_cli/_cli_finalize_run.py`
- Test: `tests/unit/cli/test_finalize_run.py`

The seam already exists and is already shared: `finalize_post_master_outputs`
(`_cli_output_manager.py:969`) is called by both the forward path (`:1526`) and the
recompile worker (imported at `_cli_recompile_worker.py:816`, called at `:843` and `:853`),
whose own comment (`:791-795`) says it is "matching the forward CLI path". What is **not**
shared is aggregation. This task widens the seam to own it.

- [ ] **Step 1: Write INV-INPUTS first — the phase's gate**

```python
def test_finalize_run_ignores_every_stale_intermediate(tmp_path):
    """INV-INPUTS / spec §7.5. Plant a stale chunk parquet, a stale shard, a stale
    _dataset_aggregated.parquet, a stale analysis_full.parquet and a stale master,
    each containing a row that exists in NO embedded table. Assert the new master
    matches a concat of the embedded tables exactly.

    Those files are outputs and intermediates of a PREVIOUS finalization, not inputs
    to this one. Under a rolling input, reusing any of them silently omits images
    that arrived since the cache was built, or retains rows for an image whose
    content changed and therefore has a new work_id.
    """
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import master_measurements_parquet_path

    _publish_two_successful_images(tmp_path)
    poison = pl.DataFrame({"Metadata_ImageFile": ["GHOST.tif"], "Shape_Circularity": [0.0]})
    _plant_stale_chunk_parquet(tmp_path, poison)
    _plant_stale_shard(tmp_path, poison)
    _plant_stale_dataset_aggregate(tmp_path, poison)
    _plant_stale_analysis_full(tmp_path, poison)
    _plant_stale_master(tmp_path, poison)

    finalize_run(tmp_path, dataset_names=["plate"])

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))

    # STANDING RULE. Both assertions below hold when both frames are EMPTY: the
    # `not in` trivially, and `.equals` because two empty frames are equal. That is
    # the CAN-22 outcome -- an empty master written with no exception -- passing the
    # test written to catch stale inputs.
    expected = _concat_of_embedded_tables(tmp_path)
    assert master.height > 0, "finalize_run wrote an empty master; both asserts are vacuous"
    assert expected.height > 0, "fixture published no embedded tables to compare against"

    assert "GHOST.tif" not in master["Metadata_ImageFile"].to_list()
    assert master.equals(expected)


def test_finalize_run_ignores_a_stale_aggregate_on_a_legacy_external_parquet_tree(tmp_path):
    """B7: the case the test above CANNOT reach, and the reason it must exist.

    The `_dataset_aggregated.parquet` preference lives on the LEGACY arm only
    (_cli_output_manager.py:1421-1431): `authorized_measurement_sources` returns
    None -> `flush_trailing_measurements_if_chunked` -> `discover_measurement_sources`,
    which prefers the aggregate (_measurement_sources.py:132-134, :161-166). On a forward tree
    `authorized_sources` is NOT None, so that arm is unreachable and the poisoned
    aggregate is ignored whether or not the arm survives into finalize_run.

    Worse: the arm calls `flush_trailing_measurements_if_chunked`, which
    MANUFACTURES `_dataset_aggregated.parquet` from `chunks/` -- the exact input
    §7.5 forbids.

    THE FIXTURE IS A LEGACY TREE, and the name says so. See the two blocks below
    for why nothing weaker works and why this one must be built exactly as
    described.
    """
    _build_legacy_external_parquet_tree(tmp_path)          # no progress payloads
    _plant_stale_dataset_aggregate(tmp_path, _poison())    # carries Metadata_ImageName
    finalize_run(tmp_path, dataset_names=["plate"])
    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    assert "GHOST.tif" not in master["Metadata_ImageFile"].to_list()


def test_finalize_run_invalidates_the_intermediates_on_success(tmp_path):
    """§7.5: so a later invocation cannot mistake them for inputs."""
    from phenotypic._cli._cli_finalize_run import finalize_run

    _publish_two_successful_images(tmp_path)
    chunk = _plant_stale_chunk_parquet(tmp_path, _poison())
    # STANDING RULE. `not chunk.exists()` is trivially true if the fixture never
    # created it, which makes the invalidation this test pins unobservable.
    assert chunk.is_file(), "fixture planted nothing; `not chunk.exists()` is vacuous"

    finalize_run(tmp_path, dataset_names=["plate"])
    assert not chunk.exists()


def test_the_master_carries_no_user_metadata(tmp_path):
    """§7.3's contract change, stated as a test.

    The one genuinely dangerous failure mode in §7 is code that filters the master
    on a user-metadata column: it returns EMPTY rather than erroring. NO schema
    stamp guards that (user ruling -- see Step 6); the guard is the v1/v2
    discrimination in sdk_/_master_io.py, applied at the readers that need it."""
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import (
        master_measurements_parquet_path,
        measurements_parquet_path,
    )

    _publish_two_successful_images(tmp_path, metadata=True)
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    assert "Metadata_Strain" not in master.columns
    assert "Metadata_Strain" in mirror.columns


def test_curation_re_keying_still_works_against_the_intrinsic_master(tmp_path):
    """§7.3 names this as needing an explicit test rather than assumption.

    Curation deliberately reads the CLEAN master so labels survive for curated-out
    objects (_curation_labels.py:408-425, reading layout.master_parquet at :417). It keys
    on dataset / image / object-label --
    all intrinsic -- so it should be unaffected. Test it; do not assume it."""
    _publish_two_successful_images(tmp_path, metadata=True)
    _finalize_and_curate(tmp_path, curated_out=["a.tif::3"])
    assert _curated_label_survives(tmp_path, "a.tif::3")


def test_master_measurements_csv_is_gone(tmp_path):
    """D8: master is parquet-only. The un-joined master is no longer the file a
    human opens -- the mirror is."""
    from phenotypic._cli._cli_finalize_run import finalize_run

    _publish_two_successful_images(tmp_path)
    finalize_run(tmp_path, dataset_names=["plate"])
    assert not (tmp_path / "deliverables" / "master_measurements.csv").exists()
```

> ### ⛔ The legacy-arm fixture has TWO preconditions; missing either makes that test green by construction
>
> **That test exists because the first draft's version was green by construction. The
> replacement was too** — for a different reason, caught in review — and there is a *second*
> trap one level below that. Both are stated here because neither is visible from the test
> body.
>
> #### Precondition 1 — `authorized_measurement_sources` must actually return `None`
>
> An earlier draft's fixture was `_publish_two_images_without_processing_state`, with the
> docstring *"makes `authorized_measurement_sources` return None (no processing state, or
> `success_markers_required` false)"*. **Neither stated condition returns `None`.** Both
> delegate to `_sources_without_state` (`_cli_completion.py:932-935`):
>
> ```python
> if state is None or not state.config.get("success_markers_required", False):
>     return _sources_without_state(output_dir)          # :932-935
> ```
>
> and `_sources_without_state` (`:852-916`) globs **both** progress trees, returning `None`
> **only when neither holds a single `*/*.json`** (`:889-895`). Its own docstring says so at
> `:876-881`: *"`None` … when neither tree exists at all — which is the signal callers read
> as 'fall back to legacy source discovery', and must not be confused with an empty mapping"*.
> A fixture that **publishes two images** — records, markers, either — gets a non-`None`
> mapping back, `finalize_run` takes the authorized arm, and the poisoned aggregate is
> ignored **for the same reason the first test already ignored it**.
>
> **So: no per-image payload under `.phenotypic/progress/` at all.** Authority comes from
> legacy external Parquets at `results/<ds>/measurements/*.parquet` — a genuine legacy tree,
> which is also the only shape that makes the aggregate preference meaningful.
>
> *(A corrupt `processing_state.json` is a second route to `None` — `load_processing_state`
> raising `KeyError`/`TypeError`/`ValueError` returns `None` directly at
> `_cli_completion.py:928-931`. **Rejected**: it reaches the arm without producing the
> legacy tree the arm exists to serve, so the test would pin the corrupt-state path rather
> than §7.5's subject. The chosen route is in the test's name so a reader cannot mistake
> which.)*
>
> #### Precondition 2 — the poisoned aggregate must survive the recovery predicate
>
> **Reaching the legacy arm is not enough; the arm must actually PREFER the aggregate.**
> `discover_measurement_sources` guards the preference (`_measurement_sources.py:161-167`):
>
> ```python
> aggregate_needs_recovery = (
>     aggregated.exists() and _aggregate_needs_image_name_recovery(aggregated)
> )
> if aggregated.exists():
>     if aggregate_needs_recovery and individual_paths:   # -> SKIPS the aggregate
> ```
>
> `_aggregate_needs_image_name_recovery` (`:51-58`) returns `True` when the frame carries no
> column owned by `IMAGE.IMAGE_NAME`, resolved through `_image_name_column` (`:39-48`) and
> `metadata_member_for_header`.
>
> **Measured: `IMAGE.IMAGE_NAME` is `Metadata_ImageName`, and
> `metadata_member_for_header("Metadata_ImageFile")` is `None`.** The plan's `_poison()`
> frame is `{"Metadata_ImageFile": [...], "Shape_Circularity": [...]}` — it carries **no**
> `IMAGE.IMAGE_NAME` column, so the predicate returns `True`, and with individual Parquets
> present (which precondition 1 requires) the aggregate is **skipped for recovery**. The
> test would then pass because `discover_measurement_sources` never chose the poison — not
> because `finalize_run` ignored it. Green by construction, one level below the bug the
> review caught.
>
> **The poison frame for this test must therefore carry `Metadata_ImageName`** — dtype
> `String`, non-null, non-empty after `strip_chars`, and not matching `_UUID_PATTERN`
> (`:24-27`). `"GHOST.tif"` satisfies all four. Keep `Metadata_ImageFile` too: it is what
> the assertion reads and what the sibling tests use.
>
> **Assert both preconditions rather than trusting them.** Before `finalize_run`, assert
> `authorized_measurement_sources(tmp_path) is None` and
> `not _aggregate_needs_image_name_recovery(<aggregate path>)`. Two lines, and they turn
> both traps from silent-green into a named failure.

- [ ] **Step 2: Run to verify failure**

Run the same command as the step that follows. **Expected: every test in the file red, and
the failure text must name the symbol under test as missing.** A collection ERROR from a
different cause — a fixture typo, a bad import elsewhere in the file — is also red, and is
not evidence this step passed. Read the reason, not the colour.

- [ ] **Step 3: Implement the six steps**

```python
def finalize_run(output_dir, *, dataset_names, include_dataset_column=True,
                 pipeline=None, metadata_csv=None, no_qc=False, study_config=None,
                 shard_paths=None, commit_guard=None):
    """Aggregate, join, publish -- one path for `full`, `measure` and `recompile`.

    Six steps (spec §7.4, minus the backfill D-A cut):

    1. select marker-authorized embedded measurement tables
    2. concat  ->  master_measurements.parquet          (un-joined, D8: no CSV)
    3. join metadata + append metadata-only phantoms + apply post ops
    4. write  ->  deliverables/measurements.{parquet,csv}
    5. persist pipeline.json, analysis outputs, per-feature splits
    6. publish aggregate proof -> run proof

    INVARIANT (INV-INPUTS, §7.5) -- **step 1 selects exactly the marker-authorized
    embedded measurement tables.** It never reads a prior master, chunk parquet,
    measurement shard, ``analysis_full.parquet`` or ``_dataset_aggregated.parquet``
    as an aggregation input. Those are outputs and intermediates of a PREVIOUS
    finalization; under a rolling input, reusing one silently omits images that
    arrived since, or retains rows for an image whose content changed and therefore
    has a new ``work_id``. Master is a pure function of the currently authorized
    embedded tables -- which is the derivability property this whole design is for.

    ``shard_paths`` is P5's fan-out hook: when supplied, step 2 merges those instead
    of reading the tables directly. It does not weaken INV-INPUTS, because the shards
    were themselves produced from authorized embedded tables **in this invocation**,
    and ``measurement_shards/`` is emptied when fan-out begins, so a prior run's
    shards can never be merged.
    """
```

> ### ⚠ CORRECTED at P2 close (user-ruled): the guarantee is CLEARING, not namespacing
>
> This docstring used to end *"namespaced by `scheduler_epoch` so a prior run's
> shards can never be merged."* **That guarantee did not hold on the local path.**
>
> `_run_state._scheduler_epoch()` returns **`None`** whenever there is no
> `slurm_lifecycle.json` — every local run — and P5 gives itself a local
> process-pool driver. So the namespace key is `None` for every local
> invocation, consecutive local runs share one directory, and a prior run's
> shards merge into this one's master. Silently, and against the INV-INPUTS this
> very paragraph invokes two lines earlier.
>
> **The ruling: empty `measurement_shards/` when fan-out begins.** Three options
> went to the user; this one was chosen, and the two rejections are the
> instructive part:
>
> - **A per-invocation shard id was rejected** — it would take the state-artifact
>   budget from four to five, which P7 Task 6's register calls a design
>   regression, and it reverses D3's whole direction.
> - **Falling back to `processing_generation` was rejected because it does not
>   deliver the guarantee.** It is content-derived, so two consecutive identical
>   local runs mint the *same* generation and collide anyway. It narrows the
>   window instead of closing it — **worse than either alternative, because it
>   looks like a fix.**
>
> Clearing is also *strictly stronger* than namespacing, which was not obvious
> going in: namespacing leaves every prior run's shards on disk forever,
> accumulating. Emptying closes the hole and stops the accretion.
>
> **`scheduler_epoch` in the path is no longer load-bearing for correctness — and
> do not remove it as part of this.** Whether the directory stays namespaced is
> P5's call; the guarantee no longer depends on it either way. The name itself is
> still correct: §5.1's collapse withdrew the rename of the CLI *writers*, never
> the reader's `RunIdentity.scheduler_epoch` (`sdk_/_state_types.py:79`).
>
> **The caveat that makes this a fix rather than a data-loss bug** is in
> `phase-5-fanout.md`, where the clearing is implemented: it happens at fan-out
> **start, before any worker writes** — never at merge time.

> **Before rewriting anything in `finalize_post_master_outputs`, inventory what it already
> does.** Its docstring numbers **five** steps at `_cli_output_manager.py:990-1017` — the
> metadata handling this task changes, `_apply_post_to_master`, `_seed_measurements`, the
> per-feature splits, and the pipeline/analysis/QC block. **Plus three side effects that are
> in no numbered list at all** (H5 — the first draft counted one):
>
> | Un-numbered side effect | Site | Why dropping it matters |
> |---|---|---|
> | `migrate_legacy_qc(output_dir)` | `:1063-1068` | Runs **first**, before the join. A legacy QC tree silently stops being migrated. |
> | `order_measurement_columns(post_df.columns)` | `:1106` | The canonical column contract; see "Keep the column ordering" below. |
> | `write_rembi_manifest(...)` → `deliverables/rembi.yaml` | `:1115-1129` | Explicitly fed from the **mirror**, not the master (`:1109-1114` says so), so dropping it also drops a mirror-rule obligation. |
>
> So the real shape is **five numbered plus three un-numbered**, and the first draft's
> citation for the numbered list (`:1023-1050`) landed on the `Args:` block (`:1023` is its
> first line) rather than on the list. A
> rewrite that names only the step under discussion drops the rest silently. That is not
> hypothetical: the column-ordering call was missing from this task until a reader asked
> whether it still existed, and two more were missing until the plan review.

Step 1 calls the existing `authorized_measurement_sources` (`_cli_completion.py:918`) —
already the right predicate, already marker-authorized, and **moved onto records by P3 Step
3b** (the CAN-22 comment at `:955-964` is that move); if that move was skipped it returns
`{}` and this step writes an empty master with no exception.

> ### The legacy arm's fate must be stated, not left to the implementer (B7)
>
> `_aggregate_measurements_unlocked` has two arms (`_cli_output_manager.py:1421-1433`):
>
> ```python
> authorized_sources = authorized_measurement_sources(output_dir)
> if authorized_sources is None:                                   # :1422
>     flush_trailing_measurements_if_chunked(output_dir)            # :1426  -- MANUFACTURES
>     path_to_dataset = measurement_sources_by_path(               #           _dataset_aggregated
>         discover_measurement_sources(output_dir, dataset_names))  #           from chunks/
> else:
>     path_to_dataset = authorized_sources                          # :1433
> ```
>
> **The `_dataset_aggregated.parquet` preference §7.5 forbids exists only on the first arm**
> (`_measurement_sources.py:132-134` documents the preference; the branch that implements it
> is `:161-166`, and `:152-160`'s `_`-prefix filter is what keeps the aggregate out of the
> individual-file list so the two cannot both be read). Two consequences the plan must resolve rather than inherit:
>
> 1. `test_finalize_run_ignores_every_stale_intermediate` plants a poisoned
>    `_dataset_aggregated.parquet` **on a forward tree**, where that arm is unreachable. It
>    is green whether or not the arm survives — the phase's own gate cannot fail for the
>    reason it exists. Hence the second test added above.
> 2. The arm calls `flush_trailing_measurements_if_chunked`, which *writes*
>    `_dataset_aggregated.parquet` from `chunks/`. Carrying it into `finalize_run` unchanged
>    means `finalize_run` both manufactures and consumes the artifact §7.5 names.
>
> ### ⚠ RULED (user, 2026-09-06): KEEP the arm. Narrow §7.5's claim to the authorized path.
>
> Two options were open — drop the arm, or keep it and narrow the invariant. **Keep it.**
>
> The reasoning is the one that decided B2: **the schema gate is disarmed until P7**
> (`_schema_shape.SCHEMA_GATE_ARMED`, flipped in `phase-7-migrate-mode.md:1279`, Task 5
> Step 1d), so a legacy tree can still reach this path for the whole P4→P7 window. Dropping
> the arm is a **behaviour change for real trees during that window**; keeping it is
> reversible and costs one branch.
>
> **§7.5's invariant is therefore narrowed to the authorized path** — amended in `design.md`
> in the same change, not only here. The legacy arm remains free to prefer
> `_dataset_aggregated.parquet`, which is what the second test above pins as *deliberate*
> rather than accidental.
>
> **The arm is a compatibility branch, so it takes the retirement condition** EXECUTION.md
> requires (user, 2026-09-06). Write it at the site, in this form:
>
> ```python
> # LEGACY AGGREGATION ARM -- DELETE WHEN: the schema gate is armed and refuses
> # legacy trees before they reach finalization (P7 Task 5 Step 1d sets
> # _schema_shape.SCHEMA_GATE_ARMED = True). Until then a pre-record tree can
> # still reach finalize_run, and its authority lives in legacy external
> # Parquets under results/<ds>/measurements/ rather than in embedded tables.
> # When that condition holds, this branch is unreachable and goes with the rest
> # of the legacy surface.
> ```
>
> **Tying the trigger to P7 Step 1d is the point**, not decoration: it makes this branch
> retire *with* the legacy surface instead of being orphaned as the one piece nobody dared
> delete. That is the failure mode the retirement-condition rule exists to prevent.
>
> The second test above is required either way, because an invariant whose gate cannot reach
> the violating branch is not gated.

### Step 3 keeps metadata on the left, and deletes the other branch (CAN-1)

The first draft said *"Step 3 onward is `finalize_post_master_outputs`, unchanged."*
**It cannot be.** That function has exactly two branches
(`_cli_output_manager.py:1077-1086`), and after the inversion both lose something:

| Branch | Condition | Does | After the inversion |
|---|---|---|---|
| legacy | `metadata_join_keys is None` (`:1077`) | `join_metadata(working_df, metadata_csv, how="left")` (`:1081-1083`) | **Correct — and it is the only branch that is.** Metadata on the left: joins user metadata onto every matched measured row **and** keeps metadata-unmatched rows as `QC_MetadataOnly` phantoms. Both halves, one call. |
| embedded | keys provided (`:1084-1087`) | `_append_metadata_only_rows(...)` only | **Broken.** Its premise (`:992-999`) is *"Measured rows already carry their publication-time metadata from the embedded tables and are not joined again"* — which **P4 falsifies**. It appends phantoms and joins nothing, so every measured row's user metadata is null. It also raises `ValueError` at `:884-893` for any join key now absent from the master. |

> **The metadata-on-the-left orientation is deliberate and stays.** `join_metadata`'s
> docstring (`_cli_output_manager.py:143-153`) is explicit: *"a left join is asymmetric by
> design: it keeps metadata-unmatched rows but still drops measurement-unmatched rows"*, and
> *"Absence of a colony is data: a strain that failed to grow, or that detection missed, is
> exactly what the user needs to see."*
>
> A measured object whose key appears in no metadata row is an object outside the described
> experiment. Dropping it is the intended semantics, not a data-loss bug — **user ruling,
> round 2.** An earlier draft of this section proposed reversing the orientation so orphan
> measurements survived; that would have changed a deliberate scientific decision on the
> strength of a reviewer's framing. Reverted.

### The surviving branch has never run on a forward tree (flow-r3 C1)

`join_metadata` is the **legacy** branch — reached only when `metadata_join_keys is None`,
which on a modern tree it never is. Deleting the embedded branch promotes a code path that
has not executed on a forward run since embedded tables landed. Three behaviours it brings,
none of them wrong in isolation and all of them changes:

1. **It casts join keys to `String` unconditionally** (`:139-142`, "casts them to ``String``
   for a safe join"). The mirror's join-key dtype changes from whatever the measurements
   carried to `String`.
2. **Row order follows the metadata frame** (its docstring says so), not the master's. The
   mirror's row order changes.
3. **Under a heterogeneous master** — `diagonal_relaxed` concat over stores with differing
   columns — a key present in some stores and absent in others can drop measured rows the
   per-image joins kept.

None of this argues against the user's ruling; metadata stays the left frame. It argues that
**"reuse the existing call" is not the no-op it reads as**, and each behaviour needs a
pinned test rather than a discovery in production:

```python
def test_the_mirrors_join_key_dtype_is_pinned(tmp_path):
    """flow-r3 C1. join_metadata is the legacy branch and has not run on a forward
    tree since embedded tables landed. Promoting it changes dtype and row order --
    both observable by the GUI and by any user script reading the mirror."""
    import polars as pl

    from phenotypic.sdk_ import measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=True, join_key_dtype=pl.Int64)
    _finalize(tmp_path)

    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    assert mirror.schema["Metadata_Well"] == pl.String, (
        "join_metadata casts join keys to String; if this changed, the GUI's "
        "filters and every downstream script keyed on the old dtype changed with it"
    )
    # flow-r4: the name said "and row order" while asserting dtype only. Renamed
    # rather than padded -- behaviour 2 states row order in prose, and a test that
    # claims to pin something it does not is worse than one that claims less. If
    # row order is worth pinning, assert it here and restore the name.


def test_a_heterogeneous_master_loses_no_measured_row(tmp_path):
    """The dangerous third behaviour: a key present in some stores and absent in
    others, concatenated diagonal_relaxed, then joined globally.

    EXPECT THIS TO FAIL under the specified implementation, and do not "fix" it by
    weakening the assertion (flow-r4 C1). Metadata is the left frame (user ruling),
    the join is global and one-shot, and null keys anti-match -- so the ragged
    image's rows are dropped. That is the forcing function this test exists to be:
    the remedy is a DESIGN DECISION about ragged join keys, taken with the user,
    not a fixture bug. `prepare_metadata_join_keys` intersects the two normalized
    frames' columns (`_metadata_join.py:166-171`) with both sides cast to
    `pl.String` (`:187-192`), so any column in both frames -- Grid_RowNum included --
    is a join key, which is what makes the fixture realistic rather than contrived.
    `diagonal_relaxed`'s per-frame null fill makes the loss all-or-nothing per
    image, which is why image-set granularity is the right granularity here.
    """
    import polars as pl

    from phenotypic.sdk_ import master_measurements_parquet_path, measurements_parquet_path

    _publish_image_with_columns(tmp_path, "a.tif", extra=["Grid_RowNum"])
    _publish_image_with_columns(tmp_path, "b.tif", extra=[])      # ragged
    _finalize(tmp_path)

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    measured = mirror.filter(pl.col("QC_MetadataOnly").fill_null(False).not_())
    assert set(measured["Metadata_ImageFile"]) == set(master["Metadata_ImageFile"])
```

### So step 3 is one existing call, not a new composite

The inversion does not need a third mode. It needs the **embedded branch deleted**:

```
step 3  =  join_metadata(master_df, metadata_csv, how="left")   →  post ops
```

That single call already does everything §7.4 step 3 asks for, in the orientation the
project intends:

- it **identifies the common columns itself** (`:139-142`), so it needs no recorded join
  keys — which is also why **CAN-2's `_consistent_embedded_join_keys` retirement falls out
  for free**: nothing downstream needs the stores' recorded keys, so their D-A-manufactured
  inconsistency stops being reachable rather than needing to be tolerated;
- it joins user metadata onto every matched measured row — the half the embedded branch lost
  once P4 made its premise false;
- it emits phantoms with `QC_MetadataOnly` — the half the embedded branch already had.

**Delete the `metadata_join_keys` parameter and its branch** (`:1077-1087`), and with it
`_consistent_embedded_join_keys`' **three** call sites (`:1435-1439`,
`_cli_recompile_worker.py:826`, and `_cli_recompile_slurm_scripts.py:192` at submission
time). The `measurement_sources`-vs-`metadata_join_keys` split in `_run_post_master_steps`
(`_cli_recompile_worker.py:819-828`) goes at the same time — §7.4 already predicted it
would, "because the two callers arrive with differently-shaped inputs", and after this they
arrive the same way.

> ### Retiring the function also deletes the mixed-AUTHORITY refusal (H6)
>
> `_consistent_embedded_join_keys` carries **two independent guards**, and the argument
> above touches only the second:
>
> | Guard | Site | Raises |
> |---|---|---|
> | mixed **authority** | `:933-936` | `ValueError("Cannot aggregate mixed embedded and legacy measurement authority")` |
> | mixed **digests / keys** | `:962-965` | `ValueError("Embedded measurement tables have mixed metadata digests or join keys")` |
>
> D-A manufactures the state that trips the *second*; the join is now global; the recorded
> keys become provenance. **None of that says anything about the first.** Deleting the call
> sites removes the mixed-authority abort with nothing named as replacing it — a tree
> holding both embedded tables and legacy external Parquets would aggregate silently instead
> of refusing.
>
> **Name the replacement or keep the check.** Two candidates already exist and neither has
> been shown to cover the same case: `_cli_recompile_tables.py:303` and
> `_cli_recompile_recovery.py:748` raise *"Legacy external measurement Parquets require
> `--mode migrate`"* on their own paths, and `datasets_needing_migration` only **advises**
> (`sdk_/_run_state.py:1287-1293`). Decide before deleting; if the answer is "keep the
> authority guard, drop only the generation guard", say that — it is a smaller change than
> retiring the function.

Update the docstring at `:992-999`, which states the now-false premise, and
`_cli/CLAUDE.md`'s master-vs-mirror rules.

Test both halves **in one frame** — a measured row carrying a non-null user column, and a
phantom row present.

### Keep the column ordering — `join_metadata` returns metadata-first

`join_metadata`'s own docstring: *"Returns: DataFrame with metadata columns first, then
measurement columns … Row order follows the metadata frame."* What restores the canonical
frame shape is **`order_measurement_columns`** (`sdk_/_metadata_helpers.py:111`), applied at
`_cli_output_manager.py:1104-1106`:

```python
post_df = post_df.select(order_measurement_columns(post_df.columns))
```

**That call is inside the function this task rewrites and must survive.** An earlier draft
of this task specified step 3 without mentioning it at all, which is how a rewrite silently
drops a sibling step: `finalize_post_master_outputs` does five numbered things
(`:990-1017`) plus three un-numbered ones, and enumerating only the one under discussion
is not a rewrite plan.

Canonical order is `[front metadata] → [measurements] → [IMAGE metadata] → [info block]`.
Verified against the real function on the two frames this change produces:

| Frame | Ordered columns |
|---|---|
| master (intrinsic only) | `Metadata_Dataset`, `Metadata_ImageFile`, `Shape_Circularity`, `Object_Label`, `Bbox_X` |
| mirror (joined) | `Metadata_Strain`, `Metadata_ImageFile`, `Shape_Circularity`, `QC_MetadataOnly`, `Object_Label`, `Bbox_X` |

Three things that follow, none of them obvious:

1. **The intrinsic identity columns are front-block, not trailing.** `Metadata_ImageFile`
   has `metadata_owner_for_header(...) is None` and `Metadata_Dataset` is `EXPERIMENT`-owned
   — **neither is `IMAGE`-owned**, so they lead the frame rather than trailing the
   measurements. §7.1's "intrinsic identity stays" therefore leaves the master's shape
   recognisable: identity, measurements, info block.
2. **The master is not ordered by this call**, because it is written before it. It inherits
   its order from the embedded tables, which the pipeline already orders through the same
   function (`_image_pipeline_core.py:1258,1275-1291`). Removing user metadata does not
   disturb the survivors' relative order — unknown-owner tags sort alphabetically at the end
   of the front block, so deleting some leaves the rest in place. **Assert that rather than
   assuming it.**
3. **`QC_MetadataOnly` sorts into the measurements block**, since it is not a metadata
   header, not the object label, and not `Bbox_`/`Grid_`. That is existing behaviour, it is
   **out of scope**, and it is recorded here only so a reviewer seeing it in the ordered
   mirror does not read it as a regression this change introduced.

```python
def test_the_mirror_keeps_canonical_column_order_after_the_join(tmp_path):
    """join_metadata returns metadata-first; order_measurement_columns restores the
    canonical shape. The call lives inside the function this phase rewrites."""
    import polars as pl

    from phenotypic.sdk_ import measurements_parquet_path, order_measurement_columns

    _publish_two_successful_images(tmp_path, metadata=True)
    _finalize(tmp_path)

    cols = pl.read_parquet(measurements_parquet_path(tmp_path)).columns

    # STANDING RULE. `cols == order_measurement_columns(cols)` is a FIXPOINT check,
    # and [] is a fixpoint -- so an absent or empty mirror passes. Its sibling test
    # is saved by pinning cols[:2]; this one has no such anchor, so it needs a guard.
    assert cols, "the mirror has no columns; the fixpoint check is vacuous"
    assert "Metadata_Strain" in cols, "the join did not happen; ordering proves nothing"

    assert cols == order_measurement_columns(cols), (
        "the mirror is not canonically ordered -- the order_measurement_columns "
        "call at _cli_output_manager.py:1106 was dropped in the rewrite"
    )


def test_the_master_inherits_canonical_order_from_the_embedded_tables(tmp_path):
    """The master is written BEFORE the ordering call, so it depends on its inputs
    already being ordered. The inversion removes columns from those inputs; assert
    that does not disturb the rest."""
    import polars as pl

    from phenotypic.sdk_ import master_measurements_parquet_path, order_measurement_columns

    _publish_two_successful_images(tmp_path, metadata=True)
    _finalize(tmp_path)

    cols = pl.read_parquet(master_measurements_parquet_path(tmp_path)).columns
    assert cols == order_measurement_columns(cols)
    assert cols[:2] == ["Metadata_Dataset", "Metadata_ImageFile"], (
        "intrinsic identity should lead the master -- neither column is IMAGE-owned, "
        "so both belong to the front block"
    )
```

**And `pht-metadata.parquet` gets the same treatment** (Task 2): order its columns with the
same function, so a third-party reader joining the two tables sees one convention rather
than two.

### Where the join keys come from (CAN-2)

`_consistent_embedded_join_keys` (`_cli_output_manager.py:914-966`) collects
`(metadata_snapshot_sha256, join_keys)` from every authorized embedded table and raises
`ValueError("Embedded measurement tables have mixed metadata digests or join keys")` at
`:962-965`. It is called unconditionally on the marker-authorized path (`:1435-1439`).

**D-A deliberately manufactures the state that trips it.** Stores keep the snapshot they
were built against, so any run that gains images after a `metadata.csv` edit has two
generations on disk, and the next aggregation aborts — while D-A's contract says divergence
is an advisory and *"an advisory is never a gate"*. This is a gate, in the finalizer, on the
normal rolling-input path.

**Retire it from the finalize path.** The stores' recorded keys become **provenance
only**. That is what the inversion implies: once the join is global, a per-store record of
how *that store* was joined is history, not input.

> **The "derive the keys from `metadata.csv` ∩ master columns" clause is struck (flow-r4
> Mod2).** It is the raw-intersection formulation, and it contradicts the resolved section
> ~100 lines above, which specifies step 3 as one `join_metadata` call that derives its own
> keys. `join_metadata` has no `keys` parameter, so the sentence could not be acted on even
> by someone who tried — but it would be read, and it is exactly the kind of leftover that
> sends an implementer to build a parameter that should not exist.

**The late-metadata case is the dangerous one, because it looks like it works.** A run with
no metadata records `join_status="not_requested"`, digest `""`, keys `()`
(`_embedded_measurement_tables.py:56-63`). Add `metadata.csv` and re-run: the recorded keys
are `()` — which is **not `None`** — so finalize takes the append-phantoms branch with an
empty key tuple and **joins no measured row at all**. Every measured row's user metadata is
null; the phantoms carry the column, so a membership assertion passes.

This **deletes recompile's separate master-merge** and collapses the `measurement_sources`
vs `metadata_join_keys` branch in `_run_post_master_steps`
(`_cli_recompile_worker.py:819-828`), which exists only because the two callers arrive with
differently-shaped inputs. After this task they arrive the same way.

- [ ] **Step 3b: Add the two tests that catch CAN-1 and CAN-2**

```python
def test_the_mirror_carries_both_joined_rows_and_phantoms(tmp_path):
    """CAN-1. Neither existing branch does both halves: one joins and drops every
    phantom, the other appends phantoms and joins nothing. Assert them in ONE
    frame, because each half passes a test that only looks at the other."""
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=True)
    _add_a_metadata_only_row(tmp_path, well="Z99")
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    measured = mirror.filter(pl.col("QC_MetadataOnly").fill_null(False).not_())
    phantoms = mirror.filter(pl.col("QC_MetadataOnly").fill_null(False))

    assert measured.height > 0 and phantoms.height == 1
    assert measured["Metadata_Strain"].null_count() == 0, "measured rows were not joined"
    assert "Z99" in phantoms["Metadata_Well"].to_list(), "phantoms were dropped"


def test_a_measured_row_absent_from_metadata_is_dropped_deliberately(tmp_path):
    """The asymmetry is by design (user ruling, round 2), so PIN it rather than
    leaving it as an accident of which frame is on the left.

    metadata.csv describes the experiment. A measured object whose key appears in
    no metadata row is an object outside that description, and `join_metadata`'s
    docstring states the intent: it keeps metadata-unmatched rows -- "a strain that
    failed to grow, or that detection missed, is exactly what the user needs to
    see" -- and drops measurement-unmatched ones.

    This test exists because an earlier draft proposed reversing the orientation.
    Without it, a future reader sees only "left join" and cannot tell which way
    round was intended.
    """
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=True)
    _add_an_object_whose_key_is_absent_from_metadata(tmp_path, image="b.tif", label=7)
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))

    # STANDING RULE. `orphan.height == 0` is satisfied by an EMPTY mirror, and by a
    # fixture that never added the orphan object. Establish both before reading the
    # zero as evidence of a deliberate drop.
    assert mirror.height > 0, "the mirror is empty; orphan.height == 0 is vacuous"
    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    assert master.filter(
        (pl.col("Metadata_ImageFile") == "b.tif") & (pl.col("Object_Label") == 7)
    ).height == 1, "the fixture never created the orphan; there is nothing to drop"

    orphan = mirror.filter(
        (pl.col("Metadata_ImageFile") == "b.tif") & (pl.col("Object_Label") == 7)
    )
    assert orphan.height == 0, (
        "an object outside the described experiment reached the mirror; the join "
        "orientation was reversed"
    )


def test_the_master_keeps_the_object_the_mirror_drops(tmp_path):
    """Where the dropped object DOES survive, and why that is the right split.

    §7.3: the master is the un-joined archival set -- intrinsic identity, every
    authorized measured row. The mirror is the post-applied, metadata-joined display
    frame. So an object outside the experiment is preserved in the master and absent
    from the mirror, which is exactly the master/mirror distinction CLAUDE.md's
    "feed analysis and dashboards from the mirror, not the master" rule rests on.
    """
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import master_measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=True)
    _add_an_object_whose_key_is_absent_from_metadata(tmp_path, image="b.tif", label=7)
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    kept = master.filter(
        (pl.col("Metadata_ImageFile") == "b.tif") & (pl.col("Object_Label") == 7)
    )
    assert kept.height == 1, "the master must retain every authorized measured row"


def test_metadata_added_after_the_stores_still_joins_every_measured_row(tmp_path):
    """CAN-2, with DF-2's assertion verbatim.

    The `measured.height > 0` guard matters: without it the assertion is vacuously
    true on an all-phantom frame, which is the failure mode the first draft's
    version already had.
    """
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=False)   # keys recorded as ()
    _add_metadata_csv(tmp_path)
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    measured = mirror.filter(pl.col("QC_MetadataOnly").fill_null(False).not_())
    assert measured.height > 0, "fixture produced no measured rows to check"
    assert "Metadata_Strain" in measured.columns
    assert measured["Metadata_Strain"].null_count() == 0, (
        "user metadata reached the mirror only as metadata-only phantoms; every "
        "measured row is null. The join keys were () rather than None, so "
        "finalize took the append-phantoms branch and joined nothing."
    )


def test_stores_with_mixed_metadata_snapshots_do_not_abort_finalization(tmp_path):
    """CAN-2. D-A manufactures this state on the normal rolling-input path; the
    kept code raises on it."""
    _publish_two_successful_images(tmp_path, metadata=True)
    _edit_metadata_csv(tmp_path)
    _publish_one_more_image(tmp_path, metadata=True)    # different snapshot digest

    # STANDING RULE. "Must not raise" is satisfied by a fixture that produced ONE
    # snapshot generation instead of two -- in which case nothing was mixed and the
    # guard this test exists to retire was never even reachable. The mixed state is
    # the whole point: it is the state D-A deliberately manufactures.
    digests = {_snapshot_sha256(store) for store in _stores(tmp_path)}
    assert len(digests) == 2, (
        f"fixture produced {len(digests)} snapshot generation(s), not 2; "
        "the mixed-digest state was never created and this test is vacuous"
    )

    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))
    # must not raise; divergence is an advisory, per D-A
```

- [ ] **Step 4: Run the tests**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_finalize_run.py -q
```

**Expected: `15 collected`, `14 passed`, `1 failed`.** The single failure must be
`test_a_heterogeneous_master_loses_no_measured_row`, which this task says up front it
expects to fail and forbids "fixing" by weakening the assertion — it is a forcing function
for a design decision about ragged join keys, taken with the user. **Any other failure, and
any collected count below 15, is this step failing.**

Fifteen is measured from the plan: 6 tests in Step 1, 4 inline in Step 3, 5 in Step 3b.
Step 6b's test is written after this step and takes the file to 16.

> **A floor is required here, not optional.** An earlier draft gave no command and no total
> — only *"every test in this task passes except X"*. The refusal to quote a fabricated
> number was right (its predecessor said "8 passed" for a set of more than eight), but
> **nothing replaced it**, and "every test passes" is satisfied vacuously by a collection of
> zero. The count above is derived from the plan's own listings, so it is checkable rather
> than asserted.

- [ ] **Step 5: Prove INV-INPUTS can fail**

**Mutate the arm that is actually reachable, and assert against the test that reaches it.**
An earlier draft said *"Add a `_dataset_aggregated.parquet` fast path to step 1"* and then,
two lines later, *"That fast path … **is in the current code**"* — both cannot be true, and
the mutation as written proves only that a *newly added* fast path is caught.

The truth, measured: the preference **is** in the current code
(`_measurement_sources.py:132-134`, `:161-166`), but only on the arm reached when
`authorized_measurement_sources` returns `None` (`_cli_output_manager.py:1421-1431`). So:

1. Run `test_finalize_run_ignores_a_stale_aggregate_on_the_legacy_arm` (Step 1) against the
   **unmodified** `finalize_run`. If option (b) was taken and the legacy arm survives, this
   test *is* the mutation — it should already be red before the fix, and green after. Record
   which. If option (a) was taken and the arm is gone, it passes trivially, and the mutation
   below is the only proof available.
2. Mutate step 1 to prefer `_dataset_aggregated.parquet` **on the authorized arm too**.
   Confirm `test_finalize_run_ignores_every_stale_intermediate` goes red. Remove the
   mutation.

Two mutations, because one test cannot reach both arms.

- [ ] **Step 6: Discriminate the master's schema — do NOT mint a stamp**

> ### ⛔ RULED (user, 2026-09-06): no `master_schema_version` stamp. The whole minting is cut.
>
> An earlier draft of this step minted `phenotypic.master_schema_version = "2"` into the
> master's Parquet KV metadata and built `read_master_measurements` to refuse an unstamped
> file. **Struck in full.** Three reasons, in ascending order of decisiveness:
>
> 1. **It does not exist yet, so minting it manufactures tracked state rather than
>    describing it** — something `read_master_measurements` would *branch on*, which by
>    EXECUTION.md's test 1 is "stop and ask" territory.
> 2. **Its reader story was already broken.** The prescribed sweep,
>    `grep -rn 'master_measurements_parquet_path' src/`, reaches **zero readers**: all four
>    `src/` hits are writers or path construction (`_cli_output_manager.py:1484`,
>    `_cli_recompile_worker.py:780`, `_cli_chunk_writer.py:236`, `_cli_completion.py:1013`).
>    Every actual master **read** goes through `BundleLayout.master_parquet`
>    (`sdk_/_io_constants.py:2681-2683`) — see the reader table below. So the stamp would
>    have shipped with a reader no consumer calls, which is precisely the
>    `DashboardManifestKey.VERSION` pattern P6 deletes *as a finding*.
> 3. **Decisively: the master already self-describes.** A v1 master carries `Metadata_*`
>    user-metadata columns because the join happened per-image; a v2 master does not, because
>    the join moved to finalization. A stamp asserting the schema is a **second home for a
>    fact the file already carries** — rule 3, one level up from the digest question.

**What ships instead.** `sdk_/_master_io.py` still exists, and it exists for exactly one
reason: to be the **single home** of the v1/v2 discrimination and — more importantly — of
its **retirement condition**. Spreading the same two-line column check across seven reader
modules would give the condition seven homes and, in practice, none.

```python
# phenotypic.sdk_._master_io

# V1/V2 MASTER DISCRIMINATION -- DELETE WHEN: no run predating this change is
# still readable, i.e. every master in the wild was written by finalize_run's
# post-inversion path. A v1 master carries Metadata_* user-metadata columns
# because the join happened per-image; a v2 master does not, because the join
# moved to finalization. Nothing else distinguishes them, and nothing stamps
# them. When that condition holds, this function and every branch on it are
# dead code and should go together.
def master_carries_user_metadata(frame: "pl.DataFrame") -> bool:
    """Return whether this master predates the §7.3 inversion."""
```

> **Why the retirement condition is mandatory and not a nicety (user, 2026-09-06).**
> "Compatibility branches outlive the thing they were compatible with, and the knowledge of
> *why* they exist evaporates first. A branch whose retirement condition is written down can
> be deleted by someone who was not there. One whose condition is not written down survives
> forever, because deleting it is an unbounded risk to whoever finds it." The comment must
> say what the check is **for** and **when it may be deleted** — not "handles v1".

**The reader set, measured — this is what the first draft's grep missed (B5).** Nine
`layout.master_parquet` references across **six** modules (the table below has six rows;
earlier prose said seven and disagreed with its own table):

| Site | What it does |
|---|---|
| `gui/results_viewer/_output_root.py:320` | reads the master |
| `gui/results_viewer/_processing_inventory.py:202`, `:373` | adds the path to an inventory (existence/mtime, not a frame read) |
| `gui/results_viewer/_curation_labels.py:417` (`_read_clean_master`), `:763` | reads the **clean** master for curation |
| `gui/results_viewer/_error_tab/_publication.py:125` | reads the master |
| `gui/results_viewer/_qc_tab/review/_data.py:84` | `normalize_viewer_frame(pl.read_parquet(...))` |
| `sdk_/_metadata_migration.py:1036`, `:1065` | **writes** — legacy-metadata canonicalization targets, P7 territory |

Which of these actually need the discrimination? Most do not: curation keys on dataset /
image / object-label, all intrinsic. The ones that matter are readers that *filter or group
on a `Metadata_*` column* — those return empty rather than raising on a v2 master, which
§7.3 calls "the one genuinely dangerous failure mode in §7".

> ### ⚠ RULED (user, 2026-09-06): P4 owns the helper and the `sdk_` readers; P6 owns the GUI readers
>
> | Owner | Scope |
> |---|---|
> | **P4 (this step)** | `master_carries_user_metadata` in `sdk_/_master_io.py`, its retirement condition, and the conversion of the **`sdk_`** readers — `_metadata_migration.py:1036`, `:1065` (which are *writes*, so the question there is which shape they may target). |
> | **P6** | The **six GUI modules**. P6 owns that surface and is already rewriting those readers, so converting them there costs one step instead of a cross-phase edit that P6 would then have to re-touch. |
>
> **This is written into both plans**, because a split recorded in only one half is how the
> second half never happens: `phase-6-consumer-migration.md` gains a step naming the helper
> and its retirement condition. Before this ruling P6 mentioned neither
> `read_master_measurements` nor `master_parquet` at all — only P7 did
> (`phase-7-migrate-mode.md:964, 1064, 1070, 1078, 1134`), and those references are to the
> **cut** stamp, not to this helper.

- [ ] **Step 6b: Settle the known ambiguity by test, not by reasoning**

**A v1 run with no `metadata.csv` also has no `Metadata_*` columns**, so column-presence
conflates it with v2. That is *expected* to be harmless — neither has anything to join — but
it is an inference, and this change has been punished for exactly that twice.

```python
def test_a_v1_metadata_free_master_is_indistinguishable_from_v2_and_that_is_harmless(tmp_path):
    """The ruling's own falsifier (user, 2026-09-06).

    If v1-no-metadata and v2 turn out BEHAVIOURALLY DISTINGUISHABLE -- i.e. some
    reader does something different, and something wrong, on one of them -- then
    the ruling flips: mint the stamp and register it as tracked state properly.
    This test is what settles that, and it is a blocker for the step, not a
    nice-to-have.

    Build both, run every reader in the table above over each, and assert the
    outcomes are equal. Assert on OUTCOMES, not on the column set -- the column
    sets are equal by construction, which is the whole premise.
    """
    v1 = _write_legacy_master_from_a_metadata_free_run(tmp_path / "v1")
    v2 = _finalize_a_metadata_free_run(tmp_path / "v2")
    outcomes_v1, outcomes_v2 = _reader_outcomes(v1), _reader_outcomes(v2)

    # STANDING RULE, and the highest stakes in the plan: this test is the
    # designated falsifier for a HARD-STOP ruling (no schema stamp). An outcome
    # collector that swallows exceptions returns {} for both, the equality holds,
    # and a false green here silently confirms "no stamp needed" -- the exact
    # question the ruling said to settle by test rather than by reasoning.
    assert outcomes_v1, "no reader outcomes collected for v1; the equality is vacuous"
    assert set(outcomes_v1) == set(outcomes_v2), "the two runs exercised different readers"
    assert len(outcomes_v1) == _EXPECTED_READER_COUNT, (
        "a reader was added or dropped without updating this falsifier"
    )

    assert outcomes_v1 == outcomes_v2
```

**If this test cannot be made to pass, stop and report it** — the answer is the stamp plus a
register row, and that raises the state-artifact count, which is a HARD STOP.

- [ ] **Step 7: Publish `source_set_digest` in the run proof (U-4)**

`publication_id` is cut. It was an opaque `uuid4().hex` binding the run proof to the
aggregate proof — replaced by comparing the two values directly. `finalize_run` step 6
publishes `source_set_digest` and `source_image_count` into **both** proofs, and P1's rule 1
compares them (CAN-4).

> ### ⛔ CORRECTED: the premise was false, and P1 already specifies the work (B8)
>
> An earlier draft said *"`source_set_digest` had no home in any phase before this step — it
> appeared only in the README's digest table and two prose mentions in P5."* **Both fields
> ship today**, written into the *aggregate* proof at `_cli_completion.py:1045-1046`:
>
> ```python
> "source_set_digest": canonical_digest(sorted(source_work_ids)),
> "source_image_count": len(source_work_ids),
> ```
>
> and read at `_cli_completion.py:786-788` (`current_aggregate_is_current`) and
> `sdk_/_run_state.py:1213-1216` (rule 1, clause 2). This step does not *introduce* them; it
> **moves them into the run proof** and cuts the hash that stood in for them.

**The work, spelled out.** P1 shipped the compatibility half already, and its docstring names
P4 as the writer (`sdk_/_run_state.py:1219-1241`):

> U-4 cuts `publication_id` and puts `source_set_digest` in the **run** proof … **That
> writer change lands in P4**; until it does, today's run proof carries neither field and the
> values live in the aggregate proof, **bound to the run proof by `publication_id`**. Both
> shapes are read here so that P1 lands on today's trees and keeps working across P4's writer
> bump, with no window in which the two comparisons silently stop being made.

1. **Add both fields to `publish_run_completion_evidence`** — the payload dict at
   `_cli_completion.py:1139-1166` — as a **COPY of the aggregate proof's values**, exactly
   as `publication_id` is copied at `:1151-1153` today. **Not recomputed.** See the block
   below; an earlier draft said "computed the same way the aggregate proof computes them",
   which is a different design.
2. **Cut `publication_id` from BOTH writers in ONE commit**: minted at `:1032` (aggregate),
   copied at `:1151-1153` (run). Cutting it from the aggregate proof alone would make
   `_source_set_binding`'s fallback compare `"abc" != None` and return `None` for every
   pre-P4 tree → `current_run_is_complete` false → **complete runs read as incomplete**.
3. **Replace it in `stable_keys`** (`_cli_completion.py:1173-1180`), the idempotence check
   that decides whether the run proof is rewritten. With `publication_id` gone the entry
   compares `None == None` and contributes nothing; `source_set_digest` is the value that
   should be there instead.
4. **Keep `_source_set_binding`'s fallback arm** (`sdk_/_run_state.py:1236-1241`) — pre-P4
   trees still carry the old shape and are read by post-P4 code. **It is a compatibility
   branch, so it takes the retirement condition** (user ruling, 2026-09-06): *delete when no
   run proof predating this change is still readable, i.e. every run proof in the wild
   carries `source_set_digest` directly.* P4 is editing P1's code here — say so in the
   commit rather than letting a reader discover it in the diff.
5. **Give the THREE remaining `publication_id` comparisons a disposition each (NEW-6).** See
   the block below. Item 3 diagnoses this degeneration correctly for `stable_keys` and then
   the first draft missed every other instance of it.

**Note the ordering property that makes this safe**: `_source_set_binding` checks
`if "source_set_digest" in proof` **first** (`:1234`), so a post-P4 run proof never reaches
the fallback, and a pre-P4 tree's aggregate proof still carries the `publication_id` it is
matched against. There is no window where both halves are half-migrated — provided (2) is
one commit.

> ### ⚠ COPY, not recompute — and this is the whole point of U-4 (NEW-7)
>
> An earlier draft of item 1 said the run proof's `source_set_digest` *"must be **computed**
> the same way the aggregate proof computes them … or the two proofs disagree on identical
> trees."* **Struck.** The two readings are not equivalent, and the trailing clause gives the
> draft away: recomputation agrees on an **unchanged** tree and diverges on a changed one —
> which is the only case that matters.
>
> | Reading | What the run proof asserts | Failure it misses |
> |---|---|---|
> | **Copy** (correct) | *"I was published against **that** aggregate."* Rule 1 then checks that assertion against **live** verification. | none — a success set that changed between the two publications is caught. |
> | **Recompute** (wrong) | *"Here is my own view, at my own moment."* | A **stale aggregate proof beside a fresh run proof passes both checks independently**, and nothing notices they disagree. The binding is gone. |
>
> **The copy IS the binding.** P1's shipped docstring says so in the sentence this step
> already quotes (`sdk_/_run_state.py:1222-1226`): U-4 states the aggregate↔run binding
> *"directly instead of through an opaque hash"* — `publication_id` restated in the clear.
> A hash restated in the clear is a copy; it cannot be a recomputation, or it would not be
> restating anything.
>
> **This also sharpens the budget row's ruling rather than contradicting it.** The two proof
> copies are **never compared to each other**. Each is compared against an independently
> derived **live** value — the aggregate's at `_cli_completion.py:786-788`, the run's at
> `sdk_/_run_state.py:1205-1215` against the verified image set. That is precisely what makes
> the pair a **cross-check** rather than a mirror, and it is why *"copy from the aggregate"*
> and *"not a second home"* are both true at once: two copies of one value, each checked
> against a different live derivation, is two checks — not one fact stored twice.

> ### ⛔ Cutting `publication_id` turns live checks into TAUTOLOGIES, not errors (NEW-6)
>
> `grep -rn publication_id src/` finds consumers beyond the two writers. **Neither of the
> comparisons below raises when the field goes — both silently stop checking.** That is the
> same degeneration item 3 identifies for `stable_keys`, so the pattern was found once and
> then missed everywhere else. **A comparison that cannot fail is worse than a deleted one,
> because it still reads as a guard.**
>
> | Site | After the cut | Disposition |
> |---|---|---|
> | `valid_run_completion` — `expected["publication_id"] = aggregate.get("publication_id")` (`_cli_completion.py:1228`), compared at `:1231-1232` | `None != None` → **False** → the entry contributes nothing; **the aggregate↔run binding stops being checked** | **Replace with `source_set_digest` + `source_image_count`.** This is the live one: **5 call sites in 4 modules** — `phenotypicCLI.py:2505`, `:2513`, `gui/run_console/_slurm_observer.py:1319`, `_cli_migrate.py:1268`, `gui/results_viewer/_output_consistency.py:381`. |
> | `valid_run_completion`'s process-only arm — `expected["publication_id"] = None` (`:1220`) | `None != None` → **False**; it was already asserting the field is absent | **Delete the line.** It asserted "no aggregate binding on a process-only run"; with no field to be absent it says nothing. Deleting is honest, keeping is a comment that looks like a check. |
> | `run_proof_is_current` — `proof.get("publication_id") == aggregate.get("publication_id")` (`sdk_/_run_state.py:1081-1085`) | `None == None` → **True unconditionally**; only the `finalization_input_digest` half survives | **Replace with `source_set_digest`.** Exported (`sdk_/_run_state.py:118`) with **no in-repo caller**, so this is public API surface: nothing in the tree will fail if it is left a tautology. |
>
> **A third degeneration in the same function, which the sweep did not name:**
> `run_proof_is_current`'s *process-only* arm (`sdk_/_run_state.py:1075-1077`) opens
> `proof.get("publication_id") is None and …`. After the cut that is `None is None` → **True
> unconditionally**, so that arm also loses half its test. Same disposition as the
> `valid_run_completion` process-only line: delete the clause, do not leave it.
>
> **One consumer needs no action, and the reason is ordering, not harmlessness.**
> `gui/results_viewer/_output_consistency.py:60`, `:442-443` carry an
> `aggregate_publication_id` field that would become permanently `None`. **P6 Task 2 deletes
> that entire module** (`phase-6-consumer-migration.md:580`, 617 lines), so the field goes
> with it. Record the dependency: if P6 Task 2 is descoped or deferred, this field becomes a
> live dead value and comes back onto P4's list.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/_cli/_cli_finalize_run.py src/phenotypic/sdk_/_master_io.py \
        tests/unit/cli/test_finalize_run.py
git commit -m "feat(cli): finalize_run -- one aggregation and publication path

Spec §7.4, §7.5, six steps (D-A cut the backfill). Step 3 is join_metadata(how="left") --
the one call that already does both halves in the orientation the project intends.
The embedded branch is deleted: P4 falsified its premise that measured rows already
carry their metadata (CAN-1). join_metadata identifies its own common columns, so
the stores' recorded join keys -- which D-A deliberately makes inconsistent -- stop
being read at all rather than needing to be tolerated (CAN-2).

NO master schema stamp is minted (user ruling): the master self-describes -- a v1 master
carries Metadata_* columns, a v2 does not -- so a stamp would be a second home for a fact
the file already carries. sdk_/_master_io.py is the one home of the v1/v2 discrimination
and of its retirement condition. source_set_digest and source_image_count move into the
run proof, replacing the cut publication_id, in the same commit that cuts it from the
aggregate proof; P1's _source_set_binding fallback stays for pre-P4 trees and now carries
its retirement condition (U-4).

INV-INPUTS was confirmed to fail on both arms: the authorized arm under an injected
_dataset_aggregated fast path, and the legacy arm under its existing one."
```

---

## Task 4: Route all three entry points through `finalize_run`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:1351`, `:1545`
- Modify: `src/phenotypic/_cli/_cli_recompile_worker.py:771-782`, `:805`
- Test: `tests/unit/cli/test_finalize_run.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("mode", ["full", "measure", "recompile"])
def test_every_mode_produces_a_byte_identical_master(tmp_path, mode):
    """§7.4: recompile becomes 'call finalize_run again', not a parallel
    implementation that must be kept in sync. Three modes, one master."""
    output = _run_mode(tmp_path, mode)
    reference = _run_mode(tmp_path / "ref", "full")

    # STANDING RULE. This is the phase's HEADLINE CLAIM, and the bare equality is
    # satisfied by two FAILED runs if _master_bytes returns b"" for a missing file.
    # Establish that both runs produced a master with rows in it, then compare.
    for out in (output, reference):
        master = master_measurements_parquet_path(out)
        assert master.is_file(), f"{out} produced no master; the comparison is vacuous"
        assert pl.read_parquet(master).height > 0, f"{out}'s master is empty"

    assert _master_bytes(output) == _master_bytes(reference)


def test_process_mode_skips_finalization_entirely(tmp_path):
    """§7.4's table: `process` writes one layer, no measurement, and
    process_only_layer already short-circuits the aggregate proof."""
    output = _run_mode(tmp_path, "process")

    # STANDING RULE. The negative below is satisfied by a `process` run that ERRORED
    # early and wrote nothing at all -- which is a different fact from "process ran
    # and correctly skipped finalization". Establish that it did its own work first.
    assert list(output.rglob("*.ome.zarr")), (
        "process produced no store; the absence of a master proves nothing"
    )

    assert not (output / "deliverables" / "master_measurements.parquet").exists()
```

- [ ] **Step 2: Run to verify failure**

Run the same command as the step that follows. **Expected: every test in the file red, and
the failure text must name the symbol under test as missing.** A collection ERROR from a
different cause — a fixture typo, a bad import elsewhere in the file — is also red, and is
not evidence this step passed. Read the reason, not the colour.

- [ ] **Step 3: Implement**

`_aggregate_measurements_unlocked` keeps its lock (`aggregate_measurements`'s
`.aggregate_publication.lock`, `_cli_output_manager.py:1559-1562`) and delegates its body.
`_run_post_master_steps` becomes a `finalize_run` call, keeping its
`generation_publication_guard` wrapper (`_cli_recompile_worker.py:841-859` — the guard is
an `if slurm_generation is None` / `else` pair around two otherwise identical calls, and
collapsing it is not part of this step).

**Thread `include_dataset_column` through (H4)** — it is a live parameter of the body being
delegated, and it crosses a serialization boundary in the recompile SLURM path. The full
site table is in the Interfaces section above; the two that a signature change would break
without a type error are `_cli_recompile_slurm_scripts.py:149,200` (written into the task
dict) and `_cli_recompile_worker.py:368` (read back out of it).

**Two callers return different things, and the deletion in Step 4 inverts one of them.**
`_aggregate_measurements_unlocked` returns `master_csv_path` (`:1542`) and
`_run_post_master_steps` returns the post-applied frame. Whatever `finalize_run` returns,
both call sites and everything downstream of them must be updated together — see Step 4.

- [ ] **Step 4: Delete the D8 surfaces — the MEASURED dependency set**

An earlier draft named four symbols. **The measured set at `869e9dee` is ten modules, two
of them in the GUI, and one semantic inversion the draft never mentioned (B6).** Re-derive
before editing:

```bash
grep -rn "MASTER_MEASUREMENTS_CSV\|master_measurements_csv_path\|load_master_measurements\|master_csv" src/
```

**Definitions to delete** — `sdk_/_io_constants.py:317` (`MASTER_MEASUREMENTS_CSV`),
`:1271-1273` (`master_measurements_csv_path`), `:2290-2302` (`load_master_measurements`,
which reads the CSV), `:2686-2688` (`BundleLayout.master_csv`), plus the module docstring
mentions at `:27`, `:47`, the `MASTER_MEASUREMENTS_CSV` in `_reserved_analysis_artifact_stems()`
at `:353`, and the `sdk_/__init__.py` export lines — **six lines: three imports** (`:91`
`MASTER_MEASUREMENTS_CSV`, `:184` `load_master_measurements`, `:187`
`master_measurements_csv_path`) **and three `__all__` entries** (`:472`, `:599`, `:602`).
An earlier draft said "four export lines … five lines: two imports", which was wrong twice
over and missed `:184`; the step's own prescribed `grep` surfaces all six.

**Dependents the draft did not name, each of which breaks on those deletions:**

| Site | Why it breaks |
|---|---|
| `gui/_config.py:67`, `:137` | imports and re-exports `MASTER_MEASUREMENTS_CSV` in `__all__` — an **import-time break** in the GUI |
| `gui/_schema_cache.py:26`, `:43` | `_FILES_BY_SOURCE["master_measurements"] = (MASTER_MEASUREMENTS_PARQUET, MASTER_MEASUREMENTS_CSV)` — the GUI's no-pyarrow CSV fallback for the master. Deleting the file removes the fallback; decide whether that is acceptable or whether the entry becomes parquet-only. |
| `sdk_/_metadata_migration.py:1036`, `:1065` | iterates `(layout.master_parquet, layout.master_csv)` as **migration targets** — P7's code, edited by P4 |
| `_cli_chunk_writer.py:47`, `:232` | writes the master CSV mid-run |
| `_cli_recompile_worker.py:39`, `:771-782` | writes it, and **re-raises** on failure at `:778` |
| `_cli_completion.py:28`, `:1012` | the `master_csv` entry in `required_paths`, which becomes the proof's `required_outputs` |
| `_cli_output_manager.py:62`, `:1483`, `:1486-1500`, `:1518`, `:1542` | the write, the success signal, the log line, the return value |

> #### The semantic change the draft never states: the CSV write **is** aggregation's success signal
>
> ```
> :1493-1500   master_csv_saved = _guarded_terminal_best_effort(..., write_master_csv, default=False)
>              if not master_csv_saved: return None
> :1508-1512   _guarded_terminal_best_effort(..., write_master_parquet,
>                  warning="Failed to save master Parquet (CSV was saved)")
> :1542        return master_csv_path
> ```
>
> **The CSV is required and the Parquet is best-effort — exactly backwards after D8.**
> Deleting the CSV requires inverting that: the Parquet write becomes the gate, and its
> failure must return `None`. Nothing in the draft said so, and the failure direction of
> getting it wrong is a run that reports success having written no master at all.
>
> **The return value propagates**, and every hop needs updating in the same commit:
> `aggregate_measurements` (`:1545`) → `OutputManager.aggregate_master_csv` (`:2004`, whose
> docstring at `:2027` promises *"Path to master_measurements.csv"*) →
> `phenotypicCLI.py:2982` (`master_path = output_manager.aggregate_master_csv(...)`). Two
> tests bind the method **by name**: `tests/integration/cli/test_staged_gpu_local.py:520,529`
> monkeypatches `OutputManager.aggregate_master_csv`, and
> `tests/unit/cli/test_cli_v2.py:2069-2086` calls it. Renaming it is optional; updating both
> is not.

**The aggregate proof's `required_outputs` drops from four descriptors to three**
(`_cli_completion.py:1011-1016`, assembled at `:1047`). `valid_aggregate_snapshot` and
`sdk_/_run_state.py` validate whatever the proof *lists*, so a three-entry proof validates
correctly on its own terms.

> ### ⚠ RULED (user, 2026-09-06): BUMP `AGGREGATE_PROOF_VERSION`
>
> **The cost was checked, not guessed.** A version mismatch at `sdk_/_run_state.py:1098` and
> `_cli_completion.py:1064` invalidates the proof, and an invalid aggregate proof forces
> **re-aggregation — not reprocessing.** Aggregation rebuilds the master from embedded
> tables that already exist on disk; no image is re-measured and no store is rewritten. The
> bump is cheap in exactly the way that matters.
>
> **And the decisive half: P4 changes the master's shape.** Every existing aggregate proof
> is therefore stale *in substance* whether or not the version moves. Not bumping leaves a
> pre-P4 proof silently certifying a post-P4 expectation — the two are indistinguishable,
> which is the same defect the master's own schema stamp was cut for creating a second home
> of. **Loud-and-cheap beats silent-and-wrong**, which is this change's own in-phase
> criterion.

Per [Q6](OPEN-QUESTIONS.md#q6-ten-test-files-depend-on-master_measurements_csv_path), ten
test files reference `master_measurements_csv_path` — **verified exactly ten** at
`869e9dee`: `tests/_output_layout.py`, `tests/unit/sdk_/test_io_constants.py`, and
`tests/unit/cli/{test_cli_chunk_writer,test_cli_completion,test_cli_migrate_authority,test_cli_output_manager,test_cli_recompile,test_cli_recompile_slurm,test_cli_v2,test_recompile_manifest_completion}.py`.
Fix each: assert on the parquet, or on `measurements.csv` where the test genuinely wanted a
human-readable file. `BundleLayout.detect` keys on `master_measurements.parquet` already
(`sdk_/_io_constants.py:2624`, `:2632`), so bundle detection is unaffected.

- [ ] **Step 4b: Delete the legacy-marker xfail tripwires**

They are `strict=True`, so they turn **red** the moment Task 4's repoint lands and stay red
until they are removed — and left in place while the repoint is *pending*, they sit at
`XFAIL`, a passing state, recording the debt as paid.

- `_RECOMPILE_READS_THE_LEGACY_MARKER_UNTIL_P4`: 2 definitions
  (`tests/unit/cli/test_cli_recompile.py:75`, `tests/unit/cli/test_cli_recompile_slurm.py:51`)
  + **28 decorations** (3 and 27 total occurrences, minus the definitions).
- `tests/unit/cli/test_embedded_measurement_recompile.py:31-40` — a separate inline
  `pytest.mark.xfail` with the same reason, carrying stale line numbers.

**Budget for the instances whose outcome is unknown** — see "Delete the xfail tripwires" in
the recompile section: the two `Cannot restore marker authority` arms
(`_cli_recompile_slurm_scripts.py:430`, `:493`) and the never-executed assertion at
`tests/unit/cli/test_cli_recompile_slurm.py:2905`.

- [ ] **Step 5: Phase gate**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix src/phenotypic/_cli/_cli_finalize_run.py \
  src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_recompile_worker.py \
  src/phenotypic/_cli/_embedded_measurement_tables.py src/phenotypic/sdk_/ tests/unit/
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit tests/integration -q
```

**Expected: compare the collected/failed counts against the recorded baseline, and report
the delta — not "PASS".** A full-suite run has pre-existing failures; a bare pass/fail
verdict on it is meaningless, and a *drop* in the collected count is the signal that
something stopped being collected. The `run-phenotypic-test` skill carries the baseline and
the sharding recipe.

**`tests/integration` is in the gate, not just `tests/unit` (M4).** Task 5's test is written
there, and `testpaths` (`pyproject.toml:220`) covers it only for a *bare* `pytest` — naming
`tests/unit` on the command line excludes it. Two of Step 4's own dependents also live
there (`tests/integration/cli/test_staged_gpu_local.py:520,529`).

This is the first phase where the full suite is warranted rather than a
selection — the master's shape changed and it is read almost everywhere. **The suite is
~65 minutes and is a Slurm job**: use the **`run-phenotypic-test`** and **`slurm-job`**
skills, with the committed script at
`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`. Never
`-n auto` (it reads node cores, not the allocation) and never `-x` (it truncates a run that
then gets recorded as a baseline).

- [ ] **Step 6: Update the docs the contract change invalidates**

- `CLAUDE.md`'s "Output layout (`deliverables/`)" bullet: `master_measurements.*` is now
  `master_measurements.parquet`, un-joined and intrinsic-only.
- `src/phenotypic/_cli/CLAUDE.md`'s master-vs-mirror rules.
- `docs/source/how_to/pages/` wherever the master is described as metadata-joined.

- [ ] **Step 7: Commit**

```bash
git add -A src/phenotypic tests docs CLAUDE.md
git commit -m "refactor(cli): route full, measure and recompile through finalize_run

Spec §7.3, §7.4, D8. Deletes recompile's separate master-merge and the
measurement_sources/metadata_join_keys branch that existed only because the two
callers arrived with differently-shaped inputs. Master is parquet-only and carries
intrinsic identity only; the mirror carries the join."
```

---

## Task 5: Verify the promote-time metadata end to end

**Files:**
- Test: `tests/integration/cli/test_promote_time_metadata.py` *(new)* — a real single-image run

- [ ] **Step 1: Run a real local run with `--metadata` and assert the store is self-describing**

```python
def test_a_real_run_leaves_stores_a_third_party_can_join(tmp_path):
    """D-A's whole justification: the store is self-describing WITHOUT any post-hoc
    rewrite. Read it back with plain pyarrow -- no phenotypic import in the assertion
    path -- and join it, the way a napari or QuPath user would."""
    import pyarrow.parquet as pq

    output = _run_full_pipeline(tmp_path, metadata=True)
    store = next(output.rglob("*.ome.zarr"))

    measurements = pq.read_table(store / "tables" / "measurements" / "table.parquet")
    metadata = pq.read_table(store / "tables" / "metadata" / "pht-metadata.parquet")
    keys = json.loads(metadata.schema.metadata[b"phenotypic.join.keys"])
    joined = measurements.to_pandas().merge(metadata.to_pandas(), on=keys, how="left")
    assert "Metadata_Strain" in joined.columns
```

- [ ] **Step 2: Run it**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/integration/cli/test_promote_time_metadata.py -q
```

**Expected: `1 passed`.** `_run_full_pipeline` is this test's own helper and appears nowhere else in
the plan — write it here, driving the real CLI entry point rather than calling `finalize_run`
directly, since the point of this task is that the **shipped** path leaves a joinable store.

- [ ] **Step 3: Commit**

```bash
git add tests/integration
git commit -m "test(cli): a promoted store is joinable by a third party with pyarrow alone

D-A. The assertion path imports no phenotypic code, which is the property that makes
'self-describing' mean something."
```
