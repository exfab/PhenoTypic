# Concern ledger

Append-only entries; statuses updated in place. A `resolved` entry naming what changed
IS the provenance lock — a later reviewer challenging it must flag `CONFLICT with <ID>`,
not raise it fresh.

---

## Settled before round 1 (user, 2026-09-03)

### D-A [settled-by-user (round 0)]
- Per-store metadata is written at promote time, not backfilled into already-proven stores.
- Cuts spec §6.3, §6.4's *generalisation*, `stages.backfilled`, the backfill half of §8,
  spike S-1, residual risk §15.4. Adds INV-IMMUTABLE.
- May not be re-argued. A consequence the plan failed to follow through on IS reportable.

### D-B [settled-by-user (round 0)]
- The verification cache is in-process (audit S1's actual proposal), not
  `.phenotypic/verification_cache.json`. Spike S-5 decides whether an on-disk tier ships.

### D-C [settled-by-user (round 0)]
- `scientific_config_digest` IS `processing_configuration_digest`, verbatim. No existing
  `work_id` changes. Spec §5.4's field list is corrected in P2.

---

## Round 1

### MIG-1 [Critical] [open]
- Raised: round 1, migration specialist
- Concern: P7's conversion table misses a **fourth tree shape that exists in the wild**. A
  pre-3.0.0 tree (`success_markers_required` absent or false) has no `image_complete/` tree
  at all; its completion evidence lives **only** in `state.datasets.<ds>.completed`. Today
  `_migrate_legacy_success_evidence` (`phenotypicCLI.py:560`, called at `:2376-2378`, gated
  by `_requires_legacy_success_migration` at `:544`) converts it in place during `--mode
  full` resume, reading completion at `:625-631` and setting `success_markers_required =
  True` / `version = "3.0.0"` at `:658-659`. Under P7 the gate refuses that tree before the
  converter can run; P7 Task 2 converts nothing (all three marker trees are empty) and P7
  Task 3 deletes `datasets.{completed,failed,started}` — the only record. Migrate reports
  success, the tree ends with zero records, the next `--mode full` reprocesses every image
  from source, and the evidence is gone. Precisely the failure
  `datasets_needing_migration`'s docstring is a post-mortem of (`_io_constants.py:1530-1533`).
  P6 ledger row 10 defers "`_legacy_*` helpers" to P7 with the correct reasoning, but P7
  never enumerates this one — the deferral has no destination.
- **Independently verified by the orchestrator** at `phenotypicCLI.py:544-560, 583, 640-660,
  2376-2378`. Live, not dead code: reachable with no flag or opt-in, echoed to the user at
  `:2380-2383`, exercised by `tests/unit/cli/test_cli_state_management.py:316`,
  `tests/unit/cli/test_cli_completion_store.py:606`,
  `tests/unit/cli/test_embedded_measurement_migration.py:312`, with a maintained pre-markers
  fixture at `tests/unit/sdk_/_migration_fixtures.py:440-447`.
- Also noted by the orchestrator: P7 Task 3's `test_work_ids_are_untouched` passes
  **vacuously** on this shape — a pre-3.0.0 tree has no `work_ids` to preserve. A check that
  can only return the answer you want.
- Resolution: —

### MIG-2 [Critical] [open]
- Raised: round 1, migration specialist
- Concern: P7 Task 4 stamps `phenotypic.master_schema_version = "2"` during migrate while
  explicitly not re-running finalization, so the file stamped is still the legacy
  metadata-joined master. It does not acquire the v2 intrinsic-only shape until the next
  `finalize_run`. Task 4's own two tests contradict each other: one asserts the v2 shape only
  after `finalize_run`, the other asserts the stamp after migrate alone. Between those points
  the stamp certifies a shape the file does not have — strictly worse than no stamp.
- Specialist's placement answer: the fix belongs in **P4 Task 3**, minting the stamp where
  `finalize_run` writes the master, so stamp and shape come from one code path.
- Resolution: —

### MIG-3 [Major] [open] `spec-change` — GATED TO USER
- Raised: round 1, migration specialist
- Concern: §7.3 calls the master's user-metadata columns disappearing "the one genuinely
  dangerous failure mode in §7" and says the schema stamp is "why the migrate step must tag
  the master with a schema version so an old reader fails loudly". **The mechanism cannot do
  that.** `grep -rn master_schema_version` finds one occurrence outside the snapshot — the
  assertion itself — and zero readers in `src/`. An *old* reader predates the key and cannot
  check it; `pd.read_parquet` / `pl.read_parquet` / `pq.read_table().to_pandas()` ignore
  Parquet KV metadata and raise nothing. The stamp is write-only and the failure mode is
  unmitigated for exactly the readers §7.3 worries about: user notebooks and exported
  analysis scripts.
- Distinct from MIG-2 (writer-side ordering); fixing MIG-2 leaves this fully open.
- Resolution: —

### MIG-4 [Major] [open]
- Raised: round 1, migration specialist
- Concern: P7 Task 4's claim that "a pre-inversion store aggregates to the same master a
  post-inversion one does" is false for two legacy store classes. **(a) Duplicate-key
  fan-out** — `prepare_embedded_measurement_table`'s right join deliberately preserves
  fan-out (`_embedded_measurement_tables.py:88-95`, warned at `:81-86`), so projecting a
  joined table onto its recorded `measurement_columns` yields the same measurement row
  repeated k times, whereas a P4-era store carries no fan-out at all (it moved to the mirror
  join). P4 Task 1 pins fan-out preservation as a test, so this is behaviour, not
  hypothesis. **(b) Join-key dtype drift** — `_restore_join_key_dtypes` (`:22-39`) logs a
  warning and leaves a column as the string-safe type on failure (`:31-38`), so a legacy
  store can carry a join key as `str` where the baseline was `int64`; concat then raises or
  silently upcasts. P7's only test on this path checks column **names**, which passes under
  both defects.
- Resolution: —

### MIG-5 [Major] [open]
- Raised: round 1, migration specialist
- Concern: **Phase ordering.** P3 Task 2 is an explicit clean break ("D1 is a clean break: no
  dual write"), so from P3 onward `publish_image_success` writes the new record and
  `valid_image_success` reads it — while the refusal gate (`requires_conversion`, P7 Task 1)
  does not land until four phases later. Between P3 and P7, on a legacy tree
  `valid_image_success` returns `False` for every image,
  `authorized_measurement_sources` yields an empty mapping, and `--mode full` silently
  reprocesses everything and rebuilds the master from nothing. P6 applies the correct
  principle to deletions (row 10) but not to the clean break that creates the same hazard.
  Not merely a bisect concern — P4 Step 5, P5 and P6 all specify runs against real trees.
- Orchestrator note: **the plan asserts the refusal already exists.** P3 Task 2 says "every
  other mode refuses it" while nothing implements that until P7 Task 1.
- Suggested direction: move P7 Task 1 to the front of P3 (or to P1). It has no dependency on
  the rest of P7.
- Resolution: —

### MIG-6 [Major] [open]
- Raised: round 1, migration specialist
- Concern: **No rollback, and the plan never says so.** §15.1 requires "the receipt/rollback
  discipline the existing metadata migration has, plus its own dry-run mode". P7 Task 5
  delivers the dry run and *resumability* — a different property: resumability guarantees a
  re-run finishes, not that the previous state is recoverable. Task 2 removes the three
  legacy trees outright, no copy, no receipt, where the existing metadata migration leaves
  receipts (`_cli_completion.py:340-350`). After a successful migrate, a user who reverts the
  code — the ordinary first response to a regression this size — has a tree the old build
  reads as entirely unprocessed. Nothing in the plan, CLI help, or the `CLAUDE.md` bullet
  says migrate is one-way.
- Suggested direction: **rename** the legacy trees to `.phenotypic/legacy-v2/` instead of
  deleting — same filesystem, a rename, no byte copied — and have `requires_conversion`
  ignore that path. Gives `migrate --revert` almost free. Otherwise make the one-way nature
  an explicit confirmation and document it.
- Resolution: —

### MIG-7 [Major] [open]
- Raised: round 1, migration specialist
- Concern: **The coexistence window is unaddressed** — nothing in the spec or plan mentions
  it. A SLURM array launched from the old build holds the old schema for its whole lifetime
  (up to 30 d on `batch`/`intel`/`epyc`), and P2's `restart_epoch` fence cannot reach it: an
  old-build worker never calls the new `publish_image_success`, it writes `image_complete/`,
  `stage2_done/` and `stage3_complete/` directly. So (1) a tree migrated while such an array
  is live re-acquires the old shape and is then refused by every writing mode including the
  array's own dependent finalizer; and (2) P7 Task 2's overwrite semantics for an existing
  record are unspecified, so a re-run migrate can let a stale legacy marker clobber a newer
  record's `stages`. `test_conversion_is_idempotent` does not cover this — after the first
  conversion the legacy trees are gone, so the both-shapes-present case is never exercised.
- Resolution: —

### MIG-8 [Minor] [open]
- Raised: round 1, migration specialist
- Concern: three migrate behaviours tabulated or asserted with no implementing step —
  (1) `master_measurements.csv` → deleted is assigned to P7 Task 4, which has no step, test
  or commit line for it; (2) P7 Task 3 asserts `deliverables/metadata.canonical.csv` exists
  after migrate, but no task builds it and the conversion table omits it; (3)
  `authorized_measurement_sources`' legacy branch (`_cli_completion.py:783-816`) is P6's
  deferred row 10, P7 never touches it, and P4 Task 3 builds `finalize_run` step 1 directly
  on it without saying what a `None` return means once D1's clean break is in force.
- Resolution: —

### MIG-9 [Minor] [open]
- Raised: round 1, migration specialist
- Concern: three shapes with no stated behaviour — a **bundle-only tree** (`deliverables/`
  with no `.phenotypic/`, explicitly supported by `BundleLayout.detect`,
  `_io_constants.py:2468-2482`) trips none of `requires_conversion`'s four signals and is
  treated as converted; a **`--mode process` tree** classifies correctly but has no master at
  all, and P7 Task 4's stamping step has no behaviour when the master is absent; a **store
  with no measurement descriptor** is documented as "a normal state, not a fault"
  (`_measurement_tables.py:340-346`) but `embedded_measurement_columns` (`:382`) raises
  `KeyError`, so P7's projection has a reachable `KeyError` with no handling.
- Resolution: — (merged into CAN-32)

---

## User rulings, round 1 (2026-09-03) — permanent, not re-raisable

### U-1 [settled-by-user] Migration floor is v0.17.3
Verified: v0.17.3 (2026-06-18) predates the marker schema (`379acee4`, 2026-08-17) and
OME-Zarr. Its `create_initial_state` writes `version="2.0.0"` with no
`success_markers_required`. **The floor IS the pre-markers shape.** Below it: refuse
explicitly. Settles the scope half of MIG-1 / SIMP-R1-14.

### U-2 [settled-by-user] §4.3 keeps BOTH clauses
`complete` = every accepted image has a valid proof **and** a valid run proof covers the
current inventory. The run-proof-subsumes-it argument rested on INV-IMMUTABLE, which
CAN-3 proved false. Completion stays O(N) in per-image proofs; the verification cache is
therefore load-bearing rather than marginal. **P1 Task 5 Step 3's "two things and no more"
is wrong and must be corrected.** Settles SIMP-R1-01 and the reader half of CAN-4.

### U-3 [settled-by-user] §7.3 — name a reader, keep the stamp
Add `read_master_measurements()` in `sdk_` raising on an unstamped or wrong-versioned
master; route every in-repo master read through it; correct §7.3 to claim only what it
delivers. Settles MIG-3 / SIMP-R1-05.

### U-4 [settled-by-user, verification-conditioned] `publication_id` is cut
User asked for experimental validation. **Verified:** nine sites; exactly one branches on
it (`_cli_completion.py:1101`). The GUI stores `aggregate_publication_id` and branches on
it nowhere — zero consumers in `src/` or `tests/`; P6 deletes that file. Today's
`uuid4().hex` is not redundant (a uuid separates executions), but §5.1 redefines it as
`sha256(source_set_digest ‖ finalization_inputs)` — a pure function of exactly the two
values the binding check compares — so once content-derived it is provably redundant.
**Cut it; the run proof carries `source_set_digest`; headline becomes 14 → 5.**
`source_set_digest` still needs a home (P4). Settles GEN-B4 / SIMP-R1-06 / DF-10.

### U-5 [settled-by-user, verification-conditioned] `RunDiagnostics`'s demoted trio is dropped
User asked to double-check for consumers. **Verified: zero after P6.** Every manifest-count
reader is in `_output_consistency.py` (deleted, P6 Task 2) or `_slurm_observer.py:1321`'s
`_manifest_is_complete` (deleted, row 5). `processing_event_log_present` is read only at
`_output_consistency.py:189`. The `total_images` hits at `_cli_types.py:246,269-271` are a
different dataclass (CLI progress), untouched. Settles SIMP-R1-09.

---

## Round 1 — canonical entries (deduped across four reviewers)

### CAN-1 [Critical] [open] `finalize_post_master_outputs` cannot be reused unchanged
- Aliases: GEN-B1, DF-1
- Both branches break on a metadata-free master (`_cli_output_manager.py:1077-1086`):
  `metadata_join_keys is None` → `join_metadata(..., how="left")`, joins but **drops every
  metadata-only phantom**; keys provided → `_append_metadata_only_rows` only, which after
  P4 raises `ValueError` at `:884-893` for any join key now absent from the master,
  deliberately re-raised at `:1092-1095`. §7.4 step 3 needs join **and** phantoms; no branch
  does both.
- Fix (DF-1's wording): step 3 becomes a composite — left-join metadata onto the master on
  resolved keys, anti-join-append phantoms with `QC_MetadataOnly = true`, then post ops.
  Test both halves in one frame. Update the `:1023-1026` docstring and `_cli/CLAUDE.md`.

### CAN-2 [Critical] [open] D-A's advisory is a hard `ValueError` in kept code
- Aliases: GEN-B2, DF-2
- `_consistent_embedded_join_keys` (`:914-966`) raises on mixed `(digest, keys)` across
  stores, called unconditionally at `:1435-1439`. D-A deliberately manufactures that state.
  **Late-metadata looks like it works:** a no-metadata run records digest `""`, keys `()`;
  adding metadata yields `()` which is **not `None`**, so finalize takes the append-phantoms
  branch with an empty key tuple and joins no measured row. P5's test asserts only column
  membership — **it passes on broken data.**
- DF-2 supplied the replacement assertion verbatim (null-count on measured rows, with a
  `measured.height > 0` guard). Use as written.
- Fix (DF-2 option a): retire `_consistent_embedded_join_keys` from finalize; derive join
  keys once from `metadata.csv` ∩ master columns; recorded keys become provenance only.

### CAN-3 [Critical] [open] INV-IMMUTABLE is false — `--mode measure` already re-promotes proven stores
- Aliases: GEN-B3, DF-27
- `_cli_process_single.py:439` → `replace_image_store_measurements` (`:1970-2001`) →
  `replace_embedded_measurement_table` (`sdk_/_measurement_tables.py:242`), whose
  descriptor-change path is a root-last transaction using
  `_clone_file_without_pixel_rewrite` (`:233`) = `os.link` with a `shutil.copy2` fallback.
  **That IS spec §6.3's hardlink re-promote, already shipping.**
- Consequences: README's "one pre-existing exception" is wrong; P4 Task 2's property test
  exercises only `finalize_run`, so it would ship a green invariant over a live violation;
  and `replace_image_store_measurements` calls `prepare_embedded_measurement_table` with the
  current snapshot (`:1993-1996`), so measure mode would write joined tables and no
  `pht-metadata.parquet` into an inverted tree. That function appears nowhere in the plan.
- **Sharpened (DF-27, verified by the orchestrator at `_measurement_tables.py:284-290`):
  there are TWO post-proof mutation paths, and the in-place one is worse.** When
  `current == descriptor`, the function calls `_write_validated_parquet(payload, table, …)`
  and returns — writing the measurement table straight into the **promoted** store with no
  `.part`, no `copytree`, and **no root `zarr.json` rewrite**. Two consequences: the
  per-image proof's store digest still matches while the table underneath changed, so the
  proof stays "valid" over mutated content; and `snapshot_sha256` lives in the root, so
  **D-A's divergence advisory reads a value this branch never refreshes** — it would report
  stale metadata as current. The advisory mechanism this plan invented is defeated by code
  already shipping.
- Fix must therefore cover all three: rewrite INV-IMMUTABLE to what is true (flow-r1's
  recommendation — do not delete `--mode measure`); bring
  `replace_image_store_measurements` into P4's inversion so it uses `prepare_image_tables`
  and refreshes the metadata table and root together; and re-anchor or repair D-A's
  advisory so the in-place branch cannot silently satisfy it.
- **Orchestrator note:** this corrects a premise given to the user for D-A — I told them
  D-A removed "the only mechanism that mutates artifacts already carrying a content proof",
  and two already ship. Reported; user did not revisit. D-A stands on its remaining grounds
  (receipt protocol, `stages.backfilled`, backfill fan-out, new partial state,
  metadata-edit-re-promotes-every-store). **Separately reported and not a request to
  reverse D-A:** D-A's stated basis for cutting spike S-1 — that no hardlink re-promote
  would exist — is false, so that cost is now unmeasured for code already running.

### CAN-4 [Critical] [open] The verdict ladder's rule 1 drops four of five comparisons
- Alias: DF-4; reader half settled by U-2
- `current_aggregate_is_current` (`:738-745`) compares five things against current config;
  plan rule 1 keeps only `inventory_digest`. Dropping `finalization_input_digest` breaks
  §7.4's late-metadata guarantee, which is real today **only** because of that comparison.
  Dropping `source_set_digest`/`source_image_count` removes the only check that the master
  covers the succeeded set — which is what makes CAN-5 reachable. All five are literal
  config fields, so keeping them costs nothing under INV-LAYER.
- Fix: rule 1 = the full five-way comparison **plus** U-2's per-image clause; one
  verdict-matrix row per comparison.

### CAN-5 [Critical] [open] A partial shard set is undetectable and gets certified
- Alias: DF-5
- No step has `TASK_FINALIZE` verify it received K shards, and it is `afterany`, so it runs
  when a shard dies. `publish_aggregate_snapshot` derives `source_set_digest` and
  `source_image_count` from `_current_success_work_ids` — **marker-derived, not
  merge-derived** (`:904-916`) — so a master missing four shards gets a proof asserting the
  full success set. With CAN-4's reduced rule 1 the run then reads `complete`.
- `test_a_prior_epochs_shards_are_never_merged` calls `finalize_run` with **no
  `shard_paths`** — the local concat path, which never reads a shard directory.

### CAN-6 [Critical] [open] The record's read-modify-write has no lost-update protection
- Alias: DF-6
- P3 Task 1 Step 3 cites a precedent that does not exist: `publish_image_success` passes
  `pre_replace` only when `expected_artifact_descriptors is not None` (`:243-249`) and the
  callback re-validates artifact descriptors (`:204-224`), never re-reading the marker.
  `atomic_write_json` (`sdk_/_atomic_io.py:209-240`) is temp-write + `os.replace`, no CAS.
- `publish_image_record` has no declared merge semantics; today's publisher writes a
  complete dict, so after the collapse it clobbers `stages.stage1`/`stage2`. Stage 3
  survives only by call ordering (`_cli_staged_slurm_worker.py:487-514`); the local paths
  need the same check.

### CAN-7 [Critical] [open] Migrate is a second, unrevised producer of everything P2/P3 change
- Aliases: MIG-1 (scope half settled by U-1), DF-22, DF-23
- **Framing correction (DF-23), superseding the orchestrator's first reading:** the two
  conversions are **not** chained producer→consumer. The HDF migrator is *itself* a producer
  of the marker schema — `publish_image_success` at `_cli_migrate.py:1413` and
  `_cli_migrate_image.py:567` — so after P3 it emits records and `convert_per_image_markers`
  finds nothing. They are **alternative producers of one shape, and only one has been
  revised.**
- `_cli_migrate.py:684-705` writes a fresh state whose `processing_generation` is
  `sha256("migration\n" + inventory_payload)` (`:686-688`) — **the inventory folded into the
  generation, exactly what D7 forbids**, already in the tree. It writes no `restart_epoch`
  while writing `work_ids`, which is precisely P7 Task 1's detection signal 4, so a freshly
  HDF-migrated tree is refused by the very next `--mode full`.
- Verified by the orchestrator: `_configured_work_id` (`_cli_migrate_image.py:125`) falls
  back to `_migration_work_id` = `sha256("migration:<ds>/<stem>")` when state has no
  `work_ids`, and `_existing_marker_identity` supplies defaults when no marker exists. So
  the migrator already mints identity and publishes records for a pre-markers tree —
  **`_migrate_legacy_success_evidence` is deleted, not folded in**, and `datasets.completed`
  is genuinely unnecessary.
- Fix: add `_cli_migrate.py:660-705` to P2 Task 3's file list; bring it to the v3 schema;
  build at least one P7 fixture through the **real** HDF migrator, not hand-planted.

### CAN-8 [Critical] [open] Spec §9's CLI depth rows are unmapped; `_cli_completion.py` is never split
- Alias: GEN-B5
- Still calling the old O(N)-hashing readers after P7: `phenotypicCLI.py:2390,2394,
  2423-2442,2872-2874,3721-3725,3735-3744`; `_cli_staged_resume.py:203-213`;
  `sdk_/_hdf_to_zarr.py:715`; `_dashboard/_manifest_builder.py:725`; `_cli_migrate.py:88-89`.
  The double walk is never removed, §11's last row is not delivered, and two completion
  predicates ship permanently — the drift hazard cited when deleting `_latest_event_states`.

### CAN-9 [Critical] [open] The master schema stamp certifies a shape the file lacks
- Alias: MIG-2; direction settled by U-3
- P7 Task 4 stamps during migrate while explicitly not re-running finalization, so the
  stamped file is still the legacy joined master; Task 4's own two tests contradict each
  other. **Fix: mint the stamp in P4 Task 3 where `finalize_run` writes the master**;
  migrate leaves the legacy master unstamped, correctly marking it pre-v2.

### CAN-10 [Critical] [open] Legacy projection does not undo fan-out; dtypes can drift
- Aliases: MIG-4, DF-3 (fan-out), DF-21 (dtype)
- The right join preserves duplicate-key fan-out by design (`:81-93`), so a legacy store
  with k metadata rows per key holds each measurement row k times; projection preserves it
  and CAN-1's global join fans it out again → k². `_restore_join_key_dtypes` (`:22-39`)
  leaves a key as the string-safe type on failure, so legacy and fresh stores disagree at
  concat. P7's only test checks column **names**.
- Fix: row-collapse with a stated, proved-safe dedup key, or accept rewriting legacy
  embedded tables and re-open against D-A with the user. Add a row-count assertion: migrated
  legacy and equivalent fresh trees produce byte-identical masters **and** mirrors.

### CAN-11 [Major] [open] The clean break lands four phases before the gate that refuses
- Alias: MIG-5
- P3 Task 2 asserts "every other mode refuses it" but `requires_conversion` is not built
  until P7 Task 1. P4 Step 5, P5 and P6 all specify runs against real trees.
- **Sharpened (GEN-G02, round 1 follow-up): the outcome is worse than a silent reprocess.**
  After P3, `authorized_measurement_sources`'s legacy branch still globs `image_complete/`
  (`_cli_completion.py:786`) and calls `valid_image_success`, which now reads `images/`.
  Every image fails validation, the mapping comes back **empty**, and P4's `finalize_run`
  writes an **empty master with no exception raised** — a successful-looking run that
  silently discards every measurement. Merges with CAN-22.
- Fix: move P7 Task 1 to the front of P3 (or P1). No dependency on P4–P6.

### CAN-12 [Major] [open] No rollback, and the plan never says migrate is one-way
- Alias: MIG-6
- Fix: **rename** the legacy trees to `.phenotypic/legacy-v2/` — same filesystem, no byte
  copied — and have `requires_conversion` ignore that path. Gives `migrate --revert` nearly
  free. Otherwise make one-way explicit and documented.

### CAN-13 [Major] [open] The coexistence window is unaddressed
- Alias: MIG-7
- A SLURM array from the old build holds the old schema for up to 30 d and writes the three
  legacy trees directly, so a tree migrated mid-array re-acquires the old shape and is then
  refused by every writing mode including its own finalizer. P7 Task 2's overwrite semantics
  are unspecified and `test_conversion_is_idempotent` cannot catch it.
- Fix: state the drain/`scancel` rule in the doc and the refusal message; make conversion
  **merge-not-overwrite**; test both shapes present. Interacts with CAN-12.

### CAN-14 [Major] [open] The shallow path still costs O(N) JSON reads
- Alias: GEN-M1. `CachedVerification` cannot serve `RunState.images`, which needs `stages`
  per image. P1 Task 6's perf test counts only `sha256`, so it passes with every JSON read
  in place. Fix: cache `ImageState`, or restate §9.2 honestly.

### CAN-15 [Major] [open] The stale-worker gate tests a mechanism that does not exist
- Alias: GEN-M2. `publish_image_success` raises only when `SLURM_JOB_ID` is set and
  `slurm_lifecycle.json` exists, comparing the **lifecycle** generation (`:181-188`). P2's
  test writes a marker and does not raise, and conflates two tokens. §14's actual
  requirement is the `generation=` fence in `aggregate_state_from_events`, untouched.

### CAN-16 [Major] [open] The 16-cell equivalence table gates ≤16 of ≥192 reachable cells
- Aliases: GEN-M3, DF-11. Real signature is keyword-only `(output_dir, dataset, image,
  input_root, process_only_layer, markers_required, expected_work_id)` (`:197-206`), with
  four store predicates behind `expected_work_id is None` (`:227-232`), `markers_required`
  (`:258-265`), and `process_only_layer` ∈ {None, "objmap", other}. For "the risky task",
  the gate does not gate.

### CAN-17 [Major] [open] P6's `core_readable` and mutation-gate replacements are non-equivalent
- Aliases: GEN-M6, DF-7, DF-8, SIMP-R1-13(2). `core_readable` = `not
  marker_authority_required or aggregate_marker_valid` (`:109-114`) — a legacy tree is
  readable with no proof, and an **active** output with a valid proof IS readable. The
  proposed predicate fails both, and it gates the live-run `skipif`, where a false `False`
  **skips** tests invisibly. Separately `completion != "active"` makes every `incomplete`
  output GUI-mutable where `is_read_only` (`:93-96`) forbids it today.

### CAN-18 [Major] [open] P6 Task 1 discards snapshot currency
- Aliases: GEN-M7, DF-9. `completion` cannot answer "does the bound in-memory snapshot still
  match disk". A re-finalize over an unchanged inventory rewrites `measurements.parquet`
  while `completion` stays `complete`, so the badge reads "Current" over a stale snapshot.
  Task 3 then deletes both fingerprints with no replacement.

### CAN-19 [Major] [open] In-array `TASK_FINALIZE` vs the existing dependent finalizer
- Aliases: GEN-M8, DF-12. `_cli_slurm_array_scripts.py:51-52` and `_cli/CLAUDE.md` make the
  dependent finalizer "the sole publisher of aggregated outputs and the completion marker".
  P5 never says whether it is removed, kept, or no-ops. Two publishers is the failure mode
  this change exists to remove.

### CAN-20 [Major] [open] `--mode measure` and `--mode process` sit outside P2's identity plumbing
- Aliases: GEN-M9, DF-16. `phenotypicCLI.py:2640` skips state creation in measure mode;
  `:2716` sets `processing_generation = uuid4().hex`. P2 requires one mint for
  `create_initial_state` and every resume path; measure is neither, yet §7.4 routes it
  through `finalize_run` and P4 Task 4 parametrizes on it.

### CAN-21 [Major] [open] `mint_run_identity` has no idempotency rule
- Alias: GEN-M10. Minting bumps and persists `restart_epoch`; nothing says it happens once
  per invocation, and `ExecutionConfig.output_dir` is `Optional[Path]` (`_cli_types.py:99`)
  so `RunIdentity` has no root to read the epoch from. Two calls burn an epoch.

### CAN-22 [Major] [open] `authorized_measurement_sources` reads the legacy tree, and is not in P3's list
- Aliases: GEN-M5, DF-24, MIG-8(3). `:786` globs `DIR_IMAGE_COMPLETE`; `:838` reads
  `image_completion_marker_path`. If missed it returns `{}` after P3 — a *valid* schema-3
  "nothing succeeded yet" — and P4 step 1 silently produces an empty master. Same for
  `refresh_success_markers_after_metadata_migration` (`:305`), with no stated ordering
  against the record conversion.

### CAN-23 [Major] [open] `valid_image_success` must reject a stage-2-only record
- Alias: GEN-M4. After the collapse a Stage-2 worker writes a record with `stages.stage2`
  and no artifacts; a missing check turns a stage-2 token into a success proof. Today the
  two facts live in two trees, so the confusion is impossible.

### CAN-24 [Major] [open] DEFERRED D-2 is a correctness requirement of the ladder
- Alias: SIMP-R1-07. §4.1 makes `gui_launch_owner.json` a liveness authority and Q2 rule 2
  reads it; audit S7 [verified] shows nothing repairs it, so a SIGKILLed GUI pins
  `status: "running"` forever. **Rule 2 is unsound as written**, and P6's `completion !=
  "active"` gate makes it worse than today. P1 Task 5's matrix has no stale-owner row.
- Fix: restate as "Q2 rule 2 requires it"; add the P1 matrix row; keep the repair in P6.

### CAN-25 [Moderate] [open] Cut spike S-4 — a set-theoretic identity
- Alias: SIMP-R1-03. Local `M ⋉ K_i` vs global `(M ⋉ K_all) ⋉ K_i` with `K_i ⊆ K_all` are
  equal for all `M`, `K`; all four variants perturb both sides identically and both are
  sorted. `S-4 PASS` is guaranteed and the FAIL branch unreachable. Move its one real
  question into P4 Task 1.

### CAN-26 [Moderate] [open] Drop S-5 as a gate on P1
- Alias: SIMP-R1-04. An on-disk tier is a cache — degrades to deep, deleted by
  `clear_machine_state`, never authoritative — so it can be added later at no penalty. The
  spike also hand-rolls the marker loop instead of calling `valid_image_success`. **With
  CAN-25 this takes P0 off P1's critical path**: P0 reduces to S-2 and S-3, gating P5 only.

### CAN-27 [Moderate] [open] Cut O-2 / `KNOWN_STAGES` — unbuildable without breaking INV-LAYER
- Alias: SIMP-R1-02. `KNOWN_STAGES` lives in `_cli`; the advisory is emitted by `sdk_`,
  which may not import `_cli`. Resolving means duplicating the frozenset or violating the
  invariant P1 Task 1 spends a task pinning. Reader side is unspecified too.
- Fix: cut. If the typo class matters, close it with shared module constants rather than
  reporting it.

### CAN-28 [Moderate] [open] Replace the cache LRU with per-output wholesale replacement
- Alias: SIMP-R1-08. Entries are already identity-fenced, so a map of `output_dir →
  (identity_digest, dict[work_id, …])` replaced wholesale on identity change is inherently
  bounded — tighter than 200k — and needs no eviction policy. Removes a magic constant, two
  tests and a mutation-proof row.

### CAN-29 [Moderate] [open] Hoist `_canonical_digest` in P1 rather than adding a third copy
- Aliases: SIMP-R1-10, DF-19 (the third copy must use `ensure_ascii=False`). Hoisting to
  `sdk_` in P1 Task 4 is less total work than add-three-then-collapse. A pure-function hoist
  moves no consumer and cannot change a verdict.

### CAN-30 [Moderate] [open] Merge INV-DEGRADE into INV-CACHE's property
- Alias: SIMP-R1-12. Both are "nothing may improve a verdict except a successful deep
  verification" — over one input vs every other. INV-DEGRADE is the only named invariant
  with no gate in the phase table.

### CAN-31 [Minor] [open] Five tests assert on text rather than behaviour
- Alias: SIMP-R1-11. Delete `test_run_state_exports_no_writer`,
  `test_no_module_still_imports_the_deleted_classifier`, `test_the_decision_tree_is_untouched`.
  Give advisories a closed set of codes if they are worth asserting on. **Keep the INV-LAYER
  AST test exactly as written.**

### CAN-32 [Minor] [open] Tabulated behaviours with no implementing step
- Aliases: MIG-8, MIG-9, DF-25. CSV deletion assigned to a task with no step;
  `metadata.canonical.csv` asserted but never built; bundle-only tree trips no signal;
  `--mode process` tree has no master and the stamp step has no behaviour for that; a store
  with no measurement descriptor is "a normal state, not a fault"
  (`_measurement_tables.py:340-346`) yet `embedded_measurement_columns` (`:382`) raises
  `KeyError`; v0.17.3 detection rests on one signal; an empty inventory must not read
  `complete`.

### CAN-33 [Minor] [open] GEN minors m1–m9
- m1 `tests/_ngff_conformance.py` defines **zero** tests; P4 Task 2 Step 4 collects nothing
  (real suites: `tests/unit/test_ngff_schema_fixtures.py`, `tests/unit/test_ome_zarr_invariants.py`).
  m2 INV-IMMUTABLE and INV-DEGRADE have no prove-it-can-fail step. m3 P4 Step 5's framing
  overstates the fast path. m4/DF-18 `aggregate_measurements` returns `master_csv_path`,
  deleted by D8. m5 `write_master(..., csv=True)` writes the CSV without naming the helper,
  so Q6's grep undercounts. m6 the brief's "authoritative config list" claim is wrong —
  `work_ids`/`metadata_sha256` are added at `phenotypicCLI.py:656-657,2695-2705`.
  m7/DF-20 `load_processing_state` calls `migrate_legacy_machine_state` on every read
  (`:106`) — a write side effect P6 removes the last GUI trigger of. m8
  `_assert_worker_generation`'s docstring says the values are supplied independently, and P2
  pre-commits to deleting the check before Step 1 confirms the finding. m9
  `refresh_success_markers_after_metadata_migration` hand-joins paths (`:340-353`).

### CAN-34 [Minor] [open] Remaining DF items
- DF-13 checkpoint aggregation is a second trigger, invalidation under-specified. DF-14
  `scientific_config_digest` names two different values. DF-17 `publish_image_record` drops
  two parameters real callers pass. DF-26 the HDF migrator synthesizes two of the three
  finalization inputs.

### CONFLICT-1 [open] `restart_epoch`'s two homes
- DF-15 files it as a defect (two homes for one fact). SIMP-R1 explicitly considered and
  **declined** to raise it: the `processing_state.json:config` copy is *the epoch this state
  was minted under*, compared against the current epoch by `assert_identity_current` — a
  fence, not a cache.
- Precedence: tier 3 (correctness) does not decide; tier 6 (simplicity) is SIMP's lane and
  SIMP is the one declining. **Resolve in round 2** with evidence: does any code path read
  the config copy for anything other than the fence comparison?

