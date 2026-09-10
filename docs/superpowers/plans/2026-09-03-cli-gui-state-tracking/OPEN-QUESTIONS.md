# Decisions and open questions

Found while grounding the plan in the real tree at `c9d1fbfc`. Each carries the evidence,
the decision, and who took it.

**D-A, D-B and D-C were taken by the user on 2026-09-03** and are binding on the phase
docs. They are recorded here rather than folded silently into the plan because each one
*departs from the spec as written*, and a reader comparing the two needs to see why.

| | Question | Decision | Taken by |
|---|---|---|---|
| **D-A** | Per-store metadata: backfill into proven stores, or write at promote time? | **Write at promote time.** §6.3, §6.4's generalisation, `stages.backfilled`, the backfill fan-out and residual risk 4 are all **cut**. | user |
| **D-B** | Verification cache on disk, or in process? | **In process first.** A spike (S-5) at implementation time decides whether the on-disk variant is needed at all; prefer in-process if it works. | user |
| **D-C** | Q1 — `scientific_config_digest`'s field list | **Keep `work_id` unchanged.** §5.4's prose is the wrong half. | user |
| Q2 | Verdict precedence unspecified | Plan decides: `complete` > `active` > `failed` > `incomplete` | plan |
| Q3 | `ImageState` used but never defined | Plan defines it (P1 Task 2) | plan |
| Q4 | `sdk_` cannot import `_cli` | Plan resolves: plain-JSON state read, INV-LAYER test | plan |
| Q6 | Ten test files depend on `master_measurements_csv_path` | Sizing note for P4 | plan |

### Round-1 rulings (2026-09-03) — permanent, not re-raisable

| | Question | Decision | Taken by |
|---|---|---|---|
| **U-1** | How far back must `--mode migrate` reach? | **v0.17.3.** Verified to predate the marker schema (`379acee4`, 2026-08-17) and OME-Zarr, writing `version="2.0.0"` with no `success_markers_required` — so **the floor IS the pre-markers shape**. **Detection amended by U-6.** | user |
| **U-6** | U-1 assumed `state.version` could detect the floor. It cannot — `"2.0.0"` is the value at v0.17.3 *and* immediately before `"3.0.0"` was introduced, and it is a state-schema version, not a package version. | **Key on the pre-markers shape instead: schema `2.0.0` with no `work_ids` key.** No sub-floor, no `BELOW_FLOOR` verdict. Older trees are the same shape and cost nothing extra; root-level machine state is already converted by `migrate_legacy_machine_state`, which ships today, so refusing it would remove support rather than bound scope. | user (after the orchestrator's premise was disproved) |
| **U-8** ⚠ | Can migrate leave the unrecoverable digest fields blank, so the remaining values still generate a matching hash? | **WITHDRAWN round 4 — my answer was wrong.** The question was sound; the mechanism I described does not exist. `work_id_for_image` **recomputes** from a live `ExecutionConfig` and never reads `state.config`; `processing_configuration_digest_from_values` writes all twelve fields as required kwargs; and a second producer `_worker_work_identity` cross-checks with a hard `RuntimeError`. Two fields are irreproducible regardless — `input_sha256` hashes the original image and is never persisted, `pipeline_fingerprint` hashes a file the tree holds only a re-serialization of. | orchestrator (self-disproved), reported to user |
| **U-10** | Given U-8 is impossible, how does a migrated tree pass `valid_image_success` without a fabricated identity? | **Mark the record, do not fabricate the identity.** Migrate writes `provenance: "migrated"`; `valid_image_success` accepts such a record on artifact validity alone, with no `work_id` comparison; `resolve_run_state` advises that the configuration fence is unavailable. Removes no guarantee (v0.17.3 has zero `work_ids`), is self-limiting (reprocessing clears the marking), and is visible. Discharges MIG-21 and MIG-22 together. | user |
| **U-2** | §4.3's `complete` — one clause or two? | **Two.** The run-proof-subsumes-it argument rested on INV-IMMUTABLE, which CAN-3 proved false. Completion stays O(N) in per-image proofs, which is what makes the cache load-bearing. | user |
| **U-3** | §7.3's master schema stamp | **Keep it, and name a reader.** `read_master_measurements()` raises on an unstamped or wrong-versioned master; §7.3 corrected to claim only in-repo readers. | user |
| **U-4** | Is `publication_id` redundant? | **Yes — cut it.** Validated: nine sites, one branching consumer, zero for the GUI copy. Today's uuid is not redundant; §5.1's content-derived redefinition makes it a pure function of two values the binding check already compares. Run proof carries `source_set_digest`. Headline **14 → 5**. | user (after verification) |
| **U-5** | Does `RunDiagnostics`'s demoted trio have consumers? | **No — drop it.** Every manifest-count reader is deleted by P6. | user (after verification) |

**O-2 is CUT** (CAN-27): `KNOWN_STAGES` could not be built without either breaking
INV-LAYER — the advisory is emitted from `sdk_`, which may not import `_cli` — or
duplicating the frozenset. Shared `STAGE_*` constants **close** the typo class instead of
reporting it, which is less code and is derivation over tracking.

**O-1 is NOT taken at all** (P2 Task 4, user-ruled). An earlier version of this line said
it was "partially taken", narrowed to the pair the audit found to be one value passed
twice. **The collapse is achievable zero times** — see §5.1's amendment
(`design.md:323-345`) for the writer-by-writer table, and O-1 below for the corrected
disposition. What Task 4 shipped is a *deletion*: the pair was not two tokens to collapse
but one value compared with itself.

**Q2's ladder is amended by U-2 and CAN-4:** rule 1 now carries both of §4.3's clauses
**and** the full five-way comparison `current_aggregate_is_current` makes today, and rule 2
requires the liveness authority to be *itself live* (CAN-24).

---

## D-A. Per-store metadata is written at promote time, not backfilled

**The spec's organising principle is "move state that is tracked to state that is
checked". §6.3/§6.4 move in the opposite direction**, and they are the largest source of
new machinery in the design:

- a hardlink re-promote protocol, itself spike-gated on S-1 (§6.3)
- a **new artifact class** — `.phenotypic/rewrites/<kind>-<digest>.json` certified
  receipts (§6.4)
- a `stages.backfilled` entry carrying its own `metadata_sha256`
- the backfill half of the SLURM fan-out (§8)
- a **new partial state**, which the spec itself records as residual risk 4:
  `deliverables/` complete and correct while `results/` is not yet self-describing
- a metadata edit re-promoting every store in the tree (§7.4)

All of it exists so that an already-promoted, already-proven store can gain
`tables/metadata/pht-metadata.parquet`. **The spec never names the consumer.** The GUI
reads the mirror; analysis reads the mirror; the master is intrinsic-identity-only by
design (§7.3). The benefit is a store that is self-describing to a third party — real, but
bought with the only mechanism in the change that mutates artifacts already carrying a
content proof.

### What is built instead

`pht-metadata.parquet` is written **during the store's original promote**, in the same
`.part` as `tables/measurements/table.parquet`, before the root `zarr.json`. The store's
existing marker-last publication protocol is untouched, and **no post-proof store mutation
exists at all** — §6.2's immutability constraint is *preserved* rather than renamed.

`deliverables/metadata.csv` is copied byte-for-byte before any per-image work starts, so
within one invocation the metadata a store is built against is already fixed.

### What is given up, and how it stays honest

A store built before `--metadata` was supplied (or against an older `metadata.csv`) keeps
the metadata it was built with. `metadata_sha256` does not participate in `work_id`
(correctly — it is a finalization input, §5.4), so a metadata edit does **not** invalidate
per-image proofs and does **not** rebuild stores. §7.4's guarantee therefore narrows:

> **Was:** a `metadata.csv` edit re-runs `finalize_run`, re-joining the mirror **and
> re-backfilling every store**, without touching a single image's measurement.
>
> **Becomes:** a `metadata.csv` edit re-runs `finalize_run`, re-joining the mirror.
> Stores keep the metadata snapshot they were built against; each store's
> `attributes.phenotypic.metadata_table.snapshot_sha256` records which one.

That divergence must be **derived and surfaced, never tracked**: `resolve_run_state` adds
one advisory when any store's recorded `snapshot_sha256` differs from the current
`metadata_sha256`. It is a `stat` + one attribute read per store on the deep path, it
reuses a value the store already carries, and — per §4.3 — **an advisory is never a gate**.
The reader is `_store_metadata_snapshot` (`sdk_/_run_state.py:641-667`, shipped in **P1
Task 5**); the writer is **P4 Task 2**.

> **⚠ CORRECTED, twice.** This entry originally spelled the root key
> `phenotypic.metadata.snapshot_sha256` and pointed at *"P4 Task 6"*.
>
> - **The key is `metadata_table`, not `metadata`.** `phenotypic.metadata` is already taken
>   by the `{protected, public, imported}` image-metadata sections
>   (`sdk_/ngff_.py:569-580`). P1 shipped the correct spelling and its reasoning at
>   `sdk_/_run_state.py:364-378`.
> - **P4 has five tasks**, so "P4 Task 6" named nothing — and the work it was reaching for
>   (the reader) had already shipped in P1. The *writer* is P4 Task 2. A pointer to a name
>   that does not exist is drift-register Entry 32's shape.
>
> Note also that the digest's **only** home is this root key. After P4's inversion the
> measurements Parquet carries no join, so its `snapshot_sha256` is `""` by that file's own
> contract; the digest is not mirrored (user ruling, 2026-09-06 — see `EXECUTION.md`).

### Consequences, phase by phase

| Spec section | Fate |
|---|---|
| §6.3 hardlink re-promote | **Cut.** Spike S-1 is no longer a gate and is dropped from P0. |
| §6.4 certified post-hoc rewrite protocol | **Generalisation cut.** The *existing* `refresh_success_markers_after_metadata_migration` (`_cli_completion.py:305`) stays exactly as it is, serving the one historical case it was written for; P7 keeps it for `--mode migrate`'s metadata-schema migration. INV-RECEIPT still holds — it is the behaviour that function already has. |
| §6.1 `stages.backfilled` | **Cut** from the record schema. `stages` stays an open map, so re-adding it later is additive. |
| §7.4 `finalize_run` step 6 | **Cut.** `finalize_run` is six steps, not seven. |
| §8 array task backfill half | **Cut.** Shard workers aggregate only. |
| §15.4 residual risk 4 | **Cut.** |
| §7.1–§7.3 measurement/metadata inversion | **Kept in full.** This half is clean subtraction and is the point of §7. |

---

## D-B. The verification cache starts in process, and a spike decides whether it moves to disk

Audit **S1** — the finding §9.1 responds to — proposed a **process-level** cache:

> Give `valid_image_success` a process-level cache keyed on the marker file's
> `(st_dev, st_ino, st_size, st_mtime_ns)` … pair it with the existing processing-inventory
> stat sweep rather than re-hashing.

§9.1 escalated that to `.phenotypic/verification_cache.json` — a new tracked artifact, in
the spec that exists to remove tracked artifacts. Every cadence the audit measured is a
**repeated call inside one long-lived process**: the observer's 2 s daemon tick, the
viewer's 5–10 s per-tab poll, `OutputRoot.discover`'s double read, `OutputMutationGuard`'s
double read. An in-memory cache serves all of them.

On-disk buys exactly one thing an in-memory cache cannot: **cold-start reuse across
processes** — a fresh GUI launch, the CLI deriving a resume worklist, each SLURM worker.
It costs an identity fence on disk, a corruption surface, the INV-VERDICT mutation suite,
`clear_machine_state` coupling, and last-wins concurrent writes.

**Decision:** implement in-process. Run **spike S-5** at implementation time to measure
whether cold start actually matters at realistic `N`; only add the on-disk variant if it
does.

INV-VERDICT and its mutation suite still apply to the in-process cache — the invariant is
about *what a cache may cause*, not about where it lives. The forged-file cases in the
suite become forged-dict cases; the corrupt-JSON cases move to S-5's on-disk variant if
one ships.

---

## D-C. `scientific_config_digest` is the existing digest, verbatim (was Q1)

### The evidence

Spec §5.4 says `scientific_config_digest` is "**not a new digest** … reused verbatim", then
lists its contents and claims they are "exactly as `work_id` does today", and separately:

> Fields that are finalization inputs rather than per-image configuration
> (`metadata_sha256`, `include_dataset_column`, `no_qc`) belong to
> `finalization_input_digest` and appear in **neither** `work_id` nor the generation.

The actual function, `processing_configuration_digest_from_values`
(`src/phenotypic/_cli/_cli_failure_tracker.py:200-243`), has this non-process branch:

```python
    else:
        payload.update(
            {
                "include_dataset_column": include_dataset_column,
                "overlay_alpha": overlay_alpha,
                "save_overlays": save_overlays,
            }
        )
```

folded into `work_id` at `_cli_failure_tracker.py:265`. So today:

- **`include_dataset_column` IS in `work_id`.** §5.4 says it is in neither.
- `overlay_alpha` and `save_overlays` are in `work_id` and appear nowhere in §5.4's list.
- `ext` and `process_format` are in the **process** branch only; §5.4 lists
  `process_format` unconditionally.

`validate_resume_compatibility` (`_cli_state_management.py:337-346`) guards the same
superset, so §5.4's appeal to "the fields `validate_resume_compatibility` already guards"
also under-lists. And §5.1 states `work_id` is "unchanged" — which §5.4 as written
contradicts.

### Decision

`scientific_config_digest := processing_configuration_digest(config)`, verbatim and
unchanged. §5.1 holds. §5.4's **argument** holds — it needs `generation ⊇ work_id`'s config
digest so the two can never disagree about what counts as scientific configuration, and
identity satisfies that maximally. §5.4's **field list** is the demonstrably wrong half and
is corrected in P2.

Consequence: `include_dataset_column` appears in both the generation and
`finalization_input_digest`. That is not incorrect — the two answer different questions —
but §5.3's "none is redundant" needs the footnote P2 adds. Flipping it still reprocesses
every image; removing the three fields from the per-image digest is a sound follow-up, and
it is a `work_id` change that deserves its own spec and its own migration rather than a
ride on this one.

---

## Q2. Verdict precedence

**Status: gap in the spec. Plan decides.**

Spec §4.3 defines four verdicts and asserts "there is exactly one path to each verdict",
but never orders them, and three of the four can hold at once on a real tree.

**Decision — first match wins, in this order:**

1. valid run proof covering the current inventory → `complete`
2. else a liveness authority reports work in flight **for the current identity** → `active`
3. else terminal-failure records exist with no superseding success proof → `failed`
4. else → `incomplete`

`complete` outranks `active` because a run proof covers the *current* inventory: a live
worker at that point is either fenced by `restart_epoch` (stale) or is a new invocation
that has already changed the inventory — in which case rule 1 does not fire. The ordering
is self-resolving rather than a tie-break.

`active` outranks `failed` so a failure journal entry from a previous attempt cannot mask
an attempt currently retrying it.

**This ordering is a test** (P1 Task 5), not a comment.

---

## Q3. `ImageState` is used but never defined

Spec §9 declares `images: Mapping[str, ImageState]` and never defines the type. P1 Task 2
defines it. With D-A, its `stages` map carries `stage1`/`stage2`/`stage3`/`measured` and
**not** `backfilled`.

---

## Q4. `sdk_` cannot import `_cli` for the readers it must host

Spec §11 moves `_cli_completion.py`'s readers into `sdk_/_run_state.py` and calls the
asymmetry "structural, not conventional". But `_cli_completion.py:14` imports
`phenotypic.sdk_` at module scope, and today's readers call `load_processing_state`
(`_cli_state_management.py:98`), which **replays the whole event log** on every load
(`:121`). `sdk_` already reaches into `_cli` from 16 sites, all lazily inside function
bodies.

**Decision:** `sdk_/_run_state.py` imports nothing from `phenotypic._cli`, at module scope
or inside a function, and reads `processing_state.json` as plain JSON. That is possible
precisely because §4.2 deletes `processing_state.datasets.{completed,failed,started}` and
demotes the event log out of the evidence set: what a verdict depends on is
`config.work_ids` and the digests, all literal JSON fields.

Pinned by **INV-LAYER**, an AST test in P1 Task 1. Without a test, "structural" is
convention with extra steps — the GUI's 25 private `_cli` imports across 9 modules are
what that looks like at scale.

---

## Q6. Ten test files depend on `master_measurements_csv_path`

Sizing note, not a contradiction. `grep -rl` gives 10 files in `tests/`, 6 in `src/`;
`MASTER_MEASUREMENTS_CSV` gives 4 and 1. D8 deletes all of them in P4.

`BundleLayout.detect` (`sdk_/_io_constants.py:2422`) keys discovery on
`master_measurements.parquet`, not the CSV, so bundle detection is unaffected — the risk is
confined to fixtures that write or assert the CSV.

---

## Still open, not blocking — raised for a later pass

None of these changes what P0–P7 build. All are recorded so they are visibly
*deferred* rather than unnoticed.

### O-1. `scheduler_epoch` may be five names collapsing to one owner, not five tokens to one

§5.1 has `scheduler_epoch` "absorb" `slurm_generation`, staged `epoch`, `lifecycle_epoch`,
`execution_epoch` and recompile's `attempt_id`. Those five are written by four subsystems
(`_cli_slurm_lifecycle`, `_cli_staged_orchestration`, the recompile worker, the local
strategy) at four different times with four different lifetimes. Collapsing the *names*
without collapsing the *writers* gives one value with four owners, which is a coupling
increase dressed as a cardinality reduction.

> **RESOLVED, and the answer is zero** (P2 Task 4, user-ruled). The paragraph below is
> the *question as posed*; it proposed collapsing "only where a single writer already
> owns the lifetime". Checked writer by writer, **no token qualifies** —
> `design.md:323-345` has the table. The deciding distinction: *ownership* says who may
> change a value, *persistence* says who else can still read its name, and a token can
> have exactly one writer and still be a public on-disk format. All five are.
>
> `slurm_generation` and `lifecycle_epoch` really are one value passed twice, but that
> made the check a **dead comparison to delete**, not two tokens to merge — deleting it
> changed no name, no key and no behaviour. The read-both-keys shim alternative was
> rejected separately: dual-key support is *more* state to keep in sync, in a change
> whose purpose is reducing it.

P2 Task 4 therefore collapses them **only where a single writer already owns the
lifetime** — `slurm_generation` and `lifecycle_epoch` are the same value passed twice today
(the audit's own finding, §11.1: `_assert_worker_generation`'s
`slurm_generation != attempt_id` check is "one value passed twice, then asserted equal") —
and leaves staged `epoch` and recompile `attempt_id` as *diagnostic* fields written under
the collapsed name. If a later measurement shows four writers never actually race, the
rest of the collapse is a follow-up.

### O-2. `stages` is an open map with no name validation

§6.1 makes `stages` open so the schema can grow additively. Nothing then validates a stage
name, so `"stage_2"` reads as "stage 2 not done" and never errors. P3 Task 2 keeps the map
open but shares `STAGE_*` constants between writer and reader, closing the typo class rather than reporting it. **Superseded by CAN-27.** The original wording proposed a `KNOWN_STAGES` frozenset and an advisory for
an unrecognised key — surfacing the typo without closing the map.

### O-3. Curating a run makes it undiscoverable — the aggregate proof fences two files the GUI owns

**Status: established by measurement, not implemented. Owed task, red-first, after P6.**

> ## Read this first: do NOT bump `AGGREGATE_PROOF_VERSION`
>
> The obvious implementation is "the proof's meaning changed, so bump the version". It is
> worse than the bug. `_valid_aggregate_proof` returns `None` on a version mismatch
> (`sdk_/_run_state.py:1122`), so a bump makes `aggregate_proof_is_current` `False` for
> **every existing schema-3 tree on disk** → `core_readable` `False` → every run
> undiscoverable until re-finalized. That converts a bug which fires when a user curates
> into one that fires when a user **upgrades**, for everyone, with no action on their
> part. The fix below needs no version change.

#### The finding

Marking one colony in the results viewer makes that run refuse to open, permanently, with
no remedy in the message. Measured end to end on a tree built by production publishers —
`docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/curation_fence_probe.py`:

```
ARM FENCED  (success_markers_required == True)   -- as published
  core_readable (before) = True     discover (before) = OK
  curating 'a' object 1 as 'oversegmented'  -> curation WRITTEN
  core_readable (after)  = False
      master_parquet        match=True   1593 -> 1593
      measurements_csv      match=False    89 -> 74
      measurements_parquet  match=False  1593 -> 1570
  discover (after) = ValueError: Core aggregate files are not authorized by a
                     valid aggregate publication marker

ARM UNFENCED (success_markers_required == False)
  ... identical byte damage ...
  core_readable (after)  = True      discover (after) = OK
```

The chain: `publish_aggregate_snapshot` fences three artifacts by size + sha256
(`_cli_completion.py:1119-1122`) — the master **and both mirror files**.
`CurationLabels._write_curated_mirror` rewrites two of the three
(`_curation_labels.py:846,848`) and republishes nothing. `publish_aggregate_snapshot` has
exactly two call sites, `_cli_finalize_run.py:484` and `sdk_/_hdf_to_zarr.py:746`, neither
reachable from the GUI, and there is no recertify path under `src/phenotypic/gui/`.

**Two arms, because the control is what makes this a measurement.** Identical byte damage,
opposite outcomes; `aggregate_proof_is_current` goes `False` in **both**, so the proof
breaking is not the deciding term — the divergence is entirely
`state_requires_success_markers`. A single-arm probe would have shown the raise and left
"would a legacy tree also break?" open, which is the question that says how many trees are
affected.

**Two hypotheses are dead, not merely unobserved:**

* `_publish_if_current`'s CAS guard does **not** save this. It refuses a write when a
  curation source changed; nothing had changed, so it wrote. `curation WRITTEN` in both
  arms.
* `master_measurements.parquet` is untouched in both arms. The damage is exactly the two
  artifacts the GUI is *designed* to rewrite.

**Pre-existing.** The retired classifier computed `core_readable` the same way and
`discover` raised at the same site with the same string. What is new is only that P6-T1's
fixture is the first to set `success_markers_required`, so it is the first to reach the
predicate's second disjunct at all. **Nothing about P6 landing makes it worse**, which is
why it is owed rather than in-scope.

#### Disposition: unfence the mirror

The aggregate proof has **two consumers asking different questions** — `resolve_run_state`
asks *"did the CLI's finalization publish a complete set?"*, `core_readable` asks *"are
these bytes safe to read now?"* — and one artifact answers both. The mirror's dual
ownership is what makes the difference visible.

**The deciding argument is the claim's truth value, not a principle about mutability.**
The fence asserts *"these are the bytes finalization published"*. After a curation click
that is **false, by design** — the curated mirror is intentionally not the CLI's output.
The response to a certificate whose statement has become false is to stop making the
statement.

* **Rejected — GUI republishes the proof.** It makes the proof mean "the bytes as of the
  last write by anyone", a weaker and *different* claim than the one the fence exists to
  make, and it puts a publisher call on the GUI side of one-writer.
* **Rejected — a second receipt certifying the current mirror.** That is a *provenance*
  record, not an integrity fence, and the GUI already has one: `curation_labels.parquet`
  is the durable authority for what changed. A receipt would be a third producer of that
  fact.

**This is audit S2's disposition not carried down one layer.** S2 excluded GUI-owned
mutable state from `snapshot_is_current()` because its writer carries its own guard and a
frozen fingerprint made the viewer report its own writes as external drift
(`_output_root.py`'s docstring says so). The aggregate proof fences the same kind of file
and never got the same treatment. That framing also predicts where else to look: **any
other place this change froze a fingerprint over a file with a live second writer.**

> **A trap for the implementer, from `gui/CLAUDE.md`'s "Master versus mirror" section.**
> `OutputRoot.master_df` holds the **mirror** (23 read sites in `gui/`);
> `clean_master_df` holds the **master** (4). Unfencing "the mirror" therefore stops
> certifying the frame the viewer reads everywhere and keeps certifying the one it barely
> touches. That is the correct outcome here — there is nothing true left to certify about
> the curated mirror — but anyone implementing this without that section in front of them
> will think they are unfencing the less important file.

#### The shape of the fix — reader-side, no version bump

`sdk_/_run_state.py:1131-1133` is:

```python
    for descriptor in outputs.values():
        if fenced_artifact_path(output_root, descriptor) is None:
            return None
```

**Keep writing all three descriptors; enforce only the CLI-owned one.** The proof stays a
complete record of what finalization published — provenance worth having — while
validation stops asserting a claim curation legitimately falsifies.

> *Record what happened; enforce what must not change.*

Being reader-side is not a stylistic preference: it **repairs trees already on disk with
no migration**, which a writer-side change cannot do.

**Rewrite `_cli_completion.py:1113-1118` in the same commit.** Its comment states the
contract this fix falsifies — *"`valid_aggregate_snapshot` and the sdk_ reader both
validate whatever the proof LISTS"*. Leaving it is a correction that leaves a false
sentence behind it.

#### Severity — higher than "an exception is raised"

* **The sidebar classifier consults neither `core_readable` nor the proof**
  (`shell/_classifier.py`), so the directory still presents as viewer-openable. The
  affordance says open; the open fails.
* **The hub** returns HTTP 400 carrying the raw string (`shell/_routes.py:376-381`).
* **The standalone launcher does not catch it at all** — `results_viewer/__main__.py:94`
  lets the `ValueError` escape, so `python -m phenotypic.gui.results_viewer` fails to
  start on a curated run.
* **The message names no remedy**, unlike its neighbour in the same function, which ends
  *"Re-run `python -m phenotypic` with the current version to regenerate the master."*
* **The remedy that exists costs the user work.** Re-running finalization republishes the
  proof over current bytes and rewrites the mirror from the master, replacing the curated
  bytes. `curation_labels.parquet` is not fenced and survives, so the labels are not lost
  — but the run is inaccessible until the user runs the CLI again, which is a heavy price
  for one click.

#### Open, needs a user ruling — not answered here

**Should the sidebar classifier consult `core_readable`?** Today it advertises as openable
a directory that cannot be opened, and that is true independently of this fix: any
`core_readable` `False` tree has the same gap. Making the classifier ask would remove the
dead affordance, at the cost of putting a run-state read on the sandbox walk — which is
the cost `_rehydrated_status` already pays elsewhere and which the boot walk is already
documented as synchronous. **Left open deliberately; it is a UX decision, not a
correctness one.**

#### What the owed task needs

The red test is the probe's FENCED arm, already written and committed beside this plan.
Then the reader-side filter, the comment rewrite, and a decision on the open question
above. It edits `sdk_/_run_state.py` and `_cli_completion.py`, so it needs its own commit
and cannot share one with consumer-migration work.

**Also fold in: `_valid_aggregate_proof` must say *why* it refused.** Today it returns a
bare `None` for four distinguishable causes, so nothing downstream — a user, a log, a
support request — can tell *"your master was tampered with"* from *"your curation broke
it"*, and those two have opposite dispositions. The enumeration **is** the specification:

| Cause | Site |
|---|---|
| no marker on disk | `_read_json_object(...) is None` |
| `version` != `AGGREGATE_PROOF_VERSION` | `:1122` |
| `required_outputs` absent, non-mapping, or empty | `:1124-1126` |
| one named descriptor no longer matches its bytes | `:1131-1133` — **say which** |

Two constraints on the shape:

* **`core_readable` stays a boolean and still refuses.** Whatever carries the cause must
  not tempt a caller into treating a proof problem as recoverable; a reason is for
  reporting, never for downgrading the verdict.
* It is a **behaviour change** to `sdk_/_run_state.py:1112-1134` and wants its own test.

Its value does not depend on how the fence question is resolved: it is useful before and
after the fix, implies no remedy, and legitimises nothing — which is why it belongs here
rather than waiting on the ruling above.

**A note for whoever writes the remedy string.** The user-facing message names no remedy
today, and improving it must wait for the fix — written now the text would describe
*curation*, written after it describes a *tampered master*, which is the same string
acquiring a different subject. When it is written: *"re-running finalization replaces the
curated mirror"* is a property of **re-running finalization**, true before and after this
fix and for every caller. That caveat belongs on the command, not on this error.
