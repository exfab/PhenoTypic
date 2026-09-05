# Run-completion contract, identity schema, and measurement/metadata layout

**Status:** design approved, spike-gated
**Companion documents:** [`audit.md`](audit.md) (the findings this responds to),
[`DEFERRED.md`](DEFERRED.md) (findings explicitly out of scope)

---

## 0. Amendments — read before any section below

**Status: approved 2026-09-03, then amended through three rounds of adversarial review.**
Nine user rulings and three pre-review decisions supersede parts of what follows. **This
section is authoritative where it conflicts with a later section**, and each row says where
the reasoning lives.

Two of these are **factual corrections** — the spec asserted something about the existing
codebase that is not true — and they are marked ⚠ because a reader who acts on the original
sentence will be wrong about shipped code.

| # | Section | Amendment |
|---|---|---|
| **D-A** | §6.3, §6.4, §6.1, §7.4, §8, §15.4 | **Per-store metadata is written at promote time, not backfilled.** Cuts the hardlink re-promote, the certified-rewrite *generalisation*, `stages.backfilled`, `finalize_run`'s step 6, the backfill half of the fan-out, and residual risk 4. §7's inversion is kept in full. |
| **D-B** | §9.1 | **The verification cache is in-process**, not `.phenotypic/verification_cache.json`. Audit S1 proposed process-level; §9.1 escalated it without cause. |
| **D-C** ⚠ | §5.4 | **§5.4's field list is wrong.** It claims `include_dataset_column` is in neither `work_id` nor the generation "exactly as `work_id` does today". `processing_configuration_digest_from_values` (`_cli_failure_tracker.py:236-243`) puts it, `overlay_alpha` and `save_overlays` in `work_id`. §5.4's *argument* survives; its field list does not. `scientific_config_digest` **is** `processing_configuration_digest`, verbatim. |
| **U-12** | §5.3 | **"None is redundant" is a claim about the four DIGESTS, not about their fields.** A field belongs to every digest whose question it can change the answer to, so a field appearing in two digests is correct rather than a defect. Settled because `include_dataset_column` is in both `scientific_config_digest` (`_cli_failure_tracker.py:230`) and `finalization_input_digest` (`_run_state.py:172`), which the alternative reading would make a bug. **No code changes; no digest value on disk changes.** |
| **U-1/U-6** | §2, new | Migration floor is the **pre-markers shape** — schema `2.0.0` with no `work_ids` key. `state.version` cannot express a package-version floor: `"2.0.0"` is the value at v0.17.3 *and* immediately before `"3.0.0"`. No `BELOW_FLOOR` verdict. |
| **U-2** | §4.3 | `complete` keeps **both** clauses. Rule 1 also carries the full five-way comparison `current_aggregate_is_current` (`_cli_completion.py:738-745`) makes — dropping `finalization_input_digest` alone breaks §7.4's late-metadata guarantee. |
| **U-3** ⚠ | §7.3 | **"tag the master … so an old reader fails loudly" is false as stated.** `pd.read_parquet` / `pl.read_parquet` / `pq.read_table` ignore Parquet KV metadata and raise nothing; an *old* reader predates the key entirely. The stamp is kept **and given a reader** — `read_master_measurements()` in `sdk_` — so the claim narrows to: in-repo readers fail loudly, external ones are unaffected. |
| **U-4** | §5.1, §5.3 | **`publication_id` is cut.** Once §5.1 redefines it as `sha256(source_set_digest ‖ finalization_inputs)` it is a pure function of two values the binding check already compares. The run proof carries `source_set_digest` instead. **Six tokens → five.** |
| **U-5** | §9 | `RunState.diagnostics` drops `manifest_completed`, `manifest_total`, `event_log_present` — verified zero consumers survive the consumer migration. |
| **U-7** | §11.1 | **Migration logic lives only in `--mode migrate`.** The legacy promoter's `--mode full` dispatch is deleted. §11.1's "move into migrate" stands, but for this helper it is a **rewrite**, not a move — migrate builds no `ExecutionConfig`. |
| **U-8** ⚠ | §5.4 | **WITHDRAWN in round 4 — the mechanism does not exist.** Migrate was to omit the unrecoverable digest fields and a forward run was to omit them too, so both hash the same reduced payload. No forward run does or can: `work_id_for_image` *recomputes* the id from `ExecutionConfig` and never reads `state.config`, and `processing_configuration_digest_from_values` writes all twelve fields unconditionally. Superseded by **U-10**. |
| **U-11** | §9.1, §9.2 | **The on-disk tier SHIPS, and the GUI stops blocking on verification.** D-B's in-process-only ruling is reversed on measured evidence: a cold deep pass is 1403 s at N=6,657 while the stat sweep is ~37 s. Separately, `OutputRoot.discover` moves off the startup path — the viewer opens immediately and verification runs behind it. |
| **U-10** | §5.4, §6.1, new | **Migrated records are marked, not fabricated.** Migrate publishes records carrying `provenance: "migrated"`; `valid_image_success` accepts such a record on **artifact validity alone** — no `work_id` comparison — and `resolve_run_state` emits an advisory that the configuration fence is unavailable for those images. Discharges MIG-21 and MIG-22 together. |
| **U-9** | §6.1 | **`stage2_done/` stays a separate file.** §6.1 folds it into `stages.stage2`; it is a consumable signal cleared by an atomic `unlink`, and folding it in replaces that with a locked read-modify-write — per-image, across a 6,000-task array, on `flock` (`sdk_/_file_locking.py:101`), whose cross-node semantics are the weaker POSIX option. Only `image_complete/` and `stage3_complete/` collapse. |
| **INV-ONEWRITER** | §6.1 | The collapsed record needs **no lock**. Disjoint work partitioning (one image → one task) plus stage sequencing give one writer per image, and **mode exclusivity** covers the two driver-side sweep writers that are not per-image at all; `atomic_write_json`'s temp-write + `os.replace` covers the crash case. |

### ⚠ §6.2's central claim is false, and was false before this spec was written

> §6.2: *"the root `zarr.json` is written **last** … and nothing writes into the store after
> publication, so a valid root implies a complete store."*

`--mode measure` already re-promotes proven stores:
`_cli_process_single.py:439` → `replace_image_store_measurements`
(`_cli_output_manager.py:1970-2001`) → `replace_embedded_measurement_table`
(`sdk_/_measurement_tables.py:242`), whose `_clone_file_without_pixel_rewrite` (`:233`) is
`os.link` with a `shutil.copy2` fallback — **that is §6.3's hardlink re-promote, shipping
today.**

Its *in-place* branch (`:284-290`) is worse, and it is worth stating precisely, because the
first phrasing of this amendment ("no `.part`") understated it. The **file** write is
atomic: `_write_validated_parquet` (`:55-81`) goes through `atomic_write_with_writer`, with
a re-read schema check before the rename. What the branch skips is the **store-level
transaction** — no `new_part_path` copytree, and **no root `zarr.json` rewrite at all**. So
`tables/measurements/table.parquet` acquires different bytes while the root the readers
validate against is byte-identical to before.

The failure that follows is therefore not a torn file. It is worse: a fully-formed,
schema-valid store whose root certifies contents it no longer has. Root-last atomicity
detects an *interrupted* write and is blind to a *completed* one, so **a valid root does not
imply unchanged contents** — and every reader that treats the root as a content proof is
reading a stale certificate.

`src/phenotypic/_cli/CLAUDE.md:251-254` repeats the same false claim and is corrected in the
same change.

The invariant is restated as **INV-PROVEN**: *an artifact carrying a content proof changes
only where the proof changes with it.*

### ⚠ U-8 was withdrawn in round 4, and U-10 replaces it

**U-8 asked a reasonable question and got a wrong answer from me.** The question — *"can't
we leave the unknown fields blank, and let the remaining values still generate a hash?"* —
is sound. A reduced payload does hash. What it does not do is **match**, and the reason is
that no reader ever consults the id migrate writes:

- `work_id_for_image` (`_cli_failure_tracker.py:329-350`) **recomputes** the identity from a
  live `ExecutionConfig` on every call. It never reads `state.config`.
- `processing_configuration_digest_from_values` (`:200-214`) takes all twelve fields as
  **required** keyword parameters and writes them into the payload unconditionally.
- A **second** producer, `_worker_work_identity` (`_cli_process_single.py:122-171`), derives
  the same id from argv, and the two are cross-checked with a hard `RuntimeError`
  (`:723-729`).

So "a forward run reading that state omits them too" describes no code path that exists. It
could be *made* to exist — by making "not asserted" a first-class value through both
producers, `ExecutionConfig`, and the cross-check — but that is an invasive change to the
identity path taken for the benefit of trees that never had an identity fence at all.

Two of the seven `work_id` fields also cannot be honestly reproduced regardless:
`input_sha256` is `file_sha256` of the **original input image** and is never persisted —
only fed into the digest — and `pipeline_fingerprint` hashes the user's original pipeline
file, where the output tree holds only a `to_json()` re-serialization.

**U-10 (round 4, user): mark the record instead of fabricating the identity.**

Migrate publishes per-image records carrying `provenance: "migrated"`. For such a record:

1. `valid_image_success` accepts it on **artifact validity alone** — the artifacts exist and
   their content proofs verify — with **no `work_id` comparison**.
2. `resolve_run_state` emits one advisory naming the affected images: the configuration
   fence is unavailable for them.
3. Every other record is unchanged. A record without the marking is fenced exactly as
   before.

**What it costs:** those images lose the configuration fence, so a later run under a
different pipeline reuses them rather than reprocessing.

**Why that is the right trade, and not merely the cheapest one:**

- It **removes no guarantee**. A v0.17.3 tree has zero `work_ids` — `git show
  v0.17.3:.../_cli_state_management.py` has no such key, and its config block holds seven
  entries of which five are digest inputs. The fence never existed there; U-10 declines to
  fabricate one rather than deleting one.
- It is **self-limiting**. The moment an image is reprocessed by a modern run it acquires a
  real `work_id` and the marking is gone. The weakening cannot propagate to any image a v2
  run has touched.
- It is **visible**. An advisory that says the fence is unavailable is strictly better than
  a fabricated id that silently asserts one.

**This is what discharges INV-DISCHARGEABLE for the pre-markers `--mode process` tree
(MIG-21).** That shape classifies `CONVERT`, and before U-10 migrate converted nothing —
so `_refuse_unmigrated_output` (`phenotypicCLI.py:1661-1662`, which runs *before*
`--restart`) refused it in every writing mode permanently, escapable only by `--overwrite`,
which deletes the outputs. `--mode process` handles that tree in place today, so it was a
regression, not a gap. With U-10 migrate emits marked records and the gate clears.

### ⚠ U-11 reverses D-B, and adds a requirement D-B never considered

**D-B ruled the verification cache in-process only.** That ruling was made before anyone
had measured the thing it was about. Measured on a fresh compute node, cold GPFS client
cache, disjoint halves so neither lane warms the other:

| Cold start at N=6,657 | Seconds | Per file |
|---|---|---|
| Today — hash every artifact (73.6 GB) | **1403** | — |
| Drop the overlay from the proof | ~256 | 19.4 ms `open+read` |
| **Stat sweep — what the on-disk tier costs** | **~37** | 1.9 ms `stat` |

The cost is **per-file latency, not bytes**: a cold `open+read` is ~10× a cold `stat`, so
replacing reads with stats beats removing 99.4% of the bytes by roughly 7×.

**The tier ships. Two things must be stated where the code lives, not only here.**

1. **It is a tenth artifact in a change that removes nine.** That is a real cost and the
   register in P7 Task 6 lists it — under *cache*, never under tracked state. Nothing may
   branch on it, and no verdict may be derived from it.
2. **It weakens the sentence §9.1's safety argument rests on.** *"An entry only lets a
   previously deep-verified result stand"* means, in process, *by this process, minutes
   ago*. On disk it means **by some process** — possibly an older build with different
   verification rules, possibly one still mid-write, possibly another user. The degradation
   story still holds (any doubt falls through to `deep`), but the guarantee is materially
   different and must say so in `_verification_cache.py`'s module docstring rather than
   inheriting wording written for the in-process case.

### And the GUI no longer blocks on verification

**New requirement, and it changes what the tier is for.** `OutputRoot.discover` is a
`deep`, binding call on the viewer's startup path, so today a cold tree hangs the GUI for as
long as verification takes. It moves to a background thread: **the viewer opens immediately
and populates as verification lands.**

The scaffolding is already half-built — `discover` takes `cancellation`
(`OutputDiscoveryCancellation`, documented thread-safe and cooperative) and
`progress_callback` (phase updates). What is synchronous is the *call site*,
`results_viewer/__main__.py:94`.

**The safety property this must not break.** The whole change exists to stop unverified
state being read as verified. A viewer that renders before verification completes is
displaying exactly that, so **the pre-verification state must be visibly distinct** — not a
spinner that resolves into indistinguishable content. It shows what it has, says that
verification is in flight, and says so in a way a user cannot mistake for a finished answer.

**Note the interaction, once.** With the GUI non-blocking, the tier's value is no longer
*"a user waits 23 minutes"* — it is *"background verification finishes in 37 s rather than
23 minutes"*. Still worth having: less shared-filesystem I/O, and a trustworthy view much
sooner. But the urgency that justified reversing D-B is reduced by the second decision, and
a later reader should know both were taken together rather than reading the tier as
load-bearing for interactivity.

### What this cost, recorded once

Three rounds of review found **six** defects of one shape: a function reading a format this
spec changes, in a file the plan never named — `authorized_measurement_sources`,
`replace_embedded_measurement_table`, a third `_consistent_embedded_join_keys` site, four of
seven proof publishers, ≥5 of ≥9 staged-engine sites, and 1,130 lines of recompile.
A complete consumer map now lives in the plan's P3 (**136 reads across 20 modules**),
regenerable by one `grep`. **Any future change to a stored format should start there.**

---

## 1. Purpose

Three coupled problems, from the audit:

1. **Nine evidence sources for "is this run done?"**, cross-checked by four
   independent classifiers that can disagree. Disagreement surfaces as
   `contradictory` → the output is flagged read-only for a reason the user
   cannot act on.
2. **Fourteen identity tokens** (thirteen counted in the audit, plus a
   fourteenth found during design hiding under the `attempt_id` alias), each
   written to disk and cross-checked against at least one other.
3. **The completion predicate is `O(N_images)` in full-file SHA-256**, and is
   invoked on a 2-second daemon tick and a 5–10-second per-browser-tab poll.

The organising principle, from the user: **move state that is tracked to state
that is checked.** Anything derivable from files on disk should be derived, not
written down and kept in sync.

Layered on top, because it is entangled with all three: the
**measurement/metadata layout inversion** — embedded per-image tables become
pure measurements, the metadata join moves to finalization, and per-image
metadata is backfilled into each store so a consumer can join in memory.

## 2. Scope

### 2.1 In scope

- Consolidated completion contract: 9 sources → 3 authorities
- Consolidated identity schema: 14 tokens → **5, two of them content-derived** (amended by U-4; §0 is authoritative — `publication_id` is cut)
- Per-image marker collapse: `image_complete` / `stage2_done` / `stage3_complete`
  → one record with a `stages` object
- Certified post-hoc store rewrite protocol (generalised from the existing
  metadata-migration special case)
- Embedded measurement tables become un-joined; metadata join moves to
  finalization; `pht-metadata.parquet` backfilled per store
- One shared `finalize_run` path for `full`, `measure`, and `recompile`
- SLURM + local fan-out of aggregation **and** backfill
- Consumer migration: viewer consistency, CLI core, `RunRegistry`
  claimability + local exit, SLURM observer **call sites only**
- `--mode migrate` conversion; every other mode refuses an unconverted tree

### 2.2 Out of scope

The SLURM observer's decision tree, grace window, and `squeue`/`sacct`
reconciliation are **not** restructured — only its two `_cli_completion` call
sites and its Stage-3 probe move. Rationale: only ~185 of its lines are
filesystem-derived; the rest is scheduler domain, it is the least testable code
in the GUI, and its failure mode ("run stuck in reconciling") is user-visible.

All other audit findings are recorded in [`DEFERRED.md`](DEFERRED.md).

## 3. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Clean break; `--mode migrate` required.** New code reads only the consolidated schema. | Deletes every legacy branch rather than quarantining it. Roughly half the complexity in the classifiers is legacy compatibility. |
| D2 | **Deep once, cheap thereafter.** Full content verification at binding, pre-mutation and finalize; routine polls re-stat only. | Detects everything the current code detects except an in-place edit preserving both size and `mtime_ns` — which §8.3 of the resume spec already declares outside the threat model. |
| D3 | **Content-derived tokens where possible.** | Same inputs → same token, so resume and fencing become emergent rather than bookkeeping. |
| D4 | **`restart_epoch` is the one added tracked value.** | Content-derived generations cannot distinguish "deliberately fresh attempt" from "same config again" — the stale-worker case. One integer, preserved by `clear_machine_state`. |
| D5 | **`--restart` keeps reusing surviving `results/` stores.** | Preserves current user-facing meaning; the epoch fixes the stale-worker hazard without turning `--restart` into `--overwrite`. |
| D6 | **`validate_resume_compatibility` keeps its refusal.** | A config change still hard-errors with the specific mismatch. The content-derived generation is a second, structural line of defence, not the primary one. A one-character pipeline edit must not silently reprocess 6,000 images. |
| D7 | **`inventory_digest` stays out of the generation digest.** | Generation fences *configuration*; `inventory_digest` fences *scope*. They change on different schedules. Conflating them would make every new image under a rolling input look like a configuration change, resetting live progress and fencing in-flight workers. |
| D8 | **Master is parquet-only; the mirror carries the CSV.** | The un-joined master is no longer the file a human opens. Drops `master_measurements.csv` and shrinks the aggregate proof's `required_outputs` from four artifacts to three. |
| D9 | **One combined spec** rather than sequenced PRs. | User decision, taken after the review/bisect cost was stated. Mitigated by explicit internal phases (§12) so the implementation plan can still land in stages. |
| D10 | **Spike-gated.** §10 must run and report before implementation begins. | The hardlink re-promote (S-1) is load-bearing: a bad result invalidates §6.3 and cascades. |

---

## 4. Authority model

### 4.1 Three written authorities

| Authority | File | Why it cannot be derived |
|---|---|---|
| **Accepted inventory** | `work_ids` in `processing_state.json` | A directory listing is a different question from "what did this run accept". |
| **Terminal failures** | `.phenotypic/terminal_failures.jsonl` | A failure leaves no artifact. Absence of output is indistinguishable from not-yet-started. |
| **Liveness & ownership** | `slurm_lifecycle.json`, `slurm_jobs.jsonl`, `gui_launch_owner.json` | External-system and process facts. A crashed worker leaves no trace. |

### 4.2 Demotions

| Was evidence | Becomes |
|---|---|
| `manifest.json` counts | `RunState.diagnostics`. Nothing branches. The resume spec already calls the manifest "itself a cache"; this enforces it. |
| `processing_state.datasets.{completed,failed,started}` | **Deleted from the file.** Already re-aggregated from the event log on every load — a cache of a cache. |
| `processing_events.log` as completion evidence | Diagnostics and live dashboard progress only. Deletes the second event-log replay. |
| `stage2_done/*.json` | `stages.stage2` in the merged record. |
| `stage3_complete/*.json` | `stages.stage3` in the merged record. |

Per-image markers, the aggregate proof and the run proof are **retained and
reclassified**: they are not tracked facts, they are content proofs — digest
manifests over artifacts that already exist.

### 4.3 The verdict type

`RunState.completion` is one of four values, replacing ~23 classification rules:

- `complete` — every accepted image has a valid proof, and a valid run proof
  covers the current inventory
- `incomplete` — proofs missing for some accepted images; safe to read, safe to resume
- `failed` — terminal failure records exist with no superseding success proof
- `active` — a liveness authority reports work in flight

**`contradictory` is deleted as a state.** It exists today only because derived
counts are cross-checked against each other. Once counts stop being evidence,
two authorities cannot disagree: there is exactly one path to each verdict.

Half-migrated trees holding unconverted `.h5` files contribute a
`RunState.advisories` entry — informational, not a gate.

## 5. Identity schema

### 5.1 The six tokens

> **AMENDED (U-4): five tokens.** `publication_id` is cut — see §0.

| Token | Kind | Derivation |
|---|---|---|
| `work_id` | content | unchanged: sha256 over schema version, dataset, input-relative path, input sha256, pipeline fingerprint, per-image config digest, mode |
| `processing_generation` | content | `sha256(pipeline_sha256 ‖ scientific_config_digest ‖ restart_epoch)` — see §5.4 |
| `publication_id` | content | `sha256(source_set_digest ‖ finalization_inputs)` |
| `restart_epoch` | tracked counter | monotonic int; preserved by `clear_machine_state` |
| `scheduler_epoch` | opaque | absorbs `slurm_generation`, staged `epoch`, `lifecycle_epoch`, `execution_epoch`, and recompile's `attempt_id` |
| `owner_generation` | opaque | GUI launch ownership |

Retained but reclassified as **diagnostic** — written, never branched on:
per-image `attempt_id` (its only consumer type-checks it), and the event-log
`generation` field.

Reclassified as **in-memory CAS counters, not identity**: `record_revision`,
registry `revision`, `binding_generation`. None reaches disk as identity.

### 5.2 Function surface

```python
from phenotypic.sdk_ import (
    run_identity,            # (output_dir) -> RunIdentity          pure read
    mint_run_identity,       # (config, *, restart) -> RunIdentity  CLI only
    assert_identity_current, # (output_dir, identity) -> None | raises
    resolve_run_state,       # (output_dir, *, depth) -> RunState
)
```

The read/write asymmetry is structural, not conventional: `sdk_/_run_state.py`
exports only readers, so the GUI cannot reach a `publish_*` function. Today it
can, and does — 25 private `phenotypic._cli` symbols across 9 GUI modules.

### 5.3 Digest composition rules

Four digests answer four different questions; none is redundant:

> ⚠ **CLARIFIED (U-12): "none is redundant" is about the DIGESTS, not their
> fields.** Each of the four answers a question no other answers, so none can be
> deleted without losing an answer. That is the claim. It says nothing about
> field membership, and a field belongs to **every digest whose question it can
> change the answer to**.
>
> The case that forced the clarification: `include_dataset_column` is in
> `scientific_config_digest` (`_cli_failure_tracker.py:230`) **and** in
> `finalization_input_digest` (`_run_state.py:172`). Under the other reading —
> that no *field* may appear twice — that overlap would be a defect and the fix
> would be to remove it from one of them. **That fix would introduce a bug in
> whichever digest lost it**, because the flag really does change both answers:
> it alters the per-image processing config, so a result computed under the other
> setting must not be reused; and it alters the join inputs, so a master built
> under the other setting must not be treated as current. A digest blind to a
> change that moves its own answer is exactly the stale-reuse failure this whole
> design removes.
>
> So overlap is not tolerated here, it is **required**, and the rule for adding a
> field to a digest is the question, not exclusivity: *can this field change the
> answer to what this digest asks?*


| Digest | Question | Lives in |
|---|---|---|
| `inventory_digest` | Did the accepted **scope** change? | aggregate + run proofs |
| `source_set_digest` | Did the **succeeded subset** change? | aggregate proof |
| `scientific_config_digest` | Did the **pipeline** change? | both proofs |
| `finalization_input_digest` | Did the **join/QC inputs** change? | both proofs |

The `inventory` / `source_set` split is load-bearing. If ten new images arrive
and three fail: `source_set_digest` is unchanged, so the existing master is
still a truthful publication of everything that succeeded; `inventory_digest`
has changed, so the run is no longer complete *for its current scope*. Without
both, one of those two answers must be wrong. This is the same failure
`manifest_completion_inventory`'s docstring is a post-mortem of.

`finalization_input_digest` is a **versioned object**, not a flat digest, so a
later change to what the join depends on is additive rather than a second
migration.


### 5.4 `scientific_config_digest` — one definition, two uses

> ⚠ **AMENDED (D-C, U-8, U-10), and the correction LANDED in P2 Task 2.** The field
> list below is now read off the code rather than asserted; the paragraph that claimed
> `include_dataset_column`, `overlay_alpha` and `save_overlays` are excluded has been
> replaced. They ARE in `work_id`, for full and measure runs
> (`_cli_failure_tracker.py:228-233`). The argument survived; the list did not.
>
> D-C's own citation drifted and its claim was **flat where the code is branched** —
> the three fields are in the payload only when `process_only_layer is None`. Recorded
> here rather than silently corrected: the plan won this argument twice, once against
> the spec and once against the amendment that was already correcting it.
>
> **And the digest is not reconstructible for a migrated tree at all** — U-8 assumed it was,
> and U-10 replaced it. A migrated record carries `provenance: "migrated"` and is accepted on
> artifact validity rather than on a digest comparison. See §0.

`scientific_config_digest` is **not a new digest**. It is the existing
per-image scientific processing-configuration digest already folded into
`work_id` (resume spec §5.4), reused verbatim for the generation. Defining it
once is the point: if the generation and `work_id` could disagree about what
counts as "scientific configuration", a change could invalidate per-image
proofs without minting a new generation, or vice versa.

**The actual payload (corrected in P2 Task 2, read off
`processing_configuration_digest_from_values`,
`_cli_failure_tracker.py:191-235`).** It is **branched, not flat** — a detail
the D-C amendment in §0 does not capture, and the reason the earlier flat list
could be wrong in both directions at once:

| Group | Fields | When |
|---|---|---|
| Base | `image_type`, `nrows`, `ncols`, `bit_depth`, `detect_mode`, `drop_originals` | always |
| Process | `process_only_layer`, `ext`, `process_format` | **only** when `process_only_layer is not None` |
| Full/measure | `include_dataset_column`, `overlay_alpha`, `save_overlays` | **only** when `process_only_layer is None` |

The two conditional groups are **mutually exclusive** — never both, never
neither. `process_format` sits beside `ext` rather than in the base payload on
purpose: a full or measure run has no process format, and folding it into the
base would change every existing run's digest and cold-start every
continuation in flight.

It **excludes** backend resources, worker counts, checkpoint settings, study
settings, `metadata_sha256` and `no_qc` — exactly as `work_id` does, because
it *is* `work_id`'s digest.

> **The branch structure is U-12's rule applied inside a single digest.**
> `include_dataset_column` is present for a full run and absent for a process
> run, and that is correct in both cases by the same test U-12 states: *can
> this field change the answer to what this digest asks?* For a full run it
> can — the flag alters per-image processing config, so a result computed
> under the other setting must not be reused. For a process run it cannot: a
> process run publishes no master, so there is no join for the flag to move.
>
> The previous text asserted the opposite — that `include_dataset_column`
> belongs to `finalization_input_digest` and appears in **neither** `work_id`
> nor the generation. It is in both, for full runs, and U-12 rules that this
> overlap is **required** rather than tolerated. What survives unchanged is
> the §7.4 guarantee the old paragraph was reaching for: `metadata_sha256` and
> `no_qc` really are finalization-only, so a **metadata** edit still triggers
> `finalize_run` without invalidating a single image's proof.

### 5.5 `finalization_input_digest` as a versioned object

Stored as an object, digested for comparison:

```json
{
  "schema_version": 1,
  "metadata_sha256": "…",
  "include_dataset_column": true,
  "no_qc": false
}
```

Comparison is over the canonical serialization, so adding a field is a
`schema_version` bump handled by the reader rather than a second tree
migration. This is the reservation the measurement/metadata rewiring needs:
its inputs can change additively.

---

## 6. Per-image record and store lifecycle

### 6.1 One record

> **AMENDED (D-A, U-9, INV-ONEWRITER).** No `stages.backfilled`. `stage2_done/` stays a
> separate file — it is a consumable signal, not a record. No lock on the record. See §0.

```
.phenotypic/progress/images/<dataset>/<stem>.json
```

```json
{
  "version": 1,
  "work_id": "…", "dataset": "…", "image_stem": "…",
  "relative_image_path": "…", "mode": "full|process|measure",
  "stages": {
    "stage1":     {"at": "…"},
    "stage2":     {"at": "…", "objmap_shape": [1024, 1024], "detector_seconds": 1.23},
    "stage3":     {"at": "…"},
    "measured":   {"at": "…"},
    "backfilled": {"at": "…", "metadata_sha256": "…"}
  },
  "artifacts": {
    "store":        {"kind": "store", "path": "…", "sha256": "<root zarr.json digest>"},
    "measurements": {"kind": "file",  "path": "…", "size": 12345, "sha256": "…"},
    "metadata":     {"kind": "file",  "path": "…", "size": 234,   "sha256": "…"},
    "overlay":      {"kind": "file",  "path": "…", "size": 67890, "sha256": "…"}
  },
  "attempt_id": "…", "scheduler_epoch": "…", "completed_at": "…"
}
```

`stages` and `artifacts` are **open maps** — that is what makes the backfill
additive rather than a schema break. `stage2_raw/<stem>.npy` remains a separate
file: it is bulk replay data, not a record.

"Is this image done?" becomes one JSON read instead of one read plus up to
three `is_file()` probes across three directory trees.

### 6.2 The store immutability constraint

> ⚠ **AMENDED: the claim below is FALSE and was false before this spec.** `--mode measure`
> already re-promotes proven stores. Its in-place branch writes the embedded table
> atomically *as a file* but skips the store transaction and the root rewrite entirely, so
> the root certifies contents that have since changed. Restated as INV-PROVEN. See §0.

The invariant is explicit in the code, not implied: `staged_store_matches_work_id`
documents that `work_id` is "written at store-build time — never patched in
afterwards, because the root `zarr.json` is written last", and `promote_store`
requires chunks → `OME/zarr.json` → root `zarr.json`, so an interrupted store
reads as absent.

Backfilling `pht-metadata.parquet` into an already-promoted, already-proven
store would otherwise break root-last atomicity, `store_publication_token`, the
GUI's HTTP 409 contract, and the processing-inventory walk's documented
assumption that a store's contents change only via a promote. **The backfill
must therefore be a re-promote.**

### 6.3 Hardlink re-promote — SPIKE-GATED (S-1)

> **CUT (D-A).** Per-store metadata is written at promote time, so there is no
> re-promote to gate. Note also that this mechanism **already ships** — see §0.

1. Build `<stem>.ome.zarr.<uuid>.part` by `os.link`-ing every existing chunk file
2. Write `tables/metadata/pht-metadata.parquet`
3. Write `OME/zarr.json`, then root `zarr.json` last, with `tables` extended
4. `promote_store(part, final)` — unchanged move-aside/retry semantics

Cost is O(chunk-file count) inode operations rather than O(bytes). **This
requires same-filesystem hardlink support**, so a copy fallback (reflink /
`copy_file_range` where available) must exist and be tested, not assumed — a
container bind-mount crossing devices will not link.

If S-1 shows linking costs the same as copying on GPFS, this design changes
shape: either a sibling file outside the store, or metadata carried in the root
`zarr.json` rather than as a table.

### 6.4 Certified post-hoc rewrite

> **GENERALISATION CUT (D-A).** The existing
> `refresh_success_markers_after_metadata_migration` stays, scoped to `--mode migrate`.

Generalise `refresh_success_markers_after_metadata_migration` from a
metadata-migration special case into the single protocol for any legal
post-proof store mutation:

```
.phenotypic/rewrites/<kind>-<digest>.json
```

declaring `kind` (`metadata_backfill` | `metadata_schema_migration`), the
affected `work_id`s, and each store's expected before/after root digest. The
rewrite executes under the receipt; marker `artifacts` re-digest against it; an
artifact that moved **without** a covering receipt still raises `RuntimeError`.

Store immutability becomes "immutable except under a certified transition" —
which is what it already is, now named honestly and with one implementation.

---

## 7. Measurement and metadata data flow

### 7.1 Two kinds of metadata

| Kind | Examples | Fate |
|---|---|---|
| **Intrinsic identity** | `Metadata_ImageFile`, `Metadata_Dataset`, object label | **Stays** in the embedded table. A concatenated row that cannot say which image it came from is unusable. |
| **User metadata** | whatever `--metadata` supplies | **Moves** to `pht-metadata.parquet` |

The boundary already has a name. `prepare_embedded_measurement_table` computes
`measurement_columns` from the baseline *before* joining and writes it as
`phenotypic.measurement_columns`. "Embedded table without user metadata" is
exactly that existing projection — this is subtraction, not invention.

### 7.2 `pht-metadata.parquet`

Contents: the user-metadata rows applicable to that image, plus the join keys.
Nothing else. The join is self-describing via the existing Parquet KV keys,
which ride along: `phenotypic.join.keys`, `phenotypic.join.kind`,
`phenotypic.metadata.snapshot_sha256`.

When `join_status` is `not_requested` or `no_common_keys`, **no metadata table
is written**. Absence is the honest signal; `stages.backfilled` still records
the metadata digest so it is not retried.

### 7.3 Contract change

> ⚠ **AMENDED (U-3): the schema stamp cannot make an *old* reader fail loudly.** Parquet
> KV metadata is ignored by every ordinary reader. The stamp is kept and given a reader
> in `sdk_`; the claim narrows to in-repo readers. See §0.

> **Was:** `master_measurements.*` is the exact pre-post concatenation of
> authorized embedded tables (already metadata-joined measured rows)
>
> **Becomes:** `master_measurements.parquet` is the exact pre-post concatenation
> of authorized embedded **measurement** tables — intrinsic identity only, no
> user metadata. `measurements.{parquet,csv}` is the post-applied,
> metadata-joined mirror.

CLAUDE.md's existing "feed analysis and dashboards from the **mirror**, not the
master" rule already points consumers at the right file. Two master consumers
need explicit tests rather than assumption:

- **Curation re-keying** deliberately reads the clean master so labels survive
  for curated-out objects. It keys on dataset / image / object-label — all
  intrinsic — so it should be unaffected. Test it; do not assume it.
- **Anything filtering master on a user-metadata column** would return empty
  rather than error. This is the one genuinely dangerous failure mode in §7,
  and it is why the migrate step must tag the master with a schema version so
  an old reader fails loudly.

### 7.4 `finalize_run` — one path, three entry points

The seam already exists and is already shared: `finalize_post_master_outputs`
is called by both the forward path and the recompile worker, whose own comment
says it is "matching the forward CLI path". What is *not* shared is aggregation.
This widens the seam to own it.

```
finalize_run(output_dir, …):
  1. select marker-authorized embedded measurement tables
  2. concat  →  master_measurements.parquet          (un-joined)
  3. join metadata + append metadata-only phantoms + apply post ops
  4. write  →  deliverables/measurements.{parquet,csv}
  5. persist pipeline.json, analysis outputs, per-feature splits
  6. backfill pht-metadata.parquet per store          (certified re-promote)
  7. publish aggregate proof → run proof
```

| Mode | Per-image work | Then |
|---|---|---|
| `full` | stage1→2→3, embed pure measurements | `finalize_run` |
| `measure` | re-measure from stores, embed | `finalize_run` |
| `recompile` | none | `finalize_run` |
| `process` | one layer, no measurement | skips 1–6 (`process_only_layer` already short-circuits the aggregate proof) |

**`recompile` becomes "call `finalize_run` again"** — not a parallel
implementation that must be kept in sync. This deletes recompile's separate
master-merge and collapses the `measurement_sources` vs `metadata_join_keys`
branch in `_run_post_master_steps`, which exists only because the two callers
arrive with differently-shaped inputs.

It also gives late-arriving metadata a first-class answer: a `metadata.csv` edit
changes `metadata_sha256`, invalidating `finalization_input_digest`, so the next
invocation re-runs `finalize_run` — re-joining the mirror and re-backfilling
every store — **without touching a single image's measurement**.

### 7.5 INVARIANT: aggregation inputs are embedded tables only

> **`finalize_run` step 1 selects exactly the marker-authorized embedded
> measurement tables. It never reads a prior master, chunk parquet,
> measurement shard, `analysis_full.parquet`, or `_dataset_aggregated.parquet`
> as an aggregation input.**

Those files are *outputs and intermediates of a previous finalization*, not
inputs to this one. Under a rolling input dataset, reusing any of them silently
omits images that arrived since the cache was built, or retains rows for an
image whose content changed and therefore has a new `work_id`.

Consequences, each of which is a test:

- `chunks/chunk_NNN.parquet`, `chunk_manifest.json`, `chunk_state.json`,
  `analysis_full.parquet` and `_dataset_aggregated.parquet` are **mid-run
  progress artifacts only**. They may feed the live dashboard. `finalize_run`
  ignores them.
- `finalize_run` invalidates them on success, so a later invocation cannot
  mistake them for inputs.
- Measurement shards are **per-invocation scratch, namespaced by
  `scheduler_epoch`**, so a prior run's shards can never be merged. Recompile
  already does this (`recompile/attempts/<attempt_id>/…`); the pattern
  generalises.
- Master is therefore a **pure function of the currently authorized embedded
  tables** — which is the derivability property this whole design is for.

---

## 8. Fan-out

Backfill has **no dependency on aggregation**: `pht-metadata.parquet` for one
image needs only that image's embedded table (for its join keys) and
`metadata.csv`. No master, no post ops, no global frame. A shard worker
therefore does one pass over its images and produces both outputs.

```
array task i ∈ [0, K):                          # aggregate + backfill
    for image in shard_i:
        read tables/measurements/table.parquet
        ├─ append → measurement_shards/<scheduler_epoch>/shard_i.parquet
        └─ project metadata rows → certified re-promote
                                   → tables/metadata/pht-metadata.parquet
                                   → record stages.backfilled

array task K (TASK_FINALIZE, dependent):        # reduce
    merge shard_*.parquet → master_measurements.parquet
    join + phantoms + post ops → measurements.{parquet,csv}
    pipeline.json, analysis outputs, per-feature splits
    publish aggregate proof → run proof
```

Recompile already has this shape — `TASK_MEASUREMENTS` (sharded by `shard_id`),
`TASK_OVERLAY`, `TASK_FINALIZE`. This promotes it to be universal.

Local `--njobs` uses the same decomposition with a process pool.

**Array-auxiliary-work rule.** `TASK_FINALIZE` stays a reserved trigger entry
*inside* the array task list, never a parallel sidecar job, and is counted when
sizing chunks against `MaxArraySize` / `MaxSubmitJobs`. This is the existing
`__PHENOTYPIC_CHECKPOINT__` / `__PHENOTYPIC_MANIFEST__` dispatch contract.

**Ordering and partial failure.** The aggregate proof asserts master + mirror;
the run proof asserts everything including backfill. A run that finishes
aggregation and dies mid-backfill has a valid aggregate proof and no run proof —
a resumable state where `RunState.completion` reports `incomplete` with only
backfill remaining.

## 9. `RunState`

```python
@dataclass(frozen=True)
class RunState:
    completion:  Literal["complete", "incomplete", "failed", "active"]
    identity:    RunIdentity
    images:      Mapping[str, ImageState]   # work_id -> stages + verdict
    advisories:  tuple[str, ...]
    diagnostics: RunDiagnostics             # counts; nothing branches on these
    depth:       Literal["shallow", "deep"]
    verified_at: datetime | None
```

`diagnostics` is a separate field on purpose: a predicate reaching into
`state.diagnostics` is visibly wrong in review.

| Caller | Depth |
|---|---|
| CLI finalize, before publishing proofs | `deep` |
| CLI resume, deriving the worklist | `deep`, cache-assisted |
| `OutputRoot.discover` (binding) | `deep` |
| `OutputMutationGuard` | `deep` |
| GUI snapshot poll (5–10 s) | `shallow` |
| SLURM observer tick (2 s) | `shallow` |

### 9.1 The verification cache

> **AMENDED (D-B): in-process, not on disk.** Audit S1 proposed process-level; this
> section escalated it without cause. See §0.

```
.phenotypic/verification_cache.json
```

Named as a cache deliberately, given how much confusion `manifest.json`
being-a-cache-but-treated-as-evidence has caused. Stores, per `work_id`: the
stat tuples (`size`, `mtime_ns`) of every deep-verified artifact, the resulting
verdict, and the `RunIdentity` it was verified under.

`shallow` re-stats those tuples. Any of these falls through to `deep`: entry
absent; stat tuple moved; recorded identity ≠ current identity (whole cache
discarded); file missing, unreadable or unparseable.

> **INVARIANT — the cache can only cause re-verification, never a wrong
> `complete`.** No code path yields a positive verdict from a cache entry
> alone; an entry only lets a *previously deep-verified* result stand while its
> stat tuples are unchanged. A stale, corrupt, truncated or forged cache
> degrades to today's behaviour, never past it.

Best-effort: on a read-only output, or if the write fails, `shallow` silently
degrades to `deep`. It must never turn an unwritable output into an error.
`clear_machine_state` deletes it. Concurrent writers are last-wins via
`atomic_write_json`, which is safe precisely because it is never authoritative.

`ctime_ns` is **dropped** from currency comparison (audit S3): it moves on
`chmod`, ownership change, hardlink, and `rsync -a`, all routine on a shared
filesystem, and `size` + `mtime_ns` already covers every write the contract makes.

### 9.2 Effect on rolling input

Adding 10 images to a 6,000-image run today re-derives the worklist by
validating 6,000 markers, each re-hashing its measurements parquet and overlay
PNG. With the cache, the 6,000 unchanged images cost one `stat()` each and the
10 arrivals are deep-verified. The same holds for the GUI's first poll after a
batch lands.

---

## 10. Spike gate — REQUIRED BEFORE IMPLEMENTATION

| Probe | Measures | Gates |
|---|---|---|
| **S-1 Hardlink re-promote** | inode count of a real `.ome.zarr` store; wall-clock of `os.link` fan-out + `promote_store` vs. full copy, **on GPFS** | §6.3. A bad result invalidates it and cascades into §6.4 and §7.4 |
| **S-2 Shard sizing** | task duration vs. shard count for combined aggregate+backfill at realistic N, against `MaxArraySize` / `MaxSubmitJobs` | The chunk-sizing formula, and whether backfill shares a task with aggregation |
| **S-3 Merge cost** | peak RSS and wall-clock merging K shard parquets vs. single-task concat at N≈6000 | Whether `TASK_FINALIZE` can hold the merge in memory or needs a streaming merge |
| **S-4 Backfill locality** | that per-image metadata projection needs no global state, across real metadata CSVs with fan-out, duplicate keys and metadata-only rows | The §8 DAG. If false, backfill moves after the merge and the finalizer becomes three roles |

**S-1 runs first.** These spikes drive the shipped code, so per CLAUDE.md they
do **not** belong in `docs/superpowers/logic_validation_scripts/` — that
directory's contract is that nothing in it imports `phenotypic`, keeping it an
independent witness. They live beside the implementation plan.

---

## 11. Consumer migration

| Consumer | Today | After |
|---|---|---|
| `_output_consistency.py` (617 lines) | 9 sources, 23 rules, 4 states | **deleted**; callers use `resolve_run_state(depth="shallow")` |
| `_snapshot_status.py` (101) | two fingerprints + full re-hash of 7 files per poll | ~30 lines mapping `completion` → badge |
| `OutputRoot` currency | `snapshot_is_current()` + `refresh_state_is_current()` | one shallow verification |
| `RunRegistry` claimability (248) | three conflict predicates | one `resolve_run_state` call |
| `RunRegistry` local exit (106) | 8-branch refusal tree | `resolve_run_state(deep)`; refusals become advisories |
| `_latest_event_states` | second event-log replay, different semantics | **deleted** |
| `_read_status_from_manifest` | infers status from manifest counts | **deleted** |
| SLURM observer | 2 × `_cli_completion` calls, `_all_stage3_markers_exist`, `_manifest_is_complete` | 1 × `resolve_run_state(shallow)`; stage probe reads `stages.stage3`. Tree, grace window, scheduler polling **untouched** |
| `_cli_completion.py` (1107) | readers + writers + legacy together | **split**: readers → `sdk_/_run_state.py`; writers stay CLI-side; legacy deleted |

### 11.1 Deleted — approximately 1,400 lines

- `classify_output_consistency`, `OutputCompletionEvidence`, `inspect_output_consistency`
- `RunRegistry`'s three claimability predicates, `_latest_event_states`,
  `_read_status_from_manifest`, `_manifest_is_complete`
- `_local_completion_evidence_conflict`'s 8-branch tree
- `stage2_done/` and `stage3_complete/` trees and their path helpers
- `MASTER_MEASUREMENTS_CSV`, `master_measurements_csv_path()`,
  `BundleLayout.master_csv`, `load_master_measurements()`
- `DashboardManifestKey.VERSION` — written at one site, read at zero; the
  manifest stops being evidence, so version-gating it is moot
- `_assert_worker_generation`'s `slurm_generation != attempt_id` check — one
  value passed twice, then asserted equal
- Every legacy read path: `_legacy_*` helpers and `resolve_*` fallbacks move
  **into** `--mode migrate` and out of the hot path
- `sdk_/monitor_slurm_jobs.py` (241, zero importers)
- `browse/_source_render.py`'s dead cache API
- Eight zero-caller resolvers in `_io_constants`

### 11.2 Folded in — inside files already being rewritten

Hand-joined `.phenotypic/aggregate_publication.json` in the GUI; the 17 shadow
state filenames moved into `_io_constants`; `DIR_PROGRESS` literals in
`phenotypicCLI.py`; the recompile `task_manifest.json` and `job_metadata.json`
naive writers made atomic.

Everything else from the audit is in [`DEFERRED.md`](DEFERRED.md).

---

## 12. Implementation phases

Sequenced so each phase is independently reviewable and leaves a working tree.

| Phase | Content | Depends on |
|---|---|---|
| **P0** | Spike gate (§10) | — |
| **P1** | `sdk_/_run_state.py`: `RunIdentity`, `RunState`, `resolve_run_state`, the cache and its invariant. No consumers moved. | P0 (S-1) |
| **P2** | Identity schema (§5): content-derived generation, `restart_epoch`, `scheduler_epoch` collapse | P1 |
| **P3** | Per-image record collapse (§6.1) + certified rewrite protocol (§6.4) | P1, P2 |
| **P4** | Embedded-table inversion + `finalize_run` + backfill (§7), local path only | P3 |
| **P5** | Fan-out (§8), SLURM + `--njobs` | P4, P0 (S-2, S-3) |
| **P6** | Consumer migration (§11) and deletions | P1–P5 |
| **P7** | `--mode migrate` conversion; refusal in every other mode | P1–P6 |

## 13. Error handling

Three rules, each of which is a test:

1. **The cache can only cause re-verification** (§9.1). No path where a cache
   entry alone yields `complete`.
2. **Uncertified artifact drift still raises.** The generalised receipt
   protocol retains `RuntimeError` for a marker-bound artifact that moved
   without a covering receipt.
3. **Unreadable ⇒ not complete, never complete.** Every parse failure degrades
   toward `incomplete`, matching the current conservative posture.

## 14. Testing

- **Mutation tests on the cache invariant** — corrupt, truncate, forge and
  stale it; assert the verdict never improves. Highest-value test in the change.
- **Partial-failure matrix** for three-phase completion: measured-not-aggregated,
  aggregated-not-backfilled, partially-backfilled. Each reports `incomplete` and
  resumes only the missing phase.
- **Rolling-input matrix**: batch added mid-run; between runs; with metadata
  arriving later; with an unready file present. Assert per-image proofs survive
  and only aggregate-level proofs invalidate.
- **Stale-worker test** for `restart_epoch`: a worker holding the pre-restart
  generation must not have its events counted.
- **Aggregation-input test** (§7.5): plant a stale chunk parquet, shard, and
  `_dataset_aggregated.parquet`, then assert the master ignores them and matches
  a concat of embedded tables exactly.
- **Hardlink fallback test**: force the copy path and assert equivalence.
- Every test above must be **proved able to fail** by reintroducing the bug it
  guards, per the project's test-integrity rule.

## 15. Residual risks

1. **The migrate step is the riskiest part of this design.** It rewrites every
   store in the tree and, unlike the rest of the change, cannot be rolled back
   by reverting code. It needs the receipt/rollback discipline the existing
   metadata migration has, plus its own dry-run mode.
2. **S-1 may invalidate §6.3.** The backfill design has a fallback shape
   (sibling file, or metadata in root `zarr.json`), but it is not designed here.
3. **One combined spec is hard to bisect** (D9). Mitigated by §12's phases, not
   eliminated.
4. **Backfill introduces a new partial state**: `deliverables/` can be complete
   and correct while `results/` is not yet self-describing.

