# Run state: what is tracked, and how to read it

This page is the reference for **a run's machine state** — what PhenoTypic
writes down about the progress of a run, what it computes instead, and which
call answers each question.

Read it before adding any counter, flag, marker or cached count under
`.phenotypic/`. The organising principle of this subsystem is **move state that
is tracked to state that is checked**: a value that can be recomputed from
artifacts on disk must be recomputed, because a stored copy can disagree with
the thing it describes and nothing will fail to say so.

```{admonition} Keep this page current
:class: important

This page is the reference other documents point at, including
`src/phenotypic/_cli/CLAUDE.md`. **If you add, remove or change a tracked
state, a proof, or a consumer, update this page in the same change** — see
the **Changing the state model** section below for the checklist. A
reference that is updated later is a reference nobody can trust in between.
```

---

## Reading state: one call

```python
from phenotypic.sdk_ import resolve_run_state

state = resolve_run_state(output_dir, depth="deep")
state.completion            # "complete" | "incomplete" | "failed" | "active"
state.diagnostics.verified  # int
state.images["<work_id>"].verdict   # "verified" | "unverified" | "failed"
```

`resolve_run_state(output_dir, *, depth="deep")` is the single answer to *"is
this run done?"*. Everything below exists to serve it.

**It never raises.** Any unreadable, absent or foreign directory degrades toward
`incomplete` rather than erroring, so callers do not need a `try` block. It
accepts a directory this package has never written to.

### `RunState`

| Field | Type | Meaning |
|---|---|---|
| `completion` | `Literal["complete", "incomplete", "failed", "active"]` | The verdict |
| `identity` | `RunIdentity` | Run-level identity tokens and digests |
| `images` | `Mapping[str, ImageState]` | Per-image state, keyed by `work_id` |
| `advisories` | `tuple[str, ...]` | Human-readable notes; **never branch on these** |
| `diagnostics` | `RunDiagnostics` | `accepted` / `verified` / `failed` counts |
| `depth` | `Literal["shallow", "deep"]` | The depth **actually performed** |
| `verified_at` | `datetime \| None` | When verification ran |

`RunIdentity` carries `processing_generation`, `restart_epoch`,
`scheduler_epoch`, `owner_generation`, `inventory_digest`,
`scientific_config_digest`, `finalization_input_digest`.

`ImageState` carries `work_id`, `dataset`, `image_stem`, `stages`, `verdict`,
`reason`.

`stages` is an **open map** — `stage1` / `stage2` / `stage3` / `measured`
today, more later. Nothing enumerates its keys. Ask `"stage3" in
state.stages`, which is what makes a future stage additive rather than a schema
break.

### Verdict precedence

Total and ordered, first match wins:

```
complete  >  active  >  failed  >  incomplete
```

There is no `contradictory`. `complete` outranks `active` because a run proof
covers the *current* inventory. `active` outranks `failed` so a failure from a
previous attempt cannot mask an attempt currently retrying it.

### Choosing a depth

| | `deep` | `shallow` |
|---|---|---|
| What it does | Re-verifies every declared artifact's content | Re-stats recorded `(size, mtime_ns)` tuples |
| Cost | O(N) hashing | O(N) `stat()` |
| Use for | Publishing, finalizing, anything that writes | Listing runs, polling, GUI status |

`shallow` falls through to a deep pass for any image absent from the cache,
moved, minted under a different identity, or unreadable — so `RunState.depth`
reports what actually happened, and a cold `shallow` call returns `"deep"`.

```{admonition} INV-VERDICT
:class: warning

A cached entry can only ever license **skipping** a re-verification the caller
already performed. It can never produce a positive verdict on its own, and the
run-level proofs are re-verified on every call regardless of depth.
```

---

## (a) Tracked state — written down, and irreducibly so

**Four things are written down.** The last column is load-bearing: if a
proposed fifth entry cannot fill it, the value is derived and belongs in
section **(c) Derived, and by what**.

| # | State | Path helper | Writer | Why it cannot be derived |
|---|---|---|---|---|
| 1 | Accepted inventory | `processing_state_path` → `config.work_ids` | `create_initial_state` (`_cli_state_management.py:206`), resume | A directory listing answers *what is here*; this answers *what this run accepted*. They differ the moment an input arrives mid-run. |
| 2 | Terminal failures | `terminal_failures_jsonl_path` | `append_terminal_failure` (`_cli_failure_tracker.py:344`) | **A failure leaves no artifact.** Absence of output is indistinguishable from not-yet-started. |
| 3 | Liveness & ownership | `slurm_lifecycle_path`, `gui_launch_owner_path`, the lifecycle ledger | CLI submitter / the GUI | Facts about external systems and live processes. A worker killed by the scheduler leaves no trace of having run. |
| 4 | `restart_epoch` | `restart_epoch_path` | `bump_restart_epoch` (`_cli_identity.py:386`) | A content-derived generation cannot distinguish *deliberately fresh attempt* from *same configuration again*. |

**Entry 2 is the one to read twice.** `completion == "failed"` has exactly one
source: an image verdict of `failed`, which comes only from this journal. Every
other fact about a run can be re-derived by looking at the tree; this cannot.

### What survives `--restart`

`clear_machine_state` deletes everything under `.phenotypic/` except
`_PRESERVED_ON_RESTART`, which has exactly three members:

```python
frozenset({TERMINAL_FAILURES_JSONL, RESTART_EPOCH_JSON, DIR_LEGACY_V2})
```

`restart_epoch.json` is preserved because **a counter that resets on the
operation it fences is not a fence**. `legacy-v2/` is preserved because a
restart is not a revert.

---

## (b) Content proofs — evidence, not tracked state

Three digest manifests over artifacts that already exist. They record nothing
that is not recoverable by re-reading what they describe; what they add is
*that it was checked, and under which identity*.

| Proof | Path helper | Covers |
|---|---|---|
| Per-image record | `image_record_path(output_dir, dataset, stem)` | One image's artifacts and stages |
| Aggregate proof | `aggregate_publication_marker_path` | The aggregated outputs |
| Run proof | `run_completion_marker_path` | The run against its accepted inventory |

**Publication order, and it is never reordered:**

```
store root zarr.json  →  per-image record  →  aggregate proof  →  run proof
```

Each step certifies only what the previous one has already made durable, so an
interruption always leaves a tree that is **behind** rather than one that is
**wrong**.

### Why the aggregate proof refuses

`aggregate_proof_refusal(output_dir)` returns why the aggregate proof is not
current, or `None` if it is. It is deliberately separate from the predicate:
folding the reason into the return value would tempt a caller into branching on
*which* cause and treating some as recoverable. They are not.

```{note}
`aggregate_proof_refusal` is **not yet exported** from `phenotypic.sdk_`, and
its user-facing consumer is not wired — the message a user sees still comes
from `OutputRoot.discover`. This is the capability, not its wiring.
```

---

## (c) Derived, and by what

One row per fact, naming the function. This is the table that stops the next
contributor writing a counter.

| Fact | Derived from | By |
|---|---|---|
| *Is this run done?* | (a) 1–4 plus the proofs in (b) | `resolve_run_state(output_dir, depth=...)` |
| `processing_generation` | `sha256(pipeline_sha256 ‖ per_image_config_digest ‖ restart_epoch)` | `derive_processing_generation` (`_cli_identity.py:148`) |
| `work_id` | schema version, dataset, input-relative path, input sha256, pipeline fingerprint, per-image config digest, mode | `work_id_for_image` (`_cli_failure_tracker.py:310`) |
| per-dataset completed / failed counts | the per-image records | `RunState.diagnostics` — **and nothing branches on these** |
| the master | the record-authorized embedded tables, each projected onto its own descriptor's `measurement_columns`, minus any store the projection excludes — and nothing else | `finalize_run` → `project_embedded_measurement_table` |
| *how many verified images the published master does not carry* | the aggregate proof's `source_image_count` vs. the live verified count | `resolve_run_state` → `RunState.advisories` (count clause) |
| *which store a re-finalization will exclude again* | each verified image's record (does it declare a `measurements` artifact?) and its store root (does it declare a projectable `measurement_columns`?) | `resolve_run_state` → `RunState.advisories` (naming clause) |

**An excluded store makes a fully verified run read `incomplete`.** The
projection (P7 Task 4) leaves out a store whose table it cannot project safely —
no measurement descriptor, or same-label rows that disagree — and the aggregate
proof certifies only the stores the master actually carries. `resolve_run_state`
then finds a verified set larger than the proof's, and reports `incomplete`.

**`resolve_run_state` says so, in two advisories, and neither is tracked state.**
The verdict is unchanged — the run *is* incomplete — but the reason is now
readable off the tree instead of surviving only as a `logger.warning` that is
gone by the time anyone looks, and re-running finalization reaches the same
verdict *and* re-derives the same advisory.

- **The count clause** reads `source_image_count` from the aggregate proof and
  compares it against the live verified count. It is the **backstop**: it fires
  for all four of the projection's exclusions, including the two the naming
  clause cannot see. It also fires for the wholly benign case of an image
  verified after the master was published, which a rolling input reaches on its
  own between finalizations — so it **reports the gap and refuses to diagnose
  it**, naming both causes and the fact that re-running finalization resolves
  one and reaches the other again. Do not reword it into an exclusion alert.
  The benign case is the common one, and this page's own argument for gating
  the schema advisory is that an advisory which is always on teaches people to
  ignore the one that matters.
- **The naming clause** names each verified image whose *record* authorizes a
  `measurements` artifact while its *store root* declares no projectable
  `measurement_columns`. That conjunction is the inconsistency: a record
  promising the finalizer a table the store does not declare. This is the
  clause that accuses, and it only accuses where the store itself is
  demonstrably inconsistent.

```{admonition} The naming clause is a SUBSET — two of the four causes
:class: important

`project_embedded_measurement_table` excludes a store for four reasons. The
naming clause covers the two that are visible in the store's root document: no
`tables.measurements` descriptor, and a descriptor whose `measurement_columns`
is not a list of strings.

The other two are properties of the Parquet payload — a metadata-joined table
that repeats rows while declaring no `target.column`, and one whose same-label
rows disagree across the projected columns — and **they are not named, by
design**. Answering them means opening a per-image Parquet from a reader the
GUI polls every few seconds, which would make reading a run's state cost what
finalizing it costs. On those two, the count clause is the only signal in the
run state and the finalization log names the store.

So: a shortfall with no store named does **not** mean nothing was excluded. It
means nothing was excluded *for a reason this reader can see*.
```

Both clauses are **depth-invariant**. The naming clause's two facts are
recorded into each image's `measured` stage during verification, out of the
same single read of the store root the metadata-snapshot advisory already pays
for, so they ride the verification cache and a warm `shallow` pass emits them
without opening a store. The count clause reads one small sidecar, O(1) in
images.

Because those facts ride the cache, **`VERIFICATION_CACHE_VERSION` went 1 → 2
in the same change**. A version-1 entry has valid stat tuples and neither fact,
so a warm shallow pass would have reused it and emitted no advisory — a
diagnostic silently switched off by a cache. That constant's comment already
required a bump when the *rules* of deep verification change rather than the
JSON shape; this is the first bump to invoke it.

```{admonition} The proof records the digest, not the set
:class: note

`source_set_digest` is `canonical_digest(sorted(work_ids))` and
`source_image_count` its arity. **The list is not recoverable from the proof**,
so "which images did the master leave out?" cannot be read back out of it and
has to be re-derived from the tree — which is what the naming clause does, and
why it answers for two of the four causes rather than all of them. Do not
"fix" this by writing the list into the proof: that is a fifth tracked state
wearing a proof's clothes, and the derivation above already answers the
question the operator is asking.
```

`processing_generation` folds **only configuration values** — a pipeline hash, a
per-image config digest, and a restart epoch. No paths, no timestamps, no
measurements. Two runs of the same configuration on different nodes mint
identical digests, which is why tests compare it **exactly**. Tolerance belongs
to measurement outputs, never to configuration identity.

**Deleted, and must not come back:**
`processing_state.datasets.{completed,failed,started}` as live state;
`manifest.json` as evidence; the event log as a completion source.

---

## (d) Retained, and read by nothing

Two directories survive on disk without belonging to any table above, and both
need saying **because** they look like counter-examples to (a)'s "four".

| Artifact | Written by | Consulted by | When it may be deleted |
|---|---|---|---|
| `.phenotypic/legacy-v2/` | `--mode migrate`, moving `image_complete/` and `stage3_complete/` aside | **Nothing.** It exists only so `migrate --revert` is a rename back. | Once the tree has been reprocessed, or the operator accepts migration is final. |
| `.phenotypic/verification_cache.json` | `persist_states` after a deep pass | Only `resolve_run_state(depth="shallow")`, and only to **skip re-hashing** an unchanged artifact | Any time. A missing cache costs one deep pass. |

**The test for whether a future artifact belongs here or is a fifth tracked
state: nothing branches on it, and no verdict is derived from it.** Delete
either of these and every answer the system gives is identical — only slower
(the cache) or one option poorer (the revert).

---

## The one that fits none of the above: `migration_manifest.json`

`.phenotypic/migration_manifest.json` is written by `--mode migrate`, is
**never unlinked** by anything in `src/`, and **is branched on** — so it fails
(d)'s test — yet it is not one of the four in (a).

It is worth understanding rather than tidying away. It arrived as migrate's
*work manifest* (tasks, offsets, merkle proofs), and its **existence** was then
re-purposed as the answer to *"was this tree migrated?"*, which gates the
continuation refusal in `_output_was_migrated` (`phenotypicCLI.py:714`).

That call is keyed on the manifest deliberately, and the docstring records why
the two more obvious signals are both wrong:

- **`PROVENANCE_MIGRATED` on the per-image records says `forward` on exactly
  these trees.** See **Provenance has two writers** below.
  Anyone reaching for provenance here will find it, and find it wrong.
- **`work_ids` keyed by bare stem is the *symptom*** of the admitted-set
  pollution the refusal exists to explain, so keying on it would be circular
  and would break the moment that defect is repaired.

It is self-limiting in the right direction: `clear_machine_state` deletes it
along with the rest of `.phenotypic/`, so the `--restart` the refusal
recommends ends the tree's migrated status rather than leaving a permanent
special case.

```{admonition} Known defect
:class: caution

This path is **hand-joined** in two places (`phenotypicCLI.py:747`,
`_cli_migrate.py:1767`) rather than resolved through an `sdk_` helper, against
this repository's own rule. If you touch either site, add a path helper in
`sdk_/_io_constants.py` and route both through it.
```

---

## Provenance has two writers

`record_rejection` (`sdk_/_image_record.py:207`) skips the `work_id` comparison
for `PROVENANCE_MIGRATED` records, because a migrated tree's identity cannot be
re-derived and is marked unavailable rather than fabricated. **Absent means
`"forward"`**, so a writer that forgets the field produces a fenced record
rather than an accepted one.

**But not every record written during a migration carries `"migrated"`**, and
the difference is the code path, not the mode:

| Path | Publisher | Provenance written |
|---|---|---|
| Converting an existing marker into a record | `_cli_migrate_state.py:350` | `migrated` |
| Minting a record from outputs (MIG-11) | `_cli_migrate_state.py:1080` via `publish_image_record` | `migrated` |
| Migrating the image artifact itself | `_cli_migrate_image.py:589` via `publish_image_success` | **`forward`** |

`publish_image_success` has **no `provenance` parameter**, so records it writes
take `publish_image_record`'s `PROVENANCE_FORWARD` default. This is why
`_cli_migrate_state.py`'s mint path calls `publish_image_record` directly and
its docstring says `publish_image_success` is *"the wrong entry despite being
the forward one"*.

**Do not use record provenance to ask whether a tree was migrated.** Use the
migration manifest.

---

## Known consumers

Every site that reads run state, what it reads, and what it decides. Add a row
here when you add a consumer.

| Consumer | Reads | Depth | Decides |
|---|---|---|---|
| `phenotypicCLI.py:2590` | `diagnostics.verified`, `diagnostics.accepted` | `deep` | Whether startup must refresh publication |
| `phenotypicCLI.py:3063` | `diagnostics.verified` | `deep` | Whether to finalize measurements |
| `phenotypicCLI.py:714` (`_output_was_migrated`) | `migration_manifest.json` | — | Whether to refuse continuation and name `--restart` |
| `_cli/_cli_checkpoint_handler.py:310` | `diagnostics.verified` | `deep` | Raise vs. close a terminal-incomplete lifecycle |
| `_cli/_cli_completion.py` | the three proofs | — | Publishes them; the writer side |
| `gui/shell/_runs_registry.py:1082` | `.completion` | `shallow` | Status of a historical output with no GUI launch generation |
| `gui/shell/_runs_registry.py:1460` | `.completion`, `.diagnostics` | `shallow` | The run's row in the runs list |
| `gui/results_viewer/_output_root.py:364` | full `RunState` | `deep` | Binds run state at discovery; gates opening via `core_readable` |
| `gui/_snapshot_status.py:121` | `.completion` | `shallow` | Live status for a bound snapshot |
| `gui/run_console/_slurm_observer.py:1334` | `.completion` | `shallow` | Whether a SLURM run is reconciling or done |

### Consumers that deliberately ask something else

Two sites look like they should call `resolve_run_state(...).completion` and
must not. Both carry a comment saying so; **do not "simplify" them**.

| Site | Asks instead | Why |
|---|---|---|
| `gui/shell/_runs_registry.py:699` | `_all_accepted_images_succeeded` | Asks *"have the accepted images succeeded?"*. `.completion` already requires the run proof, which would make the branch below it dead code. |
| `_cli/_dashboard/_manifest_builder.py:729` | `valid_aggregate_snapshot` | Runs during recompile, **before any run proof exists**. `.completion` asks whether a valid run proof covers the inventory, which is a different question. |

### `core_readable` is not `completion`

`core_readable(layout)` (`gui/results_viewer/_output_root.py:140`) decides
whether the results viewer can open a bundle. It is **not** derivable from
`completion`: a curated-but-incomplete run is core-readable, and a completion
test that lists the acceptable verdicts gets it wrong. Ask `core_readable`
directly rather than approximating it.

### Status vocabulary is hand-copied

`_runs_registry._RUN_STATUSES` is the vocabulary; other GUI sites hand-copy it.
One copy (`gui/results_viewer/_qc_tab/_rebuild.py:157`) silently fell a member
behind when `incomplete` was added, and an unrecognised status there does not
degrade — it **blocks a QC rebuild on a perfectly valid status**.

`test_every_status_vocabulary_in_the_gui_agrees_with_the_registry` walks the
package AST and is what catches this. **Do not add a member to the registry
without running it.**

---

## Where readers and writers live

```{admonition} INV-LAYER
:class: warning

`phenotypic.sdk_` may **never** import `phenotypic._cli`.
```

- **Readers live in `sdk_`.** `resolve_run_state`, the state types, the path
  helpers, `record_rejection`.
- **Writers stay in `_cli`.** `publish_image_record`, `publish_image_success`,
  `append_terminal_failure`, `bump_restart_epoch`, the migrate planners.

A GUI or notebook consumer imports from `phenotypic.sdk_` only. If you find
yourself wanting `_cli` from `sdk_`, the reader half belongs in `sdk_` and the
writer half does not move.

**Always resolve paths through the `sdk_` helpers.** Never hand-join names —
the one place that does is the known defect noted above.

---

## Changing the state model

A fifth tracked value appearing is a design regression, and the burden of proof
is on the addition. Before adding one:

1. **Try to derive it.** Can it be recomputed from artifacts plus (a) 1–4? If
   yes, it belongs in section **(c)** as a function, not on disk.
2. **Fill in the last column of (a).** State why it cannot be derived. If you
   cannot, it is derived.
3. **Apply (d)'s test.** If nothing branches on it and no verdict comes from
   it, it is a retained artifact, not tracked state — document it in (d).
4. **Decide its `--restart` fate.** Add it to `_PRESERVED_ON_RESTART` only if a
   restart must not clear it, and say why in that set's docstring.
5. **Name its reader.** Tracked state with no reader is a leak; tracked state
   read from more than one place needs an `sdk_` accessor, not a second
   hand-join.

Then, **in the same change**:

- Add a row to the relevant table on this page — (a), (b), (c) or (d).
- Add a row to the **Known consumers** table for every site that reads it.
- If it changes the status vocabulary, run
  `test_every_status_vocabulary_in_the_gui_agrees_with_the_registry`.
- If it adds a path, add an `sdk_/_io_constants.py` helper and use it
  everywhere.

Removing or renaming a state is the same checklist run backwards, plus deleting
its consumer rows. **A row that describes something no longer true is worse
than a missing row**, because it carries the authority of documentation while
being wrong.
