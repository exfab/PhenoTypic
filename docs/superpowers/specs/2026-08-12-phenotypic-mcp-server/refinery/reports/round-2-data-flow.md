# Round 2 — data-flow reviewer

Scope: `snapshots/round-1-spec.diff` (1004 lines). Flows traced end-to-end
against the real codebase at `/bigdata/iwheeldonlab/anguy344/PhenoTypic`
(branch `feat/mcp-server`). Analysis only; nothing edited.

IDs continue round-1 numbering: **FLOW-17 … FLOW-31**.

---

## 1. `§3.2 edit_previously_tried` — the trail it reads has no writer

**This is the headline finding. The mechanism USER-9 ordered reads an event type
no tool in the 26-tool catalog writes.**

Three consumers now read `pipeline.step` rows:

| Consumer | Reads | Spec |
|---|---|---|
| `exploration.steps` / `budget_note` | the step counter | `03:335` |
| `exploration.no_improvement_streak` + `tracked_signal` | per-step evidence deltas | `03:329-333`, `08:520-524` |
| `edit_previously_tried` (new this round) | the edit **and** its `decision` and `evidence` | `03:313-327` |

The row's shape (`08:526-536`):

```json
{"event":"pipeline.step","id":"pipelines/edge-v3.json.pht-pipe","step":3,
 "edit":{...},"evidence":{"num_objects":{"before":61,"after":88},...},
 "decision":"keep"}
```

**Nobody writes it.**

- §2.5's canonical event list is `pipeline.put`, `tune.start`,
  `tune.export_best`, `deploy.start` (`02-state-and-identity.md`, lineage
  block). `pipeline.step` is absent from §2.5 entirely.
- `pipeline_patch` (§3.2) is `W0` and cannot supply `evidence`: `before`/`after`
  for `num_objects` and `detect_mat.std` only exist once a probe has run, and
  the probe happens *after* the patch.
- `pipeline_probe` records something to lineage — "Per-image timing is recorded
  to lineage so `deploy_plan` estimates have a real basis" (`03:448`) — but that
  is timing, not a step row, and a probe does not know which patch preceded it.
- `decision: "keep" | "revert"` is the agent's judgment after reading evidence.
  **No tool in the catalog accepts it as an argument.** The two cut candidates
  did not carry it either.

So as written, `workspace_lineage` — kept explicitly at USER-8 *because* it is
"the only read path to the anti-repetition evidence" — reads an empty set, and
`edit_previously_tried` can never fire.

This subsumes the premise of **CONC-25** (which argued the read is an
unoffloaded 30 s-lock read and racy for siblings): both are true, but the row
being unwritten is upstream of both.

### FLOW-17 [**Critical**] — `pipeline.step` is read by three consumers and written by none · spec-change

**Decision I would make** (does not qualify for `deferred-to-2A` under USER-16 —
nothing needs observing):

1. **`pipeline_patch` writes the row at patch time**, carrying `edit` and
   `step` only, with `evidence: null`, `decision: "pending"`. Writing *before*
   the probe is what closes CONC-25's sibling race: the sibling mid-probe has
   already published its intent, so the second agent sees `decision:"pending"`
   and can back off rather than duplicate the spend. It also makes the write
   offloadable exactly as §2.5 requires (`asyncio.to_thread`), on a tool that
   is already mutating.
2. **`pipeline_probe` appends `evidence`** to the open row for that pipeline
   (it already writes a lineage row and already computes the numbers).
3. **`decision` is derived, not declared.** A step is `reverted` when a later
   patch on the same pipeline inverts it (insert↔remove of the same class at
   the resolved position, or `set_params` back to the prior value); otherwise
   it is `kept` once the next step is recorded. This needs no new tool and no
   new argument, and it makes the advisory's `hint` truthful rather than
   dependent on an agent volunteering a verdict.

Add `pipeline.step` to §2.5's event list either way — a reviewer reading §2.5
today would conclude the journal has four event types.

### FLOW-18 [Major] — the match key for "the same edit" is undefined and index-based · spec-change

`edits` are applied cumulatively and `insert_op.index` is clamped/relative
(`03`, index-semantics block). An `insert_op FocusEdgePhase at ops[1]` recorded
at step 3 is *not* the same edit as the same JSON at step 9 if a step in between
changed the slot length. `set_params {merge:true}` makes an edit a partial dict,
so equality over the literal argument matches almost nothing.

Needs a stated normalization: canonicalize `(kind, slot, class, sorted params)`
and resolve `index` to a **neighbour anchor** (the class names either side)
rather than an integer, or exclude the index from the key and say so. Also
state that the trail is per `pipeline_id`, so a `pipeline_put` fork starts with
an empty trail — the advisory will not fire for the fork of a pipeline whose
dead ends are all known.

---

## 2. `§9.3.0.2` multi-group — where `group_by` comes from, and where the
breakdown's data does not

### The namespace and the validator are both missing

`group_by: ["Metadata_Species", "Metadata_Medium"]` lives on the **profile**.
Tracing the value's origin:

- The profile artifact (§9.3) has `traits`, now `group_by` and `groups`. It
  carries **no metadata-CSV reference at all**. So at the moment `group_by` is
  written there is no file against which those column names could be checked.
- `experiment_profile_put` was **cut this round** (USER-8). The remaining
  `_get` is a read. So *no server code path validates `group_by` against
  anything, ever* — not at write, not at read, not at `campaign_put` (§9.3.5:
  `campaign_put` "stores the reference as a string without even checking the
  file resolves").
- The three CSVs §10.3 deliberately keeps apart are: `deploy_plan.metadata_csv`
  (joined onto the mirror), `MetadataGroupSubsetSelector.grouping_metadata`
  (stratification), `QCScorer.check.metadata` (expected counts). `group_by`'s
  columns must live in **the second**, but it is named on an artifact that
  references **none** of them.
- Namespace: the flat `Metadata_<Label>` form is a *canonicalization applied in
  memory on read* (`_cli/_metadata_join.py`, and CLAUDE.md's metadata-snapshot
  rule). A user's `plate_batches.csv` may literally say `Species`. So
  `"Metadata_Species"` may match no header in the file it is supposed to name.

Contrast with the one place the codebase *does* validate: `ExpectedVsDetectedCount`
raises `KeyError` at `__init__` if any `groupby` column is absent from the
resolved frame (`analysis/qc/_expected_vs_detected.py`, class docstring). That
is the standard `group_by` should meet and does not.

### FLOW-19 [**Critical**] — `group_by` is validated by nothing and names columns in a file the profile does not reference · spec-change · needs-user-input

**Answers USER-15's open question 1 (profile / subset / both): put `group_by` on
the SUBSET, not the profile.** The subset is where `grouping_metadata` already
lives, where a CSV column check is possible at `subset_generate` time (the CSV
is read there anyway), and where the images the grouping applies to are
enumerated. The profile keeps `groups` — the per-group *trait overrides* — keyed
by the same composite, because traits are human knowledge and belong with the
other traits. That split also keeps §9.3.5's "the server never acts on a trait"
true: the server acts on the subset's grouping, never on the profile's.

If `group_by` stays on the profile, then `subset_generate` must resolve the
profile, read `group_by`, and hard-fail with a named code when a column is
absent from `grouping_metadata` — otherwise a typo produces a subset stratified
on nothing, silently.

### FLOW-20 [Major] — the composite group key is undefined and the selector cannot express it · spec-change

§9.3.0.2 asserts the mechanism exists: "`MetadataGroupSubsetSelector` already
joins the CSV to images, and selecting one group is that join with a predicate."
Checked against §10.3's own parameter table:

| §9.3.0.2 needs | §10.3 provides |
|---|---|
| grouping by **N** columns | `group_key: str` — exactly one |
| **filter** to one group | `allocation` (`proportional`/`equal`) + `min_per_group` — stratify only, no predicate |
| key `"neurospora\|minimal"` | no composite key, no separator, no escaping, no ordering rule |

So this is new parameters on a class that does not exist yet (`phenotypic/subset/`
is absent from `src/` — it is P3 work), not a capability already present. Also
unspecified: value normalization (case, whitespace, NaN), what happens to an
image whose CSV row is missing a group column, and what `"a|b"` means if a
species name contains `|`.

**Recommend:** `group_by: list[str]`, `include_groups: list[list[str]] | null`
on the selector; the canonical key is the **ordered list of stringified values**
(a JSON array), rendered as a display string only. Reject any value containing
the display separator rather than escaping it.

### The per-group cost breakdown: confirmed to have no source, and CONC-24's fix is under-specified

CONC-24 is right that `QCScorer` returns one scalar per trial. Verified, and it
is worse than one scalar — group identity is destroyed at **three** levels:

1. `fold_expected_vs_detected_count` (`tune/score/_qc_scorer.py:78-82`) computes
   `QC_Count_Metric` **per group** and then `.mean()`s across groups inside a
   single image's score. Group-level detail exists for one line and is dropped.
2. `_Evaluator._aggregate` (`tune/_evaluation/_evaluator.py:398-417`)
   robust-aggregates each term's per-image cost list to
   `clamp01(median + λ·IQR)`. **Per-image costs are never persisted.**
3. Trial `user_attrs` carry `PHENO_TERMS` (term→scalar) and `PHENO_N_IMAGES`
   only (`tune/strategy/_optuna_support.py:81-91`). `campaign_status` reads the
   store; there is nothing per-group in it.

And the fix CONC-24 proposes ("belongs to the scorer, written into trial user
attrs at scoring time") is blocked by two facts the ledger has not recorded:

- **The scorer cannot know an image's group.** External CSV columns never reach
  `image.metadata` — §10.3 verified this by reproduction (`_resolve_groups` is a
  pure `image.metadata.get(group_key)` lookup; a fresh image carries only its
  five `MetadataImage_*` fields), and `join_metadata` runs on the *measurement*
  frame inside `finalize_post_master_outputs`, i.e. after a full run. The tune
  worker has no join. `TuningSpec` has no dataset or group field
  (`tune/_spec.py:162-171`, cited in §2.5/§8.3). So the scorer would need a new
  `grouping_metadata` + `group_by` parameter and its own join by `ImageName`.
- **Per-image-varying term keys break the aggregator.** If the scorer emitted
  `Count:neurospora|minimal` only for that group's images, `_aggregate` pads
  `_WORST_TERM` × `n_exceptions` into **every** term (`_evaluator.py:410-416`),
  so a group term seen on 6 of 24 images absorbs failures from the other 18.
  The term dict is assumed uniform across images.

### FLOW-21 [Major] — the per-group breakdown requires a tune-engine change that no prerequisite covers · spec-change · needs-user-input · alias CONC-24

**Answers USER-15's open question 3:** the breakdown belongs to the scorer, as
CONC-24 says — but it is **not free**, it is a new §7 prerequisite (scorer gains
a grouping join; evaluator gains group-aware aggregation that does not
cross-pad). Cost is comparable to P3, and it lands inside `phenotypic.tune`,
which §1.7's "no new science" boundary should be checked against.

The v1-sized alternative, if that is too much: **drop the breakdown and derive
the escalation signal from separate per-group campaigns.** A group-scoped subset
already gives one campaign per group (§9.3.0.2's own mechanism), and comparing
two campaigns' best costs answers "is one group failing?" without touching the
engine. It costs more compute and fits the existing contract exactly. Given
USER-15 called the signal what "the strategy rests on", this is a user call.

### FLOW-22 [Major] — flat-staging renaming silently corrupts the scorer's join · spec-change

An independent break in the same area, and it bites even with one group.

Chain: `flat/` must disambiguate `plateA/plate_001.tif` from
`plateB/plate_001.tif` (FLOW-4's collision, still open) → the staged filename
differs from the parent filename → the measurement frame's `ImageName` is
`filepath.stem` of the *staged* file (`_core/_image_parts/_image_io_handler.py:652-653`)
→ the human's layout CSV keys on the **parent** names → the group has no
metadata counterpart → `expected == 0` → `QC_Count_Metric = inf`
(`analysis/qc/_expected_vs_detected.py`, class docstring) → anchored to 0
goodness → cost `1.0` for every image of every trial.

**Failure signature:** the campaign completes, every arm's best cost is ~1.0,
`comparable: true`, no error anywhere. Days of fleet compute for a join typo.

Fix: state the flat naming function; have the staging builder emit a
`name_map.json` (parent-relative → staged stem) and require the server to stage
a rewritten copy of the scorer's layout CSV alongside; or forbid renaming and
refuse a subset whose parent-relative paths collide by stem, with a named code.
The same map is what a per-group breakdown (FLOW-21) would need to attribute a
staged image back to its group.

### FLOW-23 [Major] — `scope:"full"` on a group-scoped subset deploys the group's pipeline over every other group · spec-change

**Answers USER-15's open question 2, with a defect rather than a gap.** §5.4 and
§10.5 both define `full` as "targets `subset.parent`", and §10.5 adds "full
scope bypasses staging deliberately, running against `subset.parent` directly".
For a subset filtered to `neurospora|minimal`, the parent is the whole
experiment. So the documented behaviour is: tune per group, then promote one
group's winner across all groups' images. Nothing refuses it; the plan token
binds `(pipeline digest, parent digest, scope)` and all three are satisfied.

**Recommend:** the token binds the **group filter** too, and `scope:"full"`
means "the parent restricted by the subset's group predicate". A full-parent
deploy from a group-scoped subset requires an explicit
`group_scope:"parent"` and a named code (`group_scope_ambiguous`) if omitted,
because the two readings differ by an order of magnitude in spend and by
correctness in output.

### FLOW-24 [Major] — a multi-group experiment ends as N disjoint run trees with no single output · spec-change · needs-user-input

Follow the flow to its end. Per-group deploy = one `deploy_start` per group =
one `runs/<name>/` per group, each with its own `deliverables/measurements.*`.
Nothing in the spec joins them: `deploy_status` is per `run_id`,
`workspace_info.counts` counts runs, and there is no experiment-level surface.
`Metadata_Dataset` is derived per run from subdirectory names
(`_cli_directory_scanner`), so two runs can carry identical dataset labels for
different groups. The human gate (§10.5) also fires N times for one decision,
each with its own elicitation and its own 24 h token.

The user's experiment is one table. The spec's output is N.

**Recommend:** state the convention (`run_name = f"{base}--{group}"`), state
explicitly that joining the N mirrors is a documented post-step using existing
analysis tooling and **not** in v1, and record it in §10 rather than leaving a
reader to discover it after the compute. If a single output is required, that
is a real feature and should be scoped as such.

### One premise of §9.3.0.2 is refuted by the code

§9.3.0.2 opens: groups "can need different pipelines, different parameters, and
**different expected counts** — which is the scorer, not merely the pipeline."

The last clause is false as stated. `ExpectedVsDetectedCount` derives *expected*
from **row counts in the metadata frame per `groupby` key** (verified:
`QCScorer`'s doctest, 96 layout rows for image `p1` → expected 96). Per-image
and therefore per-group expected counts are already data-driven from one CSV,
with one scorer. What a multi-group experiment genuinely cannot express with one
scorer is a different **scorer class** or a different `fail_threshold` per
group. Worth correcting, because the sentence is the justification for
per-group descent and it overstates the need — the real drivers are
morphology-driven pipeline choice and enhancement, which the rest of the section
gets right.

---

## 3. `§10.5` the promotion fold — is the fold wrong, or its gate placement?

**The fold itself is right.** Verified against the spec's own arithmetic: a
campaign arm can mint a token only at `scope:"subset"` (§10.4), so `scope:"full"`
already had exactly one issuing path; the plan token already binds digests,
expires, and is single-use; `promotion_token` added no property. Collapsing two
tools into one is a genuine simplification and it removes a real class of
failure (two tokens with independent staleness).

**Its gate placement and its data sourcing are where it breaks.** CONC-22 owns
the class-and-ack half (declared `W0` while doing four things; token minted
alongside `pending_human_ack:true`). Three things it did not say:

### FLOW-25 [Major] — the fold makes the only human gate depend on five sources with no degraded contract · spec-change · alias CONC-22

`deploy_plan {scope:"full"}` must now assemble, in one response:

| Field | Source | If unavailable |
|---|---|---|
| `provenance.from_study`, `trial` | `lineage.jsonl` `tune.export_best` row — §2.5 lists this hop as **not recoverable from artifacts**, which is why the journal exists | **undefined** |
| `subset.score`, `gap` | the arm's study store — a killable subprocess open (§4.4) | undefined |
| `estimate.basis` | a prior subset deploy's per-image timing — see FLOW-26 | undefined |
| `extrapolation_check` | headers of **every** parent image (§10.6.1) | undefined |
| `full.digest_matches_parent` | directory digest helper that **does not exist** (§7 P3) | undefined |

§2.5's own fallback line — "worst case the journal is truncated and the
artifacts still stand alone" — was written when the journal fed provenance
displays. After the fold it feeds the human gate. A truncated journal now
degrades the *only* checkpoint before full-dataset spend, and there is no
`unknown` representation for any of these fields. This is precisely the argument
FLOW-11 made for `comparable` (a bare boolean cannot say "unknown"), applied to
five fields at once.

**Recommend:** every field in the promotion response carries a `basis`, and any
field that could not be derived is rendered as an explicit
`"unknown — <why>"` that the `ack_prompt` repeats. A gate that silently omits
the winner's provenance is worse than one that refuses.

### FLOW-26 [Major] — the measured estimate contradicts §5.3 and names a source that has no timing · spec-change · alias FLOW-10

Two bases for one tool:

- §5.3: `"basis":"probe of 2 images at 3.4 s/image"` — from a probe lineage row.
- §10.5: `"basis":"subset run: 3.4 s/image measured"` — from a subset **deploy
  run**, and this is the load-bearing claim ("the estimate is measured, not
  guessed… the strongest argument for subset-first development").

Checked what a subset run actually publishes. `DashboardManifestKey`
(`sdk_/_io_constants.py:1852-1881`) is `version, last_updated, execution_mode,
total_images, completed, failed, started, pending, success_rate, is_complete,
start_time, …` — **no duration, no per-image timing**. §5.5 names `manifest.json`
as the polling surface and names nothing else. So the only manifest-derivable
figure is `(last_updated − start_time) / completed`, which under a SLURM array
is wall-clock across N concurrent workers — **understating node-seconds by
roughly the array width**. A 100-wide array would turn 18.4 node-hours into
something near 0.2 and the human would approve a number off by two orders of
magnitude.

The real source exists but is unnamed anywhere in the spec:
`processing_events.log` carries per-image `started`/`completed` rows with
millisecond timestamps and the array task id
(`_cli/_cli_update_state.py:91,108,231`), so true per-image serial time is
derivable by pairing them per image.

**Recommend:** name `processing_events.log` as the timing source (or record
per-image timing into a `deploy.finish` lineage row at finalization), state the
pairing rule, and forbid the wall-clock derivation explicitly — it is the
obvious implementation and it is wrong.

### FLOW-27 [Major] — nothing recomputes `parent_digest`, so the window the gate exists to close stays open · spec-change

The lead's question — *what happens to a token whose parent set changed between
mint and `deploy_start`* — has no answer in the text.

§5.4 says a `scope:"full"` token "additionally binds `parent_digest`, so a parent
that gained images between the plan and the submission invalidates it". Binding
invalidates nothing on its own; something must recompute and compare. The
Validation row says "re-derive the digests from the **current request** and
compare" — but the request supplies `subset_id`, `scope`, `run_name`, `compute`.
Deriving a parent digest means walking `subset.parent` again, and no step is
assigned that walk. `deploy_start` is `W3`, so the cost is fine; the omission is
that it is unstated.

Worse, mint precedes the ack (CONC-22), so the digest bound into the token is a
**pre-approval** value. The exposure the gate is advertised to cover — images
landing while the human reads the prompt — is the one interval nothing covers
unless `deploy_start` re-walks.

**Recommend:** `deploy_start {scope:"full"}` re-derives `parent_digest` and
compares before submitting; `plan_stale` names `parent_digest` as the moved
field; and the response says how many images were added, since that is the fact
the human needs to decide whether to re-approve. Also: expiry is 24 h by
default, which is short for a gate whose whole point is that a human may not be
at the keyboard — worth stating whether an expired-but-unconsumed full-scope
token can be re-minted without repeating the elicitation (I would say no: the
digest window is exactly what expiry protects).

One flow-level note on the fold that is **not** a defect: because `deploy_plan`
is now a mint, the stale-then-replan loop produces orphaned
`.phenotypic-mcp/plans/<token>.json` records with `pending_human_ack:true` that
nothing consumes or reaps. FLOW-12 already flagged the absence of token GC; the
fold multiplies the rate at which such records accumulate.

---

## 4. `§1.6.1` the NFR table — does the design satisfy it?

No. The `W0` row contradicts itself and the design violates the strict reading
in at least five places.

The row says both: **"Returns in under one second"** and **"`W0` … does **not**
mean *is instant*, and the two must not be conflated."** Those cannot both bind.
The second clause was added to answer audit F3; it answers it by denying the
first clause in the same cell.

Under the strict reading, `W0` tools that plainly exceed one second:

| Tool | Why |
|---|---|
| `deploy_status {detail:"results"}` | parquet describe — §5.5 **itself** offloads it precisely because it is slow |
| `campaign_status` (no `since`) | one killable subprocess store-open **per arm** (§4.4) |
| `deploy_plan` (any scope) | reads every input image, per MAIN-MERGE's live defect |
| `deploy_plan {scope:"full"}` | + header sweep over 480 parent images (§10.6.1) + parent digest |
| `workspace_info {refresh}` | rehydrate; the spec's own example shows `rehydrate_ms: 184` on a small workspace |
| `subset_generate` | parent enumeration + CSV join over the whole parent |

On Lustre, 480 header opens is seconds, not milliseconds — and §10.6.1 declares
that sweep `W0`.

### FLOW-28 [Major] — §1.6.1's `W0` row is self-contradictory and the design violates its strict reading in ≥5 places · spec-change

**Recommend** splitting the two properties the row conflates, since the whole
point of the round-1 edit was to stop conflating them:

- `W0` keeps its real meaning: **takes no `LocalComputeSlot`**.
- A second, orthogonal attribute per tool: `inline` (bounded < 1 s, may run on
  the event loop) vs `offloaded` (no latency bound, **must** run in the
  executor).
- Tag every tool in §3 with both. Then F3's complaint is answered structurally
  rather than by a sentence, CONC-15's three blocking `W0` calls become
  `offloaded` by declaration, and CONC-22's `deploy_plan` objection reduces to
  "it is `W0`+`offloaded`, and separately it acquires the slot on the re-probe
  path" — which is the one part that really is a class error.

This does not qualify for `deferred-to-2A`: no measurement changes whether the
row can assert a bound it then denies.

---

## 5. `§2.3` root mandatory + must contain image data

The ruling (USER-11) is settled and I am not re-litigating it. What the round-1
edit left unfinished:

### FLOW-29 [Minor] — the layout tree omits the data the same section makes mandatory, and nothing validates the clause · spec-change

- The tree under §2.3 lists `pipelines/ tune/ profiles/ subsets/ campaigns/
  .phenotypic-mcp/ studies/ runs/`. **`data/` does not appear**, while
  `data/plates`, `data/tune_layout.csv` and `data/plate_batches.csv` are reached
  by every worked example in the spec. The tree is the normative layout; the
  reader who follows it builds a workspace that cannot start the flagship flow.
- "must contain the image data" has **no validator and no error code**. §6.2 has
  no `workspace_no_images`. Nothing checks at startup, and no tool checks later.
  The failure surfaces as `subset_generate` finding zero images, or as
  `SandboxRoot.resolve` raising `ValueError` on the user's absolute
  `/bigdata/...` path — an out-of-root rejection whose message will say nothing
  about the workspace root being wrong.
- **The data-on-another-mount case has a real cost, and it should be stated.**
  `SandboxRoot.resolve` resolves symlinks and rejects escapes
  (`_services/sandbox.py:91-120`, verified), so symlinking `data/` into the
  workspace does **not** work. The user's only options are to move the images
  under the root, or to make the data's parent the root — which puts `runs/`,
  the largest outputs in the system, on the data filesystem. On this cluster
  that is a `/rhome` vs `/bigdata` quota decision, not a detail.

**Recommend:** add `data/` to the tree marked user-owned; validate at startup
with a **bounded, depth-limited** scan for the first image-suffix hit and emit a
startup *warning* (not a refusal — a workspace may legitimately be created
before the data lands) with a named code; and add one sentence on the mount
consequence beside the existing `.git` warning.

---

## 6. The 26-tool cut — did anything leave a caller dangling?

Grepped all 11 spec docs and the plan for each cut symbol
(`promotion_request`, `promotion_approve`, `promotion_token`, `assay_put`,
`assay_get`, `pipeline_diff`, `campaign_get`, `catalog_measurements`,
`experiment_profile_put`, `assays/`, `.assay.json`).

**The spec is clean.** Every remaining hit is inside the §3.0 cut table or the
§10.5 "an earlier draft had…" paragraph — deliberate retrospective prose, not a
live caller. The §9.5 skill tool-lists, the §8.3 recovery procedure, the §10.5
flow diagram and the §10.7/§9.6 resolved-questions lists were all updated.

One live dangling reference, in the plan:

### FLOW-30 [Minor] — a plan task still scopes work for a cut tool

`plans/2026-08-14-phenotypic-mcp-server/phase-1b-engine-prerequisites.md:422`:

> Produces: `derive_columns(pipeline) -> list[str]` — what Phase 2A's
> `produces_columns` and `catalog_measurements` both call.

The interface itself survives (`produces_columns` still needs it); only the
second consumer is gone. Also note **SIMP-14's second half is still live**: the
NN-stack force-import that SIMP-14 attached to `catalog_measurements` belongs to
Task 10c, which the cut did not touch — the ledger records it "undecided", and
cutting the tool did not decide it.

(`plans/.../README.md:162` mentions `campaign_approve` vs `promotion_approve` as
a sibling-confusion example inside finding F1. That is a historical record of an
audit finding, correctly left alone. Worth noting that the cut *resolves* half
of F1's stated confusion pressure.)

---

## 7. `§3.0`'s annotations paragraph — one data-flow point only

CONC-27 owns this (a list, not a rule; `deploy_plan` breaks it). I reached the
same conclusion independently and will not restate it. The one thing to add from
the flow side: **the fold turned `deploy_plan` into a writer.** It now creates
`.phenotypic-mcp/plans/<token>.json` and records the ack on it. §2.6's
concurrency table still has a row for "promotion tokens" (CONC-22 noted the
stale name) and no row for a plan record that is written by one tool, CAS'd by
another (`consumed_by`), and mutated in between by a human answer. That is three
writers on one record, which is more than the naming question.

---

## 8. OME-Zarr cross-check — do my flows survive the move?

Judging only the MCP spec's data-flow assumptions, against
`2026-08-18-ome-zarr-image-store/design.md` @ the worktree (docs-only, moving).

**Survive unchanged:**

- `deploy_status` progress. Confirmed by reading §5.5: it reads
  `manifest.json` + the `RunRegistry` record; `DashboardManifestKey` carries
  counts and `is_complete`, nothing format-specific. The brief's "§5.5 reads
  per-image HDF as the unit of progress" is **not what §5.5 says** — this
  independently re-confirms FLOW-13/GEN-12.
- §10.6.1's header sweep and §10.5's `parent_digest`: both read **input**
  images. OME-Zarr replaces the per-image **output** store. Untouched.
- §10.3.1 subset staging (P6): symlinks/copies **input** files. The brief's
  "a zarr store is a directory" concern does not reach it, for the same reason
  the sidecar concern did not. Refuted, like FLOW-13.
- FLOW-17/18 (lineage trail), FLOW-19/20 (grouping), FLOW-22 (staging join),
  FLOW-25/26/27 (the promotion fold), FLOW-28 (NFR): all format-independent.

**Newly exposed by my flows:**

### FLOW-31 [Minor] — two of my flows touch the store boundary · spec-change

1. **§2.3's tree hardcodes the HDF shape**: `runs/<name>/results/<dataset>/{hdf,measurements}/`.
   Cosmetic as prose, but it is the layout the server enumerates and the tree a
   reader implements against. It changes under OME-Zarr and should be written
   as "engine-owned, resolved through `sdk_/_io_constants` helpers" rather than
   spelled out — which is what §2.3's own next paragraph already demands ("the
   server never hand-joins a filename").
2. **FLOW-26's timing source is inside the store's write path.**
   `processing_events.log` is written by `_cli_update_state`, and the OME-Zarr
   design reworks the staged commit protocol (rename-promote, Stage-2 sidecar
   becomes a consumable marker, §3.2–3.6). Whether per-image `started`/
   `completed` pairs keep their current semantics across a `.part`-then-promote
   commit is worth one question upstream *before* the MCP spec names that file
   as the estimate's basis. It is the only place my flows depend on a file the
   OME-Zarr change actually touches.

The real coupling remains where round 1 put it: the mode list, the migration
guard (GEN-4, FLOW-14), and `--mode migrate`'s inversion of `--input` (GEN-12).
Nothing I traced adds a sixth.

---

## Concerns

| ID | Sev | Concern | Tags |
|---|---|---|---|
| **FLOW-17** | **Critical** | `pipeline.step` is read by three consumers (`exploration.steps`, `no_improvement_streak`, `edit_previously_tried`) and **written by no tool**; §2.5's event list omits it; `decision` is not an argument anywhere. USER-9's mechanism reads an empty set | spec-change |
| **FLOW-19** | **Critical** | `group_by` is validated by nothing (`experiment_profile_put` was cut) and names columns in a CSV the profile does not reference; `Metadata_*` is an in-memory canonicalization, so the names may match no header | spec-change · needs-user-input |
| FLOW-18 | Major | `edit_previously_tried`'s match key is undefined and index-based, so it is unstable under the cumulative patches it exists to dedup; forks start with an empty trail | spec-change |
| FLOW-20 | Major | The composite group key (`"neurospora\|minimal"`) has no encoding rule, and `MetadataGroupSubsetSelector` (§10.3) has one `group_key: str` and no filter predicate — §9.3.0.2's "mechanism already exists" is false | spec-change |
| FLOW-21 | Major | The per-group breakdown needs a tune-engine change no prerequisite covers: per-image costs are discarded (`_evaluator.py:398-417`), the fold already `.mean()`s groups (`_qc_scorer.py:78-82`), and the scorer cannot learn an image's group (no CSV join in the worker). Heterogeneous term keys also break `_aggregate`'s padding | spec-change · needs-user-input · alias CONC-24 |
| FLOW-22 | Major | Flat-staging renaming breaks the scorer's layout join by `ImageName` (= file stem), producing `QC_Count_Metric = inf` → cost 1.0 on every trial with no error | spec-change |
| FLOW-23 | Major | `scope:"full"` on a group-scoped subset deploys that group's pipeline over the whole experiment — the documented behaviour, unrefused | spec-change · needs-user-input |
| FLOW-24 | Major | Multi-group ends as N disjoint run trees with no experiment-level output, colliding `Metadata_Dataset` labels, and N human gates | spec-change · needs-user-input |
| FLOW-25 | Major | The fold makes the only human gate depend on five sources (journal, study store, prior-run timing, header sweep, parent digest) with no `unknown` representation for any | spec-change · alias CONC-22 |
| FLOW-26 | Major | "subset run: 3.4 s/image measured" contradicts §5.3's probe basis and names a source with no timing; the manifest-derivable figure understates node-hours by the array width. Real source (`processing_events.log`) is unnamed | spec-change · alias FLOW-10 |
| FLOW-27 | Major | Nothing recomputes `parent_digest` at `deploy_start`; the mint precedes the ack, so the interval the gate advertises is the one interval uncovered | spec-change |
| FLOW-28 | Major | §1.6.1's `W0` row asserts "<1 s" and denies it in the same cell; ≥5 `W0` tools exceed it. Split `W0` (no slot) from `inline`/`offloaded` | spec-change |
| FLOW-29 | Minor | §2.3's tree omits the `data/` it makes mandatory; nothing validates "contains image data"; symlinked data is rejected by `SandboxRoot.resolve`, so the other-mount user pays a quota decision | spec-change |
| FLOW-30 | Minor | `phase-1b:422` still scopes `derive_columns` for the cut `catalog_measurements`; SIMP-14's Task-10c NN-import half remains undecided | — |
| FLOW-31 | Minor | §2.3 spells the HDF-shaped `results/<dataset>/{hdf,measurements}/`; FLOW-26's timing source sits inside the write path OME-Zarr reworks | spec-change |

**`deferred-to-2A`:** none of the above qualifies under USER-16. Every one still
needs a decision after any experiment returns either result; each carries the
decision I would make.

**VERDICT: REVISE**
