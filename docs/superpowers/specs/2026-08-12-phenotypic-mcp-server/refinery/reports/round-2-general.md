# Round 2 — general reviewer

**Scope:** `refinery/snapshots/round-1-spec.diff` (1004 lines) + the plan
(`plans/2026-08-14-phenotypic-mcp-server/`). Lens: spec↔plan traceability,
feasibility against the shipped `_services` tier, failure modes. IDs continue
round-1 numbering.

**Method note.** Every codebase claim below was read out of the working tree, not
inferred. Line numbers are from `feat/mcp-server` as checked out.

---

## Headline

The round-1 edits are directionally right — the cut is real, the fold's
*motivation* is sound, USER-11 closes a hole nobody had named. But the
application has three classes of defect:

1. **Half-applied rulings.** USER-8 folded promotion's *tools* but dropped
   SIMP-1's *placement* recommendation, and the placement is the half that made
   the fold coherent. This is the root of CONC-22.
2. **A blind rename.** `assay` → `experiment_profile` was applied to prose but
   not to the machine-readable contracts, leaving `workspace_info` emitting
   `counts.assays` while the paragraph beside it tells the agent to read
   `counts.profiles`.
3. **Deletions that took normative content with them.** Cutting
   `catalog_measurements` deleted the only place the spec specified
   `header_scheme()` dispatch — and three sections still cite §3.1 for it.

And one finding the diff *created*: USER-11's "the root must contain the image
data" puts the plate tree inside the directory `RunRegistry.rehydrate_from_sandbox`
walks on every boot — a walk whose own docstring carries a `TODO(perf)` about
exactly this case. That is a §1.6.1 violation introduced by a round-1 fix.

---

## Answers to the five places the orchestrator asked for an independent look

### 2. The §10.5 promotion fold — is the *fold* wrong, or only the placement?

**The fold of the CONTENT is right. The fold of the two CALLS into one is
wrong. CONC-22's incoherence is not incidental — it is forced.**

The two-tool design had one property the fold destroys: request and approve were
separate calls, so **the token was minted by the second one**. That is a clean
state machine — assemble (read, no token) → human → approve (write, mint). The
fold collapses assemble + human + mint into a single handler, and a single
handler *cannot* both hand the human a decision to read and mint only after they
answer, unless it blocks. So the example returning `plan_token` **and**
`pending_human_ack: true` together is not a drafting slip; it is the only thing a
one-call design can do.

**This is a half-applied ruling.** SIMP-1's actual recommendation, in the ledger,
was two moves: fold the tools *and* "Elicitation (USER-5) moves to
`deploy_start {full}` — better placement, at the point of spend." USER-8 adopted
the first and not the second. §10.5's new prose even argues for the second —
"the elicitation fires here, not two calls earlier — at the point of spend rather
than at the start of a sequence the agent could still abandon" — and then places
it on `deploy_plan`, which is *not* the point of spend. `deploy_start` is.

**Recommended shape (no new mechanism, and it dissolves three open concerns):**

| Tool | Class | Does |
|---|---|---|
| `deploy_plan {scope:"full"}` | `W0`, genuinely read-only | Assembles the decision: winner provenance, subset score, gap, coverage warnings, §10.6.1's header sweep, measured estimate. Returns a **`plan_id`**, not a spendable token. Mints nothing, prompts nobody. |
| `deploy_start {scope:"full", plan_id, human_response}` | `W3` | Raises the elicitation, then submits. |

Why this is strictly better and not merely different:

- **§1.6.1 stops being violated.** An unbounded human wait inside a `W0` handler
  contradicts "under one second, never blocks the event loop" head-on. The same
  wait inside `W3` is explicitly sanctioned — `W3` has "no latency requirement".
- **The ack stops being a second mutable field.** There is no window between
  "token exists" and "human answered", so `deploy_start` has nothing to race
  (CONC-22's second half) and §2.6 needs no new row.
- **`readOnlyHint: true` on `deploy_plan` becomes true again**, which is what
  `MCP-INTERFACE-AUDIT.md:794` recommends and what CONC-27 showed the fold had
  falsified. The host confirmation dialog lands on `deploy_start`, where a
  confirmation dialog belongs.
- **Provenance improves.** The ack lands in lineage next to the `run_id` rather
  than on a token that may expire unspent.

The one thing lost is an ack recorded before submit. That was never worth much:
a recorded approval for a run that never happened is not evidence of anything.

Tag: **spec-change**. It does not re-litigate USER-8 (the cut stands, the tool
count stays 26); it applies the half of SIMP-1 that USER-8 left on the table.

### 3. §1.6.1 — does the rest of the spec satisfy it?

**No, and the round-1 edits are what broke it.** Two independent violations, one
of them created this round (GEN-15 below), one of them CONC-22's (above). §1.6.1
itself is well-drafted; the design has drifted out from under it.

### 4. §3.0's annotations paragraph

CONC-27 is right that it is a list, not a rule. Two things it did not say — see
GEN-17. The more serious is that the paragraph, as written, **breaks the
unattended-campaign UX the whole design exists for**.

### 5. Did anything cut leave a caller dangling?

Yes — six sites, in both the spec and the plan. GEN-13, GEN-16, GEN-20.

### 1. §9.3.0.2 multi-group

Two orphans against Phase 1b, which is planned-but-not-started and therefore
cheap to fix now and expensive to fix in 2B. GEN-18, GEN-19.

---

## GEN-6 — CONFIRMED STILL LIVE, and the round-1 edit made it worse

CONC-28 cites GEN-6 as unresolved. Confirming, with new mechanism.

**The CAS field is unchanged.** `02-state-and-identity.md:2.6` still reads
"`approve` and any amendment CAS on `status`". §10.4 still permits an in-envelope
amendment that replaces a failed arm — `status` is `approved` on both sides, so
the CAS passes for a write that changes the arm set.

**What round 1 added.** §8.3's `campaign_start` now says it "**snapshots the
campaign it launched** rather than re-reading it during fan-out", and separately
that each arm's resolved `study_id` is "**written back into `campaign.json`**".
Those two rules together convert GEN-6 from a race into a near-certainty:

- the snapshot is taken once, at fan-out start;
- arms run for hours;
- every `study_id` write-back therefore writes an **hours-old arm set** with a
  CAS on a field that has not changed.

So an amendment applied at hour 2 is silently reverted by arm 4's write-back at
hour 5 — and the reverted document is the one `campaign_status {detail:"artifact"}`
returns as **the session-recovery entry point**. An agent recovering from
compaction is then handed the pre-amendment arm list, including the failed arm
that was replaced, and no record of the replacement's `study_id`. That study is
running, registered in `RunRegistry`, and unreachable from the campaign.

**The fix is in-house and one field.** `RunRegistry.compare_and_set(run_id,
generation, …)` (`src/phenotypic/_services/runs.py:420-446`) already implements
exactly the needed idiom: reject on a stale fence, not on a status value.
`campaign.json` should carry the same thing — a `revision` (or a digest of the
document) bumped by *every* writer: approve, amendment, and each `study_id`
write-back. §2.6's row becomes "CAS on `revision`". `campaign_start`'s snapshot
then becomes safe by construction: it holds the revision it read, and a
write-back whose revision moved re-reads, re-applies its single field, and
retries rather than clobbering.

This also gives CONC-28 its answer for free: `campaign_approve`'s elicitation
captures the revision it displayed, and an answer arriving against a moved
revision is refused with a named code rather than approving a summary that went
stale while the human read it.

Status: **open · spec-change**. Not deferrable under USER-16 — the choice of CAS
field is a decision, and it would still need making whatever a live server showed.

---

## New concerns

### GEN-13 [Major · spec-change] — the rename is prose-only; the machine contracts still say `assay`

`03-tool-catalog.md`, all in the **response and argument contracts**, not prose:

- `:509` `"workflow":{"assay":"profiles/plates.experiment.json", …}` — key still `assay`
- `:511` `"counts":{… "assays":1, …}`
- `:554` `workspace_list {kind}` enum still `"assays"`

and the paragraph at `:522-524`, which the diff *did* edit, now instructs the
agent to branch on **`counts.profiles == 0`** — a key the response two paragraphs
above does not emit. The spec contradicts itself inside one section, on the tool
§3.3 calls "the natural first call".

`workspace_list {kind:"assays"}` is additionally now unimplementable as written:
§2.3's tree has no `assays/` directory.

Fix: `workflow.experiment_profile`, `counts.profiles`, `kind:"profiles"`. Also
sweep `10-subsets-and-promotion.md:20,78` and `08-workflow-and-campaigns.md:13`
(prose, cosmetic) and `09-responsibilities-and-skills.md:449` ("A newer skill's
assay read by an older server").

### GEN-14 [Major · spec-change] — cutting `catalog_measurements` deleted the `header_scheme()` rule, and three sections still cite it

The diff removed §3.1's `catalog_measurements` block (round-0 lines 848-914).
That block was the **only** place the spec specified how columns are derived —
the `static`/`texture`/`metric_qualified` dispatch, the `TEXTURE.get_headers()`
`TypeError`, the 130-column figure, and the "do not model this on the README
generator" warning.

Three live cross-references now point at content that is gone:

| Site | Text | Now |
|---|---|---|
| `03-tool-catalog.md:253` | "`produces_columns` uses the `header_scheme()` dispatch of §3.1" | §3.1 no longer describes it |
| `05-deploy-and-slurm.md:427` | "§3.1 already establishes that one measurer … emits **130 columns**" — load-bearing: it is the justification for `deploy_status {detail:"results"}`'s column bound | the establishing text is deleted |
| `07-prerequisites.md:335` | "`TEXTURE.get_headers()` raises `TypeError` without a `scale`; see §3.1" | same |

The rule survives **only in the plan** (`phase-1b:387-419`, Task 12). That is
backwards: the plan implements the spec, not the other way round. `produces_columns`
is a Phase 2A deliverable written from §3.2, and §3.2 forwards to a section that
no longer answers.

Fix: keep the deleted block, relocated under §3.2's `produces_columns` (it was
always describing that derivation; it merely lived under the tool that has been
cut). Roughly 25 lines survive; the `catalog_measurements` argument table goes.

Also at `07-prerequisites.md:333-334`, a sed artifact from the same edit:
"column derivation for `produces_columns` and `produces_columns`".

### GEN-15 [Major · spec-change] — USER-11 puts the image tree inside the directory the boot walk scans, violating §1.6.1

Created by this round's fix to FLOW-3.

§2.2 now requires the workspace root to **contain the image data**. §2.4 has the
server call `RunRegistry.rehydrate_from_sandbox` at boot, and §3.3's
`workspace_info` reports `rehydrate_ms: 184` as a normal figure.

Verified against the shipped implementation
(`src/phenotypic/_services/runs.py:734-880`):

- `rehydrate_from_sandbox(sandbox, max_depth=3)` drives `_discover_output_dirs`,
  a synchronous DFS calling `SandboxRoot.list_children` on every directory to
  depth 3, and `classify(child)` on **every** child directory.
- Its docstring (`:819-822`) carries the authors' own note:
  `TODO(perf): on a sandbox with thousands of plate folders this walk runs
  synchronously on shell boot before the HTTP listener accepts requests.`

Before USER-11, "thousands of plate folders" was outside the root and the TODO
did not bind. After USER-11 it is the normal case: `<root>/data/plates/plateA/`
is depth 3, so `list_children` runs once per plate folder. `rehydrate_ms: 184`
was measured against a workspace with no data in it and is no longer a
representative number.

The GUI's stated remedy in that TODO — defer to a background thread, show a
"Scanning…" badge — **has no MCP analogue**: a tool call must return an answer,
and §1.6.1 gives `W0` one second.

Three options, in preference order:

1. **Scope the walk.** Rehydration only ever wants `studies/` and `runs/` — both
   server-invented, both at known relative paths. Walk those two subtrees, not
   the root. This is a `_services` change of a few lines and removes the problem
   rather than bounding it. It also makes the walk correct: today an output
   directory a user happens to leave inside `data/` is registered as a run.
2. Cache the scan and refresh on `workspace_info {refresh}` only (which CONC-15
   already flags as an unclassed blocking `W0`).
3. Report `rehydrate_ms` as a warning above a threshold and accept the cost.

Option 1 is the recommendation, and it belongs in **Phase 1b** — it is engine
work on already-shipped code, and Phase 2A's `workspace_info` is written against
whatever this does.

### GEN-16 [Major · spec-change] — USER-11 forbids symlinked data, which is the normal HPCC layout

Not a re-litigation of USER-11 (the root contains the data — settled). This is
its unstated mechanical consequence.

`SandboxRoot.resolve()` (`src/phenotypic/_services/sandbox.py:93-121`) does
`joined.resolve(strict=False)` — **following symlinks** — then
`_contains(resolved)` against a root itself resolved `strict=True` (`:86`). The
diff cites this behaviour approvingly as the reason the data must be inside.

Consequence on this cluster: `<workspace>/data → /bigdata/…/UCR_029` resolves
outside the root and **every path under it raises `ValueError`**. A home-directory
workspace with the dataset symlinked from `/bigdata` is the standard UCR HPCC
pattern (home quota is small; data lives on `/bigdata` or `/rhome`). So the
flagship flow fails on the most likely real layout, and it fails late — at the
first `pipeline_probe`, with a bare `ValueError` for which §6.2 has **no error
code**.

Two things are needed regardless of how the user rules:

- **A startup check with a named error.** At boot, if a child of the root is a
  symlink whose target escapes, refuse with a code and a stated fix
  ("`--workspace` must be the real parent of the images") rather than surfacing
  it as an opaque failure eight tool calls later. §6.2 gains a row.
- **A worked example that is actually runnable on this cluster** — i.e. the
  workspace root *is* `/bigdata/…/experiment/`, with `pipelines/`, `runs/`, and
  `studies/` created beside the data. §2.3's tree should show `data/` explicitly;
  today it does not appear at all, and the sentence under it ("Only `pipelines/`,
  `tune/`, `profiles/`, `subsets/`, `campaigns/`, `studies/`, `runs/` are this
  server's invention") reads as an exhaustive layout.

**`needs-user-input`** on one point only: USER-11 rejected a `data_roots`
allowlist as a second containment concept. A single-symlink carve-out for the
data directory is a narrower thing than that allowlist and would make the normal
HPCC layout work. Worth an explicit yes/no rather than an inference.

*Credit where it is due:* USER-11 also **closes** a hole nobody named. Staging
symlinks (§10.3.1) are enumerated by `list_children`, which filters symlinks
whose targets escape the root (`sandbox.py:173-175`, `_symlink_target_in_root`).
Had the data stayed outside, every staged subset symlink would have been silently
skipped. With the data inside, they resolve in-root and are yielded.

### GEN-17 [Major · spec-change] — §3.0's annotation paragraph contradicts itself and breaks unattended execution

Aliases CONC-27 (deploy_plan / pipeline_probe); this is disjoint content.

**(a) It contradicts itself in four lines.** "**Every tool carries MCP
annotations.** `title`, plus `readOnlyHint` and `destructiveHint`" — then, three
lines down, "annotating the read tools and **leaving** `deploy_start`,
`campaign_start`, `tune_start` and `workspace_cancel` **unannotated**".

**(b) The "leave them unannotated" strategy breaks the unattended campaign.**
`MCP-INTERFACE-AUDIT.md:714-721` records the host semantics: hosts default to
worst case, so *not* declaring `destructiveHint` on a non-read-only tool means
the host assumes `true` and **raises a confirmation dialog**. The audit's own
conclusion is the opposite of §3.0's: "the value here is mostly in declaring
`destructiveHint: **false**` on the twelve write tools that create".

`tune_start` and `campaign_start` create new studies under fresh names — §2.2
forbids auto-suffixing and a collision is an error, so neither destroys anything.
Leaving them unannotated puts a host confirmation in front of **every arm launch**
— defeating §8.1 Phase 2 ("the agent executes it across parallel subagents
without you in the loop") and §10.4 ("What runs unattended"), which is the point
of the campaign artifact.

`workspace_cancel` is the genuine `destructiveHint: true`, and it should be
*declared*, not left to a default.

**(c) It names idempotency and then never names the field.** The paragraph
argues at length that `*_put` is not idempotent and `pipeline_patch`
"emphatically is not — its edits are cumulative, so the annotation is what stops
a host retrying into a corrupted pipeline". `idempotentHint` is the field that
does that, and the paragraph's own opening enumerates only `title`,
`readOnlyHint`, `destructiveHint`. As written, the mechanism it relies on is not
in the list it declares.

Fix: replace the list with a table — one row per tool, four columns
(`title`, `readOnlyHint`, `destructiveHint`, `idempotentHint`), every cell
filled, no defaults relied on. `MCP-INTERFACE-AUDIT.md:729-758` already has
most of this table; it needs updating for the 26-tool cut (GEN-20) and lifting
into §3.0.

### GEN-18 [Major · spec-change] — §9.3.0.2's "mechanism" does not match the selector Phase 1b is about to build

§9.3.0.2 asserts: "`MetadataGroupSubsetSelector` already joins the CSV to images,
and selecting one group is that join with a predicate."

Phase 1b Task 14 (`phase-1b-engine-prerequisites.md:569-628`), which builds that
class and has **not started**, specifies:

```python
MetadataGroupSubsetSelector(n=4, seed=0, grouping_metadata=str(csv),
                            group_key="Metadata_Batch", allocation="equal")
```

Two gaps against §9.3.0.2:

1. **`group_key` is singular.** The profile's `group_by` is `list[str]`
   (`["Metadata_Species", "Metadata_Medium"]`) and the `groups` map is keyed by a
   composite (`"neurospora|minimal"`). Task 14 has no multi-column form.
2. **There is no predicate.** Task 14 stratifies (`allocation="equal"`); nothing
   filters *to* one group. §9.3.0.2's per-group campaign, per-group subset, and
   per-group deploy all rest on that filter.

Neither is expensive — a `group_by: list[str]` field and a `group_filter:
str | None` — but they are **engine work in Phase 1b**, and §9.3.0.2 was written
as though they already exist. If Task 14 ships as drafted, 2B's per-group
campaign is blocked on reopening a closed phase.

Additional, smaller: the composite key `"neurospora|minimal"` has **no stated
escaping rule**. A metadata value containing `|` collides two groups, and the
resulting wrong trait overrides are applied silently — the same failure shape
§9.3.3 calls "the worst failure in the system". Specify a separator that cannot
appear (or a JSON-array key).

Fix: amend Task 14's interface block and its `test_metadata_group_*` tests.

### GEN-19 [Major · spec-change] — the per-group cost breakdown is an engine change with no task, and the mechanism it needs already exists

§9.3.0.2 makes the breakdown load-bearing: "Without it the strategy is
unactionable." CONC-24 answers *where* it belongs (the scorer) and flags the
polling-economy conflict. Two things it did not say, both verified:

**(a) It is a Phase 1b-shaped change with no Phase 1b task.** Phase 1b's task
list (Tasks 10–18) has nothing touching `tune/score/`. Every other engine
prerequisite the spec depends on got a task; this one was added in round 1 and
did not.

**(b) The mechanism is already shipped, which makes the change small.**

- `Scorer.score_image` returns a **term dict**, not a scalar
  (`src/phenotypic/tune/score/_scorer.py:145`), and
  `finalize(terms) -> float | dict[str, float]` (`:180`).
- `set_trial_user_attrs` already writes `PHENO_TERMS` — the full term dict — into
  each trial's Optuna `user_attrs`
  (`src/phenotypic/tune/strategy/_optuna_support.py:59-82`).
- `_optuna_store.py:199-246` already round-trips `user_attrs` through the store
  on both the write and read sides.

So a per-group breakdown is a **term-per-group** extension of a contract that
exists end-to-end, not new plumbing. What is genuinely new is that `QCScorer`
must learn the grouping columns — a new field, which by CLAUDE.md's conventions
is a pydantic annotated field plus a `field_validator`, and which must be
distinguished from `QCScorer.check.metadata` (Task 14's note warns that three
different CSVs appear in this design and passing the wrong one produces a
meaningless objective rather than an error).

Fix: add a Phase 1b task. Suggested shape — `QCScorer` gains
`breakdown_by: list[str] = []`; when non-empty, `_score_terms` emits
`Count|<group>` terms alongside `Count`; `campaign_status` reads them from
`user_attrs` on the best trial. No `campaign_status` polling change is needed for
the *aggregate* path, which is CONC-24's concern — the breakdown rides the
artifact read the `detail:"artifact"` mode already performs.

### GEN-20 [Minor · spec-change] — the plan is stale against the 26-tool cut, and one register actively instructs the wrong annotation

Traceability in the other direction. The cut landed in the spec; the plan still
describes the 32-tool surface.

| Site | Says | Should say |
|---|---|---|
| `README.md:223-227` (phase map) | `2A catalog(3) + pipeline(5) + workspace(4)`; `2B assay(2) + subset(3) + tune(5) + campaign(5)`; `2C deploy(3) + promotion(2) + …` | `2A catalog(2)+pipeline(4)+workspace(4)`; `2B experiment_profile(1)+subset(3)+tune(5)+campaign(4)`; `2C deploy(3) + 4 skills + setup` |
| `README.md:162` (F1) | "for any of the **32**" | 26 |
| `README.md:149` (D3) | `phenotypic-assay-triage` | `phenotypic-experiment-triage` |
| `phase-1b:422` | "`produces_columns` and `catalog_measurements` both call" | one consumer |
| `MCP-INTERFACE-AUDIT.md:729-758` | annotation table includes `catalog_measurements`, `pipeline_diff`, `assay_put`, `assay_get`, `campaign_get`, `promotion_request`, `promotion_approve` | 26 rows |
| **`MCP-INTERFACE-AUDIT.md:794`** | "**`deploy_plan` is the strongest `readOnlyHint: true` on the surface**" | **actively wrong after the fold** — this is the register Phase 2C is written from, and it recommends the exact annotation CONC-27 identified as putting the sole full-dataset human gate behind a tool a host may auto-approve |

The historical finding registers (`MCPB-EVALUATION.md`, `review-findings.md`) are
dated artifacts and may reasonably stay frozen. `README.md`, `phase-1b`, and
`MCP-INTERFACE-AUDIT.md`'s annotation table are **live inputs to Phase 2** and are
not in that category. `:794` in particular should be struck now rather than
carried into 2C.

**A phase-boundary consequence, worth calling out separately:** the diff moves
elicitation onto `campaign_approve` (§8.2), which the phase map puts in **2B**,
while D6 schedules elicitation for **2C**. Either 2B ships `campaign_approve`
with the fabrication-only fallback and 2C retrofits it, or elicitation moves
forward to 2B. The spec's "required-unless-elicited" signature decision was taken
precisely so this is not a breaking change — but the plan should say which.

### GEN-21 [Minor · spec-change] — `edit_previously_tried` has no defined match predicate

USER-9's ruling says the advisory fires "when an edit matches one already
recorded for that pipeline". §3.2's new block inherits the phrasing and never
defines *matches*. This is not re-litigating USER-9 — it is the predicate the
ruling assumes.

The two readings fail in opposite directions:

- **Exact edit-object equality** — the advisory almost never fires. §8.7's loop
  is parameter search; `sigma=1.4` after `sigma=1.2` is a different object, and
  those are the retries an agent most needs warning about.
- **Operation-class equality** — it fires on every parameter sweep, and an
  advisory that fires every call is noise the agent learns to skip, which costs
  USER-9's benefit entirely.

The workable middle is a normalized key: `(edit_kind, op_class, position)` for
structural edits, and `(edit_kind, op_class, position, param_name)` — **not the
value** — for parameter edits, with the prior attempt's *value* carried in the
`hint` so the agent sees "you tried `sigma` at ops[0] twice, at 1.2 → reverted
and 1.4 → reverted" rather than "this exact edit was tried". That preserves the
anti-repetition signal across a sweep without firing on every step.

Also unstated: the journal is scanned for *this pipeline*, but `pipeline_patch`
mutates in place, so "this pipeline" spans every digest the id has ever had.
Recording the digest on each journal step and reporting it in the hint is what
makes "the surrounding pipeline may have changed" — the diff's own justification
for advisory-not-refusal — checkable rather than assumed.

Cost is not a concern: §8.7 caps patches at 12, so the scan is bounded by
construction. (This weakens CONC-25's cost half; its *race* half stands.)

### GEN-22 [Minor] — §3.0's naming grammar does not describe the catalog it introduces

`<group>_<verb>[_<object>]` is stated in the same paragraph as the eight groups.
`catalog_operations` has no verb; `catalog_operation_detail` has no verb;
`experiment_profile_get` has a two-word group, so the grammar cannot be parsed
unambiguously (`experiment` + `profile_get`?). Minor, but §3.0 is the section
that claims to bind every tool, and a rule with three exceptions among 26 tools
is a description, not a rule. Either state it as a preference or drop it.

### GEN-23 [Minor · spec-change] — §2.6's token row names a concept this round deleted

Noted inside CONC-22; recording separately because the fix is independent of how
CONC-22 resolves. `02-state-and-identity.md` §2.6 row reads "Plan / **promotion**
tokens", and §2.3's tree at `:136` still comments `plans/<token>.json  # §5.4
plan + promotion token records`. There is one token kind now.

### GEN-24 [Advisory] — §8.3's `campaign_status` is documented before it is defined

`campaign_status {detail:"artifact"}` gets its own `###` heading at
`08-workflow-and-campaigns.md:305`, **before** `campaign_status (W0)` at `:328`.
A reader meets the detail mode before the tool. Also `§8.3` opens "Four tools."
above five `###` headings. Cosmetic, but §8.3 is the section Phase 2B is written
from.

---

## OME-Zarr cross-check

Read `2026-08-18-ome-zarr-image-store/design.md` @ the worktree. Judging
compatibility only, per the brief.

**FLOW-13 and GEN-12 are confirmed a third time, and the round-1 cut narrowed the
coupling further.** Grepping the live spec: exactly **one** HDF reference
survives — `02-state-and-identity.md:158`, the `results/<dataset>/{hdf,measurements}/`
line in the workspace tree. One line, cosmetic, no logic reads it.

Two of the brief's five claimed exposures are now positively refuted by the
OME-Zarr design's own text:

- **The sidecar survives.** §3.4 keeps a per-image consumable Stage-2 token at
  `.phenotypic/progress/stage2_done/<dataset>/<stem>.json`, and §3.5's correction
  note keeps the **raw** `.npy` under `.phenotypic/progress/stage2_raw/` because
  Stage 3 re-promotes over the store's objmap and cannot use it as replay input.
  The three-stage resume contract is preserved, not dissolved. §3.4 states
  explicitly: "NGFF metadata never carries resume state."
- **P6 staging is untouched.** OME-Zarr replaces the per-image *output* store.
  Staging symlinks *inputs* (TIFF/PNG), which the design does not change.

**What genuinely couples, after the cut:**

1. **The mode list — and the cut makes it unrecoverable in-server.** Removing
   `mode`/`layer` from `deploy_*` is the right call for coupling, and it dissolves
   the four-vs-three enum problem. But `--mode migrate` becomes the **only** path
   from legacy `.h5` results to a store, and after the cut the MCP surface has no
   way to invoke it — the agent must drop to a shell. For a fresh agent workspace
   this never bites (all output is born as zarr). It bites on
   `deploy_start {resume: true}` against a pre-migration run directory, which is
   GEN-4's mid-run refusal reached by a second route with no in-server remedy.
   **Recommendation: accept it, and state it** — one sentence in §5.4 saying the
   server does not perform migrations and naming the CLI command, so the agent
   surfaces an actionable message instead of an opaque `sys.exit(1)`.
2. **The Python floor.** OME-Zarr locked decision #3 raises `requires-python` to
   `>=3.11, <3.13`; the repo is at `>=3.10, <3.13` (`pyproject.toml:25`). FLOW-14
   flagged `<3.13` against `fastmcp` 3.x and it remains unchecked. The *floor*
   moving to 3.11 is harmless for `fastmcp`; the *ceiling* staying at 3.13 is the
   open question, and it is now pinned by a second design. Worth resolving once,
   for both.
3. **`--durable-writes`** (locked decision #13) reaches `_services/argv.py`, which
   Phase 1a already promoted and merged. Already flagged by FLOW-14; still open.

**Nothing in the round-1 diff creates a new OME-Zarr conflict**, and the `mode`
cut removed the largest one. GEN-12's "2C waits for OME-Zarr over-blocks" stands.

---

## Round-1 resolutions verified

| Ruling | Applied? | Note |
|---|---|---|
| USER-8 — 32→26 | **partially** | Spec cut is clean and internally consistent on counts. **Plan not updated** (GEN-20). Deleted content lost (GEN-14). |
| USER-9 — `edit_previously_tried` | **yes, underspecified** | Predicate undefined (GEN-21); CONC-25's race stands |
| USER-10 — local batch suspends probing | **yes** | §1.5 states it plainly. CONC-19 correctly finds the 1–2-arm sentence contradicts it |
| USER-11 — root contains data | **yes, with two consequences unhandled** | GEN-15 (boot walk), GEN-16 (symlinks). §2.3's tree not updated to show `data/` |
| USER-12 — D1a/D5/D6 into the spec | **yes** | D1a §1.4 ✓, D6 §8.2/§10.5 ✓, D5 §3.0 ✓ but as a list, not a rule (CONC-27, GEN-17) |
| USER-13 — rename | **prose only** | GEN-13 |
| USER-15 — multi-group | **spec yes, plan no** | GEN-18, GEN-19 |
| SIMP-1 — promotion fold | **half** | tools folded, placement dropped — the root of CONC-22 |
| GEN-1 (round 1) | **resolved** | rulings are now in the spec |
| GEN-6 (round 1) | **STILL LIVE, worse** | see above |

**Round-1 concerns of mine not addressed and still open:** GEN-2 (`to_argv`
cannot emit deploy args — partially dissolved by the `mode`/`layer` cut, but
`--restart`/`--overwrite`/`--gpu-slurm` remain), GEN-3 (three mechanisms under
`phenotypic.gui`), GEN-4, GEN-5 (WAL), GEN-7 (dissolved by the promotion cut —
§10.6.1's header sweep now lives inside `deploy_plan`, so the *escalation* moved
rather than went away; it is now CONC-22's first item), GEN-8, GEN-9, GEN-10,
GEN-11.

---

## Deferral assessment (USER-16)

**None of GEN-13..24 qualifies.** Each would still need a decision after a live
server returned either result: a CAS field, an annotation table, a match
predicate, a selector signature, where a walk is rooted. I am not proposing any
`deferred-to-2A`.

---

## Concerns — severity-tagged

| ID | Sev | Concern | Tags |
|---|---|---|---|
| **GEN-6** | **Major** | *(round 1, re-confirmed)* `campaign.json` CAS on `status` cannot detect an arm-set change; round-1's snapshot + `study_id` write-back rules make the revert near-certain rather than racy. Fix: CAS on a `revision` fence, the idiom `RunRegistry.compare_and_set` already implements | spec-change |
| **GEN-14** | Major | Cutting `catalog_measurements` deleted the `header_scheme()` derivation rule; §3.2, §5.5 and §7 P3 still cite §3.1 for it. Rule now survives only in the plan | spec-change |
| **GEN-15** | Major | USER-11 puts the plate tree inside the boot walk's scope; `_discover_output_dirs` carries a `TODO(perf)` about exactly this. Violates §1.6.1. Fix: scope rehydration to `studies/` + `runs/` | spec-change |
| **GEN-16** | Major | `SandboxRoot.resolve()` follows symlinks then rejects escapes, so a symlinked `data/` — the normal HPCC layout — fails at first probe with no §6.2 code | spec-change · needs-user-input |
| **GEN-17** | Major | §3.0's annotation paragraph contradicts itself, omits `idempotentHint` while relying on it, and its "leave write tools unannotated" rule makes hosts confirm every `tune_start`/`campaign_start` — defeating unattended execution | spec-change · alias CONC-27 |
| **GEN-18** | Major | §9.3.0.2's group filter and multi-column `group_by` do not exist in Phase 1b Task 14's `MetadataGroupSubsetSelector`; composite key has no escaping rule | spec-change |
| **GEN-19** | Major | Per-group breakdown is engine work with no Phase 1b task; the `PHENO_TERMS` user-attr mechanism it needs already ships, so the change is small but must be scheduled | spec-change · alias CONC-24 |
| **GEN-13** | Major | Rename applied to prose only — `workflow.assay`, `counts.assays`, `kind:"assays"` survive in contracts, and the adjacent paragraph tells the agent to read `counts.profiles` | spec-change |
| **GEN-20** | Minor | Plan stale against the cut; `MCP-INTERFACE-AUDIT.md:794` actively recommends `readOnlyHint: true` on `deploy_plan`. Plus: elicitation lands on a 2B tool but is scheduled for 2C | spec-change |
| **GEN-21** | Minor | `edit_previously_tried` has no match predicate; both naive readings fail | spec-change |
| **GEN-22** | Minor | §3.0's `<group>_<verb>` grammar does not describe three of its own tools | spec-change |
| **GEN-23** | Minor | §2.6 and §2.3 still name "promotion tokens" | spec-change |
| **GEN-24** | Advisory | §8.3 documents `campaign_status {detail}` before defining `campaign_status`; "Four tools" over five headings | spec-change |

**On the §10.5 fold** (the orchestrator's question 2): recorded as an alias of
**CONC-22** with the resolution CONC-22 did not propose — the content fold is
right, the call fold is wrong, and the gate belongs on `deploy_start`, which is
SIMP-1's original placement recommendation that USER-8 left unapplied. Tag:
spec-change · needs-user-input.

## VERDICT: REVISE
