# Round 3 — data-flow reviewer

Scope: `round-2-spec.diff` (1367 lines) + `round-2-plan.diff` (103). Verification
first, discovery second, per the addendum. Severity ratchet honoured: two
Critical, six Major, everything else one line under Advisory.

**Headline: round 2 repeated round 1's defect.** Every applied fix I traced
landed in the section that *narrates* the change and missed the section that
*defines* the thing changed. Five of my eight concerns are that same shape at
five different sites. The new material (concurrency block, `deploy.approve`, the
fan-out background task) is mostly sound where it is written; it is unreachable
or contradicted where the surrounding spec was left alone.

---

## 0. A provenance note on my own IDs

The ledger's citations of my round-2 IDs are systematically off, and the round-3
brief inherited them. My `round-2-data-flow.md` concerns table is authoritative:
FLOW-17 is the `pipeline.step` writer gap (not the ack/token contradiction, which
was my FLOW-25/FLOW-27); FLOW-18 is the match key (not `human_response`);
FLOW-20 is the composite group key (not the token binding set); FLOW-25 is the
five-source gate (not `catalog_measurements`). I verified by content rather than
by ID, so every item the brief asked about is covered below — but the ledger's
FLOW column should be re-derived from the report before round 4, or the next
brief will mis-address the same way.

---

## 1. Verification of my round-2 concerns

| Brief's ask (by content) | My ID | Verdict |
|---|---|---|
| Token minted before the ack it records | FLOW-27 | **Half-fixed.** §10.5 and §2.5 are correct. §5.4 — the section that *defines* `deploy_start` and the token — still states the old model verbatim. See **FLOW-33 (Critical)** |
| `human_response` has no parameter | PF-3 | **Not fixed on `deploy_start`.** Present on `campaign_approve` (§8.3:3838). Absent from §5.4's argument table. See FLOW-33 |
| Binding set complete; `run_name` vs `argv_digest` don't contradict | PF-1 | The two guards do **not** contradict (§5.4 explains why both are kept, and the explanation holds). But the set is **not** complete and one of its two producers cannot satisfy it. See **FLOW-34** |
| Multi-group: does `group_filter` have a validated source; does the subset artifact record enough for USER-21 | FLOW-19/20/23 | **No, to both.** See **FLOW-36** |
| `decision` derived; match key = full edit minus index | FLOW-17/18 | Both written, both **wrong for 5 of the 6 edit kinds**. See **FLOW-35** |
| `produces_columns` restoration | PROP-2 | **Complete.** §3.2:1268-1290 restores the `header_scheme()` dispatch, the `TEXTURE.get_headers()` `TypeError`, the 130-vs-13 expansion, the `get_measurement_infoclasses()` instance-dependence, and the "do not model on `_cli_readme_generator`" warning. §7 P2's doubled name is fixed and now cites §3.2. No stale `§3.1` citation for it survives. Plan Task 10c's interface line is corrected. **Close it.** |
| Propagation pass — and did it introduce new inconsistencies | PROP-4/5 | Rename residue and CWD residue are genuinely cleared, including SIMP-25's over-replacement (the biological noun is restored at §8.7, §9.3.3, §9.4, §11). Counts in the plan are right (10+13+3 = 26). **But** the plan body was not touched (**FLOW-38**), and the pass introduced at least one new contradiction (`write_generation`, **FLOW-37**) |
| FLOW-1, FLOW-2, FLOW-5 | — | **Confirmed still open.** No hunk touches the probe's measurement frame, `produces_columns`' third frame, or the continuation/`output_not_empty` inversion. Not re-argued. |

---

## 2. Concerns

### FLOW-32 [**Critical**] — the campaign fan-out state machine has no reachable recovery path, strands arms under local routing, and its recovery signal is underivable · spec-change

This is the item the brief flagged as new and safety-critical. All three parts
are in §8.3:3882-3940.

**(a) The advertised recovery is refused by the guard advertised eight lines
above it.** The handler "transitions `approved → launching → running` and
returns", and "a concurrent second call is refused by the transition itself —
`launching` is not `approved`". Then: "**Re-calling `campaign_start` is
idempotent, and that is what makes a kill mid-fan-out recoverable.** It launches
only arms with no `study_id` recorded."

After a kill the artifact reads `running`. A recovery call and a concurrent
double-launch are **the same call arriving at the same status** — the spec offers
no discriminator between them, and then asserts the transition rejects one while
the other succeeds. Only one of the two sentences can be implemented:

- guard refuses non-`approved` → the campaign is permanently stuck with `queued`
  arms and **the recovery path does not exist**, contradicting "Nothing else in
  the design recovers a half-launched campaign";
- guard admits `running` → the double-launch protection the same paragraph claims
  is gone, and two per-campaign launchers race the same arm list. (They would
  collide in `RunRegistry.allocate` and one would take `output_generation_active`
  — but that is accidental safety from a different subsystem, not the guard the
  spec says is doing the work.)

CONC-6/FLOW-8's original finding was "`campaign_start` is not idempotent; a kill
mid-fan-out is unrecoverable". The background task fixed the *execution* model
and left the recoverability claim resting on a transition that forbids it.

**(b) `launch_state` was invented to be that discriminator and is not
derivable.** §8.3:3968 defines `fan_out_incomplete` as "the campaign is
`launching` or `running` and at least one arm is `queued` with no `study_id`,
**with no background task alive to finish it**." Nothing on disk records launcher
liveness. §2.3 explicitly anticipates overlapping server instances over one
workspace, so a peer server's *healthy* in-progress fan-out presents exactly the
same bytes as a dead one's; and within one server, a launcher legitimately parked
on `max_concurrent_arms` is indistinguishable from a launcher that no longer
exists. This is the identical problem `RunRecord` was given `(pid, create_time)`
for this round (§2.4) — the campaign artifact got no equivalent.

**(c) Under local routing the launcher has no state for arms it cannot start.**
USER-17 is unambiguous: "A second local arm does not wait. Arriving at a full
slot it is told the slot is busy and returns." §1.5's routing table gives a local
`W2` the slot for its entire subprocess lifetime. The arm-state table defines
`queued` as "waiting on `max_concurrent_arms` **or the server-wide ceiling**" —
the slot is not in that list. So with `local_slot_capacity=1` a three-arm
campaign launches arm 1, and arms 2 and 3 are refused into a state the spec does
not name, with nothing specified to retry them. The local path is precisely the
one USER-14/USER-17 exist to make safe.

**(d) Nothing wakes a blocked launcher.** `limits.max_inflight_arms` is
server-wide and "checked by the background launcher before each arm is launched".
When campaign A's launcher finds the ceiling consumed by campaign B, the spec
does not say whether it polls, sleeps, or exits; there is no wake-up on arm
completion, no poll interval, and no error code for the condition. Nor is the
task's lifecycle stated against `workspace_cancel` or server shutdown.

**Fix, minimally:** give the transition an explicit recovery arm
(`launching|running → running` **iff** `launch_state == fan_out_incomplete`),
make `launch_state` derivable by writing a launcher lease onto the artifact with
the same `(pid, create_time)` treatment §2.4 just gave `RunRecord`, add a
`blocked` arm state (or fold slot refusal into `queued` and say so), and state
the launcher's wake condition and its behaviour on cancel/shutdown.

---

### FLOW-33 [**Critical**] — §5.4 still specifies the pre-USER-18 gate, including the missing `human_response` parameter · spec-change

USER-18 moved the elicitation to `deploy_start` and USER-22 made `human_response`
unconditional. Both rulings were written into §8.2 (3733-3795) and §10.5
(5446-5480). **§5.4 — the only section that defines `deploy_start`'s signature and
the token's contents — was not touched at all.** Three separate assertions of the
retired position survive:

1. **§5.4:2478-2484, the argument table: there is no `human_response`
   parameter.** The table lists `scope`, `plan_token`, `resume`,
   `retry_failures`, `restart`, "same arguments as `deploy_plan`, plus". PF-3
   said this tool was "unimplementable as written"; it still is. §10.5:5463 says
   the parameter is required here; the signature does not have it. (§8.3:3838
   *does* carry it for `campaign_approve`, which is why this reads as a missed
   site rather than a rejected ruling.)
2. **§5.4:2480, the `scope` row:** "its `plan_token` must have been minted at
   `scope:"full"` **with the human ack recorded** (§10.5)". §10.5's own table one
   ruling later reads "`plan_token` minted at `scope:"full"`, **plus the ack at
   `deploy_start`**". The two tables state opposite contracts and both cite each
   other.
3. **§5.4:2548-2550, immediately after the new "binding set is exhaustive"
   table:** "It **also records the human ack**, which is what makes it the
   promotion gate rather than merely a plan." This is the sentence USER-18
   deleted, sitting three paragraphs below a table added in the same round that
   claims to enumerate everything the token carries.

An implementer building the deploy tools reads §5.4, not §10.5. Under §5.4 as it
stands they build a token that carries an ack, a `deploy_start` that cannot
accept a human's words, and no elicitation — i.e. exactly the design USER-18
overturned. This is the round-1 defect (PROP-1..5) reproduced on the
safety-critical path.

Also stale in the same paragraph: "a `plan_token` whose recorded `(pipeline
digest, images digest, compute)` matches the request" — a triple that no longer
matches the record (`images digest` is not a field; the record has
`subset_digest`), and that omits everything FLOW-34 covers.

---

### FLOW-34 [Major] — the binding set is declared exhaustive while omitting two fields §10.5 makes load-bearing, and its second producer cannot satisfy it · spec-change

§5.4:2506 asserts "**The binding set is exhaustive**". The record example
(§5.4:2498-2504) carries `pipeline_digest`, `subset_id`, `subset_digest`,
`compute`, `run_name`, `array`, `estimate.node_hours`, `argv_digest`.

**Omitted, both cited as binding elsewhere in the same document:**

- `parent_digest`. §5.4:2545 ("A `scope:"full"` token additionally binds
  `parent_digest`") and §10.5:5478 both rely on it, and it is the field
  `deploy_start` is supposed to check *before* prompting a human. It is in
  neither the example nor the table. This is the exact half of PF-1 that round 2
  did not close.
- `group_filter`. USER-21's ruling is "**The token binds `(parent_digest,
  group_filter)`**", restated at §10.5:5518. Not in the record, not in the table,
  and not mentioned in §5.4 at all — so the guarantee USER-21 bought ("an ack
  given for one group cannot be spent on another's images") has no carrier.

**And the growth broke the other producer.** §5.4:2551 — "The token is satisfied
two ways: a direct `deploy_plan` call, or membership in an **approved campaign**
(§8), which stamps a token per arm at approval time." At campaign-approval time
there is no `run_name` (the study is named later, `studies/<campaign>-<arm>`), no
resolved `array` (nothing has consulted `scontrol`/`sacctmgr`), no
`estimate.node_hours` (no `deploy_plan` has run for the arm) and no `argv_digest`
(no argv has been rendered). So either campaign-stamped tokens carry nulls for
four now-mandatory fields — in which case "exhaustive" is false and validation
has to special-case them — or campaign approval must run a full `deploy_plan` per
arm, which is not what §8.3 does. Neither is written down.

`run_name` and `argv_digest` do **not** contradict each other — §5.4:2516-2521
gives a coherent reason for keeping both (different failure messages; `argv_digest`
also moves on a compute-key change). That part of the brief's question is clean.

---

### FLOW-35 [Major] — the derived `decision` is right for one of six edit kinds and inverted for a second; the new match key collapses two kinds entirely · spec-change

The brief asks directly whether "is the op still present" is decidable for every
edit kind. It is not. §3.2:1317-1318 enumerates six: `insert_op {slot, index,
class, params}`, `remove_op {slot, index}`, `move_op {slot, from, to}`,
`set_params {slot, index, params, merge}`, `set_grid {nrows, ncols}`,
`set_model {class, params|null}`.

Against §3.2's rule ("an edit still present in the current pipeline was kept, one
no longer present was reverted") and §8.7:4213:

| Kind | Derivation | |
|---|---|---|
| `insert_op` | present ⇒ kept | ✅ the one case, and the one the examples use |
| `remove_op` | **inverted** — the op being *present* means the removal was reverted | ❌ reports the opposite of the truth |
| `move_op` | the op is present under both outcomes | ❌ undecidable |
| `set_params` | the op is present under both outcomes; only values changed | ❌ undecidable — and §3.2:1395 calls this "exactly what the loop is *for*" |
| `set_grid` / `set_model` | no list member to be present or absent | ❌ not expressible |

The advisory's whole value is `decision` separating "tried and reverted" (do not
repeat) from "tried and kept" (already in the pipeline). On `remove_op` it will
confidently tell the agent the opposite; on the three undecidable kinds it must
emit something, and nothing says what.

**The match key has the same shape of problem.** §3.2:1393 fixes the recorded
block as `{kind, slot, class, params}` with `index` excluded. Cross-referencing
the actual edit schemas:

- `remove_op` has no `class` and no `params` → key degenerates to
  `{remove_op, "ops"}`. **Every removal from `ops` matches every other removal.**
- `move_op` likewise → `{move_op, "ops"}`. Every move matches every other move.
- `set_params` has no `class` field either; `class` must be resolved from `index`
  at record time (nowhere stated), and dropping `index` then merges two same-class
  ops at different positions — two `BlurGauss` in one pipeline collide.
- `set_grid` has neither `slot` nor `class` nor a `params` key.

Dropping `index` was right for `insert_op` (FLOW-18's original point) and made
`remove_op`/`move_op` strictly worse than before. The key needs to be
kind-dependent: identify by *what the edit does to the pipeline* (op class +
resolved params for the op-valued kinds; the target op's class for `remove_op`;
the ordered class sequence for `move_op`), not by one schema applied to six
different shapes.

---

### FLOW-36 [Major] — `group_filter` exists only in prose; the ABC does not declare it, the artifact does not record it, and full-scope-with-filter needs the staging §10.5 says it does not · spec-change · needs-user-input

USER-24 deleted the multi-group design and kept exactly one primitive. That
primitive was written into §9.3.0.2:4428 and **nowhere else**.

- **§10.3's `SubsetSelector` ABC (5065-5077) has no `group_filter` field.** It
  declares `n` and `seed` under `model_config = ConfigDict(extra="forbid",
  validate_assignment=True)`. This is MG-1's finding unchanged — `extra="forbid"`
  means the field is not addable without a model change, and §10.3 was again not
  touched. §7 P3's subpackage item (3403-3405) does not mention it either, so no
  prerequisite covers it.
- **§10.2's subset artifact (5005-5030) does not record it.** It would ride in
  `selection.params` *if* the ABC declared it, which is the whole dependency.
  §10.5:5515 says "`deploy_plan` carries the subset's `group_filter` through to
  full scope"; there is no field on the artifact for it to read.
- **No validation and no error code.** §6.2 has `group_key_not_in_metadata` for
  the selector's `group_key`; nothing covers a filter column absent from the CSV,
  a filter value matching no row, or the `Metadata_*` canonicalization mismatch I
  raised as FLOW-19 (still open). §10.5:5519 asserts "a filter that matches
  nothing at full scale fails on the empty image set" — no code, no owner.
- **`derived_from`** (USER-24's other addition) is likewise prose-only, at
  §9.3.0.2:4448. §8.2's campaign artifact schema does not carry it.

**And the execution claim is wrong.** §10.5:5516 — "This costs nothing
structurally: the filter is a metadata predicate over the parent's images, which
is the same join the subset already performed, and **it needs no staging**." But
§1.6's new-pieces table justifies subset staging with "**neither engine accepts a
file list**", and §10.3.1 exists entirely to materialize an arbitrary image list
as a directory tree because of that. `parent ∩ group_filter` *is* an arbitrary
image list over the parent. So a group-filtered full-scope deploy either stages
(potentially the whole dataset, which §10.3.1 §4 says must never reach the parent
images) or it cannot be expressed as a `python -m phenotypic --input` invocation
at all — and `argv_digest`, which is defined as the digest of that rendered argv,
has nothing to digest. USER-21's ruling is sound in intent; its "needs no
staging" premise contradicts the two sections that own the input boundary, and
the reconciliation is a user call.

---

### FLOW-37 [Major] — §8.2's campaign artifact records none of §8.3's four new state concepts, and `write_generation` contradicts §2.6's CAS key one section away · spec-change

Round 2 added four fields to `campaign.json` in §8.3's prose. §8.2 — the
canonical artifact an implementer builds the pydantic model from (3605-3634) —
carries none of them:

| Added in §8.3 prose | In §8.2's schema? |
|---|---|
| arm `state` including the new first-class `queued` (3930-3935) | no — arms have `id`, `pipeline`, `tune_spec`, digests, `rationale`, `prefab_baseline` and no state field at all |
| per-arm `study_id` written back under CAS (3898-3903) | no |
| `write_generation` (3966) | no |
| `launch_state` (3968) | no |
| `derived_from` (§9.3.0.2:4448) | no |

**The new contradiction.** §2.6's new rule is emphatic and general: "every
mutation of an artifact CASes on the pair: the expected `status` **and** the
content digest of the bytes the caller read", with `artifact_changed` on
mismatch. §8.3:3966 then introduces `write_generation` — "the artifact's own
write counter, incremented on every CAS… **it is the value a subsequent mutation
CASes against**". Two different CAS keys for the same artifact, defined one
section apart, each stated as the rule. An implementer cannot build both. (If the
counter is meant only as a cheap staleness *read*, that is a defensible design —
but then it must not be described as the CAS key, and §2.6's digest remains the
key.)

---

### FLOW-38 [Major] — the plan still carries the cut promotion tools and the retired `human_response` signature; PROP-5 was applied to the counts only · spec-change

The plan diff is 103 lines and touches counts (32→26), the phase map, the Tech
Stack line, and the CWD default. The plan **body** was not swept. Grepping
`round-2-plan.md`:

- **Cut tools still specified in six places**, including a per-tool decision
  table and a written directive:
  - `:2849` `| promotion_request | Assemble promotion review | ✗ ¹² | ✗ | ✅ | ✗ |`
  - `:2850` `| promotion_approve | Approve promotion | … |`
  - `:2899` footnote 12 specifying `promotion_request`'s `promotion_id` return
  - `:2957` "…`promotion_approve` record a decision a human actually made — never
    call either without one" — an instruction about two tools that no longer exist
  - `:2238`, `:2263`, `:2315`, `:2405` in the interface-audit prose
- **`required-unless-elicited` survives in three places**, the form USER-22
  deleted: `:2000`, `:2061` (a "**Do now, implement later**" action row), and
  `:5636` (**decision D6**, the decision record itself).
- **D5 (`:5635`) still specifies the enumeration §3.0 replaced with a
  derivation** — "leaving `deploy_start` / `campaign_start` / `tune_start` /
  `workspace_cancel` unannotated". §3.0:1030-1045 this round says "An earlier
  draft enumerated four tools to leave unannotated, and an enumeration silently
  rots the moment a tool changes what it does — which two of them promptly did."
  The plan preserves the enumeration and the spec preserves its refutation.
- **F3 (`:5661`)** still lists `deploy_plan` among the `W0` tools doing hidden
  blocking work; §5.3 now declares it `W1` at full scope, which is F3's fix
  applied.

These are the documents Phase 2C is written from. GEN-19/PROP-5's finding was
"the plan was never updated with the spec"; it is now updated in its headline
numbers and not in its content.

---

### FLOW-39 [Major] — §8.7's canonical journal row and §3.2's examples and prose still encode the retired edit shape and the retired decision writer · spec-change

Same defect class as FLOW-33, on the `edit_previously_tried` path. The new rules
were appended; the artifacts they describe were left alone.

- **§8.7:4202-4205, the canonical `pipeline.step` row**, is
  `"edit":{"kind":"insert_op","slot":"ops","index":1,"class":"FocusEdgePhase"}`
  with `"decision":"keep"` — i.e. **`index` present, `params` absent**, and a
  written keep/revert value. Eight lines below, the new text says the recorded
  edit carries "the full edit (parameters included)", that `index` is "**not**
  part of the match", and that "the keep/revert decision is **never written by a
  tool**". The example an implementer copies is the design that was replaced.
- **§3.2:1354-1357**, the paragraph introducing the advisory, still reads "§8.7
  records every accepted step in the lineage journal with its evidence and **its
  keep/revert decision**" — three paragraphs above "**`decision` is derived, not
  reported**".
- **§3.2:1347**, `pipeline_patch`'s `diff` example, and **§3.2:1364-1366**, the
  advisory example (`"decision was 'revert'"`), carry the same retired shape.
- Residual ambiguity the new text creates: if `decision` is derived at read time,
  what is the journalled `"decision":"in_flight"` for, and is it ever updated?
  §8.7 says "Evidence is filled in when the probe returns" but says nothing about
  the field. One field, two writers, no reconciliation.

---

## Advisory (one line each, no argument)

- §1.5 admits `W1` beside a refused live orphan, which breaks the OOM invariant the same section says the slot "owns alone" — the orphan holds no slot.
- No error code covers a local `W2`/`W3` refused because the slot is busy (non-orphan case); `local_slot_timeout` is scoped to `W1` and `local_slot_orphaned` to the orphan case.
- `LocalComputeSlot` is written `asyncio.Semaphore(1)` in §1.5 and §2.6 while its capacity is stated as configurable (`local_slot_capacity`).
- `campaign_start` writes campaign status `running` before any arm launches, so `running` with zero running arms is the normal early state.
- §1.6.1's `W0` row still asserts "under one second" and "does not mean *is instant*" in the same cell (my FLOW-28, unaddressed).
- The `deploy.approve` and `deploy.start` lineage rows carry `scope` and `subset_id` but no `group_filter`, so a group-scoped full deploy is not reconstructible from the journal.
- `deploy_start` now contains an unbounded human wait inside a `W3` handler; §1.6.1's "no latency requirement" covers submission latency, not a host tool-call timeout.
- SIMP-30's memorial argument: `round-2-spec.md` has **48** "an earlier draft / the first draft" passages against 5,753 lines — the ~35-line estimate is now roughly 5× low, and eight of the largest were added this round.
- `header_sweep_cost.py` is present at `docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/`; I did not re-run it.
- §9.3.0.2 is now filed **after** §9.3.0.1 (SIMP-27 resolved).

---

## Verdict

**VERDICT: REVISE**

Not because the round-2 decisions are wrong — USER-17..24 are sound and the
concurrency block, the `deploy.approve` row, and the `(status, artifact_digest)`
CAS are real improvements. Because the decisions were written into the sections
that *explain* them and not into the sections that *define* the artifacts and
signatures, so an implementer reading §5.4, §8.2, §10.3 or the plan builds the
superseded design with nothing failing. Six of my eight concerns are that single
defect; none requires a new decision, only the propagation the round already
intended.

**`deferred-to-2A`:** none of the above qualifies under USER-16. Every one still
needs a decision after any experiment returns either result.

**`needs-user-input`:** FLOW-36 only (whether `parent ∩ group_filter` stages or
whether USER-21's scope resolution changes shape).
