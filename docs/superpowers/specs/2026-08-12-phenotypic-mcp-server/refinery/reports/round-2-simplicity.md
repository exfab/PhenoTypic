# Round 2 — simplicity reviewer

Scope: `refinery/snapshots/round-1-spec.diff` (1004 lines). Spec, plan, brief,
ledger only; no codebase traversal. Concern IDs continue round-1 numbering from
`SIMP-17`.

## Framing

Round 1 was an adding round, and it added *well* — most of the 120 net lines are
user rulings the spec genuinely lacked. But it repeated the pattern round 1's own
report named: **a mechanism was added and its justification carried forward,
while the section that retires that justification was written in the same round
and never propagated.** Three of my findings below are that exact shape, and all
three are defects in the round-1 edits rather than in the round-0 spec:

1. `experiment_profile_put` was cut on the premise that the skill writes the file
   directly, while `experiment_profile_get` was kept on the premise that the
   agent has no filesystem. Both cannot hold (**SIMP-18**).
2. `workspace_lineage` was kept in §3.0 because it is "the only read path to the
   anti-repetition evidence" — in a round that made `pipeline_patch` read that
   same trail automatically (**SIMP-22**).
3. §10.5's elicitation is placed at `deploy_plan` and justified as being "at the
   point of spend rather than at the start of a sequence the agent could still
   abandon". `deploy_plan` *is* the start of that sequence; `deploy_start` is the
   spend (**SIMP-19**).

And the largest single reduction available in the whole spec is one the spec
already argues for in its own words: §8.3 says "**a campaign is an organizing
layer, not a parallel execution engine**", two paragraphs above specifying a
bounded-concurrency scheduler with completion callbacks inside a `W2` handler
(**SIMP-20**).

---

## SIMP-20 [Major · spec-change] — Delete `campaign_start`'s executor. It dissolves six open concerns and the spec already argues against it.

**The contradiction, in §8.3's own text.** Two paragraphs apart:

> "Each arm launches through the ordinary `tune_start` path — `RunRegistry.allocate`
> → `LocalRunner.start` → CAS … A campaign is an *organizing layer*, **not a
> parallel execution engine**."

and §8.6:

> "**Launching is not fanned out.** A single `campaign_start` drives every arm …
> **Nothing about fan-out needs new machinery here.**"

But "honouring `max_concurrent_arms`" means arm 3 cannot start until arm 1
finishes. That is a scheduler: a bounded work queue with completion callbacks,
living inside one `W2` tool call that must return on submission. CONC-6 found it
has no execution model; CONC-19 found arm 2's wait is unbounded. Both are
symptoms of a component the spec says it does not have.

**The removal.** `campaign_start` stops launching. The orchestrating agent — which
already holds the whole campaign — issues one `tune_start` per arm. Everything
needed is already specified:

- Arm→study naming is **deterministic** (`studies/<campaign-name>-<arm-id>`), so
  the agent can compute each arm's ids without a write-back.
- `campaign_status` already aggregates per-arm progress by reading study
  directories, which §2.1's disk-as-authority rule says is the authoritative
  source anyway.
- `tune_start` is already a public tool with a `subset_id` requirement (§10.3.1).

**What this dissolves** (I am not re-raising these; I am claiming one edit closes
them): CONC-5 (partial `campaign_start` clobbers a running arm's `study_id`),
CONC-6 (no execution model), CONC-19 (the 1–2 local-arm cap is unachievable under
a capacity-1 semaphore), CONC-23 (no aggregate ceiling across N group campaigns),
GEN-6 (write-back silently reverts a §10.4 amendment), FLOW-8 (`campaign_start`
is not idempotent and a mid-fan-out kill is unrecoverable). The write-back is the
common mechanism in four of the six, and the write-back exists **only** because
`campaign_start` invents the study names; with the agent calling `tune_start`
under the naming rule, there is nothing to write back and nothing to CAS.

It also strictly improves USER-1 compliance. Under the fold, a second local arm
is a separate tool call that can fail fast with `local_slot_busy` and be retried
— a submit-and-poll shape. Today it is an unbounded `await` inside a handler,
racing the host's tool-call timeout, which USER-1 forbids by name.

**The spec's own counter-argument, and why it does not apply.** §8.6 rejected an
earlier draft where "subagents each launch and poll their own arm", because "a
subagent handed only its own arm could not tell which model applied, and would
waste a premature `tune_start`". That objection is about *handing an arm to a
subagent*, not about *who issues the call*. My proposal keeps launching in the
orchestrator, which holds the campaign and the model. The objection survives
intact and does not bind here.

**The real cost, stated plainly.** `campaign_approve` currently gates
`campaign_start`; after the fold nothing mechanically prevents launching arms
from a `draft` campaign. But that interlock is **already illusory**: §8.6 says
"`tune_start` remains available for a standalone study outside a campaign", so an
agent can launch every arm today without ever calling `campaign_approve`. The
fold makes an existing bypass honest rather than creating a new one. And the gate
that guards real spend is promotion (§10.5), not campaign approval — §10.4
confines campaigns to the subset.

**Two forms, pick one:**

- **(a) Full fold — 26 → 25 tools.** `campaign_start` is deleted;
  `campaign_approve` flips `status` and the skill drives the launches.
- **(b) Conservative — 26 tools, `campaign_start` becomes `W0`.** It validates
  approval, resolves and returns the ordered launch list (each arm's `spec_id`,
  computed `study_id`, and routing), and launches nothing. The approval check
  survives as a check; the scheduler is gone.

I recommend **(b)**. It keeps the precondition the campaign artifact exists to
express, costs one small handler, and still deletes the six concerns above —
because all six live in the launching, not the validating.

**Consequence for §1.5:** the "**Locally, run at most 1–2 campaign arms
concurrently**" paragraph added in round 1 is deleted rather than repaired.
CONC-19 asked how to make it achievable; the answer is that it should not exist
in the server. USER-14's ruling survives as a skill rule, which is where §9.1
puts judgment.

---

## SIMP-18 [Major · spec-change · needs-user-input] — The `experiment_profile_put` cut left its only caller unable to run

This is the dangling caller the addendum's question 5 asks about, and it is
mutual: the cut and the keep rest on **contradictory premises about the same
host**.

§3.0's cut table:

> `experiment_profile_put` — … **The triage skill writes the file**; `experiment_profile_get`
> remains so the server stays **self-sufficient on a host that gives the agent MCP and nothing else**.

A host that gives the agent MCP and nothing else is a host on which the triage
skill **cannot write the file**. So on exactly the host `_get` was kept for, §9.5
step 5 — "Write the profile with every trait carrying its `source` (the skill
writes the file directly — there is no `_put` tool)" — is unexecutable, and §9.3
describes an artifact that can be read and never created. `phenotypic-experiment-triage`
is Phase 0; the workflow does not start.

Round 1 also cut the wrong half of SIMP-2. SIMP-2's objection was that the tool
**validated** a file §9.3.5 says the server never acts on — schema machinery over
inert data. The write itself was never the complexity. Cutting the tool kept the
objection's target (validation is now simply unspecified, done by whoever writes
the file) and discarded the part that was cheap.

**Two consistent positions; the spec must pick one:**

- **(a) Assume filesystem access — 26 → 25.** Drop `experiment_profile_get` too;
  the skill reads and writes `profiles/<dataset>.experiment.json` directly. This
  is the pruning answer and it fits the deployment: USER-11 established the
  workspace root is a directory the user names and the target is a shared login
  node reached by SSH. §1.7 says HTTP is a non-goal and "nothing in v1 depends on
  it" — so justifying a v1 tool by a v2 transport (as §3.0 does) inverts the
  spec's own rule.
- **(b) Assume no filesystem access — 26 tools, `put` restored.**
  `experiment_profile_put {dataset, profile}` writes the JSON with **no
  validation at all** — no envelope check, no `source` enum, no `extra="forbid"`
  question, because §9.3.5 says the server never acts on a trait. That is a
  ten-line handler, not the schema machinery SIMP-2 objected to.

I recommend **(a)**, but this touches USER-8, which explicitly kept `_get` on the
MCP-only-host rationale. USER-8 is permanent; the new evidence is that the same
rationale forbids the accompanying cut. **Needs a user ruling on which premise
holds.** Whichever is chosen, one of the two paragraphs in §3.0 must go — they
cannot both be true.

---

## SIMP-19 [Major · spec-change] — The promotion fold is right; the gate is on the wrong tool. Moving it deletes state rather than adding it.

Answering the addendum's question 2 directly: **the fold is sound, the placement
is not.** CONC-22 (Critical) is right that the result is incoherent, and reaches
for a redesign; I think the incoherence has a one-move fix that removes a field.

**The fold is sound.** §10.5's argument is the spec's own and it holds: a
`scope:"full"` `plan_token` can only be minted by an explicit `deploy_plan`
against the parent, so `promotion_token` was a second lock on a door with one
key. The plan token already binds pipeline digest, parent digest, expiry, and
single use — every property the second token contributed.

**The placement is inverted, by the spec's own sentence.** §10.5:

> "**The elicitation fires here**, not two calls earlier — at the point of spend
> rather than at the start of a sequence the agent could still abandon."

`deploy_plan` *is* the start of a sequence the agent could still abandon.
`deploy_start` is the tool that submits — §5.4 defines it that way. Round 1
accepted SIMP-1's fold and rejected SIMP-1's placement recommendation
("elicitation moves to `deploy_start {full}` — better placement, at the point of
spend") using SIMP-1's own justification, reversed.

**Move the elicitation to `deploy_start {scope:"full"}` and CONC-22's second half
disappears without new machinery:**

| Today | With the gate at `deploy_start` |
|---|---|
| `deploy_plan` mints a token **and** returns `pending_human_ack:true` alongside it | `deploy_plan` mints a plain `scope:"full"` token. No ack field. |
| The ack is a second mutable field on the token record, racing `deploy_start` | There is no ack field. The ack is a precondition **inside** `deploy_start`'s handler, raised and consumed in one call. |
| §2.6 needs a CAS row for the ack it does not have | §2.6 needs nothing; the existing `consumed_by` CAS is the whole story. |
| The human approves, then the agent decides whether to submit — approval can sit unspent for 24 h | The human answers at the moment of submission. Approval cannot go stale. |
| The token binds `(pipeline, parent, scope, ack)` | The token binds `(pipeline, parent, scope)`. The ack is provenance in the run's lineage row, where §2.5 already puts decisions. |

This also removes the promotion analogue of CONC-28 (elicitation stretching a CAS
window from milliseconds to minutes): at `deploy_start` there is no CAS being
held open, only token consumption, which is already atomic and already
single-use.

The skill text in §9.5 barely changes — step 2 ("show the human that response and
wait") stays true, and step 3 becomes "`deploy_start {scope:"full"}`, which raises
the confirmation".

**What the fold does not fix, and I am not claiming it does:** `deploy_plan`
remains too expensive to be `W0` (the §10.6.1 header sweep over every parent
image, plus the possible `W1` re-probe). That is CONC-22's first half and SIMP-21
below; the gate move does not touch it.

---

## SIMP-21 [Minor · spec-change] — §1.6.1: keep the table, delete the number

Answering question 3. The table is **half load-bearing and half decoration**, and
the decorative half is what creates contradictions.

**The load-bearing half** is the `W0` row's second sentence — "`W0` means *takes
no compute slot*; it does **not** mean *is instant*, and the two must not be
conflated." That is the fix for audit F3 and it is worth the section on its own.

**The decoration:**

- **"Returns in under one second"** has no basis and is contradicted by three of
  the spec's own tools:
  - §10.6.1 explicitly tables the parent header sweep as **`W0`, "no decode, no
    slot"** over 480 images. Cheap per file; not sub-second.
  - §5.3 heads `deploy_plan` as `W0`, and `MAIN-MERGE.md` found it reads every
    input image twice plus the pipeline — so `deploy_plan {scope:"subset"}`
    violates the bound too, not just `scope:"full"` (CONC-22 addressed only the
    full case).
  - §3.1 requires `catalog_operations` discovery to reconcile `detect.nn` into the
    registry walk so `MicroSamDetector` is reachable. Importing the NN stack on
    first call is seconds, not milliseconds. (SIMP-14 flagged this as a cost;
    round 1 cut `catalog_measurements` and left Task 10c undecided. §1.6.1 turns
    the same cost into a spec contradiction.)
  **Recommendation: delete the number.** The invariant that matters — no compute
  slot, never blocks the event loop, real I/O goes to the executor — is already
  fully stated in the same row. Deleting one clause removes three false
  contradictions and loses nothing checkable.
- **The `W2`/`W3` row** states "no latency requirement", i.e. it is not a
  requirement. It carries one real rule — submit-and-poll — which belongs in §1.5
  where routing lives, not in a requirements table.
- **The Connection row** ("`tools/list` … is a budgeted resource, not free")
  names no budget. USER-16 correctly defers *measuring* F5, but a row with no
  number is not deferred, it is empty. Either give it a token ceiling now (the
  spec can pick one and let 2A validate it) or drop the row and keep the point in
  §3.0's existing "Token discipline" paragraph, which already says it.

Net: §1.6.1 shrinks to two rows and stops manufacturing violations. This is the
answer to "does it create obligations the design then violates" — it does, and
the obligations are the removable half.

---

## SIMP-23 [Major · spec-change] — §9.3.0.2 specifies, in the server's spec, a structure the server is forbidden to read

Answering question 1 — the shape question — and answering two of USER-15's three
deferred questions in the pruning direction.

§9.3.0.2 adds four things. Only one is server mechanism.

| Added | Is it server mechanism? |
|---|---|
| `groups: {key: {traits}}` per-group trait overrides | **No.** §9.3.5: "the server never *acts* on a trait… it is not an interlock." |
| `group_by: list[str]` on the profile | **No.** Same — the server reads it nowhere. |
| Subset selectors may filter to a group, not only stratify | **Yes.** This is the whole mechanism. |
| `campaign_status` per-group cost breakdown | Yes, but see below. |

**The `groups` block needs no spec section at all.** §9.3.0 — three subsections
above, unchanged — states the rule:

> "Adding a trait requires no server change, no schema bump, and no code — only a
> new row in the skill-owned registry (§9.3.4)."

and §9.3.0.1 states that unknown trait keys round-trip verbatim and every trait is
individually optional. A `groups` map of trait overrides is therefore *already
expressible today* in a file the server copies through without inspection. Forty
lines were spent specifying a structure the extensibility rule exists to make
unnecessary. **This is the spec containing the argument for its own reduction, in
the section immediately above the addition.**

The composite key `"neurospora|minimal"` reinforces it: a pipe-joined stringly
composite that breaks if a metadata value contains `|` — precisely the class of
collision §10.2 introduced parent-relative paths to avoid (FLOW-4). Because the
server never parses it, that is a skill-format decision, which is another way of
saying it does not belong in this spec.

**USER-15 open question 1 — profile, subset, or both? Answer: the subset.** The
subset selector is the only component that must act on the grouping;
`MetadataGroupSubsetSelector` already takes the metadata columns as parameters, so
the group predicate is an argument to `subset_generate`, not a new profile field
the server must learn. On the profile, `group_by` is human-readable provenance,
and provenance keys already round-trip. **Not both.**

**USER-15 open question 2 — what does `scope:"full"` mean for a group-scoped
subset? Answer: the parent, unchanged.** A group predicate is a *selection
method*, not a different parent; `subset.parent` is already a recorded field and
§10.5 already runs full scope against it. If "full" instead meant "this group's
images across the whole parent", the server would need a second staging pass over
the parent — new machinery for a case that is already expressible as a subset
whose selector is the group predicate applied to the parent. **Change nothing.**

**USER-15 open question 3 — where does the per-group breakdown live?** CONC-24
answered "the scorer, via trial user attrs", and I agree; I will not restate it.
What CONC-24 did not say: **the breakdown is a phase-1-only signal.** It is
meaningful only while one campaign spans several groups — which is exactly the
general-first phase USER-13 mandates. Once the agent descends per-group, each
campaign holds one group and the breakdown is a scalar again. So it does not need
to be a live-polled field at all; reporting it **once at campaign completion**
delivers the escalation evidence and never touches `campaign_status {since}`'s
stat-cursor economy. That resolves CONC-24's polling conflict by scheduling rather
than by redesign.

**Recommended shape — §9.3.0.2 shrinks from ~40 lines to about six:**

1. One sentence in §9.3: an experiment may hold several species × media groups;
   the profile may carry `group_by` and per-group trait overrides as **ordinary
   skill-owned profile content**, valid under §9.3.0's extensibility rule with no
   server change. (Delete the JSON example, or move it into the skill.)
2. One paragraph in **§10.3**, where selectors live: `MetadataGroupSubsetSelector`
   gains a group *filter* predicate alongside its existing stratification. Per-group
   campaigns, scorers and deploys fall out of that with no further change — which
   §9.3.0.2 already says correctly and is the one part worth keeping.
3. One line in §8: a group-spanning campaign reports a per-group cost breakdown at
   completion, sourced from trial user attrs (CONC-24).

Per-group trait overrides are removable as *spec surface* while remaining fully
available as *capability*. Per USER-13, general-first means the multi-group path
is the exception; the smallest shape that satisfies the ask is the selector
predicate, and everything else the user asked for follows from it.

---

## SIMP-24 [Minor · spec-change] — §3.0's annotation paragraph contradicts itself in two sentences, and overclaims what annotations do

CONC-27 says it is a list, not a rule, and that `deploy_plan` breaks it. Both
true; two things it did not say:

1. **The paragraph is internally contradictory.** "**Every tool carries MCP
   annotations**" is followed one sentence later by "…and **leaving**
   `deploy_start`, `campaign_start`, `tune_start` and `workspace_cancel`
   **unannotated**". Four tools cannot both carry annotations and be left
   unannotated. And leaving them bare is backwards for the stated goal: MCP's
   `destructiveHint` defaults to `true` only as a *default*, so the spec is
   relying on an implicit default for its four most dangerous tools while
   spelling out the harmless ones — the inverse of how this spec treats every
   other contract. If the intent is host-level confirmation, the destructive
   tools are the ones that must be annotated **explicitly**.
2. **"Enforces §9.1's server-vs-skill line at the host level" is an overclaim.**
   `readOnlyHint`/`destructiveHint` are *hints*; a host may ignore them entirely.
   Nothing is enforced. The spec now has two mechanisms both described as the
   gate — annotations and elicitation — and only one of them is real.

**Recommendation, which is a reduction:** delete the paragraph. Add three columns
(`title`, `readOnly`, `destructive`) to the per-group tool tables that already
exist in §3.1–§3.2, §4, §5, §8, §10. Prose that must be kept in sync with 26
tools by hand is the thing that goes stale; a column next to each tool's argument
table cannot. Keep only the two non-obvious notes (the `*_put` tools are not
idempotent; `pipeline_patch` is cumulative), which are genuine findings and
belong beside those tools.

---

## SIMP-22 [Minor · spec-change] — Round 1 added a second read path to the exploration trail and left the first one's justification standing

USER-8 kept `workspace_lineage` against my round-1 proposal with this rationale,
now in §3.0:

> "`workspace_lineage` was proposed for cutting and **kept**: it is **the only read
> path** to §8.7's exploration trail, which is what stops an agent repeating an
> edit it already rejected (§3.2)."

USER-9, applied in the same round, made `pipeline_patch` read that trail on every
call and return `edit_previously_tried`. It is therefore no longer the only read
path, and the anti-repetition argument no longer distinguishes it.

I am **not** proposing the cut again — USER-8 is permanent, and its other
rationale (a host granting MCP and nothing else) is untouched by this. The
finding is that §3.0's sentence is now false as written and should be trimmed to
its surviving half. It matters because a future reader deciding whether to keep
the tool will weigh a reason that has expired — the exact failure mode round 1's
report identified.

(CONC-25 attacks `edit_previously_tried` from the concurrency side; this is a
different point and does not depend on how CONC-25 resolves. Even if the advisory
is kept exactly as written, the §3.0 sentence is stale.)

---

## SIMP-25 [Minor] — Rename residue: `assays` survives in two machine contracts

The `assay` → `experiment_profile` rename (USER-13) reached the prose but not two
tool contracts in §3.3:

- `workspace_info`'s example response returns `"workflow":{"assay": …}` and
  `"counts":{…,"assays":1,…}`, while the prose two paragraphs below tells the
  agent to branch on `counts.profiles == 0`. An agent following the prose reads a
  key the tool does not return.
- `workspace_list`'s `kind` enum still contains the literal `"assays"`, while the
  directory is `profiles/`.

Also cosmetic but in the same family: §10's Phase 0 diagram still reads
"TRIAGE → assay + SUBSET"; §9.3.6's table row says "A newer skill's **assay** read
by an older server"; README claims "Fifteen user rulings" (16 now, with USER-16).

Not a design concern — but `counts.assays` and `kind:"assays"` are the response
schema and an argument enum, not prose, and they are the kind of thing that gets
implemented verbatim from the spec.

---

## SIMP-26 [Advisory] — `experiment_profile` retains a bare-stem resolution rule that nothing resolves

§3.0's id table still gives `experiment_profile` bare-stem sugar via a fixed
`.experiment.json` suffix. After the cut, exactly one tool takes it
(`experiment_profile_get`), and `campaign_put`'s reference is stored "as a string
without even checking the file resolves" (§9.3.5). A resolution rule with one
consumer is fine; it is listed here only so that if SIMP-18 resolves toward (a)
— dropping `_get` — the row goes with it rather than lingering.

---

## Deferral assessment (USER-16)

None of my concerns qualify for `deferred-to-2A`. Every one would still require a
decision after any experiment returned either result:

- SIMP-20 (fold `campaign_start`) — an architecture choice; observing a fan-out
  fail tells you nothing you do not already know from CONC-6/19.
- SIMP-18 (profile put/get premise) — a premise about the deployment host, which
  the user can state today.
- SIMP-19 (gate placement) — a design decision. *Whether elicitation surfaces from
  a subagent* is properly deferred (already in USER-16's qualifying list); *which
  tool raises it* is not.
- SIMP-21 (delete the second-count bound) — deleting an unmeasurable number does
  not require measuring it.
- SIMP-23, 24, 22, 25, 26 — all editorial or structural decisions.

---

## Concerns

| ID | Sev | Concern | Tags |
|---|---|---|---|
| **SIMP-20** | Major | `campaign_start`'s executor contradicts §8.3's own "organizing layer, not a parallel execution engine" and is the machinery §8.6 says is not needed. Deleting it (form (b): validate + emit launch list, `W0`) dissolves CONC-5, CONC-6, CONC-19, CONC-23, GEN-6, FLOW-8, deletes the round-1 "1–2 local arms" paragraph, and moves arm launching to a submit-and-poll shape USER-1 requires | spec-change |
| **SIMP-18** | Major | The `experiment_profile_put` cut and the `experiment_profile_get` keep rest on contradictory premises about the same host; on the host `_get` was kept for, the triage skill cannot write the profile and Phase 0 cannot run. Round 1 also cut the tool rather than SIMP-2's actual target, the validation | spec-change · needs-user-input |
| **SIMP-19** | Major | The promotion fold is right; the elicitation is on the wrong tool. §10.5 calls `deploy_plan` "the point of spend" when §5.4 defines `deploy_start` that way. Moving the gate to `deploy_start {scope:"full"}` deletes `pending_human_ack` as mutable state, removes the §2.6 row CONC-22 says is missing, and stops an approval going stale for 24 h. Alias CONC-22 (second half) | spec-change |
| **SIMP-23** | Major | §9.3.0.2 specifies in the server's spec a structure §9.3.5 forbids the server to read, three subsections below the rule (§9.3.0) that makes it unnecessary. Shrinks ~40 lines → ~6: one sentence in §9.3, one selector-predicate paragraph in §10.3, one completion-time breakdown line in §8. Answers USER-15 Q1 (**subset, not both**) and Q2 (**`scope:"full"` = the parent, unchanged**); on Q3 aliases CONC-24 and adds that the breakdown is completion-time, not polled | spec-change |
| **SIMP-24** | Minor | §3.0's annotation paragraph says every tool is annotated and then leaves the four dangerous ones unannotated, relying on an implicit MCP default for exactly the tools the spec is strictest about elsewhere; and "enforces at the host level" overclaims what a hint does. Replace the paragraph with three columns on the tool tables that already exist. Alias CONC-27 | spec-change |
| **SIMP-22** | Minor | §3.0 keeps `workspace_lineage` as "the only read path" to the exploration trail in the same round that gave `pipeline_patch` that read (USER-9). The keep stands on its other rationale; the sentence is stale | spec-change |
| **SIMP-21** | Minor | §1.6.1's "under one second" is contradicted by §10.6.1's header sweep, §5.3's `deploy_plan` (both scopes) and §3.1's `detect.nn` discovery. Delete the clause — the row's real invariant is already fully stated in its second half. Drop the `W2`/`W3` row (states no requirement) and either number or drop the Connection row | spec-change |
| **SIMP-25** | Minor | Rename residue in two machine contracts: `workspace_info` returns `workflow.assay` / `counts.assays` while the prose reads `counts.profiles`; `workspace_list`'s `kind` enum still contains `"assays"` | spec-change |
| **SIMP-26** | Advisory | `experiment_profile`'s bare-stem resolution row now has one consumer; it goes with `_get` if SIMP-18 resolves toward (a) | spec-change |

**Net if SIMP-20(b), 18(a), 19, 21, 23, 24 are accepted:** 26 tools → 25; one
`W2` handler becomes `W0`; ~55 spec lines removed; six open concurrency/general
concerns closed by deletion rather than by design.

**VERDICT: REVISE**
