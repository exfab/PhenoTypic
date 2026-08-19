# Round 4 — general reviewer (final round)

**Scope:** `snapshots/round-3-spec.diff` (1459 lines, 733 added / 249 removed) +
`round-3-plan.diff` (263). Verified against the working tree at `c35936606`.
The three post-diff fixes (CONC-34 `human_response` keyed on token kind,
CONC-35 `executors.compute` bounds row, CONC-36 `image_manifest`) are treated as
already applied and are not raised.

**Severity ratchet:** Critical and Major only; a Major without a concrete fix is
a one-line Advisory.

---

## Part 1 — verification of GEN-26..32

### GEN-26 [Critical] — **CLOSED**, all four parts, including the prompt source.

| Part | Where it landed | Verdict |
|---|---|---|
| `human_response` on `deploy_start` | `05:300` arg table (now `str?`, keyed on token kind by the post-diff CONC-34 fix), `note?` at `05:301` | ✅ |
| `ack_source` in the response | `05:305-307`; the third value `campaign_approved` added post-diff at `08:244` | ✅ |
| `scope` row rewritten | `05:297` — now "the human ack is taken here, not carried on the token", no longer citing §10.5 for the retired contract | ✅ |
| **Prompt source** | `05:309-321` — "**The prompt is rendered by the server from the token, never passed in by the agent**", and the token record carries `ack_prompt` + `decision_content` (`05:353-354`), both listed in the binding table as *carried but not bound* (`05:390-393`) | ✅ |

The prompt-source fix is the one I was least confident would land as a mechanism
rather than a sentence, and it did: the record example, the "carried but not
bound" rule, and the §8.2 artifact-id-first rule now reference one another. The
added ordering rule at `05:323-329` (**elicit → allocate → acquire**) is a real
improvement I did not ask for and closes the "human think-time holds the slot"
hazard the relocation could otherwise have reintroduced one call later.

### GEN-27 [Major] — **CLOSED.** One canonical binding table; the second producer can satisfy it.

`05:363-367` declares the table the only statement of the set and requires other
sections to cite it. Verified:

- The `(pipeline digest, images digest, compute)` triple in "Plan-then-submit is
  mandatory" is gone; `images_digest` now survives only at `05:427` inside the
  *rejected* self-describing-hash alternative, where it is correct.
- §10.5's two conflicting statements are gone: `10:592-596` now cites §5.4's table
  for the consequence, and the `(parent_digest, group_filter)` restatement at the
  old `10:549` is replaced by "**`group_filter` is bound on the token at full
  scope**, per §5.4's binding table".
- The table (`05:376-388`) carries a **Scope** column, with `parent_digest` and
  `group_filter` marked `full` only, and `05:357-361` states they are fields of
  the record at both scopes so an implementer does not infer optionality from the
  example.
- `estimate.node_hours` is now in §5.3's canonical response (`05:262`) alongside
  `node_seconds`, with the reason stated (`05:277-283`). Arithmetic checks:
  340 s → 0.094 h. The unit mismatch I raised is reconciled.

**Second producer:** yes, and better than I expected. `05:493-506` adds a
`kind` discriminator and a per-producer binding table — `"plan"` binds the full
set; `"campaign_arm"` binds `scope, pipeline_digest, subset_id, subset_digest,
compute, campaign_id, arm_id` and is **`subset` only**, with `run_name`, `array`,
`estimate.node_hours` and `argv_digest` *absent, not null*. `campaign_arm_scope_full`
exists at `06:51` and is exercised at `06:328`. §10.4's "Subset only" row
(`10:458`) agrees. The weaker binding set is kept off the irreversible path by
construction rather than by convention.

*One residue, Advisory:* §8.3's `campaign_approve` block (`08:336-364`) is
untouched and still says only "**mints one `plan_token` per arm**" — it never
names the `campaign_arm` kind. It states no binding set, so it does not violate
§5.4's single-statement rule, but an implementer building `campaign_approve` from
its own section mints a `"plan"`-kind token.

### GEN-28 [Major] — **CLOSED**, all three, and thoroughly.

`group_filter`: ABC field + docstring (`10:114`, `10:121`), the "declared here and
it has to be" rationale (`10:138-149`), a six-row semantics table
(`10:152-158`), the subset artifact at top level *and* in `selection.params`
(`10:46`, `10:52`, `10:84-93`), §7 P3 item 3 (`07:352-357`), two new error codes
(`06:58-59`), three §6.5 tests (`06:311-318`), §5.3's arg table and response
(`05:237`, `05:286`), §5.4's record and binding table, §2.5's two deploy rows
(`02:290-296`), and the plan's phase-1b `SubsetSelector` Interfaces block with a
contract table and two named tests.

`derived_from`: `08:88-89` in the campaign JSON, and a contract row at `08:125`.

No gap. Note also a non-finding I checked and cleared: `subset_generate`'s arg
table (`10:288-294`) needs no change, because `selector` is `{class, params}` and
the filter rides in `params` — the ABC placement is what makes the tool table
correct as it stands. (`subset_put` is a different matter — see GEN-37.)

### GEN-29 [Major] — **table added, and the previously-inverted kind is now correct.** One new inversion introduced; see GEN-35.

`03:422-429` is a six-row table against the six kinds §3.2 actually declares.
`remove_op` is explicitly `**inverted**: the op **absent** from `slot` ⇒ `keep``
— correct. `class` resolved from `index` at record time gives `remove_op` and
`move_op` the discriminator they lacked. `set_grid`/`set_model` get real rules
instead of a referent they do not have. `undetermined` is a reported value with
the evidence attached (`03:431-437`), and the §8.7 side records the canonical
edit rather than the raw arguments (§8.7's journal row, `08:738-742`, now two appends correlated by `step_id`
with `state` and no `decision` field).
§6.2's row (`06:33`) and three §6.5 tests (`06:330-343`), including "a kept `remove_op` must
report `keep`", are present.

### GEN-30 [Major] — **CLOSED**, all five sites.

`W0` examples row no longer cites the cut `pipeline_diff` (`01:198`-region);
`W1` work-class row names `deploy_plan {scope:"full"}` and metadata-only image
I/O; routing table reads "**holds slot while it computes**" with a paragraph
(`01:246-256`) explaining tier-1 takes no slot; the one-probe invariant is
re-scoped to probe *execution* (`01:271-278`); §1.6.1's `W1` row exempts the tool
from the 4-image cap and gives its real bound. §3.0's derivation now catches
`deploy_plan` on the **mutation** clause (it writes a token record) as well as
the cost clause, plus a static-annotation/union rule. The measurement arithmetic
holds: 0.081 s / 460 = 0.176 ms; 1 s / 0.176 ms = 5,682 ≈ "roughly 5,700".

### GEN-31 [Major] — **CLOSED.**

§9.5 is rewritten (`09:620-640`): step 1 says "**No human is in this step**",
step 3 is the gate and names `human_response` with the never-invent /
never-treat-a-timeout-as-yes rule, and the promotion residue in the coverage-warning
line is gone. README's multi-group sentence is deleted and replaced with USER-24's
actual outcome. The `instructions` string in `MCP-INTERFACE-AUDIT.md` now names
`deploy_start` instead of `promotion_approve`.

*Advisory:* README:6 still reads "**Twenty-four** user rulings" — there are 27.

### GEN-32 [Major] — **CLOSED**, resolved toward §2.6 as claimed.

`08:498-505` (the `campaign_status {detail:"artifact"}` block) now reads
"**It is a hint, not the CAS key**", §8.2 gains `write_generation: 7` on the
schema (`08:84`) with the same statement restated at `08:127-134`, §2.6's `(status, artifact_digest)` is unchanged,
and §6.5 gains the discriminating test — "a mutation whose `write_generation`
matches but whose `artifact_digest` does not must fail `artifact_changed`".

### GEN-18 — **confirmed open, not re-argued.** But it got more dangerous.

`to_argv(RunConsoleState)` is still the only named emitter (`05:395`, `05:519`,
`07:307`), `deploy_start` still takes `restart`, and no `_services` symbol emits
`--restart`, `--overwrite`, `--slurm k=v` or `--gpu-slurm`. Round 3 **added a
flag to that surface** (P8's manifest input) and, via CONC-36, made
`argv_digest` a bound field on a token that carries a human's consent. So the
gap moved from "a coverage hole" to "a coverage hole on the security boundary":
whatever `to_argv` cannot emit is either missing from the argv that actually
runs, or appended outside `to_argv` and therefore outside `argv_digest`. Either
way the bound field stops covering what executes. This is the one standing item I
would flag as materially worse under round 3's changes.

*(FLOW-1/2 also worsened by one degree: probe timings now feed
`estimate.node_hours` → the token → the number quoted at the human gate, so a
probe measuring a frame no engine uses is no longer only a scoring problem.
One line, no argument — it is FLOW's item, not mine.)*

---

## Part 2 — the growth question

**The spec grew 5,753 → 6,237 (+484 net; the sweep added ~548 and USER-27's trim
removed 64).** Composition of the 732 added content lines:

| Kind | Lines |
|---|---|
| Table rows | 77 |
| JSON / code | 37 |
| Blank | 69 |
| Prose | ~549 (of which 50 open a bolded normative statement) |

**It is not narration returning by another route.** The "an earlier draft…"
genre added **zero** instances and removed **seven**; every removal in the diff
is either that genre or a statement the round deliberately retired. The prose is
overwhelmingly rule text, and the sweep's own claim that it added contract
content is substantially true.

**But it is not propagation either, and that is the finding.** The sweep declared
"No ruling was made, amended, or re-litigated here." Several of the largest added
blocks are **new design decisions**, taken by a subagent sweep and not recorded as
rulings anywhere:

| Added block | Lines | What it actually is |
|---|---|---|
| `01:340-357` server-wide arm ceiling as `asyncio.Semaphore(max_inflight_arms)`, arrival-order admission, `queued_reason` ×3 | ~18 | **CONC-23's fix**, which the ledger still lists `open · "Candidate fix drafted; not yet applied pending the panel"` |
| `01:359-368` the launcher's wake condition, cancel and shutdown behaviour | ~10 | New design |
| `01:370-381` **"queued, not refused"** | ~12 | New design that **reverses USER-17's recorded corollary** — see GEN-33 |
| `01:320-328` `executors.compute.max_workers` *is* `local_slot_capacity` | ~9 | New design (subsequently propagated by CONC-35) |
| `01:330-338` probe responses carry `contended: true` | ~11 | **A new field on a tool response** — see GEN-34 |
| `01:413-430` `W1` does not run in the server process at all | ~15 | Reverses round-2's `run_in_executor` statement |
| `05:493-506` token `kind` + two-producer binding table | ~14 | New design (correct and needed — GEN-27) |
| `08:427-470` two-armed `campaign_start` transition + `launcher` lease | ~30 | **CONC-26's fix**, ledger status `open` |

So the honest answer to the addendum's question 1 is: the growth is defining-section
content that had to exist, **and** roughly 120 lines of it are unrecorded design
decisions that resolved concerns the ledger still shows as open. That is better
than narration and worse than propagation, and it is why the substance spot-check
you asked for found what it found.

Three residues of the review process itself did creep into normative text — a new
memorial genre that says "this was missing and here is the defect it caused"
rather than "an earlier draft said X". About a dozen sites, ~25 lines
(`05:360`, `05:363-367`, `05:321`, `06:386`, `08:456-457`, `10:88-93`).
Same function as USER-27's target in different clothing. **Advisory only** — most
of them are inside table cells where the justification is load-bearing.

---

## Part 3 — substantive spot-check of the sweep's edits (the gap you named)

I checked twelve of the sweep's ~30 defining-section edits for correctness rather
than presence. Nine are correct in substance. The four failures follow, plus one
that is correct but incomplete.

### GEN-33 [Major] · spec-change — §1.5 reverses USER-17's recorded corollary, and the queue it creates has no owner outside a campaign

`01:370-381` replaces round-2's "**A second local arm does not wait.** Arriving
at a full slot it is told the slot is busy and returns" with "**A second local
arm does not make its *caller* wait** … the *run* then waits for the slot in the
background launcher (below)" and "So local batch work is **queued, not
refused**".

USER-17's corollary, as recorded in `ledger.md` and in `defining-sections-map.md`'s
USER-17 row, is "a second local arm is **refused, never parked**". Round-2's
disposition of CONC-19 says the same. This is an outcome change made by a sweep
that declared it made none, and it is not in the ledger's "Applied this sweep"
table (which records only "every hard-coded capacity 1 now reads
`local_slot_capacity`").

In fairness: round 2 was already self-contradictory — `01:528` ("a queued local
run reports `queue_position: 2`") predates this diff and sat 226 lines from the
refuse statement. Round 3 resolved the contradiction, just in the direction
opposite to the ruling, without saying so.

The concrete problem is not the direction, it is that **the queue has no owner
for a non-campaign run**:

- Every "launcher" in the spec is the **per-campaign** fan-out task created by
  `campaign_start` (`01:341`, `01:614`, `06:158`, `08:123-124`, `08:449-455`).
  A standalone `deploy_start`/`tune_start` routed local has no launcher, so
  `01:372`'s "the background launcher (below)" is a dangling reference for exactly
  the case §1.5 says it exists to serve ("refusing would make the server useless
  on a workstation").
- `queued` and `queued_reason` are defined **only** on the campaign artifact
  (`08:120-124`, `08:461-466`). A standalone queued run has nowhere to store its
  state.
- §2.4's run lifecycle is `allocate → record `launching` → spawn → CAS
  `launching → running`` (`02:383`). There is no `queued` state. A run queued for
  hours therefore sits at `launching` with no pid, which §1.5's restart
  reconciliation identifies orphans by (`(pid, create_time)`); and because
  `allocate` refuses a nonterminal generation on that output directory, a restart
  leaves the directory permanently unclaimable — the same failure `02:385-393`
  was written to prevent, arriving by a different route.
- `local_slot_timeout` (`06:65`) bounds only `W1`. A queued `W2`/`W3` has no
  expiry at all.

**Fix (needs a user ruling, because it changes USER-17's outcome):** either
(a) restore refusal for the non-campaign path and scope "queued" to campaign arms
only — which is what `08:464`'s own carve-out already says ("USER-17's … applies
to a *tool call*, not to the launcher") — or (b) keep queuing and give it the
three things it needs: a named owner for the standalone queue, a `queued` state
in §2.4's `RunRecord` lifecycle with `allocate` deferred until dequeue, and a
lease so a queued run cannot outlive a server restart. (a) is one deleted
paragraph; (b) is a design.

### GEN-34 [Major] · spec-change — `contended` is declared in one paragraph and reaches no schema

`01:330-338` introduces a new field on the probe response — "Whenever more than
one CPU-heavy holder is admitted, probe responses carry **`contended: true`**,
and such timings are **not eligible as an estimate basis**" — with the stated
reason that probe timing is what §10.5 calls "measured, not guessed" in the
estimate a human approves, and what §8.7's keep/revert decisions read from.

`grep -rn contended` over all 11 spec files returns **two hits, both in that
paragraph.** It is not in:

- §3.2's canonical `pipeline_probe` response JSON (`03:570-580`) — an implementer
  building the tool from its own section emits no such field;
- §5.3's `estimate.basis` rules — the "not eligible as an estimate basis"
  constraint has no carrier, and `basis` is a free-text string;
- §6.2 — no warning code;
- §8.7's evidence row — the keep/revert half of the claim has nowhere to live;
- §6.5 — no test.

This is precisely the defect `defining-sections-map.md` exists to catch,
committed by the sweep that wrote the map, on a field the sweep itself invented,
guarding the number quoted verbatim in `ack_prompt`. Two paths admit contention
by design (`local_slot_capacity > 1`, and the orphan rule, which explicitly
admits a `W1` beside a live `n_jobs=-1` orphan), so this is not a corner.

Major rather than Critical because contention **inflates** elapsed time, so the
estimate errs conservative at the spend gate. The real loss is §8.7: a contended
probe makes a good edit look bad and the agent reverts it.

**Fix:** add `contended: bool` to §3.2's response JSON and to the per-image
record §8.7 journals; add a rule to §5.3 that an estimate whose basis probe was
`contended` reports `basis: "contended probe — not a clean measurement"` and is
excluded from `node_hours`; one §6.5 test.

### GEN-35 [Major] · spec-change — `insert_op`'s `decision` derivation inverts on the loop's most common sequence

`03:424`: `insert_op` matches on `{kind, slot, class, params}` and derives
"an op of `class` **with those params** present in `slot` ⇒ `keep`; absent ⇒
`revert`".

The dominant §8.7 sequence is *insert an op, probe, keep it, then `set_params` to
tune it* — the loop `pipeline_patch` exists for. After that sequence the recorded
attempt is `insert_op FocusEdgePhase {k:3.0}` and the pipeline holds
`FocusEdgePhase {k:4.5}`. A compacted agent re-proposing the original edit matches
the recorded key exactly, so the advisory fires — and reports **`revert`**, when
the insertion was kept and improved. That is the same inversion GEN-29 raised for
`remove_op`, surviving inside the table written to fix it, on the more common
kind.

Second, smaller: `remove_op`'s key is `{kind, slot, class}`, so removing the first
of two `BlurGauss` ops from `ops` collides with removing the second, and
"the op **absent** from `slot` ⇒ `keep`" is wrong while the sibling remains.
`set_params` handles exactly this case with `undetermined` (`03:427`);
`remove_op` does not. `set_model` (`03:429`) has `insert_op`'s shape and the same
weakness at lower stakes.

**Fix, both one cell each:** derive `insert_op` on **class presence in the slot**,
not class-plus-params — `params` belong in the *match key* (to distinguish
attempts, which is what EPT-2 asked for) but not in the *derivation* (which asks
whether the insertion survived). Give `remove_op` the `set_params` clause:
`undetermined` when `slot` holds more than one op of the recorded class.

### GEN-36 [Major] · plan-change — §7 P8 is a line, not a task (the addendum's question 3, answered)

The spec side landed well: `07:446-476` is a real prerequisite with a verified
citation, a P6-vs-P8 distinction, and the reason materializing is the thing that
must not happen. The post-diff CONC-36 fix improved it further
(`image_manifest_digest` binds contents, not argv; collection cannot race a live
run).

**Nothing else carries it.**

1. **No plan task.** `grep -rn "P8|image_manifest|manifest flag"` over
   `plans/2026-08-14-phenotypic-mcp-server/` returns one unrelated hit.
   `phase-1b-engine-prerequisites.md:1` still reads "**Engine prerequisites
   (P3–P7)**"; README:214-215 maps P6→Task 17 and P7→Task 18 and stops. Both the
   ledger and `defining-sections-map.md`'s USER-26 row assert a plan task exists.
2. **Absent from §7's own rollout order** (`07:479-492`), which is what tells an
   implementer when a prerequisite lands and which explicitly bundles
   P2/P3/P4/P6/P7 into "MCP v1". As written, v1 is reachable without P8 while
   §10.5 says a group-filtered full deploy needs it.
3. **The plan carries a contradicting citation.** `phase-1b:960` cites
   `phenotypicCLI.py:721-730` for the single-`click.Path` `--input`;
   §7 P8 and USER-26 cite `:924-929`. I checked the file: **`:924-929` is correct**
   (`-i/--input`, `click.Path`, no `multiple=True`); `:721-730` is
   `_format_param_value`, unrelated. The P8 owner following the plan looks at the
   wrong site.

**Fix:** add the task to `phase-1b` (or a Phase 1c) with files and tests, renumber
its heading to P3–P8, add P8 to `07`'s rollout diagram, and correct
`phase-1b:960`'s citation. `10:258`'s "not in §7's P1–P7" also needs its range
updated.

### GEN-37 [Major] · spec-change — a `user_named` subset can never carry a `group_filter`, so USER-21's protection has a hole on the path USER-24 recommends

`group_filter` is a field on the `SubsetSelector` ABC. `subset_put`'s argument
table (`10:296-305`) has `name`, `parent`, `images`, `note`, `coverage`,
`overwrite`/`dry_run` — no filter, and no selector runs. So a `user_named` subset
always records `group_filter: {}`.

USER-24's own reasoning names `user_named` as a way to make per-group subsets:
*"a campaign carries exactly one `subset_id` and `user_named` is first-class, so
one subset per group gives one campaign per group."* §10:309-310 makes the same
point. But `deploy_plan {scope:"full"}` resolves to `parent ∩ group_filter`, and
an empty filter means the bare parent — so a human who hand-picks one group's
plates and promotes to full scope deploys that group's pipeline **over every
group**, which is MG-3 verbatim, the failure USER-21 was written to prevent, with
every digest check passing.

Major rather than Critical because the gate still shows the truth: `ack_prompt`
quotes the parent's image count and `node_hours`, so a human who hand-picked 24
plates and is asked to approve 480 images has a visible signal. It is a silent
mechanism failure behind a non-silent gate.

**Fix, pick one and state it:** (a) `subset_put` gains an optional
`group_filter` argument recorded on the artifact exactly as the selector path
records it — cheapest, and makes the artifact field mean one thing regardless of
provenance; or (b) `deploy_plan {scope:"full"}` emits a blocking issue on a
`user_named` subset whose `group_filter` is empty and whose image set is a strict
subset of the parent, so the agent must either supply a filter or confirm the
parent is really the target.

---

## Advisory (one line each, no argument)

- README:6 says "Twenty-four user rulings"; there are 27.
- §8.3's `campaign_approve` block never names the `campaign_arm` token kind that §5.4 defines (see GEN-27 residue).
- §2.5's `deploy.approve` example and row description were not updated for the post-diff `ack_source: "campaign_approved"` value, or for the fact that a campaign-arm approval carries no `human_response`.
- `10:482` still states a partial binding set ("binds the pipeline digest and the parent digest") without citing §5.4's table, against `05:363`'s own single-statement rule.
- §5.3's canonical response shows `estimate` with both `node_seconds` and `node_hours`; §10.5's two `deploy_plan {full}` response examples (`10:499`, `10:677`) show only `node_hours`.
- `10:258` says a change "is not in §7's P1–P7" — a closed range that P8 has since extended.
- §3.0's derivation says "`pipeline_probe` mutates nothing" while §3.2 has it writing `measurements.parquet` under `.phenotypic-mcp/probes/`; the annotation outcome is unaffected.
- `move_op`'s match key includes `order_after`, so a genuine repeat of the same move after any intervening edit does not match and the advisory stays silent.

---

## Verdict

The round-3 decisions are right and the propagation is real: all six of my
round-3 concerns are closed in the defining sections, GEN-26's prompt-source fix
in particular landed as a mechanism rather than a sentence, and the two-producer
token table and the `group_filter` propagation are better work than the concerns
that prompted them. The growth is contract text, not narration.

What the substance spot-check found is that the sweep also **took new design
decisions it did not record** (~120 lines, including fixes to two concerns the
ledger still shows `open`), and that four of those decisions carry defects of
exactly the kind the sweep was convened to eliminate — a field with no schema
(GEN-34), a derivation that inverts on the common case (GEN-35), a prerequisite
with no task (GEN-36), and a ruling outcome changed without a ruling (GEN-33).
None is Critical, all five have concrete fixes, and none needs another review
cycle to specify — GEN-33 alone needs the user, because a reviewer cannot amend
USER-17.

**VERDICT: APPROVE**, with GEN-33..37 as must-fix-before-implementation.

Sequencing: GEN-33 to the user first (it is a ruling question, and GEN-34 and the
`queued_reason` rows depend on which way it goes). GEN-35 is two table cells.
GEN-36 is a plan task plus two citation corrections. GEN-34 and GEN-37 are
independent and each about ten lines.

---

