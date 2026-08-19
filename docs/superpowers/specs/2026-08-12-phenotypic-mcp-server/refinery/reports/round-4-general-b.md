# Round 4 — general reviewer (second, independent pass)

**Filename note.** A peer general-reviewer wrote `round-4-general.md` while this
pass was running; the two reports briefly merged in that file and have been
separated — `round-4-general.md` is the peer's, restored intact, and this is
mine. Our findings **overlap substantially and were reached independently**,
which is signal. Where we collide I say so and defer to their number. My new
findings are numbered **GEN-41..45** to avoid clashing with their GEN-33..37.
One shared number stands: **GEN-33** (`ack_source`), which the lead has already
committed at `a985d91ce`.

Scope: `refinery/snapshots/round-3-spec.diff` (1459) + `round-3-plan.diff` (263),
verified against the tree at **c35936606**. Ledger USER-1..27 treated as
permanent; no ruling re-litigated.

---

## Part 1 — the seven named round-3 fixes

### GEN-26 (was Critical) — `deploy_start` absorbs the human gate — **VERIFIED, with one gap**

All four claimed elements are present in `05-deploy-and-slurm.md`:

| Claim | Site | Status |
|---|---|---|
| `human_response` in §5.4 arg table | `:299` | present |
| `note` in §5.4 arg table | `:300` | present |
| `ack_source` in the response | `:305` | present |
| `scope` row rewritten (gate not on the token) | `:296` | present |
| token carries `ack_prompt` + `decision_content` | `:352-353`, `:310-320` | present |
| server renders the prompt, not the agent | `:308-320` | present, and argued |

The ordering rule at `:322-328` ("ask first, then allocate, then acquire") is a
real addition, not narration — it is the property that makes the relocation buy
anything under §1.5's suspension rule.

**The post-diff change is consistent with USER-22, not a re-litigation of it.**
`human_response` keyed on `plan_token.kind` (`:299`, argued `:453-472`) varies
with a value in the caller's own request, not with host capability — which is
what USER-22's stated objection was to. §5.4 says this itself at `:467`.

**GEN-33 (Major, already committed at `a985d91ce`) — `ack_source` had three
values in §8.2 and two in §5.4.** `08:242-244` declares
`elicited | agent_asserted | campaign_approved`; `05:305` — the defining
statement of `deploy_start`'s response, which cites §8.2 — declared only two,
while `05:464` of the same file mandates `campaign_approved`.

### GEN-27 — the token's binding set reconciled to one statement — **VERIFIED, with one leak**

`05:368-388` is now a single scope-annotated table under the explicit claim
"no other section states a binding set". §10.5's competing triple is retired
(`10:543-547` defers to it). §6.5 gained the operational test — parametrize over
the table, mutate each field, each must return `plan_stale` naming that field
(`06:322-325`). The table is now falsifiable rather than declarative.

**The second producer can satisfy it.** `05:474-506` makes the campaign-stamped
token a distinct `kind` binding a strict subset (`scope`, `pipeline_digest`,
`subset_id`, `subset_digest`, `compute`, `campaign_id`, `arm_id`), with
`run_name`/`array`/`estimate.node_hours`/`argv_digest` **absent, not null**, and
the weaker set kept off the irreversible path by `campaign_arm_scope_full` +
`"subset" only`. Consistent with §10.4's "a deploy arm targeting the full dataset
is refused" (`10:457`). No gap.

**GEN-41 (Major) — `image_manifest_digest` is a bound field that is not in the
binding table.** `05:399-411` states it "binds the resolved image set (USER-26)"
and that "`deploy_start` re-derives and compares it". It appears in **no** row of
the table at `05:375-388`, **no** field of the example record at `05:345-354`,
**no** line of §2.3's workspace tree (`02:140` has `plans/<token>.json`, not
`plans/<token>.images`), and **no** §6.5 test. The table's own claim — "Every
bound field is here" — is therefore already false, and §6.5's test ("the test
fails if a field is in the table and not in the validator") cannot catch a field
that is in neither.

*Not a criticism of the c35936606 manifest fix* — the fix is right. This is the
propagation the fix did not carry, and it is the class of gap
`defining-sections-map.md` exists to catch.

**Fix:** one table row (`image_manifest_digest` | `full` only, non-null
`group_filter` | *why*), one key in the example record, one line in §2.3's tree.

### GEN-28 — USER-24's two surviving primitives reach their artifacts — **VERIFIED (3/3), with one representation conflict**

| Primitive | Defining site | Status |
|---|---|---|
| `group_filter` on the `SubsetSelector` **ABC** | `10:114`, `10:121` (field), `10:138-158` (semantics table) | present |
| `group_filter` on the subset artifact | `10:46`, `10:52`, `10:84-93` | present |
| `derived_from` on the campaign artifact | `08:82-83` (example), `08:125` (contract row) | present |

The ABC placement is argued from `extra="forbid"` (`10:138-146`) — correct — and
the plan's `phase-1b` ABC Interfaces block carries the same field with the same
default, so the class is actually built from it. Both error codes exist
(`06:58-59`). §7 P3 item 3 (`07:352-357`) makes it ship from the first commit,
which is the right prerequisite: a later addition is a schema bump on artifacts
that by then exist on disk.

**GEN-42 (Major) — the empty `group_filter` is `{}` on the artifact and `null`
on the token, and the two are compared for equality.** Artifact side:
`Field(default_factory=dict)` (`10:121`), "or `{}` for an unfiltered subset"
(`10:85`), and §6.5 asserts "an unfiltered one writes `{}` in both" (`06:316`).
Token side: `"group_filter":null` (`05:348`), "`null` at `subset`, and `null` at
`full` for a subset with no filter" (`05:387`), `null` on both lineage rows
(`02:294`), "the map copied from the subset artifact, or `null`" (`05:286-288`).

§5.4's binding row says the value "is copied from the subset artifact's
`group_filter` (§10.2) at plan time and **re-compared at start**". A copy of `{}`
is `{}`. An implementer writing the obvious re-derivation check gets
`null != {}` and **every unfiltered full-scope deploy fails `plan_stale`** — the
common path, not a corner.

**Fix:** one sentence in §5.4 — the token normalizes an empty artifact filter to
`null` and the comparison is over the normalized form — or store `{}` on the
token too.

### GEN-29 — the per-kind `decision` derivation — **VERIFIED against the real kinds, one row still wrong**

The six kinds at `03:424-429` are exactly the six declared at `03:330-332`
(`insert_op`, `remove_op`, `move_op`, `set_params`, `set_grid`, `set_model`) —
no invented kind, none missed. The `remove_op` inversion is correctly called out
and correctly inverted. `undetermined` is a reported value that still carries
evidence (`03:431-437`), the right call for `set_params`. §6.5 gained three
matching tests (`06:330-341`). A genuine repair, not a restatement.

**GEN-43 (Major) — `remove_op`'s key and derivation break on a slot holding two
ops of one class, and unlike `set_params` there is no `undetermined` carve-out.**
`remove_op`'s key is `{kind, slot, class}` (`03:425`) with no params; its
derivation is "the op **absent** from `slot` ⇒ `keep`; present ⇒ `revert`".

Take `ops = [BlurGauss(σ=1), BlurGauss(σ=5), OtsuDetector]` and remove index 1.
The recorded key is `{remove_op, ops, BlurGauss}`. `ops` still contains a
`BlurGauss`, so the derivation reports **`revert`** for an edit that was kept —
the same inversion GEN-29 was raised about, one kind narrower. The key is also
non-discriminating: a later removal of `BlurGauss(σ=1)` canonicalizes identically
and fires a false "previously tried".

`set_params` handles the identical ambiguity (`03:427`: "`undetermined` when
`slot` holds more than one op of that class"); `remove_op` does not, and it is
worse there because it returns a confident wrong answer rather than an honest
`undetermined`.

**Fix:** the class is already resolved from `index` at record time — resolve the
removed op's `params` at the same moment. Key becomes `{kind, slot, class,
params}`, derivation becomes "an op of `class` with those params absent ⇒
`keep`". The same change makes `move_op`'s `order_after` discriminate duplicate
classes; if that is unwanted, add `set_params`' `undetermined` clause to both
rows verbatim.

*(The peer's GEN-35 finds the mirror-image problem on `insert_op`. Both rows need
the same treatment; the fixes compose.)*

### GEN-30 — `deploy_plan {scope:"full"}` is `W1` everywhere — **VERIFIED (5/5)**

| Site | Reads |
|---|---|
| §5.3 header | `05:178` — `W0` at `subset`, `W1` at `full` |
| §1.5 work-class table | `01:202` names `deploy_plan {scope:"full"}` in the `W1` row |
| §1.5 routing table | `01:283-285` "holds slot **while it computes**" |
| §1.6.1 `W1` bound | `01:598` states the parent-size bound and why the 4-image cap does not apply |
| §3.0 annotation derivation | `03:45-53` not `readOnly`, on both clauses |

No disagreeing site. `01:271-277` additionally resolves the reading that would
otherwise follow — the header sweep is `W1`-classed but is not a probe and does
not queue behind the warm worker. That distinction is load-bearing and was absent
before.

### GEN-31 — the deploy skill and the README — **VERIFIED**

`09:619-646` teaches the relocated gate: step 1 `deploy_plan` mints and says
"**No human is in this step**"; step 3 `deploy_start` is "**this call is the
gate**", with the server rendering the prompt, `human_response` required, and
"never treat a timeout or a decline as a yes". The trailing hard rule now cites
the `deploy_plan {scope:"full"}` response rather than "the promotion review"
(`09:645-647`). `README.md:24-29` describes the deleted multi-group design as
deleted and names both surviving primitives with their sections. The plan's
`instructions` string and interface-audit sibling table are corrected the same
way.

### GEN-32 — one CAS key — **VERIFIED**

`(status, artifact_digest)` at `02:335` (guard table), `02:353` (heading),
`08:366`, `08:404`, `08:501`. `write_generation` is demoted to a read hint in
both places that mention it (`08:127-133`, `08:498-503`) with the reason stated.
§6.5 has the test that fails an implementer who builds the counter as the guard
(`06:301-303`). No second CAS key survives anywhere.

### GEN-18 — **still open, and now more dangerous**

Confirmed open: §7 P2 still promotes only `to_argv` + `RunConsoleState`; no
`_services` emitter for `--restart`, `--slurm k=v` or `--gpu-slurm`. Not
re-argued.

**What changed around it:** round 3 promoted `argv_digest` from an undefined
example field to a **bound** row of the token table at *both* scopes (`05:384`),
defined as "`to_argv` plus the profile's `--slurm` pairs" (`05:395`), and §6.5
now mutates every bound field expecting a named `plan_stale` (`06:322-325`).
`deploy_start` still takes `restart: bool` (`05:302`). So the missing emitter no
longer blocks only a preview — it blocks a field carrying a human's consent and a
named acceptance test.

---

## Part 2 — substantive spot-check of the propagation sweep

The lead's first question: the ~30 defining-section edits are present; are they
*correct*? I checked the eight with the most consequence. Six are correct in
substance. Two produced GEN-41 and GEN-42 above. One more, below, is the largest
thing I found this round and I do not see it in the peer's report.

### GEN-44 (Major) — the USER-17 propagation makes `executors.compute` configurable and thereby falsifies the one-probe invariant it was meant to restate

This is the sweep's own edit, and it is where propagating a ruling into a bounds
table without re-reading the section's other invariants shows.

Four statements now stand in §1.5 and §3.2 simultaneously:

1. `01:261-264` — "**at most one `W1` probe is in flight process-wide** … §3.2's
   single warm probe worker is written assuming it." Stated as an invariant in
   its own right, precisely so it cannot vanish with a refactor.
2. `01:408-411` — "Sizing `compute` at **exactly one worker** makes the pool a
   *second expression of the same one-probe invariant*."
3. `01:321-322` (**added by the sweep**) — "`executors.compute.max_workers` is
   not an independent number — it **is** `local_slot_capacity`", i.e.
   configurable, and USER-17 explicitly sanctions 2 on a large node.
4. `01:425-427` — "`executors.compute` is the probe-dispatch slot … **one**
   in-flight probe *request*."

(3) is what the **defining** tables now carry: `01:615` and `06:157` both read
`executors.compute` workers `= local_slot_capacity`. So at
`local_slot_capacity = 2` the bounds tables admit two concurrent probe requests
into §3.2's single worker, and three §3.2 statements become false at once:
`03:535` "the server owns **one** probe worker subprocess", `03:558` "Holding the
`LocalComputeSlot` == the worker is busy", and `03:765` "probes serialize behind
one worker, so **peak memory is one probe, not three**" — which is the memory
argument that section rests on.

The spec half-knows this: `01:330-338`'s `contended: true` rule exists *because*
capacity above 1 is admitted. It addresses the measurement-quality consequence
and not the worker-exclusivity one.

**Fix (pick one, one or two sentences either way):**

- **Keep one worker.** State that `executors.compute` sizes CPU-heavy
  *admission* and that probe **dispatch** is separately serialized by the warm
  worker regardless of capacity; correct `03:558` to "holding the slot does not
  imply the worker is free at capacity > 1" and re-word `03:765`.
- **Or scale the worker.** One warm probe worker **per compute slot**; retire
  "the server owns one probe worker" for "one per slot", and §3.2's memory claim
  becomes "peak is `local_slot_capacity` probes".

Either is small. Leaving both readings in place is not, because §3.2's
exclusivity is exactly what `01:263` says is being protected.

### The six that check out

- **USER-1 → §1.6.1's `W1` row** (`01:598`). Correct and necessary: the 4-image
  cap genuinely could not bind a 480-image header sweep, and the row now says so
  in the table rather than in §5.3's prose.
- **USER-9 → §6.2's `edit_previously_tried` row** (`06:33`). Correctly rewritten
  to name the per-kind key, the derived `decision` with all four values, and
  "`undetermined` still fires with the evidence" — the row now matches §3.2
  instead of summarizing an older §3.2.
- **USER-18 → §2.5's lineage rows** (`02:290-291`). Both deploy rows carry
  `group_filter`, and the justification at `02:294-299` is right: `scope:"full"`
  + `subset_id` would otherwise read as "the whole parent", which USER-21 says it
  is not.
- **USER-21 → §10.2's artifact** (`10:84-93`). The "byte-indistinguishable"
  argument for recording the filter at top level *and* in `selection.params` is
  correct, and is the reason the field cannot live only in the selector params.
- **concurrency block → §8.2's schema** (`08:82-126`). `state`, `queued_reason`,
  `study_id`, `launcher`, `derived_from` now exist as fields with a contract
  table, and §8.3's arm-state table (`08:461-467`) is their *meaning* rather than
  a second declaration. The three-way `queued_reason` including `"local_slot"` is
  genuinely needed — at `local_slot_capacity=1` a three-arm local campaign parks
  two arms in a state that is neither of the other two reasons.
- **USER-26 → §7 P8** (`07:446-470`). The distinction from P6 is real and
  correctly stated (P6 materializes a directory for **subset** scope; P8 passes a
  list for **full** scope, where materializing is the thing that must not
  happen). Its two code citations are **verified against the shipped tree**:
  `phenotypicCLI.py:922-931` is a single `click.Path` with no `multiple=True`
  (the spec's `:924-929` resolves inside it), and
  `_cli_staged_slurm_worker.py:422` is `parser.add_argument("--manifest", …)`.

### Two findings I reached independently that the peer also filed

- **§7 P8 has no plan task** — peer's **GEN-36**. Independently confirmed:
  `phase-1b-engine-prerequisites.md` is titled "P3–P7", its "Implements:" line
  stops at P7, the README rollout ends at `P7 → Task 18`, and grepping the whole
  plan directory for `P8` or a manifest flag returns one hit — P6's verification
  note that no manifest flag exists. Meanwhile §7 P8 and `10:604` both assert
  "a new §7 prerequisite **and a new plan task**".
  *Adds one fact:* `phase-1b:960` supports the identical claim citing
  `phenotypicCLI.py:721-730`, which in the shipped tree is `_format_slurm_time`
  display code. The spec's `:924-929` is correct; the plan's citation is stale
  and should be corrected in the same edit.
- **`contended` reaches no schema** — peer's **GEN-34**. Independently
  confirmed, and I would add the enforcement site they may not have named:
  `01:330-338`'s normative half is "such timings **are not eligible as an
  estimate basis**", and the only place that can be enforced is §5.3's
  `estimate.basis` / `no_probe_evidence` machinery (`05:255-275`), which has a
  warning for *no* probe and none for *a contended* probe. So the corrupted
  timing silently becomes the `estimate.node_hours` a human approves — the exact
  corruption `01:336-338` says it prevents. Suggest the fix include one §6.2
  warning row (`estimate_basis_contended`) alongside the response field.

---

## Part 3 — the growth: 5,753 → 6,237 (now 6,313)

Measured on the diff: **+733 / −249**, net +484. Classified by line shape:

| Shape | Added | Read as |
|---|---|---|
| Table rows (`^+\|`) | 77 | defining |
| JSON schema / record lines | 34 | defining |
| Bullets (mostly §6.5 test cases) | 68 | defining |
| Blank | 69 | — |
| **Prose** | **~485** | the question |

Two-thirds of the growth is prose. **It is not narration returning by another
route** — I checked for the tell USER-27 named and it is absent: there is no new
"an earlier draft…" reconstruction anywhere in the added text. The sweep removed
those (the −249 is almost entirely them) and did not reintroduce the habit.

**But the shape of the problem changed rather than went away.** The ~485 lines
divide in two:

- **~380 lines are justification bolted to a rule that does sit in a table
  beside it** — §3.2's "'is the op still there' is the rule for exactly one of
  six kinds" ahead of the per-kind table; §5.4's "the prompt is rendered by the
  server" ahead of `ack_prompt`/`decision_content`; §10.2's
  "byte-indistinguishable" ahead of the `group_filter` bullet; §8.3's
  launcher-lease derivation ahead of §8.2's field table. This is USER-27's
  "keep the ~120 lines of rationale" category and it earns its place. It is
  *more* than USER-27 anticipated keeping, but the category is right.

- **~105 lines are new normative rules that never got a table row.** This is the
  real answer: `contended: true` and its estimate-eligibility rule (peer GEN-34 /
  ~11 lines), `image_manifest_digest` and its `.images` file (GEN-41, ~15 lines),
  `ack_source: "campaign_approved"` (GEN-33, ~20 lines), the `executors.compute`
  identity whose consequence for §3.2 was never followed through (GEN-44,
  ~11 lines), and the empty-filter representation split (GEN-42, implicit across
  four sites).

**So: the growth is defining-section content that had to exist — and the sweep
partly re-committed, at ~105 lines and one-tenth the scale, the exact defect it
was built to fix.** A rule was written into the paragraph that argues it and did
not reach the table, the schema, the error row or the test. That is not a reason
to distrust the round; it is a reason to run `defining-sections-map.md`'s own
closing instruction — "for every changed paragraph, name the argument table,
artifact schema, error row and plan decision record it implies, and grep each
one" — over **this** diff before execution starts. Four of my five findings, and
both of the peer's I confirmed, are what that pass would have returned.

---

## Concerns

| ID | Sev | Concern | Fix |
|---|---|---|---|
| **GEN-33** | Major *(committed `a985d91ce`)* | `ack_source` 3-valued in §8.2 (`08:242-244`), 2-valued in §5.4's response line (`05:305`), while `05:464` mandates the third | applied |
| **GEN-41** | Major | `image_manifest_digest` is stated to bind (`05:399-411`) but is in no row of the binding table, no field of the example record, no line of §2.3's tree, and no §6.5 test — falsifying "every bound field is here" | one table row, one record key, one tree line |
| **GEN-42** | Major | Empty `group_filter` is `{}` on the artifact (`10:85`, `10:121`, `06:316`) and `null` on the token (`05:348`, `05:387`), and §5.4 says the token value is *copied and re-compared*. Every unfiltered full-scope deploy fails `plan_stale` | state the `{}` → `null` normalization once, or store `{}` on the token |
| **GEN-43** | Major | `remove_op`'s key `{kind, slot, class}` and its "absent ⇒ keep" derivation both break when a slot holds two ops of one class — a confident inversion where `set_params` reports `undetermined` | resolve the removed op's `params` at record time alongside its `class` |
| **GEN-44** | Major | `executors.compute = local_slot_capacity` in both bounds tables (`01:615`, `06:157`) falsifies §1.5's stated one-probe invariant and three §3.2 statements, including its peak-memory claim, at any capacity > 1 | serialize probe dispatch on the warm worker regardless of capacity, **or** one worker per slot; correct `03:558` and `03:765` |
| **GEN-45** | Advisory | README's header says 24 rulings (there are 27) and `human_response` "unconditionally required" (now keyed on token kind) | — |

**Independently confirmed, filed by the peer:** GEN-34 (`contended` reaches no
schema — I add the §5.3 enforcement site and a suggested §6.2 warning row) and
GEN-36 (§7 P8 has no plan task — I add that `phase-1b:960`'s
`phenotypicCLI.py:721-730` citation is stale where the spec's `:924-929` is
correct).

**Standing open items, unchanged and not re-argued:** GEN-4, 5, 6, 8, 9, 10, 11,
12; FLOW-1, 2, 5; CONC-18. **GEN-18** is confirmed open and has become *more*
dangerous, for the reason given in Part 1. **CONC-8** correctly landed as a §7 P2
row with every clause intact.

## Verdict

All seven round-3 fixes I was asked to verify are **present and substantively
correct**: GEN-26 (four elements, plus the ask-before-allocate ordering rule),
GEN-27 (one canonical table, and the second producer can satisfy it via a
distinct `kind` kept off the full-scope path), GEN-28 (3/3, and the plan's ABC
block carries the field), GEN-29 (six kinds, matching the six the spec actually
supports, with the inversion correctly stated), GEN-30 (5/5 sites agree),
GEN-31 (skill and README both rewritten), GEN-32 (one CAS key everywhere).
The decisions were right in round 3; the propagation is what this diff is, and it
delivered that.

None of my concerns challenges a ruling or a decision. All are propagation
residue of one class — a rule that reached the paragraph and not the table — each
with a one-to-two-line fix stated above. That is an execution checklist, not a
redesign.

**VERDICT: APPROVE** — conditional on GEN-41..44 (plus the peer's GEN-34 and
GEN-36) being applied before Phase 2A opens. GEN-42, GEN-43 and GEN-44 are the
three that produce wrong behaviour rather than a missing line; GEN-36 is the one
with no owner at all.
