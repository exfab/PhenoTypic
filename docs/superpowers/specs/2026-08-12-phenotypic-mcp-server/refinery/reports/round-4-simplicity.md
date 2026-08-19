# Round 4 — simplicity review

Scope: `snapshots/round-3-spec.diff` (1459) + `round-3-plan.diff` (263), against
`ledger.md` (USER-1..27) and `defining-sections-map.md`. Live spec re-checked at
`c35936606` (6299 lines on disk; the snapshot is 6237). Continue the ledger from
`SIMP-39`.

---

## The headline arithmetic — and this time the answer is no

**The growth is defining-section content. I have been wrong to expect otherwise
twice, and I am saying plainly that round 3 broke the pattern.**

Verified totals, not the addendum's: the diff is **+732 / −248 = +484** across
**54 hunks** (5753 → 6237). Attributing every hunk to its enclosing section and
classifying by `defining-sections-map.md`'s own "what counts as a defining
section" table:

| Bucket | Net | Class |
|---|---|---|
| §5.3/§5.4 arg tables, token record, binding table | **+99** | defining |
| §7 prerequisite list (P2, P3, **new P8**) | **+49** | defining |
| §6.5 test list (three new groups) | **+43** | defining |
| §3.2 per-kind `decision` table + match key | **+36** | defining |
| §8.3 arm-state / `campaign_start` / `campaign_status` tables | **+35** | defining |
| §8.2 campaign artifact schema | **+33** | defining |
| §10.3 `SubsetSelector` ABC + param tables | **+25** | defining |
| §10.2 subset artifact schema | **+13** | defining |
| §3.0 annotation derivation | **+11** | defining |
| §2.5 lineage event list | **+8** | defining |
| §6.2 / §6.3 error + limits rows | **+2** | defining |
| **defining subtotal** | **+354** | **73%** |
| §1.5 prose | +82 | explaining |
| §8.7 construction trail | +31 | explaining |
| §10.7 open questions | +18 | meta |
| §10.5 argument | +15 | explaining |
| §9.5 deploy skill | +10 | explaining |
| README | +9 | meta |
| §1.6 / §1.6.1 | −7 | — |
| §2.6 (shipped-code subsections out) | −27 | — |
| **explaining/meta subtotal** | **+131** | **27%** |

Round 2's diff was the mirror image of this: I attributed its 672 lines and found
the only net deletion of design surface in the whole round was +23. Round 3 spent
**73% of its growth on argument tables, artifact schemas, error rows, ABC field
declarations and a prerequisite section** — content an implementer builds from and
cannot re-derive. USER-24's two primitives, USER-21's binding set and USER-26's
manifest all reached the artifacts that carry them. That is the work the round was
for, and it was done.

**Two qualifications, and neither retracts the above.**

1. **§1.5's +82 is not narration — it is six new *rules* in a prose container.**
   `executors.compute` = `local_slot_capacity`; `contended: true` on probe
   responses; the ceiling as an `asyncio.Semaphore` rather than a check;
   `queued_reason` with three values; the launcher's wake/cancel/shutdown
   condition; queued-not-refused. Each is load-bearing. The defect is the
   container, not the content — see **SIMP-46**. The concurrency specialist's read
   is right and I reach it independently: §1.5 is **339 lines** (01:191–529), its
   largest subsection is 95, and *the section's own routing table and §8.3's
   arm-state table could carry four of the six directly*.

2. **The trim did not remove 161 lines of narration.** By the identical metric I
   used for SIMP-38 — paragraphs containing `earlier draft` / `first draft` /
   `was wrong` / `got wrong` / `previously named` / `once specified` — the
   memorial mass went **309 → 213 lines across 57 → 42 sites**. That is **−96,
   not −161**; the 161 counts all removed lines in the trim, including §2.6's
   relocation and rewrite churn. **~69% of SIMP-38's mass survives**, concentrated
   in §3 (67 lines / 9 sites), §10 (36 / 6), §4 (22 / 5). The map records this
   honestly ("SIMP-38's remaining pre-round-2 memorials outside those sites are
   untouched and still open"), so this is a disclosed partial, not a concealed
   one — the thing I flagged twice before.

**So: not narration returning by another route.** The habit I called out in rounds
1 and 2 did not recur. What did recur is one level up — a decision reaching the
spec and not the plan — and that is SIMP-39.

---

## Was moving the two shipped-code fixes to §7 correct?

**Yes, and the ledger alone would have been worse.** §7 is the spec's list of work
that must be done to *existing* code before the server can exist; both fixes are
exactly that. As §2.6 subsections (~30 lines) they made the server's concurrency
design look a third larger than it is; as two §7 P2 table rows (07:331, 07:332)
tagged "Phase 1b code fixes" they have an owner the plan reads. Dropping them to
the ledger only would strip that owner — the plan generates phase tasks from §7,
not from the ledger. Net reduction, right altitude, correct call.

---

## Verification of round-3 concerns

| ID | Verdict |
|---|---|
| **SIMP-31** | **FIXED (b), as USER-25.** The exemption is in §1.6.1's `W0` row (01:597), names both gates, and grounds the wait in §8.2's single-flight rule. §10.5's 9-line "would not typecheck as a design" reconstruction is gone (−9/+2, now citing USER-18), and the "One gate, one token" memorial shortened. *Advisory:* the clause names `deploy_start`, which is `W3` and needs no `W0` exemption. |
| **SIMP-32** | **HALF-FIXED, and the unfixed half is now a live table-vs-table contradiction — see SIMP-40.** The sweep took my branch (a) (`compute` = capacity) into §1.6.1 and §6.3 and left §1.5's own executor table at `1`. The "second invariant" paragraph (01:262) also still stands and 01:329 concedes it. |
| **SIMP-33** | **FULLY FIXED.** `group_filter` reached §10.2 (top level *and* `selection.params`, 10:46/52), §10.3's ABC as a declared pydantic field (10:121), §7 P3 (07:352), §6.2 as two new codes (06:58, 06:59), §2.5 lineage (02:290) and §6.5 tests. `derived_from` reached §8.2's example (08:82) and field table (08:125). This is the round's best work. |
| **SIMP-34** | **FIXED, better than I proposed** — §5.4 keys `human_response` on **token kind** (05:299), which correctly exempts campaign-arm deploys rather than all subset-scoped ones. **But three sites still teach the retired shape — SIMP-43.** |
| **SIMP-35** | **The open half landed; the half I said "is the one that matters" did not — SIMP-41.** USER-26's manifest reached §10.5, §5.4 (`image_manifest_digest`) and §7 P8. §10.6.1 is byte-untouched. |
| **SIMP-36** | **NOT FIXED — SIMP-42.** 01:468-470, 06:66 and 06:159 are unchanged. |
| **SIMP-37** | **FIXED better than I proposed, with residue — SIMP-44.** §1.6 now says "count them there, and do not restate the number here" and names the three sections that each declared themselves the *N*th piece. Deleting the number beats my "add a row and say ten". Residue: two "3 → 4 → 5 → 7 → 9" sentences survive, README still enumerates nine, and **the table gained no row for either the concurrency substrate or P8**. |
| **SIMP-38** | **PARTIAL, honestly disclosed.** 309 → 213 lines / 57 → 42 sites. The seven itemized sites went; the pre-round-2 body did not. |

---

## SIMP-39 [Critical · plan-change] — §7 P8 exists in the spec, has no plan task, no rollout slot, no §1.6 row, and is referenced by no other section. USER-26's prerequisite has no owner.

The addendum records USER-26 as "a **new §7 prerequisite and plan task**". The
spec half landed well: §7 P8 (07:446-475) is a real section with the CLI evidence
(`phenotypicCLI.py:924-929`), the reuse precedent
(`_cli_staged_slurm_worker.py:422`), and a correct P6-vs-P8 distinction.

The plan half does not exist:

- `round-3-plan.diff` contains **zero** occurrences of `P8` or `manifest`.
- `plans/…/phase-1b-engine-prerequisites.md:1` is titled **"Engine prerequisites
  (P3–P7)"**; line 10 reads **"Implements: §7 P3, P4, P5, P6, P7."** Its task list
  ends at Task 18 (P7).
- `grep -rn "P8\|image_manifest\|--manifest"` across the entire plan directory
  returns **nothing**.
- §7's own **rollout-order diagram** (07:479-490) lists P2, P3, P4, P6, P7, P5, L1,
  P1. **P8 is absent.**
- `grep "§7 P8"` across the spec: no section outside `07-prerequisites.md` cites
  it. §1.6's table cites `§7 P6` for subset staging and has no P8 row.

§7 states the consequence itself, at 07:457-458: *"Without this flag, USER-21's
full-scope semantics are unimplementable and `argv_digest` (§5.4) has nothing to
digest."* An implementer builds P2–P7, believes v1 is reachable, and finds that
`deploy_start {scope:"full"}` on a filtered subset cannot construct an argv.

**That failure mode is memorialized in the same file, forty lines above the
rollout box** (07:461-465): *"**P6 is v1-critical, not optional infrastructure.**
An earlier version of this diagram omitted it, which would have led an implementer
to build P2–P4, believe v1 was reachable, and find that every subset-scoped tool
refuses."* The identical omission was made for P8 in the round that wrote that
paragraph.

The map is the reason this slipped. Its **USER-26** row lists the defining section
as "§7's prerequisite list; §5.4's `argv_digest`; spec README's open questions" and
marks it `partial → Y`. But the map's own *"What counts as a defining section"*
table includes **"Plan decision records + Interfaces blocks | `plans/…/README.md`
D1a–D6, F1–F5; **phase task docs**"** — and the USER-26 row names no phase task
doc. **The row was closed against an incomplete list of its own defining
sections.**

**Fix (four edits, none of them prose):**
1. `phase-1b-engine-prerequisites.md` — retitle "(P3–P8)", amend line 10's
   *Implements*, add `## P8` with `### Task 19: A top-level `--image-manifest``,
   modelled on Task 17's shape, reusing `_cli_staged_slurm_worker.py:422`'s reader.
2. Add a Phase 1b exit-gate row for it.
3. Add P8 to §7's rollout diagram, on the v1 arm (it gates USER-21, which is v1).
4. Add the §1.6 table row (folded into SIMP-44's fix).

---

## SIMP-40 [Critical · spec-change] — `executors.compute` is specified twice, with two different values, 80 lines apart inside §1.5, and the retired value keeps 20 lines of rationale

The sweep applied USER-17's configurable capacity to §1.6.1 and §6.3 and to a new
§1.5 paragraph, and did not open §1.5's own executor table.

Both of these are in §1.5, and both are stated bounds:

`01:400-402` — the executor table:

> | `compute` | **1** | `W1` pipeline execution |

`01:321-328` — added **by this diff**:

> **`executors.compute.max_workers` is not an independent number — it *is*
> `local_slot_capacity`.** Fixing the pool at 1 while the slot is configurable
> makes the two disagree at any capacity above the default…

And the two bounds tables took the second: `01:615` — "**= `local_slot_capacity`**
(default 1)"; `06:157` — "= `local_slot_capacity` … not an independent knob, or the
pool and the slot disagree above the default".

So §1.5's table says 1 and §1.6.1's table says *capacity*, while `01:412` asserts
"Both numbers are stated bounds (§1.6.1)" about a number §1.6.1 states
differently.

It is not a stray cell. Three paragraphs still argue for the fixed 1:

- `01:409-411` — "Sizing `compute` at exactly one worker makes the pool a *second
  expression of the same one-probe invariant* … so the pool and the slot cannot
  disagree about how many probes are running."
- `01:425-430` — "So `executors.compute` is **the probe-dispatch slot, not a
  compute pool**: one in-flight probe *request*, whichever process runs it."
- `01:262-268` — SIMP-32's "**Second invariant** … at most one `W1` probe is in
  flight process-wide … an invariant in its own right".

All three are false under the value the bounds tables took, and `01:329-331`
concedes it in the same section: *"Whenever more than one CPU-heavy holder is
admitted, probe responses carry `contended: true` … Two paths admit one: capacity
above 1, and the orphan rule."*

**Fix — pick one, and it is three edits either way.** I recommend keeping the
configurable value, since USER-17 made capacity configuration and §1.6.1/§6.3
already took it:
- `01:402` → `| compute | = local_slot_capacity | W1 pipeline execution |`
- `01:409-411` → "Sizing `compute` at the slot's capacity makes the pool a second
  expression of the same bound rather than an independent scheduler."
- `01:425-426` → "one in-flight probe *request per slot unit*"
- **Delete `01:262-268`** (the "second invariant" paragraph, 7 lines). The property
  worth keeping — the warm worker is verified alive before dispatch, not assumed
  usable — is stated in its own next sentence and needs no invariant to hang from.

Net −7 to −9 lines and one contradiction closed.

---

## SIMP-41 [Critical · spec-change] — §10.6.1 still computes the estimate over `parent`, after USER-26 named the run set, wrote it to a manifest, and bound its digest to the token

USER-26 is right and its landing is 90% right. §10.5:585-607 resolves
`parent ∩ group_filter` in place to an `image_manifest`; §5.4:400 binds
`image_manifest_digest` — *"a digest of the file's contents, not of the argv that
names it"*; §7 P8 owns the CLI flag. The set the human approves now has a name, a
file and a digest.

§10.6.1's tier table (10:665-666) is **byte-identical to round 2**:

> | **Always** — header sweep | `W0` … | Read dimensions, bit depth, and channel
> count from **every parent image** header. Compare the distribution against the
> subset's. |
> | **Only on mismatch** — re-probe | `W1`, 2 images | Probe 2 images drawn from
> **`parent \ subset`**, chosen from the *mismatching* stratum, and re-derive the
> estimate from that timing |

Consequences, unchanged from SIMP-35 and now sharper because the right set has a
name:

1. The header sweep reads images the run will never touch. A dimensional mismatch
   confined to an excluded group triggers a `W1` re-probe — a slot acquisition
   caused by data outside the job, in the tool whose promotion to `W1` (§5.3) was
   justified by that sweep's cost.
2. The re-probe stratum is `parent \ subset`, which contains every excluded group.
   The two images that re-derive the estimate can come from data the run excludes.
3. **`estimate.node_hours` is extrapolated over the parent's image count.** §5.4:385
   calls it *"the number quoted verbatim in `ack_prompt`. This is the figure the
   human actually approves."* The token now binds `image_manifest_digest` **and**
   `estimate.node_hours` — two bound fields computed over two different sets. The
   round hardened the binding and left the quantity wrong.
4. `parent_digest` still invalidates on additions to excluded groups → false
   `plan_stale` on a filtered deploy.

**Fix — two cells and one sentence, no machinery:**
- 10:665 → "…from every image in the **effective run set** — the `image_manifest`
  where the subset carries a `group_filter`, the parent otherwise (§10.5)."
- 10:666 → stratum is `image_manifest \ subset`.
- One sentence under the table: the extrapolation multiplier is
  `len(image_manifest)`, and `parent_digest` is the digest of that set.

§10.5's paragraph then shortens to a cross-reference.

---

## SIMP-42 [Major · spec-change] — the slot lease is unchanged: keyed on a SLURM flag a local run does not have, falling through to a maximum that appears in no table

`01:468-470`, `06:66`, `06:159` are all as filed in SIMP-36.

> `06:159` — | Slot lease | `probe_timeout_s` (`W1`) / run `--time` (local
> `W2`/`W3`) | §1.5 — auto-release; `slot_lease_expired` |

`--time` is an sbatch flag. A local `W2`/`W3` is a `python -m phenotypic`
subprocess with no scheduler, so **every local holder's lease falls through to
"a configured maximum"** — a string that appears exactly twice in the spec
(01:470, 06:66) and in no limits row. `grep "lease_max\|max_lease\|slot_lease_s"`
returns nothing. The one bound governing how long the local path can be wedged has
no number, in the round whose §1.6.1 exists so that no bound "lives only inside a
paragraph".

The two structural problems stand unaddressed: expiry **releases the slot without
killing the child**, so the subprocess keeps consuming the memory the slot exists
to serialize and the server admits a second local `W2` beside it — converting a
stuck slot into the OOM §1.5 opens by citing; and it **marks a live process's
record terminal**, which §2.6's own rule forbids in the mirror direction
("Post-start CAS never resurrects a terminal record").

**Fix, unchanged and still a reduction: the lease applies to `W1` only.** §3.2's
probe worker is killable by design and `probe_timeout_s` already bounds it. A local
`W2`/`W3` releases on reap — §1.5's routing table (01:243) already says exactly
that: *"the entire subprocess lifetime, released on reap"* — and the crash case is
covered by restart reconciliation plus the refuse-not-watch orphan rule. Deletes
the undefined maximum, one branch of `slot_lease_expired`, and the terminal-status
contradiction.

---

## SIMP-43 [Major · spec-change] — `human_response`'s fix reached §5.4's table and three sites still teach "unconditionally"

05:299 is now correct and better than what I asked for. These are now stale
against it:

- `README:19` — "with `human_response` unconditionally required" (about
  `deploy_start`).
- `09:633` — "`human_response` is **required unconditionally** (USER-22)", inside
  the `phenotypic-deploy-and-verify` skill, i.e. **the text that teaches an agent
  how to call the tool**.
- `10:523` — "with **`human_response` required unconditionally**".

(§8.2:233's "on every tool that **takes a human decision**" is still true —
`campaign_approve` does — and needs no change.)

**Fix: one clause at each of the three.** "Required when the token is a plan token;
carried forward from `campaign_approve` for a campaign arm (§5.4)."

---

## SIMP-44 [Major · spec-change] — §1.6 retired the number and then restates it twice; and the table gained no row for either the concurrency substrate or P8

The rewrite is the right call — "count them there, and do not restate the number
here" (01:566-567) — and the paragraph naming the three sections that each declared
themselves the *N*th piece (01:569-577) is worth its lines. Then:

- `01:583` — "The count went **3 → 4 → 5 → 7 → 9**", seventeen lines after the
  instruction not to restate it, and `9` is wrong: the table holds **13 rows, 12
  non-mechanical**.
- `README:111-118` — says the pieces are "deliberately not re-counted here", then
  enumerates **nine** of them and closes "the count went 3 → 4 → 5 → 7 → 9".
- **No row for the concurrency substrate** — two named executors, the slot lease,
  `(pid, create_time)` identity, artifact-digest CAS, `.complete` staging markers,
  the background launcher and its admission semaphore. Rounds 2–3's largest single
  body of new machinery is absent from the cost table.
- **No row for P8**, whose own text (07:474-475) reads *"Small, and genuinely new.
  Neither §1.6's reuse inventory nor the plan's task list contained it before
  USER-26"* — the **fourth** instance of the pattern §1.6 was just rewritten to
  memorialize, created in the same round, and §1.6 was not opened.

**Fix — two rows, two deletions:**
- `| Concurrency substrate: two executors, slot lease, process identity,
  artifact-digest CAS, staging completion markers, background launcher +
  admission semaphore | **new** | §1.5, §2.6, §8.3 |`
- `| Top-level image-manifest input on the full CLI | **new** | §7 P8 |`
- Delete `01:583`'s sentence and README's `3 → 4 → 5 → 7 → 9` + nine-item
  enumeration, replacing both with a pointer to the table. Net ≈ −8.

---

## SIMP-45 [Major · needs-user-input] — USER-17's corollary was re-scoped by a propagation sweep, and the ledger's disposition for it is now false

USER-17 (`ledger.md:387-390`) is explicit and permanent:

> **Corollary — no unbounded wait.** A second local arm arriving at a full slot is
> **told the slot is busy and returns**; it does not block awaiting it.

Round 2's §1.5 implemented it verbatim ("Arriving at a full slot it is told the
slot is busy and returns"). **This diff replaced it** (01:368-382):

> **A second local arm does not make its *caller* wait.** The handler returns
> immediately with a `run_id`, a `queued` status and a `queue_position`; the *run*
> then waits for the slot in the background launcher.
> …So local batch work is **queued, not refused**.

The spec does confront the ruling rather than ignore it — `08:4292` (round-3
snapshot numbering) argues USER-17's words apply "to a *tool call*, not to the
launcher, which holds the arm and retries" — and the reading is defensible:
USER-17's stated purposes were no unbounded wait on the call and no abandoned
reservation, and both survive (the launcher lease covers the second). `queued_reason`
did not exist in round 2 at all; it is entirely this diff's.

But a **permanent ruling's mechanism was reinterpreted by a subagent sweep in a
round whose charter was propagation, not settlement**, and the map's USER-17 row
tracks only the capacity half ("every hard-coded capacity 1 now reads
`local_slot_capacity`") and marks the row `partial → Y`. Meanwhile
`ledger.md:425` still records the disposition as *"A second arm is refused, not
parked — no unbounded wait, no orphan reservation"*, which the spec now
contradicts.

**Fix: one ledger line and one user confirmation.** Amend the CONC-19 disposition
to record the re-scoping and its date, and put the reinterpretation to the user as
a one-line confirmation of USER-17 rather than leaving a sweep to have narrowed it.
I am not arguing the merits — queueing is the better design on a workstation — only
that this is the one class of change the refinery gates.

---

## SIMP-46 [Major · spec-change] — §1.5 is 339 lines, and six of this round's new rules live in its prose rather than in the two tables that should carry them

`01:191-529`. Subsections: intro + tables 32, "One arbiter" 65, **"A locally-routed
batch job suspends interactive probing" 95**, "Blocking work never blocks the event
loop" 62, "The slot primitive" 32, "Restart reconciliation" 38, "What the agent
sees" 15.

The 95-line subsection grew **+55 this round** and its title describes none of what
was added. It now carries six independent rules as prose paragraphs:
`executors.compute` = capacity; `contended: true`; the ceiling as a semaphore;
`queued_reason`'s three values; the launcher's wake/cancel/shutdown condition;
queued-not-refused. §1.5 is the section a reader must hold entirely in mind to check
any claim about the slot, and SIMP-40 is the direct consequence — a rule stated in a
paragraph at 01:321 while its table at 01:402 says otherwise, 80 lines apart in one
section.

**Fix — move four of the six into tables that already exist, no content lost:**
- `contended: true` → a column or footnote on §1.5's **routing table**, which
  already has a per-row "takes the slot?" column.
- `queued_reason` → §8.3's **arm-state table** already enumerates the three values
  (08:4287-4292); §1.5 should cite it, not restate it.
- the ceiling semaphore and the launcher's wake condition → **one row each** on the
  routing table plus §1.6.1's bounds table, which already carries
  `max_inflight_arms`.
- `executors.compute` = capacity → §1.5's executor table (this is SIMP-40's fix).

Leaves the prose to explain the *suspension* the subsection is named for. ~25–30
lines out of §1.5, zero rules lost, and the two tables become checkable in one
place.

*Related, and the same shape:* `06:156` still says `max_inflight_arms` is "checked
by the background launcher" while `01:355-357` now says the ceiling is "a queue,
not a check" and is an `asyncio.Semaphore`. The advisory I filed in round 3 about
this became a defect in this round's diff.

---

## SIMP-47 [Major · spec-change] — the README's round-by-round changelog is a ledger rendered as spec text, and it has already decayed

`README:10-34`, three paragraphs, ~26 lines. I agree with the concurrency
specialist and can demonstrate the cost rather than assert it:

- `README:19` — "with `human_response` **unconditionally** required" — contradicts
  §5.4:299, fixed **after** this diff. The changelog went stale within one commit
  of being written.
- `README:6` — "**Twenty-four** user rulings are recorded in `refinery/ledger.md`"
  — there are **27** (USER-25, 26, 27 landed in this round, and the map's own
  summary says 27).

Both are exactly what a changelog inside the artifact it describes does: it
restates a state that then moves. `ledger.md` holds this with attribution and
dates; the README does not.

**Fix: replace the three paragraphs with two lines** — "Rounds 1–3 applied; every
ruling, with rationale and date, is in `refinery/ledger.md`" plus the status line
— and make the ruling count a pointer rather than a number. ~24 lines out, no rule
lost. This is a small item and I raise it only because it has already produced two
stale claims.

---

## Standing items — has any become more dangerous under round 3?

**Yes, one. GEN-18** (`--restart`, `--slurm k=v`, `--gpu-slurm` have no `_services`
emitter) is now adjacent to **SIMP-39**: P8 adds a *fourth* CLI surface the server
must emit and cannot, and P8 has no plan task. The two together mean the argv
construction path is under-owned in both the plan and §1.6's accounting. They should
go to the user as one item, not two.

**CONC-8** is better placed than it was — it is now a §7 P2 row with a Phase 1b tag
rather than a §2.6 subsection, which is the correct disposition.

The rest (GEN-4, 5, 6, 8, 9, 10, 11, 12; FLOW-1, 2, 5; CONC-18) are unchanged in
risk by this diff.

---

## Deferral assessment (USER-16)

None of SIMP-39..47 qualifies for `deferred-to-2A`. They are a missing plan task
(39), two tables disagreeing (40), a definition (41), a scope decision on a lease
(42), three stale restatements (43), a cost-table omission (44), a ruling
re-scoping needing a user (45), and two relocations (46, 47). No observation of a
running server settles any.

---

## Advisory (one line each, no argument)

- §1.6.1's `W0` exemption names `deploy_start`, which is `W3` and needs no `W0` exemption.
- §10.6.1's tier table labels the header sweep `W0` while §1.6.1 and §5.3 class the tool `W1` because of that sweep.
- `staged_gpu` still ships in `deploy_plan`'s response beside `requires_gpu` after §5.4 states they are equal — third round filed.
- SIMP-28 unapplied for a third round: §3.0:19 and §3.3:682 still carry the MCP-only-host rationale.
- §10.1's Phase 0 diagram still reads `TRIAGE → assay + SUBSET` against README's correct biological use.
- §8.7's `decision`-is-derived and end-writing paragraphs still duplicate §3.2's verbatim (~12 lines).
- §7's rollout diagram is the only place the v1 critical path is stated; it is not cross-referenced from §1.6 or the plan README, which is how P8 fell out of both.
- Plan diff (263 lines) is a clean count/decision-record correction pass; its only defect is the omission SIMP-39 names.

---

## Net if SIMP-39..47 are accepted

- **+1 plan task, 1 rollout row, 1 exit-gate row** (SIMP-39) — the only *addition*, and it is the one that makes USER-21 implementable
- −7 to −9 in §1.5 (SIMP-40, the false invariant and the retired rationale)
- ~2 cells + 1 sentence in §10.6.1, −4 in §10.5 (SIMP-41)
- −6 in §1.5/§6.2/§6.3 (SIMP-42, the local lease branch)
- 3 one-clause edits (SIMP-43)
- +2 table rows, −8 prose (SIMP-44)
- 1 ledger line + 1 user confirmation (SIMP-45)
- ~−25 from §1.5 into two existing tables (SIMP-46)
- ~−24 from the README (SIMP-47)

**Net ≈ −75 spec lines, three contradictions closed by deletion, one unowned
prerequisite given an owner.** Small relative to the round's +484 — which is the
finding: **this round's growth was mostly earned, and what is left to cut is
mostly the round's own seams, not its substance.**

---

## SEVERITY SUMMARY

| ID | Severity | One line |
|---|---|---|
| SIMP-39 | **Critical** | §7 P8 has no plan task, no rollout slot, no §1.6 row, no inbound reference — USER-26's prerequisite is unowned and USER-21 is unimplementable without it |
| SIMP-40 | **Critical** | `executors.compute` is `1` in §1.5's table and `= local_slot_capacity` in §1.6.1's and §6.3's, with 20 lines of rationale for the retired value still standing |
| SIMP-41 | **Critical** | §10.6.1 still sweeps `parent` and re-probes `parent \ subset` after USER-26 named the run set and §5.4 bound its digest — the human approves node-hours for a superset of the run |
| SIMP-42 | Major | Slot lease for local `W2`/`W3` keyed on `--time`, a SLURM flag a local run has not; falls through to a maximum stated in no table; expiry releases the slot without killing the child |
| SIMP-43 | Major | `human_response` fixed in §5.4's table; README:19, §9.5:633, §10.5:523 still teach "unconditionally" |
| SIMP-44 | Major | §1.6 retires the number then restates it twice, and gained no row for the concurrency substrate or for P8 |
| SIMP-45 | Major · needs-user-input | USER-17's "refused, not parked" corollary re-scoped by a propagation sweep; ledger.md:425's disposition is now false |
| SIMP-46 | Major | §1.5 is 339 lines; six new rules landed in prose instead of the routing and arm-state tables that already exist |
| SIMP-47 | Major | README's round-by-round changelog has already produced two stale claims (`human_response`, "twenty-four rulings") |

**VERDICT: REVISE**

To be plain about why, given the round cap. **The propagation itself is sound and I
would approve it on its merits** — 73% of the growth is defining-section content,
USER-21/24/26 all reached their artifacts, and three of my four round-3 structural
fixes were taken and two were improved on. The REVISE is for three Criticals, all
introduced or left by *this* diff, all with fixes of one to five lines: a
prerequisite with no owner, a defining table contradicting a defining table, and an
approval figure computed over the wrong set. None needs a round 5 — they need the
edits above before execution, not another panel.
