# Round 3 — simplicity reviewer

Scope: `refinery/snapshots/round-2-spec.diff` (817 added / 145 removed, net
**+672**) + `round-2-plan.diff` (103). Spec, plan, brief, ledger only. IDs
continue the ledger's numbering from `SIMP-31`.

**Live-spec drift, noted once and not re-raised:** the snapshot is 5,753 lines;
the files on disk are **5,889** as I write. A concurrency-fix agent has added
another 136 lines since 02:17, before the round-3 panel has reported. Where a
finding below is confirmed against the *live* files I say so; nothing below is
weakened by the drift.

---

## The arithmetic — where the 672 went

Attributing every hunk to a bucket:

| Bucket | Added | Removed | Net |
|---|---|---|---|
| **Concurrency block** (§1.5 slot/executors/reconciliation · §1.6.1 bounds · §2.4 pid · §2.6 CAS/locks/registry · §6.2 codes · §6.3 limits · §6.5 tests · §8.3 launcher/cursor/recovery) | 412 | 21 | **+391** |
| §3.2 `edit_previously_tried` rewrite | 52 | 2 | +50 |
| §8.2 three elicitation rules | 35 | 0 | +35 |
| §5.3 `deploy_plan` work class + the measurement | 32 | 1 | +31 |
| §10.5 gate-move rewrite | 38 | 12 | +26 |
| §5.4 token binding set + collection | 23 | 0 | +23 |
| §3.2 `produces_columns` restoration (PROP-2) | 29 | 5 | +24 |
| **USER-24 grouping offload** (§9.3.0.2 −56/+60, §10.5 group_filter +19) | 79 | 56 | **+23** |
| §8.7 two-part journal row | 19 | 0 | +19 |
| §3.0 annotation derivation (PROP-1/CONC-27) | 23 | 6 | +17 |
| §3.2 probe output-dir keying | 10 | 0 | +10 |
| §10.3.1 staging `.complete` | 11 | 2 | +9 |
| §8.2 `human_response` unconditional | 19 | 3 | +16 |
| Rename/residue/small edits | 34 | 37 | −3 |
| §5.4 staged-GPU collapse (PROP-3) | 7 | 10 | **−3** |

**The round's only net *deletion* of design surface is USER-24's, and it is
+23, not negative.** Deleting `group_by`, per-group trait overrides and the
breakdown removed 56 lines and cost 79 to explain the removal. The two genuine
cuts in the whole diff are PROP-3 (−3) and the `promotion_*` tools' residue.

### Is the concurrency block specification or narration?

Both, in a ratio I can name. Taking the 412 added lines:

- **~150 lines are specification the design cannot do without.** The semaphore
  primitive with `call_soon_threadsafe` and one `finally`; release-first /
  record-second; `(pid, create_time)` as identity; refuse-not-watch on orphans;
  CAS on `(status, artifact_digest)`; `launching → running` never over a
  terminal; the `.complete` staging marker; the background launcher with a
  `queued` state and idempotent re-call; the three-state cursor and `study.db-wal`;
  two named executors; the bounds table; the seven error codes. None of these is
  derivable from the rest of the spec, and each one, omitted, produces a defect
  that fails silently. Keep all of it.
- **~120 lines are rationale that earns its place** — the "why" that stops a
  later reader undoing a rule whose cost is invisible (why release-first, why
  `create_time`, why refuse rather than watch).
- **~160 lines are narration.** Specifically, and these are the cuts:

| Cut | Lines | Why it is not specification |
|---|---|---|
| §2.6 **"Two locks, one order"** | ~20 | Specifies a fix to **shipped code** (`RunRegistry.allocate` at `_services/runs.py:317-337`). The ledger itself already dispositions CONC-8 as *"open · **code fix, not a spec fix** · belongs in Phase 1b"*. It is a §7 prerequisite row, not a §2.6 subsection. |
| §2.6 **"The registry is published after it is populated"** | ~10 | Same shape — a fix to `get_registry()` / `_REGISTRY`. Not a property of the MCP server. §7 P2/P3 row. |
| §1.6.1 **"On SLURM the scheduler is not admission control"** | ~9 | An environment fact (`AssocGrpCpuLimit`), already in the repo's own `CLAUDE.md` and in the ledger. One clause on the `max_inflight_arms` row carries it. |
| §1.5 **"This is specified regardless of how cancellation delivery tests out…"** | ~8 | This paragraph argues with **USER-16's deferral criterion**. It is addressed to a reviewer, not to an implementer, and it belongs in the ledger where USER-16 lives. |
| §8.7's `decision`-is-derived + end-writing counterfactual | ~12 | Verbatim duplicates of §3.2's two paragraphs on the same two rules. One site should cross-reference the other. |
| §3.2 `edit_previously_tried` (of the +50) | ~35 | See SIMP-36. |
| "An earlier draft…" memorials added this round (§1.5 ×2, §10.5, §3.2, §8.2, §8.3, §5.3, §5.4) | ~45 | See SIMP-37. |

**~130 lines of round 2's own additions are nameable cuts**, and none of them
removes a rule.

---

## Verification of my round-2 concerns

| Ledger ID | Claim | Verdict |
|---|---|---|
| **SIMP-18** (PROP-3) | staged-GPU paragraph collapsed; `staged_gpu` derived | **CONFIRMED.** §5.4:423-427 now reads "It is unconditional… `staged_gpu` is therefore derived… it is simply `requires_gpu`." Net −3 lines. The only propagation this round that shrank the spec. *Residue, advisory:* the field still ships in `deploy_plan`'s response (`05:264`) beside `requires_gpu`, so the response carries a field the spec says equals its neighbour. |
| **SIMP-19** (PROP-2) | `produces_columns` derivation restored | **CONFIRMED, and proportionate.** +24 net (asked ~12). The overage is a 6-line code block and a 4-line "do not model on `_cli_readme_generator`" warning; both are implementation traps a reader cannot re-derive. The `MeasureColor.include_XYZ` sentence duplicates the code block's point — 3 lines, advisory. Cross-refs fixed: §7 P3 now says §3.2 and the doubled `produces_columns and produces_columns` is gone. |
| **SIMP-20** (PROP-1) | §8.2/§8.3 agree; conditional signature gone | **PARTIALLY CONFIRMED — the half I raised is fixed, the half PROP-1 named is not.** The signature is fixed everywhere: `grep "required-unless"` returns exactly one hit, and it is the memorial in §8.2:203. `ack_source` is on the response in §8.2, §5.4:291, and the §2.5 lineage row. **But PROP-1's actual finding was that §8.3 "re-argues the retracted position, citing §8.2 for a claim §8.2 just disowned" — and §8.3:316-320 is untouched and still does exactly that:** "This does not authenticate anything — §8.2 is explicit that status is provenance, not security, and **an agent could fabricate the field**", 130 lines below §8.2:189 declaring "**That constraint is no longer real.**" The round-2 pass fixed the *signature* and left the *argument*. This is round 1's defect, repeated, in the section PROP-1 named. See also **SIMP-31**, which is the same section and much worse. |
| **SIMP-21/22/23** (MG-1/2/3) | USER-24 offload + USER-21 | **The direction is right and the retained primitive is the right one. The landing is not.** See **SIMP-33** and **SIMP-35**. |
| **SIMP-24** (MG-6, `"neurospora\|minimal"` keys) | moot | **CONFIRMED** — `groups` deleted, the composite key with it. |
| **SIMP-25** (PROP-4 rename inversion) | propagation pass | **MOSTLY.** `workflow.assay` → `workflow.experiment_profile`, `counts.assays` → `counts.profiles`, `kind:"assays"` → `"profiles"`, §9.3.6's table row — all fixed. The over-replacement was correctly reverted at the two prefab sites. **Residue:** §10.1's Phase 0 diagram still reads `TRIAGE → assay + SUBSET` (`10:20`) and README:27 still says "characterize the assay" — the artifact is meant in the first, the biological noun in the second, so one is wrong and one is right and they read identically. Advisory. |
| **SIMP-26** | bare-stem row goes with `_get` if (a) is chosen | Moot — USER-8 kept `_get`. |
| **SIMP-27** | §9.3.0.2 filed above §9.3.0.1 | **CONFIRMED FIXED** — order is now 9.3.0 → 9.3.0.1 → 9.3.0.2 (`09:56,111,135`). |
| **SIMP-28** | strike the MCP-only-host rationale | **NOT APPLIED.** `03:19` still carries it verbatim, and `03:682` repeats the premise. Advisory, as filed. |
| **SIMP-29** (EPT-2) | match key gains `params`, drops `index` | **CONFIRMED** — §3.2 now specifies `{kind, slot, class, params}` and excludes `index` with a reason. |
| **SIMP-30** | ~35 lines of memorials | **CONFIRMED AND UNDERSTATED BY ~9×.** See **SIMP-37**. |

---

## SIMP-31 [Critical · spec-change · needs-user-input] — USER-18's fix was applied to one of the two human gates. `campaign_approve` is still `W0` and still blocks on a human.

§1.6.1, unchanged this round:

> **`W0`** — Returns in **under one second**, and **must never block the event
> loop**.

§8.3, added this round, about `campaign_approve` — which §8.3 labels **`W0`**:

> The digest … is captured **when the elicitation prompt is built** … and
> re-checked after the human answers … **Between those two moments a human is
> reading, so the window is minutes rather than milliseconds.**

Those are the same tool. CONC-22 (Critical) was raised because `deploy_plan` was
declared `W0` while waiting on a human; USER-18 settled it by moving the
elicitation to `deploy_start` (`W3`), and §10.5:479-483 now spells out why the
old arrangement "would not typecheck as a design". **Every word of that argument
applies unchanged to `campaign_approve`, and `campaign_approve` was not touched.**
It is the other of the two gates — §10.1's own table lists exactly two — and the
round that wrote the diagnosis in prose left the second instance in place.

It is worse than `deploy_plan` was, because round 2 *added* the wait rather than
inheriting it: the double-CAS-around-elicitation paragraph is a round-2 addition
that makes the minutes-long window explicit and load-bearing.

**Two resolutions; the second is the reduction and I recommend it.**

- **(a) Move it**, as USER-18 moved deploy's — approval flips a flag, the
  elicitation fires in `campaign_start`. Symmetric, but it is a real design
  change to a settled area and it strands `campaign_approve`'s `plan_tokens`
  mint.
- **(b) State the exemption once, in §1.6.1** — the `W0` row gains: *"except
  while a human-gate elicitation is outstanding; that wait is bounded by
  single-flight (§8.2) and is not event-loop blocking."* One clause, both gates,
  no tool moves. It also **retires ~9 lines of §10.5's relocation narrative**,
  because the `W0`-violation half of USER-18's justification stops being a
  reason — its other half (point of spend; approval cannot go stale) is
  untouched and still carries the ruling.

**Needs a ruling** because (b) narrows the *stated* basis of USER-18 while
leaving its outcome intact, and I will not re-litigate a permanent ruling by
implication. Either way, §8.3 cannot keep saying `W0` while §1.6.1 says one
second.

---

## SIMP-32 [Major · spec-change] — §1.5 asserts a "second invariant" that §1.5 falsifies forty lines later, and §1.6.1 claims a consistency that cannot hold

Both texts are round-2 additions, in one section, written to satisfy two
different findings.

`01:246-254`:

> **Second invariant, stated rather than derived: at most one `W1` probe is in
> flight process-wide.** … So it is an invariant in its own right, and it is what
> the worker's liveness check answers to.

`01:291-299` (USER-17):

> `LocalComputeSlot` owns the local-OOM invariant alone, and **its capacity is
> configuration** (`local_slot_capacity`, default `1`). A workstation with memory
> to spare may set `2`.

At capacity 2 the slot admits two holders, and nothing else in §1.5 distinguishes
a `W1` holder from a `W2` one — the routing table gives both the same slot. So
the "invariant in its own right" is false in a supported configuration, and it is
false *because of a knob introduced in the same round, in the same section*.

§1.6.1 then builds on the broken invariant:

> `executors.compute` workers | **1** | … a second expression of the one-probe
> invariant

and §1.5 says the point of sizing it at 1 is "so the pool and the slot **cannot
disagree** about how many probes are running". At capacity 2 they disagree by
construction: the slot admits two probes, the pool runs one, and the second holds
an exclusive slot while queued behind the first inside a one-worker pool — a
serialization the design never states and the `probe_timeout_s` lease then
expires under.

The live spec compounds it: `01:311-318` now admits a `W1` "contended" beside an
orphan and against capacity above 1, which concedes the invariant is not one.

**The cut: delete the "second invariant" paragraph (9 lines).** The property it
wants — that §3.2's warm probe worker is verified alive before dispatch rather
than assumed usable — is worth keeping and stands on its own; it does not need an
invariant to hang from, and the one it hangs from is not true. Then either

- drop "cannot disagree" from §1.6.1 and say `compute` is sized to
  `local_slot_capacity`, or
- keep `compute` at 1 and say plainly that a capacity above 1 buys a concurrent
  local `W2`, never a concurrent probe.

Either is one sentence. What cannot stand is three round-2 paragraphs asserting a
uniqueness the fourth one sells as configurable.

---

## SIMP-33 [Major · spec-change] — USER-24's two surviving primitives were written into the section that argues they are not server mechanism, and the four sections that must carry them are untouched

This is round 1's `catalog_measurements` defect in mirror image: last round a
**cut** took a neighbour with it; this round a **keep** lost its home.

USER-24 deleted `group_by`, per-group trait overrides and the breakdown, and
retained exactly two things: `group_filter` on the `SubsetSelector` ABC, and
`derived_from` on the campaign artifact. Both are specified **only** in
§9.3.0.2 — a section in `09-responsibilities-and-skills.md` whose thesis is *"the
grouping strategy belongs to the agent, not the server"*.

Grepping the live spec for where they land:

| Where it must be recorded | State |
|---|---|
| §10.2 subset artifact schema (`selection.params` example at `10:47-50`) | **no `group_filter`** |
| §10.3 `SubsetSelector` ABC + the per-selector param tables (`10:154-169`) | **untouched**; still documents `group_key` / `allocation` / `min_per_group` only |
| §7 P3's `phenotypic/subset/` prerequisite (`07:337`) | lists the ABC and three selectors; **no `group_filter`** |
| §6.2 error codes | `group_key_not_in_metadata` only; **no code** for a filter naming a bad column or matching zero images |
| §8.2 campaign artifact / `campaign_put` response (`08:283-296`) | **no `derived_from`** |

Meanwhile two sections *consume* `group_filter` as a recorded field of the subset:

- §10.5:540-551 — "A subset selected under a `group_filter` … `deploy_plan`
  carries **the subset's** `group_filter` through to full scope".
- §5.4:283 — "`full` targets `subset.parent` — intersected with **the subset's
  `group_filter`** where it has one".

`subset.group_filter` does not exist in any artifact schema in this spec.

And **MG-1's unanswered half was inherited verbatim**: `SubsetSelector` is
`extra="forbid"`, so `group_filter` is a pydantic model change. Moving it from
`MetadataGroupSubsetSelector` to the ABC — which is the right call on the merits,
and I said so — makes that cost *identical*, not smaller, and it is now a change
to a base class every selector inherits. No prerequisite carries it.

**Assessment of the offload itself, since it was asked:** it went the right
distance and kept the right primitive. One-subset-per-group making the campaign's
aggregate cost *be* the group's cost is a genuine dissolution — it deletes a
producer, a consumer and a polling conflict at once, which is more than my
SIMP-23 proposed. The failure is entirely in the landing.

**The fix is small and is the one I proposed in round 2:** one paragraph in
**§10.3** where selectors live (the field, its type, its position before
selection, its interaction with `allocation`), one row in the §10.2 artifact
example, one line in §7 P3, one error code, one field on §8.2's campaign schema.
Roughly 12 lines total, and §9.3.0.2 then cross-references rather than
specifying — which shortens it.

---

## SIMP-34 [Major · spec-change] — "unconditional" propagated from *host capability* to *scope*, and created a third human gate inside the band the design promises is unattended

USER-22 removed a conditional: `human_response` no longer varies with whether the
host supports elicitation. §5.4 implemented it as an unconditional **argument**:

> | `human_response` | `str` | — | **Required.** … Unconditional — see §8.2;
> there is no elicited-vs-not variant of this signature |

with no `scope` qualification, on a tool whose default scope is `"subset"`. But
§10.5's own scope table says:

> | `"subset"` (default) | Requires **`plan_token`** | the subset's image list;
> **reachable from a campaign arm** |

— `plan_token` only. And §10.1's phase diagram puts subset deploys inside
*"Phase 2 EXECUTE … (agent alone, may amend, may carry deploy arms)"*, between
the two gates the section names, with README:29 promising *"the agent executes it
across parallel subagents **without you in the loop** — bounded to the subset."*

As written, every campaign arm that carries a deploy now requires a human
utterance. That is a third gate, in the one band the design sells as unattended,
introduced by a fix about host capability. §8.2's own wording is the giveaway —
it says required "on every tool that **takes a human decision**", and
`deploy_start {scope:"subset"}` does not take one.

**The fix is one word in §5.4's table:** `human_response` is required **at
`scope:"full"`**, absent at `scope:"subset"`. That is not the conditional USER-22
retired — USER-22's objection was a signature that varies with *host
configuration the agent cannot see*; scope is an argument the agent itself
supplies, so the requirement is predictable from `tools/list` and nothing about
USER-22's reasoning is reopened.

---

## SIMP-35 [Major · spec-change] — USER-21's `parent ∩ group_filter` was written into §10.5 and into nothing that computes the number the human approves

USER-21 is better than the "refuse it" I proposed — refusing would strand the
descent, and a metadata predicate over the parent needs no staging. But it
changes the run set, and the three places that compute over the run set were not
changed with it. This is the lead's least-confident item #4, and it is real.

§10.5:546 — the run is `parent ∩ group_filter`, and the token binds
`(parent_digest, group_filter)`.

§10.6.1, untouched, defines what `deploy_plan {scope:"full"}` actually does:

> Read dimensions, bit depth, and channel count from **every parent image**
> header. Compare the distribution against the subset's.
>
> Probe 2 images drawn from **`parent \ subset`**, chosen from the mismatching
> stratum, and re-derive the estimate from that timing.

So under a group filter:

1. The **header sweep** runs over images the run will never touch, and a
   dimensional mismatch in an excluded group triggers a re-probe — a `W1` slot
   acquisition (§5.3's stated reason for the whole class change) caused by data
   outside the job.
2. The **re-probe stratum** is `parent \ subset`, which includes every other
   group. The two images that re-derive the estimate can be drawn from a group
   the run excludes.
3. **`estimate.node_hours`** — which §5.4:333 now binds into the token
   *precisely because* "this is the figure the human actually approves" — is
   extrapolated over the parent's image count, not the intersection's. The human
   approves node-hours for a job several times larger than the one that runs.
4. **`parent_digest`** invalidates the token when images are added to *any*
   group, including excluded ones — a false `plan_stale` on a filtered deploy.

(3) is the one that matters: the round hardened the token's binding set to stop
an ack being spent on the wrong quantity, and left the quantity itself computed
over the wrong set.

**The fix is a definition, not machinery:** state once, in §10.6.1, that the
"parent" every tier operates over is the **effective run set** —
`parent ∩ group_filter` where one exists, the bare parent otherwise — and that
`parent_digest` is the digest of that set. Every tier, the stratum, the estimate
and the staleness check then follow from one sentence, and §10.5's paragraph
shortens to a cross-reference.

---

## SIMP-36 [Major · spec-change] — the slot lease is specified for a holder it cannot govern, and its expiry breaks the invariant the slot exists for

§1.5:451-455:

> the slot is acquired with a wall-clock lease, unconditionally. The lease is
> `probe_timeout_s` for `W1` and, for a local `W2`/`W3`, **the run's `--time` or
> a configured maximum**. On expiry the slot auto-releases and the holder's
> record is marked `slot_lease_expired`.

Three problems, all in the `W2`/`W3` half:

1. **`--time` is a SLURM flag.** A local `W2`/`W3` is a `python -m phenotypic`
   subprocess with no scheduler and no `--time`, so the lease for every local
   holder falls through to "a configured maximum" — which appears in **no**
   table. §6.3's limits row says `Slot lease | probe_timeout_s (W1) / run --time
   (local W2/W3)` and stops. The one bound governing how long the local path can
   be wedged has no number, in a round whose §1.6.1 exists to ensure "none of
   them lives only inside a paragraph".
2. **Expiry releases the slot without killing the child.** The subprocess keeps
   running and keeps consuming the memory the slot exists to serialize; the
   server then admits a second local `W2` beside it. The lease converts a stuck
   slot into an OOM — the exact failure §1.5 opens by citing.
3. **It marks a live process's record terminal**, which §2.6's own new
   subsection forbids in the mirror direction ("Post-start CAS never resurrects a
   terminal record") for the same reason: a record's status must not disagree
   with the process.

**The reduction: the lease applies to `W1` only.** That is where it is coherent —
§3.2's probe worker is a killable request/response subprocess by design, capped
at `probe_max_images`, and `probe_timeout_s` already exists as its bound. A local
`W2`/`W3` releases on reap (§1.5's routing table already says "released on reap"),
and the crash case is already covered by restart reconciliation and the
refuse-not-watch orphan rule added in the same round. This deletes the undefined
"configured maximum", one branch of `slot_lease_expired`, and the terminal-status
contradiction — and loses no recovery path that the orphan rule does not already
own.

---

## SIMP-37 [Major · spec-change] — §1.6's cost table says "nine genuinely new pieces" after the round that added the most machinery, and the table carries its own warning about exactly this

§1.6's table ends:

> The count went **3 → 4 → 5 → 7 → 9** as successive reviews traced what the
> design actually requires. The estimate was optimistic every time, and **twice
> this table went stale because a later section grew a prerequisite that was
> never carried back here** … **This table and `README.md`'s summary are part of
> the edit whenever §7 gains a prerequisite.**

Round 2 edited exactly **one** row of it — the plan-token row's wording. Round 2
also introduced, as required mechanism: two named executors with stated worker
counts; an unconditional wall-clock lease; `(pid, create_time)` capture at spawn;
content-digest CAS on every artifact read-modify-write; `.complete` markers with
temp-dir + `os.replace` staging; a module-level discovery lock; a per-campaign
background launcher task with a `queued` arm state and idempotent recovery;
`write_generation` and `launch_state` on `campaign_status`; a WAL-aware
three-state stat cursor. **None appears in the table. §7 gained no prerequisite.
The count still reads nine.**

Two of those items are not even MCP-server work — they are fixes to **shipped
code**:

- `RunRegistry.allocate`'s lock inversion (`_services/runs.py:317-337`), which
  the ledger already dispositions as *"code fix, not a spec fix · belongs in
  Phase 1b"*, and which §2.6 nonetheless specifies in ~20 lines;
- `get_registry()`'s publish-before-`discover()` ordering, ~10 lines in §2.6.

Both are §7 prerequisite rows. Specifying repairs to existing code inside the new
server's concurrency section is what makes ~430 lines look like the server's
design when a third of it is Phase 1b's.

**This is the answer to the headline question.** The spec did not grow by 672
lines because the design got 672 lines harder; it grew because the round's
findings were written where they were found rather than where they belong, and
the one section that would have caught the imbalance — the section that
explicitly says it is part of every such edit — was not opened.

**The fix is mechanical and is a reduction:** move the two shipped-code fixes to
§7 (−26 in §2.6, +4 in §7), and add the genuinely-new concurrency machinery to
§1.6's table as **one row** ("Concurrency substrate: two executors, slot lease,
process identity, artifact-digest CAS, staging completion markers, background
launcher") taking the count to ten. One row is enough; the detail is already in
§1.5/§2.6 and does not need repeating. What is not acceptable is a cost table
that says nine.

---

## SIMP-38 [Major · spec-change] — the "an earlier draft…" memorials are now ~309 lines, not ~35, and round 2 added ~45 more

SIMP-30 filed this advisory at ~35 lines. Measured across the live spec —
paragraphs containing `earlier draft` / `first draft` / `was wrong` / `got wrong`
/ `previously named` / `once specified`:

| File | Sites | Lines |
|---|---|---|
| 03-tool-catalog | 11 | 79 |
| 10-subsets-and-promotion | 8 | 52 |
| 01-architecture | 6 | 33 |
| 08-workflow-and-campaigns | 6 | 33 |
| 05-deploy-and-slurm | 5 | 25 |
| 04-tune-integration | 5 | 22 |
| README | 3 | 18 |
| 09-responsibilities-and-skills | 4 | 16 |
| 02-state-and-identity | 3 | 14 |
| 06-errors-limits-testing | 3 | 9 |
| 07-prerequisites | 3 | 8 |
| **Total** | **57** | **309** |

That is whole paragraphs, some of which also carry a rule, so the memorial text
itself is smaller — call it 150–200 lines. Round 2 contributed ~45: §1.5's
"an earlier draft also wrote 1–2 arms" and "an earlier draft had it claim the
slot", §10.5's nine-line reconstruction of the `pending_human_ack` contradiction,
§3.2's "an earlier draft omitted `params`", §8.2's "an earlier draft made it
required-unless-elicited", §8.3/§8.7's end-writing counterfactual, §5.3's "it was
assumed to be too slow", §5.4's "it was previously named … and defined nowhere".

The argument for moving them has strengthened, not merely scaled. **A memorial is
a defence against a specific reader who would otherwise re-propose the old
design** — and this spec now has a ledger with 24 permanent user rulings and 100+
provenance-locked concerns that does that job properly, with attribution and
dates, which prose memorials do not. Every one of these paragraphs is a ledger
row rendered as normative text.

There is also a concrete cost beyond volume, and this round demonstrated it
twice: **SIMP-20's verification above** found §8.3 still arguing a position §8.2
disowned, and **SIMP-31** found the `W0`-vs-human contradiction preserved at the
second gate. In both cases the retracted position survives *because it is written
as an argument rather than as a rule* — an argument reads as content and gets
carried forward; a rule reads as a claim and gets checked. Narration is not
merely bulk here; it is where the propagation failures hide.

**Recommendation, unchanged in kind and larger in scope than SIMP-30's:** move
§3.0's cut table and every "an earlier draft…" paragraph to the ledger, leaving
at most a one-clause "(superseded; see ledger)" where a reader might otherwise
re-derive the old shape. ~150 lines, no rule lost, and the sections that most
often go stale get shorter.

---

## Deferral assessment (USER-16)

None of SIMP-31..38 qualifies for `deferred-to-2A`. Each would still require a
decision after any experiment returned either result: they are contradictions
between two texts (31, 32, 34), a missing schema field (33), a definition (35),
a scope decision about a lease (36), and two editorial relocations (37, 38). No
observation of a running server settles any of them.

---

## Advisory (one line each, no argument)

- `staged_gpu` still ships in `deploy_plan`'s response beside `requires_gpu` after §5.4 states they are equal — drop one.
- §10.1's Phase 0 diagram still reads `TRIAGE → assay + SUBSET`; README:27 "characterize the assay" is the correct sense — the two now differ silently.
- SIMP-28 unapplied: §3.0:19 and §3.3:682 still carry the MCP-only-host rationale that host cannot satisfy.
- §3.2's `MeasureColor.include_XYZ` sentence duplicates the code block above it (~3 lines).
- §8.7's `decision`-is-derived and end-writing paragraphs duplicate §3.2's verbatim in argument (~12 lines); one should cross-reference.
- §6.2 gained seven codes and §6.3 four rows; no section states the resulting total or caps it, and §6.1's envelope is unchanged.
- §1.6.1's Stated bounds says the ceiling is "checked by the background launcher"; the live spec (`01:331-341`, post-snapshot) replaces the check with `asyncio.Semaphore(max_inflight_arms)` and leaves the older sentence standing — the same propagation shape, in flight now.
- `limits.max_inflight_arms = 8` and `executors.blocking = 4` are policy defaults the spec says imply nothing; both are stated twice (§1.6.1 and §6.3) with no cross-reference.
- §5.3's cold-vs-warm caveat on the header-sweep measurement is correct and should stay, as the addendum says.
- Plan diff (103 lines) is a clean count-correction pass (32→26, D1a tech stack, phase map); PROP-5's remaining items are addressed. No simplicity concern.

---

## Net if SIMP-31(b), 32, 33, 34, 35, 36, 37, 38 are accepted

- ~**26** lines move from §2.6 to §7 (4 lines) — net −22
- ~**9** lines deleted from §1.5 (the false second invariant)
- ~**9** lines retired from §10.5 (the `W0`-violation half of the relocation narrative)
- ~**12** lines added across §10.2/§10.3/§7/§6.2/§8.2 for `group_filter` + `derived_from`; §9.3.0.2 shortens by ~10
- ~**150** lines of memorials move to the ledger
- ~**35** lines from `edit_previously_tried`, ~**12** from the §8.7/§3.2 duplication
- **1 Critical contradiction closed, 3 contradictions closed by deletion, 2 dangling schema fields given a home, 1 undefined bound removed rather than numbered**

Net ≈ **−230 spec lines**, and the spec ends the round smaller than it started
it — which is the outcome round 2 claimed and did not deliver.

**VERDICT: REVISE**
