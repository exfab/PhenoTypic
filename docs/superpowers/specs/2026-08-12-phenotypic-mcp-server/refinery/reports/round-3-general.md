# Round 3 — general reviewer

**Scope:** `snapshots/round-2-spec.diff` (1367 lines), `round-2-plan.diff` (103),
verified against the current spec files and the OME-Zarr worktree design.
**Severity ratchet observed:** Critical + Major argued; everything else is a
one-line Advisory.

**Headline:** the round-2 rulings are sound and the reasoning behind them
survives. **The propagation defect did recur** — not in the sections that were
rewritten, but in the *contracts* the rewrites depend on. Round 1's misses were
stale prose still arguing the old position. Round 2's misses are a different and
worse shape: **three of the round's decisions exist only in the paragraph that
argues for them, and never reached the argument table, the artifact schema, or
the base class that has to carry them.** Two are on the safety-critical path.

---

## Part 1 — verification of my round-2 concerns

### GEN-13 → USER-24 (multi-group offload): **the deletion did take something load-bearing.** See GEN-28.
The reasoning is right and §9.3.0.2 is now the best-argued section in the
document. But USER-24 left exactly two deliverables behind — `group_filter` on
the `SubsetSelector` ABC, and `derived_from` on the campaign artifact — and
**neither reached the section that defines the thing it lives on.** Same shape as
the `catalog_measurements` cut taking `header_scheme()` with it, and the same
cause: the surviving primitive was written into the section doing the deleting.

### GEN-14 → USER-21 (`scope:"full"` = parent ∩ group_filter): **coherent as a rule, unreachable as specified.** See GEN-27, GEN-28.
The semantics are right and §10.5's new block argues them well. But §10.5 reads
`group_filter` off "the subset", and no subset artifact or selector in §10.2/§10.3
has the field; and it states a token binding that §5.4's *self-declared exhaustive*
binding table does not contain.

### GEN-15 → PF-1..4 (token binding set + approval lineage): **partially resolved.**
`run_name`, `array`, `estimate.node_hours` and a defined `argv_digest` are real
improvements, the `deploy.approve` lineage event closes PF-2 cleanly, and token GC
closes PF-4. **PF-3 is not resolved** (GEN-26) and the binding set is now stated
four incompatible ways (GEN-27).

### GEN-16 → `deploy_plan` W0/W1: **the reasoning survived; the propagation did not.** See GEN-30.
The measurement is real, the script exists, and "the re-probe is the reason, not
the sweep" is the correct conclusion. It was then written into one section header
and nowhere else.

### GEN-17 → EPT-1 (`decision` has no writer): **the fix works for one of six edit kinds, and re-introduces the collision EPT-2 was fixed to remove.** See GEN-29. This is a *new* Major.

### GEN-18 → deploy-argument emitters: **confirmed still open, not re-argued.**
`to_argv`/`RunConsoleState` remain the only named mechanism (§5.4:370, §7:307);
`deploy_start` still takes `restart`; no `--restart`, `--overwrite`, `--slurm k=v`
or `--gpu-slurm` emitter exists in P2. OME-Zarr now locks a further flag
(`--durable-writes`, its decision #13) onto the same surface.

### GEN-19/20/21/22 → the propagation pass: **real, and it introduced nothing.**
Verified by grep across all 11 documents. `promotion_request`/`promotion_approve`
survive only as named history in three places, all correctly framed as retired.
`pending_human_ack` survives only on `campaign_put`, where it is still correct.
`counts.assays` → `counts.profiles`, `kind:"assays"` → `"profiles"`, the CWD
default, the plan's tool counts (26 everywhere; 2A/2B/2C now sums to exactly 26),
Tech Stack → `fastmcp` 3.x, D3's skill name, and the `produces_columns`
restoration are all correctly applied. PROP-2's botched `"produces_columns and
produces_columns"` substitution is fixed in both §3.2 and §7 P3, and the §7 P3
cross-reference now points at §3.2 rather than the deleted §3.1 text. The
SIMP-25 carve-out (restoring the biological noun) was applied at 08:92 and
08:701. **No fix in the pass was found to have introduced a regression.** Two
sites it did not reach are Advisory below.

---

## Concerns

### GEN-26 [**Critical**] · spec-change — `deploy_start` never absorbed the gate it now owns

USER-18 moved the human gate to `deploy_start` and USER-22 made `human_response`
unconditionally required. Both landed in §8.2 and §10.5. **§5.4, the section that
specifies `deploy_start`, was not touched.** Three consequences, in the one place
where the server spends somebody else's cluster:

1. **`human_response` is not a parameter of `deploy_start`.** §5.4's argument
   table lists `scope`, `plan_token`, `resume`, `retry_failures`, `restart`.
   `grep -n human_response 05-deploy-and-slurm.md` returns nothing. PF-3 —
   *"`human_response`, the mandatory fallback, has no parameter on either deploy
   tool"* — was dispositioned as settled by USER-22 and is still true.
2. **§5.4 re-argues the retired position, citing the section that retired it.**
   Its `scope` row reads: *"its `plan_token` must have been minted at
   `scope:"full"` **with the human ack recorded** (§10.5)"*. §10.5's own scope
   table now reads *"minted at `scope:"full"`, **plus the ack at
   `deploy_start`**"*. This is PROP-1 verbatim — two sections stating opposite
   contracts, the untouched one citing the updated one as its authority.
3. **Nothing says where the elicitation text comes from.** §10.5 mints
   `ack_prompt` at `deploy_plan` and calls it *"text to show"*; the token record
   in §5.4 has no `ack_prompt` field, so at `deploy_start` the server cannot
   reproduce it. Either the agent passes the prompt text — in which case the
   numbers the human reads are agent-chosen, which is the single thing
   elicitation was adopted to prevent (§8.2: *"a host-rendered form comes from
   the user's keyboard rather than the agent's token stream"*) — or the server
   re-renders from the token, which binds `run_name`/`array`/`node_hours` but
   **not** the subset score, held-out gap or coverage warnings that §10.5 calls
   the decision content. Neither is written down.

Critical rather than Major because (1) makes the gate unimplementable as
specified, (2) means an implementer reading §5 builds the retired design, and (3)
is a fabrication surface on the irreversible spend. §8.2's three new elicitation
rules ("artifact id first, single-flight, no non-answer approves") are declared to
bind this gate, and none of them can be satisfied by a handler whose prompt source
is undefined.

**Fix:** add `human_response: str` (required) and `note?` to §5.4's argument
table; return `ack_source`; correct the `scope` row; and state that the server
renders the elicitation message from the token record plus the plan artifact it
names, never from an agent-supplied string.

---

### GEN-27 [Major] · spec-change — the plan token's binding set is stated four times, four different ways

The token now carries a human's consent, so its binding set is the security
boundary. It is specified in four places that do not agree:

| Where | Binding set stated |
|---|---|
| §5.4 "Plan-then-submit is mandatory" | `(pipeline digest, images digest, compute)` |
| §5.4 "**The binding set is exhaustive**" + record | `scope, pipeline_digest, subset_id, subset_digest, compute, run_name, array, estimate.node_hours, argv_digest` |
| §5.4 two paragraphs later | *"A `scope:"full"` token **additionally** binds `parent_digest`"* |
| §10.5:514 | `(pipeline digest, parent digest, scope)` |
| §10.5:549 | `(parent_digest, group_filter)` |

`parent_digest` and `group_filter` appear in neither the record example nor the
table that calls itself exhaustive. `images digest` names a field nothing else
uses. §10.5:514 — unchanged context in the diff — sits ~35 lines above §10.5:549,
which contradicts it; this is the round-1 defect recurring **inside a single
section**, both statements in §10.5.

One further mismatch: the token binds `estimate.node_hours`, and §5.3's canonical
`deploy_plan` response returns `estimate.node_seconds`. The bound field is not in
the producing tool's response schema.

**Fix:** one table, in §5.4, listing every bound field with its scope condition
(`parent_digest` and `group_filter` at `full` only); delete the two prose
restatements and have §10.5 cite it; reconcile the estimate's unit.

---

### GEN-28 [Major] · spec-change — USER-24's two surviving primitives never reached the artifacts that carry them

USER-24 deleted a design and kept exactly two things. Neither reached its home.

**`group_filter`** — "a `{column: value}` map on the `SubsetSelector` **ABC**".
It appears in §9.3.0.2 (which defines it) and §10.5 (which consumes it). It does
not appear in:
- **§10.3's `SubsetSelector` base class**, whose shown model is `n`, `seed`, and
  `model_config = ConfigDict(extra="forbid")`. This is MG-1's original point
  unchanged — `extra="forbid"` means the field cannot even arrive as an extra key,
  so a selector cannot accept a filter until the ABC declares it.
- **§10.2's subset artifact**, so a filtered subset records nothing distinguishing
  it from an unfiltered one. §10.5 says *"`deploy_plan` carries **the subset's**
  `group_filter` through to full scope"* — reading a field off an artifact that
  has no such field. The token then binds it (§10.5:549), so the safety property
  USER-21 bought ("an ack given for one group cannot be spent on another's
  images") rests on a value with no defined storage location.
- **§7 P3 item 3**, the prerequisite that builds `phenotypic/subset/` — it lists
  the ABC and three selectors, and knows nothing about a filter.
- **§5.3's `deploy_plan` argument table and response**, neither of which mentions
  it, so an agent cannot see at plan time which images a full-scope run will touch.

**`derived_from: {campaign_id, reason}`** — the "one breadcrumb" that keeps a
per-group descent reconstructible. It appears once, at `09-responsibilities-and-skills.md:189`,
in a section about the skill/server boundary. **§8.2's campaign artifact JSON does
not have it.** §9 is not where campaign fields are defined.

This is the answer to the addendum's question 3: yes, the offload took something
load-bearing with it, in the same shape as the `catalog_measurements` cut. The
difference is that this time it is not a derivation that went missing but the
*declaration* of the only two mechanisms USER-24 kept.

**Fix:** declare `group_filter: dict[str, str] = {}` on §10.3's ABC with the
before-any-selector-runs semantics; add it to §10.2's artifact and §7 P3's task;
add `derived_from` to §8.2's schema.

---

### GEN-29 [Major] · spec-change — `decision` derivation and the canonical match key are written for one of six edit kinds

§3.2 declares six edit kinds:
`insert_op {slot,index,class,params}`, `remove_op {slot,index}`,
`move_op {slot,from,to}`, `set_params {slot,index,params,merge}`,
`set_grid {nrows,ncols}`, `set_model {class,params|null}`.

Round 2 added two rules, both written against `insert_op` only.

**Rule 1 — the derivation** (*"if the op is still there, the step was kept; if it
is gone, it was reverted"*):
- **`remove_op`: inverted.** A kept removal leaves the op absent, which the rule
  reads as *reverted*; a reverted removal restores the op, which the rule reads as
  *kept*. The advisory tells a compacted agent the exact opposite of what happened.
- **`move_op`: always "kept".** The op is present after a move whether or not the
  move was undone.
- **`set_params`: always "kept".** The op is present whether or not the parameters
  were reverted — and `set_params` is the dominant edit kind in a tuning loop, the
  reason EPT-2 was raised. Deciding it correctly requires comparing *current*
  params to recorded params, which the spec does not say, and `merge=true` makes
  even that ambiguous.
- **`set_grid` / `set_model`: no referent.** Neither names an op in a slot list,
  so "is the op still there" has nothing to evaluate.

**Rule 2 — the match key** (*"the full canonical edit… the recorded block is
`{kind, slot, class, params}`"*, with `index` explicitly excluded):
- `remove_op` carries neither `class` nor `params`, and its only discriminator is
  `index`, which the rule excludes. So **every `remove_op` on a slot canonicalizes
  to one edit** — removing `BlurGauss` and removing `OtsuDetector` from `ops` are
  the same recorded attempt.
- `move_op` likewise carries neither, and both its coordinates are indices. Every
  move on a slot is one edit.
- `set_grid` and `set_model` have no `slot` at all.

This is the collision EPT-2 was raised about and USER-approved to fix,
re-introduced for three of the six kinds by the fix itself. It is a *new* Major
originating in this round's diff.

**Fix:** define the canonical key and the derivation per kind — a small table.
For `remove_op` the derivation inverts; for `set_params` it compares recorded
params against current; `move_op` and `set_grid`/`set_model` either get an
explicit rule or are declared out of scope for the advisory (which is defensible —
the advisory's stated purpose is re-tried *additions*).

---

### GEN-30 [Major] · spec-change — `deploy_plan {scope:"full"}` was promoted to `W1` in one section header and nowhere else

The `W0`→`W1` correction is right. It landed in §5.3's heading and body. Four
places that define what `W1` *means* were not updated, and each now says something
false about the tool:

1. **§1.5's work-class table:** `W1 probe | Bounded image compute, interactive
   latency | apply a pipeline to 1–N images and return measurements + benchmark`.
   `deploy_plan {full}` applies no pipeline in the tier-1 path and returns no
   measurements.
2. **§1.5's routing table:** `W1 | in-process, **holds slot** | in-process,
   **holds slot**` — unconditional. §5.3 says the slot is the reason for the
   class, but the tier-1-match path (the common case: *"headers match… no re-probe
   needed"*) needs no slot. Taken literally, every full-scope preview blocks the
   agent's exploration loop and can fail `local_slot_timeout` behind a local arm.
3. **§1.6.1's `W1` row:** *"Bounded by `limits.probe_max_images` (default 4)"* —
   the sweep reads 460+ image headers, and §6.2's `probe_cap_exceeded` fires above
   4. The NFR that justifies the class is stated in units the tool violates by two
   orders of magnitude.
4. **§1.5's second invariant** (*"at most one `W1` probe is in flight
   process-wide"*, with §3.2's single warm probe worker written against it). If
   `deploy_plan {full}` is `W1`, does it dispatch through the probe worker
   subprocess? Its tier-1 sweep is header I/O, not `ImagePipeline.apply()`; §3.2's
   worker has no contract for it.

The clean resolution is probably to say `W1` applies **only to the re-probe
escalation** — tier 1 is executor-offloaded `W0` I/O, the escalation takes the
slot — which is what §5.3's own argument implies and what the measurement supports.
Either way one of these five statements has to move.

---

### GEN-31 [Major] · spec-change — §9.5's deploy skill still teaches the retired gate, and README still advertises the deleted multi-group design

§9 is the document Phase 2C's four bundled skills are written from. This is the
GEN-1 failure mode — a downstream artifact built to a superseded contract —
recurring against the round's own decisions.

`phenotypic-deploy-and-verify` (§9.5, untouched by the diff):
- Step 2: *"**Show the human that response and wait.** This is the gate"* — the
  gate is now `deploy_start`'s elicitation, not a human reading `deploy_plan`'s
  response.
- Step 3: *"using the plan token **that approval minted**"* — approval mints
  nothing now; `deploy_plan` mints the token with no human in the path.
- No step mentions `human_response`, which is a required parameter of the call the
  skill instructs. A skill that teaches an invalid call is worse than no skill.
- *"a coverage warning on the **promotion review**"* — promotion residue in the
  normative text of a skill.

`README.md` (untouched):
- *"§9.3 gained **multi-group experiments** (`group_by` + per-group trait
  overrides)"* — USER-24 deleted precisely those two things. The document a reader
  opens first advertises the design the spec no longer contains.
- Status block still reads *"refinery round 1 applied… Fifteen user rulings"*.

---

### GEN-32 [Major] · spec-change — §8.3 and §2.6 name two different CAS keys for the same artifact

Both sections were written this round.

- **§2.6:** *"every mutation of an artifact CASes on the pair: the expected
  `status` **and** the content digest of the bytes the caller read"* — the fix
  USER-approved as covering amendment-reversion, double-launch and the widened
  approval window in one.
- **§8.3:** *"**`write_generation`** — the artifact's own write counter,
  incremented on every CAS… and it is **the value a subsequent mutation CASes
  against**."*

A digest CAS and a generation-counter CAS are different mechanisms with different
failure modes, and `write_generation` appears nowhere else in the spec — not in
§2.6's guard table, not in §8.2's campaign schema (so it has no storage), and not
in §6.5's `artifact_changed` test. Either it is the CAS key and §2.6 is wrong, or
it is a read-staleness hint and §8.3 overstates it.

This is the addendum's question 2 answered from the outside: the concurrency
block's ~430 lines are internally coherent, but at least one of its boundaries
with a section written by a different hand in the same round does not close.
§2.6's rule is the better one (it survives an artifact edited by a peer server,
which a counter the reader never saw does not); §8.3 should demote
`write_generation` to a reported hint or drop it.

---

## Advisory

- §1.5 states the slot is *"a process-wide semaphore of capacity 1"* two paragraphs before saying capacity is `local_slot_capacity`, configuration.
- §1.5's `W0` work-class row still lists *"diff two pipelines"* as an example; `pipeline_diff` was cut by USER-8.
- §3.0's derivation excludes `deploy_plan` from `readOnlyHint` on cost grounds only; it also writes a token record at both scopes, which the rule's first clause already catches.
- MCP annotations are static per tool in `tools/list`, but the derivation is keyed on `scope`; harmless here (both scopes land non-readOnly) and worth one sentence.
- §5.3's response returns `estimate.node_seconds`; §10.5's returns `node_hours`; the token binds `estimate.node_hours`.
- §2.3's tree still shows `results/<dataset>/{hdf,measurements}/` — now the MCP spec's **only** OME-Zarr-coupled line (see below).
- `10-subsets-and-promotion.md:20` (`Phase 0 TRIAGE → assay + SUBSET`) and `08-workflow-and-campaigns.md:13` (*"Characterize the ASSAY"*) name the artifact, not the biological noun, so the SIMP-25 carve-out does not cover them.
- §8.2's campaign JSON example predates the `queued`/`study_id` arm states §8.3 now defines; §8.3's prose covers them, the schema does not show them.
- SIMP-30's "an earlier draft…" memorial count has grown well past 35 lines this round (§1.5 ×3, §2.6, §3.0, §3.2 ×2, §5.3, §5.4, §8.2, §9.3.0.2, §10.3.1, §10.5) — I read them as load-bearing where they prevent a re-proposal and narration where they do not; a ledger move for the latter is still defensible.
- FLOW-14's `requires-python <3.13` vs `fastmcp` 3.x is now a **locked** upstream constraint (OME-Zarr decision #3 drops Python 3.10 and caps at `<3.13`), still unverified.

---

## OME-Zarr cross-check — do the MCP spec's image-store assumptions survive?

Read `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store/docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md` (1107 lines, iterated since the brief was written).

**Yes, and the round-2 diff shrank the surface almost to nothing.** Verified by
grep over all 11 MCP documents:

- **`mode`/`layer`/`sample` are gone**, so `--mode migrate` and the changed
  `--mode recompile` have no MCP surface at all. §5.3 states the cut and names the
  storage redesign as part of its rationale — correct and now load-bearing.
- **The `.npy` sidecar is no longer named anywhere in the MCP spec.** §5.4's
  staged-GPU paragraph was rewritten this round and dropped it. This matters more
  than it did in round 1: the OME-Zarr design **withdrew** its in-store Stage-2
  write on 2026-08-19 and the sidecar is now a *consumable token* at
  `.phenotypic/progress/stage2_done/`. The MCP spec asserts nothing about it
  either way, so the change is invisible to it. FLOW-13/GEN-12 are confirmed a
  second time, and the round-2 rewrite removed the last real hook.
- **§5.3's header sweep reads source-image headers**, not the output store; the
  0.081 s measurement is unaffected by the format change.
- **§5.5 reads `manifest.json`**, format-agnostic in both designs.
- **§7 P6 stages input images**; a zarr store is an output directory, so P6's
  symlink assumption holds.
- **Remaining coupling: one cosmetic line** — `02-state-and-identity.md:158`'s
  `results/<dataset>/{hdf,measurements}/`.
- **One new constraint arrives:** the Python floor/ceiling (decision #3) and
  `--durable-writes` (decision #13). The latter joins GEN-18's list of deploy
  flags with no `_services` emitter — the argument surface the MCP spec must emit
  is still growing while nothing emits it.

**No MCP-spec assumption about the image store is falsified by the current
OME-Zarr draft.** The two specs are now coupled at one prose line, a Python
version range, and one CLI flag.

---

## Verdict

The rulings are good, the concurrency block is genuinely specification rather than
narration, and the propagation pass was real work that introduced no regressions.
But six contracts an implementer would build from — `deploy_start`'s signature,
the token's binding set, the `SubsetSelector` ABC, the campaign artifact, the `W1`
definition, and the deploy skill — still describe the pre-round-2 design, and one
of them (GEN-26) is the full-dataset spend gate.

**VERDICT: REVISE**

Sequencing: GEN-26 first — it is the only Critical, and GEN-27 and GEN-31 are its
neighbours in the same section and the same skill. GEN-28 is independent and
cheap (three declarations). GEN-29 is a table. GEN-30 and GEN-32 each require
choosing which of two written statements survives.
