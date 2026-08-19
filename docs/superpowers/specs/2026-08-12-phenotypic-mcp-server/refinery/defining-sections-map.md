# Defining-sections map — every USER ruling against the artifact that carries it

**Why this file exists.** Three review rounds independently found the same
structural defect: a decision is written into the section that *explains* it and
never reaches the section that *defines* the artifact, the argument table, the
signature, or the error code the decision touches. An implementer reading the
defining section then builds the superseded design, and nothing fails.

This map is the standing check against that. **Explaining sections argue;
defining sections are built from.** A ruling is only applied when column (d)
reads `Y`.

## What counts as a defining section

| Kind | Where |
|---|---|
| Directory tree | §2.3 |
| Lineage event list | §2.5 |
| Concurrency guard table | §2.6 |
| Tool argument tables | §3.1–§3.3, §4.1–§4.5, §5.3–§5.5, §8.3, §10.3 |
| Plan-token record + binding table | §5.4 |
| Error / limits tables | §6.2, §6.3 |
| Campaign artifact schema | §8.2 |
| Subset artifact schema | §10.2 |
| Selector ABC + selector parameter tables | §10.3 |
| Work-class, routing and bounds tables | §1.5, §1.6.1 |
| Prerequisite task list | §7 P1–P7 |
| Plan decision records + Interfaces blocks | `plans/…/README.md` D1a–D6, F1–F5; phase task docs |

Everything else — §1.5's prose, §5.3's rationale, §9.3.0.2, §10.5's argument,
§8.7's narrative, §2.6's "why" subsections — **explains**. Prose may not be the
only home of a decision.

---

## The map

Status column is as of this sweep (round 3 propagation pass). `Y` = the defining
section reflects the ruling. `partial` = some defining sections do, others do
not. `N` = no defining section carries it.

| # | Ruling (one line) | Explained in | **Defines** it | (d) |
|---|---|---|---|---|
| **USER-1** | NFR anchor: `W0` < 1 s never blocking; `W1` bounded by `probe_max_images` + `probe_timeout_s`; `W2`/`W3` submit-and-poll | §1.6.1 intro; §1.5 "Blocking work never blocks the event loop" | §1.6.1 class table; §1.6.1 stated-bounds table; §6.3 limits; §3.0's `routed` envelope; every tool heading's class label | **partial → Y** — §1.6.1's `W1` row stated a 4-image bound that `deploy_plan {full}` (now `W1`) exceeds by two orders of magnitude. Fixed this sweep. |
| **USER-2** | Rotating specialist = concurrency | ledger | *process ruling — no spec artifact* | n/a |
| **USER-3** | MCPB rejected; local stdio confirmed | §1.3; `plans/…/MCPB-EVALUATION.md` | §9.7 packaging; plan README Tech Stack | Y |
| **USER-4** | SDK = PyPI `fastmcp` 3.x (D1a), not the official SDK's bundled FastMCP 1.0 | §1.4 | plan README **D1a**; plan Tech Stack line | Y |
| **USER-5** | Elicitation adopted (D6) for `campaign_approve` and the §10.5 gate; shaped now, implemented Phase 2C | §8.2's three elicitation rules; §10.5 | §5.4 `deploy_start` arg table; §8.3 `campaign_approve` signature; plan README **D6** | **partial → Y** — spec was already Y; plan D6 still carried the retired `required-unless-elicited` shape. Fixed this sweep. |
| **USER-6** | No server-side `plate.nrows`/`ncols` backstop (D3) | §9.3.5 | §9.3.5 validation list; plan README **D3** | Y |
| **USER-7** | Reviewers run at cluster boundaries, not per task (I8) | ledger | plan README "Review protocol" | Y |
| **USER-8** | Scope cut 32 → 26 tools; `promotion_*`, `pipeline_diff`, `campaign_get`, `catalog_measurements`, `experiment_profile_put` cut; `mode`/`layer`/`sample` cut from deploy | §3.0's cut table; §10.5; §5.3 | §3.0 count line; §5.3 arg table; §1.5 `W0` examples row; §6.2 error table; plan README phase map; `MCP-INTERFACE-AUDIT.md` tool tables | **partial → Y** — §1.5's `W0` row still cited the cut `pipeline_diff`; the audit doc still specified `promotion_request`/`promotion_approve` in six places. Fixed this sweep. |
| **USER-9** | `pipeline_patch` returns a prior matching attempt as an **advisory**, never a refusal | §3.2's advisory block; §8.7 | §6.2 `edit_previously_tried` row; §8.7's `pipeline.step` row schema; §3.2's match key + `decision` derivation | **partial → Y** — the derivation and the match key were written against `insert_op` only; inverted for `remove_op`, undecidable for three more, and collapsing for two. Per-kind table added this sweep. |
| **USER-10** | One local slot; a locally-routed `W2`/`W3` suspends interactive probing | §1.5 "A locally-routed batch job suspends…" | §1.5 routing table; §2.6 guard table | Y |
| **USER-11** | Workspace root is mandatory and must contain the image data; CWD default dropped | §2.3 prose | §2.3 tree + root-selection line; §2.7 resolved-OQ list; spec README | Y (closed by PROP-4) |
| **USER-12** | D1a, D5, D6 and the reviewer cadence go into the spec now | ledger | §1.4 (D1a); §3.0 (D5); §8.2/§10.5 (D6) | **partial → Y** — §3.0 replaced D5's enumeration with a derivation; the plan's D5 still carried the enumeration, which annotates the slot-holding `pipeline_probe` as `readOnly`. Fixed this sweep. |
| **USER-13** | `assay` → `experiment_profile` | ledger | §2.3 tree; §3.3 `experiment_profile_get`; §8.2 field; §9.3 | Y (biological-noun survivals are the SIMP-25 carve-out and correct) |
| **USER-14** | Locally run 1–2 arms — **superseded by USER-17** | — | — | superseded |
| **USER-15** | Multi-group experiments — **superseded by USER-24** | — | — | superseded |
| **USER-16** | `deferred-to-2A` only when resolution depends on unobservable behaviour, and only with a written pass condition | ledger | ledger dispositions; §6.5 test list | Y |
| **USER-17** | The local arm cap **is** the slot's capacity; `local_slot_capacity` is configuration (default 1); a second local arm is refused, never parked | §1.5 "Locally, the slot *is* the cap" | §1.5's slot statement; §1.5's slot-primitive block; §2.6 guard table; §1.6.1 bounds table; §6.3 limits table; §8.3 arm-state table | **partial → Y** — §1.5 ×2, §2.6 and §6.3 all hard-coded capacity 1 beside the configurable statement. Fixed this sweep. |
| **USER-18** | The human gate lives in `deploy_start`, not `deploy_plan` | §10.5; §8.2 | §5.4 arg table + token record; §2.5 lineage `deploy.approve`; §10.5 scope table; §9.5 deploy skill | **partial → Y** — §5.4's signature was fixed just before this sweep (GEN-26/FLOW-33); two stale restatements survived inside §5.4 and §9.5's skill still taught the retired gate. Fixed this sweep. |
| **USER-19** | Per-group breakdown is the scorer's output — **superseded by USER-24** | — | — | superseded |
| **USER-20** | Handlers are `async def`; everything blocking is offloaded | §1.5 "Blocking work never blocks the event loop" | §1.6.1 bounds table (`executors.blocking`=4, `executors.compute`=1); §6.3 limits; §2.5 lineage-write rule; §2.6 guard table | Y |
| **USER-21** | Full scope on a group-filtered subset is `parent ∩ group_filter`; **the token binds `(parent_digest, group_filter)`** | §10.5 | §5.4 binding table + token record; §5.3 `deploy_plan` arg table + response; §10.2 subset artifact; §2.5 `deploy.approve` / `deploy.start` rows | **N → Y** — `group_filter` appeared in no binding table, no token record, no artifact and no lineage row, so the safety property had no carrier at all. Fixed this sweep. **One half is deliberately left open** — whether `parent ∩ group_filter` stages; see the ledger's OPEN QUESTION. |
| **USER-22** | `human_response` unconditionally required; `ack_source` in the **response** | §8.2 rule 3; §10.5 | §5.4 arg table; §8.3 `campaign_approve`; plan README **D6**; `MCPB-EVALUATION.md` | **partial → Y** — spec Y; both plan docs still carried `required-unless-elicited`. Fixed this sweep. |
| **USER-23** | Local `W2`/`W3` children are detached (`start_new_session=True`), adopted by restart reconciliation | §1.5 | §1.5 restart-reconciliation block; §2.4 run records; §6.2 `local_slot_orphaned` | Y |
| **USER-24** | The agent owns grouping; the server keeps exactly two things — `group_filter` on the `SubsetSelector` **ABC**, and `derived_from` on the campaign artifact | §9.3.0.2 | §10.3's `SubsetSelector` ABC; §10.2 subset artifact; §8.2 campaign schema; §7 P3 item 3; §5.3 arg table/response; §6.2 error table; spec README | **N → Y** — *neither* surviving primitive reached the artifact that carries it: the ABC is `extra="forbid"` so `group_filter` could not even arrive as an extra key, and the spec README still advertised the deleted `group_by` + trait-override design. Fixed this sweep. |

### Round-2 concurrency block — the same check, for concepts that are not numbered rulings

| Concept | Explained in | **Defines** it | (d) |
|---|---|---|---|
| CAS on `(status, artifact_digest)` | §2.6 "The CAS key is…" | §2.6 guard table; §8.3 approval/transition text | **partial → Y** — §8.3 named `write_generation` as "the value a subsequent mutation CASes against", a second and incompatible CAS key one section away. Demoted to a read hint this sweep; §2.6's pair remains the key. |
| Arm `state`, first-class `queued` | §8.3 prose + arm-state table | §8.2 campaign artifact schema | **N → Y** — arms in §8.2 had no state field at all. Fixed this sweep. |
| `queued_reason` (`campaign_budget` / `server_ceiling` / `local_slot`) | §1.5 "The server-wide arm ceiling is a queue" | §8.3 arm-state table; §8.2 schema | **N → Y** |
| `launcher` lease `{pid, create_time, expires}` | §8.3 "Which makes `launch_state` load-bearing" | §8.2 schema | **N → Y** |
| per-arm `study_id` written back under CAS | §8.3 | §8.2 schema | **N → Y** |
| `deploy_plan {scope:"full"}` is `W1` | §5.3 header + body | §1.5 work-class table; §1.5 routing table; §1.6.1 `W1` row; §3.0 annotation derivation | **N → Y** — promoted in one section header and nowhere else. Fixed this sweep. |

---

## Summary

**24 rulings. 4 are superseded or process-only (USER-2, 14, 15, 19), leaving 20
live rulings with at least one defining section.**

- **11 of 20 had a defining-section gap** at the start of this sweep:
  USER-1, 5, 8, 9, 12, 17, 18, 21, 22, 24, plus the six concurrency-block
  concepts tracked separately.
- **2 were total (`N`)** — USER-21 and USER-24, i.e. *both* of the rulings whose
  entire content was a new primitive. That is the pattern: **a ruling that
  deletes a design and keeps one primitive is the one most likely to leave the
  primitive undeclared**, because it is written into the section doing the
  deleting.
- **9 were partial** — the ruling reached one defining section and missed a
  sibling. Sub-pattern: the argument table gets fixed and the *record example*,
  the *bounds table*, or the *plan decision record* does not.

**The check to run before closing any future round:** for every changed
paragraph, name the argument table, artifact schema, error row and plan decision
record it implies, and grep each one. If a ruling's only home is prose, it is
not applied.
