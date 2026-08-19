# Round 4 — data-flow review

**Scope:** `snapshots/round-3-spec.diff` (1,459) + `round-3-plan.diff` (263),
read against `snapshots/round-3-spec.md` / `round-3-plan.md`, `ledger.md`
(USER-1..27), and `defining-sections-map.md`.

**Tree and provenance.** Reviewed in
`/bigdata/iwheeldonlab/anguy344/PhenoTypic` (the live tree), not the stale
`/bigdata/exfab` clone. Verified: `ledger.md` inode 245023473 / 72,460 bytes;
`round-3-spec.diff` 1,459 lines; `round-3-plan.diff` 263.

**The working copy moved under me while I read it.** Spec files carried mtimes
of 03:29–03:30, *after* the 03:22 snapshot — the post-diff commits `c35936606`,
`74f98fa96`, `a985d91ce`. I froze the round-3 snapshot and reviewed against
that, my assigned scope, and where a post-diff commit already fixes something I
found in scope I say so rather than charging it twice (FLOW-42, and the
`ack_source` advisory now closed by GEN-33).

**Every open finding below was then re-verified against HEAD `a985d91ce`**, so
none of them is an artifact of reading the snapshot: FLOW-40 (no P8 task in the
plan; `README.md:292` and `phase-1b:1` still read P3–P7), FLOW-41 (§7's only
`RunConsoleState`/`to_argv` mention is P2's promotion row at `07:307`, not P8),
FLOW-43 (`08:506-508` still requires an arm in `queued`), FLOW-44
(`03:425-426` still unguarded), FLOW-45 (`resolved_params` and `order_after`
appear nowhere in `08-workflow-and-campaigns.md`).

---

## Part 1 — verification of my round-3 concerns

### FLOW-32 (Critical, fan-out state machine) — **fixed on all four parts, one reachable hole remains**

I re-traced the whole machine, including recovery-after-kill.

| Part | Round-3 answer | Verdict |
|---|---|---|
| (a) recovery unreachable | Two-arm transition table, `launch_state` as discriminator (§8.3, snapshot 4266-4269) | ✅ |
| (b) `launch_state` underivable | `launcher: {pid, create_time, expires}` lease, `(pid, create_time)` per §2.4; declared on the schema at §8.2 3954 | ✅ |
| (c) local arms stranded | `queued_reason: "local_slot"` added, with the tool-call-vs-launcher distinction stated (§8.3 4288-4292, §1.5 351) | ✅ |
| (d) nothing wakes a blocked launcher | "The launcher's wake condition is the semaphore itself" — `asyncio.Semaphore(max_inflight_arms)`, arrival-order, plus cancel and shutdown behaviour (§1.5 359-368) | ✅ |

**Double-launch is still refused.** Live launcher ⇒ valid lease ⇒
`launch_state == clean` ⇒ the `launching|running → running` arm is not admitted.
Two simultaneous recovery callers are separated by §2.6's `(status,
artifact_digest)` CAS on the lease write. The window I expected between the
handler's status transition and the launcher's first write is closed *by the
`pending`/`queued` distinction*: in that window no arm is `queued` yet, so
`launch_state` reads `clean` and the second caller is refused. That is a good
design and it is load-bearing — which is exactly why the next finding matters.

**The hole (FLOW-43 below): the same `queued` qualifier that closes the
double-launch window opens a stranding window on recovery.**

### FLOW-33 (Critical, §5.4 pre-USER-18 gate) — **fixed, all three sites**

Verified at snapshot 2591-2612:

1. `human_response` is now in the argument table (2599), `str`, **Required**,
   "Unconditional — see §8.2; there is no elicited-vs-not variant of this
   signature". The unimplementable signature PF-3 named is gone.
2. The `scope` row (2597) now reads "**the human ack is taken here, not carried
   on the token**" — it no longer contradicts §10.5.
3. The "It **also records the human ack**" sentence is replaced by "The token
   **does not record the human ack** — USER-18 moved the ack to `deploy_start`"
   (2719-2723).

The stale `(pipeline digest, images digest, compute)` triple is also gone; §5.4
now points at the binding table as the single statement.

### FLOW-34 (Major, binding set) — **fixed, both halves**

*Half 1.* The binding table (2673-2687) gained a **Scope** column and both
missing fields: `parent_digest` (`full` only) and `group_filter` (`full` only,
with USER-21 cited by name and the copy-from-§10.2 path stated). The record
example (2646-2655) carries `"parent_digest":null,"group_filter":null`, and the
paragraph at 2657-2661 says explicitly that they are fields of the record at
*both* scopes, `null` at subset — closing the "optional key inferred from an
example" failure by name.

*Half 2 — the one I was least sure would be answered, and it is answered well.*
"The token's two producers, and what each can bind" (2751-2782) makes a
campaign-stamped token a **distinct `kind`**: `"campaign_arm"`, minted by
`campaign_approve`, binding only `scope`/`pipeline_digest`/`subset_id`/
`subset_digest`/`compute` + `campaign_id`/`arm_id`, at **`subset` scope only**.
`run_name`, `array`, `estimate.node_hours`, `argv_digest` are "**absent, not
null**", and validation does not look for them. The weaker binding set is kept
off the irreversible path by `campaign_arm_scope_full` (§6.2 2985), and §6.5
(3268-3270) has a test for exactly that. Both readings I said were unwritten are
now written, and the one I did not anticipate — that the weak token must be
barred from full scope — is written too.

One consequence to note: with `human_response` required *unconditionally*
(2599) and §10.4 letting the background launcher run a campaign deploy arm
unattended, the launcher has no human to ask. This is a real in-scope
contradiction — and the post-snapshot working copy is already fixing it (keying
`human_response` on token kind, adding `ack_source: "campaign_approved"`). I
flag it only so the fix is recorded as deliberate rather than drift; it also
touches USER-22, so it belongs in the ledger as an amendment, not a silent edit.

### FLOW-35 (Major, derived `decision`) — **fixed for four kinds of six; two residuals**

The per-kind table (§3.2 1476-1483) is a genuine fix, not a patch. Checking each
kind against the actual edit schemas:

| Kind | Key | Derivation | Verdict |
|---|---|---|---|
| `insert_op` | `{kind, slot, class, params}` | present ⇒ keep | ✅ correct |
| `remove_op` | `{kind, slot, class}`, class resolved from `index` at record time | **inverted**: absent ⇒ keep | ✅ inversion fixed; ⚠️ unguarded on duplicate classes |
| `move_op` | `{kind, slot, class, order_after}` | sequence match ⇒ keep, else `undetermined` | ✅ decidability handled; ⚠️ blind to a swap of two same-class ops |
| `set_params` | `{kind, slot, class, resolved_params}` | value match ⇒ keep/revert, `undetermined` on multiplicity | ✅ correct, and the merge resolution is the right call |
| `set_grid` | `{kind, nrows, ncols}` | pair match | ✅ correct |
| `set_model` | `{kind, class, params}`, `null` matches absent | equality | ✅ correct |

The three things I said were missing are all present: kind-dependent keys,
class resolution at record time, and `undetermined` as a *reported* value that
still carries evidence. §6.5 (3272-3283) tests the `remove_op` inversion by
name. Residual is FLOW-44.

### FLOW-36 (Major, `group_filter` prose-only) — **fixed, and USER-26 settles the half I sent up**

Every carrier I said was missing now exists:

- **ABC** — `group_filter: dict[str, str] = Field(default_factory=dict)` on
  `SubsetSelector` (5495), with the docstring line (5488) and a semantics table
  (5527-5530) covering column source, absent column, and empty match. The
  `extra="forbid"` problem is moot because the field is declared.
- **Artifact** — recorded at the top level *and* in `selection.params`
  (5420, 5426, 5458-5462).
- **Errors** — `group_filter_column_not_found` and `group_filter_matches_nothing`
  (3006-3007), the second explicitly covering the full-scope re-application.
- **Tests** — 3253-3257, including "a `RandomSubsetSelector` with a
  `group_filter` must select only from the filtered candidates", which is the
  on-the-ABC property.
- **`derived_from`** — now on §8.2's schema and its field table.
- **Ships from the first commit** — §7 P3 (3666-3671) and the plan's
  `phase-1b` ABC block (4954-5004).

The staging contradiction is resolved by USER-26 rather than papered over
(§10.5 5959-5981). See Part 2 for the end-to-end trace of what that ruling buys.

### FLOW-37 (Major, campaign artifact vs §8.3) — **fixed, both halves**

*Half 1.* §8.2's schema (3897-3956) now carries `state`, `queued_reason`,
`study_id`, `launcher`, `derived_from` and `write_generation` — in the JSON
example *and* in a field-contract table. The paragraph at 3943-3947 names the
failure directly ("an implementer building `campaign.json` from this section
produced arms with no state field at all").

*Half 2.* The CAS contradiction is resolved in the direction I said was
defensible: "**`write_generation` is a read hint, not the CAS key**… §2.6's rule
is `(status, artifact_digest)`" (3957-3963), with §8.3's own restatement
(4322-4331) corrected to match. One rule, one section, cross-referenced.

### FLOW-38 (Major, plan carries cut tools and retired signature) — **fixed for the spec-derived content**

- Cut tools: the interface-audit doc carries a retired-surface banner naming all
  six (plan 2121-2122); Appendix A's rows are struck with `~~…~~` and a
  **CUT (USER-8)** reason (2890-2891); footnote 12 is marked **RETIRED**
  (2946).
- `required-unless-elicited`: action row 2 amended in place (2084); **D6**
  rewritten with USER-22 and USER-18 both named (5723). The two surviving
  instances (2015, 2129) are inside the banner-marked audit doc, quoting the
  superseded surface — acceptable.
- **D5 → D5a** (5722): the enumeration is retired, §3.0's derivation is
  normative, and the row says "do not implement this row's original list".
- **F3** (5739): annotated `deploy_plan is no longer an instance`, with the
  §5.3 `W0`/`W1` split cited.

Residual, cosmetic and inside the banner: the audit is still titled "the 32-tool
surface" and Appendix A "all 32 tools" while D5a says 26. Not worth a fix.

### FLOW-39 (Major, §8.7 row and §3.2 examples) — **fixed**

§8.7's canonical row (4566-4568) is now
`"edit":{"kind":"insert_op","slot":"ops","class":"FocusEdgePhase","params":{"k":3.0}}`
with `"state":"in_flight"` — `index` dropped, `params` present, no `decision`
field. The two-append design with `step_id` correlation (4574-4580) resolves the
append-only contradiction, and **"There is no `decision` field on either row"**
(4598-4605) answers the residual ambiguity I raised: `state` is the row's own
lifecycle, the decision is derived per §3.2's table, and keeping it off the row
is what makes an append-only journal sufficient. §3.2's prose (1420-1425) now
reads "The journal records the edit and the evidence; it does not record a
decision". Clean.

### FLOW-1, FLOW-2, FLOW-5 — confirmed still open

**FLOW-1 and FLOW-2 are unchanged in danger.** §8.7's evidence is per-stage probe
numbers (`num_objects`, `detect_mat.std`), which sit upstream of the
`apply_post` frame mismatch, so the new derivation does not lean on it harder.

**FLOW-5 / GEN-2 became materially more dangerous, and it is the one standing
item that did.** Round 3 added a *new* CLI argument that must travel through the
argv emitter already known to be incapable of carrying `deploy_*`'s own
arguments — and put it on the full-dataset path, where the value being emitted
is the image set a human approved. This is FLOW-41 below, and it is why I am
raising it as new rather than as a re-argument of FLOW-5.

---

## Part 2 — USER-26's manifest, traced end to end

The lead asked for this specifically. Tracing the new flow — server resolves an
image set → writes a manifest → CLI consumes it via a flag that does not exist:

| Link | Where stated | Status |
|---|---|---|
| **Who computes it** | `deploy_plan` at plan time (§10.5 5965-5967) | ✅ stated |
| **When** | plan time, before the token is minted | ✅ stated |
| **Where it is stored** | — | ❌ nowhere in scope; §2.3's tree gains the token record, not the image list |
| **Lifecycle / collection** | — | ❌ no rule in scope |
| **What binds it** | §10.5 5967-5969 says "the manifest is what the token's digest binds"; §10.5 5977-5978 then resolves that to `argv_digest` | ❌ see FLOW-42 |
| **How the run consumes it** | new top-level flag, §7 P8 | ⚠️ under-scoped — FLOW-41 |
| **Who builds the flag** | — | ❌ no plan task — FLOW-40 |
| **Parent changes between plan and start** | `parent_digest` is bound at `full` scope ⇒ `plan_stale` (§5.4 2686) | ✅ correctly handled |
| **Filter changes** | `group_filter` bound at `full` scope (§5.4 2687) | ✅ |

Two of the three gaps are already being closed in the working copy. The two that
are not — the plan task and the emitter — are my Criticals.

**Verified facts** (I re-checked the ones the ruling rests on):
`phenotypicCLI.py:922-931` — `-i/--input`, `type=click.Path(exists=True,
dir_okay=True, file_okay=True)`, `default=None`, **no `multiple=True`**.
`_cli_staged_slurm_worker.py:422` — `parser.add_argument("--manifest",
type=Path, required=True)`, an `argparse` entry point on the staged worker, not
a Click option on the public CLI. Both citations in the spec are accurate.

---

## Concerns

### FLOW-40 [**Critical**] — USER-26's prerequisite has no owning task anywhere in the plan · plan-change

The ledger and the round-4 addendum both say P8 was "recorded as a **new §7
prerequisite and plan task**". The prerequisite exists (§7 P8, snapshot
3760-3789). **The plan task does not.**

Checked in the round-3 plan snapshot *and* in the live working copy:

- `phase-1b-engine-prerequisites.md:1` is titled "Engine prerequisites
  (**P3–P7**)".
- `README.md:215` — the task ledger ends `P7  distributed finalize entry point
  → Task 18`. There is no Task 19.
- `README.md:292` — "phase-1b … **P3–P7** — Tasks 10–18".
- The plan's own coverage audit still asserts "**Spec coverage: no gap.** Every
  **P2–P7** item has an owning task" (snapshot 6194) — a statement that was true
  when written and is now the thing hiding the gap, because its enumeration
  stops one short of the new prerequisite.
- No occurrence of `--manifest`, "top-level manifest", or P8 anywhere in the
  plan, in either version.

**Why Critical rather than Major.** This is not a missing line of documentation.
An implementer building Phase 1b from the plan builds P3–P7 and stops; nothing
fails, and no test covers a flag nobody was asked to write. Then `deploy_plan
{scope:"full"}` on a group-filtered subset has no argv it can render — and the
most natural thing for an implementer to do at that point is render `--input
<parent>` and drop the filter, which is precisely "deploy one group's tuned
pipeline across **every** group, with every digest check passing" (§10.5's own
words for the failure the filter exists to prevent), on the irreversible
full-dataset path.

**Fix:** add Task 19 to `phase-1b-engine-prerequisites.md` implementing §7 P8;
retitle the phase P3–P8; add the row to `README.md`'s ledger; update the
coverage claim to P2–P8. Note in Task 17 (P6) that its "there is no manifest
flag … on either" (snapshot 5330) describes the pre-P8 state.

### FLOW-41 [**Critical**] — P8 stops at the Click option; the argv emitter the server actually uses cannot express the flag · spec-change + plan-change

§7 P8 scopes the work as "promoting it to a public top-level flag on `python -m
phenotypic`". That is half of it. §5.4's Mechanism paragraph (snapshot 2769)
says the server builds argv via **`to_argv(RunConsoleState)` + the profile's
`--slurm` pairs**, and `argv_digest` (2694) is defined as the SHA-256 of exactly
that rendered list.

Verified against the merged `_services` tier:

- `src/phenotypic/_services/argv.py:53-97` — `RunConsoleState` has
  `pipeline_path`, `input_dir`, `output_dir`, `metadata_csv`, `mode`, `dry_run`,
  `retry_failures`, `advanced_args`, `slurm_args`, `gpu_slurm_args`,
  `gpu_shards`. **No manifest field, and `advanced_args` is a closed recognised
  set** (`sample`, `nrows`, `ncols`, `image_type`, `workers`, `log_level`) —
  "unknown keys are ignored at argv time", so it is not an escape hatch either.
- `src/phenotypic/_services/argv.py:326-380` — `to_argv` raises `ValueError` if
  `input_dir` is unset and emits `["--input", input_dir]` **unconditionally**.
  There is no manifest branch and no way to suppress `--input`.

So with P8 built exactly as written, the server still cannot emit the flag, and
`argv_digest` provably cannot contain it. This is GEN-2/FLOW-5's known gap —
"even with the values corrected, **no `_services` symbol can emit them**" — now
sitting under a ruling whose entire purpose is that a human's approved image set
cannot drift. The nine-month-old gap is the same; the blast radius is not.

**Fix, and it is small:** §7 P8 must name both halves — the Click option on
`phenotypicCLI.py` **and** a manifest field on `RunConsoleState` with the
corresponding branch in `to_argv` (which must also make `--input` conditional,
or state that both are passed and what `--input` means alongside a manifest).
The plan task from FLOW-40 then has two files, not one. While there: §5.4's
`resume` caveat says `input_path` is compared by literal string equality with no
normalization — say whether the manifest participates in
`validate_resume_compatibility`, or a resume can silently run a different image
set under a matching `input_path`.

### FLOW-42 [Critical **in scope**, already fixed post-snapshot — no action] — the manifest's contents were not bound

Recording this because it was in my scope and because the fix should be
recognised as answering it, not charged again.

In the snapshot, §10.5:5967-5969 asserts "the manifest is what the token's digest
binds, so **the human approves an image set that cannot subsequently drift**",
and then §10.5:5977-5978 resolves that to "`argv_digest` (§5.4) digests the argv that
names that manifest". `argv_digest` covers the argv *string*, which merely names
the file — a manifest whose contents changed across a 24 h token lifetime
re-derives an identical `argv_digest` and passes every check. §5.4's binding
table, which declares itself the exhaustive set and whose §6.5 test mutates each
row, contained no manifest field. The property USER-26 was adopted to provide
was not carried by anything. Storage location, collection rule, and
disambiguation from `deploy_status`'s `manifest.json` were likewise unstated.

The working copy (05-deploy-and-slurm.md, 03:29) adds `image_manifest_digest` as
a bound field, `.phenotypic-mcp/plans/<token>.images` as the location under the
token's lifecycle, the `argv_digest`-is-a-null-guard argument, the
`image_manifest`-never-bare-"manifest" naming rule, and — the one I would have
had to raise separately — **collection skips any token whose `consumed_by` names
a non-terminal run**, which closes the multi-hour SLURM array reading its input
list out from under itself. That is the complete answer. No further action.

### FLOW-43 [Major] — `launch_state`'s arm predicate is narrower than the relaunch predicate, so a kill that leaves arms `pending` strands them unrecoverably · spec-change

Two predicates for one condition, one section apart:

- **Relaunch** (§8.3 4255, and §6.5's test at 3235-3237): "It launches only arms
  with **no `study_id` recorded**."
- **Admission** (§8.3 4334-4339): `fan_out_incomplete` requires "at least one arm
  is **`queued`** with no `study_id`" **and** a dead lease.

They disagree exactly on arms in `pending` — which §8.3's own arm table defines
as "on the artifact, **not yet accepted by the launcher**". Whether that state
is ever occupied mid-fan-out depends on an unstated launcher policy: if the
launcher marks every arm `queued` on adoption, the predicates coincide; if it
accepts arms incrementally as `budget.max_concurrent_arms` frees — which is what
"accepted by the launcher, waiting on `max_concurrent_arms`" reads like — then a
5-arm campaign at `max_concurrent_arms=3` has arms 4 and 5 sitting `pending`.
Kill the launcher there and: status `running`, no arm `queued`, lease dead ⇒
`launch_state == clean` ⇒ the recovery arm is **not** admitted ⇒ arms 4 and 5
are stranded with no path. Restart reconciliation does not help — §1.5 (477-497)
reconciles `RunRecord`s only; nothing reconciles a campaign artifact.

§6.5's own test would fail in that configuration: it kills the launcher
mid-fan-out and requires the re-call to "launch only the arms with no
`study_id`", which the admission guard refuses.

**Fix, one word:** define `fan_out_incomplete` as "at least one arm has no
`study_id`" (dropping `queued`), matching the relaunch predicate and the test.
The double-launch protection is unaffected — it rests on the **lease**, which is
live in every non-recovery case, and the transition-vs-lease-write window stays
closed because the launcher writes its lease in the same CAS that adopts the
campaign. State that ordering explicitly while you are there.

**Also, one line:** §6.2 has no code for the refused second `campaign_start`.
`campaign_not_approved` is scoped to `draft`; §6.5 (3290-3294) requires the
refusal but names no code. Add one (`campaign_launch_in_progress`), carrying the
lease's `(pid, create_time)` so the caller can see *why* it was refused.

### FLOW-44 [Major] — `remove_op` and `move_op` carry the duplicate-class hazard that `set_params` guards explicitly · spec-change

§3.2's `set_params` row states the guard: "**`undetermined` when `slot` holds
more than one op of that class**, because dropping `index` makes the two
indistinguishable". The identical exposure exists in two other rows and is not
guarded:

- **`remove_op`** — key `{kind, slot, class}`; `ops = [BlurGauss, BlurGauss]`,
  remove index 1, keep it. Current `ops = [BlurGauss]`. Derivation: "the op
  **present** ⇒ `revert`". The advisory reports `revert` on an edit that was
  kept. This is FLOW-35's original complaint ("reports the opposite of the
  truth") surviving in the duplicate-class case, after the inversion itself was
  correctly fixed.
- **`move_op`** — key includes `order_after`, "the slot's ordered **class**
  sequence". Swapping two ops of the same class leaves the class sequence
  identical, so `order_after` matches whether the move was kept or reverted, and
  the derivation reports `keep` unconditionally.

The spec's own worked example for the hazard is "two `BlurGauss` in one
pipeline", so this is not an exotic configuration — and §6.5's test for
collision ("removing `BlurGauss` and removing `OtsuDetector`", 3278-3280) uses
two *different* classes and would pass throughout.

**Fix:** extend the `set_params` multiplicity guard to `remove_op` and
`move_op` — when the slot holds more than one op of the resolved class, the
derivation is `undetermined` and the advisory still carries the evidence. Add
the same-class case to the §6.5 test.

### FLOW-45 [Major] — §8.7's row spec does not declare the two record-time fields §3.2's keys require · spec-change

§3.2's per-kind keys need three things resolved and stored **at record time**:
the target op's `class` (for `remove_op`, `move_op`, `set_params`),
`resolved_params` for `set_params` ("the **effective** parameter set after
`merge` is applied, **not the partial map the agent sent**"), and `order_after`
for `move_op`.

§8.7 — the section that defines what the journal row contains, and the example
an implementer copies — declares only the first: "the **canonical edit** (§3.2:
the full edit with parameters, and with the target op's `class` resolved from
its index at this moment)" (4589-4597). "The full edit with parameters" is the
partial map for `set_params`; `resolved_params` and `order_after` are named
nowhere in §8.7, and the row example is an `insert_op`, which needs neither.

An implementer building the journal row from §8.7 stores the sent `params` and
no `order_after`. `set_params` — the kind §3.2 calls the one the loop runs
most — then matches on the wrong value, and `move_op`'s key has a field with no
producer. This is the round's own defect class (defining section does not carry
what the explaining section requires), one hop further out than the sweep
reached.

**Fix:** §8.7's canonical-edit paragraph should enumerate, per kind, exactly the
fields §3.2's key column names — resolved `class`, `resolved_params` for
`set_params`, `order_after` for `move_op` — and the row example should show a
`set_params` alongside the `insert_op` so the resolved form is visible.

---

## Advisory

- ~~§5.4's response line declares a two-value `ack_source` above §5.4's own use
  of a third~~ — closed by GEN-33 (`a985d91ce`); verified at HEAD.
- The `deploy.approve` and `deploy.start` lineage rows now carry `group_filter`
  (§2.5 931-935), which retires my round-3 advisory on journal reconstructibility.
  They do not carry the image-manifest digest, so the *set* is still not
  reconstructible from the journal alone.
- §1.6.1's `W0` row still asserts "under one second" and "does not mean *is
  instant*" in one cell (FLOW-28, unaddressed across three rounds).
- `deploy_start` still contains an unbounded human wait inside a `W3` handler;
  USER-25 exempts it from `W0` latency, which is a different question from the
  host's tool-call timeout.
- The interface-audit doc's "32-tool" title and Appendix A's "all 32 tools"
  disagree with D5a's 26, inside a banner-marked retired doc. Cosmetic.

---

## On the lead's four uncertainties

1. **Is the growth narration returning?** No, and I measured it rather than
   guessing. The diff is **+732 / −248** (net +484, matching 5,753 → 6,237). Of
   the 732 added lines, **77 are table rows and 34 are JSON/schema lines** — 111
   lines of pure defining-section artifact. The remaining ~620 are prose, which
   is a 5.6:1 prose-to-artifact ratio and looks alarming until you check what
   the prose *is*: only **4** added lines carry the retro-narration signature
   USER-27 retired ("an earlier draft", "had been stated", "which is how the
   defect shipped", "so an implementer building"). The rest, in every section I
   read closely (§3.2, §5.4, §8.2, §8.3, §8.7, §10.3, §10.5, §7 P8), is
   forward-looking rule-justification tied to a specific normative statement —
   the ~120 lines USER-27 explicitly preserved, at scale. It is verbose. It is
   not the habit that was cut.
2. **Are the sweep's edits correct in substance, not merely present?** For the
   nine sites I own, yes with three exceptions, all narrow and all named above:
   §8.3's `launch_state` predicate (FLOW-43), §3.2's two unguarded kinds
   (FLOW-44), §8.7's missing record-time fields (FLOW-45). The pattern in all
   three is the same and worth naming: the sweep carried each ruling to the
   defining section it was measured against, and stopped one hop short of the
   *second* defining section that consumes what the first now produces.
3. **Is P8 real work with a real owner?** **No.** It is a §7 section with no
   plan task, and its stated scope is missing the field that makes it work.
   FLOW-40 and FLOW-41. This was the right thing to be least confident about.
4. **Does the per-kind `decision` table hold against the actual edit kinds?**
   Four of six are correct as written; `remove_op` and `move_op` are correct
   except in the duplicate-class case the table already knows about and guards
   only on `set_params`. FLOW-44.

---

## Verdict

**VERDICT: REVISE**

Narrowly, and not on the spec's decisions. Everything I was asked to verify
about the round-3 diff is genuinely fixed: FLOW-33, 34, 37, 38 and 39 are closed
outright, FLOW-32's four parts and FLOW-35's derivation are closed with narrow
residuals, and FLOW-36 is closed with USER-26 answering the half I sent up. The
diff is the propagation it claimed to be.

The blocker is one ruling that did not finish landing. **USER-26 exists in the
spec and nowhere in the plan** (FLOW-40), and the prerequisite it did buy is
scoped to a Click option when the server renders argv through
`to_argv(RunConsoleState)`, which cannot carry the flag and — verified in the
merged tier — has no field for it (FLOW-41). Both sit on the full-dataset path,
and the failure mode of leaving them is not an error but a silent full-scale
deploy of one group's pipeline over every group.

Three edits close it: a Task 19 in Phase 1b, two sentences in §7 P8 naming
`RunConsoleState` and `to_argv`, and the phase/coverage headers that still read
P3–P7. FLOW-43, 44 and 45 are one line, one clause and one paragraph
respectively. If the lead prefers, all six are small enough to apply and
re-verify without a round 5 — my objection is to shipping the plan with
USER-26 unbuilt, not to the round.

**`deferred-to-2A`:** none. **`needs-user-input`:** none — USER-26 settled the
one item I sent up in round 3.
