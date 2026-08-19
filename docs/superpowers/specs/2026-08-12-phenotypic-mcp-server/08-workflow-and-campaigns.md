# PhenoTypic MCP Server — §8 Workflow, UX, and Campaigns

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 8.1 The intended UX

The server is not designed for an agent to wander a parameter space unattended.
The workflow is **collaborative planning, then delegated execution**:

```
Phase 0 — TRIAGE (human + agent)
  Characterize the ASSAY before any pipeline exists: organism morphology
  (filamentous / round / mixed), colony-vs-background contrast, colony
  separation, plate format, imaging modality. Some of this only you know;
  some is measurable from a probe. Produces the experiment profile (§9.3).

  ALSO establishes the DEVELOPMENT SUBSET (§10) — you name it, or the agent
  samples it with a recorded method. Everything below runs on the subset;
  the full dataset is touched only after a separate promotion gate.
        │
        ▼
Phase 1 — PLAN (human + agent, conversational, W0 + bounded W1)
  Driven by the experiment profile: prefab pipelines FIRST (§9.4), probed and
  compared, tuned before anything custom is authored. Nothing is submitted;
  every tool here is read-only or writes only draft artifacts.

  Phase 1 has TWO modes, and the agent should say which it is in:

    1a EXPLORE (§8.7) — the next step depends on what the last one showed.
       patch → probe{stages} → read evidence → keep/revert. Bounded and
       trailed. Use when you cannot yet name the arms.

    1b CONVERGE — the candidates are known. Write them down as a campaign.

  Exploration ends exactly when the arms can be named; that transition is
  the agent's to declare and yours to accept.
        │
        ▼
  A CAMPAIGN: the agreed set of arms, written down, reviewed by you.
        │
        ▼  ← the human checkpoint lives HERE, once, not at every submission
Phase 2 — EXECUTE (agent, autonomous, W2/W3)
  ONE `campaign_start` call launches every arm. Subagents, where used, built
  the arms during Phase 1 — they do not launch or poll individually. The
  orchestrator polls `campaign_status`. You are not in this loop.
        │
        ▼
Phase 3 — REPORT (agent → you)
  One leaderboard across arms, with the winner's provenance and the evidence
  behind the recommendation.
```

This shapes three design choices that would otherwise look arbitrary:

1. **The `deploy_plan` → `deploy_start` gate is satisfied at campaign
   granularity**, not per submission. Requiring a fresh plan for every arm would
   put a checkpoint in Phase 2, where you are deliberately absent. An approved
   campaign carries the plan token for each of its arms.
2. **Phase 1 must be cheap and side-effect-free.** Every planning tool is `W0`
   or a `dry_run`, so exploring twenty ideas conversationally costs nothing and
   touches no allocation.
3. **Phase 3 needs one call, not N.** An orchestrator polling three studies
   through three separate `tune_status` calls has to reassemble the comparison
   itself, and will compare costs whose scorers differ without noticing.

## 8.2 The campaign artifact

`<workspace>/campaigns/<name>/campaign.json`

```json
{
  "schema_version": 1,
  "name": "fungal-edge-sweep",
  "status": "approved",
  "created": "2026-08-12T13:40:02Z",
  "approved_at": "2026-08-12T13:52:17Z",
  "question": "Does phase-based edge detection beat the filamentous prefab on the low-contrast plates?",
  "experiment_profile": "profiles/plates.experiment.json",
  "subset_id": "subsets/plates-dev-24.subset.json",
  "metadata_csv": "data/tune_layout.csv",
  "derived_from": {"campaign_id": "campaigns/fungal-general-first",
                   "reason": "general-first winner failed the A_nidulans group"},
  "write_generation": 7,
  "launcher": {"pid": 48211, "create_time": 1786012345.7,
               "expires": "2026-08-12T14:10:00Z"},
  "objective": {"scorer": {"class": "QCScorer",
                           // `metadata` is CORRECT — the shipped field name (§10.3)
                           "params": {"check": {"metadata": "data/tune_layout.csv"}}},
                "sense": "cost in [0,1], lower is better"},
  "budget": {"trials_per_arm": 200, "max_concurrent_arms": 3},
  "compute": {"profile": "cpu-bulk", "n_workers": 8},
  "arms": [
    {"id": "prefab-fil", "pipeline": "pipelines/filamentous-prefab.json.pht-pipe",
     "tune_spec": "tune/filamentous-prefab.setup.json.pht-tune",
     "pipeline_digest": "sha256:3e91…", "spec_digest": "sha256:c07b…",
     "state": "running", "study_id": "studies/fungal-edge-sweep-prefab-fil",
     "queued_reason": null,
     "rationale": "baseline — FilamentousFungiPipeline, the assay-matched prefab"},
    {"id": "phase", "pipeline": "pipelines/phase-edge.json.pht-pipe",
     "tune_spec": "tune/phase-edge.setup.json.pht-tune",
     "state": "queued", "study_id": null, "queued_reason": "campaign_budget",
     "rationale": "the hypothesis",
     "prefab_baseline": {"pipeline": "FilamentousFungiPipeline", "best_cost": 0.31}},
    {"id": "watershed", "pipeline": "pipelines/watershed.json.pht-pipe",
     "tune_spec": "tune/watershed.setup.json.pht-tune",
     "state": "pending", "study_id": null, "queued_reason": null,
     "rationale": "control"}
  ]
}
```

**Five fields above are the fan-out's state, and they live here because this is
where the pydantic model is built from.** §8.3 argues each of them; none of them
was in this schema, so an implementer building `campaign.json` from this section
produced arms with no state field at all and a `campaign_status` with nothing to
read.

| Field | Where | Contract |
|---|---|---|
| `state` | per arm | `pending` → `queued` → `running` → terminal (`complete`/`failed`), per §8.3's arm-state table. `pending` is the value `campaign_put` writes |
| `queued_reason` | per arm | `"campaign_budget"` \| `"server_ceiling"` \| `"local_slot"` \| `null` (§1.5). Non-null only while `state == "queued"`; a starved arm and a healthy one are otherwise identical on disk |
| `study_id` | per arm | Written back by the launcher under CAS when the arm starts (§8.3). `null` until then, and its absence is precisely what makes the idempotent re-launch decidable |
| `launcher` | campaign | The fan-out lease — `{pid, create_time, expires}`, the same treatment §2.4 gives `RunRecord`. Absent, expired, or a pid whose `create_time` does not match a live process means no launcher is alive, which is what makes `launch_state` derivable (§8.3) |
| `derived_from` | campaign | Optional `{campaign_id, reason}`. USER-24's one breadcrumb: with grouping offloaded to the agent, N sibling per-group campaigns have no recorded relationship, and the fact that they descend from one experiment and one general-first failure would live only in the agent's context and vanish at the next compaction. The server records it and never acts on it |

**`write_generation` is a read hint, not the CAS key.** It is a monotonic counter
incremented on every successful write, and `campaign_status
{detail:"artifact"}` reports it so a recovering agent or a peer server can tell a
stale read from a current one without diffing content. **It is not what a
mutation compares against** — §2.6's rule is `(status, artifact_digest)`, and the
digest is the one that survives an artifact edited by a peer server whose counter
this reader never saw. An earlier draft of §8.3 called the counter "the value a
subsequent mutation CASes against", which put two incompatible CAS keys one
section apart, each stated as the rule; the digest wins because it is the one
that cannot be defeated by a writer the reader never observed.

**Arms reference the experiment profile, and custom arms cite the prefab they beat.**
`prefab_baseline` is the §9.4 convention: an arm whose pipeline is not a prefab
or prefab derivative records which prefab came closest and how it scored. The
server validates the field's *shape* and that the referenced study exists, but
does not require it — "custom before prefab is usually premature" is judgment,
which §9.1 places in a skill, not in the server.

**One scorer for the whole campaign.** Arms are only comparable if they are
scored the same way; a campaign whose arms carry different scorers produces a
leaderboard that means nothing. `campaign_put` takes the scorer at campaign
level and **rejects** an arm whose tune spec disagrees, with
`code: "arm_scorer_mismatch"`. This is the single most valuable invariant the
campaign concept adds.

### How that comparison is actually computed

Naming the mechanism is not pedantry — **both obvious implementations are
wrong**, and each fails in a different direction. Reproduced against the real
classes:

```python
s1 = QCScorer(check=ExpectedVsDetectedCount(metadata=<DataFrame>, groupby=[...]))
s2 = QCScorer(check=ExpectedVsDetectedCount(metadata=<same DataFrame>, groupby=[...]))

s1 == s2
# ValueError: The truth value of a DataFrame is ambiguous.
```

`ExpectedVsDetectedCount.metadata` is typed `pd.DataFrame | str`
(`analysis/qc/_expected_vs_detected.py:42-58`) — a public pydantic field, so
pydantic's generated `__eq__` walks into it and pandas raises.

The natural workaround is worse:

```python
s1.model_dump(mode="json")   # {"check": {"metadata": None, ...}}
s3.model_dump(mode="json")   # s3 built from a COMPLETELY DIFFERENT layout
s1_dump == s3_dump           # True
```

`_serialize_metadata` (`_expected_vs_detected.py:253`) emits `None` for any
DataFrame-backed metadata, so a dict comparison collapses every DataFrame-backed
scorer to one sentinel and declares them all equal — **silently producing
exactly the meaningless leaderboard this invariant exists to prevent.**

So the mechanism is ordered, and the order is load-bearing:

1. **Reject non-portable scorers first.** Any scorer whose `model_dump(mode="json")`
   contains a round-trip-lossy `None` where a source was configured is rejected
   with `scorer_not_portable` (§4.2) — *before* any comparison is attempted. This
   is already required independently, because a SLURM worker reloads the spec
   from disk and a DataFrame-backed check cannot be reconstructed
   (`model_validate({"metadata": None, …})` raises with "a check serialized from
   an in-memory DataFrame has no source path to round-trip").
2. **Then compare `model_dump(mode="json")`** — safe, because every surviving
   scorer is path-configured and serializes faithfully.
3. **Never use `==` on scorer objects.** Not anywhere, not as a shortcut.

**Validation is bound to bytes, not to a moment.** `campaign_put` checks each
arm's `.pht-tune` file, but `tune_start` re-parses that file from disk at launch
(`tune/__main__.py:200`) and `tune_put_spec {overwrite: true}` can rewrite it in
between. Without a binding, a rewritten scorer would launch unchallenged and the
drift would surface only in `campaign_status` — after the compute was spent,
which is precisely what the invariant exists to prevent.

So `campaign_put` records **`spec_digest`** and **`pipeline_digest`** per arm on
the campaign artifact, and `campaign_start` re-hashes and refuses on mismatch
(`arm_artifact_drift`). This is the same stale-digest pattern already used for
`tune_space` → `tune_put_spec` refs and for plan tokens.

The pipeline digest matters independently: `TuningSpec.pipeline` is **embedded**
(`tune/_spec.py:165`), not referenced, so `campaign_put`'s two checks — "the
arm's `pipeline` path loads" and "the arm's tune spec constructs" — validate two
different objects that nothing compares. Editing `pipelines/<name>.json.pht-pipe`
after `tune_put_spec` snapshotted it means the campaign validates a pipeline the
arm will never run.

`campaign_put`'s validation order partly self-defends today — arms' tune specs
are reloaded from `.pht-tune` JSON, and MCP arguments arrive as JSON, so a live
DataFrame cannot reach the request path. But that is incidental, not designed:
any future code comparing a campaign's declared scorer against an already-parsed
`deliverables/tuning_spec.json.pht-tune` object would hit case 1 directly.

**`status` is confirmation where the host supports it, provenance where it does
not.** An earlier draft conceded that "the server cannot verify that a human
approved anything… an agent could fabricate the field", and mitigated by making
fabrication explicit rather than silent. **That constraint is no longer real.**

`campaign_approve` raises an **elicitation** — a form the *host* renders and the
human answers. The response comes from the user's keyboard, not the agent's token
stream, so approval becomes actual confirmation for the more expensive of the two
irreversible steps rather than a field the agent fills in.

Three things this does not change, stated so the guarantee is not overread:

- **The fallback is mandatory, not optional.** A host without elicitation gets the
  previous design exactly — `human_response` required, fabrication explicit — so
  the server never depends on a capability it cannot confirm.

  **`human_response` is therefore required *unconditionally*, on every tool that
  takes a human decision.** An earlier draft made it "required-unless-elicited",
  and that is a required-field rule that varies with host capability: the agent
  cannot predict the signature from `tools/list`, and every such tool grows a
  fallback branch in its contract rather than in its implementation. The
  elicited-vs-asserted distinction is real and worth keeping — it just belongs in
  the **response**, not the signature:

  ```json
  {"ack_source":"elicited"}   // the host prompted a human and this is their answer
  {"ack_source":"agent_asserted"}   // no elicitation; the agent reports what it was told
  ```

  Same guarantee, one signature, and the distinction becomes an **auditable field
  on the artifact** instead of something implicit in host configuration — which
  is what a reader months later actually needs. It is also still not a breaking
  change to adopt elicitation, because the parameter never moves.
- **It is not authentication.** It confirms that *a* human at the host answered,
  not *which* human, and the server still runs with that user's rights (§6.4).
- **Behaviour under §1.3's shared connection is unverified.** All subagents share
  one server, and whether an elicitation raised from a subagent's call surfaces to
  the human — and to whom it is attributed when two are in flight — must be tested
  against the real host before this path is relied on.

### Three elicitation rules no test can supply

"Unverified" is the right word for *whether the prompt is delivered*. It is the
wrong word for what the prompt says, how many may be outstanding, and what a
non-answer means — those are the server's own decisions, and no observation of a
host settles any of them. They are decided here, and they hold whatever the
delivery test returns.

1. **Every elicitation message leads with the artifact id it approves.** The
   first thing in the text is `campaigns/fungal-edge-sweep` or
   `pipelines/edge-v3-tuned.json.pht-pipe`, before the numbers and before the
   question. Under §1.3 the human is being prompted by a server several subagents
   share, so "3 arms, 600 trials, ~5.7 node-hours. Approve?" is a question with no
   subject — a prompt that names no artifact cannot be answered correctly except
   by luck, and the person answering has no other channel through which to
   discover what they just agreed to.
2. **Single-flight per server: one human-gate elicitation outstanding at a
   time.** A second concurrent gate returns `human_gate_busy` (§6.2) naming the
   artifact currently on screen, and the caller retries. Two prompts in flight
   make attribution the *host's* problem, and neither the MCP protocol nor this
   design gives the server any way to tell which answer belongs to which request;
   refusing the second is the only version of this the server can reason about.
   It costs nothing real, because the gates are minutes apart in every workflow
   §8 describes.
3. **Timeout, decline, and unsupported all map to the mandatory `human_response`
   fallback — and none of them maps to approval.** These are three different
   things ("nobody answered", "somebody said no", "this host has no elicitation")
   and they share one property that matters more than their differences: not one
   of them is a human agreeing. A design in which a timeout advances the workflow
   turns the gate into a delay. Decline is a refusal that ends the call; timeout
   and unsupported fall back to the required `human_response` parameter, where
   the agent must state what it was told and `ack_source` records
   `agent_asserted` (§8.2 above) — which is auditable on the artifact afterwards,
   exactly as it would be on a host that never had elicitation.

## 8.3 Campaign tools

Four tools.

### `campaign_put` (`W0`) — draft the plan

Takes the campaign body above; defaults `status: "draft"`. Validates
**everything, submitting nothing**:

| Checked | How |
|---|---|
| Every arm's pipeline loads and is non-empty | `ImagePipeline.from_json` + the CLI's own emptiness check |
| Every arm's tune spec constructs | Real `TuningSpec` construction — all validators fire (§4.0) |
| All arms share the campaign scorer | Structural comparison (§8.2); `arm_scorer_mismatch`. **The arm's `.pht-tune` digest is recorded on the campaign**, and re-verified at `campaign_start` |
| Scorer is available and portable | `availability()` + the `QCScorer` path rule (§4.2) |
| Compute profile exists; overrides within caps | §5.2 |
| Arms resolve to distinct storage URLs | The H2 guard (§7) |
| `subset_id` resolves to a registered subset, non-empty, ≥ `min_heldout_plates` | The subset artifact (§10.2); a raw path is refused with `subset_required` |

The response is the **review document** — this is what you read before saying go:

```json
{"ok":true,"data":{
  "campaign_id":"campaigns/fungal-edge-sweep","status":"draft",
  "arms":[{"id":"otsu","n_knobs":4,"strategy":"tpe","trials":200,
           "pipeline_digest":"sha256:3e91…","spec_digest":"sha256:c07b…",
           "routed_to":"slurm","profile":"cpu-bulk",
           "estimate":{"node_seconds":6800,"basis":"probe: 3.4 s/image x 42 x 200/8"}}],
  "totals":{"arms":3,"trials":600,"est_node_hours":5.7,
            "concurrency":"3 arms x 8 workers = 24 tasks"},
  "objective":"QCScorer — cost in [0,1], lower is better",
  "pending_human_ack":true,
  "ack_prompt":"3 arms, 600 trials, ~5.7 node-hours on cpu-bulk. Approve?"},
 "issues":[{"severity":"warning","code":"needs_review_domain",
            "message":"phase arm: FocusEdgePhase.k has an inferred unbounded domain [0.5, 8.0]; inference guessed it.",
            "path":"arms[1].knobs[2]"}]}
```

`est_node_hours` is the number you actually want before agreeing to anything,
and its `basis` says whether it came from a real probe or a default.

### `campaign_approve` (`W0`)

`{campaign_id, human_response, note?}` → flips `draft` → `approved`, stamps
`approved_at`, **mints one `plan_token` per arm**, and appends a lineage row.

```json
{"ok":true,"data":{"campaign_id":"campaigns/fungal-edge-sweep","status":"approved",
  "approved_at":"2026-08-12T13:52:17Z",
  "plan_tokens":{"prefab-fil":"pl_7f3a…","phase":"pl_9c1e…","watershed":"pl_4b0a…"}}}
```

`human_response` is **required** and carries what the human actually said, which
is then recorded on the artifact and in lineage.

This does not authenticate anything — §8.2 is explicit that status is provenance,
not security, and an agent could fabricate the field. What it changes is the
failure mode: with `pending_human_ack: true` on the `campaign_put` response and a
required `human_response` here, approving without asking becomes an **explicit
fabrication** rather than an omission an agent can drift into. An agent that
never loaded the skill still gets a machine-readable signal that something is
waiting on a person. Refuses if any
blocking issue from `campaign_put` is unresolved, so approval cannot outrun
validation.

**Tokens are minted here, never by `campaign_put`.** An earlier draft showed
populated `plan_tokens` in `campaign_put`'s `draft` response — which would have
let an agent take a draft campaign's tokens straight to `deploy_start`, skipping
approval and the human checkpoint entirely. §8.2 is explicit that `status` is
provenance rather than security, so the gate has to be that the artifact an
unapproved campaign hands back contains nothing spendable.

**The approval CAS runs twice, and it CASes on `(status, artifact_digest)`.**
Elicitation is what makes this necessary. The digest of the campaign artifact is
captured **when the elicitation prompt is built** — that is, over the exact bytes
the numbers in the prompt were computed from — and re-checked after the human
answers, before `approved` is written. Between those two moments a human is
reading, so the window is minutes rather than milliseconds, and a §10.4
amendment landing inside it leaves `status` untouched at `draft`: a status-only
CAS passes and the approval attaches to an arm set nobody looked at.

On mismatch the call fails with `campaign_changed_during_approval` (§6.2), naming
the fields that moved, and re-prompts against the new content. Asking again is
the correct outcome — the human approved a specific 600 trials on a specific
three arms, and that consent does not transfer to a different plan by default.
See §2.6 for the general rule; this tool is the case that forced it.

### `campaign_start` (`W2`)

`{campaign_id, arms?}` → transitions `approved → launching → running` and
**returns**; a background task launches the arms, honouring
`max_concurrent_arms`, the server-wide ceiling (§1.6.1), and the routing rules of
§1.5. Refuses a `draft` campaign with `code: "campaign_not_approved"`, and a
concurrent second call is refused by the transition itself.

Each arm launches through the ordinary `tune_start` path — `RunRegistry.allocate`
→ `LocalRunner.start` → CAS — so campaign arms are ordinary studies, visible in
`workspace_list` and in the GUI. A campaign is an *organizing layer*, not a
parallel execution engine.

**Arm → study naming is explicit and persisted.** §2.2 forbids auto-suffixing,
so `campaign_start` does not invent names silently: each arm's study is
`studies/<campaign-name>-<arm-id>`, and the resolved `study_id` is **written back
into `campaign.json`** on the arm. A collision with an existing study is an
error naming both, not a silent rename. Without this, `campaign_status`'s
per-arm `study_id` would have no defined source.

**`campaign_start` snapshots the campaign it launched** rather than re-reading it
during fan-out, so a concurrent `campaign_approve` or an in-envelope amendment (§10.4) cannot
change the arm set mid-launch. Writes to `campaign.json` are atomic and CAS on
**`(status, artifact_digest)`** (§2.6) — status alone cannot see an amendment,
because an amendment leaves the status where it found it.

**The handler does not await the fan-out.** `campaign_start` performs the status
transition, hands back the arm list, and returns; a single **per-campaign
background task** owns launching arms up to `budget.max_concurrent_arms`,
recording each resolved `study_id` onto the artifact under CAS as it goes, and
respecting the server-wide ceiling of §1.6.1.

Awaiting the fan-out inside the handler is what USER-1's submit-and-poll contract
forbids, and the reason is not latency for its own sake: arms launch as
`max_concurrent_arms` frees up, so the await is bounded by *the campaign*, not by
submission. The caller's host times out somewhere in the middle, the coroutine is
abandoned, and the arms it had already launched keep running with nothing left
holding a reference to them — an orphan produced by a timeout, with no crash
involved.

**Arms therefore gain a first-class `queued` state**, between `pending` and
`running`, meaning *the launcher has accepted this arm and has not started it
yet*. Without it `campaign_status` has no honest word for an arm that exists, is
going to run, and has no `study_id` — and the agent reads it as failed.

**Re-calling `campaign_start` is idempotent, and that is what makes a kill
mid-fan-out recoverable.** It launches only arms with no `study_id` recorded;
arms that already have one are reported unchanged.

**The recovery call needs its own transition, because otherwise it is refused by
the same guard that stops a double launch.** After a kill the artifact reads
`running`, and a recovery call and a concurrent second caller are *the same call
arriving at the same status*. Only one of these can be true at once: either the
guard refuses everything non-`approved`, and recovery does not exist; or it
admits `running`, and the double-launch protection is gone. So the transition is
stated explicitly and takes two arms:

| From | To | Admitted when |
|---|---|---|
| `approved` | `launching` | the initial launch |
| `launching` \| `running` | `running` | **only if `launch_state == fan_out_incomplete`** — i.e. a recovery |

A second live caller finds `launch_state == clean` and is refused; a caller after
a kill finds `fan_out_incomplete` and proceeds. The discriminator is the state,
not the timing, which is what makes it implementable.

**Which makes `launch_state` load-bearing, so it must be derivable.** Defining
`fan_out_incomplete` as "…with no background task alive" is not: nothing on disk
records launcher liveness, §2.3 anticipates overlapping servers over one
workspace, and a launcher legitimately parked on the ceiling is indistinguishable
from one that no longer exists. So **the launcher writes a lease onto the
campaign artifact — `launcher: {pid, create_time, expires}` — with exactly the
`(pid, create_time)` treatment §2.4 gives `RunRecord`.** A lease that is absent,
expired, or whose pid/create_time pair does not resolve to a live process means
no launcher is alive. This is the same problem `RunRecord` was given that pair
for this round; the campaign artifact simply did not get the equivalent. Nothing else in the design recovers a half-launched
campaign: `allocate` refuses the already-claimed output directories and the agent
has no way to tell which arms got out.

| Arm state | `queued_reason` | Meaning |
|---|---|---|
| `pending` | `null` | on the artifact, not yet accepted by the launcher |
| `queued` | `"campaign_budget"` | accepted by the launcher, waiting on `budget.max_concurrent_arms`; no `study_id` yet |
| `queued` | `"server_ceiling"` | waiting on `limits.max_inflight_arms`, the server-wide semaphore (§1.5); carries a queue position |
| `queued` | `"local_slot"` | routed local and waiting on `LocalComputeSlot`. Under `local_slot_capacity=1` a three-arm campaign starts one arm and parks two here — which is neither of the other two reasons, and USER-17's "a second local arm is told the slot is busy and returns" applies to a *tool call*, not to the launcher, which holds the arm and retries |
| `running` | `null` | launched; `study_id` recorded under CAS |
| terminal | `null` | `complete` / `failed`, from the study's own record |

All four fields — `state`, `queued_reason`, `study_id` and the campaign-level
`launcher` lease — are declared on the artifact schema in §8.2. This table is
their meaning; that schema is where they are stored.

### `campaign_status {detail:"artifact"}` — read the stored campaign back

`{campaign_id, detail:"artifact"}` → the `campaign.json` artifact verbatim: arms with their
`pipeline`, `tune_spec`, `study_id`, `rationale`, `prefab_baseline`, and the
`pipeline_digest` / `spec_digest` binding (§8.2), plus the objective, budget,
compute, subset, and experiment-profile references.

**This is the session-recovery entry point**, and it is a `detail` mode of `campaign_status` rather than a separate tool — mirroring `tune_status` and `deploy_status`, and retiring a sibling pair whose names invited confusion. An agent resuming after a context
compaction typically holds one thing: a campaign id. `campaign_status` reports
*progress* per arm but not the artifact ids, so without `detail:"artifact"` the only
route back to a winning arm's pipeline was to know, unprompted, to call
`workspace_lineage {id: study_id}` and trace `tune.start`'s `parent` — a path
this spec named nowhere.

**Recovery procedure**, stated once so it is not folklore:

```
campaign_status {campaign_id, detail:"artifact"} -> arms, pipeline/tune_spec/study_id,
                                     subset_id, experiment_profile, objective
campaign_status {campaign_id}     -> where each arm actually got to
workspace_lineage {id: <study>}   -> only if you need the provenance chain
```

**Because it is the recovery entry point, it also reports how the record itself
stands.** Two fields beyond the artifact:

- **`write_generation`** — the artifact's own write counter, incremented on every
  successful write. It is what lets a recovering agent (or a second server) tell
  a stale read from a current one without diffing content. **It is a hint, not
  the CAS key**: mutations CAS on `(status, artifact_digest)` per §2.6, and the
  schema that stores the counter is §8.2's. A counter cannot be the key here
  because a peer server's write is invisible to a reader that never saw the
  counter move, while the digest of the bytes actually read cannot be defeated
  that way.
- **`launch_state`** — `clean` or `fan_out_incomplete`. `fan_out_incomplete` means
  the campaign is `launching` or `running`, at least one arm is `queued` with no
  `study_id`, and **the `launcher` lease is absent, expired, or names a pid whose
  `create_time` does not match a live process**. Liveness is read from the lease,
  never inferred from the arms.

`launch_state` exists because **"never started" and "started by a server that
died" look identical on disk** — both are a campaign with arms and no studies —
and the correct response differs completely: the first wants `campaign_start`,
the second wants `campaign_start`'s idempotent re-call plus the knowledge that
some arms are already burning compute. An agent resuming after a compaction holds
one campaign id and cannot distinguish them by looking, so the field says which
it is rather than leaving it to be inferred from a timestamp.

### `campaign_status` (`W0`) — one call, all arms

```json
{"ok":true,"data":{
 "campaign_id":"campaigns/fungal-edge-sweep","status":"running",
 "objective":"QCScorer — cost in [0,1], lower is better",
 "arms":[
  {"id":"phase","study_id":"studies/phase-edge","status":"running",
   "completed":126,"pruned":14,"failed":3,"budget":200,
   "best":{"trial":47,"score":0.081},"gap":{"value":0.06,"verdict":"ok"}},
  {"id":"otsu","study_id":"studies/otsu-base","status":"complete",
   "completed":200,"best":{"trial":180,"score":0.117},"gap":{"value":0.05,"verdict":"ok"}},
  {"id":"watershed","study_id":"studies/watershed","status":"failed",
   "completed":31,"failed":169,
   "error":"GridFinder found 0 rows on 169/200 trials"}],
 "leaderboard":[{"arm":"phase","score":0.081},{"arm":"otsu","score":0.117}],
 "comparable":true}}
```

**Polling economy.** `campaign_status` takes `since` (an opaque cursor from the
previous response). With it, arms whose state is unchanged collapse to
`{"arm":"otsu","unchanged":true}` and only movement is returned.

**The cursor is over the store artifact's stat, and it skips the store open.**
This is the load-bearing detail: §4.4 establishes that a per-arm leaderboard is a
`results`-class call requiring a killable subprocess per arm, and that this cost
is why polling must be infrequent. So the cursor embeds each arm's
`(path, mtime_ns, size)` for `trials.parquet` / `study.db` / `study.db-wal` /
`journal.log`, and an
arm whose stat is unchanged is reported `unchanged` **without opening its store at
all**.

**Three cursor states, because two is the bug.** `absent` — no artifact at that
path — is a state in its own right and **never compares equal to a stat**, so an
arm that has not yet produced a store is reported as changed (and opened) rather
than folded into `unchanged` and left invisible until something else moves it. A
cursor that treats a missing file as "nothing new" reports a queued arm frozen
forever.

**What the cursor stats depends on the storage backend, and the default backend
was the one the first draft got wrong.** `_optuna_store.py:88-89` enables WAL, so
a local SQLite study lands its trials in **`study.db-wal`** while `study.db`
itself sits stat-unchanged for long stretches — the default local path would be
reported frozen while progressing. So the cursor stats `study.db-wal` alongside
`study.db`, and under journal storage it stats `journal.log`, which is where the
trials actually are and where `study.db` never exists at all.

Trimming only the response payload would have saved context tokens while leaving
the N-subprocess-opens-per-poll cost — and the wedged-mount exposure of §7 B3 —
exactly as before. Skipping the open is what makes `campaign_status {since}`
genuinely cheaper than a bare `campaign_status`, and therefore safe to call more
often. A multi-hour
campaign polled on a human timescale otherwise accumulates dozens of
near-identical multi-KB snapshots in the agent's context — the exact long-running
unattended workflow this design is built around is also the one most able to
exhaust context. The skill instructs retaining only the latest full snapshot.

`comparable` is false — with an explanation — when arms cannot be honestly
ranked. **A leaderboard that silently ranks incomparable things is worse than no
leaderboard**, so the field is mandatory in the response rather than inferred by
the reader.

Its three causes, and where each gets its data — because one of them needed
plumbing that did not exist:

| Cause | Data source |
|---|---|
| Scorer drift between arms | The campaign-level scorer vs each arm's resolved `deliverables/tuning_spec.json.pht-tune`, compared by the ordered mechanism in §8.2 |
| An arm failed too heavily for its best trial to mean anything | `failed` / `completed` counts from the study store |
| **Arms ran on different datasets** | The `tune.start` **lineage** event (§2.5) |

That last row is not free. `TuningSpec` has **no dataset field**
(`tune/_spec.py:162-171`) — `--images` is a launch-time CLI argument recorded
nowhere in the resolved spec — and no directory-level digest helper exists in the
codebase (`bytes_fingerprint`, `file_fingerprint`, and `pipeline_content_digest`
are all single-file). So §2.5 adds a `dataset` block to the `tune.start` lineage
event, and §7 P3 adds the directory-digest helper. Until both land,
`campaign_status` must report dataset comparability as **unknown**, not assume
it — claiming a comparison the artifacts cannot support is the failure this flag
exists to prevent.

`gap` surfaces the held-out generalization check, so an arm that won by
overfitting the calibration split is visible as such rather than crowned.

## 8.4 Phase 1 in practice

What the planning conversation actually looks like, tool by tool:

```
you:   "new Aspergillus set, low-contrast plates. Otsu is under-segmenting."

agent: [skill: phenotypic-experiment-triage]
       → asks: morphology? expected colonies/plate? → you: "filamentous, 96"
       pipeline_put {name:"fil-prefab", from_prefab:"FilamentousFungiPipeline"}
       pipeline_probe {pipeline_id:"fil-prefab", …} to measure contrast/separation
       → fil-prefab now exists and is reused below; re-materializing it would
         return already_exists (§2.2 collision policy)
       → writes the experiment profile: morphology filamentous (human), contrast low (probe),
         separation touching (probe), 8x12 arrayed

agent: [skill: phenotypic-pipeline-construction — prefab-first]
       catalog_operations {category:"Prefab"}
       → the profile says filamentous + touching → candidates:
         FilamentousFungiPipeline (3 ops), HeavyWatershedPipeline (15)
       pipeline_put {name:"watershed", from_prefab:"HeavyWatershedPipeline"}
       (fil-prefab already materialized during triage — reuse, do not re-put)
       pipeline_probe both on the same 2 subset images
       → filamentous prefab: 61 objects; watershed: 44. Neither near 96.
       → "I'd tune the filamentous prefab first rather than author anything new."

you:   "do that, but also try a phase-based edge arm as the hypothesis"

agent: pipeline_put {name:"phase-edge", …, dry_run:true}     # nothing written
       tune_space   {pipeline_id:"phase-edge"}               # 9 targets, QCScorer available
       pipeline_probe {pipeline_id:"phase-edge", subset_id:"subsets/plates-dev-24.subset.json", n_images:2}
       → "94 vs 61 objects on the two low-contrast plates; phase looks promising.
          Blur sigma domain is an inference guess — I'd narrow it to [1,3]."

you:   "agreed, and cap it at 200 trials each"

agent: campaign_put {…}
       → review document: 3 arms, 600 trials, ~5.7 node-hours, one warning

you:   "go"

agent: campaign_approve {campaign_id:"campaigns/fungal-edge-sweep",
                         human_response:"go"}
       campaign_start   {campaign_id:"campaigns/fungal-edge-sweep"}
```

Everything before `campaign_start` is `W0` or `W1` — no allocation consumed, one
`LocalComputeSlot` used briefly by the probe. The expensive, irreversible step
happens once, after you have seen the number.

## 8.5 Phase 2 fan-out

**Fan-out happens in Phase 1, not Phase 2.** The orchestrator spawns one
subagent per arm to *author* it — explore, probe, settle a pipeline and a tune
spec — and each owns exactly one arm's ids, which is why §2.2 requires distinct
explicit names rather than auto-suffixing.

**Launching is not fanned out.** A single `campaign_start` drives every arm
through `RunRegistry.allocate → LocalRunner.start → CAS`, and the orchestrator
polls one `campaign_status`. An earlier draft described Phase 2 as subagents
each launching and polling their own arm, which contradicted §8.3 and the
`phenotypic-tuning-campaign` skill's tool list (which omits `tune_start` and
`tune_status` entirely). A subagent handed only its own arm could not tell which
model applied, and would waste a premature `tune_start`. `tune_start` remains
available for a standalone study outside a campaign.

What keeps this safe is entirely in §1.5 and §2.4: the subagents' `W0` calls
interleave freely, their probes serialize on the one `LocalComputeSlot`, and
`RunRegistry.allocate` refuses two claims on one output directory. Nothing
about fan-out needs new machinery here.

The orchestrator polls `campaign_status`, not three `tune_status` calls, and
reports to you on completion or on the first arm that fails hard.

## 8.7 Incremental construction — the inner loop of Phase 1

Phase 1 as described so far assumes the plan is *knowable upfront*. Often it is
not: you add an enhancer, look at what it did to `detect_mat`, and **that result
determines what the next operation should be**. The plan is discovered, not
designed.

This is not a different workflow — it is the inner loop of Phase 1, and it must
be cheap, bounded, and auditable.

```
        ┌─────────────────────────────────────────────┐
        │  pipeline_patch   (add / tune one op)       │
        │        ↓                                    │
        │  pipeline_probe {stages: true}              │
        │        ↓                                    │
        │  read per-stage numeric evidence            │
        │        ↓                                    │
        │  decide: keep · revert · try different op   │
        └──────────────┬──────────────────────────────┘
                       │  exit when the arms can be named
                       ▼
              campaign_put  →  Phase 2
```

Every tool in the loop is `W0` except the probe, which is `W1`. So the loop costs
no allocation, and sibling subagents can each run their own loop — their patches
interleave freely and their probes serialize behind the one `LocalComputeSlot`.

### Why this needs stage evidence specifically

An agent cannot see an image. Given only a final object count, a failed step is
uninterpretable: 61 objects instead of 96 could mean the enhancer destroyed the
contrast, the detector's threshold is wrong, or the refiner merged neighbours.
`pipeline_probe {stages: true}` (§3.2) makes each hypothesis checkable
separately — `detect_mat.std` collapsing after a blur, `num_objects` before and
after a refiner — which is the difference between iterating and guessing.

### Bounds

Incremental construction is where an agent can most easily wander, so it is
bounded on three axes:

| Bound | Default | Why |
|---|---|---|
| Steps per exploration | 12 patches | Beyond this, the agent is guessing rather than converging; the tool result says so |
| Images per probe | `limits.probe_max_images` (4) | Two images is usually enough to see a step's direction |
| No-improvement streak | 3 | Three consecutive steps with no movement in the tracked signal ends exploration and reports what was tried |

These are **advisory limits reported in the response**, not refusals — the agent
is told it has exhausted its exploration budget and should either commit to a
campaign or ask you. Hard-refusing would strand a legitimately long exploration
mid-way with nothing written down.

### The construction trail

Every accepted step appends a lineage row, so the resulting pipeline explains
itself:

```json
{"event":"pipeline.step","step_id":"st_4c1f…","id":"pipelines/edge-v3.json.pht-pipe","step":3,
 "edit":{"kind":"insert_op","slot":"ops","class":"FocusEdgePhase","params":{"k":3.0}},
 "agent":"sub-7","state":"in_flight","lease_expires":"2026-08-12T16:02:00Z"}
{"event":"pipeline.step.evidence","step_id":"st_4c1f…",
 "evidence":{"num_objects":{"before":61,"after":88},
             "detect_mat.std":{"before":0.04,"after":0.11}}}
```

**Two appends, not one row mutated.** §2.5's `lineage.jsonl` is **append-only** —
`atomic_append` under a file lock, with no update path — so "journal the step on
acceptance and fill in the evidence when the probe returns" cannot be implemented
as written. The step and its evidence are separate events correlated by
**`step_id`**, and readers fold by it. Without the id there is nothing for the
second append to attach to.

**`in_flight` carries a lease** (`probe_timeout_s`). A subagent that dies
mid-probe would otherwise leave a permanent *"a sibling is probing this now"* for
that edit — and the advisory would then suppress the real evidence forever, in
exactly the sibling case USER-9 added it for. Past `lease_expires`, a reader
treats the step as abandoned rather than active: the signal degrades instead of
inverting.

**The row is appended in two parts, and the order matters.** The step is
journalled the moment the edit is *accepted* — before its probe runs — carrying
the **canonical edit** (§3.2: the full edit with parameters, and with the target
op's `class` resolved from its index at this moment, since three of the six edit
kinds carry no class of their own and would otherwise be indistinguishable
later). It carries `"state":"in_flight"`. Evidence is filled in by the second
append when the probe returns.

**There is no `decision` field on either row.** `state` is the row's own
lifecycle — `in_flight` until its evidence append lands — and it is the only
status the journal stores. The keep/revert decision is the agent's choice, taken
after reading the evidence, and no tool in the catalog accepts it; the server
**derives** it at read time from the pipeline as it then stands, per §3.2's
per-kind table. Deriving beats reporting — a self-reported decision is a field an
agent can omit, while the pipeline itself cannot lie about what it contains — and
keeping it off the row is also what makes the append-only journal sufficient: a
stored decision would have to be updated, and there is no update path.

**The derivation has two limits, and the advisory states both rather than
rounding them off.**

*Attribution.* Up to 12 patches from N siblings mutate one pipeline in place, so
"present" can mean *a different agent re-added it* and "absent" can mean *a later
step removed it*. What the server can honestly report is **"still present in the
current pipeline"** — not "kept by the agent that tried it". Under a single
compacted agent, the case USER-9 actually cited, the two coincide; under
concurrent siblings they do not, and §3.2's wording says the former.

*Decidability per kind.* "Present" is the right test for `insert_op` and the
**inverse** of the right test for `remove_op`; for `move_op` and for a
`set_params` against a slot holding two ops of one class it decides nothing at
all. §3.2's table is the normative statement — this section records the edit it
needs, which is why the canonical edit and not the raw arguments go on the row —
and `undetermined` is a reported outcome, not a suppressed advisory.

Writing the whole row at the end instead would be simpler and would break §3.2's
`edit_previously_tried` in exactly the case it exists for: two sibling subagents
patching the same edit are both mid-probe, so under end-writing neither has
journalled anything, neither sees the other, and both spend the budget the
advisory was added to save. An `in_flight` row is what makes a concurrent
attempt visible while it is still concurrent.

This is what makes an incrementally-built pipeline defensible months later:
not "the agent produced this", but *which* step produced which improvement, on
what evidence. It is also what a `prefab_baseline` justification (§9.4) cites
when a custom pipeline finally does beat the prefab.

### Where it sits relative to prefab-first

The two compose rather than compete: **start from the assay-matched prefab, then
iterate from there.** The prefab is the starting point of the loop, not an
alternative to it — most explorations are "the prefab gets 61 of 96; what one
change closes the gap?", which is a far better-posed question than building from
an empty pipeline.

An exploration that begins from an empty pipeline should say why in its trail.

## 8.8 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-8.1 deploy arms~~ and ~~OQ-8.2 mid-campaign amendment~~ → **both granted,
  scoped to the development subset (§10.4)**. A campaign may carry a deploy arm
  and may replace a failed arm autonomously, provided the replacement stays
  inside the approved budget, profile, and scorer. Neither can reach the full
  dataset: that requires a separate human **promotion** (§10.5).

  Scoping development to a subset is what made both permissions safe to grant.
  The danger in "deploy arms" was never automation as such — it was unattended
  *full-dataset* compute on an unreviewed pipeline. Bound the loop to a subset
  and the campaign spends bounded, cheap compute, while the expensive
  irreversible step keeps its own gate.
