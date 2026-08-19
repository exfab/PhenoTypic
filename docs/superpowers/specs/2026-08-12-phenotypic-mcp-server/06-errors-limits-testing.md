# PhenoTypic MCP Server — §6 Errors, Limits, Safety, Testing

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 6.1 Error philosophy

**Errors are values.** A failed tool call returns `ok: false` with structured
issues — not an MCP protocol error. An agent that receives a protocol error can
only retry or give up; an agent that receives
`{"code":"unknown_param","path":"ops[0].params.sigmaa","hint":"Did you mean 'sigma'?"}`
can fix it. Protocol errors are reserved for malformed calls (bad JSON, missing
required argument).

Every issue carries `severity`, `code`, `message`, and where applicable `path`
and `hint`. `code` is a closed set — agents may branch on it; `message` is
human prose and may change.

## 6.2 Error codes

| Code | Severity | Raised when |
|---|---|---|
| `path_outside_workspace` | error | A path argument resolves outside `SandboxRoot` |
| `not_found` | error | An id names no artifact |
| `already_exists` | error | `put` without `overwrite` |
| `unknown_class` | error | Operation/scorer/strategy class not resolvable; carries did-you-mean |
| `unknown_param` | error | `extra="forbid"` rejected a field; carries did-you-mean |
| `invalid_param` | error | Pydantic validation failed (type, bound, validator) |
| `missing_param` | error | A required field was absent (`missing`) — distinct from `invalid_param` |
| `arm_artifact_drift` | error | An arm's `.pht-tune` or pipeline digest changed between `campaign_put` and `campaign_start` (§8.2) |
| `finalize_incomplete` | error | A `finalize_in_progress` marker is present; the study directory cannot be trusted (§4.5) |
| `study_not_finished` | error | `tune_export_best` on a distributed study whose budget is not drained and whose jobs are still live (§4.5) |
| `edit_previously_tried` | **advisory** | A `pipeline_patch` edit matches one already recorded in the exploration trail, under §3.2's **per-kind** canonical match key. Carries that attempt's evidence, its `step_id`, and a **derived** `decision` — `keep` \| `revert` \| `undetermined` \| `in_flight` — never one read off the journal, which stores no decision. `undetermined` still fires with the evidence. Advisory, never a refusal: the surrounding pipeline may have changed (§3.2) |
| `stage_order_hint` | **advisory** | GUI DAG validator hint; never blocks |
| `pipeline_empty` | error | Neither ops nor measurements — the CLI's own check |
| `stale_target_ref` | error | A `select` ref built against a superseded pipeline digest |
| `no_active_knobs` | error | Tune spec varies nothing |
| `grid_needs_stepped_domain` | error | `grid` strategy with a continuous `FloatRange` |
| `scorer_unavailable` | error | `scorer.availability()` is `False` |
| `scorer_not_portable` | error | `QCScorer` check not configured from a path |
| `distributed_storage_unavailable` | error | SLURM tune with no safe shared storage (§7 P1 not landed) |
| `storage_url_collision` | error | Another live study resolves to the same storage URL |
| `unknown_profile` | error | `compute.profile` names no configured profile |
| `param_not_overridable` | error | Override key absent from the profile's `overridable` |
| `cap_exceeded` | error | Override violates a cap; names key, request, cap |
| `reserved_sbatch_key` | error | Agent tried to set `array`/`output`/`error`/`job-name` |
| `profile_not_expressible` | error | A profile key the target CLI surface cannot carry — e.g. `account`/`qos`/`cpus_per_task` on the tune path, which accepts only four discrete SLURM flags (§5.2.1) |
| `plan_required` | error | `deploy_start` without a `plan_token` (§5.4) |
| `plan_stale` | error | Plan token's pipeline/images/compute digest no longer matches |
| `campaign_not_approved` | error | `campaign_start` on a `draft` campaign |
| `campaign_arm_scope_full` | error | A campaign deploy arm targeting the full dataset — campaigns are subset-scoped (§10.4) |
| `amendment_exceeds_envelope` | error | A mid-campaign amendment outside the approved budget, profile, or scorer |
| `subset_required` | error | A subset-scoped tool given a raw parent path instead of a `subset_id` (§10.3.1) |
| `subset_too_small` | error | Subset below `min_heldout_plates` (6): `derive_split` returns an EMPTY held-out set, so there is no gap at all |
| `subset_too_small_for_heldout` | **warning** | Subset under ~15: a single held-out plate makes the gap a one-sample estimate |
| `selector_unavailable` | error | `SubsetSelector.availability()` is `False` — e.g. `EmbeddingSubsetSelector`, which raises rather than falling back to random (§10.3) |
| `group_key_not_in_metadata` | error | `MetadataGroupSubsetSelector.group_key` names no column in `grouping_metadata`. **Carries the CSV's actual column list and a did-you-mean** — the agent cannot open the file itself, so an error without the columns leaves it guessing |
| `group_filter_column_not_found` | error | A `SubsetSelector.group_filter` key names no column in `grouping_metadata` (§10.3). Carries the CSV's column list and a did-you-mean, for the same reason `group_key_not_in_metadata` does. Distinct from it because the filter is on the **ABC** and fires for selectors that have no `group_key` at all |
| `group_filter_matches_nothing` | error | A `group_filter` matched no rows — at selection time, or at `deploy_plan {scope:"full"}` where the filter is re-applied to the parent (§10.5). An empty result is refused rather than returned: an empty subset passes every downstream shape check and produces a study of nothing, and an empty full-scope image set would otherwise deploy nothing while reporting success |
| `arm_scorer_mismatch` | error | A campaign arm's scorer differs from the campaign scorer (§8.2) |
| `screening_unsupported_on_slurm` | error | `--screen` + SLURM, which silently drops screening today. Ships: OQ-4.1 resolved to expose screening (default off), so the guard is needed (§7 P4). |
| `output_not_empty` | error | Deploy target non-empty; names `run_name`/`resume`/`restart` |
| `resume_incompatible` | error | Pre-validated `validate_resume_compatibility` mismatch; names the field |
| `scheduler_jobs_active` | error | Resume/restart while ledgered jobs are live |
| `local_slot_timeout` | error | `W1` waited past `probe_timeout_s` for the slot. Carries `held_by` (work class + id), `held_for_s`, and `queue_position`, so an agent can distinguish "retry in 30 s" from "a 2-hour deploy holds it" instead of retrying blind |
| `slot_lease_expired` | error | The slot's wall-clock lease elapsed while a holder still had it — `probe_timeout_s` for `W1`, the run's `--time` or the configured maximum for a local `W2`/`W3` (§1.5). The slot auto-releases and the holder's record carries this as its terminal reason, so the state is visible rather than stuck |
| `local_slot_orphaned` | error | A local `W2`/`W3` was refused because restart reconciliation found a **live orphan** from a previous server on the local path (§1.5). Carries the orphan's `pid`, `create_time`, and `run_id`, and names `workspace_cancel` as the remedy — the server refuses rather than watching a process it does not own |
| `project_config_invalid` | error | `<root>/phenotypic-mcp.toml` failed to parse, or named an unknown `default_profile`. **Raised at startup**, naming file and line — never ignored, because a config that silently does nothing is worse than no config at all (§5.2.2) |
| `project_config_widens_cap` | error | The project file set a cap **above** the site ceiling. Carries key, project value and site ceiling. Refused rather than clamped: a silent clamp teaches the author their file works (§5.2.2) |
| `artifact_lock_timeout` | error | `exclusive_path_lock` exhausted its 30 s spin (`FileLockTimeout`, `_cli/_cli_file_locking.py:50`) — a wedged mount, or a peer server holding the path. Carries the path and the elapsed wait. Retryable; it is not a validation failure and must not be reported as one |
| `artifact_changed` | error | A read-modify-write found the artifact's content digest changed since it was read (§2.6). Carries both digests and the fields that moved. **The write is refused** — never silently merged, and never `ok:true` |
| `campaign_changed_during_approval` | error | The campaign artifact changed between the elicitation prompt being built and the human answering (§8.3). Carries what moved; the gate re-prompts against the new content rather than attaching consent to a plan nobody read |
| `human_gate_busy` | error | A second human-gate elicitation was requested while one is outstanding — single-flight per server (§8.2). Names the artifact currently on screen; the caller retries |
| `output_generation_active` | error | `RunRegistry.allocate` refused an output directory that already has a nonterminal launch generation (`_services/runs.py:317-337`, today a bare `RuntimeError`). Names the holding `run_id` and its status — the agent's choices are a different `run_name` or `workspace_cancel`, and neither is discoverable from an unclassified 500 |
| `probe_cap_exceeded` | error | `n_images` above `limits.probe_max_images` |
| `environment_mismatch` | error | SLURM work requested in a `local` environment with no profiles |
| `submission_failed` | error | A subprocess exited non-zero or `sbatch` rejected a script for a reason no pre-check models; carries exit code, stderr tail, and **`retryable: bool`** classified from known transient-vs-config `sbatch` patterns, so an agent does not retry a bad account name verbatim |
| `scheduler_unreachable` | error | Detection said `slurm`, submission found otherwise; reports probe staleness |
| `not_owned` | error | Cancel targeted a run with no `RunRegistry` record here |
| `version_drift` | **warning** | Spec `phenotypic_version` ≠ installed; matches the engine's warn-only posture |

### Mapping engine exceptions onto the envelope

The codes above are the target; these are the sources.

**`pydantic.ValidationError` → one `issues` row per `.errors()` entry.** A single
`ValidationError` routinely carries several, and collapsing them would hide all
but the first. Per entry: `type` selects the code (`extra_forbidden` → `unknown_param`, `missing` →
**`missing_param`**, everything else → `invalid_param`), and `loc` becomes
`path`. `missing` gets its own code because `invalid_param` is defined as
"type, bound, validator" — none of which describe an absent required field.

**`loc` → `path` uses the agent's own addressing, not pydantic's.** Integers
become `[i]`, names join with `.`. The **nested-operation case does not carry a
second `params` segment**, which an earlier draft got wrong. Reproduced against
the real field:

```python
FilamentousFungiDetector.model_validate(
    {'inoculum_detector': {'class': 'OtsuDetector', 'params': {'sigmaa': 1.0}}})
# loc == ('inoculum_detector', 'sigmaa')      <- no 'params' between them
```

`OperationField`'s `BeforeValidator` (`sdk_/typing_.py:302-343`) calls
`cls.model_validate(value.get("params", {}))` with the params dict already
unwrapped, so the inner error's `loc` is bare and pydantic prepends only the
outer field name. The correct rendering is
`ops[3].params.inoculum_detector.sigmaa`.

This matters beyond cosmetics: §6.5 mandates one test per code asserting `path`
is populated. Written against the fictitious shape, those tests would encode the
wrong contract permanently.

Note also that §3.2's per-entry resolution keeps errors out of the final
`ImagePipeline(ops=[...])` assembly. That is what makes `ops[0]` addressing valid
at all — `ImagePipeline.ops` is a `Dict[str, ...]`, so an integer-indexed `loc`
could not arise from the assembly step, and a union-discrimination failure there
would inject validator-chain tags into `loc` and multiply the rows. The one existing formatter in the repo,
`_validation_messages` (`gui/tune/_setup_authoring.py:524-531`), instead does
`".".join(...)`, yielding `0.sigma`; that is the GUI's convention and does not
match the bracketed paths every example here uses. The projection is new code,
not reuse.

**`hint`** is populated by `difflib.get_close_matches` for `unknown_param` /
`unknown_class` (against `model_fields`) **and for any closed-set check whose
candidate set the agent cannot see** — notably `group_key_not_in_metadata`,
where the candidates live in an external CSV no tool reads back. It is otherwise
absent; never a generic string. The rule is: **if the valid values exist and the
agent has no way to obtain them, the error carries them.**

**Registry and lock failures are classified, not caught by the fallback.** Two
exceptions the reused `_services` layer raises today have no natural code and
would otherwise surface as an unclassified 500 — which is the one outcome §6.1
exists to prevent, since an agent can neither branch on it nor fix it.
`FileLockTimeout` (`_cli/_cli_file_locking.py:50`) maps to
`artifact_lock_timeout`, and `allocate`'s bare `RuntimeError` on a nonterminal
generation (`_services/runs.py:317-337`) maps to `output_generation_active`,
carrying the holding `run_id` parsed from the record rather than from the
message string. Both are **retryable in different senses** — one after a wait,
one only after a different `run_name` or a cancel — and the distinction is
exactly what the agent needs and cannot infer from a traceback.

**Subprocess and scheduler failures.** Not every failure is pre-checkable; §5.4
pre-validates resume compatibility precisely because the CLI's own failure mode
is `sys.exit(1)` inside a subprocess. Anything that escapes those checks maps to
`submission_failed`, carrying the captured stderr tail and the exit code, rather
than falling through the closed set. This covers an `sbatch` that passed the
liveness probe but rejected a specific script (an account or QoS the profile caps
did not model), and any CLI traceback the pre-checks did not anticipate.

## 6.3 Limits

| Limit | Default | Enforced by |
|---|---|---|
| `probe_max_images` | 4 | Hard cap on `pipeline_probe.n_images` |
| `probe_timeout_s` | 300 | Wall clock on `W1`, including slot wait |
| `local_slot_capacity` | 1 | `LocalComputeSlot` (§1.5) — configuration, not a fixed 1; `executors.compute` workers **is** this number |
| `executors.blocking` workers | 4 | §1.5 — filesystem, journal, subprocess, scheduler |
| `executors.compute` workers | **1**, always | §1.5 — `W1`; not an independent knob, or the pool and the slot disagree above the default |
| `limits.max_inflight_arms` | 8 | **Server-wide**, all campaigns; checked by the background launcher (§8.3) |
| Slot lease | `probe_timeout_s` (`W1`) / **`local_lease_max_s`** (local `W2`/`W3`) — *not* the run's `--time`, which is a SLURM flag a local run does not have | §1.5 — auto-release; `slot_lease_expired` |
| Catalog list size | unbounded rows, compact fields | `catalog_operations` returns no schemas |
| Measurement payloads | summary only | `describe()` + column names; parquet path for the rest |
| Log tail | 200 lines | `LocalRunner.snapshot_log` |

## 6.4 Safety boundary

The server runs as the user with the user's filesystem and scheduler rights.
There is no authentication — its boundary is the workspace sandbox plus these
explicit refusals:

1. **No `--overwrite`.** `shutil.rmtree(output_dir)` is not reachable from any
   tool. Destroying measurements, HDFs, and human QC curation stays a shell
   decision (§5.4).
2. **No raw sbatch.** Partition, account, and QoS come only from config;
   overrides are allow-listed and capped (§5.2).
3. **No cross-session cancellation.** Cancel requires a local `RunRegistry`
   record (§5.6).
4. **No path escape.** Every path resolves through `SandboxRoot` first.
5. **No credential handling.** Storage URLs are password-less by construction —
   `_reject_password_in_url` is the engine's chokepoint and the server never
   accepts a password argument. Postgres secrets stay in `~/.pgpass`.
6. **No implicit env inheritance for storage.** `$PHENOTYPIC_TUNE_STORAGE_URL`
   is never allowed to select storage silently (§4.3).

**Untrusted content.** Image filenames, dataset directory names, metadata CSV
contents, and docstrings from third-party operation classes all flow into tool
results. They are data, not instructions. Docstrings in particular are rendered
into `catalog_operation_detail` — a hostile third-party operation package could
place directives there. The server does not sanitize prose (that would corrupt
legitimate documentation), so this is recorded as a known property: **an agent
must treat catalog text as documentation, not as instruction.**

## 6.5 Testing

The project's test-integrity rules bind here, and they bind **generally**, not
only where this section calls them out:

> **Every test below must be proven able to fail** — by reintroducing the bug it
> guards, or by a one-line mutation of the code under test — before it is
> trusted. A cap-enforcement test that would keep passing with the comparison
> deleted proves nothing. A check that cannot run must **fail**, not skip.

An earlier draft applied this only to the import-purity gate and the P1 script.
That was a misreading of a project-wide rule as a per-case one: the round-trip,
error-code, cap, and refusal tests below need the same treatment, and are the
ones most likely to rot into tautologies.

### Unit — the `_services` promotion (P2)

- **Import-purity gate.** A test that imports every `phenotypic._services.*`
  module in a subprocess and asserts `dash`, `dash_bootstrap_components`,
  `flask`, and `werkzeug` are absent from `sys.modules`. This is the test that
  makes the `_space.py` split (§1.4) permanent rather than aspirational. It must
  be shown to fail when a `import dash` is reintroduced.
- **Shim equivalence.** For each promoted symbol, assert the `gui.*` re-export
  and the `_services.*` original are the *same object* — catching the
  `_REGISTRY` double-singleton failure specifically.
- **The two-lock ordering (§7 P2) has its own test**, because the registry half
  above does not cover it: hold `exclusive_path_lock` on an owner-lock path from
  one thread, then assert from another that ordinary `RunRegistry` methods remain
  callable rather than blocking for the full 30 s spin. Without it the fix is
  described and not verified, and the failure it guards is a stall — which looks
  like slowness, not like a bug.
- Existing GUI tests and the CI ledger gates (`FEATURES.md`, `WORKFLOWS.md`,
  smoke-capture) must stay green unchanged.

### Unit — tool layer

- **Round-trip:** agent-facing pipeline spec → envelope → `from_json` →
  `to_json` → the same envelope, including nested `OperationField` values and
  the `_make_unique` duplicate-class naming.
- **Every error code is reachable.** One test per code in §6.2, asserting the
  code *and* that `path`/`hint` are populated where the table promises them. A
  code with no test is a code that will regress silently.
- **Cap enforcement is exact:** at, one below, and one above each cap, including
  `time` in all three accepted formats (`240`, `04:00:00`, `0-04:00:00`).
- **Refusals are refusals:** no argument combination reaches `--overwrite`, a
  reserved sbatch key, or a non-overridable profile key.
- **The embedding placeholder actually raises.** `EmbeddingSubsetSelector`
  must be shown to raise `NotImplementedError` and to report
  `availability() == (False, ...)` — never to return a random selection. A
  placeholder that silently degrades would stamp `method: "EmbeddingSubsetSelector"`
  onto an artifact describing a subset with none of the claimed visual coverage,
  and nothing downstream could contradict it.
- **Sandbox escape, adversarially.** §6.4 has no authentication — `SandboxRoot`
  *is* the entire security boundary — so it gets an explicit hostile test rather
  than relying on the blanket one-test-per-code rule: `..` traversal, symlinks
  pointing outside, absolute paths, and the cross-platform cases this project
  must support (Windows drive letters, UNC paths, case-insensitive filesystems).
  Path-escape semantics are a classic cross-platform gap and a generic pass will
  under-cover them.
- **`deploy_plan` writes nothing under the output directory.** Assert the output
  directory is byte-identical before and after a plan call — this is what keeps
  the extracted `build_array_script_spec` (§5.3) from regressing back into the
  file-writing generator and silently breaking `output_not_empty`.

### Concurrency

- **`LocalComputeSlot` mutual exclusion:** interleave `W1` probes and a local
  `W3` submission from concurrent tasks; assert never more than one holds the
  slot. Must fail if the slot is removed from either path — this is the test
  that guards the §1.5 blocker fix.
- **Event loop stays responsive:** `W0` calls complete while a `W1` probe is in
  flight, proving the `run_in_executor` offload.
- **Restart reconciliation:** with a live orphan subprocess recorded in
  `RunRegistry`, a fresh server must refuse a local `W2`/`W3` with
  `local_slot_orphaned` — not claim the slot on the orphan's behalf, and not
  admit a second local job; with a dead one, it must CAS to `failed`.
- **PID reuse is not liveness:** a record whose `pid` resolves to a live process
  with a *different* `create_time` must reconcile as **dead**. The test needs a
  fabricated record rather than a real collision, and it must fail if the check
  is reduced to the pid alone (§2.4).
- **Slot release on every exit path:** normal return, raised exception,
  cancellation, and subprocess reap must each leave the slot free. The
  release-first ordering is pinned by injecting an `artifact_lock_timeout` into
  the exit observer's *record* step and asserting the slot is still released
  (§1.5) — this is the test that guards the strand-forever failure.
- **Lease expiry:** a holder that outlives its lease releases the slot and its
  record reads `slot_lease_expired`. This must pass whether or not cancellation
  is delivered, since the lease is specified independently of it.
- **Executor isolation:** saturating `blocking` with concurrent store-opens must
  not delay a `W1` probe the slot has already admitted — the assertion that the
  two pools are actually separate.
- **Artifact CAS:** two concurrent `pipeline_patch` calls on one artifact must
  produce one success and one `artifact_changed`; **neither may return `ok:true`
  having lost an edit**. Same shape for a `campaign.json` amendment landing
  during an approval — `campaign_changed_during_approval`, not a silent revert.
- **Terminal records are never resurrected:** a subprocess that dies before the
  post-start CAS must leave the record `failed`, and the output directory must
  remain claimable afterwards (§2.6).
- **Staging completion:** a reader must treat a staging directory without its
  `.complete` marker as absent. Killing the builder between `os.replace` and the
  marker must not let a run launch against the partial tree (§10.3.1).
- **Fan-out is idempotent:** kill the background launcher mid-fan-out; re-calling
  `campaign_start` must launch only the arms with no `study_id` and must not
  re-launch the ones already running (§8.3).
- **Recovery is distinguished from a double launch:** with the launcher lease
  live, a second `campaign_start` on a `running` campaign must be **refused**;
  with the lease expired or naming a dead `(pid, create_time)`, the same call on
  the same status must **proceed**. One test, two lease states, one status — this
  is what pins `launch_state` to the lease rather than to timing (§8.2, §8.3).
- **`write_generation` is not the CAS key:** a mutation whose `write_generation`
  matches but whose `artifact_digest` does not must fail `artifact_changed`. This
  fails if an implementer builds the counter as the guard (§2.6 owns the key;
  §8.2 stores the counter as a read hint).
- **Cursor states:** an arm with no store yet must report as changed, never
  `unchanged`; and a local SQLite study advancing only in `study.db-wal` must be
  reported as moving (§8.3).

### The group filter and the token's binding set

- **`group_filter` is on the ABC, not on one selector:** a `RandomSubsetSelector`
  with a `group_filter` must select only from the filtered candidates. The test
  fails if the filter is implemented on `MetadataGroupSubsetSelector` (§10.3).
- **A filtered subset records its filter:** `subset_generate` with a filter must
  write `group_filter` at the artifact's top level *and* in `selection.params`;
  an unfiltered one writes `{}` in both (§10.2).
- **The token binds the filter:** a `scope:"full"` `plan_token` minted from a
  filtered subset must be refused with `plan_stale` when replayed against the
  same parent with a *different* filter, and accepted with the same one. This is
  USER-21's guarantee, and without the artifact field it cannot be written at all
  (§5.4).
- **Every bound field is checked:** parametrize over §5.4's binding table and
  mutate each field in turn; each must produce `plan_stale` **naming that field**.
  The test fails if a field is in the table and not in the validator — which is
  what "§5.4's table is the binding set" has to mean operationally.
- **A campaign-arm token is a different kind:** a `"campaign_arm"` token must
  validate without `run_name`/`array`/`estimate.node_hours`/`argv_digest`, and
  must be refused at `scope:"full"` with `campaign_arm_scope_full` (§5.4, §6.2).

### `edit_previously_tried` across all six edit kinds

- **One case per kind** against §3.2's table, and specifically: a kept
  `remove_op` must report `keep` (the op is **absent**), and a reverted one must
  report `revert`. A test written only against `insert_op` passes while
  `remove_op` reports the inverse of the truth, which is how the defect shipped.
- **Two removals from one slot do not collide:** removing `BlurGauss` and
  removing `OtsuDetector` from `ops` must be two distinct recorded edits. This
  fails if the match key drops `index` without resolving `class` at record time.
- **`undetermined` still carries evidence:** a `set_params` against a slot
  holding two ops of one class must return the advisory with the numbers and
  `decision: "undetermined"`, not silence.

### Integration

- End-to-end on `load_synth_yeast_plate()`: put → probe → tune (local, grid,
  tiny budget) → export best → deploy (local, `--sample`) → status until
  terminal marker.
- **SLURM tests are marked** and require `sbatch` on `PATH`, following the
  existing `slurm` marker convention. They must **fail, not skip**, when the
  marker is selected and the scheduler is absent.

### The P1 gate (§7)

`optuna_journal_storage.py` must pass **with its temp directory on the target
cluster's shared filesystem** before the journal backend is enabled. Its C2
claim is currently proven only on local APFS. Additionally, the script must be
mutation-tested: swapping `JournalFileSymlinkLock` for a no-op lock, or removing
the pre-create step, must make it fail.

## 6.6 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-6.1 advisory volume~~ → **return every advisory kind the GUI DAG
  validator emits** (`fork`, `stub`, `required_aux`, `container_mode`,
  `missing_input`, `duplicate_input`, `stage_order_hint`, `unsupported_linear`,
  …). Curating them would mean the server deciding which of the engine's own
  findings an agent may see — a judgment call §9.1 places in a skill, not in the
  server. The skill teaches which advisories usually matter; the server reports
  all of them.
