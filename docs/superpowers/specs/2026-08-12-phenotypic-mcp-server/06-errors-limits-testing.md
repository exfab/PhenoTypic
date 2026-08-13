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
| `plan_required` | error | `deploy_start` without a `plan_token` (§5.4) |
| `plan_stale` | error | Plan token's pipeline/images/compute digest no longer matches |
| `campaign_not_approved` | error | `campaign_start` on a `draft` campaign |
| `promotion_required` | error | `deploy_start {scope:"full"}` without a `promotion_token` (§10.5) |
| `promotion_stale` | error | Promotion token's pipeline or parent-dataset digest changed since the review |
| `campaign_arm_scope_full` | error | A campaign deploy arm targeting the full dataset — campaigns are subset-scoped (§10.4) |
| `amendment_exceeds_envelope` | error | A mid-campaign amendment outside the approved budget, profile, or scorer |
| `subset_required` | error | A subset-scoped tool given a raw parent path instead of a `subset_id` (§10.3.1) |
| `subset_too_small` | error | Subset below `min_heldout_plates` (6): `derive_split` returns an EMPTY held-out set, so there is no gap at all |
| `subset_too_small_for_heldout` | **warning** | Subset under ~15: a single held-out plate makes the gap a one-sample estimate |
| `selector_unavailable` | error | `SubsetSelector.availability()` is `False` — e.g. `EmbeddingSubsetSelector`, which raises rather than falling back to random (§10.3) |
| `group_key_not_in_metadata` | error | `MetadataGroupSubsetSelector.group_key` names no column in the metadata CSV |
| `arm_scorer_mismatch` | error | A campaign arm's scorer differs from the campaign scorer (§8.2) |
| `screening_unsupported_on_slurm` | error | `--screen` + SLURM, which silently drops screening today (§7 P4). **Conditional on OQ-4.1** — if screening is not exposed at all, this code does not ship. |
| `output_not_empty` | error | Deploy target non-empty; names `run_name`/`resume`/`restart` |
| `resume_incompatible` | error | Pre-validated `validate_resume_compatibility` mismatch; names the field |
| `scheduler_jobs_active` | error | Resume/restart while ledgered jobs are live |
| `local_slot_timeout` | error | `W1` waited past `probe_timeout_s` for the slot |
| `probe_cap_exceeded` | error | `n_images` above `limits.probe_max_images` |
| `environment_mismatch` | error | SLURM work requested in a `local` environment with no profiles |
| `scheduler_unreachable` | error | Detection said `slurm`, submission found otherwise; reports probe staleness |
| `not_owned` | error | Cancel targeted a run with no `RunRegistry` record here |
| `version_drift` | **warning** | Spec `phenotypic_version` ≠ installed; matches the engine's warn-only posture |

## 6.3 Limits

| Limit | Default | Enforced by |
|---|---|---|
| `probe_max_images` | 4 | Hard cap on `pipeline_probe.n_images` |
| `probe_timeout_s` | 300 | Wall clock on `W1`, including slot wait |
| Local compute concurrency | 1 | `LocalComputeSlot` (§1.5) |
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
  `RunRegistry`, a fresh server must claim the slot rather than admit a second
  local job; with a dead one, it must CAS to `failed`.

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

- **OQ-6.1 — advisory issue volume.** The GUI DAG validator emits several
  advisory kinds (`fork`, `stub`, `required_aux`, `container_mode`,
  `missing_input`, `duplicate_input`, `unsupported_linear`, …). Should all of
  them surface on every `pipeline_put`, or only a curated subset? Returning all
  is honest but noisy, and noise trains an agent to ignore issues.
