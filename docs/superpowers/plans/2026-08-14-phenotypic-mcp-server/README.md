# PhenoTypic MCP Server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an MCP server that lets an LLM agent build `ImagePipeline`
configurations, tune them, and deploy them over datasets — locally or on SLURM —
as a thin adapter over the engines the CLI and GUI already run.

**Architecture:** Three layers, bottom-up. The existing engines (`_core`, `tune`,
`_cli`) are untouched. A new Dash-free `phenotypic/_services/` tier is promoted
out of `gui/` so two user-facing surfaces can share one tested API. A thin
`phenotypic/mcp/` tool layer sits on top of `_services`, owning transport,
dispatch, resource routing, and the structured error envelope — and importing
`phenotypic.gui` never.

**Tech Stack:** Python 3.11+, pydantic v2, the official `mcp` Python SDK
(FastMCP style, stdio transport), optuna 4.9.0, Click, pytest, `uv` as the sole
package manager and runner.

**Spec:** [`docs/superpowers/specs/2026-08-12-phenotypic-mcp-server/`](../../specs/2026-08-12-phenotypic-mcp-server/)
— eleven documents. The plan argues from the spec; executors read both. Section
references below (`§3.2`, `§7 P2`) resolve against it.

---

## Global Constraints

Every task's requirements implicitly include this section.

- **`uv` is the sole runner.** Never bare `python` or `pip`. `uv run <cmd>`,
  `uv add <pkg>`, `uv sync`.
- **Lint with explicit paths only:** `uv run ruff check --fix <paths you changed>`.
  Bare `ruff check --fix` rewrites the whole repo.
- **Type check:** `uv run mypy src/phenotypic`.
- **Operations are pydantic v2 models, keyword-only.** No hand-written
  `__init__`; parameters are annotated class-level fields; normalization goes in
  a `field_validator`.
- **Google-style docstrings everywhere.** Field descriptions in
  `model_json_schema()` are auto-derived from the `Args:` block, and §3.1 makes
  that the agent-facing contract: **docstring quality is API quality.**
- **Doctest examples must be runnable** using `load_synth_yeast_plate()`.
- **Never create example files or notebooks.** Examples live in docstrings.
- **Path helpers, never hand-joined names.** Every artifact path resolves through
  `phenotypic.sdk_._io_constants`.
- **Typed suffixes are never spelled literally:** use `CONFIG_SUFFIX_PIPELINE`
  (`.json.pht-pipe`) and `CONFIG_SUFFIX_TUNING` (`.json.pht-tune`) via
  `ensure_typed_json_suffix`, and match with `matches_any_suffix` — **never**
  `Path.suffix`, which sees only the trailing `.pht-tune`.
- **Every test must be proven able to fail** before it is trusted — by
  reintroducing the bug it guards or by a one-line mutation of the code under
  test. A check that cannot run must **fail**, not skip. This is a project-wide
  rule (§6.5) and it binds on every task here, not only where restated.
- **Assert the structural fact, not a proxy for it.** Earned three times over in
  Phase 1, each time by a test that passed while proving nothing:
  - **No substring search over source text.** `assert "phenotypic.gui" not in
    inspect.getsource(m)` fails on a module that merely *mentions* the GUI in a
    docstring and passes on one importing it under an alias. Parse the imports
    (`ast`) or probe at runtime in a subprocess.
  - **No `is` check where the value may be interned.** CPython interns short
    string literals, so `assert a.X is b.X` passes even when `b` contains a
    genuine parallel `X = "presets"` — verified: `'presets' is 'presets'` → True.
    The same hole applies to `int` in `[-5, 256]`, `bool`, and `None`. Existing
    shim-identity tests are sound only because their symbols are functions and
    classes; the idiom is one `str` constant away from worthless. For a value
    that may be interned, assert the *structure*: the name arrives by an
    `ImportFrom` of the canonical module and is bound by no module-level
    `Assign`/`AnnAssign`/`def`/`class`.
- **Any change touching a re-export shim runs an AST surface check first.** Parse
  every `.py` under `src`, `tests`, `scripts` for the names actually imported from
  the shimmed module — including multi-line parenthesised imports, aliased
  attribute access, and monkeypatch string targets — and assert the shim still
  exports all of them. **Five shim-surface defects occurred in Phase 1a alone**
  (`ColumnRefSpec`; the two `_sandbox` privates; `_normalize_setup_metadata_groupby`
  twice — once when main added it, once when a simplify edit dropped it). Four were
  caught by a failing test elsewhere; the fifth only by running this check. It is
  cheap and it is the difference between finding the problem now and finding it
  in a suite that passes.
- **Check mypy with a COLD cache, on scratch, and diff.** A warm `.mypy_cache`
  produced `416 errors in 123 files` where a cold run gives `417 / 124` — which
  is exactly main's number. Every count discrepancy chased in this phase traced
  back to cache state, not to code. Pass `--cache-dir` under `/scratch` for any
  number quoted in a gate. Then compare stashed-vs-current output,
  ignoring line order and internal typevar ids. (Instability was suspected —
  `421 errors in 125 files` and `420 errors in 124 files` were both observed —
  but C2's gate then got 421/125 on three runs including a cold cache, and
  diffed a rebuilt pre-cluster tree to 421/125 with only six typevar-id lines
  differing. The discrepancy is more likely a differing tree state than mypy
  nondeterminism. Diffing is still the right method: it is unaffected either
  way, and a count comparison tells you nothing about *which* diagnostic moved.)
- **Vendored reference sources under `docs/superpowers/specs/*/refs/` are
  read-only.** Never lint, format, or fix them.
- **Cost convention:** every tuning score is a cost in `[0, 1]`, lower is better,
  minimized. Never present one as an accuracy.
- **Two hard refusals that no task may weaken:** no `--overwrite` reachable from
  any tool (it is `shutil.rmtree`), and no raw sbatch passthrough
  (`parse_slurm_args` constrains neither keys nor values).

---

## Review protocol

**Reviewers run at cluster boundaries — six of them, one per cluster.** An
earlier version of this plan put a reviewer after each of the 18 tasks; the plan
review argued that was more machinery than the problem needs, and the user agreed.

The argument that won: nine of these tasks are `git mv` plus a re-export shim,
collectively guarded by one import-purity gate and an unchanged GUI suite. For
those, **the shim-identity test IS the review** — a reviewer reading that diff
would be checking by eye what a passing `assert shim is canonical` already proves
mechanically.

What a cluster reviewer must still check, unchanged from before:

| Check | Why it cannot be skipped |
|---|---|
| **No false greens** | Every "prove it can fail" step is a *claim by the implementer* until someone verifies it. The plan review found two tests that pass without proving anything (I1a, I1b) — both written by the plan's author. |
| **No scope leak** | A move that quietly takes a behaviour change with it is invisible in a green suite and expensive to bisect later. |
| **Interfaces hold** | Later clusters are written against earlier `Interfaces` blocks. A rename breaks work nobody is watching yet. |

`execute-plan-orchestration:implementation-test-reviewer`, Opus, high effort,
scoped to the cluster's combined diff. The cluster's own tests plus
`uv run ruff check <changed paths>` and `uv run mypy src/phenotypic` run *before*
the reviewer is dispatched, not after.

**A cluster does not hand off with an unaddressed correctness finding.** Findings
are fixed in a follow-up commit or recorded with a reason. Any finding that
conflicts with a *design* decision stops the line and returns to the user.

**End of each phase — simplify.** After 1a (C1–C3) and again after 1b (C4–C6),
`code-simplifier:code-simplifier` over the phase's combined diff: dedupe, reduce,
clarify — quality only, no behaviour change. Apply, then re-run the affected suites
plus `tests/unit/gui` and `tests/integration/gui` to prove nothing observable moved.

---

## Decisions taken before writing this plan

These were open when the plan started and are now settled. They are recorded
here because a reader of the spec alone would still find them open.

| # | Decision | Rationale |
|---|---|---|
| ~~D1~~ → **D1a** | **SUPERSEDED 2026-08-19. PyPI `fastmcp` 3.x, not the official SDK's bundled FastMCP.** | D1 originally chose `mcp.server.fastmcp`. The `build-mcp-server` skill explicitly recommends PyPI `fastmcp` 3.x and warns against "the frozen FastMCP 1.0 bundled in the official `mcp` SDK" — which is what D1 had picked. Decisive factor: **D6 adopts elicitation**, and a frozen 1.0 is exactly where that capability is likely absent. Still an optional extra; the core package gains no dependency. **Phase 2A must verify** the pinned version's `Context` exposes `elicit` and `report_progress` before the tool layer is written against them. |
| **D5** | **Tool annotations on every tool** (`readOnlyHint`, `destructiveHint`, `title`) | Absent from all ten spec files. Annotating the ~17 `W0` read tools `readOnly` lets a host auto-approve them, and leaving `deploy_start` / `campaign_start` / `tune_start` / `workspace_cancel` unannotated makes the host raise a confirmation. This enforces §9.1's server-vs-skill line **at the host level** rather than in prose. Cheap now (32 registrations + one test), materially more expensive after the tools exist. |
| **D6** | **Elicitation for the two human gates — shape now, implement in Phase 2C** | §8.2 concedes the server "cannot verify that a human approved anything… an agent could fabricate the field". That constraint is no longer real: elicitation ships in Claude Code ≥2.1.76, and a host-rendered form comes from the user's keyboard rather than the agent's token stream — turning `campaign_approve` and the §10.5 promotion gate from provenance into **actual confirmation** for the two irreversible spends. **Shape now:** `human_response` becomes required-*unless-elicited*, so adopting it later is not a breaking signature change. **Unverified and must be tested live before implementation:** whether elicitation raised from a *subagent's* tool call surfaces to the human, given §1.3's single shared connection. The current fabricate-explicitly design remains the mandated fallback when the host lacks the capability. |
| D2 | **P5 moves into Phase 1** | §7 calls it independent cleanup. On UCR HPCC `--account` is mandatory for the `exfab` and `preempt` partitions, and the tune CLI drops `account` entirely — so **no tune fleet can reach the GPU node until P5 lands**. Doing it first also retires §5.2.1's expressibility check instead of building it. |
| D3 | **No server-side `plate.nrows`/`ncols` backstop** — ship §9.3.5 as specified | A cross-check of `nrows × ncols` against the scorer's expected counts was considered and **rejected on domain grounds: grid sections are not always filled**, so the product is a poor proxy for expected colony count and the check would fire on legitimate partial layouts. The defence stays the `phenotypic-assay-triage` skill. |
| D4 | **v1 ships in three gated sub-phases** (2A / 2B / 2C) | Each leaves working, reviewable software. 32 tools behind one review gate is not reviewable. |

---

## Interface audit — five findings to fix before the first handler exists

From `MCP-INTERFACE-AUDIT.md` (13 findings). These five have asymmetric cost:
cheap now, expensive once 32 handlers exist. **F2 is verified in-tree.**

| # | Finding | Status |
|---|---|---|
| **F2** | **Nothing guards the server's own stdio protocol channel.** §3.2 makes exactly this argument one level down — the probe worker uses a dedicated pipe, "never the worker's stdout", because `tqdm` and a bare `print()` would corrupt the stream. The **server** speaks JSON-RPC on its own stdout and imports `phenotypic`, including `detect.nn`, which §3.1 *requires* be reachable. **Verified: 19 bare `print()` calls across `detect/`, `_core/`, `tune/`**, one of them at `detect/nn/_helper/_checkpoint_manager.py:830`. A single print corrupts the session for every subagent and surfaces as a host parse error nowhere near its cause. **Fix:** rebind `sys.stdout` → `stderr` before importing `phenotypic`; add as a seventh refusal in §6.4; test it. | **VERIFIED — highest priority** |
| **F1** | **No tool descriptions anywhere, for any of the 32.** `tool-design.md` is emphatic: "the description is the contract… the only thing Claude reads before deciding whether to call the tool", and requires a "does NOT do" clause. §1.2 fixes `model_json_schema()` as the contract — but that is the *payload* `catalog_operation_detail` returns, not the MCP tool description. Two different contracts; the spec writes one. Under D1a, fastmcp takes descriptions from docstrings, so all 32 get improvised during implementation. Sibling-confusion pressure is unusually high here: `deploy_plan`/`deploy_start`, three `*_status` tools, `tune_status{progress}` vs `{results}`, `campaign_approve` vs `promotion_approve`. | open |
| **F3** | **`W0` conflates "takes no compute slot" with "is instant".** §5.5 already corrects this for `deploy_status` but never carries the rule back to §1.5, and 6+ other `W0` tools do real blocking work on the shared event loop — `workspace_info` (rehydrate + `squeue`), first-call `catalog_operations` (discovery over 13 packages), `campaign_status` (N store-opens), `deploy_plan` (directory digest over a 480-image parent). Under §1.3's single connection each stalls every subagent, falsifying §3.4's "interleave freely". **Fix:** split §1.5's `W0` row into pure/inline vs I/O-bound/executor, and tag all 32. | open — compounds the `deploy_plan` defect already recorded in MAIN-MERGE.md |
| **F4** | **MCP request cancellation vs the `LocalComputeSlot` is unspecified.** The spec covers timeout, OOM and server restart, never the host cancelling a request — the likeliest case with N subagents on one connection. If `CancelledError` does not release the slot, **every subsequent probe from every subagent blocks for the session**: the exact deadlock §3.2 rejected the in-process design to avoid. Also `probe_timeout_s=300` is set with no reference to the host's tool-call timeout. | open |
| **F5** | **`outputSchema` is undecided, and under D1a may be decided for us.** fastmcp derives it from return annotations, so 32 handlers annotated `-> ToolEnvelope` publish 32 serialized copies in `tools/list` (~6.4k tokens/turn — JSON Schema has no cross-tool `$ref` sharing). **Fix:** decline explicitly in §3.0 and assert in Phase 2A that no tool publishes one. | open |

Eight further findings (F6–F13) are in the audit document, including schema-level
caps for `n_images`/`limit`, a `Literal` discriminator for `edits[].kind`, an
unbounded `workspace_list`, declaring `logging: {}`, an `instructions` string,
and versioning the tool contract independently of `phenotypic`.

**Also from the audit:** Appendix A carries the full 32-row annotation matrix for
D5 — 16 read-only, 16 write, with the non-obvious calls argued: the `*_put` tools
are **not** idempotent, `pipeline_patch` emphatically not (cumulative edits — the
annotation is what stops a host retrying into a corrupted pipeline),
`tune_export_best` is a *write* despite its name, and `openWorldHint: true` on
exactly the four tools that can trigger a gated checkpoint download.

---

## Drift register — spec citations that no longer hold

Found by verifying every load-bearing `file:line` in §1–§10 against
`feat/mcp-server` at `c847373c8`. **Fix the spec in the same change that
implements the affected task**, so the two do not diverge further.

| # | Spec says | Reality on this branch | Affects |
|---|---|---|---|
| DR1 | `IMAGE_EXTS` lives in `gui/builder/_directory_browser.py:20-21`; relocate it to `sdk_/_io_constants.py` (§1.4, §7 P2) | **Already moved.** Defined at `gui/_config.py:429`, which is **Dash-free** (imports only argparse/logging/pathlib/typing/urllib + `phenotypic.sdk_`). `_directory_browser.py:23` re-exports it for back-compat, and `_classifier.py:34` still imports through that Dash-laden shim. | Task 2 — the job shrinks to one relocation plus one repointed import, not a three-file untangle |
| DR2 | `_find_class_in_phenotypic` in `_serializable_pipeline.py` (§3.2) | Actual path is `_core/_pipeline_parts/_serializable_pipeline.py:619`; submodule list begins `:645` | Tasks 10, 14 |
| DR3 | `_submit_slurm_fleet` builds `slurm_args` at `_run.py:797-805` (§5.2.1, §7 P5) | Function at `:724`; the four `slurm_*` params at `:733-736`; the `if slurm_partition is not None:` chain at `:798-804`. Substance holds, offsets shifted | Task 16 |
| DR4 | §10.1 cites "the autonomy question **§8.6** raised" | **§8.6 does not exist.** `08-workflow-and-campaigns.md` jumps 8.5 → 8.7. Dropped or renumbered during revision | **FIXED** — §10.1 now cites OQ-8.1/OQ-8.2 (§8.8), where the question is actually recorded |
| DR5 | §4.7 resolves OQ-4.1 with "`tune_put_spec` takes `screen: false` by default" | §4.2's argument table has **no `screen` row** | **FIXED** — row added to §4.2 naming the SLURM refusal (§7 P4). Still binds Task 15 and Phase 2B |

**Confirmed unchanged** (spot-checked, all still exact): `_space.py:33-34` Dash
imports and the `:134/:161/:209` pure vs `:396/:468/:503` view split;
`_setup_authoring.py:28` importing both pure symbols; `run_console/_state.py:70`
`RunConsoleState` and `:515` `to_argv`; `_operation_registry.py:811-823`
`_REGISTRY` singleton; `gui/shell/__init__.py:17` and `gui/tune/__init__.py:18`
eager Dash imports; `discover()`'s eight-module list at `:198-205`; the
`if slurm: return _submit_slurm_fleet(...)` at `_run.py:593-595` sitting **before**
`if screen:` at `:623`; all four `_finalize_*` functions with call sites only
inside `run_tuning` (`:631`, `:637`).

---

## Phase map

```
Phase 1  PREREQUISITES — engine and refactor work, no MCP code
  1a  P2  _services promotion (9 moves + 1 split + 1 extraction)   Tasks 1–9
  1b  P3  catalog reconcile, descriptor, digest, subset/ package   Tasks 10–14
      P4  --screen + --slurm silent no-op becomes an error         Task 15
      P5  tune CLI gains --slurm key=value                         Task 16
      P6  subset staging (flat/ + nested/)                         Task 17
      P7  distributed finalize entry point                         Task 18
        │
        ▼   GATE: import-purity test green, GUI suite unchanged, ledgers green
Phase 2  v1 TOOL SURFACE
  2A  server skeleton, envelope, error mapping, probe worker,
      catalog(3) + pipeline(5) + workspace(4)                      — usable for construction
        │  GATE
  2B  assay(2) + subset(3) + tune(5) + campaign(5)
        │  GATE
  2C  deploy(3) + promotion(2) + 4 bundled skills + `phenotypic-mcp setup`
        │
        ▼
Phase 3  DISTRIBUTED TUNE (P1)  — gated on L1, see below
```

**Phase 1 has no MCP code in it at all.** Every task is engine or refactor work
that stands on its own merits and is verifiable by the existing test suite. That
is deliberate: it keeps the largest, riskiest changes away from a new tool layer
whose contract is still being exercised for the first time.

## The L1 gate — status

Phase 3 (P1, the `JournalStorage` backend) is blocked on L1: the negative
control in `optuna_journal_storage.py` must **actually lose trials** on the
target mount, or a green C2a proves nothing (§7).

**RESOLVED by measurement.** Two runs, and the second is the one that counts.

**Single-node (job 27466782, `short`, 11 s).** `DISCRIMINATION: NONE` on both
`/bigdata` and `/rhome`; both exit 1. As literally specified, L1 does not pass
here — but not because the backend is unsafe. Both mounts are **GPFS**, not the
NFS/Lustre §7 assumed, and GPFS enforces POSIX byte-range semantics cluster-wide,
so a no-op lock loses nothing. Worse, `multiprocessing` puts every worker on one
host, so the run measured the local kernel's `O_APPEND` atomicity and never
engaged the distributed token manager a fleet depends on.

**Cross-node (C7, job 27468703, four nodes `c[07,09,12,14]`).** The script gained
`init` → N × `worker` → `verify` roles so `srun` can place one worker per node,
with per-trial hostname stamping and a `--require-distinct-nodes` guard proving
the run really was distributed:

```
ok [C7-symlink] 60 trials persisted intact across 4 nodes ['c07','c09','c12','c14']
ok [C7-noop]    60 trials persisted intact across 4 nodes ['c07','c09','c12','c14']
VERDICT: NO DISCRIMINATION, cross-node.
```

**Conclusion: journal storage is safe on this cluster's shared mount — because of
the filesystem, not because of the lock.** The symlink lock is redundant on GPFS
rather than broken, and ships enabled anyway: it costs nothing and is what keeps
the same code correct on an NFS deployment elsewhere. **P1 is unblocked.**

Two limits stated plainly: this is 4 workers × 15 trials, not a 32-worker fleet
(C6's ~65× throughput headroom is the argument that contention stays irrelevant
at scale, and it is an argument, not this measurement); and absence of loss over
a finite sample is consistent with GPFS's architectural guarantee rather than
independent proof of it. **The result is filesystem-specific — re-run C7 on any
cluster whose shared mount is not GPFS.**

Artifacts: `run_l1_cross_node.sbatch` beside the script; logs under
`/bigdata/exfab/anguy344/mcp-l1-gate/logs/`. Note the first cross-node attempt
(27468686) died on `srun`'s default CPU binding against this partition's
non-contiguous masks — fixed with `--cpu-bind=none`, and the script reported
`INCONCLUSIVE` rather than inventing a verdict, which is the behaviour to keep.

## Open decisions

| # | Question | Status |
|---|---|---|
| OD1 | Given `DISCRIMINATION: NONE` on GPFS, does P1 ship on the journal backend, stay on Postgres, or wait for a cross-node test? | **CLOSED.** Cross-node test written and run (C7, job 27468703): the symlink-locked run survives a genuine 4-node fan-out on GPFS, and so does the control. P1 ships on the journal backend, documented as filesystem-dependent. |
| OD2 | Does the spec's L1 gate text get rewritten to account for GPFS? | **CLOSED.** §7 now carries the measured result and a corrected gate: the **symlink-locked** run must survive a **cross-node** fan-out with `--require-distinct-nodes`; the negative control's outcome is informative, not required. |

## Documents

| Doc | Covers |
|---|---|
| [phase-1a-services-promotion.md](phase-1a-services-promotion.md) | P2 — Tasks 1–9 |
| [phase-1b-engine-prerequisites.md](phase-1b-engine-prerequisites.md) | P3–P7 — Tasks 10–18 |
| [execution.md](execution.md) | The dependency DAG, the six Opus clusters, and where the review and simplify gates sit |
| Phase 2A/2B/2C | Written at the Phase 1 gate, against the code Phase 1 produces |

Phase 2's task documents are deliberately not written yet. They specify 32 tools
against `_services` signatures that Phase 1 creates, and writing them now would
mean inventing those signatures twice — the spec already records what the tools
must *do*; the plan's job is to say how, against real code.
