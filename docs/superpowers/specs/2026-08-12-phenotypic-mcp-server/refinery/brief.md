# Context brief — PhenoTypic MCP server refinery

Round 0. Reviewers start here and open source only to verify a specific claim.

## What is being built

An MCP server exposing PhenoTypic (arrayed-colony phenotyping on agar plates) to
an LLM agent: build `ImagePipeline` configs, tune them with `phenotypic.tune`,
deploy over datasets locally or on SLURM. **A new surface over existing engines**
— every numeric result comes from the same code the CLI and GUI already run.

**Spec:** `docs/superpowers/specs/2026-08-12-phenotypic-mcp-server/` — 11 docs,
§1 architecture · §2 state/identity · §3 tool catalog · §4 tune · §5 deploy/SLURM
· §6 errors/limits/testing · §7 prerequisites · §8 workflow/campaigns · §9
responsibilities/skills · §10 subsets/promotion.

**Plan:** `docs/superpowers/plans/2026-08-14-phenotypic-mcp-server/` — README
(constraints + decisions D1a–D6 + drift register + interface findings),
`execution.md` (DAG, six clusters, gates), `phase-1a`/`phase-1b` (19 tasks),
plus finding registers: `review-findings.md`, `MAIN-MERGE.md`,
`MCPB-EVALUATION.md`, `MCP-INTERFACE-AUDIT.md`.

## Deployment reality — not a desktop app

UCR HPCC cluster. Users SSH to a shared login node. `phenotypic` and `uv` are
already installed. The server imports `phenotypic` **in-process** and shells out
to `python -m phenotypic` and `sbatch`. Security boundary is the workspace
sandbox, not authentication — the server runs as the user with their filesystem
and scheduler rights.

## The five mechanisms most worth understanding

1. **One stdio server per session, shared by all subagents** (§1.3). Subagents
   inherit the parent's MCP connection; there is no per-subagent process. Every
   tool must be safe under concurrent invocation, and **anything that blocks
   stalls every subagent**.
2. **Work classes and one compute slot** (§1.5). `W0` introspect · `W1` probe ·
   `W2` study · `W3` deploy. Exactly one process-wide `LocalComputeSlot`
   (semaphore of capacity 1) serializes all local image compute. `W1` runs in a
   persistent killable **probe worker subprocess**, not in-process, because
   `asyncio.wait_for` cannot preempt CPU-bound work.
3. **Disk is the authority** (§2.1). Ids are sandbox-relative paths. The server
   holds no state whose loss matters; it re-reads artifacts on each call and reuses the
   GUI's `RunRegistry` for interprocess locking, generation fencing, and boot
   recovery.
4. **Two human gates** (§8, §10). Campaign approval before subset compute;
   promotion before full-dataset compute. Development is bounded to a
   **development subset**; the full dataset is touched once.
5. **Server vs skill** (§9.1). "The server makes wrong things impossible; the
   skills make right things likely." Mechanism in the server, domain judgment in
   four bundled skills.

## Codebase anchors reviewers may need

- `src/phenotypic/_services/` — **new, built in Phase 1a**: `registry.py`,
  `sandbox.py`, `runs.py`, `argv.py`, `tune_spec.py`. Dash-free tier shared by
  GUI and MCP. Guarded by `tests/unit/services/test_import_purity.py` (subprocess
  leak probe + a GUI-import allowlist pinned by equality).
- `src/phenotypic/_core/_image_pipeline.py`, `_pipeline_parts/_serializable_pipeline.py`
- `src/phenotypic/tune/` — `_spec.py` (validators are the submit-time gate),
  `_tune_cli/_run.py` (1051 lines; `run_tuning`, `_submit_slurm_fleet`, the four
  `_finalize_*`), `_search_space/`, `score/`
- `src/phenotypic/_cli/` — `_cli_slurm_array_scripts.py` (now has a pure
  `build_array_script_spec` + a writing `generate_array_job_script`),
  `_cli_execution_strategies.py`, `_cli_utils.py::parse_slurm_args`
- `src/phenotypic/phenotypicCLI.py` — note: at `src/phenotypic/`, **not** under `_cli/`
- `src/phenotypic/sdk_/_io_constants.py` — every artifact path helper

## Conventions that bind (from CLAUDE.md and the plan)

- `uv` is the sole runner; never bare `python`/`pip`. `uv run --no-sync`.
- Operations are pydantic v2, keyword-only, annotated class fields; no `__init__`.
- Google-style docstrings; `model_json_schema()` field descriptions derive from
  the `Args:` block — **docstring quality is API quality**.
- Paths resolve through `sdk_/_io_constants`; typed suffixes never spelled literally.
- **Every test must be proven able to fail** (mutation) before it is trusted.
- **Assert the structural fact, not a proxy** — no substring searches over source
  text; no `is` checks where a value may be interned. Both rules were earned by
  tests that passed while proving nothing.
- mypy checked by **diff on a cold cache**, never by error count.

## Test setup

`testpaths = ["tests/unit", "tests/smoke", "tests/integration", "tests/gui"]`,
`addopts = -m 'not slow'`. Current green baselines: `tests/unit/services` 61 ·
`tests/unit/cli` 552 · `tests/unit/gui` + `tests/integration/gui` 1746 ·
`tests/gui` 662. `pytest -n auto` oversubscribes on this box (4 cores).

## STATE OF PLAY — read before raising anything

**Phase 1a is COMPLETE and merged** (10 tasks, 3 clusters, 3 gates, one merge of
`origin/main`, one simplify pass). Phase 1b is planned but not started. **No MCP
code exists yet** — `src/phenotypic/mcp/` does not exist. So the spec's §1–§10
tool surface is entirely unimplemented and cheap to change.

Four review passes already ran. Their findings are recorded and **should not be
re-derived**; verify resolutions rather than rediscovering problems:

- `review-findings.md` — the original plan review: 9 blockers, 8 improvements.
- `MAIN-MERGE.md` — the `origin/main` merge, **plus two live spec defects**:
  `deploy_plan` is no longer `W0` (building an sbatch spec now reads every input
  image **twice** plus the pipeline), and its `sbatch_preview` can never be
  byte-identical to what is submitted (every `ATTEMPT_IDS` entry is a fresh
  `uuid4`), which also undermines §5.4's `plan_token` → `argv_digest` binding.
- `MCPB-EVALUATION.md` — packaging rejected; D1a/D5/D6 decided.
- `MCP-INTERFACE-AUDIT.md` — 13 findings against official MCP guidance. Five
  priority: **F1** no tool descriptions for any of the 32; **F2** nothing guards
  the server's own stdout JSON-RPC channel while it imports a package containing
  **19 verified bare `print()` calls** (the spec makes this exact argument for
  the probe worker one level down and misses it for the server); **F3** `W0`
  conflates "no slot" with "instant" across 6+ tools; **F4** MCP request
  cancellation vs the `LocalComputeSlot` is unspecified — if `CancelledError`
  does not release it, every later probe from every subagent blocks for the
  session; **F5** `outputSchema` under fastmcp auto-publishes from return
  annotations (~6.4k tokens/turn).

## THE OME-ZARR COUPLING — the user's specific ask this round

Branch `worktree-ome-zarr-image-store` @ `21a97d3f`, spec at
`docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md` **in the
worktree at `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store/`**.

**It is docs-only and still being iterated** — the user says major details are
outlined but it is not settled. Treat it as a *moving* constraint: flag coupling,
do not design against its details as if fixed.

What it changes: the per-image **HDF5 file is replaced by OME-Zarr (NGFF 0.5)**;
a new **`--mode migrate`** joins full/measure/process/recompile; `--mode
recompile` stops rewriting `deliverables/metadata.csv`; the "dead HDF DataFrame
layer" is retired.

Known exposure in the MCP spec — **confirm rather than assume**:
- §5.4 enumerates `mode` as `"full" | "measure" | "process"` — two more exist.
- §2.3's workspace layout documents `results/<dataset>/{hdf,measurements}/`.
- §5.5 reads per-image HDF as the unit of progress.
- **Most exposed:** §5.4 describes the staged-GPU per-image `.npy` **sidecar** as
  settled fact, while the OME-Zarr spec says that sidecar exists *only* as a
  workaround for HDF's read-only-while-open constraint. Remove the constraint and
  the sidecar — and the three-stage resume contract built on it — may not exist.
- §7 P6 subset staging symlinks image *files*; a zarr store is a directory.

## What the user wants from this loop

Convergence on **spec + plan**, with the OME-Zarr coupling explicitly weighed.
Phase 2's task documents are deliberately unwritten — the loop should leave the
spec in a state worth writing them against.
