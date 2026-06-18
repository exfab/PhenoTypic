# Code Review — `redesign/param-sweep` (legacy `sweep` → `tune` engine)

**Date:** 2026-06-06
**Branch:** `redesign/param-sweep` (HEAD `425f6e99`) vs `main` (merge-base `1d794009`)
**Scope:** 379 files, +45,286 / −10,936 (≈33 of those are in-tree design specs/plans
used as the review oracle, not code).
**Method:** 7 specialized agents over independent file scopes (6 static reviewers + 1
live verification pass), then orchestrator synthesis. Every **blocking** finding was
re-read at its cited `file:line`; every runtime claim is backed by an actual
command/console transcript. **Outcome: report only — no source was edited; worktree
confirmed clean after the fan-out.**

---

## What the branch does

Deletes the legacy `sweep` parameter-exploration subsystem outright and replaces it with
a new `tune` engine:

- **Optuna-backed** strategies (TPE/Bayesian, grid, random, enumerate) with
  pruning/screening (`tune/_strategies/`, `tune/_screening*.py`), lazily imported so the
  base package and Grid/Random paths stay Optuna-free.
- **Supervised + reference-free scoring** (`tune/_scoring/`) with object-matching
  metrics (IoU/Dice/precision/recall), composite aggregation, Pareto/multi-objective
  support.
- **SLURM-distributed workers** sharing one **Postgres-backed Optuna study** (
  `tune/_study/`, `tune/_study_store.py`, `_execution/_slurm.py`).
- **Per-operation `TuneSpec` search-hint annotations** on `enhance/*` + `detect/*` ops,
  consumed by a search-space inference layer (`tune/_search_space/`).
- A read-only **Dash "Tune co-pilot"** GUI at `/tune/` (Monitor / Curate / Space /
  Launch) (`gui/tune/`).
- New optional dep group `tune = [optuna>=4.0, sqlalchemy>=2.0, psycopg[binary]>=3.1]`;
  new pytest markers `postgres` / `slurm` (autoskip unless `PHENOTYPIC_TEST_PG_URL`
  set / `sbatch` on PATH).

---

## Verdict

A **well-engineered, genuinely large feature** with strong test discipline. It is **not
yet mergeable**: **4 blocking** correctness/operational defects plus a cluster of
should-fix items. Risk is concentrated exactly where the architecture predicted — *
*distributed write safety** and **supervised scoring correctness**.

---

## Resolution status (2026-06-06)

**All findings in this report have been addressed** in a 5-phase fix pass on
this branch (review-only audit → fixes). Final state: **1611 passed, 11
skipped**; mypy/ruff clean on the changeset; ledger gates green; lazy-import
boundary holds.

| Finding                    | Resolution                                                                                                                                                                                                                                                                        |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **B1** GT-mask collapse    | `gt_format` flag added; `_read_mask` preserves labels; **binary path uses a detection-independent geometric cell map** (`grid` edges, not `get_section_map()`) + `match_iou_greedy`, so missed cells are penalized and GT isn't clipped to the prediction. Real-data tests added. |
| **B2** credential leak     | Password-bearing storage URLs rejected at `_resolve_storage_url` (the single chokepoint feeding marker + SLURM script).                                                                                                                                                           |
| **B3** non-atomic writes   | Shared `phenotypic.sdk_.atomic_write_text/_bytes` (temp+`os.replace`) applied to **every** tune output writer, incl. `_split.py` and `_auto_space.py` (caught in final cross-phase review).                                                                                       |
| **B4** Curate race + CSS   | Overlay poll self-heals from `OverlayCache.peek` (TOCTOU fixed under lock); difference-mode CSS specificity fixed. Both reproduced and RED-verified.                                                                                                                              |
| Hub binding / ledger drift | **Chunk C run-binding implemented** — `/tune/` is reachable with data through `phenotypic-gui` (live-verified, 0 server 500s); FEATURES/WORKFLOWS rows now accurate.                                                                                                              |
| Distributed should-fix     | Stale-`RUNNING`-trial reconciliation, single shared study handle, transient-DB retry, SLURM config passthrough, `study_name` sanitize.                                                                                                                                            |
| Search-space should-fix    | Canny non-overlapping windows, enum-value round-trip, `sigma_color` None-categorical, `FocusEdgePhase.k`→0.5, `conditional_on` op_class stamping, distinct `ExcludeReason`.                                                                                                       |
| GUI should-fix             | `_BASE_PIPELINES` LRU cap; `TUNE_RUN_ROOT_STORE` hoisted to page root.                                                                                                                                                                                                            |
| Tests/CI                   | `skipif(not _OPTUNA)` guards; concurrent-open smoke; Flask-client callback-wiring tests; deflaked live-timeout test.                                                                                                                                                              |
| Nits                       | `ddof=1` CV; shape assert; raw-ρ unattended gate; `match_iou_greedy` docstring; migration doc; `.gitignore` roots.                                                                                                                                                                |

**Still pending an environment:** the Postgres/SLURM **manual-test checklist**
below remains the way to exercise the distributed path against a real
cluster/DB (those suites autoskip in CI). The full tutorial-screenshot set
(beyond `tune_copilot`) should be regenerated from a developer workstation.

---

## Verified-passing (runtime evidence)

| Check                | Command                                                                                                               | Result                                                            |
|----------------------|-----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| Type check           | `uv run --extra gui --extra tune mypy src/phenotypic/tune src/phenotypic/gui/tune`                                    | ✅ `Success: no issues found in 66 source files`                   |
| Lint                 | `uv run ruff check src/phenotypic/tune src/phenotypic/gui/tune`                                                       | ✅ `All checks passed!`                                            |
| Unit suite           | `QT_QPA_PLATFORM=offscreen uv run --extra gui --extra tune pytest tests/unit/tune tests/unit/gui/tune -p no:randomly` | ✅ 676 passed, 10 skipped (26.7s)                                  |
| GUI integration      | `… pytest tests/integration/gui -k tune`                                                                              | ✅ 44 passed; live-timeout test stable ×3                          |
| Lazy-import boundary | `python -c "sys.modules['optuna']=None; … import phenotypic.tune"`                                                    | ✅ imports OK, optuna never loaded                                 |
| Ledger gates         | `scripts/check_workflows_md.py`, `scripts/check_features_md.py --strict`                                              | ✅ 16/16 workflows, 240/240 features                               |
| Sweep removal        | grep across `src/docs/tests`                                                                                          | ✅ clean — no dangling imports/mounts/entry points                 |
| Live GUI             | 1056 `/_dash-update-component` POSTs                                                                                  | ✅ all 200, zero server tracebacks (**callback-500 risk refuted**) |
| Sandbox containment  | `../../etc`, `/etc`, symlink probes                                                                                   | ✅ refused at both UI and backend                                  |

**Invocation note (recurring footgun):** bare `uv run pytest` aborts collection (
pytest-qt needs a Qt binding) and re-syncs away extras. Always use
`QT_QPA_PLATFORM=offscreen uv run --extra gui --extra tune pytest …`.

---

## 🔴 Blocking (fix before merge)

### B1 — Supervised mask-tier scoring collapses every colony into one pseudo-object

*Confirmed by re-read.
Files: `tune/_scoring/_gt_loader.py:200-215`, `tune/_scoring/_matching.py:49-50,180`,
`tune/_scoring/_supervised.py:289-295`.*

`_read_mask` force-casts every ground-truth mask to `bool` (`.astype(bool)`).
`_object_labels` then does `np.unique(...)`, so a boolean GT yields a **single** label
`[1]` = the entire foreground union. Per-instance IoU/Dice matching — the intent in
`supervised-scorers.md` — degenerates: the whole plate becomes one "GT object", every
other predicted object is a false positive, and `match_per_grid_cell` assigns the entire
foreground to one grid cell. The mask-tier test only asserts self-match `== 1.0` (true
for any identical pair), so the bug is invisible to CI.

**Action:** decide the intended GT modality. If **instance** GT → the `bool` cast
destroys the label info (preserve the integer dtype). If **binary/semantic** GT → the
matching path must fall back to pixel-level Dice/IoU instead of `_object_labels`. Add a
partial-overlap mask-tier test on real `load_synth_yeast_plate()` data.

### B2 — Postgres credentials written in plaintext to shared-filesystem files

*Confirmed by re-read.
Files: `tune/_tune_cli/_run.py:148,163`; `_execution/_slurm.py:112`.*

`_write_run_marker` writes the resolved `storage_url` verbatim into
`.pht-tune-cache/run.json`; the SLURM generator interpolates it into `tune_workers.sh`.
A `postgresql+psycopg://user:secret@host/db` URL (the natural first instinct) lands on a
group-readable HPC filesystem. The docstring *recommends* password-less URLs but nothing
enforces it; the existing `try/except OSError` only handles over-quota, not secrecy.

**Action:** reject password-bearing URLs early (`urlsplit(...).password is not None` →
actionable error pointing at `.pgpass`/`PGPASSWORD`), or redact the password from
anything persisted while keeping the real URL only in the process env.

### B3 — No atomic writes for any deliverable or marker

*Confirmed by re-read. Files: `tune/_tune_cli/_run.py:163` + finalize
writers (`best_pipeline.json`, `tuning_spec.json`, `param_importance.json`,
`generalization.json`, `pareto_*.json`); `tune/_study_store.py:182` (`to_parquet`).*

Every critical output uses a bare `write_text` / `to_parquet` (open→write→close). The "
robust marker write" commit only added an `OSError` guard — **not** atomicity. A SLURM
walltime kill mid-write silently replaces a good file with a truncated one, and a
partial `run.json` breaks GUI auto-discovery.

**Action:** write to a temp sibling then `os.replace(tmp, dst)` (atomic on POSIX) for
all of the above.

### B4 — The core Curate A/B comparison is visibly broken

*Confirmed live by the verification agent (predicted statically by Agent 4).
Files: `gui/tune/_curate_overlays.py:158-205`, `gui/tune/_callbacks.py:1047-1213`;
`gui/tune/_assets/tune.css:72` vs `:209`.*

Two independent bugs in the headline flow:

- **Overlay race / stuck spinner:** pin trial A (renders), then pin B → slot A reverts
  to a permanent "rendering…" spinner that only a full page reload clears.
  `take_overlay` is a consume-once destructive read; re-submitting the batch on B's pin
  drops slot A's resolved future and the poll (`_poll_curate_overlays`) never resolves
  it again.
- **Difference-mode CSS leak:** the side-by-side panels stay visible in Difference mode
  because `.tune-view-hidden{display:none}` (line 72) and
  `.tune-curate-sidebyside{display:grid}` (line 209) have **equal** specificity (0,1,0)
  and the later `grid` rule wins.

**Action:** evaluate `future.done()` under `_PENDING_LOCK` (or re-submit on pin) so the
poll can always resolve; raise the hidden-rule specificity (
`.tune-curate-sidebyside.tune-view-hidden{display:none}`).

> **Context that reframes B4 — hub run-binding is unimplemented.**
`gui/shell/_app.py:299` mounts `/tune/` with `root=None`, and the sidebar run-binding is
> explicitly deferred ("Chunk C"). The verification agent confirmed that *
*via `phenotypic-gui` only the empty state is reachable** — the loaded
> Monitor/Curate/Space/Launch views (where B4 lives) cannot be opened by an end user.
> Yet
`FEATURES.md`/`WORKFLOWS.md` mark these "✅ shipping." Either Chunk C lands in this PR or
> the ledger overstates the surface (see Drift §). This lowers the *user-facing* urgency
> of B4 while the views remain hub-unreachable, but the code is still wrong.

---

## 🟠 Should-fix

### Distributed / concurrency *(static review; postgres+slurm suites autoskip, so not

runtime-verified)*

- **Stale `RUNNING` trials inflate the budget.** A killed worker leaves its trial
  `RUNNING`; the budget check (`tune/_strategies/_optuna.py:286-294`) counts only
  `COMPLETE`/`PRUNED`, so a relaunched fleet overshoots `n_trials` and never terminates
  under repeated kills. → call `optuna.storages.fail_stale_trials(...)` at worker start.
  **No kill-and-resume test exists, even under SQLite.**
- **Two live study handles per worker** — both `OptunaStudyStore.__init__` and
  `OptunaStrategy.build` call `create_study(load_if_exists=True)`. "One study" holds in
  the DB but not in-process; doubles Postgres connections. (
  `tune/_study/_optuna_store.py:73`, `tune/_strategies/_optuna.py:141`)
- **Postgres-unreachable mid-run is unhandled** — no retry/backoff around `ask`/`tell`/
  `append`; one network blip kills the worker (leaving a stale trial). (
  `tune/_tune_cli/_worker.py:54-89`)
- **SLURM config is rigid** — partition hardcoded `"batch"`, `n_workers` silently capped
  at 8, and user `slurm_args` (`--mem/--time/--qos`) are dropped (
  `tune/_tune_cli/_run.py:497-531`). Will fail on clusters without a `batch`
  partition. → add `--n-workers/--slurm-*` passthrough.
- **`study_name` not shell-sanitized** in the `#SBATCH --job-name` directive (
  `_execution/_slurm.py:134`) — safe today only because `_STUDY_NAME="tune"` is
  hardcoded.

### Search space / op bounds *(Agent 3 — see full audit table below; the large majority

of ~50 bounds are PLAUSIBLE despite the "24 unverified" commit)*

- **Canny `low`/`high` threshold windows overlap** (
  `detect/_canny_detector.py:106-107`): `low TuneSpec(0.05,0.2)` vs
  `high TuneSpec(0.1,0.4)` — a trial can sample `low > high`, wasting budget on
  apply-time rejection. → encode as `low` + `delta`, or add a `model_validator`.
- **Enum-typed categoricals lose their type on Space round-trip** (
  `tune/_search_space/_domains.py:24-39`, `_infer.py` enum branch):
  `model_dump(mode="json")` → `model_validate` returns strings, not enum members,
  breaking the round-trip guarantee for enum-backed fields. → store `m.value` (the
  `Literal`-backed `str`/`int`/`bool` categoricals are fine).
- **`LocalEdgeDenoise.sigma_color` drops its `None` (auto-estimate) mode** (
  `enhance/_local_edge_denoise.py:85`) — the Tier-1 numeric TuneSpec swallows the
  qualitatively different `None` default. → expose `None` as a category.
- **`FocusEdgePhase.k` window starts at the degenerate `0.0`** (
  `enhance/_focus_edge_phase.py:117`) — `k=0` disables noise thresholding. → raise low
  bound to ~0.5.
- **`conditional_on` parents aren't `op_class`-stamped** (
  `tune/_search_space/_infer.py:730-733`) — latent; harmless in v1 (`_tune_optional`
  always False) but breaks the moment presence-wrapping ships. → stamp `conditional_on`
  targets alongside the top-level knobs.

### GUI *(Agent 4)*

- **`_BASE_PIPELINES` is an unbounded process-wide dict** (
  `gui/tune/_curate_overlays.py:82`) — every other cache here is capped (LRU 64); this
  one leaks `ImagePipeline`s on a long-lived multi-user hub. → small LRU (8–16) or a
  documented bound.
- **`TUNE_RUN_ROOT_STORE` lives inside the Monitor subtree** but is read cross-view as
  `State` by Space/Launch (`gui/tune/_layout.py:180`, `_callbacks.py:551`) — works
  today (always rendered) but fragile if Monitor ever lazy-renders. → hoist to page
  root.

### Tests / CI *(Agent 6, corroborated by Agent 7)*

- **3–4 tune tests hard-fail without the `tune` extra** — missing `skipif(not _OPTUNA)`:
  `test_tune_cli.py::test_resolve_strategy_tpe_builds_optuna_config`,
  `::test_cli_slurm_flag_uses_slurm_executor`,
  `test_run_marker.py::test_run_marker_written_before_slurm_branch` (and
  `test_optuna_strategy.py::test_optuna_strategy_run_writes_trials_parquet` depending on
  pre-import state). The failure is a **runtime `ImportError`
  from `_optuna_support.py:144`** — collection succeeds (the lazy boundary holds), so
  Agent 6's "ModuleNotFoundError at collection" wording was inaccurate. → add the
  guards.
- **No CI-running test for the cold-DB race or concurrent writers** — covered only by a
  *mocked* single-process pre-create (`test_run_tuning_slurm.py:100`) and an autoskipped
  *sequential* Postgres test. Zero threads/processes anywhere in `tests/unit/tune`. →
  add a `ThreadPoolExecutor` concurrent-open smoke (SQLite-WAL acceptable as a smoke).
- **~8 of 15 server callbacks' wiring is unexercised in CI** — only pure helpers are
  unit-tested; the only end-to-end wiring test is the `ci_flaky`/deselected Playwright
  one. The live drive found no 500s, but a wrong-arity/store-write regression would slip
  CI. → add Flask-test-client POSTs to `/_dash-update-component` for 2–3 representative
  callbacks.
- **`test_tune_live_timeout.py`** is a wall-clock-deadline test (`_HANG_SECONDS=8.0`,
  `assert elapsed < 5.0`) in the deterministic integration lane — flake risk. → make it
  event-gated, or move to `tests/e2e/` + `ci_flaky`.

---

## Nits

- Reference-free CV uses population std `ddof=0` (
  `tune/_scoring/_reference_free_scorer.py:440`) — bias largest for 2–3-replicate
  groups; prefer `ddof=1`.
- `is_unattended_safe` can't distinguish ρ≥0.8 from ρ≥0.7 (single bool) —
  `_reference_free_scorer.py:214-227`.
- Excluded-field reason overloads `"non_numeric"` for the non-positive-default case (
  `tune/_search_space/_infer.py:210-214`) — add a distinct `ExcludeReason`.
- `_macro_average_region` builds `empty` from GT shape, not pred shape (
  `tune/_scoring/_supervised.py:291`) — add a `pred.shape == gt.shape` assert.
- `match_iou_greedy` docstring overstates uniqueness at exactly `tau=0.5` (
  `tune/_scoring/_matching.py:73`).
- Migration script `scripts/migrate_sweep_manifest.py` is correct + tested but *
  *undocumented for users** — add a "Migrating from `sweep`" note to
  `docs/source/how_to/pages/tuning.md`.
- `.pht-tune-cache` absent from `.gitignore` (consistent with existing `.phenotypic`;
  low risk).
- Stale `sweep` mentions in two **pre-existing, untouched** docs (`gui/GUI_SPEC_V1.md`,
  `docs/design_outlines/ui_full_plan_with_sweep.md`) — out of strict scope.
- Several GUI tests assert `str(app.layout)` substrings (brittle smoke) —
  `test_tune_mount.py:30`, `test_tune_curate.py:42-60`.

---

## Design / spec / ledger drift

- **Ledger vs reality:** FEATURES.md/WORKFLOWS.md mark the loaded Tune co-pilot views "✅
  shipping," but they are **unreachable through the `phenotypic-gui` hub** (`root=None`,
  Chunk C deferred — `gui/shell/_app.py:299`). Land Chunk C in this PR or downgrade the
  rows.
- **Intentional capability drops** from old `sweep` (all superseded, not accidental):
  `--save-intermediates` HDF5 dumps (no equivalent — debugging affordance), the HTML
  progress dashboard (→ GUI Monitor poll), the napari sweep viewer (→ Curate A/B;
  explicitly out of scope per `GUI_SPEC_V1.md`).

---

## TuneSpec bounds audit (Agent 3 — abridged; most bounds PLAUSIBLE)

| Op · param                                                                                                                                                         | Declared               | Default   | Verdict                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-----------|------------------------------------------|
| `CannyDetector.low_threshold` / `high_threshold`                                                                                                                   | [0.05,0.2] / [0.1,0.4] | 0.1 / 0.2 | **SUSPECT — overlap allows low>high**    |
| enum-typed categoricals (any op)                                                                                                                                   | —                      | —         | **SUSPECT — round-trip drops enum type** |
| `LocalEdgeDenoise.sigma_color`                                                                                                                                     | [0.02,0.5] log         | None      | **SUSPECT — None mode lost**             |
| `FocusEdgePhase.k`                                                                                                                                                 | [0.0,20.0]             | 2.0       | **SUSPECT — degenerate at 0.0**          |
| `FlattenIllumination.sigma`                                                                                                                                        | [40,300] log           | 200       | weak — default pinned near ceiling       |
| `FocusBlobLoG.min_radius` vs `max_radius`                                                                                                                          | [1,5] / [8,50]         | 3 / 12    | safe but narrow                          |
| `GaussianBlur.sigma`, `EnhanceBlockMatch.sigma_psd`, `MedianFilter.width` (odd via step=2), `WatershedDetector.compactness` (log), most `detect/*` geometry params | —                      | —         | PLAUSIBLE vs docstrings                  |

Full per-param table (≈50 rows) is in the agent transcript; only the four SUSPECT rows
above need action.

---

## Coverage map (Agent 6 — abridged)

| Source module                                                                                               | Assessment                                                                            |
|-------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `_study/_optuna_store.py`, `_study/_pareto.py`, `_study_store.py`                                           | **solid** (SQLite round-trips, knee geometry, run in CI)                              |
| `_scoring/_matching.py`, `_reference_free_scorer.py`, `_composite.py`, `_qc_scorer.py`, `_gt_loader.py`     | **solid** (real plates)                                                               |
| `_scoring/_supervised.py`                                                                                   | **solid-ish** — mask tier only self-match==1.0 (misses B1)                            |
| `_search_space/*`, `_engine.py`, `_evaluation/*`, `_screening*.py`, `_multi_objective.py`                   | **solid**                                                                             |
| `_tune_cli/_run.py` (753 LOC)                                                                               | **partial** — much covered only via end-to-end `run_tuning`                           |
| cold-DB race / concurrent writers / SLURM submit                                                            | **mocked-hollow / autoskipped** — not exercised in CI                                 |
| `gui/tune/_callbacks.py` (1216 LOC)                                                                         | **wiring mocked-hollow** — pure helpers solid, ~8 server callbacks' arity unexercised |
| `gui/tune/_space.py`, `_winner.py`, `_overlays.py`, `_curate_overlays.py`, `_study_read.py`, `_run_root.py` | **solid** (timeout test flaky)                                                        |

---

## Distributed (Postgres/SLURM) manual-test checklist

Run against a real cluster/DB (`PHENOTYPIC_TEST_PG_URL` set, `sbatch` on PATH,
`--extra tune` installed):

**1. Cold-start empty DB — no schema race / no double-create**

```bash
psql "$PHENOTYPIC_TEST_PG_URL" -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
uv run python -m phenotypic.tune run spec.json --images ./calibration \
  --strategy tpe --n-trials 20 --storage-url "$PHENOTYPIC_TEST_PG_URL" --slurm
psql "$PHENOTYPIC_TEST_PG_URL" -c "SELECT study_name FROM studies;"   # exactly one: "tune"
grep -i "uniqueviolation\|duplicate key" logs/slurm/tune_worker_*.log  # no matches
```

**2. Two concurrent workers on one study — no duplicate trials, bounded resume**

```bash
uv run python -m phenotypic.tune._tune_cli._worker --spec deliverables/tuning_spec.json \
  --images ./calibration --study-name tune --storage-url "$PHENOTYPIC_TEST_PG_URL" &
uv run python -m phenotypic.tune._tune_cli._worker --spec deliverables/tuning_spec.json \
  --images ./calibration --study-name tune --storage-url "$PHENOTYPIC_TEST_PG_URL" &
wait
# COMPLETE+PRUNED ≤ n_trials; no RUNNING; no duplicate trial_id
psql "$PHENOTYPIC_TEST_PG_URL" -c "SELECT state, COUNT(*) FROM trials t JOIN studies s ON t.study_id=s.study_id WHERE s.study_name='tune' GROUP BY state;"
```

**3. Kill-and-resume idempotency**  *(currently expected to FAIL — see Should-fix: stale
RUNNING)*

```bash
uv run python -m phenotypic.tune._tune_cli._worker --spec deliverables/tuning_spec.json \
  --images ./calibration --study-name tune --storage-url "$PHENOTYPIC_TEST_PG_URL" &
PID=$!; sleep 5; kill -9 $PID
psql "$PHENOTYPIC_TEST_PG_URL" -c "SELECT COUNT(*) FROM trials t JOIN studies s ON t.study_id=s.study_id WHERE s.study_name='tune' AND state='RUNNING';"  # DESIRED 0 after relaunch
# relaunch a fresh worker, then re-check RUNNING==0 and total==n_trials
```

**4. Atomic markers under interruption**  *(currently expected to print CORRUPT — see
B3)*

```bash
uv run python -m phenotypic.tune run spec.json --images ./calibration --strategy tpe \
  --n-trials 4 --storage-url "$PHENOTYPIC_TEST_PG_URL" &
PID=$!; sleep 30; kill -9 $PID   # kill during finalize
python3 -c "import json; json.load(open('deliverables/best_pipeline.json'))" && echo VALID || echo CORRUPT
python3 -c "import json; json.load(open('.pht-tune-cache/run.json'))" && echo VALID || echo CORRUPT
```

**5. Credential non-leakage**  *(currently expected to MATCH — see B2)*

```bash
uv run python -m phenotypic.tune run spec.json --images ./calibration --strategy tpe \
  --n-trials 4 --storage-url "postgresql+psycopg://user:secretpassword@localhost:5432/optuna_test" --slurm
grep -r "secretpassword" . 2>/dev/null   # DESIRED: no matches (run.json / tune_workers.sh / logs)
```

---

## Fix-first shortlist

1. **B2 credential redaction + B3 atomic writes** — small, contained, high-impact for
   the distributed story.
2. **B1 GT-modality decision** — corrupts the supervised objective; settle
   instance-vs-binary, fix the loader/matcher, add a partial-overlap test.
3. **B4 Curate overlay race + CSS leak**, and **decide hub binding (Chunk C) vs ledger
   downgrade**.
4. **Stale-`RUNNING`-trial reconciliation + a concurrent-open test** — the property the
   branch exists to guarantee.
5. **The 3–4 `skipif` test guards** — trivial; prevents a red CI lane.

---

## Appendix — review methodology

| Agent | Charter                                                           | Type                         |
|-------|-------------------------------------------------------------------|------------------------------|
| 1     | Tune engine & scoring correctness (vs specs)                      | code-reviewer                |
| 2     | Distributed execution, study store & concurrency                  | code-reviewer                |
| 3     | Search-space inference, targets & op-annotation bounds            | code-reviewer                |
| 4     | GUI Tune co-pilot (static)                                        | code-reviewer                |
| 5     | Legacy `sweep` removal, migration & packaging                     | general-purpose              |
| 6     | Test-suite quality & coverage                                     | implementation-test-reviewer |
| 7     | Verification & live GUI smoke (ran mypy/ruff/pytest + Playwright) | general-purpose              |

Static GUI/test findings were reconciled against Agent 7's runtime evidence; all
blocking findings were re-read at the cited `file:line` by the orchestrator before
inclusion. Severities reflect orchestrator judgment after verification, not raw agent
labels.
