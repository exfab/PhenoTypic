# RESUME — Tune Engine Build (param-sweep redesign)

> Handoff for continuing the multi-phase "tune engine" build in a **fresh session**.
> Last updated 2026-06-05 by the orchestrator. Read **§Critical rules** first.

---

## TL;DR — where we are

- **Branch:** `redesign/param-sweep`, worktree `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/redesign/param-sweep`.
- **HEAD:** `1bd0b2ae` — Phase 4 + 4.5 + backend simplify (#16) DONE; **Structured KnobTarget API (#18) DONE** — typed `phenotypic.tune.targets` (`Param`/`Presence`/`Nested` union + `parse_key` + `with_op_class`), `Knob.target` (string `key=` still coerced; `.key` bridge keeps `build_pipeline`/golden lock untouched), `TuningSpec` cross-validator (op-range/`op_class`/field with did-you-mean), `pipeline_targets`/`TunableParam` discovery, posture-C `op_class` stamping; spec+plan in `docs/superpowers/{specs,plans}/param-sweep-redesign/`; 690 passed, mypy clean 51 files; implementer self-review's 3 findings fixed (parse_key depth/empty-class strictness; conditional_on target invariant). **NEXT: #19 Postgres/SLURM interface discussion → 🛑 GUI brainstorm (#17) → Phase 5.** Simplify carry-overs for #11: `_run.py` finalize split + `_matching._greedy_pair`. **Unpushed** (push from your terminal; agent env has no SSH key).
- **Goal:** replace `src/phenotypic/sweep/` (deleted) with `src/phenotypic/tune/` — full feature, Phase 0→6 + annotations, ending in ONE PR to `main`.

### Done + integrated (green on the branch)
- **Phase 0** (prereqs) · **Phase 1a–1d** (MVP: search space / strategies / scoring+eval / engine+CLI; `python -m phenotypic.tune` runs grid/random; `sweep` cut over)
- **Phase 2** (full Optuna backend: `OptunaConfig`/`OptunaStrategy` lazy, ASHA pruning, `OptunaStudyStore` SQLite+Postgres, `read_pg_connection_info`, fANOVA-vs-RF dispatch, two-round screening freeze, `SlurmExecutor`+worker, `run`/`auto-space` CLI flags)
- **Phase 3** (`TuneSpec` marker + `infer_search_space` Tier-1/2, nested-op overlay grammar, `--auto-space` CLI, `ReferenceFreeScorer` + meta-validation gate)
- **Annotations** (detect/+enhance/ `TuneSpec` hints + `Field` bounds, 71.1% coverage hard-gate; bounds fact-checked, 25 unverified flagged `# TODO`)
- **Phase 4** (ALL chunks A+B+C integrated): multi-objective `objectives` **sidecar** widening (back-compat-locked); `SupervisedScorer` (Dice/IoU + count MAE, `_metrics.py`/`_matching.py`/`_gt_loader.py` `GroundTruthMasks`); `CompositeScorer` (nests `list[Scorer]`, cycle detection); `pareto_front`/knee + `deliverables/pareto/` + NSGA-II wiring + Phase-4 integration test. **Both review passes applied** (commit `0334da67`): fixed the critical multi-objective-abstention crash (finalize now emits all axes, abstainers floored to 0.0); centralized `pareto_importance_path` in `_io_constants.py`; sourced pareto axis order from the scorer; softened the ≥3-objective knee docstring.

### Remaining (in order)
1. **Phase 4.5 — Robust-eval prereqs** (NEW, user-approved 2026-06-05; task #15) — **IN PROGRESS.** Full TDD plan in `PHASE4_5_EXPANDED_PLAN.md`. **Part 1 DONE + integrated @ `743d0958`** (Chunks A+B+C: `gap`/`suspicious` fields on `Trial`/`EvaluationResult` back-compat-locked; `_split.py` 3-tier seeded resumable split + `splits/split.json`; `_held_out.py` `HeldOutConfig`+`infer_group_key`; per-trial `gap`=relative dispersion + score-vs-Count-floor `suspicious`). **Part 2 IN FLIGHT** (agent `ac11b3e07eae14296`, Chunks D+E: `_generalization.py` held-out pass + `deliverables/generalization.json`, CLI-side orchestration in `_run.py` — engine stays pure, search runs on calibration-only — `--held-out-fraction`/`--cv-group` flags, qc §7 gaming regression, DEFERRED-WORK). **Resolved decisions:** per-trial gap = calibration dispersion + winner-only held-out (Option A, literature-confirmed sklearn `std_test_score`/Cawley-Talbot); `Trial.gap` immutable (winner gap → generalization.json only); gap gate = relative-drop AND absolute-drop thresholds; suspicious = score≥0.7 & Count≤0.3. **After part 2:** code-review + annotation-adherence + a **fact-checker** over the 8 numeric defaults (min_stability_n=4, suspicious 0.7/0.3, held_out_fraction=0.2, min_heldout_plates=6, gap_margins 0.15/0.05 — all `# TODO: review`). **Deferred** (DEFERRED-WORK): group-aware-CV/StratifiedGroupKFold + stratified-rungs + incremental-frame-cache; §8 CV-estimate substituted by calibration-stability (`cv_deferred`).
2. **Backend simplify (tune/)** (NEW, user-requested 2026-06-05; task #16) — a `code-simplifier` (Opus) over the **FULL Python API / full diff added so far** = `git diff $(git merge-base main HEAD)..HEAD` restricted to the backend (all of `src/phenotypic/tune/**` Phases 0–4.5 + the tune additions to `tools_/_io_constants.py` + `_execution/**`), NOT just the recent changes — BEFORE Phase 5 adds the GUI. Quality only; **run the full test suite (`-n 8`) + mypy + ruff + doctests AFTER every simplify edit; revert any edit that breaks a gate**. **Carry-over to consolidate:** the duplicated `ScorerField` in `_spec.py` + `_scoring/_composite.py` → a shared low-level field module. Runs AFTER the Phase 4.5 review pass (operates on reviewed code). Blocks Phase 5.
3. **🛑 GUI-interface brainstorm with the user** (NEW, user-requested 2026-06-05; task #17) — STOP before Phase 5; brainstorm the `/tune/` co-pilot UX **with the user** (superpowers:brainstorming) before any implementation. Refines layout / monitor+curate+space-edit views / interaction patterns on top of the technical plan. Blocks Phase 5.
4. **Phase 5** — `/tune/` Dash co-pilot (6a monitor → 6b curate → 6c space-edit; GUI ledgers + screenshots). **Expanded TDD plan already written → `PHASE5_EXPANDED_PLAN.md`** (worktree root). Blocked by #15 + #16 + #17. **Two decisions baked in:** Phase 5 CONSUMES the real gap/suspicious/splits data from 4.5 (no `gap_available()` stub); the human winner OVERWRITES `best_pipeline.json` + a `source="human"` provenance stamp.
4. **Final simplify** (#11) — `code-simplifier` over `gui/tune/` + the GUI↔backend seams + a light whole-surface re-pass (the backend was already simplified at #16) → full regression.
5. **Phase 6** — documentation (incl. the HPCC/Postgres distributed-tuning guide)
6. **Open the PR** → `main`

### Current gate baseline (verify after fresh-session env sync)
`uv run pytest tests/unit/tune tests/unit/tools_/test_io_constants.py -n 8 -q` → **661 passed, 2 skipped** (Phase 4.5 complete @ `0373f515`); lazy-import + grid-golden + byte-compat locks green; `mypy src/phenotypic/tune` clean (47 files); `ruff` clean.

---

## ⚠️ Critical operational rules (a fresh session WILL need these)

1. **Re-enable `bypassPermissions` mode.** A session restart resets the permission mode; **background subagents then cannot write files** (no interactive-approval channel → Edit/Write/Bash-write denied). Symptom: an agent reports `WRITES STILL BLOCKED`. Toggle bypass mode before dispatching writing agents.
2. **pytest `-n 8`, NEVER `-n auto`.** `-n auto` reads the physical node's core count (64+), not your Slurm cpuset → oversubscribes → **OOM-kills the session** (this already crashed a session). Hard-cap at `-n 8`. Brief every subagent the same. Serialize only `@pytest.mark.postgres` (shared DB) + any fixed-port Playwright/e2e.
3. **Env sync:** `uv sync --group dev --extra gui --extra tune` — `gui` extra (PyQt6) is required or `pytest-qt` aborts collection; `tune` extra (optuna/sqlalchemy/psycopg) is required or mypy hits optuna missing-stub errors and the Optuna tests skip. The **lazy-import lock** must stay green even with optuna installed (`import phenotypic` must not import optuna).
4. **No push from the agent env** (SSH key not loaded). The **user pushes** `redesign/param-sweep` manually.
5. **Orphaned worktrees to clean:** `git worktree list` may show `agent-aa256e7682755664c` (the orphaned chunk-B agent — no commits; `git worktree remove --force` it). A `phase4-chunk-b-scorers` worktree may also exist (origin unclear; check before reuse).

---

## Orchestration model (how this build runs)

- Act as **orchestrator**: one subagent per phase/chunk in an **isolated git worktree** (`isolation: "worktree"`), **Opus** model for writing agents. Big phases are **chunked** (a single agent overruns one context — Phase 1d/2/4 proved this).
- **Per-task TDD** (write failing test → confirm fail → minimal impl → confirm pass → `ruff --fix` → commit), scoped `git add` (never `-A`), commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Agents do NOT push.
- **Integration (orchestrator only):** review the seam diffs → merge the agent's branch onto `redesign/param-sweep` (fast-forward if no divergence; 3-way + resolve the `tune/__init__.py` overlap otherwise) → re-run the chunk gate (`-n 8`) → `git worktree remove --force` + `git branch -D` the merged worktree.
- **Review cadence:** `feature-dev:code-reviewer` + an **annotation-adherence** reviewer per phase that adds models; **one** `code-simplifier` after Phase 5 (not per phase); the **per-phase OQ checkpoint** (if a genuine design fork surfaces, STOP and ask the user); **fact-check** numeric bounds with a `fact-checker` subagent (done for annotations).

---

## Key references

- **Orchestration plan (the runbook):** `/rhome/anguy344/.claude/plans/please-write-the-plan-ethereal-orbit.md`
  — **§0a + §0b = the binding resolved design decisions** (read these before dispatching any chunk); §5 = phase-by-phase; §6.4–6.6 = per-phase-OQ / bounds-fact-check / `-n 8` rules; §7 = gates.
- **Deferred work:** `docs/superpowers/plans/param-sweep-redesign/DEFERRED-WORK.md`.
- **Outlines + specs:** `docs/superpowers/plans/param-sweep-redesign/phase-*.md` + `workstream-operation-annotations.md`; `docs/superpowers/specs/param-sweep-redesign/*.md`. (The full per-chunk TDD task lists were expanded in-session; the §0a/§0b decisions are what bind them.)
- **Memory (auto-loaded):** `~/.claude/projects/-bigdata-exfab-anguy344-PhenoTypic/memory/` → `param-sweep-tune-engine-build.md`, `execution-working-preferences.md`, `checkpoint-before-parallel-fanout.md`, `imageproc-tuning-bounds-references.md`.

---

## Next steps (dispatch-ready specs)

### ▶ Phase 4 chunk B — `SupervisedScorer` + `CompositeScorer`  (RE-DISPATCH; isolated worktree, Opus, `-n 8`)
Builds on chunk A (`_metrics`/`_matching`/`_gt_loader`, the `finalize -> float|dict` widen, the `objectives` sidecar, `ScorerField = polymorphic_field(base=Scorer)`).
- **4.4 `SupervisedScorer`** (`tune/_scoring/_supervised.py`): fields `gt: GroundTruthMasks`, `region_metric: Literal["dice","iou"]="dice"`, `match_strategy: Literal["grid_cell","iou_greedy"]="grid_cell"`, `iou_tau=0.5`, `count_check: ExpectedVsDetectedCount | None`. `score_image` → `{"Region": …}` (mask tier, matched + macro-averaged) + `{"CountMAE": …}` (count tier, **reuses `ExpectedVsDetectedCount`** — do not re-implement counting). `availability()` reads `gt.modality()` → mask/count/none tier. `field_validator`: exactly **one** region metric. Path-configured round-trip. **GT validation DEFERRED** — tests = construct/round-trip/term-shape/availability-tier ONLY (no numeric-vs-real-GT); `# TODO(DEFERRED-WORK §1)`.
- **4.5 `CompositeScorer`** (`tune/_scoring/_composite.py`): `scorers: list[ScorerField]`, `weights: dict|None`, `multi_objective: bool=False`. `score_image` merges child terms with a per-child **prefix**. `finalize` → scalar blend by default, **`dict[str,float]`** when `multi_objective=True` (the sidecar path). `model_validator` rejects cycles/self-nesting.
- **Seams:** export `GroundTruthMasks`, `SupervisedScorer`, `CompositeScorer` from `tune/__init__.py` + `tune/_scoring/__init__.py`.

### ▶ Phase 4 chunk C — Pareto + NSGA-II + integration
- **4.6** `StudyStore.pareto_front() -> list[Trial]` (non-dominated by `objectives`) + `knee_point(front)` (max-distance-to-chord); add to the `_study/_protocol.py` Protocol + `JournalStudyStore` + `OptunaStudyStore`.
- **4.7** `deliverables/pareto/` outputs: `pareto_front.parquet` + per-objective `best_<objective>.json` + the knee as top-level `best_pipeline.json`; add `pareto_dir`/`pareto_front_parquet_path`/`pareto_best_pipeline_path` to `tools_/_io_constants.py`; branch `run_tuning` on `result.objectives is not None`. **Single-objective writes NO `pareto/`** (back-compat lock).
- **4.8** NSGA-II auto-selected when the scorer is multi-objective (`OptunaConfig` already supports `sampler="nsga2"`); grid/random + multi-objective scorer → **reject at validation** (clear error). Pass objective names/directions (all maximize) into the Optuna study.
- **4.9** Phase-4 integration test (`CompositeScorer(multi_objective=True)` over synthetic GT → `run_tuning` grid → asserts `deliverables/pareto/` + knee `best_pipeline.json`; single-objective sibling unaffected). Post: code-review + annotation-adherence; mark Phase 4 done.

### ▶ Phase 5 — `/tune/` Dash co-pilot  (expand outline first if needed; one agent, 6a→6b→6c)
Per §0a/§0b: overlays via background compute + LRU from `splits/calibration.json`; shortlist = top-5 + Pareto + gap-flagged; **GUI launch = copy-paste command card** (no `LocalRunner` spawn); 6c edits flat+presence only (nested read-only); WAL read-only monitor; gap-flag relative >0.15; one agent runs 6a→6b→6c. **CI-gated:** `gui/FEATURES.md` rows (real `Test ref`), `gui/WORKFLOWS.md` round-trip (`_capture_tune_copilot` defined+dispatched in `scripts/capture_gui_tutorial_screenshots.py` + a tutorial page), **regenerate + commit ALL PNGs**. Registration sites (per the Phase-5 expansion): `_config.py` (`MOUNT_TUNE`), `shell/_ids.py` (`SHELL_TAB_TUNE`+stores), `shell/_layout.py` (3 tab dicts, slot Home→Pipelines→Run→**Tune**→Viewer→Analysis), `shell/_app.py` `compose_hub` mount, new `gui/tune/`. Add `study_db_path` already exists. Needs Phase 2 (`study.db`) + 3 (`_param_forms`/`InferredSearchSpace`) + 4* (Pareto, feature-flag off when single-objective).

### ▶ Final simplify · Phase 6 docs · PR
- One `code-simplifier` (Opus) over `tune/`(+`gui/tune/`) → apply → **full regression** (`uv run pytest -q` with `-n 8`-safe selection; the `tests/smoke` ColorDenoise bm3d flake was already fixed).
- **Phase 6 docs (user-directed 2026-06-04):** run a **writer + fact-checker + editor/reviewer agent team** (`TeamCreate`, Opus) — apply the combo to the **Python** subsection and the **GUI** subsection each (writer drafts → fact-checker verifies every claim against the real code/CLI/specs → editor polishes). Add a **dedicated "Tuning" section under how-to** (its own section) with two subsubsections: **`### Python interface`** (the `python -m phenotypic.tune run …` CLI + `TuningSpec` Python API + auto-space/screen/deliverables/pareto/resume) and **`### GUI interface`** (the `/tune/` Dash co-pilot; copy-paste-command launch). Cover **the four scoring objectives** (the user's "four strategies" = the four `Scorer` types, NOT the search strategies): **`QCScorer`** (no-GT statistical count check) · **`ReferenceFreeScorer`** (no-GT segmentation proxies) · **`SupervisedScorer`** (with-GT Dice/IoU + count MAE) · **`CompositeScorer`** (combine/multi-objective). Search strategies (grid/random/Optuna TPE/CMA-ES/GP/NSGA-II) are the optimizer — covered under the CLI/Python `--strategy`. **GUI screenshots: capture via Playwright (drive the hub) + an automated capture script** mirroring `scripts/capture_gui_tutorial_screenshots.py` (hermetic synthetic tune run → snap → commit PNGs). Plus `tune_distributed_hpcc.md` (why Postgres on NFS; `sbatch ~/util/postgres_server/pgserver.sh`; read `connection_info.txt`; wire `--storage-url postgresql+psycopg://…:54399/…`); `cli_reference` + autodoc; README tuning section; replaces `parameter_sweeps.md`. Gate: `make -C docs html` clean; screenshot script runs + PNGs committed; no `phenotypic.sweep` left in docs/README.
- PR: `redesign/param-sweep` → `main`, after the §7.5 end-to-end acceptance.

---

## Recovery checklist for the fresh session
1. `cd /bigdata/exfab/anguy344/PhenoTypic/.worktrees/redesign/param-sweep` (enter the worktree).
2. Toggle **bypassPermissions**; `uv sync --group dev --extra gui --extra tune`.
3. `git worktree list` → remove the orphaned `agent-aa256e76…` worktree; verify HEAD is `88e83518` (or later if you'd pushed/merged more).
4. Re-run the baseline gate (`pytest tests/unit/tune -n 8 -q` → expect 470/2). 
5. Re-dispatch **Phase 4 chunk B** (spec above) → integrate → chunk C → Phase 5 → simplify → docs → PR.
6. Delete this `RESUME.md` before opening the PR (it's an untracked handoff note, not part of the deliverable).
