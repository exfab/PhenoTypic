# Parameter Tuning Engine — Design Spec

- **Date:** 2026-06-01
- **Status:** Approved design, revised after literature self-review (pre-implementation)
- **Branch:** `redesign/param-sweep`
- **Supersedes / refactors:** `src/phenotypic/sweep/` (the parameter sweep CLI feature)
- **Author:** brainstormed with Claude (Opus 4.8)
- **Revisions:** 2026-06-01 — literature self-review pass: screening switched to
  fANOVA (captures interactions; reuses optimizer trials); reference-free scoring
  gated behind meta-validation; `Scorer` gains an optimization-direction contract;
  pruning fidelity defined as calibration-set size; k-fold / metadata-stratified
  calibration; grid-over-conditional regression test mandated; the "1.42× / ~100
  points" figure softened to illustrative.

---

## 1. Motivation

### Current state

`phenotypic.sweep` enumerates the **full Cartesian product** of every parameter
combination declared via `Sweep` / `Presence` / `Fixed`, serialises them into a
JSON manifest (`generate_sweep_manifest`), and the CLI
(`python -m phenotypic.sweep`) runs *every image × every pipeline combination*
(joblib locally or a SLURM array), saving per-image / per-pipeline CSV + HDF5.

**There is no objective function, no scoring, and no optimizer.** A human
visually picks the winner in the napari/Dash sweep viewer. It is exhaustive grid
search with manual selection.

### The gap

Grid search wastes the overwhelming majority of evaluations on (a) parameters
that don't influence the output and (b) regions of the space far from any good
optimum. For an expensive black-box (a full pipeline run per image), this is the
wrong tool. We want a **robust, sample-efficient tuning engine** that:

1. goes beyond grid search (random, Bayesian/TPE, evolutionary, screening-first),
2. scores candidates against a **pluggable objective** (supervised, reference-free,
   domain-QC, or human-surfaced),
3. selects parameters that **generalise across the plate set** rather than overfit,
4. is driven interchangeably by a **human (CLI + Dash)** and an **agent (MCP)**,
   sharing one session, and
5. can eventually be exposed as an **MCP server** for agentic parameter tuning.

---

## 2. Literature basis

The design follows established practice for tuning image-analysis pipelines.

- **Screen first, then optimize.** Teodoro et al. (2016) tune a tissue/nucleus
  segmentation pipeline as a **black box**, pruning non-influential parameters
  *before* optimization. (Their screening used Morris One-At-A-Time + Sobol; we
  adopt **functional ANOVA** instead — next bullet — because our space is
  categorical/conditional, where Morris is a poor fit.) On their tissue-WSI
  workflows, searching ~100 points of a billions-to-trillions-point space
  improved Dice/Jaccard by up to 1.42× over defaults; we cite this as
  **illustrative motivation** for abandoning full grid, *not* a transferable
  promise for colony detection.
- **Measure which knobs matter with fANOVA.** Functional ANOVA (Hutter et al.,
  2014) fits a random-forest surrogate on trials *already gathered by the
  optimizer* and shows "most performance variation is attributable to just a few
  hyperparameters" — while also quantifying **interactions**, which
  one-at-a-time screening cannot. Interactions can dominate (Shehata et al.,
  2025: >90% of variation from interactions in RL tuning), so main-effect-only
  screening is risky. PED-ANOVA (Watanabe et al., 2023) extends fANOVA to
  top-performance subspaces. Optuna ships fANOVA importance natively.
- **Sample-efficient optimizers for expensive evaluations.** Teodoro compared
  Nelder–Mead, Parallel Rank Order, Genetic Algorithm, and **Bayesian
  optimization** (Snoek et al. / Spearmint), favouring model-based search "for
  objective functions whose evaluations are costly."
- **Multi-fidelity pruning needs a fidelity axis.** Hyperband/ASHA early-stopping
  presupposes cheap→expensive approximations; the natural axis here is the
  **number of calibration images** evaluated (Kandasamy et al., 2017 use "less
  data N" as exactly this). But unreliable low-fidelity subsets can make
  multi-fidelity BO *worse* than plain BO (Mikkola et al., 2022), so pruning is
  opt-in over a stratified, representative subset.
- **Tuning is naturally multi-objective.** Taveira et al. (2018) maximise
  segmentation quality **while minimising execution time** via a-priori
  scalarization, because sampling a full Pareto front is too expensive;
  importance analysis composes with scalarization (Theodorakopoulos et al.,
  2024). Caveat: fixed-weight scalarization cannot reach non-convex regions of a
  Pareto front.
- **Scoring without ground truth is possible but a *proxy of unproven
  reliability*.** Zhang et al. (2008, 1100+ citations) survey unsupervised
  metrics ("enable self-tuning of algorithm parameters") and Chen et al. (2021)
  rank cell-seg methods reference-free. But no-reference metrics can correlate
  **poorly** with true downstream quality (Deo et al., 2025), and even
  *supervised* metrics can miss the visual optimum (Jozdani et al., 2020). The
  methods that *worked* validated the proxy against ground truth first (Galdran
  et al., 2018; Chen et al., 2021). → A reference-free objective must be
  **meta-validated** against a small ground-truth set before it is trusted to
  drive optimization (see §4).
- **Mature tooling.** Optuna provides TPE Bayesian optimization, a define-by-run
  search space, **fANOVA importance**, pruning (ASHA/Hyperband), native
  multi-objective, an **ask-and-tell API**, and study persistence — the exact
  shape an MCP/agent loop needs.

Full references in §13.

---

## 3. Locked-in decisions

| # | Decision | Choice |
|---|----------|--------|
| D1 | Objective function | **Pluggable** — supervised / reference-free / domain-QC / human-surfacing are interchangeable `Scorer` implementations; the optimizer never cares how a score was produced. **`QCScorer` (domain consistency) is the primary default** for colony arrays; **`ReferenceFreeScorer` is gated behind meta-validation** against a small ground-truth set (§4), since no-reference proxies can mislead. |
| D2 | Scope | **One shared tuning engine.** Grid search is demoted to one `SearchStrategy`. CLI and MCP both wrap the same ask-and-tell core. Reuse the existing manifest / joblib / SLURM execution. |
| D3 | Optimizer backend | **Optuna** is the default behind a thin `SearchStrategy` Protocol. We own random-search and a zero-dependency importance fallback directly (no lock-in). The seam stays open for an `AxStrategy` later. |
| D4 | Robust evaluation | **Calibration set + direction-normalized aggregate + stability penalty + held-out validation.** Each trial scores over a representative, **metadata-stratified** calibration subset; for small image counts use **k-fold / leave-one-plate-out** CV. The aggregate is `level(per-image) − λ·dispersion(per-image)` on a normalized higher-is-better scale (so flat optima beat sharp ones); the winner is validated on held-out images. |
| D5 | Co-pilot UI | **Dash only** (a new GUI-hub mount), not napari. |
| D6 | Drivers | The engine is usable interchangeably by **a human (CLI + Dash) and an agent (MCP)**, collaborating on **one shared study**. |
| D7 | Search space | An **automated `infer_search_space`** derives tunable domains from the pydantic operation fields; humans/agents review and edit before tuning. |
| D8 | Screening / importance | **functional ANOVA** over the optimizer's own trials (Optuna's `get_param_importances`), capturing main effects *and* interactions; low-importance params may be frozen for a focused second round. A zero-dependency correlation/variance importance covers the no-Optuna path. |

---

## 4. Architecture

### The reframing

From *enumerate → save everything → eyeball* to *optimize an objective under a
budget* via an ask-and-tell loop:

```
study = engine.create_study(search_space, objective, strategy, budget)
while not study.done():
    params = strategy.suggest()              # grid | random | tpe | cmaes | ...
    score  = evaluator.evaluate(params)      # run on calibration imgs → score → aggregate + stability
    strategy.register_result(params, score)  # optimizer learns
return study.best()                          # or Pareto front (multi-objective)
```

Grid search is the degenerate strategy whose `suggest()` walks the full Cartesian
product, so **today's behaviour is a preserved special case**, not discarded.

### Five components (each independently testable)

1. **`SearchSpace`** — generalises `Sweep` / `Presence` / `Fixed`. A tunable
   param is one of `Categorical([...])`, `IntRange(lo, hi, step?, log?)`,
   `FloatRange(lo, hi, log?)`, `Fixed(v)`, with **conditional nesting** (a param
   exists only when its parent op is present / set a certain way — define-by-run).
   *Back-compat:* `Sweep(GaussianBlur, sigma=(1.0, 2.0))` (a tuple) is reinterpreted
   as `Categorical`; `Presence` becomes a categorical over `{present, absent}`.

2. **`SearchStrategy` (Protocol)** — `suggest()`, `register_result(params, score)`,
   `is_exhausted()`. Implementations: `GridStrategy` + `RandomStrategy`
   (homegrown, zero deps, exact migration path), `OptunaStrategy`
   (TPE / CMA-ES / GP / NSGA-II + pruning + SQLite persistence), future `AxStrategy`.

3. **`Scorer` (Protocol)** — `score(image, result) -> float | dict[str, float]`,
   plus a declared **optimization direction** (`higher_is_better: bool`) and a
   **value range / normalizer** per metric, so heterogeneous metrics can be
   combined and a stability term applied on a common higher-is-better scale.
   Implementations:
   - `SupervisedScorer` — ground truth → count error / IoU / Dice / F1 / adjusted
     Rand. (Metric choice matters: F-measure / QR / SEI track the visual optimum
     best for combined over/under-segmentation — Jozdani 2020.)
   - `ReferenceFreeScorer` — intra-colony homogeneity vs. background contrast,
     boundary gradient, shape regularity (Zhang 2008 / Chen 2021 style).
     **Requires meta-validation:** before it may *drive* optimization, the engine
     correlates it against a small ground-truth set (≥3–5 annotated plates),
     records the correlation in the report, and **warns/abstains if the
     correlation is weak** (Deo 2025). The primary failure mode is a proxy the
     optimizer can exploit but that doesn't track real quality.
   - `QCScorer` — reuse the existing `analysis/` QC checks: expected-vs-detected
     grid count, ICC replicate reliability, MAD/Tukey outlier rates, edge effects.
     The most trustworthy objective for colony arrays and the **Phase-1 default**.
   - `CompositeScorer` — weighted scalarization (one number) *or* return a dict
     (true multi-objective Pareto).

4. **`Evaluator`** — the robustness layer (D4). Builds the pipeline from params,
   runs it across the **metadata-stratified calibration subset** via the existing
   joblib/SLURM machinery, applies the `Scorer` per image, **normalizes each
   metric to a common higher-is-better scale**, then aggregates
   `score = level(per-image) − λ·dispersion(per-image)` (default `level` = median,
   `dispersion` = IQR). For small image counts it runs **k-fold /
   leave-one-plate-out** CV instead of a single split. With pruning enabled it
   evaluates calibration images **progressively** and reports intermediate scores
   via `trial.report()`, so a hopeless candidate is early-stopped after a few
   images — **fidelity = number of calibration images** (§2), over a stratified
   subset to avoid unreliable low-fidelity. Returns the aggregate plus the
   per-image breakdown, and validates the winner on a **held-out subset**.

5. **`TuningEngine`** — orchestrates the loop; owns the budget (n-trials /
   wall-clock / convergence), persistence, and reporting. **This single object is
   what the CLI and the MCP wrap.**

**Plus a `ScreeningPhase` / importance pass** (on by default above ~6 params):
**functional ANOVA** over the trials the optimizer has already run (Optuna's
`get_param_importances`) ranks parameters by influence **and interaction**, after
which low-importance params may be **frozen at their defaults for a focused
second round**. fANOVA is preferred over Morris One-At-A-Time because our space is
categorical/conditional and because interactions can dominate (§2); it also
reuses optimizer trials rather than spending a separate screening budget. A cheap
one-at-a-time pre-screen remains available as an option for purely continuous
knobs, and when Optuna is absent a zero-dependency correlation/variance importance
gives a coarse fallback. Independently valuable as a plain-English *"which knobs
matter for your plates"* answer.

### Data flow

```
SearchSpace → TuningEngine → [Screening: rank + freeze non-influential] → ask/tell loop:
     Strategy.suggest() → params → Evaluator(calibration imgs, joblib/SLURM)
         → ImagePipeline.measure() per image → Scorer per image → aggregate + stability
         → Strategy.register_result() ──► study (SQLite, resumable)
   → best params / Pareto front → validate on held-out → report + ready-to-run pipeline.json
```

---

## 5. Automated search-space derivation (`infer_search_space`)

Because operations are pydantic v2 models with typed, constrained, self-describing
fields (every op exposes `model_json_schema()` with descriptions derived from
docstrings), we mine that contract instead of hand-writing ranges. Two tiers:

**Tier 1 — per-field tuning metadata (precise, opt-in).** Operations declare
domains via `Annotated[float, TuneSpec(0.5, 3.0, log=True)]` field metadata — the
exact pattern already used for `ColumnRef` / `NdArrayField` / `OperationField`.
The GUI registry already walks `model_fields` metadata, so the machinery exists.
`TuneSpec(tunable=False)` excludes a field (seeds, paths).

**Tier 2 — type/constraint heuristics (automatic fallback).** When no `TuneSpec`:
- `bool` → `Categorical([True, False])`
- `Enum` / `Literal[...]` → `Categorical(members)` (clean given the project's
  closed-value-set convention — `FootprintShape`, `DetectMode`, gamma encodings)
- bounded `int` / `float` (pydantic `Field(ge=, le=, …)`) → `IntRange` / `FloatRange`
  (auto `log=True` when `lo > 0` and `hi/lo ≳ 100`)
- unbounded numeric with default `d` → heuristic window `[d/4, d·4]`, **flagged
  "inferred — review me"**
- `NdArrayField`, paths, names, `OperationField` → excluded (not scalar-tunable)
- optional enhancers/refiners → auto-wrapped in `Presence`, only via an opt-in
  class flag, never guessed

Output is a **proposed `SearchSpace`** that the human/agent inspects, edits, and
approves — deliberately generous, because the **screening pass prunes** the
over-inclusion. Surfaces: CLI `--auto-space` (from a `pipeline.json`) and MCP
`tune_infer_space(pipeline_json)` (returns a typed, bounded space *with each
knob's docstring description* as context).

---

## 6. Surfaces — one engine, three drivers, one shared study

The shared `TuningEngine` + persistent SQLite study means **agent and human drive
the same ask-and-tell loop against the same study** — they collaborate rather than
fork. An agent can run autonomous trials overnight while a human reviews and
curates the surfaced candidates the next morning, in the same session.

| Driver | Surface | Role |
|--------|---------|------|
| Human (batch) | CLI `python -m phenotypic.tune` | headless / SLURM runs |
| Agent | MCP `tune_*` tools | autonomous trials, or steering (re-weight objectives, widen/narrow space, stop on convergence), and *explaining* the winner |
| Human (interactive) | **Dash** co-pilot view | review suggested candidates with overlays + per-image scores, rank/accept (writes back to the study) |

### CLI

`python -m phenotypic.tune TUNING_SPEC.json INPUT_DIR [OPTIONS]`, reusing the
existing execution/SLURM/dashboard machinery. New options:
`--strategy {grid,random,tpe,cmaes}` · `--n-trials N` · `--objective {supervised,reference-free,qc,composite}` (or config path) ·
`--calibration-n/-frac` · `--stability-weight λ` · `--screen/--no-screen` ·
`--multi-objective` · `--auto-space` · plus all current
`--n-jobs/--slurm/--image-type/...` flags. **`--strategy grid` (no budget)
reproduces today's exhaustive sweep byte-compatibly** — the migration safety valve.

The input flips from a pre-enumerated manifest to a **tuning spec**
(`SearchSpace` + `Scorer` + `Strategy` + budget), authored Python-first and
serialised to `tuning_spec.json`. The manifest becomes an **output** artifact
(the record of trials run), not an input.

### MCP (later phase, same engine)

Stateful ask-and-tell tools; the Optuna study *is* the session:
`tune_create_study(space, objective, strategy, budget)` →
`tune_suggest(study_id)` → `{trial_id, params}` →
`tune_report(study_id, trial_id, score|scores)` ·
`tune_run_trial(...)` (server-side fuse of suggest→evaluate→report) ·
`tune_best` / `tune_pareto` · `tune_param_importance` · `tune_infer_space` ·
`tune_status`.

### Dash co-pilot (D5)

A new Dash view in the GUI hub (a `/tune/` mount, sibling to builder/results/run),
**not** the napari viewer. Shows suggested candidates with overlays + per-image
scores, lets the user rank/accept (the choice writes back to the study as the
objective signal), and visualises objective-vs-trial, the Pareto front, and the
screening importance bars. The napari `gui/sweep/` viewer stays for power-user
exploration of raw grid outputs but is **not** a tuning driver. *(A new Dash view
trips the `FEATURES.md` / `WORKFLOWS.md` CI gates + tutorial-screenshot round-trip
— scoped as its own GUI phase.)*

### Concurrency note

A shared SQLite study with concurrent agent + human writers needs WAL mode (fine
at this scale). Heavy parallel SLURM trials *plus* live human writes is the point
to graduate to a proper RDB backend — the Protocol seam makes that a config
change, not a rewrite.

---

## 7. Multi-objective

Two paths from the same `Scorer`:

- **Scalarized (default)** — `CompositeScorer` returns one weighted number
  (a-priori weights; Taveira 2018). Natural colony conflicts: *count accuracy vs.
  shape fidelity*, or *quality vs. runtime*.
- **True Pareto (`--multi-objective`)** — `Scorer` returns a dict; `OptunaStrategy`
  runs NSGA-II / multi-objective TPE → a Pareto front; the report draws the
  trade-off curve and the user/agent picks a knee point.

**Stability is itself an objective.** The robustness penalty in §4 collapses
*level* and *dispersion* across images into one number; when that trade-off
matters, expose it on the Pareto view (level vs. dispersion) instead of
hard-scalarizing. And because fixed-weight scalarization cannot reach non-convex
regions of a Pareto front, prefer `--multi-objective` when the conflict between
objectives is important rather than relying on weights alone.

---

## 8. Output layout

Under a `deliverables/`-style directory:

- `study.db` — SQLite, resumable
- `tuning_spec.json` — search space + objective + strategy (reproducibility)
- `trials.parquet` — per-trial params + scores + per-image breakdown
- **`best_pipeline.json`** — ready-to-run `ImagePipeline` (drops straight into
  `python -m phenotypic`)
- `pareto/` — Pareto-front pipelines (multi-objective)
- `param_importance.json` — screening / importance
- `tuning_report.html` — objective-vs-trial, importance bars, calibration-vs-held-out,
  per-image stability (extends the sweep progress dashboard)

**Disk policy:** scores for *all* trials are always kept; full per-image
measurements/HDF5 are kept only for **best + Pareto + flagged** trials by default
(`--keep-all-trial-outputs` to override). A naive optimizer saving every trial's
full output would blow up disk worse than today's grid.

---

## 9. Migration / back-compat

This is a CI-gated, GUI-coupled feature; back-compat is a hard requirement.

- **`phenotypic.sweep` public API is preserved unchanged**: `Sweep`, `Presence`,
  `Fixed`, `generate_sweep_manifest`, `load_sweep_manifest`, and the loader
  helpers. Existing user configs, the napari viewer, and the GUI keep working.
- `Sweep` / `Presence` / `Fixed` **gain** the new domain types as accepted values
  (`sigma=FloatRange(...)` alongside `sigma=(1.0, 2.0)`); a bare tuple still means
  `Categorical`. A `list[Sweep]` *is* constructible into a `SearchSpace`.
- Shared internals (`params → ImagePipeline`, the joblib/SLURM batch runner) get
  **extracted** so both `sweep` (grid) and `tune` (optimize) import them rather
  than duplicate. `sweep` becomes the grid-only facade over the engine.
- `--strategy grid` reproduces today's exhaustive output **byte-compatibly** with
  the per-image layout the GUI reads — the regression lock. This claim is
  load-bearing, so it gets an explicit test: **grid enumeration over a
  *conditional* space (`Presence` + nested params) must equal the current
  `generate_sweep_manifest` Cartesian product**, asserted against a saved fixture
  manifest.
- Old manifest JSON stays loadable; `tuning_spec.json` is a new artifact.

---

## 10. Dependency policy

`optuna` goes in an **optional extra** (`uv sync --extras tune`), not core,
matching the existing `gui` / `docs` extras and the project's care with heavy/
platform-specific deps. Grid + random + screening + robust evaluation are
**fully dependency-free** (homegrown), so Phase 1 ships with zero new deps.
Requesting `--strategy tpe` without the extra → a clear, actionable error.
`optuna` is pure-Python / cross-platform (no Windows exclusion). Promoting it to
core later (so TPE is the out-of-box default) is a one-line change.
Screening/importance uses Optuna's built-in **fANOVA** when the extra is present;
a zero-dependency correlation/variance importance is the fallback. No Morris /
Sobol or `SALib` dependency is required (an optional one-at-a-time continuous
pre-screen is also dependency-free).

---

## 11. File layout

```
src/phenotypic/tune/
  __init__.py              # public: tune(), SearchSpace, TuneSpec, infer_search_space, scorers
  _search_space/           # _space.py (domains), _infer.py (introspection), _tune_spec.py (marker)
  _strategies/             # _protocol.py, _grid.py, _random.py, _optuna.py (lazy import)
  _scorers/                # _protocol.py, _supervised.py, _reference_free.py, _qc.py, _composite.py
  _evaluator.py            # calibration set, aggregate + stability, held-out validation
  _screening.py            # fANOVA importance (Optuna) + zero-dep correlation fallback; optional 1-at-a-time pre-screen
  _engine.py               # TuningEngine: ask/tell loop, budget, convergence, reporting
  _study_store.py          # persistence (Optuna SQLite | homegrown journal fallback)
  _report.py               # tuning_report.html (extends sweep dashboard)
  _tune_cli/               # mirrors _sweep_cli/, SHARES the extracted execution/SLURM helpers
  __main__.py
src/phenotypic/sweep/      # preserved; grid facade sharing tune internals
src/phenotypic/gui/tune/   # Dash co-pilot view             (later GUI phase)
src/phenotypic/mcp/        # MCP server wrapping TuningEngine (later phase)
```

`_tune_cli` shares the extracted joblib/SLURM batch runner with `_sweep_cli`
rather than duplicating it.

---

## 12. Error handling, testing, phasing

### Error handling

- **A trial that raises doesn't kill the study** — marked failed (Optuna
  `TrialState.FAIL`) / worst-cased, logged with the same per-trial failure-log
  detail as today's sweep, loop continues.
- **Degenerate scores** (NaN — e.g. zero objects detected) → worst-case + flagged,
  never allowed to poison the surrogate model.
- **Stop conditions**: n-trials / wall-clock / early-stop (no improvement in K
  trials); always emit best-so-far.
- **Resume**: SQLite study + the existing event-log / drip-feed SLURM machinery →
  re-invoking continues from the last trial; resume reproduces an identical study
  under a fixed seed.
- **Overfitting guard**: if held-out validation ≪ calibration score, the report
  flags it loudly. Inferred unbounded ranges are flagged and caught by screening +
  validation.

### Testing

- **Unit**: domain sampling (incl. conditional/`Presence` nesting);
  `infer_search_space` on synthetic pydantic ops; each `Scorer` on
  `load_synth_yeast_plate()` incl. the `higher_is_better` direction contract;
  `ReferenceFreeScorer` **meta-validation correlation** against a GT fixture;
  Evaluator aggregate + stability + k-fold math with metric normalization;
  **`GridStrategy` enumeration over a conditional space == current
  `generate_sweep_manifest`** (regression lock, fixture-based); seeded
  `RandomStrategy` determinism.
- **Integration**: end-to-end `tune()` on the synth plate with a tiny budget,
  asserting improvement over defaults and an identical study on resume.
- The existing `sweep` test-suite must stay green.
- Fixed seeds throughout (project reproducibility requirement).
- GUI: `FEATURES.md` / `WORKFLOWS.md` + screenshot round-trip when the Dash view
  lands.

### Phasing

Each phase lands behind the review → simplifier → regression cadence.

1. **Engine core, zero deps** — `SearchSpace` (incl. conditional nesting),
   Protocol, `GridStrategy` (regression-lock) + `RandomStrategy`, `Evaluator`
   (direction-normalized robust eval + k-fold), `TuningEngine`, **`QCScorer`
   (primary)** + a zero-dep importance fallback, CLI. *Already beats grid*
   (random + importance + robust aggregation). Shippable alone.
2. **Optuna backend (extra)** — `OptunaStrategy` (TPE/CMA-ES), SQLite
   persistence/resume, **fANOVA importance**, ASHA pruning (fidelity =
   calibration images, opt-in).
3. **Auto-space + `ReferenceFreeScorer`** — `infer_search_space`, `TuneSpec`, and
   the reference-free objective **with its mandatory meta-validation gate**.
4. **Supervised scorers + multi-objective / Pareto reporting.**
5. **MCP server.**
6. **Dash co-pilot view** (FEATURES/WORKFLOWS gates + screenshots).

---

## 13. References

- Teodoro, G., Kurç, T., Taveira, L. F. R., et al. (2016). Algorithm sensitivity
  analysis and parameter tuning for tissue image segmentation pipelines.
  *Bioinformatics*, 33(7), 1064–1072. https://doi.org/10.1093/bioinformatics/btw749
- Taveira, L. F. R., Kurç, T., de Melo, A. C. M. A., et al. (2018). Multi-objective
  parameter auto-tuning for tissue image segmentation workflows. *Journal of
  Digital Imaging*, 32(3), 521–533. https://doi.org/10.1007/s10278-018-0138-z
- Zhang, H., Fritts, J. E., & Goldman, S. A. (2008). Image segmentation evaluation:
  A survey of unsupervised methods. *Computer Vision and Image Understanding*,
  110(2), 260–280. https://doi.org/10.1016/j.cviu.2007.08.003
- Chen, H., & Murphy, R. F. (2021). Evaluation of cell segmentation methods without
  reference segmentations. *Molecular Biology of the Cell*, 32(15).
  https://doi.org/10.1091/mbc.E22-08-0364
- Akiba, T., Sano, S., Yanase, T., et al. (2019). Optuna: A next-generation
  hyperparameter optimization framework. *KDD '19*.
  https://doi.org/10.1145/3292500.3330701
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter
  optimization. *Journal of Machine Learning Research*, 13, 281–305.
- Hutter, F., Hoos, H., & Leyton-Brown, K. (2014). An efficient approach for
  assessing hyperparameter importance. *ICML*, 754–762.
- Watanabe, S., Bansal, A., & Hutter, F. (2023). PED-ANOVA: Efficiently
  quantifying hyperparameter importance in arbitrary subspaces. *IJCAI*.
  arXiv:2304.10255
- Theodorakopoulos, D., et al. (2024). Hyperparameter importance analysis for
  multi-objective AutoML. arXiv:2405.07640
- Shehata, M., et al. (2025). Hyperparameter sensitivity analysis of
  reinforcement learning in autonomous driving environments.
- Deo, Y., et al. (2025). Metrics that matter: Evaluating image quality metrics
  for medical image generation. arXiv.
- Galdran, A., Costa, P., Anjos, A., et al. (2018). A no-reference quality metric
  for retinal vessel tree segmentation. *MICCAI*, 82–90.
- Jozdani, S. E., & Chen, D. (2020). On the versatility of popular and recently
  proposed supervised evaluation metrics for segmentation quality of remotely
  sensed images. *ISPRS Journal of Photogrammetry and Remote Sensing*, 160, 275–290.
- Kandasamy, K., Dasarathy, G., Schneider, J., & Póczos, B. (2017). Multi-fidelity
  Bayesian optimisation with continuous approximations. *ICML*.
- Mikkola, P., Martinelli, J., Filstroff, L., & Kaski, S. (2022). Multi-fidelity
  Bayesian optimization with unreliable information sources. *AISTATS 2023*.
  arXiv:2210.13937
- Muthusivarajan, R., et al. (2024). Evaluating the relationship between magnetic
  resonance image quality metrics and deep learning–based segmentation accuracy
  of brain tumors. *Medical Physics*.
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018).
  Hyperband: A novel bandit-based approach to hyperparameter optimization.
  *Journal of Machine Learning Research*, 18, 1–52.

---

## 14. Open questions (defer to planning)

- Default `λ` for the stability penalty (resolved: `level` = median, `dispersion`
  = IQR, on a normalized higher-is-better scale; `λ` still needs a conservative
  default + empirical calibration on real plates).
- fANOVA freezing threshold, and how many warm-up trials before the first
  importance estimate is trustworthy enough to freeze a parameter.
- Pruning low-fidelity representativeness: how many calibration images at the
  first rung before early-stopping is safe (guard against unreliable low-fidelity).
- `ReferenceFreeScorer` meta-validation: minimum ground-truth set size and the
  correlation threshold below which the engine refuses to optimize on the proxy.
- Whether `Sweep` gains the range types in-place or `SearchSpace` accepts a richer
  declaration alongside the legacy `Sweep` list.
- MCP transport / packaging location (`src/phenotypic/mcp/` vs. a standalone entry).
