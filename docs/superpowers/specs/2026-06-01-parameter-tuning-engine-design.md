# Parameter Tuning Engine — Design Spec

- **Date:** 2026-06-01
- **Status:** Approved design (pre-implementation)
- **Branch:** `redesign/param-sweep`
- **Supersedes / refactors:** `src/phenotypic/sweep/` (the parameter sweep CLI feature)
- **Author:** brainstormed with Claude (Opus 4.8)

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

The design follows established practice for tuning image-analysis pipelines:

- **Screen first, then optimize.** Teodoro et al. (2016) tune a tissue/nucleus
  segmentation pipeline as a **black box**: a cheap sensitivity-analysis pass
  (Morris One-At-A-Time, then Sobol variance decomposition) prunes
  non-influential parameters *before* optimization. Searching **~100 points** of
  a space with billions–trillions of combinations improved Dice/Jaccard by **up
  to 1.42×** over defaults — the quantitative argument against full grid.
- **Sample-efficient optimizers for expensive evaluations.** Teodoro compared
  Nelder–Mead, Parallel Rank Order, Genetic Algorithm, and **Bayesian
  optimization** (Snoek et al. / Spearmint), favouring model-based search "for
  objective functions whose evaluations are costly."
- **Tuning is naturally multi-objective.** Taveira et al. (2018) extend the same
  platform to maximise segmentation quality **while minimising execution time**
  via scalarization (a-priori weights), because the fitness function is too
  expensive to sample a full Pareto front a posteriori.
- **Scoring without ground truth is a solved-enough problem.** Zhang et al.
  (2008, 1100+ citations) survey unsupervised segmentation-evaluation metrics
  and note they "enable self-tuning of algorithm parameters." Chen et al. (2021)
  build **reference-free quality metrics for cell segmentation**, combining many
  metrics via PCA into a single quality score.
- **Mature tooling.** Optuna provides TPE Bayesian optimization, a define-by-run
  search space, pruning (ASHA/Hyperband), native multi-objective, an
  **ask-and-tell API**, and study persistence — the exact shape an MCP/agent
  loop needs.

Full references in §13.

---

## 3. Locked-in decisions

| # | Decision | Choice |
|---|----------|--------|
| D1 | Objective function | **Pluggable** — supervised / reference-free / domain-QC / human-surfacing are interchangeable `Scorer` implementations. The optimizer never cares how a score was produced. |
| D2 | Scope | **One shared tuning engine.** Grid search is demoted to one `SearchStrategy`. CLI and MCP both wrap the same ask-and-tell core. Reuse the existing manifest / joblib / SLURM execution. |
| D3 | Optimizer backend | **Optuna** is the default behind a thin `SearchStrategy` Protocol. We own random-search and the screening pass directly (no lock-in). The seam stays open for an `AxStrategy` later. |
| D4 | Robust evaluation | **Calibration set + aggregate + stability penalty + held-out validation.** Each trial scores over a representative calibration subset; the score is `median(per-image) ± λ·dispersion(per-image)` so flat optima beat sharp ones; the winner is validated on held-out images. |
| D5 | Co-pilot UI | **Dash only** (a new GUI-hub mount), not napari. |
| D6 | Drivers | The engine is usable interchangeably by **a human (CLI + Dash) and an agent (MCP)**, collaborating on **one shared study**. |
| D7 | Search space | An **automated `infer_search_space`** derives tunable domains from the pydantic operation fields; humans/agents review and edit before tuning. |

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

3. **`Scorer` (Protocol)** — `score(image, result) -> float | dict[str, float]`.
   Implementations:
   - `SupervisedScorer` — ground truth → count error / IoU / Dice / F1 / adjusted Rand.
   - `ReferenceFreeScorer` — intra-colony homogeneity vs. background contrast,
     boundary gradient, shape regularity (Zhang 2008 / Chen 2021 style).
   - `QCScorer` — reuse the existing `analysis/` QC checks: expected-vs-detected
     grid count, ICC replicate reliability, MAD/Tukey outlier rates, edge effects.
   - `CompositeScorer` — weighted scalarization (one number) *or* return a dict
     (true multi-objective Pareto).

4. **`Evaluator`** — the robustness layer (D4). Builds the pipeline from params,
   runs it across the **calibration subset** via the existing joblib/SLURM
   machinery, applies the `Scorer` per image, then aggregates
   `score = median(per-image) ± λ·dispersion(per-image)`. Returns the aggregate
   plus the per-image breakdown, and validates the winner on a **held-out subset**.

5. **`TuningEngine`** — orchestrates the loop; owns the budget (n-trials /
   wall-clock / convergence), persistence, and reporting. **This single object is
   what the CLI and the MCP wrap.**

**Plus an optional `ScreeningPhase`** (on by default above ~6 params): a Morris
One-At-A-Time pass (`r·(k+1)` cheap runs, r≈5–10) ranks parameters by influence
(`μ*` mean elementary effect; `σ` flags non-linearity) and **freezes
non-influential params at their defaults** before optimization. Optional Sobol
variance decomposition when budget allows. Independently valuable as a
plain-English *"which knobs matter for your plates"* answer.

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
  the per-image layout the GUI reads — the regression lock.
- Old manifest JSON stays loadable; `tuning_spec.json` is a new artifact.

---

## 10. Dependency policy

`optuna` goes in an **optional extra** (`uv sync --extras tune`), not core,
matching the existing `gui` / `docs` extras and the project's care with heavy/
platform-specific deps. Grid + random + screening + robust evaluation are
**fully dependency-free** (homegrown), so Phase 1 ships with zero new deps.
Requesting `--strategy tpe` without the extra → a clear, actionable error.
`optuna` is pure-Python / cross-platform (no Windows exclusion). Promoting it to
core later (so TPE is the out-of-box default) is a one-line change. Morris-OAT
screening is implemented in-house (tiny); optional Sobol uses `SALib` only if the
extra is present.

---

## 11. File layout

```
src/phenotypic/tune/
  __init__.py              # public: tune(), SearchSpace, TuneSpec, infer_search_space, scorers
  _search_space/           # _space.py (domains), _infer.py (introspection), _tune_spec.py (marker)
  _strategies/             # _protocol.py, _grid.py, _random.py, _optuna.py (lazy import)
  _scorers/                # _protocol.py, _supervised.py, _reference_free.py, _qc.py, _composite.py
  _evaluator.py            # calibration set, aggregate + stability, held-out validation
  _screening.py            # Morris OAT (+ optional SALib Sobol)
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

- **Unit**: domain sampling; `infer_search_space` on synthetic pydantic ops; each
  `Scorer` on `load_synth_yeast_plate()`; Evaluator aggregate+stability math;
  **`GridStrategy` enumeration == current manifest** (regression lock); seeded
  `RandomStrategy` determinism.
- **Integration**: end-to-end `tune()` on the synth plate with a tiny budget,
  asserting improvement over defaults and an identical study on resume.
- The existing `sweep` test-suite must stay green.
- Fixed seeds throughout (project reproducibility requirement).
- GUI: `FEATURES.md` / `WORKFLOWS.md` + screenshot round-trip when the Dash view
  lands.

### Phasing

Each phase lands behind the review → simplifier → regression cadence.

1. **Engine core, zero deps** — `SearchSpace`, Protocol, `GridStrategy`
   (regression-lock) + `RandomStrategy`, `Evaluator` (robust eval),
   `TuningEngine`, `QCScorer` + `ReferenceFreeScorer`, CLI. *Already beats grid*
   (random + screening + robust aggregation). Shippable alone.
2. **Optuna backend (extra)** — `OptunaStrategy` (TPE/CMA-ES), SQLite
   persistence/resume, param importance.
3. **Auto-space + screening** — `infer_search_space`, `TuneSpec`, Morris OAT.
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

---

## 14. Open questions (defer to planning)

- Exact stability term: dispersion as IQR vs. std vs. worst-case across images, and
  the default `λ`. (Pick a default, expose the knob.)
- Calibration-set selection: random subset vs. stratified by a metadata axis
  (e.g. plate / replicate) vs. user-specified. Default to stratified-if-metadata.
- Whether `Sweep` gains the range types in-place or `SearchSpace` accepts a richer
  declaration alongside the legacy `Sweep` list.
- MCP transport / packaging location (`src/phenotypic/mcp/` vs. a standalone entry).
