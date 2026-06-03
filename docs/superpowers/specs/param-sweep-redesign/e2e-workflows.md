# End-to-End User Workflows

Companion to the [parameter tuning engine design](2026-06-01-parameter-tuning-engine-design.md).
The concrete user journeys for the tuning engine — **backend (CLI)** and **frontend
(Dash co-pilot)** — and how they share one engine, one study, and the artifacts that flow
between them.

- **Status:** Reference (cross-cutting; synthesises the other companions into user-facing
  flows).
- **Maps to:** master §6 (drivers / shared study), D5/D6. Threads through
  [`search-space-inference.md`](search-space-inference.md),
  [`qc-objective-mapping.md`](qc-objective-mapping.md),
  [`robust-evaluation.md`](robust-evaluation.md),
  [`screening-importance.md`](screening-importance.md),
  [`optuna-integration.md`](optuna-integration.md),
  [`dash-copilot-design.md`](dash-copilot-design.md), and
  [`engine-architecture.md`](engine-architecture.md).

---

## 1. The shared model (read this first)

The frontend and backend are **not two systems** — they are two **drivers** over **one
`TuningEngine`** and **one shared `study.db`** (master D6). The backend (CLI) drives
*headlessly*; the frontend (Dash) drives *interactively* and **delegates launching to the
backend** while **attaching to the same study**. A future MCP server is the deferred third
driver on the same study.

**Three artifacts are the connective tissue:**

| Artifact | What it is | Produced by | Consumed by |
|----------|------------|-------------|-------------|
| **`tuning_spec.json`** | the run recipe: `SearchSpace` + `Scorer` + `StrategyConfig` + `Evaluator` + `Budget` (one `TuningSpec`, engine-arch §6) | CLI `--auto-space` edit, the Dash space-edit form, or Python-first | `python -m phenotypic.tune` |
| **`study.db`** | the live shared state (Optuna SQLite, WAL) — trials, scores, pruning, `user_attrs` | the engine's ask-and-tell loop | CLI resume · Dash monitor/curate · MCP |
| **`deliverables/best_pipeline.json`** | the winning `ImagePipeline`, ready to run | the winner pick (CLI auto / human curation) | `python -m phenotypic` (production run) |

---

## 2. Backend — the CLI (`python -m phenotypic.tune`)

The headless / batch / SLURM / reproducible / agent-facing path.

1. **Start from a pipeline.** A `pipeline.json` (from the builder GUI, a prefab, or
   hand-authored) is the thing whose parameters you tune.
2. **Get a search space.** `python -m phenotypic.tune --auto-space pipeline.json` runs
   `infer_search_space` (search-space §8) and prints a **review table**: `✓` reliable knobs
   (bool/enum/bounded), `⚠ needs_review` guesses (the unbounded `[d/4, d·4]` windows,
   anchored on your current values), and the *"couldn't infer — add a `TuneSpec`"* excluded
   list. Edit it (narrow ranges, drop knobs, add presence) → a **`tuning_spec.json`**. (Or
   author the spec Python-first and serialise it.)
3. **Launch.**
   ```
   python -m phenotypic.tune tuning_spec.json -i INPUT_DIR -o OUTPUT_DIR \
     --strategy {grid|random|tpe|cmaes} --n-trials N \
     --objective {qc|supervised|reference-free|composite} \
     --calibration-frac 0.3 --stability-weight 0.5 \
     --screen --multi-objective --n-jobs 8        # or --slurm
   ```
   `-i/--input` = the image dir; `-o/--output` = where the run writes (defaults to
   `./<input-name>_tune/`; re-point it at an existing run to **resume**). See master §6/§8.
   Defaults that "just work": `--objective qc` (the trustworthy Count-only/panel `QCScorer`,
   qc doc), `--strategy tpe` once the `tune` extra is installed (else `grid`/`random`).
4. **What the engine does** (you mostly watch): builds a **metadata-stratified calibration
   set** (robust-eval §6); runs the ask-and-tell loop — an **unpruned explore round** →
   **fANOVA importance** → (optional) **freeze low-importance knobs** → a **focused round**
   (ASHA-pruned) (screening §3) — distributed over joblib/SLURM against the shared
   `study.db` (optuna §7). Bad candidates are early-stopped; the winner is validated on a
   **held-out group** (robust-eval §8).
5. **Outputs land in `deliverables/`** (master §8): `study.db` (resumable), `trials.parquet`,
   **`best_pipeline.json`**, `pareto/` (multi-objective), `param_importance.json` ("which
   knobs matter"), `tuning_report.html` (objective curve, importance bars,
   calibration-vs-held-out, the generalization-gap flag).
6. **Adopt the winner:** `best_pipeline.json` drops straight into
   `python -m phenotypic INPUT_DIR` for the production run.
7. **Resume / migrate:** re-invoking continues from `study.db` (reproducible bit-for-bit
   under `--deterministic`; reproducible-in-distribution when parallel — optuna §8).
   **`--strategy grid` with no budget reproduces the (now-removed) sweep's exhaustive grid
   output byte-for-byte** — validated against a frozen golden fixture (`sweep` is deleted in
   the hard cutover; master §9).

### Minimal happy path

```
python -m phenotypic.tune --auto-space pipeline.json -i ./plates > tuning_spec.json   # review + edit
python -m phenotypic.tune tuning_spec.json -i ./plates -o ./plates_tune --strategy tpe --n-trials 100 --n-jobs 8
python -m phenotypic ./plates --pipeline ./plates_tune/deliverables/best_pipeline.json   # production run
```

### What a run writes (`OUTPUT_DIR`)

User-facing artifacts (the winner `best_pipeline.json`, `tuning_report.html`,
`param_importance.json`, the resolved `tuning_spec.json`) go in **`deliverables/`**; the
machinery (`study.db`, `trials.parquet`, selective per-trial outputs, `splits/`,
`screening/`, `progress/`, resume state) sits at the `OUTPUT_DIR` root. **See
[master §8](2026-06-01-parameter-tuning-engine-design.md) for the canonical folder tree, the
disk-retention policy, and the resume/handoff semantics** (resolve paths via the
`phenotypic.tools_` helpers, never hand-joined).

---

## 3. Frontend — the `/tune/` Dash co-pilot

The interactive / human-in-the-loop / curation path, in the GUI hub
(`phenotypic-gui --root ./images` → the **Tune** tab). Sub-phased 6a→6b→6c
(dash-copilot §5).

1. **(6c) Review/edit the space.** Load a `pipeline.json` → the `InferredSearchSpace`
   renders as forms (reusing `_param_forms`): domain editors, **provenance badges**,
   **`⚠ needs_review`** flags, the excluded list with "add a `TuneSpec`" hints. Edits emit a
   **`tuning_spec.json`**.
2. **Launch.** "Launch tuning run" **delegates to the run console's `LocalRunner`**, which
   spawns the *same* `python -m phenotypic.tune tuning_spec.json` CLI process and tails it.
   (The GUI never reimplements the optimizer — it drives the backend.)
3. **(6a) Monitor live.** Over the shared `study.db` (WAL, polled via `dcc.Interval`): the
   **objective-vs-trial curve**, the **Pareto front** (knee-point highlighted), the
   **importance bars** (with the fANOVA / RF-fallback method badge), and run status —
   counts, budget progress, the **generalization-gap flag** surfaced loudly.
4. **(6b) Review candidates + curate the winner — the co-pilot's whole point.** A
   **shortlist** appears: top-N by objective (or the Pareto front) **plus** the gap-flagged
   and anti-gaming-*suspicious* trials (dash-copilot decision B). Open a candidate →
   **on-demand detection overlays** (reusing the builder's renderer) + per-image scores.
   Because the `QCScorer` proves only *plausible + reproducible* (not *correct*, qc §7), your
   eyes on the overlays catch the blind spot. You **accept / reject / rank + notes** (stored
   as Optuna `user_attrs`, last-write-wins + attribution) and **pick the winner** →
   `best_pipeline.json`. Your accept/reject optionally feeds the **meta-validation gate**
   (reference-free §E).
5. **Co-drive a shared run (D6).** Because the view *attaches* to a `study.db`, you can
   curate a run an **agent (MCP) or a SLURM job is writing right now** — an agent tunes
   overnight, you review the surfaced candidates the next morning, same study.

### Minimal happy path

`phenotypic-gui --root ./plates` → **Tune** → load `pipeline.json` → review the space →
**Launch** → watch the curve/importance → open the top candidates → accept the best →
**best_pipeline.json** written. Then run it in the **Run** tab (or the CLI).

---

## 4. How they connect

```
   builder/prefab ─► pipeline.json
                        │  infer_search_space   (CLI --auto-space  OR  Dash 6c form)
                        ▼
                   tuning_spec.json ──► python -m phenotypic.tune ──► study.db (shared, WAL)
                                              ▲ delegates launch          │
                          Dash /tune/ ────────┘                           │ attaches (read + user_attrs)
                          (monitor 6a · curate 6b)  ◄──────────────────────┘
                                              │ pick winner
                                              ▼
                                     deliverables/best_pipeline.json ──► python -m phenotypic  (production)
```

- **Backend = drive headlessly; Frontend = drive interactively + curate** — the *same*
  `TuningEngine`, the *same* `study.db`, the *same* `best_pipeline.json`.
- **The Dash view does not optimize** — it delegates launching to the CLI runner and
  attaches to the resulting study. There is exactly one optimizer.

---

## 5. Human decision points

Four places a human (CLI or Dash) makes a call; everything else is automated:

1. **Choose the objective** (`--objective` / the spec). Default `qc` — no ground truth
   needed. Supervised when annotations exist; reference-free only behind its meta-validation
   gate.
2. **Review / edit the inferred space.** Tighten the `⚠ needs_review` guesses, drop
   irrelevant knobs, add presence, un-exclude a field by giving it a range.
3. **(Optional) confirm a freeze.** After screening, accept/adjust which low-importance
   knobs are frozen for the focused round (or let the auto-gate decide).
4. **Pick the winner from the shortlist — with overlays in view.** The machine ranks; the
   human curates (catching the proxy's "correct" gap). This is the co-pilot's core value.

---

## 6. Personas — when to use which

| You want to… | Use |
|--------------|-----|
| Run a tuning job on a cluster / overnight / in a script | **CLI** (`--slurm`) |
| Reproduce today's exhaustive sweep | **CLI** `--strategy grid` (no budget) |
| Visually judge candidate overlays + curate the winner | **Dash co-pilot** |
| Review an agent's overnight run the next morning | **Dash** attaching to the shared `study.db` |
| Programmatic / autonomous tuning | **MCP** (deferred — same engine + study) |

The CLI and Dash are interchangeable entry points to one engine; pick by *how you want to
interact*, not by *what gets tuned*.

---

## 7. The agent (MCP) — deferred third driver

Out of scope for now (per the param-sweep focus), but the architecture leaves the seam open:
the MCP `tune_*` tools (`tune_infer_space` / `tune_suggest` / `tune_report` /
`tune_run_trial` / `tune_best` / `tune_param_importance`, master §6) drive the *same*
ask-and-tell engine against the *same* `study.db` — so an agent and a human co-curate one
run. When prioritised, it slots in behind the existing `TuningEngine` with no change to the
CLI or Dash paths.
