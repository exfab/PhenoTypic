# Hyperparameter Tuning

The tune engine searches over an `ImagePipeline`'s parameters and returns the
combination that scores best on your plates. It is sample-efficient — grid and
seeded-random for small spaces, Optuna samplers (TPE, CMA-ES, GP, NSGA-II) for
larger ones — and pairs the search with pluggable scoring objectives and a
robust held-out generalization check, so a winning pipeline has to prove it
holds up on plates the search never optimized against.

The engine replaces the old `sweep` module: instead of enumerating a fixed
grid of pipeline variants, you author (or auto-infer) a search space, pick a
scoring objective, and let the optimizer propose candidates.

```{admonition} Migrating from a legacy `sweep` manifest
:class: tip

If you have an existing sweep manifest, convert it to a `tuning_spec.json` —
a `Categorical` grid over the manifest's varying parameters (plus
`__enabled__` presence knobs for optional ops) — with the migration script:

    python scripts/migrate_sweep_manifest.py old_manifest.json \
        -o tuning_spec.json --metadata layout.csv

`--metadata` is the per-image layout CSV/Parquet that feeds the count scorer
(grouped by `--groupby`, default `Metadata_ImageName`). Manifests that swept
**nested** operations raise `NotImplementedError` — author those spaces by
hand against the new `SearchSpace` API.
```

## Python interface

### Running a study from the CLI

The basic run takes a `tuning_spec.json`, an image directory, and an output
directory:

```bash
python -m phenotypic.tune run spec.json -i ./plates -o ./out
```

Override the spec's strategy and trial budget, and turn on the two-round
screening freeze, from the command line:

```bash
python -m phenotypic.tune run spec.json -i ./plates -o ./out \
    --strategy tpe \
    --n-trials 200 \
    --screen
```

`--strategy grid` and `--strategy random` use the built-in configs;
`tpe`, `cmaes`, `gp`, and `nsga2` build an `OptunaConfig` and require the
`tune` extra (`uv sync --extras tune`). `--screen` enables the two-round
screening freeze (it is off by default; `--no-screen` is the explicit
default). `--n-trials` overrides the spec's `Budget.n_trials`.

A run writes its deliverables under `<output>/deliverables/`:

- `best_pipeline.json` — the winning pipeline, ready to run with
  `python -m phenotypic`.
- `tuning_spec.json` — the fully resolved spec (pipeline + space + scorer +
  strategy + budget), so the run round-trips.
- `param_importance.json` — per-parameter importance (random-forest / fANOVA)
  showing which knobs moved the objective.
- `pareto/` — the non-dominated front, for a multi-objective
  (`CompositeScorer(multi_objective=True)`) study.
- `generalization.json` — the held-out verdict: how the winner scored on the
  reserved plates versus the search set.

Alongside the deliverables, `trials.parquet` is written at the **output
root**. It is dual-purpose: the Optuna resume store *and* the trial journal.

**Resume** is automatic. Re-running `run` with `-o` pointed at a prior run
directory *resumes* the study rather than restarting it — grid/random pick up
from `trials.parquet`, while the Optuna samplers resume from the study store
(the local `.pht-tune-cache/study.db` or the `--storage-url` study). A killed or
extended run picks up its prior trials.

### Authoring a `TuningSpec` in Python

A `TuningSpec` is one self-contained, round-trippable recipe. Its fields are:

- `pipeline` — the base `ImagePipeline` being tuned (embedded in the spec).
- `search_space` — the `SearchSpace` of `Knob`s to search over.
- `scorer` — the tuning objective (any `Scorer` subclass; see below).
- `evaluator` — the candidate-evaluation policy (`Evaluator`).
- `strategy` — the optimizer config (`GridConfig`, `RandomConfig`, or
  `OptunaConfig`).
- `budget` — the stopping criteria (`Budget`).
- `held_out` — the held-out / generalization policy (defaults to a
  conservative `HeldOutConfig`).

All of these are keyword-only pydantic models. A minimal spec — tune a
Gaussian-blur `sigma` ahead of an Otsu detector, scored against an expected
96-well colony count:

```python
>>> import pandas as pd
>>> from phenotypic import ImagePipeline
>>> from phenotypic.enhance import GaussianBlur
>>> from phenotypic.detect import OtsuDetector
>>> from phenotypic.analysis import ExpectedVsDetectedCount
>>> from phenotypic.tune import (
...     TuningSpec, Budget, Evaluator, SearchSpace, Knob, FloatRange,
...     QCScorer, GridConfig,
... )
>>> pipe = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
>>> space = SearchSpace(knobs=(
...     Knob(key="0.sigma", domain=FloatRange(low=0.5, high=8.0)),
... ))
>>> layout = pd.DataFrame(
...     {"Metadata_ImageName": ["plate1"] * 96, "Object_Label": list(range(96))}
... )
>>> scorer = QCScorer(
...     check=ExpectedVsDetectedCount(
...         metadata=layout, groupby=["Metadata_ImageName"]
...     )
... )
>>> spec = TuningSpec(
...     pipeline=pipe,
...     search_space=space,
...     scorer=scorer,
...     evaluator=Evaluator(),
...     strategy=GridConfig(),
...     budget=Budget(n_trials=8),
... )
>>> spec.search_space.keys()
['0.sigma']

```

A knob's `key` is a position-indexed path: `"0.sigma"` targets the `sigma`
field of op `0` (the `GaussianBlur`). Save the spec with
`spec.model_dump_json()` and feed it to `python -m phenotypic.tune run`. For a
spec that must round-trip through JSON, configure the count check from a
metadata *path* (`ExpectedVsDetectedCount(metadata="layout.csv", ...)`) — a
check built from an in-memory frame cannot be rebuilt from JSON.

### Inferring a search space with `auto-space`

When you don't want to hand-author every knob, `auto-space` mines a configured
pipeline's pydantic fields into a reviewable candidate space — no engine runs:

```bash
python -m phenotypic.tune auto-space pipeline.json -o ./out
```

The same inference is available in Python as `infer_search_space`, which
returns an `InferredSearchSpace` carrying:

- `.knobs` — the inferred `Knob`s (each with a `source` provenance tag and a
  `needs_review` flag).
- `.excluded` — `Excluded` records for fields inference declined to tune, each
  with a reason.
- `.needs_review` — `True` when any knob is flagged for review (or a field was
  excluded blindly).

Flagged knobs (`needs_review=True`) keep the autonomy gate conservative: an
unattended run stays cautious until a human confirms the shaky guesses, so a
generously-inferred bound never silently drives a fully-automatic study.

## The four scoring objectives

Every scorer is a `Scorer` subclass with a higher-is-better objective in
`[0, 1]`. Pick the one that matches the ground truth you have.

### `QCScorer`

Use it when you have **no ground-truth masks** but you know the expected colony
**count** per plate (e.g. a 96-well layout). It is a purely statistical check:
it wraps `phenotypic.analysis.ExpectedVsDetectedCount` and scores each
`groupby` unit on `|detected − expected| / expected`, folded to a
higher-is-better term.

It requires a configured count check on its `check` field. Configure that check
from a metadata **path** so the scorer round-trips through `tuning_spec.json`:

```python
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune import QCScorer

scorer = QCScorer(
    check=ExpectedVsDetectedCount(
        metadata="layout.csv", groupby=["Metadata_ImageName"]
    )
)
```

### `ReferenceFreeScorer`

Use it when you have **no ground truth at all** — not even expected counts — and
want to score segmentation quality from the result itself. It computes
fixed-normalized proxy terms: `ShapeRegularity` (mean shape-prior plausibility
from the `Shape_*` columns), `Contrast` (Otsu between-class separation of
foreground vs. background), and `SizeCV` (within-replicate size uniformity).
Supply an optional `count_check` to add the `Count` term.

It requires nothing to construct, but it is **gated behind meta-validation**:
`availability()` returns `False` (so the engine fails safe and degrades to
`QCScorer`) until the proxy is correlated against ground truth and clears the
enable threshold (Spearman ρ ≥ 0.7; ρ ≥ 0.8 for fully unattended tuning). Point
`gt_masks_source` at a ground-truth source to let the gate validate; without it
the gate abstains and the scorer stays unavailable.

```python
from phenotypic.tune import ReferenceFreeScorer

scorer = ReferenceFreeScorer(replicate_groupby=["Metadata_ImageName"])
```

### `SupervisedScorer`

Use it when you **have ground-truth annotations**. It is modality-tiered: a
directory of per-image masks runs the **mask tier** (per-object Dice or IoU,
macro-averaged into the `Region` term); a `.csv`/`.parquet` count table runs the
**count tier** (count divergence into the `CountMAE` term); nothing resolvable
makes it abstain.

It requires a `GroundTruthMasks` loader on its `gt` field, whose
`gt_masks_source` path selects the modality. The mask tier carries exactly one
region metric (`region_metric="dice"` or `"iou"` — they rank identically); the
count tier additionally needs a configured `count_check`:

```python
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune import GroundTruthMasks, SupervisedScorer

# Mask tier: a directory of per-image masks.
masks = SupervisedScorer(
    gt=GroundTruthMasks(gt_masks_source="./gt_masks/"),
    region_metric="dice",
)

# Count tier: a count table plus the reused count check.
counts = SupervisedScorer(
    gt=GroundTruthMasks(gt_masks_source="./counts.csv"),
    count_check=ExpectedVsDetectedCount(
        metadata="counts.csv", groupby=["Metadata_ImageName"]
    ),
)
```

### `CompositeScorer`

Use it to **combine** the above into one objective — the small complementary
panel no single metric covers. It nests a `list[Scorer]` on its `scorers` field
and composes them two ways:

- **single-objective** (default): a `weights`-weighted arithmetic blend, or
  (without weights) the geometric mean — so one weak axis cannot be masked by a
  strong one.
- **multi-objective** (`multi_objective=True`): per-child objectives kept
  separate, driving a true Pareto study (requires an Optuna `nsga2` strategy;
  pairing a multi-objective scorer with grid/random is rejected at spec
  construction). The non-dominated front lands in `deliverables/pareto/`.

```python
from phenotypic.tune import CompositeScorer

scorer = CompositeScorer(
    scorers=[qc_scorer, reference_free_scorer],
    weights={"s0": 2.0, "s1": 1.0},
)
```

## GUI interface

`phenotypic-gui` mounts a `/tune/` Dash co-pilot as a fifth tab alongside
**Pipelines · Tune · Run · Viewer · Analysis**. The co-pilot is **read-only over
a tuning run** — it never re-optimizes. It reads a finished or in-flight
`python -m phenotypic.tune` output directory and only ever *writes*
`best_pipeline.json` and `tuning_spec.json`. It has four sub-tabs:

- **Monitor** — the live study read: objective curve (raw scores + running
  best), a parameter-importance bar chart, a winner-stability
  (generalization-gap) badge, and a trials table, polled while a run is in
  flight.
- **Curate** — a shortlist of the best (and gap-flagged) trials; pin two into
  **A** / **B** slots and compare their colony overlays side-by-side on any
  plate. "Set as winner" writes `best_pipeline.json`.
- **Space** — the inferred search space as editable knob-rows; export the
  edited space back to `tuning_spec.json`.
- **Launch** — a strategy / budget / storage-URL form whose live command card
  renders a copy-paste `python -m phenotypic.tune run` invocation. The GUI does
  not launch the optimizer; you copy the command and run it yourself.

For the full walkthrough, see the
[Tune co-pilot tutorial](../../tutorials/gui/16_tune_copilot.md).

## Distributed runs

For running a single tune study across many SLURM compute nodes that share one
Optuna study (via `--slurm` and a Postgres `--storage-url`), see
[Distributed Tuning on HPCC Clusters](tune_distributed_hpcc.md).
