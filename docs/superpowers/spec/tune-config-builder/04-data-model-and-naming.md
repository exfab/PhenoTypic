# 04 — Data Model, Naming & Validation

## The spec the GUI authors

The GUI assembles a `TuningSpec` (`tune/_spec.py`), the single round-trippable
pydantic root:

```
TuningSpec
├── pipeline:      ImagePipeline        (Setup → Pipeline)
├── search_space:  SearchSpace(knobs)   (Setup → Search space)
├── scorer:        ScorerField          (Setup → Scorer)
├── evaluator:     Evaluator            (Run → Advanced)
├── strategy:      StrategyConfigField  (Run → Strategy & budget)
├── budget:        Budget               (Run → Strategy & budget)
└── held_out:      HeldOutConfig        (Run → Advanced)
```

The GUI builds this object, validates it against the live pipeline
(`TuningSpec` already cross-checks knob targets via its `model_validator`), and
serializes via pydantic. **`TuningSpec.to_json(filepath, indent)` already exists**
(`_spec.py`, with `ensure_typed_json_suffix` normalization built in) — the GUI
calls it; no new wrapper is needed. Canonical run writes route through
`tuning_spec_path(output_dir)`.

## Model changes the GUI depends on

### 1. `FloatRange.step` (detailed in doc 02)

Add `step: float | None`, a quantizing `values()`, grid/random/Optuna wiring,
and the step↔log guard. Required for the float-step affordance in the domain
editor; also makes a stepped float grid-enumerable.

### 2. `TuningSpec.phenotypic_version` (provenance stamp)

`TuningSpec` has **no top-level version field today** — only the *embedded*
pipeline carries `version`. Add a top-level provenance field:

- Name it **`phenotypic_version`** (not bare `version`) to disambiguate from the
  nested pipeline's `version`.
- Use a **`default_factory`** that reads the installed package version
  (`importlib.metadata.version("phenotypic")`), so **every** construction —
  including in-memory specs that are never explicitly serialized — is stamped,
  rather than only stamping inside `to_json()`. Field default is the running
  version; old files that lack the key load fine (the factory only fills when
  absent... so loads of old JSON get the *current* version — acceptable, since on
  load we compare the file's value when present and treat absence as "pre-stamp,
  unknown").
- On load, compare a present `phenotypic_version` against the running version and
  **warn on mismatch** (advisory provenance, not a hard gate; the spec still
  loads). The pattern is already proven for image IO
  (`_accessor_io_handler` stamps a `phenotypic_version`).

This is version *stamping*, distinct from version *history*; keeping multiple
named specs in the library (below) is the lightweight-history story for v1.

## Naming: typed config suffixes (D11)

Per the merged config-suffix migration
(`docs/superpowers/specs/2026-06-08-config-json-suffix-migration-design.md`,
implemented in `tools_/_io_constants.py`), reusable config files carry typed
terminal suffixes instead of plain `.json`:

| Artifact | Filename | Constant |
|----------|----------|----------|
| Tuning spec (full or inferred proposal) | `tuning_spec.json.pht-tune` | `TUNING_SPEC_JSON` |
| Exported tuned pipeline | `best_pipeline.json.pht-pipe` | `BEST_PIPELINE_JSON` |
| Pareto winner per objective | `pareto/best_<objective>.json.pht-pipe` | (templated) |
| Base pipeline | `pipeline.json.pht-pipe` | `PIPELINE_JSON` |

Rules the GUI must follow:

- **Never spell the suffix literals** (`.json.pht-tune`, `.json.pht-pipe`)
  anywhere outside `_io_constants.py` — import the constants/helpers.
- **Save normalization** uses `ensure_typed_json_suffix(path, CONFIG_SUFFIX_*)`
  so a user-typed `x` or `x.json` becomes `x.json.pht-tune`, never doubled.
- **File pickers** must show *both* the typed suffix and legacy `.json`, matched
  with `matches_any_suffix(path, TUNING_CONFIG_SUFFIXES / PIPELINE_CONFIG_SUFFIXES)`
  — not `Path.suffix` (which only sees `.pht-tune`).
- **Discovery/read** of an existing run's spec uses the legacy-tolerant
  `resolve_tuning_spec_path(...)` / `resolve_pipeline_config_path(...)` fallbacks.
- **UI copy** states these are JSON, type-tagged by suffix for queryability;
  legacy `.json` still loads.

## Save / load model (D12)

Two destinations with distinct purposes — do not conflate them:

### Library (explicit Save/Load)

- **Default location:** `.phenotypic-gui/presets/tune/` — the sandbox preset
  convention (`SANDBOX_GUI_DIRNAME` + `SANDBOX_PRESETS_SUBDIR`) already used by
  the builder and run-console. **Not** `.phenotypic/` (CLI machine-state cache)
  and **not** `.pht-tune-cache/` (per-run Optuna/study state) — those are
  disposable machine state and would mis-file a user's reusable specs.
  - Add a **`SANDBOX_TUNE_PRESETS_SUBDIR = "tune"`** constant in
    `gui/_config.py` (per the CLAUDE.md "no re-spelled strings" rule); the
    library path is `sandbox / SANDBOX_GUI_DIRNAME / SANDBOX_PRESETS_SUBDIR /
    SANDBOX_TUNE_PRESETS_SUBDIR`. The existing run-console saves presets to
    `presets/` directly, so the `tune/` subfolder is the new disambiguator.
- **Browse…** opens a file picker to save/load anywhere outside the library.
- This is the cross-run reuse library; it holds many named `.json.pht-tune`
  specs (lightweight versioning).

### Run copy (automatic on Deploy)

- Deploy **always** writes the run's own canonical copy to
  `deliverables/tuning_spec.json.pht-tune` via `tuning_spec_path(output_dir)` —
  the reproducibility record bundled with the run output, independent of whether
  the user clicked Save.

So "Save" (library, optional) and "Deploy persistence" (run record, automatic)
are separate concerns and both exist.

## Validation contract (blocked-deploy)

Validation runs on every relevant edit and aggregates into the footer that gates
the forward action. Issues, by section:

| Issue | Section | Blocks | Kind |
|-------|---------|--------|------|
| No active knobs | Search space | Continue + Deploy | spec-level |
| `low ≥ high` on an active range knob | Search space | Continue + Deploy | spec-level, **client-side only** |
| QC scorer with no metadata CSV | Scorer | Continue + Deploy | spec-level |
| `grid` strategy + active continuous float | Run (pre-flight) | Deploy only | run-level |

Surfacing is consistent at three altitudes:

- **Section** — red left border + a header badge counting that section's issues.
- **Field** — inline red message at the offending control.
- **Footer** — aggregates *all* blocking issues into one line and disables the
  forward action. **Setup's footer gates Continue** (Setup-local issues);
  **Run's footer gates Deploy** and sees the **union** (Setup spec-level issues
  + Run-level issues), so a spec broken upstream can never deploy. Deploy is
  additionally hard-guarded in code, not only via the disabled style.

Note the deliberate split: spec-level issues block on *both* footers; the
grid+float conflict blocks only on Run because it is a strategy×space
interaction, not a spec defect.

### What validation cannot do

Relational constraints between knobs are **client-side validation only**. There
is no Optuna `constraints_func` wired, so the GUI can block an obviously
infeasible spec (`low ≥ high`) but cannot steer the sampler away from an
infeasible region defined across knobs. Cross-knob constraints are an explicit
non-goal (doc 02).
