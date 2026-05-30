# CLAUDE.md

## Quick Start

**`uv` is the sole package manager and runner.** Never use bare `python` or `pip`.

- `uv run <cmd>` — run commands
- `uv add <package>` (or `--group dev`) — add dependencies
- `uv sync` — sync env (after checkout or in new worktrees)
- `uv sync --group dev --group docs --extras gui` — full dev env
- `source .venv/bin/activate` — manual venv activation

### Linting & Type Checking

- `uv run mypy src/phenotypic` — type checking
- `uv run ruff check --fix` — format and lint

### CLI

- `uv run python -m phenotypic` — single pipeline on images/directories (parallel,
  SLURM, resume)
- `uv run python -m phenotypic.sweep` — parameter sweeps across pipeline variants

### GUI hub

- `uv run phenotypic-gui --root ./images --port 8050` — unified hub: builder +
  results viewer + run console mounted under one URL via Werkzeug
  `DispatcherMiddleware`. SSH-tunnel from a workstation:
  `ssh -L 8050:localhost:8050 user@cluster`.
- `uv run python -m phenotypic.gui --root ./images` — equivalent module entry.
- Standalone tools still work: `python -m phenotypic.gui.builder`,
  `python -m phenotypic.gui.results_viewer`, `python -m phenotypic.gui.run_console`.
- Note: `phenotypic gui` (no hyphen, as a subcommand of the existing CLI) is NOT
  supported. Use `phenotypic-gui` or `python -m phenotypic.gui`.

#### Adding GUI features

Two ledgers track the GUI surface; both are CI-gated:

- **`src/phenotypic/gui/FEATURES.md`** — every individual user-visible
  affordance (button, badge, store, callback, route). The
  `gui-checks` workflow's `features-md-gate` job rejects any PR that
  touches `src/phenotypic/gui/` without modifying `FEATURES.md`.
  Pre-commit also validates `Test ref` on `✅ shipping` rows.
- **`src/phenotypic/gui/WORKFLOWS.md`** — every end-to-end user flow
  worth a tutorial page. Adding a row here REQUIRES adding a matching
  `_capture_<id>` function in `scripts/capture_gui_tutorial_screenshots.py`
  and a walkthrough page under `docs/source/tutorials/gui/`.
  The `gui-checks` workflow's `workflows-md-gate` job runs
  `scripts/check_workflows_md.py` (also available as a pre-commit
  hook) to enforce the round-trip.

Run `uv run python scripts/capture_gui_tutorial_screenshots.py` after
any visible chrome change and commit the refreshed PNGs alongside the
source change. The `gui-checks` workflow's `smoke-capture` job
regenerates them on Ubuntu and uploads as a build artifact for
spot-checking, but cross-platform font rendering means committed PNGs
should come from a developer workstation, not CI.

The capture regenerates the **full** screenshot set, so unrelated
tutorials' PNGs shift by a few bytes (font-rendering noise) on every
run. **Commit them all — do not cherry-pick or `git checkout --` the
collateral.** Full regeneration + commit-everything keeps the workflow
simple and the committed render internally consistent; the accepted
cost is occasional binary churn in history.

---

## Architecture

**Purpose:** Modular image processing for arrayed colony phenotyping on solid media (
agar plates).
**Philosophy:** Accuracy over speed. Be mindful of memory — images are large and
operations copy data; avoid unnecessary intermediate allocations.

### Five Layers

1. **Image Data** — `Image`/`GridImage` with accessor pattern, lazy evaluation, caching
   (`image.rgb[:]`, `image.detect_mat[:]`, `image.color.Lab[:]`).
   See [_core/CLAUDE.md](src/phenotypic/_core/CLAUDE.md).
2. **Operation ABCs** — `_operate(image) -> image` interface.
   See [abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md) for hierarchy and reference
   implementations.
3. **Pipeline** — `ImagePipeline` chains operations, batch execution, YAML/JSON
   serialization, automatic benchmarking.
4. **Enhancement** — preprocessing ops on `detect_mat`; RGB/gray unchanged.
5. **Post-Measurement** — `post/` transforms DataFrames in the final stage of
   `ImagePipeline.measure()`.
   `analysis/` provides standalone statistical tools (edge correction, growth curves,
   outlier removal) for exported data.

### Design Decisions

- **Operations are pydantic v2 models:** every operation and analyzer is a
  `pydantic.BaseModel` rooted at `BaseOperation`. Parameters are **annotated
  class-level fields** — there is no hand-written `__init__`; construction is
  **keyword-only**; invalid input raises `pydantic.ValidationError` (a `ValueError`
  subclass). Algorithm bodies (`_operate`/`apply`/`measure`) are unchanged. Every class
  exposes a machine-readable contract via `model_json_schema()`, with field
  descriptions auto-derived from the Google-style `Args:` docstring. To add a
  parameter, declare a typed field; put input normalization and guards in a
  `field_validator`, never an `__init__`. Raw-array params use the reusable
  `NdArrayField` type; operation-valued params use `OperationField` (both in
  `tools_/typing_.py`).
- **Public API:** only `__init__.py` exports are public; `_implementation.py` files are
  private.
- **Immutability:** operations return copies; never modify `image.rgb`/`image.gray`
  directly.
- **Explicit:** use `ImagePipeline` for multi-step workflows; no hidden state.
- **Domain-specific:** built for microbe phenotyping; use microbiology context in
  docs/examples.
- **Duck typing** for type checks; **explicit matplotlib** (no implicit pyplot).
- **Reproducibility:** `to_json()`/`from_json()` serialization; fixed random seeds.
- **Cross-platform:** macOS, Windows, Linux; use try/except for platform-specific
  imports.

---

## Code Style

- **Google-style docstrings** everywhere. Order and ImageOperation conventions live
  in [abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md).
- All doctest examples must be **runnable** using `load_synth_yeast_plate()`; use
  microbiology context (colony visibility, edge sharpness, mask quality).
- **Never create** separate example files/notebooks — examples go in docstrings.
- Don't create summary documents unless explicitly asked.
- **Explicit naming:** no generic `main()`, `run()`, `process()` — name after what it
  does.
- Break large functions into smaller, testable helpers with private methods.
- For batch processing, use the CLI (`python -m phenotypic`) not custom scripts.
- Import `phenotypic.settings_` before other modules when modifying settings.

---

## Module Guides

- [_core/CLAUDE.md](src/phenotypic/_core/CLAUDE.md) — Image class, accessors
- [abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md) — ABC hierarchy, implementation
- [schema/CLAUDE.md](src/phenotypic/schema/CLAUDE.md) — public measurement schema (`MeasurementInfo` base + header enums)
- [tools_/CLAUDE.md](src/phenotypic/tools_/CLAUDE.md) — mixins, utilities
- [settings_/CLAUDE.md](src/phenotypic/settings_/CLAUDE.md) — global config
- [enhance/CLAUDE.md](src/phenotypic/enhance/CLAUDE.md) — enhancer conventions
- [gui/CLAUDE.md](src/phenotypic/gui/CLAUDE.md) — GUI sub-apps, shared `_config.py`
  constants, `_design.py` tokens
- [DESIGN.md](DESIGN.md) — dashboard & plot style guide
- `src/phenotypic/post/`, `src/phenotypic/analysis/` — no sub-CLAUDE.md

## Key Files

- `src/phenotypic/_core/_image.py` — `Image` class
- `src/phenotypic/_core/_image_pipeline.py` — Pipeline implementation
- `src/phenotypic/abc_/` — Operation interfaces
- `src/phenotypic/__main__.py` — CLI entry point

## Code Style

Public parameters with closed value sets: type as `EnumType | Literal["a", "b", ...]`,
normalize in a `field_validator(mode="before")` with `value = EnumType(value)`, and use
only enum members internally.
Define the `Literal` alias once as a `TypeAlias` and reuse. If both `Enum` and `Literal`
exist, add a test asserting their values match.

**Closed value sets needing user-visible documentation: prefer `MeasurementInfo` /
`ConstantLabels`** (in `phenotypic.tools_.constants_`). Each member is a
`(label, description)` tuple, the description is accessible to callers, and the existing
pattern (override `category()` classmethod, optionally `__new__` for bare-label values)
is the project convention. The `MeasurementInfo` base class lives in the public
`phenotypic.schema` package; framework-config constant enums (`GAMMA_ENCODINGS`,
`PIPE_STATUS`, `METADATA`) stay in `phenotypic.tools_.constants_`, while the
per-feature measurement-column enums live in `phenotypic.schema`. Do not modify these
classes' internals to satisfy the generic `MyEnum(value)` normalization — their bespoke
coercion (e.g. `_GAMMA_COERCE` for `GAMMA_ENCODINGS`) is intentional.

For **type-only enforcement** of a closed set with no documentation surface (CLI dispatch
keys, internal mode flags), a `Literal[...]` `TypeAlias` in `tools_/typing_.py` is
sufficient — no Enum needed. Examples: `FootprintShape`, `DetectMode`, `ExecutionMode`,
`ImageTypeName`, `ProcessingStatus`.

Pair an Enum with a `Literal` alias only when both forms are used at boundary code
(string-typed external input + enum-typed internal storage), and add an alignment test
(`set(get_args(MyLiteral)) == {m.value for m in MyEnum}` — see
`tests/unit/tools_/test_io_constants.py::TestEnumLiteralAlignment::test_image_type_literal_covers_base_and_grid_enum_values`).
When the Literal intentionally covers only a subset of the Enum's members (e.g.
`ImageTypeName` exposes only `BASE` and `GRID`, not the internal-only `CROP`/`OBJECT`/
`GRID_SECTION`), assert with `issubset` instead and document the partial coverage in
the test docstring.

Parameterized strings are not enumerations: keep the template as a private `Final[str]`
and expose a typed render function whose parameters are the public API.

Never accept bare `str` for closed sets, never propagate raw strings past the boundary,
never derive `Literal` from runtime expressions.

## Gotchas

- Some packages excluded on Windows: `rawpy`, `pympler`, `jupyter` — use try/except.
- External tools: ExifTool (raw metadata), Pandoc (doc builds).
- **Operations use `.apply()`, not `__call__`:** `op.apply(image)` is correct;
  `op(image)` raises `TypeError`.
- **Operations are keyword-only constructed:** `OtsuDetector(ignore_zeros=True)`, not
  `OtsuDetector(True)` — pydantic models take no positional args. Unknown kwargs and
  invalid values raise `pydantic.ValidationError`.
- **Measurement columns are category-prefixed:** `Size_Area`, `Shape_Circularity`,
  `Intensity_MeanIntensity`, etc. The header enums are the **public**
  `phenotypic.schema` package (`from phenotypic.schema import SHAPE, SIZE, ...`);
  the old `phenotypic.tools_.measurement_info` path was removed.
  `MeasurementInfo.get_labels()` returns unprefixed names; `get_headers()` returns the
  prefixed column names used in DataFrames.
- **Analysis classes use `.analyze()`:** `EdgeCorrector.analyze(df)`,
  `LogGrowthModel.analyze(df)` — not `.fit()` or `.correct()`.
- **`num_objects` is on `Image`**, not on the `objmap` accessor: use
  `image.num_objects`.
- **Master vs. mirror outputs:** `master_measurements.{csv,parquet}` is a
  **clean, pre-post, metadata-free archive** of what per-image runs
  measured; `measurements.{csv,parquet}` is the **post-applied mirror**
  the GUI viewer reads/curates. Per-image parquets in
  `results/<ds>/measurements/` are also clean — the CLI calls
  `pipeline.measure(image, apply_post=False)` on the per-image path.
  Post is applied once at the end of aggregation against the merged
  master, and the post-applied frame is what `analysis.{csv,parquet}`
  and `measurements_by_feature/<feature>.{csv,parquet}` are derived
  from. The external `--metadata` CSV inner-join also lands on the
  post-applied frame (inside `finalize_post_master_outputs`), so the
  mirror, per-feature splits, and `analysis.{csv,parquet}` carry the
  metadata columns while the master archive stays both post-free and
  metadata-free. Code paths that need to feed a frame to analysis
  plugins or dashboards should read `measurements.parquet`, not
  `master_measurements.*`.
- **Finalize via `finalize_post_master_outputs` for FINAL master writes:**
  any code path that writes `master_measurements.{csv,parquet}` *as the
  run's final output* must immediately call
  `phenotypic._cli._cli_output_manager.finalize_post_master_outputs(
  output_dir, master_df, pipeline)`. The `aggregate_measurements`
  (forward CLI) and `--recompile` worker (`_run_post_master_steps`)
  callers already do this.

  Mid-run intermediate writers (`_aggregate_chunks_locked` in
  `_cli_chunk_writer.py`) intentionally bypass
  `finalize_post_master_outputs`: chunks publish partial results so users
  can download mid-run, but the post pipeline, per-feature splits,
  analysis chain, and `pipeline.json` persistence are deferred to the
  run's final aggregation. Don't add `finalize_post_master_outputs` to
  the chunk writer — it would re-run expensive finalize work on every
  checkpoint.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
