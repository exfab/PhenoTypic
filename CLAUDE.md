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
- [tools_/CLAUDE.md](src/phenotypic/tools_/CLAUDE.md) — mixins, utilities
- [settings_/CLAUDE.md](src/phenotypic/settings_/CLAUDE.md) — global config
- [enhance/CLAUDE.md](src/phenotypic/enhance/CLAUDE.md) — enhancer conventions
- [FRONTEND_STYLE_GUIDE.md](DESIGN.md) — dashboard & plot style guide
- `src/phenotypic/post/`, `src/phenotypic/analysis/` — no sub-CLAUDE.md

## Key Files

- `src/phenotypic/_core/_image.py` — `Image` class
- `src/phenotypic/_core/_image_pipeline.py` — Pipeline implementation
- `src/phenotypic/abc_/` — Operation interfaces
- `src/phenotypic/__main__.py` — CLI entry point

## Gotchas

- Some packages excluded on Windows: `rawpy`, `pympler`, `jupyter` — use try/except.
- External tools: ExifTool (raw metadata), Pandoc (doc builds).
- **Operations use `.apply()`, not `__call__`:** `op.apply(image)` is correct;
  `op(image)` raises `TypeError`.
- **Measurement columns are category-prefixed:** `Size_Area`, `Shape_Circularity`,
  `Intensity_MeanIntensity`, etc.
  `MeasurementInfo.get_labels()` returns unprefixed names; `get_headers()` returns the
  prefixed column names used in DataFrames.
- **Analysis classes use `.analyze()`:** `EdgeCorrector.analyze(df)`,
  `LogGrowthModel.analyze(df)` — not `.fit()` or `.correct()`.
- **`num_objects` is on `Image`**, not on the `objmap` accessor: use
  `image.num_objects`.
