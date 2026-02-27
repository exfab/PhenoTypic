# CLAUDE.md

## Quick Start

**`uv` is the sole package manager and runner.** Never use bare `python` or `pip`.

- `uv run <cmd>` — run commands
- `uv add <package>` (or `--group dev`) — add dependencies
- `uv sync` — sync env (after checkout or in new worktrees)
- `uv sync --group dev --group docs --extras gui` — full dev env
- `source .venv/bin/activate` — manual venv activation

### Testing

See [tests/CLAUDE.md](tests/CLAUDE.md)

### Linting & Type Checking

- `uv run mypy src/phenotypic` — type checking
- `uv run ruff check --fix` — format and lint

### Documentation

- `cd docs && uv run sphinx-build -b html source build` — build docs
- `uv run sphinx-autobuild source build` — auto-rebuild on changes

### CLI

- **`python -m phenotypic`** — single pipeline on images/directories (parallel, SLURM, resume)
- **`python -m phenotypic.sweep`** — parameter sweeps across pipeline variants

```bash
uv run python -m phenotypic pipeline.json ./images -o ./results \
    --image-type GridImage --nrows 8 --ncols 12 --n-jobs -1

uv run python -m phenotypic.sweep manifest.json ./images -o ./sweep_results
```

See [src/phenotypic/_cli/CLAUDE.md](src/phenotypic/_cli/CLAUDE.md) for full CLI docs.

---

## Architecture

**Purpose:** Modular image processing for arrayed colony phenotyping on solid media (agar plates).

### Four Layers

1. **Image Data Layer** — `Image`/`GridImage` with accessor pattern, lazy evaluation,
   caching (`image.rgb[:]`, `image.detect_mat[:]`, `image.color.Lab[:]`).
   See [_core/CLAUDE.md](src/phenotypic/_core/CLAUDE.md).

2. **Operation ABCs** — `_operate(image) -> image` interface:

```
BaseOperation
├── ImageOperation → ImageEnhancer, ImageCorrector, ObjectDetector
├── GridOperation → GridFinder, GridCorrector, GridObjectRefiner
├── MeasureFeatures / GridMeasureFeatures
└── PrefabPipeline
```

   See [abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md).

3. **Pipeline Layer** — `ImagePipeline` chains operations, batch execution, YAML/JSON
   serialization, automatic benchmarking.

4. **Enhancement Layer** — 19+ preprocessing ops on `detect_mat`; RGB/gray unchanged.

### Module Organization

```
phenotypic/module_name/
├── __init__.py          # Public exports only
├── _implementation.py   # Private (leading underscore)
```

Only `__init__.py` exports are public API.

### Key Modules

- `detect` — 11+ detectors | `enhance` — 19+ preprocessors | `refine` — post-detection
- `measure` — feature extraction | `grid` — grid detection/alignment
- `correction` — image quality | `analysis` — statistics | `prefab` — pre-built pipelines
- `tools_` — mixins/helpers | `settings_` — global config

### Design Decisions

- **Immutability:** Operations return copies; never modify `image.rgb`/`image.gray` directly
- **Explicit:** Use `ImagePipeline` for multi-step workflows; no hidden state
- **Domain-specific:** Built for microbe phenotyping; use microbiology context in docs/examples
- **Duck typing** for type checks; **explicit matplotlib** (no implicit pyplot)
- **Reproducibility:** `to_json()`/`from_json()` serialization; fixed random seeds
- **Cross-platform:** macOS, Windows, Linux; use try/except for platform-specific imports

---

## Code Style

- **Google-style docstrings** everywhere
- **Never create** separate example files/notebooks — examples go in docstrings
- Don't create summary documents unless explicitly asked
- **Explicit naming:** No generic `main()`, `run()`, `process()` — name after what it does
- Break large functions into smaller, testable helpers with private methods
- For batch processing, use the CLI (`python -m phenotypic`) not custom scripts
- Import `phenotypic.settings_` before other modules when modifying settings

### Docstring Order

(1) One-line summary → (2) Args → (3) Returns → (4) Raises → (5) Longer description → (6) Examples (doctest format)

- All examples must be **runnable** doctest using `load_synth_yeast_plate()`
- Use microbiology context (colony visibility, edge sharpness, mask quality)
- For ImageOperation subclass docstrings, see [abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md)

---

## Module Guides

- [tests/CLAUDE.md](tests/CLAUDE.md) — testing
- [src/phenotypic/_core/CLAUDE.md](src/phenotypic/_core/CLAUDE.md) — Image class, accessors
- [src/phenotypic/abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md) — ABC hierarchy, implementation
- [src/phenotypic/_cli/CLAUDE.md](src/phenotypic/_cli/CLAUDE.md) — CLI, SLURM
- [src/phenotypic/tools_/CLAUDE.md](src/phenotypic/tools_/CLAUDE.md) — mixins, utilities
- [src/phenotypic/settings_/CLAUDE.md](src/phenotypic/settings_/CLAUDE.md) — global config

## Key Files

- `src/phenotypic/_core/_image.py` — Main `Image` class
- `src/phenotypic/_core/_image_pipeline.py` — Pipeline implementation
- `src/phenotypic/abc_/` — Operation interfaces
- `src/phenotypic/__main__.py` — CLI entry point

**Reference implementations:** `detect/_otsu_detector.py` (detector),
`enhance/_gaussian_blur.py` (enhancer), `enhance/_gray_opening.py` (FootprintMixin)

## Gotchas

- Some packages excluded on Windows: `rawpy`, `pympler`, `jupyter` — use try/except
- External tools: ExifTool (raw metadata), Pandoc (doc builds)
- CI docs: Sphinx → GitHub Pages on release publish or manual dispatch

## Links

- https://github.com/exfab/PhenoTypic
- https://exfab.github.io/PhenoTypic/
- https://exfab.engineering.ucsb.edu/
- https://colour.readthedocs.io/
