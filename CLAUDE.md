# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

---

## Quick Start: Common Development Commands

All commands should be run from the repository root using `uv`.

**`uv` is the sole package manager and runner for this project.** Use it for everything:
- **Run commands:** `uv run <cmd>` (never bare `python` or `pip`)
- **Add dependencies:** `uv add <package>` (or `uv add --group dev <package>` for dev deps)
- **Sync environments:** `uv sync` (always run after checkout or in new worktrees)

### Testing

reference @tests/CLAUDE.md for testing details

### Linting and Type Checking

```bash
# Type checking with mypy
uv run mypy src/phenotypic

# Format and lint
uv run ruff check --fix

# Follow Google-style docstrings
```

### Documentation

```bash
# Build Sphinx documentation locally
cd docs
uv run sphinx-build -b html source build
# Then open build/index.html in your browser

# Auto-rebuild on changes (useful for development)
uv run sphinx-autobuild source build
```

### Development Setup

```bash
# Install dev prototying env
uv sync --group dev --group docs --extras gui

# Run a Python script from the venv
uv run python script.py

# Activate the venv manually
source .venv/bin/activate
```

### Command-Line Interface (CLI)

Two CLIs serve different workflows:

- **`python -m phenotypic`** — Run a single pipeline on images or directories. Supports
  local parallel processing (joblib), SLURM cluster submission, resume/restart, and
  multi-layer output (RGB, detection matrices, object masks, overlays, CSV).
- **`python -m phenotypic.sweep`** — Run parameter sweeps: execute multiple pipeline
  configurations against every image in a flat directory. For hyperparameter exploration
  and algorithm comparison. Outputs one subdirectory per pipeline with aggregated CSV.

```bash
# Single pipeline on a directory of images
uv run python -m phenotypic pipeline.json ./images -o ./results \
    --image-type GridImage --nrows 8 --ncols 12 --n-jobs -1

# Parameter sweep across pipeline variants
uv run python -m phenotypic.sweep manifest.json ./images -o ./sweep_results
```

**See [src/phenotypic/_cli/CLAUDE.md](src/phenotypic/_cli/CLAUDE.md) for complete CLI
documentation** including all flags, output structure, and SLURM cluster execution

---

## High-Level Code Architecture

### Project Purpose

PhenoTypic is a modular image processing framework for **arrayed colony phenotyping on
solid media** (agar plates). The framework provides:

- High-level `Image` and `GridImage` classes for easy access to image data and operations
- Extensible operation classes for custom detectors, enhancers, measurers, and correctors
- Pre-built pipelines (`ImagePipeline`) for sequential processing and batch operations
- Grid-aware analysis for plate-based experiments (96-well, 384-well, etc.)

### Core Architecture

The framework has four layers:

1. **Image Data Layer** — `Image` class uses composition with handler classes and exposes
   data through an **accessor pattern** with lazy evaluation and caching (e.g.,
   `image.rgb[:]`, `image.detect_mat[:]`, `image.color.Lab[:]`). See
   [src/phenotypic/_core/CLAUDE.md](src/phenotypic/_core/CLAUDE.md) for details.

2. **Operation Classes (ABC System)** — All algorithms inherit from ABCs that provide a
   consistent `_operate(image) -> image` interface:

```
BaseOperation (root)
├── ImageOperation
│   ├── ImageEnhancer (preprocessing: blur, contrast, etc.)
│   ├── ImageCorrector (quality improvements: rotation, etc.)
│   └── ObjectDetector (colony detection algorithms)
├── GridOperation (grid-aware: GridFinder, GridCorrector, GridObjectRefiner)
├── MeasureFeatures / GridMeasureFeatures (feature extraction)
└── PrefabPipeline (pre-built pipeline templates)
```

   See [src/phenotypic/abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md) for the full
   hierarchy, decision matrix, and implementation guide.

3. **Pipeline Layer** — `ImagePipeline` chains operations with sequential processing,
   batch execution via worker pools, YAML/JSON serialization, and automatic benchmarking.

4. **Enhancement Layer** — 19+ preprocessing operations (denoising, background
   correction, contrast enhancement, morphological operations). All enhancers operate on
   `image.detect_mat[:]`; original RGB and grayscale remain unchanged.

### Module Organization

Each module in `src/phenotypic/` follows a consistent pattern:

```
phenotypic/module_name/
├── __init__.py           # Module docstring + public class exports only
├── _implementation.py    # Private implementation (leading underscore)
└── _another_impl.py      # All implementation files are private
```

Only classes exported in `__init__.py` are public API. All implementation files are
private (leading `_`).

#### Key Modules

- `phenotypic.detect` — Object detection (11+ detectors)
- `phenotypic.enhance` — Image preprocessing (19+ ops)
- `phenotypic.refine` — Post-detection refinement
- `phenotypic.measure` — Feature extraction
- `phenotypic.grid` — Grid detection and alignment
- `phenotypic.correction` — Image quality improvements
- `phenotypic.analysis` — Downstream statistical analysis
- `phenotypic.prefab` — Pre-built pipelines
- `phenotypic.tools_` — Utility mixins and helpers
- `phenotypic.settings_` — Global configuration

### Design Decisions

1. **Immutability:** Operations return modified copies; originals unchanged. Never modify
   `image.rgb` or `image.gray` directly.
2. **Explicit over implicit:** Use `ImagePipeline` for multi-step workflows. No hidden
   state.
3. **Domain-specific:** Purpose-built for microbe phenotyping on agar plates. Examples
   use microbiology context. Code should be intuitive for entry-level data scientists.
4. **Cross-platform:** macOS, Windows, Linux. Watch for platform-specific optional deps.
5. **Reproducibility:** Serialize pipelines via `to_json()` / `from_json()`. Fixed
   random seeds for stochastic operations.
6. **Duck typing:** Follow duck typing principles when reasonable.
7. **Explicit matplotlib:** Never use implicit pyplot.

---

## Code Style and Documentation Standards

### Code Style Rules

- Use **Google-style docstrings** for all documentation
- Use `uv run` to execute Python code or Python-dependent functions
- Activate venv with: `source .venv/bin/activate`
- Follow **duck typing** for type checks
- **Never create separate example files/notebooks** — put all examples in docstrings
- Don't create summary documents unless explicitly asked
- For batch processing, use the CLI: `uv run python -m phenotypic` rather than custom
  scripts
- When modifying settings, import `phenotypic.settings_` before other modules
- Break large functions into smaller helper functions that are independently unit-testable
  and easier to maintain. Within classes, use private helper methods (e.g.,
  `_compute_threshold`) to keep public methods focused and readable.
- **Explicit naming:** Always use descriptive, explicit names for functions, methods, and
  classes. Avoid generic names like `main()`, `run()`, `process()`, or `handle()`. Name
  things after what they actually do (e.g., `sweep_worker_cli()` instead of `main()`,
  `process_plate_image()` instead of `process()`).

### Docstring Format

Use **Google-style docstrings** with this order: (1) one-line summary, (2) Args, (3)
Returns, (4) Raises, (5) longer description with intuition/use cases/limitations, (6)
Examples in doctest format.

```python
def function_name(param):
    """One-line summary.

    Args:
        param: Parameter description.

    Returns:
        Return value description.

    Raises:
        ValueError: When and why.

    Longer explanation with use cases and limitations.

    Examples:
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> result = function_name(image, param=value)
    """
```

All examples must be **fully runnable** doctest format. Use `load_synth_yeast_plate()`
from `phenotypic.data` for image examples. Use real microbiology context — document
parameter effects on colony visibility, edge sharpness, or mask quality.

For **ImageOperation subclasses** (detectors, enhancers, correctors), see
[src/phenotypic/abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md) for the specialized
docstring ordering rules and formatting guidelines.

---

## Detailed Module Guides

- [tests/CLAUDE.md](tests/CLAUDE.md) — Test organization, configuration, writing new tests
- [src/phenotypic/_core/CLAUDE.md](src/phenotypic/_core/CLAUDE.md) — Image class, accessor pattern, color spaces
- [src/phenotypic/abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md) — ABC hierarchy, which ABC to subclass, implementation patterns
- [src/phenotypic/_cli/CLAUDE.md](src/phenotypic/_cli/CLAUDE.md) — CLI flags, output structure, SLURM execution
- [src/phenotypic/tools_/CLAUDE.md](src/phenotypic/tools_/CLAUDE.md) — FootprintMixin, GridInferenceMixin, other utilities
- [src/phenotypic/settings_/CLAUDE.md](src/phenotypic/settings_/CLAUDE.md) — VALIDATE_OPS, MPL defaults, configuration pattern

---

## Key Files

- `src/phenotypic/_core/_image.py` — Main `Image` class, user entry point
- `src/phenotypic/_core/_image_pipeline.py` — Pipeline implementation
- `src/phenotypic/abc_/` — All operation interface definitions
- `src/phenotypic/__main__.py` — CLI entry point (`python -m phenotypic`)

**Reference implementations for new operations:**

- `src/phenotypic/detect/_otsu_detector.py` — Simple detector
- `src/phenotypic/enhance/_gaussian_blur.py` — Simple enhancer
- `src/phenotypic/enhance/_gray_opening.py` — Morphological pattern with FootprintMixin

---

## Extending the Framework

### Creating New Operations

1. Inherit from appropriate ABC in `phenotypic.abc_` (e.g., `ImageEnhancer`,
   `ObjectDetector`)
2. Implement `_operate(self, image: Image) -> Image` as an instance method
3. Access data via accessors: `image.rgb[:]`, `image.detect_mat[:]`, `image.objects`
4. Never modify `image.rgb` or `image.gray` directly (only enhancers modify `detect_mat`)
5. Return modified `Image` instance (immutability principle)
6. Add to module `__init__.py` exports
7. Add tests in `tests/test_*.py`
8. Document with Google-style docstrings including microbiology context

See [src/phenotypic/abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md) for the full
implementation pattern with code examples, the instance method requirement, and
FootprintMixin usage for morphological operations.

---

## Platform-Specific Considerations

Some packages are excluded on Windows: `rawpy`, `pympler`, `jupyter`/`ipykernel`. Use
try/except for platform-specific imports if needed.

**External tools:** ExifTool (raw image metadata, https://exiftool.org/install.html),
Pandoc (documentation builds).

---

## CI/CD

### Documentation Pipeline (documentation.yml)

- **Triggers:** Release published, manual dispatch
- **Builds:** Sphinx documentation with Pandoc for notebook conversion
- **Deploys to:** GitHub Pages (gh-pages branch)

---

## Additional Resources

- **Repository:** https://github.com/exfab/PhenoTypic
- **Documentation:** https://exfab.github.io/PhenoTypic/
- **ExFAB BioFoundry:** https://exfab.engineering.ucsb.edu/
- **Color Science Lib:** https://colour.readthedocs.io/
