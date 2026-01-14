# AGENTS.md

This file provides guidance to agentic coding when working with code in
this repository.

---

## Quick Start: Common Development Commands

All commands should be run from the repository root using `uv`.

### Testing

```bash
# Run all tests
uv run pytest

# Run tests in parallel (faster)
uv run pytest -n auto

# Run a single test file
uv run pytest tests/test_image.py

# Run a specific test function
uv run pytest tests/test_image.py::test_image_load

# Run tests with coverage
uv run pytest --cov=src/phenotypic --cov-report=html
```

### Linting and Type Checking

```bash
# Type checking with mypy
uv run mypy src/phenotypic

# Format and lint (manual - project doesn't have auto-formatting configured)
# Follow Google-style docstrings and the rules in .cursor/rules/
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
# Install with dev dependencies
uv sync --group dev

# Install with docs dependencies
uv sync --group docs --group dev

# Install dev + jupyter for prototyping
uv sync --group dev --extras jupyter

# Run a Python script from the venv
uv run python script.py

# Activate the venv manually
source .venv/bin/activate
```

### Command-Line Interface (CLI)

PhenoTypic includes a CLI for batch processing directories of images:

```bash
# Basic usage
uv run python -m phenotypic pipeline.json ./input_images ./output_results

# With options (parallel processing, grid configuration)
uv run python -m phenotypic pipeline.json ./input_images ./output_results \
    --image-type GridImage \
    --nrows 8 \
    --ncols 12 \
    --n-jobs 4

# Process standard images (non-grid)
uv run python -m phenotypic pipeline.json ./input_images ./output_results \
    --image-type Image \
    --n-jobs -1  # Use all available cores

# Save intermediate processing layers
uv run python -m phenotypic pipeline.json ./input_images ./output_results \
    --save-rgb \
    --save-gray \
    --save-objmask \
    --rgb-ext png
```

**CLI Features:**

- Parallel processing with joblib (specify `--n-jobs`)
- Automatic generation of measurement CSVs per image
- Visual QC overlays saved as PNGs
- Master measurements file aggregating all results
- Supports both `Image` and `GridImage` types
- Configurable grid dimensions for plate-based experiments
- Optional intermediate outputs (RGB, grayscale, enhanced grayscale, masks, object maps)

**Output Structure:**

```
output_results/
├── measurements/
│   ├── image1.csv
│   ├── image2.csv
│   └── ...
├── overlays/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── rgb/                     # Optional: created with --save-rgb
│   ├── image1.tiff
│   └── ...
├── gray/                    # Optional: created with --save-gray
│   └── ...
├── enh_gray/                # Optional: created with --save-enh-gray
│   └── ...
├── objmask/                 # Optional: created with --save-objmask
│   └── ...
├── objmap/                  # Optional: created with --save-objmap
│   └── ...
├── objmap_rgb/              # Optional: created with --save-objmap-rgb
│   └── ...
└── master_measurements.csv  # Aggregated results
```

**Breaking Change (v2.0):**

The CLI now uses boolean flags instead of user-specified directories for optional outputs:

- **Old (v1.x)**: `--save-rgb-dir ./custom/path/rgb`
- **New (v2.0)**: `--save-rgb` (auto-creates `OUTPUT_DIR/rgb/`)

This change:
- Simplifies command-line usage
- Enforces consistent directory structure
- Aligns with `measurements/` and `overlays/` pattern

If you need custom paths, use symlinks: `ln -s /custom/path OUTPUT_DIR/rgb`

---

## High-Level Code Architecture

### Project Purpose

PhenoTypic is a modular image processing framework for **arrayed colony phenotyping on
solid media** (agar plates). The
framework provides:

- High-level `Image` and `GridImage` classes for easy access to image data and
  operations
- Extensible operation classes for custom detectors, enhancers, measurers, and
  correctors
- Pre-built pipelines (`ImagePipeline`) for sequential processing and batch operations
- Grid-aware analysis for plate-based experiments (96-well, 384-well, etc.)

### Core Architecture: Layers and Components

#### 1. **Image Data Layer** (src/phenotypic/core/_image_parts/)

The `Image` class is the central data structure. It uses composition with handler
classes:

```
Image (main class)
├── ImageIOHandler (file I/O, metadata, color spaces)
│   ├── ImageDataManager (raw image arrays)
│   └── ImageColorHandler (RGB, grayscale, color spaces: XYZ, Lab, HSV)
├── ImageObjectsHandler (detection results: masks, labels)
└── ImageGridHandler (grid information)
```

The Image class does **not expose raw attributes**. Instead, it uses an **Accessor
Pattern** for data access (see
section 3 below).

#### 2. **Accessor Pattern** (src/phenotypic/core/_image_parts/accessors/)

Accessors provide a unified interface to image components with lazy evaluation, caching,
and transparent format
conversion:

```python
image.rgb[:]  # Raw RGB array
image.gray[:]  # Grayscale (automatic luminance)
image.enh_gray[:]  # Enhanced grayscale for processing
image.objmask[:]  # Binary mask of detected objects
image.objmap[:]  # Labeled object map
image.objects  # ObjectsAccessor (high-level interface)
image.color  # ColorAccessor (XYZ, Lab, HSV, xy chromaticity)
image.grid  # GridAccessor (grid layout and alignment)
image.metadata  # MetadataAccessor (EXIF, file info)
```

**Key principle:** Data is accessed through accessors, not direct attributes. This
ensures consistency and enables lazy
evaluation.

#### 3. **Operation Classes (ABC System)** (src/phenotypic/abc_/)

All algorithms inherit from abstract base classes (ABCs) that provide a consistent
interface:

```
BaseOperation (root class)
├── Automatic memory/performance tracking
├── Logging integration
│
├── ImageOperation (single-image operations)
│   ├── ImageEnhancer (preprocessing: blur, contrast, etc.)
│   ├── ImageCorrector (quality improvements: rotation, etc.)
│   └── ObjectDetector (colony detection algorithms)
│
├── GridOperation (grid-aware operations)
│   ├── GridFinder (automatic grid detection)
│   ├── GridCorrector (grid alignment)
│   └── GridObjectRefiner (mask refinement)
│
├── MeasureFeatures (feature extraction)
│   └── GridMeasureFeatures (per-well measurements)
│
└── PrefabPipeline (pre-built pipeline templates)
```

**Standard interface:** All operations implement `_operate(image) -> image` to enable
chainable processing.

#### 4. **Pipeline Layer** (src/phenotypic/core/_image_pipeline.py)

The `ImagePipeline` class chains operations together:

- **Sequential processing:** Operations execute in order, each passes result to next
- **Lazy execution:** Operations don't execute until needed (results accessed)
- **Batch processing:** Process multiple images with worker pools
- **Serialization:** Save/load pipelines as YAML/JSON for reproducibility
- **Benchmarking:** Automatic timing of each operation
- **Memory profiling:** Track memory usage per operation

### Module Organization

Each module in `src/phenotypic/` follows a consistent pattern:

```
phenotypic/module_name/
├── __init__.py           # Module docstring + public class exports only
├── _implementation.py    # Private implementation (leading underscore)
└── _another_impl.py      # All implementation files are private
```

**Philosophy:**

- **Public API:** Only classes exported in `__init__.py` are considered stable
- **Private Implementation:** Files starting with `_` are internal details
- **Single Responsibility:** Each class does one thing well
- **Standardized Interfaces:** All implementations inherit from appropriate ABC

#### Key Modules

| Module                     | Purpose                             | Key Classes                                                                                                                      |
|----------------------------|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `phenotypic.detect`        | Object detection (11+ detectors)    | `ObjectDetector` (ABC), `OtsuDetector`, `CannyDetector`, etc.                                                                    |
| `phenotypic.enhance`       | Image preprocessing (19+ enhancers) | `ImageEnhancer` (ABC), `GaussianBlur`, `CLAHE`, `MorphologicalOpening`, `BilateralDenoise`, `UnsharpMask`, `BlackTophatEnhancer` |
| `phenotypic.refine`        | Post-detection refinement           | `GridObjectRefiner`, morphology operations, mask editing                                                                         |
| `phenotypic.measure`       | Feature extraction                  | `MeasureFeatures` (ABC), color composition, morphology, etc.                                                                     |
| `phenotypic.grid`          | Grid detection and alignment        | `GridFinder`, `GridCorrector`                                                                                                    |
| `phenotypic.correction`    | Image quality improvements          | `ImageCorrector` (ABC), rotation, edge correction, etc.                                                                          |
| `phenotypic.analysis`      | Downstream statistical analysis     | Growth curves, clustering, outlier detection                                                                                     |
| `phenotypic.prefab`        | Pre-built pipelines                 | Complete workflows at ExFAB                                                                                                      |
| `phenotypic.settings_`     | Global configuration                | `VALIDATE_OPS`, `MPL` (matplotlib defaults)                                                                                      |
| `phenotypic.phenotypicCLI` | Command-line batch processing       | `main()`, `process_single_image()` for parallel pipeline execution                                                               |

### Design Patterns

1. **Strategy Pattern:** ABC classes allow swappable algorithms (e.g., different
   detectors)
2. **Chain of Responsibility:** `ImagePipeline` chains operations
3. **Lazy Evaluation:** Accessors compute data on-demand with caching
4. **Composite Pattern:** `Image` composes multiple handler components
5. **Decorator Pattern:** `ImagePipeline` wraps operations with benchmarking/logging

### Important Design Decisions

1. **Immutability Philosophy:** Operations return modified copies (except direct
   attribute assignment)
    - `enhanced = detector.operate(image)` creates a new Image with detection results
    - Original `image` is unchanged

2. **Explicit Over Implicit:** Pipeline operations are explicit and traceable
    - Use `ImagePipeline` to define multi-step workflows
    - Avoid hidden state or implicit operations

3. **Domain-Specific:** The framework is purpose-built for microbe phenotyping on agar
   plates
    - Examples should use microbiology context (colony growth, well plates)
    - Code should be intuitive to entry-level data scientists

4. **Cross-Platform Support:** Code must work on macOS, Windows, and Linux
    - Watch for platform-specific optional dependencies in pyproject.toml
    - Test code paths on multiple platforms

5. **Reproducibility:** Pipelines can be serialized and re-executed
    - Use `pipeline.to_yaml()` / `pipeline.from_yaml()` for reproducible workflows
    - Fixed random seeds for stochastic operations

---

## Documentation and Code Standards

This section consolidates all code style, design, and documentation rules for the project.

### Design Principles

- Package features should be **intuitive for entry-level data scientists**
- Framework is **extendable** and **standalone** (no external extensions required)
- Examples should have **microbiology context** (arrayed microbe growth, agar plates)
- Follow **duck typing** principles when reasonable
- **Cross-platform support:** macOS, Windows, and Linux (watch for platform-specific optional dependencies in pyproject.toml)
- Use **explicit matplotlib interface** (never use implicit pyplot)

### Code Style Rules

- Use **Google-style docstrings** for all documentation
- Use `uv run` to execute Python code or Python-dependent functions
- Activate venv with: `source .venv/bin/activate`
- Follow **duck typing** for type checks
- **Never create separate example files/notebooks** - put all examples in docstrings
- Don't create summary documents unless explicitly asked
- For batch processing, use the CLI: `uv run python -m phenotypic` rather than custom scripts
- When modifying settings, import `phenotypic.settings_` before other modules

### Docstring Format (All Classes)

Use **Google-style docstrings** with this exact order:

1. **One-line summary** - What does it do?
2. **Args** - Parameters and their effects
3. **Returns** - What is returned
4. **Raises** - Exceptions that can occur
5. **Longer description** - Include intuition, use cases, limitations (especially for ImageOperation subclasses)
6. **Examples** - Doctest-formatted, copy-pasteable code with microbiology context

**Quick template:**

```python
def function_name(param):
    """One-line summary.

    Args:
        param: Parameter description.

    Returns:
        Return value description.

    Raises:
        ValueError: When and why.

    Longer explanation. For operations: why is this useful for colony analysis?
    Include limitations and how parameters affect results.

    Examples:
        >>> from phenotypic.data import load_synth_plate
        >>> image = load_synth_plate()
        >>> result = function_name(image, param=value)
    """
```

### Doctest Format Requirements

- Use **doctest format** for all code examples (lines starting with `>>>` are code)
- Output from code should appear on the next line(s) without prefix
- All examples must be **fully runnable** and **copy-pasteable**
- Use `load_synth_plate()` from `phenotypic.data` for image examples (returns a GridImage with detected colonies)
- Use real microbiology context (colony detection, plate images) - not synthetic/abstract examples
- Document parameter effects on colony visibility, edge sharpness, background suppression, or mask quality

### ImageOperation Subclasses (Special Rules)

For `ObjectDetector`, `ImageEnhancer`, `ImageCorrector`, and related operation classes, use this specific order:

1. **One-line summary** at the top (what does it do?)
2. **Args/Attributes section** - Concise parameter descriptions (1-2 sentences per param) with effects on image processing
3. **Returns section** - What the operation returns
4. **Raises section** - Exceptions that can be raised
5. **Detailed explanation** - Comes AFTER exceptions. Include:
   - **Use cases** (3-5 bullet points): Key scenarios when to use this operation
   - **Limitations** (3-5 bullet points): Critical limitations, failure modes, when NOT to use
   - **Parameter effects** (1-2 sentences per parameter): How tuning impacts results
6. **Examples section** - 2 practical doctest code examples:
   - Basic usage
   - Pipeline/advanced usage

**Documentation format:** Moderate conciseness (100-150 lines total per class)
- Clear and informative without excessive verbosity
- Concise parameter descriptions (1-2 sentences per param)
- Use cases and limitations as bullet-point lists
- Examples use doctest format and are fully runnable
- Use `load_synth_plate()` from `phenotypic.data` when an image is needed
- Reference: [HysteresisDetector](src/phenotypic/detect/_hysteresis_detector.py) in `phenotypic.detect` module as an example

---

## Testing Strategy

### Test Organization

- Test files in `tests/` mirror the module structure under `src/phenotypic/`
- 28 test modules covering all major functionality
- Test resources (sample images, data) in `tests/resources/`

### Key Test Files to Know

- `test_image.py` - Core Image class functionality
- `test_image_pipeline.py` - Pipeline operations and chaining
- `test_image_pipeline_batch.py` - Batch processing
- `test_pipeline_serialization.py` - Pipeline save/load (YAML/JSON)
- `test_accessor_numpy_interface.py` - Accessor NumPy compatibility
- `test_gitter_detector.py` - Detection algorithm tests
- `test_measure_color_composition.py` - Color measurement tests
- `test_morphological_enhancers.py` - Morphological operations (opening, closing, etc.)
- `test_phenotypic_cli.py` - CLI batch processing tests

### Test Configuration

- **Framework:** Pytest with parallel execution support (`pytest-xdist`)
- **Pytest config:** `pyproject.toml [tool.pytest.ini_options]`
- **Conftest:** `conftest.py` enables specific loggers and benchmarking
- **Run in parallel:** `uv run pytest -n auto` (faster for CI)

---

## Global Settings and Configuration

### Settings Module (src/phenotypic/settings_.py)

The `settings_` module provides global configuration for PhenoTypic:

```python
from phenotypic import settings_

# Operation validation
settings_.VALIDATE_OPS = True  # Default: validate operation inputs/outputs

# Matplotlib defaults
settings_.MPL.FIGSIZE = (8, 6)  # Default figure size for plots
```

**Key Configuration Options:**

- **`VALIDATE_OPS`**: When `True` (default), operations validate inputs and outputs to
  ensure data consistency. Set to `False` for performance-critical applications where
  validation overhead is unacceptable.

- **`MPL`**: Container class for matplotlib defaults:
    - `FIGSIZE`: Default figure size for visualizations `(width, height)` in inches

**Usage Pattern:**

```python
# Disable validation for batch processing performance
import phenotypic.settings_ as settings

settings.VALIDATE_OPS = False

# Customize default plot size
settings.MPL.FIGSIZE = (12, 8)

# Then proceed with normal operations
from phenotypic import ImagePipeline
from phenotypic.data import load_synth_plate

image = load_synth_plate()
```

**Important Notes:**

- Settings should be configured **before** importing other PhenoTypic modules
- Changes to settings affect global behavior across all operations
- For reproducibility, document any non-default settings in your analysis scripts

---

## Color Space and Image Data Handling

### Color Space Support

The framework handles sophisticated color operations with lazy evaluation:

- **Native RGB:** Input format (e.g., from JPEG, PNG)
- **Grayscale:** Automatic weighted luminance conversion
- **Color Spaces:**
    - **XYZ / XYZ-D65:** Standard illuminant-based
    - **Lab:** Perceptually uniform color space
    - **HSV:** Hue/Saturation/Value
    - **xy chromaticity:** 2D color without brightness

### Key Implementation Detail: ImageColorHandler

Color conversions are managed by `ImageColorHandler` in
`core/_image_parts/_image_color_handler.py`. It handles:

- Gamma correction (sRGB support with linear fallback)
- Illuminant specifications (D65 standard, D50 for imaging)
- CIE Observer (1931 2° Standard Observer)
- Lazy evaluation with caching

**Example:** `image.color.Lab[:]` triggers XYZ → Lab conversion only when accessed,
results are cached.

---

## Image Enhancement Operations

The `phenotypic.enhance` module provides 19+ preprocessing operations: denoising
(GaussianBlur, MedianFilter, BilateralDenoise), background correction (RollingBallRemoveBG),
contrast enhancement (CLAHE, UnsharpMask), morphological operations (MorphologicalOpening,
MorphologicalClosing, etc.), and edge detection (SobelFilter).

**Key Principles:**

- All enhancers operate on `image.enh_gray[:]` (enhanced grayscale)
- Original RGB and grayscale remain unchanged (immutability)
- Chain multiple enhancers in `ImagePipeline` for preprocessing

**Example:**

```python
from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur, CLAHE, MorphologicalOpening
from phenotypic.data import load_synth_plate

image = load_synth_plate()
pipeline = ImagePipeline([
    GaussianBlur(sigma=1.5),
    CLAHE(clip_limit=2.0),
    MorphologicalOpening(shape='disk', radius=2)
])
result = pipeline.operate(image)
```

See `phenotypic.enhance` module docstrings for parameter tuning guidance and use cases.

---

## Performance and Memory Optimization

### Built-In Monitoring

All `BaseOperation` subclasses automatically track:

- **Memory usage:** Via `tracemalloc` (Python built-in)
- **Execution time:** Automatic benchmarking per operation
- **Object memory:** Via `pympler` (detailed object analysis)
- **Process memory:** Via `psutil` (system-wide memory)

### Memory Tracking in Pipelines

When running `ImagePipeline`:

```python
results = pipeline.operate(images)
# Memory and timing automatically logged per operation
```

This allows profiling bottlenecks without explicit instrumentation.

---

## GitHub Actions and CI/CD

### Testing Pipeline (run-pytest.yml)

- **Triggers:** PRs to main/dev branches, manual dispatch
- **Matrix Testing:**
    - **Linux:** Python 3.10, 3.11, 3.12
    - **Windows:** Python 3.12 only
    - **macOS:** Python 3.12 only
- **Key Steps:**
    1. Install exiftool on Linux (for metadata extraction)
    2. Install Python and uv
    3. Install dependencies with `uv`
    4. Run tests: `pytest -n auto` (parallel)
    5. Build documentation with Pandoc support

### Documentation Pipeline (documentation.yml)

- **Triggers:** Release published, manual dispatch
- **Builds:** Sphinx documentation with Pandoc for notebook conversion
- **Deploys to:** GitHub Pages (gh-pages branch)

---

## Key Files to Know

### Core Image Class

- `src/phenotypic/core/_image.py` - Main `Image` class, entry point for users
- `src/phenotypic/core/_image_parts/` - Handler classes and accessors
- `src/phenotypic/core/_image_pipeline.py` - Pipeline implementation

### Abstract Base Classes

- `src/phenotypic/abc_/` - All operation interface definitions
- Study these to understand how to extend the framework

### Example Implementations

- `src/phenotypic/detect/_otsu_detector.py` - Simple detector example
- `src/phenotypic/enhance/_gaussian_blur.py` - Simple enhancer example
- `src/phenotypic/enhance/_morphological_opening.py` - Morphological operation example
- Study these to understand the pattern for new operations

### Command-Line Interface and Configuration

- `src/phenotypic/phenotypicCLI.py` - CLI for batch processing pipelines
- `src/phenotypic/__main__.py` - Module entry point (`python -m phenotypic`)
- `src/phenotypic/settings_.py` - Global configuration (validation, matplotlib defaults)

### Documentation

- `docs/source/` - Sphinx configuration and custom templates
- `docs/source/examples/` - Example notebooks and scripts
- Auto-generated from docstrings via Sphinx Gallery

---

## Working with the `Image` Class

### Basic Usage Pattern

```python
from phenotypic import Image
from phenotypic.detect import OtsuDetector
from phenotypic.data import load_synth_plate

# Load image
image = load_synth_plate()

# Apply operations
detector = OtsuDetector()
image_with_detection = detector.operate(image)

# Access data through accessors
colony_mask = image_with_detection.objmask[:]
detected_colonies = image_with_detection.objects
```

### Important: Never Modify image.rgb or image.gray Data Directly

```python
# ✗ DON'T do this:
image.rgb[:] = modified_data  # Violates consistency

# ✓ DO this:
image.color_handler.set_rgb(modified_data)  # Use handler methods if modification needed
# Or better yet, use operations that return new Image instances
```

### Accessing Different Image Representations

```python
# All return numpy arrays
rgb_data = image.rgb[:]  # Original color data
gray_data = image.gray[:]  # Luminance grayscale
enhanced = image.enh_gray[:]  # Enhanced for processing

# Color spaces (lazy-evaluated, cached)
lab_data = image.color.Lab[:]
hsv_data = image.color.HSV[:]
xyz_data = image.color.XYZ[:]

# Grid and detection data
if image.grid is not None:
    rows, cols = image.grid.shape

if image.objects is not None:
    for obj in image.objects:
        area = obj.area
```

---

## Platform-Specific Considerations

### Optional Dependencies

Some packages are excluded on Windows:

```python
rawpy  # Raw image support (not Windows)
pympler  # Memory profiling (not Windows)
jupyter, ipykernel  # Development (not Windows)
```

When writing code:

- Don't assume `rawpy` is available on Windows
- Use try/except for platform-specific imports if needed
- Test cross-platform code paths

### External Tools

- **ExifTool:** Required for extracting metadata from raw images. Install
  from: https://exiftool.org/install.html
- **Pandoc:** Required for documentation builds. Automatically installed in CI, install
  locally if building docs.

---

## Batch Processing with the CLI

**1. Design and test pipeline interactively:**

```python
from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur, CLAHE
from phenotypic.detect import OtsuDetector
from phenotypic.data import load_synth_plate

image = load_synth_plate()
pipeline = ImagePipeline([GaussianBlur(sigma=1.5), CLAHE(clip_limit=2.0), OtsuDetector()])
result = pipeline.apply_and_measure(image)
pipeline.to_json("my_pipeline.json")  # Save for batch processing
```

**2. Run batch processing:**

```bash
uv run python -m phenotypic my_pipeline.json ./raw_plates ./results \
    --image-type GridImage --nrows 8 --ncols 12 --n-jobs -1
```

**Best Practices:**

- Test on diverse sample images before batch processing
- Always review overlay images in `results/overlays/` for visual QC
- Disable validation for large batches: set `settings_.VALIDATE_OPS = False` before importing operations
- Use `--n-jobs -1` to utilize all CPU cores
- Version control your pipeline JSON files with image acquisition settings documented

---

## Extending the Framework

### Creating New Operations

**Pattern for any operation (detector, enhancer, measurer):**

1. Inherit from appropriate ABC in `phenotypic.abc_` (e.g., `ImageEnhancer`, `ObjectDetector`)
2. Implement `_operate(self, image: Image) -> Image`
3. Access data via accessors: `image.rgb[:]`, `image.enh_gray[:]`, `image.objects`, etc.
4. Never modify `image.rgb` or `image.gray` directly (only enhancers work on `enh_gray`)
5. Return modified `Image` instance (immutability principle)
6. Add to module `__init__.py` exports
7. Add tests in `tests/test_*.py`
8. Document with Google-style docstrings including microbiology context

**Reference implementations:**

- `src/phenotypic/detect/_otsu_detector.py` - Simple detector
- `src/phenotypic/enhance/_gaussian_blur.py` - Simple enhancer
- `src/phenotypic/enhance/_morphological_opening.py` - Morphological pattern

### Key Principles

- Leverage automatic memory/performance tracking from `BaseOperation`
- Use duck typing for type checks
- Follow the accessor pattern for consistent data access
- Document parameter effects on colony visibility, edge sharpness, and mask quality (for operations)

---

## Additional Resources

- **Repository:** https://github.com/exfab/PhenoTypic
- **Documentation:** https://exfab.github.io/PhenoTypic/
- **ExFAB BioFoundry:** https://exfab.engineering.ucsb.edu/
- **Color Science Lib:** https://colour.readthedocs.io/
