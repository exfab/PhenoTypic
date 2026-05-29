# Abstract Base Classes

## Hierarchy

```
BaseOperation
├── ImageOperation
│   ├── ImageEnhancer (+FootprintMixin)  # Modifies only detect_mat
│   ├── ImageCorrector                   # Transforms entire image
│   │   └── GridCorrector (+GridOperation)
│   ├── ObjectDetector                   # Produces objmask/objmap
│   │   ├── ThresholdDetector            # Threshold-based
│   │   └── GridObjectDetector (+GridOperation)
│   ├── ObjectRefiner                    # Modifies objmask/objmap
│   │   └── GridObjectRefiner (+GridOperation)
│   └── GridOperation                    # Marker mixin for GridImage input
├── MeasureFeatures                      # Extracts → DataFrame
│   └── GridMeasureFeatures
│       └── GridFinder                   # Detects grid structure

Standalone: MeasurementInfo (enum base, now in `phenotypic.schema`; re-exported here for back-compat), PrefabPipeline (inherits ImagePipeline)
```

---

## Which ABC to Subclass

| Goal | Subclass | Modifies |
|------|----------|----------|
| Preprocess (blur, contrast) | `ImageEnhancer` | `detect_mat` only |
| Detect colonies | `ObjectDetector` | `objmask`, `objmap` |
| Threshold detection | `ThresholdDetector` | `objmask`, `objmap` |
| Clean detection results | `ObjectRefiner` | `objmask`, `objmap` |
| Rotate/crop/transform | `ImageCorrector` | All components |
| Grid-aware transform | `GridCorrector` | All + grid state |
| Grid-aware refinement | `GridObjectRefiner` | `objmask`, `objmap` |
| Extract measurements | `MeasureFeatures` | Returns DataFrame |
| Grid measurements | `GridMeasureFeatures` | Returns DataFrame |
| Detect grid structure | `GridFinder` | Grid metadata |
| Pre-built workflow | `PrefabPipeline` | Complete pipeline |

---

## Implementation Rules

- **`_operate()` must be an instance method** (not static); access params via `self`. Static `_operate(image, **params)` is deprecated.
  Canonical: [`enhance/_gaussian_blur.py`](../enhance/_gaussian_blur.py).
- **Operations must be instantiable with no required args** for `from_json()` to work. Operations that need mandatory args (e.g., `ColorCorrector` needs a fitted profile) cannot be generically deserialized and must be excluded from round-trip tests.
- **Tuple attributes survive JSON round-trip** only if you coerce them back: JSON has no tuple type, so tuples become lists on `from_json()`. Add a `__setattr__` that re-coerces lists back to tuples whether set in `__init__` or via `setattr()`.
  See [`detect/_manual_grid_point_detector.py`](../detect/_manual_grid_point_detector.py), `enhance/_frangi_vesselness.py`, `enhance/_hessian_filter.py` for the pattern.
- **Optional `inspect()` on `MeasureFeatures`** — implementing `def inspect(self, image=None, *, for_save=False, **kwargs)` returning an mpl or plotly Figure opts a subclass into the CLI's `--save-inspect` auto-discovery (saves a PNG per image under `results/<ds>/inspect/<step>/<stem>.png`). Contract details in the `MeasureFeatures` class docstring; reference impl in [`measure/_measure_symmetric_zones.py`](../measure/_measure_symmetric_zones.py).

---

## FootprintMixin

`ImageEnhancer` already inherits from `FootprintMixin` — just call `self._make_footprint(shape, width)`.
For other ABCs (e.g., `ObjectRefiner`), add it explicitly: `class MyOp(FootprintMixin, ObjectRefiner)`.
See [tools_/CLAUDE.md](../tools_/CLAUDE.md) for the full mixin reference.

---

## Integrity Validation

`@validate_operation_integrity` protects components (only when `settings.VALIDATE_OPS = True`):

| ABC | Protected |
|-----|-----------|
| `ImageEnhancer` | `rgb`, `gray` (only `detect_mat` modified) |
| `ObjectDetector` | `rgb`, `gray`, `detect_mat` |
| `ObjectRefiner` | `rgb`, `gray`, `detect_mat` |
| `ImageCorrector` | None (transforms all) |
| `MeasureFeatures` | All (read-only) |

---

## Docstring Pattern

All ABC class docstrings include: (1) one-line summary → (2) Quick Decision Guide (this ABC vs alternatives, 8–15 bullets) → (3) context blocks (purpose, pipeline role, when to use) → (4) implementation guide with code template → (5) known implementations → (6) formal API → (7) two doctest examples (basic + advanced).

ImageOperation **subclass** docstrings follow: (1) one-line summary → (2) Args/Attributes → (3) Returns → (4) Raises → (5) detailed explanation (use cases, limitations, parameter effects) → (6) two doctest examples.

- Doctest format (`>>>`); use `load_synth_yeast_plate()` for image examples.
- Target 100–150 lines per ImageOperation subclass docstring.
- Canonical: [`detect/_hysteresis_detector.py`](../detect/_hysteresis_detector.py).

---

## Best Examples

| ABC | Example | Location |
|-----|---------|----------|
| `ImageEnhancer` | GaussianBlur | `enhance/_gaussian_blur.py` |
| `ImageEnhancer` + FootprintMixin | GrayOpening | `enhance/_gray_opening.py` |
| `ObjectDetector` | OtsuDetector | `detect/_otsu_detector.py` |
| `ThresholdDetector` | HysteresisDetector | `detect/_hysteresis_detector.py` |
| `ObjectRefiner` | SmallObjectRemover | `refine/_small_object_remover.py` |
| `ObjectRefiner` + FootprintMixin | MaskDilator | `refine/_mask_dilation.py` |
| `ImageCorrector` | GridAligner | `correction/_grid_aligner.py` |
| `GridFinder` | AutoGridFinder | `grid/_auto_grid_finder.py` |
| `MeasureFeatures` | MeasureSize | `measure/_measure_size.py` |
| `PrefabPipeline` | HeavyOtsuPipeline | `prefab/_heavy_otsu_pipeline.py` |
