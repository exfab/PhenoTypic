# Abstract Base Classes

## Hierarchy

```
BaseOperation
├── ImageOperation
│   ├── ImageEnhancer (+FootprintMixin)  # Modifies only detect_mat
│   │   ├── ImageDenoiser                # Noise-estimate-driven restoration
│   │   ├── FocusEdge                # Output is an edge/ridge response map
│   │   ├── FocusBlob                # Output is a blob/scale-space response
│   │   ├── Smoothing                    # Kernel/diffusion blur
│   │   ├── BackgroundSubtraction        # Removes slow-varying background
│   │   ├── MorphologicalFiltering       # Structuring-element small-feature ops
│   │   └── ContrastAdjustment           # Intensity/contrast remapping
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

Standalone: MeasurementInfo (enum base, now in `phenotypic.schema`; re-exported here for back-compat), PrefabPipeline (inherits ImagePipeline), and plotting extension contracts under `phenotypic.abc_.plotting` (`PhtPlot`, `PlotImage`, `PlotMeas`, `PlotAnalysis`, `PlotQc`, `PlotOutput`, `PlotPage`)
```

---

## Which ABC to Subclass

| Goal                    | Subclass                                 | Modifies            |
|-------------------------|------------------------------------------|---------------------|
| Preprocess `detect_mat` | an `ImageEnhancer` purpose-group (below) | `detect_mat` only   |
| Detect colonies         | `ObjectDetector`                         | `objmask`, `objmap` |
| Threshold detection     | `ThresholdDetector`                      | `objmask`, `objmap` |
| Clean detection results | `ObjectRefiner`                          | `objmask`, `objmap` |
| Rotate/crop/transform   | `ImageCorrector`                         | All components      |
| Grid-aware transform    | `GridCorrector`                          | All + grid state    |
| Grid-aware refinement   | `GridObjectRefiner`                      | `objmask`, `objmap` |
| Extract measurements    | `MeasureFeatures`                        | Returns DataFrame   |
| Grid measurements       | `GridMeasureFeatures`                    | Returns DataFrame   |
| Detect grid structure   | `GridFinder`                             | Grid metadata       |
| Pre-built workflow      | `PrefabPipeline`                         | Complete pipeline   |

### Enhancer purpose-groups

Concrete enhancers subclass one of the marker ABCs in the hierarchy above
(`ImageDenoiser`, `FocusEdge`, `FocusBlob`, `Smoothing`, `BackgroundSubtraction`,
`MorphologicalFiltering`, `ContrastAdjustment`) instead of `ImageEnhancer` directly
— all subclass `ImageEnhancer`, add no methods/params, and are picked by what
`_operate` produces. They are intentionally **not** re-exported from
`phenotypic.enhance` (keeps them out of the GUI builder dropdown). The per-group
example classes + the pinning test (`tests/unit/abc_/test_enhancer_taxonomy.py`)
live in [`enhance/CLAUDE.md`](../enhance/CLAUDE.md).

---

## Implementation Rules

- **Operations are pydantic v2 models** rooted at `BaseOperation`: keyword-only
  construction, class-level annotated fields, **no `__init__`**, and `.apply()`
  (not `__call__`). Normalize inputs in a `field_validator`. For parameter,
  closed-value-set, and tune-annotation conventions, use the `adding-an-operation`
  skill.
- **`_operate()` must be an instance method** (not static); access params via `self`.
  Static `_operate(image, **params)` is deprecated.
  Canonical: [`enhance/_blur_gauss.py`](../enhance/_blur_gauss.py).
- **Operations must be instantiable with no required args** for `from_json()` to work.
  Operations that need mandatory args (e.g., `ColorCorrector` needs a fitted profile)
  cannot be generically deserialized and must be excluded from round-trip tests.
- **`ImageOperation` image caches must use weak references only.** If an operation
  needs to remember which `Image` populated a reusable cache, store a
  `weakref.ref[Image]`, never the `Image` itself or one of its accessors. Treat a
  released weak reference as a cache miss and recompute. Compact derived arrays may
  be cached only when they own their buffers and cannot pin a whole-image allocation.
- **Tuple attributes survive JSON round-trip** only if you coerce them back: JSON has no
  tuple type, so tuples become lists on `from_json()`. Add a
  `field_validator(..., mode="before")` that coerces the incoming list back to a tuple.
  See [`enhance/_focus_edge_frangi.py`](../enhance/_focus_edge_frangi.py) (`_coerce_sigmas`)
  and `enhance/_focus_edge_hessian.py` for the pattern.
- **Plotting is an explicit lifecycle capability.** Combine an existing Pydantic
  operation or analyzer with exactly one of `PlotImage`, `PlotMeas`,
  `PlotAnalysis`, or `PlotQc`, then configure that same object under
  `ImagePipeline.plots`. `inspect(subject, *, for_save=False)` returns the primary
  saveable figure; `report(subject)` returns the complete notebook report.
  `PlotImage` receives an `Image` at call time and must never cache it strongly.
  Compact derived measurements may be cached, but cached NumPy crops must own
  their buffers rather than retain a whole-image backing array. CLI output is
  published under `deliverables/plots/<binding-id>/`. Reference implementation:
  [`measure/_measure_symmetric_zones.py`](../measure/_measure_symmetric_zones.py).

---

## FootprintMixin

`ImageEnhancer` already inherits from `FootprintMixin` — just call
`self._make_footprint(shape, width)`.
For other ABCs (e.g., `ObjectRefiner`), add it explicitly:
`class MyOp(FootprintMixin, ObjectRefiner)`.
See [sdk_/CLAUDE.md](../sdk_/CLAUDE.md) for the full mixin reference (`FootprintMixin`
lives in `sdk_/mixin/_footprint_mixin.py`).

---

## Integrity Validation

`@validate_operation_integrity` protects components (only when
`phenotypic.settings.VALIDATE_OPS = True`):

| ABC               | Protected                                  |
|-------------------|--------------------------------------------|
| `ImageEnhancer`   | `rgb`, `gray` (only `detect_mat` modified) |
| `ObjectDetector`  | `rgb`, `gray`, `detect_mat`                |
| `ObjectRefiner`   | `rgb`, `gray`, `detect_mat`                |
| `ImageCorrector`  | None (transforms all)                      |
| `MeasureFeatures` | All (read-only)                            |

---

## Docstring Pattern

ImageOperation and ABC docstrings follow the canonical **layered
progressive-disclosure** template — see
[`docs/source/contrib_guide/docstring_style.md`](../../../docs/source/contrib_guide/docstring_style.md)
(single source of truth; the `pht-docwriter` agent automates it). ABC base classes
additionally lead with a Quick Decision Guide (this ABC vs alternatives) plus a
code template before the formal API. Canonical subclass example:
[`detect/_hysteresis_detector.py`](../detect/_hysteresis_detector.py).

---

## Best Examples

| ABC                              | Example            | Location                          |
|----------------------------------|--------------------|-----------------------------------|
| `ImageEnhancer`                  | BlurGauss       | `enhance/_blur_gauss.py`       |
| `ImageEnhancer` + FootprintMixin | GrayOpening        | `enhance/_gray_opening.py`        |
| `ObjectDetector`                 | OtsuDetector       | `detect/_otsu_detector.py`        |
| `ThresholdDetector`              | HysteresisDetector | `detect/_hysteresis_detector.py`  |
| `ObjectRefiner`                  | SmallObjectRemover | `refine/_small_object_remover.py` |
| `ObjectRefiner` + FootprintMixin | MaskDilation       | `refine/_mask_dilation.py`        |
| `ImageCorrector`                 | GridAligner        | `correction/_grid_aligner.py`     |
| `GridFinder`                     | AutoGridFinder     | `grid/_auto_grid_finder.py`       |
| `MeasureFeatures`                | MeasureSize        | `measure/_measure_size.py`        |
| `PrefabPipeline`                 | HeavyOtsuPipeline  | `prefab/_heavy_otsu_pipeline.py`  |
