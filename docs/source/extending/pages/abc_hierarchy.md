# ABC Hierarchy Deep-Dive

PhenoTypic's operation system is built on a hierarchy of abstract base
classes (ABCs). Understanding this hierarchy is essential for writing
correct custom operations.

## The Hierarchy

```
ImageOperation (root ABC)
├── ImageEnhancer          — modifies detect_mat
├── ImageCorrector         — transforms the entire image
├── ObjectDetector         — writes to objmask and objmap
│   └── ThresholdDetector  — convenience base for threshold methods
│   └── GridObjectDetector — grid-aware detection
├── ObjectRefiner          — cleans up objmask and objmap
├── MeasureFeatures        — extracts features → DataFrame
│   └── GridMeasureFeatures — grid-aware measurements
└── PostMeasurement        — transforms measurement DataFrames
```

## What Each ABC Enforces

| ABC | Must implement | Reads | Writes |
|-----|---------------|-------|--------|
| `ImageEnhancer` | `_operate(image)` | `detect_mat` | `detect_mat` |
| `ImageCorrector` | `_operate(image)` | all | all |
| `ObjectDetector` | `_operate(image)` | `detect_mat` | `objmask`, `objmap` |
| `ObjectRefiner` | `_operate(image)` | `objmask`, `objmap` | `objmask`, `objmap` |
| `MeasureFeatures` | `_operate(image)` | `objmap`, image data | (returns DataFrame) |

## The `_operate()` Contract

Every operation implements a single method:

```python
def _operate(self, image: Image) -> Image:
```

- **Input:** An `Image` or `GridImage` instance
- **Output:** The same image (possibly modified in place)
- **Parameters:** Accessed via `self` (set in `__init__`)

The base class handles:
- Copying (when `inplace=False`)
- GridImage preservation (grid state, nrows, ncols)
- Serialization (parameters → JSON)
- Benchmarking

## Serialization

Operations are serializable to JSON by default. The serializer captures
all `__init__` parameters that are not prefixed with `_`. Ensure your
parameters are JSON-compatible types (int, float, str, list, dict, bool,
None).

## Registration

Custom operations are automatically discoverable by `ImagePipeline`
when they are importable. The pipeline's `from_json()` method resolves
class names using Python's import system.
