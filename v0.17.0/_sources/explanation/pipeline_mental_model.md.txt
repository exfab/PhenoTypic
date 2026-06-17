# The Pipeline Mental Model

PhenoTypic processes plate images through a linear sequence of operations.
Understanding this model is essential for designing effective workflows.

## The Three-Stage Flow

Every analysis follows the same pattern:

1. **Enhance** — Improve `detect_mat` for better colony/background separation
2. **Detect** — Find colonies and write to `objmask` and `objmap`
3. **Measure** — Extract features from detected colonies into a DataFrame

```
Image → [Enhancers] → [Detector] → [Refiners] → [Measurements] → DataFrame
         modify          write        clean up      read
         detect_mat      objmask      objmask       objmap
                         objmap       objmap
```

Each stage reads from specific accessors and writes to others. This
separation means enhancers cannot corrupt detection results, and
measurements always see the final refined mask.

## What Each Stage Touches

| Stage | Reads | Writes | Leaves unchanged |
|-------|-------|--------|-----------------|
| Enhancer | `detect_mat` | `detect_mat` | `rgb`, `gray`, `objmask`, `objmap` |
| Detector | `detect_mat` | `objmask`, `objmap` | `rgb`, `gray`, `detect_mat` |
| Refiner | `objmask`, `objmap` | `objmask`, `objmap` | `rgb`, `gray`, `detect_mat` |
| Corrector | all components | all components | (transforms the entire image) |
| Measurement | `objmap`, `gray`/`rgb` | (none — returns DataFrame) | everything |

This contract is enforced by the abstract base classes. A custom enhancer
that accidentally writes to `objmap` would violate the contract and produce
unpredictable results.

## Immutability and Copies

By default, `pipeline.apply(image)` creates a copy of the input image. The
original is never modified. This makes it safe to apply multiple pipelines
to the same image for comparison.

Pass `inplace=True` to modify the image directly when memory is constrained.

## Serialization

Because each operation stores its configuration as simple parameters, the
entire pipeline can be serialized to JSON and reconstructed exactly. This
is the foundation of PhenoTypic's reproducibility model — the JSON file
captures everything needed to reproduce an analysis.

## References

[1] D. Procida, "Diataxis: A systematic framework for technical
documentation," 2023. [Online]. Available: https://diataxis.fr/
