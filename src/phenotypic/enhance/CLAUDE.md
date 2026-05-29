# phenotypic.enhance

## Implementation Conventions

- Hessian-based filters default to `black_ridges=False` (bright structures on dark
  background), matching the `detect_mat` convention where colonies appear bright.
- `SetDetectMode` is the only class inheriting `ImageOperation` instead of
  `ImageEnhancer` — it resets `detect_mat` rather than modifying it. Don't use
  it as a reference implementation for new enhancers.
