# A09 candidate-core drift register

These rows apply to the private `clark_rolling_hough` implementation authorized by
`SOURCE_CONTRACT.md`. Shared detector integration, wrappers, and public exports do not exist yet.

| ID | Candidate deviation | Reason and consequence | Evidence required |
|---|---|---|---|
| D01 | Accept only nonempty two-dimensional NumPy `float64` arrays; reject every other dtype and container with explicit Python exceptions. | The source relies on input dtype during SciPy correlation, so implicit integer/float32 conversion would create an unreviewed numerical extension. Nonfinite float64 values retain source masking behavior. | Invalid-input tests covering integer, Boolean, float32, complex, list, empty, and non-2-D inputs |
| D02 | Use named `smoothing_radius`, not the paper's diameter or the current plan's `smoothing_diameter`. | This removes an unresolved conversion and matches executable `smr` exactly. | Parameter forwarding test |
| D03 | Expose dense raw-count and residual cubes with zeros outside the source rolling-window mask. | The source persists only positive sparse residuals, but raw counts are required for auditability. `eligible` prevents adapter zeros from being mistaken for evaluated values. | Sparse-to-dense fixture comparison |
| D04 | Freeze source platform integers to int64 and retain in-memory float64 instead of persistence float32. | Cross-platform output dtype becomes stable without changing integer values or in-memory arithmetic. | Dtype tests and exact fixture counts |
| D05 | Return NaN orientation where no positive residual exists. | The source emits no sparse row there; directly calling its angle helper on zero weights returns pi. NaN distinguishes invalid from a valid axial zero/pi normal. | Zero-response and invalid-sentinel tests |
| D06 | Return a deterministic empty numerical result instead of reproducing source persistence `IndexError`. | The source's empty `Hthets` has shape `(0,)`, then indexes `shape[1]`. The candidate retains all defined zero products and reports false validity. | Constant fixture error plus safe-empty behavioral test |
| D07 | Omit global backprojection normalization from the core response. | Source `backproj` is locally accumulated before persistence, then globally normalized. The raw sum is stable, local, and avoids all-zero division; later normalization requires a separately approved wrapper boundary. | Raw response equals residual sum exactly |
| D08 | Use the nondeprecated `scipy.ndimage.correlate` entry point while retaining default reflect semantics. | The pinned spelling `scipy.ndimage.filters.correlate` is deprecated but aliases the same operation in the locked environment. | Source fixture convolution comparison |
| D09 | Convert the executable's sparse positive-residual emission rule to a dense Boolean `valid` array. | The source has no dense validity product. The adapter representation is exactly `np.any(threshold_residual > 0, axis=2)` and must not use raw-count truthiness or integer labels. | Exact fixture validity reconstruction, Boolean dtype assertion, and validity mutant |
