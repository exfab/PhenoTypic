# Reconnection Numerical Helpers

This package contains pure array and graph algorithms used to reconnect filamentary
image structure. It must remain import-cheap and domain-independent.

## Contracts

- Accept arrays and scalar parameters, never `Image`, pydantic operations, or GUI types.
- Validate public Python entry points before entering compiled kernels.
- Do not mutate caller-owned arrays. Document every output shape, dtype, axis, unit,
  polarity, boundary rule, sentinel, and tie-break.
- Use `(row, column)` coordinates and axial angles in radians unless an algorithm's
  frozen contract explicitly says otherwise.
- Keep optional dependencies lazy. Importing this package must not add GUDHI,
  FilFinder, FilterPy, Astropy, plotting, or UI modules beyond the modules already
  loaded by the existing eager `phenotypic.sdk_` parent-package import.
- Do not use Numba `fastmath` without a committed numerical proof. After changing a
  cached Numba kernel, clear only that module's affected `__pycache__` directory.
- Keep domain-specific cost construction, rasterization, image-layer access, and
  operation normalization in the detector or operation adapter.

## Reference ports

Before implementing a port, assemble the complete reference corpus locally under the
matching specification's `refs/<method>/` directory. Cite `file:line` in the
reconciliation, compare production code line by line, and record one drift row per
deviation.

Every numerical helper requires all of the following:

1. A source-generated golden fixture containing every public result field.
2. Behavioral controls independent of the source fixture.
3. A standalone validation script using only the standard library, NumPy, and SciPy.
4. A derived tolerance mechanism rather than an unexplained fixed tolerance.
5. A mutant-by-test matrix proving each plausible single-change defect is detected.
6. Independent source-fidelity review before this package re-exports the helper.
