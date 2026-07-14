# A09 Pinned Source Probe

From the repository root, run:

```bash
uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/rolling_hough/source_contract_probe.py
```

The harness exits nonzero if the pinned Clark discretization or stable FilFinder v1.8 simple-line
outputs drift. It requires only the project's locked NumPy, SciPy, and Matplotlib environment. It
does not import `phenotypic` and does not require Astropy because FITS I/O is stubbed.

This is a G0 contract probe, not the required all-output golden fixture. The latter remains blocked
until the public A09 result equations are approved.
