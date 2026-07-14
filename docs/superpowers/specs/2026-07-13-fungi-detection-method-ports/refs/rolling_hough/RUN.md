# A09 Pinned Source Probe

From the repository root, run:

```bash
uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/rolling_hough/source_contract_probe.py
```

The harness exits nonzero if the pinned Clark discretization or stable FilFinder v1.8 simple-line
outputs drift. It requires only the project's locked NumPy, SciPy, and Matplotlib environment. It
does not import `phenotypic` and does not require Astropy because FITS I/O is stubbed.

Generate and independently verify the all-output fixture:

```bash
uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/rolling_hough/generate_fixture.py
uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/rolling_hough/verify_fixture.py
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rolling_hough.py
```

The generator executes the pinned Clark and FilFinder files directly and does not import
`phenotypic`. The verifier imports neither reference source and checks the canonical fixture hash,
sparse-to-dense reconciliation, exact residual sums, invalid sentinels, empty-output failure, and
FilFinder endpoint behavior.

Verify the byte-pinned evidence from the reference directory:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

The evidence supports only the candidate narrow core in `SOURCE_CONTRACT.md`. The broader planned
result remains blocked pending a separate coherence/wrapper design and independent G0 review.
