# A11 reference and logic commands

Run from the repository root. `uv` is the only Python runner.

## Verify pinned files

```bash
uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/persistence/verify_checksums.py
```

## Regenerate the executable-source fixture

The exact macOS CPython 3.12 wheel is local and checksum-pinned. This command must print fixture
SHA-256 `5d8c0a17c1467fc8095cbb26b1b149fcd8b43bac551bec04f127ee565f25f35b` twice in succession:

```bash
uv run --with docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/persistence/gudhi-3.13.0-cp312-cp312-macosx_14_0_universal2.whl \
  python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/persistence/generate_fixture.py
```

The PyPI metadata records equivalent CPython 3.10-3.14 wheels for macOS, Linux, and Windows. The
committed oracle artifact is deliberately one exact platform wheel; cross-platform optional-extra
testing remains the integrator's G8 responsibility.

## Run the structurally independent numerical oracle

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/persistence.py
```

The script imports neither GUDHI nor `phenotypic`. It explicitly constructs every vertex, edge,
and square, reduces the boundary matrix over `F_2`, and separately checks thresholded Betti curves
using 8-connected foreground and 4-connected background flood fills.

## Static G0 checks

```bash
uv run ruff check \
  docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/persistence/generate_fixture.py \
  docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/persistence/verify_checksums.py \
  docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/persistence.py
uv run python -m py_compile \
  docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/persistence/generate_fixture.py \
  docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/persistence/verify_checksums.py \
  docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/persistence.py
```

Production tests, mypy, missing-dependency controls, and actual mutation killing start only after an
independent reviewer marks this exact G0 commit PASS.
