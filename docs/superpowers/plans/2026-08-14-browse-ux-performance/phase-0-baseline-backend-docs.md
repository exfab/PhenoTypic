# Phase 0: Baseline, backend contract, and install documentation

**Shipping boundary:** macOS and Windows receive bundled libvips, Linux/HPC retains the
system-library path, Pillow remains the explicit portable mode, and later performance
claims have a recorded baseline.

## Task 0.1: Capture representative baselines

- [ ] Add deterministic Browse preparation fixtures for small JPEG/PNG, large TIFF, and
  RAW when rawpy is available. Keep generated fixtures small enough for CI; maintain a
  separate documented local fixture set for peak-memory measurements.
- [ ] Add a benchmark harness under `tests/performance/gui/` or the repository's existing
  performance-test location. It must measure source probe, first preview, normalization,
  DZI completion, manifest response, first OSD open, repeat navigation, and peak RSS.
- [ ] Record p50 and p95 for cold and warm paths, Pillow and libvips backend selection,
  and a burst of ten navigation actions. Record versions, image dimensions, tile
  parameters, platform, and command.
- [ ] Confirm the current number of `Image.imread`/`.rgb[:]` calls per cold selected image
  with spies. Use this as a regression count, not wall-clock inference.

## Task 0.2: Make backend selection inspectable

Files:

- `src/phenotypic/gui/results_viewer/_dzi_tiler.py`
- `src/phenotypic/gui/browse/_app.py`
- `src/phenotypic/gui/results_viewer/_app.py` if Results needs the same startup report
- `tests/gui/results_viewer/test_dzi_tiler.py`

Steps:

- [ ] Add immutable `DziBackendInfo` and `resolve_dzi_backend(mode="auto")`.
- [ ] Preserve automatic fallback for `ImportError` and native-library `OSError`.
- [ ] Add dependency injection or a backend parameter so tests can force Pillow without
  manipulating module imports.
- [ ] Log one startup record with backend, native version, and a sanitized fallback
  reason. Do not log the same warning for every image.
- [ ] If pyvips loads but `dzsave()` raises its backend-specific runtime exception,
  discard staging output and retry once with Pillow. Do not fallback on permission,
  disk-full, or publication errors.
- [ ] Test forced Pillow on every platform. Test a fake pyvips implementation
  deterministically and a real libvips smoke conditionally.
- [ ] Compare manifest geometry, image dimensions, levels, and tile coverage across
  backends. Use pixel tolerance where necessary; do not assert byte identity.

## Task 0.3: Bundle desktop libvips and document every path

Use the official `pyvips` binary extra rather than naming `pyvips-binary` directly:

```toml
"pyvips[binary]>=3.1.1; sys_platform == 'darwin' or sys_platform == 'win32'",
"pyvips>=2.2; sys_platform != 'darwin' and sys_platform != 'win32'",
```

This keeps old Linux/HPC environments away from incompatible glibc-tagged wheels. Do
not remove Pillow.

- [x] Add bundled desktop behavior, Linux/HPC system installation, verification,
  intentional-Homebrew troubleshooting, and fallback behavior to
  [`README.md`](../../../../README.md).
- [x] Add the detailed setup to
  [`docs/source/tutorials/getting_started.rst`](../../../source/tutorials/getting_started.rst)
  and [`gui_hub.md`](../../../source/how_to/pages/gui_hub.md).
- [ ] Keep commands aligned with the official
  [pyvips installation guide](https://libvips.github.io/pyvips/README.html#non-conda-install)
  when the implementation ships.
- [ ] Regenerate `uv.lock` and prove macOS resolves `pyvips-binary` while Linux markers do
  not request it.

## Verification

```bash
uv run pytest tests/gui/results_viewer/test_dzi_tiler.py -v
uv run mypy src/phenotypic
uv run ruff check --fix src/phenotypic/gui/results_viewer/_dzi_tiler.py \
  src/phenotypic/gui/browse/_app.py tests/gui/results_viewer/test_dzi_tiler.py
```

Also run both local probes:

```bash
uv run python -c "from phenotypic.gui.results_viewer import _dzi_tiler; print(_dzi_tiler.resolve_dzi_backend())"
DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix vips)/lib" \
  uv run python -c "import pyvips; print(tuple(pyvips.version(i) for i in range(3)))"
```

## Exit criteria

- Pillow is independently selectable and tested.
- Native libvips availability and fallback reason are inspectable without starting a
  conversion.
- The baseline report is reproducible and contains no absolute user paths.
- README and full docs explain bundled desktop libvips, the Linux/HPC system path, and
  automatic Pillow fallback.
