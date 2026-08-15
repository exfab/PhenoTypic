# Phase 5: Instrumentation, GUI ledgers, documentation, and release gate

**Shipping boundary:** The optimization is measurable, both DZI backends remain supported,
and every user-facing contract, screenshot, feature row, and workflow description agrees
with the implementation.

## Task 5.1: Add local structured timings

Files:

- `src/phenotypic/gui/browse/_preparation.py`
- `src/phenotypic/gui/browse/_tile_routes.py`
- `src/phenotypic/gui/browse/_assets/browse.js`
- Focused unit and E2E tests

Steps:

- [ ] Record source-probe, preview, normalization, queue-wait, DZI, manifest, and
  selection-to-OSD-open durations.
- [ ] Include cache outcome, source dimensions, revision, preparation reason, requested
  backend, actual backend/version, and sanitized fallback reason.
- [ ] Keep only a bounded in-memory window for p50/p95 display or logs. Add no remote
  telemetry or analytics dependency.
- [ ] Avoid absolute paths and source EXIF in telemetry.
- [ ] Add `Server-Timing` assertions for cold and warm manifest responses.
- [ ] Prove a warm manifest has no decode/tiling stages.

## Task 5.2: Re-run performance acceptance

- [ ] Run the Phase 0 harness on the same fixtures and environment.
- [ ] Compare cold/warm p50, p95, duplicate decode count, peak RSS, and key-repeat burst
  behavior.
- [ ] Fail acceptance if selected-image latency regresses materially while background
  work is present, even if total conversion throughput improves.
- [ ] Confirm Pillow peak memory does not regress from its recorded baseline.
- [ ] Record results with versions, commands, image dimensions, backend, cache state, and
  limitations. Do not generalize one workstation's timings to all systems.

## Task 5.3: Reconcile GUI feature and workflow ledgers

Files:

- `src/phenotypic/gui/FEATURES.md`
- `src/phenotypic/gui/WORKFLOWS.md`
- `src/phenotypic/gui/CLAUDE.md` if cache constants or lifecycle guidance changed

Steps:

- [ ] Replace the ephemeral-cache and symmetric-prefetch rows with revisioned persistent
  cache, bounded preparation, and fallback behavior.
- [ ] Add shipping rows with real tests for shortcuts, viewer reuse, preview,
  keep-position, filmstrip, dataset progress/stop, and cache clear.
- [ ] Update the existing `browse` workflow row. Do not add a duplicate workflow slug.
- [ ] Leave no `🚧 in progress` row at merge.

## Task 5.4: Update tutorial and screenshots

Files:

- `docs/source/tutorials/gui/18_browse.md`
- `docs/source/tutorials/gui/index.md`
- `docs/source/how_to/pages/gui_hub.md`
- `docs/source/tutorials/getting_started.rst`
- `README.md`
- `scripts/capture_gui_tutorial_screenshots.py`
- `docs/source/_static/gui_images/browse/`

Steps:

- [ ] Replace the promise that Browse cache data is wiped at exit with the bounded,
  revisioned, persistent policy and clear behavior.
- [ ] Document J/K and shifted jumps, arrow-key panning, preview semantics, exact-size
  keep-position behavior, filmstrip state, explicit preparation, Stop limits, and the
  Pillow fallback.
- [ ] Keep native libvips commands aligned across README, Getting Started, and GUI Hub.
- [ ] Extend `_capture_browse` to capture the loaded Single view with filmstrip, position,
  shortcut hints, and preparation status. Add a prepare/progress capture only if the
  tutorial teaches that interaction separately.
- [ ] Run the full capture script and commit the complete regenerated PNG set required by
  the project gate.

## Task 5.5: CI coverage for both backends

- [ ] Install `libvips-dev --no-install-recommends` in one GUI E2E/screenshot CI lane so
  the real fast path runs.
- [ ] Retain a Linux lane without native libvips so automatic fallback runs in a real
  environment.
- [ ] Run deterministic forced-Pillow tests in every environment.
- [ ] Run conditional real-libvips manifest geometry and OpenSeadragon smoke tests where
  native libvips is present.
- [ ] Do not add NOTICE or license files solely for a user-installed system dependency.
  Reassess licensing only if the project later redistributes native binaries.

## Task 5.6: Final functional and failure matrix

- [ ] Standard JPEG/PNG/TIFF and supported RAW.
- [ ] Missing rawpy, missing pyvips binding, missing native libvips loader, and forced
  Pillow.
- [ ] pyvips runtime error retries Pillow once; disk and permission errors do not.
- [ ] Writable sandbox, read-only sandbox/user cache, and temporary fallback.
- [ ] Source replacement during preview, normalization, and DZI.
- [ ] Process death with a partial entry, restart recovery, quota pruning, and cache clear.
- [ ] Two tabs with independent generations and two processes requesting one revision.
- [ ] Empty, single-image, large dataset, Timeline pause, hidden tab, offline/resume, rapid
  J/K repeat, and prefix-mounted hub.
- [ ] Chromium real-browser tests for keyboard scoping, preview swap, viewer reuse,
  position retention, filmstrip, batch priority, Stop, and Clear.

## Required commands

```bash
uv run pytest tests/gui/browse tests/gui/results_viewer/test_dzi_tiler.py \
  tests/integration/gui/test_browse_refresh.py -v
PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_single.py \
  tests/e2e/gui/test_browse_timeline.py \
  tests/e2e/gui/test_browse_compare_strip.py -v
uv run mypy src/phenotypic
uv run python scripts/check_features_md.py
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
uv run --group docs sphinx-build -b html -W --keep-going \
  docs/source docs/_build/html
uv run python scripts/capture_gui_tutorial_screenshots.py
```

Run Ruff only with the explicit changed paths from all phases. After all fixes and
independent reviews, rerun the full project test commands required by CI.

## Final acceptance

- Cold preparation is atomic, revision-correct, and deduplicated.
- Warm entries survive restart, remain bounded, and require no new decode or tiling.
- Selected-image work outranks speculative and dataset work.
- Dataset preparation is explicit, stoppable between stages, and accurately reported.
- J/K, viewer reuse, preview, keep-position, and filmstrip pass real-browser tests.
- Native libvips is optional, verifiable, and visible; forced Pillow remains green.
- FEATURES, WORKFLOWS, tutorial text, screenshots, README, and full docs match the shipped
  behavior.
