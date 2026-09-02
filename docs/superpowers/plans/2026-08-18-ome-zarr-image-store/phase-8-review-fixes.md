# OME-Zarr Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the accepted OME-Zarr review findings while retaining full pyramids for third-party consumers.

**Architecture:** Keep the public writer and loader interfaces stable. Make promotion retries and cleanup attempt-owned, derive NGFF transforms from the actual repeated 2x sampling operation, and restore explicit loader overrides after stored metadata is applied.

**Tech Stack:** Python 3.12, NumPy, Zarr v3, OME-NGFF 0.5, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md`

## Global Constraints

- `uv` is the sole package manager and runner; never use bare `python` or `pip`.
- Full-layer builder snapshots retain every pyramid level for third-party consumers.
- `promote_store(part, final, *, fsync)` and all public CLI/Python signatures remain unchanged.
- Cleanup may remove only paths allocated by the current write attempt.
- Spatial transforms describe repeated 2x sampling, saturating independently after an axis reaches one pixel; channel axes remain scale 1 and translation 0.
- Do not increment `STORE_SCHEMA_VERSION`; the private PhenoTypic metadata schema is unchanged.
- Vendored upstream references are read-only and byte-identical to their recorded source.

---

### Task 1: Concurrency-safe promotion and attempt-owned cleanup

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`
- Modify: `src/phenotypic/_cli/_cli_output_manager.py`
- Test: `tests/unit/sdk_/test_ngff_promote.py`
- Test: `tests/unit/core/test_image_zarr_roundtrip.py`
- Test: `tests/unit/cli/test_cli_output_manager.py`
- Test: `tests/integration/cli/test_commit_protocol.py`

**Interfaces:**
- Preserve: `promote_store(part: Path, final: Path, *, fsync: bool) -> Path`.
- Retire internal-only `discard_parts_for` after cleanup moves into `_save_store`.

- [ ] Add deterministic failing tests for the move-aside/concurrent-winner race, rollback collision, and same-target live-part preservation.
- [ ] Run the focused tests and verify failures reproduce the reviewed defects.
- [ ] Give each promotion retry a fresh UUID trash path and reconcile only that attempt's trash before retrying or raising.
- [ ] Wrap `_save_store` so an ordinary exception removes only its own `part`; remove the broad OutputManager cleanup.
- [ ] Run unit promotion/output-manager tests and the real multiprocessing commit-protocol test.

### Task 2: NGFF sampling transforms and validation artifacts

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Modify: `tests/unit/sdk_/test_ngff_geometry.py`
- Modify: `tests/unit/core/test_image_zarr_roundtrip.py`
- Modify: `tests/unit/core/test_ngff_conformance.py`
- Modify: `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md`
- Modify: `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/refinery/ledger.md`
- Modify: `docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py`
- Create: `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/refs/ngff-0.5.html`
- Create: `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/refs/SOURCE.md`
- Create: `tests/fixtures/phenotypic/ngff_multiscales_odd.json`

**Interfaces:**
- On disk, `datasets[].coordinateTransformations` contains scale and, for downsampled levels, translation after scale.
- For axis length `size` at level `n`, scale is `2 ** min(n, (size - 1).bit_length())`; translation is `(scale - 1) / 2`.

- [ ] Vendor the official published NGFF 0.5 document byte-for-byte and record URL, retrieval date, and SHA-256.
- [ ] Add failing unit/golden tests for odd extents, singleton-axis saturation, channel axes, image layers, and labels.
- [ ] Run the focused tests and verify the old shape-ratio implementation fails them.
- [ ] Implement sampling-factor scale and block-center translation metadata for every pyramid member.
- [ ] Amend the design, ledger, and numeric validation script; record the mutation matrix for the load-bearing claims.
- [ ] Run geometry, round-trip, conformance, golden-fixture, and logic-validation checks.

### Task 3: Loader override precedence and deterministic naming coverage

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`
- Modify: `tests/unit/core/test_image_zarr_roundtrip.py`
- Modify: `tests/unit/core/test_grid_image_zarr_roundtrip.py`
- Modify: `tests/unit/sdk_/test_ngff_promote.py`

**Interfaces:**
- `Image.load_zarr` and `GridImage.load_zarr` keep their signatures; explicit non-`None` `name` and `bit_depth` values win over stored metadata.

- [ ] Add failing Image/GridImage tests for explicit `name` and `bit_depth`, retaining existing gamma/illuminant coverage.
- [ ] Run the focused tests and verify stored protected metadata currently overwrites the explicit values.
- [ ] Record caller-supplied kwargs before stored defaults, restore stored metadata, then reapply explicit values through validated properties.
- [ ] Replace the probabilistic PID-substring assertion with UUID-component parsing.
- [ ] Run the loader and promotion test files.

### Task 4: Preserve full-layer pyramids and complete verification

**Files:**
- Modify: `tests/unit/core/test_full_layers_intermediates.py`
- Modify: `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/README.md`
- Modify: `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/EXECUTION.md`

- [ ] Add a regression assertion that `full_layers=True` stores contain multiple levels and remain conformant; do not change the production full-layer path.
- [ ] Index this phase in the multi-document plan and record its execution status.
- [ ] Run all focused OME-Zarr unit/integration tests with the repository test-runner environment.
- [ ] Run explicit-path ruff checks, compare mypy against its recorded baseline, and submit the complete unit suite through the committed Slurm runner.
- [ ] Re-run each covering regression after temporarily applying its named mutant, then restore the correct implementation and confirm green.
