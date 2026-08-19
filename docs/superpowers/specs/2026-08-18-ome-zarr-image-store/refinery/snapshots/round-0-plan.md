# OME-Zarr per-image store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-image HDF5 file with an OME-Zarr (NGFF 0.5 / Zarr format v3)
store — one directory per input image — carrying image layers as named sibling
multiscale series, `objmap` as a first-class NGFF label image, and all PhenoTypic state
in a namespaced `attributes.phenotypic` block.

**Architecture:** A new `sdk_/ngff_.py` owns layout constants, pyramid geometry, the
`attributes.phenotypic` contract, the write-only OME projection, the rename-promote
commit primitive, and `valid_staged_store`. `Image`/`GridImage` gain `save2zarr` /
`load_zarr` / `load_layer_zarr` / `save_intermediate_zarr`, which fully replace the HDF
quartet. The CLI's staged-GPU engine keeps its three-stage shape: Stage 2 writes the objmap
into the promoted store for interop, and its resume state — a consumable token plus the
retained raw detector output Stage 3 replays from — moves under `.phenotypic/progress/`,
where the rest of the run's resume state already lives. The GUI reads pyramid levels
instead of decoding whole layers.
Legacy `.h5` runs are converted by an explicit `--mode migrate`, which also absorbs the
metadata-schema header migration.

**Tech Stack:** Python 3.11–3.12, `zarr>=3.0` (Zarr format v3 + sharding codec + zstd),
`numpy`, `jsonschema` (NGFF conformance, test-time), `h5py` (migration read path only),
`click` (CLI), Dash/Flask (GUI tile routes).

**Spec:** [`docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md)

**Logic validation:** [`ngff_store_geometry.py`](../../logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py)
re-derives the pyramid level count, label level parity, shard/chunk divisibility, shard
write-buffer size, per-setting file counts, and the label-downsampling requirement from
numpy alone. Run it before Phase 1 and after any change to the geometry helpers:

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py
```

---

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim
from the spec; where a task restates one, the value here is authoritative.

### Format and versions

- **NGFF 0.5 on Zarr format v3.** No NGFF 0.6, no Zarr v2 stores written.
- `requires-python = ">=3.11, <3.13"`. Python 3.10 is dropped. The `<3.13` ceiling is
  caused by `mahotas` 1.4.18 (no cp313 wheel), **not** by zarr — record that wherever the
  cap is edited so it does not read as unexamined inheritance.
- **`store_schema_version = 3`** and **`metadata_schema_version = 2`** are two separate
  markers in `attributes.phenotypic`. Never collapse them.
- **Neither `ome-zarr` nor `ome-zarr-models` is adopted, in any dependency group.**
  `ome-zarr-models` 1.7 pins `pydantic<2.13`; pydantic 2.13 has shipped. There is no
  `[tool.uv] conflicts` block, so a dev-group-only cap would still bind the whole locked
  environment.

### Store layout

- Path: `results/<dataset>/zarr/<stem>.ome.zarr/`. Never hand-join `f"{stem}.ome.zarr"`;
  always go through `zarr_store_path(output_dir, dataset, stem)`.
- Root `zarr.json` carries `ome: {version:"0.5", "bioformats2raw.layout":3}` and the
  `phenotypic` block. `OME/zarr.json` carries `ome: {version:"0.5", series:[…]}`.
- Series are named `rgb`, `gray`, `detect_mat`. **`rgb` is omitted entirely when empty.**
- The **primary series** is `rgb` when present, `gray` otherwise, and is always first in
  `series`. Labels attach to the primary series. Readers MUST resolve the objmap path
  from `phenotypic.labels.objmap` and MUST NOT hard-code `rgb/labels/objmap`.
- **`objmap` is always present**, including after Stage 1, where it is a zeros array.
- Axes: `rgb` → `["c","y","x"]` (`channel`,`space`,`space`); `gray`, `detect_mat`,
  `objmap` → `["y","x"]` (`space`,`space`). `dimension_names` is set on each level
  array's own `zarr.json` and must match `axes`.

### Geometry (all re-derived by `ngff_store_geometry.py`)

- `levels = ceil(log2(max(H, W) / 512)) + 1`, and `1` when `max(H, W) <= 512`.
  **`ceil`, not `floor`** — a floor-based draft stopped one level early. Assertion C1.
- Per-level shape is **ceil-halving**: `(h + 1) // 2, (w + 1) // 2`.
- `coordinateTransformations.scale` is derived from the **actual level shape ratio**,
  never from `2 ** n` — odd extents make the two diverge.
- **The pyramid depth is fixed, not tunable.** `pyramid_level_count(h, w)` is the whole
  policy — a pure function of the level-0 shape. The spec's `--pyramid-levels auto|N`
  (§1.3) is **descoped** and can land later as its own change; with no lever, two stores in
  one tree cannot disagree, which dissolves OPEN-QUESTIONS **P3**. The depth applies
  uniformly to every series (NGFF requires a label image to carry its parent's level count).
  The resolved count and the downsample methods are persisted in `phenotypic.pyramid`. A
  single-level store is still reachable internally, via the private `levels=` argument used
  by `save_intermediate_zarr` for builder node previews.
- Image layers downsample by **local mean**; `objmap` downsamples by
  **nearest-neighbour**. Mean-downsampling a label map fabricates label values present at
  no level-0 pixel. Assertion C5.

### Chunking, sharding, compression

- Chunks `(1, 1024, 1024)` for `rgb`; `(1024, 1024)` for 2-D arrays.
- Shards `(C, 4096, 4096)`. The shard shape must be an exact multiple of the chunk shape
  **in every dimension including the channel axis** (`3 % 1 == 0`).
- Codec `zstd` (replacing `gzip-4`).
- **Chunk key encoding uses the `"."` separator**, store-wide and uniform, so a chunk key
  is one path segment (`c.0.0.0`) rather than four nested directories. This is a Windows
  `MAX_PATH` measure and is not optional.

### Metadata

- `attributes.phenotypic` is the **sole source of truth on read**. The OME projection is
  **write-only** and is never read back.
- `series` and `labels` are **separate keys**. Never merge them into one `layers` map.
- `image_class` (`Image` / `GridImage`, drives loader dispatch) and `Metadata_ImageType`
  (`Base`/`Grid`/`Crop`/`Object`/`GridSection`, user-visible schema metadata) are
  distinct and both persisted.
- `work_id` is written into the block **at write time**, never patched in afterwards —
  the root `zarr.json` is written last, so a post-hoc patch would violate the ordering
  invariant.
- Metadata keys are canonical flat `Metadata_<Label>` headers *by convention*, and semantic
  ownership is recovered with `metadata_owner_for_header()` / `metadata_member_for_header()`
  — **never** by `startswith("Metadata_")`, prefix splitting, or category-name comparison.
  **This is not a write-time gate.** Real images legitimately carry `Metadata_PlateNum`
  (which `metadata_member_for_header` does not resolve) and bare public keys that
  `_remap_legacy_metadata_key` preserves verbatim; the store writes metadata unvalidated,
  exactly as the HDF writer does. See OPEN-QUESTIONS **D3**.
- **`omero` is emitted completely or not at all**: every channel carries a 6-hex-digit
  `color` and a `window` with all four of `min`, `max`, `start`, `end`, with `max`/`end` =
  `2**bit_depth - 1`. `rgb` emits three channels, `gray` one white channel.
- **`omero` is omitted entirely from `detect_mat` and from label groups.** NGFF makes it
  conditional and the whole-or-nothing rule is per group. `detect_mat` is a float layer
  typically in `[0, 1]`; a `2**bit_depth - 1` window over it renders solid black in any
  viewer that honours `omero`. This supersedes the spec's §2.2. See OPEN-QUESTIONS **P2**.
- **`image-label` is always emitted** (the NGFF `label.schema` requires it even though the
  prose says SHOULD), with `colors` carrying **only** the transparent background entry
  `{"label-value": 0, "rgba": [0,0,0,0]}`. This supersedes the spec's §2.3 per-value
  palette: Stage 2 overwrites the objmap in place without re-promoting, so a per-value
  palette would describe a zeros array while the array held ~1536 labels — non-conforming
  for the whole Stage-2 → Stage-3 window. Nothing in PhenoTypic reads `colors` (the GUI uses
  `skimage.color.label2rgb`). See OPEN-QUESTIONS **P1**. `properties` is never emitted.

### Commit protocol

- Every publishing stage builds `.<stem>.ome.zarr.<uuid4hex>.part/` as a **sibling** of
  the target and promotes by directory rename. The **uuid4 hex** — not a PID — is what
  makes duplicate/concurrent execution benign.
- Write order inside the `.part`: all arrays and chunks → `OME/zarr.json` → root
  `zarr.json` **last**. An interrupted store has no valid root and reads as absent.
- Promote is a **two-step move-aside**: `os.replace(final, .trash)` then
  `os.replace(part, final)` then `rmtree(trash)`. This is mandatory, not an optimization —
  `os.replace` onto a non-empty directory raises `OSError ENOTEMPTY` on POSIX, and on
  Windows `MoveFileEx`'s `MOVEFILE_REPLACE_EXISTING` cannot name a directory at all.
- Orphaned `.part` / `.trash` directories are swept at the start of each run **by uuid**,
  never by PID.
- **Resume state is carried by consumable markers under `.phenotypic/progress/`, never by
  NGFF metadata.** That includes Stage 2's **raw** detector output
  (`stage2_raw/<ds>/<stem>.npy`): Stage 3 re-promotes the store over its own objmap, so the
  store cannot be its own replay input — see OPEN-QUESTIONS **D1**.
- **`fsync` is on under SLURM and off locally**, detected from `SLURM_CPUS_PER_TASK` /
  `SLURM_JOB_ID` exactly as `resolve_worker_count` (`_cli/_cli_utils.py:65`) does. The
  resolved mode is **logged at run start** and is overridable with `--durable-writes` /
  `--no-durable-writes`. Both mitigations are required, not optional. On POSIX this means
  `fsync` on each chunk file and on the `.part` directory; on Windows the directory
  `fsync` is skipped.

### Windows

Windows is a **supported CLI platform for staged runs**.

1. No directory `fsync`; the directory step is POSIX-guarded.
2. The move-aside is wrapped in **retry-with-backoff** (`ERROR_SHARING_VIOLATION`),
   reusing the shape of `_open_hdf_with_recovery` (`sdk_/hdf_.py:34`).
3. The two-step move-aside is mandatory (see above).
4. Store paths are `\\?\`-prefixed on Windows; the `"."` chunk-key separator keeps a chunk
   key to one path segment.
5. NTFS case-insensitivity: the store's path segments (`OME`, `rgb`, `gray`,
   `detect_mat`, `objmap`, `labels`) contain no case-only collisions — **asserted by
   test**, not left to inspection.
6. Per-file antivirus overhead is documented, not mitigated.

### Testing

- **No check may skip on a missing fixture or optional dependency.** A check that cannot
  run must fail. NGFF conformance failure fails the suite; it is never downgraded to a
  warning.
- Commit-protocol tests run in the **PR lane on Linux** and the **nightly lane on
  Windows**. The one-day latency on a Windows-specific promote regression is accepted.

---

## File structure

| File | Responsibility | Phase |
|---|---|---|
| `src/phenotypic/sdk_/ngff_.py` | **New.** Layout constants, pyramid geometry, chunk/shard policy, `attributes.phenotypic` construction, OME projection, promote primitive, sweep, `valid_staged_store` | 1 |
| `src/phenotypic/sdk_/_io_constants.py` | `DIR_ZARR`, `dataset_zarr_dir`, `zarr_store_path`, `BundleLayout.store_path`, `PhenotypicAttr`, `load_image_from_store` | 2 |
| `src/phenotypic/_core/_image_parts/_image_io_handler.py` | `save2zarr` / `load_zarr` / `load_layer_zarr` / `save_intermediate_zarr` replace the HDF quartet; legacy HDF readers survive privately for migration | 2, 6 |
| `src/phenotypic/_core/_image_parts/_grid_image_handler.py` | Writes/reads `phenotypic.grid` (`nrows`, `ncols`, serialized `grid_finder`) | 2 |
| `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py` | Builder node previews via `save_intermediate_zarr` | 2 |
| `src/phenotypic/_cli/_cli_output_manager.py` | `save_image_store` replaces `save_image_hdf` | 3 |
| `src/phenotypic/_cli/_cli_stage2_token.py` | **New**, replaces deleted `_cli_sidecar.py`. Consumable Stage-2 token plus the retained raw detector output Stage 3 replays from | 3 |
| `src/phenotypic/_cli/_cli_staged_resume.py` | `valid_staged_store`, `staged_store_matches_work_id`, classifier | 3 |
| `src/phenotypic/_cli/_cli_staged_{workers,strategy,controller,slurm_worker,orchestration}.py` | Store paths, in-store label write, token lifecycle | 3 |
| `src/phenotypic/_cli/_cli_directory_scanner.py`, `_cli_recompile_slurm_scripts.py` | Non-recursive `*.ome.zarr` directory scan | 3 |
| `src/phenotypic/gui/_shared/tiles.py`, `gui/results_viewer/_tile_routes.py`, `gui/builder/_preview_{cache,tiles}.py` | Pyramid-level tile reads, fingerprint/mtime fixes | 4 |
| `src/phenotypic/gui/results_viewer/_output_root.py` | Non-recursive store discovery | 4 |
| `src/phenotypic/_cli/_cli_migrate.py` | **New.** `--mode migrate` driver | 5 |
| `src/phenotypic/sdk_/_hdf_to_zarr.py` | **New.** `migrate_hdf_to_zarr`, `migrate_run_hdf_to_zarr` | 5 |
| `src/phenotypic/sdk_/hdf_.py` | ~1,463 lines of dead DataFrame layer removed; keeper list preserved | 6 |
| `tests/fixtures/ngff/0.5/*.schema` | **New**, vendored, read-only NGFF JSON schemas | 0 |

---

## Phase DAG

```text
Phase 0  Foundation: deps, Python floor, CI, vendored NGFF schemas
   |
Phase 1  sdk_/ngff_.py — geometry, attributes, projection, promote, validity
   |
Phase 2  Image/GridImage store I/O + path constants + conformance harness
   |
   +----------------+----------------+
   |                |                |
Phase 3          Phase 4          Phase 5
CLI + staged     GUI read paths   --mode migrate
   |                |                |
   +----------------+----------------+
   |
Phase 6  Retirement: HDF write path, dead DataFrame layer, docs, supersessions
   |
Phase 7  Verification: commit protocol, differential resume, Windows lane, release note
```

Phases 3, 4, and 5 are independent of one another and may be executed in parallel by
separate agents. **Phase 5 must land before Phase 6** — migration reads legacy HDF, and
Phase 6 is what removes the public HDF surface. Phase 6 keeps the private legacy readers
(`_load_v2_grouped`, `_load_legacy_flat_group`) exactly because Phase 5 depends on them.

## Phase documents

| | Document | Tasks |
|---|---|---|
| 0 | [`phase-0-foundation.md`](phase-0-foundation.md) | 2 |
| 1 | [`phase-1-ngff-core.md`](phase-1-ngff-core.md) | 6 |
| 2 | [`phase-2-image-io.md`](phase-2-image-io.md) | 5 |
| 3 | [`phase-3-cli-staged.md`](phase-3-cli-staged.md) | 8 |
| 4 | [`phase-4-gui-read.md`](phase-4-gui-read.md) | 4 |
| 5 | [`phase-5-migrate.md`](phase-5-migrate.md) | 5 |
| 6 | [`phase-6-retirement.md`](phase-6-retirement.md) | 4 |
| 7 | [`phase-7-verification.md`](phase-7-verification.md) | 4 |

## Existing-test inventory

**32 test files reference the HDF surface this change removes.** Verified with
`grep -rlE 'save2hdf5|load_hdf5|load_layer_hdf5|save_intermediate_layers|dataset_hdf_dir|save_image_hdf|hdf_path|\.h5' tests/`.
An earlier draft named 8 of them, which left the largest single block of work in the plan
unestimated. Recorded as OPEN-QUESTIONS **G7/P20**.

**`tests/gui` is in `testpaths`** (`pyproject.toml:200`), so eleven of these run in the
default lane — yet no phase's exit criteria run `tests/gui`, which is why the breakage would
first surface at Phase 7 Task 7.4 rather than in the phase that caused it. Each phase's exit
criteria below now name the files it owns.

| Phase | Files it must update | Disposition |
|---|---|---|
| 2 | `tests/unit/core/test_image_hdf_roundtrip.py`, `test_load_layer_hdf5.py`, `test_image_pipeline.py`, `test_delta_intermediates.py`, `test_full_layers_intermediates.py`, `test_image_dtype_conversion.py`, `tests/unit/test_fixtures.py` | Port to `save2zarr`/`load_zarr`; the two `*_intermediates` files follow `save_intermediate_zarr` |
| 2 | `tests/unit/sdk_/test_io_constants.py`, `test_bundle_layout.py` | Extend (already named in Task 2.1) |
| 3 | `tests/unit/cli/test_staged_resume.py`, `test_staged_controller.py`, `test_cli_v2.py`, `tests/integration/cli/test_cli_hdf_output.py`, `test_staged_gpu_local.py` | Port; `test_cli_hdf_output.py` becomes `test_cli_store_output.py` wholesale; `test_staged_gpu_local.py:742` monkeypatches `save_image_hdf` **by name** |
| 4 | `tests/gui/_shared/test_tiles.py`, `tests/gui/results_viewer/test_tile_routes.py`, `test_output_root.py`, `test_mutation_guard.py`, `test_output_discovery_contracts.py`, `colony_view/{test_cropper,test_crop_routes,test_grid}.py`, `tests/gui/builder/{test_preview_cache,test_preview_compute_scope,test_preview_tile_blueprint}.py`, `tests/unit/gui/results_viewer/test_output_root.py` | Port. **Twelve files** — the bulk of Phase 4's real cost, and none were previously counted |
| 5 | `tests/migration/test_metadata_schema_migration.py`, `tests/unit/cli/{test_cli_recompile,test_cli_recompile_slurm,test_cli_recompile_metadata_migration_slurm}.py`, `tests/unit/sdk_/test_metadata_io.py` | Port to `--mode migrate`; the `_slurm` file loses its subject entirely (Task 5.4 deletes it) |
| 6 | `tests/unit/sdk_/test_hdf_open_recovery.py` | Must keep passing **unchanged** — it is what pins the keeper list |

Every phase's exit criteria must run the files it owns, not only `tests/unit/<area>`.

## Open questions

Tracked in [`OPEN-QUESTIONS.md`](OPEN-QUESTIONS.md) — **P1–P12** raised while grounding the
spec against the code, **D1–D16** from an independent data-flow review, every one
re-verified in this worktree before being recorded.

**Decided:**

- **D1** — the raw Stage-2 detector output moves to
  `.phenotypic/progress/stage2_raw/<ds>/<stem>.npy`, beside the token. Restores today's
  exact retry idempotency; without it a retried Stage 3 deletes a real colony via
  `drop_frame_background`.
- **P1** — `image-label.colors` carries the background entry only. Nothing in PhenoTypic
  reads it, and a per-value palette would be stale for the whole Stage-2 → Stage-3 window.
- **P2** — `omero` is omitted entirely from `detect_mat`; a `2**bit_depth - 1` window over
  a float layer in `[0,1]` renders solid black.
- **P3** — `--pyramid-levels` is descoped. Depth is derived from shape, so mixed geometry
  is unreachable.

**Still undecided, non-blocking:** **D9** (`metadata.csv` rewrite vs `metadata_sha256`),
**D10** (`_metadata_migration.py`'s HDF-target machinery is uncosted).
# Phase 0 — Foundation: dependencies, Python floor, CI, vendored NGFF schemas

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §6, §7.

**Why first:** every later phase imports `zarr`, and the conformance harness in Phase 2
imports `jsonschema` and reads the vendored schemas. Nothing else can start until the
resolution universe contains them.

**Blocks:** Phases 1–7.

---

### Task 0.1: Raise the Python floor, add `zarr` and `jsonschema`, update CI

**Files:**
- Modify: `pyproject.toml` (`requires-python` line 25, classifiers lines 27–34,
  `dependencies` line 45 region, `[dependency-groups]`, `[[tool.mypy.overrides]]`)
- Modify: `uv.lock` (regenerated, not hand-edited)
- Modify: `.github/workflows/run-pytest.yml` (header prose lines 4–7; matrix `3.10` entry)
- Modify: `.github/workflows/run-pytest-full.yml` (header prose line 10; matrix `3.10`
  entry at line 46)
- Modify: `.github/workflows/package-integrity.ci.yml` (comment line 43; matrix line 44)
- Modify: `.github/workflows/publish_to_pypi.yml` (lines 17 and 20)
- Test: `tests/unit/test_packaging_floor.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: an environment in which `import zarr` and `import jsonschema` succeed, and
  `zarr.__version__` starts with `3.`. Every later task assumes both.

**Constraints specific to this task:**
- `jsonschema` is currently **transitive only** — it is not named anywhere in
  `pyproject.toml` (verified). Spec §7 forbids a conformance check that skips on a
  missing dependency, so it must become a **declared** dependency of the test group, not
  left to chance.
- Do **not** add `ome-zarr` or `ome-zarr-models` to any group (Global Constraints).
- Ruff sets no `target-version` and mypy no `python_version`; both follow
  `requires-python`, so raising the floor may surface new `UP` lints. Fix them in this
  task, with `uv run ruff check --fix` on **explicit paths only**.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_packaging_floor.py`:

```python
"""Guards on the declared dependency universe for the OME-Zarr store.

These are packaging assertions, not behaviour tests: they fail loudly if a
future edit reintroduces Python 3.10, adopts an ome-zarr package, or lets the
NGFF conformance dependency drift back to transitive-only.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_requires_python_floor_is_311(pyproject: dict) -> None:
    assert pyproject["project"]["requires-python"] == ">=3.11, <3.13"


def test_classifiers_drop_310(pyproject: dict) -> None:
    classifiers = pyproject["project"]["classifiers"]
    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers


def test_zarr_is_a_runtime_dependency(pyproject: dict) -> None:
    deps = pyproject["project"]["dependencies"]
    assert any(dep.startswith("zarr") for dep in deps), deps


def test_h5py_is_retained_for_migration(pyproject: dict) -> None:
    deps = pyproject["project"]["dependencies"]
    assert any(dep.split(">")[0].split("=")[0].strip() == "h5py" for dep in deps)


def test_ome_zarr_packages_are_not_adopted_anywhere(pyproject: dict) -> None:
    """`ome-zarr-models` pins pydantic<2.13; uv resolves one universe."""
    banned = {"ome-zarr", "ome-zarr-models"}
    pools: list[list[str]] = [list(pyproject["project"]["dependencies"])]
    for group in pyproject.get("dependency-groups", {}).values():
        pools.append([item for item in group if isinstance(item, str)])
    for extra in pyproject["project"].get("optional-dependencies", {}).values():
        pools.append(list(extra))
    for pool in pools:
        for requirement in pool:
            name = (
                requirement.split(";")[0]
                .split("[")[0]
                .split(">")[0]
                .split("<")[0]
                .split("=")[0]
                .strip()
                .lower()
            )
            assert name not in banned, requirement


def test_jsonschema_is_declared_not_transitive(pyproject: dict) -> None:
    """Spec §7: a conformance check may never skip on a missing dependency."""
    groups = pyproject.get("dependency-groups", {})
    declared = {
        requirement.split(";")[0].split(">")[0].split("<")[0].split("=")[0].strip().lower()
        for group in groups.values()
        for requirement in group
        if isinstance(requirement, str)
    }
    assert "jsonschema" in declared


def test_zarr_v3_is_importable_at_runtime() -> None:
    import zarr

    assert zarr.__version__.startswith("3."), zarr.__version__
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
uv run pytest tests/unit/test_packaging_floor.py -v
```

Expected: `test_requires_python_floor_is_311`, `test_classifiers_drop_310`,
`test_zarr_is_a_runtime_dependency`, `test_jsonschema_is_declared_not_transitive`, and
`test_zarr_v3_is_importable_at_runtime` all FAIL (the last with `ModuleNotFoundError:
No module named 'zarr'`). `test_h5py_is_retained_for_migration` and
`test_ome_zarr_packages_are_not_adopted_anywhere` PASS already — that is correct; they are
regression guards, not new work.

- [ ] **Step 3: Edit `pyproject.toml`**

Line 25:

```toml
requires-python = ">=3.11, <3.13"
```

Classifiers — delete the `3.10` line, keep `3.11` and `3.12`:

```toml
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12"
```

Add to `[project].dependencies`, beside the existing `h5py` entry:

```toml
    "zarr>=3.0",
```

Leave `"h5py"` in place — it is the `--mode migrate` read path (Phase 5).

Add `jsonschema` to the **test** dependency group (the group the PR lane installs):

```toml
    "jsonschema>=4.0",
```

Add a mypy override beside the existing `h5py` / `mahotas` ones so an untyped `zarr`
does not fail the type gate:

```toml
[[tool.mypy.overrides]]
module = [
    "zarr",
    "zarr.*",
]
ignore_missing_imports = true
```

- [ ] **Step 4: Edit the four CI workflows**

`.github/workflows/run-pytest.yml` — header prose (lines 4–7) and the matrix. Replace the
`3.10` floor entry with `3.11` in **both** places:

```yaml
# Matrix: Linux x Python {3.11 (floor), 3.12 (ceiling)}. Windows, macOS, and
# any intermediate Python move to the nightly full lane in
# ``run-pytest-full.yml``.
```

```yaml
          # Ubuntu: floor (3.11) + ceiling (3.12).
          - os: ubuntu-latest
            python-version: "3.11"
          - os: ubuntu-latest
            python-version: "3.12"
```

`.github/workflows/run-pytest-full.yml` — line 10 prose becomes
`#   * Linux x Python {3.11, 3.12}`; delete the `python-version: "3.10"` matrix entry at
line 46 together with its sibling `os:` key.

`.github/workflows/package-integrity.ci.yml` lines 43–44:

```yaml
        # Matches requires-python (>=3.11, <3.13). The <3.13 ceiling is
        # mahotas 1.4.18 (no cp313 wheel), not zarr.
        python-version: ["3.11", "3.12"]
```

`.github/workflows/publish_to_pypi.yml` lines 17 and 20:

```yaml
       - name: Set up Python 3.11
         uses: actions/setup-python@v5
         with:
           python-version: '3.11'
```

Leave the testmon cache key at `run-pytest.yml:153` alone — it already keys on
`hashFiles('uv.lock')`, and Step 5 changes `uv.lock`, which invalidates it correctly
without an edit.

- [ ] **Step 5: Regenerate the lock and sync**

```bash
uv lock
uv sync --group dev --group test-qt --group docs --extra gui --extra napari
uv run python -c "import zarr, jsonschema; print(zarr.__version__)"
```

Expected: a `3.x` version string. Markers resolve zarr to 3.1.6 on 3.11 and 3.3.x on
3.12 with no pinning; do **not** pin it.

- [ ] **Step 6: Run the packaging test and the lint/type gates**

```bash
uv run pytest tests/unit/test_packaging_floor.py -v
uv run ruff check --fix pyproject.toml src/phenotypic tests/unit/test_packaging_floor.py
uv run mypy src/phenotypic
```

Expected: all packaging tests PASS. Ruff may report new `UP` lints now that the floor is
3.11 (e.g. `UP007` union syntax) — fix them; that churn belongs to this task, not to a
later one. If ruff rewrites files outside `src/phenotypic`, `git status` and revert them
before committing.

- [ ] **Step 7: Run the full unit suite to catch 3.11-floor fallout**

```bash
uv run pytest tests/unit -x -q
```

Expected: PASS. A failure here is real fallout from the floor raise (removed 3.10
compatibility shims), not from zarr, which nothing imports yet.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .github/workflows tests/unit/test_packaging_floor.py src/phenotypic
git commit -m "build: raise Python floor to 3.11 and add zarr>=3.0

Drops Python 3.10 across pyproject, the four CI workflows, and the lock.
Adds zarr>=3.0 as a runtime dependency and promotes jsonschema from a
transitive to a declared test dependency, because spec §7 forbids a
conformance check that skips on a missing dependency. h5py is retained
for the --mode migrate read path. The <3.13 ceiling is mahotas 1.4.18,
not zarr; that is now stated where the cap is spelled."
```

---

### Task 0.2: Vendor the NGFF 0.5 JSON schemas as read-only reference material

**Files:**
- Create: `tests/fixtures/ngff/0.5/image.schema`
- Create: `tests/fixtures/ngff/0.5/label.schema`
- Create: `tests/fixtures/ngff/0.5/ome.schema`
- Create: `tests/fixtures/ngff/0.5/_version.schema`
- Create: `tests/fixtures/ngff/0.5/SOURCE.md`
- Modify: `pyproject.toml` (`[tool.ruff] extend-exclude`)
- Test: `tests/unit/test_ngff_schema_fixtures.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/fixtures/ngff/0.5/{image,label,ome,_version}.schema` — read by the
  conformance harness `assert_store_conforms(...)` introduced in Task 2.5.

**Constraints specific to this task:**
- **Four files, not three.** All three of `image`, `label`, and `ome` carry exactly one
  **remote** `$ref` — `https://ngff.openmicroscopy.org/0.5/schemas/_version.schema`
  (verified by parsing the downloaded files). `jsonschema` >= 4.18 does **not** fetch remote
  refs; it raises `referencing.exceptions.Unresolvable`, which is not a `ValidationError`, so
  the harness would **error** rather than fail and offline CI would have no fallback.
  `_version.schema` is 280 bytes: `{"type": "string", "enum": ["0.5"]}`. Task 2.5 resolves it
  through a `referencing.Registry` keyed on each file's `$id`.
- These are **vendored upstream sources**. Per CLAUDE.md they must stay byte-identical to
  upstream: never lint, format, autofix, tidy, or "fix" them. Add the directory to
  `[tool.ruff] extend-exclude` in the same commit that adds the files, so no later bare
  `ruff check --fix` can touch them.
- `SOURCE.md` records the exact upstream URL, the retrieval date, and the sha256 of each
  file, so a future reader can prove the copy is unmodified.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ngff_schema_fixtures.py`:

```python
"""The vendored NGFF 0.5 schemas must be present, parseable, and unmodified.

Spec §7 forbids a conformance check that skips on a missing fixture, so the
absence of these files is a hard failure here rather than a skip downstream.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ngff" / "0.5"
SCHEMA_NAMES = ("image.schema", "label.schema", "ome.schema", "_version.schema")


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_is_present_and_parses(name: str) -> None:
    path = SCHEMA_DIR / name
    assert path.is_file(), f"vendored NGFF schema missing: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_matches_recorded_digest(name: str) -> None:
    """SOURCE.md pins each file's sha256; a mismatch means someone edited it."""
    recorded = dict(
        re.findall(
            r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|",
            (SCHEMA_DIR / "SOURCE.md").read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )
    actual = hashlib.sha256((SCHEMA_DIR / name).read_bytes()).hexdigest()
    assert recorded.get(name) == actual, (
        f"{name} does not match the digest recorded in SOURCE.md; the vendored "
        "upstream copy must stay byte-identical."
    )


def test_every_schema_is_rooted_at_the_attributes_object() -> None:
    """All three are ``{"required": ["ome"], "properties": {"ome": …}}``.

    This is what the conformance harness must validate against: the whole
    ``attributes`` mapping, NOT ``attributes["ome"]``. Passing the inner block
    fails with "'ome' is a required property" on every store.
    """
    for name in ("image.schema", "label.schema", "ome.schema"):
        payload = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
        assert payload["required"] == ["ome"], name
        assert list(payload["properties"]) == ["ome"], name
        assert payload["description"] == "The zarr.json attributes key", name


def test_ome_schema_requires_series() -> None:
    """Stricter than the prose — §7 calls this out explicitly."""
    payload = json.loads((SCHEMA_DIR / "ome.schema").read_text(encoding="utf-8"))
    assert payload["properties"]["ome"]["required"] == ["series", "version"]


def test_label_schema_requires_image_label() -> None:
    payload = json.loads((SCHEMA_DIR / "label.schema").read_text(encoding="utf-8"))
    assert payload["properties"]["ome"]["required"] == ["image-label", "version"]


def test_image_label_does_not_require_exhaustive_colors() -> None:
    """Pins the fact that re-graded P1: `colors` is OPTIONAL.

    `$defs/image-label` has no `required` list at all, so nothing obliges one
    entry per unique label value. The spec's §2.3 "MUST" is a PhenoTypic
    invention, not an NGFF rule.
    """
    payload = json.loads((SCHEMA_DIR / "label.schema").read_text(encoding="utf-8"))
    image_label = payload["$defs"]["image-label"]
    assert "required" not in image_label
    assert "colors" in image_label["properties"]


def test_every_remote_ref_is_vendored() -> None:
    """A remote $ref raises Unresolvable, which is not a ValidationError."""
    import re

    ids = {
        json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))["$id"]
        for name in SCHEMA_NAMES
    }
    for name in SCHEMA_NAMES:
        raw = (SCHEMA_DIR / name).read_text(encoding="utf-8")
        for ref in re.findall(r'"\$ref"\s*:\s*"(https?://[^"]+)"', raw):
            assert ref in ids, f"{name} references un-vendored {ref}"
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
uv run pytest tests/unit/test_ngff_schema_fixtures.py -v
```

Expected: FAIL with `vendored NGFF schema missing: .../image.schema`.

- [ ] **Step 3: Fetch the schemas**

```bash
mkdir -p tests/fixtures/ngff/0.5
BASE=https://ngff.openmicroscopy.org/0.5/schemas
for name in image label ome _version; do
  curl -fsSL "$BASE/$name.schema" -o "tests/fixtures/ngff/0.5/$name.schema"
done
sha256sum tests/fixtures/ngff/0.5/*.schema
```

Do not reformat the downloaded bytes. If a URL 404s, resolve the correct one from
<https://ngff.openmicroscopy.org/0.5/> and record what you used in `SOURCE.md` — do
**not** hand-write a schema.

- [ ] **Step 4: Write `SOURCE.md`**

```markdown
# Vendored NGFF 0.5 JSON schemas

Read-only upstream reference material. **Never lint, format, autofix, or edit
these files.** They are the artifact every conformance assertion resolves
against; editing one silently invalidates every claim ever checked against it.

- Upstream: <https://ngff.openmicroscopy.org/0.5/schemas/>
- Retrieved: 2026-08-18

| file | sha256 |
|---|---|
| `image.schema` | `<paste from sha256sum>` |
| `label.schema` | `<paste from sha256sum>` |
| `ome.schema` | `<paste from sha256sum>` |
| `_version.schema` | `<paste from sha256sum>` |

Three facts about these files that the spec gets wrong or omits:

- `ome.schema` **requires** `["series", "version"]`, though the prose presents
  named series as optional.
- `label.schema` **requires** `["image-label", "version"]`, though the prose says
  SHOULD — **but `$defs/image-label` has no `required` list**, so `colors` is
  optional and nothing requires one entry per unique label value. The spec's
  §2.3 "MUST" is a PhenoTypic policy, not an NGFF rule.
- `$defs/omero` requires only `["channels"]`; the channel item has no `required`
  list and `color` is an unconstrained string. Only `window`, **if present**,
  requires all four of `start`/`min`/`end`/`max`. Emitting the full block is
  PhenoTypic policy too.
- All three reference `_version.schema` remotely, which is why it is vendored
  here and resolved through a `referencing.Registry` rather than fetched.
```

- [ ] **Step 5: Exclude the directory from ruff**

In `pyproject.toml`, extend the existing `[tool.ruff] extend-exclude` list (the one
already protecting `docs/superpowers/**/refs`):

```toml
extend-exclude = [
    "docs/superpowers/**/refs",
    "tests/fixtures/ngff",
]
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
uv run pytest tests/unit/test_ngff_schema_fixtures.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/ngff pyproject.toml tests/unit/test_ngff_schema_fixtures.py
git commit -m "test: vendor the NGFF 0.5 JSON schemas as read-only fixtures

Conformance is validated against the published schemas via jsonschema
rather than ome-zarr-models, which pins pydantic<2.13. SOURCE.md pins a
sha256 per file and a test asserts the digests still match, so a stray
formatter cannot silently invalidate every conformance assertion. The
directory is added to ruff's extend-exclude in the same commit."
```

---

## Phase 0 exit criteria

- [ ] `uv run python -c "import zarr; print(zarr.__version__)"` prints a `3.x` version.
- [ ] `uv run pytest tests/unit/test_packaging_floor.py tests/unit/test_ngff_schema_fixtures.py -v` is all green.
- [ ] `grep -rn "3\.10" .github/workflows/` returns no `python-version` matches.
- [ ] `uv run pytest tests/unit -q` passes at the raised floor.
- [ ] `uv run mypy src/phenotypic` passes.
# Phase 1 — `sdk_/ngff_.py`: geometry, attributes, projection, promote, validity

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §1, §2, §3.2, §3.6, §3.7, §3.8.

**Depends on:** Phase 0 (needs `import zarr`).
**Blocks:** Phase 2 and everything downstream.

This phase builds one new module and nothing else. It touches no existing behaviour, so
the whole phase can land while the HDF path still works. Every task is pure-function or
filesystem-local and testable without a pipeline.

**Before starting, run the logic validation script and confirm it passes:**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py
```

Expected: `All store-geometry claims hold.` and exit 0. The geometry helpers you write in
Task 1.1 must agree with `level_count` and `level_shapes` in that script — Task 1.1's test
imports the script and asserts equality against it, so the script is the reference, not a
parallel implementation.

---

### Task 1.1: Layout constants and pyramid geometry

**Files:**
- Create: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_geometry.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```python
  NGFF_VERSION: Final[str] = "0.5"
  BIOFORMATS2RAW_LAYOUT: Final[int] = 3
  STORE_SCHEMA_VERSION: Final[int] = 3
  STORE_SUFFIX: Final[str] = ".ome.zarr"
  PYRAMID_STOP_PX: Final[int] = 512
  OME_GROUP: Final[str] = "OME"
  OME_XML_NAME: Final[str] = "METADATA.ome.xml"
  LABELS_GROUP: Final[str] = "labels"
  OBJMAP_LABEL: Final[str] = "objmap"
  SERIES_ORDER: Final[tuple[str, str, str]] = ("rgb", "gray", "detect_mat")
  AXES_3D: Final[tuple[str, str, str]] = ("c", "y", "x")
  AXES_2D: Final[tuple[str, str]] = ("y", "x")

  def pyramid_level_count(height: int, width: int, *, stop_px: int = PYRAMID_STOP_PX) -> int
  def pyramid_level_shapes(shape: tuple[int, ...], levels: int) -> tuple[tuple[int, ...], ...]
  def level_scale_vector(level0: tuple[int, ...], level_n: tuple[int, ...]) -> list[float]
  def downsample_image(array: np.ndarray) -> np.ndarray
  def downsample_label(array: np.ndarray) -> np.ndarray
  def build_pyramid(array: np.ndarray, levels: int, *, kind: Literal["image", "label"]) -> list[np.ndarray]
  def axes_for(series: str) -> tuple[str, ...]
  ```

**Constraints specific to this task:**
- `pyramid_level_count` must be `ceil(log2(max(H,W)/stop_px)) + 1`, and `1` when
  `max(H,W) <= stop_px`. **`ceil`, never `floor`.**
- `pyramid_level_shapes` uses ceil-halving `(h+1)//2` with a floor of 1 per axis, and
  leaves any leading channel axis unchanged.
- `level_scale_vector` divides level-0 extent by level-n extent **per axis, from the
  actual shapes** — never `2 ** n`. The channel axis, if present, gets scale `1.0`.
- `downsample_label` is `array[::2, ::2]` (top-left of each block). Never mean.
- `downsample_image` is a 2×2 block mean over an **edge-replicated** pad, so an odd
  extent yields `(h+1)//2` without a zero-padded darkened edge. It preserves dtype:
  integer inputs are rounded with `np.rint` and cast back.
- **The pyramid depth is fixed, not tunable.** `pyramid_level_count(h, w)` is the whole
  policy: a pure function of the level-0 shape, with no user lever and no stored choice to
  disagree with. The spec's `--pyramid-levels auto|N` (§1.3) is **descoped** — see
  OPEN-QUESTIONS **P3**. A single-level store is still reachable internally (builder node
  previews, Phase 2 Task 2.4) via the private `levels=` argument on `_save_store`; it is
  simply not a CLI surface.

  Descoping it also dissolves P3 outright: with geometry a pure function of image shape,
  two stores in one tree cannot disagree, so `valid_staged_store` needs no level check and
  a resumed run cannot produce mixed geometry.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_geometry.py`:

```python
"""Pyramid geometry, checked against the committed logic-validation script.

The script under docs/superpowers/logic_validation_scripts/ is the reference
implementation for level counts and level shapes; it depends only on numpy and
has already refuted a floor-based formula. These tests assert the shipped
helpers agree with it, so the two can never drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from phenotypic.sdk_ import ngff_

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "superpowers"
    / "logic_validation_scripts"
    / "2026-08-18-ome-zarr-image-store"
    / "ngff_store_geometry.py"
)


def _load_reference():
    spec = importlib.util.spec_from_file_location("ngff_store_geometry", _SCRIPT)
    assert spec is not None and spec.loader is not None, _SCRIPT
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = _load_reference()

PLATES = [(2048, 2048), (4000, 3000), (6000, 4000), (512, 512), (300, 200), (513, 100)]


@pytest.mark.parametrize(("height", "width"), PLATES)
def test_level_count_matches_reference(height: int, width: int) -> None:
    assert ngff_.pyramid_level_count(height, width) == REFERENCE.level_count(
        height, width
    )


def test_level_count_uses_ceil_not_floor() -> None:
    """floor(log2(4000/512)) + 1 == 3, which stops one level early at 1000x750."""
    assert ngff_.pyramid_level_count(4000, 3000) == 4


def test_single_level_at_or_below_stop_px() -> None:
    assert ngff_.pyramid_level_count(512, 512) == 1
    assert ngff_.pyramid_level_count(100, 100) == 1


@pytest.mark.parametrize(("height", "width"), PLATES)
def test_level_shapes_match_reference(height: int, width: int) -> None:
    levels = ngff_.pyramid_level_count(height, width)
    shapes = ngff_.pyramid_level_shapes((height, width), levels)
    assert [tuple(s) for s in shapes] == [
        tuple(s) for s in REFERENCE.level_shapes(height, width)
    ]


def test_level_shapes_ceil_halve_odd_extents() -> None:
    assert ngff_.pyramid_level_shapes((1025, 7), 3) == ((1025, 7), (513, 4), (257, 2))


def test_level_shapes_leave_channel_axis_alone() -> None:
    assert ngff_.pyramid_level_shapes((3, 1025, 7), 2) == ((3, 1025, 7), (3, 513, 4))


def test_scale_vector_comes_from_actual_shapes_not_powers_of_two() -> None:
    """1025 -> 513 is a ratio of 1025/513, which is NOT 2.0."""
    scale = ngff_.level_scale_vector((1025, 7), (513, 4))
    assert scale == pytest.approx([1025 / 513, 7 / 4])
    assert scale[0] != pytest.approx(2.0)


def test_scale_vector_pins_channel_axis_to_one() -> None:
    assert ngff_.level_scale_vector((3, 1024, 1024), (3, 512, 512)) == pytest.approx(
        [1.0, 2.0, 2.0]
    )


def test_label_downsample_invents_no_new_values() -> None:
    rng = np.random.default_rng(20260818)
    labels = rng.choice(np.array([0, 3, 7, 11, 40], dtype=np.uint16), size=(64, 64))
    small = ngff_.downsample_label(labels)
    assert set(np.unique(small)).issubset(set(np.unique(labels)))
    assert small.shape == (32, 32)
    assert small.dtype == labels.dtype


def test_mean_downsample_would_invent_values() -> None:
    """Guards C5: proves the rejected method really is wrong, not merely unchosen."""
    labels = np.array([[0, 40], [40, 40]], dtype=np.uint16)
    meaned = ngff_.downsample_image(labels)
    assert set(np.unique(meaned)) - set(np.unique(labels))


def test_image_downsample_odd_extent_uses_edge_pad_not_zero_pad() -> None:
    array = np.full((3, 3), 100, dtype=np.uint8)
    small = ngff_.downsample_image(array)
    assert small.shape == (2, 2)
    assert (small == 100).all(), "a zero pad would darken the trailing row/column"


def test_image_downsample_preserves_dtype() -> None:
    array = np.arange(16, dtype=np.uint16).reshape(4, 4)
    assert ngff_.downsample_image(array).dtype == np.uint16
    assert ngff_.downsample_image(array.astype(np.float64)).dtype == np.float64


def test_build_pyramid_shapes_and_count() -> None:
    array = np.zeros((1025, 7), dtype=np.uint16)
    levels = ngff_.build_pyramid(array, 3, kind="label")
    assert [lvl.shape for lvl in levels] == [(1025, 7), (513, 4), (257, 2)]


def test_build_pyramid_channel_first_rgb() -> None:
    array = np.zeros((3, 1024, 1024), dtype=np.uint8)
    levels = ngff_.build_pyramid(array, 2, kind="image")
    assert [lvl.shape for lvl in levels] == [(3, 1024, 1024), (3, 512, 512)]


def test_level_count_is_a_pure_function_of_shape() -> None:
    """No user lever, so two stores in one tree cannot disagree (P3)."""
    assert not hasattr(ngff_, "resolve_pyramid_levels")
    assert ngff_.pyramid_level_count(4000, 3000) == ngff_.pyramid_level_count(4000, 3000)


def test_axes_for_series() -> None:
    assert ngff_.axes_for("rgb") == ("c", "y", "x")
    assert ngff_.axes_for("gray") == ("y", "x")
    assert ngff_.axes_for("detect_mat") == ("y", "x")
    assert ngff_.axes_for("objmap") == ("y", "x")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_geometry.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'phenotypic.sdk_.ngff_'`.

- [ ] **Step 3: Write the module**

Create `src/phenotypic/sdk_/ngff_.py`:

```python
"""OME-Zarr (NGFF 0.5 / Zarr format v3) store layout, geometry, and commit protocol.

Single source of truth for everything about the on-disk shape of a per-image
store: the directory layout, the pyramid geometry, the chunk/shard/codec
policy, the ``attributes.phenotypic`` contract, the write-only OME projection,
and the rename-promote commit primitive.

Nothing here reads or writes an :class:`~phenotypic.Image`; the layer that does
is :mod:`phenotypic._core._image_parts._image_io_handler`. Keeping the geometry
free of the image model is what lets the committed logic-validation script
(``docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/``)
re-derive every numeric claim from numpy alone.

See also:
    ``docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md``
"""

from __future__ import annotations

import math
from typing import Final, Literal, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

#: NGFF specification version written into every ``ome`` block.
NGFF_VERSION: Final[str] = "0.5"

#: ``bioformats2raw.layout`` marker on the root group (named-series collection).
BIOFORMATS2RAW_LAYOUT: Final[int] = 3

#: Version of the PhenoTypic *group and array* layout. Distinct from
#: ``metadata_schema_version``, which versions the header namespace.
STORE_SCHEMA_VERSION: Final[int] = 3

#: Directory suffix for one per-image store.
STORE_SUFFIX: Final[str] = ".ome.zarr"

#: Halve pyramid levels until ``max(H, W) <= PYRAMID_STOP_PX``.
PYRAMID_STOP_PX: Final[int] = 512

OME_GROUP: Final[str] = "OME"
OME_XML_NAME: Final[str] = "METADATA.ome.xml"
LABELS_GROUP: Final[str] = "labels"
OBJMAP_LABEL: Final[str] = "objmap"

#: Canonical series order. ``rgb`` is omitted from a store when empty; the
#: remaining names keep this relative order.
SERIES_ORDER: Final[tuple[str, str, str]] = ("rgb", "gray", "detect_mat")

AXES_3D: Final[tuple[str, str, str]] = ("c", "y", "x")
AXES_2D: Final[tuple[str, str]] = ("y", "x")

#: NGFF axis ``type`` per dimension name.
AXIS_TYPES: Final[dict[str, str]] = {"c": "channel", "y": "space", "x": "space"}


def axes_for(series: str) -> tuple[str, ...]:
    """Return the ``dimension_names`` tuple for one series or label name.

    Args:
        series: ``"rgb"``, ``"gray"``, ``"detect_mat"``, or ``"objmap"``.

    Returns:
        ``("c", "y", "x")`` for ``rgb``; ``("y", "x")`` otherwise.
    """
    return AXES_3D if series == "rgb" else AXES_2D


# ---------------------------------------------------------------------------
# Pyramid geometry
# ---------------------------------------------------------------------------


def pyramid_level_count(
    height: int, width: int, *, stop_px: int = PYRAMID_STOP_PX
) -> int:
    """Number of pyramid levels when halving until ``max(H, W) <= stop_px``.

    ``ceil``, not ``floor``: a floor-based formula terminates one level early
    and leaves a 4000x3000 plate's smallest level at 1000x750.

    Args:
        height: Level-0 height in pixels.
        width: Level-0 width in pixels.
        stop_px: Longest-edge threshold at which halving stops.

    Returns:
        A level count of at least 1.
    """
    longest = max(height, width)
    if longest <= stop_px:
        return 1
    return int(math.ceil(math.log2(longest / stop_px))) + 1


def pyramid_level_shapes(
    shape: tuple[int, ...], levels: int
) -> tuple[tuple[int, ...], ...]:
    """Explicit shape per pyramid level, ceil-halving the two spatial axes.

    A leading channel axis (3-D input) is carried through unchanged.

    Args:
        shape: Level-0 shape, ``(y, x)`` or ``(c, y, x)``.
        levels: Number of levels to emit, including level 0.

    Returns:
        A tuple of ``levels`` shapes, starting with *shape*.
    """
    shapes: list[tuple[int, ...]] = [tuple(shape)]
    for _ in range(levels - 1):
        previous = shapes[-1]
        lead, (h, w) = previous[:-2], previous[-2:]
        shapes.append((*lead, max(1, (h + 1) // 2), max(1, (w + 1) // 2)))
    return tuple(shapes)


def level_scale_vector(
    level0: tuple[int, ...], level_n: tuple[int, ...]
) -> list[float]:
    """Per-axis downsample factor from the *actual* level shapes.

    NGFF requires ``coordinateTransformations.scale`` to describe the real
    relationship between levels. Odd extents make the true ratio diverge from
    ``2 ** n``, so this is derived from shapes and never from the level index.

    Args:
        level0: Level-0 shape.
        level_n: Shape of the level being described.

    Returns:
        One float per axis, in axis order. Any leading channel axis is 1.0.
    """
    return [float(a) / float(b) for a, b in zip(level0, level_n, strict=True)]


def downsample_image(array: np.ndarray) -> np.ndarray:
    """2x block-mean downsample with edge replication, preserving dtype.

    Edge replication (rather than zero padding) is what keeps an odd trailing
    row or column at its own brightness instead of darkening it toward zero.
    The spatial axes are the last two; any leading channel axis is preserved.

    Args:
        array: 2-D ``(y, x)`` or 3-D ``(c, y, x)`` array.

    Returns:
        An array whose spatial extents are ``(n + 1) // 2``.
    """
    h, w = array.shape[-2:]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        pad_width = [(0, 0)] * (array.ndim - 2) + [(0, pad_h), (0, pad_w)]
        array = np.pad(array, pad_width, mode="edge")
    lead = array.shape[:-2]
    ph, pw = array.shape[-2:]
    blocks = array.astype(np.float64).reshape(*lead, ph // 2, 2, pw // 2, 2)
    reduced = blocks.mean(axis=(-3, -1))
    if np.issubdtype(array.dtype, np.integer):
        return np.rint(reduced).astype(array.dtype)
    return reduced.astype(array.dtype)


def downsample_label(array: np.ndarray) -> np.ndarray:
    """2x nearest-neighbour downsample (top-left of each 2x2 block).

    A label map must never be mean-downsampled: averaging fabricates label
    values present at no level-0 pixel. Verified by claim C5 of the committed
    logic-validation script.

    Args:
        array: 2-D ``(y, x)`` integer label array.

    Returns:
        An array whose extents are ``(n + 1) // 2``, with dtype preserved and
        no label value absent from *array*.
    """
    return array[..., ::2, ::2]


def build_pyramid(
    array: np.ndarray, levels: int, *, kind: Literal["image", "label"]
) -> list[np.ndarray]:
    """Materialise every pyramid level for one array.

    Args:
        array: Level-0 array.
        levels: Level count, including level 0.
        kind: ``"image"`` downsamples by local mean; ``"label"`` by
            nearest-neighbour.

    Returns:
        A list of ``levels`` arrays, starting with *array*.
    """
    reduce = downsample_image if kind == "image" else downsample_label
    out = [array]
    for _ in range(levels - 1):
        out.append(reduce(out[-1]))
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_geometry.py -v
uv run python docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py
```

Expected: all tests PASS; script exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_geometry.py
git commit -m "feat(sdk): add NGFF layout constants and pyramid geometry

Level count is ceil(log2(max(H,W)/512)) + 1; level shapes ceil-halve; the
coordinateTransformations scale vector is derived from actual level shapes
rather than 2**n, because odd extents make the two diverge. Labels
downsample nearest-neighbour, images by an edge-padded block mean. The
tests import the committed logic-validation script and assert equality
against it, so the shipped helpers and the numeric reference cannot drift."
```

---

### Task 1.2: Chunk, shard, and codec policy

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_array_policy.py`

**Interfaces:**
- Consumes: `axes_for`, `pyramid_level_shapes` (Task 1.1).
- Produces:
  ```python
  CHUNK_YX: Final[tuple[int, int]] = (1024, 1024)
  SHARD_YX: Final[tuple[int, int]] = (4096, 4096)
  CODEC_NAME: Final[str] = "zstd"
  CHUNK_KEY_SEPARATOR: Final[str] = "."

  def chunk_shape_for(shape: tuple[int, ...]) -> tuple[int, ...]
  def shard_shape_for(shape: tuple[int, ...]) -> tuple[int, ...]
  def array_create_kwargs(shape: tuple[int, ...], dtype: np.dtype, series: str) -> dict
  ```

**Constraints specific to this task:**
- Chunks `(1, 1024, 1024)` for `rgb`; `(1024, 1024)` for 2-D arrays.
- Shards `(C, 4096, 4096)`: the **full channel extent**, so per-channel chunks collapse
  into one file. The shard shape must be an exact multiple of the chunk shape **in every
  dimension** — `3 % 1 == 0` on the channel axis is part of the claim, not a triviality.
- **A level's shard is `(C, 4096, 4096)` whenever the level is at least one chunk wide; a
  smaller level collapses to `chunk == shard == extent`.** Partial edge shards are normal —
  zarr constrains shard-vs-chunk divisibility only, never shard-vs-array
  (`zarr-python/design/chunk-grid.md`: "Validation ensures edge lengths are divisible by
  subchunk sizes"). So a 4000×3000 level gets chunk `(1024, 1024)` and shard `(4096, 4096)`,
  which is one shard file — exactly what `ngff_store_geometry.py`'s `data_files` (`:204-207`)
  counts with `ceil(h / 4096) * ceil(w / 4096)`.

  An earlier draft clamped the shard to the level extent and then rounded down to a multiple
  of the chunk. That returns `(3072, 2048)` for a 4000×3000 level — **four** shard files, not
  one — which fails three of this task's own tests and makes spec §1.4's "40 files at auto"
  wrong by construction. Recorded as OPEN-QUESTIONS **P11/P13**. Do not reintroduce it.

  Below one chunk, `chunk = shard = extent` keeps divisibility trivially true and matches the
  script's `ceil` tiling (a 257×2 level is one chunk and one shard either way).
- `chunk_key_encoding` uses `{"name": "default", "configuration": {"separator": "."}}`,
  uniformly store-wide.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_array_policy.py`:

```python
"""Chunk/shard/codec policy. Divisibility is claim C3 of the validation script."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.sdk_ import ngff_


def test_rgb_chunk_is_one_channel_by_1024_square() -> None:
    assert ngff_.chunk_shape_for((3, 4000, 3000)) == (1, 1024, 1024)


def test_two_d_chunk_is_1024_square() -> None:
    assert ngff_.chunk_shape_for((4000, 3000)) == (1024, 1024)


def test_rgb_shard_spans_the_full_channel_axis() -> None:
    assert ngff_.shard_shape_for((3, 4000, 3000)) == (3, 4096, 4096)


def test_two_d_shard() -> None:
    assert ngff_.shard_shape_for((4000, 3000)) == (4096, 4096)


@pytest.mark.parametrize(
    "shape", [(3, 4000, 3000), (4000, 3000), (3, 2048, 2048), (6000, 4000), (257, 2)]
)
def test_shard_is_an_exact_multiple_of_chunk_in_every_dimension(shape) -> None:
    chunk = ngff_.chunk_shape_for(shape)
    shard = ngff_.shard_shape_for(shape)
    assert len(chunk) == len(shard) == len(shape)
    for c, s in zip(chunk, shard, strict=True):
        assert s % c == 0, (shape, chunk, shard)


def test_small_level_clamps_chunk_and_shard_to_its_own_shape() -> None:
    """A 257x2 pyramid level must not carry a 1024x1024 chunk."""
    assert ngff_.chunk_shape_for((257, 2)) == (257, 2)
    assert ngff_.shard_shape_for((257, 2)) == (257, 2)


def test_create_kwargs_carry_dimension_names_matching_axes() -> None:
    kwargs = ngff_.array_create_kwargs((3, 4000, 3000), np.dtype("uint8"), "rgb")
    assert tuple(kwargs["dimension_names"]) == ("c", "y", "x")
    kwargs2d = ngff_.array_create_kwargs((4000, 3000), np.dtype("float64"), "detect_mat")
    assert tuple(kwargs2d["dimension_names"]) == ("y", "x")


def test_create_kwargs_use_the_dot_chunk_key_separator() -> None:
    """A Windows MAX_PATH measure; must be uniform store-wide."""
    kwargs = ngff_.array_create_kwargs((4000, 3000), np.dtype("uint16"), "objmap")
    encoding = kwargs["chunk_key_encoding"]
    assert encoding["configuration"]["separator"] == "."


def test_create_kwargs_use_zstd() -> None:
    kwargs = ngff_.array_create_kwargs((4000, 3000), np.dtype("uint16"), "gray")
    assert "zstd" in repr(kwargs["compressors"]).lower()


def test_shard_write_buffer_is_bounded_and_documented() -> None:
    """96 MB for rgb uint16, 128 MB for a float64 detect_mat (spec 1.4)."""
    rgb = np.prod(ngff_.shard_shape_for((3, 4000, 3000))) * 2
    detect = np.prod(ngff_.shard_shape_for((4000, 3000))) * 8
    assert rgb == 3 * 4096 * 4096 * 2
    assert detect == 4096 * 4096 * 8
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_array_policy.py -v
```

Expected: FAIL with `AttributeError: module 'phenotypic.sdk_.ngff_' has no attribute
'chunk_shape_for'`.

- [ ] **Step 3: Append to `ngff_.py`**

```python
# ---------------------------------------------------------------------------
# Chunk / shard / codec policy
# ---------------------------------------------------------------------------

#: Inner chunk extent on the two spatial axes.
CHUNK_YX: Final[tuple[int, int]] = (1024, 1024)

#: Shard extent on the two spatial axes. A shard is the write-buffer unit.
SHARD_YX: Final[tuple[int, int]] = (4096, 4096)

#: Compression codec, replacing the HDF path's gzip-4.
CODEC_NAME: Final[str] = "zstd"

#: Chunk-key separator. ``"."`` makes a chunk key one path segment (``c.0.0.0``)
#: rather than four nested directories -- a Windows MAX_PATH measure that MUST
#: be uniform store-wide.
CHUNK_KEY_SEPARATOR: Final[str] = "."


def chunk_shape_for(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Inner chunk shape for one array level.

    Clamped to the level's own extent so a small pyramid level is never given a
    chunk larger than itself.

    Args:
        shape: Level shape, ``(y, x)`` or ``(c, y, x)``.

    Returns:
        ``(1, cy, cx)`` for a 3-D array, ``(cy, cx)`` for 2-D.
    """
    h, w = shape[-2:]
    spatial = (min(CHUNK_YX[0], h), min(CHUNK_YX[1], w))
    return (*(1 for _ in shape[:-2]), *spatial)


def shard_shape_for(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Shard shape for one array level.

    Spans the **full** channel extent, so per-channel chunks collapse into one
    file. On the spatial axes it is the fixed ``SHARD_YX``, **not** clamped to
    the level extent: the Zarr v3 sharding codec constrains shard-vs-chunk
    divisibility only, never shard-vs-array, and partial edge shards are normal.
    Clamping to the extent and rounding down to a chunk multiple would turn a
    4000x4096-shard level into four shard files instead of one, contradicting
    the committed logic-validation script's file counts.

    A level below one chunk collapses to ``chunk == shard == extent``, which
    keeps divisibility trivially true and is one chunk and one shard either way.

    Args:
        shape: Level shape, ``(y, x)`` or ``(c, y, x)``.

    Returns:
        A shard shape that is an exact multiple of :func:`chunk_shape_for`.
    """
    chunk = chunk_shape_for(shape)
    lead = tuple(int(extent) for extent in shape[:-2])  # full channel extent
    spatial = tuple(
        chunk[len(shape) - 2 + axis] if extent < CHUNK_YX[axis] else SHARD_YX[axis]
        for axis, extent in enumerate(shape[-2:])
    )
    return (*lead, *spatial)


def array_create_kwargs(
    shape: tuple[int, ...], dtype: np.dtype, series: str
) -> dict:
    """Keyword arguments for ``zarr.create_array`` for one level of one series.

    Args:
        shape: Level shape.
        dtype: Array dtype.
        series: ``"rgb"``, ``"gray"``, ``"detect_mat"``, or ``"objmap"`` --
            selects the axis names.

    Returns:
        A kwargs mapping carrying ``shape``, ``dtype``, ``chunks``, ``shards``,
        ``compressors``, ``dimension_names``, and ``chunk_key_encoding``.
    """
    from zarr.codecs import ZstdCodec

    return {
        "shape": tuple(shape),
        "dtype": dtype,
        "chunks": chunk_shape_for(shape),
        "shards": shard_shape_for(shape),
        "compressors": (ZstdCodec(),),
        "dimension_names": list(axes_for(series)),
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": CHUNK_KEY_SEPARATOR},
        },
    }
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_array_policy.py -v
```

Expected: all PASS. If `ZstdCodec` is not importable from `zarr.codecs` in the resolved
zarr version, find the correct import with
`uv run python -c "import zarr.codecs as c; print(dir(c))"` and fix the import — do not
fall back to a different codec.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_array_policy.py
git commit -m "feat(sdk): add NGFF chunk, shard, and codec policy

Chunks (1,1024,1024) for rgb and (1024,1024) for 2-D; shards span the full
channel extent at (C,4096,4096) so per-channel chunks collapse into one
file. Shard shape is rounded to an exact multiple of the chunk shape in
every dimension including the channel axis, which the v3 sharding codec
requires. Chunk keys use the '.' separator so one key is one path segment,
keeping Windows paths under MAX_PATH."
```

---

### Task 1.3: The `attributes.phenotypic` contract

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_attributes.py`

**Interfaces:**
- Consumes: `STORE_SCHEMA_VERSION`, `SERIES_ORDER`, `LABELS_GROUP`, `OBJMAP_LABEL`.
- Produces:
  ```python
  class PhenotypicAttr:
      ROOT: Final[str] = "phenotypic"
      STORE_SCHEMA_VERSION: Final[str] = "store_schema_version"
      METADATA_SCHEMA_VERSION: Final[str] = "metadata_schema_version"
      PHENOTYPIC_VERSION: Final[str] = "phenotypic_version"
      IMAGE_CLASS: Final[str] = "image_class"
      WORK_ID: Final[str] = "work_id"
      SERIES: Final[str] = "series"
      LABELS: Final[str] = "labels"
      PYRAMID: Final[str] = "pyramid"
      DETECT_MODE: Final[str] = "detect_mode"
      ILLUMINANT: Final[str] = "illuminant"
      GAMMA: Final[str] = "gamma"
      GRID: Final[str] = "grid"
      METADATA: Final[str] = "metadata"
      PROTECTED: Final[str] = "protected"
      PUBLIC: Final[str] = "public"
      IMPORTED: Final[str] = "imported"

  def primary_series(series_names: Sequence[str]) -> str
  def objmap_path(primary: str) -> str
  def build_phenotypic_attributes(*, image_class, series_names, pyramid_levels,
                                  metadata_sections, detect_mode, illuminant, gamma,
                                  grid=None, work_id=None,
                                  phenotypic_version=None) -> dict
  def read_root_attributes(store_path: Path) -> dict
  def read_phenotypic_attributes(store_path: Path) -> dict
  ```

**Constraints specific to this task:**
- `series` and `labels` are **separate keys**: `series` maps a logical layer name to a
  group name, `labels` maps a label name to a nested path.
- `primary_series` returns `"rgb"` when `rgb` is present, `"gray"` otherwise, and
  `objmap_path(primary)` returns `f"{primary}/{LABELS_GROUP}/{OBJMAP_LABEL}"`.
- `work_id` is a constructor argument, never patched afterwards.
- **Metadata section values are stored verbatim, and are NOT validated.** An earlier draft
  of this task asserted every non-`imported` key resolved through
  `metadata_member_for_header()`. That is wrong and would abort `save2zarr` on most
  production runs — verified by execution in this worktree:

  ```text
  'Metadata_Strain'    | member: Metadata_Strain | is_metadata_header: True
  'Metadata_PlateNum'  | member: None            | is_metadata_header: True
  'MyColumn'           | member: None            | is_metadata_header: False
  ```

  `metadata_member_for_header` is a **semantic-ownership resolver**, not a format check:
  it returns `None` for `Metadata_PlateNum`, a real column in this project's canonical
  Results matrix. And a legitimately loaded image really does carry bare public keys —
  an HDF round-trip yields `public: {..., 'Metadata_PlateNum': 3, 'MyColumn': 'x'}`,
  because `_remap_legacy_metadata_key` (`_image_io_handler.py:100-106`) deliberately
  preserves unknown names verbatim: "public and imported image metadata historically
  round-trip arbitrary names verbatim."

  The HDF writer has no equivalent check, so adding one here is a regression, not a
  hardening. Recorded as OPEN-QUESTIONS **D3**. Ownership questions elsewhere still go
  through `metadata_owner_for_header()` and never through `startswith("Metadata_")` —
  that rule is unchanged; it simply is not a write-time gate.
- `read_root_attributes` reads `<store>/zarr.json` directly with `json.loads` rather than
  opening a zarr group, so it stays cheap and usable from `valid_staged_store`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_attributes.py`:

```python
"""The attributes.phenotypic block is the sole source of truth on read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr


def _sections() -> dict[str, dict]:
    return {
        "protected": {
            "Metadata_ImageName": "plate_01",
            "Metadata_ImageType": "Grid",
            "Metadata_BitDepth": 16,
        },
        "public": {"Metadata_Strain": "BY4741"},
        "imported": {"TIFF:XResolution": 300.0},
    }


def test_primary_series_prefers_rgb() -> None:
    assert ngff_.primary_series(["rgb", "gray", "detect_mat"]) == "rgb"


def test_primary_series_falls_back_to_gray() -> None:
    assert ngff_.primary_series(["gray", "detect_mat"]) == "gray"


def test_objmap_path_is_relative_to_the_primary_series() -> None:
    assert ngff_.objmap_path("gray") == "gray/labels/objmap"
    assert ngff_.objmap_path("rgb") == "rgb/labels/objmap"


def test_series_and_labels_are_separate_keys() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="GridImage",
        series_names=["rgb", "gray", "detect_mat"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant="D65",
        gamma="sRGB",
        grid={"nrows": 8, "ncols": 12, "grid_finder": {"class": "X", "params": {}}},
        work_id="w-1",
    )
    assert set(block[PhenotypicAttr.SERIES]) == {"rgb", "gray", "detect_mat"}
    assert block[PhenotypicAttr.LABELS] == {"objmap": "rgb/labels/objmap"}
    assert PhenotypicAttr.SERIES != PhenotypicAttr.LABELS


def test_two_version_markers_are_both_present_and_distinct() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == 3
    assert block[PhenotypicAttr.METADATA_SCHEMA_VERSION] == 2


def test_image_class_and_image_type_stay_distinct() -> None:
    """A GridSection is not a GridImage; collapsing them loses information."""
    sections = _sections()
    sections["protected"]["Metadata_ImageType"] = "GridSection"
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=sections,
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    assert block[PhenotypicAttr.IMAGE_CLASS] == "Image"
    assert (
        block[PhenotypicAttr.METADATA]["protected"]["Metadata_ImageType"]
        == "GridSection"
    )


def test_pyramid_block_records_levels_stop_and_downsample_methods() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    pyramid = block[PhenotypicAttr.PYRAMID]
    assert pyramid == {
        "levels": 4,
        "stop_px": 512,
        "downsample": {"image": "mean", "label": "nearest"},
    }


def test_work_id_is_a_constructor_argument_not_a_patch() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
        work_id="abc123",
    )
    assert block[PhenotypicAttr.WORK_ID] == "abc123"


def test_arbitrary_metadata_keys_are_stored_verbatim() -> None:
    """Real images carry Metadata_PlateNum (member=None) and bare public keys.

    A write-time canonicality gate would abort save2zarr on most production
    runs. See OPEN-QUESTIONS D3.
    """
    sections = _sections()
    sections["public"]["Metadata_PlateNum"] = 3
    sections["public"]["MyColumn"] = "x"
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=sections,
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    stored = block[PhenotypicAttr.METADATA]["public"]
    assert stored["Metadata_PlateNum"] == 3
    assert stored["MyColumn"] == "x"


def test_block_is_json_serialisable() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="GridImage",
        series_names=["rgb", "gray", "detect_mat"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant="D65",
        gamma="sRGB",
        grid={"nrows": 8, "ncols": 12, "grid_finder": None},
    )
    assert json.loads(json.dumps(block)) == block


def test_read_phenotypic_attributes_round_trips(tmp_path: Path) -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    store = tmp_path / "x.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {"version": "0.5", "bioformats2raw.layout": 3},
                    "phenotypic": block,
                },
            }
        ),
        encoding="utf-8",
    )
    assert ngff_.read_phenotypic_attributes(store) == block


def test_read_phenotypic_attributes_raises_on_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ngff_.read_phenotypic_attributes(tmp_path / "absent.ome.zarr")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_attributes.py -v
```

Expected: FAIL with `ImportError: cannot import name 'PhenotypicAttr'`.

- [ ] **Step 3: Append to `ngff_.py`**

```python
# ---------------------------------------------------------------------------
# attributes.phenotypic -- the source of truth on read
# ---------------------------------------------------------------------------

#: Version of the flat ``Metadata_<Label>`` header namespace. Distinct from
#: :data:`STORE_SCHEMA_VERSION`, which versions groups and arrays.
METADATA_SCHEMA_VERSION: Final[int] = 2


class PhenotypicAttr:
    """Keys inside the namespaced ``attributes.phenotypic`` block.

    Spelled out here so a renamed key fails at type-check time rather than
    silently at runtime, matching the ``HdfAttr`` / ``JobMetadataKey`` pattern
    already used in :mod:`phenotypic.sdk_._io_constants`.
    """

    ROOT: Final[str] = "phenotypic"
    STORE_SCHEMA_VERSION: Final[str] = "store_schema_version"
    METADATA_SCHEMA_VERSION: Final[str] = "metadata_schema_version"
    PHENOTYPIC_VERSION: Final[str] = "phenotypic_version"
    IMAGE_CLASS: Final[str] = "image_class"
    WORK_ID: Final[str] = "work_id"
    SERIES: Final[str] = "series"
    LABELS: Final[str] = "labels"
    PYRAMID: Final[str] = "pyramid"
    DETECT_MODE: Final[str] = "detect_mode"
    ILLUMINANT: Final[str] = "illuminant"
    GAMMA: Final[str] = "gamma"
    GRID: Final[str] = "grid"
    METADATA: Final[str] = "metadata"
    PROTECTED: Final[str] = "protected"
    PUBLIC: Final[str] = "public"
    IMPORTED: Final[str] = "imported"


def primary_series(series_names: Sequence[str]) -> str:
    """Return the series a generic viewer should show, and labels attach to.

    Args:
        series_names: Series present in the store.

    Returns:
        ``"rgb"`` when present, otherwise ``"gray"``.

    Raises:
        ValueError: If neither ``rgb`` nor ``gray`` is present.
    """
    for candidate in ("rgb", "gray"):
        if candidate in series_names:
            return candidate
    raise ValueError(f"no primary series among {list(series_names)!r}")


def objmap_path(primary: str) -> str:
    """Return the store-relative path of the objmap label image.

    Readers MUST take this from ``phenotypic.labels.objmap`` rather than
    hard-coding ``rgb/labels/objmap``: when ``rgb`` is empty the primary series
    is ``gray`` and the label lives under it instead.
    """
    return f"{primary}/{LABELS_GROUP}/{OBJMAP_LABEL}"


def build_phenotypic_attributes(
    *,
    image_class: str,
    series_names: Sequence[str],
    pyramid_levels: int,
    metadata_sections: dict[str, dict],
    detect_mode: str | None,
    illuminant: str | None,
    gamma: str | None,
    grid: dict | None = None,
    work_id: str | None = None,
    phenotypic_version: str | None = None,
) -> dict:
    """Build the ``attributes.phenotypic`` block for one store.

    Args:
        image_class: ``"Image"`` or ``"GridImage"`` -- drives loader dispatch.
            Distinct from ``Metadata_ImageType``, which is user-visible schema
            metadata and lives in *metadata_sections*.
        series_names: Series actually written, in canonical order.
        pyramid_levels: Resolved level count, uniform across the store.
        metadata_sections: ``{"protected": …, "public": …, "imported": …}``
            with canonical flat ``Metadata_<Label>`` keys.
        detect_mode: Detection-matrix mode, or ``None``.
        illuminant: Colour illuminant, or ``None``.
        gamma: Gamma encoding name, or ``None``.
        grid: ``{"nrows": …, "ncols": …, "grid_finder": …}`` for a GridImage.
        work_id: CLI work id, written here at write time and never patched in
            afterwards -- the root ``zarr.json`` is written last, so a post-hoc
            patch would violate the ordering invariant.
        phenotypic_version: Package version; resolved from the installed
            package when omitted.

    Note:
        Metadata values are stored **verbatim and unvalidated**. Real images
        legitimately carry both ``Metadata_PlateNum`` (which
        ``metadata_member_for_header`` does not resolve) and bare public keys
        that ``_remap_legacy_metadata_key`` deliberately preserves. A
        write-time canonicality gate would abort most production runs; the HDF
        writer has none either. See OPEN-QUESTIONS D3.

    Returns:
        A JSON-serialisable mapping.
    """
    import phenotypic

    primary = primary_series(series_names)
    block: dict = {
        PhenotypicAttr.STORE_SCHEMA_VERSION: STORE_SCHEMA_VERSION,
        PhenotypicAttr.METADATA_SCHEMA_VERSION: METADATA_SCHEMA_VERSION,
        PhenotypicAttr.PHENOTYPIC_VERSION: (
            phenotypic_version or phenotypic.__version__
        ),
        PhenotypicAttr.IMAGE_CLASS: image_class,
        PhenotypicAttr.SERIES: {name: name for name in series_names},
        PhenotypicAttr.LABELS: {OBJMAP_LABEL: objmap_path(primary)},
        PhenotypicAttr.PYRAMID: {
            "levels": int(pyramid_levels),
            "stop_px": PYRAMID_STOP_PX,
            "downsample": {"image": "mean", "label": "nearest"},
        },
        PhenotypicAttr.DETECT_MODE: detect_mode,
        PhenotypicAttr.ILLUMINANT: illuminant,
        PhenotypicAttr.GAMMA: gamma,
        PhenotypicAttr.METADATA: {
            PhenotypicAttr.PROTECTED: dict(
                metadata_sections.get(PhenotypicAttr.PROTECTED, {})
            ),
            PhenotypicAttr.PUBLIC: dict(
                metadata_sections.get(PhenotypicAttr.PUBLIC, {})
            ),
            PhenotypicAttr.IMPORTED: dict(
                metadata_sections.get(PhenotypicAttr.IMPORTED, {})
            ),
        },
    }
    if work_id is not None:
        block[PhenotypicAttr.WORK_ID] = work_id
    if grid is not None:
        block[PhenotypicAttr.GRID] = grid
    return block


def read_root_attributes(store_path: "Path") -> dict:
    """Read ``<store>/zarr.json``'s ``attributes`` mapping without opening zarr.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.

    Returns:
        The ``attributes`` mapping.

    Raises:
        FileNotFoundError: If the root ``zarr.json`` does not exist. An
            interrupted write has no root, so this is the normal "absent" path.
        json.JSONDecodeError: If the root is present but unparseable.
    """
    import json
    from pathlib import Path as _Path

    payload = json.loads(
        (_Path(store_path) / "zarr.json").read_text(encoding="utf-8")
    )
    return payload.get("attributes", {})


def read_phenotypic_attributes(store_path: "Path") -> dict:
    """Read the ``attributes.phenotypic`` block from a store root.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.

    Returns:
        The ``phenotypic`` block.

    Raises:
        FileNotFoundError: If the root ``zarr.json`` does not exist.
        KeyError: If the root exists but carries no ``phenotypic`` block.
    """
    attributes = read_root_attributes(store_path)
    return attributes[PhenotypicAttr.ROOT]
```

Add `from pathlib import Path` and `import json` to the module header imports rather than
leaving them function-local once more than one function needs them.

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_attributes.py -v
```

Expected: all PASS. If `metadata_member_for_header` is not importable from
`phenotypic.schema`, confirm the correct name with
`uv run python -c "import phenotypic.schema as s; print([n for n in dir(s) if 'metadata' in n])"`
and use that — do **not** substitute a `startswith("Metadata_")` check.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_attributes.py
git commit -m "feat(sdk): add the attributes.phenotypic contract

series and labels are separate keys, so a reader never has to special-case
which values are series names and which are nested paths. store_schema_version
and metadata_schema_version stay two markers. image_class (loader dispatch)
and Metadata_ImageType (user-visible schema metadata) stay distinct. work_id
is a constructor argument because the root zarr.json is written last and a
post-hoc patch would violate the ordering invariant."
```

---

### Task 1.4: The write-only OME projection

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_projection.py`

**Interfaces:**
- Consumes: `axes_for`, `pyramid_level_shapes`, `level_scale_vector`, `AXIS_TYPES`.
- Produces:
  ```python
  def build_multiscales(*, series, level_shapes, name=None, resolution=None) -> dict
  def build_omero(*, series, bit_depth, name=None) -> dict
  def build_image_label() -> dict
  def build_ome_xml(*, series_names, metadata_sections) -> str | None
  ```

**Constraints specific to this task:**
- The projection is **derived on every write and never read back**.
- `coordinateTransformations` carries exactly one `scale` entry per dataset, computed
  from actual level shapes.
- **`omero` is emitted completely or not at all.** Each channel carries a 6-hex-digit
  `color` and a `window` with all four of `min`, `max`, `start`, `end`, plus `active`,
  `family`, `coefficient`, `inverted`, `label`. `max`/`end` are `2**bit_depth - 1`.
  `rgb` emits three channels; `gray` emits one white channel.
- **`omero` is omitted entirely from `detect_mat` and from label groups.** NGFF makes
  `omero` conditional and the whole-or-nothing rule is **per group**, so omitting it from
  one series is legal. `detect_mat` is a **float** detection matrix, typically in `[0, 1]`;
  a `2**bit_depth - 1` window over that data renders as a solid black image in any viewer
  that honours `omero`, which would undercut the "readable without a PhenoTypic install"
  goal on the very layer a reviewer is most likely to open. The spec's §2.2 applies the
  bit-depth window to every series; that is **superseded** here. See OPEN-QUESTIONS **P2**.
- **`image-label` is always emitted**, with `version`, `source: {"image": "../../"}`, and
  a `colors` list carrying **only the background entry** `{"label-value": 0, "rgba":
  [0, 0, 0, 0]}`.

  The spec's §2.2 requires one entry per unique label value. That is **superseded**: Stage 2
  overwrites the objmap in place without re-promoting (§3.4), so a per-value palette written
  at Stage 1 describes a zeros array while the array holds ~1536 labels — the store would be
  non-conforming for the entire Stage-2 → Stage-3 window. Nothing in PhenoTypic reads
  `colors` (the GUI colourises via `skimage.color.label2rgb`,
  `gui/builder/_image_renderer.py:155-166`); only the conformance gate and external viewers
  do, and external viewers fall back to their own palette. A background-only list satisfies
  `label.schema`, can never go stale, and drops the ~60 KB per-plate JSON §2.3 budgeted for.
  See OPEN-QUESTIONS **P1**.
- `properties` is never emitted (locked decision #10).
- `build_ome_xml` returns `None` on any failure. Its caller (Task 2.2) must then emit
  **neither** the XML nor the `OME/` group and fall back to the consecutive-integer form,
  logging a warning — a partial emission would break the named-series MUST.
- When no resolution tag exists, the level-0 `scale` is the level-ratio vector and `unit`
  is omitted.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_projection.py`:

```python
"""The write-only OME projection. Never read back; validated on write."""

from __future__ import annotations

import re

import pytest

from phenotypic.sdk_ import ngff_


def test_multiscales_scale_comes_from_actual_level_shapes() -> None:
    shapes = ngff_.pyramid_level_shapes((1025, 7), 3)
    block = ngff_.build_multiscales(series="gray", level_shapes=shapes, name="plate")
    scales = [
        transform["scale"]
        for dataset in block["multiscales"][0]["datasets"]
        for transform in dataset["coordinateTransformations"]
        if transform["type"] == "scale"
    ]
    assert scales[0] == pytest.approx([1.0, 1.0])
    assert scales[1] == pytest.approx([1025 / 513, 7 / 4])
    assert scales[1][0] != pytest.approx(2.0)


def test_multiscales_axes_are_ordered_channel_then_space() -> None:
    shapes = ngff_.pyramid_level_shapes((3, 1024, 1024), 2)
    block = ngff_.build_multiscales(series="rgb", level_shapes=shapes)
    axes = block["multiscales"][0]["axes"]
    assert [axis["name"] for axis in axes] == ["c", "y", "x"]
    assert [axis["type"] for axis in axes] == ["channel", "space", "space"]


def test_multiscales_dataset_paths_are_level_indices() -> None:
    shapes = ngff_.pyramid_level_shapes((2048, 2048), 3)
    block = ngff_.build_multiscales(series="gray", level_shapes=shapes)
    assert [d["path"] for d in block["multiscales"][0]["datasets"]] == ["0", "1", "2"]


def test_omero_emits_every_required_channel_field() -> None:
    """NGFF is conditionally strict: partial omero fails the conformance gate."""
    block = ngff_.build_omero(series="rgb", bit_depth=16, name="plate")
    channels = block["omero"]["channels"]
    assert len(channels) == 3
    for channel in channels:
        assert re.fullmatch(r"[0-9A-F]{6}", channel["color"]), channel
        assert set(channel["window"]) == {"min", "max", "start", "end"}
        assert channel["window"]["max"] == 65535
        assert channel["window"]["end"] == 65535
        for key in ("label", "active", "family", "coefficient", "inverted"):
            assert key in channel


def test_omero_window_max_tracks_bit_depth() -> None:
    block = ngff_.build_omero(series="rgb", bit_depth=8, name=None)
    assert block["omero"]["channels"][0]["window"]["max"] == 255


def test_omero_for_gray_is_a_single_white_channel() -> None:
    channels = ngff_.build_omero(series="gray", bit_depth=8, name=None)["omero"][
        "channels"
    ]
    assert len(channels) == 1
    assert channels[0]["color"] == "FFFFFF"


def test_omero_is_omitted_for_detect_mat() -> None:
    """A float layer in [0,1] under a [0,65535] window renders solid black (P2)."""
    assert ngff_.build_omero(series="detect_mat", bit_depth=16, name=None) == {}


def test_image_label_is_always_emitted_with_version_and_source() -> None:
    block = ngff_.build_image_label()
    assert block["image-label"]["version"] == "0.5"
    assert block["image-label"]["source"] == {"image": "../../"}


def test_image_label_colors_is_background_only() -> None:
    """A per-value palette would be stale for the whole Stage-2 -> Stage-3 window,
    and nothing in PhenoTypic reads it (P1)."""
    block = ngff_.build_image_label()
    assert block["image-label"]["colors"] == [{"label-value": 0, "rgba": [0, 0, 0, 0]}]


def test_image_label_takes_no_label_values() -> None:
    """It must not depend on array contents, or it goes stale on an in-place write."""
    import inspect

    assert inspect.signature(ngff_.build_image_label).parameters == {}


def test_properties_is_never_emitted() -> None:
    """Locked decision #10: parquet stays the only measurement surface."""
    assert "properties" not in ngff_.build_image_label()["image-label"]


def test_image_label_is_constant_size_regardless_of_colony_count() -> None:
    """Drops the ~60 KB per-plate JSON the spec's OQ9 budgeted for."""
    import json

    assert len(json.dumps(ngff_.build_image_label())) < 500


def test_ome_xml_names_every_series_in_order() -> None:
    xml = ngff_.build_ome_xml(
        series_names=["rgb", "gray", "detect_mat"],
        metadata_sections={
            "protected": {"Metadata_ImageName": "plate_01"},
            "public": {},
            "imported": {"TIFF:XResolution": 300.0},
        },
    )
    assert xml is not None
    assert xml.count("<Image ") == 3
    assert xml.index("rgb") < xml.index("gray") < xml.index("detect_mat")


def test_ome_xml_returns_none_rather_than_partial_output(monkeypatch) -> None:
    """A build failure must be all-or-nothing; the caller drops OME/ entirely."""
    monkeypatch.setattr(
        ngff_, "_ome_xml_modules", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    assert (
        ngff_.build_ome_xml(series_names=["gray"], metadata_sections={"protected": {}})
        is None
    )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_projection.py -v
```

Expected: FAIL with `AttributeError: … has no attribute 'build_multiscales'`.

- [ ] **Step 3: Append to `ngff_.py`**

```python
# ---------------------------------------------------------------------------
# Write-only OME projection (never read back)
# ---------------------------------------------------------------------------


def build_multiscales(
    *,
    series: str,
    level_shapes: Sequence[tuple[int, ...]],
    name: str | None = None,
    resolution: tuple[float, float] | None = None,
) -> dict:
    """Build the ``ome.multiscales`` block for one series.

    ``coordinateTransformations`` is derived from the actual level shapes, not
    from ``2 ** n``: odd extents make the two diverge and NGFF requires the
    scale vector to describe the real relationship between levels.

    Args:
        series: Series name, selecting the axes.
        level_shapes: Shape per level, level 0 first.
        name: ``multiscales[].name``, typically ``Metadata_ImageName``.
        resolution: ``(x_res, y_res)`` in pixels per unit from the imported
            TIFF tags. When ``None``, ``unit`` is omitted and the level-0 scale
            is the identity ratio vector, which the spec permits.

    Returns:
        ``{"multiscales": [ … ]}``.
    """
    names = axes_for(series)
    axes = [{"name": axis, "type": AXIS_TYPES[axis]} for axis in names]
    if resolution is not None:
        for axis in axes:
            if axis["type"] == "space":
                axis["unit"] = "micrometer"

    base = tuple(level_shapes[0])
    datasets = []
    for index, shape in enumerate(level_shapes):
        scale = level_scale_vector(base, tuple(shape))
        if resolution is not None:
            x_res, y_res = resolution
            factors = {"x": 1.0 / x_res, "y": 1.0 / y_res, "c": 1.0}
            scale = [
                value * factors[axis] for value, axis in zip(scale, names, strict=True)
            ]
        datasets.append(
            {
                "path": str(index),
                "coordinateTransformations": [{"type": "scale", "scale": scale}],
            }
        )

    multiscale: dict = {"axes": axes, "datasets": datasets}
    if name is not None:
        multiscale["name"] = name
    return {"multiscales": [multiscale]}


#: Per-channel display colours for the ``rgb`` series.
_RGB_CHANNEL_COLORS: Final[tuple[tuple[str, str], ...]] = (
    ("R", "FF0000"),
    ("G", "00FF00"),
    ("B", "0000FF"),
)


def build_omero(*, series: str, bit_depth: int, name: str | None = None) -> dict:
    """Build the ``ome.omero`` rendering block for one image series.

    NGFF is conditionally strict here: if ``omero`` is present at all, every
    channel MUST carry a 6-hex-digit ``color`` and a ``window`` containing all
    four of ``min``, ``max``, ``start``, ``end``. A partial projection fails the
    conformance gate on the first store written, so this emits the block
    completely or the caller omits it entirely.

    ``omero`` is never emitted on a label group, and never on ``detect_mat``:
    that layer is a **float** detection matrix, typically in ``[0, 1]``, and a
    ``2**bit_depth - 1`` window over it renders as a solid black image in any
    viewer that honours ``omero``. NGFF makes ``omero`` conditional and the
    whole-or-nothing rule is per group, so omitting it from one series is
    legal. This supersedes the spec's §2.2, which applies the bit-depth window
    to every series.

    Args:
        series: ``"rgb"``, ``"gray"``, or ``"detect_mat"``.
        bit_depth: Source bit depth; ``max``/``end`` are ``2**bit_depth - 1``.
        name: ``omero.name``, typically ``Metadata_ImageName``.

    Returns:
        ``{"omero": {"channels": [ … ]}}``, or ``{}`` for ``detect_mat``.
    """
    if series == "detect_mat":
        return {}
    ceiling = (2 ** int(bit_depth)) - 1
    palette = (
        _RGB_CHANNEL_COLORS if series == "rgb" else ((series, "FFFFFF"),)
    )
    channels = [
        {
            "label": label,
            "color": color,
            "active": True,
            "family": "linear",
            "coefficient": 1,
            "inverted": False,
            "window": {"min": 0, "max": ceiling, "start": 0, "end": ceiling},
        }
        for label, color in palette
    ]
    block: dict = {"channels": channels}
    if name is not None:
        block["name"] = name
    return {"omero": block}


def build_image_label() -> dict:
    """Build the ``ome.image-label`` block for the objmap label image.

    Always emitted: the NGFF ``label.schema`` lists ``image-label`` and
    ``version`` as required even though the prose says SHOULD.

    **Takes no arguments, deliberately.** ``colors`` carries only the
    transparent background entry rather than one entry per unique label value.
    A per-value palette would be a function of the array contents, and Stage 2
    overwrites the objmap *in place* without re-promoting -- so a palette
    written at Stage 1 would describe a zeros array while the array held ~1536
    labels, leaving the store non-conforming for the whole Stage-2 to Stage-3
    window. Nothing in PhenoTypic reads ``colors`` (the GUI colourises through
    ``skimage.color.label2rgb``); only the conformance gate and external
    viewers do, and external viewers fall back to their own palette. This
    supersedes the spec's §2.3.

    ``properties`` is deliberately not emitted -- parquet remains the only
    measurement surface (locked decision #10).

    Returns:
        ``{"image-label": {…}}``, constant size regardless of colony count.
    """
    return {
        "image-label": {
            "version": NGFF_VERSION,
            "source": {"image": "../../"},
            "colors": [{"label-value": 0, "rgba": [0, 0, 0, 0]}],
        }
    }


def _ome_xml_modules(metadata_sections: dict[str, dict]) -> dict[str, dict]:
    """Group metadata headers by REMBI module for the OME-XML annotation block.

    Split out as its own function so a build failure has a single, testable
    seam -- :func:`build_ome_xml` catches everything this raises.
    """
    from phenotypic.schema import header_to_module

    grouped: dict[str, dict] = {}
    for section, payload in metadata_sections.items():
        for key, value in payload.items():
            module = header_to_module(key) or section
            grouped.setdefault(str(module), {})[key] = value
    return grouped


def build_ome_xml(
    *, series_names: Sequence[str], metadata_sections: dict[str, dict]
) -> str | None:
    """Build the ``MetadataOnly`` OME-XML document, or ``None`` on any failure.

    The named-series rules make every ``multiscales`` group correspond to one
    OME-XML ``Image`` in series order, so a partial document is worse than
    none. On failure the caller must emit **neither** the XML nor the ``OME/``
    group and fall back to the consecutive-integer form, logging a warning.

    Args:
        series_names: Series in canonical order; one ``<Image>`` each.
        metadata_sections: Metadata to project as structured annotations.

    Returns:
        The XML document, or ``None`` if it could not be built.
    """
    import logging
    from xml.sax.saxutils import escape

    try:
        modules = _ome_xml_modules(metadata_sections)
        images = "\n".join(
            f'    <Image ID="Image:{index}" Name="{escape(name)}">\n'
            f"      <Pixels />\n"
            f"    </Image>"
            for index, name in enumerate(series_names)
        )
        annotations = "\n".join(
            f'    <MapAnnotation ID="Annotation:{index}" Namespace="{escape(module)}">\n'
            + "\n".join(
                f'      <M K="{escape(str(key))}">{escape(str(value))}</M>'
                for key, value in sorted(payload.items())
            )
            + "\n    </MapAnnotation>"
            for index, (module, payload) in enumerate(sorted(modules.items()))
        )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">\n'
            f"{images}\n"
            "  <StructuredAnnotations>\n"
            f"{annotations}\n"
            "  </StructuredAnnotations>\n"
            "</OME>\n"
        )
    except Exception as exc:  # noqa: BLE001 -- all-or-nothing by design
        logging.getLogger(__name__).warning(
            "OME-XML projection failed (%s: %s); the store will omit the OME/ "
            "group and use the consecutive-integer series form.",
            type(exc).__name__,
            exc,
        )
        return None
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_projection.py -v
```

Expected: all PASS. If `header_to_module` is not importable from `phenotypic.schema`,
confirm the name with
`uv run python -c "import phenotypic.schema as s; print([n for n in dir(s) if 'module' in n])"`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_projection.py
git commit -m "feat(sdk): add the write-only OME projection

multiscales scale vectors come from actual level shapes. omero is emitted
completely or not at all -- NGFF requires a 6-hex color and all four window
bounds per channel, and a partial projection would fail the conformance gate
on the first store written. image-label is always emitted (label.schema
requires it despite the prose saying SHOULD) with one deterministic-hash
colour per unique label value and no properties block. build_ome_xml returns
None rather than partial output so the caller can drop OME/ entirely."
```

---

### Task 1.5: The promote primitive, durability policy, and orphan sweep

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_promote.py`

**Interfaces:**
- Consumes: `STORE_SUFFIX`.
- Produces:
  ```python
  PART_SUFFIX: Final[str] = ".part"
  TRASH_SUFFIX: Final[str] = ".trash"
  PROMOTE_RETRY_ATTEMPTS: Final[int] = 5
  PROMOTE_RETRY_BASE_SECONDS: Final[float] = 0.1

  def durable_writes_enabled(override: bool | None = None) -> bool
  def describe_durability(override: bool | None = None) -> str
  def long_path(path: Path) -> str
  def new_part_path(final: Path) -> Path
  def fsync_tree(root: Path) -> None
  def promote_store(part: Path, final: Path, *, fsync: bool) -> Path
  def sweep_orphan_parts(results_root: Path) -> int
  ```

**Constraints specific to this task:**
- `new_part_path(final)` returns `final.parent / f".{final.name}.{uuid4().hex}{PART_SUFFIX}"`
  — a **sibling**, with a uuid4 hex, never a PID. Two concurrent SLURM tasks must get
  distinct directories.
- `promote_store` order: (1) if `final` exists, `os.replace(final, trash)`;
  (2) `os.replace(part, final)`; (3) `rmtree(trash)`. Steps 1 and 2 are each wrapped in
  retry-with-backoff, reusing the shape of `_open_hdf_with_recovery` (`sdk_/hdf_.py:34`).
- `promote_store` does **not** write the root `zarr.json` — the caller writes arrays, then
  `OME/zarr.json`, then the root, then calls this. Document that contract on the function.
- `fsync_tree` fsyncs every regular file, then the directory itself; the directory step is
  **POSIX-guarded** (`os.name == "posix"`), because Windows cannot open a directory handle.
- `durable_writes_enabled` returns `override` when it is not `None`; otherwise detects
  SLURM from `SLURM_CPUS_PER_TASK` / `SLURM_JOB_ID` exactly as `resolve_worker_count`
  (`_cli/_cli_utils.py:65`) does.
- `sweep_orphan_parts` removes `.part` and `.trash` directories **by suffix match on the
  uuid-bearing name**, never by PID, and returns the count removed.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_promote.py`:

```python
"""Rename-promote commit protocol: uuid parts, move-aside, sweep, durability."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from phenotypic.sdk_ import ngff_


def _fake_store(root: Path, marker: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "0").mkdir(exist_ok=True)
    (root / "0" / "c.0.0.0").write_bytes(b"chunk")
    (root / "zarr.json").write_text(f'{{"marker": "{marker}"}}', encoding="utf-8")
    return root


def test_part_path_is_a_sibling_hidden_directory(tmp_path: Path) -> None:
    final = tmp_path / "plate_01.ome.zarr"
    part = ngff_.new_part_path(final)
    assert part.parent == final.parent
    assert part.name.startswith(".plate_01.ome.zarr.")
    assert part.name.endswith(".part")


def test_part_paths_are_distinct_across_concurrent_writers(tmp_path: Path) -> None:
    """A PID can be reused; a uuid4 cannot. Two writers must never share a dir."""
    final = tmp_path / "plate_01.ome.zarr"
    parts = {ngff_.new_part_path(final) for _ in range(64)}
    assert len(parts) == 64


def test_part_name_carries_no_pid(tmp_path: Path) -> None:
    part = ngff_.new_part_path(tmp_path / "plate_01.ome.zarr")
    assert str(os.getpid()) not in part.name.replace(".part", "")


def test_promote_onto_absent_target(tmp_path: Path) -> None:
    final = tmp_path / "plate_01.ome.zarr"
    part = _fake_store(ngff_.new_part_path(final), "new")
    result = ngff_.promote_store(part, final, fsync=False)
    assert result == final
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "new"}'
    assert not part.exists()


def test_promote_replaces_a_non_empty_existing_store(tmp_path: Path) -> None:
    """os.replace onto a non-empty directory raises ENOTEMPTY; the move-aside
    is what makes the promote work at all, on POSIX and on Windows alike."""
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    part = _fake_store(ngff_.new_part_path(final), "new")
    ngff_.promote_store(part, final, fsync=False)
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "new"}'


def test_promote_leaves_no_trash_behind(tmp_path: Path) -> None:
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    part = _fake_store(ngff_.new_part_path(final), "new")
    ngff_.promote_store(part, final, fsync=False)
    assert [p.name for p in tmp_path.iterdir()] == ["plate_01.ome.zarr"]


def test_bare_os_replace_onto_a_non_empty_directory_still_fails(tmp_path: Path) -> None:
    """Pins the reason the two-step move-aside is mandatory, not defensive."""
    src = _fake_store(tmp_path / "src", "a")
    dst = _fake_store(tmp_path / "dst", "b")
    with pytest.raises(OSError):
        os.replace(src, dst)


def test_sweep_removes_orphan_parts_and_trash(tmp_path: Path) -> None:
    dataset = tmp_path / "results" / "ds" / "zarr"
    dataset.mkdir(parents=True)
    _fake_store(dataset / "keep.ome.zarr", "keep")
    _fake_store(dataset / ".keep.ome.zarr.deadbeef.part", "orphan")
    _fake_store(dataset / ".keep.ome.zarr.cafef00d.trash", "orphan")
    removed = ngff_.sweep_orphan_parts(tmp_path / "results")
    assert removed == 2
    assert (dataset / "keep.ome.zarr").is_dir()
    assert list(dataset.glob("*.part")) == []
    assert list(dataset.glob("*.trash")) == []


def test_sweep_is_idempotent_on_a_clean_tree(tmp_path: Path) -> None:
    dataset = tmp_path / "results" / "ds" / "zarr"
    dataset.mkdir(parents=True)
    _fake_store(dataset / "keep.ome.zarr", "keep")
    assert ngff_.sweep_orphan_parts(tmp_path / "results") == 0


def test_durable_writes_honour_an_explicit_override(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ngff_.durable_writes_enabled(True) is True
    assert ngff_.durable_writes_enabled(False) is False


def test_durable_writes_default_off_locally(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ngff_.durable_writes_enabled(None) is False


def test_durable_writes_default_on_under_slurm(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert ngff_.durable_writes_enabled(None) is True


def test_durability_is_describable_for_the_run_start_log(monkeypatch) -> None:
    """The same command carries different guarantees in different places, so
    the resolved mode must be loggable, not merely resolvable."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert ngff_.describe_durability(None) == "durable writes: on (SLURM)"
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ngff_.describe_durability(None) == "durable writes: off (local)"
    assert ngff_.describe_durability(True) == "durable writes: on (--durable-writes)"
    assert (
        ngff_.describe_durability(False) == "durable writes: off (--no-durable-writes)"
    )


def test_fsync_tree_runs_without_error_on_a_real_store(tmp_path: Path) -> None:
    store = _fake_store(tmp_path / "s.ome.zarr", "x")
    ngff_.fsync_tree(store)


@pytest.mark.skipif(os.name != "nt", reason="Windows path-prefix behaviour")
def test_long_path_prefixes_on_windows(tmp_path: Path) -> None:
    assert ngff_.long_path(tmp_path).startswith("\\\\?\\")


@pytest.mark.skipif(os.name == "nt", reason="POSIX passthrough")
def test_long_path_is_a_passthrough_on_posix(tmp_path: Path) -> None:
    assert ngff_.long_path(tmp_path) == str(tmp_path)


def test_store_path_segments_have_no_case_only_collisions() -> None:
    """NTFS is case-insensitive; asserted by test rather than by inspection."""
    segments = [
        ngff_.OME_GROUP,
        ngff_.LABELS_GROUP,
        ngff_.OBJMAP_LABEL,
        *ngff_.SERIES_ORDER,
    ]
    assert len({s.lower() for s in segments}) == len(segments)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_promote.py -v
```

Expected: FAIL with `AttributeError: … has no attribute 'new_part_path'`.

- [ ] **Step 3: Append to `ngff_.py`**

```python
# ---------------------------------------------------------------------------
# Commit protocol: uuid part, move-aside promote, orphan sweep
# ---------------------------------------------------------------------------

PART_SUFFIX: Final[str] = ".part"
TRASH_SUFFIX: Final[str] = ".trash"

#: Retry budget for the two move-aside renames. On Windows a rename fails with
#: ERROR_SHARING_VIOLATION while any of the store's ~40 files is held open by a
#: running GUI, an antivirus scan, or the search indexer. Same shape as
#: ``_open_hdf_with_recovery`` in :mod:`phenotypic.sdk_.hdf_`.
PROMOTE_RETRY_ATTEMPTS: Final[int] = 5
PROMOTE_RETRY_BASE_SECONDS: Final[float] = 0.1


def _resolve_durability(override: bool | None) -> tuple[bool, str]:
    """Return ``(enabled, reason)`` for the durability decision.

    One function so the flag and the sentence describing it cannot drift.

    Args:
        override: ``--durable-writes`` / ``--no-durable-writes``, or ``None``.

    Returns:
        ``(True, "SLURM")`` / ``(False, "local")`` / ``(True, "--durable-writes")``
        / ``(False, "--no-durable-writes")``.
    """
    import os as _os

    if override is True:
        return True, "--durable-writes"
    if override is False:
        return False, "--no-durable-writes"
    on_slurm = bool(
        _os.environ.get("SLURM_JOB_ID") or _os.environ.get("SLURM_CPUS_PER_TASK")
    )
    return (True, "SLURM") if on_slurm else (False, "local")


def durable_writes_enabled(override: bool | None = None) -> bool:
    """Resolve whether the promote fsyncs before renaming.

    ``write()`` returns once data is in the page cache. Without ``fsync`` the
    kernel may flush the root ``zarr.json`` *before* the chunk data it
    describes, so a node crash can leave a store that passes
    :func:`valid_staged_store` -- metadata parses, shapes agree -- while
    reading ``fill_value``. That is silent wrong data, not a visible failure,
    and no amount of metadata validation catches it.

    The dominant failure mode does not need it: a SLURM timeout kills the
    process, and the kernel survives and flushes normally. ``fsync`` buys
    protection only against node loss, power failure, and filesystem crash --
    which is exactly what a cluster job is exposed to and a laptop run is not.

    Args:
        override: ``--durable-writes`` / ``--no-durable-writes``, or ``None``
            to auto-detect.

    Returns:
        ``True`` when the promote should fsync.

    Note:
        This checks ``SLURM_JOB_ID`` **as well as** ``SLURM_CPUS_PER_TASK``.
        ``resolve_worker_count`` (``_cli_utils.py:65-72``) reads only the
        latter, so this is deliberately broader -- not "exactly as" that helper
        does, which is what the spec's §3.7 claims. A job that sets
        ``SLURM_JOB_ID`` without a per-task CPU count still gets durable writes.
    """
    return _resolve_durability(override)[0]


def describe_durability(override: bool | None = None) -> str:
    """One-line description of the resolved durability mode, for the start log.

    The same command carries different guarantees in different places, which is
    a genuinely surprising thing to debug. Logging the resolved mode at run
    start is a required mitigation, not a nicety.

    Shares :func:`_resolve_durability` with :func:`durable_writes_enabled`, so
    the flag and the sentence describing it cannot drift apart.
    """
    enabled, reason = _resolve_durability(override)
    return f"durable writes: {'on' if enabled else 'off'} ({reason})"


def long_path(path: "Path") -> str:
    """Return an OS-appropriate path string, ``\\\\?\\``-prefixed on Windows.

    An output root, dataset name, and image stem plus a store-internal path can
    exceed Windows' 260-character ``MAX_PATH``. The ``"."`` chunk-key separator
    keeps a chunk key to one segment; this prefix covers the rest.
    """
    import os as _os
    from pathlib import Path as _Path

    resolved = _Path(path).resolve()
    if _os.name != "nt":
        return str(resolved)
    text = str(resolved)
    return text if text.startswith("\\\\?\\") else "\\\\?\\" + text


def new_part_path(final: "Path") -> "Path":
    """Return a fresh, uuid-suffixed ``.part`` sibling of *final*.

    The uuid -- matching the ``attempt_id = uuid4().hex`` convention already
    used in ``_cli_staged_strategy.py`` (lines 148, 192, 225, 359) -- is what
    keeps two concurrent writers from interleaving chunks into one directory.
    It is NOT what makes the promote itself benign; that is the retry loop in
    :func:`promote_store`. An un-suffixed ``.part`` would let two concurrent SLURM tasks
    interleave chunks into one directory and produce a store that *validates*.
    A PID is not enough: PIDs are reused.
    """
    from pathlib import Path as _Path
    from uuid import uuid4

    final = _Path(final)
    return final.parent / f".{final.name}.{uuid4().hex}{PART_SUFFIX}"


def fsync_tree(root: "Path") -> None:
    """``fsync`` every regular file under *root*, then *root* itself.

    The directory step is POSIX-guarded: Windows cannot open a directory handle
    for flushing, and relies on NTFS journaling instead.
    """
    import os as _os
    from pathlib import Path as _Path

    root = _Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            handle = _os.open(path, _os.O_RDONLY)
            try:
                _os.fsync(handle)
            finally:
                _os.close(handle)
    if _os.name == "posix":
        handle = _os.open(root, _os.O_RDONLY)
        try:
            _os.fsync(handle)
        finally:
            _os.close(handle)


#: errno / winerror values worth retrying. Everything else fails fast: retrying
#: a genuine ENOSPC five times with exponential backoff burns 3.1 s per image
#: before surfacing, which at 10k images is an hour of sleeping.
_RETRYABLE_WINERROR: Final[frozenset[int]] = frozenset({32, 33})  # SHARING_VIOLATION, LOCK_VIOLATION


def _is_retryable(exc: OSError) -> bool:
    """Whether *exc* is a transient contention error rather than a hard failure.

    Windows refuses to rename a directory while any file inside it is held open
    (``ERROR_SHARING_VIOLATION``); with ~40 files per store instead of one
    ``.h5``, that exposure is 40x larger. On POSIX, ``ENOTEMPTY``/``ENOENT`` on
    the target mean a concurrent promoter moved under us, which the retry loop
    resolves by re-evaluating.
    """
    import errno

    if getattr(exc, "winerror", None) in _RETRYABLE_WINERROR:
        return True
    return exc.errno in {errno.ENOTEMPTY, errno.ENOENT, errno.EEXIST}


def promote_store(part: "Path", final: "Path", *, fsync: bool) -> "Path":
    """Atomically promote a fully written ``.part`` directory to *final*.

    The caller is responsible for the write **order** inside *part*: all arrays
    and chunks first, then ``OME/zarr.json``, then the root ``zarr.json`` last.
    An interrupted store therefore has no valid root and reads as absent.

    The move-aside is mandatory, not an optimization: ``os.replace`` onto a
    non-empty directory raises ``OSError`` (``ENOTEMPTY``) on POSIX, and on
    Windows ``MoveFileEx``'s ``MOVEFILE_REPLACE_EXISTING`` cannot name a
    directory at all.

    The whole ``exists -> move-aside -> replace`` sequence sits inside one
    retry loop and re-evaluates existence on every attempt. That is what makes
    duplicate execution benign: a uuid ``.part`` prevents two writers
    *interleaving chunks*, but it does nothing for the promote itself, where a
    check-then-act done once lets writer B skip the move-aside because A had not
    yet renamed, then hit ``ENOTEMPTY`` on a now-non-empty target.

    On failure after a successful move-aside, the previous store is **rolled
    back** into place before retrying or raising. Deleting it in a ``finally``
    would leave no copy at any path -- a data-loss mode the single-file HDF
    rename never had, since a failed ``os.replace(tmp, final)`` left ``final``
    untouched.

    Known weakening versus the single-file rename: the two renames are still not
    one atomic step, so a crash *between* them (as opposed to a raised error)
    leaves the image absent plus an orphaned ``.trash``. Both are recoverable --
    absence reclassifies to the rebuilding stage, and :func:`sweep_orphan_parts`
    clears the leftovers.

    Args:
        part: Fully written ``.part`` directory.
        final: Target store path.
        fsync: Whether to flush *part* before renaming
            (see :func:`durable_writes_enabled`).

    Returns:
        *final*.
    """
    import os as _os
    import shutil
    import time
    from pathlib import Path as _Path

    part, final = _Path(part), _Path(final)
    if fsync:
        fsync_tree(part)

    trash = final.parent / f"{part.name[:-len(PART_SUFFIX)]}{TRASH_SUFFIX}"
    last: OSError | None = None
    for attempt in range(PROMOTE_RETRY_ATTEMPTS):
        moved_aside = False
        try:
            # Re-evaluate existence EVERY attempt. A concurrent promoter can
            # create or remove `final` between the check and either rename, so
            # a check-then-act done once outside the loop turns a benign
            # duplicate execution into a hard failure.
            if final.exists():
                _os.replace(final, trash)
                moved_aside = True
            _os.replace(part, final)
        except OSError as exc:
            last = exc
            if not _is_retryable(exc):
                if moved_aside and trash.exists() and not final.exists():
                    _os.replace(trash, final)
                raise
            if moved_aside and trash.exists() and not final.exists():
                # Roll back. Without this the previous store is already in
                # `trash` and is about to be deleted, leaving NO copy at any
                # path -- a data-loss mode the single-file HDF rename never had
                # (a failed os.replace(tmp, final) left `final` untouched).
                _os.replace(trash, final)
            time.sleep(PROMOTE_RETRY_BASE_SECONDS * (2**attempt))
            continue
        # Success: only now is the previous store safe to discard.
        if trash.exists():
            shutil.rmtree(trash, ignore_errors=True)
        return final
    assert last is not None
    raise last


#: A `.part` younger than this may still be being written. The sweep never
#: touches one. Generous by design: the cost of skipping a genuine orphan is one
#: stale directory until the next run; the cost of deleting a live one is a
#: destroyed in-flight image.
SWEEP_MIN_AGE_SECONDS: Final[float] = 6 * 60 * 60


def sweep_orphan_parts(
    results_root: "Path", *, min_age_seconds: float = SWEEP_MIN_AGE_SECONDS
) -> int:
    """Remove *stale* orphaned ``.part`` / ``.trash`` directories.

    **A uuid identifies the attempt, not whether its process is alive.** The
    staged SLURM engine explicitly assumes stale workers can still be running --
    that is what ``assert_active_epoch`` exists for -- and under an array the
    tasks share one output root and start at different times. A sweep with no
    liveness signal would ``rmtree`` the ``.part`` directories its siblings are
    actively filling, which is the same defect a PID-based sweep has.

    Two guards, both required:

    * **age**: only directories whose mtime is older than *min_age_seconds* are
      removed;
    * **placement**: the caller must run this from the controller before any
      worker is submitted, not from each worker's start-up (see Phase 3).

    The scan is bounded to ``results/<dataset>/zarr/`` rather than recursive:
    ``rglob`` would descend into every store, which is the same ~400k-stat
    pathology the spec flags for the GUI's discovery path.

    Args:
        results_root: The run's ``results/`` directory.
        min_age_seconds: Minimum age before a leftover is considered orphaned.

    Returns:
        Number of directories removed.
    """
    import os as _os
    import shutil
    import time
    from pathlib import Path as _Path

    removed = 0
    root = _Path(results_root)
    if not root.is_dir():
        return 0
    cutoff = time.time() - min_age_seconds
    for dataset_dir in root.iterdir():
        zarr_dir = dataset_dir / "zarr"
        if not zarr_dir.is_dir():
            continue
        for path in zarr_dir.iterdir():
            if not path.is_dir():
                continue
            if not (path.name.endswith(PART_SUFFIX) or path.name.endswith(TRASH_SUFFIX)):
                continue
            if STORE_SUFFIX not in path.name:
                continue
            if _os.stat(path).st_mtime > cutoff:
                continue  # may still be in flight
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_promote.py -v
```

Expected: all PASS on Linux, with the two `long_path` tests split by `skipif`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_promote.py
git commit -m "feat(sdk): add the rename-promote commit primitive

.part directories carry a uuid4, not a PID, so two concurrent SLURM tasks
can never interleave chunks into one directory and produce a store that
validates. The promote is a two-step move-aside because os.replace onto a
non-empty directory raises ENOTEMPTY on POSIX and MOVEFILE_REPLACE_EXISTING
cannot name a directory on Windows; both renames retry with backoff for
ERROR_SHARING_VIOLATION. fsync is on under SLURM and off locally, with an
explicit override and a describable mode for the run-start log."
```

---

### Task 1.6: `valid_staged_store`

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_validity.py`

**Interfaces:**
- Consumes: `read_phenotypic_attributes`, `PhenotypicAttr`.
- Produces:
  ```python
  def store_level0_shape(store_path: Path, member_path: str) -> tuple[int, ...] | None
  def valid_staged_store(path: Path) -> bool
  ```

**Constraints specific to this task:**
`valid_staged_store` mirrors `valid_staged_hdf` (`_cli/_cli_staged_resume.py:69`) case for
case:

- the root `zarr.json` parses and carries `phenotypic.store_schema_version`;
- **every** entry in `phenotypic.series` **and** `phenotypic.labels` opens as a Zarr array
  group — objmap included, which Stage 1's zeros write guarantees;
- level-0 `(y, x)` extents agree across all of them **and are non-zero** — a zero-size
  Zarr array is legal and must not pass;
- it catches `OSError`, `KeyError`, `ValueError`, `TypeError`, `json.JSONDecodeError`,
  `FileNotFoundError`, **and `zarr.errors.BaseZarrError`**. The HDF version's
  `(OSError, TypeError, ValueError)` set is insufficient — none of zarr's error types are
  `ValueError` subclasses.

`staged_store_matches_work_id` is **not** defined here; it stays in
`_cli_staged_resume.py` beside the classifier, mirroring today's placement of
`staged_hdf_matches_work_id` (Task 3.4).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_validity.py`:

```python
"""valid_staged_store mirrors valid_staged_hdf case for case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from phenotypic.sdk_ import ngff_


def _write_store(
    root: Path,
    *,
    shapes: dict[str, tuple[int, ...]],
    series: list[str],
    with_root: bool = True,
    store_schema_version: int | None = 3,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    primary = ngff_.primary_series(series) if series else "gray"
    members = {name: name for name in series}
    labels = {"objmap": ngff_.objmap_path(primary)}
    for name, path in [*members.items(), *labels.items()]:
        if name not in shapes:
            continue
        array = np.zeros(shapes[name], dtype=np.uint16)
        zarr.create_array(
            store=str(root / path / "0"),
            **ngff_.array_create_kwargs(array.shape, array.dtype, name),
        )
    if with_root:
        block = {
            ngff_.PhenotypicAttr.SERIES: members,
            ngff_.PhenotypicAttr.LABELS: labels,
        }
        if store_schema_version is not None:
            block[ngff_.PhenotypicAttr.STORE_SCHEMA_VERSION] = store_schema_version
        (root / "zarr.json").write_text(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {"ome": {"version": "0.5"}, "phenotypic": block},
                }
            ),
            encoding="utf-8",
        )
    return root


def test_complete_store_is_valid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is True


def test_missing_store_is_invalid(tmp_path: Path) -> None:
    assert ngff_.valid_staged_store(tmp_path / "absent.ome.zarr") is False


def test_missing_root_zarr_json_is_invalid(tmp_path: Path) -> None:
    """Interrupted after chunks, before the root: reads as absent, by design."""
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
        with_root=False,
    )
    assert ngff_.valid_staged_store(store) is False


def test_root_without_store_schema_version_is_invalid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
        store_schema_version=None,
    )
    assert ngff_.valid_staged_store(store) is False


def test_missing_objmap_is_invalid(tmp_path: Path) -> None:
    """Stage 1 writes a zeros objmap, so its absence means an incomplete write."""
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_missing_detect_mat_is_invalid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_disagreeing_extents_are_invalid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 47), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_zero_extent_is_invalid(tmp_path: Path) -> None:
    """A zero-size Zarr array is legal; it must not pass validity."""
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (0, 48), "detect_mat": (0, 48), "objmap": (0, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_rgb_store_attaches_labels_under_rgb(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={
            "rgb": (3, 64, 48),
            "gray": (64, 48),
            "detect_mat": (64, 48),
            "objmap": (64, 48),
        },
        series=["rgb", "gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is True
    block = ngff_.read_phenotypic_attributes(store)
    assert block[ngff_.PhenotypicAttr.LABELS]["objmap"] == "rgb/labels/objmap"


def test_corrupt_root_json_is_invalid_not_raising(tmp_path: Path) -> None:
    store = tmp_path / "a.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{not json", encoding="utf-8")
    assert ngff_.valid_staged_store(store) is False


def test_a_file_where_a_store_should_be_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "a.ome.zarr"
    path.write_bytes(b"not a directory")
    assert ngff_.valid_staged_store(path) is False


def test_zarr_errors_are_caught_not_propagated(tmp_path: Path, monkeypatch) -> None:
    """None of zarr's error types subclass ValueError, so the HDF version's
    (OSError, TypeError, ValueError) set would have let them escape."""
    import zarr.errors

    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )

    def _boom(*args, **kwargs):
        raise zarr.errors.BaseZarrError("synthetic")

    monkeypatch.setattr(ngff_, "store_level0_shape", _boom)
    assert ngff_.valid_staged_store(store) is False
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_validity.py -v
```

Expected: FAIL with `AttributeError: … has no attribute 'valid_staged_store'`.

- [ ] **Step 3: Append to `ngff_.py`**

```python
# ---------------------------------------------------------------------------
# Resume validity
# ---------------------------------------------------------------------------


def store_level0_shape(store_path: "Path", member_path: str) -> tuple[int, ...] | None:
    """Return the level-0 shape of one member array, or ``None`` if absent.

    Args:
        store_path: Store root.
        member_path: Store-relative group path, e.g. ``"gray"`` or
            ``"rgb/labels/objmap"``.

    Returns:
        The level-0 array shape, or ``None`` when the level-0 array is missing.
    """
    import zarr
    from pathlib import Path as _Path

    level0 = _Path(store_path) / member_path / "0"
    if not level0.is_dir():
        return None
    return tuple(zarr.open_array(store=str(level0), mode="r").shape)


def valid_staged_store(path: "Path") -> bool:
    """Return whether *path* holds the image layers Stage 2 requires.

    Mirrors ``valid_staged_hdf`` case for case:

    * the root ``zarr.json`` parses and carries ``store_schema_version``;
    * every entry in ``phenotypic.series`` **and** ``phenotypic.labels`` opens
      as a Zarr array group -- objmap included, which Stage 1's zeros write
      guarantees;
    * level-0 ``(y, x)`` extents agree across all of them and are non-zero. A
      zero-size Zarr array is legal and must not pass.

    The exception set is the HDF version's ``(OSError, TypeError,
    ValueError)`` **plus ``KeyError``** -- which the attribute lookups need and
    the HDF version did not.

    It does **not** need ``zarr.errors.BaseZarrError``. The spec's §3.6 argues
    the opposite ("none of zarr's error types are ``ValueError`` subclasses");
    that is inverted. ``BaseZarrError`` inherits **directly from
    ``ValueError``** (https://zarr.readthedocs.io/en/stable/api/zarr/errors/),
    as do ``MetadataValidationError`` and every other zarr error except the four
    ``IndexError`` ones, none of which this function can raise.
    ``json.JSONDecodeError`` is likewise a ``ValueError`` and
    ``FileNotFoundError`` an ``OSError``, so both are already covered. Keeping
    the shorter tuple also avoids importing ``zarr.errors`` in a function the
    resume planner calls once per image.

    Args:
        path: Candidate ``*.ome.zarr`` directory.

    Returns:
        ``True`` only for a store Stage 2 can consume.
    """
    from pathlib import Path as _Path

    try:
        store = _Path(path)
        if not store.is_dir():
            return False
        block = read_phenotypic_attributes(store)
        if PhenotypicAttr.STORE_SCHEMA_VERSION not in block:
            return False
        members = [
            *block[PhenotypicAttr.SERIES].values(),
            *block[PhenotypicAttr.LABELS].values(),
        ]
        if not members:
            return False
        shapes: list[tuple[int, ...]] = []
        for member in members:
            shape = store_level0_shape(store, member)
            if shape is None:
                return False
            shapes.append(shape)
        spatial = [shape[-2:] for shape in shapes]
        if any(len(yx) < 2 or yx[0] <= 0 or yx[1] <= 0 for yx in spatial):
            return False
        return all(yx == spatial[0] for yx in spatial[1:])
    except (OSError, KeyError, TypeError, ValueError):
        return False
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_validity.py -v
```

Expected: all PASS. If `zarr.errors.BaseZarrError` does not exist in the resolved version,
find the actual base with
`uv run python -c "import zarr.errors as e; print([n for n in dir(e) if n.endswith('Error')])"`
and use the true base class — do **not** widen the handler to bare `Exception`.

- [ ] **Step 4a: Add `write_objmap_in_place`**

Store surgery belongs here, not in the CLI. **Both Phase 3 (Stage 2) and Phase 4 (the tile
cache-invalidation test) import it**, so defining it in Phase 3 would make Phase 4 depend on
Phase 3 while the DAG declares them parallel. Recorded as OPEN-QUESTIONS **B10**.

```python
def write_objmap_in_place(store_path: "Path", objmap: "np.ndarray") -> None:
    """Overwrite every pyramid level of a promoted store's objmap.

    An intermediate write, not a publish -- the store is **not** re-promoted, so
    the root ``zarr.json`` is untouched and the store's published identity does
    not change. A concurrent reader may therefore observe a torn objmap; the
    completion marker, not the store's shape, is what gates consumers, and the
    GUI deliberately does not invalidate on this (Phase 4 Task 4.3).

    Every level is rewritten, not just level 0: a stale level-1 objmap under a
    fresh level 0 renders as a silently wrong overlay at any zoomed-out view,
    with no error path.

    Args:
        store_path: Promoted store.
        objmap: New level-0 label array.
    """
    import zarr

    block = read_phenotypic_attributes(store_path)
    member = block[PhenotypicAttr.LABELS][OBJMAP_LABEL]
    levels = int(block[PhenotypicAttr.PYRAMID]["levels"])
    for index, level in enumerate(build_pyramid(objmap, levels, kind="label")):
        handle = zarr.open_array(
            store=long_path(Path(store_path) / member / str(index)), mode="r+"
        )
        handle[...] = level
```

Add to `tests/unit/sdk_/test_ngff_validity.py`:

```python
def test_write_objmap_in_place_rewrites_every_level(tmp_path: Path) -> None:
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    labels = Image.load_layer_zarr(store, "objmap")
    labels[:4, :4] = 7
    ngff_.write_objmap_in_place(store, labels)
    level0 = Image.load_layer_zarr(store, "objmap", level=0)
    level1 = Image.load_layer_zarr(store, "objmap", level=1)
    np.testing.assert_array_equal(level1, level0[::2, ::2])


def test_write_objmap_in_place_does_not_touch_the_root(tmp_path: Path) -> None:
    """It is an intermediate, not a publish. Phase 4 relies on this."""
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    before = (store / "zarr.json").read_bytes()
    ngff_.write_objmap_in_place(store, Image.load_layer_zarr(store, "objmap"))
    assert (store / "zarr.json").read_bytes() == before
```

- [ ] **Step 5: Export the public surface from `sdk_/__init__.py`**

Add `valid_staged_store`, `promote_store`, `new_part_path`, `sweep_orphan_parts`,
`durable_writes_enabled`, `describe_durability`, `PhenotypicAttr`, and `STORE_SUFFIX` to
`src/phenotypic/sdk_/__init__.py`'s imports and `__all__`, beside the existing `HDF`
export. Keep the list alphabetised as the file already is.

- [ ] **Step 6: Run the whole new module's suite plus the type gate**

```bash
uv run pytest tests/unit/sdk_/test_ngff_*.py -v
uv run ruff check --fix src/phenotypic/sdk_/ngff_.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_
uv run mypy src/phenotypic/sdk_/ngff_.py
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_/test_ngff_validity.py
git commit -m "feat(sdk): add valid_staged_store

Mirrors valid_staged_hdf case for case: root parses and carries
store_schema_version, every series AND label opens, level-0 extents agree
and are non-zero (a zero-size Zarr array is legal and must not pass). The
exception set is wider than the HDF version's because none of zarr's error
types subclass ValueError, so the old set would have let them escape."
```

---

## Phase 1 exit criteria

- [ ] `uv run pytest tests/unit/sdk_/test_ngff_*.py -q` is all green.
- [ ] `uv run python docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py` exits 0.
- [ ] `uv run mypy src/phenotypic/sdk_/ngff_.py` passes.
- [ ] `grep -n "2 \*\* n\|2\*\*n" src/phenotypic/sdk_/ngff_.py` finds nothing in scale computation.
- [ ] `grep -n "getpid" src/phenotypic/sdk_/ngff_.py` finds nothing.
- [ ] No existing test changed behaviour — the HDF path is untouched in this phase.
# Phase 2 — Image/GridImage store I/O, path constants, conformance harness

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §1, §2, §3.1, §3.3, §4.1, §7.

**Depends on:** Phase 1.
**Blocks:** Phases 3, 4, 5.

This phase makes an `Image` round-trip through a store. The HDF quartet is left in place
and still works — it is removed in Phase 6, after migration (Phase 5) has been built on top
of its readers.

---

### Task 2.1: Path constants and the store locator

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (line 656 `DIR_HDF`, line 1447
  `dataset_hdf_dir`, line 1886 `HdfAttr`, line 1936 `load_image_from_hdf`, line 2063
  `BundleLayout.hdf_path`)
- Test: `tests/unit/sdk_/test_io_constants.py` (extend), `tests/unit/sdk_/test_bundle_layout.py` (extend)

**Interfaces:**
- Consumes: `ngff_.STORE_SUFFIX`, `ngff_.PhenotypicAttr`, `ngff_.read_phenotypic_attributes`.
- Produces:
  ```python
  DIR_ZARR: Final[str] = "zarr"
  def dataset_zarr_dir(output_dir: Path, dataset: str) -> Path
  def zarr_store_path(output_dir: Path, dataset: str, stem: str) -> Path
  def load_image_from_store(store_path: Path, *, fallback: ImageTypeName = "Image") -> "_Image | _GridImage"
  BundleLayout.store_path(dataset: str, stem: str) -> Optional[Path]
  ```

**Constraints specific to this task:**
- `DIR_HDF` / `dataset_hdf_dir` / `HdfAttr` / `load_image_from_hdf` / `BundleLayout.hdf_path`
  are **kept** through Phase 5 — migration reads legacy trees and needs them. They are
  removed in Phase 6. Do not delete them here.
- `zarr_store_path` is the **only** place `STORE_SUFFIX` is joined to a stem. Nothing
  anywhere else may write `f"{stem}.ome.zarr"`. A grep gate in Phase 7 enforces this.
- `BundleLayout.store_path` returns the path only when it `is_dir()` — the store is a
  directory, so `is_file()` (as `hdf_path` uses) would always be `None`. This is the exact
  bug shape that makes a naive port silently disable every full-res GUI read.
- `load_image_from_store` dispatches on `phenotypic.image_class` — **not** on
  `Metadata_ImageType`, which is a different, user-visible field.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/sdk_/test_io_constants.py`:

```python
def test_zarr_store_path_is_the_only_suffix_joiner(tmp_path) -> None:
    from phenotypic.sdk_ import dataset_zarr_dir, zarr_store_path
    from phenotypic.sdk_.ngff_ import STORE_SUFFIX

    path = zarr_store_path(tmp_path, "plates", "plate_01")
    assert path == dataset_zarr_dir(tmp_path, "plates") / f"plate_01{STORE_SUFFIX}"
    assert path.parent.name == "zarr"


def test_dataset_zarr_dir_sits_beside_the_other_result_dirs(tmp_path) -> None:
    from phenotypic.sdk_ import dataset_results_dir, dataset_zarr_dir

    assert dataset_zarr_dir(tmp_path, "ds") == dataset_results_dir(tmp_path, "ds") / "zarr"
```

Append to `tests/unit/sdk_/test_bundle_layout.py`:

```python
def test_store_path_resolves_a_directory_not_a_file(tmp_path) -> None:
    """A store is a directory; an is_file() check would always return None."""
    from phenotypic.sdk_ import BundleLayout, zarr_store_path

    store = zarr_store_path(tmp_path, "ds", "img")
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "deliverables").mkdir()
    (tmp_path / "deliverables" / "master_measurements.parquet").write_bytes(b"")
    layout = BundleLayout.resolve(tmp_path)
    assert layout.store_path("ds", "img") == store


def test_store_path_returns_none_when_absent(tmp_path) -> None:
    from phenotypic.sdk_ import BundleLayout

    (tmp_path / "deliverables").mkdir()
    (tmp_path / "deliverables" / "master_measurements.parquet").write_bytes(b"")
    layout = BundleLayout.resolve(tmp_path)
    assert layout.store_path("ds", "img") is None
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/unit/sdk_/test_io_constants.py tests/unit/sdk_/test_bundle_layout.py -k "zarr or store_path" -v
```

Expected: `ImportError: cannot import name 'zarr_store_path'`.

- [ ] **Step 3: Add the constants and helpers**

In `_io_constants.py`, beside `DIR_HDF` (line 656):

```python
#: OME-Zarr image-state subdirectory: ``<output>/results/<ds>/zarr/``.
DIR_ZARR: Final[str] = "zarr"
```

Beside `dataset_hdf_dir` (line 1447):

```python
def dataset_zarr_dir(output_dir: Path, dataset: str) -> Path:
    """Return ``<output>/results/<dataset>/zarr/``."""
    return dataset_results_dir(output_dir, dataset) / DIR_ZARR


def zarr_store_path(output_dir: Path, dataset: str, stem: str) -> Path:
    """Return ``<output>/results/<dataset>/zarr/<stem>.ome.zarr/``.

    The single place ``.ome.zarr`` is joined to an image stem. Callers must
    never hand-join the suffix; a grep gate in the test suite enforces this.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        stem: Image filename without extension.

    Returns:
        The per-image store path. Existence is not checked.
    """
    from phenotypic.sdk_.ngff_ import STORE_SUFFIX

    return dataset_zarr_dir(output_dir, dataset) / f"{stem}{STORE_SUFFIX}"
```

Beside `load_image_from_hdf` (line 1936):

```python
def load_image_from_store(
    store_path: Path,
    *,
    fallback: ImageTypeName = "Image",
) -> "_Image | _GridImage":
    """Read ``phenotypic.image_class`` from a store root and dispatch the loader.

    Dispatches on ``image_class`` (``Image`` / ``GridImage``), which is the
    loader-dispatch field. It is **not** ``Metadata_ImageType``, which is
    user-visible schema metadata and may be ``GridSection`` on a plain
    :class:`Image`.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.
        fallback: Class name used when the block carries no ``image_class``.

    Returns:
        An :class:`Image` or :class:`GridImage` loaded from the store.
    """
    from phenotypic import GridImage, Image
    from phenotypic.sdk_.constants_ import IMAGE_TYPES
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

    block = read_phenotypic_attributes(store_path)
    class_name = block.get(PhenotypicAttr.IMAGE_CLASS, fallback)
    image_cls = GridImage if class_name == IMAGE_TYPES.GRID.value else Image
    return image_cls.load_zarr(store_path)
```

On `BundleLayout`, beside `hdf_path` (line 2063):

```python
    def store_path(self, dataset: str, stem: str) -> Optional[Path]:
        """Full-res per-image OME-Zarr store for ``(dataset, stem)``, or ``None``.

        Args:
            dataset: Dataset name (subdirectory under ``results/``).
            stem: Image stem (filename without extension).

        Returns:
            Resolved store path if the **directory** exists, otherwise ``None``.
            Note the ``is_dir`` check: a store is a directory, so the
            ``is_file`` test used by :meth:`hdf_path` would always return
            ``None`` here.
        """
        if self.output_root is None:
            return None
        candidate = zarr_store_path(self.output_root, dataset, stem)
        return candidate if candidate.is_dir() else None
```

Export `DIR_ZARR`, `dataset_zarr_dir`, `zarr_store_path`, and `load_image_from_store`
from `src/phenotypic/sdk_/__init__.py` beside their HDF counterparts.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/sdk_/test_io_constants.py tests/unit/sdk_/test_bundle_layout.py -v
```

Expected: PASS except `test_store_path_resolves_a_directory_not_a_file`, which still fails
because `Image.load_zarr` does not exist yet — that is Task 2.2. Mark that one
`@pytest.mark.xfail(reason="Image.load_zarr lands in Task 2.2", strict=True)` and remove
the marker in Task 2.2's Step 4. Do **not** skip it.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_
git commit -m "feat(sdk): add zarr store path constants and locator

zarr_store_path is the only place .ome.zarr is joined to a stem.
BundleLayout.store_path checks is_dir, not is_file -- a store is a
directory, so a copy-paste of hdf_path would return None for every image
and silently disable full-res GUI reads. The HDF helpers stay until
Phase 6, because --mode migrate is built on their readers."
```

---

### Task 2.2: `save2zarr` and `load_zarr` on `ImageIOHandler`

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`
  (add beside `save2hdf5` at line 871 and `load_hdf5` at line 1155)
- Test: `tests/unit/core/test_image_zarr_roundtrip.py` (create)

**Interfaces:**
- Consumes: everything from Phase 1.
- Produces:
  ```python
  def save2zarr(self, path, *, work_id: str | None = None, durable: bool | None = None) -> Path
  @classmethod
  def load_zarr(cls, path, **kwargs) -> Image
  @classmethod
  def load_layer_zarr(cls, path, layer: str, level: int = 0) -> np.ndarray
  def _write_series(self, part: Path, series: str, array, levels: int) -> None
  def _series_names(self) -> list[str]
  ```

**Constraints specific to this task:**
- `save2zarr` builds into a `.part` via `ngff_.new_part_path`, writes **arrays first, then
  `OME/zarr.json`, then the root `zarr.json` last**, and promotes with
  `ngff_.promote_store(part, final, fsync=ngff_.durable_writes_enabled(durable))`.
- `rgb` is **omitted entirely** when `self.rgb.isempty()`.
- `objmap` is **always** written, zeros included.
- `work_id` goes into the attributes block at build time — no post-write patch.
- On `build_ome_xml` returning `None`, emit **neither** `OME/zarr.json` nor
  `OME/METADATA.ome.xml`, drop `series` from the root's `ome` block, and log a warning.
  Never emit a half-built `OME/` group.
- `load_zarr` reads only level 0, restores the three metadata sections verbatim, and warns
  (without upcasting) when `Image.load_zarr` is called on a store whose `image_class` is
  `GridImage` — mirroring `load_hdf5`'s existing behaviour at line 1170.
- `load_layer_zarr(path, layer, level)` resolves `objmap` through
  `phenotypic.labels.objmap`, never by hard-coded path.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_image_zarr_roundtrip.py`:

```python
"""Image -> store -> Image must be bit-exact in layers and equal in metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic import GridImage, Image
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes
from phenotypic.data import load_synth_yeast_plate


@pytest.fixture
def plate() -> Image:
    return Image(load_synth_yeast_plate())


def test_layers_round_trip_bit_exact(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    np.testing.assert_array_equal(back.rgb[:], plate.rgb[:])
    np.testing.assert_array_equal(back.gray[:], plate.gray[:])
    np.testing.assert_array_equal(back.detect_mat[:], plate.detect_mat[:])
    np.testing.assert_array_equal(back.objmap[:], plate.objmap[:])


def test_metadata_sections_round_trip(plate: Image, tmp_path: Path) -> None:
    plate._metadata.public["Metadata_Strain"] = "BY4741"
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    assert dict(back._metadata.public) == dict(plate._metadata.public)
    assert dict(back._metadata.protected) == dict(plate._metadata.protected)
    assert dict(back._metadata.imported) == dict(plate._metadata.imported)


def test_objmap_is_written_even_when_nothing_is_detected(
    plate: Image, tmp_path: Path
) -> None:
    """Stage 1 relies on this: valid_staged_store requires objmap to exist."""
    assert plate.num_objects == 0
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    block = read_phenotypic_attributes(store)
    objmap = block[PhenotypicAttr.LABELS]["objmap"]
    assert (store / objmap / "0").is_dir()
    assert (Image.load_zarr(store).objmap[:] == 0).all()


def test_rgb_is_omitted_entirely_when_empty(tmp_path: Path) -> None:
    gray_only = Image(load_synth_yeast_plate())
    gray_only.rgb.clear()
    store = gray_only.save2zarr(tmp_path / "g.ome.zarr")
    assert not (store / "rgb").exists()
    block = read_phenotypic_attributes(store)
    assert "rgb" not in block[PhenotypicAttr.SERIES]
    assert block[PhenotypicAttr.LABELS]["objmap"] == "gray/labels/objmap"


def test_primary_series_is_first_in_the_ome_series_list(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    ome = json.loads((store / "OME" / "zarr.json").read_text(encoding="utf-8"))
    assert ome["attributes"]["ome"]["series"][0] == "rgb"


def test_root_zarr_json_is_written_last(plate: Image, tmp_path: Path, monkeypatch) -> None:
    """An interrupted store has no valid root and must read as absent."""
    from phenotypic.sdk_ import ngff_

    written: list[str] = []
    real_promote = ngff_.promote_store

    def _record(part: Path, final: Path, *, fsync: bool):
        for path in sorted(Path(part).rglob("zarr.json")):
            written.append(str(path.relative_to(part)))
        return real_promote(part, final, fsync=fsync)

    monkeypatch.setattr(ngff_, "promote_store", _record)
    plate.save2zarr(tmp_path / "p.ome.zarr")
    assert written[-1] == "zarr.json", written


def test_work_id_is_written_into_the_block_not_patched(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr", work_id="w-42")
    assert read_phenotypic_attributes(store)[PhenotypicAttr.WORK_ID] == "w-42"


def test_pyramid_depth_is_uniform_across_every_series(
    plate: Image, tmp_path: Path
) -> None:
    """NGFF requires a label image to carry its parent's level count."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    block = read_phenotypic_attributes(store)
    levels = block[PhenotypicAttr.PYRAMID]["levels"]
    members = [
        *block[PhenotypicAttr.SERIES].values(),
        *block[PhenotypicAttr.LABELS].values(),
    ]
    for member in members:
        found = sorted(p.name for p in (store / member).iterdir() if p.name.isdigit())
        assert found == [str(i) for i in range(levels)], member


def test_pyramid_depth_is_a_pure_function_of_shape(plate: Image, tmp_path: Path) -> None:
    """Fixed, not tunable: save2zarr takes no pyramid argument at all (P3)."""
    import inspect

    from phenotypic.sdk_ import ngff_

    assert "pyramid_levels" not in inspect.signature(Image.save2zarr).parameters
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    gray = plate.gray[:]
    assert read_phenotypic_attributes(store)[PhenotypicAttr.PYRAMID]["levels"] == (
        ngff_.pyramid_level_count(gray.shape[0], gray.shape[1])
    )


def test_load_layer_zarr_reads_one_layer_without_a_full_image(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    np.testing.assert_array_equal(Image.load_layer_zarr(store, "gray"), plate.gray[:])


def test_load_layer_zarr_resolves_objmap_via_the_labels_key(
    tmp_path: Path,
) -> None:
    """rgb-less stores put the label under gray; a hard-coded path would 404."""
    gray_only = Image(load_synth_yeast_plate())
    gray_only.rgb.clear()
    store = gray_only.save2zarr(tmp_path / "g.ome.zarr")
    assert Image.load_layer_zarr(store, "objmap").shape == gray_only.gray[:].shape


def test_load_layer_zarr_can_read_a_pyramid_level(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    full = Image.load_layer_zarr(store, "gray", level=0)
    half = Image.load_layer_zarr(store, "gray", level=1)
    assert half.shape == ((full.shape[0] + 1) // 2, (full.shape[1] + 1) // 2)


def test_load_layer_zarr_raises_keyerror_for_an_unknown_layer(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    with pytest.raises(KeyError):
        Image.load_layer_zarr(store, "not_a_layer")


def test_image_load_zarr_on_a_griddimage_store_warns_without_upcasting(
    tmp_path: Path,
) -> None:
    grid = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12)
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    with pytest.warns(UserWarning, match="GridImage"):
        back = Image.load_zarr(store)
    assert type(back) is Image


def test_image_class_and_image_type_are_independent(tmp_path: Path) -> None:
    plain = Image(load_synth_yeast_plate())
    plain._metadata.protected["Metadata_ImageType"] = "GridSection"
    store = plain.save2zarr(tmp_path / "s.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert block[PhenotypicAttr.IMAGE_CLASS] == "Image"
    assert (
        block[PhenotypicAttr.METADATA]["protected"]["Metadata_ImageType"]
        == "GridSection"
    )
    assert type(Image.load_zarr(store)) is Image


def test_detect_mode_illuminant_and_gamma_survive(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    assert back.illuminant == plate.illuminant
    assert str(back.gamma) == str(plate.gamma)
    assert back._data.detect_mode == plate._data.detect_mode


def test_ome_group_is_dropped_entirely_when_the_xml_cannot_be_built(
    plate: Image, tmp_path: Path, monkeypatch, caplog
) -> None:
    """Partial OME output would break the named-series MUST; drop it wholesale."""
    from phenotypic.sdk_ import ngff_

    monkeypatch.setattr(ngff_, "build_ome_xml", lambda **kwargs: None)
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    assert not (store / "OME").exists()
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert "series" not in root["attributes"]["ome"]
    assert any("OME" in record.message for record in caplog.records)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/core/test_image_zarr_roundtrip.py -v
```

Expected: FAIL with `AttributeError: 'Image' object has no attribute 'save2zarr'`.

- [ ] **Step 3: Implement `save2zarr` / `load_zarr` / `load_layer_zarr`**

Add to `ImageIOHandler` (place `save2zarr` immediately after `save2hdf5` at line 871, and
`load_zarr` / `load_layer_zarr` after `load_layer_hdf5` at line 1194), so the HDF and store
paths sit side by side until Phase 6 removes the former.

```python
    def _series_names(self) -> list[str]:
        """Series this image will write, in canonical order.

        ``rgb`` is omitted entirely when empty; ``gray`` then becomes the
        primary series and the objmap label attaches under it.
        """
        from phenotypic.sdk_.ngff_ import SERIES_ORDER

        return [
            name
            for name in SERIES_ORDER
            if name != "rgb" or not self.rgb.isempty()
        ]

    def _write_series(
        self, part: Path, series: str, array: np.ndarray, levels: int
    ) -> None:
        """Write every pyramid level of one series (or label) into *part*.

        Args:
            part: The ``.part`` directory being built.
            series: Group path relative to *part*.
            array: Level-0 array.
            levels: Level count, uniform across the store.
        """
        import zarr

        from phenotypic.sdk_ import ngff_

        kind = "label" if series.endswith(ngff_.OBJMAP_LABEL) else "image"
        name = ngff_.OBJMAP_LABEL if kind == "label" else series
        for index, level in enumerate(ngff_.build_pyramid(array, levels, kind=kind)):
            handle = zarr.create_array(
                store=ngff_.long_path(part / series / str(index)),
                **ngff_.array_create_kwargs(level.shape, level.dtype, name),
            )
            handle[...] = level

    def save2zarr(
        self,
        path,
        *,
        work_id: str | None = None,
        durable: bool | None = None,
    ) -> Path:
        """Save the image as an OME-Zarr (NGFF 0.5 / Zarr v3) store.

        Builds a uuid-suffixed ``.part`` sibling, writes every array, then
        ``OME/zarr.json``, then the root ``zarr.json`` **last**, then promotes
        by directory rename. An interrupted write leaves no valid root, so the
        store reads as absent rather than as partial.

        Args:
            path: Target ``*.ome.zarr`` directory. Created or replaced.
            work_id: CLI work id, written into ``attributes.phenotypic`` at
                build time. Never patched in afterwards.
            durable: ``fsync`` before promoting. ``None`` auto-detects SLURM.

        Returns:
            The promoted store path.

        Examples:
            Save a synthetic plate and read it back:

            >>> from phenotypic.data import load_synth_yeast_plate
            >>> img = Image(load_synth_yeast_plate())
            >>> store = img.save2zarr('plate.ome.zarr')
            >>> reloaded = Image.load_zarr(store)
            >>> reloaded.gray[:].shape == img.gray[:].shape
            True
        """
        import json
        import logging
        from pathlib import Path as _Path

        from phenotypic.sdk_ import ngff_

        final = _Path(path)
        final.parent.mkdir(parents=True, exist_ok=True)
        part = ngff_.new_part_path(final)

        series_names = self._series_names()
        primary = ngff_.primary_series(series_names)
        gray = self.gray[:]
        levels = ngff_.pyramid_level_count(gray.shape[0], gray.shape[1])

        arrays: dict[str, np.ndarray] = {"gray": gray, "detect_mat": self.detect_mat[:]}
        if "rgb" in series_names:
            arrays["rgb"] = np.moveaxis(self.rgb[:], -1, 0)  # (H,W,C) -> (c,y,x)

        # 1. arrays and chunks
        for name in series_names:
            self._write_series(part, name, arrays[name], levels)
        objmap = self.objmap[:]
        self._write_series(part, ngff_.objmap_path(primary), objmap, levels)

        # 2. per-group ome metadata
        bit_depth = int(self.bit_depth or 8)
        name = self._metadata.protected.get("Metadata_ImageName")
        for series in series_names:
            shapes = ngff_.pyramid_level_shapes(arrays[series].shape, levels)
            block = {
                "version": ngff_.NGFF_VERSION,
                **ngff_.build_multiscales(
                    series=series, level_shapes=shapes, name=name
                ),
                **ngff_.build_omero(series=series, bit_depth=bit_depth, name=name),
            }
            self._write_group_json(part / series, {"ome": block})
        label_shapes = ngff_.pyramid_level_shapes(objmap.shape, levels)
        self._write_group_json(
            part / primary / ngff_.LABELS_GROUP,
            {"ome": {"version": ngff_.NGFF_VERSION, "labels": [ngff_.OBJMAP_LABEL]}},
        )
        self._write_group_json(
            part / ngff_.objmap_path(primary),
            {
                "ome": {
                    "version": ngff_.NGFF_VERSION,
                    **ngff_.build_multiscales(
                        series=ngff_.OBJMAP_LABEL, level_shapes=label_shapes
                    ),
                    **ngff_.build_image_label(),
                }
            },
        )

        # 3. OME/ group -- all or nothing
        sections = {
            "protected": dict(self._metadata.protected),
            "public": dict(self._metadata.public),
            "imported": dict(self._metadata.imported),
        }
        xml = ngff_.build_ome_xml(
            series_names=series_names, metadata_sections=sections
        )
        ome_root: dict = {
            "version": ngff_.NGFF_VERSION,
            "bioformats2raw.layout": ngff_.BIOFORMATS2RAW_LAYOUT,
        }
        if xml is not None:
            (part / ngff_.OME_GROUP).mkdir(parents=True, exist_ok=True)
            (part / ngff_.OME_GROUP / ngff_.OME_XML_NAME).write_text(
                xml, encoding="utf-8"
            )
            self._write_group_json(
                part / ngff_.OME_GROUP,
                {
                    "ome": {
                        "version": ngff_.NGFF_VERSION,
                        "series": series_names,
                    }
                },
            )
        else:
            logging.getLogger(__name__).warning(
                "OME/ group omitted for %s; falling back to the "
                "consecutive-integer series form.",
                final.name,
            )

        # 4. root zarr.json LAST
        self._write_group_json(
            part,
            {
                "ome": ome_root,
                ngff_.PhenotypicAttr.ROOT: self._build_store_attributes(
                    series_names=series_names,
                    levels=levels,
                    sections=sections,
                    work_id=work_id,
                ),
            },
        )
        return ngff_.promote_store(
            part, final, fsync=ngff_.durable_writes_enabled(durable)
        )

    @staticmethod
    def _write_group_json(group_dir: Path, attributes: dict) -> None:
        """Write a Zarr v3 group ``zarr.json`` carrying *attributes*."""
        import json
        from pathlib import Path as _Path

        group_dir = _Path(group_dir)
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "zarr.json").write_text(
            json.dumps(
                {"zarr_format": 3, "node_type": "group", "attributes": attributes},
                indent=2,
            ),
            encoding="utf-8",
        )

    def _build_store_attributes(
        self, *, series_names, levels, sections, work_id
    ) -> dict:
        """Assemble ``attributes.phenotypic``. Overridden by ``GridImage``."""
        from phenotypic.sdk_ import ngff_

        return ngff_.build_phenotypic_attributes(
            image_class=type(self).__name__,
            series_names=series_names,
            pyramid_levels=levels,
            metadata_sections=sections,
            detect_mode=self._data.detect_mode,
            illuminant=str(self.illuminant) if self.illuminant is not None else None,
            gamma=(
                self.gamma.name if hasattr(self.gamma, "name") else str(self.gamma)
            )
            if self.gamma is not None
            else None,
            work_id=work_id,
        )
```

`load_zarr` / `load_layer_zarr`:

```python
    @classmethod
    def load_zarr(cls, path, **kwargs) -> Image:
        """Load an image from an OME-Zarr store.

        Reads only level 0. ``attributes.phenotypic`` is the sole source of
        truth; the OME projection is never read back.

        Args:
            path: Path to a ``*.ome.zarr`` directory.
            **kwargs: Forwarded to the constructor, taking priority over
                anything recovered from the store.

        Returns:
            An :class:`Image` (or :class:`GridImage`, via the subclass).
        """
        import warnings

        from phenotypic.sdk_ import ngff_

        block = ngff_.read_phenotypic_attributes(path)
        saved_class = block.get(ngff_.PhenotypicAttr.IMAGE_CLASS)
        if saved_class == "GridImage" and cls.__name__ != "GridImage":
            warnings.warn(
                "Store was saved as GridImage; use GridImage.load_zarr to "
                "preserve grid state",
                UserWarning,
                stacklevel=2,
            )
        return cls._load_from_store(path, block, **kwargs)

    @classmethod
    def load_layer_zarr(cls, path, layer: str, level: int = 0) -> np.ndarray:
        """Read one layer at one pyramid level without building an Image.

        ``objmap`` is resolved through ``phenotypic.labels.objmap``, never by a
        hard-coded ``rgb/labels/objmap`` -- an rgb-less store puts the label
        under ``gray``.

        Args:
            path: Store path.
            layer: ``"rgb"``, ``"gray"``, ``"detect_mat"``, or ``"objmap"``.
            level: Pyramid level index; 0 is full resolution.

        Returns:
            The layer array. ``rgb`` is returned as ``(H, W, C)``.

        Raises:
            KeyError: If *layer* is not present in the store.
        """
        import zarr

        from phenotypic.sdk_ import ngff_

        block = ngff_.read_phenotypic_attributes(path)
        member = block[ngff_.PhenotypicAttr.SERIES].get(layer) or block[
            ngff_.PhenotypicAttr.LABELS
        ].get(layer)
        if member is None:
            raise KeyError(f"Layer {layer!r} not found in {path}")
        from pathlib import Path as _Path

        array = zarr.open_array(
            store=ngff_.long_path(_Path(path) / member / str(level)), mode="r"
        )[...]
        return np.moveaxis(array, 0, -1) if layer == "rgb" else array
```

`_load_from_store` follows the *shape* of `_load_v2_grouped` (line 984) — read each series'
level-0 array, restore the metadata sections, apply `detect_mode`, `illuminant`, `gamma`,
and `bit_depth` — but **must not copy its metadata merge**, which drops values. Write it
directly below `_load_v2_grouped` so the two read paths are visibly parallel.

> **Do not mirror `_image_io_handler.py:1071-1073`:**
>
> ```python
> for mapped, value in decoded.items():
>     if mapped in target and target[mapped] is not None:
>         continue          # <-- the constructor already set it; stored value is DROPPED
>     target[mapped] = value
> ```
>
> The constructor sets `Metadata_ImageType` before the loader runs
> (`_image_data_manager.py:104,156`; `_grid_image_handler.py:101`), so the stored value
> never lands. **Verified by execution** on the HDF path in this worktree:
> `before ImageType: GridSection` → `after ImageType: Image`.
>
> Spec §2.1 requires `image_class` and `Metadata_ImageType` to stay distinct and §7 mandates
> a test asserting both survive independently — so mirroring this bug makes
> `test_image_class_and_image_type_are_independent` (below) fail.
>
> `_load_from_store` therefore **assigns the three stored sections verbatim**, overwriting
> constructor defaults rather than deferring to them. This makes the store read path more
> correct than the HDF one; that divergence is deliberate and recorded as OPEN-QUESTIONS
> **D7**. The HDF loader is not fixed here — it is retired in Phase 6.
>
> `_metadata` also has a fourth section, `private`, which the HDF writer does not persist
> and which the store does not persist either. That is a deliberate carry-over, not an
> oversight; say so in the docstring so a later reader does not "complete" the set.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/core/test_image_zarr_roundtrip.py -v
```

Expected: all PASS. Remove the `xfail` marker added in Task 2.1 Step 4 and re-run
`tests/unit/sdk_/test_bundle_layout.py`.

- [ ] **Step 5: Run the doctest for the new example**

```bash
uv run pytest --doctest-modules src/phenotypic/_core/_image_parts/_image_io_handler.py -q
```

Expected: PASS. The `save2zarr` docstring example must actually run — CLAUDE.md requires
runnable doctests using `load_synth_yeast_plate()`.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_core/_image_parts/_image_io_handler.py tests/unit/core/test_image_zarr_roundtrip.py tests/unit/sdk_/test_bundle_layout.py
git commit -m "feat(core): add save2zarr / load_zarr / load_layer_zarr

Arrays, then OME/zarr.json, then the root zarr.json last, then a rename
promote -- so an interrupted write reads as absent, not as partial. rgb is
omitted entirely when empty, which moves the primary series and the objmap
label to gray; readers resolve the label through phenotypic.labels.objmap
rather than a hard-coded path. objmap is always written, zeros included,
because valid_staged_store requires it after Stage 1. A failed OME-XML
build drops the whole OME/ group rather than emitting a partial one."
```

---

### Task 2.3: `GridImage` grid state

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_grid_image_handler.py`
  (mirror `_save_image2hdfgroup` at line 464 and `_load_from_hdf5_group` at line 509)
- Test: `tests/unit/core/test_grid_image_zarr_roundtrip.py` (create)

**Interfaces:**
- Consumes: `Image._build_store_attributes`, `Image._load_from_store`.
- Produces: `GridImage._build_store_attributes` (override, adds `phenotypic.grid`) and
  `GridImage._load_from_store` (override, reads it back).

**Constraints specific to this task:**
- `phenotypic.grid` is `{"nrows": int, "ncols": int, "grid_finder": {"class": …, "params": …}}`,
  serialized with `SerializablePipeline._serialize_single_operation` exactly as the HDF
  path does at line 495.
- Deserialization failure warns and falls back to the default finder — mirroring line 543 —
  rather than raising. Grid state is recoverable; the image is not worth losing over it.
- `grid.nrows`/`ncols` are **not** projected as NGFF `plate` metadata (locked §2.5): HCS
  requires one image group per well, which would multiply group count by `nrows × ncols`
  for no reader benefit.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_grid_image_zarr_roundtrip.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic import GridImage
from phenotypic.grid import AutoGridFinder
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes
from phenotypic.data import load_synth_yeast_plate


@pytest.fixture
def grid() -> GridImage:
    return GridImage(load_synth_yeast_plate(), nrows=16, ncols=24)


def test_grid_dimensions_round_trip(grid: GridImage, tmp_path: Path) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store)
    assert (back.nrows, back.ncols) == (16, 24)


def test_grid_finder_round_trips_by_class_and_params(tmp_path: Path) -> None:
    grid = GridImage(load_synth_yeast_plate(), grid_finder=AutoGridFinder(nrows=8, ncols=12))
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store)
    assert type(back.grid_finder).__name__ == "AutoGridFinder"
    assert (back.grid_finder.nrows, back.grid_finder.ncols) == (8, 12)


def test_grid_block_lives_under_phenotypic_not_ome(grid: GridImage, tmp_path: Path) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert PhenotypicAttr.GRID in root["attributes"]["phenotypic"]
    assert "plate" not in json.dumps(root["attributes"]["ome"])


def test_no_hcs_plate_metadata_is_emitted(grid: GridImage, tmp_path: Path) -> None:
    """HCS would need one image group per well: 16x24 = 384 groups, no benefit."""
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    groups = [p.name for p in store.iterdir() if p.is_dir()]
    assert set(groups) <= {"OME", "rgb", "gray", "detect_mat"}


def test_corrupt_grid_finder_warns_and_falls_back(grid: GridImage, tmp_path: Path) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    root_path = store / "zarr.json"
    payload = json.loads(root_path.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"][PhenotypicAttr.GRID]["grid_finder"] = {
        "class": "NoSuchFinder",
        "params": {},
    }
    root_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.warns(UserWarning, match="GridFinder"):
        back = GridImage.load_zarr(store)
    assert back.grid_finder is not None


def test_explicit_kwargs_take_priority_over_stored_grid(
    grid: GridImage, tmp_path: Path
) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store, nrows=8, ncols=12)
    assert (back.nrows, back.ncols) == (8, 12)


def test_image_class_records_gridimage(grid: GridImage, tmp_path: Path) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert block[PhenotypicAttr.IMAGE_CLASS] == "GridImage"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/core/test_grid_image_zarr_roundtrip.py -v
```

Expected: FAIL — `phenotypic.grid` is absent from the block.

- [ ] **Step 3: Implement the overrides**

```python
    def _build_store_attributes(self, *, series_names, levels, sections, work_id) -> dict:
        """Add ``phenotypic.grid`` to the base attributes block.

        ``nrows``/``ncols`` are deliberately **not** projected as NGFF ``plate``
        metadata: HCS requires each well to be a separate image group, while
        PhenotypicTypic's grid is a virtual partition of one array.
        """
        from phenotypic._core._pipeline_parts._serializable_pipeline import (
            SerializablePipeline,
        )
        from phenotypic.sdk_.ngff_ import PhenotypicAttr

        block = super()._build_store_attributes(
            series_names=series_names,
            levels=levels,
            sections=sections,
            work_id=work_id,
        )
        grid: dict = {"nrows": int(self.nrows), "ncols": int(self.ncols)}
        if self.grid_finder is not None:
            grid["grid_finder"] = {
                "class": type(self.grid_finder).__name__,
                "params": SerializablePipeline._serialize_single_operation(
                    self.grid_finder
                ),
            }
        block[PhenotypicAttr.GRID] = grid
        return block

    @classmethod
    def _load_from_store(cls, path, block, **kwargs):
        """Restore ``nrows``/``ncols``/``grid_finder`` before the base loader.

        Uses ``setdefault`` so explicit caller kwargs take priority, mirroring
        the HDF path. A deserialization failure warns and falls back to the
        default finder rather than raising: grid state is recoverable, the
        image is not worth losing over it.
        """
        import warnings

        from phenotypic.sdk_.ngff_ import PhenotypicAttr

        grid = block.get(PhenotypicAttr.GRID) or {}
        for key in ("nrows", "ncols"):
            if key in grid:
                try:
                    kwargs.setdefault(key, int(grid[key]))
                except (TypeError, ValueError):
                    pass
        payload = grid.get("grid_finder")
        if payload:
            from phenotypic._core._pipeline_parts._serializable_pipeline import (
                SerializablePipeline,
            )

            try:
                kwargs.setdefault(
                    "grid_finder",
                    SerializablePipeline._deserialize_operations({"__gf__": payload})[
                        "__gf__"
                    ],
                )
            except (KeyError, AttributeError, TypeError, ValueError) as exc:
                warnings.warn(
                    f"GridFinder deserialization failed "
                    f"({type(exc).__name__}: {exc}); falling back to the default "
                    "AutoGridFinder. Grid configuration may be incorrect.",
                    UserWarning,
                    stacklevel=2,
                )
        return super()._load_from_store(path, block, **kwargs)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/core/test_grid_image_zarr_roundtrip.py tests/unit/core/test_image_zarr_roundtrip.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_core/_image_parts/_grid_image_handler.py tests/unit/core/test_grid_image_zarr_roundtrip.py
git commit -m "feat(core): persist grid state in attributes.phenotypic.grid

nrows/ncols/grid_finder round-trip through the phenotypic block, not
through NGFF plate metadata: HCS requires one image group per well, which
would multiply a 16x24 store's group count by 384 for no reader benefit.
A corrupt grid_finder warns and falls back, as the HDF path already does."
```

---

### Task 2.4: `save_intermediate_zarr` for builder node previews

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`
  (replace `save_intermediate_layers`, line 905)
- Modify: `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py`
  (lines 1021, 1024, 1042, 1046, 1052 — **five** call sites, not three: two use
  `save2hdf5` and three use `save_intermediate_layers`)
- Modify: `src/phenotypic/gui/builder/_preview_cache.py` (lines 208, 212, 217, 284–286 —
  the DAG manifest's `"hdf"` key and the `base_00.h5` / `{i:02d}_{key}.h5` filenames)
- Test: `tests/unit/core/test_save_intermediate_zarr.py` (create),
  `tests/unit/gui/builder/test_preview_cache_manifest.py` (extend)

**Interfaces:**
- Consumes: `save2zarr` internals.
- Produces:
  ```python
  def save_intermediate_zarr(self, path, layers: tuple[str, ...]) -> Path
  ```
  and a builder DAG manifest whose per-node key is `"store"` (not `"hdf"`) holding
  `"<name>.ome.zarr"`.

**Constraints specific to this task:**
- `save_intermediate_zarr` writes a **single-level, no-pyramid** store —
  `levels=1` — a **private** argument on `_save_store`, not a CLI flag. Node previews are
  transient and small; pyramiding them would multiply builder-cache inodes for no gain.
- It still goes through the promote primitive: builder previews are written concurrently by
  Dash callbacks, so a half-written directory is a live failure mode there too.
- The manifest key rename `"hdf"` → `"store"` is a **GUI-visible contract change**.
  `_preview_cache.py:284-286` reads `parent_manifest["nodes"][pred_id]["hdf"]`, and
  `_preview_tiles.py:124` reads `node["hdf"]`. Both must change in the same commit, and any
  on-disk manifest from a previous session is invalidated — bump the manifest version
  constant so a stale cache is rebuilt rather than misread.
- Unknown layer names still raise `ValueError`, as `save_intermediate_layers` does at
  line 934.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/core/test_save_intermediate_zarr.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes
from phenotypic.data import load_synth_yeast_plate


@pytest.fixture
def plate() -> Image:
    return Image(load_synth_yeast_plate())


def test_writes_only_the_requested_layers(plate: Image, tmp_path: Path) -> None:
    store = plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("gray",))
    block = read_phenotypic_attributes(store)
    assert set(block[PhenotypicAttr.SERIES]) == {"gray"}


def test_is_single_level_by_design(plate: Image, tmp_path: Path) -> None:
    """Node previews are transient; pyramiding them multiplies cache inodes."""
    store = plate.save_intermediate_zarr(
        tmp_path / "n.ome.zarr", layers=("gray", "detect_mat")
    )
    assert read_phenotypic_attributes(store)[PhenotypicAttr.PYRAMID]["levels"] == 1
    assert not (store / "gray" / "1").exists()


def test_round_trips_through_load_layer_zarr(plate: Image, tmp_path: Path) -> None:
    store = plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("gray",))
    np.testing.assert_array_equal(Image.load_layer_zarr(store, "gray"), plate.gray[:])


def test_unknown_layer_names_raise(plate: Image, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown layer"):
        plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("nope",))


def test_uses_the_promote_primitive(plate: Image, tmp_path: Path, monkeypatch) -> None:
    """Dash callbacks write these concurrently; a half-written dir is live risk."""
    from phenotypic.sdk_ import ngff_

    calls: list[str] = []
    real = ngff_.promote_store
    monkeypatch.setattr(
        ngff_,
        "promote_store",
        lambda part, final, *, fsync: (calls.append(final.name), real(part, final, fsync=fsync))[1],
    )
    plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("gray",))
    assert calls == ["n.ome.zarr"]
```

Extend `tests/unit/gui/builder/test_preview_cache_manifest.py`:

```python
def test_manifest_node_key_is_store_not_hdf(tmp_path) -> None:
    """GUI-visible contract change; a stale 'hdf' key must not be read."""
    from phenotypic.gui.builder import _preview_cache

    manifest = _preview_cache.build_manifest_for_test(tmp_path)  # existing helper
    for node in manifest["nodes"].values():
        assert "hdf" not in node
        assert node["store"].endswith(".ome.zarr")


def test_manifest_version_bumped_so_stale_caches_rebuild() -> None:
    from phenotypic.gui.builder import _preview_cache

    assert _preview_cache.MANIFEST_VERSION >= 2
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/unit/core/test_save_intermediate_zarr.py tests/unit/gui/builder/test_preview_cache_manifest.py -v
```

Expected: `AttributeError: 'Image' object has no attribute 'save_intermediate_zarr'` and a
manifest still carrying `"hdf"`.

- [ ] **Step 3: Implement**

Replace `save_intermediate_layers` with:

```python
    def save_intermediate_zarr(self, path, layers: tuple[str, ...]) -> Path:
        """Save only *layers* as a single-level OME-Zarr store.

        Used for GUI builder node previews. No pyramid: previews are transient
        and small, and pyramiding them would multiply builder-cache inodes for
        no reader benefit. The promote primitive is still used, because Dash
        callbacks write these concurrently.

        Args:
            path: Target ``*.ome.zarr`` directory.
            layers: Subset of ``("rgb", "gray", "detect_mat", "objmap")``.

        Returns:
            The promoted store path.

        Raises:
            ValueError: If *layers* contains unknown names.
        """
        valid = {"rgb", "gray", "detect_mat", "objmap"}
        unknown = set(layers) - valid
        if unknown:
            raise ValueError(f"Unknown layer names: {unknown}")
        return self._save_store(
            path, series=tuple(layers), levels=1, work_id=None, durable=False
        )
```

Refactor `save2zarr`'s body into `_save_store(path, *, series, levels, work_id, durable)`
so both entry points share one implementation. `save2zarr` calls it with
`series=tuple(self._series_names())` and `levels=ngff_.pyramid_level_count(...)`;
`save_intermediate_zarr` calls it with `levels=1`. **`levels` is private** — there is no
CLI or public-API lever, so two stores in one tree can never disagree (P3).

In `_image_pipeline_core.py`, change all five call sites. The two `save2hdf5(...)` calls at
lines 1021 and 1042 become `save2zarr(output_dir / "base_00.ome.zarr")` and
`save2zarr(output_dir / f"{i:02d}_{key}.ome.zarr")`; the three `save_intermediate_layers`
calls become `save_intermediate_zarr` with the same `layers=` argument and the
`.ome.zarr` name.

In `_preview_cache.py`, rename the manifest key and bump the version:

```python
#: Manifest schema version. Bumped when the per-node artifact moved from a
#: single ``.h5`` to an ``.ome.zarr`` store, so a manifest written by an older
#: session is rebuilt rather than misread.
MANIFEST_VERSION: Final[int] = 2
```

`_describe` writes `"store": filename` instead of `"hdf": filename`, and the predecessor
lookup at lines 284–286 reads `parent_manifest["nodes"][pred_id]["store"]`.

- [ ] **Step 4: Run the tests plus the builder GUI suite**

```bash
uv run pytest tests/unit/core/test_save_intermediate_zarr.py tests/unit/gui/builder -v
```

Expected: all PASS. `_preview_tiles.py:124`'s `node["hdf"]` will fail here if missed — fix
it in this task, not in Phase 4.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_core src/phenotypic/gui/builder tests/unit/core/test_save_intermediate_zarr.py tests/unit/gui/builder
git commit -m "feat(core): replace save_intermediate_layers with save_intermediate_zarr

Node previews become single-level stores -- transient and small, so a
pyramid would only multiply builder-cache inodes. All five pipeline call
sites move (three save_intermediate_layers plus two save2hdf5, which an
earlier count missed). The builder DAG manifest's per-node key becomes
'store'; MANIFEST_VERSION is bumped so a manifest from an older session is
rebuilt rather than misread through a key that no longer exists."
```

---

### Task 2.5: NGFF conformance harness

**Files:**
- Create: `tests/_ngff_conformance.py`
- Test: `tests/unit/core/test_ngff_conformance.py`

**Interfaces:**
- Consumes: `tests/fixtures/ngff/0.5/*.schema` (Task 0.2).
- Produces:
  ```python
  def assert_store_conforms(store_path: Path) -> None
  ```
  imported by every later phase that writes a store.

**Constraints specific to this task:**
- Validation failure **fails the suite**. It is never downgraded to a warning and never
  skipped on a missing dependency or fixture — a check that cannot run must fail.
- Validate three surfaces: every image series' group `zarr.json` against `image.schema`,
  the label group's against `label.schema`, and `OME/zarr.json` against `ome.schema`.
- **Validate `payload["attributes"]`, NOT `payload["attributes"]["ome"]`.** Every one of the
  three schemas is rooted at the *attributes* object — verified by download:

  ```json
  {"$id": ".../image.schema", "description": "The zarr.json attributes key",
   "type": "object", "properties": {"ome": {…}}, "required": ["ome"]}
  ```

  Passing the inner `ome` block fails with `'ome' is a required property` on **every**
  store, so all seven tests here plus every downstream `assert_store_conforms` call in
  Phases 3 and 5 would fail. Recorded as OPEN-QUESTIONS **B2**.
- **Resolve `_version.schema` from the vendored copy, never over the network.** All three
  schemas carry exactly one remote `$ref` to it. `jsonschema` >= 4.18 does not fetch remote
  refs — it raises `referencing.exceptions.Unresolvable`, which is **not** a
  `ValidationError`, so an unregistered ref makes the suite *error* rather than fail and
  leaves offline CI with no fallback. Build a `referencing.Registry` keyed on each file's
  `$id` and pass it to the validator.
- The published schemas are stricter than the prose in two places (`ome.schema` requires
  `["series", "version"]`; `label.schema` requires `["image-label", "version"]`) and
  **looser** in two others (`$defs/image-label` has no `required` list, so `colors` is
  optional; `$defs/omero` requires only `channels`, and the channel item has no `required`
  list at all). The harness reports what the schemas say; PhenoTypic policy that goes beyond
  them is asserted separately, in Phase 1 Task 1.4's unit tests.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_ngff_conformance.py`:

```python
"""Every written store must validate against the vendored NGFF schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic import GridImage, Image
from phenotypic.data import load_synth_yeast_plate
from tests._ngff_conformance import assert_store_conforms


def test_a_written_image_store_conforms(tmp_path: Path) -> None:
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    assert_store_conforms(store)


def test_a_written_grid_store_conforms(tmp_path: Path) -> None:
    store = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12).save2zarr(
        tmp_path / "g.ome.zarr"
    )
    assert_store_conforms(store)


def test_an_rgb_less_store_conforms(tmp_path: Path) -> None:
    image = Image(load_synth_yeast_plate())
    image.rgb.clear()
    assert_store_conforms(image.save2zarr(tmp_path / "gray.ome.zarr"))


def test_a_single_level_node_preview_store_conforms(tmp_path: Path) -> None:
    """The levels=1 path (builder node previews) must conform too."""
    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "n.ome.zarr", layers=("gray", "detect_mat", "objmap")
    )
    assert_store_conforms(store)


def test_a_remote_ref_resolves_from_the_vendored_copy(tmp_path: Path) -> None:
    """An unregistered $ref raises Unresolvable, which is NOT a ValidationError,
    so the suite would error rather than fail. Verified: all three schemas
    reference _version.schema by absolute URL."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    assert_store_conforms(store)  # would raise Unresolvable without the registry


def test_a_wrong_version_string_is_rejected(tmp_path: Path) -> None:
    """Proves the resolved _version.schema enum is actually enforced."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    group = store / "gray" / "zarr.json"
    payload = json.loads(group.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["version"] = "0.4"
    group.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_partial_omero_is_rejected(tmp_path: Path) -> None:
    """Proves the gate has teeth: the exact defect an earlier draft would ship.

    Note what is and is not schema-enforced here. `$defs/omero` requires only
    `channels`, and the channel item has no `required` list -- so a channel
    missing `color` would validate. But `window`, *if present*, requires all
    four of start/min/end/max, which is what this truncation violates. Emitting
    the complete block is PhenoTypic policy (asserted in Phase 1 Task 1.4);
    this test covers the part the schema does enforce.
    """
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    group = store / "gray" / "zarr.json"
    payload = json.loads(group.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["omero"]["channels"][0]["window"] = {"max": 255}
    group.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_missing_image_label_is_rejected(tmp_path: Path) -> None:
    """label.schema requires image-label even though the prose says SHOULD."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    label_json = store / "rgb" / "labels" / "objmap" / "zarr.json"
    payload = json.loads(label_json.read_text(encoding="utf-8"))
    del payload["attributes"]["ome"]["image-label"]
    label_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_missing_series_is_rejected(tmp_path: Path) -> None:
    """ome.schema requires series."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    ome_json = store / "OME" / "zarr.json"
    payload = json.loads(ome_json.read_text(encoding="utf-8"))
    del payload["attributes"]["ome"]["series"]
    ome_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/core/test_ngff_conformance.py -v
```

Expected: `ModuleNotFoundError: No module named 'tests._ngff_conformance'`.

- [ ] **Step 3: Write the harness**

Create `tests/_ngff_conformance.py`:

```python
"""Validate a written store against the vendored NGFF 0.5 JSON schemas.

Conformance is checked with ``jsonschema`` rather than ``ome-zarr-models``,
which pins ``pydantic<2.13``; pydantic 2.13 has already shipped, so adopting it
would hold the project a release behind today. There is no ``[tool.uv]
conflicts`` block, so even a dev-group-only cap would bind the whole locked
environment.

A failure here fails the suite. It is never downgraded to a warning and never
skipped: a check that cannot run must fail.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import jsonschema
import referencing
import referencing.jsonschema

SCHEMA_DIR = Path(__file__).resolve().parent / "fixtures" / "ngff" / "0.5"


SCHEMA_NAMES = ("image.schema", "label.schema", "ome.schema", "_version.schema")


def _schema(name: str) -> dict:
    path = SCHEMA_DIR / name
    if not path.is_file():
        raise AssertionError(
            f"vendored NGFF schema missing: {path}. A conformance check that "
            "cannot run must fail, never skip."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _registry() -> referencing.Registry:
    """Resolve every ``$ref`` from the vendored copies, never over the network.

    All three schemas reference ``_version.schema`` by absolute URL. jsonschema
    >= 4.18 does not fetch remote refs; it raises
    ``referencing.exceptions.Unresolvable``, which is not a ``ValidationError``
    -- so without this the suite errors instead of failing, and offline CI has
    no fallback at all.
    """
    registry = referencing.Registry()
    for name in SCHEMA_NAMES:
        payload = _schema(name)
        resource = referencing.Resource.from_contents(
            payload, default_specification=referencing.jsonschema.DRAFT202012
        )
        registry = resource @ registry
    return registry


def _attributes(group_dir: Path) -> dict:
    """Return the group's whole ``attributes`` mapping.

    NOT ``attributes["ome"]``: each vendored schema is rooted at the attributes
    object itself (``"description": "The zarr.json attributes key"``,
    ``"required": ["ome"]``). Passing the inner block fails with
    ``'ome' is a required property`` on every store ever written.
    """
    payload = json.loads((group_dir / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"]


def _validate(attributes: dict, schema_name: str, where: Path) -> None:
    schema = _schema(schema_name)
    validator = jsonschema.Draft202012Validator(schema, registry=_registry())
    try:
        validator.validate(attributes)
    except jsonschema.ValidationError as exc:
        raise AssertionError(
            f"{where} does not conform to {schema_name}: {exc.message} "
            f"(at {list(exc.absolute_path)})"
        ) from exc


def assert_store_conforms(store_path: Path) -> None:
    """Validate every conformance surface of one written store.

    Args:
        store_path: Path to a promoted ``*.ome.zarr`` directory.

    Raises:
        AssertionError: On any schema violation, missing schema fixture, or
            structurally absent group.
    """
    from phenotypic.sdk_.ngff_ import (
        OME_GROUP,
        PhenotypicAttr,
        read_phenotypic_attributes,
    )

    store = Path(store_path)
    block = read_phenotypic_attributes(store)

    for member in block[PhenotypicAttr.SERIES].values():
        _validate(_attributes(store / member), "image.schema", store / member)

    for member in block[PhenotypicAttr.LABELS].values():
        _validate(_attributes(store / member), "label.schema", store / member)

    ome_group = store / OME_GROUP
    if ome_group.is_dir():
        _validate(_attributes(ome_group), "ome.schema", ome_group)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/core/test_ngff_conformance.py -v
```

Expected: all PASS. If a real store fails validation, **fix the writer**, not the harness —
that is the harness doing its job.

- [ ] **Step 5: Commit**

```bash
git add tests/_ngff_conformance.py tests/unit/core/test_ngff_conformance.py
git commit -m "test: validate written stores against the vendored NGFF schemas

jsonschema, not ome-zarr-models, which pins pydantic<2.13. Three negative
tests prove the gate has teeth: a truncated omero window, a missing
image-label, and a missing series each fail, which are exactly the three
places the published schemas are stricter than the prose."
```

---

## Phase 2 exit criteria

- [ ] `uv run pytest tests/unit/core/test_image_zarr_roundtrip.py tests/unit/core/test_grid_image_zarr_roundtrip.py tests/unit/core/test_ngff_conformance.py tests/unit/core/test_save_intermediate_zarr.py -q` is green.
- [ ] `uv run pytest tests/unit/gui/builder -q` is green (manifest key rename).
- [ ] `uv run pytest --doctest-modules src/phenotypic/_core/_image_parts/_image_io_handler.py -q` is green.
- [ ] `grep -rn '\.ome\.zarr"' src/phenotypic --include='*.py' | grep -v 'ngff_.py\|_io_constants.py'` returns nothing.
- [ ] The HDF write path still works: `uv run pytest tests/unit/core/test_image_hdf_roundtrip.py -q` is green.
# Phase 3 — CLI write path and the staged-GPU engine

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §3.2–§3.7, §4.3, §4.4.

**Depends on:** Phase 2.
**Runs in parallel with:** Phases 4 and 5.

This is the safety-critical phase. The staged engine's resume classifier is what decides
whether a finished image is reprocessed or skipped, and all three defects the spec's
independent review caught lived here. **Task 3.4's differential test is the gate** — it is
the test that would have caught all three, and it must be written before the classifier is
touched.

---

### Task 3.1: `save_image_store` on `OutputManager`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_output_manager.py`
  (add beside `save_image_hdf`, line 1633; `save_image_layers` at line 1688 is already
  deprecated and is **not** ported)
- Test: `tests/unit/cli/test_cli_output_manager.py` (extend)

**Interfaces:**
- Consumes: `Image.save2zarr`, `zarr_store_path`, `ngff_.durable_writes_enabled`.
- Produces:
  ```python
  def save_image_store(
      self,
      image: "Image",
      dataset_name: str,
      image_stem: str,
      *,
      work_id: str | None = None,
      durable: bool | None = None,
  ) -> Optional[Path]
  ```

**Constraints specific to this task:**
- The old signature's `root_attributes: Mapping[str, str] | None` is **replaced by an
  explicit `work_id` argument**. Today the CLI patches `phenotypic_work_id` in post-write
  via `h5py.File(tmp, "r+")` (line 1666); under the new ordering invariant the root
  `zarr.json` is written last, so a post-hoc patch is impossible by construction.
- **`save_image_hdf` has three callers, not two.** Verified:
  `_cli_staged_workers.py:125` and `:225` pass
  `root_attributes={"phenotypic_work_id": work_id}` (at `:129` and `:229`) and become
  `work_id=work_id`; **`_cli_process_single.py:183` passes none** and becomes a bare
  `save_image_store(image, dataset_name, image_stem)`. The spec's §4.4 lists
  `_cli_process_single.py` only as a "loader swap", so its writer swap is under-specified
  there. There is also a name-monkeypatch at
  `tests/integration/cli/test_staged_gpu_local.py:742` that must be renamed with it.
- Failure semantics are preserved exactly: log a warning and return `None`, never raise.
  The staged workers turn `None` into a `RuntimeError` themselves (lines 133 and 231), and
  that layering must not change.
- On failure, clean up the `.part` directory rather than the file — `tmp_path.unlink()` at
  line 1676 becomes `shutil.rmtree(part, ignore_errors=True)`.
- `save_image_hdf` is **kept** in this phase and removed in Phase 6, so a half-migrated
  tree never has two writers fighting.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/cli/test_cli_output_manager.py`:

```python
def test_save_image_store_writes_under_results_dataset_zarr(tmp_path) -> None:
    from phenotypic import Image
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.data import load_synth_yeast_plate

    manager = _make_manager(tmp_path)  # existing helper in this module
    saved = manager.save_image_store(
        Image(load_synth_yeast_plate()), "ds", "img"
    )
    assert saved == zarr_store_path(tmp_path, "ds", "img")
    assert saved.is_dir()


def test_save_image_store_writes_work_id_at_write_time(tmp_path) -> None:
    """The root zarr.json is written last, so a post-hoc patch is impossible."""
    from phenotypic import Image
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes
    from phenotypic.data import load_synth_yeast_plate

    manager = _make_manager(tmp_path)
    saved = manager.save_image_store(
        Image(load_synth_yeast_plate()), "ds", "img", work_id="w-7"
    )
    assert read_phenotypic_attributes(saved)[PhenotypicAttr.WORK_ID] == "w-7"


def test_save_image_store_returns_none_and_logs_on_failure(tmp_path, monkeypatch, caplog) -> None:
    """Preserves save_image_hdf's contract: the workers raise, not the manager."""
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    manager = _make_manager(tmp_path)
    monkeypatch.setattr(
        Image, "save2zarr", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    assert manager.save_image_store(Image(load_synth_yeast_plate()), "ds", "img") is None
    assert any("Failed to save" in record.message for record in caplog.records)


def test_save_image_store_cleans_up_the_part_directory_on_failure(tmp_path, monkeypatch) -> None:
    from phenotypic import Image
    from phenotypic.sdk_ import dataset_zarr_dir
    from phenotypic.data import load_synth_yeast_plate

    manager = _make_manager(tmp_path)
    monkeypatch.setattr(
        Image, "save2zarr", lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
    )
    manager.save_image_store(Image(load_synth_yeast_plate()), "ds", "img")
    leftovers = list(dataset_zarr_dir(tmp_path, "ds").glob("*.part"))
    assert leftovers == []


def test_save_image_store_result_passes_valid_staged_store(tmp_path) -> None:
    from phenotypic import Image
    from phenotypic.sdk_.ngff_ import valid_staged_store
    from phenotypic.data import load_synth_yeast_plate

    manager = _make_manager(tmp_path)
    saved = manager.save_image_store(Image(load_synth_yeast_plate()), "ds", "img")
    assert valid_staged_store(saved) is True
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/cli/test_cli_output_manager.py -k save_image_store -v
```

Expected: `AttributeError: 'OutputManager' object has no attribute 'save_image_store'`.

- [ ] **Step 3: Implement**

```python
    def save_image_store(
        self,
        image: "Image",
        dataset_name: str,
        image_stem: str,
        *,
        work_id: str | None = None,
          durable: bool | None = None,
    ) -> Optional[Path]:
        """Save a processed image as an OME-Zarr store under ``results/<ds>/zarr/``.

        Atomicity comes from :func:`phenotypic.sdk_.ngff_.promote_store`: the
        image is built into a uuid-suffixed ``.part`` sibling and promoted by
        directory rename.

        ``work_id`` is a first-class argument rather than the old
        ``root_attributes`` mapping. The store's root ``zarr.json`` is written
        last so an interrupted write reads as absent, which makes the previous
        post-write patch (``h5py.File(tmp, "r+")``) impossible by construction.

        Args:
            image: Image object with processing results.
            dataset_name: Dataset name.
            image_stem: Image filename without extension.
            work_id: CLI work id, written into ``attributes.phenotypic``.
            durable: ``fsync`` before promoting; ``None`` auto-detects SLURM.

        Returns:
            Path where the store was promoted, or ``None`` if saving failed.
            Callers that require publication (the staged workers) turn ``None``
            into a ``RuntimeError`` themselves; that layering is deliberate.
        """
        import shutil

        from phenotypic.sdk_ import zarr_store_path
        from phenotypic.sdk_.ngff_ import new_part_path

        final_path = zarr_store_path(self.output_dir, dataset_name, image_stem)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            saved = image.save2zarr(
                final_path,
                work_id=work_id,
                durable=durable,
            )
            logger.info("Saved OME-Zarr store for %s/%s", dataset_name, image_stem)
            return saved
        except Exception as e:
            for orphan in final_path.parent.glob(f".{final_path.name}.*.part"):
                shutil.rmtree(orphan, ignore_errors=True)
            logger.warning(
                "Failed to save OME-Zarr store for %s/%s: %s: %s",
                dataset_name,
                image_stem,
                type(e).__name__,
                e,
            )
            return None
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/cli/test_cli_output_manager.py -v
```

Expected: all PASS, including the pre-existing `save_image_hdf` tests — that path is
untouched here.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_output_manager.py tests/unit/cli/test_cli_output_manager.py
git commit -m "feat(cli): add save_image_store

work_id becomes an explicit argument instead of a root_attributes mapping
patched in post-write: the store's root zarr.json is written last, so
patching it afterwards would violate the ordering invariant that makes an
interrupted write read as absent. Failure still returns None and logs --
the staged workers are what turn that into a RuntimeError."
```

---

### Task 3.2: The consumable Stage-2 token

**Files:**
- Create: `src/phenotypic/_cli/_cli_stage2_token.py`
- Test: `tests/unit/cli/test_cli_stage2_token.py` (create)
- (`src/phenotypic/_cli/_cli_sidecar.py` and `tests/unit/cli/test_cli_sidecar.py` are
  deleted in Task 3.5, once every caller has moved.)

**Interfaces:**
- Consumes: `progress_dir`, `atomic_write_with_writer`.
- Produces:
  ```python
  def stage2_token_path(output_dir: Path, dataset: str, image_stem: str) -> Path
  def write_stage2_token(output_dir, dataset, image_stem, *, work_id: str | None, objmap_shape: tuple[int, int]) -> Path
  def stage2_token_exists(output_dir, dataset, image_stem) -> bool
  def read_stage2_token(output_dir, dataset, image_stem) -> dict
  def delete_stage2_token(output_dir, dataset, image_stem) -> None

  def stage2_raw_path(output_dir: Path, dataset: str, image_stem: str) -> Path
  def write_stage2_raw(output_dir, dataset, image_stem, array: np.ndarray) -> Path
  def load_stage2_raw(output_dir, dataset, image_stem) -> np.ndarray
  def delete_stage2_raw(output_dir, dataset, image_stem) -> None
  ```

**The raw array is retained, deliberately (OPEN-QUESTIONS D1).**

`<output>/.phenotypic/progress/stage2_raw/<dataset>/<stem>.npy` holds Stage 2's **raw**
detector output — pre-`_write_object_output`, pre-`drop_frame_background`, pre-relabel — and
Stage 3 consumes it. This is not a leftover of the old sidecar; it is the property the old
sidecar provided and that nothing else does.

Without it, Stage 3's input is the store's own objmap, which Stage 3 then re-promotes over —
so the raw output is destroyed the moment Stage 3 first succeeds. The retry window is real:
`save_image_store` lands at `_cli_staged_workers.py:225` but the completion marker is not
written until `:251`, with `save_overlay` and `PlotCoordinator.emit_image` in between. A
timeout there leaves the classifier reading `"stage3"`, and the second pass runs
`_write_object_output` on already-refined labels. `drop_frame_background`
(`_objmap_accessor.py:498-509`) zeroes the label owning the plurality of border pixels
**after excluding the already-zeroed background**, so the plurality falls to whichever real
colony touches the frame most — and that colony is silently deleted, once per retry.

The store's objmap is still written in place by Stage 2 (§3.4) for interop; the raw `.npy`
is what makes Stage 3 replayable.

**Constraints specific to this task:**
- Path is `<output>/.phenotypic/progress/stage2_done/<dataset>/<stem>.json`, i.e.
  `progress_dir(output_dir) / "stage2_done" / dataset / f"{stem}.json"` — the same shape as
  `stage3_completion_marker_path` (`_cli_staged_resume.py:113`), which uses
  `"stage3_complete"`.
- Written atomically (temp + rename) via `atomic_write_with_writer`, exactly as
  `write_sidecar` does today.
- The token carries `work_id` and the objmap's level-0 shape. It is **consumable**:
  `delete_stage2_token` mirrors `delete_sidecar`, and the resume planner's `"complete"`
  branch tests its **absence**.
- **NGFF metadata never carries resume state.** In particular `ome.labels` is not a
  substitute: a durable labels list makes the `"complete"` conjunct permanently false, so
  every finished image is reprocessed forever and `migrate_legacy_stage3_markers` is
  silently disabled. `zarr.Group.members()` also enumerates children by store listing and
  returns a partially written `objmap`, so the labels list is not even the only discovery
  path.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/cli/test_cli_stage2_token.py`:

```python
"""The Stage-2 token replaces the .npy sidecar. It must be consumable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic._cli._cli_stage2_token import (
    delete_stage2_token,
    read_stage2_token,
    stage2_token_exists,
    stage2_token_path,
    write_stage2_token,
)
from phenotypic.sdk_ import progress_dir


def test_token_lives_under_progress_not_in_the_store(tmp_path: Path) -> None:
    """Resume state lives where the rest of it already lives."""
    path = stage2_token_path(tmp_path, "ds", "img")
    assert path == progress_dir(tmp_path) / "stage2_done" / "ds" / "img.json"
    assert ".ome.zarr" not in str(path)


def test_write_then_exists_then_delete(tmp_path: Path) -> None:
    assert stage2_token_exists(tmp_path, "ds", "img") is False
    write_stage2_token(tmp_path, "ds", "img", work_id="w-1", objmap_shape=(64, 48))
    assert stage2_token_exists(tmp_path, "ds", "img") is True
    delete_stage2_token(tmp_path, "ds", "img")
    assert stage2_token_exists(tmp_path, "ds", "img") is False


def test_delete_is_idempotent(tmp_path: Path) -> None:
    delete_stage2_token(tmp_path, "ds", "img")
    delete_stage2_token(tmp_path, "ds", "img")


def test_token_carries_work_id_and_objmap_shape(tmp_path: Path) -> None:
    write_stage2_token(tmp_path, "ds", "img", work_id="w-9", objmap_shape=(64, 48))
    payload = read_stage2_token(tmp_path, "ds", "img")
    assert payload["work_id"] == "w-9"
    assert tuple(payload["objmap_shape"]) == (64, 48)


def test_token_is_written_atomically(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []
    import phenotypic._cli._cli_stage2_token as module

    real = module.atomic_write_with_writer
    monkeypatch.setattr(
        module,
        "atomic_write_with_writer",
        lambda final, writer: (seen.append(str(final)), real(final, writer))[1],
    )
    write_stage2_token(tmp_path, "ds", "img", work_id=None, objmap_shape=(2, 2))
    assert seen == [str(stage2_token_path(tmp_path, "ds", "img"))]


def test_token_is_valid_json(tmp_path: Path) -> None:
    write_stage2_token(tmp_path, "ds", "img", work_id=None, objmap_shape=(2, 2))
    json.loads(stage2_token_path(tmp_path, "ds", "img").read_text(encoding="utf-8"))


def test_read_missing_token_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_stage2_token(tmp_path, "ds", "img")


# --- the retained raw array (D1) -------------------------------------------


def test_raw_array_lives_beside_the_token(tmp_path: Path) -> None:
    from phenotypic._cli._cli_stage2_token import stage2_raw_path

    assert stage2_raw_path(tmp_path, "ds", "img") == (
        progress_dir(tmp_path) / "stage2_raw" / "ds" / "img.npy"
    )


def test_raw_array_round_trips_exactly(tmp_path: Path) -> None:
    """Stage 3 replays from this, so it must be bit-exact."""
    import numpy as np

    from phenotypic._cli._cli_stage2_token import load_stage2_raw, write_stage2_raw

    array = np.arange(64, dtype=np.uint16).reshape(8, 8)
    write_stage2_raw(tmp_path, "ds", "img", array)
    np.testing.assert_array_equal(load_stage2_raw(tmp_path, "ds", "img"), array)
    assert load_stage2_raw(tmp_path, "ds", "img").dtype == array.dtype


def test_raw_array_is_written_atomically(tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    import phenotypic._cli._cli_stage2_token as module

    seen: list[str] = []
    real = module.atomic_write_with_writer
    monkeypatch.setattr(
        module,
        "atomic_write_with_writer",
        lambda final, writer: (seen.append(str(final)), real(final, writer))[1],
    )
    module.write_stage2_raw(tmp_path, "ds", "img", np.zeros((2, 2), dtype=np.uint16))
    assert seen == [str(module.stage2_raw_path(tmp_path, "ds", "img"))]


def test_raw_delete_is_idempotent(tmp_path: Path) -> None:
    from phenotypic._cli._cli_stage2_token import delete_stage2_raw

    delete_stage2_raw(tmp_path, "ds", "img")
    delete_stage2_raw(tmp_path, "ds", "img")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/cli/test_cli_stage2_token.py -v
```

Expected: `ModuleNotFoundError: No module named 'phenotypic._cli._cli_stage2_token'`.

- [ ] **Step 3: Write the module**

```python
"""Consumable Stage-2 completion token for the staged GPU engine.

Replaces the ``.npy`` objmap sidecar. Stage 2 now writes the detector output
directly into the promoted store's label array and drops this token; Stage 3
consumes the token, exactly as it used to consume the sidecar.

The token is deliberately **not** NGFF metadata. Using ``ome.labels`` as the
"Stage 2 done" signal is not an exact replacement for ``sidecar_exists()`` and
would break resume in two ways:

* The sidecar is consumable -- ``delete_sidecar`` ran at the end of Stage 3 and
  the resume planner's ``"complete"`` branch tests its **absence**. A durable
  labels list makes that conjunct permanently false, so ``"complete"`` never
  fires and every finished image is reprocessed. It also silently disables
  ``migrate_legacy_stage3_markers``.
* The labels list is not the only discovery path: ``zarr.Group.members()``
  enumerates children by store listing and returns a partially written
  ``objmap``, which reads as a mix of real labels and ``fill_value``. NGFF only
  says label images SHOULD be listed; it grants no exclusivity.

Consequently, NGFF metadata never carries resume state. Resume state lives in
``.phenotypic/progress/``, where the rest of it already lives.
"""

from __future__ import annotations

import json
from pathlib import Path

from phenotypic.sdk_ import atomic_write_with_writer, progress_dir

_STAGE2_DIR = "stage2_done"


def stage2_token_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """``<output>/.phenotypic/progress/stage2_done/<dataset>/<stem>.json``."""
    return progress_dir(output_dir) / _STAGE2_DIR / dataset / f"{image_stem}.json"


def write_stage2_token(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    *,
    work_id: str | None,
    objmap_shape: tuple[int, int],
) -> Path:
    """Atomically record that Stage 2 published this image's label array.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Image stem.
        work_id: Work id this Stage-2 result belongs to, or ``None``.
        objmap_shape: Level-0 ``(y, x)`` extent of the written objmap.

    Returns:
        The token path.
    """
    final = stage2_token_path(output_dir, dataset, image_stem)
    payload = {
        "work_id": work_id,
        "objmap_shape": [int(objmap_shape[0]), int(objmap_shape[1])],
    }

    def _write(path: str) -> None:
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    atomic_write_with_writer(final, _write)
    return final


def stage2_token_exists(output_dir: Path, dataset: str, image_stem: str) -> bool:
    """Return whether Stage 2 has published and Stage 3 has not yet consumed."""
    return stage2_token_path(output_dir, dataset, image_stem).is_file()


def read_stage2_token(output_dir: Path, dataset: str, image_stem: str) -> dict:
    """Read the token payload.

    Raises:
        FileNotFoundError: If the token does not exist.
    """
    return json.loads(
        stage2_token_path(output_dir, dataset, image_stem).read_text(encoding="utf-8")
    )


def delete_stage2_token(output_dir: Path, dataset: str, image_stem: str) -> None:
    """Consume the token. Idempotent, mirroring ``delete_sidecar``."""
    stage2_token_path(output_dir, dataset, image_stem).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The retained raw detector output
# ---------------------------------------------------------------------------

_STAGE2_RAW_DIR = "stage2_raw"


def stage2_raw_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """``<output>/.phenotypic/progress/stage2_raw/<dataset>/<stem>.npy``."""
    return progress_dir(output_dir) / _STAGE2_RAW_DIR / dataset / f"{image_stem}.npy"


def write_stage2_raw(
    output_dir: Path, dataset: str, image_stem: str, array: "np.ndarray"
) -> Path:
    """Atomically retain Stage 2's **raw** detector output for Stage 3 to replay.

    This is what makes Stage 3 idempotent under retry. Stage 3 re-promotes the
    store over its own objmap, so the store cannot serve as its own input a
    second time: on a replay ``_write_object_output`` would run again on
    already-refined labels, and ``drop_frame_background`` would zero whichever
    real colony touches the frame most -- silently, once per retry.

    Written before the token, so a crash between them leaves no token and
    Stage 2 simply re-runs.
    """
    final = stage2_raw_path(output_dir, dataset, image_stem)

    def _write(path: str) -> None:
        import numpy as np

        with open(path, "wb") as handle:
            np.save(handle, array)

    atomic_write_with_writer(final, _write)
    return final


def load_stage2_raw(output_dir: Path, dataset: str, image_stem: str) -> "np.ndarray":
    """Load the retained raw detector output.

    Raises:
        FileNotFoundError: If Stage 2 did not retain one.
    """
    import numpy as np

    return np.load(stage2_raw_path(output_dir, dataset, image_stem))


def delete_stage2_raw(output_dir: Path, dataset: str, image_stem: str) -> None:
    """Consume the raw array. Idempotent; always paired with the token."""
    stage2_raw_path(output_dir, dataset, image_stem).unlink(missing_ok=True)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/cli/test_cli_stage2_token.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_stage2_token.py tests/unit/cli/test_cli_stage2_token.py
git commit -m "feat(cli): add the consumable Stage-2 token

Replaces the .npy sidecar without moving resume state into NGFF metadata.
A durable ome.labels list would make the resume planner's 'complete'
conjunct permanently false -- every finished image reprocessed forever --
and would silently disable migrate_legacy_stage3_markers. The token lives
under .phenotypic/progress/ beside the Stage-3 marker it pairs with."
```

---

### Task 3.3: Stage 1, Stage 2, and Stage 3 workers

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py`
  (`stage1_preprocess_core` line 99, `stage2_detect_core` line 139,
  `ensure_staged_overlay` line 168, `stage3_merge_measure_core` line 193)
- Test: `tests/integration/cli/test_staged_store_stages.py` (create)

**Interfaces:**
- Consumes: `save_image_store` (3.1), the Stage-2 token (3.2), `valid_staged_store` (1.6).
- Produces: the same four function signatures, with `hdf` locals replaced by `store`.

**Constraints specific to this task:**
- **Stage 1** writes a complete store including a **zeros `objmap`** with its `ome.labels`
  list and `image-label` block. That is what lets `valid_staged_store` mirror
  `valid_staged_hdf` exactly, and it is what today's HDF writer already does.
- **Stage 2** opens the promoted store, overwrites `labels/objmap` **in place** with the
  detector output, then writes the Stage-2 token. It does **not** promote — an in-store
  label write is an intermediate, not a publish. Concurrent readers may observe a torn
  objmap; the completion marker, not the store's shape, is what gates consumers.
- **Stage 3 re-promotes the entire store.** This is not optional. Post-ops (refiners, size
  filters) mutate the objmap, and this re-save is what publishes the **post-refined**
  segmentation. Removing it would leave the label image holding raw detector output that
  disagrees with the parquet and with a single-pass run, violating the
  byte-identical-to-single-pass contract in `_cli/CLAUDE.md`.
- **Preserve the existing `work_id is None` guard verbatim.** Today
  `write_stage3_completion_marker` and `delete_sidecar` run only when `work_id is None`
  (`_cli_staged_workers.py:250-258`); the work-id path publishes markers elsewhere, in
  `_cli_staged_slurm_worker.py:409`. Port the token deletion into the **same** guard.
  Making it unconditional here would double-delete against the SLURM worker and change
  resume classification — this is exactly the kind of silent divergence Task 3.4's
  differential test exists to catch.
- Stage 2 must **not** rewrite pyramid levels 1..n of the objmap incorrectly: it rewrites
  every level, downsampling with `ngff_.downsample_label`. A stale level-1 objmap under a
  fresh level 0 is a silently wrong GUI overlay.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/cli/test_staged_store_stages.py`:

```python
"""Stage 1/2/3 against a real store. The post-refined objmap test is the point."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic._cli._cli_stage2_token import stage2_token_exists
from phenotypic.schema import OBJECT
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_.ngff_ import valid_staged_store

#: The measurement column is ``Object_Label``, not ``ObjectLabel`` --
#: ``schema/_object.py:7,22`` with ``category() == "Object"``. Resolve it
#: through the schema rather than spelling it, so a rename cannot silently
#: turn the most load-bearing test in this plan into a KeyError.


def test_stage1_publishes_a_store_with_a_zeros_objmap(staged_run) -> None:
    """valid_staged_store requires objmap; Stage 1 must emit it, zeros and all."""
    staged_run.run_stage1()
    store = zarr_store_path(staged_run.output_dir, "ds", "img")
    assert valid_staged_store(store) is True
    assert (Image.load_layer_zarr(store, "objmap") == 0).all()


def test_stage1_store_conforms(staged_run) -> None:
    from tests._ngff_conformance import assert_store_conforms

    staged_run.run_stage1()
    assert_store_conforms(zarr_store_path(staged_run.output_dir, "ds", "img"))


def test_stage2_writes_the_label_in_place_without_promoting(staged_run, monkeypatch) -> None:
    from phenotypic.sdk_ import ngff_

    staged_run.run_stage1()
    promotes: list[str] = []
    real = ngff_.promote_store
    monkeypatch.setattr(
        ngff_,
        "promote_store",
        lambda p, f, *, fsync: (promotes.append(f.name), real(p, f, fsync=fsync))[1],
    )
    staged_run.run_stage2()
    assert promotes == [], "Stage 2's label write is an intermediate, not a publish"


def test_stage2_drops_a_token_and_the_objmap_is_readable(staged_run) -> None:
    staged_run.run_stage1()
    staged_run.run_stage2()
    assert stage2_token_exists(staged_run.output_dir, "ds", "img") is True
    store = zarr_store_path(staged_run.output_dir, "ds", "img")
    assert Image.load_layer_zarr(store, "objmap").any()


def test_stage2_rewrites_every_pyramid_level_of_the_objmap(staged_run) -> None:
    """A stale level-1 under a fresh level 0 is a silently wrong GUI overlay."""
    staged_run.run_stage1()
    staged_run.run_stage2()
    store = zarr_store_path(staged_run.output_dir, "ds", "img")
    level0 = Image.load_layer_zarr(store, "objmap", level=0)
    level1 = Image.load_layer_zarr(store, "objmap", level=1)
    np.testing.assert_array_equal(level1, level0[::2, ::2])


def test_stage3_publishes_the_post_refined_objmap(staged_run_with_size_filter) -> None:
    """The round-trip test is blind to this: it never goes through the stages.

    Post-ops mutate the objmap. Without Stage 3's re-promote the stored label
    image holds raw detector output that disagrees with the parquet.
    """
    run = staged_run_with_size_filter  # post-op removes exactly one colony
    run.run_stage1()
    run.run_stage2()
    raw_labels = set(
        np.unique(
            Image.load_layer_zarr(
                zarr_store_path(run.output_dir, "ds", "img"), "objmap"
            )
        )
    ) - {0}
    run.run_stage3()
    published = set(
        np.unique(
            Image.load_layer_zarr(
                zarr_store_path(run.output_dir, "ds", "img"), "objmap"
            )
        )
    ) - {0}
    parquet_labels = set(run.read_measurements()[str(OBJECT.LABEL)].tolist())
    assert published == parquet_labels
    assert published < raw_labels, "the size filter should have removed a colony"


def test_stage3_consumes_the_token_and_the_raw_array(staged_run) -> None:
    from phenotypic._cli._cli_stage2_token import stage2_raw_path

    staged_run.run_stage1()
    staged_run.run_stage2()
    staged_run.run_stage3()
    assert stage2_token_exists(staged_run.output_dir, "ds", "img") is False
    assert not stage2_raw_path(staged_run.output_dir, "ds", "img").exists()


def test_stage3_is_idempotent_under_retry(staged_run_with_border_colony) -> None:
    """The D1 guard. A timeout between the promote and the completion marker
    leaves the classifier reading "stage3", so Stage 3 runs a second time.
    Replaying from the retained raw array must produce an identical result.

    Replaying from the STORE instead re-runs _write_object_output on
    already-refined labels, and drop_frame_background then zeroes whichever
    real colony touches the frame most -- silently, once per retry.
    """
    run = staged_run_with_border_colony  # a colony provably touches the frame
    run.run_stage1()
    run.run_stage2()
    run.run_stage3()
    store = zarr_store_path(run.output_dir, "ds", "img")
    once = Image.load_layer_zarr(store, "objmap").copy()
    measurements_once = run.read_measurements()

    run.simulate_timeout_after_promote()  # removes the marker, keeps token + raw
    run.run_stage3()

    np.testing.assert_array_equal(Image.load_layer_zarr(store, "objmap"), once)
    assert set(run.read_measurements()[str(OBJECT.LABEL)]) == set(
        measurements_once[str(OBJECT.LABEL)]
    )


def test_stage3_replays_from_the_raw_array_not_the_store(staged_run, monkeypatch) -> None:
    """Pins the input source, so a later 'simplification' cannot swap it back."""
    from phenotypic._cli import _cli_stage2_token

    staged_run.run_stage1()
    staged_run.run_stage2()
    reads: list[str] = []
    real = _cli_stage2_token.load_stage2_raw
    monkeypatch.setattr(
        _cli_stage2_token,
        "load_stage2_raw",
        lambda *a: (reads.append("raw"), real(*a))[1],
    )
    staged_run.run_stage3()
    assert reads == ["raw"]


def test_stage3_leaves_the_token_alone_on_the_work_id_path(staged_run_with_work_id) -> None:
    """Preserves today's guard: with a work_id, markers are published by the
    SLURM worker, not here. Making this unconditional double-deletes."""
    run = staged_run_with_work_id
    run.run_stage1()
    run.run_stage2()
    run.run_stage3()
    assert stage2_token_exists(run.output_dir, "ds", "img") is True


def test_stage3_republished_store_still_conforms(staged_run) -> None:
    from tests._ngff_conformance import assert_store_conforms

    staged_run.run_stage1()
    staged_run.run_stage2()
    staged_run.run_stage3()
    assert_store_conforms(zarr_store_path(staged_run.output_dir, "ds", "img"))


def test_stage3_raises_when_publication_fails(staged_run, monkeypatch) -> None:
    staged_run.run_stage1()
    staged_run.run_stage2()
    monkeypatch.setattr(
        "phenotypic._cli._cli_output_manager.OutputManager.save_image_store",
        lambda *a, **k: None,
    )
    with pytest.raises(RuntimeError, match="Stage 3"):
        staged_run.run_stage3()
```

Add four fixtures to `tests/integration/cli/conftest.py`. Each builds a one-image dataset
from `load_synth_yeast_plate()` and a `StagePlan` with a trivial CPU stand-in for the
`GpuDetector` (the stages take the detector as an argument, so no GPU is needed), and
exposes `run_stage1/2/3`, `output_dir`, `store(dataset, stem)`, and `read_measurements()`.

| Fixture | What it adds |
|---|---|
| `staged_run` | The baseline. |
| `staged_run_with_size_filter` | A post-op that provably removes exactly one colony, so `test_stage3_publishes_the_post_refined_objmap` has something to detect. |
| `staged_run_with_work_id` | A non-`None` `work_id`, exercising the guarded tail. |
| `staged_run_with_border_colony` | A detector stand-in whose output has **a real colony touching the frame** plus a background blob, so `drop_frame_background` has a second victim available on a replay. Also exposes `simulate_timeout_after_promote()`, which removes the Stage-3 completion marker while leaving the token and the raw array — reproducing the exact `_cli_staged_workers.py:225`-to-`:251` window. Without a border-touching colony the D1 idempotency test is **vacuous**: `drop_frame_background` returns early at `_objmap_accessor.py:503` when no non-zero label reaches the border, so a second pass would be a harmless no-op and the test would pass even with the defect present. Assert in the fixture that a border colony exists. |

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/integration/cli/test_staged_store_stages.py -v
```

Expected: FAIL — the stages still write HDF.

- [ ] **Step 3: Port the three stage cores**

`stage1_preprocess_core` (line 125):

```python
    saved_store = output_manager.save_image_store(
        image, dataset_name, image_stem, work_id=work_id
    )
    if saved_store is None or not valid_staged_store(saved_store):
        raise RuntimeError(
            f"Stage 1 store publication failed for {dataset_name}/{image_stem}"
        )
```

`stage2_detect_core` (lines 152–166) — load the input layer from the store, then write
every objmap level in place and drop the token:

```python
    store = zarr_store_path(output_dir, dataset_name, image_stem)
    image = image_cls.load_zarr(store)  # read-only use; never re-promoted here
    array = getattr(image, detector.input_layer)[:]
    try:
        sample = detector._preprocess(array)
        batch = detector._collate([sample])
        result = detector._infer_batch(batch)[0]
    except MemoryError:
        raise
    except Exception as exc:
        raise PerImageScientificError(STAGE_GPU_DETECT, exc) from exc
    _check_active(active_check)
    # Order matters: the store write and the raw retention both precede the
    # token, so a crash anywhere before the token leaves no "Stage 2 done"
    # signal and Stage 2 simply recomputes.
    write_objmap_in_place(store, result)
    write_stage2_raw(output_dir, dataset_name, image_stem, result)
    write_stage2_token(
        output_dir,
        dataset_name,
        image_stem,
        work_id=None,
        objmap_shape=(int(result.shape[0]), int(result.shape[1])),
    )
```

`write_objmap_in_place` is **already available from Phase 1 Task 1.6** — it is store surgery,
not CLI logic, and Phase 4 Task 4.3 imports it too, so defining it here would make Phase 4
depend on Phase 3 while the DAG says they are parallel. Recorded as OPEN-QUESTIONS **B10**.
Import it: `from phenotypic.sdk_.ngff_ import write_objmap_in_place`.

`ensure_staged_overlay` (line 184): `dataset_hdf_dir(...)/f"{stem}.h5"` →
`zarr_store_path(output_dir, dataset_name, image_stem)`; `load_hdf5` → `load_zarr`.

`stage3_merge_measure_core` (lines 205–258): replace the `load_sidecar` merge with a
store read, keep everything after it, and port the guarded tail **verbatim in shape**:

```python
    store = zarr_store_path(output_dir, dataset_name, image_stem)
    image = image_cls.load_zarr(store)
    image.name = image_stem

    # Replay from Stage 2's RETAINED RAW output, never from the store's own
    # objmap. Stage 3 re-promotes over that objmap, so using it as input makes
    # a retried Stage 3 re-run _write_object_output on already-refined labels
    # -- and drop_frame_background then deletes a real colony. See D1.
    result = load_stage2_raw(output_dir, dataset_name, image_stem)
    try:
        plan.gpu_detector._write_object_output(image, result)
        plan.post_pipeline.apply(image, inplace=True)
        measurements = plan.post_pipeline.measure(image, apply_post=False)
    except MemoryError:
        raise
    except Exception as exc:
        raise PerImageScientificError(STAGE_MEASURE, exc) from exc

    _check_active(active_check)
    output_manager.save_measurements(measurements, dataset_name, image_stem)
    _check_active(active_check)
    # Re-promote: post-ops mutate the objmap, and this is what publishes the
    # POST-REFINED segmentation. Without it the stored label image disagrees
    # with the parquet and with a single-pass run.
    saved_store = output_manager.save_image_store(
        image, dataset_name, image_stem, work_id=work_id
    )
    if saved_store is None or not valid_staged_store(saved_store):
        raise RuntimeError(
            f"Stage 3 store publication failed for {dataset_name}/{image_stem}"
        )
    ...
    if work_id is None:
        _check_active(active_check)
        write_stage3_completion_marker(
            output_dir, dataset_name, image_name or image_stem, image_stem
        )
        _check_active(active_check)
        # Consume both. The completion marker is already written above, so a
        # crash between these two deletes classifies "complete" either way and
        # the survivor is inert garbage -- but delete the token FIRST, so the
        # only reachable intermediate state is "no token, orphan raw" (Stage 2
        # would recompute and overwrite it) rather than "token present, raw
        # missing" (Stage 3 would replay into a FileNotFoundError).
        delete_stage2_token(output_dir, dataset_name, image_stem)
        delete_stage2_raw(output_dir, dataset_name, image_stem)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/integration/cli/test_staged_store_stages.py -v
```

Expected: all PASS, in particular `test_stage3_publishes_the_post_refined_objmap`.

- [ ] **Step 5: Prove the post-refined test has teeth**

Temporarily delete the `save_image_store` re-promote from `stage3_merge_measure_core` and
re-run:

```bash
uv run pytest tests/integration/cli/test_staged_store_stages.py::test_stage3_publishes_the_post_refined_objmap -v
```

Expected: FAIL. Restore the re-promote and confirm PASS. Record the observed failure
message in the commit body — a test that cannot be shown to fail is not a guard.

- [ ] **Step 5a: Prove the D1 idempotency test has teeth**

Temporarily change Stage 3's input back to the store's own objmap:

```python
    result = image.objmap[:]          # instead of load_stage2_raw(...)
```

and re-run:

```bash
uv run pytest tests/integration/cli/test_staged_store_stages.py::test_stage3_is_idempotent_under_retry -v
```

Expected: FAIL, with the second pass's objmap missing the border-touching colony that the
first pass kept. Restore `load_stage2_raw` and confirm PASS.

If it **passes** with the defect in place, the fixture is wrong — almost certainly no real
colony touches the frame, so `drop_frame_background` returns early at
`_objmap_accessor.py:503` and the second pass is a no-op. Fix the fixture before moving on;
a green test here would otherwise certify exactly the bug it exists to catch.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_workers.py src/phenotypic/sdk_/ngff_.py tests/integration/cli
git commit -m "feat(cli): port the three staged workers to the OME-Zarr store

Stage 1 emits a zeros objmap so valid_staged_store mirrors valid_staged_hdf
exactly. Stage 2 overwrites every objmap level in place -- a stale level-1
under a fresh level 0 is a silently wrong overlay -- retains its RAW output
under .phenotypic/progress/stage2_raw/, and drops the token last.

Stage 3 replays from that raw array, not from the store. The store is what
Stage 3 re-promotes over, so using it as input makes a retried Stage 3
re-run _write_object_output on already-refined labels, and
drop_frame_background then zeroes whichever real colony touches the frame
most -- silently, once per retry. Verified by swapping the input back and
watching test_stage3_is_idempotent_under_retry fail.

Stage 3 still re-promotes the whole store, because post-ops mutate the
objmap and that re-save is what publishes the post-refined segmentation;
verified by deleting it and watching
test_stage3_publishes_the_post_refined_objmap fail. The work_id is None
guard around the marker and the two deletions is preserved verbatim."
```

---

### Task 3.4: Resume classifier and the differential parity test

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`
  (`valid_staged_hdf` line 69, `staged_hdf_matches_work_id` line 99,
  `classify_staged_image` line 167, `migrate_legacy_stage3_markers` line 287,
  `clear_downstream_artifacts_for_stage1` line 314, `reconcile_stage3_publications` line 322)
- Test: `tests/unit/cli/test_staged_resume_parity.py` (create),
  `tests/unit/cli/test_staged_resume.py` (extend)

**Interfaces:**
- Consumes: `valid_staged_store` (1.6), `stage2_token_exists` (3.2), `zarr_store_path` (2.1).
- Produces:
  ```python
  def staged_store_matches_work_id(path: Path, work_id: str) -> bool
  ```
  and a `classify_staged_image` with an unchanged signature and unchanged return values.

**Constraints specific to this task:**
- `classify_staged_image`'s **signature and return values do not change**. Only the
  artifact probes underneath it change: `hdf` → store, `valid_staged_hdf` →
  `valid_staged_store`, `staged_hdf_matches_work_id` → `staged_store_matches_work_id`,
  `sidecar_exists` → `stage2_token_exists`.
- Every branch is preserved **in order**, including the `process_only_layer == "objmap"`
  early return (line 190), both `stage3_completion_exists` branches (lines 205 and 211),
  and the `not markers_required and parquet and not sidecar → "complete"` branch (line
  221). The last one is the branch a durable labels list would have broken.
- `migrate_legacy_stage3_markers` keeps working. It depends on the token's **absence**
  marking completion; if it stops firing, resume state is wrong for every legacy tree.
- This task's differential test is the phase gate. Write it **first**, run it against the
  unmodified HDF classifier to confirm it passes, and only then port.

- [ ] **Step 1: Write the differential parity test**

Create `tests/unit/cli/test_staged_resume_parity.py`:

```python
"""Differential resume parity: the zarr classifier must agree with the HDF one.

This is the test that would have caught all three resume defects the spec's
independent review found. It enumerates every combination
classify_staged_image currently distinguishes and asserts the two artifact
worlds produce the same stage, rather than asserting a hand-written table that
could itself encode the bug.

tests/unit/cli/test_staged_resume.py already parameterizes markers_required at
:57, :86, :108, :128, :146, :165; this mirrors that shape across all four axes.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from phenotypic._cli._cli_staged_resume import classify_staged_image

PROCESS_ONLY_LAYERS = [None, "objmap", "gray"]
MARKERS_REQUIRED = [True, False]
WORK_IDS = [None, "w-1"]
#: Which durable artifacts exist, as
#: (image_state, stage2_signal, parquet, stage3_marker, image_success_marker).
#:
#: The FIFTH axis is load-bearing. classify_staged_image's first branch
#: (_cli_staged_resume.py:182) consults valid_image_success, which reads the
#: per-image completion marker. Without this axis that branch is never
#: exercised -- valid_image_success returns False in both worlds -- and the
#: parity test passes while production breaks. See Task 3.8 / OPEN-QUESTIONS D2.
ARTIFACTS = list(itertools.product([False, True], repeat=5))

CASES = [
    pytest.param(layer, markers, work_id, artifacts, id=f"{layer}-{markers}-{work_id}-{artifacts}")
    for layer, markers, work_id, artifacts in itertools.product(
        PROCESS_ONLY_LAYERS, MARKERS_REQUIRED, WORK_IDS, ARTIFACTS
    )
]


@pytest.mark.parametrize(("layer", "markers", "work_id", "artifacts"), CASES)
def test_zarr_classifier_matches_the_hdf_classifier(
    layer, markers, work_id, artifacts, hdf_world, zarr_world
):
    """hdf_world / zarr_world build the same artifact set in the two formats."""
    hdf_root = hdf_world(artifacts, work_id=work_id)
    zarr_root = zarr_world(artifacts, work_id=work_id)
    common = dict(
        dataset="ds",
        image=Path("img.tif"),
        input_root=Path("/in"),
        process_only_layer=layer,
        markers_required=markers,
        expected_work_id=work_id,
    )
    assert classify_staged_image(output_dir=zarr_root, **common) == (
        hdf_world.classify(output_dir=hdf_root, **common)
    )
```

`hdf_world` and `zarr_world` are fixtures in `tests/unit/cli/conftest.py`. `hdf_world`
pins the **pre-port** HDF classifier: copy `classify_staged_image` and its four probe
functions into `tests/_legacy_staged_resume.py` **before** touching the source, and have
`hdf_world.classify` call that frozen copy. Freezing it is the point — a differential test
against a classifier that moves with the code proves nothing.

- [ ] **Step 2: Run it against the unmodified source to confirm it passes**

```bash
uv run pytest tests/unit/cli/test_staged_resume_parity.py -q
```

Expected: PASS (both sides are still effectively the HDF classifier via `zarr_world`
building HDF artifacts). If it fails here, the fixtures are wrong — fix them before
porting. **Do not proceed until this is green.**

- [ ] **Step 3: Port the classifier**

Replace `valid_staged_hdf` with a re-export and port the work-id probe:

```python
from phenotypic.sdk_.ngff_ import valid_staged_store  # noqa: F401 -- public re-export


def staged_store_matches_work_id(path: Path, work_id: str) -> bool:
    """Return whether a valid staged store is bound to ``work_id``.

    Replaces ``staged_hdf_matches_work_id``. The work id lives in
    ``attributes.phenotypic.work_id``, written at store-build time.
    """
    if not valid_staged_store(path):
        return False
    try:
        from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

        block = read_phenotypic_attributes(path)
        return block.get(PhenotypicAttr.WORK_ID) == work_id
    except (OSError, KeyError, ValueError, TypeError):
        return False
```

In `classify_staged_image`, change only the probes (lines 196–221):

```python
    store = zarr_store_path(output_dir, dataset, stem)
    if expected_work_id is not None:
        store_valid = staged_store_matches_work_id(store, expected_work_id)
    else:
        store_valid = valid_staged_store(store)
    if not store_valid:
        return "stage1"
    ...
    stage2_done = stage2_token_exists(output_dir, dataset, stem)
    ...
    if (
        process_only_layer is None
        and not markers_required
        and parquet.is_file()
        and not stage2_done
    ):
        return "complete"

    return "stage3" if stage2_done else "stage2"
```

Update `migrate_legacy_stage3_markers`, `clear_downstream_artifacts_for_stage1`, and
`reconcile_stage3_publications` to use the store path and the token.

> **`clear_downstream_artifacts_for_stage1` deletes nothing extra.** An earlier draft of
> this task said it must `rmtree` the store because "an `unlink` there raises
> `IsADirectoryError`". That rests on a misreading — **verified**: the function
> (`_cli_staged_resume.py:314-319`) deletes only the `.npy` sidecar and the `.json` Stage-3
> marker. It never unlinks an image artifact, so no `IsADirectoryError` is possible, and
> both deletions become plain `.json` unlinks under the new design.
>
> Adding an `rmtree(store)` would **introduce** behaviour that does not exist today: at its
> two call sites (`_cli_staged_strategy.py:145`, `_cli_staged_slurm_worker.py:141`, both
> immediately before Stage 1) it opens a window where the image is absent, whereas today the
> previous HDF survives until Stage 1's atomic replace — and it removes the only fallback if
> Stage 1 then fails. Stage 1's promote already replaces the store atomically. Recorded as
> OPEN-QUESTIONS **D13**.

- [ ] **Step 4: Point `zarr_world` at the real artifacts and re-run**

Switch the `zarr_world` fixture to build stores + tokens, and re-run:

```bash
uv run pytest tests/unit/cli/test_staged_resume_parity.py tests/unit/cli/test_staged_resume.py -q
```

Expected: all PASS. A failure names the exact `(layer, markers, work_id, artifacts)`
combination that diverged.

- [ ] **Step 5: Prove the parity test catches the three known defects**

Apply each defect in turn, confirm the parity test fails, then revert:

1. Make `valid_staged_store` require `objmap` to be non-zeros → Stage 1 stores classify
   `"stage1"` forever.
2. Replace `stage2_token_exists` with a durable `ome.labels` probe → the
   `not markers_required` `"complete"` branch never fires.
3. Delete Stage 3's re-promote (Task 3.3) → parity holds but
   `test_stage3_publishes_the_post_refined_objmap` fails; note in the commit that the
   third defect is caught by that test, not this one.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_resume.py tests/unit/cli tests/_legacy_staged_resume.py
git commit -m "feat(cli): port the staged resume classifier to the store

Signature and return values are unchanged; only the artifact probes move.
A differential test enumerates every (process_only_layer, markers_required,
expected_work_id, artifacts) combination the classifier distinguishes and
asserts the zarr world agrees with a frozen copy of the HDF classifier --
freezing it is what makes the comparison mean anything. Verified to fail
under both resume defects the spec's review found. The artifact axis
includes the per-image completion marker, without which branch 1 is never
exercised and the test passes while production breaks (see Task 3.8)."
```

---

### Task 3.5: Staged strategy, controller, SLURM worker, orchestration; delete the sidecar

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py` (lines 22, 33, 37–38, 89, 93, 140,
  142, 171–182, 219–246, 328–336)
- Modify: `src/phenotypic/_cli/_cli_staged_slurm_worker.py` (lines 19, 28, 45–46, 128–134,
  196, 223–227, 276, 316–317, 332–333, 354–360, 382–383, 409)
- Modify: `src/phenotypic/_cli/_cli_staged_controller.py`, `_cli_staged_orchestration.py`
- Delete: `src/phenotypic/_cli/_cli_sidecar.py`, `tests/unit/cli/test_cli_sidecar.py`
- Test: `tests/unit/cli/test_staged_routing.py` (extend),
  `tests/unit/cli/test_staged_controller.py` (extend)

**Constraints specific to this task:**
- `_cli_staged_strategy.py:328` is `--mode process --layer objmap`: it merges the Stage-2
  result and exports, then deletes the token. With the objmap now in the store, the merge
  is a store read — but the **token deletion must stay**, or a subsequent full run
  misclassifies.
- ⚠️ **…and that path must not leave raw detector output published forever.**
  `_cli_staged_strategy.py:360-382` applies `_write_object_output`, writes the exported
  layer, deletes the signal, and **never re-saves the image**. Today the residue is Stage 1's
  zeros inside a non-user-facing `.h5`. Under this design the residue is Stage 2's **raw**
  objmap — pre-`drop_frame_background`, pre-relabel, possibly one giant background label
  covering the plate — sitting in a first-class NGFF label image that napari and Vizarr will
  render. Either re-promote the store after the export or restore the zeros objmap.
  Recorded as OPEN-QUESTIONS **D11**.
- ⚠️ **The run-start sweep must not delete a live writer's `.part`.** The uuid identifies the
  *attempt*, not whether its process is alive, and the staged SLURM engine explicitly
  assumes stale workers can still be running — that is what `assert_active_epoch`
  (`_cli_staged_slurm_worker.py:346-348`, `_cli_staged_orchestration.py:679`) exists for. A
  recovery controller sweeping while a prior-epoch task is mid-write into its `.part` would
  `rmtree` under it. Gate the sweep on age (mtime older than this run's epoch start) or on a
  lifecycle epoch recorded inside the `.part`. Recorded as OPEN-QUESTIONS **D14**.
- **Delete `clear_stage2_sidecars`** (`_cli_staged_orchestration.py:661-674`, called from
  `phenotypicCLI.py:1590` on `--restart`). It globs `results/*/objmap/*.npy` and becomes a
  permanent no-op. Not a correctness hole — `clear_machine_state` on the same path wipes
  `.phenotypic/`, where the new token lives — but leaving a no-op named after a deleted
  concept is how the next reader concludes sidecars still exist.
- `_cli_staged_slurm_worker.py:409` deletes the token on the work-id path. That is the
  counterpart to Task 3.3's preserved `work_id is None` guard; both must remain.
- **Every site that deleted the sidecar must now delete BOTH the token and the raw array.**
  There are five, all verified present: `_cli_staged_workers.py:258` (guarded),
  `_cli_staged_strategy.py:246` (local Stage 3, unconditional) and `:382`
  (`_export_objmap_layer`), `_cli_staged_slurm_worker.py:409`, and
  `_cli_staged_resume.py:364` (`reconcile_stage3_publications`). Deleting the token and
  leaving the raw array orphans a `.npy` per image; deleting the raw array and leaving the
  token makes the next Stage 3 replay into a `FileNotFoundError`. Delete the **token
  first** at every site, for the reason given in Task 3.3.
- The run start must **log the resolved durability mode** (`ngff_.describe_durability`) —
  a required mitigation, not a nicety.
- The run start must **sweep orphaned `.part` / `.trash` directories** and log the count.
- ⚠️ **Both belong on every execution path, not only the staged-GPU one.** Spec §3.7 and
  §3.2 are unqualified, and a plain `--mode full` CPU run writes stores through the same
  promote. Wiring them into `_cli_staged_strategy`'s setup alone leaves the common case with
  no durability log and no sweep. Put both in the shared run-setup that
  `_cli_execution_strategies.create_execution_strategy` dispatches through, so every strategy
  inherits them. Recorded as OPEN-QUESTIONS **G6/P21**.
- ⚠️ **The sweep runs from the controller, once, before any worker is submitted — never
  from a worker's own start-up.** A uuid identifies the attempt, not whether its process is
  alive; under a SLURM array the tasks share one output root and start at different times, so
  a per-worker sweep would `rmtree` the `.part` directories its siblings are actively
  filling. `ngff_.sweep_orphan_parts` additionally refuses anything younger than
  `SWEEP_MIN_AGE_SECONDS`, but that age guard is a backstop, not a licence to call it from a
  worker. Recorded as OPEN-QUESTIONS **B6/P16**.
- Delete `_cli_sidecar.py` only after `grep -rn "_cli_sidecar\|sidecar_exists\|write_sidecar\|load_sidecar\|delete_sidecar" src/ tests/` is empty.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/cli/test_staged_routing.py`:

```python
def test_run_start_logs_the_resolved_durability_mode(staged_strategy, caplog) -> None:
    """The same command carries different guarantees in different places."""
    staged_strategy.prepare()
    assert any("durable writes:" in record.message for record in caplog.records)


def test_controller_sweeps_stale_orphaned_part_directories(staged_strategy, tmp_path) -> None:
    import os
    import time

    from phenotypic.sdk_ import dataset_zarr_dir

    orphan = dataset_zarr_dir(tmp_path, "ds") / ".img.ome.zarr.deadbeef.part"
    orphan.mkdir(parents=True)
    old = time.time() - 24 * 60 * 60
    os.utime(orphan, (old, old))
    staged_strategy.prepare()
    assert not orphan.exists()


def test_the_sweep_spares_a_recent_part(staged_strategy, tmp_path) -> None:
    """A uuid says nothing about liveness; under a SLURM array a sibling task
    may be mid-write into exactly this directory."""
    from phenotypic.sdk_ import dataset_zarr_dir

    live = dataset_zarr_dir(tmp_path, "ds") / ".img.ome.zarr.cafef00d.part"
    live.mkdir(parents=True)
    staged_strategy.prepare()
    assert live.is_dir()


def test_workers_never_sweep(staged_run, tmp_path) -> None:
    """Only the controller sweeps, and only before submitting anything."""
    from phenotypic.sdk_ import dataset_zarr_dir

    live = dataset_zarr_dir(tmp_path, "ds") / ".other.ome.zarr.deadbeef.part"
    live.mkdir(parents=True)
    staged_run.run_stage1()
    assert live.is_dir()


def test_a_plain_full_run_also_logs_durability_and_sweeps(cli_runner, tiny_run, caplog) -> None:
    """Spec §3.7 and §3.2 are unqualified; the CPU path uses the same promote."""
    from phenotypic.phenotypicCLI import main

    cli_runner.invoke(main, tiny_run.args())  # --mode full, no GpuDetector
    assert any("durable writes:" in record.message for record in caplog.records)


def test_sidecar_module_is_gone() -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phenotypic._cli._cli_sidecar")


def test_process_only_objmap_still_consumes_the_token(staged_run) -> None:
    from phenotypic._cli._cli_stage2_token import stage2_token_exists

    staged_run.run_stage1()
    staged_run.run_stage2()
    staged_run.export_objmap_layer()
    assert stage2_token_exists(staged_run.output_dir, "ds", "img") is False
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/cli/test_staged_routing.py -v
```

- [ ] **Step 3: Port the four modules and delete the sidecar**

Mechanical, one import block and one path expression at a time. After each file:

```bash
uv run pytest tests/unit/cli -q
```

Then:

```bash
grep -rn "_cli_sidecar\|sidecar_exists\|write_sidecar\|load_sidecar\|delete_sidecar" src/ tests/
git rm src/phenotypic/_cli/_cli_sidecar.py tests/unit/cli/test_cli_sidecar.py
```

- [ ] **Step 4: Run the CLI suite**

```bash
uv run pytest tests/unit/cli tests/integration/cli -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli tests/integration/cli
git commit -m "refactor(cli): route the staged engine through the store; drop the sidecar

The run start now logs the resolved durability mode and sweeps orphaned
.part/.trash directories by uuid. --mode process --layer objmap still
consumes the Stage-2 token after export, and the SLURM worker still deletes
it on the work-id path; both are load-bearing for resume classification.
_cli_sidecar.py is deleted only after every caller moved."
```

---

### Task 3.6: Directory scanning, recompile scripts, single-pass, tune

**Files:**
- Modify: `src/phenotypic/_cli/_cli_directory_scanner.py` (`scan_hdf_outputs` line 173,
  glob at line 217)
- Modify: `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py` (lines 217–231)
- Modify: `src/phenotypic/_cli/_cli_process_single.py`, `_cli_execution_strategies.py`
- Modify: `src/phenotypic/tune/_tune_cli/_run.py`
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py` (lines 113, 131)
- Test: `tests/unit/cli/test_directory_scanner.py` (extend)

**Constraints specific to this task:**
- `scan_hdf_outputs` becomes `scan_store_outputs`; the glob `hdf_dir.glob("*.h5")` becomes
  `zarr_dir.glob(f"*{STORE_SUFFIX}")` and must be **non-recursive** and match
  **directories**. A store contains files, so a recursive glob or an `is_file()` filter
  finds nothing.
- The AppleDouble guard at line 218 (`not p.name.startswith(".")`) must be **kept**: it now
  also excludes the `.part` and `.trash` directories, which is exactly right.
- `_cli_readme_generator.py` documents the layout to users: `hdf/` → `zarr/`, `.h5` →
  `.ome.zarr`, `Image.load_hdf5` → `Image.load_zarr`, and add a line saying the store is
  readable by napari, QuPath, and Vizarr without a PhenoTypic install — that is a headline
  user-facing benefit of this change.

- [ ] **Step 1: Write the failing test**

```python
def test_scan_finds_store_directories_not_files(tmp_path) -> None:
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs
    from phenotypic.sdk_ import zarr_store_path

    for stem in ("a", "b"):
        store = zarr_store_path(tmp_path, "ds", stem)
        store.mkdir(parents=True)
        (store / "zarr.json").write_text("{}", encoding="utf-8")
    datasets = scan_store_outputs(tmp_path)
    assert [p.name for p in datasets[0].images] == ["a.ome.zarr", "b.ome.zarr"]


def test_scan_is_non_recursive(tmp_path) -> None:
    """A recursive scan walks INTO every store: 400k stat calls at 10k images."""
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs
    from phenotypic.sdk_ import zarr_store_path

    store = zarr_store_path(tmp_path, "ds", "a")
    (store / "gray" / "0").mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    (store / "gray" / "0" / "nested.ome.zarr").mkdir()
    assert len(scan_store_outputs(tmp_path)[0].images) == 1


def test_scan_skips_part_and_trash_directories(tmp_path) -> None:
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs
    from phenotypic.sdk_ import dataset_zarr_dir, zarr_store_path

    store = zarr_store_path(tmp_path, "ds", "a")
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    (dataset_zarr_dir(tmp_path, "ds") / ".a.ome.zarr.deadbeef.part").mkdir()
    assert len(scan_store_outputs(tmp_path)[0].images) == 1
```

- [ ] **Step 2: Run to verify failure, then port each file**

```bash
uv run pytest tests/unit/cli/test_directory_scanner.py -v
```

- [ ] **Step 3: Run the full CLI + tune suites**

```bash
uv run pytest tests/unit/cli tests/unit/tune tests/integration/cli -q
```

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/_cli src/phenotypic/tune tests/unit/cli
git commit -m "refactor(cli): scan for store directories, not .h5 files

The glob is non-recursive and matches directories: a store contains files,
so a recursive scan walks into all ~40 of them (400k stat calls at 10k
images) and an is_file() filter finds nothing at all. The AppleDouble
dotfile guard is kept and now also excludes .part/.trash. The README
generator documents the new layout and that the output is readable by
napari, QuPath, and Vizarr without a PhenoTypic install."
```

---

### Task 3.7: The `--durable-writes` CLI option

**Files:**
- Modify: `src/phenotypic/phenotypicCLI.py` (option block beside `--mode` at line 942; the
  module docstring's option documentation)
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py`,
  `src/phenotypic/_cli/_cli_process_single.py` (thread the value to `save_image_store`)
- Test: `tests/unit/cli/test_cli_store_options.py` (create)

**Interfaces:**
- Consumes: `ngff_.durable_writes_enabled`, `ngff_.describe_durability`,
  `OutputManager.save_image_store`.
- Produces: one new top-level CLI option and its config plumbing.

**Why this task exists:** spec §3.7 requires `--durable-writes` / `--no-durable-writes`, but
the flag appears in no section that enumerates CLI options, so it had no owning task and
would have shipped unimplemented. Recorded as OPEN-QUESTIONS **P12**.

**`--pyramid-levels` is descoped.** Spec §1.3 also introduces `--pyramid-levels auto|N`;
that lever is **not** implemented. The pyramid depth is a pure function of the level-0 shape
(`ngff_.pyramid_level_count`), which dissolves OPEN-QUESTIONS **P3** — with no user lever,
two stores in one tree cannot disagree, so `valid_staged_store` needs no level check and a
resumed run cannot produce mixed geometry. A single-level store is still reachable
internally, via the private `levels=` argument used by `save_intermediate_zarr`. The lever
can be added later as its own change; the spec's §1.3 should record it as deferred.

**Constraints specific to this task:**
- `--durable-writes` / `--no-durable-writes` is a **tri-state**: unset means auto-detect. A
  plain `click.option(..., is_flag=True)` collapses that to two states and silently loses
  the SLURM detection. Use a paired `--durable-writes/--no-durable-writes` option with
  `default=None`.
- The resolved durability mode is logged at run start (already required by Task 3.5); this
  task is what gives that log line something other than the auto-detection to report.
- The option applies to `--mode full`, `--mode process`, and `--mode measure`, and is
  **rejected** on `--mode recompile` and `--mode migrate`, which do not write image stores
  from a pipeline. Reuse the existing per-mode rejection pattern at lines 1231–1244.

- [ ] **Step 1: Write the failing test**

```python
"""The durability flag must be genuinely tri-state."""

from __future__ import annotations

import pytest

from phenotypic.phenotypicCLI import main


def test_durable_writes_is_tri_state(cli_runner, tiny_run, monkeypatch, caplog) -> None:
    """Unset must mean auto-detect, not 'off'. A plain is_flag loses that."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    cli_runner.invoke(main, tiny_run.args())
    assert any("durable writes: off (local)" in r.message for r in caplog.records)

    caplog.clear()
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    cli_runner.invoke(main, tiny_run.args())
    assert any("durable writes: on (SLURM)" in r.message for r in caplog.records)

    caplog.clear()
    cli_runner.invoke(main, [*tiny_run.args(), "--no-durable-writes"])
    assert any(
        "durable writes: off (--no-durable-writes)" in r.message for r in caplog.records
    )

    caplog.clear()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    cli_runner.invoke(main, [*tiny_run.args(), "--durable-writes"])
    assert any(
        "durable writes: on (--durable-writes)" in r.message for r in caplog.records
    )


def test_durable_writes_is_rejected_on_recompile_and_migrate(cli_runner, tiny_run) -> None:
    for mode in ("recompile", "migrate"):
        result = cli_runner.invoke(
            main,
            ["--mode", mode, "--output", str(tiny_run.output_dir), "--durable-writes"],
        )
        assert result.exit_code != 0
        assert "--durable-writes" in result.output


def test_no_pyramid_levels_option_exists(cli_runner) -> None:
    """Descoped: the pyramid depth is a pure function of shape (P3)."""
    result = cli_runner.invoke(main, ["--help"])
    assert "--pyramid-levels" not in result.output


def test_pyramid_depth_is_derived_not_configured(cli_runner, tiny_run) -> None:
    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

    cli_runner.invoke(main, tiny_run.args())
    store = tiny_run.store("ds", "img")
    shape = tiny_run.image_shape
    assert read_phenotypic_attributes(store)[PhenotypicAttr.PYRAMID]["levels"] == (
        ngff_.pyramid_level_count(*shape)
    )
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/cli/test_cli_store_options.py -v
```

Expected: `Error: No such option: --durable-writes`.

- [ ] **Step 3: Add the option**

```python
@click.option(
    "--durable-writes/--no-durable-writes",
    "durable_writes",
    default=None,
    help=(
        "fsync each image store before promoting it. Unset auto-detects: on "
        "under SLURM, off locally. The resolved mode is logged at run start."
    ),
)
```

Reject it on the two modes that do not write stores from a pipeline, mirroring the existing
guards:

```python
    if durable_writes is not None and cli_mode in {"recompile", "migrate"}:
        raise click.UsageError(
            f"--durable-writes is not accepted with --mode {cli_mode}; that mode "
            "does not write image stores from a pipeline."
        )
```

Thread the value through the run config to every `save_image_store` call in
`_cli_staged_strategy.py` and `_cli_process_single.py`.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/cli/test_cli_store_options.py tests/unit/cli -v
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/phenotypicCLI.py src/phenotypic/_cli tests/unit/cli/test_cli_store_options.py
git commit -m "feat(cli): add --durable-writes

A spec §3.7 requirement that no section enumerated as a CLI flag, so it had
no owning task. It is tri-state: unset means auto-detect, and a plain
is_flag would collapse that to 'off' and silently lose the SLURM detection.

--pyramid-levels (spec §1.3) is deliberately NOT added. The pyramid depth
is a pure function of the level-0 shape, which means two stores in one tree
can never disagree -- so valid_staged_store needs no level check and a
resumed run cannot produce mixed geometry. The lever can land later as its
own change."
```

---

### Task 3.8: Per-image completion markers must describe a store, not a file

**Files:**
- Modify: `src/phenotypic/_cli/_cli_completion.py` (`SUCCESS_MARKER_VERSION` line 26,
  `_sha256` lines 29–34, `publish_image_success` line 36, `valid_image_success` lines 117–130,
  `refresh_success_markers_after_metadata_migration` lines 136–155)
- Modify the five `"hdf"` artifact declarations: `phenotypicCLI.py:400`,
  `_cli_staged_slurm_worker.py:332` and `:382`, `_cli_process_single.py:640`,
  `_cli_execution_strategies.py:167`
- Test: `tests/unit/cli/test_cli_completion_store.py` (create)

**Why this task exists — this is a silent production break, not a refactor.**

`grep -rn 'publish_image_success|valid_image_success|_cli_completion|SUCCESS_MARKER_VERSION'`
over the spec and the entire plan directory returns **nothing**. The surface was uncosted
until an independent data-flow review found it. Both halves fail on a directory:

```python
def _sha256(path: Path) -> str:          # _cli_completion.py:29
    with path.open("rb") as handle:      # IsADirectoryError on a store -- UNCAUGHT
```

```python
if (not artifact.is_file()               # _cli_completion.py:126 -- False for a store
        or artifact.stat().st_size != descriptor.get("size")
        or _sha256(artifact) != descriptor.get("sha256")):
    return False
```

So `publish_image_success` **kills the publishing worker**, and `valid_image_success` makes
`classify_staged_image`'s first branch (`_cli_staged_resume.py:182`) return `"stage3"` for
every already-finished image on the work-id path, forever. Recorded as OPEN-QUESTIONS **D2**.

**Interfaces:**
- Produces: a `kind`-tagged artifact descriptor, and `SUCCESS_MARKER_VERSION = 2`.

**Constraints specific to this task:**
- A store descriptor is
  `{"path": <relative>, "kind": "store", "fingerprint": paths_fingerprint([store / "zarr.json"])}`.
  Key on the **root `zarr.json`**, not the directory: `paths_fingerprint` on a directory
  emits one sentinel byte and does not recurse (`_io_constants.py:215-217`), so a directory
  fingerprint is a constant function of the path and would validate a store whose contents
  changed. This is the same trap as OPEN-QUESTIONS **D4/D5**.
- File descriptors keep their existing `{"size", "sha256"}` shape and gain
  `"kind": "file"`. `valid_image_success` dispatches on `kind`, defaulting to `"file"` when
  absent so a marker written by an older version still parses.
- **`SUCCESS_MARKER_VERSION` must be bumped to `2`.** A v1 marker describes an `.h5` that no
  longer exists; without the bump those markers are read and fail validation, silently
  reprocessing every image in every legacy tree.
- `refresh_success_markers_after_metadata_migration` (`:136-155`) exists because rewriting a
  per-image HDF invalidates the marker's `sha256`. Header-only **store** migration
  (Phase 5 Task 5.5) rewrites `zarr.json` and does exactly the same thing, so this bridge
  must handle store descriptors too. Recorded as OPEN-QUESTIONS **D10**.

- [ ] **Step 1: Write the failing test**

```python
"""Per-image completion markers over a store directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic._cli._cli_completion import (
    SUCCESS_MARKER_VERSION,
    publish_image_success,
    valid_image_success,
)
from phenotypic.sdk_ import zarr_store_path


def test_marker_version_is_bumped() -> None:
    """A v1 marker describes an .h5 that no longer exists."""
    assert SUCCESS_MARKER_VERSION >= 2


def test_publishing_a_store_artifact_does_not_raise(published_store) -> None:
    """_sha256 opens its argument as a file; on a directory that is fatal."""
    assert published_store.marker.is_file()


def test_a_published_store_validates(published_store) -> None:
    assert valid_image_success(
        published_store.output_dir,
        dataset="ds",
        image_stem="img",
        work_id="w-1",
    ) is True


def test_a_rewritten_store_invalidates_the_marker(published_store) -> None:
    """Keying on the directory instead of zarr.json would miss this."""
    root = published_store.store / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"]["work_id"] = "different"
    root.write_text(json.dumps(payload), encoding="utf-8")
    assert valid_image_success(
        published_store.output_dir, dataset="ds", image_stem="img", work_id="w-1"
    ) is False


def test_a_deleted_store_invalidates_the_marker(published_store) -> None:
    import shutil

    shutil.rmtree(published_store.store)
    assert valid_image_success(
        published_store.output_dir, dataset="ds", image_stem="img", work_id="w-1"
    ) is False


def test_a_file_descriptor_without_kind_still_validates(legacy_file_marker) -> None:
    """Defaulting kind to 'file' keeps older markers parseable."""
    assert valid_image_success(**legacy_file_marker) is True


def test_every_hdf_artifact_declaration_is_ported() -> None:
    """The five sites that declare the per-image image-state artifact."""
    import re
    from pathlib import Path as _Path

    src = _Path(__file__).resolve().parents[3] / "src" / "phenotypic"
    hits = [
        f"{p.relative_to(src)}:{n}"
        for p in src.rglob("*.py")
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r'"hdf"\s*:', line)
    ]
    assert hits == [], hits
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/cli/test_cli_completion_store.py -v
```

Expected: `test_publishing_a_store_artifact_does_not_raise` fails with `IsADirectoryError`,
and `test_marker_version_is_bumped` fails at `1 >= 2`.

- [ ] **Step 3: Implement the `kind` dispatch, bump the version, port the five sites.**

- [ ] **Step 4: Re-run the differential parity test**

```bash
uv run pytest tests/unit/cli/test_cli_completion_store.py tests/unit/cli/test_staged_resume_parity.py -q
```

Expected: green — and the parity test now actually exercises branch 1, because Task 3.4's
fifth artifact axis makes `valid_image_success` return `True` in some combinations.

- [ ] **Step 5: Prove the fifth axis matters**

Temporarily revert `ARTIFACTS` to `repeat=4` and re-introduce the `is_file()` check. The
parity test should PASS despite the broken classifier — demonstrating the blind spot. Restore
both and confirm the parity test now FAILS under the same defect.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli tests/unit/cli/test_cli_completion_store.py
git commit -m "fix(cli): describe a store, not a file, in per-image completion markers

_sha256 opened its argument as a file (IsADirectoryError kills the
publishing worker) and valid_image_success required is_file() (so every
finished image reclassified stage3 forever on the work-id path). Artifact
descriptors gain a kind tag; store descriptors fingerprint the root
zarr.json rather than the directory, because paths_fingerprint emits one
sentinel byte for a directory and does not recurse. SUCCESS_MARKER_VERSION
is bumped so v1 markers describing a vanished .h5 are not read and failed.
This surface appeared in neither the spec nor the plan until a data-flow
review found it, and the resume parity test could not see it -- hence the
fifth artifact axis in Task 3.4."
```

---

## Phase 3 exit criteria

- [ ] `uv run pytest tests/unit/cli tests/integration/cli -q` is green.
- [ ] `uv run pytest tests/unit/cli/test_staged_resume_parity.py -q` is green, and has been
      demonstrated to fail under both injected resume defects.
- [ ] `test_stage3_publishes_the_post_refined_objmap` has been demonstrated to fail when
      Stage 3's re-promote is removed.
- [ ] `grep -rn "sidecar" src/phenotypic/_cli/` returns nothing.
- [ ] `grep -rn "dataset_hdf_dir\|\.h5" src/phenotypic/_cli/` returns only migration-path
      references (Phase 5) — nothing in the forward run path.
- [ ] A run start log line contains `durable writes:`.
# Phase 4 — GUI read paths: pyramid tiles and the staleness traps

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §4.2, §4.4.

**Depends on:** Phase 2.
**Runs in parallel with:** Phases 3 and 5.

This is where the pyramid pays for itself: today the GUI decodes an **entire** layer to
render one whole-plate tile. It is also where the change is most likely to fail silently —
a store directory's `st_mtime_ns` does **not** change when a nested chunk is rewritten, so
every mtime-based staleness check becomes a stale-tile bug rather than an error.

---

### Task 4.1: Store discovery in `OutputRoot`

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py`
  (line 53 import, `hdf_path` line 494, `has_*` line 630, `rglob("*.h5")` line 888,
  the consistency-report path build at lines 1146–1152)
- Test: `tests/unit/gui/results_viewer/test_output_root_stores.py` (create)

**Interfaces:**
- Consumes: `BundleLayout.store_path` (Task 2.1).
- Produces: `OutputRoot.store_path(dataset, stem) -> Path | None`, replacing `hdf_path`.

**Constraints specific to this task:**

- `store_path` uses `is_dir()`, not `is_file()`.

- **Line 886–889 is a correctness problem, not just a cost problem.**
  `_processing_snapshot_paths` does
  `paths.extend(path for path in layout.results_dir.rglob("*.h5") if path.is_file())`, and
  those paths feed `_cancellable_paths_fingerprint`, whose directory branch (`:832-834`)
  emits **one sentinel byte and does not recurse**. If the port yields store *directories*,
  `snapshot.processing_fingerprint` — and therefore `OutputRoot.source_fingerprint`
  (`:512`) — **stops changing when per-image results change**, so a viewer open across a run
  never notices new or rewritten images. The port must enumerate each store's
  **`zarr.json`**, which also happens to solve the cost problem (a naive
  `rglob("*.ome.zarr")` recurses into every store — ~400k stat calls at 10k images).
  OPEN-QUESTIONS **D5**.

- **Line 1146–1152 is a staleness fingerprint, not a report label.** An earlier draft of
  this task described it as "an `("hdf", hdf_path)` pair for the output-consistency report…
  update whatever label the report renders". That is wrong. `_image_source_token`
  (`:1138-1178`) hashes, per source path:

  ```python
  f"{stat.st_dev}\0{stat.st_ino}\0{stat.st_size}\0{stat.st_mtime_ns}\0{stat.st_ctime_ns}\n"
  ```

  **None of those five fields moves** when a chunk inside a store is rewritten. The token
  drives `bound_image_source_token` (`:649`) and `_capture_image_source_tokens` (`:405`,
  `:1093`) — i.e. whether the viewer's binding to an image's pixel source is still valid.
  Ported as a relabel it goes silently blind. It must key on **`store / "zarr.json"`**.
  OPEN-QUESTIONS **D4**.

- **The rule underneath D4 and D5:** anywhere the old code took a fingerprint or a stat of
  the `.h5`, the store equivalent is the **root `zarr.json`** — never the store directory.
  `paths_fingerprint` "handles directories" only in the sense that it does not raise; it
  ignores their contents. The spec's §4.2 table is misleading on exactly this point and
  should be corrected.

- [ ] **Step 1: Write the failing test**

```python
"""OutputRoot discovers store directories without walking into them."""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.sdk_ import zarr_store_path


def _seed(root: Path, stems: list[str]) -> None:
    (root / "deliverables").mkdir(parents=True, exist_ok=True)
    (root / "deliverables" / "master_measurements.parquet").write_bytes(b"")
    for stem in stems:
        store = zarr_store_path(root, "ds", stem)
        (store / "gray" / "0").mkdir(parents=True)
        (store / "zarr.json").write_text("{}", encoding="utf-8")


def test_store_path_resolves_a_directory(tmp_path: Path) -> None:
    _seed(tmp_path, ["a"])
    root = OutputRoot.discover(tmp_path)
    assert root.store_path("ds", "a") == zarr_store_path(tmp_path, "ds", "a")


def test_store_path_is_none_when_absent(tmp_path: Path) -> None:
    _seed(tmp_path, ["a"])
    assert OutputRoot.discover(tmp_path).store_path("ds", "missing") is None


def test_discovery_does_not_walk_into_stores(tmp_path: Path, monkeypatch) -> None:
    """A recursive scan costs 400k stat calls at 10k images."""
    _seed(tmp_path, ["a", "b"])
    visited: list[str] = []
    real_iterdir = Path.iterdir

    def _counting(self):
        visited.append(str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _counting)
    OutputRoot.discover(tmp_path)
    assert not any("/gray/" in seen for seen in visited), visited


def test_discovery_finds_every_store(tmp_path: Path) -> None:
    _seed(tmp_path, ["a", "b", "c"])
    root = OutputRoot.discover(tmp_path)
    assert all(root.store_path("ds", stem) is not None for stem in "abc")
```

- [ ] **Step 2: Run it to verify it fails.** Expected: `AttributeError: … 'store_path'`.

Add two tests for the fingerprint sites:

```python
def test_processing_fingerprint_changes_when_a_store_changes(tmp_path: Path) -> None:
    """Enumerating directories would freeze this permanently (D5)."""
    _seed(tmp_path, ["a"])
    root = OutputRoot.discover(tmp_path)
    before = root.source_fingerprint
    (zarr_store_path(tmp_path, "ds", "a") / "zarr.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    assert OutputRoot.discover(tmp_path).source_fingerprint != before


def test_image_source_token_changes_when_a_store_changes(tmp_path: Path) -> None:
    """It is a staleness fingerprint, not a report label (D4)."""
    from phenotypic.gui.results_viewer._output_root import _image_source_token

    _seed(tmp_path, ["a"])
    store = zarr_store_path(tmp_path, "ds", "a")
    before = _image_source_token([store / "zarr.json"])
    (store / "zarr.json").write_text('{"changed": true}', encoding="utf-8")
    assert _image_source_token([store / "zarr.json"]) != before
```

- [ ] **Step 3: Port the four sites.** Replace `hdf_path` with `store_path` (delegating to
`BundleLayout.store_path`); replace `rglob("*.h5")` with a bounded
`(results / dataset / DIR_ZARR).glob(f"*{STORE_SUFFIX}")` per dataset directory that yields
**each store's `zarr.json`**, not the store directory; and point `_image_source_token`'s
source path at `store / "zarr.json"` likewise.

- [ ] **Step 4: Run** `uv run pytest tests/unit/gui/results_viewer tests/gui/results_viewer -q`. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_output_root.py tests/unit/gui/results_viewer
git commit -m "refactor(gui): discover store directories without walking into them

rglob('*.h5') becomes a bounded per-dataset glob for *.ome.zarr. A naive
rglob port recurses into every store, which is ~400k stat calls at 10k
images -- pathological on exactly the runs the viewer is for. The
output-consistency report's hdf path pair is ported too; it is a real call
site the spec's affected-module table did not list."
```

---

### Task 4.2: Pyramid-level tile reads

**Files:**
- Modify: `src/phenotypic/gui/_shared/tiles.py`
  (`_load_hdf_layer_rgb` line 291, `_hdf_layer_array_to_rgb` line 333, `crop_hdf_rgb`
  line 349, `_crop_hdf_layer_window` line 396, the caller at lines 509–518, `__all__` at
  line 1155)
- Test: `tests/unit/gui/shared/test_tiles_zarr.py` (create)

**Interfaces:**
- Produces:
  ```python
  def _load_zarr_layer_rgb(store: str, content_token: str, layer: LayerName, target_px: int) -> Image
  def crop_store_rgb(store_path, layer, window, content_token, ...) -> Image
  def select_pyramid_level(store_path: Path, layer: str, target_px: int) -> int
  ```

**Constraints specific to this task:**
- `select_pyramid_level` returns the **smallest level whose longest edge still covers**
  `target_px` — i.e. the coarsest level that does not under-sample the request. Reading a
  level finer than needed is the current behaviour and wastes the whole point of the change;
  reading one coarser produces a visibly soft tile.
- Level metadata comes from `phenotypic.pyramid.levels` plus the per-level array shapes.
  **Never infer the level count from directory listing** — a `.part` sweep or a partially
  written store would give a wrong answer.
- **Read amplification is real, and the docstring must say so.** Slicing a 64×64 colony
  crop from a sharded level 0 costs a shard-index read plus one full `1024×1024` inner
  chunk. It is cheap, but it is not "the same as h5py", which an earlier draft implied.
- The existing LRU cache keys stay content-keyed; only the source changes.

- [ ] **Step 1: Write the failing test**

```python
"""Tile reads select a pyramid level instead of decoding the whole layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.gui._shared.tiles import select_pyramid_level
from phenotypic.data import load_synth_yeast_plate


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")


def test_full_size_request_selects_level_zero(store: Path) -> None:
    level0 = Image.load_layer_zarr(store, "gray", level=0)
    assert select_pyramid_level(store, "gray", max(level0.shape)) == 0


def test_small_request_selects_a_coarse_level(store: Path) -> None:
    assert select_pyramid_level(store, "gray", 64) > 0


def test_selected_level_still_covers_the_request(store: Path) -> None:
    """Coarser than the request would render visibly soft."""
    for target in (64, 128, 256, 512, 1024):
        level = select_pyramid_level(store, "gray", target)
        shape = Image.load_layer_zarr(store, "gray", level=level).shape
        assert max(shape) >= target or level == 0


def test_selection_never_reads_finer_than_necessary(store: Path) -> None:
    level = select_pyramid_level(store, "gray", 256)
    if level > 0:
        finer = Image.load_layer_zarr(store, "gray", level=level - 1).shape
        assert max(finer) > 256


def test_level_count_comes_from_metadata_not_directory_listing(
    store: Path, monkeypatch
) -> None:
    """A .part sweep or a partial write would make a listing lie."""
    import shutil

    shutil.rmtree(store / "gray" / "1")
    with pytest.raises(Exception):
        select_pyramid_level(store, "gray", 64)


def test_single_level_store_always_selects_zero(tmp_path: Path) -> None:
    """The levels=1 path: builder node previews."""
    flat = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "f.ome.zarr", layers=("gray",)
    )
    assert select_pyramid_level(flat, "gray", 32) == 0


def test_crop_matches_the_full_resolution_slice(store: Path) -> None:
    from phenotypic.gui._shared.tiles import crop_store_rgb

    full = Image.load_layer_zarr(store, "gray", level=0)
    crop = np.asarray(crop_store_rgb(store, "gray", (10, 10, 74, 74), "tok"))
    np.testing.assert_array_equal(crop[..., 0], full[10:74, 10:74])
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.** Rename the four functions, add `select_pyramid_level` reading
`phenotypic.pyramid.levels` and each level's shape, and update `__all__` at line 1155.

- [ ] **Step 4: Run** `uv run pytest tests/unit/gui tests/gui/_shared -q`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/tiles.py tests/unit/gui
git commit -m "perf(gui): read a pyramid level instead of decoding the whole layer

select_pyramid_level picks the coarsest level that still covers the request,
from phenotypic.pyramid.levels rather than a directory listing (a .part
sweep or a partial write would make a listing lie). Read amplification is
documented honestly: a 64x64 crop from a sharded level 0 costs a shard-index
read plus one full 1024x1024 inner chunk -- cheap, but not free."
```

---

### Task 4.3: The staleness traps

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_tile_routes.py`
  (`_ensure_hdf_layer_source_png` lines 462–477: `file_fingerprint` at 473,
  `stat().st_mtime_ns` compare at 466/469, `os.utime` at 477)
- Modify: `src/phenotypic/gui/builder/_preview_tiles.py` (line 76 mtime compare)
- Modify: `src/phenotypic/gui/_shared/tiles.py` (line 518 mtime-keyed crop path)
- Test: `tests/unit/gui/results_viewer/test_tile_cache_invalidation.py` (create)

**Constraints specific to this task:**

**Three live staleness sites, plus one that only needs a port.** An earlier draft of this
task counted four traps and included `_shared/tiles.py:518` among them; that is wrong, and
it also omitted the two sites that genuinely go content-blind (now in Task 4.1).

| Site | Problem |
|---|---|
| `_tile_routes.py:473` | `file_fingerprint()` opens its argument as a file → `IsADirectoryError` on a store. Must key on `paths_fingerprint([store / "zarr.json"])`. |
| `_tile_routes.py:466,469,477` | `stat().st_mtime_ns` compare and `os.utime` against the store |
| `_preview_tiles.py:76` | same mtime compare |
| `_shared/tiles.py:518` | **not a staleness site.** `crop_hdf_rgb` opens with `del mtime_ns` (`:386`) and its docstring says the parameter is "accepted for caller/API compatibility; crop reads are windowed and not full-layer cached" (`:375-376`). The `os.stat(h5).st_mtime_ns` feeds a discarded parameter, and `os.stat` works fine on a directory. Port it to the store path; do **not** add a staleness fix. OPEN-QUESTIONS **D15**. |

- **A store directory's `st_mtime_ns` does not change when a nested chunk is rewritten**
  (verified). Every staleness check must move to the **root `zarr.json`**, which the
  promote writes last on every publish — so its mtime and contents change exactly when the
  store's contents do.
- **`paths_fingerprint` does not "handle directories" in the sense the spec implies.** It
  emits a single sentinel byte for a directory and does not recurse
  (`_io_constants.py:215-217`), so `paths_fingerprint([store])` is a constant function of
  the path and would freeze the tile cache **permanently**. Always pass
  `[store / "zarr.json"]`.
- `os.utime` at line 477 must be applied against the root `zarr.json`'s mtime, not the
  directory's.
- ⚠️ **The content token must include the root's mtime, not only its bytes.** A Stage-3
  re-promote whose metadata did not change produces a **byte-identical** root `zarr.json`.
  The PNG is regenerated (its mtime is now older) — but `_load_zarr_layer_rgb` is
  `@functools.lru_cache`d on `(path, token, layer)` (`gui/_shared/tiles.py:290-292`), and an
  unchanged token means the "regenerated" PNG is written from the **old decoded array**. So
  the token is
  `paths_fingerprint([root_json]) + str(root_stat.st_mtime_ns)`, which changes on every
  promote whether or not the metadata did. Recorded as OPEN-QUESTIONS **B7/P17**.
- **The GUI deliberately does not see Stage 2's in-place objmap write.** Root-keying makes
  the cache invalidate on **promotes**, and Stage 2 does not promote — so the tile stays at
  Stage 1's zeros objmap until Stage 3 publishes. That is the correct behaviour, not a
  regression: the completion marker, not the store's shape, is what gates consumers, and a
  torn mid-Stage-2 objmap is exactly what a viewer should not be shown. It does mean spec
  §3.5's claim that the in-store write buys "the GUI can render a real objmap mid-run" is
  **false**, and §3.5 should drop it — the in-store write's remaining justification is
  third-party interop. Recorded as OPEN-QUESTIONS **D6**; the D1 decision (raw array
  retained outside the store) is what makes the in-store write purely an interop
  convenience rather than a correctness dependency.

- [ ] **Step 1: Write the failing test**

```python
"""Cache invalidation across the mtime/fingerprint traps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.sdk_ import file_fingerprint, paths_fingerprint
from phenotypic.data import load_synth_yeast_plate


def test_file_fingerprint_raises_on_a_store_directory(tmp_path: Path) -> None:
    """Pins the exact reason the tile route must switch helpers."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    with pytest.raises(IsADirectoryError):
        file_fingerprint(store)


def test_paths_fingerprint_handles_a_store_directory(tmp_path: Path) -> None:
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    assert paths_fingerprint([store / "zarr.json"]).startswith("sha256:")


def test_store_directory_mtime_does_not_change_when_a_chunk_is_rewritten(
    tmp_path: Path,
) -> None:
    """The verified fact the whole task exists for."""
    import os

    from phenotypic.sdk_.ngff_ import write_objmap_in_place

    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    before = os.stat(store).st_mtime_ns
    labels = Image.load_layer_zarr(store, "objmap")
    labels[0, 0] = 7
    write_objmap_in_place(store, labels)
    assert os.stat(store).st_mtime_ns == before


def test_root_zarr_json_changes_on_every_publish(tmp_path: Path) -> None:
    image = Image(load_synth_yeast_plate())
    store = image.save2zarr(tmp_path / "p.ome.zarr")
    before = paths_fingerprint([store / "zarr.json"])
    image._metadata.public["Metadata_Strain"] = "BY4742"
    image.save2zarr(tmp_path / "p.ome.zarr")
    assert paths_fingerprint([store / "zarr.json"]) != before


def test_served_tile_changes_after_a_promote(live_viewer) -> None:
    """End-to-end: republish under a live cache and assert the tile changes.

    Keyed on a PROMOTE, not an in-place write: the promote is what rewrites the
    root zarr.json, and it is what publishes a store consumers should see.
    """
    first = live_viewer.get_tile("ds", "img", layer="objmap")
    live_viewer.republish_with_objmap("ds", "img", value=7)  # goes through save2zarr
    second = live_viewer.get_tile("ds", "img", layer="objmap")
    assert first != second


def test_served_tile_is_unchanged_by_an_in_place_write(live_viewer) -> None:
    """Stage 2's in-place write is deliberately invisible to the GUI (D6).

    The completion marker, not the store's shape, gates consumers -- and a torn
    mid-Stage-2 objmap is exactly what a viewer must not be shown. Pinning this
    keeps a later 'fix' from re-introducing per-chunk cache invalidation.
    """
    first = live_viewer.get_tile("ds", "img", layer="objmap")
    live_viewer.rewrite_objmap_in_place("ds", "img", value=7)
    assert live_viewer.get_tile("ds", "img", layer="objmap") == first


def test_builder_preview_invalidates_on_store_change(builder_preview, tmp_path) -> None:
    first = builder_preview.png_bytes("block-1", "gray")
    builder_preview.rewrite_node_store("block-1")
    assert builder_preview.png_bytes("block-1", "gray") != first
```

- [ ] **Step 2: Run to verify failure.** `test_store_directory_mtime_does_not_change…` and
`test_file_fingerprint_raises…` should PASS immediately — they pin facts, not new
behaviour. The two cache-invalidation tests should FAIL.

- [ ] **Step 3: Port all four sites.** In `_ensure_hdf_layer_source_png` → rename to
`_ensure_store_layer_source_png` and key on `store / "zarr.json"`:

```python
def _ensure_store_layer_source_png(
    store: Path, layer: LayerName, source_png: Path, target_px: int
) -> None:
    """Materialise a source PNG for one store layer, invalidating on the root.

    A store directory's ``st_mtime_ns`` does **not** change when a nested chunk
    is rewritten, so staleness is keyed on the root ``zarr.json`` -- which the
    promote writes last on every publish, and therefore changes exactly when
    the store's contents do.
    """
    root_json = store / "zarr.json"
    root_stat = os.stat(root_json)
    if (
        source_png.exists()
        and source_png.stat().st_mtime_ns >= root_stat.st_mtime_ns
    ):
        return
    # Bytes AND mtime: a re-promote with unchanged metadata is byte-identical,
    # and a stale LRU key would serve the previously decoded array.
    content_token = (
        paths_fingerprint([root_json]).removeprefix("sha256:")[:16]
        + f"-{root_stat.st_mtime_ns}"
    )
    _load_zarr_layer_rgb(str(store), content_token, layer, target_px).save(source_png)
    os.utime(source_png, ns=(root_stat.st_mtime_ns, root_stat.st_mtime_ns))
```

Apply the same root-keying at `_preview_tiles.py:76` and `_shared/tiles.py:518`.

- [ ] **Step 4: Run** `uv run pytest tests/unit/gui tests/gui tests/e2e/gui -q`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui tests/unit/gui
git commit -m "fix(gui): key tile staleness on the root zarr.json, not the store dir

Four sites, two failure modes. file_fingerprint opens its argument as a
file and raises IsADirectoryError on a store; paths_fingerprint handles
directories. Separately, a store directory's st_mtime_ns does NOT change
when a nested chunk is rewritten -- verified by test -- so every mtime
compare and every os.utime moves to the root zarr.json, which the promote
writes last on every publish. The production route keys on a content
fingerprint and only the crop path uses mtime; both needed fixing."
```

---

### Task 4.4: Builder preview tiles

**Files:**
- Modify: `src/phenotypic/gui/builder/_preview_tiles.py`
  (`_channel_to_rgb_uint8` lines 50–63, `stage_channel_png` line 73, the manifest read at
  lines 124–128)
- Modify: `src/phenotypic/gui/builder/_preview_cache.py`
  (the h5py class probe at lines 160–170, `_describe` at lines 181–208)
- Test: `tests/unit/gui/builder/test_preview_tiles_zarr.py` (create)

**Constraints specific to this task:**
- `_preview_cache.py:160-170` opens the node artifact with `h5py` to read
  `phenotypic_class` and dispatch `GridImage.load_hdf5` vs `Image.load_hdf5`. Replace the
  whole probe with `load_image_from_store` (Task 2.1), which does exactly this against
  `attributes.phenotypic.image_class`.
- `_preview_cache.py:197-208` reads layer names and shape from the HDF to build the
  manifest node description. Read them from `phenotypic.series` and the level-0 array
  shapes instead — do **not** open a full `Image` for a manifest entry.
- Task 2.4 already renamed the manifest key to `"store"` and bumped `MANIFEST_VERSION`;
  this task consumes that rename at `_preview_tiles.py:124`.

- [ ] **Step 1: Write the failing test**

```python
def test_channel_png_renders_from_a_node_store(tmp_path) -> None:
    from phenotypic import Image
    from phenotypic.gui.builder._preview_tiles import stage_channel_png
    from phenotypic.data import load_synth_yeast_plate

    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr", layers=("gray", "detect_mat", "objmap")
    )
    png = stage_channel_png(tmp_path, "block-1", "gray", store)
    assert png.is_file() and png.stat().st_size > 0


def test_manifest_describe_does_not_load_a_full_image(tmp_path, monkeypatch) -> None:
    from phenotypic import Image
    from phenotypic.gui.builder import _preview_cache
    from phenotypic.data import load_synth_yeast_plate

    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr", layers=("gray",)
    )
    monkeypatch.setattr(
        Image, "load_zarr", lambda *a, **k: pytest.fail("manifest must not load an Image")
    )
    node = _preview_cache._describe("block-1", store.name, base_dir=tmp_path)
    assert node["layers"] == ["gray"]


def test_class_dispatch_uses_image_class_not_h5py(tmp_path) -> None:
    from phenotypic import GridImage
    from phenotypic.sdk_ import load_image_from_store
    from phenotypic.data import load_synth_yeast_plate

    store = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12).save2zarr(
        tmp_path / "g.ome.zarr"
    )
    assert type(load_image_from_store(store)).__name__ == "GridImage"
```

- [ ] **Step 2: Run to verify failure. Step 3: port. Step 4: run**
`uv run pytest tests/unit/gui/builder tests/gui/builder tests/e2e/gui -q`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/builder tests/unit/gui/builder
git commit -m "refactor(gui): render builder previews from node stores

The h5py class probe becomes load_image_from_store, which dispatches on
attributes.phenotypic.image_class. Manifest descriptions read layer names
and shapes from phenotypic.series and the level-0 arrays rather than
opening a full Image for a cache entry."
```

---

## Phase 4 exit criteria

- [ ] `uv run pytest tests/unit/gui tests/gui tests/e2e/gui -q` is green. **`tests/gui`
      is not optional here** — it is in `testpaths` (`pyproject.toml:200`) and holds twelve
      of this phase's files (see the README's test inventory). Omitting it defers the
      breakage to Phase 7.
- [ ] `grep -rn "file_fingerprint" src/phenotypic/gui/` returns nothing pointed at a store.
- [ ] `grep -rn "\.h5\|load_hdf5\|hdf_path\|_load_hdf_layer_rgb\|crop_hdf_rgb" src/phenotypic/gui/` returns nothing.
- [ ] The three live staleness sites plus the two Task 4.1 fingerprints all key on `zarr.json`, verified by
      `grep -rn 'zarr.json' src/phenotypic/gui/ | wc -l` being at least 5.
- [ ] A whole-plate tile request measurably reads fewer bytes than level 0 — assert in
      `test_small_request_selects_a_coarse_level`.
# Phase 5 — `--mode migrate`

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §5, plus the supersession notes in the spec header.

**Depends on:** Phase 2.
**Runs in parallel with:** Phases 3 and 4.
**Must land before Phase 6** — migration is built on the legacy HDF readers that Phase 6
retires from the public surface.

`--mode migrate` converts an existing output tree in place: per-image `.h5` → `.ome.zarr`,
legacy per-topic metadata headers → canonical flat `Metadata_<Label>`, and
`deliverables/metadata.csv` → canonical headers with the original bytes preserved beside it.

---

### Task 5.1: `migrate_hdf_to_zarr` and `migrate_run_hdf_to_zarr`

**Files:**
- Create: `src/phenotypic/sdk_/_hdf_to_zarr.py`
- Test: `tests/unit/sdk_/test_hdf_to_zarr.py` (create)
- Fixtures: `tests/fixtures/legacy_hdf/{v1_flat,v2_grouped,v2_enh_gray}/` (create)

**Interfaces:**
- Consumes: `Image._load_v2_grouped` (line 984), `Image._load_legacy_flat_group` (line 1079),
  `Image.save2zarr`, `load_image_from_hdf`.
- Produces:
  ```python
  def migrate_hdf_to_zarr(src: Path, dst: Path | None = None, *, keep_source: bool = True) -> Path
  def migrate_run_hdf_to_zarr(output_dir: Path, *, keep_source: bool = True, njobs: int = 1, dry_run: bool = False) -> MigrationReport
  @dataclass(frozen=True)
  class MigrationReport:
      converted: int
      skipped: int
      failed: tuple[tuple[Path, str], ...]
  ```

**Constraints specific to this task:**
- Reuse the **existing** v1-flat and v2-grouped loaders. Do not write a third HDF reader.
- **The legacy `enh_gray` layer maps to `detect_mat`.** It is the pre-rename name still
  handled at `_cli_staged_resume.py:82` and must not be dropped silently. A fixture with an
  `enh_gray` layer is mandatory, not optional.
- ⚠️ **`_load_v2_grouped` cannot currently read that fixture.** Verified: the v2 loader does
  a bare `layers["detect_mat"]` (`_image_io_handler.py:1035-1036`) with **no fallback**;
  only the v1-flat loader has one (`:1100-1108`, `# Backward compat: try 'detect_mat'
  first, fall back to 'enh_gray'`). Meanwhile `valid_staged_hdf`
  (`_cli_staged_resume.py:81-83`) accepts `enh_gray` at `schema_version >= 2`, so the code
  believes such files exist in the wild. **Step 3a below adds the fallback**, which is a
  change to a legacy reader that must land here — before Phase 6 retires it. Recorded as
  OPEN-QUESTIONS **D8**.
- Header canonicalization happens **in the same pass**: a converted store is canonical by
  construction. There is no separate header pass for anything that goes through conversion.
- Sources are **retained by default** (`keep_source=True`). Deletion is opt-in.
- Migration is **resumable and restartable**: a store that already exists and passes
  `valid_staged_store` is skipped, so re-running after an interruption is the recovery
  procedure. There is no `--resume` flag.
- Conversion goes through the §3.2 promote, so an interrupted conversion leaves no valid
  root and is simply redone.

- [ ] **Step 1: Build the golden fixtures**

Write a one-off generator under `tests/fixtures/legacy_hdf/_generate.py` that produces
three `.h5` files from `load_synth_yeast_plate()` — one v1-flat, one v2-grouped, one
v2-grouped with the layer named `enh_gray` instead of `detect_mat` — and commit the
resulting `.h5` files. Commit the generator too, so the fixtures can be rebuilt after the
HDF writer is removed in Phase 6.

- [ ] **Step 2: Write the failing test**

```python
"""Legacy HDF -> store conversion must equal a freshly written store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.sdk_._hdf_to_zarr import migrate_hdf_to_zarr, migrate_run_hdf_to_zarr
from phenotypic.sdk_.ngff_ import (
    PhenotypicAttr,
    read_phenotypic_attributes,
    valid_staged_store,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_hdf"


@pytest.mark.parametrize("layout", ["v1_flat", "v2_grouped", "v2_enh_gray"])
def test_conversion_produces_a_valid_store(layout: str, tmp_path: Path) -> None:
    store = migrate_hdf_to_zarr(FIXTURES / layout / "img.h5", tmp_path / "img.ome.zarr")
    assert valid_staged_store(store) is True


@pytest.mark.parametrize("layout", ["v1_flat", "v2_grouped", "v2_enh_gray"])
def test_converted_store_conforms(layout: str, tmp_path: Path) -> None:
    from tests._ngff_conformance import assert_store_conforms

    assert_store_conforms(
        migrate_hdf_to_zarr(FIXTURES / layout / "img.h5", tmp_path / "img.ome.zarr")
    )


def test_enh_gray_maps_to_detect_mat(tmp_path: Path) -> None:
    """The pre-rename layer name; dropping it silently loses the detection matrix."""
    store = migrate_hdf_to_zarr(
        FIXTURES / "v2_enh_gray" / "img.h5", tmp_path / "img.ome.zarr"
    )
    block = read_phenotypic_attributes(store)
    assert "detect_mat" in block[PhenotypicAttr.SERIES]
    assert "enh_gray" not in block[PhenotypicAttr.SERIES]
    assert Image.load_layer_zarr(store, "detect_mat").any()


def test_converted_equals_a_freshly_written_store(tmp_path: Path) -> None:
    converted = migrate_hdf_to_zarr(
        FIXTURES / "v2_grouped" / "img.h5", tmp_path / "converted.ome.zarr"
    )
    fresh = Image.load_hdf5(FIXTURES / "v2_grouped" / "img.h5").save2zarr(
        tmp_path / "fresh.ome.zarr"
    )
    for layer in ("gray", "detect_mat", "objmap"):
        np.testing.assert_array_equal(
            Image.load_layer_zarr(converted, layer),
            Image.load_layer_zarr(fresh, layer),
        )
    a = read_phenotypic_attributes(converted)
    b = read_phenotypic_attributes(fresh)
    for key in (PhenotypicAttr.SERIES, PhenotypicAttr.LABELS, PhenotypicAttr.METADATA):
        assert a[key] == b[key]


def test_legacy_headers_are_canonicalized_in_the_same_pass(tmp_path: Path) -> None:
    store = migrate_hdf_to_zarr(
        FIXTURES / "v1_flat" / "img.h5", tmp_path / "img.ome.zarr"
    )
    block = read_phenotypic_attributes(store)
    for section in ("protected", "public"):
        for key in block[PhenotypicAttr.METADATA][section]:
            assert key.startswith("Metadata_"), key


def test_source_is_retained_by_default(tmp_path: Path) -> None:
    src = tmp_path / "img.h5"
    src.write_bytes((FIXTURES / "v2_grouped" / "img.h5").read_bytes())
    migrate_hdf_to_zarr(src, tmp_path / "img.ome.zarr")
    assert src.is_file()


def test_source_deletion_is_opt_in(tmp_path: Path) -> None:
    src = tmp_path / "img.h5"
    src.write_bytes((FIXTURES / "v2_grouped" / "img.h5").read_bytes())
    migrate_hdf_to_zarr(src, tmp_path / "img.ome.zarr", keep_source=False)
    assert not src.exists()


def test_run_migration_is_idempotent(legacy_run: Path) -> None:
    """Re-running after an interruption is the recovery procedure."""
    first = migrate_run_hdf_to_zarr(legacy_run)
    second = migrate_run_hdf_to_zarr(legacy_run)
    assert first.converted > 0
    assert second.converted == 0
    assert second.skipped == first.converted


def test_dry_run_writes_nothing(legacy_run: Path) -> None:
    from phenotypic.sdk_ import dataset_zarr_dir

    report = migrate_run_hdf_to_zarr(legacy_run, dry_run=True)
    assert report.converted > 0
    assert not dataset_zarr_dir(legacy_run, "ds").exists()


def test_a_failed_conversion_is_reported_not_raised(legacy_run: Path) -> None:
    corrupt = next((legacy_run / "results").rglob("*.h5"))
    corrupt.write_bytes(b"not an hdf")
    report = migrate_run_hdf_to_zarr(legacy_run)
    assert len(report.failed) == 1
    assert report.failed[0][0] == corrupt
```

- [ ] **Step 3: Run to verify failure.** Expected: `ModuleNotFoundError`, plus a `KeyError:
'detect_mat'` on the `v2_enh_gray` fixture once the module exists — that second failure is
what Step 3a fixes.

- [ ] **Step 3a: Give `_load_v2_grouped` the `enh_gray` fallback**

In `_image_io_handler.py`, replace the bare lookup at line 1035 with the same
backward-compat shape the v1-flat loader already uses at `:1100-1108`:

```python
        # Detection matrix + mode. Backward compat: 'enh_gray' is the
        # pre-rename name, still accepted by valid_staged_hdf at
        # _cli_staged_resume.py:82, so schema-2 files carrying it exist.
        if "detect_mat" in layers:
            detect_mat_ds = layers["detect_mat"]
            detect_matrix_data = detect_mat_ds[()]
            detect_mode = detect_mat_ds.attrs.get("detect_mode", "gray")
            if isinstance(detect_mode, bytes):
                detect_mode = detect_mode.decode("utf-8", errors="replace")
        else:
            detect_matrix_data = layers["enh_gray"][()]
            detect_mode = "gray"
```

Add a matching test in `tests/unit/core/` asserting a v2-grouped HDF carrying `enh_gray`
loads with a populated `detect_mat`. Phase 6 Task 6.2 keeps `_load_v2_grouped` as a private
migration reader, so this fallback survives the retirement.

- [ ] **Step 4: Implement.** `migrate_hdf_to_zarr` opens the source through
`load_image_from_hdf` (which already dispatches `Image`/`GridImage` on `phenotypic_class`),
renames `enh_gray` → `detect_mat` if present, canonicalizes headers via the existing
metadata-migration helpers, and calls `save2zarr`. `migrate_run_hdf_to_zarr` walks
`results/*/hdf/*.h5`, skips any stem whose store already passes `valid_staged_store`, and
fans out over `njobs` processes.

- [ ] **Step 5: Run** `uv run pytest tests/unit/sdk_/test_hdf_to_zarr.py -v`. Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_hdf_to_zarr.py tests/unit/sdk_/test_hdf_to_zarr.py tests/fixtures/legacy_hdf
git commit -m "feat(sdk): convert legacy per-image HDFs to OME-Zarr stores

Reuses the existing v1-flat and v2-grouped loaders rather than adding a
third reader. The legacy enh_gray layer maps to detect_mat -- it is the
pre-rename name still handled in the staged resume path, and a fixture
carrying it is part of the suite. Header canonicalization happens in the
same pass, so a converted store is canonical by construction. Conversion
goes through the promote, so an interrupted run is simply re-run."
```

---

### Task 5.2: `deliverables/metadata.csv` rewrite with preserved original

**Files:**
- Modify: `src/phenotypic/sdk_/_hdf_to_zarr.py`
- Test: `tests/unit/sdk_/test_metadata_csv_migration.py` (create)

**Constraints specific to this task:**
- This **narrows a locked decision of the flat-metadata design** (its decision #7: "the
  startup metadata snapshot is immutable provenance… never rewritten"). The narrowing is
  "never rewritten **as a side effect**". Finalization, chunk writers, and recompile still
  never rewrite it. Only `--mode migrate` does, and only after copying the untouched bytes.
- Order is load-bearing: copy `metadata.csv` → `metadata.original.csv` **byte-for-byte
  first**, then rewrite. A crash between the two leaves both the original and the
  unrewritten file, which is recoverable; the reverse order loses the provenance.
- `metadata.original.csv` is written once and never overwritten — a second migration must
  not clobber the first run's original with an already-canonical file.
- ⚠️ **`metadata.csv` is not inert provenance — its SHA-256 is load-bearing state.**
  `phenotypicCLI.py:276` and `:1338-1341` write `state.config["metadata_sha256"]`;
  `_cli_completion.py:541-547` folds it into `finalization_input_digest`; `:391-399`
  recomputes `expected_finalization` from it to decide whether the published aggregate is
  still valid. Rewriting the file therefore forces a choice, and **both obvious answers are
  wrong**: leave the recorded digest → the aggregate publication marker stops validating and
  the next run re-finalizes everything; update it → the digest no longer matches
  `metadata.original.csv`, which is the provenance this task exists to preserve. Recorded as
  OPEN-QUESTIONS **D9**, undecided. The plan's working assumption is that migrate updates
  `state.config["metadata_sha256"]` to the canonicalized file **and** records a new
  `metadata_original_sha256` beside it, so provenance is still verifiable — but this needs a
  decision, and the test below pins whichever answer is chosen.
- ⚠️ **`metadata.csv` is read again after migration.** `_snapshot_metadata_csv`
  (`phenotypicCLI.py:241-282`) runs at the start of `full`, `recompile` (`:1329`), and
  incremental startup (`:294-303`). If a user passes `--metadata <the original csv>` again
  after migrating, `destination.read_bytes() != payload` (`:270`) and the canonicalized file
  is **overwritten with the raw original** (`:280`) — silently reverting the migration. The
  snapshot must recognize an already-canonicalized destination and not revert it.

- [ ] **Step 1: Write the failing test**

```python
def test_original_bytes_are_preserved_verbatim(legacy_run: Path) -> None:
    from phenotypic.sdk_._hdf_to_zarr import migrate_metadata_csv

    source = legacy_run / "deliverables" / "metadata.csv"
    before = source.read_bytes()
    migrate_metadata_csv(legacy_run)
    assert (legacy_run / "deliverables" / "metadata.original.csv").read_bytes() == before


def test_metadata_csv_is_rewritten_with_canonical_headers(legacy_run: Path) -> None:
    import csv

    from phenotypic.sdk_._hdf_to_zarr import migrate_metadata_csv

    migrate_metadata_csv(legacy_run)
    with (legacy_run / "deliverables" / "metadata.csv").open(encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert all(column.startswith("Metadata_") for column in header if column), header


def test_original_is_never_overwritten_by_a_second_migration(legacy_run: Path) -> None:
    from phenotypic.sdk_._hdf_to_zarr import migrate_metadata_csv

    migrate_metadata_csv(legacy_run)
    first = (legacy_run / "deliverables" / "metadata.original.csv").read_bytes()
    migrate_metadata_csv(legacy_run)
    assert (legacy_run / "deliverables" / "metadata.original.csv").read_bytes() == first


def test_copy_happens_before_the_rewrite(legacy_run: Path, monkeypatch) -> None:
    """A crash between the two must leave provenance, not lose it."""
    order: list[str] = []
    import phenotypic.sdk_._hdf_to_zarr as module

    monkeypatch.setattr(module.shutil, "copyfile", lambda *a: order.append("copy"))
    monkeypatch.setattr(module, "_write_canonical_csv", lambda *a: order.append("write"))
    module.migrate_metadata_csv(legacy_run)
    assert order == ["copy", "write"]


def test_finalization_still_never_rewrites_metadata_csv(finished_run: Path) -> None:
    """The narrowing is 'never as a side effect', not 'sometimes'."""
    from phenotypic._cli._cli_output_manager import finalize_post_master_outputs

    before = (finished_run / "deliverables" / "metadata.csv").read_bytes()
    finalize_post_master_outputs(finished_run)
    assert (finished_run / "deliverables" / "metadata.csv").read_bytes() == before


def test_migration_keeps_the_published_aggregate_valid(finished_run: Path) -> None:
    """metadata_sha256 feeds finalization_input_digest (D9)."""
    from phenotypic._cli._cli_completion import aggregate_publication_is_valid
    from phenotypic.sdk_._hdf_to_zarr import migrate_metadata_csv

    assert aggregate_publication_is_valid(finished_run) is True
    migrate_metadata_csv(finished_run)
    assert aggregate_publication_is_valid(finished_run) is True, (
        "leaving metadata_sha256 stale re-finalizes every image on the next run"
    )


def test_provenance_stays_verifiable_after_the_digest_update(finished_run: Path) -> None:
    """Updating metadata_sha256 must not orphan metadata.original.csv (D9)."""
    import hashlib
    import json

    from phenotypic.sdk_._hdf_to_zarr import migrate_metadata_csv

    migrate_metadata_csv(finished_run)
    state = json.loads(
        (finished_run / ".phenotypic" / "state.json").read_text(encoding="utf-8")
    )
    original = (finished_run / "deliverables" / "metadata.original.csv").read_bytes()
    assert (
        state["config"]["metadata_original_sha256"]
        == hashlib.sha256(original).hexdigest()
    )


def test_resnapshotting_the_original_does_not_revert_the_migration(
    finished_run: Path, cli_runner
) -> None:
    """_snapshot_metadata_csv:270-280 would otherwise overwrite the canonical file."""
    from phenotypic.phenotypicCLI import main
    from phenotypic.sdk_._hdf_to_zarr import migrate_metadata_csv

    migrate_metadata_csv(finished_run)
    canonical = (finished_run / "deliverables" / "metadata.csv").read_bytes()
    cli_runner.invoke(
        main,
        [
            "--mode",
            "recompile",
            "--output",
            str(finished_run),
            "--metadata",
            str(finished_run / "deliverables" / "metadata.original.csv"),
        ],
    )
    assert (finished_run / "deliverables" / "metadata.csv").read_bytes() == canonical
```

- [ ] **Step 2–4: Run to verify failure, implement, re-run.**

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_hdf_to_zarr.py tests/unit/sdk_/test_metadata_csv_migration.py
git commit -m "feat(sdk): canonicalize deliverables/metadata.csv under --mode migrate

Narrows flat-metadata decision #7 from 'never rewritten' to 'never
rewritten as a side effect'. The untouched bytes are copied to
metadata.original.csv first, so a crash between the copy and the rewrite
leaves provenance rather than losing it, and the original is never
overwritten by a second migration. Finalization, chunk writers, and
recompile still never touch it, asserted by test."
```

---

### Task 5.3: `--mode migrate` CLI wiring

**Files:**
- Create: `src/phenotypic/_cli/_cli_migrate.py`
- Modify: `src/phenotypic/phenotypicCLI.py` (`--mode` choices line 943; the mode-validation
  block at lines 1217–1244; the module docstring's mode list at lines 71–80 and 1183)
- Test: `tests/unit/cli/test_cli_migrate_mode.py` (create)

**Interfaces:**
- Produces: `--mode migrate --output <dir> [--njobs N] [--dry-run]`.

**Constraints specific to this task:**
- `migrate` joins `{full, measure, recompile, process}` in the existing `click.Choice` and
  **reuses `recompile`'s argument validation**: no `--pipeline`, no `--input`, operates on
  an existing output root. Extend the existing `cli_mode in (...)` guards rather than
  adding a parallel branch — the guards at lines 1231, 1236, and 1240 each name modes
  explicitly.
- **Local-only, parallel via `--njobs`.** No SLURM controller, no array, no chunking, no
  `MaxArraySize` accounting. Migration is one-time, resumable, and restartable, so it does
  not justify another scheduler surface. A test asserts no `sbatch` is invoked.
- **A run whose output contains only `.h5` results fails with a pointer to this mode**
  rather than auto-migrating. Format conversion rewrites the entire results tree; that is
  typed deliberately, not triggered as a side effect of an unrelated `--mode full`.

- [ ] **Step 1: Write the failing test**

```python
def test_migrate_is_an_accepted_mode(cli_runner) -> None:
    result = cli_runner.invoke(main, ["--mode", "migrate", "--help"])
    assert result.exit_code == 0


def test_migrate_rejects_pipeline_and_input(cli_runner, legacy_run) -> None:
    for flag, value in (("--pipeline", "p.json"), ("--input", "imgs")):
        result = cli_runner.invoke(
            main, ["--mode", "migrate", "--output", str(legacy_run), flag, value]
        )
        assert result.exit_code != 0
        assert "migrate" in result.output


def test_migrate_converts_a_legacy_tree(cli_runner, legacy_run) -> None:
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_.ngff_ import valid_staged_store

    result = cli_runner.invoke(main, ["--mode", "migrate", "--output", str(legacy_run)])
    assert result.exit_code == 0
    assert valid_staged_store(zarr_store_path(legacy_run, "ds", "img"))


def test_migrate_never_submits_a_slurm_job(cli_runner, legacy_run, monkeypatch) -> None:
    """One-time, resumable work does not justify another scheduler surface."""
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("migrate must not shell out")
    )
    assert cli_runner.invoke(
        main, ["--mode", "migrate", "--output", str(legacy_run)]
    ).exit_code == 0


def test_a_legacy_only_output_fails_with_a_pointer(cli_runner, legacy_format_run) -> None:
    """Conversion rewrites the whole results tree; it must be typed deliberately."""
    result = cli_runner.invoke(
        main,
        ["--mode", "recompile", "--output", str(legacy_format_run)],
    )
    assert result.exit_code != 0
    assert "--mode migrate" in result.output


def test_dry_run_reports_without_writing(cli_runner, legacy_run) -> None:
    from phenotypic.sdk_ import dataset_zarr_dir

    result = cli_runner.invoke(
        main, ["--mode", "migrate", "--output", str(legacy_run), "--dry-run"]
    )
    assert result.exit_code == 0
    assert not dataset_zarr_dir(legacy_run, "ds").exists()
```

- [ ] **Step 2–4: Run to verify failure, implement, re-run.**

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_migrate.py src/phenotypic/phenotypicCLI.py tests/unit/cli/test_cli_migrate_mode.py
git commit -m "feat(cli): add --mode migrate

Joins the existing mode choice list and reuses recompile's argument
validation. Local-only with --njobs: migration is one-time, resumable, and
restartable, so it does not justify another SLURM controller/array surface
with its own chunking and MaxArraySize accounting. A legacy-only output
root fails with a pointer to this mode instead of auto-migrating, because
format conversion rewrites the whole results tree."
```

---

### Task 5.4: Move the metadata-schema migration under `migrate` and drop its SLURM fan-out

**Files:**
- Modify: `src/phenotypic/_cli/_cli_recompile_metadata_migration.py` (98 lines)
- Delete: `src/phenotypic/_cli/_cli_recompile_metadata_migration_slurm.py` (345 lines),
  `_cli_recompile_metadata_migration_worker.py` (534 lines) — **879** lines deleted. The
  spec's "~950" counts `_cli_recompile_metadata_migration.py` (98) too, but that file is
  *modified*, not deleted
- Modify: `src/phenotypic/phenotypicCLI.py` (recompile's migration hook)
- Test: `tests/unit/cli/test_recompile_no_longer_migrates.py` (create)

**Constraints specific to this task:**
- This **supersedes flat-metadata decision #1** ("Every recompile migrates automatically…
  not restricted to a special command"). `recompile` **stops rewriting** legacy headers but
  **keeps reading** them — its decision #3 (permanent stored-data compatibility) is
  untouched, so no existing output directory breaks. Recompile simply no longer mutates one
  as a side effect.
- The SLURM fan-out existed only because copying large HDFs is slow. Conversion through the
  promote is not, and migration is local-only (Task 5.3), so the fan-out has no remaining
  justification. Delete it; do not port it.
- Record the supersession in the flat-metadata spec in Task 6.4.

- [ ] **Step 1: Write the failing test**

> **Two distinct fixtures, or these tests contradict each other.** An earlier draft had
> `test_a_legacy_only_output_fails_with_a_pointer` (Task 5.3) asserting
> `--mode recompile --output <legacy_run>` exits **non-zero**, while
> `test_recompile_still_reads_legacy_headers` asserted the same command exits **zero** — and
> its body referenced an undefined `legacy_run_v2`. The intended distinction is real but the
> fixtures did not encode it. Recorded as OPEN-QUESTIONS **D16**. Define both in
> `tests/unit/cli/conftest.py`:
>
> - **`legacy_format_run`** — output tree whose `results/*/hdf/*.h5` exist and whose
>   `results/*/zarr/` does not. `recompile` must **fail** with a pointer to `--mode migrate`.
> - **`legacy_headers_run`** — output tree already converted to stores, but whose metadata
>   headers are still legacy per-topic names. `recompile` must **succeed**, read those
>   headers, and **not** rewrite them.

```python
def test_recompile_still_reads_legacy_headers(legacy_headers_run, cli_runner) -> None:
    """Decision #3 is untouched: no existing output directory breaks."""
    result = cli_runner.invoke(
        main, ["--mode", "recompile", "--output", str(legacy_headers_run)]
    )
    assert result.exit_code == 0


def test_recompile_does_not_rewrite_headers(legacy_headers_run, cli_runner) -> None:
    before = _read_headers(legacy_headers_run)
    cli_runner.invoke(main, ["--mode", "recompile", "--output", str(legacy_headers_run)])
    assert _read_headers(legacy_headers_run) == before


def test_the_slurm_fanout_modules_are_gone() -> None:
    import importlib

    for name in (
        "phenotypic._cli._cli_recompile_metadata_migration_slurm",
        "phenotypic._cli._cli_recompile_metadata_migration_worker",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_migrate_performs_the_header_migration(legacy_headers_run, cli_runner) -> None:
    before = _read_headers(legacy_headers_run)
    cli_runner.invoke(main, ["--mode", "migrate", "--output", str(legacy_headers_run)])
    after = _read_headers(legacy_headers_run)
    assert after != before
    assert all(h.startswith("Metadata_") for h in after)
```

- [ ] **Step 2–4: Run to verify failure, implement, re-run** `uv run pytest tests/unit/cli -q`.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "refactor(cli): move metadata-schema migration into --mode migrate

Supersedes flat-metadata decision #1. Recompile stops REWRITING legacy
headers but keeps READING them, so decision #3 (permanent stored-data
compatibility) is untouched and no existing output directory breaks --
recompile simply no longer mutates one as a side effect. The ~950-line
SLURM fan-out existed only because copying large HDFs is slow; conversion
through the promote is not, and migration is local-only, so it is deleted
rather than ported."
```

---

### Task 5.5: Header-only migration via hard-link promote

**Files:**
- Modify: `src/phenotypic/sdk_/_hdf_to_zarr.py`
- Test: `tests/unit/sdk_/test_header_only_migration.py` (create)

**Interfaces:**
- Produces: `migrate_store_headers(store: Path) -> Path`

**Constraints specific to this task:**
- A header rename is **not** "rewrites one small `zarr.json`". It also touches
  `OME/METADATA.ome.xml` (derived from `header_to_module()`) and each series'
  `multiscales[].name` / `omero.name`. It is a **multi-file publish**, and there is no
  atomic multi-file primitive.
- Flat-metadata decision #2 requires the original to survive a failed publication.
  Therefore: rebuild only the metadata files into a `.part` copy, **hard-link the unchanged
  chunk files** so the copy is cheap, and promote via §3.2.
- Hard-linking must degrade gracefully: on a filesystem that refuses `os.link` (some
  network mounts), fall back to a copy and log it. A failure to hard-link must not fail the
  migration. **Never a symlink** — a symlinked `.part` would not survive the promote's
  `rmtree(trash)`, whereas a hard link does (the link-count walk is in OPEN-QUESTIONS,
  "Data-flow conclusions that came back clean").
- **While a hard-linked `.part` exists, it and the live store share bytes.** Any in-place
  chunk write during that window (`write_objmap_in_place` opens `mode="r+"`) lands in both.
  Migration is offline so this cannot happen today; put it in the function's docstring so it
  stays true by intent rather than by accident.
- After promote, the store must still pass `assert_store_conforms`.
- ⚠️ **Rewriting a store's `zarr.json` invalidates its per-image completion marker.**
  `refresh_success_markers_after_metadata_migration` (`_cli_completion.py:136-155`) exists
  precisely because rewriting a per-image HDF invalidated the marker's `sha256` descriptor.
  A header-only store migration does exactly the same thing to the store descriptor
  introduced in Phase 3 Task 3.8, so that bridge must handle store descriptors too. Add a
  test asserting a migrated store still passes `valid_image_success`.
- ⚠️ **`sdk_/_metadata_migration.py` (~2,500 lines) is HDF-shaped and appears in no phase's
  file list.** `TargetKind` includes `"hdf"` (`:44`), targets are built from
  `dataset_root / "hdf"` (`:797`), and there are hdf-specific rollback fingerprints and
  receipts (`:1601-1604`, `:1792-1796`, `:1878-1885`, `:2415-2461`). Task 5.1 says
  canonicalization happens "via the existing metadata-migration helpers" without costing
  any of it. Recorded as OPEN-QUESTIONS **D10**. Scope this before starting: either the
  helpers gain a `"store"` `TargetKind`, or `_hdf_to_zarr.py` uses only their pure header
  mapping and leaves the target machinery to the HDF path it was written for.

- [ ] **Step 1: Write the failing test**

```python
def test_header_migration_rewrites_every_derived_file(canonical_store: Path) -> None:
    """Not one zarr.json: also OME-XML and each series' multiscales/omero name."""
    import json

    from phenotypic.sdk_._hdf_to_zarr import migrate_store_headers

    store = migrate_store_headers(canonical_store)
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert root["attributes"]["phenotypic"]["metadata_schema_version"] == 2
    assert "Metadata_" in (store / "OME" / "METADATA.ome.xml").read_text(encoding="utf-8")
    gray = json.loads((store / "gray" / "zarr.json").read_text(encoding="utf-8"))
    assert "name" in gray["attributes"]["ome"]["multiscales"][0]


def test_chunks_are_hard_linked_not_copied(canonical_store: Path) -> None:
    """Decision #2 needs a .part; a full copy would cost the whole store."""
    import os

    from phenotypic.sdk_._hdf_to_zarr import migrate_store_headers

    chunk = next((canonical_store / "gray" / "0").glob("*"))
    before = os.stat(chunk).st_ino
    store = migrate_store_headers(canonical_store)
    assert os.stat(next((store / "gray" / "0").glob("*"))).st_ino == before


def test_hard_link_failure_falls_back_to_copy(canonical_store: Path, monkeypatch, caplog) -> None:
    import os

    from phenotypic.sdk_._hdf_to_zarr import migrate_store_headers

    monkeypatch.setattr(os, "link", lambda *a: (_ for _ in ()).throw(OSError("nolink")))
    store = migrate_store_headers(canonical_store)
    assert (store / "gray" / "0").is_dir()
    assert any("hard link" in record.message.lower() for record in caplog.records)


def test_a_failed_publish_leaves_the_original_intact(canonical_store: Path, monkeypatch) -> None:
    """Flat-metadata decision #2."""
    import json

    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_._hdf_to_zarr import migrate_store_headers

    before = (canonical_store / "zarr.json").read_bytes()
    monkeypatch.setattr(
        ngff_, "promote_store", lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
    )
    with pytest.raises(OSError):
        migrate_store_headers(canonical_store)
    assert (canonical_store / "zarr.json").read_bytes() == before


def test_migrated_store_still_conforms(canonical_store: Path) -> None:
    from tests._ngff_conformance import assert_store_conforms
    from phenotypic.sdk_._hdf_to_zarr import migrate_store_headers

    assert_store_conforms(migrate_store_headers(canonical_store))
```

- [ ] **Step 2–4: Run to verify failure, implement, re-run.**

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_hdf_to_zarr.py tests/unit/sdk_/test_header_only_migration.py
git commit -m "feat(sdk): header-only migration through a hard-linked promote

A header rename is a multi-file publish -- the root zarr.json, the OME-XML
derived from header_to_module(), and each series' multiscales/omero name --
and there is no atomic multi-file primitive. Flat-metadata decision #2
requires the original to survive a failed publication, so the metadata
files are rebuilt into a .part whose chunk files are hard-linked, then
promoted. A filesystem that refuses os.link falls back to a copy with a log
line rather than failing the migration."
```

---

## Phase 5 exit criteria

- [ ] `uv run pytest tests/unit/sdk_/test_hdf_to_zarr.py tests/unit/sdk_/test_metadata_csv_migration.py tests/unit/sdk_/test_header_only_migration.py tests/unit/cli/test_cli_migrate_mode.py -q` is green.
- [ ] `uv run python -m phenotypic --mode migrate --output <a real legacy run> --dry-run` reports a non-zero conversion count and writes nothing.
- [ ] Running migrate twice on the same tree converts zero on the second pass.
- [ ] `grep -rn "sbatch" src/phenotypic/_cli/_cli_migrate.py src/phenotypic/sdk_/_hdf_to_zarr.py` returns nothing.
- [ ] The three golden fixtures (`v1_flat`, `v2_grouped`, `v2_enh_gray`) are committed with their generator.
# Phase 6 — Retirement: the HDF write path, the dead DataFrame layer, docs

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §5.4, §9, and the supersession notes in the spec header.

**Depends on:** Phases 3, 4, **and 5**. Phase 5 is built on the legacy HDF readers, so
nothing here may run until migration exists and is green.

This is a **public-API removal, not an internal cleanup**. `HDF` is re-exported in
`sdk_/__init__.py:240` and in `__all__`, and `phenotypic.sdk_` is published via a
`:recursive:` autosummary with `undoc-members`, so every public name here appears in the
rendered docs. `save2hdf5` / `load_hdf5` / `load_layer_hdf5` additionally appear in
`docs/source/api_reference/core/*.rst` and in runnable doctests.

---

### Task 6.1: Delete the dead HDF DataFrame layer

**Files:**
- Modify: `src/phenotypic/sdk_/hdf_.py` (1,984 lines → roughly 520)
- Test: `tests/unit/sdk_/test_hdf_open_recovery.py` (must keep passing unchanged)

**What is removed** — verified as having **no** call sites in `src/` or `tests/`, and
corroborated by commits `66734e8e9`, `3e8b58aa0`, `da9eb6dd8` removing the last consumers:

`preallocate_series_layout` (1107), `save_series_new` (1276), `save_series_update` (1367),
`save_series_append` (1506), `load_series` (1544), `_convert_categorical_columns` (1600),
`preallocate_frame_layout` (1622), `save_frame_new` (1727), `save_frame_update` (1797),
`save_frame_append` (1880), `load_frame` (1909), the fixed-length-string codecs
(`_get_string_dtype` 620, `_pad_or_truncate_string` 637, `_apply_fixed_length_to_strings`
658, `_trim_trailing_whitespace` 686, `_decode_fixed_length_strings` 698,
`_encode_values_for_hdf5` 727, `_encode_index_for_hdf5` 853, `_decode_values_from_hdf5`
965, `_decode_index_from_hdf5` 1009, `_create_resizable_dataset` 1085), plus the three
additional dead statics `assert_swmr_on` (525), `get_uncompressed_sizes_for_group` (540),
and `close_handle` (1967).

**What is kept — the keeper list.** Deleting past it breaks live tests:

| Keeper | Why |
|---|---|
| `_open_hdf_with_recovery` (34) | Migration read path; also the retry-with-backoff shape reused by `promote_store` |
| `_clear_hdf_consistency_flags` (113) | Called by the above |
| `safe_writer` (252), `swmr_writer` (277) | **Live callers** at `tests/unit/sdk_/test_hdf_open_recovery.py:104` and `:141`. A literal "delete the remainder" breaks that file. |
| `strict_writer` (314), `swmr_reader` (335) | **No call sites** — verified by grep over `src/` and `tests/`. Kept anyway: they are three-line properties on the same public `HDF` class as the two writers above, and removing half a symmetric writer/reader set is a worse public surface than keeping it. Recorded so the justification is honest rather than borrowed from the two that do have callers. |
| `reader` (338), `get_group` (341) and the group accessors | Migration read path |
| `save_array2hdf5` (493) | Used by `tests/fixtures/legacy_hdf/_generate.py` (Task 5.1), which must remain able to rebuild the golden fixtures after the production writer is gone |

**Constraints specific to this task:**
- The spec's earlier figure of 1,700 lines was **wrong**; the correct enumeration is
  ~1,346 lines, reaching ~1,463 with the three additional dead statics. Reaching 1,700 would
  require deleting the keeper list itself. Do not chase a line count.
- Re-verify emptiness immediately before deleting each symbol:
  `grep -rn "<symbol>" src/ tests/ docs/`. A Phase 3/4/5 change may have added a caller.

- [ ] **Step 1: Write the guard test**

Create `tests/unit/sdk_/test_hdf_surface.py`:

```python
"""Pins the HDF keeper list so a future cleanup cannot delete past it."""

from __future__ import annotations

import pytest

from phenotypic.sdk_.hdf_ import HDF, _clear_hdf_consistency_flags, _open_hdf_with_recovery

KEEPERS = [
    "safe_writer",
    "swmr_writer",
    "strict_writer",
    "swmr_reader",
    "reader",
    "get_group",
    "save_array2hdf5",
]

REMOVED = [
    "preallocate_series_layout",
    "save_series_new",
    "save_series_update",
    "save_series_append",
    "load_series",
    "preallocate_frame_layout",
    "save_frame_new",
    "save_frame_update",
    "save_frame_append",
    "load_frame",
    "assert_swmr_on",
    "get_uncompressed_sizes_for_group",
    "close_handle",
]


@pytest.mark.parametrize("name", KEEPERS)
def test_keeper_survives(name: str) -> None:
    assert hasattr(HDF, name), (
        f"{name} has live callers; deleting it breaks test_hdf_open_recovery.py "
        "or the legacy-fixture generator."
    )


@pytest.mark.parametrize("name", REMOVED)
def test_dead_dataframe_layer_is_gone(name: str) -> None:
    assert not hasattr(HDF, name)


def test_recovery_helpers_survive() -> None:
    assert callable(_open_hdf_with_recovery)
    assert callable(_clear_hdf_consistency_flags)
```

- [ ] **Step 2: Run to confirm the REMOVED half fails.**

```bash
uv run pytest tests/unit/sdk_/test_hdf_surface.py -v
```

- [ ] **Step 3: Delete, re-verifying each symbol's emptiness first.**

```bash
for sym in preallocate_series_layout save_series_new save_series_update \
           save_series_append load_series preallocate_frame_layout save_frame_new \
           save_frame_update save_frame_append load_frame assert_swmr_on \
           get_uncompressed_sizes_for_group close_handle; do
  echo "--- $sym"; grep -rn "$sym" src/ tests/ docs/ | grep -v test_hdf_surface.py
done
```

Expected: only `hdf_.py`'s own definitions. Then delete them.

- [ ] **Step 4: Run the suite**

```bash
uv run pytest tests/unit/sdk_/test_hdf_surface.py tests/unit/sdk_/test_hdf_open_recovery.py -v
uv run pytest tests/unit -q
wc -l src/phenotypic/sdk_/hdf_.py
```

Expected: green, and roughly 520 lines remaining.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/hdf_.py tests/unit/sdk_/test_hdf_surface.py
git commit -m "refactor(sdk): delete the dead HDF DataFrame layer

~1,463 lines with no call sites in src/ or tests/, corroborated by
66734e8e9, 3e8b58aa0, and da9eb6dd8 removing the last consumers. The
keeper list is pinned by test: safe_writer and swmr_writer have live
callers in test_hdf_open_recovery.py, and save_array2hdf5 is what the
legacy-fixture generator uses to rebuild the migration goldens after the
production writer is gone. An earlier estimate of 1,700 lines was wrong;
reaching it would mean deleting the keepers."
```

---

### Task 6.2: Remove the public HDF image API

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`
  (delete `_get_hdf5_group` 693, `_save_array2hdf5` 715, `_save_image2hdfgroup` 751,
  `_save_hdf5_metadata` 846, `save2hdf5` 871, `load_hdf5` 1155, `load_layer_hdf5` 1194;
  the doctests at lines 274 and 895; `save_intermediate_layers` was already replaced in
  Task 2.4)
- Modify: `src/phenotypic/_core/_image_parts/_grid_image_handler.py`
  (`_save_image2hdfgroup` 464)
- Modify: `docs/source/api_reference/core/image_methods.rst`,
  `docs/source/api_reference/core/grid_image_methods.rst`
- Create: `docs/source/release_notes/<next-version>.md` entry
- Test: `tests/unit/core/test_image_hdf_roundtrip.py` (rewrite as a removal guard),
  `tests/unit/core/test_load_layer_hdf5.py` (delete)

**Constraints specific to this task:**
- **`_load_from_hdf5_group` (971), `_load_v2_grouped` (984), and `_load_legacy_flat_group`
  (1079) are KEPT as private readers.** `--mode migrate` calls them. They are removed from
  no public surface because they were never on one.
- `load_hdf5` itself is removed from the public API, but migration needs a reader entry
  point: keep it as `_load_hdf5_for_migration`, called from `sdk_/_hdf_to_zarr.py` and
  `_io_constants.load_image_from_hdf`. Rename in this task and update **all three** call
  sites — the two in `src/`, plus **`phase-5-migrate.md`'s
  `test_converted_equals_a_freshly_written_store`**, which calls
  `Image.load_hdf5(FIXTURES / "v2_grouped" / "img.h5")` directly. An earlier draft counted
  two and would have left that test raising `AttributeError` at a phase whose exit criterion
  is a green full suite. Recorded as OPEN-QUESTIONS **B11**.
- Re-run `grep -rn "load_hdf5" src/ tests/` after the rename and confirm the only hits are
  the private name.
- The doctests at lines 274 and 895 are **runnable** and will fail collection if left
  pointing at removed methods. Rewrite them against `save2zarr` / `load_zarr`.
- A release note is required — this removes names from a published autosummary.

- [ ] **Step 1: Rewrite the round-trip test as a removal guard**

```python
"""The public HDF image API is gone; the private migration readers remain."""

from __future__ import annotations

import pytest

from phenotypic import GridImage, Image


@pytest.mark.parametrize(
    "name", ["save2hdf5", "load_hdf5", "load_layer_hdf5", "save_intermediate_layers"]
)
@pytest.mark.parametrize("cls", [Image, GridImage])
def test_public_hdf_api_is_removed(cls, name: str) -> None:
    assert not hasattr(cls, name)


@pytest.mark.parametrize(
    "name", ["_load_v2_grouped", "_load_legacy_flat_group", "_load_hdf5_for_migration"]
)
def test_private_migration_readers_survive(name: str) -> None:
    """--mode migrate is built on these; Phase 5 breaks without them."""
    assert hasattr(Image, name)


def test_migration_can_still_read_every_golden_fixture(tmp_path) -> None:
    from pathlib import Path

    from phenotypic.sdk_._hdf_to_zarr import migrate_hdf_to_zarr
    from phenotypic.sdk_.ngff_ import valid_staged_store

    fixtures = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_hdf"
    for layout in ("v1_flat", "v2_grouped", "v2_enh_gray"):
        store = migrate_hdf_to_zarr(
            fixtures / layout / "img.h5", tmp_path / f"{layout}.ome.zarr"
        )
        assert valid_staged_store(store) is True
```

- [ ] **Step 2: Run to verify it fails. Step 3: delete the write path and rename the reader.**

- [ ] **Step 4: Fix the docs**

In both `.rst` files, replace `save2hdf5` / `load_hdf5` / `load_layer_hdf5` with
`save2zarr` / `load_zarr` / `load_layer_zarr`. Add a release-note entry:

```markdown
### Removed

- `Image.save2hdf5`, `Image.load_hdf5`, `Image.load_layer_hdf5`, and
  `Image.save_intermediate_layers` (and their `GridImage` counterparts). Per-image
  storage is now an OME-Zarr (NGFF 0.5 / Zarr v3) store; use `save2zarr`,
  `load_zarr`, and `load_layer_zarr`.
- The DataFrame half of `phenotypic.sdk_.HDF` (`save_series_*`, `load_series`,
  `save_frame_*`, `load_frame`, `preallocate_*`, and their fixed-length-string
  codecs). These had no remaining call sites.

### Migration

Existing `.h5` output directories are converted with:

    uv run python -m phenotypic --mode migrate --output <previous-output-dir>

A run whose output contains only `.h5` results now fails with a pointer to this
command rather than converting as a side effect.

### Requires

- Python 3.11 or 3.12. Python 3.10 is no longer supported.
```

- [ ] **Step 5: Run the doc build and the doctests**

```bash
uv run pytest --doctest-modules src/phenotypic/_core -q
uv run sphinx-build -W -b html docs/source docs/_build/html
```

Expected: both green. `-W` turns the autosummary's dangling-reference warnings into errors,
which is how a missed `.rst` reference surfaces.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/_core docs tests/unit/core
git commit -m "refactor!: remove the public HDF image API

save2hdf5 / load_hdf5 / load_layer_hdf5 / save_intermediate_layers are
removed from Image and GridImage. This is a public-API removal, not an
internal cleanup: HDF is re-exported from sdk_ and phenotypic.sdk_ is
published via a :recursive: autosummary with undoc-members, so a release
note and .rst updates ship with it. The private v1-flat and v2-grouped
readers are kept -- --mode migrate is built on them -- and the runnable
doctests are rewritten against save2zarr/load_zarr."
```

---

### Task 6.3: Remove the HDF path constants

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (`DIR_HDF` 656, `dataset_hdf_dir` 1447,
  `HdfAttr` 1886, `load_image_from_hdf` 1936, `BundleLayout.hdf_path` 2063)
- Modify: `src/phenotypic/sdk_/__init__.py`

**Constraints specific to this task:**
- `load_image_from_hdf` and `dataset_hdf_dir` are still needed by `_hdf_to_zarr.py`. Move
  both **into** `sdk_/_hdf_to_zarr.py` as private helpers rather than deleting them, and
  drop them from `_io_constants.py` and `__all__`. That keeps the legacy layout knowledge
  in the one module that is allowed to know it.
- `HdfAttr` is dead once `load_image_from_hdf` moves; the moved copy carries its own
  `_PHENOTYPIC_CLASS` constant.
- `BundleLayout.hdf_path` is deleted outright — nothing reads legacy trees through
  `BundleLayout`.

- [ ] **Step 1: Write the guard**

```python
@pytest.mark.parametrize(
    "name", ["DIR_HDF", "dataset_hdf_dir", "HdfAttr", "load_image_from_hdf"]
)
def test_hdf_path_constants_are_gone(name: str) -> None:
    import phenotypic.sdk_ as sdk

    assert not hasattr(sdk, name)


def test_bundle_layout_has_no_hdf_path() -> None:
    from phenotypic.sdk_ import BundleLayout

    assert not hasattr(BundleLayout, "hdf_path")


def test_migration_still_resolves_legacy_directories(tmp_path) -> None:
    from phenotypic.sdk_._hdf_to_zarr import _dataset_hdf_dir

    assert _dataset_hdf_dir(tmp_path, "ds").name == "hdf"
```

- [ ] **Step 2–4: Run to verify failure, move the helpers, re-run**
`uv run pytest tests/unit -q`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_ tests/unit/sdk_
git commit -m "refactor(sdk): retire the HDF path constants

DIR_HDF, dataset_hdf_dir, HdfAttr, load_image_from_hdf, and
BundleLayout.hdf_path leave the shared layout module. The two that
migration still needs move into _hdf_to_zarr.py as private helpers, so
knowledge of the legacy tree layout lives only in the module allowed to
have it."
```

---

### Task 6.4: Documentation, CLAUDE.md, and supersessions

**Files:**
- Modify: `CLAUDE.md` (the `--mode` list, the output-layout section, the Gotchas entries
  naming `.h5` and `Image.load_hdf5`)
- Modify: `src/phenotypic/_cli/CLAUDE.md` (file inventory, staged-GPU sidecar description,
  master-vs-mirror section)
- Modify: `src/phenotypic/_core/CLAUDE.md`, `src/phenotypic/gui/CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-08-17-flat-metadata-namespace/design.md`
  (record decisions #1 and #7 as superseded)
- Modify: `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md`
  (**Status:** Draft → Implemented, with the plan linked)

**Constraints specific to this task:**
- The flat-metadata spec's decisions #1 and #7 must be **annotated in place as superseded**,
  with a link back to this design — not silently edited. The reasoning has to survive.
  Decision #3 is explicitly **not** superseded.
- `_cli/CLAUDE.md` currently describes the `.npy` objmap sidecar as the Stage-2 signal in
  several places, and the root `CLAUDE.md` repeats it. Both must move to the in-store label
  write plus the consumable token, or the next agent will reintroduce the sidecar.
- The staged-GPU `.npy` sidecar sentence in the root `CLAUDE.md`'s "SLURM array auxiliary
  work" section explicitly carves the sidecar out of the no-parallel-sidecar-jobs rule. That
  carve-out is now obsolete; replace it rather than deleting the surrounding rule, which
  still stands.

- [ ] **Step 1: Update the two specs**

In `2026-08-17-flat-metadata-namespace/design.md`, append to decisions #1 and #7:

```markdown
> **Superseded (2026-08-18)** by
> [2026-08-18-ome-zarr-image-store](../2026-08-18-ome-zarr-image-store/design.md).
> Metadata-schema migration moves out of `--mode recompile` and into `--mode migrate`.
> `recompile` stops rewriting legacy headers but keeps reading them — **decision #3
> (permanent stored-data compatibility) is untouched**, so no existing output directory
> breaks.
```

```markdown
> **Narrowed (2026-08-18)** by
> [2026-08-18-ome-zarr-image-store](../2026-08-18-ome-zarr-image-store/design.md) to
> "never rewritten *as a side effect*". `--mode migrate` rewrites
> `deliverables/metadata.csv` with canonical headers after first copying the untouched
> bytes to `deliverables/metadata.original.csv`. Finalization, chunk writers, and
> recompile still never rewrite it.
```

- [ ] **Step 2: Update the four CLAUDE.md files.** Verify no stale reference remains:

```bash
grep -rn "\.h5\|save2hdf5\|load_hdf5\|load_layer_hdf5\|save_intermediate_layers\|npy.*sidecar\|objmap sidecar" CLAUDE.md AGENTS.md src/phenotypic/*/CLAUDE.md docs/source
```

Expected: only migration-context mentions.

- [ ] **Step 3: Build the docs**

```bash
uv run sphinx-build -W -b html docs/source docs/_build/html
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md src/phenotypic/*/CLAUDE.md docs
git commit -m "docs: record the OME-Zarr layout and the two supersessions

The flat-metadata spec's decisions #1 and #7 are annotated in place rather
than edited away, so the reasoning survives; decision #3 is explicitly not
superseded. Every CLAUDE.md that described the .npy objmap sidecar as the
Stage-2 signal now describes the in-store label write plus the consumable
token, including the carve-out in the SLURM auxiliary-work rule -- which is
obsolete as written, while the surrounding rule still stands."
```

---

## Phase 6 exit criteria

- [ ] `uv run pytest tests -q` is green (whole suite, not just unit).
- [ ] `uv run sphinx-build -W -b html docs/source docs/_build/html` succeeds.
- [ ] `grep -rn "save2hdf5\|load_layer_hdf5\|save_intermediate_layers" src/ docs/source` returns nothing.
- [ ] `grep -rn "\.h5" src/phenotypic --include='*.py' | grep -v _hdf_to_zarr` returns nothing.
- [ ] `wc -l src/phenotypic/sdk_/hdf_.py` is roughly 520.
- [ ] A release note entry exists naming every removed public symbol and the migration command.
# Phase 7 — Verification: commit protocol, invariant gates, Windows lane, sign-off

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §3.8, §7.

**Depends on:** Phases 3, 4, 5, 6.

Everything here is a test or a gate. No production behaviour changes in this phase — if a
test written here fails, the fix belongs in the phase that owns the code, not here.

---

### Task 7.1: Commit-protocol tests

**Files:**
- Test: `tests/integration/cli/test_commit_protocol.py` (create)

**Constraints specific to this task:**
Three cases, **not one**:

- **(a)** Interrupt after chunks but before the root `zarr.json` → the store reads as
  absent, and the resume classifier returns `"stage1"`.
- **(b)** Two concurrent writers on the same stem → distinct `.part` directories and one
  coherent winner. Neither a merged directory nor a store failing `valid_staged_store`.
- **(c)** A stale `.part` from a killed process is **removed, not merged into**.

Case (a) must be **proven to be able to fail**: reverse the write order (root first, then
chunks) and confirm the test goes red. A guard that cannot be shown to fail is not a guard.

- [ ] **Step 1: Write the tests**

```python
"""The three commit-protocol cases. Case (a) is proven to have teeth."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from phenotypic import Image
from phenotypic.sdk_ import ngff_, zarr_store_path
from phenotypic.sdk_.ngff_ import valid_staged_store
from phenotypic.data import load_synth_yeast_plate


# --- (a) interrupted before the root ---------------------------------------


def test_interrupt_before_the_root_reads_as_absent(tmp_path: Path, monkeypatch) -> None:
    image = Image(load_synth_yeast_plate())
    final = tmp_path / "p.ome.zarr"

    def _die(*args, **kwargs):
        raise KeyboardInterrupt("killed after chunks, before the root")

    monkeypatch.setattr(ngff_, "promote_store", _die)
    with pytest.raises(KeyboardInterrupt):
        image.save2zarr(final)
    assert not final.exists()
    assert valid_staged_store(final) is False


def test_a_part_without_a_root_never_validates(tmp_path: Path) -> None:
    """Even if a sweep were skipped, a rootless directory must not pass."""
    image = Image(load_synth_yeast_plate())
    final = tmp_path / "p.ome.zarr"
    image.save2zarr(final)
    (final / "zarr.json").unlink()
    assert valid_staged_store(final) is False


def test_interrupted_store_classifies_stage1(staged_run, monkeypatch) -> None:
    from phenotypic._cli._cli_staged_resume import classify_staged_image

    staged_run.run_stage1()
    store = zarr_store_path(staged_run.output_dir, "ds", "img")
    (store / "zarr.json").unlink()
    assert (
        classify_staged_image(
            output_dir=staged_run.output_dir,
            dataset="ds",
            image=Path("img.tif"),
            input_root=Path("/in"),
            process_only_layer=None,
            markers_required=False,
        )
        == "stage1"
    )


# --- (b) concurrent writers ------------------------------------------------


def _write_one(args) -> str:
    final, marker = args
    image = Image(load_synth_yeast_plate())
    image._metadata.public["Metadata_Strain"] = marker
    image.save2zarr(final)
    return marker


def test_two_concurrent_writers_produce_one_coherent_winner(tmp_path: Path) -> None:
    final = tmp_path / "p.ome.zarr"
    with mp.get_context("spawn").Pool(2) as pool:
        pool.map(_write_one, [(final, "A"), (final, "B")])
    assert valid_staged_store(final) is True
    winner = Image.load_zarr(final)._metadata.public["Metadata_Strain"]
    assert winner in {"A", "B"}
    assert list(tmp_path.glob("*.part")) == []
    assert list(tmp_path.glob("*.trash")) == []


def test_concurrent_writers_use_distinct_part_directories(tmp_path: Path) -> None:
    """A shared .part would let two writers interleave chunks and still validate."""
    final = tmp_path / "p.ome.zarr"
    parts = {ngff_.new_part_path(final).name for _ in range(2)}
    assert len(parts) == 2


# --- (c) stale part from a killed process ----------------------------------


def test_a_stale_part_is_removed_not_merged_into(tmp_path: Path) -> None:
    results = tmp_path / "results" / "ds" / "zarr"
    results.mkdir(parents=True)
    stale = results / ".p.ome.zarr.deadbeefdeadbeef.part"
    (stale / "gray" / "0").mkdir(parents=True)
    (stale / "gray" / "0" / "0.0").write_bytes(b"garbage from a killed worker")
    assert ngff_.sweep_orphan_parts(tmp_path / "results") == 1
    assert not stale.exists()


def test_a_new_write_does_not_reuse_a_stale_part(tmp_path: Path) -> None:
    image = Image(load_synth_yeast_plate())
    final = tmp_path / "p.ome.zarr"
    stale = final.parent / ".p.ome.zarr.deadbeefdeadbeef.part"
    (stale / "gray" / "0").mkdir(parents=True)
    (stale / "gray" / "0" / "0.0").write_bytes(b"garbage")
    image.save2zarr(final)
    assert valid_staged_store(final) is True
    assert (Image.load_layer_zarr(final, "gray") == image.gray[:]).all()


def test_promote_leaves_no_trash_on_success(tmp_path: Path) -> None:
    image = Image(load_synth_yeast_plate())
    final = tmp_path / "p.ome.zarr"
    image.save2zarr(final)
    image.save2zarr(final)
    assert [p.name for p in tmp_path.iterdir()] == ["p.ome.zarr"]
```

- [ ] **Step 2: Run them.**

```bash
uv run pytest tests/integration/cli/test_commit_protocol.py -v
```

Expected: all PASS.

- [ ] **Step 3: Prove case (a) can fail**

Temporarily reverse the write order in `save2zarr` — write the root `zarr.json` **first**,
before the arrays — and re-run:

```bash
uv run pytest tests/integration/cli/test_commit_protocol.py -k "interrupt or rootless" -v
```

Expected: `test_a_part_without_a_root_never_validates` still passes (it deletes the root
explicitly) but `test_interrupted_store_classifies_stage1` **fails**, because a rootful
chunk-less store now validates. Restore the order, confirm green, and paste the observed
failure into the commit body.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/cli/test_commit_protocol.py
git commit -m "test: pin the three commit-protocol cases

(a) interrupted before the root reads as absent and classifies stage1;
(b) two concurrent writers get distinct uuid .part directories and produce
one coherent winner with no leftovers; (c) a stale .part from a killed
process is removed rather than merged into. Case (a) was proven to have
teeth by reversing the write order and watching it go red."
```

---

### Task 7.2: Windows nightly lane and platform assertions

**Files:**
- Modify: `.github/workflows/run-pytest.yml` (PR lane: ensure the commit-protocol tests are
  not excluded by `-m 'not slow'`)
- Modify: `.github/workflows/run-pytest-full.yml` (nightly Windows job at lines 129–144)
- Test: `tests/unit/sdk_/test_ngff_windows.py` (create)

**Constraints specific to this task:**
- Commit-protocol tests run **in the PR lane on Linux** and **the nightly lane on Windows**.
  The spec accepts a one-day latency on a Windows-specific promote regression rather than
  promoting the whole Windows suite to PR time.
- Windows facts to assert rather than assume:
  1. no directory `fsync` (POSIX-guarded);
  2. the move-aside retries;
  3. the two-step move-aside is the only path (no single-call replace fallback);
  4. `\\?\` prefixing;
  5. **no case-only collisions** among store path segments;
  6. per-file overhead is documented, not mitigated — no test.

- [ ] **Step 1: Write the platform tests**

```python
"""Windows-specific promote behaviour, asserted rather than assumed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from phenotypic.sdk_ import ngff_


def test_no_case_only_collisions_in_store_path_segments() -> None:
    """NTFS is case-insensitive."""
    segments = [
        ngff_.OME_GROUP,
        ngff_.LABELS_GROUP,
        ngff_.OBJMAP_LABEL,
        *ngff_.SERIES_ORDER,
    ]
    assert len({s.lower() for s in segments}) == len(segments)


def test_directory_fsync_is_posix_guarded(tmp_path: Path, monkeypatch) -> None:
    """Windows cannot open a directory handle for flushing."""
    store = tmp_path / "s"
    store.mkdir()
    (store / "f").write_bytes(b"x")
    monkeypatch.setattr(os, "name", "nt")
    opened: list[str] = []
    real_open = os.open
    monkeypatch.setattr(
        os, "open", lambda p, f, *a: (opened.append(str(p)), real_open(p, f, *a))[1]
    )
    ngff_.fsync_tree(store)
    assert str(store) not in opened


def test_move_aside_retries_before_giving_up(tmp_path: Path, monkeypatch) -> None:
    """ERROR_SHARING_VIOLATION while any of ~40 files is held open."""
    calls = {"n": 0}
    real = os.replace

    def _flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError(32, "The process cannot access the file")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", _flaky)
    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir()
    ngff_._replace_with_retry(src, dst)
    assert calls["n"] == 3
    assert dst.is_dir()


def test_move_aside_eventually_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        os, "replace", lambda s, d: (_ for _ in ()).throw(OSError(32, "locked"))
    )
    with pytest.raises(OSError):
        ngff_._replace_with_retry(tmp_path / "a", tmp_path / "b")


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_long_path_prefix_is_applied(tmp_path: Path) -> None:
    assert ngff_.long_path(tmp_path).startswith("\\\\?\\")


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_a_deep_store_path_still_writes(tmp_path: Path) -> None:
    """MAX_PATH: an output root + dataset + stem + store-internal path."""
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    deep = tmp_path.joinpath(*["longish_directory_name_segment"] * 6)
    deep.mkdir(parents=True)
    store = Image(load_synth_yeast_plate()).save2zarr(deep / "p.ome.zarr")
    assert ngff_.valid_staged_store(store) is True


def test_chunk_keys_are_one_path_segment(tmp_path: Path) -> None:
    """The '.' separator is what keeps a chunk key from being four directories."""
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    chunk_dirs = [p for p in (store / "gray" / "0").iterdir() if p.is_dir()]
    assert chunk_dirs == [], "chunk keys must not nest into directories"
```

- [ ] **Step 2: Wire the CI lanes.** In `run-pytest-full.yml`'s Windows job, ensure
`tests/integration/cli/test_commit_protocol.py` and `tests/unit/sdk_/test_ngff_windows.py`
are collected. In `run-pytest.yml`, confirm neither is marked `slow`.

- [ ] **Step 3: Run on Linux**

```bash
uv run pytest tests/unit/sdk_/test_ngff_windows.py -v
```

Expected: the four platform-independent tests PASS, the three Windows-only tests SKIP.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows tests/unit/sdk_/test_ngff_windows.py
git commit -m "test: assert the six Windows consequences of the store layout

Case-only collisions, the POSIX-guarded directory fsync, the move-aside
retry, and the one-path-segment chunk key are asserted on every platform;
the \\?\ prefix and a deep-path write are Windows-only. Commit-protocol
tests run in the PR lane on Linux and the nightly lane on Windows -- the
spec accepts the one-day latency rather than promoting the whole Windows
suite to PR time."
```

---

### Task 7.3: Architectural invariant gates

**Files:**
- Test: `tests/unit/test_ome_zarr_invariants.py` (create)

**Constraints specific to this task:**
These are grep-style gates over the source tree, in the same spirit as
`tests/unit/schema/test_no_metadata_literals.py`. Each one guards an invariant that a
future edit could plausibly violate without any other test noticing.

- [ ] **Step 1: Write the gates**

```python
"""Source-tree invariants for the OME-Zarr store. Each guards a silent-failure mode."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "phenotypic"
PY = sorted(SRC.rglob("*.py"))


def _hits(pattern: str, *, allow: set[str] = frozenset()) -> list[str]:
    rx = re.compile(pattern)
    out = []
    for path in PY:
        if path.name in allow:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if rx.search(line):
                out.append(f"{path.relative_to(SRC)}:{number}: {line.strip()}")
    return out


def test_store_suffix_is_joined_in_exactly_one_place() -> None:
    assert _hits(r'\.ome\.zarr"', allow={"ngff_.py", "_io_constants.py"}) == []


def test_objmap_path_is_never_hard_coded() -> None:
    """An rgb-less store puts the label under gray."""
    assert _hits(r'rgb/labels/objmap') == []


def test_no_module_still_writes_hdf() -> None:
    """Phase 6 keeps h5py READERS for migration; only WRITE paths must be gone.

    `_image_io_handler.py` retains `_load_v2_grouped` / `_load_legacy_flat_group`
    / `_load_hdf5_for_migration`, which open `h5py.File(..., "r")` at :1172 and
    :1211. Gating on a bare `h5py.File` would be red on day one against code the
    plan deliberately keeps.
    """
    assert (
        _hits(
            r'save2hdf5|save_intermediate_layers|h5py\.File\([^)]*["\']w',
            allow={"_hdf_to_zarr.py", "hdf_.py", "_image_io_handler.py"},
        )
        == []
    )


def test_metadata_ownership_is_never_a_prefix_check() -> None:
    """CLAUDE.md: use metadata_owner_for_header, never string parsing.

    `_metadata_migration.py:210` is the one sanctioned carve-out: it is the
    centralized canonicalization helper, which is exactly where string handling
    is allowed to live.
    """
    assert (
        _hits(r'startswith\(\s*["\']Metadata_', allow={"_metadata_migration.py"}) == []
    )


def test_file_fingerprint_is_never_called_on_a_store() -> None:
    """It opens its argument as a file and raises IsADirectoryError."""
    assert _hits(r"file_fingerprint\(\s*store") == []


def test_no_recursive_glob_for_stores() -> None:
    """rglob walks INTO every store: ~400k stat calls at 10k images.

    Matches the f-string form too, so `sweep_orphan_parts` -- which lives in
    `ngff_.py` and once used exactly this pattern -- cannot exempt itself.
    """
    assert _hits(r'rglob\(\s*f?["\'][^"\']*\.ome\.zarr') == []
    assert _hits(r'rglob\(\s*f["\'][^"\']*\{STORE_SUFFIX\}') == []
```

**Four candidate gates were considered and dropped as unable to fail.** Recording them so
they are not "helpfully" re-added:

| Dropped gate | Why it could never fail |
|---|---|
| `test_no_pid_in_a_part_directory_name` | The regex needs `getpid` and `part` on one **physical** line; the only real instance (`_cli_output_manager.py:1658`) is wrapped across lines, and Phase 6 deletes it anyway. |
| `test_scale_vectors_are_never_powers_of_two` | `r'"scale":\s*\[?\s*2\s*\*\*'` matches a JSON-literal-with-Python-exponent form no implementation emits. Phase 1 Task 1.1's `test_scale_vector_comes_from_actual_shapes_not_powers_of_two` is the real guard. |
| `test_resume_state_never_lives_in_ngff_metadata` | `r'labels.*stage2\|stage2.*ome\.labels'` matches nothing plausible. Phase 3 Task 3.4's differential parity test is what actually catches this defect. |
| `test_zarr_errors_are_caught_not_propagated` (Task 1.6) | `BaseZarrError` subclasses `ValueError`, so the assertion holds with or without it in the tuple. |

- [ ] **Step 2: Run.** Any hit is a real defect; fix it in the owning phase's module, not by
relaxing the gate.

```bash
uv run pytest tests/unit/test_ome_zarr_invariants.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_ome_zarr_invariants.py
git commit -m "test: gate the OME-Zarr source-tree invariants

Five grep gates, each guarding a failure mode no other test would notice:
a hand-joined store suffix, a hard-coded rgb/labels/objmap, an HDF write
path surviving retirement, prefix-parsed metadata ownership, and
file_fingerprint or a recursive glob pointed at a store directory.

The allow lists matter: _image_io_handler.py keeps h5py READERS for
migration, and _metadata_migration.py:210 is the sanctioned string-handling
carve-out, so gating on either without an exemption is red on day one.

Four candidate gates were dropped as unable to fail -- a PID-in-.part regex
needing two tokens on one physical line, a 2**n scale-vector regex matching
a form no implementation emits, an ome.labels regex matching nothing, and
the zarr-error catch test (BaseZarrError subclasses ValueError, so it
passed either way)."
```

---

### Task 7.4: Full-suite sign-off

- [ ] **Step 1: Run everything**

```bash
uv run pytest tests -q
uv run mypy src/phenotypic
uv run ruff check src/phenotypic tests
uv run sphinx-build -W -b html docs/source docs/_build/html
uv run python docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py
```

- [ ] **Step 2: Run the GUI ledger gates** (CLAUDE.md's `gui-tutorial-capture` skill)

```bash
uv run python scripts/check_features_md.py
uv run python scripts/check_workflows_md.py
```

The GUI's user-visible chrome is unchanged by this work — tile rendering is faster, not
different — so no `FEATURES.md` / `WORKFLOWS.md` row should need adding. If a gate fails,
that means a chrome change slipped in during Phase 4 and needs the skill's full procedure.

- [ ] **Step 3: End-to-end on a real dataset**

```bash
uv run python -m phenotypic --mode full --pipeline <pipeline.json> --input <images> -o /tmp/zarr_run
uv run python -m phenotypic --mode full --pipeline <pipeline.json> --input <images> -o /tmp/zarr_run   # resume: converges immediately
uv run phenotypic-gui --root /tmp/zarr_run --port 8050
```

Confirm: the second run reports every image `complete`; the viewer renders whole-plate tiles
and colony crops; `results/<ds>/zarr/` holds one `.ome.zarr` directory per input and no
`.part` or `.trash` leftovers.

- [ ] **Step 4: Open the store in a third-party viewer**

```bash
uv run python -c "import napari, zarr; v = napari.Viewer(); v.open('/tmp/zarr_run/results/<ds>/zarr/<stem>.ome.zarr', plugin='napari-ome-zarr'); napari.run()"
```

This is the headline external claim of the design — a PhenoTypic output directory readable
without a PhenoTypic install. If it fails, the store does not conform in a way the schema
gate missed, and that is a Phase 1/2 defect.

- [ ] **Step 5: Update the spec status and commit**

Set `**Status:**` in the design doc to `Implemented` and link this plan.

```bash
git add docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md
git commit -m "docs: mark the OME-Zarr store design implemented"
```

---

## Phase 7 exit criteria

- [ ] `uv run pytest tests -q` green.
- [ ] `uv run mypy src/phenotypic` and `uv run ruff check src/phenotypic tests` clean.
- [ ] `uv run sphinx-build -W` succeeds.
- [ ] `ngff_store_geometry.py` exits 0.
- [ ] Commit-protocol case (a) demonstrated to fail under a reversed write order.
- [ ] A real run resumes to `complete` on a second invocation with no reprocessing.
- [ ] A written store opens in napari via `napari-ome-zarr`.
# Open questions raised during planning

The spec's own §10 records "none blocking". These are questions the **plan** surfaced by
resolving the spec against the actual code — either gaps the spec does not cover, or places
where the spec's statement and the code disagree. Each says what the plan currently assumes,
so implementation is not blocked on any of them; a different answer means editing the named
task.

**Status key:** `OPEN` — needs a decision. `ASSUMED` — plan proceeds on a stated
assumption. `RESOLVED` — decided, with the decision recorded.

---

## P1 — `image-label.colors` goes stale after Stage 2, breaking conformance mid-run

**Status:** RESOLVED — **option (2): emit the background entry only.** Re-graded from
data-integrity to conformance/interop first (see D12: nothing in PhenoTypic reads `colors`),
then decided. `build_image_label()` now takes **no arguments**, so it cannot depend on array
contents and cannot go stale. Applied in Phase 1 Task 1.4; the ~60 KB per-plate JSON the
spec's OQ9 budgeted for disappears with it.

Spec §2.3 requires `image-label.colors` to carry **one entry per unique label value**, and
§7 requires **every written store** to validate against `label.schema`. Spec §3.4 has
Stage 2 overwrite `labels/objmap` **in place**, without re-promoting.

Those two cannot both hold. After Stage 1 the objmap is zeros, so `colors` has exactly one
entry (background). After Stage 2 the array holds up to ~1536 distinct labels while the
group's `zarr.json` still says one colour. Any conformance check run against a mid-run store
— and the spec asks the GUI to render exactly that store — sees a `colors` list that does
not describe the array.

Three ways out, none free:

1. **Stage 2 also rewrites the label group's `zarr.json`.** Cheapest, but it makes Stage 2 a
   two-file write with no atomicity between them, in a step the spec deliberately defined as
   an intermediate.
2. **Relax `colors` to background-only** and drop the per-value requirement. Loses the
   viewer-friendly palette that motivated it, and needs a re-read of whether `label.schema`
   actually requires exhaustiveness or only well-formedness.
3. **Stage 2 promotes after all**, which contradicts locked decision #4.

**Plan currently assumes (1)**, implemented inside `write_objmap_in_place`
(Phase 3, Task 3.3). `test_stage2_drops_a_token_and_the_objmap_is_readable` and
`test_stage1_store_conforms` need a Stage-2 conformance sibling either way.

---

## P2 — `omero.window` bounds are wrong for `detect_mat`

**Status:** RESOLVED — **omit `omero` from `detect_mat` entirely.** NGFF makes `omero`
conditional and the whole-or-nothing rule is per group, so `rgb` and `gray` keep their
blocks and `detect_mat` simply has none. No wrong window can be emitted, and `detect_mat`
is a derived analysis layer no viewer has a meaningful default rendering for. Applied in
Phase 1 Task 1.4 (`build_omero` returns `{}` for `detect_mat`); supersedes spec §2.2.

Spec §2.2 fixes `max`/`end` at `2**bit_depth - 1` for every series. `detect_mat` is a
**float** detection matrix, typically in `[0, 1]` (the spec's own §10 notes it is float64
and 96 MB). A window of `[0, 65535]` over data in `[0, 1]` renders as a black image in every
viewer that honours `omero`.

Options: emit `{"min": 0, "max": 1, "start": 0, "end": 1}` for float series; or compute the
window from the actual array min/max; or omit `omero` from `detect_mat` entirely (NGFF makes
it conditional, and the whole-or-nothing rule is per-group).

**Plan currently assumes** the literal spec text (`2**bit_depth - 1` everywhere), which is
almost certainly wrong for `detect_mat`. Phase 1 Task 1.4's `build_omero` is the single
place to change. Flagged rather than silently "fixed" because it is a spec statement, not an
oversight in the plan.

---

## P3 — Changing `--pyramid-levels` between runs produces a mixed-geometry tree

**Status:** RESOLVED by **descoping the lever**. `--pyramid-levels` is not implemented; the
pyramid depth is `pyramid_level_count(h, w)`, a pure function of the level-0 shape. With no
user lever, two stores in one tree cannot disagree — so `valid_staged_store` needs no level
check, a resumed run cannot produce mixed geometry, and the tile-request crash this question
described is unreachable. `resolve_pyramid_levels` is removed from Phase 1; Phase 3 Task 3.7
is now `--durable-writes` only. A single-level store stays reachable internally via the
private `levels=` argument for builder node previews. The lever can land later as its own
change; spec §1.3 should record it as deferred.

`valid_staged_store` (§3.6) checks only **level-0** extents. A resumed run with a different
`--pyramid-levels` therefore leaves every already-written store at the old level count while
new ones use the new one. Nothing detects it, and the GUI's `select_pyramid_level` reads
`phenotypic.pyramid.levels` per store, so it will not crash — it will just serve tiles at
inconsistent resolutions.

Options: record the resolved level count in the run manifest and refuse a resume that
changes it; or add level count to `valid_staged_store` so a mismatched store reclassifies to
`stage1`; or accept mixed geometry and document it.

**Plan currently assumes** mixed geometry is accepted and undocumented, because the spec says
nothing. Phase 3 Task 3.4 is where a validity change would go; Phase 5's
`migrate_run_hdf_to_zarr` has the same question for `--njobs`-parallel conversions.

---

## P4 — Stage 2 must rewrite **every** objmap pyramid level, which the spec does not say

**Status:** ASSUMED (plan is stricter than the spec).

Spec §3.4 says Stage 2 "overwrites `labels/objmap` in place" without saying at which levels.
Rewriting only level 0 leaves levels 1..n holding Stage 1's zeros, which the GUI then serves
as a blank overlay for any zoomed-out view — silently wrong, never an error.

**Plan assumes** all levels are rewritten (`write_objmap_in_place`, Phase 1/3), and
`test_stage2_rewrites_every_pyramid_level_of_the_objmap` pins it. This should be folded back
into the spec's §3.4 rather than living only in the plan.

---

## P5 — `save_intermediate_layers` has **five** call sites, not three

**Status:** RESOLVED (plan uses the verified count).

Spec §3.1 says `save_intermediate_layers` "has three live callers in
`_image_pipeline_core.py` and two in tests". Verified against the code: that file has
**five** relevant calls — `save_intermediate_layers` at lines 1024, 1046, 1052, **and
`save2hdf5` at lines 1021 and 1042**. The two `save2hdf5` calls are the same builder-preview
path and must move together, or node previews write HDF into a zarr tree.

Phase 2 Task 2.4 covers all five. The spec's §3.1 sentence should be corrected.

---

## P6 — The builder DAG manifest's `"hdf"` key is a GUI-visible contract change

**Status:** ASSUMED.

`gui/builder/_preview_cache.py` writes a per-node manifest with a `"hdf"` key
(lines 208, 212, 217) that `_preview_cache.py:284-286` and `_preview_tiles.py:124` read
back. Renaming the per-node artifact from `base_00.h5` to `base_00.ome.zarr` changes an
on-disk contract the spec's §3.1 mention of `save_intermediate_zarr` does not discuss.

**Plan assumes** the key becomes `"store"` and `MANIFEST_VERSION` is bumped so a stale
manifest is rebuilt rather than misread (Phase 2, Task 2.4). Alternative: keep the key name
`"hdf"` to avoid the version bump — rejected, because a key named `hdf` holding a zarr path
is exactly the kind of lie that costs an afternoon later.

---

## P7 — Stage 3 deletes the Stage-2 signal only when `work_id is None`

**Status:** RESOLVED (plan preserves the code's behaviour, not the spec's sentence).

Spec §3.5 says Stage 3 "writes the completion marker and **deletes the Stage-2 token**".
In the code (`_cli_staged_workers.py:250-258`) both actions sit inside `if work_id is None:`;
on the work-id path the SLURM worker publishes and deletes instead
(`_cli_staged_slurm_worker.py:409`).

**Plan preserves the guard verbatim** (Phase 3, Task 3.3). Making the deletion unconditional
would double-delete against the SLURM worker and change resume classification — which is
precisely what Task 3.4's differential test exists to catch. The spec sentence should gain
the qualifier.

---

## P8 — `jsonschema` is not a declared dependency

**Status:** RESOLVED.

Spec §7 forbids a conformance check that skips on a missing dependency, and §6 rules out
`ome-zarr-models`. But `jsonschema` appears nowhere in `pyproject.toml` — it is available
today only transitively. A transitive dependency can vanish on any lock refresh, and the
check would then be unrunnable, which §7 says must fail rather than skip.

**Plan declares it** in the test dependency group and pins that with
`test_jsonschema_is_declared_not_transitive` (Phase 0, Task 0.1).

---

## P9 — `save_array2hdf5`'s "eight live call sites" are removed by this change

**Status:** ASSUMED.

Spec §5.4 keeps `save_array2hdf5` on the grounds that it "has eight live call sites". Those
sites are in `_image_io_handler._save_image2hdfgroup` and `save_intermediate_layers` — both
of which **this change deletes** (Phase 2 Task 2.4, Phase 6 Task 6.2). After Phase 6 the
count is zero.

**Plan keeps it anyway**, for a different and explicit reason: `tests/fixtures/legacy_hdf/_generate.py`
(Phase 5, Task 5.1) needs an HDF writer to rebuild the migration golden fixtures after the
production writer is gone. That reason is recorded in Phase 6 Task 6.1's keeper table. If the
fixtures are instead frozen as committed bytes with no regeneration path, `save_array2hdf5`
can go too — but then the fixtures can never be extended.

---

## P10 — Two GUI/SDK call sites are missing from the spec's affected-module table

**Status:** RESOLVED (plan covers them).

Spec §4.4 lists 24 files. Two real call sites are not among them:

- `sdk_/_io_constants.py:2063` — `BundleLayout.hdf_path`, the accessor
  `OutputRoot.hdf_path` delegates to. **Note the trap:** it checks `is_file()`. A copy-paste
  port to a store returns `None` for every image, silently disabling every full-res GUI
  read, with no error anywhere. Phase 2 Task 2.1 uses `is_dir()` and pins it by test.
- `gui/results_viewer/_output_root.py:1146-1152` — an `("hdf", hdf_path)` pair built for the
  output-consistency report. Phase 4 Task 4.1 ports it.

---

## P11 — Chunk and shard shapes for pyramid levels smaller than one chunk

**Status:** ASSUMED.

Spec §1.4 fixes chunks at `(1024, 1024)` and shards at `(4096, 4096)` but does not say what a
`257 × 2` level gets. Zarr rejects a chunk larger than the array in some code paths, and the
sharding codec requires exact divisibility regardless.

**Plan assumes** both are clamped to the level's own extent, with the shard then rounded down
to an exact multiple of the clamped chunk (Phase 1, Task 1.2). This means the small levels
are single-chunk, single-shard — which is what the file-count table in §1.4 already implies
(8 files per additional level, flat across plate sizes), so the assumption is consistent with
the committed validation script. Worth confirming the script's `data_files`/`metadata_files`
functions agree with the clamped policy rather than with an unclamped one.

---

## P12 — `--pyramid-levels` and `--durable-writes` are not wired into the CLI argument list

**Status:** RESOLVED — **Phase 3 Task 3.7** added.

§1.3 introduces `--pyramid-levels auto|N` and §3.7 requires `--durable-writes /
--no-durable-writes`, but neither appears in §5's interface section or anywhere else that
enumerates CLI flags, and `phenotypicCLI.py`'s option block is untouched by the spec. Both
would therefore have shipped unimplemented.

**Resolved** by adding Phase 3 Task 3.7, which creates both options, validates
`--pyramid-levels` through `resolve_pyramid_levels` so the CLI and writer cannot disagree,
makes `--durable-writes` genuinely tri-state (unset = auto-detect; a plain `is_flag` would
collapse that to "off" and silently lose the SLURM detection), rejects both on `recompile`
and `migrate`, and documents that builder node previews are always single-level and ignore
`--pyramid-levels`.

The spec should gain these two flags in its §5 interface enumeration.

---

# Round 1 — data-flow review findings (D1–D16)

Raised by an independent data-flow review that traced five flows (staged write, resume
state, metadata, GUI reads, migration) against the real code. **Every claim reproduced
below was independently re-verified in this worktree before being recorded here**; the
verification method is stated per item. Findings the review raised that I could not
reproduce are marked as such.

## D1 — Stage 3 is not idempotent under retry, and a retry can delete a colony

**Status:** RESOLVED — **option (1): keep the raw Stage-2 array outside the store.**
Stage 2 retains its raw detector output at
`.phenotypic/progress/stage2_raw/<ds>/<stem>.npy`, written *before* the token so a crash
between them just re-runs Stage 2; Stage 3 replays from it and consumes both. Restores
today's exact idempotency with no new NGFF surface. Applied in Phase 3 Tasks 3.2 and 3.3,
with `test_stage3_is_idempotent_under_retry` and a `staged_run_with_border_colony` fixture
that makes the test non-vacuous (without a border-touching colony `drop_frame_background`
returns early and a second pass is a harmless no-op).

The full analysis follows, retained because it is the reasoning behind the decision.

**Verified** by reading `abc_/_gpu_detector.py:242-249` and
`_core/_image_parts/accessors/_objmap_accessor.py:498-509`.

Today Stage 3's input is the **raw** detector output, preserved in the `.npy` sidecar
(`_cli_staged_workers.py:210`), which survives Stage 3's re-save. A Stage 3 killed and
re-run replays from the same raw input and produces an identical result, any number of
times.

The plan replaces that input with the store's own objmap (Phase 3 Task 3.3) and then
re-promotes the store over it — so the raw detector output is destroyed the moment Stage 3
first succeeds. The retry window is real: `save_image_store` lands at
`_cli_staged_workers.py:225`, but the completion marker is not written until `:251`, with
`save_overlay` (`:239`) and `PlotCoordinator.emit_image` (`:243`) in between. A timeout
anywhere in that span leaves store re-promoted, parquet written, token present, marker
absent — which `classify_staged_image` (`_cli_staged_resume.py:233`) reads as `"stage3"`.

On the second pass, `_write_object_output` runs again on already-refined labels:

```python
if self.output_kind == "instance":
    image.objmap[:] = result.astype(np.uint16)
    if self.drop_frame_background:
        image.objmap.drop_frame_background()
```

`drop_frame_background` zeroes the label owning the **plurality of border pixels**, after
`border = border[border > 0]` excludes the already-zeroed background. So on the second pass
the plurality falls to whichever **real colony** touches the frame most — and that colony is
deleted. Silently, no error, once per retry. `post_pipeline.apply` also runs twice, which is
harmless for a size filter and not for erosion, border refiners, or watershed.

This breaks the byte-identical-to-single-pass contract in `_cli/CLAUDE.md` on the first
retry, and the resume classifier cannot tell a first Stage 3 from a second.

**Options:**

1. **Keep the raw Stage-2 array outside the store**, under
   `.phenotypic/progress/stage2_raw/<ds>/<stem>.npy`, paired with the token. Restores
   today's semantics exactly, adds no NGFF surface, and is a small change. Costs the spec's
   claim that "the `.npy` sidecar format disappears" — but see D6, which refutes the *other*
   stated benefit of the in-store write, so that claim is worth re-examining anyway.
2. Add a second label image `labels/objmap_raw`, written by Stage 2 and dropped by Stage 3's
   promote. Same semantics, but more inodes and more conformance surface.
3. Have Stage 2 apply `_write_object_output` itself so Stage 3 skips it. Fixes the
   colony-deletion half but leaves `post_pipeline.apply` running twice.
4. Make Stage 3's tail atomic so the marker cannot lag the promote. Narrows the window
   without closing it.

**Recommendation: (1).** It is the only cheap option that fully restores the current
guarantee. **This is a decision for you** — it partially reverses a stated design goal.

## D2 — The per-image completion marker breaks on a store directory

**Status:** OPEN — **needs a new task. Currently uncosted in both spec and plan.**

**Verified** by reading `_cli/_cli_completion.py:29-34` and `:117-130`, and by
`grep -rn 'publish_image_success|valid_image_success|_cli_completion|SUCCESS_MARKER_VERSION'`
over the spec and the whole plan directory → **no matches**.

`publish_image_success` records `{"size": ..., "sha256": _sha256(resolved)}` per declared
artifact, and `_sha256` does `path.open("rb")` — **`IsADirectoryError` on a store**,
uncaught, so the publishing worker dies. `valid_image_success` mirrors it with
`not artifact.is_file()` → **False for every store**, so branch 1 of `classify_staged_image`
returns `"stage3"` for every already-finished image on the work-id path, forever.

Five sites declare an `"hdf"` artifact, all confirmed present:
`phenotypicCLI.py:400`, `_cli_staged_slurm_worker.py:332` and `:382`,
`_cli_process_single.py:640`, `_cli_execution_strategies.py:167`.

**It is also invisible to the Phase 3 Task 3.4 gate.** That test parameterizes
`(image_state, stage2_signal, parquet, stage3_marker)` — there is no *image-completion
marker* axis, so `valid_image_success` returns `False` in both worlds and branch 1 is never
exercised. The parity test passes while production breaks.

**Plan now assumes:** a store gets its own descriptor kind
(`{"path": ..., "kind": "store", "fingerprint": paths_fingerprint([store / "zarr.json"])}`)
with `valid_image_success` dispatching on `kind`, and `SUCCESS_MARKER_VERSION`
(`_cli_completion.py:26`, currently `1`) bumped to `2`. Added as **Phase 3 Task 3.8**, and
the Task 3.4 differential test gains a fifth artifact axis.

## D3 — `_assert_canonical_metadata` rejects real production metadata

**Status:** RESOLVED — check removed. **Verified by execution** in this worktree:

```text
'Metadata_Strain'    | member: Metadata_Strain | is_metadata_header: True
'Metadata_PlateNum'  | member: None            | is_metadata_header: True
'MyColumn'           | member: None            | is_metadata_header: False
```

`metadata_member_for_header` is a **semantic-ownership resolver**, not a format check: it
returns `None` for `Metadata_PlateNum`, a real column in this project's canonical Results
matrix. And a legitimately loaded image really can carry a bare public key — verified by
HDF round-trip, `public after: {..., 'Metadata_PlateNum': 3, 'MyColumn': 'x'}`, because
`_remap_legacy_metadata_key` (`_image_io_handler.py:100-106`) deliberately preserves unknown
bare names verbatim.

So the check as written aborts `save2zarr` for most production runs, and the review's
suggested replacement (`is_metadata_header`) still rejects `MyColumn`.

**Resolved by deleting the assertion.** The HDF writer has no equivalent check, so adding
one is a regression, not a hardening. Phase 1 Task 1.3 loses `_assert_canonical_metadata`
and its `test_non_canonical_metadata_headers_are_rejected` test; the docstring records why.

## D7 — `Metadata_ImageType` does not survive the read path

**Status:** OPEN — **a spec §7 mandated test will fail as the plan is written.**

**Verified by execution:**

```text
before ImageType: GridSection
after  ImageType: Image
```

Spec §2.1 requires `image_class` and `Metadata_ImageType` to stay distinct, and §7 requires
a round-trip test asserting both "preserved independently". Phase 2 Task 2.2 instructs
`_load_from_store` to "mirror `_load_v2_grouped`" — **and `_load_v2_grouped` loses it**. The
cause is `_image_io_handler.py:1071-1073`:

```python
for mapped, value in decoded.items():
    if mapped in target and target[mapped] is not None:
        continue
    target[mapped] = value
```

The constructor has already set `Metadata_ImageType`, so the stored value never lands.
Mirroring this inherits the bug, and `test_image_class_and_image_type_are_independent`
(Phase 2 Task 2.2) fails.

**Plan now assumes** `_load_from_store` restores the three metadata sections **verbatim**
rather than by that skip-if-present merge, and Phase 2 Task 2.2 says so explicitly. Note
this makes the store read path **more** correct than the HDF one — a deliberate divergence,
flagged rather than silently introduced. Whether the HDF loader should be fixed too is out
of scope here (it is retired in Phase 6 anyway).

## D8 — The mandated `v2_enh_gray` fixture cannot be read by the loader the plan mandates

**Status:** OPEN — **needs a budgeted change to a legacy reader.**

**Verified** by reading `_image_io_handler.py:1035-1036` (v2 loader: bare
`layers["detect_mat"]`, no fallback) against `:1100-1108` (v1-flat loader: has the
`enh_gray` fallback).

Phase 5 Task 5.1 requires a **v2-grouped** fixture carrying `enh_gray` ("mandatory, not
optional") *and* requires reusing the existing loaders ("Do not write a third HDF reader").
Those are incompatible: `_load_v2_grouped` raises `KeyError` on that fixture. Meanwhile
`valid_staged_hdf` (`_cli_staged_resume.py:81-83`) accepts `enh_gray` at
`schema_version >= 2`, so the code believes such files exist in the wild.

**Options:** add the fallback to `_load_v2_grouped` (a change to a legacy reader that must
land **before** Phase 6 retires it, and that Phase 5 does not currently budget); or make the
fixture v1-flat only, leaving the schema-2 `enh_gray` case unmigratable.

**Plan now assumes** the fallback is added, as a new step in Phase 5 Task 5.1.

## D9 — `--mode migrate`'s `metadata.csv` rewrite collides with `metadata_sha256`

**Status:** OPEN.

`deliverables/metadata.csv` is not inert provenance — its SHA-256 is load-bearing state
(`phenotypicCLI.py:276`, `:1338-1341` write `state.config["metadata_sha256"]`;
`_cli_completion.py:541-547` folds it into `finalization_input_digest`; `:391-399`
recomputes `expected_finalization` from it). Task 5.2 rewrites the file and says nothing
about the digest. Leave it → the aggregate publication marker stops validating and the next
run re-finalizes everything. Update it → the recorded digest no longer matches
`metadata.original.csv`, which is the provenance the task exists to preserve.

Separately, and answering the review's own question: `metadata.csv` **is** read after
migration. `_snapshot_metadata_csv` (`phenotypicCLI.py:241-282`) runs at the start of
`full`, `recompile`, and incremental startup; if a user passes `--metadata <original.csv>`
again after migrating, `destination.read_bytes() != payload` and the canonicalized file is
**overwritten with the raw original**, silently reverting the migration.

**Plan currently assumes** neither problem exists. Both need a step in Task 5.2.

## D4, D5 — Two GUI fingerprints go content-blind, and I mis-scoped one of them

**Status:** OPEN — Phase 4 Task 4.1 understates both.

- `_image_source_token` (`_output_root.py:1138-1178`) hashes
  `st_dev/st_ino/st_size/st_mtime_ns/st_ctime_ns` per source path. **None of those five
  moves** when a chunk inside a store is rewritten. It drives `bound_image_source_token`
  (`:649`) and `_capture_image_source_tokens` (`:405`, `:1093`) — i.e. whether the viewer's
  binding to an image's pixel source is still valid. My Task 4.1 calls it "a label the
  report renders". That is wrong; it is a staleness fingerprint and must key on
  `store / "zarr.json"`.
- `_processing_snapshot_paths` (`:886-889`) feeds `_cancellable_paths_fingerprint`, whose
  directory branch (`:832-834`) emits a constant byte and does not recurse. If the port
  yields store **directories**, `snapshot.processing_fingerprint` — and therefore
  `OutputRoot.source_fingerprint` (`:512`) — stops changing when per-image results change. I
  framed line 888 purely as a cost problem (400k stat calls); the correctness problem is
  larger. The port must enumerate each store's `zarr.json`.

**Also a spec correction:** §4.2's table says "Use `paths_fingerprint()`, which handles
directories". It handles them **by ignoring their contents** (`_io_constants.py:215-217`
emits a single sentinel byte and does not recurse). `paths_fingerprint([store])` is a
constant function of the path and would freeze the tile cache permanently. The plan already
keys on `store / "zarr.json"`, but the spec sentence must be corrected before anyone
implements from it.

## D6 — The GUI cannot see the Stage-2 objmap at all, and Task 4.3 contradicted itself

**Status:** RESOLVED — **accepted as correct behaviour, and the contradiction removed.**

With D1 decided (the raw array is retained outside the store), the in-store Stage-2 write is
purely an interop convenience, not a correctness dependency — so the GUI not seeing it costs
nothing. Root-keying the cache means it invalidates on **promotes**, which is what should
gate consumers: the completion marker, not the store's shape, and a torn mid-Stage-2 objmap
is exactly what a viewer must not be shown.

Phase 4 Task 4.3's tests now pin **both** directions —
`test_served_tile_changes_after_a_promote` and
`test_served_tile_is_unchanged_by_an_in_place_write` — so a later "fix" cannot re-introduce
per-chunk invalidation. Spec §3.5 should drop its claim that the in-store write buys "the
GUI can render a real objmap mid-run"; it does not.

With Task 4.3's fix in place: Stage 2 rewrites `objmap/*` levels; the root `zarr.json` is
untouched; `_ensure_store_layer_source_png` returns early because the cached PNG's mtime was
`os.utime`'d to that same unchanged root. So the GUI serves the **Stage-1 zeros objmap** for
the entire Stage-2 → Stage-3 window.

Nothing is corrupted — Stage 3's promote fixes it — but it refutes spec §3.5's claim that
the in-store write buys "the GUI can render a real objmap mid-run". That was one of only two
stated justifications for writing in place; D1 attaches to the other. **Together, D1 and D6
mean the in-store Stage-2 write currently buys nothing and costs idempotency.** That is the
core of the D1 decision above.

Within Phase 4 Task 4.3, `test_served_tile_changes_after_an_in_place_rewrite` asserts the
tile *does* change after an in-place write, while Step 3 implements a check guaranteeing it
does not. One of the two has to give.

## D12 — P1's stated resolution is not in the plan text

**Status:** RESOLVED as a bookkeeping error; P1 itself re-graded below.

P1 says the plan resolves it "inside `write_objmap_in_place` (Phase 3, Task 3.3)". The
actual code block there writes array levels only — no `zarr.json` rewrite, no
`build_image_label` call, which appears only at write time. P1 is **unresolved in the plan
text**, not resolved-as-stated.

The review also traced what consumes `colors`: **nothing in PhenoTypic**. The GUI colourises
via `skimage.color.label2rgb` (`gui/builder/_image_renderer.py:155-166`); neither
`load_layer_zarr` nor `_load_from_store` reads it. The only consumers are the conformance
gate and external viewers. **P1 is therefore re-graded from data-integrity to
conformance-and-interop** — still real, because third-party readability is a headline goal
of this design, but no longer the most severe item. With no internal consumer, option (2)
from P1 (background-only `colors`) costs least.

## D11, D13, D14, D15, D16 — smaller corrections, all accepted

- **D11** (`--mode process --layer objmap` leaves raw detector output published forever).
  Today the residue is Stage 1's zeros in a non-user-facing HDF; under the plan it is a
  first-class NGFF label image that napari and Vizarr will render. Task 3.5 must re-promote
  after the export or restore the zeros objmap.
- **D13** (my `rmtree(store)` in `clear_downstream_artifacts_for_stage1` rests on a
  misreading). **Verified:** that function deletes only the `.npy` sidecar and the `.json`
  marker (`_cli_staged_resume.py:314-319`) — it never unlinks an image artifact, so no
  `IsADirectoryError` is possible. Adding an `rmtree` **introduces** behaviour: at its two
  call sites it would open a window where the image is absent, whereas today the previous
  HDF survives until Stage 1's atomic replace. Reverting to "delete nothing extra".
- **D14** (the run-start `.part`/`.trash` sweep can delete a live writer's directory). The
  uuid identifies the *attempt*, not whether its process is alive, and the staged SLURM
  engine explicitly assumes stale workers can still be running — that is what
  `assert_active_epoch` exists for. Gate the sweep on age or on a lifecycle epoch recorded
  inside the `.part`.
- **D15** (`tiles.py:518` is not a live staleness site). `crop_hdf_rgb` opens with
  `del mtime_ns` and its docstring says the parameter is accepted for API compatibility
  only. Calling it one of "four traps" spends attention on a non-issue while D4 and D5 —
  the two sites that genuinely go content-blind — were absent from the list. Task 4.3's
  framing is corrected: the site still needs the zarr port, but not a staleness fix.
- **D16** (Phase 5's two recompile tests contradict each other, and one references an
  undefined `legacy_run_v2`). The intended distinction — legacy *format* vs legacy *headers*
  — is real but the fixtures do not encode it. Both tests need distinct fixtures.

## Dead code the change strands

`clear_stage2_sidecars` (`_cli_staged_orchestration.py:661-674`, called from
`phenotypicCLI.py:1590` on `--restart`) globs `results/*/objmap/*.npy` and becomes a
permanent no-op. Not a correctness hole — `clear_machine_state` on the same path wipes
`.phenotypic/`, where the new token lives — but Phase 6 must remove it.

## Corrections to my own plan text, found while verifying

- Every test snippet imported `load_synth_yeast_plate` from `phenotypic.util`. It lives in
  `phenotypic.data`. Fixed across four phase documents.
- Every test snippet used `image.metadata.public[...]`. The accessor exposes
  `by_module/get/items/keys/table/...`, not the three sections; the established test idiom
  is `image._metadata.public[...]` (`tests/unit/sdk_/test_metadata_io.py:824`). Fixed
  across five phase documents.
- `_metadata` has a **fourth** section, `private`, which the HDF writer does not persist.
  The plan's three-section model is right for storage, but the plan should say `private` is
  deliberately not stored rather than leaving it unmentioned.

## Data-flow conclusions that came back clean

Recorded because a clean trace is a result:

- **Hard-link / promote / `rmtree(trash)` does not lose data.** Link-count walk:
  link → 2; `os.replace(store, trash)` → still 2 (a rename moves a dirent, not an inode);
  `os.replace(part, store)` → still 2; `rmtree(trash)` unlinks a *name* → 1, data survives.
  A crash mid-promote leaves an orphan `.trash` holding the second link, which the sweep
  decrements. The copy fallback on `os.link` failure keeps this sound; a **symlink** fallback
  would break it, and the plan correctly does not use one.
- **Metadata collision handling already exists and is correct.**
  `_normalize_stored_metadata_items` (`_image_io_handler.py:154-189`) raises `ValueError`
  when two source keys collide on one target with different values, and coalesces when
  equal. Migration surfaces that as `report.failed` — the image is named and skipped, never
  silently merged.
- **Token consumability is complete.** The sidecar is deleted at five sites
  (`_cli_staged_workers.py:258`, `_cli_staged_strategy.py:246` and `:382`,
  `_cli_staged_slurm_worker.py:409`, `_cli_staged_resume.py:364`) and all five are covered by
  the plan's file lists.
- **`migrate_legacy_stage3_markers` still fires**, because reaching `"complete"` for a legacy
  tree goes through the branch requiring the token's *absence*, and legacy trees have no
  token. This is exactly why the token must not live in `ome.labels` — P7's reasoning holds.
- **Pyramid dtype and value integrity, and the `rgb` moveaxis round-trip**, are sound.

**One correction to P7:** it implies the local path relies on the `work_id is None` guard.
It does not — `_cli_staged_strategy.py:243-246` writes the marker and deletes the sidecar
**unconditionally**. Preserving the guard verbatim is still right, but P7 should name that
third site so a future reader does not "simplify" the guard away.

---

# Round 2 — plan review findings (B1–B11, F1–F12, G1–G7, P13–P22, S1–S8)

From an independent plan review that verified every `file:line` against the worktree, ran
the logic-validation script (exit 0), and reached the zarr 3.x docs. **Every claim acted on
below was independently re-verified here first**; the method is stated per item.

## Verified and fixed

| | Finding | Verification | Disposition |
|---|---|---|---|
| **B1** | `shard_shape_for` returned `(3072, 2048)` for a 4000×3000 level — failing three of its own tests and making spec §1.4's file counts wrong | Arithmetic, then a script re-deriving shard-file counts at every level of three plate sizes against `ngff_store_geometry.py` | **Fixed.** Clamping removed: shard is `(C, 4096, 4096)` unless an extent is below one chunk, where `chunk == shard == extent`. Now agrees with the script at every level. Closes **P11/P13** |
| **B2** | Conformance harness validated `attributes["ome"]`; all three schemas are rooted at `attributes` | Downloaded and parsed all three: `required: ["ome"]`, `description: "The zarr.json attributes key"` | **Fixed.** Validates `payload["attributes"]` |
| **B3** | `_version.schema` not vendored; its `$ref` is remote, and `Unresolvable` is not a `ValidationError` | Parsed all three — each has exactly one remote `$ref` to it; fetched it (280 bytes) | **Fixed.** Vendored as a fourth file, resolved through a `referencing.Registry` |
| **B4** | `promote_store`'s `finally` deleted the previous store when the second rename failed — data loss the HDF path never had | Read the plan's own code | **Fixed.** Rolls back `trash → final` and re-raises; `rmtree` only on success. Closes **P14** |
| **B5** | Check-then-act made a concurrent promote raise, so "duplicate execution is benign" was not restored | Traced both interleavings | **Fixed.** The whole `exists → move-aside → replace` sequence is inside one retry loop that re-evaluates each attempt. Closes **P15** |
| **B6** | A uuid says nothing about liveness; a run-start sweep would `rmtree` a sibling SLURM task's in-flight `.part` | Read `assert_active_epoch`'s existence as proof the engine assumes live stale workers | **Fixed.** Controller-only, plus a `SWEEP_MIN_AGE_SECONDS` age guard, plus a bounded non-recursive scan. Closes **P16** |
| **B7** | A Stage-3 re-promote with unchanged metadata yields a byte-identical root, so the LRU key never moves and the "regenerated" PNG comes from the old array | Read `tiles.py:290-292`'s cache key | **Fixed.** Token is bytes **plus** `st_mtime_ns`. The in-place half of B7 dissolved when **D6** was decided. Closes **P17** |
| **B8** | Two Phase 7 gates were red on day one against code the plan keeps | `grep` → one `startswith("Metadata_")` at `_metadata_migration.py:210`; `h5py.File` at `_image_io_handler.py:1172,1211` | **Fixed.** Allow lists corrected; the HDF gate now matches write modes only |
| **B9** | `read_measurements()["ObjectLabel"]` — the column is `Object_Label` | `str(OBJECT.LABEL)` → `Object_Label` | **Fixed.** Resolved through `phenotypic.schema.OBJECT` so a rename cannot silently `KeyError` the most load-bearing test in the plan |
| **B10** | `write_objmap_in_place` was defined in Phase 3 but imported by Phase 4, contradicting the parallel DAG | Read both phase docs | **Fixed.** Moved to Phase 1 Task 1.6 with its own two tests |
| **B11** | Phase 6's `load_hdf5` rename named two call sites; Phase 5 has a third | Read `phase-5-migrate.md:113` | **Fixed.** All three named, plus a post-rename grep |
| **F1** | Spec §3.6's "none of zarr's error types are `ValueError` subclasses" is **inverted** — `BaseZarrError` inherits directly from `ValueError` | zarr readthedocs error hierarchy | **Fixed.** Tuple reduced to `(OSError, KeyError, TypeError, ValueError)`; the tautological test dropped; the spec's claim corrected in place. Closes **S2** |
| **F3** | `resolve_worker_count` reads only `SLURM_CPUS_PER_TASK`, never `SLURM_JOB_ID` | `_cli_utils.py:65-72` | **Fixed.** "exactly as" replaced with what the function actually does |
| **F5** | `save_image_hdf` has **three** callers, not two | `grep` → `_cli_process_single.py:183` | **Fixed**, plus the by-name monkeypatch at `test_staged_gpu_local.py:742` |
| **F6** | `strict_writer` and `swmr_reader` have **zero** call sites; the keeper justification was borrowed | `grep` over `src/` and `tests/` | **Fixed.** Kept for a stated reason (symmetric public surface), not a false one |
| **F7** | "977 lines total" counts a file the same task modifies | `wc -l` → 345 + 534 = 879 | **Fixed** |
| **F8** | `$defs/image-label` has **no `required` list** — `colors` is optional and nothing requires exhaustiveness | Parsed `label.schema` | **Confirmed.** This retroactively justifies the **P1** decision on schema grounds, not just cost grounds |
| **F9** | `$defs/omero` requires only `channels`; the channel item has no `required` list and `color` has no pattern | Parsed `image.schema` | **Recorded.** Emitting the full block stays PhenoTypic policy; `window`-if-present *is* schema-enforced, so `test_partial_omero_is_rejected` keeps its teeth |
| **G6** | Durability logging and the sweep were wired into the staged strategy only | Read Task 3.5 | **Fixed.** Both move to shared run-setup so a plain `--mode full` gets them. Closes **P21** |
| **G7** | **32** existing test files reference the removed HDF surface; the plan named 8, and `tests/gui` is in `testpaths` | `grep -rl` over `tests/`; `pyproject.toml:200` | **Fixed.** Full inventory in the README, assigned per phase; Phase 4's commands and exit criteria now run `tests/gui`. Closes **P20** |

## Still open

| | Question | Why it is still open |
|---|---|---|
| **G1/P19** | `build_multiscales(resolution=...)` is normative in spec §2.2 but **no caller passes it**, so the projection ships dead — and the hard-coded `"micrometer"` is a fabrication (TIFF carries `ResolutionUnit`, usually inch or cm) | Either wire it (read the tags from `metadata.imported`, handle the unit and the missing-tag case) or delete the parameter. Shipping an untested branch is the one option to reject |
| **G2/P18** | Spec §2.4's OME-XML failure fallback ("the consecutive-integer form") is not what Task 2.2 does — it keeps named groups and drops `series`, which is *less* conformant than either form | Recommend making an XML build failure **fatal**: it is pure string formatting over already-validated data, so the fallback exists for a path that cannot realistically happen, and removing it also removes the `_ome_xml_modules` seam that exists only to test it |
| **G3/P22** | `long_path` is applied at 3 of ~8 filesystem entry points — not in `_write_group_json`, `promote_store`, `fsync_tree`, `read_root_attributes`, or `sweep_orphan_parts`, i.e. most paths that actually approach `MAX_PATH` | Apply uniformly behind a helper so a new site cannot forget, or state that only array I/O is long-path-safe |
| **G4** | Spec §3.2 step 1 ("`rmtree` any pre-existing `.part` for this stem") is dropped; the plan relies on a fresh uuid | The plan is right and the spec is wrong — a stem-wide `rmtree` is exactly the sibling-eating behaviour **B6** guards against — but the divergence should be recorded as a deviation in the spec |
| **G5** | Nothing asserts the `"."` chunk-key separator is uniform **store-wide**; only `array_create_kwargs`'s return value is unit-tested and Task 7.2 checks `gray/0` | Needs one test walking every level of every series of a written store |
| **D9** | `--mode migrate`'s `metadata.csv` rewrite vs `metadata_sha256` | Carried from round 1 |
| **D10** | `sdk_/_metadata_migration.py`'s ~2,500 lines of HDF-target machinery are uncosted | Carried from round 1 |

## Adopted simplifications

**S1** (delete the clamping) — done via B1. **S2** (reduce the exception tuple) — done via
F1. **S3** (errno-discriminating retry) — done; `_is_retryable` gates on Windows
`ERROR_SHARING_VIOLATION`/`LOCK_VIOLATION` and POSIX `ENOTEMPTY`/`ENOENT`/`EEXIST`, so a
genuine `ENOSPC` fails fast instead of burning 3.1 s per image. **S4** (collapse the two
durability functions) — done via `_resolve_durability`. **S6** (drop four tautological
gates) — done, with the reasons recorded so they are not re-added.

**Not yet adopted:** **S5** (a `store_writer` context manager that makes the write-order
invariant unforgeable — genuinely attractive, but it restructures three callers and is worth
its own pass), **S7** (hoist `ngff_.py`'s function-local imports), **S8** (deduplicate the
two identical rootless-store assertions in Tasks 1.6 and 7.1).
