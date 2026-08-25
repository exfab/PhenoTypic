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
quartet. The CLI's staged-GPU engine keeps its three-stage shape. Stage 2 **does not write into the
store** — only the final store needs third-party interop, and an in-store write would be
visible to the uncached crop route as raw pre-`drop_frame_background` labels. Its resume
state — a consumable token plus the retained raw detector output Stage 3 replays from —
moves under `.phenotypic/progress/`, where the rest of the run's resume state already lives. The GUI reads pyramid levels
instead of decoding whole layers.
Legacy `.h5` runs are converted by an explicit `--mode migrate`, which also absorbs the
metadata-schema header migration.

**Tech Stack:** Python 3.11–3.12, `zarr>=3.0` (Zarr format v3 + sharding codec + zstd),
`numpy`, `jsonschema` + `xmlschema` (NGFF and OME-XML conformance, test-time),
`h5py` (migration read path only),
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
- `coordinateTransformations.scale` records the repeated **2x sampling factor**,
  independently saturating each spatial axis after it reaches one pixel; channel axes
  remain scale 1. The downsampled level is translated to its block center by
  `(scale - 1) / 2` on each spatial axis (and translation 0 on channel or saturated
  axes). It is never derived from the stored shape ratio, which diverges for odd
  extents.
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
  `2**bit_depth - 1`. `rgb` emits three channels.
- **`omero` is omitted from every FLOAT series and from label groups** — keyed on **dtype**,
  not on the series name. Both `gray` and `detect_mat` are float (verified: `gray` is
  `float32` in `[0.545, 0.955]` while `bit_depth` is 8), so a bit-depth window would render
  them near-black in any viewer honouring `omero` — and `gray` is the **primary series in
  every rgb-less store**. In practice `rgb` is the only series carrying a block, and an
  rgb-less store carries none; §2.5 makes `omero` optional and the whole-or-nothing rule is
  per group, so this is conformant. Supersedes spec §2.2. See OPEN-QUESTIONS **P2** and
  **ALGO-2**.
  **Deferred (user ruling):** making these layers render — by integer conversion or a
  range-derived window — waits on data about the effect on analysis quality. NGFF mandates
  integer pixels only for **label** images (§2.6), so nothing is being worked around.
- **`image-label` is always emitted** (the NGFF `label.schema` requires it even though the
  prose says SHOULD), with `colors` carrying **only** the transparent background entry
  `{"label-value": 0, "rgba": [0,0,0,0]}`. This supersedes the spec's §2.3 per-value
  palette: `$defs/image-label` carries no `required` list, so `colors` is optional and a
  background-only entry conforms. Nothing in PhenoTypic reads `colors` (the GUI uses
  `skimage.color.label2rgb`), only the conformance gate and external viewers — and those fall
  back to their own palette. A per-value list is a function of the array contents and can go
  stale; a background-only one cannot, and it drops the ~60 KB per-plate JSON §2.3 budgeted
  for. See OPEN-QUESTIONS **P1**. `properties` is never emitted.

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
| `src/phenotypic/_cli/_cli_staged_{workers,strategy,controller,slurm_worker,orchestration}.py` | Store paths, raw-`.npy` + token lifecycle, Stage-3 replay | 3 |
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

Phases 3 and 4 are independent of one another and may be executed in parallel by separate
agents. **Phase 5 is only partly independent of Phase 3**: Task 5.1 needs Task 3.4's
`staged_store_matches_work_id`, and Tasks 5.2/5.6/5.7 need Task 3.8's `kind`-tagged marker
descriptors, so only Tasks 5.3 and 5.4 may start before Phase 3 reaches those tasks.
`phase-5-migrate.md` carries the edge table (ledger MIG-5 / GEN-23 / SIMP-19). **Phase 5 must land before Phase 6** — migration reads legacy HDF, and
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
| 5 | [`phase-5-migrate.md`](phase-5-migrate.md) | 6 live + 1 tombstone (5.5 **cut**; 5.6, 5.7 added) |
| 6 | [`phase-6-retirement.md`](phase-6-retirement.md) | 4 (6.3a folded into 6.4 — ledger SIMP-13) |
| 7 | [`phase-7-verification.md`](phase-7-verification.md) | 4 (7.3a folded into `assert_store_conforms` — ledger SIMP-12) |
| 8 | [`phase-8-review-fixes.md`](phase-8-review-fixes.md) | 4 review repairs |

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
| 2 | `tests/unit/core/test_image_pipeline.py`, `test_delta_intermediates.py`, `test_full_layers_intermediates.py` | **Task 2.4** — they assert on the `base_00.h5` / `NN_<Op>.h5` artifacts the five `_image_pipeline_core.py` call sites write; they follow `save_intermediate_zarr` |
| 2 | `tests/gui/builder/{test_preview_cache,test_preview_compute_scope,test_preview_tile_blueprint}.py` | **Task 2.4** owns the manifest-key half (also on the Phase 4 row — Task 4.4 owns only the rendering half) |
| 2 | `tests/unit/sdk_/test_io_constants.py`, `test_bundle_layout.py` | Extend (already named in Task 2.1) |
| 3 | `tests/unit/cli/test_staged_resume.py`, `test_staged_controller.py`, `test_cli_v2.py`, `tests/integration/cli/test_cli_hdf_output.py` | Port; `test_cli_hdf_output.py` becomes `test_cli_store_output.py` wholesale |
| 3 | `tests/integration/cli/test_staged_gpu_local.py` (969 lines) | **Task 3.5** — the only file in this change that becomes an **`ImportError` at collection** (`:19` imports `_cli_sidecar`, which Task 3.5 deletes). Task 3.3 additionally defeats its `save_image_hdf` name-monkeypatch at `:742` and changes the message its `pytest.raises` matches at `:746` |
| 3 | `tests/unit/test_docs_staged_cli.py` | **Task 3.5** — asserts `"sidecar"` appears in root `CLAUDE.md` and `docs/source/how_to/pages/gpu_detection_setup.md`, both of which Task 3.5 rewrites |
| 4 | `tests/gui/results_viewer/{test_output_root,test_output_discovery_contracts,test_mutation_guard}.py`, `tests/unit/gui/results_viewer/test_output_root.py` | **Task 4.1** |
| 4 | `tests/gui/_shared/test_tiles.py`, `colony_view/{test_cropper,test_grid}.py` | **Task 4.2** |
| 4 | `tests/gui/results_viewer/test_tile_routes.py`, `colony_view/test_crop_routes.py` | **Task 4.3** — both carry the content-changes-under-one-path assertions |
| 4 | `tests/gui/builder/{test_preview_cache,test_preview_compute_scope,test_preview_tile_blueprint}.py` | **Task 4.4**, rendering half only; the manifest half is Phase 2 Task 2.4 (see above) |
| 5 | `tests/migration/test_metadata_schema_migration.py`, `tests/unit/cli/{test_cli_recompile,test_cli_recompile_slurm}.py`, `tests/unit/sdk_/test_metadata_io.py` | Port to `--mode migrate` |
| 5 | `tests/unit/cli/test_cli_recompile_metadata_migration_slurm.py` (2121 lines), `tests/unit/schema/test_no_metadata_literals.py` | **Task 5.4** — the first loses its subject entirely (26 import sites into two deleted modules) and is deleted with them; the second holds an allowlist entry keyed on the first file's path |
| 6 | `tests/unit/core/test_image_hdf_roundtrip.py`, `test_load_layer_hdf5.py`, `test_image_dtype_conversion.py`, `tests/unit/test_fixtures.py` | **Task 6.2**, not Phase 2 — see the ownership note below |
| 6 | `tests/unit/sdk_/test_hdf_open_recovery.py` | Must keep passing **unchanged** — it is what pins the keeper list |

Every phase's exit criteria must run the files it owns, not only `tests/unit/<area>`.

> **Ownership corrected (missing-owner review, 2026-08-19).** Four rows above changed hands.
>
> - **`test_image_hdf_roundtrip.py` and `test_load_layer_hdf5.py` move from Phase 2 to
>   Phase 6.** They were double-claimed: this table assigned them to Phase 2 while
>   [`phase-6-retirement.md`](phase-6-retirement.md) Task 6.2 rewrites the first as a removal
>   guard and **deletes** the second. Phase 6 wins, and the plan documents already agree with
>   Phase 6 rather than with this table — `phase-2-image-io.md` never mentions
>   `test_load_layer_hdf5.py` at all, and mentions `test_image_hdf_roundtrip.py` exactly once,
>   in an **exit criterion requiring it to stay green**. That is the correct shape: Phase 2
>   adds `save2zarr` beside `save2hdf5` and removes nothing, so there is nothing in Phase 2
>   for these two to be ported *to*, and any port done there would be deleted four phases
>   later.
> - **`test_image_dtype_conversion.py` moves from Phase 2 to Phase 6** for the same reason:
>   its only hit is `:590-592`, a direct `save2hdf5` → `load_hdf5` round-trip, and both names
>   survive Phase 2 by design.
> - **`tests/unit/test_fixtures.py` moves to Phase 6 and is nearly a no-op.** Its only hit is
>   the `temp_hdf5_file` fixture at `:135-146`, and `grep -rn "temp_hdf5_file" tests/` returns
>   **only its own definition** — nothing consumes it. Delete the dead fixture with the API it
>   was written for; do not port it.
> - **Phase 4's twelve files are now split across Tasks 4.1–4.4**, and Phase 3's and Phase 5's
>   rows name their owning task. Before this review, not one of the twelve appeared in any
>   Task 4.x `Files:` list, so no agent was authorized to touch them.

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
  reads it, and a per-value palette is a function of the array contents, which a
  background-only one is not.
- **P2** — `omero` is omitted from **every float series** (`gray` and `detect_mat` alike),
  not from `detect_mat` by name; a `2**bit_depth - 1` window over a float layer in `[0,1]`
  renders solid black, and `gray` has the identical dtype and range. Keyed on dtype so the
  deferred integer conversion would restore the block automatically.
- **P3** — `--pyramid-levels` is descoped. Depth is derived from shape, so mixed geometry
  is unreachable.

- **D9** — `deliverables/metadata.csv` is **never rewritten** (user ruling, ledger FLOW-4).
  It is immutable input provenance; migration emits `metadata.canonical.csv` beside it, and
  no `metadata.original.csv` is created.
- **D10** — `_metadata_migration.py`'s `"hdf"` `TargetKind` is **retained**, and the reason is
  that it stays *reachable* for legacy trees, not that it goes harmlessly empty (ledger
  FLOW-8, corrected by FLOW-32/MIG-21: `keep_source=True` is the default, so the retained
  `.h5` files are still enumerated). Recorded in Task 6.4; `--mode migrate` pass 1 skips an
  `.h5` whose stem already has a valid store.

**Nothing is still undecided.** Everything above is settled — by a user ruling, by the
precedence table, or by evidence. (An earlier draft's "still undecided, non-blocking" list
named D9 and D10, both of which had been settled a round earlier; ledger GEN-28 / SIMP-19.)
