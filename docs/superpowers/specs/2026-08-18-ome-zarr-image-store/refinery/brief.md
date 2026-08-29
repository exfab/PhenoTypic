# Context brief — OME-Zarr per-image store

Written at round 0 so reviewers do not each re-pay discovery. **Every
`file:line` below was verified in this worktree.** Open source only to check a
specific claim.

- **Worktree:** `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store`
- **Spec:** `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md` (837 lines)
- **Plan:** `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/` — `README.md` +
  `phase-0`…`phase-7` + `OPEN-QUESTIONS.md`
- **Logic validation:** `docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py`
  — runs clean (`uv run python <path>` → exit 0, "All store-geometry claims hold")

## What the change is

Replace the per-image HDF5 file with an OME-Zarr store (NGFF 0.5 / Zarr v3):

```text
results/<dataset>/hdf/<stem>.h5   ->   results/<dataset>/zarr/<stem>.ome.zarr/
```

Image layers become named sibling multiscale series (`rgb`, `gray`,
`detect_mat`); `objmap` becomes an NGFF label image nested under the primary
series; all PhenoTypic state moves into `attributes.phenotypic` in the root
`zarr.json`. Legacy trees convert via a new `--mode migrate`. Python floor
rises to 3.11. The HDF *write* path is removed; private HDF *readers* survive
for migration.

**Spec anchors** (required by the refinery): Objective = the spec's `## Summary`
+ `## Locked decisions`; Non-goals = `## 8. Non-goals`; NFR line = §1.4
("Write-buffer cost is not a constraint… `--njobs` sizing is governed by
processing, exactly as it is today, and this change does not move it") and
OQ10. There is **no stated latency or throughput requirement**, so performance
concerns without a spec anchor are precedence tier 8.

## Architecture map — the five surfaces this touches

### 1. Image model (`src/phenotypic/_core/_image_parts/`)

`_image_io_handler.py` (1377 lines) — the HDF quartet being replaced:

| Symbol | Line | Fate |
|---|---|---|
| `_save_image2hdfgroup` | 751 | deleted (Phase 6) |
| `_save_hdf5_metadata` | 846 | deleted |
| `save2hdf5` | 871 | deleted; `save2zarr` replaces |
| `save_intermediate_layers` | 905 | replaced by `save_intermediate_zarr` (Phase 2) |
| `_load_from_hdf5_group` | 971 | **kept private** (migration) |
| `_load_v2_grouped` | 984 | **kept private** (migration) |
| `_load_legacy_flat_group` | 1079 | **kept private** (migration) |
| `load_hdf5` | 1155 | renamed `_load_hdf5_for_migration` |
| `load_layer_hdf5` | 1194 | deleted |

Two verified traps in the read path:

- **`_load_v2_grouped` drops `Metadata_ImageType`.** `:1071-1073` skips any key
  the constructor already set. Verified by execution: `GridSection` in →
  `Image` out. Spec §7 mandates a test asserting it survives.
- **`_load_v2_grouped` has no `enh_gray` fallback** (bare `layers["detect_mat"]`
  at `:1035-1036`); only `_load_legacy_flat_group` does (`:1100-1108`). But
  `valid_staged_hdf` accepts `enh_gray` at `schema_version >= 2`
  (`_cli_staged_resume.py:81-83`), so schema-2 files carrying it exist.

`_grid_image_handler.py` — `_save_image2hdfgroup` :464, `_load_from_hdf5_group`
:509. Grid state = `nrows`, `ncols`, serialized `grid_finder`.

**Metadata API:** sections are `image._metadata.{protected,public,imported,private}`
— **four**, and `private` is *not* persisted by the HDF writer. The public
`image.metadata` accessor exposes `by_module/get/items/keys/table`, **not** the
sections; the established test idiom is `image._metadata.public[...]`
(`tests/unit/sdk_/test_metadata_io.py:824`).

**Metadata canonicality is NOT a write-time invariant.** Verified by execution:

```text
'Metadata_Strain'    | metadata_member_for_header: Metadata_Strain | is_metadata_header: True
'Metadata_PlateNum'  | metadata_member_for_header: None            | is_metadata_header: True
'MyColumn'           | metadata_member_for_header: None            | is_metadata_header: False
```

`Metadata_PlateNum` is a real column in this project's canonical Results matrix,
and `_remap_legacy_metadata_key` (`:100-106`) deliberately preserves unknown
bare names verbatim. An HDF round-trip really does yield
`public: {..., 'Metadata_PlateNum': 3, 'MyColumn': 'x'}`.

Synthetic fixture: `from phenotypic.data import load_synth_yeast_plate`
(**not** `phenotypic.util`).

### 2. CLI staged-GPU engine (`src/phenotypic/_cli/`)

Three stages, per image: Stage 1 preprocess → Stage 2 GPU detect → Stage 3
post-ops + measure.

- `_cli_staged_workers.py`: `stage1_preprocess_core` :99, `stage2_detect_core`
  :139, `ensure_staged_overlay` :168, `stage3_merge_measure_core` :193.
  `save_image_hdf` calls at **:125** and **:225** (with
  `root_attributes=` at :129/:229). **`if work_id is None:` at :249** guards
  both `write_stage3_completion_marker` (:251) and `delete_sidecar` (:258).
- `_cli_output_manager.py`: `save_image_hdf` :1633, `.{pid}.part` :1655-1657,
  post-write `h5py.File(tmp,"r+")` patch :1666, `save_image_layers`
  (deprecated) :1688.
- `_cli_process_single.py:183` — a **third** `save_image_hdf` caller, no
  `root_attributes`.
- `_cli_sidecar.py` — the `.npy` objmap sidecar. Deleted at **five** sites:
  `_cli_staged_workers.py:258`, `_cli_staged_strategy.py:246` and `:382`,
  `_cli_staged_slurm_worker.py:409`, `_cli_staged_resume.py:364`.
- `_cli_staged_resume.py` — `valid_staged_hdf` :69 (`enh_gray` :82, exception
  set `(OSError, TypeError, ValueError)` :95), `staged_hdf_matches_work_id`
  :99, `classify_staged_image` **:167**, `migrate_legacy_stage3_markers` :287,
  `clear_downstream_artifacts_for_stage1` :314 (deletes only the sidecar and
  the marker — **no image artifact**), `reconcile_stage3_publications` :322.
- **`_cli_completion.py`** — the per-image completion marker, and the surface
  the spec and the plan both originally missed. `SUCCESS_MARKER_VERSION = 1`
  :26; `_sha256` :29-34 does `path.open("rb")`; `valid_image_success` :117-130
  requires `artifact.is_file()`. **Both fail on a directory.** Five sites
  declare an `"hdf"` artifact: `phenotypicCLI.py:400`,
  `_cli_staged_slurm_worker.py:332` and `:382`, `_cli_process_single.py:640`,
  `_cli_execution_strategies.py:167`.
- `_cli_directory_scanner.py`: `scan_hdf_outputs` :173, glob :218 (with an
  AppleDouble dotfile guard).
- `phenotypicCLI.py`: `--mode` choices :943, mode validation :1217-1244,
  `_snapshot_metadata_csv` :241-282 (writes `metadata_sha256` at :276).

`abc_/_gpu_detector.py:242-249` — `_write_object_output`. For
`output_kind == "instance"` it assigns the objmap, then
`drop_frame_background()`, then `relabel()`. **`drop_frame_background`
(`_core/_image_parts/accessors/_objmap_accessor.py:498-509`) zeroes the label
owning the plurality of border pixels, after excluding already-zeroed
background** — so a second pass over already-refined labels deletes a real
colony. For `"semantic"` the result is a boolean mask routed through
`image.objmask[:]`.

### 3. GUI read paths (`src/phenotypic/gui/`)

- `_shared/tiles.py`: `_load_hdf_layer_rgb` :291 (**`@lru_cache` on
  `(path, mtime_ns, layer)`**, :290-292), `crop_hdf_rgb` :349 (opens with
  `del mtime_ns` at :386 — its mtime parameter is **discarded**),
  `_crop_hdf_layer_window` :396, caller :509-518, `__all__` :1155.
- `results_viewer/_tile_routes.py`: `_ensure_hdf_layer_source_png` :462-477 —
  `stat` compare :466/:469, `file_fingerprint(h5)` :473, `os.utime` :477.
- `results_viewer/_output_root.py`: `hdf_path` :494, `:630`,
  `rglob("*.h5")` :888, `_image_source_token` :1138-1178 (hashes
  `st_dev/st_ino/st_size/st_mtime_ns/st_ctime_ns` — **none of which move when a
  nested chunk is rewritten**), the `("hdf", …)` pair :1146-1152.
- `builder/_preview_tiles.py` :50-78, :124-128; `builder/_preview_cache.py`
  :160-170 (h5py class probe), :208/:212/:217 (manifest `"hdf"` key), :284-286.
- **`paths_fingerprint` (`sdk_/_io_constants.py:182`) does NOT hash directory
  contents** — `:214-217` emits the name plus one `b"\x02"` marker and
  `continue`s. `paths_fingerprint([store_dir])` is a constant function of the
  path. The spec's §4.2 wording ("handles directories") is misleading.

### 4. sdk\_ (`src/phenotypic/sdk_/`)

- `_io_constants.py` (2225 lines): `file_fingerprint` :166, `paths_fingerprint`
  :182, `DIR_HDF` :656, `dataset_hdf_dir` :1447, `HdfAttr` :1886,
  `load_image_from_hdf` :1936, **`BundleLayout.hdf_path` :2063 — uses
  `is_file()`**, which returns `None` for every store if copy-pasted.
- `hdf_.py` (1984 lines): `_open_hdf_with_recovery` :34 (the retry-with-backoff
  shape), `save_array2hdf5` :493 (**eight** live call sites, all inside the two
  writers being deleted), and ~1,463 lines of dead DataFrame layer.
  `strict_writer` :314 and `swmr_reader` :335 have **zero** call sites;
  `safe_writer` :252 and `swmr_writer` :277 are used by
  `tests/unit/sdk_/test_hdf_open_recovery.py:104,141`.
- `_metadata_migration.py` (~2500 lines) — **HDF-shaped throughout**:
  `TargetKind` includes `"hdf"` :44, targets built from `dataset_root/"hdf"`
  :797, hdf-specific rollback/receipts :1601-1604, :1792-1796, :1878-1885,
  :2415-2461. `:210` holds the **one sanctioned** `startswith("Metadata_")`.

### 5. Packaging and CI

`pyproject.toml`: `requires-python = ">=3.10, <3.13"` :25, 3.10 classifier :32,
`extend-exclude` :260, **`testpaths = ["tests/unit","tests/smoke","tests/integration","tests/gui"]` :200**.
`jsonschema` is present at `uv.lock:2014` but **absent from `pyproject.toml`** —
transitive only. CI: `run-pytest.yml:110`, `run-pytest-full.yml:46` (+ Windows
job :129), `package-integrity.ci.yml:44`, `publish_to_pypi.yml:20`.

**zarr is NOT installed in this worktree** (`ModuleNotFoundError`). Verify zarr
API claims against docs, not by import.

## External references (verified this session)

Schemas downloaded and parsed. **All three are rooted at the `attributes`
object**, not at `ome`:

```json
{"$id": ".../image.schema", "description": "The zarr.json attributes key",
 "type": "object", "properties": {"ome": {…}}, "required": ["ome"]}
```

- `ome.schema` → `properties.ome.required == ["series", "version"]`
- `label.schema` → `properties.ome.required == ["image-label", "version"]`,
  but **`$defs/image-label` has NO `required` list** — `colors` is optional and
  nothing requires one entry per unique label value
- `image.schema` → `$defs/omero.required == ["channels"]`; the channel item has
  **no `required` list**; `color` is an unconstrained string; only `window`, if
  present, requires `["start","min","end","max"]`
- All three carry exactly one **remote** `$ref` to `_version.schema`
  (`{"type":"string","enum":["0.5"]}`). `jsonschema` >= 4.18 does not fetch
  remote refs — it raises `referencing.exceptions.Unresolvable`, which is not a
  `ValidationError`.

zarr 3.x (from the docs, not import): `create_array(store, *, shape, dtype,
chunks, shards, compressors, chunk_key_encoding, dimension_names, …)` — all
real; `compressors` is plural; `from zarr.codecs import ZstdCodec` correct;
`zarr.errors.BaseZarrError` **inherits from `ValueError`** (so the spec's §3.6
argument is inverted). The dict form of `chunk_key_encoding` is the one symbol
not directly confirmed.

## Project conventions that bind this plan

From `CLAUDE.md` and `src/phenotypic/_cli/CLAUDE.md`:

- **`uv` only** — never bare `python`/`pip`. `uv run ruff check --fix` must be
  given **explicit paths**.
- Google-style docstrings; **runnable** doctests using `load_synth_yeast_plate()`.
- Metadata ownership via `metadata_owner_for_header()` / `metadata_member_for_header()`
  — never `startswith("Metadata_")`, prefix splitting, or category comparison.
- Operations are keyword-only pydantic models; `.apply()`, never `__call__`.
- Vendored reference sources under `docs/superpowers/**/refs` are **read-only**
  and byte-identical to upstream.
- Staged-GPU runs must be **byte-identical to a single-pass run**.
- No SLURM sidecar jobs beside an active array; route ancillary work through
  reserved array trigger entries.
- GUI chrome changes must keep `FEATURES.md` / `WORKFLOWS.md` and the tutorial
  screenshots in sync (`gui-tutorial-capture` skill).

## Test inventory

**32 files** reference the HDF surface being removed
(`grep -rlE 'save2hdf5|load_hdf5|load_layer_hdf5|save_intermediate_layers|dataset_hdf_dir|save_image_hdf|hdf_path|\.h5' tests/`).
Twelve are under `tests/gui`, which **is** in `testpaths`. The plan's README
carries the full per-phase assignment.

## Prior review rounds — do not re-litigate

`OPEN-QUESTIONS.md` holds two merged rounds:

- **Round 1 (data-flow):** D1–D16. Decided by the user: **D1** (raw Stage-2
  array retained at `.phenotypic/progress/stage2_raw/`), **P1**
  (`image-label.colors` background-only), **P2** (`omero` omitted from
  `detect_mat`), **P3** (`--pyramid-levels` descoped; depth derived from shape).
  **D6** resolved (the GUI deliberately does not see Stage 2's in-place write).
- **Round 2 (plan review):** B1–B11 all fixed, F1–F12, G1–G7, P13–P22, S1–S8.

**Still open, and the most useful targets for this round:** G1/P19 (dead
`resolution=` projection), G2/P18 (OME-XML failure fallback), G3/P22
(`long_path` at 3 of ~8 entry points), G5 (no store-wide chunk-key uniformity
test), D9 (`metadata.csv` rewrite vs `metadata_sha256`), D10
(`_metadata_migration.py`'s HDF-target machinery uncosted), S5/S7/S8
(unadopted simplifications).
