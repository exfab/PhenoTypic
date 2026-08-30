# OME-Zarr per-image store

**Date:** 2026-08-18
**Status:** Draft — revised after independent review
**Scope:** Per-image CLI image storage, the staged-GPU commit protocol, GUI tile
reads, the legacy-HDF migration mode, and the Python support floor

## Summary

PhenoTypic will replace the per-image HDF5 file with an **OME-Zarr (NGFF 0.5 /
Zarr format v3) store**, one per input image:

```text
results/<dataset>/hdf/<stem>.h5   ->   results/<dataset>/zarr/<stem>.ome.zarr/
```

Image layers become named sibling NGFF multiscale images inside a
`bioformats2raw.layout` collection; `objmap` becomes a first-class NGFF **label
image**; and all PhenoTypic-specific state — including grid `nrows`/`ncols`,
`Metadata_ImageType`, and the three metadata sections — lives in a namespaced
`attributes.phenotypic` JSON block beside the `ome` block.

The replacement is total. `save2hdf5` / `load_hdf5` / `load_layer_hdf5` /
`save_intermediate_layers` are removed from the write and read paths. Existing
`.h5` runs are converted by an explicit `--mode migrate`, not by a permanent
dual reader.

## Locked decisions

1. **Full replacement, migration by explicit mode.** One writer, one reader.
   Legacy `.h5` is reachable only through `--mode migrate` and the `sdk_`
   functions behind it. No format flag, no runtime sniffing in the hot path.
2. **Layout is a named-series collection.** Root carries
   `bioformats2raw.layout: 3`; `OME/zarr.json` carries
   `series: ["rgb", "gray", "detect_mat"]`.
3. **NGFF 0.5 on Zarr format v3**, raising `requires-python` to
   `>=3.11, <3.13`. Python 3.10 is dropped.
4. **Commit is rename-promote.** Every publishing stage builds a `.part`
   directory and promotes it by directory rename. Stage 2's in-store label
   write is an intermediate, not a publish.
5. **Resume state is carried by consumable markers**, never by NGFF metadata.
6. **OME projection is write-only.** `attributes.phenotypic` is the sole source
   of truth on read.
7. **Layout version and metadata-schema version stay separate** (inherited from
   the flat-metadata design, decision #6).
8. **Pyramids are in scope and tunable** via `--pyramid-levels`; the GUI reads
   them.
9. **The dead HDF DataFrame layer is retired.**
10. **Per-colony measurements are NOT projected into `image-label.properties`.**
    Considered and deferred; parquet remains the only measurement surface.
11. **Conformance is validated against the published NGFF JSON schemas via
    `jsonschema`**, not via `ome-zarr-models`.
12. **Windows is a supported CLI platform for staged runs** (§3.8).
13. **`fsync` before promote is on under SLURM and off locally**, logged at run
    start and overridable with `--durable-writes` (§3.7).

### Supersessions

This design amends two locked decisions of
[2026-08-17-flat-metadata-namespace](../2026-08-17-flat-metadata-namespace/design.md).
Both changes must be recorded there as superseded when this design is approved.

- **Its decision #1** ("Every recompile migrates automatically… not restricted
  to a special command") is superseded. Metadata-schema migration moves out of
  `--mode recompile` and into `--mode migrate`. `recompile` **stops rewriting**
  legacy headers but **keeps reading** them — its decision #3 (permanent
  stored-data compatibility) is untouched, so no existing output directory
  breaks; recompile simply no longer mutates one as a side effect.
- **Its decision #7** ("The startup metadata snapshot is immutable provenance…
  never rewritten") is narrowed to *never rewritten as a side effect*.
  `--mode migrate` rewrites `deliverables/metadata.csv` with canonical
  `Metadata_<Label>` headers, after first copying the untouched bytes to
  `deliverables/metadata.original.csv`. Finalization, chunk writers, and
  recompile still never rewrite it.

---

## Context

### What exists today

Per-image `results/<dataset>/hdf/<stem>.h5`, `schema_version = 2`:

```text
/                     attrs: version, schema_version, metadata_schema_version,
                             phenotypic_class, bit_depth, illuminant, gamma
                             (+ CLI-injected phenotypic_work_id)
/layers/rgb           (H,W,3) uint8|uint16, optional
/layers/gray          (H,W)
/layers/detect_mat    (H,W) float, attrs: detect_mode
/layers/objmap        (H,W) uint16 labels — ALWAYS written, zeros if undetected
/metadata/{protected,public,imported}
/grid/                attrs: nrows, ncols; dataset: grid_finder_json
```

Four properties are load-bearing beyond storage:

- **Atomicity.** `save_image_hdf` writes `.{name}.{pid}.part` and `os.replace`s
  it ([`_cli_output_manager.py:1658`](../../../../src/phenotypic/_cli/_cli_output_manager.py)).
  The PID in that name is what makes duplicate execution benign.
- **Content-defined resume.** `valid_staged_hdf` requires `gray`,
  `detect_mat`/`enh_gray`, and `objmap` to exist with agreeing, non-zero
  `(H, W)` ([`_cli_staged_resume.py:69`](../../../../src/phenotypic/_cli/_cli_staged_resume.py)).
- **A consumable Stage-2 token.** The `.npy` sidecar signals "Stage 2 done" and
  is **deleted** by Stage 3 (`delete_sidecar`,
  [`_cli_staged_workers.py:257`](../../../../src/phenotypic/_cli/_cli_staged_workers.py)).
  The resume planner's `"complete"` branch depends on its *absence*.
- **Stage 3 re-publishes.** After `post_pipeline.apply(image, inplace=True)`,
  Stage 3 performs a full atomic `save_image_hdf`
  ([`_cli_staged_workers.py:225`](../../../../src/phenotypic/_cli/_cli_staged_workers.py)).
  Post-ops mutate the objmap, so this re-save is what publishes the
  **post-refined** segmentation.

A fifth HDF write path exists outside the CLI: `save_intermediate_layers`,
writing the v1-flat layout for GUI builder node previews
([`_image_pipeline_core.py:1024`](../../../../src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py)).

### Why change

- `objmap` is a segmentation label map with no standard representation. NGFF
  models exactly this.
- The GUI decodes an **entire** layer to render one whole-plate tile. Pyramids
  reduce that to a small read.
- Stage 2's sidecar is a workaround for HDF's read-only-while-open constraint.
- A PhenoTypic output directory becomes readable by napari, QuPath, and Vizarr
  without a PhenoTypic install.

---

## 1. Store layout

```text
results/<dataset>/zarr/<stem>.ome.zarr/
├── zarr.json          ome:        {version:"0.5", "bioformats2raw.layout":3}
│                      phenotypic: {…}                        ← §2
├── OME/
│   ├── zarr.json      ome: {version:"0.5", series:["rgb","gray","detect_mat"]}
│   └── METADATA.ome.xml
├── rgb/                            omitted entirely when rgb is empty
│   ├── zarr.json      ome: {version, multiscales, omero}
│   ├── 0/ 1/ 2/ 3/                 uint8|uint16, axes (c,y,x)
│   └── labels/
│       ├── zarr.json  ome: {version, labels:["objmap"]}
│       └── objmap/
│           ├── zarr.json  ome: {version, multiscales, image-label}
│           └── 0/ 1/ 2/ 3/         uint16, nearest-neighbour downsampled
├── gray/              zarr.json + levels, axes (y,x)
└── detect_mat/        zarr.json + levels, axes (y,x)
```

### 1.1 Series and the primary image

`rgb` is optional. The **primary series** is `rgb` when present and `gray`
otherwise, and is always first in `OME/zarr.json`'s `series` list. Labels attach
to the primary series so a generic viewer overlays the segmentation on the right
image. The resolved path is recorded in `phenotypic.labels.objmap`; readers MUST
NOT hard-code `rgb/labels/objmap`.

**`objmap` is always present, including after Stage 1**, matching today's writer,
which emits a zeros objmap when nothing is detected. This is what lets
`valid_staged_store` (§3.6) mirror `valid_staged_hdf` exactly.

### 1.2 Axes

| Series | Axes | `dimension_names` |
|---|---|---|
| `rgb` | `channel`, `space`, `space` | `["c", "y", "x"]` |
| `gray`, `detect_mat`, `objmap` | `space`, `space` | `["y", "x"]` |

NGFF 0.5 requires `dimension_names` on each level array's own `zarr.json`
matching `axes`, requires 2–5 axes with 2 or 3 of `type: "space"`, and requires
ordering time → channel → space. All hold.

### 1.3 Pyramid

Levels halve until `max(H, W) <= 512`:

```text
levels = ceil(log2(max(H, W) / 512)) + 1        (1 when max(H, W) <= 512)
```

`ceil`, not `floor` — a draft used `floor`, which terminates one level early and
leaves a 4000×3000 plate's smallest level at 1000×750. Caught by
[`ngff_store_geometry.py`](../../logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py)
claim C1, whose assertion is retained because it has already failed once.

**Tunable.** `--pyramid-levels auto|N`, default `auto`; `1` disables pyramiding.
The value applies **uniformly to every series in a store** — NGFF requires a
label image to carry the same level count as its parent, so this cannot be
per-layer. The resolved count and downsample methods are persisted in
`phenotypic.pyramid` so readers never infer them.

Per-level shapes use **ceil-halving**: `(h+1)//2, (w+1)//2`. This is normative,
not incidental — §2.2 derives `coordinateTransformations.scale` from the
**actual** level shape ratio, not from `2**n`, because odd extents make the two
diverge and NGFF requires the scale vector to describe the real relationship.

Image layers downsample by local mean. `objmap` downsamples by
**nearest-neighbour**; mean-downsampling fabricates label values present at no
level-0 pixel (script claim C5).

### 1.4 Chunking, sharding, compression

- Chunks `(1, 1024, 1024)` for `rgb`; `(1024, 1024)` for 2-D arrays.
- Shards `(C, 4096, 4096)`. The Zarr v3 sharding codec requires the shard shape
  to be an exact multiple of the chunk shape **in every dimension**, including
  the channel axis (`3 % 1 == 0`); a shard spanning the full channel axis
  collapses per-channel chunks into one file, verified empirically.
- Codec `zstd`, replacing `gzip-4`.

**Chunk key encoding** uses the `"."` separator, so a chunk key is one path
segment (`c.0.0.0`) rather than four nested directories. NGFF 0.5 permits all
Zarr features including chunk key encodings unless it explicitly disallows them.
This is a Windows path-length measure (§3.8) and MUST be uniform store-wide.

**Write-buffer cost is not a constraint.** A shard is the write-buffer unit:
`3 × 4096 × 4096 × 2 B` = 96 MB for `rgb`, and 128 MB worst-case for a float64
`detect_mat`. That is measured against a per-worker **processing** peak of up to
**24 GB** — Stage 1 workers run `imread` plus the whole pre-pipeline, Stage 3
runs post-ops and measurement, and the transient allocations there dominate
everything this design adds. The shard buffer is ~0.5% of that peak and the
resident layers ~1.2%, so **`--njobs` sizing is governed by processing, exactly
as it is today, and this change does not move it.** The figures are recorded
only so a future reader does not have to re-derive them.

**File counts, verified**, against a baseline of exactly 1 HDF file:

| levels | data | metadata | total (4000×3000) |
|---|---|---|---|
| 1 | 4 | 12 | **16** |
| 4 (auto) | 16 | 24 | **40** |
| 4, unsharded | 108 | 24 | **132** |

Each additional level costs exactly **8 files** (4 data + 4 metadata), flat
across plate sizes. Two honest qualifications:

1. **Sharding does not restore HDF parity.** ~40 inodes per image, not ~1; 400k
   rather than 10k at 10k images. It is what makes the format viable on
   Lustre/GPFS, not what makes it free.
2. **Metadata files dominate a sharded store** (24 of 40). An earlier estimate
   of "15–20 sharded" counted only data files; the script now asserts that
   figure is refuted so the mistake cannot recur.

---

## 2. Metadata contract

### 2.1 `attributes.phenotypic` — the source of truth

```json
{"attributes": {
  "ome": {"version": "0.5", "bioformats2raw.layout": 3},
  "phenotypic": {
    "store_schema_version": 3,
    "metadata_schema_version": 2,
    "phenotypic_version": "…",
    "image_class": "GridImage",
    "work_id": "…",
    "series": {"rgb": "rgb", "gray": "gray", "detect_mat": "detect_mat"},
    "labels": {"objmap": "rgb/labels/objmap"},
    "pyramid": {"levels": 4, "stop_px": 512,
                "downsample": {"image": "mean", "label": "nearest"}},
    "detect_mode": "…",
    "illuminant": "D65",
    "gamma": "sRGB",
    "grid": {"nrows": 8, "ncols": 12,
             "grid_finder": {"class": "…", "params": {…}}},
    "metadata": {
      "protected": {"Metadata_ImageName": "…",
                    "Metadata_ImageType": "Grid",
                    "Metadata_BitDepth": 16},
      "public":   {"Metadata_Strain": "…"},
      "imported": {"TIFF:XResolution": …}
    }
  }}}
```

`series` and `labels` are **separate keys**. An earlier draft put both in one
`layers` map, mixing series names with nested paths so readers could not tell
which was which without special-casing.

**Two version markers, not one** — `store_schema_version` describes groups and
arrays, `metadata_schema_version` the header namespace. Flat-metadata decision
#6, carried forward.

**`image_class` and `Metadata_ImageType` remain distinct.** `image_class`
(`Image` / `GridImage`) drives loader dispatch; `Metadata_ImageType`
(`Base` / `Grid` / `Crop` / `Object` / `GridSection`) is user-visible schema
metadata. A `GridSection` is not a `GridImage`; collapsing them loses
information.

**`work_id` is written into this block at write time**, never injected
afterwards. Today it is patched in post-write via `h5py.File(tmp, "r+")`; under
§3 the root `zarr.json` is written last, so a post-hoc patch would violate the
ordering invariant.

Metadata keys are canonical flat `Metadata_<Label>` headers. Semantic ownership
is recovered via `metadata_owner_for_header()` / `metadata_member_for_header()`,
never by prefix parsing.

### 2.2 Write-only OME projection

Derived on every write, **never read back**:

| Source | Projected into |
|---|---|
| actual level shape ratios | `datasets[].coordinateTransformations` (one `scale`) |
| `TIFF:XResolution` / `YResolution` | `axes[].unit` + the level-0 `scale` |
| `Metadata_ImageName` | `multiscales[].name`, `omero.name` |
| `Metadata_BitDepth`, channel identity | the full `omero.channels` block |

**`omero` is emitted completely or not at all.** NGFF makes it conditionally
strict: if present, every channel MUST carry a 6-hex-digit `color` and a
`window` containing **all four** of `min`, `max`, `start`, `end`. An earlier
draft projected only `window.max` and would have failed the conformance gate on
the first store written. The projection is:

```json
{"label": "R", "color": "FF0000", "active": true, "family": "linear",
 "coefficient": 1, "inverted": false,
 "window": {"min": 0, "max": 65535, "start": 0, "end": 65535}}
```

with `max`/`end` = `2**bit_depth - 1`. `gray` and `detect_mat` emit a single
white channel. `omero` is omitted entirely from label groups.

When no resolution tag exists, `scale` is the level-ratio vector with `unit`
omitted, which the spec permits.

### 2.3 `image-label` metadata

The NGFF `label.schema` lists `image-label` and `version` as **required**, even
though the prose says SHOULD. The store therefore always emits it:

```json
"image-label": {"version": "0.5",
                "source": {"image": "../../"},
                "colors": [{"label-value": 0, "rgba": [0,0,0,0]},
                           {"label-value": 1, "rgba": […]}, …]}
```

`colors` MUST carry one entry per unique label value. Colours come from a
deterministic hash of the label value, so they are reproducible and require no
stored palette. **Size cost:** a 1536-colony plate yields 1537 entries, roughly
60 KB of JSON in `labels/objmap/zarr.json`. This is the largest metadata file in
the store and is accounted for in §1.4's per-image byte budget, though not in
its *file count* (it is one file regardless of colony count).

`properties` is deliberately not emitted (decision #10).

### 2.4 `OME/METADATA.ome.xml`

A `MetadataOnly` OME-XML document carrying imported TIFF/EXIF tags and a
REMBI-module-grouped view built from `phenotypic.schema.header_to_module()`.

NGFF marks this SHOULD, **but** the named-series rules say every `multiscales`
group MUST correspond to one OME-XML `Image` in series order. A build failure
therefore cannot simply be warned past while keeping the `series` list. On
failure the writer emits **neither** the XML nor the `OME/` group, falling back
to the consecutive-integer form the spec requires in that case, and logs a
warning. This keeps every emitted store conforming.

### 2.5 What is deliberately not mapped

`grid.nrows` / `ncols` are **not** NGFF `plate` metadata: HCS requires each well
to be a separate image group, while PhenoTypic's grid is a virtual partition of
one array. The mapping would multiply group count by `nrows × ncols` for no
reader benefit.

`detect_mode`, `illuminant`, `gamma`, and `work_id` have no standard equivalent.

---

## 3. Write path and commit protocol

### 3.1 Modules

- **New** `src/phenotypic/sdk_/ngff_.py`: layout constants, pyramid builder,
  chunk/shard policy, `attributes.phenotypic` construction, the atomic-promote
  helper, and `valid_staged_store`.
- `_image_io_handler.py`: `save2zarr` / `load_zarr` / `load_layer_zarr` replace
  the HDF trio, and **`save_intermediate_zarr` replaces
  `save_intermediate_layers`** — a single-level, no-pyramid store for GUI
  builder node previews. This fourth write path was missed in an earlier draft;
  it has three live callers in `_image_pipeline_core.py` and two in tests.
- `_grid_image_handler.py` writes `phenotypic.grid`.
- `_io_constants.py`: `DIR_HDF` → `DIR_ZARR`, `dataset_hdf_dir` →
  `dataset_zarr_dir`, plus `zarr_store_path(output_dir, dataset, stem)` so no
  caller hand-joins `f"{stem}.ome.zarr"`.

### 3.2 The promote primitive

Used by every publishing stage (Stage 1, single-pass, and Stage 3):

1. `shutil.rmtree` any pre-existing `.part` for this stem, then build
   `.<stem>.ome.zarr.<uuid4hex>.part/` as a **sibling** of the target.
   The uuid — matching the `attempt_id = uuid4().hex` convention already used at
   [`_cli_staged_strategy.py:158`](../../../../src/phenotypic/_cli/_cli_staged_strategy.py) —
   replaces an earlier draft's un-suffixed `.part`, which would have let two
   concurrent SLURM tasks interleave chunks into one directory and produce a
   store that *validates*. Today's `.{pid}.part` makes duplicate execution
   benign; this restores that property with a stronger identifier than a
   reusable PID.
2. Write all arrays and chunks; `fsync` per §3.7.
3. Write `OME/zarr.json`, then the root `zarr.json` **last**. The root carries
   `work_id`, `series`, and `labels`, so an interrupted store has no valid root
   and reads as absent.
4. If the target exists, `os.replace` it to `.<stem>.ome.zarr.<uuid>.trash`;
   then `os.replace(part, final)`; then `rmtree` the trash.

**Why the move-aside is mandatory, not an optimization.** `os.replace` onto a
**non-empty directory** raises `OSError ENOTEMPTY` on POSIX (verified), and on
Windows `MoveFileEx`'s `MOVEFILE_REPLACE_EXISTING` cannot name a directory at
all. The design is cross-platform only because step 4 makes the target
nonexistent first.

**Known weakening versus HDF.** Steps 4a and 4b are two non-atomic renames.
Between them the store exists at no path a reader knows — a transient-absence
window the single-file rename did not have. A crash there leaves the image
absent plus an orphaned `.trash` directory. Both are recoverable: absence
reclassifies to the rebuilding stage, and orphaned `.part`/`.trash` directories
are swept at the start of each run by uuid (never by PID, which can collide).

**Windows caveat.** The move-aside fails with `ERROR_SHARING_VIOLATION` if any
of the ~40 files is held open by a running GUI or antivirus — a 40× larger
surface than the single `.h5`. Documented as a known limitation.

### 3.3 Stage 1

Builds and promotes a complete store: `rgb` (when present), `gray`,
`detect_mat`, and a **zeros `objmap`** label with its `ome.labels` list and
`image-label` block. Mirrors today's writer, which always emits an objmap.

### 3.4 Stage 2 — the sidecar becomes a consumable marker

Stage 2 opens the promoted store and overwrites `labels/objmap` in place with
the detector output, then writes a **consumable Stage-2 token**:

```text
<output>/.phenotypic/progress/stage2_done/<dataset>/<stem>.json
```

written atomically (temp + rename), carrying `work_id` and the objmap's level-0
shape.

**Why not `ome.labels`.** An earlier draft used the labels list as the "Stage 2
done" signal, claiming an exact replacement for `sidecar_exists()`. That is
false in two ways, both confirmed against the code:

- The sidecar is **consumable** — `delete_sidecar` runs at the end of Stage 3
  and the resume planner's `"complete"` branch tests its **absence**
  ([`_cli_staged_resume.py:225`](../../../../src/phenotypic/_cli/_cli_staged_resume.py)).
  A durable labels list makes that conjunct permanently false, so `"complete"`
  never fires and every finished image is reprocessed. It also silently
  disables `migrate_legacy_stage3_markers`.
- The labels list is **not** the only discovery path. `zarr.Group.members()`
  enumerates children by store listing and returns a partially written
  `objmap`, which reads as a mix of real labels and `fill_value`. NGFF only says
  label images SHOULD be listed; it grants no exclusivity.

Consequently, **NGFF metadata never carries resume state.** Resume state lives
in `.phenotypic/progress/`, where the rest of it already lives.

### 3.5 Stage 3 — re-publishes, as it does today

Stage 3 loads the store, applies `post_pipeline` in place, writes measurements,
and then **re-promotes the entire store via §3.2**, exactly mirroring today's
atomic re-save. It then writes the completion marker and **deletes the Stage-2
token**, mirroring `delete_sidecar`.

This is not optional. Post-ops (refiners, size filters) mutate the objmap, and
the re-save is what publishes the **post-refined** segmentation. An earlier
draft declared Stage 3 "unchanged" while removing it, which would have left the
store's label image holding raw detector output that disagrees with the parquet
and with a single-pass run — violating the byte-identical-to-single-pass
contract in [`_cli/CLAUDE.md`](../../../../src/phenotypic/_cli/CLAUDE.md).

Stage 2's in-store write therefore buys two things, not atomicity: the `.npy`
sidecar format disappears, and the GUI can render a real objmap mid-run.

### 3.6 Resume validity

`valid_staged_store(path)` mirrors `valid_staged_hdf` case for case:

- root `zarr.json` parses and carries `phenotypic.store_schema_version`;
- every entry in `phenotypic.series` **and** `phenotypic.labels` opens as a Zarr
  array group (objmap included — §3.3 guarantees it exists after Stage 1);
- level-0 `(y, x)` extents agree across all of them **and are non-zero** (a
  zero-size Zarr array is legal and must not pass);
- `phenotypic.work_id` matches when expected, replacing
  `staged_hdf_matches_work_id`.

It catches `OSError`, `KeyError`, `ValueError`, `TypeError`,
`json.JSONDecodeError`, `FileNotFoundError`, and `zarr.errors.BaseZarrError`.
The HDF version's `(OSError, TypeError, ValueError)` set is insufficient because
none of zarr's error types are `ValueError` subclasses.

### 3.7 Concurrency and durability

Zarr has no locking. Safety rests on: the uuid-suffixed `.part` (§3.2), Stage 2
being the sole writer of the label array between promotes, and resume state
living outside the store. Concurrent readers during Stage 2 may observe a torn
`objmap`; the completion marker, not the store's shape, is what gates consumers.

**Durability is environment-dependent.** `write()` returns once data is in the
page cache; without `fsync` the kernel may flush the root `zarr.json` **before**
the chunk data it describes, so a node crash can leave a store that passes
`valid_staged_store` — metadata parses, shapes agree — while reading
`fill_value`. Silent wrong data, not a visible failure. Strengthening validation
does not help: it checks metadata and shapes, never chunk contents, and
checksums would cost a full read of everything just written.

The dominant failure mode does not need `fsync`. A SLURM timeout kills the
*process*; the kernel survives and flushes normally. `fsync` buys protection
only against node loss, power failure, and filesystem crash.

The promote therefore **`fsync`s under SLURM and not locally**, detected from
`SLURM_CPUS_PER_TASK` / `SLURM_JOB_ID` exactly as `resolve_worker_count`
([`_cli_utils.py:65`](../../../../src/phenotypic/_cli/_cli_utils.py)) already
does. Because this makes the same command carry different guarantees in
different places — a genuinely surprising thing to debug — two mitigations are
required, not optional:

- the resolved mode is **logged at run start** ("durable writes: on (SLURM)");
- `--durable-writes / --no-durable-writes` overrides the detection explicitly.

On POSIX this means `fsync` on each chunk file and on the `.part` directory. On
Windows the directory `fsync` is skipped (§3.8).

### 3.8 Windows

Windows is a supported CLI platform for staged runs, and it is already exercised
by the `tests-windows-full` lane
([`run-pytest-full.yml:129`](../../../../.github/workflows/run-pytest-full.yml)).
Moving from one file per image to ~40 changes six things:

1. **No directory `fsync`.** Windows cannot open a directory handle for
   flushing. §3.7's per-file `fsync` applies; the directory step is
   POSIX-guarded, and Windows relies on NTFS journaling for the rest.
2. **The move-aside can fail while files are open.** Windows refuses to rename a
   directory when any file inside it is held open (`ERROR_SHARING_VIOLATION`) —
   a running GUI, an antivirus scan, or the search indexer will do it. The
   exposure is ~40× the single `.h5`. The promote wraps steps 4a/4b in
   retry-with-backoff, reusing the shape of `_open_hdf_with_recovery`
   ([`hdf_.py:34`](../../../../src/phenotypic/sdk_/hdf_.py)), which already does
   exactly this for HDF lock conflicts.
3. **The two-step move-aside is mandatory, not defensive.** `MoveFileEx`'s
   `MOVEFILE_REPLACE_EXISTING` cannot name a directory, so there is no
   single-call replace to fall back to on Windows.
4. **`MAX_PATH`.** Zarr's default nested chunk keys plus an output root, dataset
   name, and image stem can exceed 260 characters. Mitigated twice: the `"."`
   chunk-key separator (§1.4) makes a chunk key one path segment instead of
   four, and store paths are `\\?\`-prefixed on Windows.
5. **NTFS is case-insensitive.** The store's path segments (`OME`, `rgb`,
   `gray`, `detect_mat`, `objmap`, `labels`) contain no case-only collisions.
   Asserted by test rather than left to inspection.
6. **Per-file overhead.** Windows Defender scans each newly created file; 40
   files × 10k images is 400k scans. Documented, not mitigated.

Because Windows runs nightly rather than per-PR, a Windows-specific promote
regression surfaces a day late. The commit-protocol tests (§7) should therefore
run in the PR lane on Linux and the nightly lane on Windows, and the spec accepts
the latency rather than promoting the whole Windows suite.

---

## 4. Read paths

### 4.1 Image loaders

`Image.load_zarr` / `GridImage.load_zarr` read the root `zarr.json`, dispatch on
`phenotypic.image_class`, load level 0 of each series, restore the metadata
sections, then apply `phenotypic.grid`. `load_layer_zarr(path, layer, level=0)`
replaces `load_layer_hdf5`.

### 4.2 GUI tile server

`_load_hdf_layer_rgb` becomes `_load_zarr_layer_rgb(store, key, layer,
target_px)`, selecting the smallest pyramid level covering `target_px`. This is
where the pyramid pays for itself.

**Four mtime/fingerprint traps, not one.** A store directory's `st_mtime_ns` does
**not** change when a nested chunk is rewritten (verified). Every staleness check
against the old `.h5` must move to the **root `zarr.json`**, which §3.2 writes
last on every promote:

| Site | Problem |
|---|---|
| [`_tile_routes.py:471`](../../../../src/phenotypic/gui/results_viewer/_tile_routes.py) | `file_fingerprint()` opens the path as a file → `IsADirectoryError` on a store. Use `paths_fingerprint()`, which handles directories. |
| `_tile_routes.py:469,477` | `stat().st_mtime_ns` compare + `os.utime` against the store |
| [`_preview_tiles.py:76`](../../../../src/phenotypic/gui/builder/_preview_tiles.py) | same compare |
| `tiles.py:518` | mtime-keyed crop path |

Note the production tile route keys its cache on a **content fingerprint**, not
an mtime; only the crop path uses mtime. Both need the fix, for different
reasons.

**Read amplification.** Slicing a 64×64 colony crop from a sharded level 0 costs
a shard-index read plus one full `1024×1024` inner chunk — cheap, but not the
"same as h5py" an earlier draft implied.

### 4.3 Unchanged externally

`--mode process --layer {rgb|gray|detect_mat|objmap}` keeps its behaviour and
output formats.

### 4.4 Affected modules

24 files reference `.h5`, `h5py`, or `dataset_hdf_dir`. Beyond §3.1:

| File | Change |
|---|---|
| `_cli_directory_scanner.py`, `_cli_recompile_slurm_scripts.py` | `*.h5` glob → `*.ome.zarr` **non-recursive** directory scan |
| `gui/results_viewer/_output_root.py` | `rglob("*.h5")` → non-recursive scan. A naive port recurses **into** every store: 400k stat calls at 10k images. |
| `_cli_staged_{resume,strategy,workers,controller,slurm_worker}.py` | store paths, `valid_staged_store`, Stage-2 token |
| `_cli_output_manager.py` | `save_image_hdf` → `save_image_store` (§3.2) |
| `_cli_process_single.py`, `_cli_execution_strategies.py`, `tune/_tune_cli/_run.py` | loader swap |
| `_core/_pipeline_parts/_image_pipeline_core.py` | `save_intermediate_zarr` |
| `gui/_shared/tiles.py`, `gui/results_viewer/_tile_routes.py`, `gui/builder/_preview_{cache,tiles}.py` | §4.2 |
| `_cli_readme_generator.py` | documents the new layout |
| `docs/source/api_reference/core/{image,grid_image}_methods.rst` | public-API change (§5.4) |

---

## 5. `--mode migrate`

### 5.1 Interface

```bash
uv run python -m phenotypic --mode migrate --output <previous-output-dir> [--njobs N] [--dry-run]
```

`migrate` joins `{full, measure, recompile, process}` in the existing
`--mode` choice list ([`phenotypicCLI.py:943`](../../../../src/phenotypic/phenotypicCLI.py)),
reusing `recompile`'s argument validation: no `--pipeline`, no `--input`,
operates on an existing output root.

**Local-only, parallel via `--njobs`.** Migration is one-time, resumable, and
restartable — a partially migrated tree is simply migrated again — so it does
not justify another SLURM controller/array surface with its own chunking and
`MaxArraySize` accounting.

Behind it, `sdk_` exposes `migrate_hdf_to_zarr(src, dst=None)` and
`migrate_run_hdf_to_zarr(output_dir, *, keep_source=True)`.

**A run whose output contains only `.h5` results fails with a pointer to this
mode** rather than auto-migrating. Format conversion rewrites the entire results
tree; that should be typed deliberately, not triggered as a side effect of an
unrelated `--mode full`.

### 5.2 What it converts

1. **Per-image stores.** `results/*/hdf/*.h5` → `results/*/zarr/*.ome.zarr`,
   reusing the existing v1-flat and v2-grouped loaders. The legacy
   **`enh_gray`** layer maps to `detect_mat` — it is the pre-rename name still
   handled at `_cli_staged_resume.py:82` and must not be dropped silently.
   Sources are retained by default.
2. **Metadata-schema headers**, in the same pass. A converted store is canonical
   by construction: legacy per-topic headers are read (permanently supported per
   flat-metadata decision #3) and written as flat `Metadata_<Label>`. There is
   no separate header pass for anything that goes through conversion.
3. **`deliverables/metadata.csv`.** The untouched bytes are copied to
   `deliverables/metadata.original.csv`, then the file is rewritten with
   canonical headers. See the supersession note on decision #7.

### 5.3 Header-only migration is now multi-file

An earlier draft claimed a header rename "rewrites one small `zarr.json`". That
is wrong: a rename also touches `OME/METADATA.ome.xml` (§2.4, derived from
`header_to_module()`) and each series' `multiscales[].name` / `omero.name`
(§2.2). It is a **multi-file publish**, and there is no atomic multi-file
primitive — which matters because flat-metadata decision #2 requires the
original to survive a failed publication.

Header-only migration therefore uses the §3.2 promote: rebuild the store's
metadata files into a `.part` copy (hard-linking unchanged chunk files, so the
copy is cheap) and promote. This preserves decision #2 at a cost far below the
full-HDF copy it replaces, but it is not the trivial operation the draft
described.

### 5.4 Retirement

**Verified:** the DataFrame half of `hdf_.py` is dead. `save_array2hdf5` has
eight live call sites; `save_series_*`, `load_series`, `save_frame_*`,
`load_frame`, `preallocate_*`, and their fixed-length-string codecs have none in
`src/` or `tests/`. Corroborated by commits `66734e8e9`, `3e8b58aa0`, `da9eb6dd8`
removing the last consumers.

Two corrections to an earlier draft:

- **The figure is ~1,346 enumerated lines**, reaching ~1,463 with three
  additional dead statics (`assert_swmr_on`, `get_uncompressed_sizes_for_group`,
  `close_handle`) — not 1,700. Reaching 1,700 would require deleting the keeper
  list itself.
- **`safe_writer`, `swmr_writer`, and `strict_writer` are keepers.** They have
  live callers in `tests/unit/sdk_/test_hdf_open_recovery.py:104,141`. A literal
  "delete the remainder" breaks that file.

Keepers: `_open_hdf_with_recovery`, `_clear_hdf_consistency_flags`, the three
writer properties, and the reader properties the migration path uses.

**This is a public-API removal, not an internal cleanup.** `HDF` is re-exported
in `sdk_/__init__.py:240` and `__all__`, and `phenotypic.sdk_` is published via
a `:recursive:` autosummary with `undoc-members`. `save2hdf5` / `load_hdf5` /
`load_layer_hdf5` additionally appear in `docs/source/api_reference/core/*.rst`
and in runnable doctests at `_image_io_handler.py:274,895`. Removal requires
a release note and doctest updates.

`_cli_sidecar.py` is deleted (§3.4). `_cli_recompile_metadata_migration.py`,
`_slurm.py`, and `_worker.py` (~950 lines) move under `--mode migrate` and lose
their SLURM fan-out, which existed only because copying large HDFs is slow.

---

## 6. Packaging, Python floor, CI

- `requires-python = ">=3.11, <3.13"`. Add `zarr>=3.0`; markers resolve it to
  3.1.6 on 3.11 and 3.3.x on 3.12 with no pinning. Retain `h5py` for migration.
- **The `<3.13` ceiling is not caused by zarr** — zarr 3.3.0 classifies through
  3.14. It is `mahotas` 1.4.18 ([`pyproject.toml:46`](../../../../pyproject.toml)),
  which ships no cp313 wheel. Stated here so the cap does not read as unexamined
  inheritance. Moving it is a separate decision (OQ6).
- Edits: `run-pytest.yml:110`, `run-pytest-full.yml:46`,
  `package-integrity.ci.yml:44`, `publish_to_pypi.yml:20`, **plus**
  `uv.lock:3` (the resolution universe — and `run-pytest.yml:153` keys the
  testmon cache on its hash), the `pyproject.toml:32` classifier, and the stale
  prose comments at `run-pytest.yml:4,107`, `run-pytest-full.yml:10`,
  `package-integrity.ci.yml:43`.
- Ruff sets no `target-version` and mypy no `python_version`; both follow
  `requires-python`, so raising the floor may surface new `UP` lints.

**Neither `ome-zarr` nor `ome-zarr-models` is adopted, in any dependency group.**
`ome-zarr-models` 1.7 pins `pydantic<2.13`; pydantic 2.13 has already shipped,
so it would hold the project a release behind **today**, not hypothetically. A
dev-group-only cap does not help: there is no `[tool.uv] conflicts` block, so
uv produces a single resolution and the cap binds the whole locked environment.
`ome-zarr` 0.18 depends on `ome-zarr-models>=1.6` and so inherits the same cap.

Conformance is instead validated against the **published NGFF JSON schemas**
using `jsonschema` (§7), which has no pydantic constraint. The schemas are
vendored under `tests/fixtures/ngff/0.5/` and treated as read-only reference
material.

---

## 7. Testing

- **Round-trip.** `Image` and `GridImage` → store → back: layers bit-exact,
  metadata sections equal, `nrows`/`ncols`/`grid_finder` equal, `image_class`
  and `Metadata_ImageType` preserved independently.
- **NGFF conformance.** Every written store validates against the vendored
  `image.schema`, `label.schema`, and `ome.schema` via `jsonschema`. Note both
  schemas are stricter than the prose: `ome.schema` requires `series`,
  `label.schema` requires `image-label`. Validation failure fails the suite; it
  is never downgraded to a warning or skipped on a missing dependency.
- **Resume parity — differential.** For every
  `(process_only_layer, markers_required, expected_work_id, artifacts-present)`
  combination `classify_staged_image` currently distinguishes, assert the zarr
  classifier returns the same stage as the HDF classifier.
  `tests/unit/cli/test_staged_resume.py` already parameterizes
  `markers_required` at `:57,86,108,128,146,165`; mirror that shape. This is the
  test that would have caught all three resume breaks in the first draft.
- **Post-refined objmap.** Drive the *staged* pipeline with a post-op that
  provably mutates the objmap (e.g. a size filter that removes one colony);
  assert the published `labels/objmap` matches the parquet's label set. The
  round-trip test is blind to this because it never goes through the stages.
- **Stage-2 token is consumable.** Assert the token exists after Stage 2 and is
  gone after Stage 3, and that a finished image classifies `"complete"`.
- **Commit protocol.** Three cases, not one: (a) interrupt after chunks but
  before the root `zarr.json`; (b) two concurrent writers on the same stem —
  assert distinct `.part` directories and one coherent winner; (c) a stale
  `.part` from a killed process is removed, not merged into. Prove (a) can fail
  by reversing the write order.
- **Pyramid correctness.** No label value at level *n* absent from level 0.
  Mutate the downsampler to `mean` and confirm failure (already demonstrated by
  script claim C5).
- **Cache invalidation.** Rewrite a store in place under a live tile cache and
  assert the served tile changes. Separately assert `paths_fingerprint` handles
  a store directory where `file_fingerprint` raises.
- **Migration.** Golden fixtures in v1-flat and v2-grouped layouts, **including
  one with an `enh_gray` layer**, migrate to stores equal to freshly written
  ones. Assert `metadata.original.csv` is byte-identical to the pre-migration
  `metadata.csv`.

No check may skip on a missing fixture or optional dependency; a check that
cannot run must fail.

**Deliberately not tested:** that a partially written `objmap` is
"undiscoverable". It is discoverable — `zarr.Group.members()` returns it — and a
test asserting otherwise would either fail or be narrowed until tautological.
§3.4 is designed around that fact instead.

### Logic validation

[`ngff_store_geometry.py`](../../logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py)
re-derives the pyramid level count, label level parity, shard/chunk divisibility
across **all** dimensions including the channel axis, shard write-buffer size,
file counts at each `--pyramid-levels` setting, and the label-downsampling
requirement — from numpy alone. It has already refuted three claims made during
design and each refutation is retained as a regression assertion.

---

## 8. Non-goals

- Ingesting third-party OME-Zarr as pipeline input (the projection is
  write-only).
- Measurements in `image-label.properties`.
- Changing the measurement storage format; parquet is unaffected.
- HCS `plate` metadata for the colony grid (§2.5).
- Adopting NGFF 0.6. Its `scene` layout is structurally this directory tree, so
  migration will mean adding a `scene` object to the root, not moving arrays.

## 9. Resolved questions

Recorded rather than deleted, so the reasoning survives.

| | Question | Resolution |
|---|---|---|
| OQ1 | `bioformats2raw.layout` is transitional | **Accept the sunset.** The 0.6 `scene` layout is structurally this directory tree, so migration rewrites the root `zarr.json`, not the arrays. |
| OQ2 | Cost of dropping Python 3.10 | **None.** Floor moves to 3.11. |
| OQ3 | Inode count acceptable? | **Yes, with a knob.** `--pyramid-levels` makes the cost linear at 8 files/level; 1 level is 16 files/image, auto is 40. |
| OQ4 | Path layout | **Confirmed** as `results/<dataset>/zarr/<stem>.ome.zarr/`. |
| OQ5 | Migration ergonomics | **`--mode migrate`**, local-only with `--njobs`, absorbing the metadata-schema migration; a legacy-only output root fails with a pointer rather than auto-migrating. |
| OQ6 | Move the `<3.13` ceiling? | **No.** Keep `mahotas`; the cap stays and its cause is now documented (§6). |
| OQ7 | Windows support level | **Supported for staged runs.** Six consequences specified in §3.8. |
| OQ8 | `fsync` before promote | **On under SLURM, off locally**, with explicit logging and a `--durable-writes` override (§3.7). |
| OQ9 | `image-label.colors` at scale | **Acceptable.** ~60 KB for a 1536-colony plate; always emitted (§2.3). |
| OQ10 | Shard buffer vs `--njobs` | **Not a constraint.** Per-worker processing peaks at ~24 GB; the shard buffer is ~0.5% of that. `--njobs` sizing is unchanged by this design (§1.4). |

## 10. Open questions

None blocking. One adjacent observation, recorded but out of scope: `detect_mat`
as float64 accounts for 96 MB of the resident image and 128 MB of the worst-case
shard buffer. Narrowing it to float32 would halve both. That is a change to the
`Image` data model with accuracy implications well beyond storage, and belongs in
its own design.
