# Process-mode OME-Zarr output and plain-image ingest

**Date:** 2026-08-27
**Status:** Design approved; implementation not started.
**Branch:** `process-mode-ome-zarr`, stacked on `worktree-ome-zarr-image-store`.
**Scope:** The `--mode process` output format, `Image.imread` on an OME-Zarr
store, and the CLI input scanner. Nothing in the full-run bundle store, the
staged-GPU commit protocol, or `--mode migrate` changes.

Builds on [2026-08-18-ome-zarr-image-store](../2026-08-18-ome-zarr-image-store/design.md),
whose `sdk_/ngff_.py` toolkit and `Image._save_store` writer this design reuses
rather than extends.

## Summary

`--mode process` stops writing a flat TIFF/PNG per image and writes a
**single-series OME-Zarr directory store** instead:

```text
out/<rel>/<stem>.tiff        ->   out/<rel>/<stem>.ome.zarr/
```

The store carries the processed pixels of exactly one layer, a full pyramid,
conformant NGFF image metadata, and an `attributes.phenotypic.provenance` block
recording the pipeline that produced it. It opens in napari, QuPath, and Vizarr
with no PhenoTypic install.

`Image.imread` learns to read such a store — and any third-party OME-Zarr — as
**plain pixels**, exactly as it reads a TIFF. It never restores PhenoTypic run
state; that remains `Image.load_zarr`'s job. The CLI input scanner learns to
find stores, so a tree of process-mode output is directly runnable as the input
of a later run.

This **reverses non-goal #1 of the 2026-08-18 design** ("Ingesting third-party
OME-Zarr as pipeline input (the projection is write-only)"). That must be
recorded as superseded there when this design is approved. The projection
remains write-only in the sense that matters: nothing reads NGFF metadata back
to reconstruct PhenoTypic *state*. What `imread` reads is pixels and geometry,
which is what it reads from a TIFF.

## Locked decisions

1. **The artifact is a single-series store plus provenance.** One series — the
   requested `--layer` — with no objmap, no grid, and no `image_class`. The
   image's own metadata sections *are* written (§2.2). What the store does not
   carry is load-bearing; see decision 4.
2. **Directory store, never zipped.** The destination is NAS then cloud object
   storage, where a directory store supports HTTP range reads and a zip does
   not. NGFF RFC-9 (zipped stores) is out of scope.
3. **The distinction between a bundle and a plain image is carried by the verb,
   not by the file.** `imread` always reads pixels; `load_zarr` always restores
   state. No role marker is written and nothing sniffs. This preserves the
   2026-08-18 design's decision #1 ("no runtime sniffing in the hot path").
4. **Process-mode stores omit `image_class`.** It is the one key `load_zarr`
   dispatches on, so omitting it is what makes `load_zarr` refuse a store that
   is not a run bundle (§3.3). This requires a writer change; see §2.2.
5. **Provenance is the existing operation journal, reused unchanged**, except
   that process-mode stores record the pipeline **basename** rather than its
   resolved absolute path (§2.3).
6. **No `kind` marker.** Nothing would read it. The 2026-08-18 design deleted
   `metadata_schema_version` for exactly this reason — a hard-coded constant
   asserting something no code path enforces — and writing `kind` would repeat
   the mistake that ruling exists to prevent.
7. **Zarr is the default output format, per layer.** `--process-format` defaults
   to `zarr` for `rgb` and `gray`, and to `tiff` for `detect_mat` and `objmap`.
   No command that works today starts failing.
8. **`objmap` and `detect_mat` have no OME-Zarr form**, for two different
   reasons. An explicit `--process-format zarr` on either is a `UsageError`.
   See §5.3.
9. **Float series still emit no `omero`.** The 2026-08-18 ruling stands
   unchanged; `gray` and `detect_mat` carry no rendering block. See §2.6.
10. **`omero.rdefs.model` is added on integer series.** The one new field in the
   writer.
11. **Consolidated metadata is written.** Legal under the Zarr v3 extension
   mechanism via `must_understand: false`; see §6.
12. **Provenance carries no *input-image* digest.** See §2.3.3.
13. **`imread` refuses rather than silently projects.** A store it cannot map
    onto PhenoTypic's 2-D image model raises; it never quietly takes index 0 of
    an axis it does not understand. See §4.

---

## Context

### What exists today

`--mode process --layer {rgb|gray|detect_mat|objmap}` runs `pipeline.apply()`
and writes one file per input, mirroring the input tree
([`_cli_process_only.py`](../../../../src/phenotypic/_cli/_cli_process_only.py),
120 lines):

- `process_only_output_path` (`:24`) hardcodes the extension:
  `ext = ".png" if layer == "objmap" else ".tiff"` (`:31`).
- `write_process_only_layer` (`:38`) delegates to the accessor's `imsave`
  through `atomic_write_with_writer`, so the write is already atomic.
- The `--ext` CLI option
  ([`_cli_process_single.py:474`](../../../../src/phenotypic/_cli/_cli_process_single.py),
  `default="tiff"`, free text with no `click.Choice`) is threaded through the
  worker but **process mode never consults it**. It is inert on this path.

`Image.imread`
([`_image_io_handler.py:602`](../../../../src/phenotypic/_core/_image_parts/_image_io_handler.py))
dispatches purely on `filepath.suffix` against `IO.ACCEPTED_FILE_EXTENSIONS`
([`constants_.py:96`](../../../../src/phenotypic/sdk_/constants_.py)) and raises
`UnsupportedFileTypeError` otherwise. A `.ome.zarr` directory is rejected today.

The CLI input scanner
([`_cli_directory_scanner.py`](../../../../src/phenotypic/_cli/_cli_directory_scanner.py))
matches candidate inputs by suffix only (`:37`, `:85`, `:304`) and never
considers directories.

The 2026-08-18 branch already provides everything the writer needs:

| Symbol | Location | Role here |
|---|---|---|
| `Image._save_store(path, *, series, write_objmap, levels, …)` | `_image_io_handler.py:931` | The single writer. Already parameterised by series subset and objmap on/off — a process-mode store is a third caller, not new machinery. |
| `ngff_.array_create_kwargs` | `ngff_.py:385` | Chunk/shard/codec/`dimension_names` policy. |
| `ngff_.axes_for` | `ngff_.py:133` | `("c","y","x")` for `rgb`, `("y","x")` otherwise. |
| `ngff_.build_omero` | `ngff_.py:737` | R/G/B palette (`:730`); returns `{}` for float dtype (`:777`). |
| `ngff_.promote_store` / `new_part_path` | `ngff_.py:1235` / `:1139` | Rename-commit. |
| `ngff_.pyramid_level_count` / `pyramid_level_shapes` | `ngff_.py:150` / `:172` | Pyramid geometry. |
| `sdk_.store_stem` | `_io_constants.py:1531` | Correct stem extraction; documents why `Path.stem` yields `img.ome`. |

### Why change

Three reasons, each observed in a live pipeline rather than assumed. The
reference case is
[`AutoConvertRaw`](file:///rhome/anguy344/bigdata_exfab/software/AutoConvertRaw),
which converts CR3 captures to 16-bit TIFF and colour-corrects them with
`phenotypic --mode process --layer rgb` before publishing to a NAS.

**1. Provenance does not travel with the pixels.** AutoConvertRaw stamps it
per-*directory* and out-of-band: `push_stamp_dirs` (`src/worker_push.sh:43-95`)
drops a `.pp.profile.<hash>` marker plus a copy of the pipeline JSON into
`<dir>/.auto-convert-raw/`, and only once every key under that directory has
reached a terminal state. A corrected TIFF therefore carries no record of the
`ColorCorrector` profile that produced it — and that profile is a 3x13 fitted
`correction_matrix`, 39 coefficients. Move the file, or merge two directories
corrected under different profiles, and the association is gone. Roughly 60
lines of hashing, resolution, and two failure paths exist to approximate what
one JSON key inside each image gives directly.

**2. A flat image file has no atomic commit.** AutoConvertRaw's
`src/verify_lib.sh` exists because, in its own words:

> rawtherapee-cli and tifffile.imwrite both write their TIFF IN PLACE, so a
> SIGKILL mid-write (a `preempt` preemption, a walltime kill, a node failure)
> leaves a TRUNCATED file at the final path. The reapers used to test only for
> existence, so the stub was promoted, published to the NAS as the deliverable,
> and the source CR3 deleted by push_publish. Silent, permanent quality loss.

That gate is deliberately **fail-open**: when the verifier cannot run, every
path is accepted unchecked. A store promoted by `ngff_.promote_store` is built
in a `.part` sibling and committed by directory rename, so a store either has
its root `zarr.json` or does not exist. Truncation at the final path is
unreachable rather than detected. (PhenoTypic's own TIFF writer already uses
`atomic_write_with_writer`, so the live truncation risk on that path is
RawTherapee's — but the reaper cannot tell the stages apart, so it gates both.)

**3. Downstream consumption is lossy.** The corrected TIFFs feed later
PhenoTypic runs. Pixels survive; the operations that produced them do not.

---

## 1. Store layout

```text
out/<rel>/<stem>.ome.zarr/
├── zarr.json          ome:        {version:"0.5", "bioformats2raw.layout":3}
│                      phenotypic: {…}                              ← §2
│                      consolidated_metadata: {…, must_understand:false}  ← §6
├── OME/
│   ├── zarr.json      ome: {version:"0.5", series:["rgb"]}
│   └── METADATA.ome.xml
└── rgb/               the one requested layer, and only it
    ├── zarr.json      ome: {version, multiscales, omero}
    ├── 0/  zarr.json + c.0.0.0
    ├── 1/  zarr.json + c.0.0.0
    ├── 2/  zarr.json + c.0.0.0
    └── 3/  zarr.json + c.0.0.0
```

### 1.1 File count, measured

Enumerated by execution against `ngff_.array_create_kwargs` and
`ngff_.pyramid_level_shapes` at 4000x3000, `--layer rgb`, uint16:

| | levels | shapes | chunks | shards | files |
|---|---|---|---|---|---|
| single-series `rgb` | 4 | (4000,3000) (2000,1500) (1000,750) (500,375) | (1,1024,1024) | (3,4096,4096) | **12** |

Against the bundle store's 40 (three series plus a label), and against exactly 1
for the retired HDF. The 12 are: root `zarr.json`; `OME/zarr.json`;
`OME/METADATA.ome.xml`; `rgb/zarr.json`; and per level a `zarr.json` plus one
shard file `c.0.0.0`. Chunk keys are one path segment because
`CHUNK_KEY_SEPARATOR` is `"."` (`ngff_.py:334`).

At AutoConvertRaw's scale this is a 12x multiplication of object count, not the
bundle store's 40x. On object storage that is a request-cost question, not an
inode-limit one. This figure is re-derived from scratch by the logic-validation
script (§9).

### 1.2 Geometry, chunking, sharding

Unchanged from the 2026-08-18 design §1.3-1.4 and inherited wholesale through
`array_create_kwargs`: chunks `(1,1024,1024)` for `rgb` and `(1024,1024)` for
2-D, shards `(C,4096,4096)`, `zstd`, `"."` chunk-key separator. Pyramid depth is
`pyramid_level_count(h, w)`, a pure function of the level-0 shape, with no user
lever — the 2026-08-19 ruling that descoped `--pyramid-levels` applies here
unchanged.

### 1.3 Naming and tree mirroring

`process_only_output_path` keeps mirroring `image_path` relative to
`input_root`, with the extension resolved by §5. A store is
`<stem>.ome.zarr`; the stem is recovered with `sdk_.store_stem`, never
`Path.stem` — that helper already documents why (`_io_constants.py:1531-1556`).

---

## 2. Metadata contract

### 2.1 RGB versus grayscale is the canonical NGFF encoding

NGFF 0.5 has **no RGB type**. In the pinned reference copy
(`../2026-08-18-ome-zarr-image-store/refs/ngff-0.5.html`) the string "RGB"
occurs in normative prose exactly once, describing the hex format of
`omero.channels[].color`. There is no colour-space attribute, no photometric
field, no interleaved-samples flag. §2.4:

> The `axes` MUST contain 2 or 3 entries of `type:space` and MAY contain one
> additional entry of `type:time` and MAY contain one additional entry of
> `type:channel` or a null / custom type. […] the entries MUST be ordered by
> `type` where the `time` axis must come first (if present), followed by the
> `channel` or custom axis (if present) and the axes of type `space`.

The channel axis is `MAY`, so a bare `["y","x"]` grayscale image is conformant.
RGB is therefore a 3-length `channel` axis ordered before the space axes —
**planar `(3,H,W)`, not interleaved `(H,W,3)`**. The community position on the
gap is explicit: as of v0.5 there is no formal specification for distinguishing
RGB components from conventional optical channels, and the recommendation is to
flatten them into the channel dimension
([yaozarrs guide](https://imaging-formats.github.io/yaozarrs/ome_zarr_guide/)).
The long-running issues ([#23](https://github.com/ome/ngff/issues/23),
[#78](https://github.com/ome/ngff/issues/78)) and the active RFCs (6, 8, 9) do
not address it. No RGB type is coming; flatten-into-`c` is the canon, and it is
what `bioformats2raw` emits.

`ngff_.axes_for` (`:133`) and `build_omero` (`:737`) already implement exactly
this. **No change is required to make PhenoTypic canonical** — it already is.

| Layer | axes | `dimension_names` | shape | `omero` |
|---|---|---|---|---|
| `rgb` | channel, space, space | `["c","y","x"]` | `(3,H,W)` | R/G/B + `rdefs.model:"color"` |
| `gray`, `detect_mat` | space, space | `["y","x"]` | `(H,W)` | omitted (float) |

### 2.2 `attributes.phenotypic`

```json
{"attributes": {
  "ome": {"version": "0.5", "bioformats2raw.layout": 3},
  "phenotypic": {
    "store_schema_version": 3,
    "phenotypic_version": "0.18.0",
    "series": {"rgb": "rgb"},
    "pyramid": {"levels": 4, …},
    "illuminant": "D65",
    "gamma": "sRGB",
    "detect_mode": "gray",
    "metadata": {"protected": {…}, "public": {…}, "imported": {…}},
    "provenance": {
      "schema_version": 1,
      "status": "complete",
      "retry_base_length": 0,
      "pipeline": {"source_path": "preprocess_pipeline.json.pht-pipe",
                   "sha256": "a3f2…"},
      "operations": [
        {"sequence": 1,
         "operation_name": "ColorCorrector",
         "operation_class": "phenotypic.correction._color_corrector.ColorCorrector",
         "phenotypic_version": "0.18.0",
         "parameters": {"profile": {"correction_matrix": [[…], […], […]], …}},
         "applied_at_utc": "2026-08-27T18:04:11.212Z",
         "duration_seconds": 4.117,
         "pipeline_step_path": ["ColorCorrector"]},
        {"sequence": 2, "operation_name": "ColorDenoise", …}
      ]
    }
  }}}
```

**What is absent is the contract.** No `image_class` — its absence is what makes
`load_zarr` raise on this store (§3.3), and the writer omits it deliberately
(§2.3.2). No `grid`, no `labels`, no `work_id`. And no `kind` marker
(decision 4).

`metadata` sections **are** written: they are the image's own imported TIFF/EXIF
tags and schema metadata, which a processed image legitimately carries. Omitting
them would discard capture provenance for no benefit. It is `image_class` alone
that gates `load_zarr`.

`store_schema_version` stays gated by value through
`ngff_.require_readable_store` (`:626`) on every path that decodes store content
**as PhenoTypic state**. `imread` is deliberately not such a path; see §4.6.

### 2.3 Provenance is the existing journal, not a new block

An earlier draft of this section specified a bespoke `provenance` mapping
carrying `ImagePipeline.to_json()`, a `pipeline_sha256`, and a `source` path,
and ruled out timestamps. **That was written in ignorance of an existing
system and is withdrawn in full.**

#### 2.3.1 What already exists

[`_core/_provenance.py`](../../../../src/phenotypic/_core/_provenance.py) (384
lines) maintains a per-image **operation journal**, and
`_build_store_attributes` already passes it into every store automatically
([`_image_io_handler.py:868`](../../../../src/phenotypic/_core/_image_parts/_image_io_handler.py)):

```python
provenance=deepcopy(self._metadata.provenance_journal),
```

`build_phenotypic_attributes` already accepts it (`ngff_.py:498`) and writes it
under `PhenotypicAttr.PROVENANCE` (`:572`). `_load_from_store` already reads it
back (`_image_io_handler.py:1462`), and `_cli_staged_resume.py:94` already
depends on it.

The journal is **strictly better than the withdrawn design**. It records what
actually ran, with resolved parameters, rather than what was configured:
`append_operation_provenance` (`_provenance.py:326`) captures
`operation.model_dump(mode="json")` per operation, so `ColorCorrector`'s 39
fitted coefficients are recorded as applied. `pipeline.sha256`
(`pipeline_source_identity`, `:276`) already is the `pipeline_sha256` the
withdrawn design proposed.

Three consequences:

- **The `pipeline` key of the withdrawn design is deleted.** The journal's
  `operations[]` supersedes it.
- **The "no timestamp" ruling is withdrawn.** The journal emits
  `applied_at_utc` and `duration_seconds` per operation. The reproducibility
  argument behind that ruling still holds in the abstract, but not at the price
  of maintaining a second provenance mechanism beside this one. Reuse wins.
- **`Image` gains no new writer parameter.** The journal rides the existing
  path.

#### 2.3.2 The two changes that are needed

**(a) Populate the journal on the process path.** `initialize_cli_provenance`
is called from `_cli_process_single.py:263` on the full path and from
`_cli_staged_workers.py:88`, but
[`_cli_process_only.py`](../../../../src/phenotypic/_cli/_cli_process_only.py)
never touches provenance — verified by grep. So a process-mode store would today
carry the empty journal `new_provenance_journal()` returns. The fix is one call
before `pipeline.apply()`, not a subsystem.

**(b) Record the pipeline basename, not the resolved absolute path.**
`pipeline_source_identity` (`_provenance.py:276-282`) does `Path(path).resolve()`
and stores the result, so a published store would carry, e.g.,
`/rhome/<user>/bigdata_exfab/software/AutoConvertRaw/config/preprocess_pipeline.json.pht-pipe`
— cluster filesystem layout, username, and project directory names, inside an
artifact bound for a NAS and then object storage.

Process-mode stores record `Path(path).name` instead. `sha256` is unchanged and
still pins the pipeline's identity exactly, so nothing is lost but the ability
to point at the file on the cluster — which a published artifact should not be
asserting anyway.

The journal therefore means slightly different things by mode, and that is
deliberate: a bundle store stays inside the run directory and benefits from the
absolute path; a process-mode store is a publication artifact and does not. The
difference is carried by one explicit parameter, not by inference (§8).

**Provenance travels with the pixels for exactly one hop.** Because
`initialize_cli_provenance` opens with `new_provenance_journal()` and `imread`
reads a store as plain pixels rather than restoring state (§3.2), feeding a
process-mode store back in as input yields a second store whose journal records
only the *second* pipeline — a chain of process-mode runs does not accumulate a
lineage. That follows from decision 3 and is intended, but it is not what "the
store carries the operations that ran" reads like at first glance, so it is
stated here rather than left to be discovered. Pinned by
`test_a_store_round_trips_store_in_to_store_out`.

#### 2.3.3 A published store is bit-reproducible

**Added 2026-08-28 (user ruling), after a defect in §7.3 exposed the need.**

A process-mode store omits two fields from every entry in
`provenance.operations[]`:

```json
"applied_at_utc": "2026-08-28T19:05:39.448Z",
"duration_seconds": 0.00850173132494092
```

They are written by `append_operation_provenance`
([`_provenance.py:377,380`](../../../../src/phenotypic/_core/_provenance.py))
from `datetime.now(timezone.utc)` and a `perf_counter` delta, on every
operation apply. **They are the entire source of non-reproducibility in a
store.** Everything else in the block is a pure function of the inputs —
`operation_name`, `operation_class`, `phenotypic_version`, the resolved
`parameters`, `pipeline_step_path`, and the `pipeline` digest. Measured across
two runs of one image through one pipeline:

```text
files differing:            ['zarr.json']
phenotypic keys differing:  ['provenance']
operation fields differing: ['applied_at_utc', 'duration_seconds']
```

**Only the published artifact drops them.** The in-memory journal keeps both,
and so does the bundle store, which never leaves the run directory. The switch
is threaded to the writer exactly as `write_image_class` and `consolidate` are,
rather than mutating the caller's image.

Two things this buys. It makes §7.3's whole-tree digest **stable** across an
identical regeneration, so the tree walk can stay a dumb complete hash with no
exclusion list, no JSON round-trip, and no coupling to the metadata schema. And
it makes the artifact byte-identical across identical runs, which is what
content-addressed storage, server-side dedup, and "did these two runs agree?"
all require of an object-storage artifact.

Nothing reads either field on this path: `_cli_staged_resume.py:94-101` reads
the journal but only `status`, and the two tests asserting on
`duration_seconds` (`test_staged_store_stages.py:122`,
`test_cli_provenance_original.py:72`) are both on the staged/bundle path. What
is lost is human-facing: when a store was processed, and how long each
operation took. `duration_seconds` is telemetry rather than provenance — it
says nothing about *what* was computed — and the filesystem still records mtime.

#### 2.3.4 A published store reports a terminal status

**Defect found 2026-08-28 while implementing §2.3.3.** Every published
process-mode store said:

```json
"status": "in_progress"
```

`initialize_cli_provenance` defaults `status="in_progress"`
(`_provenance.py:305`), and every other CLI path calls
`set_provenance_status(image, "complete")` on success —
`_cli_process_single.py:311`, `_cli_staged_workers.py:493`. The process-only
path never touched status at all.

That is not cosmetic. `status` is the field that says whether an artifact is
trustworthy, and `_cli_staged_resume.py:101` already gates on
`status in {"staged", "complete"}` — so a consumer following that same
convention would reject every store PhenoTypic publishes. The process path sets
`"complete"` after a successful apply, and `"failed"` on the error path, matching
its siblings.

#### 2.3.3 Still deferred

**No `source_sha256`** — a digest of the *input image*, distinct from the
journal's `pipeline.sha256` (a digest of the *pipeline file*). Digesting a 72 MB
TIFF per image is real I/O at AutoConvertRaw's scale, no current consumer needs
it, and AutoConvertRaw does not digest its inputs today. It may land later
behind a flag.

### 2.4 Write-only OME projection

`multiscales`, `omero`, and `OME/METADATA.ome.xml` are built by the existing
`ngff_` functions and are **never read back** (2026-08-18 decision #6). The
projection for a single-series store is the same code with a one-element series
tuple: `OME/zarr.json` carries `series: ["rgb"]` and the OME-XML carries one
`<Image>`, preserving the named-series rule that every `multiscales` group
corresponds to one OME-XML `Image` in series order.

Physical resolution remains unprojected (2026-08-19 ruling); `scale` vectors are
pure sampling factors and `axes[].unit` is omitted.

### 2.5 `omero.rdefs` — the one writer addition

`build_omero` gains `"rdefs": {"model": "color"}` for `rgb` and
`{"model": "greyscale"}` for a single-channel **integer** series. Verified
absent today: `grep -rn rdefs src/ tests/` returns nothing. NGFF §2.5 documents
`rdefs.model` as taking exactly `"color"` or `"greyscale"`; it is the only field
in the format that states the rendering model outright, and OMERO and Vizarr
read it. It is emitted only where `omero` itself is emitted, so the
whole-or-nothing rule per group is unaffected.

### 2.6 Float series still carry no `omero`

The 2026-08-19 ruling stands: `build_omero` returns `{}` for any float dtype
(`:777-778`), so `gray` and `detect_mat` — both float32 — emit no rendering
block, and consequently no `rdefs.model: "greyscale"` either. Their
grayscale-ness is carried by the absence of a channel axis, which §2.4 of the
spec makes sufficient and unambiguous.

This was re-examined for this design, because the original deferral was made
while the store was internal machine state and this output is explicitly for
external viewers. Two considered alternatives were rejected:

- **A window derived from the actual data range** would give viewers a declared
  range while keeping float pixels bit-exact, but makes the metadata
  data-dependent: two stores of the same layer carry different windows.
- **A fixed `[0,1]` window** is data-independent but wrong where it matters —
  the 2026-08-18 design records that `detect_mat` values are not bounded to
  `[0,1]`, so it would clip silently.

Omission is conformant (§2.5 makes `omero` optional) and costs nothing on the
`rgb` path, which is the only layer the reference use case exercises. Viewers
fall back to auto-scaling. Recorded as deferred, not resolved: see §11.

---

## 3. Read paths

### 3.1 Three store kinds

| | Origin | `attributes.phenotypic` | Shape |
|---|---|---|---|
| **A. Bundle** | a full run's `results/<ds>/zarr/` | `image_class`, `grid`, `metadata`, `labels`, `work_id` | 1-3 series + label |
| **B. Processed** | `--mode process` (this design) | `provenance`, `series`, no `image_class` | 1 series |
| **C. Third-party** | napari, QuPath, `bioformats2raw` | absent entirely | arbitrary: 5-D `tczyx`, N series, HCS plate |

### 3.2 The distinction is the verb

```text
Image.imread(p)            Image.load_zarr(p)
────────────────           ──────────────────
A bundle    -> pixels      A -> full state restore
B processed -> pixels      B -> raises (no image_class)
C 3rd-party -> pixels      C -> raises (no phenotypic block)
```

Nothing inspects the store to choose behaviour. `imread` reading a bundle store
yields its primary series' pixels and discards run state; that is the documented
meaning of the verb, not an accident, and it is the same relationship `imread`
has to a TIFF that happens to have been written by a PhenoTypic run.

This is the only option consistent with the 2026-08-18 decision #1. It is also
the only one that serves case C at all: a marker-gated design has no marker to
read on a QuPath export.

### 3.3 `load_zarr` needs an explicit guard

`load_zarr` raises on a process-mode store because the writer **omits**
`image_class` there. Two corrections to an earlier draft underlie that sentence.

**First: `load_zarr` does not raise on a missing `image_class` today.** Verified
by reading the code:

- `load_zarr` does `block.get(PhenotypicAttr.IMAGE_CLASS)`
  ([`_image_io_handler.py:1673`](../../../../src/phenotypic/_core/_image_parts/_image_io_handler.py)).
  A missing key yields `None`, which is not `"GridImage"`, so the
  subclass-mismatch warning does not fire and control falls through to
  `_load_from_store` unimpeded.
- `require_readable_store` passes, because a process-mode store *does* carry a
  correct `store_schema_version`.
- `_load_from_store` (`:1369`) then reads the *header* fields with defaulting
  `.get()` calls — `series`, `metadata`, `labels`, `bit_depth`, `illuminant`,
  `gamma` — so nothing stops it there.

**Corrected again, 2026-08-28 (found during implementation, verified by
execution).** An earlier version of this section concluded that `load_zarr`
would therefore *silently succeed*, returning a degraded `Image`. It does not.
Having defaulted the header, `_load_from_store` goes on to subscript the series
mapping **bare** — `series["gray"]` (`:1640`), `series["detect_mat"]` (`:1650`)
— so a single-series store raises:

```text
KeyError: 'detect_mat'
```

The guard is still required, and for a reason that survives the correction: an
error naming an internal series key tells a user nothing about what they did
wrong or what to do instead. What changes is only the severity of the
pre-existing behaviour — an obscure error, not a plausible wrong object. Both
justify replacing it with one that names `imread`.

**Second: `image_class` is written unconditionally today**, so its absence was
not something the design could simply rely on. `_build_store_attributes`
hardcodes `image_class=type(self).__name__`
([`_image_io_handler.py:850`](../../../../src/phenotypic/_core/_image_parts/_image_io_handler.py)),
and `build_phenotypic_attributes` declares it a **required** keyword
(`ngff_.py:489`). Every store carries it, `save_intermediate_zarr`'s GUI preview
stores included.

Both halves are therefore in scope:

1. **The writer gains a way to omit it.** `_save_store` takes
   `write_image_class: bool = True`, threads it to `_build_store_attributes`,
   and `build_phenotypic_attributes`'s `image_class` widens from `str` to
   `str | None`, omitting the key when `None`. Only the process-mode caller
   passes `False`; `save2zarr` and `save_intermediate_zarr` are unchanged.
2. **`load_zarr` gains the guard**: absent `image_class` raises, naming
   `imread` as the remedy.

```text
ValueError: <path> carries no phenotypic.image_class and is not a PhenoTypic
run bundle. It was written by --mode process or by another tool. Use
Image.imread() to read its pixels.
```

The guard is on `image_class` specifically, not on the absence of `provenance`
or on any `kind` marker: `image_class` is the field `load_zarr` actually
dispatches on, so gating the read on the same key it dispatches on keeps the
check and its purpose from drifting apart. **Corrected 2026-08-27:** an earlier draft claimed the guard also gives case C
(a third-party store, no `phenotypic` block) a clear error instead of a
`KeyError`. Placed after `require_readable_store`, it does not — that call
raises `KeyError` first, at `ngff_.py:623` (`return
attributes[PhenotypicAttr.ROOT]`), before the guard is reached.

`load_zarr` therefore reads the root attributes directly and raises the **same**
`ValueError` for both shapes of "not a run bundle": no `phenotypic` block at
all, and a block with no `image_class`. One error, one remedy, one message — a
caller does not care which of the two applies, and splitting them would leave
the more common case (a third-party store) with the worse error.

This is the one place the design lets the file carry a signal, and it is a
deliberate narrowing of locked decision 3 rather than a hole in it: `imread`
still never inspects the block to decide behaviour. Only `load_zarr` does, on
the single key it already dispatches on.

---

## 4. The `imread` projection rule

The substance of ecosystem compatibility. `Image` models a 2-D image, optionally
with three colour channels. An arbitrary NGFF store does not. The mapping is
explicit, ordered, and refuses rather than guesses.

Implemented as `ngff_.read_ngff_image_spec(store, …)` — a pure resolver in
`sdk_` with no `Image` import, returning the array plus the hints `imread` needs.
`imread` stays a thin caller, matching how it already delegates to `ski.io` and
`rawpy`.

### 4.1 Resolve the series

In order:

1. The root carries `ome.plate` -> **raise.** An HCS plate is a collection of
   wells, not one image; the error names the well-path form to pass instead.
2. `OME/zarr.json` carries `ome.series` -> take `series[0]`.
3. The root itself carries `ome.multiscales` -> the root **is** the image.
4. The root carries `bioformats2raw.layout` but no `series` -> the
   consecutive-integer form NGFF §2.2.3 mandates in that case; take group `"0"`.
5. Otherwise -> **raise**: not an OME-Zarr image.

**The plate check is first, and the ordering is load-bearing.** An earlier draft
of this section numbered it 4, after the series list. That is wrong against the
real artifact: a `bioformats2raw` plate carries **both** a root `ome.plate`
**and** an `OME/zarr.json` series list of its well fields, so under the
series-first ordering the resolver returns a single well field and reads it as
the image — contradicting §9's own "HCS plate -> raises" row. The
implementation has always checked `plate` first; this is the spec catching up to
it. `test_resolver_refuses_an_hcs_plate` now builds its fixture with both keys
present and fails under the series-first ordering, so the two cannot drift again.

A `series=` keyword bypasses the whole ladder, step 1 included — which is
precisely what step 1's own error message instructs, "pass
`series=<row>/<col>/<field>`". Reading one well field out of a plate is
supported; only *guessing* which one is refused. An unknown `series=` raises
`ValueError` naming the series the store actually declares — not
`FileNotFoundError`, which in this codebase means "interrupted write, store
absent".

### 4.2 Resolve the level

`datasets[0]`. NGFF §2.4 makes the ordering normative: *"The `path`s MUST be
ordered from largest (i.e. highest resolution) to smallest."* A `level=` keyword
overrides, mirroring `load_layer_zarr(path, layer, level=0)`.

### 4.3 Project the axes

Read `axes` from the resolved `multiscales` and map onto `(H,W)` or `(H,W,3)`:

| Axis | Size 1 | Size > 1 |
|---|---|---|
| `time` | squeeze | **raise** (override: `t=`) |
| third `space` (`z`) | squeeze | **raise** (override: `z=`) |
| `channel` | squeeze | 3 -> transpose `(c,y,x)` to `(y,x,c)`; 2 or >=4 -> **raise** (override: `c=`) |
| no `channel` axis | — | 2-D as-is |

Refusal is the point. Silently reading `t=0` of a timelapse, or channel 0 of a
5-channel acquisition, produces a plausible image and a wrong result that
nothing downstream can detect. The override keywords make the choice explicit
and auditable in the caller.

### 4.4 Derive `bit_depth`

**Corrected 2026-08-27.** An earlier draft read `phenotypic.bit_depth`. **No
writer emits that key.** `build_phenotypic_attributes` (`ngff_.py:539-570`)
emits `store_schema_version`, `phenotypic_version`, `image_class`, `series`,
`pyramid`, `detect_mode`, `illuminant`, `gamma`, `metadata`, and the optional
`provenance` / `labels` / `work_id` / `grid` — and nothing else. Bit depth lives
in `metadata.protected[IMAGE.BIT_DEPTH]`, which is where `_load_from_store`
reads it (`_image_io_handler.py:1406`).

Resolution order:

1. `phenotypic.metadata.protected[Metadata_BitDepth]` when present;
2. otherwise inferred from dtype (`uint8` -> 8, `uint16` -> 16);
3. otherwise left to the `Image` constructor's default.

A third-party store has neither, so step 3 is the normal path for case C. Note
step 2 has no answer for a float dtype, which is why a `gray` or `detect_mat`
store relies on step 1 — and why reading the wrong key would have silently lost
bit depth on every float round trip.

### 4.5 Name and metadata

`name` comes from `sdk_.store_stem(path)`. `metadata[IMAGE.SUFFIX]` is
`".ome.zarr"`.

`imread` carries across exactly two things, and only when present:
`phenotypic.provenance`, and `phenotypic.metadata.imported`. Both are what the
file says about itself, which is precisely what `imread` already extracts from a
TIFF via `_extract_tiff_metadata`. The `protected` and `public` sections are
PhenoTypic run state and are never carried — that is the line that keeps
`imread` from becoming a partial `load_zarr`.

**The imported section is written through `_metadata.imported`, never through
`image.metadata[key]`.** `MetadataAccessor.__setitem__`
([`_metadata_accessor.py:210-218`](../../../../src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py))
routes an unrecognised key into `_public_metadata` and raises `ValueError` on
any non-scalar value — so the obvious loop would land imported tags in the
`public` section, contradicting the paragraph above, and would raise on a
structured TIFF tag. The TIFF branch already does it correctly
(`_image_io_handler.py:727`: `image._metadata.imported.update(…)`) and the store
branch follows it, normalising through the same helper `_load_from_store` uses
(`:1466-1472`).

### 4.6 `imread` does not gate on `store_schema_version`

`require_readable_store` raises `KeyError` when the `phenotypic` block is
absent, which is the normal condition for every third-party store — the exact
case `imread` exists to serve. `imread` therefore reads NGFF structure only and
treats the `phenotypic` block as optional enrichment. The version gate stays
where it belongs: on the paths that decode a store as PhenoTypic state.

A consequence worth stating: a store written by a *newer* PhenoTypic is readable
by `imread` as pixels, and correctly so — its NGFF geometry is still NGFF.

---

## 5. CLI

### 5.1 `--process-format`

```python
@click.option(
    "--process-format",
    type=click.Choice(["tiff", "zarr"]),
    default=None,        # resolved per-layer; see 5.2
    help="Output format for --mode process. Default: zarr for "
         "rgb/gray/detect_mat, tiff for objmap.",
)
```

`--ext` is left untouched. It is inert on this path today (§Context), and wiring
it in would change TIFF/PNG naming as a side effect of a zarr change.
Overloading a free-text option as a format discriminator would also make every
typo a silent mis-selection.

### 5.2 The default is layer-dependent

| `--layer` | default format | output |
|---|---|---|
| `rgb`, `gray` | `zarr` | `<stem>.ome.zarr/` |
| `detect_mat` | `tiff` | `<stem>.tiff` (float, unchanged) |
| `objmap` | `tiff` | `<stem>.png` (16-bit raw labels, unchanged) |

Resolved in one function beside the existing `--mode process requires --layer`
check (`_cli_process_single.py:591`), so the rule has one home. `--help` states
it; it is not left to be inferred.

The alternative — a uniform `zarr` default with `--layer objmap` erroring until
the user passes `--process-format tiff` — was rejected: it breaks a command that
works today, and the error is pure ceremony since exactly one answer is legal.

### 5.3 Two layers have no OME-Zarr form

NGFF §2.6 is structural about label images:

> In OME-Zarr, Zarr arrays representing pixel-annotation data are stored in a
> group called "labels". […] The "labels" group is nested within an image group,
> at the same level of the Zarr hierarchy as the resolution levels for the
> original image. The "labels" group is not itself an image.

A standalone objmap store therefore has no conformant single-series form. Two
alternatives were considered and rejected:

- **objmap as a plain image series** (`uint16`, `["y","x"]`) is a valid NGFF
  image but not a label image: viewers render a grey ramp rather than a
  segmentation, and nothing links it to what it segments. That discards the
  reason the 2026-08-18 design gave for moving objmap to NGFF at all.
- **A carrier image plus a nested `labels/objmap`** is conformant and overlays
  correctly, but makes this store carry two arrays and breaks decision 1.

The refusal is explicit and names the remedy:

```text
UsageError: --layer objmap has no single-series OME-Zarr form (NGFF 0.5 §2.6:
a labels group is nested inside an image group and is not itself an image).
Use --process-format tiff for the 16-bit raw-label PNG, or --layer rgb.
```

Because the default for `objmap` is already `tiff` (§5.2), this fires only on an
explicit request, never on a bare command.

#### `detect_mat` — a PhenoTypic guard, not an NGFF rule

**Amended 2026-08-27 (user ruling), correcting decision 7 as first written.**
`detect_mat` was specified as defaulting to `zarr`. It cannot: verified by
execution,

```text
rgb          OK
gray         OK
detect_mat   RAISES: ValueError: no primary series among ['detect_mat']
```

`_write_store_part` calls `ngff_.primary_series(series_names)` unconditionally
([`_image_io_handler.py:1019`](../../../../src/phenotypic/_core/_image_parts/_image_io_handler.py)),
and that function accepts only `rgb` or `gray` (`ngff_.py:471-475`).
`_save_store`'s own docstring already states the constraint: *"series: … Must
contain a primary series (`rgb` or `gray`)"*.

Two alternatives were considered and rejected. **Widening `primary_series`** to
return the sole series of a one-series store is defensible — the function exists
only to decide where labels attach, and a process store sets
`write_objmap=False` so it has none — but it edits a function the bundle write
path also depends on, to serve a float analysis intermediate that is not an
interop artifact. **Co-writing `gray`** (the existing precedent at
`_image_io_handler.py:1240`, `wanted = set(layers) | {"gray"}`) would break
decision 1 and roughly double the store for a layer the user did not ask for.

Be honest about the asymmetry in the error text: `objmap` is refused because
NGFF says a labels group is not an image; `detect_mat` is refused because
PhenoTypic's writer requires a primary series. The first is a format rule, the
second is ours, and a user reading the message deserves to know which:

```text
UsageError: --layer detect_mat has no single-series OME-Zarr form: PhenoTypic's
store writer requires a primary series (rgb or gray) and detect_mat is neither.
Use --process-format tiff for the float TIFF, or --layer gray.
```

If a `detect_mat` store is ever wanted, widening `primary_series` is the change
to make, and it belongs in its own design.

### 5.4 The option must reach the user-facing CLI, not just the worker

**Added 2026-08-27.** An earlier draft wired `--process-format` only into
`_cli_process_single.py`. **That file is the per-image SLURM worker**
(`@click.command()` at `:420`, function `main` at `:545`), not the command a
user runs. `python -m phenotypic` is `phenotypicCLI.py`, which declares its own
`--layer` (`:1235-1240`), validates it (`:1331-1338`), and builds an
`ExecutionConfig` (`:1663`). Wiring only the worker leaves
`--mode process --process-format zarr` an unknown-option error.

Three consequences, all in scope:

1. **`--process-format` is declared on `phenotypicCLI.py` and resolved once**,
   beside the existing `--layer` guard, then carried in
   `ExecutionConfig` (`_cli_types.py:185`).
2. **`process_only_output_path` has seven call sites** outside
   `_cli_process_only.py`, every one of which currently computes a `.tiff` or
   `.png` path: `phenotypicCLI.py:450` (the continuation artifact) and `:954`
   (dry-run summary), `_cli_execution_strategies.py:159` (local completion
   marker), `_cli_process_single.py:721` (the worker's own artifact
   publication), and `_cli_staged_{strategy.py:128,402, resume.py:220}`
   (objmap-only, so safe today, but they should pass the format explicitly
   rather than inherit a default). Because the parameter defaults to `"tiff"`,
   missing one is silent: continuation hunts for a file that was never written.
   The format must also reach `_cli_execution_strategies.py:578` (the local
   strategy's `process_single_apply_only_core` call) and the worker command line
   built at `_cli_slurm_array_scripts.py:297-303`.
3. **The format joins the continuation identity.** It is added to
   `processing_configuration_digest_from_values`
   (`_cli_failure_tracker.py:101-113`) and the state-compatibility check
   (`_cli_state_management.py:210,336`), so switching format invalidates
   continuation rather than silently reusing outputs of the other kind.

### 5.5 The full-run bundle store is not affected

`--mode full` and `--mode measure` are untouched. `--process-format` is rejected
with a `UsageError` outside `--mode process`, mirroring how `--layer` already
behaves (`_cli_process_single.py:589-596`).

---

## 6. Consolidated metadata

Written into the root `zarr.json` on promote.

Opening a store costs one GET per metadata file — 8 of the 12 — which is the
latency that matters when the destination is object storage and a viewer
enumerates many stores. `zarr.consolidate_metadata` collapses that to one.

**It is legal, and the reason is precise.** The Zarr v3 core specification:

> An implementation MUST fail to open Zarr groups or arrays if any metadata
> fields are present which (a) the implementation does not recognize and (b) are
> not explicitly set to `"must_understand": false`.
>
> An extension object MAY explicitly set `must_understand=False` if
> implementations can ignore its presence.

Verified by execution against zarr 3.1.5 that the serialised block carries it:

```text
consolidated_metadata KEYS: ['kind', 'metadata', 'must_understand']
must_understand present : True -> False
```

So a conformant reader that does not recognise `consolidated_metadata` is
*required by the specification to ignore it*, not to fail. `zarr-python` emits a
warning that the feature "is currently not part in the Zarr format 3
specification"; that is accurate but narrow — it means *not core spec*, not
*non-conformant*. It is a proposed formal extension
([zarr-specs #309](https://github.com/zarr-developers/zarr-specs/pull/309),
[issue #136](https://github.com/zarr-developers/zarr-specs/issues/136)).

Three further properties, all verified by execution:

- The key is a **top-level sibling of `attributes`**, not nested inside it. So
  `attributes.ome` and `attributes.phenotypic` survive untouched and
  `ngff_.read_root_attributes` (`:589`), which does
  `payload.get("attributes", {})`, needs no change.
- It **adds no files**. Per-node `zarr.json` documents all remain, so a reader
  that ignores the key walks the tree correctly.
- The staleness failure mode does not arise. A process-mode store is written
  once into a `.part` and promoted by rename; it is never mutated in place, so
  the consolidated view cannot drift from the tree it describes.

**Consolidation happens inside the `.part`, before the promote — never after.**
An earlier draft consolidated the returned store path, which rewrites the root
`zarr.json` **at the final path** and so reintroduces exactly the failure this
design claims to make unreachable (§"Why change" #2: *"a store either has its
root `zarr.json` or does not exist"*). It would also land after
`promote_store`'s optional `fsync`, leaving the consolidated root non-durable
under SLURM. `_save_store` / `_write_store_part` therefore take
`consolidate: bool = False`, applied immediately before `ngff_.promote_store`.

Two `zarr-python` warnings are suppressed at that call site, with a comment
citing this section rather than a global filter: the consolidated-metadata
notice, and `"Object at METADATA.ome.xml is not recognized as a component of a
Zarr hierarchy"` — the latter fires once per image at AutoConvertRaw scale.
Both are `ZarrUserWarning`, a `UserWarning` subclass, so filtering on the class
covers both.

---

## 7. Re-ingest

Both halves are required for the loop to close; either alone leaves the artifact
unusable as input.

### 7.1 `imread` gains a directory branch

`imread`'s dispatch becomes: a directory whose name ends in `ngff_.STORE_SUFFIX`
-> the §4 resolver; otherwise the existing suffix dispatch, unchanged. The
`UnsupportedFileTypeError` path is preserved for everything else.

### 7.2 The scanner gains a non-recursive store match

`_cli_directory_scanner` matches `*.ome.zarr` **directories**, non-recursively,
alongside its existing suffix match on files. The 2026-08-18 design records this
exact trap against a sibling site (§4.4): *"a naive port recurses INTO every
store: 400k stat calls at 10k images."* A store is a directory full of files, so
`rglob` descends into all of them. The same bug class applies verbatim here, and
a test asserts non-recursion by stat count rather than by output equality — an
`rglob` port produces the same file list and only differs in cost.

### 7.3 Work IDs, and two ways a store input breaks them

Work IDs are derived from the input path relative to `--input`. Two things break
on a directory input, both found 2026-08-27 and both in scope.

**`file_sha256` opens the input as a file.**
[`_cli_failure_tracker.py:92-98`](../../../../src/phenotypic/_cli/_cli_failure_tracker.py)
does `with path.open("rb")`, and is called with the input image path from
`work_id_for_image` (`:205`), the SLURM identity ledger
(`_cli_slurm_array_scripts.py:381`, once per image at submit time), and
`_cli_process_single.py:141,624`. A `*.ome.zarr` input is a directory, so this
raises `IsADirectoryError` — which would break the whole of §7 and this design's
own end-to-end criterion.

The store branch digests the **whole tree**: every member's store-relative path
and content, in sorted path order.

> **Corrected 2026-08-28 (found by execution during the CLI phase).** This
> section originally specified digesting the **root `zarr.json`** alone, on the
> reasoning that the promote protocol writes it last so it fingerprints
> completeness, and that it "changes whenever any published content does".
>
> The first half is true. **The second is false.** The root carries the schema
> version, the series map, the pyramid geometry, the metadata sections and the
> provenance journal — none of which move when pixels do. Two stores holding
> entirely different images produced one digest:
>
> ```text
> pixels genuinely differ : True (mean 0.640 vs 0.500)
> shard bytes differ      : True
> root zarr.json identical: True
> file_sha256 differs     : False
> ```
>
> That silently breaks content-change detection for a store input. The digest
> feeds `work_id_for_image` and the SLURM identity ledger, so an edited store
> would keep its work ID and continuation would reuse stale output — while the
> flat-file path digests every pixel byte. A weaker guarantee on the newer path
> is exactly backwards.
>
> The dismissed cost was also wrong. A store holds roughly a dozen files whose
> bytes are the bytes an equivalent TIFF would carry, so digesting the tree
> reads about as much as digesting that TIFF, plus a directory walk — which is
> what buys the guarantee, not what wastes effort.

Paths are folded in alongside content, so moving a chunk between members or
renaming one changes the digest; sorting makes it independent of filesystem
iteration order. A directory that is not a store still raises
`IsADirectoryError` rather than acquiring an invented fingerprint.

**`Path.stem` yields `img.ome`.** `process_only_output_path` and the work-ID
sites must derive a store's stem with `sdk_.store_stem`
(`_io_constants.py:1531`), never `Path.stem` — otherwise a store-input run
writes `p01.ome.ome.zarr`. The helper's own docstring records why: `img.ome` is
*"a plausible-looking wrong name rather than an error"* that propagates into
parquet filenames and completion markers, so every image reprocesses forever.

### 7.4 Mixed input trees

A tree containing both flat images and stores scans as the union. No ordering or
precedence rule is introduced; if a `<stem>.tiff` and a `<stem>.ome.zarr` sit in
one directory they are two distinct inputs with two distinct work IDs, exactly
as `<stem>.tiff` and `<stem>.png` are today.

---

## 8. Affected modules

| File | Change |
|---|---|
| `sdk_/ngff_.py` | New `read_ngff_image_spec()` (the §4 resolver -- pure, no `Image` import); `rdefs` added to `build_omero` (§2.5); `build_phenotypic_attributes`'s `image_class` widens to `str \| None` and omits the key when `None` (§3.3). No `build_provenance()` -- the journal already exists (§2.3). |
| `_core/_provenance.py` | `pipeline_source_identity` gains a switch to record the pipeline **basename** instead of the resolved absolute path; `initialize_cli_provenance` threads it (§2.3.2b). |
| `_core/_image_parts/_image_io_handler.py` | `imread` gains the store branch (§7.1); `load_zarr` gains the `image_class` guard (§3.3); `_save_store` and `_build_store_attributes` thread `write_image_class`. |
| `_cli/_cli_process_only.py` | `process_only_output_path` becomes format-aware; `write_process_only_layer` gains the zarr branch, calling `_save_store(series=(layer,), write_objmap=False, write_image_class=False, levels=pyramid_level_count(h, w))`; `process_single_apply_only_core` calls `initialize_cli_provenance` before `pipeline.apply()` (§2.3.2a). |
| `_cli/_cli_process_single.py` | The `--process-format` option, the layer-dependent default resolution, and the objmap guard (§5). |
| `_cli/_cli_directory_scanner.py` | Non-recursive `*.ome.zarr` directory match beside the existing suffix match (§7.2), on both `scan_directory_structure` and `get_input_structure_summary` — the dry-run path a user runs first. |
| `phenotypicCLI.py` | Declares and resolves `--process-format`; passes it to the two `process_only_output_path` sites at `:450` and `:954` (§5.4). |
| `_cli/_cli_types.py` | `ExecutionConfig` carries the resolved format (`:185`). |
| `_cli/_cli_execution_strategies.py` | Threads the format to `process_only_output_path` (`:159`) and to `process_single_apply_only_core` (`:578`) (§5.4). |
| `_cli/_cli_slurm_array_scripts.py` | Passes `--process-format` on the worker command line (`:297-303`); store-aware digest in the identity ledger (`:381`) (§7.3). |
| `_cli/_cli_failure_tracker.py` | `file_sha256` gains a store branch (§7.3); the format joins `processing_configuration_digest_from_values` (§5.4). |
| `_cli/_cli_state_management.py` | The format joins the continuation compatibility check (`:210,336`) (§5.4). |

`_cli/_cli_readme_generator.py` was listed here in an earlier draft and is
**not** affected: it documents nothing about process mode (`grep -n process`
returns only unrelated hits), and `phenotypicCLI.py:2324` skips
`output_manager.create_structure` for process runs, so the generator is never
reached on this path.

No new dependency is added. `zarr>=3.0` and `jsonschema` are already declared;
no `fsspec`/`s3fs`/`gcsfs` is introduced (§11).

---

## 9. Testing

**Conformance.** Every emitted store is validated against the published NGFF
JSON schemas via `jsonschema`, reusing the existing gate (2026-08-18 decision
#11). Covers each `--layer` and both the rgb (3-axis) and 2-D forms.

**Round trip.** `imread(save(x))` is bit-exact against the source array for each
layer, including the `(3,H,W)` -> `(H,W,3)` transpose and `bit_depth` recovery.

**Third-party fixtures.** Synthetic stores exercising case C, each asserting the
specific §4 outcome rather than merely "raises":

| Fixture | Expected |
|---|---|
| 5-D `tczyx`, `t=1 z=1 c=3` | reads as RGB |
| 5-D, `t=10` | raises; `t=` override reads |
| `c=5` | raises; `c=` override reads |
| HCS `ome.plate` root | raises, naming the well-path remedy |
| `bioformats2raw.layout`, no `series` | reads group `"0"` |
| no `phenotypic` block at all | reads; `require_readable_store` never called |
| `store_schema_version: 4` | `imread` reads; `load_zarr` raises |

**Refusals.** `load_zarr` on a process-mode store raises, and on a
third-party store with no `phenotypic` block raises the same guard rather than a
bare `KeyError` (§3.3). A regression test pins that it does **not** return a
degraded `Image`, which is today's behaviour. `--layer objmap --process-format
zarr` raises `UsageError`. `--process-format` outside `--mode process` raises.

**Scanner.** Non-recursion asserted by stat count against a tree of stores.

**Consolidation.** `attributes.phenotypic` survives consolidation byte-identical;
file count is unchanged; a reader ignoring the key resolves the same arrays.

### Logic validation

Per the repository's Agentic AI File Rules, an executable script at
`docs/superpowers/logic_validation_scripts/2026-08-27-process-mode-ome-zarr/store_geometry.py`
re-derives, stdlib + numpy only and importing no `phenotypic` code:

- the 12-file count for a single-series `rgb` store at 4000x3000;
- `pyramid_level_count` and the level shapes it implies;
- that the shard shape is an exact multiple of the chunk shape in every
  dimension including the channel axis, which the Zarr v3 sharding codec
  requires.

It exits non-zero on failure and is committed alongside this spec.

---

## 10. Non-goals

- Multi-series or objmap-bearing process output (decisions 1 and 6).
- Zipped stores / NGFF RFC-9 (decision 2).
- Reading NGFF metadata back to reconstruct PhenoTypic *state* from a
  third-party store. `imread` reads pixels and geometry; nothing more.
- Changing the full-run bundle store, the staged-GPU commit protocol, or
  `--mode migrate`.
- HCS `plate` support. Detected and refused with a pointer, not handled.
- Reading stores over a URL (`s3://`, `https://`). `imread` takes a local path.
  See §11.
- Measurements in `image-label.properties` (inherited from 2026-08-18 #10).

## 11. Deferred, with reasons

**`omero` on float series.** §2.6. Reopened for this design and re-deferred: the
two candidate fixes are data-dependent metadata or a demonstrably wrong fixed
window, and the layer that matters for the reference use case is integer. Revisit
if a float layer becomes a primary external deliverable.

**`source_sha256` in provenance.** §2.3. Costs a full read per image at
AutoConvertRaw's scale with no current consumer.

**NGFF 0.4 / Zarr v2 input.** Refused by name, not converted. A v2 group is
spelled `.zgroup`/`.zattrs` where v3 writes `zarr.json`, so a 0.4 store has no
root by this design's reader and would otherwise surface as the bare
`FileNotFoundError` that means "interrupted write, store absent" — a misdiagnosis
of exactly the §3.1 case C stores that are most common in the wild, since
`bioformats2raw`'s default output and QuPath's export are both 0.4/v2 today. The
reader therefore detects the v2 marker and raises `ValueError` naming NGFF 0.5 /
Zarr v3 as the requirement. Converting in-process is out of scope: it is a
whole-store rewrite with its own chunking, sharding and pyramid decisions, all of
which §1 settles for *written* stores only, and mature external converters
already exist. Revisit if 0.4 input turns out to be the common case rather than
the legacy one.

**Cloud URL input.** The destination is object storage, so
`imread("s3://bucket/…/p01.ome.zarr")` is the natural next step, and zarr 3 plus
`fsspec` makes it reachable. It is out of scope here: PhenoTypic declares no
`fsspec`/`s3fs`/`gcsfs` dependency today (verified), the CLI's path handling,
work-ID derivation, and output mirroring all assume local paths, and remote
input is a change to the run model rather than to this artifact. Nothing in this
design forecloses it — the §4 resolver takes a store root and does not care how
its bytes arrive.

## 12. Consequences outside this repository

Recorded because this design's default (decision 5) changes the output of a
command a live pipeline runs.

AutoConvertRaw invokes `phenotypic --mode process --layer rgb` and reaps
`<batch>_<NNNN>.tiff` (`src/worker_correct.sh:278,306`). Under decision 5 that
command emits `<stem>.ome.zarr/`, the reap finds nothing, and every image is
marked `cc_failed`.

It is insulated today: `ACR_PHENOTYPIC_PYTHON` points at AutoConvertRaw's own
non-editable venv, pinned to PhenoTypic 0.18.0 (`src/config.sh:455-473`). But
that file states the limit of the insulation itself:

> NOTE: the pin lives in .venv only. A future `uv sync` rebuilds it from the
> live checkout referenced by [tool.uv.sources] in pyproject.toml — freeze
> PhenoTypic to a git tag/commit or a wheel to make it durable.

The same file already documents this precise failure for the 0.16.0 -> 0.18.0
flag change: *"a downgrade reaps every image as cc_failed."*

**Action, outside this repository and not a task of this design:** pin
AutoConvertRaw to a PhenoTypic git tag or wheel before this lands. Migrating it
to consume stores is a separate change, and one this design's §7 makes
straightforward but does not perform.

## 13. Supersession to record

The 2026-08-18 design's **non-goal #1** — "Ingesting third-party OME-Zarr as
pipeline input (the projection is write-only)" — is superseded by this design
and must be marked as such there on approval, pointing here.

Its decision #6 ("OME projection is write-only") is **not** superseded and
remains true as written: nothing reads the OME projection back to reconstruct
PhenoTypic state. §4 reads NGFF geometry to place pixels, which is the same
class of act as reading a TIFF's IFD.
