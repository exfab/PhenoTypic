# Browse UX and preparation performance implementation plan

**Status:** Implemented on `codex/browse-ux-libvips`

**Goal:** Make Browse feel immediate during plate-series review while keeping source
rendering faithful, resource use bounded, and the portable Pillow tiler fully
supported.

**Scope:** Single-view navigation, OpenSeadragon lifecycle, progressive previews,
nearby-image preparation, explicit dataset preparation, persistent prepared-artifact
caching, source metadata probing, DZI backend selection, telemetry, tests, and the GUI
documentation ledgers.

**Planning basis:** The original behavior below was established by repository inspection.
The selected defaults are engineering decisions validated by the tests and local
performance fixtures listed in Phase 0. No external empirical performance claims are
assumed.

## Outcome

After this plan lands, a user can move through an image series with `J` and `K`, jump
ten images with shifted shortcuts, select from a bounded plate contact sheet, reuse one
OpenSeadragon viewer, optionally retain a same-sized plate viewport, and see a preview
while the deep-zoom pyramid becomes ready. Selected-image work always outranks bounded
background preparation. Users can explicitly prepare the rest of a dataset, stop queued
work, inspect progress, clear prepared data, and continue using the GUI when native
libvips is absent or fails.

## Why the current design needs a coordinated change

Established behavior:

- Browse currently sends up to six neighboring DZI requests concurrently after the
  selected image opens, without a server-side priority queue or concurrency cap
  ([`_callbacks.py`](../../../../src/phenotypic/gui/browse/_callbacks.py),
  [`browse.js`](../../../../src/phenotypic/gui/browse/_assets/browse.js)).
- A DZI request performs synchronous full-source normalization before tiling, and the
  thumbnail route repeats that normalization path
  ([`_tile_routes.py`](../../../../src/phenotypic/gui/browse/_tile_routes.py),
  [`_thumb_routes.py`](../../../../src/phenotypic/gui/browse/_thumb_routes.py)).
- Metadata currently performs another full `Image.imread(...).rgb[:]` decode
  ([`_metadata.py`](../../../../src/phenotypic/gui/browse/_metadata.py)).
- Browse wipes its prepared-image cache at app creation and exit
  ([`_source_render.py`](../../../../src/phenotypic/gui/browse/_source_render.py),
  [`_app.py`](../../../../src/phenotypic/gui/browse/_app.py)).
- Single-view navigation destroys and recreates OpenSeadragon, and the single view and
  Timeline popout share one JavaScript viewer handle
  ([`browse.js`](../../../../src/phenotypic/gui/browse/_assets/browse.js)).
- The pyvips path already falls back to Pillow on a missing Python binding or native
  loader error, but backend information is not exposed until tiling starts
  ([`_dzi_tiler.py`](../../../../src/phenotypic/gui/results_viewer/_dzi_tiler.py)).

The user proposal to continuously convert every DZI is therefore only partly effective.
It improves later cache hits, but unbounded conversion can compete with the selected
image for CPU, memory, and disk. The implementation will use constant background work
only while useful queued work exists, cap it at one speculative image, prioritize the
navigation direction, and require an explicit action before preparing the entire
dataset. [Based on general systems reasoning.]

## Architecture

```text
canonical picker selection
  -> revisioned source descriptor
  -> preview GET and foreground DZI GET
       -> BrowsePreparationManager
            priority 0: selected image
            priority 10: directional neighbors
            priority 20: explicit dataset remainder
       -> BrowseCache
            header-only metadata
            lightweight preview
            one faithful full normalization
            libvips DZI or Pillow fallback
            atomic immutable publication
  -> reused OpenSeadragon viewer
  -> bounded filmstrip and progress state
```

Two Browse-owned services are constructed once in
[`_app.py`](../../../../src/phenotypic/gui/browse/_app.py) and injected into routes and
callbacks:

1. `BrowseCache` in `browse/_cache.py` owns revision identity, cache paths,
   readiness, atomic publication, access markers, pruning, and persistent-to-temporary
   fallback.
2. `BrowsePreparationManager` in a new `browse/_preparation.py` owns per-revision
   deduplication, foreground coordination, the one-worker speculative queue, client
   generations, dataset jobs, progress snapshots, and bounded timing samples.

Do not add Redis, Celery, WebSockets, SQLite, a service worker, or a separate process.
The current GUI is a process-owned application. Filesystem locks and immutable entries
make artifact publication safe across processes, but scheduling limits and cancellation
remain process-local in this release.

## Fixed contracts

### Source identity

Add an immutable `SourceRevision` with these fields:

```python
@dataclass(frozen=True)
class SourceRevision:
    sandbox_id: str
    relative_path: str
    token: str
    revision: str
    original: Path
    size: int
    mtime_ns: int
    ctime_ns: int | None
```

`revision` is a fixed-length hash of the canonical sandbox identity, relative path,
`st_size`, `st_mtime_ns`, `st_ctime_ns` when available, the render-schema version, and
DZI parameters. The hash is a filesystem revision key, not a content hash. Preparation
must stat before and after conversion, publish only if the descriptors match, retry once
under the new revision, and return a safe `409` if the source keeps changing.

### Cache location and layout

Generalize the cache-root conventions in
[`results_viewer/_output_root.py`](../../../../src/phenotypic/gui/results_viewer/_output_root.py)
into a shared GUI cache-path helper. Resolve the Browse owner in this order:

1. `<sandbox>/.phenotypic-gui/browse_cache` when writable.
2. The platform user cache, namespaced by the canonical sandbox hash.
3. A session temporary directory with a warning when neither persistent location works.

The source lister must always exclude `.phenotypic-gui`, including when hidden entries
are otherwise shown.

```text
<cache-owner>/
  locks/<entry-key>.lock
  entries/<entry-key>/
    entry.json
    preview.png
    dzi/
      image.dzi
      image_files/...
```

Use a 10 GiB high-water mark and prune to an 8 GiB low-water mark at startup and after
successful publication. Touch access metadata at most once per hour. Pruning uses one
global eviction lock, skips locked/current/in-flight entries, and removes older revisions
before the newest revision of a source. A single entry larger than the quota remains
usable and causes older entries to be pruned. These first-release limits are constants,
not new CLI flags.

Generate artifacts under a sibling staging path. Publish preview files atomically,
publish the completed DZI directory atomically, and write `entry.json` with
`phase="ready"` last. Never serve a DZI without the ready marker. Remove the full-size
normalized PNG after complete DZI publication so persistent storage does not duplicate
the source pixels.

### Artifact and job state

Keep persistent artifact state separate from process-local work state:

```python
ArtifactPhase = Literal[
    "absent", "normalizing", "preview_ready", "tiling", "ready", "failed"
]
JobState = Literal[
    "queued", "running", "cancel_requested", "completed", "failed", "superseded"
]
```

Only stable artifact facts belong in `entry.json`. Queue generations, clients, and
cancellation are never restored after restart.

### Preparation manager

```python
class BrowsePreparationManager:
    def ensure_preview(self, source: SourceRevision) -> PreparedPreview: ...
    def preview_if_ready(self, source: SourceRevision) -> PreparedPreview | None: ...
    def ensure_dzi(self, source: SourceRevision) -> PreparedDzi: ...
    def replace_navigation_queue(
        self,
        *,
        client_id: str,
        generation: int,
        sources: Sequence[SourceRevision],
    ) -> None: ...
    def start_dataset_job(
        self,
        *,
        client_id: str,
        job_id: str,
        sources: Sequence[SourceRevision],
    ) -> None: ...
    def cancel_job(self, job_id: str) -> None: ...
    def snapshot(self, job_id: str | None = None) -> PreparationSnapshot: ...
    def shutdown(self) -> None: ...
```

Foreground preview and DZI requests run through one per-revision preparation record and
lock. That record has separate preview-ready and DZI-ready events/futures plus shared
failure and cancellation state; a completed preview must never satisfy a DZI waiter.
There is one speculative worker. It must not begin another task while foreground work is
waiting or active. A backend call already in progress may finish; cancellation removes
queued work and checks between preview, normalization, and tiling stages. The UI must say
“Stopping after current image” rather than promise immediate interruption.

Each browser tab gets a `client_id` in `sessionStorage`. Replacing navigation work or
cancelling generations is scoped to that client so one tab cannot cancel another.

### Route contract

Use revision-addressed, sandbox-validated same-origin routes. The paths below are relative
to the Browse app's configured `requests_pathname_prefix`; JavaScript must construct them
with the existing app-prefix helper so standalone `/`, hub `/browse/`, and proxy-prefixed
launches all work:

```text
GET    <browse-prefix>assets/<token>/<revision>/preview.png
GET    <browse-prefix>assets/<token>/<revision>/preview-if-ready.png
GET    <browse-prefix>assets/<token>/<revision>/image.dzi
GET    <browse-prefix>assets/<token>/<revision>/image_files/<level>/<tile>.png
POST   <browse-prefix>api/prepare/nearby
POST   <browse-prefix>api/prepare/dataset
GET    <browse-prefix>api/prepare/status/<job-id>
DELETE <browse-prefix>api/prepare/dataset/<job-id>
POST   <browse-prefix>api/cache/clear
```

Completed revisioned assets receive `Cache-Control: private, max-age=31536000,
immutable`. Status and error responses do not. Preparation endpoints re-resolve every
token under the configured sandbox and reject a revision that no longer matches.

The selected preview and DZI requests may run concurrently. The preview route uses a
bounded direct decoder where possible: pyvips thumbnailing for standard images, Pillow
draft/thumbnail behavior as the portable path, and an embedded RAW preview where rawpy
provides one. If no cheap preview is available, publish a preview after faithful
normalization and before DZI tiling. The final DZI remains authoritative because a RAW
embedded preview or platform decoder can differ from the faithful normalized pixels.
Filmstrip images use the cache-only preview route. A miss returns a no-store placeholder
or response and relies on the one-worker neighbor queue to prepare it; mounting nine
filmstrip images must never create nine foreground decodes.

### Backend contract

Refactor import-time backend selection behind an immutable, testable accessor:

```python
@dataclass(frozen=True)
class DziBackendInfo:
    name: Literal["pyvips", "pillow"]
    version: str | None
    fallback_reason: str | None
```

`auto` chooses pyvips when its binding and native library load. Missing binding or
native-library `OSError` selects Pillow. If `dzsave()` raises the pyvips-specific runtime
error, discard the staging output and retry that image once with Pillow. Permission,
disk, publication, and unexpected errors fail directly. Always record the actual backend
in `entry.json` and timings. Never mix both backends in one published entry.

The GUI extra installs `pyvips[binary]>=3.1.1` on macOS and Windows, while Linux and
other platforms retain `pyvips>=2.2` with optional system
[libvips](https://libvips.github.io/pyvips/README.html#non-conda-install). This avoids
loader-path setup on supported desktop wheels without making older Linux/HPC systems
resolve an incompatible binary wheel. Pillow remains installed and required.

### Navigation and display contract

- `BROWSE_IMAGE_PICKER.value` remains the only authoritative selection. Buttons,
  shortcuts, and filmstrip clicks only update it.
- `J` selects the previous image, `K` the next, `Shift+J` jumps back ten, and `Shift+K`
  jumps forward ten. Boundaries clamp without wrapping.
- Do not bind arrows globally. OpenSeadragon retains arrow-key panning.
- Ignore shortcuts in input, textarea, select, contenteditable, Dash combobox, visible
  modal, modified chord, hidden Single mode, or an already-prevented event. Coalesce
  repeat events at approximately 80 ms.
- Automatic candidates are `+1,+2,+3,-1,-2` while moving forward,
  `-1,-2,-3,+1,+2` while moving backward, and alternate outward when direction is
  unknown.
- Split `ns.singleViewer` and `ns.popoutViewer`. Reuse the mounted single viewer with
  `viewer.open(...)`; destroy a viewer only when its host is removed.
- Guard preview, `open`, and `open-failed` handlers with the active client generation.
- “Keep position” is opt-in and stored locally. Restore normalized center and zoom only
  when decoded dimensions match exactly; otherwise use `goHome()`.
- Render a centered filmstrip containing at most the active item plus four neighbors on
  each side. Use real buttons, lazy preview images, `aria-current`, visible filenames,
  focus rings, and quiet Ready/Preparing/Queued/Failed indicators.
- Pause starting speculative work when Timeline is active, the document is hidden, or
  the browser is offline.

## Instrumentation and success criteria

Record structured local logs and a bounded in-memory timing window. Do not add remote
telemetry or an analytics dependency. Avoid absolute source paths in logs.

Fields:

- source probe, preview, normalization, queue wait, DZI, manifest, and
  selection-to-OpenSeadragon-open durations;
- cache hit/miss, revision, decoded dimensions, task priority/reason;
- requested backend, actual backend/version, and sanitized fallback reason.

Acceptance is measured against the Phase 0 baseline on the same fixture set:

- each cold revision performs one faithful full decode;
- simultaneous preview, DZI, metadata, and warm requests publish one artifact set;
- a warm manifest request performs no decode or tiling;
- warm-navigation p50 and p95 improve relative to baseline;
- selected-image work is never queued behind dataset preparation;
- speculative preparation concurrency never exceeds one;
- cache pruning returns below the low-water mark except for one oversize entry;
- forced Pillow and native libvips both create valid, OpenSeadragon-readable DZI geometry.

Do not require byte-identical tiles across Pillow and libvips because their resampling
paths can differ. Validate dimensions, levels, coverage, and declared pixel tolerances.

## Phase files and order

| Phase | File | Shipping boundary |
|---|---|---|
| 0 | [`phase-0-baseline-backend-docs.md`](phase-0-baseline-backend-docs.md) | Benchmark baseline, backend contract, optional libvips docs |
| 1 | [`phase-1-cache-and-source-probe.md`](phase-1-cache-and-source-probe.md) | Persistent revisioned cache, atomic artifacts, metadata fix |
| 2 | [`phase-2-preparation-manager.md`](phase-2-preparation-manager.md) | Deduplicated priority queue, revisioned routes, dataset jobs |
| 3 | [`phase-3-navigation-and-viewer.md`](phase-3-navigation-and-viewer.md) | J/K navigation, viewer reuse, preview, keep-position |
| 4 | [`phase-4-filmstrip-and-controls.md`](phase-4-filmstrip-and-controls.md) | Plate contact sheet, progress/stop/clear chrome, accessibility |
| 5 | [`phase-5-instrumentation-docs-release.md`](phase-5-instrumentation-docs-release.md) | Metrics, ledgers, screenshots, CI, cross-platform release gate |

Implement in order. Phase 1 is the correctness prerequisite for every additional source
request. Phase 2 must land before the browser stops using its current manifest-fetch
warming behavior. Phases 3 and 4 then build on one canonical selection and preparation
state model.

## Deliberate exclusions

- No silent full-dataset conversion.
- No more than one speculative worker until measurements justify a change.
- No direct libvips tiling of original or RAW sources. Faithful `Image.imread`
  normalization remains the authoritative DZI input.
- No global arrow, Home, or End shortcuts.
- No persistent job database or cross-process queue coordination.
- No service worker, IndexedDB, or remote telemetry.
- No claim that Stop interrupts an opaque in-progress native encode.

## Execution and independent review protocol

For each phase:

1. Implement the phase tasks test-first and run its focused verification.
2. Run Ruff only against explicit changed paths.
3. Run mypy against `src/phenotypic` at the phase boundary.
4. Dispatch an independent code-review subagent for the phase diff, as required by the
   project instructions. Fix findings and rerun the focused tests before proceeding.
5. Update affected `FEATURES.md` rows in the same shipping phase. Do not leave a shipping
   row without a resolvable test reference.

After all phases, run a full independent review, then the final commands in Phase 5.
