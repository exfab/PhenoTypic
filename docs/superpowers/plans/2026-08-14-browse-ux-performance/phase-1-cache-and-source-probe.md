# Phase 1: Persistent cache, atomic artifacts, and source probing

**Shipping boundary:** Current on-demand Browse remains visually unchanged, but prepared
entries persist safely, simultaneous requests deduplicate, source replacement invalidates
old URLs, and metadata no longer performs a full pixel decode.

## Task 1.1: Introduce shared GUI cache paths

Files:

- New `src/phenotypic/gui/_cache_paths.py`
- `src/phenotypic/gui/results_viewer/_output_root.py`
- `src/phenotypic/gui/_config.py`
- New `tests/gui/test_cache_paths.py`

Steps:

- [ ] Extract the existing platform user-cache convention into the shared module without
  changing Results paths.
- [ ] Add `SANDBOX_BROWSE_CACHE_SUBDIR`, schema version, 10 GiB high-water, 8 GiB
  low-water, access-touch interval, crash-cleanup grace period, nearby radius, and one
  speculative worker as shared constants.
- [ ] Resolve Browse cache ownership in the fixed fallback order from the index.
- [ ] Namespace user-cache entries by canonical sandbox hash.
- [ ] Test macOS, Windows, XDG Linux, writable sandbox, read-only sandbox, and temporary
  fallback path selection without mutating real user directories.

## Task 1.2: Add immutable source revisions and cache entries

Files:

- New `src/phenotypic/gui/browse/_cache.py`
- `src/phenotypic/gui/browse/_source_render.py`
- `src/phenotypic/gui/browse/_source_lister.py`
- `tests/gui/browse/test_source_render.py`
- New `tests/gui/browse/test_cache.py`

Steps:

- [ ] Implement `SourceRevision`, fixed-length entry keys, cache paths, and revisioned
  public URLs. Tokens remain opaque routing capabilities, not cache filenames.
- [ ] Use `st_size`, `st_mtime_ns`, optional `st_ctime_ns`, sandbox identity, relative
  path, render schema, and DZI parameters in the revision.
- [ ] Use the existing cross-platform `exclusive_path_lock` for entry publication.
- [ ] Stat before and after conversion. Discard and retry once if the source changes;
  never publish a stale conversion under the new revision.
- [ ] Generate under a sibling staging path, atomically publish preview and DZI, and
  publish `entry.json` ready state last.
- [ ] Remove abandoned staging entries after the grace period, not immediately while a
  different process may still own a lock.
- [ ] Stop wiping the cache in `_app.py` and at `atexit`.
- [ ] Always exclude `.phenotypic-gui` from source discovery.
- [ ] Delete the normalized full-size PNG only after complete DZI publication.

Required tests:

- [ ] Two threads and two processes requesting one revision publish one complete entry.
- [ ] More distinct concurrent keys than the old process-lock cache bound cannot bypass
  same-entry serialization.
- [ ] Mutation during normalization and during tiling cannot publish a stale entry.
- [ ] A simulated crash after preview and before DZI is cleaned or resumed safely.
- [ ] Fixed-length cache components remain Windows-safe for long relative source paths.
- [ ] A completed entry survives cache-object and app recreation.

## Task 1.3: Add bounded eviction and explicit clear semantics

- [ ] Track access at selected-manifest granularity, touching at most once per hour.
- [ ] Prune under one eviction lock at startup and after successful publication, never
  on each tile response.
- [ ] Skip locked, selected, and in-flight entries. Prefer stale source revisions.
- [ ] Prune from 10 GiB to 8 GiB. Define the oversize-single-entry behavior from the
  index and test it.
- [ ] Add a cache `clear(protected_revisions=...)` method for the later UI. It preserves
  the current revision and returns removed bytes, removed entries, and failures.
- [ ] Unit-test quota accounting, access ordering, locked-entry skipping, partial-entry
  grace, clear behavior, and an unwritable persistent owner.

## Task 1.4: Remove metadata's second full decode

Files:

- `src/phenotypic/gui/browse/_metadata.py`
- `src/phenotypic/gui/browse/_capture_time.py`
- New `src/phenotypic/gui/browse/_source_probe.py`
- `tests/gui/browse/test_metadata.py`
- `tests/gui/browse/test_capture_time.py`

Steps:

- [ ] Build one cached, best-effort source probe keyed by the same `SourceRevision`
  identity fields as prepared artifacts, including optional `ctime_ns`.
- [ ] Read size from stat, standard image dimensions from Pillow headers, capture/camera
  data from the existing EXIF path, and RAW dimensions from header metadata when
  available.
- [ ] Leave an unavailable field blank until normalization rather than forcing a pixel
  decode.
- [ ] Reuse the probe for Timeline capture-time ordering and selected-image metadata.
- [ ] Monkeypatch `Image.imread` to fail and prove standard metadata still succeeds.
- [ ] Prove cache invalidation on size, `mtime_ns`, or available `ctime_ns` change.

## Verification

```bash
uv run pytest tests/gui/browse/test_source_render.py \
  tests/gui/browse/test_cache.py tests/gui/browse/test_metadata.py \
  tests/gui/browse/test_capture_time.py tests/gui/test_cache_paths.py -v
uv run mypy src/phenotypic
```

Run Ruff with the explicit source and test paths changed in this phase.

## Exit criteria

- Warm prepared entries survive process restart and remain revision-correct.
- Incomplete entries are never served.
- Every cold revision has one faithful full decode regardless of simultaneous route
  requests.
- Cache usage is bounded and a read-only sandbox degrades to user or temporary cache.
- Metadata does not access `Image.imread(...).rgb[:]`.
