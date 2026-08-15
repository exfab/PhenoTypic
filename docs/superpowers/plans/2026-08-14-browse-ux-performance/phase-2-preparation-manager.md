# Phase 2: Priority preparation manager and revisioned routes

**Shipping boundary:** Browse no longer launches an unbounded burst of manifest requests.
Foreground work, directional neighbors, and explicit dataset jobs share one deduplicated
state model with bounded background concurrency.

## Task 2.1: Implement deterministic manager state

Files:

- New `src/phenotypic/gui/browse/_preparation.py`
- `src/phenotypic/gui/browse/_app.py`
- `src/phenotypic/gui/_config.py`
- New `tests/gui/browse/test_preparation.py`

Steps:

- [ ] Implement separate artifact and job states from the plan index.
- [ ] Inject cache, clock, backend, and executor. Provide a synchronous fake executor for
  deterministic tests.
- [ ] Deduplicate work by revision across foreground, neighbor, filmstrip, and dataset
  callers with one staged record containing distinct preview-ready and DZI-ready
  events/futures plus shared failure/cancellation state.
- [ ] Use priority order: selected image, directional neighbors, dataset remainder.
- [ ] Run one speculative worker. Do not begin another speculative task while foreground
  work is queued or active.
- [ ] Scope generations and cancellation to a `sessionStorage` browser-tab client ID.
- [ ] Replacing a navigation generation drops superseded queued work and reprioritizes
  matching revisions already queued.
- [ ] Cancellation checks between preview, normalization, and DZI stages. An opaque
  running backend call may finish.
- [ ] Register orderly shutdown from `_app.py` without deleting prepared entries.

Required scheduler tests:

- [ ] Selected work always begins before queued dataset work.
- [ ] Directional order changes when navigation reverses.
- [ ] Duplicate revisions create one future and one published result.
- [ ] At most one speculative task executes.
- [ ] Two tabs cannot cancel each other's generations.
- [ ] Stop removes pending batch work and reports `cancel_requested` until the active
  stage finishes.
- [ ] Failures are terminal in the job count but retryable through a later request.
- [ ] `ready + failed == total` when a dataset job reaches a terminal state.

## Task 2.2: Add lightweight preview preparation

Files:

- `src/phenotypic/gui/browse/_source_render.py`
- `src/phenotypic/gui/browse/_thumb_routes.py`
- `tests/gui/browse/test_thumb_routes.py`

Steps:

- [ ] Add a bounded direct preview decoder: pyvips thumbnailing for supported standard
  images, Pillow draft/thumbnail fallback, and rawpy embedded thumbnails where present.
- [ ] Normalize preview orientation and cap both dimensions and output bytes.
- [ ] Treat previews as transient approximations; never use them as the DZI source.
- [ ] If no cheap preview path is available, create the preview after faithful
  normalization and publish it before tiling begins.
- [ ] Make selected-preview requests foreground. Filmstrip previews must be cache-only or
  queue-backed and must never become independent foreground decodes when their `<img>`
  elements mount.
- [ ] Test standard, RAW-with-embedded-preview, RAW-without-preview, corrupt source,
  missing optional decoder, and source mutation cases.

The lightweight preview may perform its own bounded decode. The invariant is one
faithful full normalization per source revision, not zero auxiliary preview reads.

## Task 2.3: Migrate DZI and preview routes

Files:

- `src/phenotypic/gui/browse/_tile_routes.py`
- `src/phenotypic/gui/browse/_thumb_routes.py`
- `src/phenotypic/gui/browse/_app.py`
- `tests/gui/browse/test_tile_routes.py`
- `tests/gui/browse/test_thumb_routes.py`

Steps:

- [ ] Add the revisioned asset URLs from the index and preserve traversal/symlink guards.
  Register internal paths relative to the Browse app and construct browser URLs from its
  configured prefix; cover standalone, hub, and custom proxy prefixes.
- [ ] Re-resolve the source token and compare its current revision on every route entry.
- [ ] Make the selected-preview GET wait only for preview readiness, the cache-only
  filmstrip preview return without starting work, and DZI GET wait for complete DZI.
- [ ] Add immutable private cache headers only for completed revisioned assets.
- [ ] Return safe status semantics: `404` unknown token/artifact, `409` source changed,
  `422` unsupported source, and fixed client-safe `500` messages.
- [ ] Add `Server-Timing` for cache lookup, queue wait, normalization, and DZI when the
  request owns those stages.
- [ ] Keep detailed paths and tracebacks in server logs only.

## Task 2.4: Add preparation APIs

Files:

- New `src/phenotypic/gui/browse/_preparation_routes.py`
- `src/phenotypic/gui/browse/_app.py`
- `tests/gui/browse/test_preparation_routes.py`

Steps:

- [ ] Implement nearby queue replacement, dataset start, status, dataset stop, and cache
  clear routes using same-origin JSON.
- [ ] Resolve every submitted token inside the configured sandbox; never trust a client
  path, revision, count, or state.
- [ ] Make dataset start idempotent per client/job ID.
- [ ] Return exact item counts but indeterminate per-image native-stage progress.
- [ ] Protect the current revision from cache clear and report removed bytes/entries.
- [ ] Test malformed JSON, stale revision, mixed valid/invalid tokens, job ownership,
  source refresh, and prefix-mounted URLs.

## Task 2.5: Replace browser manifest warming

Files:

- `src/phenotypic/gui/browse/_assets/browse.js`
- `src/phenotypic/gui/browse/_callbacks.py`
- `tests/gui/browse/test_callbacks_helpers.py`

Steps:

- [ ] Remove the loop that fetches every neighbor `.dzi` concurrently.
- [ ] Publish active index, total, revision, client generation, inferred direction, and
  ordered candidates in the current-image payload.
- [ ] Send one nearby queue replacement request per settled selection.
- [ ] Pause new speculative work in Timeline mode, while the document is hidden, or while
  the browser reports offline; resume by sending the latest generation.
- [ ] Debounce rapid selection changes so intermediate generations are retired before
  expensive work starts.
- [ ] Reuse the immutable source inventory from the initial directory scan instead of
  rescanning it for Timeline activation.

## Verification

```bash
uv run pytest tests/gui/browse/test_preparation.py \
  tests/gui/browse/test_preparation_routes.py tests/gui/browse/test_tile_routes.py \
  tests/gui/browse/test_thumb_routes.py tests/gui/browse/test_callbacks_helpers.py -v
uv run mypy src/phenotypic
```

## Exit criteria

- Browser navigation creates no background manifest-fetch burst.
- Foreground and speculative callers deduplicate through one revision future.
- One speculative conversion is the hard default under both backends.
- Dataset jobs are explicit, observable, stoppable between stages, and lower priority
  than active navigation.
- Revisioned routes remain sandbox-safe and never serve partial artifacts.
