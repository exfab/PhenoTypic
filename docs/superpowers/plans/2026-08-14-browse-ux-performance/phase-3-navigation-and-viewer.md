# Phase 3: Keyboard navigation, viewer reuse, preview, and viewport retention

**Shipping boundary:** High-frequency Single-view review is keyboard-friendly, avoids
OpenSeadragon reconstruction, shows preparation progress visually, and remains safe under
rapid navigation.

## Task 3.1: Keep one canonical navigation state

Files:

- `src/phenotypic/gui/_shared/_picker_navigation.py`
- `src/phenotypic/gui/browse/_ids.py`
- `src/phenotypic/gui/browse/_layout.py`
- `src/phenotypic/gui/browse/_callbacks.py`
- `tests/gui/browse/test_callbacks_helpers.py`
- `tests/gui/browse/test_layout.py`

Steps:

- [ ] Extend the shared picker helper with a clamped arbitrary delta/target operation
  while preserving the one-step wrapper used by Results.
- [ ] Keep `BROWSE_IMAGE_PICKER.value` authoritative. Previous/next buttons, keyboard
  events, and future filmstrip buttons must only update that value.
- [ ] Add an `N of M` readout and preparation status region. Announce selected position
  politely; do not announce every background item transition.
- [ ] Add stores for keyboard events, client generation, current source revision,
  preparation status, and the keep-position preference.
- [ ] Test empty, one-image, boundary, jump-ten, stale-generation, and source-refresh
  behavior.

## Task 3.2: Add scoped J/K shortcuts

Files:

- `src/phenotypic/gui/browse/_assets/browse.js`
- `src/phenotypic/gui/browse/_layout.py`
- New `tests/e2e/gui/test_browse_single.py`

Steps:

- [ ] Bind `J`, `K`, `Shift+J`, and `Shift+K` to the canonical picker path.
- [ ] Clamp at dataset ends; do not wrap.
- [ ] Ignore Single shortcuts while Single mode is hidden, a visible modal is open, or
  the target is input, textarea, select, contenteditable, or a Dash combobox.
- [ ] Ignore Ctrl/Meta/Alt chords and already-prevented events.
- [ ] Leave arrow, Home, and End behavior unchanged so OpenSeadragon and the page retain
  their native controls.
- [ ] Act on the first physical press immediately and coalesce repeat events at roughly
  80 ms so a held key does not enqueue every intermediate conversion.
- [ ] Put visible `J`/`K` hints in the stepper controls and descriptive titles/ARIA labels.

## Task 3.3: Reuse and isolate OpenSeadragon instances

Files:

- `src/phenotypic/gui/browse/_assets/browse.js`
- `tests/e2e/gui/test_browse_single.py`
- `tests/e2e/gui/test_browse_timeline.py`

Steps:

- [ ] Replace shared `ns.viewer` with `ns.singleViewer` and `ns.popoutViewer`.
- [ ] Construct the Single viewer once per mounted host and navigate with
  `singleViewer.open(dziUrl)`.
- [ ] Destroy only when the host is removed. Keep Timeline popout lifecycle independent.
- [ ] Increment a client image generation before every open. Ignore stale `open`,
  `open-failed`, and preview-load handlers.
- [ ] Expose a stable, test-only viewer seam analogous to the Compare viewer seam.
- [ ] Prove twenty changes retain one Single viewer and a bounded canvas count.
- [ ] Prove Single and Timeline popout can coexist without destroying each other.

## Task 3.4: Add progressive preview

Files:

- `src/phenotypic/gui/browse/_layout.py`
- `src/phenotypic/gui/browse/_assets/browse.css`
- `src/phenotypic/gui/browse/_assets/browse.js`
- `tests/e2e/gui/test_browse_single.py`

Steps:

- [ ] Add a decorative preview image behind the OSD canvas with a revisioned URL.
- [ ] On selection, show the previous frame until the matching new preview loads, then
  report “Preview shown, preparing deep zoom.”
- [ ] Fade only after the matching DZI fires `open`; keep the preview visible on DZI
  failure.
- [ ] Respect `prefers-reduced-motion` and avoid exposing duplicate alternative text.
- [ ] Test a delayed DZI, fast preview, stale preview arrival, preview failure, DZI
  failure, and reduced-motion behavior.

## Task 3.5: Add opt-in keep-position

- [ ] Add a locally persisted “Keep position” toggle, default off.
- [ ] Before `viewer.open`, capture normalized center, zoom, and decoded dimensions.
- [ ] Restore after the matching open only when the old and new dimensions match exactly.
  Otherwise call `goHome()`.
- [ ] Reset retained state after an open failure or explicit OSD Home action.
- [ ] Test same-size restore, different-size reset, disabled behavior, and stale open
  callbacks.

## E2E assertions

The new `test_browse_single.py` must cover:

- J/K and shifted jumps update dropdown, metadata, position, preview, and viewer.
- Shortcuts do nothing inside editing controls, Timeline, or a visible modal.
- Arrow keys pan OSD without changing selection.
- Held-key bursts coalesce to the final selection with bounded manifest requests.
- Viewer reuse and generation fencing prevent stale UI.
- Preview remains until the matching DZI opens.
- Keep-position restores only across exact dimensions.

## Verification

```bash
uv run pytest tests/gui/browse/test_layout.py \
  tests/gui/browse/test_callbacks_helpers.py -v
PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_single.py \
  tests/e2e/gui/test_browse_timeline.py -v
uv run mypy src/phenotypic
```

## Exit criteria

- Every navigation surface converges on one picker value.
- Rapid navigation cannot display an older preview, success, or error state.
- OpenSeadragon arrows still pan.
- Single and popout viewers have independent lifecycles.
- Preview and keep-position behaviors pass real-browser tests.
