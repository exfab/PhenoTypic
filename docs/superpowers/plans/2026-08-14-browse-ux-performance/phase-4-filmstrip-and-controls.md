# Phase 4: Plate contact sheet and preparation controls

**Shipping boundary:** Nearby images and preparation state are visible without rendering
the whole dataset, and explicit prepare, stop, and clear actions are understandable and
accessible.

## Task 4.1: Add the bounded plate contact sheet

Files:

- `src/phenotypic/gui/browse/_ids.py`
- `src/phenotypic/gui/browse/_layout.py`
- `src/phenotypic/gui/browse/_callbacks.py`
- `src/phenotypic/gui/browse/_assets/browse.css`
- `src/phenotypic/gui/browse/_assets/browse.js`
- `tests/gui/browse/test_layout.py`
- `tests/gui/browse/test_callbacks_helpers.py`
- `tests/e2e/gui/test_browse_single.py`

Steps:

- [ ] Render a centered window of at most four images before and after the active image.
- [ ] Use real buttons with visible/truncated filenames, decorative preview images,
  `aria-label="Open <filename>"`, and `aria-current="true"` on exactly one item.
- [ ] Use lazy revisioned cache-only previews and preparation indicators for Ready,
  Preparing, Queued, and Failed. A cache miss must wait for the one-worker neighbor queue,
  not promote each mounted `<img>` to foreground preparation.
- [ ] Filmstrip clicks update `BROWSE_IMAGE_PICKER.value` and do not introduce another
  selected-image store.
- [ ] Center where possible and clamp at dataset ends. Scroll the active button into
  view only when selection or focus changes, not on every status update.
- [ ] Keep visible focus rings, touch-sized controls, responsive horizontal overflow,
  and reduced-motion behavior.
- [ ] Prove the mounted DOM stays at nine or fewer image buttons regardless of dataset
  size.

## Task 4.2: Add explicit dataset preparation

Files:

- `src/phenotypic/gui/browse/_layout.py`
- `src/phenotypic/gui/browse/_callbacks.py`
- `src/phenotypic/gui/browse/_assets/browse.js`
- `tests/gui/browse/test_layout.py`
- `tests/e2e/gui/test_browse_single.py`

Steps:

- [ ] Add “Prepare dataset” with an estimate of image count and, when cheaply available,
  approximate cache impact. Do not promise a precise compressed size.
- [ ] Require explicit confirmation when estimated work cannot fit below the current
  quota. Explain that old prepared entries may be evicted.
- [ ] Show completed/total/failed counts and current stage. Per-image native progress is
  indeterminate.
- [ ] Replace Prepare with Stop while a batch is active. After Stop, say “Stopping after
  current image” until the opaque stage exits.
- [ ] Keep completed entries after Stop.
- [ ] Ensure selected-image and directional work can overtake remaining dataset work.
- [ ] Keep progress scoped to the browser tab/job generation so an old job cannot update
  a new source selection.

## Task 4.3: Add cache clear and backend details

- [ ] Add “Clear prepared images” to secondary Browse controls, not the primary
  navigation row.
- [ ] Require confirmation that reports the current prepared size.
- [ ] Preserve the currently displayed/in-flight revision and report entries/bytes that
  could not be removed.
- [ ] Surface active backend, cache location class, used/quota bytes, and fallback reason
  in a status popover or tooltip. Do not make backend detail permanent visual noise.
- [ ] Do not expose absolute cache or source paths to the browser.

## Task 4.4: Accessibility and interaction QA

- [ ] Tab through stepper, dropdown, keep-position, prepare/stop, filmstrip, and clear
  controls in logical order.
- [ ] Confirm status updates do not steal focus and high-frequency background progress is
  not repeatedly announced.
- [ ] Confirm the active filmstrip item is visible at 320 px, 768 px, and desktop widths.
- [ ] Confirm empty, one-image, first-image, last-image, failed-preview, and failed-DZI
  states.
- [ ] Confirm button labels and shortcuts remain understandable without color or icons.

## Verification

```bash
uv run pytest tests/gui/browse/test_layout.py \
  tests/gui/browse/test_callbacks_helpers.py \
  tests/gui/browse/test_preparation.py -v
PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_single.py -v
uv run mypy src/phenotypic
```

## Exit criteria

- No more than nine filmstrip items mount at the default radius.
- Exactly one filmstrip item and one dropdown option represent the active image.
- Prepare, Stop, and Clear reflect actual server state and never claim immediate native
  cancellation.
- Background status updates preserve focus and do not flood assistive announcements.
