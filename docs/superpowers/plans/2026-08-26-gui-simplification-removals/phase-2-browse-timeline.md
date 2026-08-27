# Phase 2 — Browse Timeline mode: delete

**Spec:** §1.1, §1.2, §1.3, §6. **Depends on:** nothing (phase 1 touched a different
sub-app). **Blocks:** phase 3.

**Deliverable:** the Browse tab has no view-mode toggle and no timeline body. Single mode
is the whole tab, with **behaviour unchanged**. The four timeline-only browse helpers and
the browse copy of `timeline.js` / `timeline.css` are gone. `_shared/timeline/` still
exists — phase 3 removes it.

> **The asymmetry to hold on to:** phase 1 removed a tab from a tab bar. This phase removes
> a *mode* from a two-mode tab, which means the surviving mode stops being conditional.
> `BROWSE_SINGLE_BODY` currently renders with a `display` style driven by the toggle
> callback (`browse/_callbacks.py:1353-1360`); after this phase it renders unconditionally.
> Getting that wrong leaves Single permanently hidden and every Browse e2e test failing at
> once.

---

### Task 2.1: Pin Single mode's behaviour before touching it

**Files:**
- Test: `tests/unit/gui/browse/test_single_mode_survives_removal.py` (create)

**Interfaces:**
- Produces: `test_browse_layout_has_no_view_mode_toggle`,
  `test_browse_single_body_is_unconditional` — the executable statement of this phase's
  "Single is unchanged" contract.

- [ ] **Step 1: Enumerate what Single mode owns today**

```bash
uv run grep -n "BROWSE_SINGLE\|BROWSE_VIEW_MODE\|BROWSE_TIMELINE_BODY" \
  src/phenotypic/gui/browse/_layout.py src/phenotypic/gui/browse/_callbacks.py
```
Record the result. `BROWSE_VIEW_MODE_TOGGLE` and `BROWSE_TIMELINE_BODY` disappear;
`BROWSE_SINGLE_BODY` **survives** and must stop being style-driven.

- [ ] **Step 2: Write the failing test**

```python
"""Browse keeps Single mode and loses the mode toggle.

The point of this file is the *second* test: a removal that hides the surviving
mode instead of unhiding it is the failure mode with the widest blast radius,
and it is invisible in a unit test that only checks for absence.
"""

from phenotypic.gui.browse import _ids as ids
from phenotypic.gui.browse._layout import build_browse_layout


def _ids_in(component) -> set[str]:
    found: set[str] = set()

    def walk(node) -> None:
        node_id = getattr(node, "id", None)
        if isinstance(node_id, str):
            found.add(node_id)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                walk(child)
        elif children is not None:
            walk(children)

    walk(component)
    return found


def test_browse_layout_has_no_view_mode_toggle():
    present = _ids_in(build_browse_layout())
    assert ids.BROWSE_SINGLE_BODY in present
    assert not any(name.startswith("browse-tl-") for name in present)
    assert "browse-view-mode-toggle" not in present
    assert "browse-timeline-body" not in present


def test_browse_single_body_is_unconditional():
    """Single must not be hidden by a leftover ``display: none``."""

    def find(node, target):
        if getattr(node, "id", None) == target:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                hit = find(child, target)
                if hit is not None:
                    return hit
        elif children is not None:
            return find(children, target)
        return None

    body = find(build_browse_layout(), ids.BROWSE_SINGLE_BODY)
    assert body is not None
    assert (getattr(body, "style", None) or {}).get("display") != "none"
```

Note the string literals `"browse-view-mode-toggle"` / `"browse-timeline-body"` — the
constants are being deleted, so the test cannot import them.

- [ ] **Step 3: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/browse/test_single_mode_survives_removal.py -v
```
Expected: `test_browse_layout_has_no_view_mode_toggle` FAILS (the toggle is present).
`test_browse_single_body_is_unconditional` may pass already — that is fine; it is a guard
against what task 2.3 could break, not a statement about today.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/unit/gui/browse/test_single_mode_survives_removal.py
git commit -m "test(gui): pin Browse Single mode against the timeline removal"
```

---

### Task 2.2: Delete the four browse helpers, the assets and their tests

**Files:**
- Delete: `src/phenotypic/gui/browse/_thumb_routes.py`, `_timeline_records.py`,
  `_capture_time.py`, `_plate_pattern.py`
- Delete: `src/phenotypic/gui/browse/_assets/timeline.js`, `_assets/timeline.css`
- Delete: `tests/gui/browse/test_timeline_callbacks_helpers.py`,
  `test_timeline_records.py`, `test_capture_time.py`, `test_plate_pattern.py`
- Delete: `tests/e2e/gui/test_browse_timeline.py`, `tests/e2e/gui/test_browse_compare_strip.py`

- [ ] **Step 1: Re-verify each helper is timeline-only**

The spec (§1.1) records call-site evidence for all four. Re-run it — this is the single
highest-risk claim in the phase, because three of the four have names that sound like
general Browse infrastructure:

```bash
for m in _thumb_routes _timeline_records _capture_time _plate_pattern; do
  echo "--- $m"
  uv run grep -rn "$m" src/ tests/ --include='*.py' | grep -v "browse/$m.py"
done
```

Expected, per spec:
- `_thumb_routes` → `browse/_app.py:33, :84` only.
- `_timeline_records` → `browse/_callbacks.py:50` only.
- `_capture_time` → `browse/_callbacks.py:44` only, feeding `_capture_time_of` at
  `:1531` inside `build_browse_records`. **It is not the EXIF chip** on the front metadata
  row — that resolves through `browse/_metadata.py`. Confirm with
  `uv run grep -n "capture\|exif" src/phenotypic/gui/browse/_metadata.py`.
- `_plate_pattern` → `browse/_callbacks.py:46` and `_timeline_records.py:17` only.

**If any grep shows a consumer outside those, stop.** A helper with a Single-mode consumer
is not deletable and the spec needs an amendment, not a workaround.

- [ ] **Step 2: Delete**

```bash
git rm src/phenotypic/gui/browse/_thumb_routes.py \
       src/phenotypic/gui/browse/_timeline_records.py \
       src/phenotypic/gui/browse/_capture_time.py \
       src/phenotypic/gui/browse/_plate_pattern.py \
       src/phenotypic/gui/browse/_assets/timeline.js \
       src/phenotypic/gui/browse/_assets/timeline.css
git rm tests/gui/browse/test_timeline_callbacks_helpers.py \
       tests/gui/browse/test_timeline_records.py \
       tests/gui/browse/test_capture_time.py \
       tests/gui/browse/test_plate_pattern.py \
       tests/e2e/gui/test_browse_timeline.py \
       tests/e2e/gui/test_browse_compare_strip.py
```

`test_browse_compare_strip.py` goes despite its name: its own module docstring records
that it drives the Browse **Timeline** surface (`#browse-tl-compare-btn`,
`window.__phenotypicTimeline`). Confirm before deleting:

```bash
git show HEAD:tests/e2e/gui/test_browse_compare_strip.py | head -20
```

Do not commit yet — the tree does not import.

---

### Task 2.3: Strip the timeline surface out of Browse

**Files:**
- Modify: `src/phenotypic/gui/browse/_ids.py` — 64 `BROWSE_TL_*` names + `:47-49`
- Modify: `src/phenotypic/gui/browse/_layout.py` — toggle `:274`, single body `:296`, `build_timeline_body` `:320-585`
- Modify: `src/phenotypic/gui/browse/_callbacks.py` — imports `:30, :39, :44, :46, :50`; TL callbacks
- Modify: `src/phenotypic/gui/browse/_cache.py` — timeline-thumb cache entries
- Modify: `src/phenotypic/gui/browse/_app.py:33, :84` — `_thumb_routes` registration

- [ ] **Step 1: Remove the ids**

Delete all 64 `BROWSE_TL_*` constants, plus `BROWSE_VIEW_MODE_TOGGLE` (`:47`) and
`BROWSE_TIMELINE_BODY` (`:49`), and each of their `__all__` entries (`:130-132` and the
`BROWSE_TL_*` block). **Keep `BROWSE_SINGLE_BODY` (`:48`).**

```bash
uv run grep -c "BROWSE_TL" src/phenotypic/gui/browse/_ids.py
```
Expected after the edit: `0`. Before: `64`.

- [ ] **Step 2: Remove the layout surface**

- Delete `build_timeline_body()` entirely (`_layout.py:320` to the end of that function,
  which closes with `id=ids.BROWSE_TIMELINE_BODY` at `:584`).
- Delete the view-mode toggle component at `:274`.
- At `:296`, make `BROWSE_SINGLE_BODY` unconditional: remove any `style` that the toggle
  callback drove, and remove the wrapper that switched between the two bodies so Single is
  a direct child.
- Remove `build_timeline_body` from `__all__` if present.

- [ ] **Step 3: Remove the callbacks and imports**

In `_callbacks.py` remove:
- `:30` `stepped_timeline_tile_size_from_trigger` (from `_shared.timeline`)
- `:39` `from phenotypic.gui._shared.timeline import build_matrix, build_timeline_grid`
- `:44` `from phenotypic.gui.browse._capture_time import read_capture_time`
- `:46` the `_plate_pattern` import block
- `:50` the `_timeline_records` import block
- the timeline helpers and callbacks: `timeline_thumb_url` (`:505`),
  `timeline_revision_token` (`:510`), `render_timeline_grid` (`:527`),
  `_reset_timeline_for_source` (`:931`), the `_capture_time_of` helper at `:1531`, and the
  mode-switch callback at `:1353-1360` **including its clientside counterpart at `:1370`**.

```bash
uv run grep -n "timeline\|BROWSE_TL\|view_mode" src/phenotypic/gui/browse/_callbacks.py
```
Expected after the edit: no hits. Grep is the completion check — the callback bodies are
long and interleaved, so read the whole file once after editing.

- [ ] **Step 4: Remove the cache entries and the thumb route**

```bash
uv run grep -n "timeline\|thumb" src/phenotypic/gui/browse/_cache.py \
                                  src/phenotypic/gui/browse/_app.py
```
Remove the timeline-thumb cache entries in `_cache.py`, and in `_app.py` the
`_thumb_routes` import (`:33`) and its `register` call (`:84`).

**Do not remove the Browse DZI/`BrowseCache` path.** Browse keeps libvips → DZI → OSD as
its only pixel path (viv-rebuild spec §9); only the *timeline thumbnail* route goes.

- [ ] **Step 5: Run the tests written in task 2.1**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/browse/test_single_mode_survives_removal.py -v
```
Expected: both PASS.

- [ ] **Step 6: Run the whole browse unit suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/browse -n 4 -q
```
Expected: PASS. **Any failure in a non-timeline browse test is a regression in Single
mode** and must be fixed here, not deferred.

- [ ] **Step 7: Run the Browse e2e suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k browse -q
```
Expected: PASS for the Single-mode tests. This is the check that catches a Single body
left hidden — the unit test in 2.1 checks the built layout, this checks the running app.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/browse/
git add -A src/phenotypic/gui/browse tests/unit/gui/browse tests/gui/browse tests/e2e/gui
git commit -m "refactor(gui): delete Browse Timeline mode, keep Single unchanged"
```

---

### Task 2.4: Retire the Browse Timeline ledger rows, tutorial and capture function

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md:104-126` (Browse Timeline block — remove)
- Modify: `src/phenotypic/gui/WORKFLOWS.md:54` (`browse_timeline` row — remove)
- Modify: `scripts/capture_gui_tutorial_screenshots.py:1156` and its call sites
- Delete: `docs/source/tutorials/gui/19_browse_timeline.md` and its image directory
- Modify: `docs/source/tutorials/gui/index.md`

- [ ] **Step 1: Remove the ledger rows and the tutorial**

Follow phase 1 task 1.4 steps 1-4 verbatim, substituting: `FEATURES.md:104-126`,
`WORKFLOWS.md:54`, `_capture_browse_timeline` (`:1156`), page
`19_browse_timeline.md`, images `docs/source/_static/gui_images/browse_timeline/`.

**Do not touch the `18_browse.md` page** — that is Single mode's tutorial and it survives.

- [ ] **Step 2: Run the three gates**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --smoke
```
Expected: all exit 0.

- [ ] **Step 3: Commit**

```bash
git add -A src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
           scripts/capture_gui_tutorial_screenshots.py docs/source
git commit -m "docs(gui): retire the Browse Timeline ledger rows and tutorial"
```
