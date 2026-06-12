# Deferred Tickets — Source Image Browse Tab

**Date:** 2026-06-11
**Branch:** `worktree-gui-source-image-viewer`
**Feature:** GUI Browse tab (`src/phenotypic/gui/browse/`) — see
`docs/superpowers/specs/2026-06-11-gui-source-image-browse-tab-design.md`
and `docs/superpowers/plans/2026-06-11-gui-source-image-browse-tab.md`.

These were surfaced by the per-phase reviews + the final Fable review (verdict:
SHIP). **None block merge** — the feature is correct and tested. Captured here
so they aren't lost. Tickets 1 and 3 have real user-facing impact; the rest are
quality/cleanup.

---

## Ticket 1 — Browse metadata does a redundant full image decode  *(P1, user-facing)*

**Where:** `gui/browse/_metadata.py:71-72` + `gui/browse/_source_render.py:98`

**Problem:** Viewing one image triggers **two full decodes** of the same file:
the tile-route manifest endpoint → `normalize_to_png` → `Image.imread(original).rgb[:]`
(to build the PNG to tile), and the metadata callback → `_metadata.read` →
`Image.imread(original).rgb[:]` (just to read `width`/`height` + EXIF). For a
**large RAW** (`.nef/.cr2/…`), `rawpy` postprocessing is multi-second, so it's
paid twice per first view. The spec anticipated a *cheap* dims path; the impl
always takes the heavy one.

**Fix:** Make `_metadata.read` a true cheap read — no full pixel decode:
- **Dims:** `PIL.Image.open(original).size` (header-only) for standard formats;
  `rawpy.imread(str(original)).sizes` (header metadata, no `postprocess()`) for
  RAW, swapping W/H when `sizes.flip in (5, 6)`. RAW on a host without `rawpy`
  (Windows) → dims `None` (panel shows `—`, which is fine since RAW can't render
  there anyway).
- **EXIF:** `exifread.process_file(open(original, "rb"), details=False)` reads
  tags without decoding pixels — and it's the same library `Image._metadata.imported`
  is populated from, so the exifread keys (`EXIF DateTimeOriginal`,
  `Image Make/Model`) and the existing `_extract_exif` stay unchanged.
- Keep `_metadata.read` a **pure read** (no PNG-writing side effects).

**Effort:** ~30–60 min. **Risk:** low (read path; `test_metadata.py` covers it —
add a regression asserting the module no longer imports `phenotypic.Image`, i.e.
no full decode). Watch the `rawpy.sizes` flip/orientation handling.

---

## Ticket 2 — `_index_string_with_prefix` triplicated across three sub-apps  *(P2, cleanup)*

**Where:** `gui/builder/_app.py:40`, `gui/results_viewer/_app.py:72`,
`gui/browse/_app.py:29`.

**Problem:** Three copies of the same function — **verified identical except for
docstrings** (same DOCTYPE template, same security-relevant escaping
`\\`→`\\\\`, `"`→`\\"`, `</`→`<\/`, same injected
`window.__phenotypicAppPrefix`). Three places a future XSS-hardening fix must
land; they can silently drift.

**Fix:** Lift one canonical `index_string_with_app_prefix(url_prefix)` into
`gui/_shared/` (e.g. a new `_shared/_index.py`, exported from
`_shared/__init__.py`); import it in all three `_app.py` and delete the local
copies.

**Effort:** ~30 min. **Risk:** low-moderate — touches builder + results_viewer +
browse factories; each `test_app`/index test must still assert the prefix is
injected. **Do as a standalone PR** (edits non-Browse apps; shouldn't ride a
feature branch).

---

## Ticket 3 — RAW/image-extension literals drift in `run_console` + `tune`  *(P1, user-facing)*

**Where:** `gui/run_console/_callbacks.py:607-618` and
`gui/tune/_callbacks.py:1757-1759` (`_PLATE_EXTS`).

**Problem:** Both hand-spell their own image-extension set instead of importing
the now-canonical `gui/_config.IMAGE_EXTS`, and after this feature's RAW work
they're **out of sync**:
- `run_console` = `{".png",".tif",".tiff",".jpg",".jpeg",".raw",".nef",".cr2",".arw",".dng"}`
  — includes `.raw` (the core can't decode it) and **lacks `.cr3`** (the core
  *can*).
- `tune._PLATE_EXTS` = `{".png",".tif",".tiff",".jpg",".jpeg",".nef",".cr2",".arw",".dng"}`
  — **lacks `.cr3`**.

So the Tune and Run-console image pickers won't surface `.cr3` plates while
Browse/Builder will — a real "my `.cr3` doesn't show up" inconsistency introduced
by extending the core RAW set.

**Fix:** Replace both inline sets with `from phenotypic.gui._config import IMAGE_EXTS`
(`{…, ".cr2", ".cr3", ".nef", ".arw", ".dng"}` — single source of truth).

**Effort:** ~15 min. **Risk:** low (broadens/corrects two pickers; confirm no
test pins the old literal set). Cheapest fix with real impact.

---

## Ticket 4 — `%2f`-mangled tile URLs return 200 app-shell instead of 404  *(P3, recommend WON'T-FIX)*

**Where:** `gui/browse/_tile_routes.py` tile route, under the hub's
`DispatcherMiddleware`.

**Observation:** `GET /browse/tiles/<token>_files/0/..%2f..%2fx.png` returns
Dash's 200 app-shell HTML, not a 404 — because Werkzeug percent-decodes `%2f`→`/`
and normalizes the `..` *before* routing, so the path escapes the `/tiles`
prefix entirely and falls through to Dash's catch-all.

**Why WON'T-FIX (not just "decline"):** It cannot be fixed cleanly. By the time
routing runs, the normalized path (`/browse/x.png`) is no longer under `/tiles`,
so a `/tiles` `before_request` guard never sees it; the only catch-all that would
404 it is one that 404s *all* unknown paths, which breaks Dash's own routing. And
it's **benign** — no sandbox file is ever served (it's the app-shell HTML), and
the real client never generates such URLs (OSD derives well-formed tile paths).
The adversarial test suite already proves single-segment traversal vectors
(absolute/NUL/non-UTF8/directory/symlink/`%2e%2e`) all return JSON 404.

**Recommendation:** **Won't-fix.** Documented so the behavior is a known,
deliberate choice, not an oversight.

---

## Ticket 5 — results_viewer vendored OSD is committed CR-stripped (byte-parity)  *(P3, cleanup)*

**Where:** `gui/results_viewer/_assets/openseadragon/openseadragon.min.js`
(+ `LICENSE`).

**Problem:** The repo has `core.autocrlf=true` and no top-level `.gitattributes`,
so the pre-existing results_viewer `openseadragon.min.js` was committed
**CR-stripped** (277226 vs upstream 277305 bytes). Harmless for JS execution but
byte-divergent from upstream/our Browse copy. (Browse's copy is byte-clean — it
ships a scoped `_assets/openseadragon/.gitattributes`; see the
`gui-vendored-assets-autocrlf` note in memory.)

**Fix:** Add the same scoped `.gitattributes` (`* binary`, `*.png binary`,
`openseadragon.min.js -text`) under `results_viewer/_assets/openseadragon/`,
restore the clean bytes (e.g.
`git cat-file blob HEAD:…/browse/…/openseadragon.min.js > …/results_viewer/…/openseadragon.min.js`,
same for `LICENSE`), then `git add` so git stores them verbatim. Verify with
`git check-attr -a` (`binary: set`) and a staged-blob size of 277305.

**Effort:** ~15 min. **Risk:** low (line-ending-only; functionally identical).
Cosmetic byte-parity; do alongside Ticket 2's `_shared` cleanup PR.
