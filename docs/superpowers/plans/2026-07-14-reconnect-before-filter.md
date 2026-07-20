# Reconnect-Before-Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Dijkstra reconnection in `TwoKFilamentousDetector` and `FilamentousFungiDetector` bridge genuinely-disconnected branch fragments (faded/severed hyphae) instead of deleting them before reconnection runs, behind a `reconnect_scope` flag defaulting to the new behavior.

**Architecture:** A new pure function `sdk_.reconnect.select_reconnect_fragments(...)` chooses the fragment set: `scope="pseudo"` reproduces `identify_pseudo_fragments` exactly (legacy, bit-identical); `scope="branches"` (default) also admits the disconnected branch components the overlap filter drops (`colony_mask & ~structure_mask`). Both detectors gain a `reconnect_scope` field and swap their `identify_pseudo_fragments` call for the new function. Everything downstream (cost build, tiled Dijkstra, final-mask union + re-Voronoi) is unchanged; unbridged fragments are dropped by the existing `final_mask` step. FFD's golden regression is re-baselined with the old golden retained as a pseudo-scope legacy lock.

**Tech Stack:** Python 3.10+, NumPy, scikit-image, pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-07-14-reconnect-before-filter-design.md`

## Global Constraints

- **Branch:** work on `branch-reconnection` (already checked out).
- **`sdk_.reconnect` import purity:** functions take raw arrays only — never an `Image`, operation, or `_PhaseCong3Result`. `select_reconnect_fragments` takes `np.ndarray`s only. (Enforced by `tests/.../test_import_rules.py`.)
- **Docstrings:** Google-style, matching the surrounding module.
- **TDD + frequent commits:** each task is failing test → implement → passing test → commit.
- **`scope="pseudo"` MUST be bit-identical** to today's behavior — it is a pure pass-through to `identify_pseudo_fragments`. The FFD legacy-lock test enforces this.
- **PhenoTypic git ops run under the sandbox override** (`.git` is outside the write allowlist): use `dangerouslyDisableSandbox: true` for `git` commands, and `uv run` for tests.
- **Never review/verify with a weaker model than implemented.**

---

## File Structure

- `src/phenotypic/sdk_/reconnect/_colony_reconnect.py` — add `select_reconnect_fragments`; add `remove_small_objects` import.
- `src/phenotypic/sdk_/reconnect/__init__.py` — export `select_reconnect_fragments`.
- `src/phenotypic/detect/_two_k_filamentous_detector.py` — add `reconnect_scope` field; swap the fragment-selection call.
- `src/phenotypic/detect/_filamentous_fungi_detector.py` — add `Literal` import + `reconnect_scope` field; swap the call.
- `tests/unit/sdk_/reconnect/test_colony_reconnect.py` — unit tests for `select_reconnect_fragments`.
- `tests/unit/detect/test_two_k_filamentous_detector.py` — severed-hypha branches-vs-pseudo behavior test.
- `tests/unit/detect/test_filamentous_fungi_regression.py` — legacy-lock (pseudo) test + repointed default (branches) golden + regeneration entrypoint.
- `tests/fixtures/filamentous_fungi_regression_objmap_branches.npy` — new default baseline (added).
- `tests/fixtures/filamentous_fungi_regression_objmap.npy` — existing, frozen as the pseudo/legacy golden.

## Execution DAG (cluster-and-isolate)

- **Task 1** (Keystone — sdk core): implement first. Everything depends on it.
- **Task 2** (Seam — TwoK wiring) and **Task 3** (Seam — FFD wiring + golden re-baseline): both depend only on Task 1; independent of each other (different detector + test files); may run in parallel worktrees.
- **Task 4** (verification gate): after 1–3.

Each Seam is isolated for a focused gate (risky wiring / golden change). Model: frontier/high-effort for all tasks (judgment-heavy: quality-filter activation, golden diff review).

---

### Task 1: `select_reconnect_fragments` in `sdk_.reconnect`

**Files:**
- Modify: `src/phenotypic/sdk_/reconnect/_colony_reconnect.py`
- Modify: `src/phenotypic/sdk_/reconnect/__init__.py`
- Test: `tests/unit/sdk_/reconnect/test_colony_reconnect.py`

**Interfaces:**
- Consumes: `identify_pseudo_fragments(colony_labels, center_objmask) -> (central_mask, fragment_labels)` (unchanged), `skimage.measure.label`, `skimage.morphology.remove_small_objects`.
- Produces:
  ```python
  def select_reconnect_fragments(
      colony_labels: np.ndarray,
      center_mask: np.ndarray,
      colony_mask: np.ndarray,
      structure_mask: np.ndarray,
      *,
      scope: Literal["branches", "pseudo"] = "branches",
      min_fragment_size: int = 1,
  ) -> tuple[np.ndarray, np.ndarray]:   # (central_mask, fragment_labels int32)
  ```
  - `scope="pseudo"` returns `identify_pseudo_fragments(colony_labels, center_mask)` verbatim.
  - `scope="branches"` unions the pseudo-fragments with `colony_mask & ~structure_mask`, optionally drops components `< min_fragment_size` px, relabels. `central_mask` is unchanged from the pseudo path.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/sdk_/reconnect/test_colony_reconnect.py`:

```python
def test_select_reconnect_fragments_pseudo_matches_identify():
    from phenotypic.sdk_.reconnect import select_reconnect_fragments, identify_pseudo_fragments
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[4:8, 4:8] = 1            # CC touching a center
    labels[20:24, 20:24] = 1        # CC not touching a center -> pseudo-fragment
    center = np.zeros((30, 30), dtype=bool); center[6, 6] = True
    colony_mask = labels > 0        # ignored by the pseudo path
    structure_mask = labels > 0
    c0, f0 = identify_pseudo_fragments(labels, center)
    c1, f1 = select_reconnect_fragments(labels, center, colony_mask, structure_mask, scope="pseudo")
    assert np.array_equal(c0, c1)
    assert np.array_equal(f0, f1)


def test_select_reconnect_fragments_branches_admits_disconnected():
    from phenotypic.sdk_.reconnect import select_reconnect_fragments
    center = np.zeros((30, 40), dtype=bool); center[10, 10] = True
    structure_mask = np.zeros((30, 40), dtype=bool); structure_mask[8:13, 8:20] = True  # body on center
    colony_labels = np.where(structure_mask, 1, 0).astype(np.int32)
    colony_mask = structure_mask.copy()
    colony_mask[8:13, 28:36] = True        # disconnected branch fragment, dropped by the overlap filter
    central, frags = select_reconnect_fragments(
        colony_labels, center, colony_mask, structure_mask, scope="branches")
    assert central[10, 10]                 # body is central
    assert frags[10, 30] > 0               # the disconnected fragment is a reconnect candidate
    # pseudo scope must NOT admit it
    _, frags_pseudo = select_reconnect_fragments(
        colony_labels, center, colony_mask, structure_mask, scope="pseudo")
    assert frags_pseudo[10, 30] == 0


def test_select_reconnect_fragments_min_size_drops_specks():
    from phenotypic.sdk_.reconnect import select_reconnect_fragments
    center = np.zeros((30, 40), dtype=bool); center[10, 10] = True
    structure_mask = np.zeros((30, 40), dtype=bool); structure_mask[8:13, 8:20] = True
    colony_labels = np.where(structure_mask, 1, 0).astype(np.int32)
    colony_mask = structure_mask.copy()
    colony_mask[8:13, 28:36] = True        # 40-px real fragment (kept)
    colony_mask[0, 0] = True               # 1-px speck (dropped)
    _, frags = select_reconnect_fragments(
        colony_labels, center, colony_mask, structure_mask, scope="branches", min_fragment_size=5)
    assert frags[10, 30] > 0
    assert frags[0, 0] == 0


def test_select_reconnect_fragments_rejects_bad_scope():
    import pytest
    from phenotypic.sdk_.reconnect import select_reconnect_fragments
    z = np.zeros((5, 5), dtype=np.int32); b = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        select_reconnect_fragments(z, b, b, b, scope="nonsense")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_colony_reconnect.py -k select_reconnect_fragments -q`
Expected: FAIL with `ImportError: cannot import name 'select_reconnect_fragments'`.

- [ ] **Step 3: Add the `remove_small_objects` import**

In `src/phenotypic/sdk_/reconnect/_colony_reconnect.py`, change line 18:
```python
from skimage.morphology import dilation, disk
```
to:
```python
from skimage.morphology import dilation, disk, remove_small_objects
```

Also ensure `Literal` is importable for the type hint — add to the top-of-file typing import (the module uses `from __future__ import annotations`, so a bare `Literal` in the signature is a string annotation and needs no runtime import; still, add `from typing import Literal` for clarity if not present).

- [ ] **Step 4: Implement `select_reconnect_fragments`**

Add to `src/phenotypic/sdk_/reconnect/_colony_reconnect.py`, immediately after `identify_pseudo_fragments` (ends at line 91):

```python
def select_reconnect_fragments(
    colony_labels: np.ndarray,
    center_mask: np.ndarray,
    colony_mask: np.ndarray,
    structure_mask: np.ndarray,
    *,
    scope: str = "branches",
    min_fragment_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the fragment set the Dijkstra reconnection is allowed to bridge.

    ``scope="pseudo"`` reproduces :func:`identify_pseudo_fragments` exactly: only the
    Voronoi-cut pieces of ``colony_labels`` that miss the inoculum. ``scope="branches"``
    additionally admits the disconnected branch components the overlap filter drops
    (``colony_mask & ~structure_mask``), so genuinely-severed hyphae reach reconnection
    instead of being deleted before it runs. Fragments the reconnection cannot bridge are
    still dropped downstream by the caller's ``final_mask`` step (they are never painted
    into ``colony_labels``), so no extra deletion is needed here.

    Args:
        colony_labels: Voronoi colony labels built from ``structure_mask`` (the Dijkstra
            targets). Zero is background.
        center_mask: Boolean inoculum-center mask.
        colony_mask: The pre-filter union ``branch_mask | center_mask``.
        structure_mask: ``filter_mask_by_overlap(colony_mask, center_mask)`` — the
            center-connected bodies kept today.
        scope: ``"branches"`` (default) admits disconnected branch fragments;
            ``"pseudo"`` restricts to Voronoi-cut pseudo-fragments (legacy behavior).
        min_fragment_size: Drop connected fragments smaller than this many pixels
            (``scope="branches"`` only). ``1`` keeps all.

    Returns:
        ``(central_mask, fragment_labels)``. ``central_mask`` is the trusted colony
        bodies (unchanged vs the pseudo path); ``fragment_labels`` is an int32 relabeled
        map of the fragments to reconnect.

    Raises:
        ValueError: If ``scope`` is not ``"branches"`` or ``"pseudo"``.
    """
    central_mask, fragment_labels = identify_pseudo_fragments(colony_labels, center_mask)
    if scope == "pseudo":
        return central_mask, fragment_labels
    if scope != "branches":
        raise ValueError(f"scope must be 'branches' or 'pseudo', got {scope!r}")

    media_frag_mask = (
        np.asarray(colony_mask, dtype=bool) & ~np.asarray(structure_mask, dtype=bool)
    )
    fragment_mask = (fragment_labels > 0) | media_frag_mask
    if min_fragment_size > 1:
        fragment_mask = remove_small_objects(fragment_mask, min_size=min_fragment_size)
    return central_mask, label(fragment_mask).astype(np.int32)
```

- [ ] **Step 5: Export it**

In `src/phenotypic/sdk_/reconnect/__init__.py`, add `select_reconnect_fragments` to the `from ._colony_reconnect import (...)` block (alphabetical, after `reconnect_fragments_tiled`) and to `__all__`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/sdk_/reconnect/test_colony_reconnect.py -q`
Expected: PASS (all, including the pre-existing tests).

- [ ] **Step 7: Prove the tests can fail (mutation)**

Temporarily change `fragment_mask = (fragment_labels > 0) | media_frag_mask` to `fragment_mask = (fragment_labels > 0)` (drop the media union). Run the select tests → `test_select_reconnect_fragments_branches_admits_disconnected` must FAIL. Revert.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/sdk_/reconnect/_colony_reconnect.py src/phenotypic/sdk_/reconnect/__init__.py tests/unit/sdk_/reconnect/test_colony_reconnect.py
git commit -m "feat(reconnect): select_reconnect_fragments admits disconnected branch fragments"
```

---

### Task 2: Wire `reconnect_scope` into `TwoKFilamentousDetector`

**Files:**
- Modify: `src/phenotypic/detect/_two_k_filamentous_detector.py`
- Test: `tests/unit/detect/test_two_k_filamentous_detector.py`

**Interfaces:**
- Consumes: `select_reconnect_fragments(colony_labels, center_mask, colony_mask, structure_mask, *, scope, min_fragment_size)` from Task 1. Local vars `colony_mask`, `structure_mask`, `center_mask` already exist in `_operate` (lines 190–191, 175).
- Produces: `TwoKFilamentousDetector(reconnect_scope="branches"|"pseudo")`; default `"branches"`.

- [ ] **Step 1: Write the failing behavior test**

Append to `tests/unit/detect/test_two_k_filamentous_detector.py`:

```python
def _plate_colony_with_severed_hypha():
    """1x2 grid: colony 0 = core + tendril + a SEPARATE bright bar just beyond the tendril
    tip (a severed hypha, ~9px gap, inside colony 0's Voronoi cell). colony 1 = plain core.
    Returns (GridImage, well0, well1, frag_bbox)."""
    H, W = 200, 400
    g = np.full((H, W), 60, dtype=np.uint8)
    yy, xx = np.ogrid[:H, :W]

    def disk(cy, cx, r, val):
        g[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = val

    well0, well1 = (100, 100), (100, 300)
    disk(*well0, 22, 235); g[97:104, 100:150] = 215        # colony 0: core + tendril to col ~149
    g[97:104, 158:194] = 215                                # severed hypha: separate bar, ~9px gap
    disk(*well1, 22, 235)                                   # colony 1: plain core
    frag_bbox = (slice(95, 106), slice(158, 194))           # tight around the severed bar
    rgb = np.repeat(g[..., None], 3, axis=2)
    return GridImage(rgb, nrows=1, ncols=2), well0, well1, frag_bbox


def test_reconnect_scope_branches_recovers_severed_hypha():
    img, well0, well1, frag_bbox = _plate_colony_with_severed_hypha()

    def run(scope):
        d = TwoKFilamentousDetector(
            center_detector=ManualGridPointDetector(coord1=well0, coord2=well1,
                                                    shape="disk", width=40),
            reconnect_scope=scope,
        )
        return np.asarray(d.apply(img.copy(), inplace=False).objmap[:])

    branches = run("branches")
    pseudo = run("pseudo")
    # pseudo drops the severed hypha (overlap filter deletes it before reconnection)
    assert pseudo[frag_bbox].max() == 0
    # branches reconnects it: the fragment region is now labeled, and total coverage grows
    assert branches[frag_bbox].max() > 0
    assert (branches > 0).sum() > (pseudo > 0).sum()


def test_reconnect_scope_defaults_to_branches():
    d = TwoKFilamentousDetector()
    assert d.reconnect_scope == "branches"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/detect/test_two_k_filamentous_detector.py -k reconnect_scope -q`
Expected: FAIL — `reconnect_scope` is not a field (`ValidationError`/`AttributeError`).

- [ ] **Step 3: Add the field**

In `src/phenotypic/detect/_two_k_filamentous_detector.py`, in the reconnection-fields block (after `gap_crossing_penalty`, ~line 80), add:
```python
    reconnect_scope: Literal["branches", "pseudo"] = "branches"
```
(`Literal` is already imported at line 4.)

- [ ] **Step 4: Swap the import + call**

Change the import block (lines 21–29): replace `identify_pseudo_fragments,` with `select_reconnect_fragments,`.

In `_operate`, replace line 198:
```python
        central_mask, fragment_labels = identify_pseudo_fragments(colony_labels, center_mask)
```
with:
```python
        central_mask, fragment_labels = select_reconnect_fragments(
            colony_labels, center_mask, colony_mask, structure_mask,
            scope=self.reconnect_scope,
            min_fragment_size=max(1, self.min_branch_width_px),
        )
```

- [ ] **Step 5: Run the behavior tests**

Run: `uv run pytest tests/unit/detect/test_two_k_filamentous_detector.py -k reconnect_scope -q`
Expected: PASS. **If `test_reconnect_scope_branches_recovers_severed_hypha` fails on the `branches[frag_bbox].max() > 0` line**, the detected gap is out of reach: narrow the drawn gap (raise the bar start toward col 152) until the fragment reconnects. **If it fails on `pseudo[frag_bbox].max() == 0`**, the drawn bar is not actually disconnected in the detected mask: widen the gap. Re-run until both hold. (The mechanism itself is already proven by Task 1's sdk tests; this is fixture geometry tuning only.)

- [ ] **Step 6: Run the full TwoK suite (regression)**

Run: `uv run pytest tests/unit/detect/test_two_k_filamentous_detector.py -q`
Expected: PASS — in particular `test_final_objmap_excludes_objects_not_overlapping_centers` (the distant stray at `(32,200)` is ~84 px from any colony, well beyond `frag_reach_px=10`/`max_gap_length=30`, so it stays excluded under the new default) and `test_reconnection_reduces_fragments`.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/detect/_two_k_filamentous_detector.py tests/unit/detect/test_two_k_filamentous_detector.py
git commit -m "feat(detect): TwoKFilamentousDetector.reconnect_scope reconnects severed hyphae"
```

---

### Task 3: Wire `reconnect_scope` into `FilamentousFungiDetector` + re-baseline golden

**Files:**
- Modify: `src/phenotypic/detect/_filamentous_fungi_detector.py`
- Modify: `tests/unit/detect/test_filamentous_fungi_regression.py`
- Add: `tests/fixtures/filamentous_fungi_regression_objmap_branches.npy`
- Keep (frozen): `tests/fixtures/filamentous_fungi_regression_objmap.npy` (the pseudo/legacy golden)

**Interfaces:**
- Consumes: `select_reconnect_fragments(...)` (Task 1). Local vars `overall_objmask` (line 441), `inoculum_structure_mask` (line 449), `inoculum_objmask` (line 398) in scope at line 483.
- Produces: `FilamentousFungiDetector(reconnect_scope="branches"|"pseudo")`; default `"branches"`; orthogonal to `reconnect_strategy`.

- [ ] **Step 1: Add `Literal` import + the field**

In `src/phenotypic/detect/_filamentous_fungi_detector.py` line 2, add `Literal`:
```python
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Optional, Union
```
Add the field next to `reconnect_strategy` (~line 301):
```python
    reconnect_scope: Literal["branches", "pseudo"] = "branches"
```

- [ ] **Step 2: Swap the import + call**

In the `from phenotypic.sdk_.reconnect import (...)` block (lines ~32–41), replace `identify_pseudo_fragments,` with `select_reconnect_fragments,`.

Replace line 483:
```python
        central_mask, fragment_labels = identify_pseudo_fragments(
                colony_labels, inoculum_objmask,
        )
```
with:
```python
        central_mask, fragment_labels = select_reconnect_fragments(
                colony_labels, inoculum_objmask, overall_objmask, inoculum_structure_mask,
                scope=self.reconnect_scope,
                min_fragment_size=max(1, self.min_branch_width_px),
        )
```

- [ ] **Step 3: Rewrite the regression test around the two scopes**

Replace the body of `tests/unit/detect/test_filamentous_fungi_regression.py` with:

```python
# tests/unit/detect/test_filamentous_fungi_regression.py
"""Golden characterization tests pinning FilamentousFungiDetector's objmap.

Two goldens:
- ``FIXTURE_PSEUDO`` (frozen, pre-reconnect-scope): FFD with ``reconnect_scope="pseudo"``
  must reproduce it bit-for-bit — the legacy-behavior lock.
- ``FIXTURE_BRANCHES``: FFD with the default ``reconnect_scope="branches"``.
On the synthetic plate the two may be byte-identical (no reachable disconnected fragments);
that is the correct, reassuring result — branches only adds structure when a bridgeable
fragment exists. The behavioral proof lives in the TwoK severed-hypha test.
"""
from pathlib import Path

import numpy as np

from phenotypic.data import load_synth_filamentous_plate
from phenotypic.detect import FilamentousFungiDetector, OtsuDetector

_FIX = Path(__file__).parent.parent.parent / "fixtures"
FIXTURE_PSEUDO = _FIX / "filamentous_fungi_regression_objmap.npy"
FIXTURE_BRANCHES = _FIX / "filamentous_fungi_regression_objmap_branches.npy"


def _run(reconnect_scope: str) -> np.ndarray:
    image = load_synth_filamentous_plate().copy()
    detector = FilamentousFungiDetector(
        inoculum_detector=OtsuDetector(ignore_zeros=True), reconnect_scope=reconnect_scope,
    )
    return np.asarray(detector.apply(image, inplace=False).objmap[:])


def test_pseudo_scope_matches_legacy_golden():
    # A missing fixture must FAIL loudly, not skip.
    assert FIXTURE_PSEUDO.exists(), f"Legacy golden missing: {FIXTURE_PSEUDO}."
    expected = np.load(FIXTURE_PSEUDO)
    actual = _run("pseudo")
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.array_equal(actual, expected), (
        f"reconnect_scope='pseudo' is NOT bit-identical to the legacy golden: "
        f"{int((actual != expected).sum())} pixels differ"
    )


def test_branches_default_matches_golden():
    assert FIXTURE_BRANCHES.exists(), (
        f"Branches golden missing: {FIXTURE_BRANCHES}. Regenerate with "
        f"`uv run python -m tests.unit.detect.test_filamentous_fungi_regression`."
    )
    expected = np.load(FIXTURE_BRANCHES)
    actual = _run("branches")
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.array_equal(actual, expected), (
        f"objmap changed: {int((actual != expected).sum())} pixels differ"
    )


if __name__ == "__main__":  # regenerate the BRANCHES default only; the pseudo golden is frozen.
    FIXTURE_BRANCHES.parent.mkdir(parents=True, exist_ok=True)
    np.save(FIXTURE_BRANCHES, _run("branches"))
    print(f"wrote {FIXTURE_BRANCHES}")
```

- [ ] **Step 4: Run the legacy-lock test — MUST pass bit-identical**

Run: `uv run pytest tests/unit/detect/test_filamentous_fungi_regression.py::test_pseudo_scope_matches_legacy_golden -q`
Expected: PASS. **If it fails, `scope="pseudo"` is not a pure pass-through — stop and fix `select_reconnect_fragments`/the FFD wiring before continuing** (this is the backward-compat guarantee).

- [ ] **Step 5: Regenerate the branches golden + eyeball the diff**

Run: `uv run python -m tests.unit.detect.test_filamentous_fungi_regression`
Then compare against the legacy golden and confirm any change is *added reconnected structure*, not corruption:
```bash
uv run python -c "
import numpy as np
a=np.load('tests/fixtures/filamentous_fungi_regression_objmap.npy')
b=np.load('tests/fixtures/filamentous_fungi_regression_objmap_branches.npy')
d=(a!=b)
print('changed px:', int(d.sum()), 'of', a.size)
print('added-fg px (bg->fg):', int(((a==0)&(b>0)).sum()), '| removed-fg px (fg->bg):', int(((a>0)&(b==0)).sum()))
"
```
Expected: `removed-fg px` is 0 (branches never deletes a colony pixel pseudo kept); `added-fg px` ≥ 0 (0 is fine — see the module docstring). If `removed-fg px > 0`, investigate before committing the fixture.

- [ ] **Step 6: Run the full regression test**

Run: `uv run pytest tests/unit/detect/test_filamentous_fungi_regression.py -q`
Expected: PASS (both tests).

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/detect/_filamentous_fungi_detector.py tests/unit/detect/test_filamentous_fungi_regression.py tests/fixtures/filamentous_fungi_regression_objmap_branches.npy
git commit -m "feat(detect): FilamentousFungiDetector.reconnect_scope + re-baselined golden"
```

---

### Task 4: Verification gate

**Files:** none (validation only).

- [ ] **Step 1: Full affected-suite run**

Run: `uv run pytest tests/unit/sdk_/reconnect tests/unit/detect -q`
Expected: PASS. Investigate any failure before proceeding.

- [ ] **Step 2: Import-purity guard**

Run: `uv run pytest -k import_rules -q`
Expected: PASS — `select_reconnect_fragments` takes only arrays, so `sdk_.reconnect` stays free of `Image`/operation/`_PhaseCong3Result` types.

- [ ] **Step 3: Cross-cutting mutation check**

Reintroduce the Task 1 mutation (drop the `| media_frag_mask` union) and confirm BOTH the sdk `branches` test AND the TwoK `severed_hypha` test go red; then revert and confirm green. This proves the new tests actually guard the behavior end-to-end.

- [ ] **Step 4: (Optional) Visual A/B on real data**

In the Neurospora repo, re-run the diagnostic overlay (`scratchpad/diagnose_reconnect_loss.py`, extended to render `reconnect_scope="pseudo"` vs `"branches"` on the rows-1–2/cols-1–4 crop) to eyeball that severed hyphae in row-2/col-4 are visibly reconnected under `branches`. Not a gate; a sanity check on real imagery before considering tuning of `max_gap_length`/`frag_reach_px`.

- [ ] **Step 5: Simplify pass**

Review the four changed source hunks for duplication/clarity (quality only, no behavior change); if anything is simplified, re-run `uv run pytest tests/unit/sdk_/reconnect tests/unit/detect -q`.

---

## Self-Review (checked against the spec)

- **Spec coverage:** shared `select_reconnect_fragments` (Task 1) ✓; `reconnect_scope` flag default-new on both detectors (Tasks 2, 3) ✓; `scope="pseudo"` bit-identical + FFD legacy lock (Task 3 Step 4) ✓; new default golden re-baselined with diff review (Task 3 Step 5) ✓; severed-hypha behavior + distant-noise-not-reconnected (Task 2 Steps 1, 6) ✓; `min_fragment_size` guard (Task 1) ✓; mutation checks (Tasks 1, 4) ✓; import purity (Task 4) ✓.
- **Type consistency:** `select_reconnect_fragments(colony_labels, center_mask, colony_mask, structure_mask, *, scope, min_fragment_size)` used identically in Tasks 1–3; both detectors pass `min_fragment_size=max(1, self.min_branch_width_px)` (a real field on both, default 3).
- **No placeholders:** every code step has complete code; the only tuning is the fixture geometry in Task 2 Step 5, with an explicit adjustment procedure.
