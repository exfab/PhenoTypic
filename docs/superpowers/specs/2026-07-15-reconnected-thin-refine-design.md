# Provenance-Aware Reconnected-Thin Refine — Design

**Date:** 2026-07-15
**Status:** Approved (design), pending implementation plan
**Affects:** `TwoKFilamentousDetector`, `FilamentousFungiDetector`, `sdk_.reconnect`

## Problem

Filamentous-colony masks carry **residual agar-media branching** — faint agar texture the phase-congruency
detector traces as thin spurious hyphae. A provenance-blind cleanup (trim thin+isolated pixels anywhere,
"D8") reduces the agar but also trims **already-connected, confident colony structure**, because thin
real hyphae and thin agar tendrils are structurally near-identical.

We want a cleanup that **only touches the uncertain part of the colony and never the confident core** —
i.e. provenance-aware: protect the colony established by the grid centers + overlap, trim only the
branches added by reconnection.

## Empirical grounding (why this design, not the alternatives)

Two provenance signals were prototyped and compared per-colony on the full production crop (r1c4/r1c5/r2c4):

| filter | keep / cover (r1c4, r1c5, r2c4) | verdict |
|---|---|---|
| blind D8 (provenance-blind) | 92/.996, 90/1.0, 86/.998 | trims thin everywhere, incl. confident core |
| **seed vs candidate** (protect strict-seed pixels) | 92/.996, 91/1.0, 86/.999 | **≈ blind D8 — dead end** |
| **anchored vs reconnected** (protect `structure_mask`) | **97/.999, 97/1.0, 94/1.0** | **safe: protects the core, trims only reconnected tendrils** |

- **Seed vs candidate fails.** Only ~16–19% of a colony is strict-seed-backed, and those pixels are the
  *ridge cores* (which D8 already spares). The thin agar tendrils **and** the thin real hyphae are both
  loose-*candidate-only*, so seed-backing does not separate them — seed-aware ≈ blind D8.
- **Anchored vs reconnected works.** ~80–89% of each colony is anchored (`structure_mask` — the overlap-
  connected core). Protecting it and trimming only the non-anchored reconnected region (~10–20%, the
  outer bridged tendrils) keeps `cover ≈ 1.0` — it *cannot* damage the confident colony.
- **Correction to an earlier assumption:** reconnection is **not** a no-op on these anchored colonies; it
  bridges ~10–20% (the outer tendrils). So there is a real, substantial reconnected region to filter.

**Honest tradeoff:** the anchored-provenance refine removes *less* agar than blind D8 — it leaves any
agar woven *into* the anchored fan. But everything it removes is provenance-justified (the least-certain,
reconnection-added pixels) and it can never touch the confident colony. This is the intended trade:
safety over completeness.

## Goal

Add an **opt-in, post-reconnection refine step** to both detectors that trims **thin + locally isolated**
pixels **only in the reconnected region** (`colony_labels > 0` AND NOT `structure_mask`), leaving the
anchored colony untouched. Off by default (current behavior + FFD golden preserved).

## Key decisions

1. **Provenance = anchored vs reconnected** (`structure_mask`). Seed/candidate rejected (empirically ≈ blind).
2. **Both detectors**, via a shared pure `sdk_.reconnect` helper.
3. **Opt-in, default off** (`refine_reconnected="none"`), so default behavior and FFD's golden regression
   are unchanged.
4. **Trim criterion = the density-aware "thin AND isolated" test** (D8), the best colony-preserving
   discriminator from the agar-hardening analysis, applied *only* to reconnected-region pixels.

## Architecture

### New pure function (`src/phenotypic/sdk_/reconnect/_colony_reconnect.py`)

```python
def refine_reconnected_thin(
    colony_labels: np.ndarray,     # final labeled objmap (post reconnection + Voronoi)
    structure_mask: np.ndarray,    # anchored (overlap-connected) region — PROTECTED, never trimmed
    *,
    max_width_px: float = 1.5,     # medial-axis half-width ≤ this ⇒ "thin"
    min_density: float = 0.35,     # local foreground fraction < this ⇒ "isolated"
    density_window: int = 15,      # window (px) for the local-density estimate
) -> np.ndarray:
    """Zero out thin + locally-isolated pixels that lie OUTSIDE ``structure_mask``.

    Provenance-aware residual-agar cleanup: only the reconnected region
    (``colony_labels > 0 & ~structure_mask``) is eligible for trimming; the anchored colony
    is preserved exactly. Returns a copy of ``colony_labels`` with the trimmed pixels set to 0
    (labels are preserved; only pixels are removed).
    """
    fg = colony_labels > 0
    dt = distance_transform_edt(fg)
    dens = uniform_filter(fg.astype(float), density_window)
    reconnected = fg & ~np.asarray(structure_mask, dtype=bool)
    drop = reconnected & (dt > 0) & (dt <= max_width_px) & (dens < min_density)
    out = colony_labels.copy()
    out[drop] = 0
    return out
```

- Pure array function (no Image/operation/`_PhaseCong3Result`) — satisfies the `sdk_.reconnect` import
  contract. Needs `from scipy.ndimage import distance_transform_edt, uniform_filter` added to the module.
- Exported from `sdk_/reconnect/__init__.py`.
- **Label-preserving:** it only zeros pixels; it does not re-Voronoi or relabel. Any fragment left
  disconnected by trimming its thin bridge is naturally dropped downstream / is a tiny orphan — acceptable
  and consistent with the prototype (which held `cover ≈ 1.0`).

### Per-detector wiring

Add one field to **both** detectors:
```python
refine_reconnected: Literal["none", "thin"] = "none"
```

Insert the refine **after the final Voronoi**, just before writing the objmap:

- **TwoK** (`_two_k_filamentous_detector.py`, after line 219 `colony_labels = partition_by_grid_voronoi(markers, final_mask)`):
  ```python
  if self.refine_reconnected == "thin":
      colony_labels = refine_reconnected_thin(
          colony_labels, structure_mask,
          max_width_px=self.min_branch_width_px / 2,
          density_window=max(3, int(round(5 * self.min_branch_width_px))),
      )
  ```
- **FFD** (`_filamentous_fungi_detector.py`, after line 527 `colony_labels = partition_by_grid_voronoi(centroid_markers, final_mask)`, before the dtype cast at 530): same call with `inoculum_structure_mask`.

**Parameter derivation** (from each detector's existing `min_branch_width_px`, default 3):
- `max_width_px = min_branch_width_px / 2` → **1.5** at the default (a hypha ~`min_branch_width_px` wide
  has medial-axis half-width ~`min_branch_width_px/2`; thinner ⇒ sub-branch-width thread).
- `density_window = round(5 × min_branch_width_px)` → **15** at the default.
- `min_density = 0.35` — a constant (helper default), not derived.

These reproduce the validated D8 defaults exactly at `min_branch_width_px = 3`. Orthogonal to
`reconnect_scope` and (FFD) `reconnect_strategy` — the refine acts on whatever final objmap they produce.

### Data flow (`refine_reconnected="thin"`)

```
... reconnect → final_mask = (colony_labels>0) | structure_mask → Voronoi → colony_labels
                                                                              │
                          structure_mask (anchored, protected) ──────────────┤
                                                                              ▼
                                              refine_reconnected_thin(colony_labels, structure_mask)
                                              trims (thin & isolated & ~structure_mask) → objmap
```

## Testing

### New sdk unit tests (`tests/unit/.../test_colony_reconnect.py`)
- Anchored pixels are **never** zeroed: with a `structure_mask` covering the whole colony, output equals input.
- A thin, isolated pixel run *outside* `structure_mask` is zeroed; a *thick* non-anchored blob is kept.
- `max_width_px`/`min_density` monotonicity: raising `max_width_px` trims at least as much.
- Labels preserved on kept pixels (no relabeling; only zeros introduced).

### Detector behavior tests
- **TwoK**: on a synthetic plate with a deliberately-thin reconnected tendril, `refine_reconnected="thin"`
  removes it while `"none"` keeps it, and **no anchored (`structure_mask`) pixel is removed** (assert the
  refined objmap ⊇ the anchored core). Confirm `refine_reconnected="none"` leaves the objmap identical to
  today.
- **FFD golden regression:** default `refine_reconnected="none"` must stay **bit-identical** to the current
  golden (add/keep the legacy-lock). No new default baseline needed — the default output is unchanged.

### Mutation check
Reintroduce a one-line mutation (e.g. drop the `& ~structure_mask` guard so it trims anchored pixels too)
and confirm the "anchored pixels never removed" test fails; revert.

## Risks / notes

- **Removes less agar than blind D8** — by design (it never touches the anchored fan). If a future need
  calls for cleaning the anchored region too, that is a *separate, less-safe* mode, not this one.
- **Disconnection:** trimming a thin reconnected bridge can orphan the fragment beyond it. Acceptable
  (that fragment was joined only by a dubious thin/agar-like bridge); the prototype kept `cover ≈ 1.0`.
- `min_density`/`max_width_px` are heuristics validated on this dataset; exposed via the helper for future
  tuning. Only the mode field is on the detectors initially (YAGNI on extra tunables).

## Files affected

- `src/phenotypic/sdk_/reconnect/_colony_reconnect.py` — add `refine_reconnected_thin` + scipy imports.
- `src/phenotypic/sdk_/reconnect/__init__.py` — export it.
- `src/phenotypic/detect/_two_k_filamentous_detector.py` — `refine_reconnected` field + call.
- `src/phenotypic/detect/_filamentous_fungi_detector.py` — `refine_reconnected` field + call.
- `tests/unit/.../test_colony_reconnect.py` — helper unit tests.
- `tests/unit/detect/test_two_k_filamentous_detector.py` — thin-tendril behavior + none-is-noop tests.
- `tests/unit/detect/test_filamentous_fungi_regression.py` — none-scope legacy-lock stays green.

## Out of scope

- Cleaning agar woven into the *anchored* fan (a separate, less-safe mode).
- Seed-vs-candidate provenance (empirically ≈ blind — rejected).
- Re-Voronoi/relabel after trimming (the helper only zeros pixels).
- Renaming: `refine_reconnected` / `"thin"` / `refine_reconnected_thin` are proposals, open to review.
