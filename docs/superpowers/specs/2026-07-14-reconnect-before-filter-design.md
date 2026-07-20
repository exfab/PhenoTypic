# Reconnect-Before-Filter for Filamentous Detectors — Design

**Date:** 2026-07-14
**Status:** Approved (design), pending implementation plan
**Affects:** `TwoKFilamentousDetector`, `FilamentousFungiDetector`, `sdk_.reconnect`

## Problem

Both `TwoKFilamentousDetector` and `FilamentousFungiDetector` run their overlap filter
(`filter_mask_by_overlap`) **before** the Dijkstra reconnection. The overlap filter keeps
only branch components that physically touch an inoculum center and discards everything
else — which is exactly the set of disconnected hyphal fragments that fade out in the media
and that the reconnection exists to bridge. By the time `identify_pseudo_fragments` and
`reconnect_fragments_tiled` run, those fragments have already been deleted, so reconnection
can only ever repair *pseudo-fragments* (pieces a Voronoi boundary cuts off between two
adjacent wells), never a genuine gap where a hypha faded in the agar.

### Measured evidence

Stage-by-stage replay of `TwoKFilamentousDetector._operate` on a 2×4 colony crop
(`d000273_300_001`, rows 1–2 / cols 1–4), measured on the row-2/col-4 colony:

| stage | r2c4 fg px | Δ | notes |
|---|---|---|---|
| `branch_mask = gated>0` (pre-reconnect) | 11,409 | — | full colony incl. outer hyphal halo + a detached blob |
| `structure_mask = filter_mask_by_overlap` | 8,373 | **−31%** | overlap filter strips the disconnected pieces |
| `partition_by_grid_voronoi` (1st) | 7,033 | −16% | Voronoi + connectivity correction trims tendrils |
| after `reconnect_fragments_tiled` | 7,033 | **+0%** | Dijkstra is a **no-op** |
| final | 7,033 | — | 62% survived; 38% lost |

Globally on this crop, `identify_pseudo_fragments` returned **0 fragments** and reconnection
added **0 pixels**. The reconnection machinery is inert for the case it was built for.

## Goal

Let the Dijkstra reconnection actually see and bridge the disconnected branch fragments,
so faded/severed hyphae are reconnected to their colony instead of being deleted — while
keeping a bit-identical path to today's behavior for backward compatibility and A/B
comparison, and regenerating `FilamentousFungiDetector`'s golden regression baseline.

## Key decisions

1. **Scope: both detectors.** Apply the reorder to `TwoKFilamentousDetector` *and*
   `FilamentousFungiDetector`. FFD's golden regression baseline will be regenerated (see
   Testing), with the current baseline retained as a legacy-behavior lock.
2. **Rollout: flag, default = new.** Add a `reconnect_scope` field to both detectors,
   defaulting to the new fragment-reconnecting behavior, with the legacy behavior reachable
   for comparison.
3. **Shared logic in `sdk_.reconnect`.** A single pure function selects the fragment set;
   both detectors call it. No fragment-selection logic is duplicated in the detectors.

## Core insight — the fix is small and reuses everything

The cost machinery already spans the full image:

- `build_reconnect_cost` consumes the full-frame `pc_sum`/`M`/`m`/`orientation`.
- `reconnect_fragments_tiled` tiles the whole image.
- `prescreen_fragments` + `calibrate_thresholds` + `apply_filter_cascade` already exist to
  reject fragments with no low-cost, PCT-supported path (i.e. agar noise).

The *only* reason nothing reconnects is that `fragment_labels` is empty. So the fix is to
**expand the fragment set to include the disconnected branch components** — precisely the
pixels the overlap filter drops, `colony_mask & ~structure_mask` — and let the existing
prescreen + quality-filter cascade decide which get bridged.

Unbridged fragments require **no new deletion logic**: `reconnect_fragments_tiled` only paints
fragments that pass the quality filter into `colony_labels`. Fragments that fail are never
added, so the existing `final_mask = (colony_labels > 0) | structure_mask` step drops them
automatically (they are neither in `colony_labels` nor in the center-connected
`structure_mask`).

## Architecture

### New pure function (`src/phenotypic/sdk_/reconnect/_colony_reconnect.py`)

```python
def select_reconnect_fragments(
    colony_labels: np.ndarray,      # Voronoi labels built from structure_mask (Dijkstra targets)
    center_mask: np.ndarray,        # inoculum center mask
    colony_mask: np.ndarray,        # branch_mask | center_mask (pre-filter union)
    structure_mask: np.ndarray,     # filter_mask_by_overlap(colony_mask, center_mask)
    *,
    scope: Literal["branches", "pseudo"] = "branches",
    min_fragment_size: int = 1,     # small-object guard for scope="branches"
) -> tuple[np.ndarray, np.ndarray]:  # (central_mask, fragment_labels)
    ...
```

**Semantics**

- `scope="pseudo"` → returns **exactly** `identify_pseudo_fragments(colony_labels, center_mask)`.
  This is a pure pass-through: the legacy path is bit-identical, `min_fragment_size` ignored.
- `scope="branches"` (default) →
  1. `central_mask, pseudo_frag_labels = identify_pseudo_fragments(colony_labels, center_mask)`
  2. `media_frag_mask = colony_mask & ~structure_mask` (the disconnected branch components)
  3. `fragment_mask = (pseudo_frag_labels > 0) | media_frag_mask`
  4. if `min_fragment_size > 1`: `fragment_mask = remove_small_objects(label(fragment_mask), min_fragment_size) > 0`
  5. `fragment_labels = label(fragment_mask)`
  6. return `(central_mask, fragment_labels)`

  `central_mask` (the trusted colony bodies, used to mark near-zero traversal cost) is
  **unchanged** vs the pseudo path. Pseudo-fragments (⊆ `structure_mask`) and media
  fragments (⊆ `~structure_mask`) are disjoint, so the union relabels cleanly.

Exported from `src/phenotypic/sdk_/reconnect/__init__.py`. `identify_pseudo_fragments`
remains public and unchanged.

### Per-detector wiring

Add one field to **both** detectors:

```python
reconnect_scope: Literal["branches", "pseudo"] = "branches"
```

In each `_operate`, replace the `identify_pseudo_fragments(...)` call with:

```python
central_mask, fragment_labels = select_reconnect_fragments(
    colony_labels, center_mask, colony_mask, structure_mask,
    scope=self.reconnect_scope, min_fragment_size=<derived>,
)
```

- **TwoK** (`_two_k_filamentous_detector.py:190-198`): `colony_mask`/`structure_mask`/
  `center_mask` already exist as local names.
- **FFD** (`_filamentous_fungi_detector.py:449-485`): pass `overall_objmask` as `colony_mask`,
  `inoculum_structure_mask` as `structure_mask`, `inoculum_objmask` as `center_mask`.

Everything downstream is unchanged: `build_reconnect_cost`, `reconnect_fragments_tiled`
(incl. FFD's `app2_gwdt` path), and the `final_mask` union + re-Voronoi. `reconnect_scope`
is **orthogonal** to FFD's existing `reconnect_strategy` (scope picks *which* fragments;
strategy picks *edge cost*).

`min_fragment_size` derivation: a small default that drops single/near-single-pixel specks
before Dijkstra (cheap noise + performance win; the prescreen would reject them anyway).
Proposed: derived from `min_branch_width_px` (e.g. `max(1, min_branch_width_px)`), tunable
via a field if needed. Legacy path (`scope="pseudo"`) ignores it entirely.

### Data flow (before → after, `scope="branches"`)

```
branch_mask ─┐
             ├─► colony_mask ─► structure_mask ─► colony_labels (Voronoi)  ─┐
center_mask ─┘        │              │                                      │
                      │              └───────────────┐                      │
                      └─ media frags (colony_mask & ~structure_mask) ─┐     │
                                                                      ▼     ▼
                        pseudo frags (Voronoi cuts) ──────────► select_reconnect_fragments
                                                                      │
                                                                      ▼
                                                 fragment_labels ─► reconnect_fragments_tiled
                                                                      │ (bridged frags painted in;
                                                                      │  unbridged never added)
                                                                      ▼
                                              final_mask = (colony_labels>0) | structure_mask
                                                                      ▼
                                                       partition_by_grid_voronoi → objmap
```

## Testing

### New sdk unit tests (`tests/unit/.../test_colony_reconnect.py`)

- `scope="pseudo"` returns arrays equal to `identify_pseudo_fragments` on the same inputs
  (element-wise equality of both `central_mask` and `fragment_labels`).
- `scope="branches"` with a planted disconnected fragment: `fragment_labels` contains that
  fragment and `central_mask` is unchanged vs the pseudo call.
- Disjointness/labeling: pseudo ∪ media fragments relabel without collision.
- `min_fragment_size` drops sub-threshold specks; `min_fragment_size=1` keeps all.

### Detector behavior tests

- **TwoK** (`tests/unit/detect/test_two_k_filamentous_detector.py`): on a synthetic plate
  with a deliberately-severed hypha (a fixture with a real gap), `reconnect_scope="branches"`
  reconnects it (strictly more recovered branch coverage / fewer fragments) while
  `reconnect_scope="pseudo"` does not. This proves the flag does real work and that the
  quality filter accepts a genuine near-colony fragment.
- A distant agar speck placed far from any colony is **not** reconnected under
  `scope="branches"` (prescreen/quality-filter rejects it) — guards against noise inflation.
- Keep `test_final_objmap_excludes_objects_not_overlapping_centers` green: unbridged
  fragments are still excluded from the final objmap.

### FFD golden regression — re-baseline

`tests/unit/detect/test_filamentous_fungi_regression.py` currently pins FFD's objmap on
`load_synth_filamentous_plate()` to `tests/fixtures/filamentous_fungi_regression_objmap.npy`,
regenerated via `python -m tests.unit.detect.test_filamentous_fungi_regression`.

1. **Legacy lock:** add a test running `FilamentousFungiDetector(reconnect_scope="pseudo", ...)`
   against the **existing** `.npy`, asserting bit-identical. Because `scope="pseudo"` is a pure
   pass-through, this must pass unchanged — it proves the flag preserves legacy behavior.
   (If it does *not* match, the pass-through has a bug — fix before proceeding.)
2. **New default baseline:** regenerate a new fixture
   (`filamentous_fungi_regression_objmap_branches.npy`) with the default
   `reconnect_scope="branches"`, and repoint the default-config golden test at it. Before
   committing, **eyeball the diff** against the legacy golden to confirm the change is *added
   reconnected structure*, not corruption.
3. If `load_synth_filamentous_plate()` contains no disconnected fragments, the two goldens may
   coincide; the golden's job is stability-pinning, and the *behavioral* proof lives in the
   severed-hypha detector test on a fixture built with a gap. Do not force synthetic-plate
   changes solely to differentiate the goldens.

### Mutation check

Reuse the existing gwdt-mutation harness pattern to confirm the new tests can fail
(reintroduce a one-line mutation that empties the media-fragment set and confirm the
severed-hypha test goes red).

## Risks and tuning

- **Latent quality-filter activation.** Feeding real fragments to Dijkstra exercises the
  prescreen/calibration/cost-threshold cascade that was effectively dead in practice. Reach
  is bounded by `max_gap_length` (30) and `frag_reach_px` (10); genuine near-colony hyphae
  pass, distant specks get screened. If defaults over- or under-connect, they are tunable —
  the implementation plan includes a visual A/B on the rows-1–2/cols-1–4 crop before the FFD
  golden is finalized.
- **Performance.** More fragments = more prescreen/Dijkstra work. The `min_fragment_size`
  guard and the existing prescreen keep this bounded; the crop's ~71 dropped components are
  mostly tiny specks screened cheaply before per-tile Dijkstra.

## Files affected

- `src/phenotypic/sdk_/reconnect/_colony_reconnect.py` — add `select_reconnect_fragments`.
- `src/phenotypic/sdk_/reconnect/__init__.py` — export it.
- `src/phenotypic/detect/_two_k_filamentous_detector.py` — add `reconnect_scope` field; swap call.
- `src/phenotypic/detect/_filamentous_fungi_detector.py` — add `reconnect_scope` field; swap call.
- `tests/unit/.../test_colony_reconnect.py` — new unit tests.
- `tests/unit/detect/test_two_k_filamentous_detector.py` — severed-hypha + noise tests.
- `tests/unit/detect/test_filamentous_fungi_regression.py` — legacy-lock test + repointed default golden.
- `tests/fixtures/filamentous_fungi_regression_objmap_branches.npy` — new default baseline (add).

## Out of scope

- The full-orchestrator refactor (Approach C: moving filter→voronoi→reconnect→final into a
  single `sdk_.reconnect` function). Deferred as YAGNI.
- Retaining high-confidence *unbridged* isolated fragments in the output (today's behavior —
  drop anything not center-connected — is preserved).
- Any change to the reconnection cost model, tiling, or quality-filter internals.
- The `"branches"` / `"pseudo"` value labels are open to review; the field name
  `reconnect_scope` is settled (it pairs with FFD's existing `reconnect_strategy`).
