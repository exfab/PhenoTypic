# Two-`k` phase hysteresis — enhancer + reconstructing detector — design

Status: proposal (not committed). Two new classes packaging the two-`k` phase-congruency
hysteresis developed this session:

- **Class 1 — `FocusEdgeTwoKPhase`** (a `FocusEdge` *enhancer*): runs phase congruency at a strict
  and a loose `k`, builds the hysteresis mask, and returns the **loose-`k` PCT response selected by
  that mask** — a continuous, agar-denoised, reconnected branch response (center hole preserved).
- **Class 2 — `TwoKFilamentousDetector`** (a `GridObjectDetector`): contains Class 1, binarizes its
  non-zero response, fills the inoculum centers, and reconnects fragments with the **existing
  Dijkstra machinery** from `FilamentousFungiDetector`, then labels per colony.

Names are placeholders (`FocusEdgeTwoKPhase` parallels `FocusEdgePhase`/`FocusEdgeMonogenicPhase`).

---

## 1 · Class 1 — `FocusEdgeTwoKPhase` (enhancer)

A `FocusEdge` subclass. Operates on `detect_mat` (assumed already illumination-flattened +
contrast-stretched upstream, exactly like `FocusEdgePhase` assumes a prepared `detect_mat`).

**`_operate`:**
```
strict = _phasecong3(detect_mat, k=k_strict)          # clean, fragmented    (full _PhaseCong3Result)
loose  = _phasecong3(detect_mat, k=k_loose)           # full branches + agar (full _PhaseCong3Result)
seed   = strict.pc_sum > seed_thresh(strict.pc_sum)   # otsu  (strict × strict)
cand   = loose.pc_sum  > cand_thresh(loose.pc_sum)    # triangle (loose × loose)
mask   = reconstruction(seed ∩ cand, mask=cand, "dilation")   # two-k hysteresis: seeds grown into candidates
detect_mat[:] = clip(loose.pc_sum * mask, 0, 1)       # SELECT loose-k response where the mask confirms branch
```

- **Output is continuous, not binary** — the loose-`k` PCT magnitudes wherever the hysteresis mask
  says "real branch", `0` elsewhere. Keeping the magnitude (not a 0/1 mask) preserves it as a
  proper `FocusEdge` response and leaves downstream thresholding / cost-surface / skeleton weighting
  free. (Measured this session: this recovers the faint connectors — ~12 fragments/colony vs 22
  strict-only — while agar stays clean.)
- **Gate the *loose* map, seed from the *strict* one.** The loose map has the fuller branch
  structure; the strict seeds certify which of it is real. Binarize with *opposite* thresholds
  (`otsu` seed / `triangle` cand) — verified this is the only combo that both reconnects and stays
  clean.
- **Center hole preserved.** Phase congruency is an edge detector, so the solid inoculum core stays
  a hole — deliberately (skeleton/network analysis needs it missing; §2).

**Fields:** `k_strict=6.0`, `k_loose=4.5` (4.0 recovers more at a little more agar),
`seed_thresh: Literal["otsu","triangle"]="otsu"`, `cand_thresh=…"triangle"`, `n_orient=8`,
`min_wavelength=5.0`, plus the `NormalizedOutputMixin` `norm` policy.

**Shared kernel (the DRY seam that Class 2 needs).** Factor the body into a helper
```
two_k_phase(detect_mat, k_strict, k_loose, seed_thresh, cand_thresh)
    -> (gated_response: ndarray, loose: _PhaseCong3Result)
```
Class 1 returns only `gated_response` (writes `detect_mat`); **Class 2 reuses the same call to get
`loose`** — whose `M`/`m`/`orientation` the Dijkstra cost surface needs — so the two-PCT cost is
paid **once**, not again in the detector (§3).

---

## 2 · Design fork — center-fill: opt-in on Class 1, or a separate class?

**Assuming** Class 1 is a maintained `FocusEdge` primitive reused by both the detector (Class 2)
*and* the skeleton/network measure (`MeasureHyphalNetwork`, sibling spec). The question: does Class 1
gain an opt-in `include_center` (fill the inoculum core = `ManualGridPointDetector` ∩
`SubtractGaussian`), or is the center-fill a **separate** operation?

**Option A (baseline, recommended): center-fill is a separate operation.** Class 1 stays a pure
branch-response enhancer; the center-fill is its own class (e.g. `FillInoculumCenters`, a
`BackgroundSubtraction`/marker enhancer or an `ObjectRefiner`), and the *detector* unions the two.
- *Deciding axis — the two consumers want opposite things.* This is what actually settles it, and
  it's not one of the generic axes: `MeasureHyphalNetwork` **needs the center absent**
  (`skeletonize` on a solid core manufactures a spurious tip/junction rosette — established in that
  spec), while detection/area **needs it filled**. A pure center-hole-free branch response is the
  more valuable invariant, so it should be the enhancer's *only* behaviour.
- *Coupling / layering.* Center-fill needs a **grid detector** (`ManualGridPointDetector`) and a
  **background subtractor**. Putting those inside a `FocusEdge` makes an *enhancer depend on a
  detector* — a layering inversion and a wide, mostly-unused parameter surface. Separate class
  keeps each concern's dependencies where they belong.
- *Composability.* Two clean ops compose in a pipeline / in Class 2; the detector already
  orchestrates multiple signals, so the union lives naturally there.
- *Cost:* two ops to wire instead of one flag. Minor.

**Option B: opt-in `include_center=False` on Class 1.** One op yields branches-plus-cores when asked.
- *Upside:* one-call convenience for the detection path.
- *Downsides:* Class 1's output type becomes conditional ("pure branch response" *unless* the flag is
  set), so every downstream must know whether the center was filled — precisely the ambiguity that
  hurts the skeleton measure. Plus the detector-dependency layering issue above. Even done well the
  default must be **off**, which means the common path still composes the center-fill separately —
  so the flag mostly adds surface without removing the separate op.

**Recommendation: Option A (separate class).** Confidence: the "skeleton needs no center" half is
**established** (measured/argued in the sibling spec); the layering half is a **judgment call**, but
"a `FocusEdge` enhancer holding a `GridObjectDetector`" is a clear smell. What would change it: if a
GUI/one-shot workflow demanded a single "branches + cores" enhancer badly enough to accept the
conditional-output cost — then add B as **default-off sugar over** the separate class, never as the
only path. Reversibility: high either way (both are additive), so ship A now; B can follow if demand
appears.

---

## 3 · Class 2 — `TwoKFilamentousDetector` (detector)

A `GridObjectDetector`. It is `FilamentousFungiDetector` with its **Phase-2 branch stage replaced**
by Class 1 + the center-fill, and **Phases 3–5 reused**. `_operate`:

```
1. BRANCH        gated, loose = two_k_phase(image.detect_mat, ...)   # Class 1's kernel; keep `loose`
                 branch_mask   = gated > 0                            # "non-zero pixels" = the two-k mask
2. CENTERS       grid_mask     = center_detector(image).objmask       # ManualGridPointDetector at wells
                 center_mask   = grid_mask & (SubtractGaussian(image) > otsu)   # §2 fill, gated to wells
                 markers       = centroids(center_detector.objmap)     # inoculum seeds (Voronoi + Dijkstra source)
3. FILTER+VORONOI  keep branch_mask components overlapping centers; grid-Voronoi seed → colony_labels
4. RECONNECT     cost = assemble_composite_cost(loose.pc_sum, anisotropy(loose.M, loose.m),
                                                coherence(loose.orientation), mad(enhanced), …)
                 colony_labels = multisource_dijkstra + assign + quality-filter + paint   # tiled
5. FINAL VORONOI colony_labels = separate_colonies(markers, reconnected_mask | center_mask)
6. WRITE         image.objmap[:] = colony_labels
```

**The cost-surface constraint is satisfied for free.** FFD's Phase-4 cost surface
(`_build_cost_surface`) consumes the *full* `_PhaseCong3Result` (`pc_sum`, `M`, `m`, `orientation`)
plus a local-MAD map — verified against the source. A generic enhancer output (just `detect_mat`)
wouldn't expose `M`/`m`/`orientation`; but Class 1's **shared kernel already returns `loose`**, so
Class 2 threads it straight into the cost surface. **Use the *loose* (`k=4.5`) result** — its fuller
structure gives Dijkstra more real hyphae to route along than the strict map would.

**What is reused vs replaced (vs `FilamentousFungiDetector`):**
- **Replaced — Phase 2 branch detection.** FFD's dual mask (`ContrastStretching` + `SubtractGaussian`
  mask A + `FocusEdgePhase._phasecong3` mask B + `HysteresisDetector`) → Class 1's two-`k` gated
  response, binarized by non-zero. Cleaner (the two-`k` hysteresis *is* the denoise + reconnect).
- **Replaced — inoculum source.** FFD's `InoculumDetector` pipeline → the grid center-fill
  (`ManualGridPointDetector` ∩ `SubtractGaussian`), for plates with known well coords. (Keep
  `inoculum_detector` swappable for coord-free plates.)
- **Reused verbatim — Phases 3–5.** `_filter_mask_by_overlap`, `_create_markers_from_centroids`,
  `_separate_colonies` (Euclidean Voronoi + connectivity correction), `_build_cost_surface`,
  `_reconnect_fragments_tiled` and all of `sdk_.branch_pathfinding`, plus the scene-derivation of
  `gauss_sigma`/`tile_size`/`mad_window`/… from `max_colony_radius_px`/`min_branch_width_px`.

**Reuse mechanism (decide at implementation):**
- **(a) Extract a mixin/base** holding FFD Phases 3–5 (`_ReconnectAndLabelMixin`), consumed by both
  FFD and Class 2 — DRY, but a refactor of FFD.
- **(b) Subclass `FilamentousFungiDetector`** and override the branch-detection step (and the
  inoculum step) to call Class 1's kernel + the center-fill, leaving Phases 3–5 inherited. Less
  refactor; needs FFD's `_operate` to expose a branch-detection hook (it is currently monolithic, so
  a small `_detect_branches(image) -> (branch_mask, pct_result, enhanced_arr, enhanced_gray)`
  extraction is the minimal enabling change).
- Recommend **(b)** with the one-method extraction — smallest change, and it makes FFD's own branch
  stage overridable, which is generally useful.

---

## 4 · Registration & tests

- `enhance/_focus_edge_two_k_phase.py` → `FocusEdgeTwoKPhase`; re-export + add to the `FocusEdge`
  entry of `tests/unit/abc_/test_enhancer_taxonomy.py`. Shared kernel in
  `enhance/_two_k_phase_kernel.py` (or beside `_focus_edge_phase.py`).
- `detect/_two_k_filamentous_detector.py` → `TwoKFilamentousDetector`; re-export from
  `detect/__init__.py`.
- Tests:
  - **Class 1 gating (load-bearing):** a synthetic strict-fragmented / loose-connected pair →
    output is continuous, non-zero exactly on the reconnected mask, `0` on isolated loose-only agar;
    mutation to `otsu`/`otsu` binarization raises the fragment count (proves the loose-`triangle`
    threshold carries the reconnection).
  - **Center hole preserved:** solid-core synthetic colony → Class 1 output has a `0` core.
  - **Shared kernel:** `two_k_phase` returns a `loose` result whose `M`/`m`/`orientation` are finite
    and shaped like `detect_mat` (the cost-surface contract).
  - **Class 2 end-to-end:** synthetic plate of N wells → N labels; fewer isolated fragments than the
    same run with reconnection disabled (mutation).
  - **Cost-surface reuse:** spy that Phase-4 receives the *loose* `_PhaseCong3Result` (no second PCT
    pass beyond Class 1's two).

## 5 · Open questions

1. **Reuse mechanism (§3 a vs b).** Confirm the minimal FFD `_detect_branches` extraction is
   acceptable, else go with the mixin.
2. **Flatten/stretch placement.** Class 1 assumes an enhanced `detect_mat` (like `FocusEdgePhase`).
   Confirm Class 2 runs `FlattenIllumination(300) → ContrastStretching(70,99)` (no gamma — settled
   this session) before Class 1, either inline or as a `branch_base` pipeline field.
3. **`k_loose` default** (4.0 vs 4.5): 4.0 recovers more connectors (~9 vs 12 frags/colony) at a
   little more agar — pick per whether the downstream leans on Dijkstra to clean up (then 4.5 is
   safer) or wants maximal recall pre-Dijkstra (then 4.0).
4. **Does Class 2 even need Dijkstra after two-`k`?** Two-`k` already gets to ~12 frags/colony; the
   residual are *true empty gaps* (the tested niche where Dijkstra still helps). Worth measuring the
   marginal fragment reduction Dijkstra buys on top of two-`k` before committing to the heavier
   Class 2 — it may be small enough that `LightDetectFungi` (two-`k` + grid-Voronoi, no Dijkstra)
   suffices and Class 2 is only for the hardest plates.
