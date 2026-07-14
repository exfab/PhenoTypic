# A01: APP2 GWDT and detector cost integration

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn that did not author A01
**Shape:** Keystone core followed by detector Seam integration
**Blocked by:** C1, C2, C14 and S00

## Corrected contract required

The current design's inverse-intensity geodesic is not the GWDT described by APP2. APP2's
GWDT is background-seeded and accumulates image intensity along the shortest path; a later
fast-marching stage applies a separate image-derived cost ([Xiao and Peng 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3661058/)). Freeze these as separate functions after a
line-by-line Vaa3D reconciliation:

```python
def grey_weighted_distance(
    image: np.ndarray,
    background: np.ndarray,
    *,
    connectivity: Literal[4, 8] = 8,
) -> np.ndarray: ...

def app2_gwdt_cost(
    distance: np.ndarray,
) -> np.ndarray: ...
```

`app2_gwdt_cost` uses the selected executable source's fixed 256-entry `givals`
lookup. It has no tunable lambda. Continuous exponential evaluation is a different
adaptation and must not be accepted through a quantization tolerance.

If `seeds`, `weight_exp`, or `eps` are retained, label the result a PhenoTypic adaptation and
write one drift row per parameter. Do not claim exact EDT reduction. Keep threshold/background
policy in the detector and make the new term explicitly opt-in so the default Dijkstra output is
unchanged.

The integrated detector policy uses the complement of the full-image dual-mask fungal foreground
as APP2 background and rejects all-background or all-foreground masks. It computes the GWDT and
fixed GI lookup once before tiles are generated. `reconnect_strategy="app2_gwdt"` then replaces
only candidate-path propagation with the APP2 endpoint-average GI recurrence. The legacy
composite surface remains the prescreening and path-quality evidence, but its destination-only
edge cost, EDT gap penalty, and radial-retreat multiplier are not added to APP2 edges. The default
`reconnect_strategy="dijkstra"` retains the prior path exactly. These detector-specific decisions
are recorded separately from the source-faithful helper in drift rows D10-D12.

Require a real-valued, 2-D, finite, nonnegative image and a same-shaped boolean background mask.
Reject negative values, NaNs, infinities, empty arrays, shape mismatches, and non-boolean masks.
The explicit mask is a contract adaptation of Vaa3D's `image <= threshold` seed selection; the
detector owns threshold construction. Preserve all remaining executable behavior, including
float32 state, input-valued seed initialization, the initialization-frontier omission of diagonal
length, the ordinary recurrence's diagonal length, all-background output equal to the seed input
values, and the no-background float32 `1e20` sentinel map. Detector adapters validate that both
background and foreground exist before calling the source-faithful helper.

## Owned files

```text
src/phenotypic/sdk_/reconnect/_gwdt.py
tests/unit/sdk_/reconnect/test_gwdt.py
tests/fixtures/reconnect/gwdt/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/gwdt.py
refs/gwdt source corpus and reconciliation
```

The integrator owns exports, detector fields, `_build_cost_surface`, tune annotations,
serialization fixtures, and the drift-register merge. Compute GWDT on the full image before
tiled reconnection. The current detector builds its cost at
`src/phenotypic/detect/_filamentous_fungi_detector.py:616-669` and already applies an EDT gap
penalty at lines 598-614.

## Tasks

1. Pin the APP2 paper and Vaa3D files `fastmarching_dt.h`, `fastmarching_tree.h`, related heap/
   macro files, connector, license, commit, and hashes.
2. Reconcile seed initialization, destination/source intensity, neighbor lengths, connectivity,
   threshold semantics, parent map, dtype, tie-breaking, and output transform line by line.
   Record the 3-D-to-2-D reduction as a drift row. Prove with a one-slice source-harness case which
   in-plane neighbors and diagonal weights correspond to each selected 2-D connectivity.
3. Capture full distance and transformed-cost maps from a small source harness.
4. Write failing helper/oracle/invalid-input tests and the standalone script.
5. Implement the exact bounded two-phase Vaa3D heap recurrence with no `phenotypic` dependency.
6. Prove the golden fixture fails for the source-versus-destination intensity mutant.
7. Integrator adds an opt-in detector field, one defined blend/replacement equation, serialization,
   and exact disabled-mode regression.
8. Reviewer reruns the source harness, fixture, mutations, full-image/tile seam check, and focused
   detector tests.

## Logic-validation script

Use repeated whole-grid relaxation on tiny arrays with explicit source-frontier initialization,
followed by ordinary recurrence relaxation. This is structurally independent of the production
heap. Also exhaustively enumerate simple paths after the initialized frontier for the smallest
two-route cases. Verify: analytic 1-D cumulative sums; axis cost 1; initialization diagonal cost 1;
post-frontier diagonal cost \(\sqrt2\); multiple-background source behavior; a two-route analytic choice;
intensity monotonicity; added-seed monotonicity; positive scaling; transpose/reflection/rotation
equivariance only where the source's initialization order permits it; 4-connectivity no smaller
than 8-connectivity on controlled cases; center enhancement in a thick bar; fixed lookup cost
preference; all-background input-valued behavior; no-background sentinel behavior; and
deterministic failures for negative, nonfinite, empty, or mismatched arrays/masks.

Derive tolerance from the longest asserted path's accumulation bound plus square-root rounding.
Fixture keys include inputs, masks, both connectivities, complete maps, parent map if public, and
the transformed cost.

## Required mutants

- background versus colony seeds;
- inverse versus raw intensity in GWDT;
- source versus destination intensity;
- endpoint average inserted;
- add a diagonal factor to the initialization frontier or remove it from ordinary recurrence;
- 4-connectivity forced;
- additive recurrence replaced;
- threshold `<` versus `<=`;
- hidden normalization;
- downstream transform inverted/removed;
- cumulative GWDT used directly as a local Dijkstra term;
- per-tile instead of full-image transform;
- disabled mode changes legacy output.
- accept negative/nonfinite intensity or change source all/no-background behavior;
- change the documented one-slice 3-D-to-2-D neighbor reduction.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/gwdt.py
uv run pytest tests/unit/sdk_/reconnect/test_gwdt.py tests/unit/detect/test_filamentous_fungi_detector.py -q
uv run pytest tests/unit/tune/test_detect_annotations.py -q
uv run mypy src/phenotypic/sdk_/reconnect src/phenotypic/detect/_filamentous_fungi_detector.py
uv run ruff check src/phenotypic/sdk_/reconnect tests/unit/sdk_/reconnect
```
