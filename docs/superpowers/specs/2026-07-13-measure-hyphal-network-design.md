# `MeasureHyphalNetwork` — design

Status: proposal (not committed). A new `MeasureFeatures` subclass that turns the phase-congruency
branch response into a **hyphal network graph** and reports per-colony **FFT-style** branching /
network metrics: skeletonize the branch segmentation → Dijkstra-reconnect fragments → build the
skeleton graph → measure it.

Reference for the analysis: **Fungal Feature Tracker (FFT)** — Vidal-Diez de Ulzurrun, Huang, Chang,
Lin, Hsueh, *PLoS Comput. Biol.* 15(10):e1007428, 2019 (doi:10.1371/journal.pcbi.1007428). FFT
quantifies filamentous-fungus morphology from the skeletonized mycelium: total length, number of
hyphal tips, number of branch points, number of branches, covered area, and derived ratios; the
mycology-standard **hyphal growth unit** (total length / #tips) and box-counting fractal dimension
round out the network descriptors.

## 1 · Scoring target and why the center hole helps

`MeasureFeatures` subclasses read `image.detect_mat` / `objmask` / `objmap` and return a
**per-object DataFrame** (first column `OBJECT.LABEL`, one row per colony label). This one measures
the **branch skeleton** of each colony.

**Input is the PCT *branch* segmentation, not the center-filled composite.** Phase congruency is an
edge detector, so the inoculum core reads as an open hole — and that is exactly what we want here.
`skimage.morphology.skeletonize` on a *solid* disc collapses it into a messy central medial-axis
rosette that manufactures dozens of spurious tips and junctions. With the **core absent**, the
hyphae skeletonize into clean lines that radiate toward (but do not fill) the center, so tips and
branch points are real hyphal features, not core artifacts. Concretely: feed this measure the
**un-filled** branch mask (`LightDetectFungi`'s branch term / `FocusEdgePhase` segmentation), *not*
the `center-fill ∪ branch` composite used for area/detection.

## 2 · Pipeline (inside `_operate`, per colony)

```
objmap (colony labels)  +  branch response (detect_mat = PCT pc_sum)
  │
  ├─ per label ℓ:
  │    mask_ℓ      = objmask & (objmap == ℓ)              # fragmented branch mask, center hole
  │    mask_ℓ'     = dijkstra_reconnect(mask_ℓ, seed=centroid_ℓ, cost=f(response))   # §3
  │    skel_ℓ      = skeletonize(mask_ℓ')                 # clean radiating skeleton
  │    G_ℓ         = skeleton_to_graph(skel_ℓ)            # nodes=tips/junctions, edges=segments
  │    row_ℓ       = network_metrics(G_ℓ, skel_ℓ, mask_ℓ')  # §4 FFT-style
  └─ DataFrame(rows, first col OBJECT.LABEL)
```

Order matters: **reconnect the mask first, then skeletonize.** Reconnecting the *binary mask*
(bridging fragment gaps with a thin painted path) lets `skeletonize` produce a single connected
medial axis; skeletonizing first and trying to stitch skeleton endpoints afterward duplicates the
graph-building work and is more fragile. (Alternative — skeleton-endpoint bridging — noted in §6.)

## 3 · Reconnection — reuse `sdk_.branch_pathfinding`

The Dijkstra machinery is already extracted and algorithm-agnostic
(`run_multisource_dijkstra`, `assign_fragments_to_colonies`, `extract_fragment_paths`,
`assemble_connected_mask`, `prescreen_fragments`, `_path_quality`), exactly what
`FilamentousFungiDetector` Phase 4 uses. Reuse it here rather than reimplementing.

**The one real dependency:** the detector builds its cost surface from the *full*
`_PhaseCong3Result` (`pc_sum`, `M`, `m`, `orientation`) plus a local-MAD map
(`assemble_composite_cost`). A `MeasureFeatures` only receives the finished `detect_mat` (the
clipped `pc_sum`), not `M`/`m`/`orientation`. Two options, decide before implementing:

- **(A, lean)** cost = `1 − normalize(detect_mat)` with the border/distance-gap penalties from
  `branch_pathfinding._cost_surface`. Cheap; no re-run; loses the anisotropy/coherence terms that
  bias paths along hyphae. Good enough for short-gap reconnection.
- **(B, faithful)** re-run `FocusEdgePhase._phasecong3` on the colony's `detect_mat` to rebuild the
  full cost surface exactly as the detector does. Costs one PCT pass per measure; reuses
  `assemble_composite_cost` verbatim.

Recommend **A** as the default (this is a *measurement*, not the detector; a good-enough bridge is
fine) with **B** available via a flag. Seeds are the colony centroids
(`center_of_mass(objmap == ℓ)`), matching `_create_markers_from_centroids`.

## 4 · Metrics (FFT-style), per colony

Build the skeleton graph, then emit one row per label. Nodes classified by 8-neighbour count on the
skeleton: **degree 1 = tip (endpoint)**, **degree ≥ 3 = branch point (junction)**; edges are the
pixel chains between nodes.

| Feature (`NETWORK.*`) | Definition | FFT analogue |
|---|---|---|
| `TOTAL_LENGTH` | Σ edge arc-length (px; ×`px_size` if calibrated) | total mycelium length |
| `N_TIPS` | # degree-1 nodes | number of tips |
| `N_BRANCH_POINTS` | # degree-≥3 nodes | number of branch points |
| `N_BRANCHES` | # edges | number of branches |
| `MEAN_BRANCH_LENGTH` / `MEDIAN_BRANCH_LENGTH` | over edges | segment length stats |
| `MEAN_TORTUOSITY` | mean(edge arc-length / endpoint-euclidean) | segment straightness |
| `MEAN_BRANCH_ANGLE` | mean angle between edges meeting at each junction | branching angle |
| `HYPHAL_GROWTH_UNIT` | `TOTAL_LENGTH / N_TIPS` | HGU (mycology standard) |
| `MYCELIAL_AREA` | reconnected-mask pixel count | covered area |
| `MEAN_HYPHAL_WIDTH` | 2 × mean(distance-transform at skeleton pixels), or `AREA/TOTAL_LENGTH` | hyphal width |
| `FRACTAL_DIMENSION` | box-counting slope on the skeleton | network complexity |
| `N_ISOLATED_FRAGMENTS` | # connected components still unreconnected | reconnection QC |

Add a `NETWORK` feature enum beside the existing `SHAPE`/`INTENSITY` enums (the `str(ENUM.X)`
column-name convention). Units: default pixels; multiply lengths by a `px_size_mm` field when set,
so downstream tables are in mm (matching the pipeline's calibrated measures).

## 5 · Graph backend — add `skan`

`skan` (the standard skeleton-analysis library) is **not** currently installed; `networkx` **is**.
`skan.Skeleton` / `skan.summarize` give exactly the per-branch table FFT needs — branch length,
euclidean distance, `branch-type` (0 tip–tip, 1 junction–tip, 2 junction–junction, 3 cycle), and
node coordinates — in one call, and is the lightest path to the §4 metrics.

- **Recommend adding `skan`** as a dependency (`pyproject` optional-extra `network` if we want to
  keep it out of the core install). Fractal dimension and branch angle are computed on top of its
  node/edge tables.
- **Fallback (no new dep):** build the graph with `networkx` from a neighbour-count classification
  (degree-1 tips, degree-≥3 junctions, trace edges between them). More code, same metrics; keep as
  the `backend="networkx"` option if adding `skan` is undesirable.

## 6 · Architecture note — measure vs refiner (flag for the reviewer)

Bundling **reconnection** (which *mutates* a mask) inside a `MeasureFeatures` is slightly against the
grain: measures are otherwise read-only and return a DataFrame. Two clean shapes:

- **(bundled, as directed)** `MeasureHyphalNetwork` does reconnect→skeletonize→measure internally,
  keeping the reconnected mask/skeleton local (optionally surfaced via a `FigureProvider` overlay,
  not persisted to `objmask`). Simplest to wire into an existing `meas={}` block.
- **(split, cleaner reuse)** a `ReconnectBranches(ObjectRefiner)` produces the connected `objmask`
  upstream (reusing `FilamentousFungiDetector`'s exact Phase-4 path), and `MeasureHyphalNetwork`
  then only skeletonizes + measures. This makes reconnection reusable outside measurement and keeps
  the measure read-only, but adds a pipeline step.

Recommend **bundled** for the first cut (matches the directive and a single `meas` entry), with the
refiner split called out as the refactor once reconnection is needed elsewhere.

## 7 · Files, registration, tests

- `measure/_measure_hyphal_network.py` → `MeasureHyphalNetwork(MeasureFeatures)`; re-export from
  `measure/__init__.py` (import + `__all__`). Add the `NETWORK` enum to the features-enum module.
- `pyproject.toml`: `skan` (core or `network` extra); guard the import with a clear
  `ImportError` message if kept optional.
- Tests (`tests/unit/measure/`):
  - **Synthetic Y:** a hand-built 3-branch "Y" skeleton → assert `N_TIPS==3`, `N_BRANCH_POINTS==1`,
    `N_BRANCHES==3`, and `TOTAL_LENGTH` within 1 px of the drawn length. Pins the graph metrics.
  - **Center-hole benefit (the load-bearing one):** skeletonize a solid disc vs the same disc with
    a hole; assert the holed version yields **far fewer spurious tips** — the justification for §1.
  - **Reconnection effect:** a colony split into 2 fragments by a small gap → `N_ISOLATED_FRAGMENTS`
    drops to 0 after reconnection and `N_BRANCHES` increases by the bridge; mutation-disable the
    reconnect and confirm the fragment count rises.
  - **Contract:** returns a DataFrame with `OBJECT.LABEL` first, one row per `objmap` label, no
    mutation of `objmask`/`objmap`.
  - **Calibration:** `px_size_mm` scales lengths linearly; areas by its square.

## 8 · Open questions

1. **Cost-surface fidelity (§3 A vs B).** Start with A (normalized-response cost); measure whether
   B's anisotropy-aware cost meaningfully changes reconnection on real plates before paying the
   per-colony PCT re-run.
2. **Which mask feeds it.** Confirm the pipeline can hand this measure the *branch-only* (un-filled)
   segmentation while other measures (area/shape) use the center-filled one — i.e. this measure may
   need its own `detect_mat`/`objmask` source, or run before the center-fill union. Resolve against
   `LightDetectFungi`'s output contract.
3. **Fractal dimension definition.** Box-counting on the skeleton vs on the filled mask gives
   different numbers; FFT uses the mycelium image. Pick one and document it (skeleton is the more
   reproducible across thresholds).
4. **`skan` as core vs optional dep.** If core-install weight matters, ship it behind a `network`
   extra with a `networkx` fallback; otherwise core is simpler.
