# A05: Kalman predict-and-coast reconnection

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Keystone core followed by detector Seam
**Blocked by:** C6 and S00

## Contract decision

Choose a bounded combinatorial tracker or replace this plan with a separately named
`kalman_nearest_coast` plan. A true combinatorial
filter branches hypotheses and therefore requires scoring, pruning, tie-breaking, missed-hit
penalty, and a hypothesis cap. The design's `(endpoints, theta, gate_chi2)` cannot determine the
described tracks. The exact port source must be selected from primary Kalman/track-finding sources,
all of which are older than five years, such as [Frühwirth 1987](https://doi.org/10.1016/0168-9002(87)90887-4), [Mankel 1997](https://doi.org/10.1016/S0168-9002(97)00705-5), and [Bertacchi et al. 2021](https://doi.org/10.1016/j.cpc.2020.107610).

Recommended bounded interface:

```python
@dataclass(frozen=True)
class KalmanCoastResult:
    hypothesis_offsets: np.ndarray       # int64, (H + 1,)
    hypothesis_seed_indices: np.ndarray  # int64, (H,)
    hypothesis_component_ids: np.ndarray # int64, (H,)
    states: np.ndarray                   # float64, (T, 4)
    covariances: np.ndarray              # float64, (T, 4, 4)
    innovations: np.ndarray              # float64, (T, 2), zero on a miss
    innovation_covariances: np.ndarray   # float64, (T, 2, 2)
    mahalanobis_sq: np.ndarray           # float64, (T,), zero on a miss
    measurement_valid: np.ndarray        # bool, (T,)
    coast_flags: np.ndarray              # bool, (T,)
    selected_hit_indices: np.ndarray     # int64, (T,), -1 on a miss
    scores: np.ndarray                   # float64, (H,)
    termination_codes: tuple[KalmanTermination, ...]  # length H
    selected_hypothesis_indices: np.ndarray  # int64, one per accepted output track

def kalman_coast(
    seed_endpoints_rc: np.ndarray,       # (N, 2)
    seed_directions_rc: np.ndarray,      # (N, 2) outward unit vectors
    seed_component_ids: np.ndarray,      # (N,)
    candidate_hits_rc: np.ndarray,       # (M, 2)
    candidate_layers: np.ndarray,        # (M,) ordered pseudo-time/layer
    candidate_component_ids: np.ndarray, # (M,)
    *,
    step_px: float,
    gate_chi2: float,
    initial_position_var: float,
    initial_direction_var: float,
    process_var: float,
    measurement_var: float,
    max_coast_steps: int,
    max_steps: int,
    max_hypotheses: int,
    miss_penalty: float,
) -> KalmanCoastResult: ...
```

`hypothesis_offsets` partitions every length-`T` history array into `H` hypotheses. Hypotheses are
ordered by `(seed_component_id, seed_index, score, selected_hit_index_sequence)`, and selected
tracks preserve that order. All arrays are C-contiguous and read-only in the frozen result. An
empty input returns offsets `[0]`, no termination codes, and correctly shaped zero-length arrays.
Reject nonfinite coordinates/directions/parameters, non-unit or zero seed directions, shape or
length mismatches, negative variances/penalties, nonpositive limits/step/gate, unsorted layers,
unknown component IDs, and candidate indices that cannot be represented as `int64`.

Use state `[row, col, v_row, v_col]`, position measurement, source-selected constant-velocity
matrices, Joseph covariance update, linear solves instead of explicit inverse, inclusive squared
Mahalanobis gate, deterministic score/index order, and a result containing all states,
covariances, innovations, gates, coast flags, hypotheses, selected hit indices, score, and
termination. Keep continuous-to-pixel rasterization in the detector adapter, where image shape and
clipping policy exist.
Raw phase orientation is axial; the detector adapter must resolve outward direction from endpoint
geometry. Do not reuse `max_gap_length`, which currently controls path-quality windows
(`src/phenotypic/detect/_filamentous_fungi_detector.py:830-848`).

## Owned files and tasks

```text
src/phenotypic/sdk_/reconnect/_kalman.py
tests/unit/sdk_/reconnect/test_kalman.py
tests/fixtures/reconnect/kalman/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/kalman.py
refs/kalman corpus and reconciliation
```

1. Pin exact equations/source implementation and decide combinatorial versus nearest.
2. Freeze matrices, units, direction convention, gate dimension, coast/hypothesis limits,
   continuous result, and invalid/singular behavior. Freeze rasterization separately in the
   detector adapter.
3. Capture all result fields from the selected oracle.
4. Write closed-form script and failing helper tests.
5. Implement one NumPy backend. Do not add optional FilterPy numerical drift.
6. Integrator extracts seed endpoints/directions and ordered candidate hits/layers. It freezes which
   roles are colony versus fragment, rejects self/ambiguous multi-colony links, rasterizes continuous
   states inside the image, samples the shared raw cost surface to construct `cost_profile`,
   `total_cost`, `mean_cost`, and `path_length`, then adapts tracks to existing
   `FragmentAssignment` and `FragmentPath`
   (`src/phenotypic/sdk_/branch_pathfinding/_dataclasses.py:36-76`) before reusing quality and paint.
7. Prefer global sparse tracking. If tiled, specify sufficient halo, core-only commits, and
   deterministic deduplication.
8. Preserve byte-identical Dijkstra behavior when selected.
9. Reviewer audits matrices, every golden field, mutants, and detector seam.

## Logic-validation script

For `m` misses derive:

\[
x_m=F^m x_0,\qquad
P_m=F^mP_0(F^m)^T+\sum_{i=0}^{m-1}F^iQ(F^i)^T.
\]

Derive `S`, squared Mahalanobis distance, boundary innovations from eigenvectors of correlated `S`,
gain, updated state, and Joseph covariance. Check straight/diagonal tracks, an anisotropic-gate
distractor, empty gates, gap counter reset, axial equivalence plus outward sign, PSD/symmetry,
bounded termination, deterministic ties, and all fixture fields. Derive tolerance from float64
epsilon, matrix norms, condition numbers, and operation counts.

## Required core mutants

- remove velocity coupling, Q, or R;
- substitute Euclidean gate or compare `d` to a squared threshold;
- `<` versus `<=` boundary;
- row/column or sine/cosine swap;
- treat axial orientation as directed;
- update on an empty gate or freeze coast covariance;
- fail to reset/limit coast count;
- remove hypothesis branching, pruning, or deterministic tie-break;
- replace Joseph update with an asymmetric form;

## Required post-S01 seam mutants

- admit self-component endpoints or ambiguous multi-colony ownership;
- rasterize with swapped row/column coordinates or wrong clipping;
- omit raw-cost sampling or miscompute `FragmentPath` summary fields;
- remove tile halo/core ownership/deduplication if tiling remains;
- change disabled Dijkstra behavior.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/kalman.py
uv run pytest tests/unit/sdk_/reconnect/test_kalman.py -q
uv run mypy src/phenotypic/sdk_/reconnect/_kalman.py
uv run ruff check src/phenotypic/sdk_/reconnect/_kalman.py tests/unit/sdk_/reconnect/test_kalman.py
```

After S01, the algorithm reviewer returns for:

```bash
uv run pytest tests/unit/detect/test_filamentous_fungi_detector.py tests/unit/sdk_/test_branch_pathfinding.py -q
uv run mypy src/phenotypic
uv run ruff check
```
