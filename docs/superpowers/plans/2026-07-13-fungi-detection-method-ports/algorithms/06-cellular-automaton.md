# A06: cellular-automaton track finder

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Keystone graph core followed by detector Seam
**Blocked by:** C7 and S00

## Corrected contract

The current `(points, theta, tol)` helper lacks the directed graph that makes CA state meaningful.
HEP trackers get direction from detector layers; a fungal ridge field does not. The selected
layer definition is therefore a PhenoTypic capability adaptation and must be explicit. Standard
CA tracking increments states synchronously on a directed friendship graph and then extracts
descending-state chains ([Track Finding, 2021](https://link.springer.com/chapter/10.1007/978-3-030-65771-0_5)).

Prefer splitting source-faithful evolution from fungal graph construction:

```python
@dataclass(frozen=True)
class CATrackResult:
    segments: np.ndarray                  # int64, (M, 2), point-index pairs
    compatibility: np.ndarray             # int64, (K, 2), segment-index pairs
    segment_seed_component_ids: np.ndarray # int64, (M,), -1 means unowned
    states: np.ndarray                    # int64, (M,)
    predecessor_offsets: np.ndarray       # int64, (M + 1,)
    predecessors: np.ndarray              # int64, (P,)
    track_offsets: np.ndarray             # int64, (R + 1,)
    track_segment_indices: np.ndarray     # int64, (Q,)
    track_seed_component_ids: np.ndarray  # int64, (R,)
    convergence_iterations: int

def build_ca_segment_graph(
    points: np.ndarray,
    theta: np.ndarray,
    layers: np.ndarray,
    component_ids: np.ndarray,
    endpoint_roles: np.ndarray,
    *,
    angle_tol: float,
    max_link_distance: float,
    max_layer_skip: int = 1,
) -> tuple[np.ndarray, np.ndarray]: ...  # segments (M,2), compatibility pairs (K,2)

def evolve_ca(
    edges: np.ndarray,          # (M,2) directed point-index segments
    compatibility: np.ndarray,  # (K,2) directed compatible segment-index pairs
    segment_seed_component_ids: np.ndarray,  # (M,), -1 means unowned
) -> CATrackResult: ...
```

Offsets encode ragged predecessors and tracks without object arrays. Segments are ordered by
`(source_point, destination_point)`, compatibility pairs lexicographically, and tracks by
`(seed_component_id, descending segment-index sequence)`. All arrays are C-contiguous and read-only.
An empty graph returns offsets `[0]`, zero iterations, and correctly shaped zero-length arrays.
Reject nonfinite point/orientation data, invalid shapes/dtypes/indices, non-axial angles, negative
or unknown component IDs other than the `-1` sentinel, nonpositive distances, invalid tolerances,
unsorted layers, duplicate segments/pairs, cycles, and seed ownership conflicts. Freeze `(row,col)` coordinates, axial tangent convention, radians,
layer DAG, component/self-link rules, endpoint-role vocabulary, inclusive boundaries, snapshot
updates, cycle rejection, and deterministic ordering. CA nodes are directed point-to-point segments;
compatibility pairs identify predecessor/successor segment cells.

## Owned files and tasks

```text
src/phenotypic/sdk_/reconnect/_cellular_automaton.py
tests/unit/sdk_/reconnect/test_cellular_automaton.py
tests/fixtures/reconnect/cellular_automaton/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/cellular_automaton.py
refs/cellular_automaton corpus and reconciliation
```

1. Pin the selected CATS/TrickTrack/related implementation, license, update rule, and extraction.
2. Define fungal points, layer construction, link distance/skip, orientation compatibility, cycle
   handling, fork enumeration, and result ordering.
3. Capture straight, equal/unequal fork, skipped layer, distractor, boundary, permutation, and
   isolated cases with every graph/result field.
4. Write independent synchronous and topological-DP oracles, then red tests.
5. Implement graph builder and Numba-compatible snapshot evolution.
6. Integrator passes detector orientation/components/roles into reconnection, rejects ambiguous
   multi-colony tracks, rasterizes accepted links, samples the shared raw cost surface to construct
   complete `FragmentPath` cost profiles and `FragmentAssignment` costs, and reuses quality
   filtering and path painting at
   `src/phenotypic/detect/_filamentous_fungi_detector.py:830-898`.
7. Specify ambiguous multi-colony rejection and deterministic overlap-tile conflict handling.
8. Preserve byte-identical Dijkstra behavior by default.
9. Reviewer reruns source, oracles, mutants, and detector integration.

## Logic-validation script

Compare snapshot CA iteration against independent longest-path dynamic programming on the same
DAG. Check exact chain depth and convergence iterations, equal/unequal forks, isolated cells,
layer-skip gate, exact angle/distance boundaries, input-permutation invariance, sequential-update
counterexample, cycle rejection, canonical output, and every fixture key. State/track assertions are
exact; derive only the graph-construction cosine/distance ulp bound.

## Required core mutants

- in-place rather than synchronous update;
- wrong predecessor-state rule or min instead of max;
- strict/inclusive angle or distance boundary change;
- degrees/radians or \(\pi\)-fold error;
- row/column swap or distance gate removal;
- layer skip/backward-link error or early convergence;
- keep one predecessor, drop endpoint, or nondeterministic set order;

## Required post-S01 seam mutants

- fail to interpolate long raster links or swap row/column rasterization;
- assign an ambiguous track to the first colony;
- omit raw-cost sampling or miscompute `FragmentPath` summaries;
- change tile conflict ownership or disabled Dijkstra behavior.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/cellular_automaton.py
uv run pytest tests/unit/sdk_/reconnect/test_cellular_automaton.py -q
uv run mypy src/phenotypic/sdk_/reconnect/_cellular_automaton.py
uv run ruff check src/phenotypic/sdk_/reconnect/_cellular_automaton.py tests/unit/sdk_/reconnect/test_cellular_automaton.py
```

After S01/S02, the algorithm reviewer returns for:

```bash
uv run pytest tests/unit/detect/test_filamentous_fungi_detector.py -q
uv run pytest tests/unit/tune/test_detect_annotations.py tests/unit/tune/test_annotation_coverage.py tests/unit/sdk_/test_typing_aliases.py -q
uv run mypy src/phenotypic/detect/_filamentous_fungi_detector.py
uv run ruff check src/phenotypic/detect/_filamentous_fungi_detector.py
```
