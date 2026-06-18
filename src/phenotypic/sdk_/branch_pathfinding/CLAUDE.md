# Branch Pathfinding

Multi-source Dijkstra pathfinding over image cost surfaces. Extracted
from `FilamentousFungiDetector` and reused by `MeasureRadialExpansion`
(and any future callers). Algorithm-agnostic: **cost surfaces are the
caller's responsibility** — assemble your own from whatever image
features you have (phase congruency, skeletons, distance transforms,
etc.).

## Public API

See `__init__.py` for exports. Functions group by pipeline stage:
cost surface (`_cost_surface.py`) → multi-source Dijkstra
(`_dijkstra_kernels.py`) → fragment prescreening
(`_fragment_prescreening.py`) → path quality (`_path_quality.py`) →
Voronoi partition (`_voronoi_partition.py`) → diagnostics
(`_diagnostics.py`). Dataclasses live in `_dataclasses.py`. Plotly is
imported lazily inside diagnostic function bodies; the subpackage
itself remains import-cheap.

## Numba cache

`_dijkstra_kernels.py` compiles via `@numba.njit(cache=True)`. After
edits to that file (including moving it), delete the on-disk cache:

```bash
find src/phenotypic/tools_/branch_pathfinding -name __pycache__ -exec rm -rf {} +
```

Otherwise numba may try to reuse a stale signature and throw a
`SystemError` at import.

## Direction and `delta` conventions

- **Seed** Dijkstra where you want cost-zero (colony interior for fungi
  detection, core zone for radial expansion).
- **Backtrack** from the points you want routed (fragment pixels for
  fungi, skeleton tips for radial expansion).
- `delta > 0` adds a radial-retreat penalty (wavefronts pay extra for
  steps that move closer to their source centroid). Use a nonzero
  `delta` when the problem has a growth-direction prior; use
  `delta = 0.0` for geometric shortest-cost paths where direction is
  symmetric.

## Adding a new caller

Compose your cost surface outside this package, then call
`run_multisource_dijkstra` directly. Keep domain knowledge (what
"good structure" means in your image, what seeds to use) in your
caller's file; `branch_pathfinding` should stay a pure algorithm
library.
