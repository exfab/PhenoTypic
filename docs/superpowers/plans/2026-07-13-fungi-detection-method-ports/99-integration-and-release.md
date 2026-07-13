# Shared integration and release plan

## S00: scaffold

Create the import-cheap package only after corrected signatures are frozen:

```text
src/phenotypic/sdk_/reconnect/
  __init__.py
  CLAUDE.md
```

`CLAUDE.md` records array purity, Google docstrings, validation at Python entry points, no pydantic/
`Image`, lazy optional imports, no `fastmath` without a numeric proof, Numba cache clearing, result
dtype/axis documentation, and the reference/drift/mutation obligations. Add an import test that
fails if GUDHI, FilFinder, FilterPy, Astropy, or plotting/UI modules enter `sys.modules`.
Create any corrected-contract aliases needed by parallel modules, including `BoundaryMode`, in
`sdk_/typing_.py` during S00 and pin them in `tests/unit/sdk_/test_typing_aliases.py`.

Do not force unrelated methods into one return type. Use method-specific frozen dataclasses when a
method has multiple observable outputs. Detector strategy adapters convert their results into the
existing `FragmentAssignment`/`FragmentPath` contracts at
`src/phenotypic/sdk_/branch_pathfinding/_dataclasses.py:36-76`.

## S01: detector strategy Seam

Refactor without behavior change before adding strategy branches:

1. Extract the calibration, `apply_filter_cascade`, and paint/dilate block from
   `FilamentousFungiDetector._process_tile` (`src/phenotypic/detect/_filamentous_fungi_detector.py:830-898`)
   into a private method consuming assignments and paths.
2. Keep the current Dijkstra candidate-generation block at lines 812-827 as one adapter.
3. Add orientation to the reconnection data flow without recomputing it.
4. Add Kalman and CA adapters that emit the existing assignments/paths. They do not bypass shared
   path-quality rejection or painting.
5. Add the NFA adapter and its owned detector test
   `tests/unit/detect/test_filamentous_fungi_nfa.py`; it consumes live phase orientation, preserves
   accepted label IDs, and rejects invalid-label rows according to A07.
6. Apply GWDT only in the Dijkstra cost adapter according to the corrected blend/replacement
   equation. Compute nonlocal transforms before tiling.
7. Resolve tensor voting's detector contract at D0. If retained, compute stick/ball on the full
   phase response/orientation field, feed only stick saliency through one explicitly specified
   opt-in cost/evidence equation, keep ball diagnostic, and add disabled-mode and forwarding tests.
   If that equation is not accepted, explicitly defer detector use rather than silently omitting it.
8. Keep `reconnect_strategy="dijkstra"` and all new strengths/modes disabled by default so old JSON
   and default construction reproduce the prior output.
9. Add deterministic global/tile ownership, halo, and conflict rules for any nonlocal strategy.

Tests first pin the extracted Dijkstra path byte-for-byte, then add per-strategy spy and behavioral
tests. Extend back-compat JSON fixtures, pipeline round-trip, prefab forwarding only for intentionally
public fields, GUI registry schema, Literal aliases, and tune annotations.

## S02: public exports, topology extra, and shared ledgers

After individual core review gates:

1. Merge helper exports into `sdk_.reconnect.__init__` in dependency-free order.
2. Add public operations to `enhance`, `refine`, or `detect`, and the A11 path-1 function to
   `analysis`, only when their contract is complete.
3. Add closed-set Literal aliases and tests.
4. Update enhancer taxonomy and detect/enhance tune-coverage gates.
5. Add the topology extra with the pinned stable FilFinder release. Add GUDHI only if A11 path 1,
   2, or 3 enters release scope. Verify compatible bounds at implementation time, then regenerate
   `uv.lock` with `uv` only.
6. Add FilFinder to the dev group only if fixture regeneration needs it. Runtime modules still
   import it lazily.
7. Merge one drift-register row per deviation, preserving per-method IDs.
8. Add a CI/base test that imports and serializes every public operation with the topology extra
   absent.

Official GUDHI installation documentation currently supports Python 3.10+ and major desktop
platform toolchains ([GUDHI installation](https://gudhi.inria.fr/python/latest/installation.html)),
which is compatible with the repository's declared Python 3.10-3.12 range
(`pyproject.toml:24`). Wheel availability must still be verified for every CI platform and pinned
version at implementation time.

## S03: combined gates

### Numerical suite

```bash
for script in docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/*.py; do
  uv run python "$script"
done
```

Every expected script must be enumerated in a manifest test so a missing script is a failure. The
FilFinder adapter script is explicitly scoped to PhenoTypic-owned translations. A11 may remain
absent only if persistence denoising is explicitly deferred and removed from the release scope.

### Focused regression

```bash
uv run pytest tests/unit/sdk_/reconnect -q
uv run pytest tests/unit/enhance tests/unit/refine tests/unit/detect -q
uv run pytest tests/unit/abc_/test_enhancer_taxonomy.py -q
uv run pytest tests/unit/tune/test_annotation_coverage.py tests/unit/tune/test_enhance_annotations.py tests/unit/tune/test_detect_annotations.py -q
uv run pytest tests/unit/sdk_/test_typing_aliases.py tests/unit/gui/test_operation_registry.py -q
uv run mypy src/phenotypic
uv run ruff check
```

### Optional dependency matrix

1. Base environment: import/schema/serialization succeeds; calling optional methods raises targeted
   `ImportError`; no optional module is imported eagerly.
2. Topology environment: real FilFinder and any approved persistence tests run and do not skip.
3. Supported Python/platform matrix: resolve/install wheels or document a guarded platform
   exclusion before merging.

### Final review and simplify

A fresh phase reviewer checks the combined diff for cross-method axis, polarity, normalization,
dtype, label, and default-value inconsistencies. High-confidence findings are fixed and all gates
rerun. A separate quality agent then performs behavior-preserving simplification only, followed by
the full regression suite.

## Release evidence

The pull request must link:

- corrected design revision;
- this plan bundle;
- per-method reference provenance and reconciliation;
- fixture manifests;
- standalone logic-script output;
- mutant-by-test matrix;
- drift register;
- base/topology dependency matrix;
- independent reviewer sign-off for every A01-A11 cluster and the combined seams.

No performance or fungal-detection benefit is claimed without a separately designed benchmark with
ground truth. This release establishes faithful numerical implementations and integration safety.
