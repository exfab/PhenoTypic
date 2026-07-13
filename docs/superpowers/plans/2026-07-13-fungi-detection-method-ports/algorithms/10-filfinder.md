# A10: FilFinder diagnostic wrapper

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Keystone external-wrapper integration
**Blocked by:** C12 and S02 topology extra

## Corrected wrapper contract

Pin the latest stable version selected at implementation time. The July 2026 review found stable
`fil-finder==1.8` ([PyPI release metadata](https://pypi.org/project/fil-finder/)); verify that again
when locking because package metadata is temporally unstable.
The moving development documentation must not be the fixture authority. The maintained API uses a
multi-stage `FilFinder2D` workflow with skeletonization and pruning
([official API](https://fil-finder.readthedocs.io/en/latest/api/fil_finder.FilFinder2D.html)).

`ObjectDetector` has no graph output channel. This cluster returns one selected raster product and
labels its connected components. A future graph product needs a separate analysis design.

```python
class FilFinderDetector(ObjectDetector):
    threshold: float = 0.5
    output: FilFinderOutput = "mask"  # mask, skeleton, longest_path
    beamwidth_px: float = 1.0
    prune_criteria: FilFinderPruneCriteria = "all"
    relative_intensity_threshold: float = 0.2
    skeleton_threshold_px: float | None = None
    branch_threshold_px: float | None = None
    max_prune_iterations: int = 10
    rng_seed: int = 0
```

Freeze threshold comparison, output attributes, beam/pixel units, pruning parameter mapping,
connectivity, label ordering, empty-mask behavior, and one-pixel skeleton warning. Instantiate a
fresh FilFinder object per application, use the supplied threshold mask, skip FilFinder flattening/
adaptive segmentation, pass pixel quantities, label the selected raster, and preserve all image
source layers.

Freeze an explicit stage graph per output. `output="mask"` runs mask creation only and must not call
`medskel` or graph analysis. `output="skeleton"` runs mask creation plus `medskel` and must not run
branch pruning/longest-path analysis. `output="longest_path"` runs all three stages. Parameters for
skipped stages remain serializable but inactive and must not alter output or trigger validation
against an unavailable downstream stage. Add call-spy assertions for the exact ordered calls and
inactive-field behavior of each output.

## Owned files and tasks

```text
src/phenotypic/detect/_filfinder_detector.py
tests/unit/detect/test_filfinder_detector.py
tests/fixtures/reconnect/filfinder/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/filfinder.py
refs/filfinder corpus and adapter reconciliation
```

1. Pin stable source/tag, wheel hash, license, API calls/attributes, Astropy and NetworkX/scikit-
   image transitive versions.
2. Freeze threshold, outputs, units, pruning, seed, connectivity, labels, and dependency message.
3. Implement class with lazy imports and actionable call-time `ImportError`, following the pattern
   at `src/phenotypic/detect/nn/_microsam_detector.py:170-189`.
4. Add fake-object forwarding tests before the real optional dependency tests.
5. Capture straight, Y-spur, disconnected, loop/branch, noise, threshold-boundary, empty, and
   symmetric-tie fixtures with every wrapper-visible intermediate.
6. Integrator pins `fil-finder` in the topology extra and dev oracle group, updates lockfile,
   exports, Literal aliases, tune coverage, GUI discovery, and serialization.
7. Reviewer runs base and topology environments separately and audits every mutation.

## Honest logic-validation scope

Do not reimplement FilFinder and call it independent validation. The standalone adapter script may
only verify PhenoTypic-owned threshold comparison, threshold monotonicity, deterministic connected-
component labeling, `objmask == objmap > 0`, empty input, and pixel-value pass-through. It must state
that skeletonization, graph pruning, longest path, widths, and orientations are validated only by
the pinned external fixture and behavioral controls.

Fixture every selected output plus threshold mask, FilFinder mask, distance, pre/post-prune
skeleton, longest path, label map, lengths, parameters, dependency versions, seed, and platform.

## Required mutants

- strict versus inclusive threshold or wrong image layer;
- omit existing-mask/skip-flatten forwarding;
- pass bare floats rather than pixel quantities;
- swap/ignore prune fields;
- reuse one stateful FilFinder object or omit seed;
- select pre-pruned/full rather than requested raster;
- label original mask for skeleton output or wrong connectivity;
- update objmask without objmap;
- modify detect_mat;
- import optional dependency at module import;
- swallow missing dependency and emit empty output.
- execute a skipped downstream stage or let an inactive parameter change an earlier output.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/filfinder.py
uv run pytest tests/unit/detect/test_filfinder_detector.py -k 'not real_filfinder' -q
uv run pytest tests/unit/tune/test_detect_annotations.py tests/unit/tune/test_annotation_coverage.py tests/unit/sdk_/test_typing_aliases.py tests/unit/gui/test_operation_registry.py -q
uv run mypy src/phenotypic/detect/_filfinder_detector.py
uv run ruff check src/phenotypic/detect/_filfinder_detector.py tests/unit/detect/test_filfinder_detector.py
uv sync --extra topology --group dev
uv run pytest tests/unit/detect/test_filfinder_detector.py -q
```
