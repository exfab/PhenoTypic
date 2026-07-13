# A11: persistence-based ridge denoising

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** blocked Keystone research/contract cluster
**Blocked by:** C11 and S02 topology extra

## Review outcome and required scope choice

Do not implement the proposed enhancer from the current spec. GUDHI's cubical API computes
persistence intervals and representative cells, but does not reconstruct a scalar image after
feature cancellation ([GUDHI cubical reference](https://gudhi.inria.fr/python/3.8.0/cubical_complex_ref.html)). The claimed “small cancellation/Morse step” is therefore not established.

Choose one path and revise the design before coding:

1. **Analysis-only:** expose persistence pairs/Betti summaries, remove the denoising enhancer.
2. **Narrow 0-D denoiser:** select and pin a merge-tree/component-tree pruning and reconstruction
   algorithm for superlevel bright ridges; validate only beta-0.
3. **Full beta-0/beta-1 cancellation:** select an exact discrete-Morse cancellation and scalar-field
   reconstruction source, with executable oracle and compatible license.

The plan recommends path 2 only if a source-faithful reconstruction is found. Otherwise defer A11
rather than shipping an operation that computes pairs but performs an ad hoc intensity edit.
If path 1 is selected, the work belongs in a standalone analysis surface, not a `FocusEdge`
operation. If path 2 or 3 is selected, decide from the reconstructed output whether the wrapper is
a ridge-response `FocusEdge` or a general `ImageDenoiser`; update the taxonomy only after that
decision.

Path 1 has this concrete analysis-only surface:

```python
@dataclass(frozen=True)
class PersistencePairsResult:
    birth_values: tuple[np.ndarray, ...]
    death_values: tuple[np.ndarray, ...]
    lifetimes: tuple[np.ndarray, ...]
    birth_cells: tuple[np.ndarray, ...]
    death_cells: tuple[np.ndarray, ...]
    essential_cells: tuple[np.ndarray, ...]
    filtration: str

def cubical_persistence(
    image: np.ndarray,
    *,
    filtration: Literal["sublevel", "superlevel"] = "superlevel",
    min_persistence: float = 0.0,
) -> PersistencePairsResult: ...
```

It is exported as `phenotypic.analysis.cubical_persistence`; it is not an operation and is not
serialized in an image pipeline. Paths 2 and 3 add the blocked reconstruction helper only after a
source is selected:

```python

def persistence_denoise(
    image: np.ndarray,
    threshold: float,
    *,
    reconstruction: PersistenceReconstruction,
) -> np.ndarray: ...  # remains blocked until reconstruction is sourced
```

All returned values use the original image's intensity coordinates. For superlevel filtration,
GUDHI receives `-image`, but results are converted back so `birth_value >= death_value` and
`lifetime = birth_value - death_value`. An essential superlevel class has
`death_value = -np.inf` and `lifetime = np.inf`. For sublevel filtration,
`birth_value <= death_value`, `lifetime = death_value - birth_value`, and an essential class has
`death_value = np.inf`. `min_persistence` is compared with these nonnegative public lifetimes.

Freeze whether GUDHI receives vertices or top-dimensional cells, row/column versus Fortran
flattening, bright-ridge superlevel sign transform and inverse coordinate mapping, beta dimensions,
essential intervals, strict
`min_persistence`, threshold equality, connectivity/duality, plateau ties, output dtype, and exact
reconstruction. GUDHI documents different pair-cell APIs and warns that vertex/top-cell pair
representatives are not interchangeable; the contract must pick one
([GUDHI API](https://gudhi.inria.fr/python/3.8.0/cubical_complex_ref.html)).

Representative cell IDs are diagnostic, version-pinned data, not a mathematical invariant on a
plateau. Cross-implementation validation compares interval multisets and topology independently.
The pinned-GUDHI fixture may compare exact cells for drift detection, but a version change must not
be rejected solely because it selects an equivalent plateau representative. If stable public cells
are required, add and separately validate a PhenoTypic canonicalization rule.

## Owned files and tasks

Path 1 owns:

```text
src/phenotypic/analysis/_cubical_persistence.py
tests/unit/analysis/test_cubical_persistence.py
tests/fixtures/analysis/persistence/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/persistence.py
refs/persistence corpus and reconciliation
```

Paths 2 or 3 additionally own:

```text
src/phenotypic/sdk_/reconnect/_persistence.py
src/phenotypic/enhance/_focus_edge_persistence_denoise.py  # only if taxonomy selects FocusEdge
tests/unit/sdk_/reconnect/test_persistence_denoise.py
tests/unit/enhance/test_focus_edge_persistence_denoise.py
tests/fixtures/reconnect/persistence_denoise/
```

1. Run a spike that pins GUDHI version/wheels/API/license plus the exact reconstruction source.
2. Decide path 1, 2, or 3. Stop if no source-faithful reconstruction with compatible license exists.
3. Freeze filtration polarity, cell convention/order, topology dimensions, threshold semantics,
   reconstruction, plateau ties, boundaries, dtype, and outputs.
4. Capture public-coordinate birth/death/lifetime arrays, pair cells, essential cells,
   cancel/keep flags, every reconstruction
   intermediate, and denoised output from the selected oracle.
5. Write an independent tiny-grid filtration oracle and red tests.
6. Implement lazy GUDHI import for pair computation. Implement denoising only from the selected
   source; keep package import cheap and call-time errors actionable.
7. If an enhancer exists, add wrapper forwarding, doctest, serialization, taxonomy, tune, and base/
   topology environment tests.
8. Reviewer audits the scope choice, GUDHI cell semantics, source reconstruction, every fixture
   output, mutations, and dependency isolation.

## Logic-validation script

The standalone script cannot import GUDHI. For persistence reporting, construct the tiny
top-dimensional-cell cubical complex explicitly, assign face filtrations, and reduce boundary
columns over \(\mathbb F_2\) with Python integer bitsets. Cross-check its Betti curves by enumerating
unique filtration levels and using SciPy connected-component labeling; derive 2-D holes only under
an explicitly documented foreground/background digital-topology duality. For the narrowed 0-D
path, also implement an independent union-find merge tree and verify
birth/death/lifetime in original intensity coordinates, essential component, sign reversal between
superlevel and sublevel, plateau-equivalent representatives, exact threshold keep/cancel,
monotonicity, and reconstruction on a 1-D/tiny-2-D landscape.

The design's “one ridge plus N bumps” claim is valid only after polarity and pairing are defined.
Use controlled peaks with analytically known merge levels and assert exact lifetimes. Do not claim
that counting persistence pairs alone proves a denoised image is numerically correct. Fixture all
intervals, representative cells, flags, reconstructed levels, and final output. Treat exact plateau
cell IDs as pinned-version drift evidence, while the independent oracle asserts interval multisets
and topological equivalence.

## Required mutants

- sublevel/superlevel sign reversed;
- vertices versus top-dimensional cells or C/Fortran flattening swapped;
- wrong homology dimension or essential interval handling;
- strict/inclusive persistence threshold changed;
- birth/death or lifetime sign swapped;
- plateau tie nondeterminism;
- wrong foreground/background connectivity dual;
- cancel strong rather than weak pair;
- change reconstruction support or level;
- compute pairs but return original/thresholded/smoothed image;
- eager GUDHI import or swallowed missing dependency;
- wrapper hardcodes threshold/reconstruction.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/persistence.py
uv sync --extra topology --group dev
uv run pytest tests/unit/analysis/test_cubical_persistence.py -q
uv run mypy src/phenotypic/analysis/_cubical_persistence.py
uv run ruff check src/phenotypic/analysis/_cubical_persistence.py tests/unit/analysis/test_cubical_persistence.py
```

Only for path 2 or 3, append:

```bash
uv run pytest tests/unit/sdk_/reconnect/test_persistence_denoise.py tests/unit/enhance/test_focus_edge_persistence_denoise.py -q
uv run mypy src/phenotypic/sdk_/reconnect/_persistence.py src/phenotypic/enhance/_focus_edge_persistence_denoise.py
uv run ruff check src/phenotypic/sdk_/reconnect/_persistence.py src/phenotypic/enhance/_focus_edge_persistence_denoise.py
```
