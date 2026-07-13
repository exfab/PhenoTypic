# A08: RORPO and robust path openings

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Keystone graph-morphology core and Leaf wrapper
**Blocked by:** C9, licensing decision, and S00

## Corrected contract

Use the four canonical 2-D path-adjacency DAGs, not an arbitrary `n_orientations` bank. The
published 2-D residual ranks four path-opening responses and subtracts the fourth-ranked from the
largest; the rank threshold is not free ([Merveille et al. 2018](https://perso.esiee.fr/~perretb/I5FM/TAI/_downloads/5bd7490acbb12babbdf95418c2d5cba7/RORPO.pdf)). Pin the 2-D IPOL release, not the primarily 3-D repository.

The IPOL software is GPL-3.0-or-later while PhenoTypic is Apache-2.0. Do not copy or transcribe GPL
code into production without a licensing decision. Use separate roles: an oracle/reconciliation
agent may inspect and execute the GPL source, then publish fixtures, provenance, and a source-free
behavioral contract; the production implementer must not inspect that source and works only from
the paper, non-GPL algorithm sources, and the reviewed source-free contract. If that separation
cannot be maintained, do not describe the work as clean-room and stop for a licensing decision
([IPOL article/software](https://www.ipol.im/pub/art/2017/207/)).

```python
@dataclass(frozen=True)
class RorpoResult:
    response: np.ndarray
    direction_vector: np.ndarray
    direction_valid: np.ndarray
    winning_scale: np.ndarray

def rorpo(
    image: np.ndarray,
    path_lengths: tuple[int, ...],
    robustness: int = 0,
    *,
    black_ridges: bool = False,
) -> RorpoResult: ...
```

`direction_vector` has shape `image.shape + (2,)`, component order `(row, column)`, unit norm where
valid, and `(0.0, 0.0)` where invalid. `direction_valid` has `image.shape` and is false for zero
response, isotropic ties, or any tie for which the selected reference does not define a unique
direction. Freeze the sign convention for this axial vector and the pointwise tie-break across
the four directions and scales. Do not collapse the source-visible vector to a scalar angle.

Freeze path length as vertices/edges, DAG tables, upper-level grayscale reconstruction,
robustness footprint, anti-extensive gap behavior, border rules, rank/ties, axial direction,
winning scale, polarity, dtype, and absence of hidden normalization. Robust mode may preserve
flanks across a gap but must not claim to paint missing pixels if the source remains anti-extensive.
The helper preserves source intensity units. The wrapper must apply a declared
`NormalizedOutputMixin` policy at the `detect_mat` boundary and test that policy separately from
the port.

## Owned files and tasks

```text
src/phenotypic/sdk_/reconnect/_rorpo.py
src/phenotypic/enhance/_focus_edge_rorpo.py
tests/unit/sdk_/reconnect/test_rorpo.py
tests/unit/enhance/test_focus_edge_rorpo.py
tests/fixtures/reconnect/rorpo/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rorpo.py
refs/rorpo corpus and reconciliation
```

1. Resolve license boundary; pin IPOL paper/archive/SWHID, TPAMI paper, Luengo Hendriks source,
   and 3-D repository only as a comparison. Assign distinct oracle and production implementer
   identities and record the information barrier before either role starts.
2. Freeze the exact four DAGs, path length, robustness, ranks, direction correction, ties, borders,
   dtype, polarity, and public outputs.
3. The oracle agent writes a tiny exhaustive path/upper-level-set oracle, captures every source
   output, and publishes only the source-free contract and fixtures to the implementer.
4. Add red tests for all intermediates/public fields and invalid inputs.
5. Implement clean-room path-opening kernels, robust construction, pointwise rank/direction, and
   multiscale merge.
6. Implement response wrapper, forwarding spy, doctest, serialization, taxonomy, and tune fields.
7. Reviewer audits the GPL boundary, clean-room derivation, source fixture, mutations, and drift.

## Logic-validation script

On tiny arrays, enumerate every admissible length-`L` directed path at every unique upper-level set,
reconstruct grayscale openings, apply the exact robust dilation/min construction, rank four maps
pointwise, and derive the direction vector, validity, and winning scale independently. Check exact
`L-1/L/L+1` lines in all
orientations, curved admissible path, constant-array zero rank gap, a digitized blob compared to a
source-derived nonzero lattice bound, multilevel reconstruction,
anti-extensive robust gap, borders, 90-degree covariance, affine intensity law, multiscale
monotonicity, tie behavior, polarity, and no-path behavior. Min/max morphology is exact unless an
explicit dtype conversion requires an ulp bound.

## Required mutants

- interpret length as edges or change boundary comparator;
- omit/swap one DAG adjacency or row/column axis;
- substitute straight-line opening;
- process levels/order incorrectly or wrap borders;
- omit robust dilation/final minimum or fill gap pixels;
- rank globally or use max-minus-second/minus wrong rank;
- detach direction from sorted orientation;
- omit vector normalization/validity, swap vector components, omit axial correction, or change
  equal-scale/equal-direction tie;
- normalize per scale, truncate dtype, or mishandle polarity;
- ignore robustness or substitute an arbitrary rotated bank.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rorpo.py
uv run pytest tests/unit/sdk_/reconnect/test_rorpo.py tests/unit/enhance/test_focus_edge_rorpo.py -q
uv run pytest tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune/test_enhance_annotations.py -q
uv run mypy src/phenotypic
uv run ruff check
```
