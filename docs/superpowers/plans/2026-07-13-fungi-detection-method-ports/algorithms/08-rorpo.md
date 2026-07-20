# A08: RORPO and robust path openings

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Keystone graph-morphology core only; wrapper and polarity adaptations deferred
**Blocked by:** C9, licensing decision, and S00

## Corrected contract

Use the four canonical 2-D path-adjacency DAGs, not an arbitrary `n_orientations` bank. The
reviewed 2-D IPOL v1.0 oracle and the 2022 article revision are pinned in the G0 evidence. The
paper's atomic upper-level-set definition controls equal-level plateaus. The executable is
compatible away from plateau-order ambiguity and has two named plateau drifts in the fixture.

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
) -> RorpoResult: ...
```

The release slice accepts only a nonempty 2-D `numpy.uint8` bright-ridge image and returns a
`numpy.uint8` response. Float conversion, normalization, dark-ridge inversion, and an enhancer
wrapper are deferred. `winning_scale` is int64 and equals `-1` where the merged response is zero.
Caller order for nonempty positive `path_lengths` is load-bearing; unsorted values and duplicates
are accepted, and the first caller-supplied scale owns a strict equal-response tie.

`direction_vector` has shape `image.shape + (2,)`, component order `(row, column)`, unit norm where
valid, and `(0.0, 0.0)` where invalid. `direction_valid` is true only under the frozen five-part
unique-direction predicate: response greater than one, unique truncated split cost, strict
low/high boundary, unique truncated correction-angle assignment, and nonzero corrected vector
sum. Canonicalize the axial sign to positive row, then nonnegative column when row is zero.

A path length `L` counts vertices. Equal-valued upper-level sets are processed atomically;
boundaries clip and never wrap. Robustness uses square radius `floor(R / 2)` followed by the
anti-extensive minimum with the original image, so `R=1` is exactly `R=0`. Sort the four robust
responses pointwise and return `largest - smallest` without normalization.

## Owned files and tasks

```text
src/phenotypic/sdk_/reconnect/_rorpo.py
tests/unit/sdk_/reconnect/test_rorpo.py
tests/fixtures/reconnect/rorpo/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rorpo.py
refs/rorpo corpus and reconciliation
```

1. Resolve license boundary; pin IPOL paper/archive/SWHID, TPAMI paper, Luengo Hendriks source,
   and 3-D repository only as a comparison. Assign distinct oracle and production implementer
   identities and record the information barrier before either role starts.
2. Freeze the exact four DAGs, vertex-count path length, robustness, ranks, direction correction,
   five-part validity predicate, caller-order scale ownership, borders, uint8 dtype, and outputs.
3. The oracle agent writes a tiny exhaustive path/upper-level-set oracle, captures every source
   output, and publishes only the source-free contract and fixtures to the implementer.
4. Add red tests for all intermediates/public fields and invalid inputs.
5. Implement clean-room path-opening kernels, robust construction, pointwise rank/direction, and
   multiscale merge.
6. Keep the enhancer wrapper, float conversion, normalization, dark-ridge inversion,
   serialization, taxonomy, and tune fields deferred.
7. Reviewer audits the GPL boundary, clean-room derivation, source fixture, mutations, and drift.

## Logic-validation script

On tiny arrays, enumerate every admissible length-`L` directed path at every unique upper-level set,
reconstruct grayscale openings, apply the exact robust dilation/min construction, rank four maps
pointwise, and derive the direction vector, validity, and winning scale independently. Check exact
`L-1/L/L+1` lines in all
orientations, curved admissible path, constant-array zero rank gap, multilevel reconstruction,
the two named plateau counterexamples, anti-extensive robust gap, `R=1 == R=0`, borders,
90-degree covariance, uint8 intensity behavior, caller-order multiscale ownership, duplicate
scales, five-part direction validity, and no-path behavior. Min/max morphology is exact.

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
- normalize per scale, accept a non-uint8 input, or change zero-response winning-scale sentinel;
- change strict-first scale ownership or reject unsorted/duplicate positive lengths;
- relax any of the five direction-validity conditions;
- ignore robustness, make `R=1` nonidentity, or substitute an arbitrary rotated bank.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rorpo.py
uv run pytest tests/unit/sdk_/reconnect/test_rorpo.py -q
uv run mypy src/phenotypic/sdk_/reconnect/_rorpo.py
uv run ruff check src/phenotypic/sdk_/reconnect/_rorpo.py tests/unit/sdk_/reconnect/test_rorpo.py
```
