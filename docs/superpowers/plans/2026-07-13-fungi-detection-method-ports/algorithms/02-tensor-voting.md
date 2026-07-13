# A02: tensor voting

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Keystone, high numerical and complexity risk
**Blocked by:** C3, C4 and S00

## Contract decision

Pin the exact Medioni/Tang edition, Risser paper, executable C++/MATLAB source, license, and
revision before choosing equations. The 2008 Risser source is older than five years and comes
from a different imaging domain; it remains useful as a historical algorithm oracle, not evidence
of fungal benefit ([Risser et al. 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC3298375/)).

Freeze:

```python
def tensor_vote(
    response: np.ndarray,
    theta: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unnormalized (stick_saliency, ball_saliency)."""
```

Private seams return the three accumulated tensor components before decomposition. Validate
same-shaped finite 2-D arrays, nonnegative response, finite `sigma > 0`, axis convention, and no
mutation. Pin tangent/normal interpretation, \(\pi\)-periodicity, active-token threshold, self-votes,
finite support, boundary behavior, accumulator dtype, and loop order.

The helper remains unnormalized. The wrapper and optional detector seam independently clip only
negative roundoff to zero, then map stick saliency by `stick / max(stick)` and ball saliency by
`ball / max(ball)`; a zero maximum returns an all-zero field. Values are clipped to `[0, 1]` only
after division. Record this independent per-field normalization as a PhenoTypic adaptation and
fixture the pre-clip and normalized arrays. Never normalize stick and ball jointly.

The wrapper is blocked until its orientation source is decided. Preferred options are: compute a
documented structure-tensor orientation through a reusable helper, or initially omit the standalone
wrapper and integrate only where the fungi detector already owns orientation
(`src/phenotypic/detect/_filamentous_fungi_detector.py:641-647`). Never store an image-sized
orientation parameter in serialized operation state.

If the reusable-orientation choice is accepted, A02 also owns
`src/phenotypic/sdk_/reconnect/_orientation.py` and
`tests/unit/sdk_/reconnect/test_orientation.py`; it must pin the estimator, scale, tangent/normal
conversion, axial interval, boundaries, fixtures, and mutations. If the wrapper is deferred, all
wrapper/export/taxonomy/doctest steps and commands below are explicitly not part of A02's green
gate. D0 must record which branch applies before dispatch.

## Owned files

```text
src/phenotypic/sdk_/reconnect/_tensor_voting.py
src/phenotypic/enhance/_focus_edge_tensor_voting.py  # conditional on accepted orientation source
tests/unit/sdk_/reconnect/test_tensor_voting.py
tests/unit/enhance/test_focus_edge_tensor_voting.py
tests/fixtures/reconnect/tensor_voting/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/tensor_voting.py
refs/tensor_voting corpus and reconciliation
```

## Tasks

1. Resolve the source, 3-D-to-2-D mapping, token field, equations, constants, and license.
2. Write line reconciliation and drift rows before production code.
3. Implement a slow direct float64 oracle and capture all reference outputs.
4. Write red tests for tensor components, stick, ball, boundaries, normalization separation,
   complexity guard, and invalid inputs.
5. Implement the pure helper and Numba kernel without `fastmath`; stream votes and document memory.
6. Benchmark sparse/dense token behavior on the 800x600 synthetic plate and set a documented active
   token/support contract before exposing a wrapper.
7. If C3 selects a wrapper orientation source, implement wrapper normalization outside the helper,
   output Literal, spy forwarding, doctest, serialization, taxonomy, and tune fields. Otherwise
   record standalone wrapper deferral and run only helper plus detector-contract gates.
8. Resolve detector use at D0: specify an opt-in stick-saliency cost/evidence equation and tests, or
   explicitly defer it. Ball saliency remains diagnostic.
9. Reviewer reruns the external oracle and every mutation before integration.

## Logic-validation script

Independently re-derive for tensor \(A=[[a,b],[b,d]]\):

\[
\Delta=\operatorname{hypot}(a-d,2b),\quad
\lambda_{1,2}=(a+d\pm\Delta)/2,\quad
s=\Delta,\quad b_{sal}=\lambda_2.
\]

Check collinear tokens \(2nn^T\), orthogonal tokens \(I\), one spatial vote, published
distance/curvature ratios, a gap ROI against lateral controls, crossing ball/stick ratios,
\(\pi\)-periodicity, rotation covariance, positive linearity only with a fixed support mask or zero
active-token threshold, a separate threshold-crossing discontinuity case, zero/boundary behavior, and every
fixture output. Use the float-sum \(\gamma_n\) bound plus Weyl's eigenvalue bound; use absolute stick
tolerance near isotropy.

## Required mutants

- remove curvature or distance decay;
- swap row/column or rotate orientation by \(\pi/2\);
- use wrong vote-direction sign or periodicity;
- omit response amplitude or overwrite accumulator;
- change self-vote or support-radius boundary;
- wrap rather than clip image boundaries;
- mis-map eigenvalues to stick/ball;
- accumulate prematurely in float32;
- normalize inside the helper.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/tensor_voting.py
uv run pytest tests/unit/sdk_/reconnect/test_tensor_voting.py -q
uv run mypy src/phenotypic/sdk_/reconnect
uv run ruff check src/phenotypic/sdk_/reconnect
# Run wrapper/taxonomy/tune/doctest commands only when C3 selects the wrapper branch.
```
