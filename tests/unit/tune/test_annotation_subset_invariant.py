"""Wave-0 contract: the ``TuneSpec ⊆ Field`` invariant across all annotated ops.

For any ``detect/`` + ``enhance/`` field carrying **both** a ``TuneSpec`` search
window (``low``/``high``) **and** an ``annotated_types`` validity bound
(``Ge``/``Gt``/``Le``/``Lt``/``Interval``), the search window must lie inside the
validity envelope, strictness-aware:

- lower edge:  ``low >= ge``  /  ``low > gt``
- upper edge:  ``high <= le`` /  ``high < lt``

A ``TuneSpec`` that escapes its co-located ``Field`` constraint is an author bug
caught here at *test* time (``infer_search_space`` also raises on it, but this
gate sweeps the whole annotated surface, not just a sampled pipeline).

The ``_BadSubsetOp`` fixture proves the check has teeth: its ``TuneSpec`` low is
below the ``Field(ge=...)`` bound, so the strictness-aware assertion must flag it.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from phenotypic.tune import TuneSpec

from ._annotation_introspect import (
    field_key,
    field_metadata,
    iter_numeric_tunable_fields,
    tune_spec_of,
)
import annotated_types as at


def _interval_to_bounds(interval: at.Interval) -> list[object]:
    out: list[object] = []
    if interval.ge is not None:
        out.append(at.Ge(interval.ge))
    if interval.gt is not None:
        out.append(at.Gt(interval.gt))
    if interval.le is not None:
        out.append(at.Le(interval.le))
    if interval.lt is not None:
        out.append(at.Lt(interval.lt))
    return out


def _subset_violations(key: str, spec: TuneSpec, metadata: list[object]) -> list[str]:
    """Return human-readable ``⊆`` violations for one field (empty when clean)."""
    if spec.low is None and spec.high is None:
        return []
    bounds: list[object] = []
    for m in metadata:
        if isinstance(m, at.Interval):
            bounds.extend(_interval_to_bounds(m))
        else:
            bounds.append(m)

    problems: list[str] = []
    for m in bounds:
        if isinstance(m, at.Ge) and spec.low is not None and spec.low < float(m.ge):
            problems.append(f"{key}: low {spec.low} < ge {float(m.ge)}")
        if isinstance(m, at.Gt) and spec.low is not None and spec.low <= float(m.gt):
            problems.append(f"{key}: low {spec.low} <= gt {float(m.gt)} (strict)")
        if isinstance(m, at.Le) and spec.high is not None and spec.high > float(m.le):
            problems.append(f"{key}: high {spec.high} > le {float(m.le)}")
        if isinstance(m, at.Lt) and spec.high is not None and spec.high >= float(m.lt):
            problems.append(f"{key}: high {spec.high} >= lt {float(m.lt)} (strict)")
    return problems


def test_every_tune_spec_is_subset_of_its_field_bound():
    """No annotated field's ``TuneSpec`` escapes its co-located ``Field`` bound."""
    violations: list[str] = []
    for cls, field_name, field_info in iter_numeric_tunable_fields():
        spec = tune_spec_of(field_info)
        if spec is None:
            continue
        key = field_key(cls, field_name)
        violations.extend(
            _subset_violations(key, spec, field_metadata(field_info))
        )
    assert not violations, "TuneSpec ⊆ Field invariant violated:\n" + "\n".join(
        violations
    )


# --------------------------------------------------------------------------- #
# Deliberately-bad fixture — proves the strictness-aware check fails first.
# --------------------------------------------------------------------------- #
class _BadSubsetOp(BaseModel):
    """A fixture whose search window escapes its validity bound (low < ge)."""

    bad: Annotated[float, TuneSpec(0.1, 5.0)] = Field(2.0, ge=1.0)


def test_subset_check_catches_a_deliberate_violation():
    """The strictness-aware comparison flags a ``low < ge`` escape."""
    field_info = _BadSubsetOp.model_fields["bad"]
    spec = tune_spec_of(field_info)
    assert spec is not None
    problems = _subset_violations(
        "_BadSubsetOp.bad", spec, field_metadata(field_info)
    )
    assert problems, "expected the bad fixture to violate the ⊆ invariant"


def test_subset_check_passes_a_legal_window():
    """A window inside the bound (low >= ge, high <= le) is clean."""

    class _GoodOp(BaseModel):
        ok: Annotated[float, TuneSpec(1.5, 4.0)] = Field(2.0, ge=1.0, le=5.0)

    field_info = _GoodOp.model_fields["ok"]
    spec = tune_spec_of(field_info)
    assert spec is not None
    assert not _subset_violations(
        "_GoodOp.ok", spec, field_metadata(field_info)
    )
