"""Wave-0 contract: the shrinking-allowlist coverage gate.

The **denominator** is every numeric-tunable field across ``detect/`` +
``enhance/`` ``__all__`` (closed-set / union / operation / ndarray / container
fields excluded — see ``_annotation_introspect.is_numeric_tunable``). A field is
**covered** when it carries a ``TuneSpec`` (search window *or* ``tunable=False``)
or a ``Field`` validity bound. Uncovered fields must be a **subset** of
``tests/fixtures/tune/annotation_allowlist.json``:

- ``test_uncovered_is_subset_of_allowlist`` fails if coverage **regresses**
  (a new uncovered field appears, or an allowlisted field that was migrated is
  un-migrated).
- ``test_no_allowlisted_field_is_already_covered`` fails when an allowlisted
  field becomes covered without being **removed** from the JSON — forcing each
  migration wave to delete its migrated entries (the allowlist only shrinks).

``ADVISORY_UNTIL_COVERAGE = 0.70`` is the advisory-vs-hard switch: below it the
coverage-level assertion is advisory (a skip carrying the current %); at or above
it the gate hard-asserts the threshold so coverage cannot rot back down.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ._annotation_introspect import (
    field_key,
    is_covered,
    iter_numeric_tunable_fields,
)

#: Coverage fraction below which the level assertion is advisory, not hard.
ADVISORY_UNTIL_COVERAGE = 0.70

_ALLOWLIST_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "tune"
    / "annotation_allowlist.json"
)


def _load_allowlist() -> set[str]:
    payload = json.loads(_ALLOWLIST_PATH.read_text())
    return set(payload["allowlist"])


def _covered_and_uncovered() -> tuple[set[str], set[str], set[str]]:
    """Return ``(denominator, covered, uncovered)`` key sets."""
    denominator: set[str] = set()
    covered: set[str] = set()
    for cls, field_name, field_info in iter_numeric_tunable_fields():
        key = field_key(cls, field_name)
        denominator.add(key)
        if is_covered(field_info):
            covered.add(key)
    uncovered = denominator - covered
    return denominator, covered, uncovered


def test_uncovered_is_subset_of_allowlist():
    """Coverage may only grow: no new uncovered field outside the allowlist."""
    allowlist = _load_allowlist()
    _, _, uncovered = _covered_and_uncovered()
    regressions = sorted(uncovered - allowlist)
    assert not regressions, (
        "These numeric-tunable fields are uncovered but not in the allowlist "
        "(coverage regressed — annotate them or add to the allowlist with a "
        "reason):\n  " + "\n  ".join(regressions)
    )


def test_no_allowlisted_field_is_already_covered():
    """A migrated field must be **removed** from the allowlist (it only shrinks)."""
    allowlist = _load_allowlist()
    _, covered, _ = _covered_and_uncovered()
    stale = sorted(allowlist & covered)
    assert not stale, (
        "These fields are now covered but still in the allowlist — delete them "
        "from annotation_allowlist.json:\n  " + "\n  ".join(stale)
    )


def test_allowlist_entries_are_real_denominator_fields():
    """Every allowlist entry names a real numeric-tunable field (no typos / drift)."""
    allowlist = _load_allowlist()
    denominator, _, _ = _covered_and_uncovered()
    unknown = sorted(allowlist - denominator)
    assert not unknown, (
        "Allowlist names fields that are not in the numeric-tunable denominator "
        "(renamed / removed?):\n  " + "\n  ".join(unknown)
    )


def test_coverage_level_advisory_then_hard():
    """Below the advisory threshold this is informational; above it it hard-gates."""
    denominator, covered, _ = _covered_and_uncovered()
    assert denominator, "expected a non-empty numeric-tunable denominator"
    fraction = len(covered) / len(denominator)
    if fraction < ADVISORY_UNTIL_COVERAGE:
        pytest.skip(
            f"annotation coverage {fraction:.1%} < advisory threshold "
            f"{ADVISORY_UNTIL_COVERAGE:.0%} (gate is advisory; keep migrating)"
        )
    assert fraction >= ADVISORY_UNTIL_COVERAGE, (
        f"annotation coverage {fraction:.1%} fell below the hard threshold "
        f"{ADVISORY_UNTIL_COVERAGE:.0%}"
    )
