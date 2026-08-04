"""Two-sided byte-compat lock for ``_parse_key`` (P3-5a).

The Phase-3 nested-key grammar is *additive*: introducing the
``FlatKey`` / ``PresenceKey`` / ``NestedKey`` parse types must leave every
pre-existing flat / presence key projecting onto the **identical**
``(position, field)`` / ``(position, "__enabled__")`` tuple Phase 1 emitted.

This locks all four flat+presence shapes against a frozen expectation, so the
grammar change is proven byte-identical for existing keys (the guard the golden
manifest lock relies on — the golden's space is flat-only, so an additive
grammar leaves it untouched).
"""
from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.tune._evaluation._builder import (
    FlatKey,
    PresenceKey,
    _parse_key,
)


def _ordered_ops() -> list:
    base = ImagePipeline(ops=[BlurGauss(sigma=2.0), OtsuDetector()])
    return list(base.get_ops().values())


def _project(parsed) -> tuple[int, str]:
    """Collapse a parse result to the Phase-1 ``(position, field)`` projection.

    Phase 1 returned ``(position, field)`` where ``field`` is the scalar field
    name or the literal ``"__enabled__"`` for a presence toggle. This re-derives
    that exact tuple from the new typed parse result.
    """
    if isinstance(parsed, FlatKey):
        return parsed.position, parsed.field
    if isinstance(parsed, PresenceKey):
        return parsed.position, "__enabled__"
    raise AssertionError(f"unexpected parse type for flat/presence: {parsed!r}")


def test_flat_scalar_key_projects_to_phase1_tuple():
    ops = _ordered_ops()
    assert _project(_parse_key("0.sigma", ops)) == (0, "sigma")


def test_flat_scalar_key_second_position_projects_to_phase1_tuple():
    ops = _ordered_ops()
    assert _project(_parse_key("1.ignore_zeros", ops)) == (1, "ignore_zeros")


def test_bare_presence_key_projects_to_phase1_enabled_tuple():
    """``"<pos>.__enabled__"`` (no class segment) — the two-part presence form."""
    ops = _ordered_ops()
    assert _project(_parse_key("0.__enabled__", ops)) == (0, "__enabled__")


def test_classed_presence_key_projects_to_phase1_enabled_tuple():
    """``"<pos>.<Class>.__enabled__"`` — the three-part presence form."""
    ops = _ordered_ops()
    assert _project(_parse_key("0.BlurGauss.__enabled__", ops)) == (
        0,
        "__enabled__",
    )


def test_all_four_flat_presence_shapes_freeze_together():
    ops = _ordered_ops()
    frozen = {
        "0.sigma": (0, "sigma"),
        "1.ignore_zeros": (1, "ignore_zeros"),
        "0.__enabled__": (0, "__enabled__"),
        "0.BlurGauss.__enabled__": (0, "__enabled__"),
    }
    for key, expected in frozen.items():
        assert _project(_parse_key(key, ops)) == expected
