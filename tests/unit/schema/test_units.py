"""The closed UNITS list, unit resolution, and per-member units on MeasurementInfo."""

from __future__ import annotations

import ast
import inspect
import sys

import pytest

from phenotypic.schema import UNITS, Entry, MeasurementInfo, ResolvedUnit


def test_static_units_resolve_to_themselves():
    assert UNITS.PX2.resolve(None) == ResolvedUnit("px²", 2)
    assert UNITS.UNITLESS.resolve(ResolvedUnit("px", 1)) == ResolvedUnit("", 0)


def test_input_dependent_units_scale_the_input():
    radius = ResolvedUnit("px", 1)
    assert UNITS.OF_INPUT.resolve(radius) == ResolvedUnit("px", 1)
    assert UNITS.OF_INPUT_SQUARED.resolve(radius) == ResolvedUnit("px²", 2)
    assert UNITS.OF_INPUT_PER_TIME.resolve(radius) == ResolvedUnit("px/time", 1)
    assert UNITS.OF_INPUT_PER_TIME.resolve(ResolvedUnit("px²", 2)) == ResolvedUnit("px²/time", 2)


def test_unitless_input_gives_one_over_time():
    # K0-15: a growth fit on circularity or a count reads "1/time", not "/time".
    unitless = ResolvedUnit("", 0)
    assert UNITS.OF_INPUT_PER_TIME.resolve(unitless) == ResolvedUnit("1/time", 0)
    assert UNITS.OF_INPUT.resolve(unitless) == ResolvedUnit("", 0)
    assert UNITS.OF_INPUT_SQUARED.resolve(unitless) == ResolvedUnit("", 0)


def test_input_dependent_unit_needs_a_known_input():
    with pytest.raises(ValueError, match="unknown"):
        UNITS.OF_INPUT_PER_TIME.resolve(None)
    with pytest.raises(ValueError, match="unknown"):
        UNITS.OF_INPUT.resolve(ResolvedUnit("", None))


def test_unresolved_length_power_says_to_resolve():
    with pytest.raises(TypeError, match="output_unit"):
        UNITS.OF_INPUT_PER_TIME.length_power ** 1


def test_not_a_quantity_has_no_length_power():
    assert UNITS.NOT_A_QUANTITY.length_power is None
    assert not UNITS.NOT_A_QUANTITY.is_input_dependent


def test_members_are_the_closed_list_in_order():
    # The value is a unique lower-case key: equal values would alias members (K0-14).
    assert [m.name for m in UNITS] == [
        "PX", "PX2", "INTENSITY", "INTENSITY_PX2", "RADIAN", "DEGREE",
        "CIELAB", "CIELAB2", "TURN", "RADIAN_PER_PX", "RADIAN_PER_PX2",
        "RADIAN_PX", "UNITLESS", "TIME", "PER_TIME", "FRAME", "NOT_A_QUANTITY",
        "OF_INPUT", "OF_INPUT_SQUARED", "OF_INPUT_PER_TIME",
    ]
    assert all(m.value == m.name.lower() for m in UNITS)


def test_units_module_imports_only_stdlib():
    # Located through the class rather than a spelled private path, so a schema
    # file move cannot turn this guard into an ImportError.
    tree = ast.parse(inspect.getsource(sys.modules[UNITS.__module__]))
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith("phenotypic") for m in imported)


class _Demo(MeasurementInfo):
    @classmethod
    def metric_family(cls): return "Demo"
    @classmethod
    def default_unit(cls): return UNITS.PX
    WIDTH = Entry("Width", "A width.")
    AREA = Entry("Area", "An area.", unit=UNITS.PX2)


def test_class_default_and_override():
    assert _Demo.WIDTH.unit is UNITS.PX
    assert _Demo.AREA.unit is UNITS.PX2


def test_entry_refuses_a_string_unit():
    with pytest.raises(TypeError, match="UNITS"):
        Entry("X", "x", unit="px")
