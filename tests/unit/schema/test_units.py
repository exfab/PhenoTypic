"""The closed UNITS list, unit resolution, and per-member units on MeasurementInfo."""

from __future__ import annotations

import ast
import inspect
import sys

import pytest

from phenotypic.schema import UNITS, Entry, MeasurementInfo, ResolvedUnit, UnitEntry


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
        "RADIAN_PX", "DEGREE_PER_PX", "DEGREE_PER_PX2", "UNITLESS", "TIME",
        "PER_TIME", "FRAME", "NOT_A_QUANTITY", "OF_INPUT", "OF_INPUT_SQUARED",
        "OF_INPUT_PER_TIME",
    ]
    assert all(m.value == m.name.lower() for m in UNITS)


def _non_stdlib_imports(source: str) -> list[str]:
    """Every import in *source* that is relative or not from the standard library."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.level != 0:
                found.append("." * node.level + (node.module or ""))
            elif node.module.split(".")[0] not in sys.stdlib_module_names:
                found.append(node.module)
        elif isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] not in sys.stdlib_module_names]
    return found


def _units_source() -> str:
    # Located through the class rather than a spelled private path, so a schema
    # file move cannot turn this guard into an ImportError.
    return inspect.getsource(sys.modules[UNITS.__module__])


def test_units_module_imports_only_stdlib():
    assert _non_stdlib_imports(_units_source()) == []


@pytest.mark.parametrize(
    "line",
    ["import numpy", "import numpy.linalg as la", "from ._measurement_info import Entry",
     "from . import _categories", "from phenotypic.schema import SIZE"],
)
def test_the_stdlib_guard_catches_a_non_stdlib_import(line):
    # A scratch copy of the source with one bad import appended; the real file is untouched.
    assert _non_stdlib_imports(_units_source() + "\n" + line + "\n") != []


@pytest.mark.parametrize(
    "build, error",
    [
        pytest.param(lambda: UnitEntry(1, "px", "d"), TypeError, id="key-not-a-string"),
        pytest.param(lambda: UnitEntry("PX", "px", "d"), ValueError, id="key-not-lower-case"),
        pytest.param(lambda: UnitEntry("k", 1, "d"), TypeError, id="symbol-not-a-string"),
        pytest.param(lambda: UnitEntry("k", "s", None), TypeError, id="desc-not-a-string"),
        pytest.param(lambda: UnitEntry("k", "s", " "), ValueError, id="desc-blank"),
        pytest.param(lambda: UnitEntry("k", "s", "d", 1), TypeError, id="power-positional"),
        pytest.param(lambda: UnitEntry("k", "s", "d", length_power=1.0), TypeError, id="length-power-float"),
        pytest.param(lambda: UnitEntry("k", "s", "d", length_power=True), TypeError, id="length-power-bool"),
        pytest.param(lambda: UnitEntry("k", "s", "d", input_power=1.0), TypeError, id="input-power-float"),
        pytest.param(lambda: UnitEntry("k", "s", "d", input_power=True), TypeError, id="input-power-bool"),
        pytest.param(lambda: UnitEntry("k", "s", "d", input_power=-1, length_power=None), ValueError,
                     id="input-power-negative"),
        pytest.param(lambda: UnitEntry("k", "s", "d", per_time=1), TypeError, id="per-time-not-a-bool"),
        pytest.param(lambda: UnitEntry("k", "s", "d", input_power=1), ValueError,
                     id="input-dependent-with-a-length-power"),
        pytest.param(lambda: ResolvedUnit(1, 1), TypeError, id="resolved-symbol-not-a-string"),
        pytest.param(lambda: ResolvedUnit("px", 1.0), TypeError, id="resolved-power-float"),
        pytest.param(lambda: ResolvedUnit("px", True), TypeError, id="resolved-power-bool"),
        pytest.param(lambda: UNITS._new_member_(UNITS, "px"), TypeError, id="member-not-a-unit-entry"),
    ],
)
def test_unit_declarations_refuse_bad_values(build, error):
    with pytest.raises(error):
        build()


def test_per_time_wraps_a_compound_input():
    # Two solidi without parentheses are ambiguous (S4).
    per_time = UNITS.OF_INPUT_PER_TIME
    assert per_time.resolve(ResolvedUnit("rad/px", -1)) == ResolvedUnit("(rad/px)/time", -1)
    assert per_time.resolve(ResolvedUnit("1/time", 0)) == ResolvedUnit("(1/time)/time", 0)
    assert per_time.resolve(ResolvedUnit("time", 0)) == ResolvedUnit("time/time", 0)
    assert per_time.resolve(ResolvedUnit("(rad/px)²", -2)) == ResolvedUnit("(rad/px)²/time", -2)


def test_input_dependent_units_carry_a_placeholder_symbol():
    # Never "" like UNITLESS, so a renderer can't show a fitted rate as unitless.
    assert UNITS.OF_INPUT.symbol == "input unit"
    assert UNITS.OF_INPUT_SQUARED.symbol == "(input unit)²"
    assert UNITS.OF_INPUT_PER_TIME.symbol == "input unit/time"
    for member in UNITS:
        if member.is_input_dependent:
            with pytest.raises(TypeError, match="output_unit"):
                member.length_power


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



def test_a_subclass_defining_unit_is_refused():
    # A unit() classmethod would shadow the member property, so member.unit
    # would be a bound method (K0-01, U1 review S1).
    with pytest.raises(TypeError, match="default_unit"):
        class _TrapClassmethod(MeasurementInfo):
            @classmethod
            def metric_family(cls): return "Trap"
            @classmethod
            def unit(cls): return UNITS.PX
            X = Entry("X", "x")

    with pytest.raises(TypeError, match="default_unit"):
        class _TrapMemberless(MeasurementInfo):
            @classmethod
            def unit(cls): return UNITS.PX

    with pytest.raises(TypeError, match="default_unit"):
        class _TrapProperty(MeasurementInfo):
            @classmethod
            def metric_family(cls): return "Trap"
            @property
            def unit(self): return UNITS.PX
            X = Entry("X", "x")

    with pytest.raises(TypeError, match="default_unit"):
        class _TrapMember(MeasurementInfo):
            @classmethod
            def metric_family(cls): return "Trap"
            unit = Entry("Unit", "A member named unit.")
