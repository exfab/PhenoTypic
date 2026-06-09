import pytest

from phenotypic.gui.tune._domain_editor import (
    domain_from_editor,
    domain_summary,
    grid_feasibility,
)
from phenotypic.tune._search_space import (
    Categorical,
    FloatRange,
    IntRange,
    Knob,
    SearchSpace,
)
from phenotypic.tune._search_space._targets import Param


def test_range_mode_continuous_float():
    d = domain_from_editor(
        mode="range", low=1.0, high=6.0, step=None, log=False,
        choices=None, is_int=False,
    )
    assert d == FloatRange(low=1.0, high=6.0)


def test_range_mode_stepped_float_by_magnitude_off():
    d = domain_from_editor(
        mode="range", low=0.0, high=1.0, step=0.25, log=False,
        choices=None, is_int=False,
    )
    assert d == FloatRange(low=0.0, high=1.0, step=0.25)


def test_range_mode_int_log():
    d = domain_from_editor(
        mode="range", low=20, high=400, step=1, log=True,
        choices=None, is_int=True,
    )
    assert d == IntRange(low=20, high=400, step=1, log=True)


def test_range_mode_int_rejects_fractional_step():
    with pytest.raises(ValueError, match="integer"):
        domain_from_editor(
            mode="range", low=1, high=5, step=0.5, log=False,
            choices=None, is_int=True,
        )


def test_range_mode_int_rejects_non_positive_step():
    with pytest.raises(ValueError, match="positive"):
        domain_from_editor(
            mode="range", low=1, high=5, step=0, log=False,
            choices=None, is_int=True,
        )


def test_choices_mode_builds_categorical():
    d = domain_from_editor(
        mode="choices", low=None, high=None, step=None, log=False,
        choices=[0.5, 1, 2], is_int=False,
    )
    assert d == Categorical(choices=(0.5, 1, 2))


def test_choices_mode_requires_values():
    with pytest.raises(ValueError, match="Choices"):
        domain_from_editor(
            mode="choices", low=None, high=None, step=None, log=False,
            choices=[], is_int=False,
        )


def test_summary_strings():
    assert domain_summary(IntRange(low=20, high=400, step=1, log=True)) == (
        "20-400 · step 1 · by-magnitude"
    )
    assert domain_summary(FloatRange(low=1.0, high=6.0)) == "1.0-6.0 · float"
    assert domain_summary(FloatRange(low=0.0, high=1.0, step=0.25)) == (
        "0.0-1.0 · step 0.25"
    )
    assert domain_summary(Categorical(choices=(0.5, 1, 2))) == "{0.5, 1, 2}"


def test_grid_feasibility_blocks_on_continuous_float():
    knob = Knob(
        target=Param(op=0, field="sigma"),
        domain=FloatRange(low=1.0, high=6.0),
    )
    ok, msg = grid_feasibility(SearchSpace(knobs=(knob,)))
    assert ok is False
    assert "continuous float" in msg.lower()


def test_grid_feasibility_ok_when_all_enumerable():
    knob = Knob(
        target=Param(op=0, field="sigma"),
        domain=FloatRange(low=1.0, high=6.0, step=0.5),
    )
    ok, _msg = grid_feasibility(SearchSpace(knobs=(knob,)))
    assert ok is True
