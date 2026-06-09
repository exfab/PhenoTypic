from __future__ import annotations

import pytest

from phenotypic.tune import (
    Categorical,
    FloatRange,
    IntRange,
    Knob,
    SearchSpace,
)
from phenotypic.tune._strategies._enumerate import enumerate_grid, grid_values


def test_grid_values_per_domain():
    assert grid_values(Categorical(choices=(True, False))) == [True, False]
    assert grid_values(IntRange(low=2, high=8, step=2)) == [2, 4, 6, 8]
    assert grid_values(FloatRange(low=0.0, high=1.0, step=0.5)) == [
        0.0, 0.5, 1.0,
    ]
    from phenotypic.tune import Fixed
    assert grid_values(Fixed(value="x")) == ["x"]


def test_grid_values_rejects_floatrange():
    with pytest.raises(ValueError, match="continuous|FloatRange"):
        grid_values(FloatRange(low=0.0, high=1.0))


def test_enumerate_conditional_absent_collapses():
    # Mirrors the golden config: Presence(GaussianBlur, sigma=(1,2)) + Sweep(Otsu, ignore_zeros=(T,F))
    space = SearchSpace(knobs=(
        Knob(key="0.GaussianBlur.__enabled__",
             domain=Categorical(choices=(True, False)), source="presence_optin"),
        Knob(key="0.sigma",
             domain=Categorical(choices=(1.0, 2.0)),
             conditional_on=(("0.GaussianBlur.__enabled__", True),)),
        Knob(key="1.ignore_zeros",
             domain=Categorical(choices=(True, False))),
    ))
    combos = enumerate_grid(space)
    # absent: enabled=False → sigma omitted → 2 (× ignore_zeros); present: 2 sigmas × 2 = 4 → total 6
    assert len(combos) == 6
    absent = [c for c in combos if c["0.GaussianBlur.__enabled__"] is False]
    assert len(absent) == 2
    assert all("0.sigma" not in c for c in absent)
    present = [c for c in combos if c["0.GaussianBlur.__enabled__"] is True]
    assert len(present) == 4
    assert all("0.sigma" in c for c in present)


def test_enumerate_unconditional_only():
    space = SearchSpace(knobs=(
        Knob(key="0.a", domain=Categorical(choices=(1, 2))),
        Knob(key="1.b", domain=IntRange(low=1, high=2)),
    ))
    assert len(enumerate_grid(space)) == 4
