"""The ``desc`` text plan 03 touched is technical and says what the code computes.

Biological interpretation belongs in ``bio_desc``, which a person writes; the
sentences removed from ``desc`` went to the maintainer as a list (design §8).
"""

from __future__ import annotations

import pytest

from phenotypic.schema import INTENSITY, SHAPE, SIZE


@pytest.mark.parametrize(
    ("member", "phrase"),
    [
        (SHAPE.CIRCULARITY, "genetic mutations"),
        (SHAPE.CIRCULARITY, "mixed microbial populations"),
        (SHAPE.ECCENTRICITY, "healthy bacterial colonies"),
        (SHAPE.ECCENTRICITY, "motility"),
        (SIZE.AREA, "biomass"),
    ],
)
def test_biological_claims_are_not_in_desc(member, phrase) -> None:
    assert phrase not in member.desc


def test_circularity_does_not_promise_one_for_a_round_colony() -> None:
    assert "value = 1" not in SHAPE.CIRCULARITY.desc
    assert "0.9" in SHAPE.CIRCULARITY.desc


def test_compactness_does_not_promise_one_for_a_round_colony() -> None:
    # Compactness is 1 / Circularity, so it carries the same pixel effect.
    assert "1 for perfect circles" not in SHAPE.COMPACTNESS.desc
    assert "1.1" in SHAPE.COMPACTNESS.desc


@pytest.mark.parametrize("member", [SHAPE.CIRCULARITY, SHAPE.ECCENTRICITY, SIZE.AREA])
def test_desc_opens_with_a_plain_sentence_not_a_formula(member) -> None:
    assert not member.desc.startswith(("Calculated as", ":math:"))


@pytest.mark.parametrize("member", [INTENSITY.DENSITY, INTENSITY.CONVEX_DENSITY])
def test_density_desc_says_what_the_code_divides_by(member) -> None:
    # MeasureIntensity divides integrated intensity by an area; nothing is
    # compared with a maximum possible intensity. The arithmetic itself is
    # pinned in tests/unit/measure/test_measure_size.py.
    assert "max possible" not in member.desc
    assert "divided by" in member.desc


def test_density_desc_names_it_a_mean() -> None:
    assert "MeanIntensity" in INTENSITY.DENSITY.desc
