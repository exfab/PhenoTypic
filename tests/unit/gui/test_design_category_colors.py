# tests/unit/gui/test_design_category_colors.py
from phenotypic.gui._design import ERROR_CATEGORY_COLORS, category_color, OKABE_ITO
from phenotypic.schema import ErrorCategory


def test_every_core_category_has_an_oi_color():
    for token in ErrorCategory.labels():
        assert token in ERROR_CATEGORY_COLORS
        assert ERROR_CATEGORY_COLORS[token] in set(OKABE_ITO) | {"#BBBBBB"}


def test_other_is_grey():
    assert ERROR_CATEGORY_COLORS["other"] == "#BBBBBB"  # OI_GREY


def test_custom_color_cycles_palette_and_is_deterministic():
    assert category_color("halo", custom_index=0) == category_color("halo", custom_index=0)
    assert category_color("halo", custom_index=0) != category_color("halo", custom_index=1)
