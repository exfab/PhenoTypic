"""The measurement-prefix exclusion list must come from the schema."""

from __future__ import annotations

from phenotypic.gui.results_viewer.colony_view._grid import _MEASUREMENT_PREFIXES


def test_texture_is_excluded_and_texturegray_is_not_invented() -> None:
    """``TEXTURE.category()`` is ``Texture``; ``TextureGray`` is nothing."""
    assert "Texture_" in _MEASUREMENT_PREFIXES
    assert "TextureGray_" not in _MEASUREMENT_PREFIXES


def test_continuous_measurement_families_are_excluded() -> None:
    for prefix in ("Shape_", "Intensity_", "Size_", "Bbox_", "ColorLab_"):
        assert prefix in _MEASUREMENT_PREFIXES, prefix


def test_grouping_families_stay_selectable_as_axes() -> None:
    """Metadata, Grid, Object and Curation are what an axis IS."""
    for prefix in ("Metadata_", "Grid_", "Object_", "Curation_"):
        assert prefix not in _MEASUREMENT_PREFIXES, prefix
