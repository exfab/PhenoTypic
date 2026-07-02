"""Tests for B3: metadata-prefix predicates routed through is_metadata_header.

Decouple discipline: column names are derived from the LIVE enum values
(``str(GENETIC_METADATA.STRAIN)`` today = ``"Metadata_Strain"``, post-B2 flip
= ``"MetadataGenetic_Strain"``).  Tests are GREEN before and after B3, and
stay GREEN after the B2 flip — the old ``startswith("Metadata_")`` predicate
would silently mis-bucket ``"MetadataGenetic_Strain"`` after the flip; the new
``is_metadata_header`` handles both strings correctly.
"""
from __future__ import annotations

import polars as pl

from phenotypic.schema import GENETIC_METADATA
from phenotypic.gui.results_viewer.colony_view._grid import selectable_axis_columns


def test_metadata_columns_bucket_first() -> None:
    """Metadata columns must sort into bucket 0 (before Grid_ and Color_)."""
    meta_col = str(GENETIC_METADATA.STRAIN)
    # Use two unique values so cardinality == 2, satisfying the [2, 50] gate.
    df = pl.DataFrame(
        {
            meta_col: ["a", "b"],
            "Grid_Row": [1, 2],
            "Color_Hue": [0.1, 0.9],
        }
    )
    col_value_sets: dict[str, list[str]] = {
        meta_col: ["a", "b"],
        "Grid_Row": ["1", "2"],
        "Color_Hue": ["0.1", "0.9"],
    }
    out = selectable_axis_columns(df, col_value_sets)

    assert meta_col in out, f"{meta_col!r} not returned by selectable_axis_columns"
    assert "Grid_Row" in out
    assert "Color_Hue" in out

    # Bucket 0 (metadata) < Bucket 1 (Grid_) < Bucket 2 (everything else).
    assert out.index(meta_col) < out.index("Grid_Row"), (
        f"Expected {meta_col!r} before 'Grid_Row' in {out}"
    )
    assert out.index("Grid_Row") < out.index("Color_Hue"), (
        f"Expected 'Grid_Row' before 'Color_Hue' in {out}"
    )
