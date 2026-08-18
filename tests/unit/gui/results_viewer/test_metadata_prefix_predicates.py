"""Metadata-prefix predicates accept flat and exact historical headers."""
from __future__ import annotations

import polars as pl
import pytest

from phenotypic.schema import GENETIC
from phenotypic.gui.results_viewer.colony_view._grid import selectable_axis_columns


@pytest.mark.parametrize(
    "meta_col",
    [str(GENETIC.STRAIN), "MetadataGenetic_Strain"],
)
def test_metadata_columns_bucket_first(meta_col: str) -> None:
    """Metadata columns must sort into bucket 0 (before Grid_ and Color_)."""
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

    normalized_meta_col = str(GENETIC.STRAIN)
    assert normalized_meta_col in out
    assert "Grid_Row" in out
    assert "Color_Hue" in out

    # Bucket 0 (metadata) < Bucket 1 (Grid_) < Bucket 2 (everything else).
    assert out.index(normalized_meta_col) < out.index("Grid_Row"), (
        f"Expected {normalized_meta_col!r} before 'Grid_Row' in {out}"
    )
    assert out.index("Grid_Row") < out.index("Color_Hue"), (
        f"Expected 'Grid_Row' before 'Color_Hue' in {out}"
    )
