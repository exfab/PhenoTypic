"""Retirement of the ad-hoc per-image-file column.

The chunk-writer identity helper now emits the canonical
``Metadata_ImageName`` plus ``Metadata_FileSuffix`` instead of the retired
ad-hoc stem column. Assertions reference the live enum so they stay correct
across the metadata-category namespace flip.
"""

import polars as pl

from phenotypic._cli._cli_chunk_writer import _attach_image_identity
from phenotypic.schema import IMAGE


def test_chunk_writer_emits_imagename_and_suffix() -> None:
    df = pl.DataFrame({"x": [1]})
    out = _attach_image_identity(df, stem="plate1", suffix=".tif")
    # Exactly the input column plus the two canonical identity columns —
    # no retired ad-hoc stem column survives.
    assert set(out.columns) == {"x", str(IMAGE.IMAGE_NAME), str(IMAGE.SUFFIX)}
    assert out[str(IMAGE.IMAGE_NAME)][0] == "plate1"
    assert out[str(IMAGE.SUFFIX)][0] == ".tif"


def test_attach_identity_preserves_existing_suffix() -> None:
    # Non-clobbering: a Metadata_FileSuffix already emitted upstream by
    # ``insert_metadata`` is preserved rather than overwritten.
    df = pl.DataFrame({str(IMAGE.SUFFIX): [".png"]})
    out = _attach_image_identity(df, stem="p2", suffix=".tif")
    assert out[str(IMAGE.SUFFIX)][0] == ".png"
    assert out[str(IMAGE.IMAGE_NAME)][0] == "p2"
