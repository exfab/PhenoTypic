"""Rename-on-load migration for legacy curation state.

Older ``curation_labels.parquet`` files were keyed on the retired ad-hoc
image-stem column. The shim renames that column to the canonical
``Metadata_ImageName`` on read so durable curation state survives the
consolidation. Assertions reference the live enum so they stay correct
across the metadata-category namespace flip.
"""

import polars as pl

from phenotypic.gui.results_viewer._curation_labels import (
    _LEGACY_IMAGE_FILE,
    _migrate_legacy_imagefile,
)
from phenotypic.schema import METADATA


def test_legacy_imagefile_column_renamed() -> None:
    legacy = pl.DataFrame({_LEGACY_IMAGE_FILE: ["p1"], "Object_Label": [3]})
    out = _migrate_legacy_imagefile(legacy)
    assert str(METADATA.IMAGE_NAME) in out.columns
    assert _LEGACY_IMAGE_FILE not in out.columns
    assert out[str(METADATA.IMAGE_NAME)][0] == "p1"


def test_new_frame_unchanged() -> None:
    new = pl.DataFrame({str(METADATA.IMAGE_NAME): ["p1"], "Object_Label": [3]})
    out = _migrate_legacy_imagefile(new)
    assert out.columns == new.columns


def test_frame_without_either_column_unchanged() -> None:
    other = pl.DataFrame({"Object_Label": [3]})
    out = _migrate_legacy_imagefile(other)
    assert out.columns == other.columns
