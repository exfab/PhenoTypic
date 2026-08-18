import numpy as np
import pandas as pd
import pytest

from phenotypic import Image
from phenotypic.schema import GENETIC, IMAGE, REMBI_MODULE


def _img():
    img = Image(np.zeros((8, 8, 3), dtype=np.uint8), name="sample")
    img.metadata["Strain"] = "BY4741"       # public tag
    return img


def test_by_module_groups_image_data():
    img = _img()
    image_data = img.metadata.by_module(REMBI_MODULE.IMAGE_DATA)
    # framework private/protected keys (e.g. ImageName) land in ImageData
    assert any("ImageName" in k for k in image_data)


def test_by_module_groups_public_tag_to_biosample():
    img = _img()
    biosample = img.metadata.by_module(REMBI_MODULE.BIOSAMPLE)
    assert any("Strain" in k for k in biosample)


def test_by_module_accepts_str_module():
    img = _img()
    image_data = img.metadata.by_module("ImageData")
    assert any("ImageName" in k for k in image_data)


def test_insert_metadata_orders_by_cluster():
    img = Image(np.zeros((8, 8, 3), dtype=np.uint8), name="sample")
    # Insertion order deliberately scrambles the canonical cluster order so the
    # test discriminates the cluster sort from the raw ChainMap iteration order:
    #   Strain -> MetadataGenetic (cluster 2), Media -> MetadataCondition (3),
    #   Dataset -> MetadataExperiment (4), framework ImageName/... ->
    #   MetadataImage (last).
    img.metadata["Strain"] = "BY4741"
    img.metadata["Media"] = "YPD"
    img.metadata["Dataset"] = "plateA"

    result = img.metadata.insert_metadata(pd.DataFrame({"Size_Area": [1, 2]}))
    meta_cols = [str(c) for c in result.columns if str(c).startswith("Metadata")]

    def _pos(needle: str) -> int:
        return next(i for i, c in enumerate(meta_cols) if needle in c)

    # Canonical cluster order: Strain (Genetic) < Media (Condition)
    # < Dataset (Experiment) < ImageName (framework Image, last).
    assert _pos("Strain") < _pos("Media") < _pos("Dataset") < _pos("ImageName")


def test_framework_metadata_aliases_share_permissions_and_lookup_identity():
    """Bare, current, and flat spellings cannot bypass framework protections."""
    img = Image(np.zeros((8, 8, 3), dtype=np.uint8), name="original")

    for alias in ("ImageName", str(IMAGE.IMAGE_NAME), "MetadataImage_ImageName"):
        assert alias in img.metadata
        assert img.metadata[alias] == img.name
        assert img.metadata.get(alias) == img.name
        img.metadata[alias] = "renamed"
        assert img.name == "renamed"
        with pytest.raises(PermissionError):
            del img.metadata[alias]

    for alias in ("UUID", str(IMAGE.UUID), "MetadataImage_UUID"):
        assert alias in img.metadata
        assert img.metadata[alias] == img.uuid
        with pytest.raises(PermissionError):
            img.metadata[alias] = "shadow"
        with pytest.raises(PermissionError):
            del img.metadata[alias]

    assert "Metadata_ImageName" not in img._metadata.public
    assert "Metadata_UUID" not in img._metadata.public


def test_insert_metadata_recognizes_flat_framework_alias_already_in_frame():
    """A flat frame input satisfies the IMAGE field instead of receiving a duplicate."""
    img = Image(np.zeros((8, 8, 3), dtype=np.uint8), name="sample")
    source = pd.DataFrame({"Metadata_ImageName": ["external"]})

    result = img.metadata.insert_metadata(source)

    assert list(result.columns).count("Metadata_ImageName") == 1
    assert result[str(IMAGE.IMAGE_NAME)].tolist() == ["external"]


def test_insert_metadata_preserves_bare_known_metadata_column() -> None:
    """A bare known frame column is an existing schema member, not a new target."""
    img = Image(np.zeros((8, 8, 3), dtype=np.uint8), name="sample")
    img.metadata["Strain"] = "metadata-value"
    source = pd.DataFrame({"Strain": ["frame-value"]})

    result = img.metadata.insert_metadata(source)

    assert result.columns.tolist().count("Strain") == 1
    assert str(GENETIC.STRAIN) not in result.columns
    assert result["Strain"].tolist() == ["frame-value"]
