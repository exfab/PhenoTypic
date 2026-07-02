import numpy as np

from phenotypic import Image
from phenotypic.schema import REMBI_MODULE


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


def test_insert_metadata_orders_by_rembi_module():
    import pandas as pd

    img = Image(np.zeros((8, 8, 3), dtype=np.uint8), name="sample")
    # Insertion order deliberately scrambles the canonical REMBI order so the
    # test discriminates the REMBI sort from the raw ChainMap iteration order:
    #   Dataset -> STUDY (0), Strain -> BIOSAMPLE (1), Media -> SPECIMEN_PREP (2),
    #   framework ImageName/... -> IMAGE_DATA (4).
    img.metadata["Strain"] = "BY4741"
    img.metadata["Media"] = "YPD"
    img.metadata["Dataset"] = "plateA"

    result = img.metadata.insert_metadata(pd.DataFrame({"Size_Area": [1, 2]}))
    meta_cols = [str(c) for c in result.columns if str(c).startswith("Metadata")]

    def _pos(needle: str) -> int:
        return next(i for i, c in enumerate(meta_cols) if needle in c)

    # Canonical REMBI order: Study < Biosample < SpecimenPreparation < ImageData.
    assert _pos("Dataset") < _pos("Strain") < _pos("Media") < _pos("ImageName")
