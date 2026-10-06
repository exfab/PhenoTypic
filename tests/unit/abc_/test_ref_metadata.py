"""RefMetadata: column discovery, context requirement, provenance, tree walk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic import Image, ImagePipeline, ReferenceContext
from phenotypic._core._provenance import current_application_operations
from phenotypic._core._reference_context import RefMetadataUnavailableError
from phenotypic.abc_ import ImageEnhancer, RefMetadata
from phenotypic.enhance import CompositeEnhance
from phenotypic.sdk_ import RefColumn, RefImageColumn

SEEN: list[dict] = []


class _ReadsStrain(ImageEnhancer, RefMetadata):
    """Test op: records what it read."""

    strain_column: RefColumn = "Metadata_Strain"
    blank_column: RefImageColumn = "Metadata_BlankImage"

    def _operate(self, image):
        SEEN.append(self._ref_values(image))
        return image


def _image(name: str = "t04") -> Image:
    return Image(arr=np.full((6, 6), 0.5, dtype=np.float32), name=name)


def _layout() -> pd.DataFrame:
    return pd.DataFrame({
        "Metadata_ImageName": ["t04"],
        "Metadata_Strain": ["WT"],
        "Metadata_BlankImage": ["t00"],
    })


def test_columns_are_discovered_from_marked_fields():
    op = _ReadsStrain(strain_column="Metadata_Strain")
    assert op._ref_columns() == ("Metadata_Strain", "Metadata_BlankImage")
    assert op._ref_image_columns() == ("Metadata_BlankImage",)


def test_without_context_the_error_names_both_fixes():
    with pytest.raises(RefMetadataUnavailableError) as info:
        _ReadsStrain().apply(_image())
    message = str(info.value)
    assert "_ReadsStrain" in message
    assert "with phenotypic.ReferenceContext(" in message
    assert "--metadata" in message


def test_with_context_the_op_reads_its_own_columns():
    SEEN.clear()
    with ReferenceContext(_layout()):
        _ReadsStrain().apply(_image())
    assert SEEN == [{"Metadata_Strain": "WT", "Metadata_BlankImage": "t00"}]


def test_provenance_records_the_resolved_values():
    with ReferenceContext(_layout()):
        out = ImagePipeline(ops={"reads": _ReadsStrain()}).apply(_image())
    records = [
        r for r in current_application_operations(out)
        if r["operation_name"] == "_ReadsStrain"
    ]
    assert records[-1]["parameters"]["_references"]["values"] == {
        "Metadata_Strain": "WT",
        "Metadata_BlankImage": "t00",
    }
    assert records[-1]["parameters"]["blank_column"] == "Metadata_BlankImage"


def test_only_image_operations_may_mix_it_in():
    with pytest.raises(TypeError, match="ImageOperation"):
        type("Bad", (RefMetadata,), {})


def test_pipeline_reports_reference_columns_tree_wide():
    pipe = ImagePipeline(ops={"comp": CompositeEnhance(ops=[_ReadsStrain()])})
    found = pipe.reference_columns()
    assert list(found.values()) == [("Metadata_Strain", "Metadata_BlankImage")]
    assert next(iter(found)).startswith("comp")
    assert list(pipe.reference_columns(images_only=True).values()) == [("Metadata_BlankImage",)]
    assert ImagePipeline(ops={}).reference_columns() == {}


class _Boom(ImageEnhancer):
    """Test op: fails with an ordinary (non-reference) error."""

    def _operate(self, image):
        raise KeyError("not a reference failure")


def test_reference_errors_pass_through_apply_and_pipeline_wraps_once():
    with pytest.raises(RefMetadataUnavailableError):
        _ReadsStrain().apply(_image())
    with pytest.raises(RuntimeError) as info:
        ImagePipeline(ops={"reads": _ReadsStrain()}).apply(_image())
    assert isinstance(info.value.__cause__, RefMetadataUnavailableError)


def test_other_errors_are_still_double_wrapped_by_apply():
    with pytest.raises(RuntimeError) as info:
        _Boom().apply(_image())
    assert type(info.value.__cause__) is Exception
    assert isinstance(info.value.__cause__.__cause__, KeyError)
