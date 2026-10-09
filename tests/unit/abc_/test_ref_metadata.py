"""RefMetadata: column discovery, context requirement, provenance, tree walk."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

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
    # The exact spelling: the CLI preflight names the op by this path.
    assert found == {"comp/ops[0]": ("Metadata_Strain", "Metadata_BlankImage")}
    assert pipe.reference_columns(images_only=True) == {"comp/ops[0]": ("Metadata_BlankImage",)}
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


class _ReadsBlank(ImageEnhancer, RefMetadata):
    """Test op: looks up its blank and loads it."""

    blank_column: RefImageColumn = "Metadata_BlankImage"

    def _operate(self, image):
        self._ref_image(self._ref_values(image)[self.blank_column])
        return image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_provenance_records_the_table_and_the_blank_it_used(tmp_path):
    """Success criterion 4: the journal says which table and which blank file."""
    root = tmp_path / "imgs"
    root.mkdir()
    blank_path = root / "t00.tif"
    tifffile.imwrite(blank_path, np.full((6, 6), 40, dtype=np.uint8))
    csv = tmp_path / "layout.csv"
    pd.DataFrame({"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]}).to_csv(
        csv, index=False
    )
    with ReferenceContext(csv, image_root=root):
        out = ImagePipeline(ops={"reads": _ReadsBlank()}).apply(_image())
    records = [
        r for r in current_application_operations(out) if r["operation_name"] == "_ReadsBlank"
    ]
    assert records[-1]["parameters"]["_references"] == {
        "table_sha256": _sha256(csv),
        "values": {"Metadata_BlankImage": "t00"},
        "images": {"t00": {"sha256": _sha256(blank_path)}},
    }


def test_ref_image_loads_the_reference_once(monkeypatch):
    """Image and digest come from one load, so they describe the same bytes."""
    calls: list[str] = []
    real = ReferenceContext._load

    def counting(self, name):
        calls.append(name)
        return real(self, name)

    monkeypatch.setattr(ReferenceContext, "_load", counting)
    with ReferenceContext(_layout(), images={"t00": _image("t00")}):
        _ReadsBlank().apply(_image())
    assert calls == ["t00"]


class _FailsAfterLookup(ImageEnhancer, RefMetadata):
    """Test op: resolves its columns, then fails (ordinary or reference error)."""

    strain_column: RefColumn = "Metadata_Strain"
    reference_failure: bool = False

    def _operate(self, image):
        self._ref_values(image)
        if self.reference_failure:
            self._ref_image("not_in_the_context")
        raise KeyError("after the lookup")


@pytest.mark.parametrize("reference_failure", [False, True])
def test_a_failed_apply_leaves_no_resolved_record(reference_failure):
    """A record left by a failed apply could be inherited by a later op that
    reuses the same id() and skips its lookup."""
    from phenotypic.abc_._ref_metadata import _resolved

    store = _resolved()
    before = dict(store)
    with ReferenceContext(_layout()):
        for _ in range(3):
            with pytest.raises((RuntimeError, ValueError)):
                _FailsAfterLookup(reference_failure=reference_failure).apply(_image())
    assert store == before


class _ComputedColumns(ImageEnhancer, RefMetadata):
    """Test op: derives its column from a parameter instead of marked fields."""

    channel: str = "Red"

    def _ref_columns(self) -> tuple[str, ...]:
        return (f"Metadata_Exposure{self.channel}",)

    def _operate(self, image):
        SEEN.append(self._ref_values(image))
        return image


def test_a_subclass_may_compute_its_columns():
    pipe = ImagePipeline(ops={"exposure": _ComputedColumns(channel="Green")})
    assert pipe.reference_columns() == {"exposure": ("Metadata_ExposureGreen",)}
    assert pipe.reference_columns(images_only=True) == {}
    layout = pd.DataFrame({"Metadata_ImageName": ["t04"], "Metadata_ExposureGreen": ["120ms"]})
    SEEN.clear()
    with ReferenceContext(layout):
        pipe.apply(_image())
    assert SEEN == [{"Metadata_ExposureGreen": "120ms"}]
