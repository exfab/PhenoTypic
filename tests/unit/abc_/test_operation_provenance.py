"""Applied-operation provenance follows the complete public ``apply`` boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import phenotypic
from phenotypic import Image, ImagePipeline
from phenotypic.abc_ import ImageCorrector, ImageEnhancer
from phenotypic.enhance import BlurGauss
from phenotypic.settings import validation


class _FailingCorrector(ImageCorrector):
    """Operation used to prove failed algorithms do not claim completion."""

    def _operate(self, image: Image) -> Image:
        raise ValueError("deliberate operation failure")


class _IntegrityBreakingEnhancer(ImageEnhancer):
    """Operation used to prove post-apply integrity checks precede recording."""

    def _operate(self, image: Image) -> Image:
        image.rgb[:] = np.zeros_like(image.rgb[:])
        return image


class _PostApplyFailureCorrector(ImageCorrector):
    """Operation whose public override fails after its parent has returned."""

    def apply(self, image: Image, inplace: bool = False) -> Image:
        super().apply(image=image, inplace=inplace)
        raise RuntimeError("deliberate post-apply failure")

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = 0.5
        return image


class _PostApplySuccessCorrector(ImageCorrector):
    """Operation whose public override post-processes its parent's result."""

    def apply(self, image: Image, inplace: bool = False) -> Image:
        result = super().apply(image=image, inplace=inplace)
        result.name = "post-processed"
        return result

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = 0.5
        return image


class _ReplacementCorrector(ImageCorrector):
    """Return a fresh image to exercise logical image-owned state carry-over."""

    def _operate(self, image: Image) -> Image:
        return Image(np.zeros_like(image.rgb[:]), name=image.name)


def _plate() -> Image:
    pixels = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
    return Image(pixels, name="provenance-plate")


def test_successful_apply_appends_one_json_safe_record_to_returned_image() -> None:
    source = _plate()

    result = BlurGauss(sigma=1.5).apply(source)

    assert source.provenance == ()
    assert len(result.provenance) == 1
    record: Any = result.provenance[0]
    assert record["sequence"] == 1
    assert record["operation_name"] == "BlurGauss"
    assert record["operation_class"] == "phenotypic.enhance._blur_gauss.BlurGauss"
    assert record["phenotypic_version"] == phenotypic.__version__
    assert record["parameters"]["sigma"] == 1.5
    assert record["pipeline_step_path"] is None
    assert record["duration_seconds"] >= 0
    assert record["applied_at_utc"].endswith("Z")


def test_successful_inplace_apply_appends_exactly_once_to_source_image() -> None:
    image = _plate()

    result = BlurGauss(sigma=1.5).apply(image, inplace=True)

    assert result is image
    assert [entry["operation_name"] for entry in image.provenance] == ["BlurGauss"]


def test_provenance_property_does_not_allow_mutating_internal_journal() -> None:
    result = BlurGauss(sigma=1.5).apply(_plate())

    with pytest.raises(TypeError):
        result.provenance[0]["operation_name"] = "tampered"

    assert result.provenance[0]["operation_name"] == "BlurGauss"


def test_image_copy_inherits_existing_provenance_without_sharing_it() -> None:
    first = BlurGauss(sigma=1.0).apply(_plate())

    second = BlurGauss(sigma=2.0).apply(first)

    assert [entry["sequence"] for entry in first.provenance] == [1]
    assert [entry["sequence"] for entry in second.provenance] == [1, 2]
    assert first.provenance[0]["parameters"]["sigma"] == 1.0
    assert second.provenance[1]["parameters"]["sigma"] == 2.0


def test_replacement_operation_carries_prior_journal_and_retained_original() -> None:
    source = BlurGauss(sigma=1.0).apply(_plate())
    source._metadata.provenance_journal.update(
        {
            "status": "in_progress",
            "pipeline": {
                "source_path": "/resolved/pipeline.json",
                "sha256": "a" * 64,
            },
            "retry_base_length": 1,
        }
    )
    source._retain_original()
    retained = source._original.copy()

    result = _ReplacementCorrector().apply(source)

    journal = result._metadata.provenance_journal
    assert journal["status"] == "in_progress"
    assert journal["pipeline"]["source_path"] == "/resolved/pipeline.json"
    assert journal["retry_base_length"] == 1
    assert [entry["operation_name"] for entry in result.provenance] == [
        "BlurGauss",
        "_ReplacementCorrector",
    ]
    np.testing.assert_array_equal(result._original, retained)
    assert [entry["operation_name"] for entry in source.provenance] == ["BlurGauss"]


def test_successful_public_override_appends_one_leaf_record_after_postprocessing() -> None:
    result = _PostApplySuccessCorrector().apply(_plate())

    assert result.name == "post-processed"
    assert [entry["operation_name"] for entry in result.provenance] == [
        "_PostApplySuccessCorrector"
    ]


def test_nested_pipeline_uses_configured_dictionary_keys_as_step_path() -> None:
    pipeline = ImagePipeline(
        ops={
            "first-prep": BlurGauss(sigma=1.0),
            "nested-block": ImagePipeline(
                ops={"inner-prep": BlurGauss(sigma=2.0)}
            ),
        }
    )

    result = pipeline.apply(_plate())

    assert [entry["pipeline_step_path"] for entry in result.provenance] == [
        ["first-prep"],
        ["nested-block", "inner-prep"],
    ]


def test_failed_operate_appends_nothing() -> None:
    image = _plate()

    with pytest.raises(RuntimeError, match="_FailingCorrector failed"):
        _FailingCorrector().apply(image, inplace=True)

    assert image.provenance == ()


def test_integrity_failure_appends_nothing() -> None:
    image = _plate()

    with validation(True), pytest.raises(RuntimeError):
        _IntegrityBreakingEnhancer().apply(image, inplace=True)

    assert image.provenance == ()


def test_failure_after_parent_apply_returns_appends_nothing() -> None:
    image = _plate()

    with pytest.raises(RuntimeError, match="deliberate post-apply failure"):
        _PostApplyFailureCorrector().apply(image, inplace=True)

    assert image.provenance == ()
