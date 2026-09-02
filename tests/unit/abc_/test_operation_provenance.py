"""Applied-operation provenance follows the complete public ``apply`` boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import phenotypic
from phenotypic import Image, ImagePipeline
from phenotypic.abc_ import ImageCorrector, ImageEnhancer
from phenotypic._core._provenance import provenance_success_sink
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


class _DiscardedNestedCorrector(ImageCorrector):
    """Use a distinct nested operation whose returned copy is discarded."""

    def _operate(self, image: Image) -> Image:
        nested = BlurGauss(sigma=0.75).apply(image, inplace=False)
        image.detect_mat[:] = nested.detect_mat[:]
        return image


class _NestedFailureCorrector(ImageCorrector):
    """Invoke a distinct failing operation inside an outer operation."""

    def _operate(self, image: Image) -> Image:
        _FailingCorrector().apply(image, inplace=True)
        return image


class _OuterFailureAfterNestedCorrector(ImageCorrector):
    """Fail at the outer public boundary after a nested operation succeeds."""

    def apply(self, image: Image, inplace: bool = False) -> Image:
        super().apply(image=image, inplace=inplace)
        raise RuntimeError("outer failure after nested success")

    def _operate(self, image: Image) -> Image:
        nested = BlurGauss(sigma=0.75).apply(image, inplace=False)
        image.detect_mat[:] = nested.detect_mat[:]
        return image


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
    source_journal = source._metadata.provenance_journal
    application = source_journal["applications"][-1]
    source_journal["status"] = "in_progress"
    application["status"] = "in_progress"
    application["pipeline"] = {
        "source_path": "pipeline.json",
        "sha256": "a" * 64,
    }
    application["retry_base_length"] = 1
    source._retain_original()
    retained = source._original.copy()

    result = _ReplacementCorrector().apply(source)

    journal = result._metadata.provenance_journal
    assert journal["status"] == "in_progress"
    assert len(journal["applications"]) == 1
    current = journal["applications"][-1]
    assert current["pipeline"]["source_path"] == "pipeline.json"
    assert current["retry_base_length"] == 1
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


def test_distinct_nested_operation_records_inner_then_outer_and_sink_prefixes(
) -> None:
    snapshots: list[list[str]] = []

    def _capture(updated: Image) -> None:
        snapshots.append(
            [entry["operation_name"] for entry in updated.provenance]
        )

    with provenance_success_sink(_capture):
        result = ImagePipeline(
            ops={"composite": _DiscardedNestedCorrector()}
        ).apply(_plate())

    assert [entry["operation_name"] for entry in result.provenance] == [
        "BlurGauss",
        "_DiscardedNestedCorrector",
    ]
    assert [entry["sequence"] for entry in result.provenance] == [1, 2]
    assert [entry["pipeline_step_path"] for entry in result.provenance] == [
        ["composite"],
        ["composite"],
    ]
    assert snapshots == [
        ["BlurGauss"],
        ["BlurGauss", "_DiscardedNestedCorrector"],
    ]


def test_distinct_nested_failure_records_no_operation_or_sink_prefix() -> None:
    image = _plate()
    snapshots: list[list[str]] = []

    with provenance_success_sink(
        lambda updated: snapshots.append(
            [entry["operation_name"] for entry in updated.provenance]
        )
    ), pytest.raises(RuntimeError):
        _NestedFailureCorrector().apply(image, inplace=True)

    assert image.provenance == ()
    assert snapshots == []


def test_outer_failure_rolls_back_nested_memory_and_persisted_sink_prefix() -> None:
    image = _plate()
    snapshots: list[list[str]] = []

    with provenance_success_sink(
        lambda updated: snapshots.append(
            [entry["operation_name"] for entry in updated.provenance]
        )
    ), pytest.raises(RuntimeError, match="outer failure after nested success"):
        _OuterFailureAfterNestedCorrector().apply(image, inplace=True)

    assert image.provenance == ()
    assert snapshots == [["BlurGauss"], []]


def test_deferred_wrapper_factory_is_cached_once_per_concrete_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    from phenotypic._core import _provenance as provenance_module

    owners: list[type] = []
    original_factory = provenance_module.wrap_image_operation_apply

    def _counting_factory(apply_method: Any, owner: type) -> Any:
        owners.append(owner)
        return original_factory(apply_method, owner)

    monkeypatch.setattr(
        provenance_module,
        "wrap_image_operation_apply",
        _counting_factory,
    )

    class _DynamicallyWrappedCorrector(ImageCorrector):
        def _operate(self, image: Image) -> Image:
            return image

    operation = _DynamicallyWrappedCorrector()
    operation.apply(_plate())
    operation.apply(_plate())

    assert owners.count(_DynamicallyWrappedCorrector) == 1
    assert tuple(inspect.signature(operation.apply).parameters) == (
        "image",
        "inplace",
    )
