"""SubtractBlank: per-polarity arithmetic, detect-mode matching, guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic import GridImage, Image, ImagePipeline, ReferenceContext
from phenotypic._core._image_parts.detection_modes import get_detection_mode
from phenotypic._core._reference_context import ReferenceImageError, ReferenceLookupError
from phenotypic.abc_ import ImageCorrector
from phenotypic.detect import CompositeDetector, OtsuDetector
from phenotypic.enhance import BlurGauss, CompositeEnhance, SetDetectMode, SubtractBlank
from phenotypic.enhance._subtract_blank import StaleDetectMatError


def _gray(values: np.ndarray, name: str) -> Image:
    return Image(arr=values.astype(np.float32), name=name)


def _pair():
    blank = np.full((8, 8), 0.40, dtype=np.float32)
    frame = blank.copy()
    frame[1:3, 1:3] = 0.85   # white mycelium
    frame[5:7, 5:7] = 0.15   # dark pigmented colony
    return _gray(frame, "t04"), _gray(blank, "t00")


def _ctx(target_name="t04", blank: Image | None = None, blank_name="t00"):
    layout = pd.DataFrame({"Metadata_ImageName": [target_name], "Metadata_BlankImage": [blank_name]})
    return ReferenceContext(layout, images={blank_name: blank} if blank is not None else None)


@pytest.mark.parametrize(
    "polarity,bright,dark",
    [("brighter", 0.45, 0.0), ("darker", 0.0, 0.25), ("both", 0.45, 0.25)],
)
def test_polarity_arithmetic(polarity, bright, dark):
    target, blank = _pair()
    with _ctx(blank=blank):
        out = SubtractBlank(polarity=polarity).apply(target)
    dm = out.detect_mat[:]
    assert dm[1, 1] == pytest.approx(bright, abs=1e-6)
    assert dm[5, 5] == pytest.approx(dark, abs=1e-6)
    assert dm[0, 7] == pytest.approx(0.0, abs=1e-6)   # bare agar cancels


def test_rgb_and_gray_are_untouched():
    target, blank = _pair()
    gray_before = target.gray[:].copy()
    with _ctx(blank=blank):
        out = SubtractBlank().apply(target)
    np.testing.assert_array_equal(out.gray[:], gray_before)


def test_blank_is_taken_in_the_targets_detect_mode():
    rng = np.random.default_rng(0)
    blank_rgb = rng.integers(0, 120, size=(8, 8, 3), dtype=np.uint8)
    frame_rgb = blank_rgb.copy()
    frame_rgb[2:5, 2:5] = 230
    target = Image(arr=frame_rgb, name="t04")
    blank = Image(arr=blank_rgb, name="t00")
    target.set_detect_mode("LabL")
    mode = get_detection_mode("LabL")
    expected = np.clip(mode.compute(target) - mode.compute(blank), 0.0, 1.0)
    with _ctx(blank=blank):
        out = SubtractBlank().apply(target)
    np.testing.assert_allclose(out.detect_mat[:], expected, atol=1e-6)


def test_runs_after_set_detect_mode_following_an_enhancer():
    target, blank = _pair()
    pipe = ImagePipeline(ops={"blur": BlurGauss(sigma=1.0), "mode": SetDetectMode(mode="gray"), "sb": SubtractBlank()})
    with _ctx(blank=blank):
        pipe.apply(target)


def test_refuses_after_an_enhancer():
    target, blank = _pair()
    pipe = ImagePipeline(ops={"blur": BlurGauss(sigma=1.0), "sb": SubtractBlank()})
    # ImagePipeline re-raises an op's exception as RuntimeError from it
    # (_image_pipeline_core.py:934-941), so assert on the cause.
    with _ctx(blank=blank), pytest.raises(RuntimeError) as info:
        pipe.apply(target)
    assert isinstance(info.value.__cause__, StaleDetectMatError)
    assert "SetDetectMode" in str(info.value.__cause__)


def test_bare_op_after_a_blur_refuses_directly():
    target, blank = _pair()
    blurred = BlurGauss(sigma=1.0).apply(target)
    with _ctx(blank=blank), pytest.raises(StaleDetectMatError):
        SubtractBlank().apply(blurred)


def test_runs_inside_a_branch_pipeline_and_composites():
    """The ucr_033 placement: inside a branch, after its own SetDetectMode."""
    target, blank = _pair()
    branch = ImagePipeline(ops={"mode": SetDetectMode(mode="gray"), "sb": SubtractBlank()})
    with _ctx(blank=blank):
        ImagePipeline(ops={"branch": branch}).apply(target)
        CompositeEnhance(ops=[SubtractBlank()]).apply(target)
        CompositeDetector(
            ops=[ImagePipeline(ops={"sb": SubtractBlank(), "det": OtsuDetector()})]
        ).apply(target)


def _cause_chain(exc: BaseException) -> list[type[BaseException]]:
    chain: list[type[BaseException]] = []
    current: BaseException | None = exc
    while current is not None:
        chain.append(type(current))
        current = current.__cause__
    return chain


def test_a_composites_direct_child_refusal_arrives_unwrapped():
    """Composites do not catch a child's exception (``apply_child`` is a bare
    ``apply``), and the composite's own ``ImageOperation.apply`` lets a
    ``ReferenceContextError`` through, so a bare composite raises the typed
    error itself, exactly like a bare SubtractBlank."""
    target, blank = _pair()
    blurred = BlurGauss(sigma=1.0).apply(target)
    with _ctx(blank=blank), pytest.raises(StaleDetectMatError):
        CompositeEnhance(ops=[SubtractBlank()]).apply(blurred)


def test_a_refusal_inside_a_composites_branch_pipeline_arrives_wrapped():
    """A branch ``ImagePipeline`` wraps its op's error in ``RuntimeError``; the
    composite then sees a plain ``RuntimeError`` and double-wraps it like any
    other failure. The typed error survives only deep in the cause chain."""
    target, blank = _pair()
    blurred = BlurGauss(sigma=1.0).apply(target)
    detector = CompositeDetector(
        ops=[ImagePipeline(ops={"sb": SubtractBlank(), "det": OtsuDetector()})]
    )
    with _ctx(blank=blank), pytest.raises(RuntimeError) as info:
        detector.apply(blurred)
    assert _cause_chain(info.value) == [
        RuntimeError,         # CompositeDetector's ImageOperation.apply
        Exception,            # its _apply_to_single_image
        RuntimeError,         # the branch ImagePipeline
        StaleDetectMatError,
    ]


def test_a_refusal_inside_a_nested_branch_pipeline_is_wrapped_per_level():
    target, blank = _pair()
    blurred = BlurGauss(sigma=1.0).apply(target)
    pipe = ImagePipeline(ops={"branch": ImagePipeline(ops={"sb": SubtractBlank()})})
    with _ctx(blank=blank), pytest.raises(RuntimeError) as info:
        pipe.apply(blurred)
    assert _cause_chain(info.value) == [RuntimeError, RuntimeError, StaleDetectMatError]


def test_refuses_bit_depth_mismatch():
    target = Image(arr=np.full((8, 8), 40, dtype=np.uint8), name="t04")
    blank = Image(arr=np.full((8, 8), 40 * 257, dtype=np.uint16), name="t00")
    with _ctx(blank=blank), pytest.raises(ReferenceImageError, match="bit"):
        SubtractBlank().apply(target)


class _NoopCorrector(ImageCorrector):
    """A corrector that changes nothing; its presence alone must trip the guard."""

    def _operate(self, image):
        return image


def test_refuses_after_a_corrector():
    target, blank = _pair()
    pipe = ImagePipeline(ops={"fix": _NoopCorrector(), "sb": SubtractBlank()})
    with _ctx(blank=blank), pytest.raises(RuntimeError) as info:
        pipe.apply(target)
    assert isinstance(info.value.__cause__, StaleDetectMatError)
    assert "_NoopCorrector" in str(info.value.__cause__)


def test_refuses_a_corrector_from_an_earlier_application():
    """Stage 2's probe copy opens a fresh provenance application; a Stage-1
    corrector must still be seen (review minor: scan every application)."""
    target, blank = _pair()
    corrected = ImagePipeline(ops={"fix": _NoopCorrector()}).apply(target)
    with _ctx(blank=blank), pytest.raises(RuntimeError) as info:
        ImagePipeline(ops={"sb": SubtractBlank()}).apply(corrected)
    assert isinstance(info.value.__cause__, StaleDetectMatError)


def test_refuses_self_reference():
    target, _ = _pair()
    with _ctx(blank_name="t04", blank=target), pytest.raises(ReferenceLookupError) as info:
        SubtractBlank().apply(target)
    assert info.value.reason == "self"


def test_refuses_self_reference_written_with_its_extension(tmp_path):
    """'t04.tif' for image 't04' resolves to the frame's own file; subtracting
    it would zero the frame silently, so it must be refused like 't04'."""
    import tifffile

    tifffile.imwrite(tmp_path / "t04.tif", np.full((8, 8), 100, dtype=np.uint8))
    target = Image.imread(tmp_path / "t04.tif")
    layout = pd.DataFrame({"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t04.tif"]})
    with ReferenceContext(layout, image_root=tmp_path), pytest.raises(ReferenceLookupError) as info:
        SubtractBlank().apply(target)
    assert info.value.reason == "self"


def test_refuses_shape_mismatch():
    target, _ = _pair()
    small = _gray(np.zeros((4, 4)), "t00")
    with _ctx(blank=small), pytest.raises(ReferenceImageError, match="shape"):
        SubtractBlank().apply(target)


# Review Focus 5
def test_grid_image_target():
    rgb = np.full((48, 48, 3), 30, dtype=np.uint8)
    frame = rgb.copy()
    frame[10:14, 10:14] = 220
    target = GridImage(arr=frame, name="t04")
    blank = Image(arr=rgb, name="t00")
    with _ctx(blank=blank):
        out = SubtractBlank().apply(target)
    assert out.detect_mat[:][11, 11] > 0.5
    assert out.detect_mat[:][40, 40] == pytest.approx(0.0, abs=1e-6)


def test_round_trips_through_json():
    pipe = ImagePipeline(ops={"sb": SubtractBlank(blank_column="Metadata_Frame0", polarity="darker")})
    again = ImagePipeline.from_json(pipe.to_json())
    op = again.get_ops()["sb"]
    assert (op.blank_column, op.polarity) == ("Metadata_Frame0", "darker")
