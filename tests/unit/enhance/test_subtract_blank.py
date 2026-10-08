"""SubtractBlank: per-polarity arithmetic, detect-mode matching, guards."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from phenotypic import GridImage, Image, ImagePipeline, ReferenceContext
from phenotypic._core._image_parts.detection_modes import get_detection_mode
from phenotypic._core._reference_context import ReferenceImageError, ReferenceLookupError
from phenotypic.abc_ import ImageCorrector
from phenotypic.detect import CompositeDetector, OtsuDetector
from phenotypic.enhance import BlurGauss, CompositeEnhance, SetDetectMode, SubtractBlank
from phenotypic.enhance._subtract_blank import _RECORDED_CLASSES, StaleDetectMatError


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


def _rgb_pair():
    blank_rgb = np.full((8, 8, 3), 60, dtype=np.uint8)
    frame_rgb = blank_rgb.copy()
    frame_rgb[2:5, 2:5] = 220
    return Image(arr=frame_rgb, name="t04"), Image(arr=blank_rgb, name="t00")


def test_rgb_and_gray_of_an_rgb_target_are_untouched():
    target, blank = _rgb_pair()
    rgb_before, gray_before = target.rgb[:].copy(), target.gray[:].copy()
    with _ctx(blank=blank):
        out = SubtractBlank().apply(target)
    np.testing.assert_array_equal(out.rgb[:], rgb_before)
    np.testing.assert_array_equal(out.gray[:], gray_before)
    assert float(out.detect_mat[:][3, 3]) > 0.5   # the subtraction did happen


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
        out = pipe.apply(target)
    dm = out.detect_mat[:]
    assert dm[1, 1] == pytest.approx(0.45, abs=1e-6)   # mycelium minus blank, unblurred
    assert dm[0, 7] == pytest.approx(0.0, abs=1e-6)


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
        via_branch = ImagePipeline(ops={"branch": branch}).apply(target)
        via_composite = CompositeEnhance(ops=[SubtractBlank()]).apply(target)
        detected = CompositeDetector(
            ops=[ImagePipeline(ops={"sb": SubtractBlank(), "det": OtsuDetector()})]
        ).apply(target)
    dm = via_branch.detect_mat[:]
    assert dm[1, 1] == pytest.approx(0.45, abs=1e-6)
    assert dm[5, 5] == pytest.approx(0.0, abs=1e-6)   # darker colony dropped by "brighter"
    assert dm[0, 7] == pytest.approx(0.0, abs=1e-6)
    dm = via_composite.detect_mat[:]
    assert dm[1, 1] > dm[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert dm[5, 5] == pytest.approx(0.0, abs=1e-6)
    # Only the mycelium survives the subtraction, so only it is detected.
    mask = detected.objmask[:]
    assert mask[1, 1] and not mask[5, 5] and not mask[0, 7]


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


@pytest.mark.parametrize("dtype,blank_value,frame_value", [(np.uint8, 100, 110), (np.uint16, 30000, 30100)])
def test_a_single_channel_integer_pair_subtracts_in_unit_range(dtype, blank_value, frame_value):
    """A 2-D integer scan is normalised by its dtype max at construction, so the
    pair subtracts in [0, 1] like any other: no saturation, no wrap-around."""
    frame = np.full((8, 8), blank_value, dtype=dtype)
    frame[1:3, 1:3] = frame_value          # brighter colony
    frame[5:7, 5:7] = 2 * blank_value - frame_value   # darker colony, same depth
    target = Image(arr=frame, name="t04")
    blank = Image(arr=np.full((8, 8), blank_value, dtype=dtype), name="t00")
    step = (frame_value - blank_value) / np.iinfo(dtype).max
    with _ctx(blank=blank):
        dm = SubtractBlank(polarity="both").apply(target).detect_mat[:]
    assert dm.dtype == np.float32
    assert dm[1, 1] == pytest.approx(step, abs=1e-6)
    assert dm[5, 5] == pytest.approx(step, abs=1e-6)
    assert dm[0, 7] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("target_is_rgb", [True, False])
def test_refuses_a_single_channel_and_rgb_pair(target_is_rgb):
    """A single-channel scan's gray and an RGB frame's luminance are different
    quantities; subtracting one from the other is meaningless."""
    single = np.full((8, 8), 0.4, dtype=np.float32)            # bit_depth 16
    rgb = np.full((8, 8, 3), 100 * 257, dtype=np.uint16)       # bit_depth 16
    target = Image(arr=rgb if target_is_rgb else single, name="t04")
    blank = Image(arr=single if target_is_rgb else rgb, name="t00")
    with _ctx(blank=blank), pytest.raises(ReferenceImageError, match="RGB"):
        SubtractBlank().apply(target)


@pytest.mark.parametrize("gamma,illuminant", [("sRGB", "D50"), (None, "D65"), (None, "D50")])
def test_blank_is_projected_through_the_targets_colour_configuration(gamma, illuminant):
    """Identical pixels cancel exactly whatever the target's colour configuration;
    the blank must not be projected under its own D65/sRGB defaults.
    inplace=True keeps the target's configuration (Image.copy drops it -- a
    separate core issue)."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(30, 220, size=(16, 16, 3), dtype=np.uint8)
    target = Image(arr=rgb.copy(), name="t04", gamma=gamma, illuminant=illuminant)
    blank = Image(arr=rgb.copy(), name="t00")
    target.set_detect_mode("LabL")
    with _ctx(blank=blank):
        out = SubtractBlank(polarity="both").apply(target, inplace=True)
    assert float(out.detect_mat[:].max()) == 0.0


@pytest.mark.parametrize("polarity", ["brighter", "darker", "both"])
def test_a_blank_projecting_outside_the_unit_range_is_refused(monkeypatch, polarity):
    """Unclipped 'both' once returned values far outside [0, 1] (P14); a
    projection outside [0, 1] is refused rather than clipped into a plausible map."""
    target, blank = _pair()
    mode = get_detection_mode("gray")
    real = type(mode).compute

    def projected(self, image):
        out = real(self, image)
        return out - 2.0 if image.name == "t00" else out

    monkeypatch.setattr(type(mode), "compute", projected)
    with _ctx(blank=blank), pytest.raises(ReferenceImageError, match=r"outside \[0, 1\]"):
        SubtractBlank(polarity=polarity).apply(target)


@pytest.mark.parametrize("scale", [200.0, -1.0])
def test_a_float_single_channel_frame_outside_the_unit_range_is_refused(scale):
    """A 2-D float array carries no scale, so one outside [0, 1] is refused when
    the Image is built -- before it can reach SubtractBlank at all."""
    target, _ = _pair()
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        _gray(target.gray[:] * scale, "t04")


@pytest.mark.parametrize("polarity", ["brighter", "darker", "both"])
def test_output_is_within_the_unit_range(polarity):
    target, blank = _pair()
    with _ctx(blank=blank):
        dm = SubtractBlank(polarity=polarity).apply(target).detect_mat[:]
    assert float(dm.min()) >= 0.0 and float(dm.max()) <= 1.0


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


class _GrandchildCorrector(_NoopCorrector):
    """A corrector two levels below ImageCorrector (like a GridCorrector subclass)."""


def test_refuses_after_a_grandchild_corrector():
    target, blank = _pair()
    pipe = ImagePipeline(ops={"fix": _GrandchildCorrector(), "sb": SubtractBlank()})
    with _ctx(blank=blank), pytest.raises(RuntimeError) as info:
        pipe.apply(target)
    assert isinstance(info.value.__cause__, StaleDetectMatError)
    assert "_GrandchildCorrector" in str(info.value.__cause__)


def _with_recorded_class(image: Image, operation_class: str) -> Image:
    """Append a journal record naming *operation_class*, as a stored history would."""
    from phenotypic._core._provenance import _operations

    journal = image._metadata.provenance_journal
    operations = journal["applications"][-1]["operations"]
    record = dict(operations[-1])
    record.update(
        sequence=len(_operations(journal)) + 1,
        operation_name=operation_class.rsplit(".", 1)[-1],
        operation_class=operation_class,
    )
    operations.append(record)
    return image


def _recorded_target(operation_class: str) -> Image:
    target, _ = _pair()
    target = ImagePipeline(ops={"mode": SetDetectMode(mode="gray")}).apply(target)
    return _with_recorded_class(target, operation_class)


_FOREIGN_MODULE = "refmeta_foreign_corrector_mod"


@pytest.fixture
def foreign_corrector_module(tmp_path, monkeypatch):
    """An importable, never-imported module defining a corrector, outside phenotypic.

    Its top-level code records that it ran, so a test can prove it was not imported.
    """
    import sys

    (tmp_path / f"{_FOREIGN_MODULE}.py").write_text(
        "import os\n"
        "from phenotypic.abc_ import ImageCorrector\n\n"
        "os.environ['REFMETA_FOREIGN_MODULE_RAN'] = '1'\n\n\n"
        "class RefmetaForeignCorrector(ImageCorrector):\n"
        '    """A corrector defined where nothing imports it."""\n\n'
        "    def _operate(self, image):\n"
        "        return image\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("REFMETA_FOREIGN_MODULE_RAN", raising=False)
    assert _FOREIGN_MODULE not in sys.modules
    recorded = f"{_FOREIGN_MODULE}.RefmetaForeignCorrector"
    _RECORDED_CLASSES.pop(recorded, None)
    yield recorded
    _RECORDED_CLASSES.pop(recorded, None)
    sys.modules.pop(_FOREIGN_MODULE, None)


def test_a_journal_cannot_make_subtract_blank_import_a_foreign_module(
    foreign_corrector_module,
):
    """A journal may come from a downloaded store: importing the module it names
    would run that module's code. Unloaded and outside phenotypic, the class is
    unresolvable, so the history is refused -- and nothing is imported."""
    import sys

    target = _recorded_target(foreign_corrector_module)
    _, blank = _pair()
    with _ctx(blank=blank), pytest.raises(StaleDetectMatError, match="cannot resolve") as info:
        SubtractBlank().apply(target)
    assert "RefmetaForeignCorrector" in str(info.value)
    assert _FOREIGN_MODULE not in sys.modules
    assert "REFMETA_FOREIGN_MODULE_RAN" not in os.environ


def test_a_foreign_corrector_already_imported_is_recognised(foreign_corrector_module):
    import importlib

    importlib.import_module(_FOREIGN_MODULE)
    target = _recorded_target(foreign_corrector_module)
    _, blank = _pair()
    with _ctx(blank=blank), pytest.raises(StaleDetectMatError, match="an ImageCorrector") as info:
        SubtractBlank().apply(target)
    assert "RefmetaForeignCorrector" in str(info.value)


def test_a_foreign_corrector_listed_for_preload_is_recognised(foreign_corrector_module, monkeypatch):
    """PHENOTYPIC_PRELOAD_MODULES is the trusted custom-op list; its modules load first."""
    monkeypatch.setenv("PHENOTYPIC_PRELOAD_MODULES", _FOREIGN_MODULE)
    target = _recorded_target(foreign_corrector_module)
    _, blank = _pair()
    with _ctx(blank=blank), pytest.raises(StaleDetectMatError, match="an ImageCorrector") as info:
        SubtractBlank().apply(target)
    assert "RefmetaForeignCorrector" in str(info.value)


_FRESH_PROCESS_SCRIPT = """
import sys
import numpy as np
import pandas as pd
from phenotypic import Image, ImagePipeline, ReferenceContext
from phenotypic._core._provenance import _operations
from phenotypic.enhance import SetDetectMode, SubtractBlank
from phenotypic.enhance._subtract_blank import StaleDetectMatError

target = ImagePipeline(ops={"mode": SetDetectMode(mode="gray")}).apply(
    Image(arr=np.full((8, 8), 0.4, dtype=np.float32), name="t04"))
journal = target._metadata.provenance_journal
operations = journal["applications"][-1]["operations"]
record = dict(operations[-1])
record.update(sequence=len(_operations(journal)) + 1, operation_name="PadImage",
              operation_class="phenotypic.correction._image_padder.PadImage")
operations.append(record)
assert "phenotypic.correction" not in sys.modules, "precondition: correction imported"
layout = pd.DataFrame({"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]})
blank = Image(arr=np.full((8, 8), 0.4, dtype=np.float32), name="t00")
try:
    with ReferenceContext(layout, images={"t00": blank}):
        SubtractBlank().apply(target)
except StaleDetectMatError as exc:
    print("REFUSED:", exc)
else:
    print("ACCEPTED")
"""


def test_a_fresh_process_recognises_a_recorded_phenotypic_corrector():
    """The review's P5: a notebook that never imports phenotypic.correction runs
    SubtractBlank on a store whose history ran a corrector. PhenoTypic's own
    modules are imported to resolve the record."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_SCRIPT],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("REFUSED:"), result.stdout + result.stderr
    assert "an ImageCorrector" in result.stdout


def test_refuses_a_history_naming_an_unresolvable_class():
    """The guard cannot vouch for a history it cannot read."""
    target = _recorded_target("phenotypic.no_such_module.GhostOperation")
    _, blank = _pair()
    with _ctx(blank=blank), pytest.raises(
        StaleDetectMatError, match="phenotypic.no_such_module.GhostOperation"
    ):
        SubtractBlank().apply(target)


def test_accepts_a_recorded_history_without_correctors():
    """Control for the recorded-class tests: a resolvable non-corrector passes."""
    target = _recorded_target("phenotypic.enhance._blur_gauss.BlurGauss")
    _, blank = _pair()
    with _ctx(blank=blank):
        SubtractBlank().apply(target)


def test_refuses_a_corrector_from_an_earlier_application():
    """Stage 2's probe copy opens a fresh provenance application; a Stage-1
    corrector must still be seen (review minor: scan every application)."""
    target, blank = _pair()
    corrected = ImagePipeline(ops={"fix": _NoopCorrector()}).apply(target)
    with _ctx(blank=blank), pytest.raises(RuntimeError) as info:
        ImagePipeline(ops={"sb": SubtractBlank()}).apply(corrected)
    assert isinstance(info.value.__cause__, StaleDetectMatError)


# ---- the reference plate: treated like every other plate (user decision) ----


def test_the_reference_plate_naming_itself_subtracts_to_zero():
    """No special case: the plate is its own blank, so nothing remains."""
    target, _ = _pair()
    with _ctx(blank_name="t04", blank=target):
        out = SubtractBlank().apply(target)
    np.testing.assert_array_equal(out.detect_mat[:], np.zeros_like(out.detect_mat[:]))


@pytest.mark.parametrize("value", ["t04.tif", "path"])
def test_the_reference_plate_named_by_its_file_subtracts_to_zero(tmp_path, value):
    """'t04.tif', or a path to its own file, is the plate's own file: an
    ordinary blank, so the plate subtracts to an empty image."""
    import tifffile

    own = tmp_path / "t04.tif"
    frame = np.full((8, 8), 100, dtype=np.uint8)
    frame[2:4, 2:4] = 220
    tifffile.imwrite(own, frame)
    target = Image.imread(own)
    name = str(own) if value == "path" else value
    layout = pd.DataFrame({"Metadata_ImageName": ["t04"], "Metadata_BlankImage": [name]})
    with ReferenceContext(layout, image_root=tmp_path):
        out = SubtractBlank().apply(target)
    np.testing.assert_array_equal(out.detect_mat[:], np.zeros_like(out.detect_mat[:]))


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


# ---- a blank value that is a file path -------------------------------------


def _null_ctx(target_name="t04"):
    """The image's row exists, but its blank cell is empty."""
    layout = pd.DataFrame({"Metadata_ImageName": [target_name], "Metadata_BlankImage": [None]})
    return ReferenceContext(layout)


def test_a_same_stem_blank_in_another_folder_is_an_ordinary_blank(tmp_path):
    """``blanks/plate1.tif`` for ``images/plate1.tif``: a folder of blanks named
    like the plates. Same stem, different file, subtracted as usual."""
    import tifffile

    own = tmp_path / "images" / "plate1.tif"
    blank = tmp_path / "blanks" / "plate1.tif"
    for path in (own, blank):
        path.parent.mkdir(parents=True)
    tifffile.imwrite(blank, np.full((8, 8), 40, dtype=np.uint8))
    frame = np.full((8, 8), 40, dtype=np.uint8)
    frame[2:4, 2:4] = 200
    tifffile.imwrite(own, frame)
    layout = pd.DataFrame({"Metadata_ImageName": ["plate1"], "Metadata_BlankImage": [str(blank)]})
    with ReferenceContext(layout):
        out = SubtractBlank().apply(Image.imread(own))
    assert out.detect_mat[:][3, 3] > 0.5
    assert out.detect_mat[:][6, 6] == pytest.approx(0.0, abs=1e-6)


def test_a_file_path_blank_is_subtracted(tmp_path):
    import tifffile

    blank = tmp_path / "blanks" / "b0.tif"
    blank.parent.mkdir()
    tifffile.imwrite(blank, np.full((8, 8), 40, dtype=np.uint8))
    frame = np.full((8, 8), 40, dtype=np.uint8)
    frame[2:4, 2:4] = 200
    target = Image(arr=frame, name="t04")
    layout = pd.DataFrame({"Metadata_ImageName": ["t04"], "Metadata_BlankImage": [str(blank)]})
    with ReferenceContext(layout):
        out = SubtractBlank().apply(target)
    assert out.detect_mat[:][3, 3] > 0.5
    assert out.detect_mat[:][6, 6] == pytest.approx(0.0, abs=1e-6)


# ---- what still refuses ------------------------------------------------------


def test_refuses_an_empty_blank_cell():
    target, _ = _pair()
    with _null_ctx(), pytest.raises(ReferenceLookupError) as info:
        SubtractBlank().apply(target)
    assert (info.value.reason, info.value.column) == ("null", "Metadata_BlankImage")


@pytest.mark.parametrize("case", ["unmatched", "ambiguous"])
def test_refuses_a_missing_or_disagreeing_row(case):
    target, blank = _pair()
    names, blanks = {
        "unmatched": (["t09"], ["t00"]),
        "ambiguous": (["t04", "t04"], ["t00", "t01"]),
    }[case]
    layout = pd.DataFrame({"Metadata_ImageName": names, "Metadata_BlankImage": blanks})
    with ReferenceContext(layout, images={"t00": blank}), pytest.raises(ReferenceLookupError) as info:
        SubtractBlank().apply(target)
    assert info.value.reason == case


def test_refuses_a_blank_that_does_not_resolve():
    target, _ = _pair()
    with _ctx(blank=None), pytest.raises(ReferenceImageError):
        SubtractBlank().apply(target)
