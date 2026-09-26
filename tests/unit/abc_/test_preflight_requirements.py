"""Operations declare what they need from the run (spec 2026-09-24-cli-preflight §3).

``BaseOperation.preflight_requirements()`` is what the CLI's run preflight
reads to refuse, before any image is processed, a grid operation under
``--image-type Image`` or an RGB-reading operation on grayscale inputs.
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys

import pytest

from phenotypic.abc_ import BaseOperation, OperationRequirements

#: Every concrete class that raises ``GridImageInputError`` on a plain Image at
#: 81d19ec (review R13). Each derives from one of the four raising ABCs.
GRID_CLASSES = (
    ("phenotypic.detect", "FilamentousFungiDetector"),
    ("phenotypic.detect", "TwoKFilamentousDetector"),
    ("phenotypic.detect", "ManualGridPointDetector"),
    ("phenotypic.correction", "GridAligner"),
    ("phenotypic.grid", "GridApply"),
    ("phenotypic.refine", "GridOversizedObjectRemover"),
    ("phenotypic.refine", "KeepSectionLargest"),
    ("phenotypic.refine._merge_within_section", "MergeWithinSection"),
    ("phenotypic.refine", "ReduceSectionsByLine"),
    ("phenotypic.refine", "RemoveGridOutliers"),
    ("phenotypic.measure", "MeasureGridSpread"),
    ("phenotypic.measure", "MeasureGridLinRegStats"),
    ("phenotypic.grid", "AutoGridFinder"),
    ("phenotypic.grid", "CenteredAutoGridFinder"),
    ("phenotypic.grid", "ManualGridFinder"),
)

#: Grid-aware operations that fall back on a plain Image, so must NOT be flagged.
GRID_TOLERANT_CLASSES = (
    ("phenotypic.detect", "RoundPeaksDetector"),
    ("phenotypic.detect", "SinePeakDetector"),
    ("phenotypic.refine", "RefineBySineFit"),
    ("phenotypic.refine", "GridAlignmentRefiner"),
    ("phenotypic.detect", "InoculumDetector"),
)

#: Classes that read RGB on every call.
UNCONDITIONAL_RGB_CLASSES = (
    ("phenotypic.measure", "MeasureColor"),
    ("phenotypic.measure._measure_color_composition", "MeasureColorComposition"),
    ("phenotypic.enhance", "FocusEdgeColorPhase"),
    ("phenotypic.correction", "ColorDenoise"),
    ("phenotypic.correction._color_correction._color_corrector", "ColorCorrector"),
    ("phenotypic.correction", "CalibrateColorRpcc"),
)

#: Gray-tolerant readers that declare ``False`` explicitly.
GRAY_TOLERANT_RGB_READERS = (
    ("phenotypic.correction", "BayesShrinkCorrector"),
    ("phenotypic.correction", "VisuShrinkCorrector"),
    ("phenotypic.correction", "DenoiseBlockMatch"),
    ("phenotypic.measure", "MeasureSymZones"),
    ("phenotypic.correction", "PadImage"),
)

#: Packages whose concrete operations the ratchet scans.
OPERATION_PACKAGES = (
    "phenotypic.detect",
    "phenotypic.enhance",
    "phenotypic.refine",
    "phenotypic.grid",
    "phenotypic.correction",
    "phenotypic.measure",
    "phenotypic.post",
    "phenotypic.prefab",
    "phenotypic.detect.nn",
)


def _cls(module: str, name: str) -> type:
    return getattr(importlib.import_module(module), name)


def _requirements_of_class(cls: type) -> OperationRequirements:
    """Requirements of an instance built without validation or post-init hooks.

    ``cls.__new__`` bypasses pydantic entirely, so classes with required fields
    (``CalibrateColorRpcc(rois=...)``) or a ``model_post_init`` that reads them
    (``ColorCorrector``) are inspectable too. Only type-derived and
    class-variable requirements can be read this way, which is exactly what
    the callers below ask about.
    """
    return cls.__new__(cls).preflight_requirements()


def test_a_plain_operation_requires_nothing() -> None:
    from phenotypic.detect import OtsuDetector

    assert OtsuDetector().preflight_requirements() == OperationRequirements()


@pytest.mark.parametrize(("module", "name"), GRID_CLASSES)
def test_grid_operations_require_a_grid_image(module: str, name: str) -> None:
    assert _requirements_of_class(_cls(module, name)).grid_image


@pytest.mark.parametrize(("module", "name"), GRID_TOLERANT_CLASSES)
def test_grid_tolerant_operations_do_not(module: str, name: str) -> None:
    assert not _requirements_of_class(_cls(module, name)).grid_image


@pytest.mark.parametrize(
    "name", ["ContrastGamma", "ContrastLog", "ContrastSigmoid", "ContrastStretching"]
)
def test_input_layer_ops_require_rgb_only_when_reading_rgb(name: str) -> None:
    cls = _cls("phenotypic.enhance", name)

    assert cls(input_layer="rgb").preflight_requirements().rgb_input
    assert not cls().preflight_requirements().rgb_input


def test_set_detect_mode_requires_rgb_for_colour_modes() -> None:
    from phenotypic.enhance import SetDetectMode

    assert SetDetectMode(mode="red").preflight_requirements().rgb_input
    assert SetDetectMode(mode="LabL").preflight_requirements().rgb_input
    assert not SetDetectMode(mode="gray").preflight_requirements().rgb_input


@pytest.mark.parametrize(("module", "name"), UNCONDITIONAL_RGB_CLASSES)
def test_unconditional_rgb_readers_require_rgb(module: str, name: str) -> None:
    assert _requirements_of_class(_cls(module, name)).rgb_input


@pytest.mark.parametrize(("module", "name"), GRAY_TOLERANT_RGB_READERS)
def test_gray_tolerant_readers_declare_false(module: str, name: str) -> None:
    cls = _cls(module, name)
    assert cls.__dict__["_requires_rgb_input"] is False
    assert not _requirements_of_class(cls).rgb_input


def test_gpu_detectors_follow_their_input_layer() -> None:
    from tests._fakes.fake_gpu_detector import FakeGpuDetector

    assert FakeGpuDetector(input_layer="rgb").preflight_requirements().rgb_input
    assert not FakeGpuDetector(input_layer="gray").preflight_requirements().rgb_input


def _concrete_operation_classes() -> list[type]:
    for package in OPERATION_PACKAGES:
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", ()):
            getattr(module, name, None)  # resolve lazy exports
    seen: set[type] = set()
    stack = list(BaseOperation.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return sorted(
        (
            cls
            for cls in seen
            if cls.__module__.startswith("phenotypic.")
            and not inspect.isabstract(cls)
            and not cls.__name__.startswith("_")
        ),
        key=lambda cls: (cls.__module__, cls.__qualname__),
    )


def test_ratchet_rgb_readers_declare_their_requirement_explicitly() -> None:
    """Every class whose module reads ``.rgb[`` or ``.color.`` sets the flag itself.

    At 81d19ec this named ``BayesShrinkCorrector``, ``VisuShrinkCorrector``,
    ``DenoiseBlockMatch`` (a false positive: a docstring mentions
    ``skimage.color``) and ``MeasureSymZones`` (a plot-only RGB read), all
    gray-tolerant, which now declare ``False``.

    A heuristic, not a proof: it catches the common spellings of an RGB read,
    not every one. A gray-tolerant reader satisfies it by setting
    ``_requires_rgb_input = False`` explicitly, which records that someone
    checked. Classes whose requirement depends on a field (``InputLayerMixin``,
    ``GpuDetector``, ``SetDetectMode``) override ``preflight_requirements`` and
    are exempt.
    """
    undeclared = []
    for cls in _concrete_operation_classes():
        source = inspect.getsource(sys.modules[cls.__module__])
        if ".rgb[" not in source and ".color." not in source:
            continue
        declared = any(
            "_requires_rgb_input" in vars(klass)
            or "preflight_requirements" in vars(klass)
            for klass in cls.__mro__
            if klass is not BaseOperation and klass.__module__.startswith("phenotypic.")
            and klass.__module__ == cls.__module__
        )
        if not declared:
            undeclared.append(f"{cls.__module__}.{cls.__qualname__}")
    assert not undeclared, (
        "these operations read RGB (by the heuristic) but do not declare "
        f"_requires_rgb_input explicitly: {undeclared}"
    )


def test_requirements_are_not_fields() -> None:
    from phenotypic.measure import MeasureColor

    fields = set(MeasureColor.model_fields)
    assert not {"_requires_rgb_input", "_requires_modules", "_requires_extra"} & fields
    assert "_requires_rgb_input" not in MeasureColor.model_json_schema()["properties"]


def test_importing_abc_loads_no_deferred_runtime_module() -> None:
    code = (
        "import sys, phenotypic.abc_ as a; a.OperationRequirements; "
        "from phenotypic._startup_perf import DEFERRED_RUNTIME_MODULES as D; "
        "print(sorted(m for m in (*D, 'torch', 'PIL') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout
