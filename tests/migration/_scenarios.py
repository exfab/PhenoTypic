"""Scenario registry for the pydantic-migration golden harness.

A *scenario* is a single ``(operation class, keyword arguments)`` pair
together with the frozen input it runs against. The harness derives
scenarios in two layers:

1. **Auto-default scenarios** -- :func:`discover_operations` walks the
   eight operation subpackages plus ``phenotypic.analysis`` and collects
   every concrete ``BaseOperation`` / analyzer class. Each gets a
   ``"defaults"`` scenario constructed with no arguments. A small
   :data:`REQUIRED_ARGS` table supplies the minimal keyword set for the
   few classes whose constructors have required parameters.
2. **Curated extras** -- :data:`CURATED_EXTRAS` adds hand-picked
   non-default scenarios for important or hard operations (stochastic
   detectors, kernel-array enhancers, the symmetric-zone measurer, ...).

Every scenario is keyword-only constructed, so the registry is valid on
both the pre-migration code and the migrated pydantic models.

Input-category routing
----------------------

Each operation's *input category* is derived from its abstract base
class (see :func:`category_for`):

* enhancers / detectors / correctors -> ``raw_plate``
* refiners -> ``detected_plate``
* non-grid measurers -> ``detected_plate``
* grid measurers -> ``detected_grid``
* grid finders -> ``raw_grid``
* analyzers -> ``reference_measurements``

The category in turn fixes the *invocation kind* (see
:func:`invocation_for`): ``ImageOperation`` subclasses use
``.apply(image)``, ``MeasureFeatures`` subclasses (including grid
finders) use ``.measure(image)``, ``PostMeasurement`` subclasses use
``.apply(df)`` against the reference frame, and analyzers use
``.analyze(df)``.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import Any

# --- Subpackages walked for operation discovery --------------------------

OPERATION_SUBPACKAGES: tuple[str, ...] = (
    "detect",
    "enhance",
    "refine",
    "correction",
    "grid",
    "post",
    "measure",
    "nn",
)

# Concrete analyzer classes in phenotypic.analysis (scope = these 5).
ANALYZER_NAMES: tuple[str, ...] = (
    "TukeyOutlierRemover",
    "EdgeCorrector",
    "LogGrowthModel",
    "ExpectedVsDetectedCount",
    "ReplicateAgreement",
)

# Abstract base names that must never themselves be treated as scenarios.
_ABC_NAMES: frozenset[str] = frozenset(
        {
            "BaseOperation",
            "ImageOperation",
            "ImageEnhancer",
            "ImageDenoiser",
            "ImageCorrector",
            "ObjectDetector",
            "ObjectRefiner",
            "ThresholdDetector",
            "GpuDetector",
            "GridOperation",
            "GridFinder",
            "GridCorrector",
            "GridObjectRefiner",
            "GridMeasureFeatures",
            "GridObjectDetector",
            "MeasureFeatures",
            "PostMeasurement",
            "PrefabPipeline",
        }
)

# --- Input-category constants -------------------------------------------

CATEGORY_RAW_PLATE = "raw_plate"
CATEGORY_DETECTED_PLATE = "detected_plate"
CATEGORY_RAW_GRID = "raw_grid"
CATEGORY_DETECTED_GRID = "detected_grid"
CATEGORY_REFERENCE = "reference_measurements"

# --- Invocation-kind constants ------------------------------------------

INVOKE_APPLY_IMAGE = "apply_image"  # op.apply(image[, inplace])
INVOKE_MEASURE = "measure"  # op.measure(image)
INVOKE_APPLY_DF = "apply_df"  # op.apply(dataframe)
INVOKE_ANALYZE = "analyze"  # analyzer.analyze(dataframe)

# --- Image-component constants ------------------------------------------
#
# The array(s) an image operation may legitimately modify -- captured and
# compared per scenario. Storing only the relevant components keeps the
# golden fixtures small (an enhancer's untouched objmap is not stored)
# and makes the equivalence assertion a tighter contract.

COMPONENT_DETECT_MAT = "detect_mat"
COMPONENT_OBJMASK = "objmask"
COMPONENT_OBJMAP = "objmap"

# Components touched by an ``ImageEnhancer`` (detection matrix only).
_COMPONENTS_ENHANCER: tuple[str, ...] = (COMPONENT_DETECT_MAT,)
# Components touched by detectors / refiners (object mask + map).
_COMPONENTS_OBJECTS: tuple[str, ...] = (
    COMPONENT_OBJMASK,
    COMPONENT_OBJMAP,
)
# Components an ``ImageCorrector`` / unknown op may touch (everything).
_COMPONENTS_ALL: tuple[str, ...] = (
    COMPONENT_DETECT_MAT,
    COMPONENT_OBJMASK,
    COMPONENT_OBJMAP,
)


# --- Required-argument table --------------------------------------------
#
# Only classes whose constructors have *required* parameters appear here.
# Values are produced lazily by zero-argument factories so importing this
# module never touches the frozen inputs or builds heavy objects.


def _manual_grid_finder_args() -> dict[str, Any]:
    """Minimal valid args for ``ManualGridFinder`` (8x12 plate)."""
    import numpy as np

    return {
        "row_edges": np.linspace(0, 600, 9, dtype=int),
        "col_edges": np.linspace(0, 800, 13, dtype=int),
    }


def _expected_vs_detected_args() -> dict[str, Any]:
    """Minimal valid args for ``ExpectedVsDetectedCount``.

    The metadata frame is the unique ``(plate, grid-cell)`` index of the
    frozen reference measurements -- one expected colony per well.
    """
    from tests.migration._inputs import load_frozen_input

    ref = load_frozen_input(CATEGORY_REFERENCE)
    metadata = (
        ref[["Metadata_ImageName", "Grid_RowNum", "Grid_ColNum"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return {
        "metadata": metadata,
        "groupby" : ["Metadata_ImageName"],
    }


# Maps class name -> zero-arg factory returning required ctor kwargs.
REQUIRED_ARGS: dict[str, Any] = {
    "ManualGridFinder"       : _manual_grid_finder_args,
    "ExpectedVsDetectedCount": _expected_vs_detected_args,
}

# Maps class name -> kwargs for classes whose all-default constructor is
# *technically valid but degenerate* -- i.e. it instantiates but cannot
# run. The four ``post/`` transforms default ``column=""`` (no target
# column), so their "defaults" scenario must name a real column from the
# reference frame. ``_ensure_prefix`` adds the ``Metadata_`` prefix, so
# bare names like ``"ImageName"`` resolve to ``"Metadata_ImageName"``.
DEFAULT_ARG_OVERRIDES: dict[str, dict[str, Any]] = {
    "AppendString"  : {"column": "ImageName", "value": "_suffix"},
    "PrependString" : {"column": "ImageName", "value": "prefix_"},
    "ExpandMetadata": {
        "column"   : "Tag",
        "labels"   : ["Plate", "Hour", "Rep"],
        "delimiter": "-",
    },
    "MergeMetadata" : {
        "columns"  : ["ImageName", "Time"],
        "label"    : "Combined",
        "delimiter": "_",
    },
}

# Per-class frozen-input-category overrides. A handful of plain
# ``ObjectRefiner`` classes are not ``GridOperation`` subclasses yet read
# ``image.grid`` inside ``_operate``, so the ABC-based router would route
# them to a plain ``Image`` that has no grid. Pin them to a grid input.
CATEGORY_OVERRIDES: dict[str, str] = {
    "KeepSectionLargest": CATEGORY_DETECTED_GRID,
}

# Standard analyzer keyword set (column-based analyzers share this).
_ANALYZER_BASE_ARGS: dict[str, Any] = {
    "on"     : "Size_Area",
    "groupby": ["Metadata_ImageName"],
}

# Per-analyzer default keyword overrides (merged onto _ANALYZER_BASE_ARGS).
_ANALYZER_DEFAULT_ARGS: dict[str, dict[str, Any]] = {
    "TukeyOutlierRemover": dict(_ANALYZER_BASE_ARGS),
    "EdgeCorrector"      : dict(_ANALYZER_BASE_ARGS),
    "ReplicateAgreement" : dict(_ANALYZER_BASE_ARGS),
    "LogGrowthModel"     : {
        "on"     : "Size_Area",
        "groupby": ["Metadata_ImageName", "Grid_RowNum", "Grid_ColNum"],
    },
    # ExpectedVsDetectedCount's required args come from REQUIRED_ARGS.
}

# Classes that cannot be auto-defaulted and have no curated minimal arg
# set -- recorded as skipped by the capture script with this reason.
UNCAPTURABLE: dict[str, str] = {
    "ColorCorrector"    : (
        "requires a fitted ColorCheckerProfile (a color-chart "
        "calibration object); no synthetic plate carries one"
    ),
    "MergeWithinSection": (
        "pre-existing library bug: MergeWithinSection._operate never "
        "returns the image (implicitly returns None), so there is no "
        "result to freeze as a golden on the current code"
    ),
    "GridApply"         : (
        "not part of the original 137-class migration scope: it was a "
        "plain wrapper class promoted to a pydantic GridCorrector during "
        "Phase 6 green-up so it could slot into a pydantic ImagePipeline. "
        "No pre-migration golden was captured (the un-migrated class was "
        "not an operation), and it has a required operation-valued "
        "image_op argument with no synthetic default. GridSectionPipeline "
        "round-trip coverage in tests/smoke/test_serialization.py "
        "exercises GridApply end-to-end instead."
    ),
}


@dataclass(frozen=True)
class Scenario:
    """A single capture/equivalence scenario.

    The constructor keyword arguments are held as a *zero-argument
    factory* (:attr:`kwargs_factory`) rather than a resolved dict, so
    assembling the scenario list never touches the frozen inputs.
    ``ExpectedVsDetectedCount`` in particular needs the reference frame
    to build its ``metadata`` argument; deferring resolution lets the
    pytest harness parametrize over scenarios at collection time.

    Attributes:
        scenario_id: Unique slug, ``"<subpackage>.<Class>[.<variant>]"``.
        subpackage: Owning subpackage (``detect``, ..., ``analysis``).
        class_name: Operation/analyzer class name.
        variant: ``"defaults"`` or a curated-extra label.
        kwargs_factory: Zero-argument callable returning the constructor
            keyword-argument dict. Call :meth:`resolve_kwargs` to invoke
            it.
        category: Frozen-input category (see ``CATEGORY_*``).
        invocation: Invocation kind (see ``INVOKE_*``).
        components: Image components to capture/compare, for image
            scenarios only (see ``COMPONENT_*`` and
            :func:`components_for`). Empty for frame scenarios.
        stochastic: Whether the operation needs a seeded numpy RNG.
        structural_only: Whether only shape/dtype/columns are captured
            (true for ``nn/`` model-backed detectors).
        tolerance: Absolute float tolerance for the golden comparison.
            ``0.0`` (default) means bit-exact. A small positive value is
            used for operations that are non-deterministic at the C-
            library level (the ``bm3d`` backend), where numpy's global
            RNG seed cannot pin the result -- see :data:`TOLERANT_OPS`.
    """

    scenario_id: str
    subpackage: str
    class_name: str
    variant: str
    kwargs_factory: Any
    category: str
    invocation: str
    components: tuple[str, ...] = ()
    stochastic: bool = False
    structural_only: bool = False
    tolerance: float = 0.0

    def resolve_kwargs(self) -> dict[str, Any]:
        """Resolve and return the constructor keyword arguments.

        Returns:
            A fresh keyword-argument dict (the factory is invoked anew on
            every call so mutable defaults are never shared).
        """
        return dict(self.kwargs_factory())


@dataclass
class _CuratedExtra:
    """A curated non-default scenario before class resolution.

    Attributes:
        class_name: Target operation/analyzer class name.
        variant: Short label distinguishing this from ``defaults``.
        kwargs: Constructor keyword arguments (or a zero-arg factory).
        stochastic: Whether the operation needs a seeded RNG.
    """

    class_name: str
    variant: str
    kwargs: Any
    stochastic: bool = False


# --- Curated extras ------------------------------------------------------
#
# ~20 hand-picked non-default scenarios for important / hard operations.
# kwargs may be a dict or a zero-arg factory (deferred construction).


def _manual_point_detector_args() -> dict[str, Any]:
    """ManualPointDetector seeded with a few plate-interior centers."""
    return {
        "centers": [(150, 200), (300, 400), (450, 600)],
        "shape"  : "disk",
        "width"  : 20,
    }


def _manual_selector_args() -> dict[str, Any]:
    """ManualRefine keeping objects near three plate-interior points."""
    return {"centers": [(150, 200), (300, 400), (450, 600)]}


CURATED_EXTRAS: tuple[_CuratedExtra, ...] = (
    # -- detect --
    _CuratedExtra("OtsuDetector", "ignore_zeros", {"ignore_zeros": True}),
    _CuratedExtra(
            "OtsuDetector", "keep_borders", {"ignore_borders": False}
    ),
    _CuratedExtra("LiDetector", "ignore_zeros", {"ignore_zeros": True}),
    _CuratedExtra(
            "UserThreshold", "high_thresh", {"threshold": 0.7}
    ),
    _CuratedExtra(
            "CannyDetector", "tight_sigma", {"sigma": 2.0, "min_size": 30}
    ),
    _CuratedExtra(
            "WatershedDetector",
            "compact_small",
            {"min_size": 30, "compactness": 0.01},
            stochastic=True,
    ),
    _CuratedExtra(
            "InoculumDetector",
            "gmm_off",
            {"enable_gmm": False},
            stochastic=True,
    ),
    _CuratedExtra(
            "ManualPointDetector",
            "seeded_centers",
            _manual_point_detector_args,
    ),
    _CuratedExtra(
            "ManualGridPointDetector",
            "two_corners",
            {"coord1": (60, 70), "coord2": (540, 730), "width": 18},
    ),
    # -- enhance --
    _CuratedExtra("GaussianBlur", "sigma4", {"sigma": 4.0}),
    _CuratedExtra(
            "GaussianBlur",
            "constant_mode",
            {"sigma": 2.0, "mode": "constant", "cval": 0.0},
    ),
    _CuratedExtra("CLAHE", "small_kernel", {"kernel_size": 32}),
    _CuratedExtra("MedianFilter", "wide", {"width": 9}),
    _CuratedExtra(
            "SubtractRollingBall", "small_ball", {"radius": 50}
    ),
    _CuratedExtra("UnsharpMask", "strong", {"amount": 2.0}),
    # -- refine --
    _CuratedExtra(
            "SmallObjectRemover", "aggressive", {"min_size": 256}
    ),
    _CuratedExtra(
            "MaskDilator", "disk3x", {"width": 5, "n_iter": 2}
    ),
    _CuratedExtra(
            "RemoveBorderObjects", "wide_margin", {"border_size": 60}
    ),
    _CuratedExtra(
            "ManualRefine", "seeded_centers", _manual_selector_args
    ),
    # -- measure --
    _CuratedExtra(
            "MeasureColor", "with_xyz", {"include_XYZ": True}
    ),
    _CuratedExtra(
            "MeasureSymmetricZones",
            "intensity_method",
            {"method": "intensity", "n_annuli": 60},
    ),
    # -- analysis --
    _CuratedExtra(
            "TukeyOutlierRemover", "k3", {**_ANALYZER_BASE_ARGS, "k": 3.0}
    ),
)


# --- Discovery -----------------------------------------------------------


def discover_operations() -> dict[str, list[type]]:
    """Discover concrete operation classes per subpackage.

    Walks every module in each subpackage of
    :data:`OPERATION_SUBPACKAGES` and collects classes that subclass
    ``BaseOperation``, are concrete (not abstract), are defined in the
    walked module, and are not one of the shared ABCs.

    Returns:
        A mapping from subpackage name to a name-sorted list of
        operation classes.
    """
    from phenotypic.abc_ import BaseOperation

    discovered: dict[str, list[type]] = {}
    for subpkg in OPERATION_SUBPACKAGES:
        pkg = importlib.import_module(f"phenotypic.{subpkg}")
        found: dict[str, type] = {}
        for _finder, modname, _ispkg in pkgutil.walk_packages(
                pkg.__path__, f"{pkg.__name__}."
        ):
            try:
                module = importlib.import_module(modname)
            except Exception:  # noqa: BLE001 - optional deps may be absent
                continue
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                        issubclass(obj, BaseOperation)
                        and obj.__module__ == modname
                        and not inspect.isabstract(obj)
                        and name not in _ABC_NAMES
                ):
                    found[name] = obj
        discovered[subpkg] = [found[n] for n in sorted(found)]
    return discovered


def discover_analyzers() -> list[type]:
    """Resolve the five concrete analyzer classes in scope.

    Returns:
        A name-sorted list of the analyzer classes named in
        :data:`ANALYZER_NAMES`.
    """
    analysis = importlib.import_module("phenotypic.analysis")
    classes: list[type] = []
    for name in sorted(ANALYZER_NAMES):
        classes.append(getattr(analysis, name))
    return classes


# --- Category / invocation routing --------------------------------------


def category_for(cls: type, subpackage: str) -> str:
    """Return the frozen-input category for an operation/analyzer class.

    Args:
        cls: Operation or analyzer class.
        subpackage: Owning subpackage name (``"analysis"`` for analyzers).

    Returns:
        One of the ``CATEGORY_*`` constants.
    """
    if subpackage == "analysis":
        return CATEGORY_REFERENCE

    if cls.__name__ in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[cls.__name__]

    from phenotypic.abc_ import (
        GridFinder,
        GridMeasureFeatures,
        GridObjectDetector,
        GridOperation,
        ImageCorrector,
        MeasureFeatures,
        ObjectDetector,
        ObjectRefiner,
        PostMeasurement,
    )

    if issubclass(cls, PostMeasurement):
        return CATEGORY_REFERENCE
    if issubclass(cls, GridFinder):
        return CATEGORY_RAW_GRID
    if issubclass(cls, GridMeasureFeatures):
        return CATEGORY_DETECTED_GRID
    # Grid-aware operations (GridObjectDetector / GridObjectRefiner /
    # GridCorrector) require a GridImage. Grid detectors run on a raw
    # grid; grid refiners/correctors need objects already present.
    if issubclass(cls, GridOperation):
        if issubclass(cls, GridObjectDetector):
            return CATEGORY_RAW_GRID
        return CATEGORY_DETECTED_GRID
    if issubclass(cls, MeasureFeatures):
        return CATEGORY_DETECTED_PLATE
    if issubclass(cls, ObjectRefiner):
        return CATEGORY_DETECTED_PLATE
    if issubclass(cls, (ObjectDetector, ImageCorrector)):
        return CATEGORY_RAW_PLATE
    # ImageEnhancer / ImageDenoiser / plain ImageOperation.
    return CATEGORY_RAW_PLATE


def invocation_for(cls: type, subpackage: str) -> str:
    """Return the invocation kind for an operation/analyzer class.

    Args:
        cls: Operation or analyzer class.
        subpackage: Owning subpackage name (``"analysis"`` for analyzers).

    Returns:
        One of the ``INVOKE_*`` constants.
    """
    if subpackage == "analysis":
        return INVOKE_ANALYZE

    from phenotypic.abc_ import MeasureFeatures, PostMeasurement

    if issubclass(cls, PostMeasurement):
        return INVOKE_APPLY_DF
    if issubclass(cls, MeasureFeatures):
        # Grid finders are MeasureFeatures too -> .measure().
        return INVOKE_MEASURE
    return INVOKE_APPLY_IMAGE


def components_for(cls: type) -> tuple[str, ...]:
    """Return the image components a scenario should capture/compare.

    Mirrors PhenoTypic's own integrity contract (see ``ImageOperation``):
    an ``ImageEnhancer`` modifies only ``detect_mat``; detectors and
    refiners modify only the object mask + map; an ``ImageCorrector``
    (or any operation not matching the above) may modify everything, so
    all three components are captured.

    Args:
        cls: An ``ImageOperation`` subclass (used only for image
            scenarios).

    Returns:
        The ordered tuple of component names to persist and compare.
    """
    from phenotypic.abc_ import (
        ImageEnhancer,
        ObjectDetector,
        ObjectRefiner,
    )

    if issubclass(cls, ImageEnhancer):
        return _COMPONENTS_ENHANCER
    if issubclass(cls, (ObjectDetector, ObjectRefiner)):
        return _COMPONENTS_OBJECTS
    # ImageCorrector and any plain ImageOperation: capture everything.
    return _COMPONENTS_ALL


# --- Stochastic / structural-only classification ------------------------

# Operations whose default scenario depends on a numpy RNG draw.
_STOCHASTIC_DEFAULTS: frozenset[str] = frozenset(
        {
            "InoculumDetector",
            "WatershedDetector",
            "FilamentousFungiDetector",
            "GMMCoreExtractor",
            "ChanVeseDetector",
        }
)

# nn/ detectors need model checkpoints -> structural-only capture.
_STRUCTURAL_ONLY: frozenset[str] = frozenset(
        {"Sam2Detector", "MicroSamDetector"}
)

# Operations that are *not bit-reproducible* even on the unmigrated
# code. Both are backed by the ``bm3d`` C library, whose internal
# numerics jitter at the ~1e-7 level on repeat runs; numpy's global RNG
# seed has no effect on it. The migration must keep these numerically
# equivalent within this absolute tolerance rather than bit-exact.
# Confirmed by running each twice during harness development: the
# detect_mat differed by < 1e-6.
TOLERANT_OPS: dict[str, float] = {
    "BM3DDenoiser" : 1e-5,
    "StableDenoise": 1e-5,
}


def _as_factory(kwargs: Any) -> Any:
    """Coerce a kwargs spec into a zero-argument factory callable.

    Args:
        kwargs: Either a ready dict or a callable returning one.

    Returns:
        A zero-argument callable yielding the keyword-argument dict.
    """
    if callable(kwargs):
        return kwargs
    snapshot = dict(kwargs)
    return lambda: dict(snapshot)


# --- Scenario assembly ---------------------------------------------------


def build_scenarios() -> list[Scenario]:
    """Assemble the full ordered scenario list.

    Produces one ``"defaults"`` scenario per discovered operation and
    in-scope analyzer, then appends every resolvable curated extra.
    Classes listed in :data:`UNCAPTURABLE` are skipped (the capture
    script reports them separately).

    Constructor keyword arguments are stored as deferred factories on
    each :class:`Scenario` -- this function never resolves them, so it is
    safe to call before the frozen inputs have been captured (e.g. at
    pytest collection time).

    Returns:
        The ordered list of :class:`Scenario` objects.
    """
    scenarios: list[Scenario] = []
    name_to_subpkg: dict[str, str] = {}
    name_to_cls: dict[str, type] = {}

    # 1. Auto-default scenarios for operations.
    discovered = discover_operations()
    for subpkg, classes in discovered.items():
        for cls in classes:
            name = cls.__name__
            name_to_subpkg[name] = subpkg
            name_to_cls[name] = cls
            if name in UNCAPTURABLE:
                continue
            if name in REQUIRED_ARGS:
                kwargs_factory = _as_factory(REQUIRED_ARGS[name])
            else:
                kwargs_factory = _as_factory(
                        DEFAULT_ARG_OVERRIDES.get(name, {})
                )
            invocation = invocation_for(cls, subpkg)
            components = (
                components_for(cls)
                if invocation == INVOKE_APPLY_IMAGE
                else ()
            )
            scenarios.append(
                    Scenario(
                            scenario_id=f"{subpkg}.{name}",
                            subpackage=subpkg,
                            class_name=name,
                            variant="defaults",
                            kwargs_factory=kwargs_factory,
                            category=category_for(cls, subpkg),
                            invocation=invocation,
                            components=components,
                            stochastic=name in _STOCHASTIC_DEFAULTS,
                            structural_only=name in _STRUCTURAL_ONLY,
                            tolerance=TOLERANT_OPS.get(name, 0.0),
                    )
            )

    # 2. Auto-default scenarios for analyzers.
    for cls in discover_analyzers():
        name = cls.__name__
        name_to_subpkg[name] = "analysis"
        name_to_cls[name] = cls
        if name in REQUIRED_ARGS:
            kwargs_factory = _as_factory(REQUIRED_ARGS[name])
        else:
            kwargs_factory = _as_factory(
                    _ANALYZER_DEFAULT_ARGS.get(name, _ANALYZER_BASE_ARGS)
            )
        scenarios.append(
                Scenario(
                        scenario_id=f"analysis.{name}",
                        subpackage="analysis",
                        class_name=name,
                        variant="defaults",
                        kwargs_factory=kwargs_factory,
                        category=CATEGORY_REFERENCE,
                        invocation=INVOKE_ANALYZE,
                )
        )

    # 3. Curated extras.
    for extra in CURATED_EXTRAS:
        cls = name_to_cls.get(extra.class_name)
        if cls is None:
            # Curated extra naming a class that did not discover (e.g.
            # an optional-dependency module failed to import). Skipped
            # silently here; the capture summary still covers defaults.
            continue
        subpkg = name_to_subpkg[extra.class_name]
        invocation = invocation_for(cls, subpkg)
        components = (
            components_for(cls)
            if invocation == INVOKE_APPLY_IMAGE
            else ()
        )
        scenarios.append(
                Scenario(
                        scenario_id=(
                            f"{subpkg}.{extra.class_name}.{extra.variant}"
                        ),
                        subpackage=subpkg,
                        class_name=extra.class_name,
                        variant=extra.variant,
                        kwargs_factory=_as_factory(extra.kwargs),
                        category=category_for(cls, subpkg),
                        invocation=invocation,
                        components=components,
                        stochastic=(
                                extra.stochastic
                                or extra.class_name in _STOCHASTIC_DEFAULTS
                        ),
                        structural_only=extra.class_name in _STRUCTURAL_ONLY,
                        tolerance=TOLERANT_OPS.get(extra.class_name, 0.0),
                )
        )

    return scenarios


def resolve_class(scenario: Scenario) -> type:
    """Return the operation/analyzer class for a scenario.

    Args:
        scenario: The scenario whose class to resolve.

    Returns:
        The class object named by ``scenario.class_name``.

    Raises:
        LookupError: If the class cannot be found in its subpackage.
    """
    if scenario.subpackage == "analysis":
        module = importlib.import_module("phenotypic.analysis")
        return getattr(module, scenario.class_name)

    for cls in discover_operations().get(scenario.subpackage, []):
        if cls.__name__ == scenario.class_name:
            return cls
    raise LookupError(
            f"Class {scenario.class_name!r} not found in "
            f"subpackage {scenario.subpackage!r}."
    )
