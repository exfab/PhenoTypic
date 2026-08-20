"""One list, two consumers. Two lists is how ``detect.nn`` went missing.

Task 10a lifts the ``submodules`` literal out of
``SerializablePipeline._find_class_in_phenotypic`` into a module-level
constant, with **zero behaviour change**: the loader keeps resolving exactly
the modules it resolved before, in exactly the same order. Wiring the second
consumer (``OperationRegistry.discover``) is Task 10b's job, because
``discover`` needs a category and a base class per module and cannot read a
bare list of module names.
"""

from __future__ import annotations

import pytest

from phenotypic._core._pipeline_parts import _serializable_pipeline
from phenotypic._core._pipeline_parts._serializable_pipeline import (
    PHENOTYPIC_CLASS_MODULES,
    SerializablePipeline,
)

# Verbatim copy of the ``submodules`` literal as it stood before the lift
# (``_serializable_pipeline.py:645``). Resolution is first-match, so this is
# compared as an ordered sequence: a reordering silently changes which module
# wins a duplicate class name, and a membership-only check would not see it.
_SUBMODULES_BEFORE_THE_LIFT = (
    "phenotypic.detect",
    "phenotypic.measure",
    "phenotypic.enhance",
    "phenotypic.refine",
    "phenotypic.grid",
    "phenotypic.correction",
    "phenotypic.analysis",
    "phenotypic.prefab",
    "phenotypic.post",
    "phenotypic.detect.nn",
    "phenotypic.tune",
    "phenotypic.tune.score",
    "phenotypic.tune.strategy",
)


def test_one_shared_module_list():
    """The constant is the old literal, in order."""
    assert tuple(PHENOTYPIC_CLASS_MODULES) == _SUBMODULES_BEFORE_THE_LIFT


@pytest.mark.parametrize(
    "class_name, module_name",
    [
        ("BlurGauss", "phenotypic.enhance"),
        ("MeasureTexture", "phenotypic.measure"),
        ("QCScorer", "phenotypic.tune.score"),
    ],
)
def test_loader_resolves_through_the_constant(monkeypatch, class_name, module_name):
    """The lift wired the loader to the constant, not to a stale copy.

    Emptying the constant must break resolution for every class that is only
    reachable through the submodule walk. If ``_find_class_in_phenotypic``
    kept its own literal, resolution would survive the monkeypatch.
    """
    resolved = SerializablePipeline._find_class_in_phenotypic(class_name)
    assert resolved is not None
    assert resolved.__module__.startswith(module_name.rsplit(".", 1)[0])

    monkeypatch.setattr(_serializable_pipeline, "PHENOTYPIC_CLASS_MODULES", ())
    assert SerializablePipeline._find_class_in_phenotypic(class_name) is None


# --------------------------------------------------------------------------
# Task 10b: the catalog reaches the families the loader could always resolve.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def discovered_registry():
    """A freshly discovered registry (never the ``get_registry`` singleton)."""
    from phenotypic._services.registry import OperationRegistry

    registry = OperationRegistry()
    registry.discover()
    return registry


def test_registry_reaches_prefabs(discovered_registry):
    """``PrefabPipeline`` is not an ``ImageOperation``, so nothing found it."""
    names = set(discovered_registry.get_all())
    assert "FilamentousFungiPipeline" in names, "prefabs unreachable from the catalog"
    assert "HeavyWatershedPipeline" in names
    assert discovered_registry.get("FilamentousFungiPipeline").category == "Prefab"


def test_scorers_and_strategies_are_catalog_citizens(discovered_registry):
    """§3.1: without these, an agent authoring a spec can only guess."""
    names = set(discovered_registry.get_all())
    assert {"QCScorer", "SupervisedScorer", "ReferenceFreeScorer"} <= names
    assert {"GridConfig", "RandomConfig", "OptunaConfig"} <= names
    assert discovered_registry.get("QCScorer").category == "Scorer"
    assert discovered_registry.get("GridConfig").category == "Strategy"


def test_abstract_bases_are_not_registered(discovered_registry):
    """The walk excludes the base it filters on, for the new families too."""
    names = set(discovered_registry.get_all())
    assert "Scorer" not in names
    assert "StrategyConfig" not in names
    assert "PrefabPipeline" not in names


def test_the_pre_existing_categories_are_untouched(discovered_registry):
    """Rewiring ``discover`` must not drop anything it already found.

    One representative per category, not per *sounding* category:
    ``EdgeCorrector`` is the ``analysis`` edge-correction class (category
    ``Edge Correction``), so with it as the only C-name the eight-operation
    ``Corrector`` family had no representative at all — and dropping
    ``phenotypic.correction`` from discovery passed the whole suite.
    """
    registry = discovered_registry
    names = set(registry.get_all())
    assert {
        "BlurGauss",  # Enhancer      <- phenotypic.enhance
        "OtsuDetector",  # Detector   <- phenotypic.detect
        "MeasureSize",  # Measure     <- phenotypic.measure
        "RemoveGridOutliers",  # Refiner  <- phenotypic.refine
        "GridApply",  # Grid          <- phenotypic.grid
        "CropImage",  # Corrector     <- phenotypic.correction
        "AppendString",  # Post       <- phenotypic.post
        "EdgeCorrector",  # Edge Correction <- phenotypic.analysis
        "LogGrowthModel",  # Model    <- phenotypic.analysis
        "MADOutlierRemover",  # Filter <- phenotypic.analysis
        "GridOccupancy",  # quality_check <- phenotypic.analysis
    } <= names

    for name, category in (
        ("GridApply", "Grid"),
        ("CropImage", "Corrector"),
        ("EdgeCorrector", "Edge Correction"),
    ):
        assert registry.get(name).category == category, name


def test_every_walked_module_contributes_at_least_one_class(discovered_registry):
    """Self-maintaining twin of the named list above.

    Named representatives go stale as classes move; this asserts the
    property that actually matters — no module in ``_discovery_targets``
    walks to nothing — so a target dropped from the map fails here even if
    nobody remembers to add a name.
    """
    defining_modules = {info.module for info in discovered_registry.get_all().values()}
    for module_name in discovered_registry._discovery_targets():
        assert any(
            defined == module_name or defined.startswith(f"{module_name}.")
            for defined in defining_modules
        ), f"{module_name} is a discovery target but registered nothing"


def test_a_duplicate_name_resolves_the_way_the_loader_resolves_it():
    """First match wins, because that is what the pipeline loader does.

    Unfalsifiable against the shipped modules — there are no duplicate
    names — so the two competing exports are built here. Last-match
    registration would have the catalog describe the *second* class while
    ``ImagePipeline.from_json`` deserializes the first.
    """
    import types

    from phenotypic._services.registry import OperationRegistry
    from phenotypic.abc_ import ObjectDetector
    from phenotypic.detect import OtsuDetector, TriangleDetector

    first = types.ModuleType("fake_first")
    first.Clashing = type("Clashing", (OtsuDetector,), {"__module__": "fake_first"})
    second = types.ModuleType("fake_second")
    second.Clashing = type("Clashing", (TriangleDetector,), {"__module__": "fake_second"})

    registry = OperationRegistry()
    registry._discover_from_module(first, "Detector", ObjectDetector)
    registry._discover_from_module(second, "Detector", ObjectDetector)

    assert registry.get("Clashing").cls is first.Clashing
    assert [op.name for op in registry.get_by_category("Detector")] == ["Clashing"], (
        "the loser must not linger in the category listing either"
    )


def test_no_class_name_resolves_to_two_classes():
    """The catalog and the pipeline loader must not disagree on a name.

    The loader takes the **first** module in ``PHENOTYPIC_CLASS_MODULES``
    that exports a name; registration is first-match for the same reason.
    There are no duplicates today — this keeps a new export from
    introducing one unnoticed, which is the only way the two surfaces can
    resolve one name to two different classes.
    """
    import importlib

    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        PHENOTYPIC_CLASS_MODULES,
    )
    from phenotypic._services.registry import OperationRegistry

    registry = OperationRegistry()
    targets = registry._discovery_targets()

    exporters: dict[str, set[type]] = {}
    for module_name in PHENOTYPIC_CLASS_MODULES:
        if module_name not in targets:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name, obj in registry._iter_public_classes(module):
            exporters.setdefault(name, set()).add(obj)

    clashes = {name: classes for name, classes in exporters.items() if len(classes) > 1}
    assert not clashes, f"one name, two classes: {sorted(clashes)}"


def test_discover_derives_from_the_shared_constant(monkeypatch):
    """Emptying the constant must empty the catalog.

    If ``discover`` kept its own hard-coded module walk — the state this
    task exists to end — discovery would sail on regardless.
    """
    from phenotypic._services.registry import OperationRegistry

    monkeypatch.setattr(_serializable_pipeline, "PHENOTYPIC_CLASS_MODULES", ())
    registry = OperationRegistry()
    registry.discover()
    assert registry.get_all() == {}


def test_an_unimportable_module_is_recorded_not_fatal(monkeypatch):
    """An absent optional dependency degrades the catalog, never breaks it."""
    from phenotypic._services import registry as registry_mod
    from phenotypic._services.registry import OperationRegistry

    real_import = registry_mod.importlib.import_module

    def _fail_on_nn(name, *args, **kwargs):
        if name == "phenotypic.detect.nn":
            raise ImportError("no module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(registry_mod.importlib, "import_module", _fail_on_nn)
    registry = OperationRegistry()
    registry.discover()

    assert "BlurGauss" in registry.get_all(), "one bad module must not empty the rest"
    assert "phenotypic.detect.nn" in registry.skipped_imports
    assert "torch" in registry.skipped_imports["phenotypic.detect.nn"]


# --------------------------------------------------------------------------
# Task 10c: lazily-exported classes are catalog citizens too.
# --------------------------------------------------------------------------


def test_lazy_detect_nn_detectors_are_discoverable(discovered_registry):
    """``detect.nn`` exports through ``__getattr__`` with no ``__dir__``.

    ``inspect.getmembers`` reads ``dir(module)``, so an eager-only walk
    finds nothing there and the entire staged-GPU path stays invisible.
    """
    names = set(discovered_registry.get_all())
    assert "MicroSamDetector" in names, "detect.nn unreachable — staged GPU is invisible"
    assert {"Sam2", "Sam3", "DinoSam2Detector"} <= names
    assert discovered_registry.get("MicroSamDetector").category == "Detector"


def test_the_eager_walk_alone_would_not_find_them():
    """Pin *why* the ``__all__`` walk exists, so nobody simplifies it away."""
    import inspect

    import phenotypic.detect.nn as nn_module

    eager = {name for name, _ in inspect.getmembers(nn_module, inspect.isclass)}
    assert "MicroSamDetector" not in eager
    assert "MicroSamDetector" in nn_module.__all__


def test_a_failing_lazy_export_is_guarded_at_getattr_time(monkeypatch):
    """The heavy import fires on ``getattr``, not on importing the module.

    A ``try/except`` around ``import_module`` therefore never sees it; the
    guard has to sit around the attribute access.
    """
    from phenotypic._services.registry import OperationRegistry

    import phenotypic.detect.nn as nn_module

    real_getattr = nn_module.__getattr__

    def _fail_on_microsam(name):
        if name == "MicroSamDetector":
            raise ImportError("no module named 'micro_sam'")
        return real_getattr(name)

    # The lazy loader never binds the class onto the module, so patching
    # ``__getattr__`` is enough — there is no cached attribute shadowing it.
    assert "MicroSamDetector" not in vars(nn_module)
    monkeypatch.setattr(nn_module, "__getattr__", _fail_on_microsam)

    registry = OperationRegistry()
    registry.discover()

    names = set(registry.get_all())
    assert "MicroSamDetector" not in names
    assert "Sam2" in names, "one unavailable export must not sink its siblings"
    assert "micro_sam" in registry.skipped_imports["phenotypic.detect.nn.MicroSamDetector"]


def test_non_class_all_entries_are_ignored(discovered_registry):
    """``detect.nn.__all__`` also carries the ``*_AVAILABLE`` booleans."""
    names = set(discovered_registry.get_all())
    assert "SAM2_AVAILABLE" not in names
    assert "MICROSAM_AVAILABLE" not in names


def test_a_broken_tune_install_costs_only_the_tuning_categories(monkeypatch):
    """Naming the tuning bases must not couple the GUI's catalog to ``tune``.

    ``Scorer`` and ``StrategyConfig`` are the only base classes resolved
    outside ``phenotypic.abc_``, and they are resolved before the per-module
    guard in ``discover`` can help — an unguarded import there would take
    the whole registry down with them.
    """
    import sys

    from phenotypic._services.registry import OperationRegistry

    monkeypatch.setitem(sys.modules, "phenotypic.tune.score", None)

    registry = OperationRegistry()
    registry.discover()

    names = set(registry.get_all())
    assert "QCScorer" not in names
    assert "GridConfig" not in names
    assert {"BlurGauss", "OtsuDetector", "FilamentousFungiPipeline"} <= names
    assert "phenotypic.tune" in registry.skipped_imports
