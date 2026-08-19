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
    """Rewiring ``discover`` must not drop anything it already found."""
    names = set(discovered_registry.get_all())
    assert {"BlurGauss", "OtsuDetector", "MeasureSize", "EdgeCorrector"} <= names


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

