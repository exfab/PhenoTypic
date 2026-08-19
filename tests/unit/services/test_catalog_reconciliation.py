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
