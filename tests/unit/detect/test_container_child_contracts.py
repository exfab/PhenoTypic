"""Behavioural probes for every ``"parallel"`` entry in ``_CHILD_CONTRACT``.

The table in ``_cli_validation`` says what each container hands its children,
and ``_branch_prefix`` computes the Stage-2 prefix from that answer. A table
entry that is merely *stated* can be wrong and still pass every test that reads
the table. A probe cannot: it puts two recording operations in the container's
children and asserts the second did **not** observe the first's output.

Coverage of the table itself is asserted in
``tests/unit/cli/test_gpu_detection_tree_wide.py`` -- the set is closed by rule,
so anything absent from it is refused by default. This file covers the other
half: that each entry present says the truth.
"""

import ast
import inspect
import threading

import numpy as np
import pytest

from phenotypic._cli import _cli_validation
from phenotypic._cli._cli_validation import (
    _CHILD_CONTRACT,
    UnstageableGpuDetectorError,
    _child_contract,
    _populate_child_contract,
)
from phenotypic.abc_ import ImageEnhancer, ObjectDetector
from phenotypic.abc_.plotting import PlotImage
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import CompositeDetector
from phenotypic.enhance import CompositeEnhance


#: What a probe stamps: ONE square, so a chained second branch would observe an
#: objmap max of exactly 1 -- a value nothing else in this fixture produces.
_PROBE_OBJECT_COUNT = 1


class _DetectProbe(ObjectDetector):
    """Record what this branch was handed, then write a mask of its own.

    The mask is a single deliberate square rather than a threshold of the
    image: a threshold on the synthetic plate can itself yield ~96 objects,
    which is the same number the fixture arrives with, so it would not
    discriminate chaining from parallelism.
    """

    tag: str

    def _operate(self, image):
        _SEEN.append((self.tag, int(image.objmap[:].max())))
        mask = np.zeros(image.gray[:].shape, dtype=bool)
        mask[10:30, 10:30] = True
        image.objmask[:] = mask
        return image


class _EnhanceProbe(ImageEnhancer):
    """Record the incoming ``detect_mat``, then overwrite it distinctively."""

    tag: str
    fill: float

    def _operate(self, image):
        _SEEN.append((self.tag, float(image.detect_mat[:].mean())))
        image.detect_mat[:] = self.fill  # scalar broadcast; detect_mat is [0,1]
        return image


_SEEN: list = []


@pytest.fixture(autouse=True)
def _clear_probe_log():
    _SEEN.clear()
    yield
    _SEEN.clear()


def test_every_parallel_entry_in_the_table_has_a_probe_here():
    """A new ``"parallel"`` entry must arrive with its own probe.

    Without this, someone adds a third composite to the table and the probes
    silently cover only the two that came before it.
    """
    _populate_child_contract()
    probed = {CompositeDetector, CompositeEnhance}
    parallel = {
        cls for cls, contract in _CHILD_CONTRACT.items() if contract == "parallel"
    }
    assert parallel == probed


def test_composite_detector_branches_each_receive_the_composites_own_input():
    """Assert the PROPERTY, not a literal.

    ``load_synth_yeast_plate()`` arrives with its 8x12 ground-truth objmap
    already populated, so the value a branch observes on entry is 96, not 0. A
    test hard-coding 0 would only pass on a cleared image -- it would be
    testing the fixture, and a spike on this change was derailed by exactly
    this number once already. What the contract actually claims is that every
    branch sees *the composite's own input*, whatever that happens to be.
    """
    image = load_synth_yeast_plate()
    baseline = int(image.objmap[:].max())
    assert baseline != _PROBE_OBJECT_COUNT, (
        "fixture collision: the plate's own object count equals what a probe "
        "writes, so this test could no longer tell chaining from parallelism"
    )

    CompositeDetector(
        ops=[_DetectProbe(tag="a"), _DetectProbe(tag="b")], mode="union"
    ).apply(image)

    # Sequential branches would have "b" observing "a"'s objmap, i.e. 1.
    assert _SEEN == [("a", baseline), ("b", baseline)]


def test_composite_enhance_branches_each_receive_the_composites_own_input():
    image = load_synth_yeast_plate()
    baseline = float(image.detect_mat[:].mean())
    fill_a = 0.25
    assert baseline != fill_a, "fixture collision: see the detector probe above"

    CompositeEnhance(
        ops=[
            _EnhanceProbe(tag="a", fill=fill_a),
            _EnhanceProbe(tag="b", fill=0.75),
        ],
        mode="max",
    ).apply(image)

    # "b" must see the composite's own input, not "a"'s 0.25 fill.
    assert _SEEN == [("a", baseline), ("b", baseline)]


# --------------------------------------------------------------------------
# Subclass matching: accepted by isinstance, refused on an `_operate` override
# --------------------------------------------------------------------------


class _PlottedComposite(CompositeDetector, PlotImage):
    """A mixin that adds plotting and does not touch ``_operate``."""


class _ChainingComposite(CompositeDetector):
    """A subclass that REPLACES ``_operate``, verified against nothing."""

    def _operate(self, image):  # pragma: no cover - never executed
        return image


def test_a_mixin_subclass_of_a_listed_container_is_accepted():
    """An exact-type lookup refused this, and a user really does hold it.

    Same reasoning the table's own note gives for keeping ``ImagePipeline``
    out: matched by subclass, not by exact type.
    """
    assert _child_contract(_PlottedComposite()) == "parallel"


def test_a_subclass_that_overrides_operate_is_refused():
    """A bare ``isinstance`` would admit it.

    The contract was verified against ``CompositeDetector._operate``. A
    subclass replacing it could chain its branches, and nothing here has
    checked that it does not -- which is the exact class of error the
    "composition primitives only" narrowing exists to prevent.
    """
    with pytest.raises(UnstageableGpuDetectorError, match="overrides _operate"):
        _child_contract(_ChainingComposite())


def test_the_override_refusal_names_the_base_it_could_not_inherit_from():
    """The message's job is to tell the user what to use instead."""
    with pytest.raises(UnstageableGpuDetectorError) as exc:
        _child_contract(_ChainingComposite())
    assert "CompositeDetector" in str(exc.value)


def test_an_unrelated_container_still_gets_the_composition_primitives_refusal():
    """The subclass branch must not swallow the default refusal."""
    from phenotypic.detect import FilamentousFungiDetector

    with pytest.raises(
        UnstageableGpuDetectorError, match="only composition primitives"
    ):
        _child_contract(FilamentousFungiDetector())


# --------------------------------------------------------------------------
# The table is installed in ONE operation, so no thread sees it half-built
# --------------------------------------------------------------------------


def test_the_table_is_installed_in_a_single_operation():
    """Structural, and deterministic — this is the guard that matters.

    ``_populate_child_contract``'s early-out is ``if _CHILD_CONTRACT: return``.
    Filling the table one key at a time made a window in which that guard is
    already truthy while the table holds one entry, so a second thread returns
    early and ``_child_contract`` then refuses a ``CompositeEnhance`` — a
    placement the design permits. The GUI reaches here on threaded Werkzeug and
    already swallows ``ValueError``, so the symptom is not an error: the run
    silently routes to CPU.

    A "call it twice and check the table" test passes on **both** shapes, and a
    threaded test is timing-dependent. Asserting the shape of the population is
    neither: this fails on the old sequential-``__setitem__`` code and cannot
    flake.
    """
    tree = ast.parse(inspect.getsource(_cli_validation))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_populate_child_contract"
    )

    subscript_writes = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript) for target in node.targets)
    ]
    assert not subscript_writes, (
        "_CHILD_CONTRACT is populated key-by-key again. A thread arriving "
        "between the writes sees a truthy partial table, returns at the "
        "`if _CHILD_CONTRACT` guard, and gets a wrong refusal. Build the dict "
        "literal first, then install it with a single .update()."
    )

    updates = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
    ]
    assert len(updates) == 1, "expected exactly one atomic .update() install"


def test_concurrent_first_population_gives_every_thread_the_right_answer():
    """Behavioural companion to the structural guard above.

    Both threads race an unpopulated table from a cold start. This passes on
    the fixed shape always; on the old shape it fails only when the window is
    hit, which is why it is the *companion* and not the guard.
    """
    _CHILD_CONTRACT.clear()

    barrier = threading.Barrier(2)
    results: dict[str, object] = {}

    def ask(name, container):
        barrier.wait()
        try:
            results[name] = _child_contract(container)
        except Exception as exc:  # noqa: BLE001 - recording, not handling
            results[name] = type(exc).__name__

    threads = [
        threading.Thread(target=ask, args=("detector", CompositeDetector())),
        threading.Thread(target=ask, args=("enhance", CompositeEnhance())),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == {"detector": "parallel", "enhance": "parallel"}
