"""The import-laziness lock -- the refactor's standing invariant.

Deferring an ``import`` from module scope into a function body cannot be caught
by an ordinary test: the library still loads the first time anything calls the
function, so every test passes whether the deferral worked or not. Equally, a
lazy ``__getattr__`` that returns a *different* object than the eager import
did, or that quietly stops serving a name, passes everything that does not
happen to touch that name.

These tests close both holes against goldens captured before any import moved
(``tests/fixtures/import_laziness/``, written by
``scripts/capture_import_laziness_goldens.py``). Re-run them at every later
gate, as ``tests/unit/tune/test_lazy_import_lock.py`` does for optuna.

When a stage legitimately changes what is eager, re-capture the goldens and
commit the diff -- do not relax an assertion.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests import _import_surface as surface
from tests import _legacy_store_digest as legacy


@pytest.fixture(scope="module")
def golden() -> dict:
    """The committed import-surface golden."""
    return surface.load_golden()


@pytest.fixture(scope="module")
def measured_exports() -> dict:
    """Export surface measured in a clean subprocess (one probe for the module)."""
    return surface.measure_export_surface()


@pytest.fixture(scope="module")
def measured_eager() -> dict:
    """Eager ``sys.modules`` sets measured in clean subprocesses."""
    return {
        target: surface.measure_eager_imports(target)
        for target in surface.EAGER_TARGETS
    }


# ---------------------------------------------------------------------------
# Export surface -- the public contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", surface.PUBLIC_PACKAGES)
def test_public_exports_are_unchanged(package, golden, measured_exports):
    """``__all__`` must be byte-identical to the pre-refactor capture.

    ``dir()`` is deliberately not asserted: it reads ``__dict__`` and does not
    trigger PEP-562 ``__getattr__``, so it shrinks legitimately whenever a name
    becomes lazy. ``__all__`` is the contract users import against.
    """
    assert (
        measured_exports[package]["__all__"] == golden["exports"][package]["__all__"]
    ), f"{package}.__all__ changed; re-capture the golden if this is intended"


@pytest.mark.parametrize(
    "package", [name for name in surface.PUBLIC_PACKAGES if name not in surface.RESOLVE_SKIP]
)
def test_every_exported_name_resolves_to_the_same_object(
    package, golden, measured_exports
):
    """Each exported name must still resolve, and to the same object.

    This is what a lazy ``__getattr__`` can silently break: the name is still
    in ``__all__``, ``import`` still succeeds, but the attribute now yields a
    different class, a stub, or raises. Identity is recorded as
    ``kind:module:qualname``, which moves if the object does.
    """
    expected = golden["exports"][package].get("resolved")
    if expected is None:
        pytest.skip(f"{package} has no __all__")
    assert measured_exports[package]["resolved"] == expected


def test_pipeline_class_resolution_is_unchanged(golden):
    """Every operation name must still resolve through ``_find_class_in_phenotypic``.

    That resolver's first branch is ``hasattr(phenotypic, class_name)``, which
    under PEP-562 laziness works only while ``__getattr__`` raises a clean
    ``AttributeError`` for unknown names. Anything else -- notably an
    ``ImportError`` leaking out of a missing optional dependency -- breaks
    pipeline deserialization for every class that falls through to the
    submodule list.
    """
    assert surface.measure_class_resolution() == golden["class_resolution"]


# ---------------------------------------------------------------------------
# Eager imports -- the ratchet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", surface.EAGER_TARGETS)
def test_the_deferral_measurement_was_not_compromised(target, measured_eager):
    """A forbidden library preloaded before the probe voids the measurement.

    The probe reports what importing the target *added* to ``sys.modules``, so a
    library already loaded when the subprocess started is invisible to it and
    the deferral check below passes for the wrong reason. That is not
    hypothetical: pytest-cov's ``.pth`` starts coverage in every subprocess it
    spawns, which would silence the ``coverage`` entry precisely under
    ``pytest --cov`` — the configuration where you would most want it.

    Failing here means the environment, not the library, is wrong.
    """
    preloaded = measured_eager[target]["forbidden_preloaded"]
    assert not preloaded, (
        f"{preloaded} were already in sys.modules before 'import {target}' ran, "
        "so the deferral check cannot see them; the measurement is void"
    )


@pytest.mark.parametrize("target", surface.EAGER_TARGETS)
def test_forbidden_libraries_are_not_imported_eagerly(target, measured_eager):
    """Libraries a stage has already evicted must never come back.

    :data:`surface.FORBIDDEN_EAGER` starts empty and grows by one entry per
    stage. Nothing is ever removed from it.

    Asserted against the *final* ``sys.modules``, not against what the import
    added: a preload cannot hide from the final state. The companion test above
    separately rejects a compromised measurement, so a failure here is a real
    re-introduced module-scope import rather than a dirty environment.
    """
    offenders = measured_eager[target]["forbidden_in_final"]
    assert not offenders, (
        f"'import {target}' eagerly loads {offenders}, which an earlier stage "
        "deferred; something re-introduced a module-scope import"
    )


@pytest.mark.parametrize("target", surface.EAGER_TARGETS)
def test_the_eager_module_set_never_grows(target, golden, measured_eager):
    """A ratchet, not an equality: shrinking is the goal, growing is a regression.

    Asserting equality would fail on every stage that succeeds. Asserting an
    upper bound catches a newly added module-scope import without standing in
    the way of the refactor.
    """
    measured = measured_eager[target]
    expected = golden["eager"][target]

    # A subset, not a count. A change that drops five eager modules and adds
    # five different ones leaves the count identical while quietly making
    # something new eager -- and the golden already stores the full list, so
    # the stronger check costs nothing.
    newly_eager = sorted(
        set(measured["phenotypic_modules"]) - set(expected["phenotypic_modules"])
    )
    assert not newly_eager, (
        f"'import {target}' newly loads {len(newly_eager)} phenotypic modules "
        f"at import time: {newly_eager[:10]}"
    )
    new_roots = sorted(
        set(measured["third_party_roots"]) - set(expected["third_party_roots"])
    )
    assert not new_roots, f"'import {target}' newly loads {new_roots} at import time"


def test_module_scope_imports_stay_deferred():
    """Static backstop for files that are off the eager path today.

    The runtime probe above only sees what ``import phenotypic`` actually
    reaches. A module-scope import in a file that nothing currently imports
    eagerly would pass it, then start costing on the day something imports that
    file. Walking only the AST's top-level statements is what makes
    "imported at runtime" precise -- ``if TYPE_CHECKING:`` blocks and
    in-function imports are correctly ignored.
    """
    import phenotypic

    src = Path(phenotypic.__file__).parent
    offenders = [
        (relative, target)
        for relative, target in surface.FORBIDDEN_MODULE_SCOPE_IMPORTS
        if surface.imports_at_module_scope(src / relative, target)
    ]
    assert not offenders, f"module-scope imports that must be deferred: {offenders}"


# ---------------------------------------------------------------------------
# Legacy migration -- the h5py anchor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", legacy.LAYOUTS)
def test_legacy_migration_output_is_unchanged(layout):
    """Converting a committed legacy layout must produce the same store.

    The existing migration tests are differential -- migrated store versus
    freshly written store, or versus the source ``.h5`` -- so a defect present
    on both sides of the comparison passes them. This is an independent anchor:
    structure, pixel content hashes and metadata, all captured before any
    import moved.
    """
    expected = legacy.load_golden()["layouts"][layout]
    with tempfile.TemporaryDirectory() as scratch:
        measured = legacy.migrate_and_digest(
            layout, Path(scratch) / "img.ome.zarr"
        )

    assert measured["paths"] == expected["paths"], "store layout changed"
    assert measured["series"] == expected["series"], "pixel content changed"
    # Labels are hashed separately from series because they are not a series:
    # `objmap` lives under `rgb/labels/`, so wrong object labels would leave
    # paths, series and attributes all identical.
    assert measured["labels"] == expected["labels"], "label content changed"
    assert measured["attributes"] == expected["attributes"], "store metadata changed"


# ---------------------------------------------------------------------------
# Deferred constants must keep their identity
# ---------------------------------------------------------------------------


def test_the_lazily_built_colourspace_is_one_shared_object():
    """``sRGB_D50`` must be the same object on every access, as it was eagerly.

    This is not a caching nicety. ``rgb_to_xyz`` assigns
    ``sRGB_D50.whitepoint = ...`` and then passes ``sRGB_D50.whitepoint`` into
    the conversion, so a ``__getattr__`` that built a fresh colourspace per
    access would silently change what every D50 conversion in the library
    computes -- while every existing test still passed, because each call would
    mutate and read its own private copy and look self-consistent.
    """
    from phenotypic.sdk_ import colourspace

    first = colourspace.sRGB_D50
    second = colourspace.sRGB_D50
    assert first is second

    # And the mutation the conversion path performs must be visible to the next
    # reader, which is only true while they are the same object. `whitepoint` is
    # a validated property on colour's RGB_Colourspace, so the probe has to be a
    # real chromaticity pair rather than a sentinel.
    import numpy as np

    original = np.array(first.whitepoint, copy=True)
    probe = original + 0.01
    try:
        first.whitepoint = probe
        assert np.allclose(colourspace.sRGB_D50.whitepoint, probe)
    finally:
        first.whitepoint = original


def test_the_lazily_built_correction_constants_are_shared_too():
    """The two colour constants inside colour-correction share the same rule."""
    from phenotypic.correction._color_correction import _color_checker_profile, _helpers

    assert _helpers._srgb_cs() is _helpers._srgb_cs()
    assert (
        _color_checker_profile._illuminant_xy_table()
        is _color_checker_profile._illuminant_xy_table()
    )


# ---------------------------------------------------------------------------
# _startup_perf -- the existing lazy stub this refactor generalizes
# ---------------------------------------------------------------------------


def test_colour_plotting_stub_is_installed_and_stays_lazy():
    """``import phenotypic`` must not pull in ``colour.plotting``.

    ``colour``'s own ``__init__`` does ``from colour import plotting``, which
    drags matplotlib and the ASTM G-173 spectral datasets. ``_startup_perf``
    pre-registers a lazy stand-in so that binding resolves to a stub instead.
    Nothing covered this before, which is uncomfortable for the one mechanism
    the whole refactor imitates.
    """
    script = """
import json, sys
import phenotypic  # noqa: F401

module = sys.modules.get("colour.plotting")
print(json.dumps({
    "present": module is not None,
    "is_stub": getattr(module, "__phenotypic_lazy_stub__", False),
}))
"""
    result = surface._probe(script)
    assert result["present"], "colour.plotting missing from sys.modules entirely"
    assert result["is_stub"], (
        "colour.plotting was genuinely imported; the lazy stub did not take effect"
    )


def test_the_stub_declines_dunder_lookups_without_importing_colour_plotting():
    """A dunder probe must not detonate the real import.

    ``inspect.getmodule`` asks for ``__file__`` while formatting unrelated
    tracebacks. Forwarding that would import the real ``colour.plotting`` at the
    worst possible moment -- during error formatting -- and in a venv where
    colour is broken it would surface a confusing secondary exception.
    """
    script = """
import json, sys
import phenotypic  # noqa: F401  -- installs the stub

stub = sys.modules["colour.plotting"]
raised = False
try:
    stub.__file__
except AttributeError:
    raised = True

print(json.dumps({
    "raised": raised,
    "still_stub": getattr(
        sys.modules["colour.plotting"], "__phenotypic_lazy_stub__", False
    ),
    "matplotlib_pyplot_loaded_by_colour": "matplotlib.pyplot" in sys.modules,
}))
"""
    result = surface._probe(script)
    assert result["raised"], "the stub forwarded a dunder lookup instead of declining"
    assert result["still_stub"], (
        "the dunder probe swapped in the real colour.plotting"
    )


def test_installing_the_stub_is_a_no_op_once_colour_is_imported():
    """Importing phenotypic *after* colour must not disturb the real module."""
    script = """
import json
import colour  # noqa: F401  -- imported first, on purpose
from phenotypic._startup_perf import install_lazy_colour_plotting

print(json.dumps({"installed": install_lazy_colour_plotting()}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=surface.REPO_ROOT,
    )
    import json

    assert json.loads(completed.stdout.strip().splitlines()[-1])["installed"] is False


def test_no_new_file_imports_a_deferred_library_at_module_scope():
    """Scan every source file, not just the ones a stage happened to touch.

    :data:`surface.FORBIDDEN_MODULE_SCOPE_IMPORTS` is a hand-maintained list, so
    it is blind to a *new* file. Concretely: add
    ``correction/_color_correction/_swatch_locator.py`` with ``import colour`` at
    module scope, imported only from inside a corrector's ``_operate``. The
    runtime probe passes (colour still is not eager), the static per-file check
    passes (the file is not in the list), the golden passes (nothing changed).
    Months later someone hoists that import into ``correction/__init__.py``,
    colour is back on every CLI startup, and nothing said a word at any point.

    Scanning everything and allowing three named exceptions inverts the default:
    a new offender fails immediately, and adding it to the allowlist is a
    deliberate act with a reason attached.
    """
    import phenotypic

    src = Path(phenotypic.__file__).parent
    offenders = []
    for path in sorted(src.rglob("*.py")):
        relative = path.relative_to(src).as_posix()
        for library in surface.FORBIDDEN_EAGER:
            if (relative, library) in surface.MODULE_SCOPE_IMPORT_ALLOWLIST:
                continue
            if surface.imports_at_module_scope(path, library):
                offenders.append(f"{relative} imports {library}")

    assert not offenders, (
        "these files import a deferred library at module scope:\n  "
        + "\n  ".join(offenders)
        + "\nDefer the import into the function that uses it, or add the file to "
        "MODULE_SCOPE_IMPORT_ALLOWLIST with a reason."
    )
