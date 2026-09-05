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

    This pins a documented behavioural contract, not a numeric one. Callers
    mutate the returned object in place (``rgb_to_xyz`` assigns
    ``.whitepoint``), and the eager module-level constant they were written
    against was shared process-wide; anything reading that attribute outside the
    mutating call sees the last writer's value, and that stays true only while
    every access yields one object.

    Be precise about what this is *not*: rebuilding per access does not change
    what ``rgb_to_xyz`` returns. Measured, not assumed -- simulating a
    per-access rebuild moves the output by exactly 0.0, because the function
    imports the colourspace once per call and then mutates and reads it inside
    that same call, so two objects never straddle the mutation. An earlier
    version of this docstring claimed otherwise.
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
            if any(
                library == allowed_lib and relative.startswith(prefix)
                for prefix, allowed_lib in surface.MODULE_SCOPE_ALLOWED_PREFIXES
            ):
                continue
            if surface.imports_at_module_scope(path, library):
                offenders.append(f"{relative} imports {library}")

    assert not offenders, (
        "these files import a deferred library at module scope:\n  "
        + "\n  ".join(offenders)
        + "\nDefer the import into the function that uses it, or add the file to "
        "MODULE_SCOPE_IMPORT_ALLOWLIST with a reason."
    )


# ---------------------------------------------------------------------------
# Numeric anchors -- what the shared mutable colourspace actually computes
# ---------------------------------------------------------------------------


def test_rgb_to_xyz_matches_the_pre_refactor_numbers():
    """Pin the conversion output against the source as it was before deferral.

    The golden was captured by running the *pre-refactor* source out of the
    pinned base worktree, so it is an outside witness rather than a photograph
    of current behaviour. It catches a real change to what the conversion
    computes -- a wrong whitepoint, a dropped cctf, a transposed matrix.

    **Compared at a tolerance, not bit-exactly.** A sha256 of the raw float64
    bytes was tried first and failed in CI: the goldens were captured on an
    AMD EPYC 7713 and shard 17 ran on an Intel Xeon Gold 5220R, whose BLAS
    kernels differ in the last ulp. Measured across all four combinations, the
    worst cross-node relative difference is 3.4e-16, against a float64 epsilon
    of 2.2e-16 -- roughly 1.5 ulp. ``rtol`` below is ~3000x that noise floor and
    still far tighter than any real algorithm change.

    **What this does NOT guard, measured rather than assumed.** It does not
    detect ``sRGB_D50`` being rebuilt per access. That was the stated reason for
    adding it; the claim was wrong. ``rgb_to_xyz`` imports the colourspace once
    per call, then mutates ``.whitepoint`` and reads it back inside that same
    call, so two different objects can never straddle the mutation -- simulating
    a per-access rebuild (and a non-default observer) changes the output by
    exactly 0.0. The ``lru_cache`` governs object identity and allocation, not
    these numbers, and
    :func:`test_the_lazily_built_colourspace_is_one_shared_object` is what pins
    the sharing contract.
    """
    import json

    import numpy as np

    from phenotypic._core._image_parts.color_space_accessors._xyz_conversion import (
        rgb_to_xyz,
    )
    from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS

    golden = json.loads(
        (surface.GOLDEN_DIR / "rgb_to_xyz.json").read_text(encoding="utf-8")
    )
    shape = tuple(golden["shape"])
    rgb = np.linspace(0.0, 1.0, int(np.prod(shape)), dtype=np.float64).reshape(shape)
    observer = "CIE 1931 2 Degree Standard Observer"

    # Twice per combination: the in-place whitepoint assignment must leave the
    # shared colourspace in a state the next call reproduces exactly.
    for _ in range(2):
        for gamma in (GAMMA_ENCODINGS.SRGB, GAMMA_ENCODINGS.LINEAR):
            for illuminant in ("D50", "D65"):
                key = f"{gamma.name}:{illuminant}"
                out = np.asarray(
                    rgb_to_xyz(
                        rgb, gamma=gamma, illuminant=illuminant, observer=observer
                    )
                ).ravel()
                np.testing.assert_allclose(
                    out,
                    np.array(golden["values"][key]),
                    rtol=1e-12,
                    atol=1e-15,
                    err_msg=f"{key} no longer matches the pre-refactor conversion",
                )


def test_the_colour_plotting_stub_swaps_itself_out_on_a_real_attribute():
    """The stub's other half: it must actually forward, and install the real module.

    Three tests covered install, dunder-decline and no-op-when-late; none ever
    triggered ``__getattr__`` on a real name, which is the path that does the
    swap. Leaving half of it uncovered is an odd gap in the one mechanism the
    whole refactor imitates.
    """
    script = """
import json, sys
import phenotypic  # noqa: F401  -- installs the stub

stub = sys.modules["colour.plotting"]
assert getattr(stub, "__phenotypic_lazy_stub__", False)

value = stub.plot_chromaticity_diagram_CIE1931  # a real plotting attribute
real = sys.modules["colour.plotting"]
import colour

print(json.dumps({
    "forwarded": value is not None,
    "swapped_in": not getattr(real, "__phenotypic_lazy_stub__", False),
    "parent_repointed": colour.plotting is real,
}))
"""
    result = surface._probe(script)
    assert result["forwarded"], "the stub did not forward a real attribute"
    assert result["swapped_in"], "sys.modules still holds the stub after a real access"
    assert result["parent_repointed"], "colour.plotting was not repointed on the parent"


def test_a_failing_swap_in_leaves_the_stub_installed():
    """If the real import raises, the stub must stay — not vanish.

    It pops itself from ``sys.modules`` before importing. Without a rollback a
    failed import leaves ``colour.plotting`` absent entirely, so the *second*
    access raises a different and more confusing error than the first, and the
    laziness is silently gone for the rest of the process.
    """
    script = """
import importlib, json, sys
import phenotypic  # noqa: F401

stub = sys.modules["colour.plotting"]


def boom(name):
    raise ImportError("simulated matplotlib backend failure")


importlib.import_module = boom
raised = False
try:
    stub.plot_chromaticity_diagram_CIE1931
except ImportError:
    raised = True

print(json.dumps({
    "raised": raised,
    "stub_still_installed": sys.modules.get("colour.plotting") is stub,
}))
"""
    result = surface._probe(script)
    assert result["raised"], "the failing import should propagate"
    assert result["stub_still_installed"], (
        "a failed swap-in removed the stub from sys.modules; the next access "
        "would fail differently and laziness would be gone"
    )


def test_public_util_annotations_still_resolve_at_runtime():
    """Deferring a library must not break runtime annotation resolution.

    ``util/_measurement_outputs.py`` annotates its public functions with
    ``MeasurementFrame``, a union over pandas and polars. Moving polars under
    ``TYPE_CHECKING`` first left the runtime alias as the *string*
    ``"pd.DataFrame | pl.DataFrame"``, which made
    ``typing.get_type_hints(split_measurements)`` raise
    ``NameError: name 'pl' is not defined`` — where the pre-refactor source
    returned the resolved union. Sphinx autodoc resolves annotations at runtime,
    so this would have surfaced as a docs-build failure, not a test failure.

    Deliberately in-process rather than in a subprocess. The concern would be
    that another test importing polars could rescue this one, making it the same
    order-dependent false green as the theme test — but checked, and it cannot:
    ``get_type_hints`` resolves names in the *defining module's* globals, and
    ``pl`` is never bound there at runtime, so the failure is unconditional. The
    broken alias raises with polars already imported.
    """
    import typing

    from phenotypic.util import generate_output_key, split_measurements

    for function in (split_measurements, generate_output_key):
        hints = typing.get_type_hints(function)
        assert "df" in hints, f"{function.__name__} lost its parameter annotation"


def test_the_slurm_worker_still_gets_a_headless_backend():
    """`matplotlib.use("Agg")` must still run before anything imports pyplot.

    `_cli_process_single` sets the non-interactive backend at module scope. That
    is load-bearing: a SLURM worker has no display, and a pyplot import that
    picks an interactive backend first would fail or hang. Deferring the worker
    import out of `_cli_execution_strategies` (so `--help` stops paying 274 ms
    for matplotlib) moves *when* that runs, so this pins that it still runs
    early enough.

    The order asserted is the real one: import the CLI, then the worker, then
    pyplot -- and matplotlib must not have been loaded before the worker, or the
    backend selection would be racing whatever loaded it.
    """
    script = """
import json, sys

import phenotypic.phenotypicCLI  # noqa: F401
before_worker = "matplotlib" in sys.modules

import phenotypic._cli._cli_process_single  # noqa: F401  -- calls use("Agg")
import matplotlib

after_worker = matplotlib.get_backend()

import matplotlib.pyplot  # noqa: F401

print(json.dumps({
    "matplotlib_loaded_before_worker": before_worker,
    "backend_after_worker": after_worker,
    "backend_after_pyplot": matplotlib.get_backend(),
}))
"""
    result = surface._probe(script)
    assert not result["matplotlib_loaded_before_worker"], (
        "matplotlib was already imported before the worker set its backend"
    )
    assert result["backend_after_worker"].lower() == "agg", (
        f"worker left the backend at {result['backend_after_worker']!r}, not Agg"
    )
    assert result["backend_after_pyplot"].lower() == "agg", (
        "importing pyplot changed the backend away from Agg"
    )
