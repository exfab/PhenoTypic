"""Each gui.* shim must re-export the same object, not a parallel one."""

from __future__ import annotations


def test_get_registry_is_one_function():
    from phenotypic._services.registry import get_registry as canonical
    from phenotypic.gui._operation_registry import get_registry as shim

    assert shim is canonical


def test_get_registry_is_one_singleton():
    from phenotypic._services.registry import get_registry as canonical
    from phenotypic.gui._operation_registry import get_registry as shim

    assert shim() is canonical()


def test_registry_shim_reexports_every_public_name():
    """Five names are imported from the shim path across the repo.

    ``ColumnRefSpec`` is the one the plan's shim sketch omitted;
    ``tests/unit/gui/test_param_forms.py`` imports it six times.
    """
    from phenotypic import _services
    from phenotypic.gui import _operation_registry as shim

    for name in (
        "ColumnRefSpec",
        "OperationInfo",
        "OperationRegistry",
        "ParamInfo",
        "get_registry",
    ):
        assert getattr(shim, name) is getattr(_services.registry, name), name


def test_sandbox_root_is_one_class():
    from phenotypic._services.sandbox import SandboxRoot as canonical
    from phenotypic.gui.shell._sandbox import SandboxRoot as shim

    assert shim is canonical


def test_sandbox_shim_reexports_the_privates_two_modules_import():
    """``_setup_authoring.py:22`` and ``_source_context.py:24`` import these."""
    from phenotypic import _services
    from phenotypic.gui.shell import _sandbox as shim

    for name in (
        "SandboxRoot",
        "_is_safe_relative_path",
        "_v1_selection_matches_sandbox",
    ):
        assert getattr(shim, name) is getattr(_services.sandbox, name), name


def test_discovery_stays_lazy():
    """Importing the module must not walk eight packages.

    Probed in a subprocess rather than via ``importlib.reload``. Reloading
    rebinds this module's classes and functions while the shim keeps
    references to the originals, so ``shim.OperationRegistry is
    registry.OperationRegistry`` becomes False for the rest of the session —
    measured, not assumed. That silently breaks the identity invariant the
    other tests in this file assert.
    """
    import subprocess
    import sys

    probe = (
        "import phenotypic._services.registry as r; print(r._REGISTRY is None)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True", (
        "importing _services.registry eagerly built the registry"
    )
