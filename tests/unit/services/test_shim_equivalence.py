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


def test_run_registry_is_one_class():
    from phenotypic._services.runs import RunRegistry as canonical
    from phenotypic.gui.shell._runs_registry import RunRegistry as shim

    assert shim is canonical


def test_local_runner_is_one_class():
    from phenotypic._services.runs import LocalRunner as canonical
    from phenotypic.gui.run_console._runner import LocalRunner as shim

    assert shim is canonical


def test_runs_shims_reexport_every_public_name():
    """Both originals' full ``__all__`` must survive the merge into one module."""
    from phenotypic import _services
    from phenotypic.gui.run_console import _runner as runner_shim
    from phenotypic.gui.shell import _runs_registry as registry_shim

    for name in (
        "RunMode",
        "RunStatus",
        "RunRecord",
        "RunRegistry",
        "run_status_is_nonterminal",
    ):
        assert getattr(registry_shim, name) is getattr(_services.runs, name), name

    for name in ("LocalRunHandle", "LocalRunner"):
        assert getattr(runner_shim, name) is getattr(_services.runs, name), name


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


def test_export_is_one_function():
    from phenotypic._services.tune_spec import export_best_from_run as canonical
    from phenotypic.gui.tune._export import export_best_from_run as shim

    assert shim is canonical


def test_export_shim_reexports_every_public_name():
    """The shim surface is derived from the imports the repo actually makes.

    ``_callbacks.py`` and ``tests/unit/gui/tune/test_export.py`` between them
    pull all five functions plus ``PreparedPipelineExport``; the private
    ``_params_from_best_params_payload`` travels too so the shim stays a
    complete view of the original module rather than of its ``__all__``.
    """
    from phenotypic._services import tune_spec
    from phenotypic.gui.tune import _export as shim

    for name in (
        "PreparedPipelineExport",
        "_params_from_best_params_payload",
        "export_best_from_run",
        "export_pareto_pipeline",
        "export_winning_pipeline",
        "prepare_best_from_run",
        "publish_prepared_export",
    ):
        assert getattr(shim, name) is getattr(tune_spec, name), name


def test_command_is_one_function():
    from phenotypic._services.tune_spec import build_tune_command as canonical
    from phenotypic.gui.tune._command import build_tune_command as shim

    assert shim is canonical


def test_command_shim_reexports_every_public_name():
    """``_layout.py:38``, ``_launch.py:26`` and ``_callbacks.py:55`` import these."""
    from phenotypic._services import tune_spec
    from phenotypic.gui.tune import _command as shim

    for name in (
        "DEFAULT_STORAGE_ENV",
        "ExecutionTarget",
        "StorageMode",
        "ValidatedTuneCommand",
        "build_tune_command",
        "render_launch_command",
        "render_tokens",
        "storage_url_preflight_issue",
    ):
        assert getattr(shim, name) is getattr(tune_spec, name), name


def test_validation_shim_reexports_every_public_name():
    """``_callbacks.py:87`` and ``test_validation.py`` import through here."""
    from phenotypic._services import tune_spec
    from phenotypic.gui.tune import _validation as shim

    for name in (
        "Blocks",
        "Issue",
        "can_deploy",
        "preflight_issues",
        "spec_path_issue",
        "validate_setup",
    ):
        assert getattr(shim, name) is getattr(tune_spec, name), name


def test_setup_authoring_shim_reexports_every_imported_name():
    """Derived from the imports, not from the old ``__all__``.

    ``write_setup_draft`` and ``build_authored_setup_spec`` reach this module
    only through multi-line parenthesised imports in
    ``tests/integration/gui/tune/test_setup_view.py`` — the shape that produced
    the X2 and X3 incidents.
    """
    from phenotypic._services import tune_spec
    from phenotypic.gui.tune import _setup_authoring as shim

    for name in (
        "SETUP_DRAFT_VERSION",
        "SetupAuthoringResult",
        "SetupDraft",
        "SetupDraftCache",
        "SetupPathPayload",
        "SetupPathResolution",
        "SetupWriteReceipt",
        "authored_content_fingerprint",
        "authored_setup_spec_path",
        "build_authored_setup_spec",
        "build_setup_draft",
        "load_pipeline_or_spec",
        "path_content_fingerprint",
        "resolve_picker_payload",
        "resolve_setup_path",
        "setup_draft_from_store",
        "setup_path_payload",
        "setup_path_resolution_from_store",
        "write_authored_setup_spec",
        "write_setup_draft",
        "write_setup_draft_receipt",
    ):
        assert getattr(shim, name) is getattr(tune_spec, name), name


def test_grid_feasibility_is_one_function():
    """Promoted out of ``_domain_editor`` so ``_validation`` could travel."""
    from phenotypic._services.tune_spec import grid_feasibility as canonical
    from phenotypic.gui.tune._domain_editor import grid_feasibility as shim

    assert shim is canonical


def test_sandbox_fingerprint_is_one_function():
    """Promoted beside the ``SandboxRoot`` it hashes."""
    from phenotypic._services.sandbox import sandbox_fingerprint as canonical
    from phenotypic.gui.shell._source_context import sandbox_fingerprint as shim

    assert shim is canonical


def test_tune_presets_dir_is_one_function_and_three_constants():
    """``gui/_config`` re-exports all four rather than defining them.

    Task 2's ``IMAGE_EXTS`` move repeated for the tune presets path, so the six
    other ``SANDBOX_GUI_DIRNAME`` consumers are untouched by the relocation.

    **Identity alone was a false green here** — measured, not assumed. Three of
    the four are short string literals, which CPython interns, so a parallel
    ``SANDBOX_PRESETS_SUBDIR = "presets"`` added to ``_config`` satisfies ``is``
    while being a second definition free to drift.

    So the binding itself is asserted, from the parsed module: each name must
    arrive by an ``ImportFrom`` of ``phenotypic.sdk_._io_constants``, and must
    not be bound by any module-level assignment, ``def``, or ``class``. Both are
    AST facts about how the module is written, never a substring of it.
    """
    import ast
    import inspect

    from phenotypic.gui import _config
    from phenotypic.sdk_ import _io_constants

    tree = ast.parse(inspect.getsource(_config))
    locally_defined: set[str] = set()
    reexported: set[str] = set()
    for node in tree.body:  # module level only; a nested def is not a rebinding
        if isinstance(node, ast.Assign):
            locally_defined.update(
                t.id for t in node.targets if isinstance(t, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            locally_defined.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            locally_defined.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "phenotypic.sdk_._io_constants":
                reexported.update(alias.name for alias in node.names)

    for name in (
        "SANDBOX_GUI_DIRNAME",
        "SANDBOX_PRESETS_SUBDIR",
        "SANDBOX_TUNE_PRESETS_SUBDIR",
        "tune_presets_dir",
    ):
        assert getattr(_config, name) is getattr(_io_constants, name), name
        assert name in reexported, f"{name} is not re-exported from _io_constants"
        assert name not in locally_defined, (
            f"gui/_config defines {name} in parallel with _io_constants; "
            "interned string literals make this invisible to an identity check"
        )


def test_tune_presets_dir_still_resolves_the_same_path():
    """The relocation is a move, not a redefinition."""
    from pathlib import Path

    from phenotypic.sdk_._io_constants import tune_presets_dir

    assert tune_presets_dir(Path("/sandbox")) == Path(
        "/sandbox/.phenotypic-gui/presets/tune"
    )
