"""``to_argv`` cannot travel without ``RunConsoleState``; neither may import Dash."""

from __future__ import annotations


def test_state_and_builder_move_together():
    from phenotypic._services.argv import RunConsoleState, to_argv

    assert to_argv.__annotations__["state"] in (
        RunConsoleState,
        "RunConsoleState",
    )


def test_shims_are_the_same_objects():
    from phenotypic._services.argv import RunConsoleState as canonical
    from phenotypic.gui.run_console._state import RunConsoleState as shim

    assert shim is canonical


def test_tune_argv_shim_is_the_same_objects():
    from phenotypic import _services
    from phenotypic.gui.tune import _run_argv as shim

    for name in ("tune_run_argv", "tune_run_argv_from_tail", "tune_run_tail"):
        assert getattr(shim, name) is getattr(_services.argv, name), name


def test_state_shim_reexports_every_public_name():
    """``_state.__all__`` must keep resolving through the shim."""
    from phenotypic.gui.run_console import _state

    for name in (
        "RunConsoleState",
        "run_state_to_json",
        "run_state_from_json",
        "state_from_controls",
        "to_argv",
    ):
        assert getattr(_state, name) is not None, name


def test_argv_module_does_not_import_gui():
    """No import in this module may reach up into ``phenotypic.gui``.

    Checked against the parsed import statements rather than a substring of
    the source: a raw ``"phenotypic.gui" not in source`` also matches prose in
    a docstring (it does — this module's docstring explains *why* it must not
    import the GUI), while missing ``from phenotypic import gui``.
    """
    import ast
    import inspect

    from phenotypic._services import argv

    tree = ast.parse(inspect.getsource(argv))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.append(module)
            imported += [f"{module}.{alias.name}" for alias in node.names]

    offenders = [
        name
        for name in imported
        if name == "phenotypic.gui" or name.startswith("phenotypic.gui.")
    ]
    assert not offenders, f"_services.argv imports from the GUI: {offenders}"
