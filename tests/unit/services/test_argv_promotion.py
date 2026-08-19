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
    import the GUI), while missing ``from phenotypic import gui``. ``_boundary``
    owns that walk; the tier-wide gate in ``test_import_purity.py`` runs the
    same one, and sharing it is what stops the two from drifting into
    disagreeing about what "reaches" means.
    """
    from phenotypic._services import argv

    from ._boundary import gui_modules_reached

    offenders = sorted(gui_modules_reached(argv))
    assert not offenders, f"_services.argv imports from the GUI: {offenders}"


def test_slurm_emitters_are_one_object_each():
    """The GUI submitter must not keep a parallel copy of either emitter.

    Identity, not equality: two functions that agree today drift silently, and
    the drift that matters here is invisible — spec 05 §5.4 digests the
    composed argv, so a second renderer produces a digest the server cannot
    reproduce.
    """
    from phenotypic._services.argv import (
        slurm_argv_extension,
        to_subprocess_argv,
    )
    from phenotypic.gui.run_console import _slurm

    assert _slurm._slurm_argv_extension is slurm_argv_extension
    assert _slurm._build_subprocess_argv is to_subprocess_argv


def test_the_gui_module_no_longer_defines_the_emitters_or_their_key_order():
    """AST, not identity: a rebinding to an equal object still satisfies ``is``.

    ``_SLURM_DIRECT_KEYS`` travels with the emitter that reads it. A copy left
    behind in ``gui/`` is free to gain or reorder a key, and the ordering is
    load-bearing — it fixes the order of the ``--slurm`` pairs inside
    ``argv_digest``.
    """
    import ast
    import inspect

    from phenotypic.gui.run_console import _slurm

    tree = ast.parse(inspect.getsource(_slurm))
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            defined.add(node.target.id)

    for name in (
        "_slurm_argv_extension",
        "_build_subprocess_argv",
        "_SLURM_DIRECT_KEYS",
    ):
        assert name not in defined, (
            f"gui/run_console/_slurm.py defines {name} in parallel with "
            "_services/argv.py"
        )
