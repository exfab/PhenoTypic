"""The tune-GUI lazy-import lock — optuna stays out of the import surface.

The ``/tune/`` co-pilot reads a study only through the parquet journal at
import / build time; the live ``OptunaStudyStore`` is opened lazily INSIDE the
Monitor poll callback (gated on the ``tune`` extra). So ``import
phenotypic.gui.tune`` — which pulls the Dash factory, the layout, and the
callbacks module — must NEVER drag ``optuna`` into ``sys.modules``. This test
mirrors :mod:`tests.unit.tune.test_lazy_import_lock` and is re-run in every
later gate.
"""
from __future__ import annotations

import importlib
import sys


def test_import_tune_gui_does_not_import_optuna() -> None:
    sys.modules.pop("optuna", None)
    importlib.import_module("phenotypic.gui.tune")
    # The factory, layout, study-read helpers, and callbacks are all reachable
    # from the package import; none may pull optuna.
    importlib.import_module("phenotypic.gui.tune._app")
    importlib.import_module("phenotypic.gui.tune._layout")
    importlib.import_module("phenotypic.gui.tune._callbacks")
    importlib.import_module("phenotypic.gui.tune._study_read")
    assert "optuna" not in sys.modules, (
        "importing phenotypic.gui.tune must not import optuna"
    )
