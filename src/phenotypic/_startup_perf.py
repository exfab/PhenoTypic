"""Process-wide startup optimizations applied before the heavy submodule import.

This module is imported at the very top of :mod:`phenotypic` — before any
submodule (``abc_``, ``_core``, ``correction`` …) pulls in third-party
scientific libraries — so the tweaks here run exactly once, ahead of the
expensive import chain.

The single optimization today is :func:`install_lazy_colour_plotting`.
``import colour`` (colour-science) is one of the largest costs on the import
path, and ``-X importtime`` profiling shows ~90% of that is colour's own
``__init__`` eagerly doing ``from colour import plotting`` (line 72 of
``colour/__init__.py``), which drags in matplotlib *and* the ASTM G-173
spectral datasets. PhenoTypic never uses ``colour.plotting`` (verified by a
repo-wide grep), so loading it is pure startup tax paid by every entry point
(CLI, tune, GUI launcher).

We avoid it without monkeypatching colour's source or its numeric behaviour:
a **lazy** stand-in module is pre-registered in :data:`sys.modules` so colour's
``from colour import plotting`` binds the stub instead of importing the real
thing. If any caller ever *does* touch a ``colour.plotting`` attribute, the
stub transparently imports the genuine submodule on first access and forwards
to it — so this is an optimization, not a feature removal.
"""
from __future__ import annotations

import importlib
import sys
import time
import types

#: ``perf_counter`` stamped the moment this module is first imported. Because
#: :mod:`phenotypic` imports this module before anything else, it marks the
#: start of the (heavy) library import — the only point early enough to measure
#: it, since the console-script import chain runs before any ``main()``.
IMPORT_STARTED_AT: float = time.perf_counter()

__all__ = ["IMPORT_STARTED_AT", "install_lazy_colour_plotting"]


class _LazyColourPlotting(types.ModuleType):
    """Stand-in for ``colour.plotting`` that loads the real module on demand.

    Pre-registered in :data:`sys.modules` so colour's eager
    ``from colour import plotting`` binds this instead of importing
    matplotlib + the spectral datasets. The first genuine attribute access
    swaps in the real submodule and forwards the lookup, so behaviour is
    unchanged for the (currently nonexistent) callers that need plotting.
    """

    #: Marker so tests / introspection can recognise the stub.
    __phenotypic_lazy_stub__ = True

    def __getattr__(self, name: str):
        # Introspection probes (e.g. inspect.getmodule asking for ``__file__``
        # while formatting an unrelated traceback) must not force the heavy
        # import — only genuine plotting attribute access should. Forwarding a
        # dunder lookup would detonate the real ``colour.plotting`` import even
        # in venvs where colour is broken, surfacing a confusing secondary
        # error during traceback formatting.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # First real use: replace ourselves with the genuine submodule.
        sys.modules.pop("colour.plotting", None)
        real = importlib.import_module("colour.plotting")
        colour = sys.modules.get("colour")
        if colour is not None:
            setattr(colour, "plotting", real)  # repoint parent-package attr
        return getattr(real, name)


def install_lazy_colour_plotting() -> bool:
    """Pre-register a lazy ``colour.plotting`` stub to skip its eager load.

    No-op (returns ``False``) when it is already too late to help — i.e.
    ``colour`` or ``colour.plotting`` is already imported — so importing
    :mod:`phenotypic` after colour never disturbs an existing import.

    Returns:
        ``True`` if the lazy stub was installed, ``False`` if skipped
        because colour was already (partially) imported.
    """
    if "colour" in sys.modules or "colour.plotting" in sys.modules:
        return False
    sys.modules["colour.plotting"] = _LazyColourPlotting("colour.plotting")
    return True


# Apply the optimization as an import side effect so :mod:`phenotypic` only
# needs a single (E402-clean) ``from ._startup_perf import …`` line ahead of
# its heavy submodule chain, rather than a bare function-call statement.
install_lazy_colour_plotting()
