"""Import custom operation modules named by ``PHENOTYPIC_PRELOAD_MODULES``.

A pipeline JSON records each operation by bare class name, and
``ImagePipeline.from_json`` resolves names only inside the ``phenotypic``
namespace. A custom operation defined elsewhere therefore resolves only after a
module has been imported that attaches the class to that namespace (a
*self-registering* module, e.g. ``tests/_fakes/register_fake_gpu.py``).

Lives in ``sdk_`` rather than ``_cli`` because class resolution in ``_core``
calls it (``_find_class_in_phenotypic``), and ``_core`` may not import
``_cli``. ``phenotypic._cli._cli_preload`` re-exports it for existing callers.
"""

from __future__ import annotations

import importlib
import os

PRELOAD_MODULES_ENV = "PHENOTYPIC_PRELOAD_MODULES"


def preload_module_names() -> tuple[str, ...]:
    """The module names ``PHENOTYPIC_PRELOAD_MODULES`` lists, in order.

    Empty comma-separated entries and surrounding whitespace are ignored.
    """
    return tuple(
        name
        for name in (
            value.strip()
            for value in os.environ.get(PRELOAD_MODULES_ENV, "").split(",")
        )
        if name
    )


def preload_custom_operation_modules() -> None:
    """Import every module ``PHENOTYPIC_PRELOAD_MODULES`` names.

    Idempotent, because ``importlib.import_module`` returns an already-imported
    module from ``sys.modules``. Import failures intentionally propagate so a
    process reports the missing registration module instead of a later, less
    specific pipeline deserialization error.
    """
    for module_name in preload_module_names():
        importlib.import_module(module_name)
