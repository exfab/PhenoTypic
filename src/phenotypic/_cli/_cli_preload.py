"""Preload custom operation modules before pipeline deserialization."""

from __future__ import annotations

import importlib
import os

_PRELOAD_MODULES_ENV = "PHENOTYPIC_PRELOAD_MODULES"


def preload_custom_operation_modules() -> None:
    """Import custom operation modules named by the preload environment.

    Empty comma-separated entries and surrounding whitespace are ignored.
    Import failures intentionally propagate so a remote process reports the
    missing registration module instead of a later, less specific pipeline
    deserialization error.
    """
    for value in os.environ.get(_PRELOAD_MODULES_ENV, "").split(","):
        module_name = value.strip()
        if module_name:
            importlib.import_module(module_name)
