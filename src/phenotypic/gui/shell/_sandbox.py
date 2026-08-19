"""Back-compat shim. Implementation in :mod:`phenotypic._services.sandbox`.

Re-exports the *same* objects, so ``SandboxRoot`` is one class no matter which
path a caller imports it through. The two private helpers travel with it
because :mod:`phenotypic.gui.tune._setup_authoring` and
:mod:`phenotypic.gui.shell._source_context` import them by name.
"""

from __future__ import annotations

from phenotypic._services.sandbox import (  # noqa: F401 - re-exported
    SandboxRoot,
    _is_safe_relative_path,
    _v1_selection_matches_sandbox,
)

__all__ = ["SandboxRoot"]
