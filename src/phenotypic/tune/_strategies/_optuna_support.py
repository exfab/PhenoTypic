"""The Optuna dependency boundary — lazy import behind the ``tune`` extra.

``import optuna`` happens **lazily inside this module's functions**, never at
package import (optuna-integration.md §10). The umbrella ``import phenotypic``
and every Grid/Random tuning path must stay Optuna-free; only an
explicitly-requested Optuna strategy (a later chunk) calls
:func:`_require_optuna`. Requesting it without the extra raises a clear,
actionable :class:`ImportError` pointing at ``uv sync --extras tune``.
"""
from __future__ import annotations

from types import ModuleType

#: The actionable message shown when the ``tune`` extra is not installed.
_MISSING_OPTUNA_MSG = (
    "Optuna is required for this strategy. Install the 'tune' extra: "
    "uv sync --extras tune"
)


def _require_optuna() -> ModuleType:
    """Import and return the ``optuna`` module, or raise an actionable error.

    The import is deliberately inside the function body so importing this
    module (and therefore ``phenotypic.tune``) never pulls in Optuna; only an
    actual call resolves the dependency.

    Returns:
        The imported ``optuna`` module.

    Raises:
        ImportError: If Optuna is not installed, with a message pointing at
            ``uv sync --extras tune``.
    """
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(_MISSING_OPTUNA_MSG) from exc
    return optuna
