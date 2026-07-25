"""Per-page generation fencing for replaceable Dash sub-apps."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from flask import g, jsonify, request
from dash.types import RendererHooks

from phenotypic.gui._config import CFG_BINDING_GENERATION

if TYPE_CHECKING:
    import dash

__all__ = [
    "BINDING_GENERATION_PAYLOAD_KEY",
    "BindingFenceTimeoutError",
    "BindingRequestFence",
    "binding_generation_hooks",
    "install_bound_output_callback_guard",
    "install_binding_generation_guard",
]

BINDING_GENERATION_PAYLOAD_KEY = "__phenotypic_binding_generation"
_DASH_CALLBACK_PATH = "/_dash-update-component"
_FENCE_G_KEY = "_phenotypic_binding_fence_entered"


class BindingFenceTimeoutError(TimeoutError):
    """Raised when an old binding cannot drain within its publication budget."""


class BindingRequestFence:
    """Close one binding to new callbacks and drain admitted requests."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._accepting = True
        self._active_requests = 0

    def try_enter(self) -> bool:
        """Admit one callback unless publication has closed this binding."""
        with self._condition:
            if not self._accepting:
                return False
            self._active_requests += 1
            return True

    def leave(self) -> None:
        """Release one admitted callback."""
        with self._condition:
            if self._active_requests <= 0:
                raise RuntimeError("binding request fence leave without enter")
            self._active_requests -= 1
            if self._active_requests == 0:
                self._condition.notify_all()

    def close_and_wait(self, *, timeout_seconds: float) -> None:
        """Reject new callbacks and wait boundedly for admitted callbacks.

        Args:
            timeout_seconds: Positive maximum drain duration.

        Raises:
            ValueError: If ``timeout_seconds`` is not positive.
            BindingFenceTimeoutError: If a callback remains active at timeout.
        """
        if timeout_seconds <= 0:
            raise ValueError("binding fence timeout must be positive")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            self._accepting = False
            while self._active_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BindingFenceTimeoutError(
                        "Timed out waiting for the previous Results/Analysis "
                        "binding to finish active callbacks."
                    )
                self._condition.wait(timeout=remaining)

    def reopen(self) -> None:
        """Re-admit callbacks after a failed publication rollback."""
        with self._condition:
            self._accepting = True
            self._condition.notify_all()

    @property
    def active_requests(self) -> int:
        """Return the number of currently admitted callback requests."""
        with self._condition:
            return self._active_requests


def binding_generation_hooks(generation: str | None) -> RendererHooks | None:
    """Return Dash's public request hook for one bound page generation."""
    if generation is None:
        return None
    return {
        "request_pre": (
            "function(payload) {"
            f'payload["{BINDING_GENERATION_PAYLOAD_KEY}"] = '
            "window.__phenotypicBindingGeneration;"
            "}"
        )
    }


def install_bound_output_callback_guard(
    app: "dash.Dash",
    *,
    mutation_is_safe: Callable[[], bool],
    status_output_id: str,
) -> None:
    """Reject non-status callbacks when the bound output is unsafe to mutate."""

    @app.server.before_request
    def _reject_unsafe_bound_output_callback() -> Any:
        if request.path.rstrip("/") != _DASH_CALLBACK_PATH:
            return None
        if mutation_is_safe():
            return None
        payload = request.get_json(silent=True)
        output = payload.get("output") if isinstance(payload, dict) else None
        if isinstance(output, str) and status_output_id in output:
            return None
        return (
            jsonify(
                {
                    "status": "bound_output_read_only",
                    "error": (
                        "This output is active or changed on disk. Mutation "
                        "callbacks are disabled until one stable terminal "
                        "snapshot is refreshed."
                    ),
                }
            ),
            423,
        )


def install_binding_generation_guard(
    app: "dash.Dash",
    generation: str | None,
    fence: BindingRequestFence | None = None,
) -> None:
    """Reject callbacks issued by a page from another binding generation.

    The shell replaces complete Dash applications when Results/Analysis is
    rebound. A browser tab can retain the old page and otherwise send its
    callback payload to the newly published app. Bound pages therefore add
    their generation to every Dash callback transport request, and the
    destination app checks it before Dash dispatch can invoke a callback.

    Standalone apps pass ``None`` and retain their normal callback transport.

    Args:
        app: Dash app whose Flask server receives callback requests.
        generation: Immutable shell binding UUID, or ``None`` outside the
            replaceable hub sessions.
        fence: Shared Results/Analysis callback fence for this binding.
    """
    if generation is None:
        return
    app.server.config[CFG_BINDING_GENERATION] = generation
    request_fence = fence if fence is not None else BindingRequestFence()

    def _leave_request_fence_if_entered() -> None:
        if getattr(g, _FENCE_G_KEY, False):
            setattr(g, _FENCE_G_KEY, False)
            request_fence.leave()

    @app.server.before_request
    def _reject_stale_binding_callback() -> Any:
        if request.path.rstrip("/") != _DASH_CALLBACK_PATH:
            return None
        payload = request.get_json(silent=True)
        supplied = (
            payload.get(BINDING_GENERATION_PAYLOAD_KEY)
            if isinstance(payload, dict)
            else None
        )
        if supplied == generation:
            if request_fence.try_enter():
                setattr(g, _FENCE_G_KEY, True)
                return None
            return (
                jsonify(
                    {
                        "status": "stale_binding",
                        "error": (
                            "This output binding is being replaced. Reload "
                            "before retrying."
                        ),
                    }
                ),
                409,
            )
        return (
            jsonify(
                {
                    "status": "stale_binding",
                    "error": (
                        "This page belongs to an older output binding. "
                        "Reload before retrying."
                    ),
                    "expected_generation": generation,
                }
            ),
            409,
        )

    @app.server.after_request
    def _leave_binding_callback(response: Any) -> Any:
        _leave_request_fence_if_entered()
        return response

    @app.server.teardown_request
    def _leave_failed_binding_callback(_error: BaseException | None) -> None:
        _leave_request_fence_if_entered()
