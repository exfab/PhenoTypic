"""``/sandbox/api/*`` JSON blueprint (Phase 2).

Phase 0 placeholder — implementation lands in Phase 2. See ``GUI_SPEC_V1.md``
section 3.

Per plan: Flask blueprints (NOT Dash callbacks) so a future ``--mode=cloud``
auth gate can attach a single ``@before_request`` hook. Routes:
``/sandbox/api/root``, ``/sandbox/api/children``, ``/sandbox/api/classify``.
Registered on ``shell_app.server`` directly — NOT under
``DispatcherMiddleware``.
"""
from __future__ import annotations

# TODO(Phase 2): ``register_sandbox_api(server, sandbox)`` builder.

__all__: list[str] = []
