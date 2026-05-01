"""Process-wide ``RunRegistry`` for live + historical pipeline runs (Phase 4).

Phase 0 placeholder — implementation lands in Phase 4. See ``GUI_SPEC_V1.md``
section 5 (Run console internals).

Per plan: ``RunRecord`` dataclass + ``RunRegistry`` with explicit
``threading.Lock`` on ``register``/``get``/``list``/``update_status`` (multiple
Dash callback threads + the runner's daemon thread mutate it concurrently).
Rehydrates from sandbox scan on boot. Stashed on
``app.server.config["runs_registry"]``.
"""
from __future__ import annotations

# TODO(Phase 4): ``RunRecord``, ``RunRegistry`` (locked), ``rehydrate_from_sandbox``.

__all__: list[str] = []
