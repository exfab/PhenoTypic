"""``LocalRunner`` — ``Popen`` + deque ring buffer + SIGTERM-on-stop (Phase 4).

Phase 0 placeholder — implementation lands in Phase 4. See ``GUI_SPEC_V1.md``
section 5.

Per plan: ``Popen`` with stdout pipe, daemon thread tees to
``<output_dir>/.gui_log/stdout.log`` and appends to
``collections.deque(maxlen=5000)`` under a ``buffer_lock`` (NOT
``queue.Queue`` — a bounded queue with blocking ``put()`` would back-fill the
subprocess's stdout pipe and can deadlock the run). ``stop()`` sends SIGTERM
→ SIGKILL after 10s. ``atexit`` hook SIGTERMs all live handles.
"""
from __future__ import annotations

# TODO(Phase 4): LocalRunner class + _atexit_cleanup hook.

__all__: list[str] = []
