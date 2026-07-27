"""``ToolSession`` lifecycle primitive + idle-release daemon.

Each tool the shell hosts (Builder, Viewer, Run console) wraps its heavy state
in a :class:`ToolSession`. The shell drops Python references on ``release()``;
the next ``get()`` rebuilds from disk. RSS may not return to the OS — that's
documented in the Release-button tooltip — but the in-process object graph
shrinks predictably.

Threading model
    * ``get`` and ``release`` are both serialised through ``self._lock`` so an
      in-flight ``get`` cannot observe a half-torn-down state, and a
      ``release`` cannot run concurrently with ``get``'s build call.
    * ``get`` sets ``_last_access`` *before* invoking ``build``, while still
      holding the lock. This pairing prevents the idle daemon from observing
      ``_state`` set with a stale ``_last_access`` and spuriously releasing
      a session that was just built (see ``test_idle_thread_does_not_release``
      ``_during_active_get``).
    * ``touch`` skips the lock and writes a single attribute. Under CPython
      the ``STORE_ATTR`` bytecode is indivisible and Python ``float``
      objects are immutable, so a concurrent reader sees either the previous
      timestamp or the new one — never a torn value. A stale read costs at
      most one idle-poll iteration of inaccuracy.
    * ``idle_seconds`` reads ``_state`` and ``_last_access`` without the lock
      to keep the polling daemon cheap. The values may be racy but, given
      the ordering rule above, the worst case is "release fires one tick
      late," which is harmless.

Idle release uses a single background daemon thread polling ``time.monotonic()``
every poll-interval seconds (default 60). One-shot ``threading.Timer`` is
deliberately avoided: a Timer scheduled at touch-time and cancelled on every
subsequent touch races with the daemon thread that the timer might also try
to spawn. Polling is simpler and idempotent.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Callable, Generic, Iterable, TypeVar

from phenotypic.gui._config import THREAD_NAME_PREFIX

logger = logging.getLogger(__name__)

__all__ = [
    "ToolSession",
    "start_idle_release_thread",
    "swap_tool_session_states",
]

T = TypeVar("T")


class ToolSession(Generic[T]):
    """Lazy, releasable wrapper around a heavy tool's state object.

    ``ToolSession`` is generic over the state type ``T`` (e.g. ``dash.Dash``
    for the viewer, ``OutputRoot`` for measurement loading). The shell
    constructs a session at boot but does NOT call ``build`` — that happens
    on first ``get()``.

    Args:
        name: Human-readable label for log lines and ``__repr__``.
        build: Zero-argument callable that constructs the heavy state. Called
            once per release cycle, under ``self._lock``.
        teardown: Optional cleanup callable invoked with the
            (now-detached) state **after** the session has cleared its
            reference. Runs outside the session's lock; concurrent ``get``
            calls may have already triggered a rebuild before ``teardown``
            returns. Therefore ``teardown`` must only touch state owned by
            the released object, never shared global state that the rebuild
            repopulates — e.g. clearing keys on the *released*
            ``viewer_app.server.config`` is fine, but clearing keys on a
            process-wide singleton (``shell_app.server.config``) is a bug
            because the rebuild re-installs them. Defaults to a no-op.

    Example:
        >>> from phenotypic.gui.shell._session import ToolSession
        >>> def _build():
        ...     return {"big_dataframe": "loaded"}
        >>> session = ToolSession("viewer", build=_build)
        >>> session.get()  # triggers _build()
        {'big_dataframe': 'loaded'}
        >>> session.release()
        >>> session.get()  # _build() runs again
        {'big_dataframe': 'loaded'}
    """

    def __init__(
        self,
        name: str,
        *,
        build: Callable[[], T],
        teardown: Callable[[T], None] | None = None,
    ) -> None:
        self.name = name
        self._build = build
        self._teardown = teardown if teardown is not None else _noop_teardown
        self._state: T | None = None
        self._last_access: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self) -> T:
        """Return the live state, building it on first access.

        Acquires ``self._lock`` so concurrent first-access calls don't both
        run ``build``. Updates ``_last_access`` to the current monotonic
        time **before** invoking ``build`` so that the idle daemon (which
        reads ``_last_access`` lock-free) cannot observe a freshly-set
        ``_state`` paired with a stale ``_last_access`` and spuriously
        decide to release.
        """
        with self._lock:
            self._last_access = time.monotonic()
            if self._state is None:
                logger.debug("ToolSession[%s]: building", self.name)
                self._state = self._build()
            return self._state

    def touch(self) -> None:
        """Bump activity timestamp without forcing a build.

        Called by Flask blueprints (``/runs/...``, ``/sandbox/api/...``)
        whose hits indicate the user is actively using the tool. Without
        ``touch`` the idle daemon would release tools while the user is
        reading an iframed dashboard (the dashboard polls files via the
        ``/runs/`` blueprint, NOT through Dash callbacks).

        Lock-free by design — the only mutation is a 64-bit float write.
        """
        self._last_access = time.monotonic()

    def idle_seconds(self) -> float:
        """Seconds since the last ``get`` or ``touch``.

        Returns 0.0 when the session is unbuilt — there is nothing to
        release. Reading ``_state`` without the lock is intentional; a
        race only changes the answer by one polling cycle.
        """
        if self._state is None:
            return 0.0
        return time.monotonic() - self._last_access

    def is_built(self) -> bool:
        """True iff the session currently holds built state."""
        return self._state is not None

    def set_build(self, build: Callable[[], T]) -> None:
        """Replace the build callable and drop any existing state.

        Used by hand-off endpoints (e.g. the viewer's
        ``/sandbox/api/viewer/output-root``) that need the next ``get`` to
        construct the heavy state with new arguments. The swap and the
        release are performed under ``self._lock`` so an in-flight ``get``
        cannot observe a half-swapped state, and the next ``get`` is
        guaranteed to call the freshly-installed ``build``.

        ``teardown`` runs outside the lock with the previously-held state,
        following the same contract as :meth:`release`.

        Args:
            build: The new zero-argument constructor for the heavy state.
        """
        with self._lock:
            self._build = build
            stale = self._state
            self._state = None
        if stale is not None:
            try:
                self._teardown(stale)
            except Exception:
                logger.exception(
                    "ToolSession[%s]: teardown raised during set_build; "
                    "swap completed anyway",
                    self.name,
                )
            del stale
            gc.collect()
            logger.debug(
                "ToolSession[%s]: build swapped + state released", self.name
            )

    def release(self) -> None:
        """Drop the in-memory state. Idempotent.

        Runs ``teardown(state)`` first, then drops the reference and calls
        ``gc.collect()``. The next ``get`` will rebuild from scratch. Held
        under ``self._lock`` so an in-flight ``get`` cannot observe a
        torn-down state.
        """
        with self._lock:
            if self._state is None:
                return
            state = self._state
            self._state = None
        # Teardown + gc happen outside the lock. ``state`` is a local
        # reference; nothing else points at the old value once ``self._state``
        # is cleared. Holding the lock through teardown would block ``touch``
        # in the unlikely-but-possible case where teardown itself is slow.
        try:
            self._teardown(state)
        except Exception:
            logger.exception(
                "ToolSession[%s]: teardown raised; releasing anyway",
                self.name,
            )
        del state
        gc.collect()
        logger.debug("ToolSession[%s]: released", self.name)

    # ------------------------------------------------------------------
    # Dunders
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        built = "built" if self.is_built() else "empty"
        return f"<ToolSession name={self.name!r} {built}>"


def swap_tool_session_states(
    updates: Iterable[tuple["ToolSession[Any]", Any]],
    *,
    commit: Callable[[], None] | None = None,
) -> None:
    """Atomically replace a group of already-built tool states.

    Every session lock is acquired before the shared ``commit`` callback or
    any state assignment runs. A request can therefore observe the complete
    old group or the complete new group, never one updated tool paired with
    another tool's prior revision. Candidate states must be constructed
    before calling this function.

    Args:
        updates: ``(session, candidate_state)`` pairs to publish together.
        commit: Optional shared-state update executed while every session is
            locked and before candidate states become visible.

    Raises:
        ValueError: If the same session appears more than once.
        Exception: Any exception raised by ``commit``. No session state is
            changed when that happens.
    """
    pairs = list(updates)
    sessions = [session for session, _state in pairs]
    if len({id(session) for session in sessions}) != len(sessions):
        raise ValueError("each ToolSession may appear only once in a swap")

    ordered_locks = sorted(
        {id(session._lock): session._lock for session in sessions}.values(),
        key=id,
    )
    for lock in ordered_locks:
        lock.acquire()

    stale_states: list[tuple[ToolSession[Any], Any]] = []
    try:
        if commit is not None:
            commit()
        accessed_at = time.monotonic()
        for session, candidate in pairs:
            if session._state is not None:
                stale_states.append((session, session._state))
            session._state = candidate
            session._last_access = accessed_at
    finally:
        for lock in reversed(ordered_locks):
            lock.release()

    for session, stale in stale_states:
        try:
            session._teardown(stale)
        except Exception:
            logger.exception(
                "ToolSession[%s]: teardown raised after atomic swap",
                session.name,
            )
    if stale_states:
        del stale_states
        gc.collect()


def _noop_teardown(_: object) -> None:
    return None


# ---------------------------------------------------------------------------
# Idle release daemon
# ---------------------------------------------------------------------------

def start_idle_release_thread(
    sessions: "list[ToolSession[object]]",
    *,
    idle_release_seconds: float,
    poll_interval_seconds: float = 60.0,
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    """Start a daemon thread that releases idle sessions.

    The daemon walks ``sessions`` every ``poll_interval_seconds`` and calls
    ``release()`` on any session whose ``idle_seconds()`` exceeds
    ``idle_release_seconds``. The thread is marked daemon so it does not
    block process shutdown.

    Args:
        sessions: A list of :class:`ToolSession` instances to monitor. The
            list itself may be mutated by the caller — the daemon snapshots
            it on each iteration. **Caller retains ownership.**
        idle_release_seconds: Sessions whose ``idle_seconds()`` exceeds this
            threshold are released. The shell exposes this via
            ``--idle-release-minutes`` (multiplied by 60).
        poll_interval_seconds: How often the daemon wakes up. Default 60s.
            Polling is intentional — a one-shot ``threading.Timer`` races
            with ``touch()`` resets, so we trade one wake-up per minute for
            a much simpler concurrency model.
        stop_event: Optional :class:`threading.Event` that, when set,
            causes the daemon to exit. Used in tests to terminate the
            thread cleanly. In production this is typically ``None`` and
            the thread runs until process exit.

    Returns:
        The started ``threading.Thread`` instance. Caller can reach into it
        to ``.join()`` after setting ``stop_event``, but in production the
        daemon is fire-and-forget.
    """
    if stop_event is None:
        stop_event = threading.Event()

    def _loop() -> None:
        while not stop_event.is_set():
            for session in list(sessions):
                if not session.is_built():
                    continue
                if session.idle_seconds() < idle_release_seconds:
                    continue
                logger.info(
                    "ToolSession[%s]: idle for %.0fs >= %.0fs; releasing",
                    session.name,
                    session.idle_seconds(),
                    idle_release_seconds,
                )
                try:
                    session.release()
                except Exception:
                    logger.exception(
                        "ToolSession[%s]: release raised in idle daemon",
                        session.name,
                    )
            stop_event.wait(timeout=poll_interval_seconds)

    thread = threading.Thread(
        target=_loop,
        name=f"{THREAD_NAME_PREFIX}-idle-release",
        daemon=True,
    )
    thread.start()
    return thread
