"""Unit tests for ``phenotypic.gui.shell._session``.

Covers the lifecycle contract:

    * lazy build (``build`` not called at construction)
    * idempotent release (re-release is a no-op)
    * ``touch`` updates ``idle_seconds`` without forcing build
    * ``release`` runs ``teardown`` exactly once with the live state
    * ``get`` is thread-safe under concurrent first-access
    * idle-release daemon thread releases stale sessions and exits cleanly
"""
from __future__ import annotations

import threading
import time
from typing import List

from phenotypic.gui.shell._session import (
    ToolSession,
    start_idle_release_thread,
)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_build_is_lazy() -> None:
    calls: List[int] = []

    def _build() -> str:
        calls.append(1)
        return "state"

    session: ToolSession[str] = ToolSession("t", build=_build)
    assert calls == []
    assert session.is_built() is False

    assert session.get() == "state"
    assert calls == [1]
    assert session.is_built() is True

    # Subsequent gets do not rebuild.
    assert session.get() == "state"
    assert calls == [1]


def test_release_is_idempotent() -> None:
    teardowns: List[str] = []

    session: ToolSession[str] = ToolSession(
        "t",
        build=lambda: "state",
        teardown=teardowns.append,
    )
    session.get()
    session.release()
    session.release()  # no-op
    assert teardowns == ["state"]


def test_release_then_get_rebuilds() -> None:
    counter = {"n": 0}

    def _build() -> int:
        counter["n"] += 1
        return counter["n"]

    session: ToolSession[int] = ToolSession("t", build=_build)
    assert session.get() == 1
    session.release()
    assert session.get() == 2  # rebuilt


def test_teardown_exception_does_not_block_release() -> None:
    """A misbehaving teardown must not leave the session half-released."""
    def _bad_teardown(_: str) -> None:
        raise RuntimeError("boom")

    session: ToolSession[str] = ToolSession(
        "t",
        build=lambda: "state",
        teardown=_bad_teardown,
    )
    session.get()
    session.release()  # logs + swallows the RuntimeError
    assert session.is_built() is False
    # Next get rebuilds successfully.
    assert session.get() == "state"


# ---------------------------------------------------------------------------
# Touch + idle_seconds
# ---------------------------------------------------------------------------

def test_idle_seconds_zero_when_unbuilt() -> None:
    session: ToolSession[str] = ToolSession("t", build=lambda: "state")
    assert session.idle_seconds() == 0.0


def test_touch_resets_idle_without_building() -> None:
    calls: List[int] = []

    def _build() -> str:
        calls.append(1)
        return "state"

    session: ToolSession[str] = ToolSession("t", build=_build)
    # Touch on an unbuilt session is a no-op as far as build goes; idle stays 0.
    session.touch()
    assert calls == []
    assert session.idle_seconds() == 0.0

    session.get()
    time.sleep(0.05)
    pre = session.idle_seconds()
    assert pre > 0.0
    session.touch()
    post = session.idle_seconds()
    assert post < pre


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_get_runs_build_once() -> None:
    """Many threads racing on first ``get()`` only call build once."""
    builds: List[int] = []
    barrier = threading.Barrier(8)

    def _build() -> str:
        builds.append(1)
        return "state"

    session: ToolSession[str] = ToolSession("t", build=_build)
    results: List[str] = []
    lock = threading.Lock()

    def _worker() -> None:
        barrier.wait()
        out = session.get()
        with lock:
            results.append(out)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == ["state"] * 8
    assert builds == [1]


def test_release_does_not_collide_with_get() -> None:
    """``release`` cannot tear down a state mid-``get`` rebuild.

    We force the build to be slow so a concurrent release races against the
    in-flight build. The lock must serialise: the get either returns a
    fully-built state, or rebuilds after the release dropped the state.
    Either way, no caller observes a torn-down half-state.
    """
    build_started = threading.Event()
    build_continue = threading.Event()
    builds: List[int] = []
    teardowns: List[str] = []

    def _build() -> str:
        builds.append(1)
        build_started.set()
        build_continue.wait(timeout=2.0)
        return "state"

    session: ToolSession[str] = ToolSession(
        "t", build=_build, teardown=teardowns.append,
    )

    got: List[str] = []

    def _getter() -> None:
        got.append(session.get())

    t = threading.Thread(target=_getter)
    t.start()
    build_started.wait(timeout=2.0)
    # Another thread tries to release while build is still running. The
    # release must block on the lock until build finishes, then run
    # teardown on the just-built state.
    release_thread = threading.Thread(target=session.release)
    release_thread.start()
    # Let the build complete.
    build_continue.set()

    t.join(timeout=2.0)
    release_thread.join(timeout=2.0)

    assert got == ["state"]
    assert teardowns == ["state"]
    assert session.is_built() is False


# ---------------------------------------------------------------------------
# Idle release daemon
# ---------------------------------------------------------------------------

def test_idle_thread_releases_stale_session() -> None:
    teardowns: List[str] = []

    session: ToolSession[str] = ToolSession(
        "t",
        build=lambda: "state",
        teardown=teardowns.append,
    )
    session.get()  # mark built; idle clock starts now

    stop = threading.Event()
    sessions: list[ToolSession[object]] = [session]  # type: ignore[list-item]
    thread = start_idle_release_thread(
        sessions,
        idle_release_seconds=0.05,
        poll_interval_seconds=0.02,
        stop_event=stop,
    )
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and session.is_built():
            time.sleep(0.02)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    assert session.is_built() is False
    assert teardowns == ["state"]


def test_idle_thread_skips_unbuilt_sessions() -> None:
    """Daemon must not call build to check idleness."""
    builds: List[int] = []

    def _build() -> str:
        builds.append(1)
        return "state"

    session: ToolSession[str] = ToolSession("t", build=_build)

    stop = threading.Event()
    sessions: list[ToolSession[object]] = [session]  # type: ignore[list-item]
    thread = start_idle_release_thread(
        sessions,
        idle_release_seconds=0.01,
        poll_interval_seconds=0.01,
        stop_event=stop,
    )
    try:
        time.sleep(0.05)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    # The daemon polled multiple times, but never forced a build.
    assert builds == []
    assert session.is_built() is False


def test_idle_thread_respects_touch() -> None:
    """``touch`` must keep the session alive past the idle threshold."""
    session: ToolSession[str] = ToolSession("t", build=lambda: "state")
    session.get()

    stop = threading.Event()
    sessions: list[ToolSession[object]] = [session]  # type: ignore[list-item]
    thread = start_idle_release_thread(
        sessions,
        idle_release_seconds=0.05,
        poll_interval_seconds=0.01,
        stop_event=stop,
    )
    try:
        # Touch repeatedly for longer than the idle threshold.
        for _ in range(15):
            session.touch()
            time.sleep(0.01)
        # Session should still be built — touch kept resetting the clock.
        assert session.is_built() is True
    finally:
        stop.set()
        thread.join(timeout=2.0)


def test_idle_thread_exits_on_stop_event() -> None:
    session: ToolSession[str] = ToolSession("t", build=lambda: "state")
    stop = threading.Event()
    sessions: list[ToolSession[object]] = [session]  # type: ignore[list-item]
    thread = start_idle_release_thread(
        sessions,
        idle_release_seconds=10.0,
        poll_interval_seconds=0.5,
        stop_event=stop,
    )
    stop.set()
    thread.join(timeout=2.0)
    assert thread.is_alive() is False


def test_idle_thread_does_not_release_during_active_get() -> None:
    """Regression for the get() ordering race.

    Before the fix, ``get()`` set ``_state`` first and ``_last_access`` second,
    leaving a window where the lock-free idle daemon could observe
    ``is_built() == True`` paired with ``_last_access == 0.0`` (stale) and
    schedule a release on a session that was just built. The fix sets
    ``_last_access`` *first*, under the lock; this test pins that ordering.

    On the buggy code, the daemon polled hundreds of times while ``slow_build``
    sat in ``Event.wait``, eventually taking the lock and tearing down the
    state the moment the build finished — so ``is_built()`` is False after
    the getter returned.
    """
    proceed = threading.Event()
    teardowns: List[str] = []

    def slow_build() -> str:
        proceed.wait(timeout=2.0)
        return "state"

    session: ToolSession[str] = ToolSession(
        "t", build=slow_build, teardown=teardowns.append,
    )

    stop = threading.Event()
    sessions: list[ToolSession[object]] = [session]  # type: ignore[list-item]
    thread = start_idle_release_thread(
        sessions,
        idle_release_seconds=0.0,
        poll_interval_seconds=0.001,
        stop_event=stop,
    )
    try:
        getter_result: List[str] = []

        def _getter() -> None:
            getter_result.append(session.get())

        getter = threading.Thread(target=_getter)
        getter.start()
        # Let the daemon poll many times while the build is still in flight.
        time.sleep(0.05)
        proceed.set()
        getter.join(timeout=2.0)

        # Right after get() returns, the session must still hold the state
        # the caller just received. ``release()`` may run on a *later* poll
        # (idle_seconds=0, so we expect that), but the immediate post-get
        # invariant is what protects callers from racing rebuilds.
        assert getter_result == ["state"]
        assert teardowns == [], (
            "daemon released a session whose build had just completed; "
            "the get()-vs-daemon ordering race regressed"
        )
    finally:
        stop.set()
        thread.join(timeout=2.0)


def test_touch_is_lock_free() -> None:
    """``touch`` must not contend on the session lock.

    If a future refactor accidentally puts ``touch`` under ``self._lock``,
    iframe polling would block whenever the lock is held by ``get()`` or
    ``release()`` — exactly the case the lock-free design avoids.
    """
    session: ToolSession[str] = ToolSession("t", build=lambda: "state")
    session._lock.acquire()  # type: ignore[attr-defined]
    try:
        start = time.monotonic()
        session.touch()
        elapsed = time.monotonic() - start
        # Generous bound: the call should be ~microseconds. Allow 50 ms
        # for slow CI runners. A buggy locked impl would block forever.
        assert elapsed < 0.05
    finally:
        session._lock.release()  # type: ignore[attr-defined]


def test_idle_thread_observes_session_list_mutations() -> None:
    """Daemon snapshots ``sessions`` per iteration; appends are observed.

    The implementation does ``for session in list(sessions):`` so the caller
    can register new sessions after launch and the daemon picks them up on
    the next poll without restart.
    """
    sessions: list[ToolSession[object]] = []
    stop = threading.Event()
    thread = start_idle_release_thread(
        sessions,
        idle_release_seconds=0.01,
        poll_interval_seconds=0.005,
        stop_event=stop,
    )
    try:
        teardowns: List[str] = []
        late = ToolSession[str](
            "late", build=lambda: "state", teardown=teardowns.append,
        )
        late.get()  # mark built; idle clock starts
        sessions.append(late)  # type: ignore[arg-type]

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and late.is_built():
            time.sleep(0.005)
        assert late.is_built() is False
        assert teardowns == ["state"]
    finally:
        stop.set()
        thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# set_build (viewer hand-off)
# ---------------------------------------------------------------------------

def test_set_build_swaps_build_and_releases_state() -> None:
    """set_build must drop existing state and install the new build."""
    teardowns: List[str] = []
    session: ToolSession[str] = ToolSession(
        "viewer",
        build=lambda: "first",
        teardown=teardowns.append,
    )
    assert session.get() == "first"

    session.set_build(lambda: "second")
    # Existing state was released, including teardown.
    assert teardowns == ["first"]
    assert session.is_built() is False

    # Next get uses the new build.
    assert session.get() == "second"
    assert session.is_built() is True


def test_set_build_with_no_state_is_safe() -> None:
    """Calling set_build before any get must not invoke teardown."""
    teardowns: List[str] = []
    session: ToolSession[str] = ToolSession(
        "viewer",
        build=lambda: "first",
        teardown=teardowns.append,
    )
    session.set_build(lambda: "second")
    assert teardowns == []
    assert session.get() == "second"
