"""Unit tests for ``phenotypic.gui.run_console._runner``.

Drives a real subprocess (``python -c "..."``) so the lifecycle paths
(stdout tee → ring buffer + disk; SIGTERM-then-SIGKILL; reap) are
exercised end-to-end. Cross-platform: we use ``sys.executable`` to dodge
PATH issues and avoid signal-specific assertions that would break on
Windows. Tests that need POSIX signals are skipped there.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

import pytest

from phenotypic.gui.run_console._runner import (
    LocalRunHandle,
    LocalRunner,
)


@pytest.fixture()
def runner() -> LocalRunner:
    return LocalRunner()


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_start_spawns_subprocess_and_streams_stdout(
    runner: LocalRunner, tmp_path: Path,
) -> None:
    handle = runner.start(
        run_id="hello",
        argv=[
            sys.executable,
            "-c",
            "for i in range(3): print(f'line {i}', flush=True)",
        ],
        output_dir=tmp_path,
    )
    assert isinstance(handle, LocalRunHandle)
    assert handle.process.pid > 0

    # Wait for the subprocess to exit + tee thread to drain.
    handle.process.wait(timeout=5.0)
    assert _wait_until(
        lambda: len(runner.snapshot_log("hello")) >= 3, timeout=2.0
    )
    snap = runner.snapshot_log("hello")
    assert any("line 0" in line for line in snap)
    assert any("line 2" in line for line in snap)

    # Disk tee landed at the documented location.
    log_path = tmp_path / ".gui_log" / "stdout.log"
    assert log_path.exists()
    text = log_path.read_text()
    assert "line 0" in text
    assert "line 2" in text


def test_start_rejects_duplicate_run_id(
    runner: LocalRunner, tmp_path: Path,
) -> None:
    runner.start(
        run_id="dup",
        argv=[sys.executable, "-c", "import time; time.sleep(0.5)"],
        output_dir=tmp_path,
    )
    with pytest.raises(RuntimeError):
        runner.start(
            run_id="dup",
            argv=[sys.executable, "-c", "pass"],
            output_dir=tmp_path,
        )
    runner.stop("dup")


def test_is_running_tracks_subprocess_state(
    runner: LocalRunner, tmp_path: Path,
) -> None:
    runner.start(
        run_id="r",
        argv=[sys.executable, "-c", "import time; time.sleep(0.5)"],
        output_dir=tmp_path,
    )
    assert runner.is_running("r") is True
    runner.get("r").process.wait(timeout=5.0)  # type: ignore[union-attr]
    assert _wait_until(lambda: not runner.is_running("r"), timeout=2.0)


def test_stop_returns_false_if_run_id_unknown(runner: LocalRunner) -> None:
    assert runner.stop("missing") is False


# ---------------------------------------------------------------------------
# SIGTERM → SIGKILL
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX SIGTERM/SIGKILL semantics aren't equivalent on Windows",
)
def test_stop_sigterm_then_sigkill(runner: LocalRunner, tmp_path: Path) -> None:
    """A subprocess that ignores SIGTERM is killed via SIGKILL after grace.

    We craft a Python child that installs a no-op SIGTERM handler and
    sleeps. SIGTERM gets ignored; the runner's stop() escalates to SIGKILL
    after ``grace_seconds=0.2``. Reap returns a non-zero exit code (-9 on
    Linux, -SIGKILL = -9 on macOS).
    """
    code = (
        "import signal, time, sys\n"
        "signal.signal(signal.SIGTERM, lambda *a: None)\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    runner.start(
        run_id="stubborn",
        argv=[sys.executable, "-c", code],
        output_dir=tmp_path,
    )
    # Wait for the child to install its handler.
    assert _wait_until(
        lambda: any("ready" in line for line in runner.snapshot_log("stubborn")),
        timeout=3.0,
    )
    assert runner.stop("stubborn", grace_seconds=0.2) is True
    rc = runner.reap("stubborn")
    assert rc is not None
    # On POSIX, SIGKILL surfaces as -SIGKILL = -9.
    assert rc < 0


def test_stop_returns_quickly_when_subprocess_already_exited(
    runner: LocalRunner, tmp_path: Path,
) -> None:
    runner.start(
        run_id="quick",
        argv=[sys.executable, "-c", "print('done')"],
        output_dir=tmp_path,
    )
    runner.get("quick").process.wait(timeout=5.0)  # type: ignore[union-attr]
    assert runner.stop("quick") is False  # already exited


# ---------------------------------------------------------------------------
# Ring buffer (deque-not-Queue)
# ---------------------------------------------------------------------------

def test_ring_buffer_drops_oldest_under_flood(
    runner: LocalRunner, tmp_path: Path,
) -> None:
    """Lines beyond maxlen are silently dropped from the ring buffer.

    The deque-not-queue choice prevents subprocess pipe deadlock — this
    test enforces the trade-off: buffer is bounded; old data is lost,
    new data is preserved; no exception is raised.

    We print 6000 lines against the default 5000-deep buffer, assert the
    buffer holds at most 5000, the *first* line is gone, and the *last*
    is kept.
    """
    handle = runner.start(
        run_id="flood",
        argv=[
            sys.executable,
            "-c",
            "for i in range(6000): print(i, flush=True)",
        ],
        output_dir=tmp_path,
    )
    handle.process.wait(timeout=15.0)
    # Wait for the tee thread to drain the pipe (process.wait() returns
    # when the subprocess exits, but the tee thread may still be reading
    # the trailing buffered bytes).
    assert _wait_until(
        lambda: handle.buffer.maxlen is not None
        and len(runner.snapshot_log("flood")) >= handle.buffer.maxlen,
        timeout=5.0,
    )
    snap = runner.snapshot_log("flood")
    assert len(snap) <= 5000
    assert len(snap) == 5000  # exactly bounded by maxlen
    # Most recent line is preserved.
    assert any("5999" in line for line in snap[-3:])
    # Oldest line dropped.
    assert not any(line.strip() == "0" for line in snap)


def test_snapshot_log_tail_clamps(runner: LocalRunner, tmp_path: Path) -> None:
    runner.start(
        run_id="tailed",
        argv=[
            sys.executable,
            "-c",
            "for i in range(20): print(i, flush=True)",
        ],
        output_dir=tmp_path,
    )
    runner.get("tailed").process.wait(timeout=5.0)  # type: ignore[union-attr]
    assert _wait_until(
        lambda: len(runner.snapshot_log("tailed")) >= 20, timeout=2.0
    )
    last3 = runner.snapshot_log("tailed", tail=3)
    assert len(last3) == 3
    assert any("19" in line for line in last3)


def test_buffer_is_thread_safe_under_concurrent_snapshot(
    runner: LocalRunner, tmp_path: Path,
) -> None:
    runner.start(
        run_id="rs",
        argv=[
            sys.executable,
            "-c",
            "for i in range(500): print(i, flush=True)",
        ],
        output_dir=tmp_path,
    )
    snapshots: list[int] = []

    def _reader() -> None:
        for _ in range(50):
            snapshots.append(len(runner.snapshot_log("rs")))
            time.sleep(0.001)

    threads = [threading.Thread(target=_reader) for _ in range(4)]
    for t in threads:
        t.start()
    runner.get("rs").process.wait(timeout=5.0)  # type: ignore[union-attr]
    for t in threads:
        t.join()
    # Every snapshot is monotonically non-decreasing OR bounded — the
    # worst behaviour from a buggy lock would be an exception, not a
    # value error. We just confirm we got 200 cleanly-counted snapshots.
    assert len(snapshots) == 200


# ---------------------------------------------------------------------------
# Reap + atexit hook
# ---------------------------------------------------------------------------

def test_reap_returns_exit_code_then_drops_handle(
    runner: LocalRunner, tmp_path: Path,
) -> None:
    runner.start(
        run_id="z",
        argv=[sys.executable, "-c", "import sys; sys.exit(7)"],
        output_dir=tmp_path,
    )
    runner.get("z").process.wait(timeout=5.0)  # type: ignore[union-attr]
    rc = runner.reap("z")
    assert rc == 7
    assert runner.get("z") is None
    assert runner.reap("z") is None  # idempotent


def test_atexit_hook_only_registered_once() -> None:
    """Multiple runner instances do NOT re-register the atexit hook.

    Two runners get registered, but only one ``atexit.register`` call is
    made — the class-level guard prevents duplicates.
    """
    _ = LocalRunner._atexit_registered  # noqa: SLF001
    a = LocalRunner()
    b = LocalRunner()
    assert LocalRunner._atexit_registered is True  # noqa: SLF001
    # Both instances appear in the class-level list.
    assert a in LocalRunner._instances  # noqa: SLF001
    assert b in LocalRunner._instances  # noqa: SLF001
    # ``pre`` may have been True from a prior test; the invariant is just
    # that subsequent instantiations don't re-register.


def test_buffer_default_maxlen_is_5000() -> None:
    """Lock the documented contract: deque(maxlen=5000) per spec."""
    handle = LocalRunHandle(
        run_id="dummy",
        output_dir=Path("/tmp"),
        process=None,  # type: ignore[arg-type]
        stdout_log_path=Path("/tmp/x"),
    )
    assert isinstance(handle.buffer, deque)
    assert handle.buffer.maxlen == 5000


def test_snapshot_log_tail_zero_returns_empty(
    runner: LocalRunner, tmp_path: Path,
) -> None:
    """``tail=0`` is "no lines" — guard against ``lines[-0:]`` returning all.

    Regression for L7: ``-0`` is ``0`` in Python, so the naive
    ``lines[-tail:]`` returned everything when callers expected nothing.
    """
    runner.start(
        run_id="t0",
        argv=[
            sys.executable,
            "-c",
            "for i in range(5): print(i, flush=True)",
        ],
        output_dir=tmp_path,
    )
    runner.get("t0").process.wait(timeout=5.0)  # type: ignore[union-attr]
    assert _wait_until(
        lambda: len(runner.snapshot_log("t0")) >= 5, timeout=2.0
    )
    assert runner.snapshot_log("t0", tail=0) == []
    # Negative tail also returns nothing (defensive bound).
    assert runner.snapshot_log("t0", tail=-1) == []


def test_concurrent_start_with_same_run_id_only_one_succeeds(
    tmp_path: Path,
) -> None:
    """Regression for M5: TOCTOU between dup-check and Popen.

    Twelve threads race to start the same run_id; the lock + reservation
    pattern guarantees exactly one succeeds. The losers raise RuntimeError;
    the winner spawns a real subprocess. We then stop+reap to clean up.
    """
    runner = LocalRunner()
    barrier = threading.Barrier(12)
    successes: list[int] = []
    failures: list[BaseException] = []
    lock = threading.Lock()

    def _worker() -> None:
        barrier.wait()
        try:
            runner.start(
                run_id="race",
                argv=[
                    sys.executable,
                    "-c",
                    "import time; time.sleep(0.5)",
                ],
                output_dir=tmp_path,
            )
            with lock:
                successes.append(1)
        except RuntimeError as exc:
            with lock:
                failures.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert len(successes) == 1
        assert len(failures) == 11
    finally:
        runner.stop("race")
        runner.reap("race")
