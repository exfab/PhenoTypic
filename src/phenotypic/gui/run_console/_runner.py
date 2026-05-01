"""``LocalRunner`` — Popen + deque ring buffer + SIGTERM-on-stop.

Spawns ``python -m phenotypic ...`` in a subprocess, tees stdout to disk
AND to an in-memory ring buffer that the Run console's log-tail callback
samples on a ``dcc.Interval``. Owns the subprocess lifecycle but NOT the
UI scratch state (the registry persists across UI release).

Why ``collections.deque`` not ``queue.Queue``?
    A bounded ``queue.Queue`` with blocking ``put()`` back-fills the
    subprocess's stdout pipe — once the queue fills, the tee thread
    blocks on ``put()``, the kernel buffer fills, and the subprocess
    blocks on its own ``write()``. This is a textbook deadlock.

    A ``deque(maxlen=N)`` discards the oldest entries silently when
    full, so the pipe drains forever and the subprocess never blocks.
    Trade-off: a flooded log loses old lines, not new ones — exactly
    what a UI log-tail wants.

Process termination
    ``stop()`` sends ``SIGTERM`` first, waits up to 10 s, then escalates
    to ``SIGKILL``. ``atexit`` registers an emergency SIGTERM-everyone
    hook so a Ctrl-C on the GUI propagates without orphaning subprocesses.
"""
from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "LocalRunHandle",
    "LocalRunner",
]

#: Maximum lines retained in the in-memory ring buffer per run. Beyond this
#: the deque silently drops the oldest entry. 5000 is enough for the Run
#: console's UI log tail, which only renders the last ~100 lines anyway.
_LOG_BUFFER_MAXLEN = 5000

#: Seconds between SIGTERM and SIGKILL. The CLI's atexit handlers and
#: progress-flush logic typically finish in well under a second; 10 s is
#: a generous grace window.
_TERM_GRACE_SECONDS = 10.0

#: How often the tee thread polls when ``Popen.stdout.readline`` returns
#: empty bytes. We use ``readline``'s blocking semantics instead, but a
#: timeout-style fall-back exists for race conditions on Windows.
_TEE_THREAD_NAME_PREFIX = "phenotypic-runner-tee-"


@dataclass
class LocalRunHandle:
    """Immutable view of a live local subprocess.

    Attributes:
        run_id: Stable identifier within the registry.
        output_dir: Directory the run is writing into.
        process: The :class:`subprocess.Popen` object. ``poll()`` reveals
            exit status; ``returncode`` is set after termination.
        stdout_log_path: ``<output_dir>/.gui_log/stdout.log`` — the on-disk
            tee target. Stays valid even after the deque has been GC'd.
        buffer: In-memory ring buffer (``collections.deque(maxlen=...)``).
            Caller MUST acquire ``buffer_lock`` before reading.
        buffer_lock: Guards the deque against concurrent
            ``append`` (tee thread) and ``snapshot`` (Dash callback)
            calls.
    """

    run_id: str
    output_dir: Path
    process: subprocess.Popen[bytes]
    stdout_log_path: Path
    buffer: "deque[str]" = field(default_factory=lambda: deque(maxlen=_LOG_BUFFER_MAXLEN))
    buffer_lock: threading.Lock = field(default_factory=threading.Lock)


class LocalRunner:
    """Spawns + tracks local pipeline subprocesses.

    A single :class:`LocalRunner` owns the ``atexit`` hook for the process
    so multiple registry entries don't double-register the cleanup. The
    Phase 5 composer instantiates one per Flask app.
    """

    _atexit_registered: bool = False
    _instances: list["LocalRunner"] = []
    _atexit_lock = threading.Lock()

    def __init__(self) -> None:
        self._handles: dict[str, LocalRunHandle] = {}
        self._lock = threading.Lock()
        with LocalRunner._atexit_lock:
            LocalRunner._instances.append(self)
            if not LocalRunner._atexit_registered:
                atexit.register(LocalRunner._atexit_cleanup_all)
                LocalRunner._atexit_registered = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        run_id: str,
        argv: list[str],
        *,
        output_dir: Path,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> LocalRunHandle:
        """Spawn ``argv`` and tee its stdout.

        Args:
            run_id: Stable id (typically the registry key).
            argv: Command + args. Caller is responsible for building the
                CLI invocation (``["python", "-m", "phenotypic", ...]``).
            output_dir: Where the CLI is writing its results. The tee
                thread creates ``<output_dir>/.gui_log/`` and writes
                ``stdout.log`` there.
            cwd: Subprocess working directory. Defaults to ``output_dir``.
            env: Subprocess environment. Defaults to inheriting the
                parent's full environment.

        Returns:
            :class:`LocalRunHandle` with the live process + ring buffer.

        Raises:
            FileNotFoundError: If ``argv[0]`` cannot be found.
            RuntimeError: If a handle already exists for ``run_id``.
        """
        # Reserve the run_id under the lock BEFORE the (slow) Popen so a
        # concurrent ``start()`` with the same id can't slip past the
        # duplicate check. We use a sentinel ``None`` placeholder; if the
        # spawn fails we drop the reservation in the ``except`` block so
        # a retry can proceed.
        with self._lock:
            if run_id in self._handles:
                raise RuntimeError(f"run_id already running: {run_id!r}")
            self._handles[run_id] = None  # type: ignore[assignment]

        try:
            log_dir = output_dir / ".gui_log"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "stdout.log"

            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge so dashboard sees both
                cwd=str(cwd or output_dir),
                env=env,
                # ``bufsize=0`` (unbuffered binary) so ``readline``
                # returns full lines as the subprocess flushes them.
                bufsize=0,
            )

            handle = LocalRunHandle(
                run_id=run_id,
                output_dir=output_dir,
                process=process,
                stdout_log_path=log_path,
            )
        except BaseException:
            with self._lock:
                self._handles.pop(run_id, None)
            raise

        with self._lock:
            self._handles[run_id] = handle

        thread = threading.Thread(
            target=self._tee_loop,
            args=(handle,),
            name=f"{_TEE_THREAD_NAME_PREFIX}{run_id}",
            daemon=True,
        )
        thread.start()
        logger.debug("LocalRunner.start: run_id=%s pid=%d", run_id, process.pid)
        return handle

    def get(self, run_id: str) -> LocalRunHandle | None:
        """Return the handle for ``run_id``, or ``None`` if absent.

        Returns ``None`` for both "never registered" and "reservation
        held but ``Popen`` not yet completed" (the reservation window
        in ``start()``). Callers that need to distinguish should use
        :meth:`list_run_ids`.
        """
        with self._lock:
            handle = self._handles.get(run_id)
            return handle if handle is not None else None

    def list_run_ids(self) -> list[str]:
        """Return live + reserved run_ids. May include reservations from
        in-flight ``start()`` calls."""
        with self._lock:
            return list(self._handles.keys())

    def is_running(self, run_id: str) -> bool:
        handle = self.get(run_id)
        if handle is None:
            return False
        return handle.process.poll() is None

    def stop(self, run_id: str, *, grace_seconds: float = _TERM_GRACE_SECONDS) -> bool:
        """Send SIGTERM, wait ``grace_seconds``, escalate to SIGKILL.

        Note:
            This method **does not** update the :class:`RunRegistry`. The
            runner and registry are deliberately decoupled (the registry
            persists across UI release while the runner does not). Callers
            that need the registry to reflect a cancellation MUST call
            ``registry.update_status(run_id, "cancelled")`` after this
            method returns ``True``. The Phase 6 ``Cancel`` callback owns
            that wiring.

        Returns:
            ``True`` if a handle was found and termination was attempted;
            ``False`` if no live handle exists for ``run_id``.
        """
        handle = self.get(run_id)
        if handle is None:
            return False
        proc = handle.process
        if proc.poll() is not None:
            return False  # already exited

        logger.debug("LocalRunner.stop: SIGTERM %s pid=%d", run_id, proc.pid)
        try:
            proc.terminate()
        except ProcessLookupError:
            return False
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            logger.warning(
                "LocalRunner.stop: SIGTERM grace expired (%ss); SIGKILL %s",
                grace_seconds,
                run_id,
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover
                logger.error(
                    "LocalRunner.stop: process %s did not exit after SIGKILL",
                    run_id,
                )
        return True

    def snapshot_log(self, run_id: str, *, tail: int | None = None) -> list[str]:
        """Return a snapshot of the in-memory log buffer for ``run_id``.

        Args:
            run_id: Registry key.
            tail: If non-None, return at most this many trailing lines.

        Returns:
            List of decoded log lines (newline-terminated). Empty list if
            ``run_id`` is unknown.
        """
        handle = self.get(run_id)
        if handle is None:
            return []
        with handle.buffer_lock:
            lines = list(handle.buffer)
        if tail is None:
            return lines
        if tail <= 0:
            # ``lines[-0:]`` is ``lines[0:]`` which returns everything;
            # we want "no lines". Match the docstring contract.
            return []
        return lines[-tail:]

    def reap(self, run_id: str) -> int | None:
        """Drop the handle for a finished run; return its exit code.

        Idempotent: returns ``None`` if no handle exists or the process
        is still running.
        """
        with self._lock:
            handle = self._handles.get(run_id)
            if handle is None:
                return None
            rc = handle.process.poll()
            if rc is None:
                return None
            del self._handles[run_id]
        return rc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tee_loop(self, handle: LocalRunHandle) -> None:
        """Read stdout line by line, append to deque + write to disk.

        Runs on a daemon thread per handle. Exits when ``stdout`` is
        closed by the subprocess (i.e. it has exited).
        """
        stdout: IO[bytes] | None = handle.process.stdout
        if stdout is None:  # pragma: no cover - never happens with PIPE
            return
        try:
            with handle.stdout_log_path.open(
                "ab", buffering=0
            ) as disk_log:
                for raw in iter(stdout.readline, b""):
                    line = raw.decode("utf-8", errors="replace")
                    with handle.buffer_lock:
                        handle.buffer.append(line)
                    try:
                        disk_log.write(raw)
                    except OSError:  # pragma: no cover - disk full etc.
                        logger.exception(
                            "tee disk write failed for run_id=%s",
                            handle.run_id,
                        )
        except Exception:  # pragma: no cover - belt-and-braces
            logger.exception(
                "tee loop crashed for run_id=%s", handle.run_id
            )
        finally:
            try:
                stdout.close()
            except OSError:
                pass

    @classmethod
    def _atexit_cleanup_all(cls) -> None:
        """SIGTERM every live subprocess across every runner instance.

        Snapshots both the instances list and each runner's run-id list
        once, before either loop. Otherwise a run registered between the
        first (terminate) loop and the second (wait) loop would land on
        the SIGKILL fast-path without first receiving a SIGTERM.
        """
        with cls._atexit_lock:
            instances = list(cls._instances)
        runner_snapshots: dict[LocalRunner, list[str]] = {
            runner: runner.list_run_ids() for runner in instances
        }
        for runner, run_ids in runner_snapshots.items():
            for run_id in run_ids:
                handle = runner.get(run_id)
                if handle is None or handle.process.poll() is not None:
                    continue
                try:
                    handle.process.terminate()
                except (ProcessLookupError, OSError):
                    pass
        # Best-effort short wait so terminating cleanly beats SIGKILL.
        deadline = time.monotonic() + 2.0
        for runner, run_ids in runner_snapshots.items():
            for run_id in run_ids:
                handle = runner.get(run_id)
                if handle is None:
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    handle.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    try:
                        handle.process.kill()
                    except (ProcessLookupError, OSError):
                        pass


# Pacify type-checker on Iterable import (kept for symmetry with future
# typing tweaks).
_ = (Iterable, os, signal)
