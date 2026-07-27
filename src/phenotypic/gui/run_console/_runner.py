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
from typing import IO, Callable, Iterable
from uuid import UUID

from phenotypic.gui._config import RUN_LOG_DIRNAME, STDOUT_LOG

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
_EXIT_THREAD_NAME_PREFIX = "phenotypic-runner-exit-"
_DEFAULT_RETENTION_LIMIT = 64
#: A descendant may inherit the child's stdout descriptor and keep the pipe
#: open after the observed process exits. Terminal state must not wait for
#: that unrelated descendant, so log-drain joining is deliberately bounded.
_TEE_DRAIN_GRACE_SECONDS = 0.25


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
        generation: Durable launch generation used by every lifecycle access
            and the exit callback.
        exit_thread: Observer that waits for process termination.
        finished_at: Monotonic completion timestamp used for bounded eviction.
    """

    run_id: str
    output_dir: Path
    process: subprocess.Popen[bytes]
    stdout_log_path: Path
    generation: UUID
    buffer: "deque[str]" = field(default_factory=lambda: deque(maxlen=_LOG_BUFFER_MAXLEN))
    buffer_lock: threading.Lock = field(default_factory=threading.Lock)
    #: Daemon thread teeing ``process.stdout`` into ``buffer`` and disk.
    #: Exposed so callers (and tests) can ``join()`` and know the pipe is
    #: fully drained — ``process.wait()`` only signals subprocess exit; the
    #: OS pipe may still hold buffered bytes the tee thread has yet to read.
    tee_thread: threading.Thread | None = None
    exit_thread: threading.Thread | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None


ExitCallback = Callable[[LocalRunHandle, int], None]
ThreadFactory = Callable[..., threading.Thread]


class LocalRunner:
    """Spawns + tracks local pipeline subprocesses.

    A single :class:`LocalRunner` owns the ``atexit`` hook for the process
    so multiple registry entries don't double-register the cleanup. The
    Phase 5 composer instantiates one per Flask app.
    """

    _atexit_registered: bool = False
    # TODO(perf): every ``LocalRunner()`` appends itself to this list and
    # never removes itself. In production exactly one runner exists per
    # process so the list stays bounded, but pytest fixtures that build
    # many run-console apps will accumulate instances and retain their
    # ``_handles`` dicts. If/when a long-running test suite exhibits the
    # leak, switch to ``weakref.WeakSet``. Bounded enough today.
    _instances: list["LocalRunner"] = []
    _atexit_lock = threading.Lock()

    def __init__(
        self,
        *,
        retention_limit: int = _DEFAULT_RETENTION_LIMIT,
        thread_factory: ThreadFactory = threading.Thread,
    ) -> None:
        if retention_limit < 1:
            raise ValueError("retention_limit must be at least 1")
        self._handles: dict[str, LocalRunHandle | None] = {}
        self._lock = threading.Lock()
        self._retention_limit = retention_limit
        self._thread_factory = thread_factory
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
        generation: UUID,
        on_exit: ExitCallback | None = None,
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
            generation: Required durable launch generation. Every later
                lifecycle access must present the same value.
            on_exit: Callback invoked exactly once by the lifecycle observer
                after the process exits. It receives the retained handle and
                integer return code.

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
            existing = self._handles.get(run_id)
            if run_id in self._handles:
                can_replace_generation = (
                    existing is not None
                    and existing.process.poll() is not None
                    and generation != existing.generation
                )
                if not can_replace_generation:
                    raise RuntimeError(
                        f"run_id already running or retained: {run_id!r}"
                    )
            # Starting a new generation for this output evicts its finished
            # predecessor only now, preserving logs/handles until replacement.
            self._handles[run_id] = None  # type: ignore[assignment]

        process: subprocess.Popen[bytes] | None = None
        handle: LocalRunHandle | None = None
        try:
            log_dir = output_dir / RUN_LOG_DIRNAME
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / STDOUT_LOG

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
                generation=generation,
            )
        except BaseException:
            with self._lock:
                self._handles.pop(run_id, None)
            raise

        with self._lock:
            self._handles[run_id] = handle

        try:
            tee_thread = self._thread_factory(
                target=self._tee_loop,
                args=(handle,),
                name=f"{_TEE_THREAD_NAME_PREFIX}{run_id}",
                daemon=True,
            )
            exit_thread = self._thread_factory(
                target=self._observe_exit,
                args=(handle, on_exit),
                name=f"{_EXIT_THREAD_NAME_PREFIX}{run_id}",
                daemon=True,
            )
            handle.tee_thread = tee_thread
            handle.exit_thread = exit_thread
            tee_thread.start()
            exit_thread.start()
        except BaseException:
            # A child without an exit observer is unsafe. Terminate and reap
            # it synchronously, release the reservation/handle, and preserve
            # any log bytes already drained by the tee thread.
            self._terminate_and_reap(handle, grace_seconds=1.0)
            if handle.tee_thread is not None and handle.tee_thread.is_alive():
                handle.tee_thread.join(timeout=1.0)
            with self._lock:
                if self._handles.get(run_id) is handle:
                    self._handles.pop(run_id, None)
            raise

        with self._lock:
            self._evict_finished_handles_locked(protected_run_id=run_id)
        logger.debug("LocalRunner.start: run_id=%s pid=%d", run_id, process.pid)
        return handle

    def get(self, run_id: str, *, generation: UUID) -> LocalRunHandle | None:
        """Return the exact generation handle, or ``None`` if absent/stale.

        Returns ``None`` for both "never registered" and "reservation
        held but ``Popen`` not yet completed" (the reservation window
        in ``start()``), as well as a mismatched retained generation.
        """
        with self._lock:
            handle = self._handles.get(run_id)
            if handle is None or handle.generation != generation:
                return None
            return handle

    def is_running(
        self,
        run_id: str,
        *,
        generation: UUID,
    ) -> bool:
        """Return whether the exact retained generation is still running."""
        handle = self.get(run_id, generation=generation)
        if handle is None:
            return False
        return handle.process.poll() is None

    def stop(
        self,
        run_id: str,
        *,
        generation: UUID,
        grace_seconds: float = _TERM_GRACE_SECONDS,
    ) -> bool:
        """Send SIGTERM, wait ``grace_seconds``, escalate to SIGKILL.

        Args:
            run_id: Stable registry identity.
            generation: Required exact generation fence. A mismatched retained
                handle is left untouched.
            grace_seconds: Seconds to wait before escalating to SIGKILL.

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
        handle = self.get(run_id, generation=generation)
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

    def snapshot_log(
        self,
        run_id: str,
        *,
        generation: UUID,
        tail: int | None = None,
    ) -> list[str]:
        """Return a snapshot of the in-memory log buffer for ``run_id``.

        Args:
            run_id: Registry key.
            generation: Required exact generation fence.
            tail: If non-None, return at most this many trailing lines.

        Returns:
            List of decoded log lines (newline-terminated). Empty list if
            ``run_id`` is unknown.
        """
        handle = self.get(run_id, generation=generation)
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

    def reap(self, run_id: str, *, generation: UUID) -> int | None:
        """Explicitly drop a finished retained handle and return its exit code.

        Idempotent: returns ``None`` if no matching generation exists or the
        process is still running. A stale generation cannot reap a replacement.
        """
        with self._lock:
            handle = self._handles.get(run_id)
            if handle is None or handle.generation != generation:
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

    def _observe_exit(
        self,
        handle: LocalRunHandle,
        on_exit: ExitCallback | None,
    ) -> None:
        """Wait for terminal process evidence and notify the owner once."""
        returncode = handle.process.wait()
        if handle.tee_thread is not None:
            handle.tee_thread.join(timeout=_TEE_DRAIN_GRACE_SECONDS)
        handle.finished_at = time.monotonic()
        if on_exit is not None:
            try:
                on_exit(handle, returncode)
            except Exception:
                logger.exception(
                    "local exit callback failed for run_id=%s generation=%s",
                    handle.run_id,
                    handle.generation,
                )
        with self._lock:
            self._evict_finished_handles_locked(
                protected_run_id=handle.run_id
            )

    @staticmethod
    def _terminate_and_reap(
        handle: LocalRunHandle,
        *,
        grace_seconds: float,
    ) -> None:
        """Best-effort terminate and synchronously reap one child."""
        process = handle.process
        if process.poll() is None:
            try:
                process.terminate()
            except (ProcessLookupError, OSError):
                pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                logger.error(
                    "process %s did not exit after observer-start cleanup",
                    handle.run_id,
                )

    def _evict_finished_handles_locked(
        self,
        *,
        protected_run_id: str,
    ) -> None:
        """Evict oldest finished handles until the retention bound holds."""
        excess = len(self._handles) - self._retention_limit
        if excess <= 0:
            return
        candidates = sorted(
            (
                handle
                for run_id, handle in self._handles.items()
                if run_id != protected_run_id
                and handle is not None
                and handle.process.poll() is not None
            ),
            key=lambda item: (
                item.finished_at
                if item.finished_at is not None
                else item.started_at
            ),
        )
        for handle in candidates[:excess]:
            if self._handles.get(handle.run_id) is handle:
                self._handles.pop(handle.run_id, None)

    def _snapshot_handles_for_shutdown(self) -> list[LocalRunHandle]:
        """Return current handles for process-exit cleanup only.

        This private adapter is intentionally unscoped because process exit
        must terminate every child generation. Interactive lifecycle callers
        must use the generation-fenced public methods.
        """
        with self._lock:
            return [
                handle
                for handle in self._handles.values()
                if handle is not None
            ]

    @classmethod
    def _atexit_cleanup_all(cls) -> None:
        """SIGTERM every live subprocess across every runner instance.

        Snapshots both the instances list and each runner's retained handles
        once, before either loop. Otherwise a run registered between the
        first (terminate) loop and the second (wait) loop would land on
        the SIGKILL fast-path without first receiving a SIGTERM.
        """
        with cls._atexit_lock:
            instances = list(cls._instances)
        runner_snapshots: dict[LocalRunner, list[LocalRunHandle]] = {
            runner: runner._snapshot_handles_for_shutdown()
            for runner in instances
        }
        for handles in runner_snapshots.values():
            for handle in handles:
                if handle.process.poll() is not None:
                    continue
                try:
                    handle.process.terminate()
                except (ProcessLookupError, OSError):
                    pass
        # Best-effort short wait so terminating cleanly beats SIGKILL.
        deadline = time.monotonic() + 2.0
        for handles in runner_snapshots.values():
            for handle in handles:
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
