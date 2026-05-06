"""Pytest fixtures + gating for browser-driven GUI E2E tests.

These tests require Playwright + Chromium. They are skipped unless the
``PLAYWRIGHT`` environment variable is set to ``1`` (matches the CI gate
in ``.github/workflows/gui-e2e.yml``).

Fixtures shipped:

* :func:`fake_sandbox` (module-scoped) — temp directory pre-populated
  with one image directory and one CLI output (master parquet +
  ``results/`` + ``dashboard.html`` + ``progress/manifest.json``) so
  the file browser has something interesting to render and the Recent
  Runs panel rehydrates a row. Module scope so all tests in a single
  test module share one sandbox build (~0.5–1s saved per test).
* :func:`live_server` (module-scoped) — spawns ``phenotypic-gui --root
  <fake_sandbox>`` on an OS-assigned ephemeral port via a child
  process. Yields the base URL; teardown SIGTERMs the child. Module
  scope so all tests in a single test module share one Werkzeug boot
  (~5–15s saved per test).
* :func:`hub_url` — string alias for ``live_server`` for tests that
  only need the URL.

For tests that mutate the sandbox (e.g. writing a preset file under
``<sandbox>/.phenotypic-gui/``), declare local function-scoped
overrides via ``_build_sandbox`` and ``_start_live_server`` — see
``test_save_preset.py`` for the canonical pattern.

The pytest-playwright plugin contributes ``page`` (a fresh browser
page per test) and ``browser_context`` automatically.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

if os.environ.get("PLAYWRIGHT") != "1":
    pytest.skip(
        "Set PLAYWRIGHT=1 to run browser E2E tests "
        "(CI sets this automatically when gui-e2e workflow triggers).",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Sandbox helpers + fixture
# ---------------------------------------------------------------------------

def _write_sample_dashboard(output_dir: Path) -> None:
    """Generate a real ``dashboard.html`` (with the postMessage hook) in
    ``output_dir`` so the iframe assertion has live HTML to load."""
    from phenotypic._cli._dashboard._generator import generate_dashboard

    generate_dashboard(output_dir, execution_mode="local")


def _build_sandbox(parent_dir: Path) -> Path:
    """Populate a sandbox directory with the standard E2E layout.

    Public to test modules that need a function-scoped override of
    :func:`fake_sandbox` (e.g. when a test mutates ``<sandbox>/.phenotypic-gui/``
    and needs isolation). Pass a unique parent (e.g. ``tmp_path``) per call.

    Layout::

        parent_dir/
            sandbox/
                plate1/
                    image.tif
                results/
                    CliOutputExample/
                        master_measurements.parquet
                        results/
                            Run_0/
                        dashboard.html
                        progress/
                            manifest.json

    Returns:
        Path to the populated ``sandbox`` directory.
    """
    sandbox = parent_dir / "sandbox"
    sandbox.mkdir()

    # Image dir — populates the sandbox capability summary's "Image dirs" count.
    plate = sandbox / "plate1"
    plate.mkdir()
    (plate / "image.tif").write_bytes(b"")

    # CLI output dir under ``results/``. The classifier checks for
    # master_measurements.parquet + a ``results/`` subdir.
    output_dir = sandbox / "results" / "CliOutputExample"
    output_dir.mkdir(parents=True)
    (output_dir / "master_measurements.parquet").write_bytes(b"")
    (output_dir / "results").mkdir()  # the inner results/ subdir
    (output_dir / "results" / "Run_0").mkdir()

    # Manifest — Recent Runs reads this to compute status.
    progress = output_dir / "progress"
    progress.mkdir()
    (progress / "manifest.json").write_text(
        '{"version":1,"execution_mode":"local","is_complete":true,'
        '"total_images":2,"completed":2,"failed":0}',
        encoding="utf-8",
    )

    # Real dashboard.html (with postShellEvent JS) so the iframe + postMessage
    # tests have something live to load.
    _write_sample_dashboard(output_dir)

    return sandbox


@pytest.fixture(scope="module")
def fake_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped sandbox shared across all tests in a module.

    Use a function-scoped override (see ``test_save_preset.py``) for tests
    that mutate the sandbox.
    """
    parent = tmp_path_factory.mktemp("e2e_sandbox")
    return _build_sandbox(parent)


# ---------------------------------------------------------------------------
# Live-server fixture
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Ask the OS for a free TCP port and release it before returning."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_http_200(url: str, *, timeout: float = 20.0) -> None:
    """Block until ``url`` returns 2xx or the timeout expires."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, ConnectionRefusedError, OSError) as err:
            last_err = err
        time.sleep(0.2)
    raise RuntimeError(
        f"Live server did not respond at {url} within {timeout}s "
        f"(last error: {last_err!r})"
    )


def _start_live_server(sandbox: Path) -> Iterator[str]:
    """Generator that spawns ``phenotypic-gui`` against ``sandbox`` and yields
    the base URL. SIGTERMs the child on teardown; SIGKILLs after 5s.

    Public so a test module that overrides :func:`fake_sandbox` with a
    function-scoped variant can also override :func:`live_server` and reuse
    the same boot logic.
    """
    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "phenotypic.gui",
        "--root",
        str(sandbox),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_http_200(base_url + "/", timeout=20.0)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


@pytest.fixture(scope="module")
def live_server(fake_sandbox: Path) -> Iterator[str]:
    """Module-scoped Werkzeug live server backed by :func:`fake_sandbox`.

    Boots once per test module. ~5–15s of Werkzeug start-up amortized
    across every test in the file.
    """
    yield from _start_live_server(fake_sandbox)


@pytest.fixture(scope="module")
def hub_url(live_server: str) -> str:
    """Convenience alias when a test only needs the base URL string."""
    return live_server


# ---------------------------------------------------------------------------
# Browser context overrides for pytest-playwright
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args: dict) -> dict:
    """Override the default pytest-playwright context args.

    ``ignore_https_errors`` is irrelevant here (loopback-only HTTP), but
    setting ``viewport`` makes captured screenshots reproducible across
    machines without depending on the host display.
    """
    return {
        **browser_context_args,
        "viewport": {"width": 1400, "height": 900},
    }
