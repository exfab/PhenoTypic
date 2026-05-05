"""Pytest fixtures + gating for browser-driven GUI E2E tests.

These tests require Playwright + Chromium. They are skipped unless the
``PLAYWRIGHT`` environment variable is set to ``1`` (matches the CI gate
in ``.github/workflows/gui-e2e.yml``).

Fixtures shipped:

* :func:`fake_sandbox` — temp directory pre-populated with one image
  directory and one CLI output (master parquet + ``results/`` +
  ``dashboard.html`` + ``progress/manifest.json``) so the file browser
  has something interesting to render and the Recent Runs panel
  rehydrates a row.
* :func:`live_server` — spawns ``phenotypic-gui --root <fake_sandbox>``
  on an OS-assigned ephemeral port via a child process. Yields the
  base URL; teardown SIGTERMs the child.
* :func:`hub_url` — string alias for ``live_server`` for tests that
  only need the URL.

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
# Sandbox fixture
# ---------------------------------------------------------------------------

def _write_sample_dashboard(output_dir: Path) -> None:
    """Generate a real ``dashboard.html`` (with the postMessage hook) in
    ``output_dir`` so the iframe assertion has live HTML to load."""
    from phenotypic._cli._dashboard._generator import generate_dashboard

    generate_dashboard(output_dir, execution_mode="local")


@pytest.fixture()
def fake_sandbox(tmp_path: Path) -> Path:
    """Build a tmp sandbox the GUI hub can browse + iframe.

    Layout::

        tmp_path/
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
    """
    sandbox = tmp_path / "sandbox"
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


@pytest.fixture()
def live_server(fake_sandbox: Path) -> Iterator[str]:
    """Spawn ``phenotypic-gui`` against ``fake_sandbox`` on an ephemeral port.

    The child inherits the current ``uv``/``.venv`` environment via
    ``sys.executable``. A SIGTERM is sent during teardown; if the child
    still hasn't exited within 5s we escalate to SIGKILL so test cleanup
    cannot be blocked by a stuck Werkzeug server.

    Yields:
        The base URL (``http://127.0.0.1:<port>``). No trailing slash.
    """
    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "phenotypic.gui",
        "--root",
        str(fake_sandbox),
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


@pytest.fixture()
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
