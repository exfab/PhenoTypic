"""Pytest fixtures + gating for browser-driven GUI E2E tests.

These tests require Playwright + Chromium. They are skipped unless the
``PLAYWRIGHT`` environment variable is set to ``1`` (matches the CI gate
in the ``e2e-tests`` job of ``.github/workflows/gui-checks.yml``).

Fixtures shipped:

* :func:`fake_sandbox` (module-scoped) — temp directory pre-populated
  with one image directory and one CLI output (master parquet +
  ``results/`` + ``dashboard.html`` +
  ``.phenotypic/progress/manifest.json``) so
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

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from phenotypic.gui._config import DELIVERABLES_DIRNAME
from phenotypic.sdk_ import manifest_json_path

if os.environ.get("PLAYWRIGHT") != "1":
    pytest.skip(
        "Set PLAYWRIGHT=1 to run browser E2E tests "
        "(the e2e-tests job in the gui-checks workflow sets this automatically).",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Sandbox helpers + fixture
# ---------------------------------------------------------------------------

def _write_sample_dashboard(output_dir: Path) -> None:
    """Generate a real ``dashboard.html`` (with the postMessage hook) under
    ``output_dir/deliverables/`` so the iframe assertion has live HTML to load."""
    from phenotypic._cli._dashboard._generator import generate_dashboard

    generate_dashboard(output_dir, execution_mode="local")


def publish_coherent_terminal_evidence(
    output_dir: Path,
    *,
    total_images: int,
) -> Path:
    """Publish a minimal successful manifest for a terminal E2E fixture.

    Mutation-capable Results fixtures must model a completed run rather than
    relying on the viewer to infer write authority from deliverables.  Write
    through the canonical SDK path so a generated dashboard's
    ``.phenotypic/progress`` directory cannot shadow a legacy manifest.

    Args:
        output_dir: Full-run output root.
        total_images: Number of successfully completed input images.

    Returns:
        Canonical manifest path.
    """
    target = manifest_json_path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "version": 1,
                "execution_mode": "local",
                "is_complete": True,
                "total_images": total_images,
                "completed": total_images,
                "failed": 0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return target


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
                        deliverables/
                            master_measurements.parquet
                            dashboard.html
                        results/
                            Run_0/
                        .phenotypic/
                            progress/
                                manifest.json

    Returns:
        Path to the populated ``sandbox`` directory.
    """
    sandbox = parent_dir / "sandbox"
    sandbox.mkdir()

    # Image dir — populates the sandbox capability summary's "Image dirs" count.
    from PIL import Image as PILImage

    plate = sandbox / "plate1"
    plate.mkdir()
    PILImage.new("RGB", (32, 32), (40, 80, 120)).save(plate / "image.tif")

    # CLI output dir under ``results/``. The classifier checks for
    # deliverables/master_measurements.parquet + a ``results/`` subdir.
    output_dir = sandbox / "results" / "CliOutputExample"
    output_dir.mkdir(parents=True)
    deliverables = output_dir / DELIVERABLES_DIRNAME
    deliverables.mkdir()
    (deliverables / "master_measurements.parquet").write_bytes(b"")
    (output_dir / "results").mkdir()  # the inner results/ subdir
    (output_dir / "results" / "Run_0").mkdir()

    # Real dashboard.html (with postShellEvent JS) so the iframe + postMessage
    # tests have something live to load.
    _write_sample_dashboard(output_dir)
    # Publish after dashboard generation because the generator creates the
    # canonical ``.phenotypic/progress`` tree.
    publish_coherent_terminal_evidence(output_dir, total_images=2)

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


def _start_live_server(
    sandbox: Path,
    *,
    env_overrides: dict[str, str] | None = None,
) -> Iterator[str]:
    """Generator that spawns ``phenotypic-gui`` against ``sandbox`` and yields
    the base URL. SIGTERMs the child on teardown; SIGKILLs after 5s.

    Public so a test module that overrides :func:`fake_sandbox` with a
    function-scoped variant can also override :func:`live_server` and reuse
    the same boot logic.

    Args:
        sandbox: Path passed as ``--root`` to ``phenotypic-gui``.
        env_overrides: Optional mapping merged on top of ``os.environ``
            for the subprocess. Lets a calling test module flip env
            vars on the spawned hub without re-implementing the boot
            logic. (The ``PHENOTYPIC_GUI_DAG`` feature flag this
            originally serviced was retired in Phase 8; the parameter
            stays for future use.)
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
    env = {**os.environ, **(env_overrides or {})} if env_overrides else None
    # Redirect the GUI subprocess' stdout+stderr to a temp log file, NOT
    # an in-process ``subprocess.PIPE``.  Werkzeug logs every request to
    # stderr; an undrained pipe fills its ~64 KB OS buffer after a few
    # hundred requests, at which point the GUI blocks on its next stderr
    # write and every later ``page.goto`` in the module times out.  A
    # module-scoped server serves enough requests to hit this reliably.
    log_path = Path(tempfile.gettempdir()) / f"phenotypic-gui-e2e-{port}.log"
    with log_path.open("w", encoding="utf-8") as gui_log:
        proc = subprocess.Popen(
            cmd,
            stdout=gui_log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
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


def bind_results_output(
    page: Any,
    hub_url: str,
    output_rel: str,
    *,
    destination: str | None = "/results/",
    timeout_ms: int = 30_000,
) -> dict[str, Any]:
    """Bind one Results output and wait for atomic publication.

    The production endpoint accepts the request asynchronously (HTTP 202).
    This helper polls the returned job path until the paired Results and
    Analysis publication succeeds, then navigates to ``destination``.  The
    legacy synchronous HTTP 200 response remains supported for isolated
    route adapters.

    Args:
        page: Playwright page whose browser context sends the request.
        hub_url: Origin of the running GUI hub.
        output_rel: Sandbox-relative output or deliverables path.
        destination: Hub path to load after publication. Pass ``None`` to
            leave the page on the Shell landing route.
        timeout_ms: Maximum time to wait for a terminal binding job.

    Returns:
        The successful synchronous response or terminal job payload.

    Raises:
        AssertionError: If submission, polling, or publication fails.
    """
    page.goto(hub_url + "/")
    page.wait_for_load_state("networkidle")
    outcome = page.evaluate(
        """
        async ({path, timeoutMs}) => {
            const decode = async (response) => {
                const text = await response.text();
                let payload = null;
                try {
                    payload = text ? JSON.parse(text) : {};
                } catch (_error) {
                    return {
                        ok: false,
                        error: `non-JSON response: ${text}`,
                        status: response.status,
                    };
                }
                return {ok: true, payload, status: response.status};
            };

            const submitted = await fetch(
                '/sandbox/api/viewer/output-root',
                {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({path}),
                },
            );
            const accepted = await decode(submitted);
            if (!accepted.ok) {
                return accepted;
            }
            if (accepted.status === 200) {
                const status = accepted.payload?.status;
                const absPath = accepted.payload?.abs_path;
                const fingerprint = (
                    accepted.payload?.snapshot?.processing_fingerprint
                );
                if (
                    status !== 'ok' ||
                    typeof absPath !== 'string' ||
                    absPath.length === 0 ||
                    typeof fingerprint !== 'string' ||
                    fingerprint.length === 0
                ) {
                    return {
                        ok: false,
                        status: accepted.status,
                        error: 'synchronous binding acknowledgement was incomplete',
                        payload: accepted.payload,
                    };
                }
                return {ok: true, payload: accepted.payload};
            }
            if (accepted.status !== 202) {
                return {
                    ok: false,
                    status: accepted.status,
                    error: 'binding submission was not accepted',
                    payload: accepted.payload,
                };
            }

            const jobId = accepted.payload?.job_id;
            const acceptedJob = accepted.payload?.job;
            const pollPath = accepted.payload?.poll_path;
            const cancelPath = accepted.payload?.cancel_path;
            const expectedSuffix = (
                typeof jobId === 'string' && jobId.length > 0
            ) ? `/jobs/${encodeURIComponent(jobId)}` : null;
            const activeJob = (
                acceptedJob &&
                acceptedJob.job_id === jobId &&
                ['queued', 'running'].includes(acceptedJob.status) &&
                acceptedJob.terminal === false
            );
            const consistentPaths = (
                expectedSuffix &&
                typeof pollPath === 'string' &&
                pollPath.length > 0 &&
                pollPath === cancelPath &&
                pollPath.split(/[?#]/, 1)[0].endsWith(expectedSuffix)
            );
            if (!activeJob || !consistentPaths) {
                return {
                    ok: false,
                    status: accepted.status,
                    error: 'binding acceptance had an incomplete polling contract',
                    payload: accepted.payload,
                };
            }

            const deadline = Date.now() + timeoutMs;
            while (Date.now() < deadline) {
                const polled = await fetch(pollPath, {
                    headers: {'Accept': 'application/json'},
                });
                const snapshot = await decode(polled);
                if (!snapshot.ok || snapshot.status !== 200) {
                    return {
                        ok: false,
                        status: snapshot.status,
                        error: snapshot.error || 'binding poll failed',
                        payload: snapshot.payload,
                    };
                }
                const polledJob = snapshot.payload?.job;
                const polledStatus = polledJob?.status;
                const authoritative = (
                    snapshot.payload?.job_id === jobId &&
                    snapshot.payload?.status === polledStatus &&
                    polledJob?.job_id === jobId &&
                    [
                        'queued',
                        'running',
                        'succeeded',
                        'failed',
                        'cancelled',
                        'superseded',
                    ].includes(polledStatus) &&
                    typeof polledJob?.terminal === 'boolean'
                );
                if (!authoritative) {
                    return {
                        ok: false,
                        status: snapshot.status,
                        error: 'binding progress response was not authoritative',
                        payload: snapshot.payload,
                    };
                }
                const terminal = polledJob.terminal;
                if (terminal) {
                    const fingerprint = (
                        snapshot.payload?.snapshot?.processing_fingerprint
                    );
                    if (
                        polledStatus !== 'succeeded' ||
                        typeof snapshot.payload?.abs_path !== 'string' ||
                        snapshot.payload.abs_path.length === 0 ||
                        typeof fingerprint !== 'string' ||
                        fingerprint.length === 0
                    ) {
                        return {
                            ok: false,
                            status: snapshot.status,
                            error: (
                                snapshot.payload?.error ||
                                polledJob?.error ||
                                `binding ended as ${polledStatus}`
                            ),
                            payload: snapshot.payload,
                        };
                    }
                    return {ok: true, payload: snapshot.payload};
                }
                if (!['queued', 'running'].includes(polledStatus)) {
                    return {
                        ok: false,
                        status: snapshot.status,
                        error: (
                            `nonterminal binding reported status ${polledStatus}`
                        ),
                        payload: snapshot.payload,
                    };
                }
                await new Promise((resolve) => setTimeout(resolve, 50));
            }
            return {
                ok: false,
                status: 202,
                error: `binding did not finish within ${timeoutMs}ms`,
                payload: accepted.payload,
            };
        }
        """,
        {"path": output_rel, "timeoutMs": timeout_ms},
    )
    assert outcome["ok"], (
        f"viewer hand-off failed for {output_rel!r}: "
        f"HTTP {outcome.get('status')} error={outcome.get('error')!r} "
        f"payload={outcome.get('payload')!r}"
    )
    if destination is not None:
        page.goto(hub_url + destination)
    return outcome["payload"]


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
# Builder palette helpers
# ---------------------------------------------------------------------------

def expand_palette_accordions(page) -> None:
    """Expand every collapsed palette accordion section on the builder.

    The builder palette groups operations under ``dbc.Accordion``
    sections.  ``always_open=True`` auto-expands only the *first* item;
    the rest start ``collapsed`` (``display: none``).  A palette button
    in a non-first category (e.g. ``GaussianBlur`` under Enhancer) is
    therefore present in the DOM but not *visible*, so a Playwright
    ``hover`` / drag against it times out with "element is not visible".

    Builder-canvas E2E modules call this from their ``_open_builder``
    helper so palette buttons in any category are interactable.  The
    accordion is plain Bootstrap collapse chrome — independent of
    ``palette_dnd.js`` — so this works even in the asset-failure
    resilience tests.
    """
    for header_text in ("Corrector", "Detector", "Enhancer", "Refiner", "Measure"):
        header = page.locator(
            f'button.accordion-button:has-text("{header_text}")'
        ).first
        if header.count() > 0:
            try:
                cls = header.get_attribute("class") or ""
                if "collapsed" in cls:
                    header.click()
                    page.wait_for_timeout(150)
            except Exception:  # pragma: no cover - best-effort
                pass
    page.wait_for_timeout(150)


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
