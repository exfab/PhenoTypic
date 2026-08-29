"""E2E: Curate linked-zoom mirror between the two side-by-side graphs (B4).

Spawns a STANDALONE tune Dash server (the hub mounts tune empty-state, so the
Curate view with its two graphs only exists on a run-bound app) over a small
parquet-journal fixture with an Image Source set, then drives the browser:

1. switch to the Curate sub-tab,
2. ``Plotly.relayout`` graph A to an explicit x-range,
3. assert graph B's layout x-range mirrors it (via ``page.evaluate`` on the
   rendered figure layout).

Marked ``ci_flaky``: it DOM/figure-polls a Dash clientside callback chain on a
freshly-booted Werkzeug subprocess, whose first-render + relayout budget is
reliable locally but stochastically exceeds the poll on GHA shared runners.
Run it locally with ``PLAYWRIGHT=1 uv run pytest
tests/e2e/gui/test_tune_overlay_sync.py -n0``.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import Page

# DOM-polls a clientside relayout chain on a cold Werkzeug subprocess — see the
# module docstring + tests/CLAUDE.md "ci_flaky convention".
pytestmark = [
    pytest.mark.ci_flaky,
    pytest.mark.skip(
        reason=(
            "Tune is unmounted by "
            "docs/superpowers/specs/2026-08-26-gui-simplification-removals "
            "(spec section 2). These tests are the acceptance suite for the "
            "re-mount; delete this marker when /tune/ is mounted again."
        )
    ),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_http_200(url: str, *, timeout: float = 25.0) -> None:
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
    raise RuntimeError(f"tune server did not respond at {url}: {last_err!r}")


_LAUNCHER = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from phenotypic.gui.tune import create_app
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.gui.shell import SandboxRoot
    from phenotypic.sdk_ import trials_parquet_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    sandbox_dir = Path(sys.argv[1])
    port = int(sys.argv[2])
    images = sandbox_dir / "calibration"
    images.mkdir(exist_ok=True)
    parquet = trials_parquet_path(sandbox_dir)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(trials=[
        Trial(number=0, params={}, score=0.4, terms={}, n_images=1),
        Trial(number=1, params={}, score=0.6, terms={}, n_images=1),
    ]).to_parquet(parquet)
    root = TuneRunRoot.discover(sandbox_dir)
    sandbox = SandboxRoot.from_path(sandbox_dir)
    app = create_app(root=root, url_prefix="/", sandbox=sandbox)
    app.run(host="127.0.0.1", port=port, debug=False)
    """
)


@pytest.fixture()
def tune_server(tmp_path: Path) -> Iterator[str]:
    """Spawn a standalone run-bound tune server; yield its base URL."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    port = _free_port()
    log_path = Path(tempfile.gettempdir()) / f"tune-e2e-{port}.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", _LAUNCHER, str(sandbox_dir), str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ},
        )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_http_200(base_url + "/", timeout=25.0)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


def test_linked_zoom_mirrors_graph_a_range_to_graph_b(
    page: Page, tune_server: str
) -> None:
    page.goto(tune_server + "/")
    page.wait_for_selector("#tune-subtab-curate", timeout=20_000)

    # Switch to the Curate view (Monitor is the default visible view).
    page.click("#tune-subtab-curate")
    page.wait_for_selector("#tune-graph-a .js-plotly-plot", timeout=20_000)
    page.wait_for_selector("#tune-graph-b .js-plotly-plot", timeout=20_000)

    # Drive an explicit x-range relayout on graph A via Plotly's JS API. This
    # is exactly what a user drag-zoom produces (the ``xaxis.range[*]`` keys in
    # ``relayoutData``), so it exercises the same clientside mirror.
    target_lo, target_hi = 1.0, 5.0
    page.evaluate(
        """
        ([lo, hi]) => {
            const gd = document.querySelector('#tune-graph-a .js-plotly-plot');
            return Plotly.relayout(gd, {'xaxis.range[0]': lo, 'xaxis.range[1]': hi});
        }
        """,
        [target_lo, target_hi],
    )

    # The clientside callback mirrors A's range onto B's figure. Poll B's
    # rendered x-axis range until it matches (Dash round-trips the figure prop).
    def _b_range() -> list | None:
        return page.evaluate(
            """
            () => {
                const gd = document.querySelector('#tune-graph-b .js-plotly-plot');
                if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis) return null;
                return gd._fullLayout.xaxis.range;
            }
            """
        )

    deadline = time.monotonic() + 15.0
    matched = False
    while time.monotonic() < deadline:
        rng = _b_range()
        if rng is not None and abs(rng[0] - target_lo) < 0.5 and abs(rng[1] - target_hi) < 0.5:
            matched = True
            break
        time.sleep(0.2)
    assert matched, f"graph B x-range did not mirror graph A (last: {_b_range()})"
