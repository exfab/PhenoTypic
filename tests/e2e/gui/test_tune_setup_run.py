"""E2E coverage for Tune Setup authoring and Run navigation."""
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
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.ci_flaky


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_200(url: str, *, timeout: float = 25.0) -> None:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, ConnectionRefusedError, OSError) as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"Tune server did not respond at {url}: {last_error!r}")


_LAUNCHER = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.enhance import GaussianBlur
    from phenotypic.gui.shell import SandboxRoot
    from phenotypic.gui.tune import create_app

    sandbox_dir = Path(sys.argv[1])
    port = int(sys.argv[2])
    pipeline = sandbox_dir / "pipeline.json.pht-pipe"
    pipeline.write_text(
        ImagePipeline(
            ops=[GaussianBlur(sigma=2.0), OtsuDetector()]
        ).to_json(),
        encoding="utf-8",
    )
    (sandbox_dir / "layout.csv").write_text(
        "MetadataImage_ImageName,Object_Label\\nplate.cr3,1\\n",
        encoding="utf-8",
    )
    (sandbox_dir / "plate images").mkdir()
    app = create_app(
        root=None,
        url_prefix="/",
        sandbox=SandboxRoot.from_path(sandbox_dir),
    )
    app.run(host="127.0.0.1", port=port, debug=False)
    """
)


@pytest.fixture()
def tune_setup_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """Spawn a sandbox-bound standalone Tune app."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    port = _free_port()
    log_path = Path(tempfile.gettempdir()) / f"tune-setup-e2e-{port}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", _LAUNCHER, str(sandbox_dir), str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ},
        )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_http_200(base_url + "/")
        yield base_url, sandbox_dir
    finally:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def test_setup_authors_spec_and_navigates_to_run(
    page: Page,
    tune_setup_server: tuple[str, Path],
) -> None:
    base_url, sandbox_dir = tune_setup_server
    page.goto(base_url + "/")
    page.wait_for_selector("#tune-setup-pipeline-input", timeout=20_000)

    page.fill("#tune-setup-pipeline-input", "pipeline.json.pht-pipe")
    page.press("#tune-setup-pipeline-input", "Enter")
    page.fill("#tune-setup-metadata-input", "layout.csv")
    page.press("#tune-setup-metadata-input", "Enter")

    page.wait_for_selector(
        '[id*="tune-setup-space-knob-row"]',
        timeout=20_000,
    )
    continue_button = page.locator("#tune-setup-continue")
    expect(continue_button).to_be_enabled(timeout=20_000)
    continue_button.click()

    page.wait_for_selector(
        "#tune-destview-run:not(.tune-view-hidden)",
        timeout=20_000,
    )
    expect(page.locator("#tune-setup-gate")).to_contain_text(
        "Authored tuning spec",
        timeout=20_000,
    )
    authored = list(
        (sandbox_dir / ".phenotypic-gui" / "presets" / "tune").glob(
            "*.json.pht-tune"
        )
    )
    assert len(authored) == 1
