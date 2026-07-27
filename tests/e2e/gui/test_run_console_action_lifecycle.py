"""Browser regression coverage for Run Console action acknowledgements.

The SLURM-shaped case injects a submitter that fails in memory. No scheduler
command, SSH connection, or fake scheduler executable is used by this module.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from dash import dcc, html
from playwright.sync_api import Page, Route, expect
from werkzeug.serving import make_server

from phenotypic.gui.run_console._app import create_app
from phenotypic.gui.run_console._slurm import SlurmSubmitError
from phenotypic.gui.shell._ids import (
    SHELL_METADATA_CSV_STORE,
    SHELL_SOURCE_IMAGE_ROOT_STORE,
)
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.sdk_ import gui_launch_owner_path
from tests.e2e.gui.test_run_console import (
    _prepare_action_paths,
    _set_action_controls,
    _wait_for_owner_status,
)


def _mock_submit_failure(*_args: object, **_kwargs: object) -> None:
    """Fail at the injected submit seam without invoking scheduler code."""
    raise SlurmSubmitError("mock submit seam rejected the request")


@pytest.fixture()
def action_hub(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """Serve a standalone Run app with a no-scheduler submit dependency."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    plate = sandbox / "plate1"
    plate.mkdir()
    (plate / "image.tif").write_bytes(b"one-image")
    app = create_app(
        SandboxRoot.from_path(sandbox),
        slurm_submitter=_mock_submit_failure,
        start_slurm_observer=False,
    )
    app.layout = html.Div(
        [
            dcc.Store(id=SHELL_SOURCE_IMAGE_ROOT_STORE, data=None),
            dcc.Store(id=SHELL_METADATA_CSV_STORE, data=None),
            app.layout,
        ]
    )
    server = make_server(
        "127.0.0.1",
        0,
        app.server,
        threaded=True,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        name="run-action-e2e-server",
        daemon=True,
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", sandbox
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _capture_browser_failures(
    page: Page,
) -> tuple[list[str], list[str], list[int]]:
    """Capture page errors and callback-network outcomes for one test."""
    page_errors: list[str] = []
    failed_callbacks: list[str] = []
    callback_statuses: list[int] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on(
        "requestfailed",
        lambda request: (
            failed_callbacks.append(request.url)
            if "_dash-update-component" in request.url
            else None
        ),
    )
    page.on(
        "response",
        lambda response: (
            callback_statuses.append(response.status)
            if "_dash-update-component" in response.url
            else None
        ),
    )
    return page_errors, failed_callbacks, callback_statuses


def test_same_page_validate_then_local_run_records_each_generation(
    page: Page,
    action_hub: tuple[str, Path],
) -> None:
    """Validate and Run share one callback without a same-page silent no-op."""
    hub_url, sandbox = action_hub
    page_errors, failed_callbacks, callback_statuses = (
        _capture_browser_failures(page)
    )
    validate_pipeline, input_dir, validate_output = _prepare_action_paths(
        sandbox,
        name="ValidateAction",
    )
    local_pipeline, _, local_output = _prepare_action_paths(
        sandbox,
        name="LocalAction",
    )
    page.goto(hub_url + "/")
    page.wait_for_selector("#rc-btn-validate")

    _set_action_controls(
        page,
        pipeline=validate_pipeline,
        input_dir=input_dir,
        output_dir=validate_output,
        modes=["local"],
    )
    page.locator("#rc-btn-validate").click()
    validate_owner = _wait_for_owner_status(validate_output, terminal=True)
    expect(page.locator("#rc-action-feedback")).to_contain_text(
        str(validate_owner["generation"])
    )
    expect(page.locator("#rc-btn-cancel")).to_be_disabled(timeout=10_000)

    _set_action_controls(
        page,
        pipeline=local_pipeline,
        input_dir=input_dir,
        output_dir=local_output,
        modes=["local"],
    )
    page.locator("#rc-btn-run").click()
    local_owner = _wait_for_owner_status(local_output, terminal=True)
    expect(page.locator("#rc-action-feedback")).to_contain_text(
        str(local_owner["generation"])
    )
    expect(page.locator("#rc-status-banner")).to_contain_text(
        f"generation={local_owner['generation']}"
    )
    expect(page.locator("#rc-status-banner")).to_contain_text("status=failed")
    expect(page.locator("#rc-btn-cancel")).to_be_disabled(timeout=10_000)
    expect(page.locator("#rc-log-tail")).not_to_have_text("(no log yet)")

    assert validate_owner["generation"] != local_owner["generation"]
    assert page_errors == []
    assert failed_callbacks == []
    assert callback_statuses
    assert all(status < 400 for status in callback_statuses)


def test_fresh_page_mock_submit_seam_records_failed_generation(
    page: Page,
    action_hub: tuple[str, Path],
) -> None:
    """A fresh SLURM-shaped click reaches the injected seam and stays visible."""
    hub_url, sandbox = action_hub
    page_errors, failed_callbacks, callback_statuses = (
        _capture_browser_failures(page)
    )
    pipeline, input_dir, output_dir = _prepare_action_paths(
        sandbox,
        name="MockSubmitAction",
    )
    page.goto(hub_url + "/")
    page.wait_for_selector("#rc-btn-run")
    _set_action_controls(
        page,
        pipeline=pipeline,
        input_dir=input_dir,
        output_dir=output_dir,
        modes=["slurm"],
    )
    page.locator("#rc-btn-run").click()

    owner = _wait_for_owner_status(output_dir, terminal=True)
    assert owner["mode"] == "slurm"
    assert owner["status"] == "failed"
    expect(page.locator("#rc-action-feedback")).to_contain_text(
        str(owner["generation"])
    )
    expect(page.locator("#rc-status-banner")).to_contain_text(
        f"generation={owner['generation']}"
    )
    expect(page.locator("#rc-status-banner")).to_contain_text("status=failed")
    expect(page.locator("#rc-btn-cancel")).to_be_disabled(timeout=10_000)

    assert page_errors == []
    assert failed_callbacks == []
    assert callback_statuses
    assert all(status < 400 for status in callback_statuses)


def test_local_action_callback_network_failure_becomes_visible(
    page: Page,
    action_hub: tuple[str, Path],
) -> None:
    """A dropped action callback cannot leave a silent enabled-button no-op."""
    hub_url, sandbox = action_hub
    page_errors, failed_callbacks, _callback_statuses = (
        _capture_browser_failures(page)
    )
    pipeline, input_dir, output_dir = _prepare_action_paths(
        sandbox,
        name="DroppedLocalAction",
    )
    aborted = False

    def _abort_action_callback(route: Route) -> None:
        nonlocal aborted
        request = route.request
        payload = request.post_data_json if request.post_data else {}
        output = json.dumps(payload.get("output", ""))
        if (
            not aborted
            and request.method == "POST"
            and "rc-store-action-result.data" in output
        ):
            aborted = True
            route.abort("failed")
            return
        route.continue_()

    page.route("**/_dash-update-component", _abort_action_callback)
    page.goto(hub_url + "/")
    page.wait_for_selector("#rc-btn-run")
    _set_action_controls(
        page,
        pipeline=pipeline,
        input_dir=input_dir,
        output_dir=output_dir,
        modes=["local"],
    )
    page.locator("#rc-btn-run").click()

    expect(page.locator("#rc-action-feedback")).to_contain_text(
        "callback request failed",
        timeout=10_000,
    )
    assert aborted is True
    assert not gui_launch_owner_path(output_dir).exists()
    assert page_errors == []
    assert any("_dash-update-component" in url for url in failed_callbacks)
