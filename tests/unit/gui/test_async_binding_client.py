"""Behavioral tests for the shared asynchronous binding browser callbacks."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from phenotypic.gui._async_binding_client import (
    async_binding_callback_source,
    binding_cancel_callback_source,
    binding_poll_callback_source,
)

_NODE = shutil.which("node")


def _evaluate_callback(
    source: str,
    *,
    invocation: str,
    responses: list[dict],
) -> dict:
    """Execute generated callback source against deterministic fetch mocks."""
    if _NODE is None:
        pytest.skip("Node.js is required for JavaScript callback behavior tests")
    script = f"""
    const assert = require("node:assert/strict");
    const callback = ({source});
    const responses = {json.dumps(responses)};
    const calls = [];
    const navigations = [];
    const noUpdate = Symbol("no_update");
    global.window = {{
        dash_clientside: {{no_update: noUpdate}},
        location: {{assign: (url) => navigations.push(url)}},
        setTimeout: (fn, _delay) => fn(),
    }};
    global.fetch = async (url, options = {{}}) => {{
        assert.equal(
            navigations.length,
            0,
            "navigation occurred before terminal success"
        );
        calls.push({{url, method: options.method || "GET"}});
        const response = responses.shift();
        assert.ok(response, "unexpected fetch");
        return {{
            ok: response.ok,
            status: response.status,
            json: async () => response.body,
        }};
    }};
    (async () => {{
        const result = await {invocation};
        process.stdout.write(JSON.stringify({{
            result: result === noUpdate ? "NO_UPDATE" : result,
            calls,
            navigations,
            remaining: responses.length,
        }}));
    }})().catch((error) => {{
        console.error(error);
        process.exit(1);
    }});
    """
    completed = subprocess.run(
        [_NODE, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("selection_required", [False, True])
def test_submission_returns_pollable_state_without_waiting(
    selection_required: bool,
) -> None:
    """The POST callback returns after 202 and leaves polling to the interval."""
    source = async_binding_callback_source(
        api_url="/node/abc/sandbox/api/viewer/output-root",
        redirect_url="/node/abc/results/",
        selection_required=selection_required,
    )
    invocation = (
        'callback(1, {path: "large-output"}, null)'
        if selection_required
        else "callback(1, null)"
    )
    observed = _evaluate_callback(
        source,
        invocation=invocation,
        responses=[
            {
                "ok": True,
                "status": 202,
                "body": {
                    "status": "running",
                    "job_id": "job-1",
                    "poll_path": (
                        "/node/abc/sandbox/api/viewer/output-root/jobs/job-1"
                    ),
                    "cancel_path": (
                        "/node/abc/sandbox/api/viewer/output-root/jobs/job-1"
                    ),
                    "job": {
                        "job_id": "job-1",
                        "status": "running",
                        "phase": "inventory",
                        "terminal": False,
                    },
                },
            }
        ],
    )
    assert observed["calls"] == [
        {
            "url": "/node/abc/sandbox/api/viewer/output-root",
            "method": "POST",
        }
    ]
    assert observed["result"]["job"]["phase"] == "inventory"
    assert observed["result"]["redirect_url"] == "/node/abc/results/"
    assert observed["navigations"] == []
    assert observed["remaining"] == 0


def test_poll_updates_progress_then_navigates_only_on_success() -> None:
    """Each interval performs one GET; only terminal success reloads."""
    source = binding_poll_callback_source()
    active = {
        "job_id": "job-2",
        "poll_path": "/sandbox/api/viewer/output-root/jobs/job-2",
        "cancel_path": "/sandbox/api/viewer/output-root/jobs/job-2",
        "redirect_url": "/analysis/",
        "job": {
            "job_id": "job-2",
            "status": "running",
            "phase": "inventory",
            "terminal": False,
        },
    }
    running = _evaluate_callback(
        source,
        invocation=f"callback(1, {json.dumps(active)})",
        responses=[
            {
                "ok": True,
                "status": 200,
                "body": {
                    "job": {
                        "job_id": "job-2",
                        "status": "running",
                        "phase": "measurements",
                        "completed": 40,
                        "total": 100,
                        "terminal": False,
                    }
                },
            }
        ],
    )
    assert running["result"]["job"]["phase"] == "measurements"
    assert running["result"]["job"]["completed"] == 40
    assert running["navigations"] == []

    succeeded = _evaluate_callback(
        source,
        invocation=f"callback(2, {json.dumps(running['result'])})",
        responses=[
            {
                "ok": True,
                "status": 200,
                "body": {
                    "job": {
                        "job_id": "job-2",
                        "status": "succeeded",
                        "phase": "complete",
                        "terminal": True,
                    }
                },
            }
        ],
    )
    assert succeeded["result"]["job"]["status"] == "succeeded"
    assert succeeded["navigations"] == ["/analysis/"]


def test_cancel_uses_delete_and_returns_terminal_state() -> None:
    """The sidebar cancellation action uses the advertised DELETE path."""
    source = binding_cancel_callback_source()
    active = {
        "job_id": "job-3",
        "poll_path": "/prefix/sandbox/api/viewer/output-root/jobs/job-3",
        "cancel_path": "/prefix/sandbox/api/viewer/output-root/jobs/job-3",
        "redirect_url": "/prefix/results/",
        "job": {
            "job_id": "job-3",
            "status": "running",
            "phase": "measurements",
            "terminal": False,
        },
    }
    observed = _evaluate_callback(
        source,
        invocation=f"callback(1, {json.dumps(active)})",
        responses=[
            {
                "ok": True,
                "status": 200,
                "body": {
                    "status": "cancelled",
                    "job": {
                        "job_id": "job-3",
                        "status": "cancelled",
                        "phase": "cancelled",
                        "terminal": True,
                        "detail": "Results binding cancelled.",
                    },
                },
            }
        ],
    )
    assert observed["calls"] == [
        {
            "url": "/prefix/sandbox/api/viewer/output-root/jobs/job-3",
            "method": "DELETE",
        }
    ]
    assert observed["result"]["job"]["status"] == "cancelled"
    assert observed["navigations"] == []
