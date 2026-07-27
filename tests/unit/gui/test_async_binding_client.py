"""Behavioral tests for the shared asynchronous binding browser callback."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from phenotypic.gui._async_binding_client import (
    async_binding_callback_source,
)

_NODE = shutil.which("node")


def _evaluate_callback(
    source: str,
    *,
    responses: list[dict],
    selection_required: bool,
) -> dict:
    """Execute generated callback source against deterministic fetch mocks."""
    if _NODE is None:
        pytest.skip("Node.js is required for JavaScript callback behavior tests")
    invocation = (
        'callback(1, {path: "output"})'
        if selection_required
        else "callback(1)"
    )
    script = f"""
    const assert = require("node:assert/strict");
    const callback = ({source});
    const responses = {json.dumps(responses)};
    const calls = [];
    const navigations = [];
    global.window = {{
        dash_clientside: {{no_update: Symbol("no_update")}},
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
            result,
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
def test_navigation_waits_for_polled_terminal_success(
    selection_required: bool,
) -> None:
    """A 202 and running poll cannot trigger the page reload."""
    source = async_binding_callback_source(
        api_url="/sandbox/api/viewer/output-root",
        redirect_url="/results/",
        selection_required=selection_required,
    )
    observed = _evaluate_callback(
        source,
        selection_required=selection_required,
        responses=[
            {
                "ok": True,
                "status": 202,
                "body": {"poll_path": "/sandbox/api/viewer/output-root/jobs/1"},
            },
            {
                "ok": True,
                "status": 200,
                "body": {
                    "job": {
                        "status": "running",
                        "terminal": False,
                    }
                },
            },
            {
                "ok": True,
                "status": 200,
                "body": {
                    "job": {
                        "status": "succeeded",
                        "terminal": True,
                    }
                },
            },
        ],
    )
    assert observed == {
        "result": "",
        "calls": [
            {
                "url": "/sandbox/api/viewer/output-root",
                "method": "POST",
            },
            {
                "url": "/sandbox/api/viewer/output-root/jobs/1",
                "method": "GET",
            },
            {
                "url": "/sandbox/api/viewer/output-root/jobs/1",
                "method": "GET",
            },
        ],
        "navigations": ["/results/"],
        "remaining": 0,
    }


def test_terminal_failure_is_rendered_without_navigation() -> None:
    """A failed job returns its error to Dash and leaves the page in place."""
    source = async_binding_callback_source(
        api_url="/sandbox/api/viewer/output-root",
        redirect_url="/analysis/",
        selection_required=False,
    )
    observed = _evaluate_callback(
        source,
        selection_required=False,
        responses=[
            {
                "ok": True,
                "status": 202,
                "body": {"poll_path": "/sandbox/api/viewer/output-root/jobs/2"},
            },
            {
                "ok": True,
                "status": 200,
                "body": {
                    "job": {
                        "status": "failed",
                        "terminal": True,
                        "error": "candidate analysis failed",
                    }
                },
            },
        ],
    )
    assert observed["result"] == "candidate analysis failed"
    assert observed["navigations"] == []
    assert observed["remaining"] == 0
