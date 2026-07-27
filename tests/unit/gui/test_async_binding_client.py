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
    const acceptedJobs = [];
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
        if (response.accepted_job_id) {{
            acceptedJobs.push(response.accepted_job_id);
        }}
        if (response.throw_error) {{
            throw new TypeError(response.throw_error);
        }}
        return {{
            ok: response.ok,
            status: response.status,
            json: async () => {{
                if (response.json_error) {{
                    throw new SyntaxError(response.json_error);
                }}
                return response.body;
            }},
        }};
    }};
    (async () => {{
        const result = await {invocation};
        process.stdout.write(JSON.stringify({{
            result: result === noUpdate ? "NO_UPDATE" : result,
            calls,
            navigations,
            acceptedJobs,
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


def test_response_lost_after_submission_is_not_terminal_or_authoritative() -> None:
    """A lost POST acknowledgement cannot prove non-publication."""
    source = async_binding_callback_source(
        api_url="/sandbox/api/viewer/output-root",
        redirect_url="/results/",
        selection_required=True,
    )
    observed = _evaluate_callback(
        source,
        invocation='callback(1, {path: "accepted-output"}, null)',
        responses=[
            {
                # The server accepted this job before the transport lost the
                # response, so the browser never learns its job identifier.
                "accepted_job_id": "accepted-but-unacknowledged",
                "throw_error": "connection closed before acknowledgement",
            }
        ],
    )

    assert observed["acceptedJobs"] == ["accepted-but-unacknowledged"]
    assert observed["result"]["status"] == "unknown"
    assert observed["result"]["submission_outcome"] == "unknown"
    assert observed["result"]["job"]["status"] == "unknown"
    assert observed["result"]["job"]["authoritative"] is False
    assert observed["result"]["job"]["terminal"] is False
    assert "unchanged" not in observed["result"]["job"]["detail"]
    assert observed["navigations"] == []


def test_response_lost_submission_retains_prior_authoritative_active_job() -> None:
    """An ambiguous retry does not replace an already monitored job."""
    source = async_binding_callback_source(
        api_url="/sandbox/api/viewer/output-root",
        redirect_url="/analysis/",
        selection_required=False,
    )
    active = {
        "job_id": "authoritative-active",
        "poll_path": (
            "/sandbox/api/viewer/output-root/jobs/authoritative-active"
        ),
        "cancel_path": (
            "/sandbox/api/viewer/output-root/jobs/authoritative-active"
        ),
        "redirect_url": "/analysis/",
        "job": {
            "job_id": "authoritative-active",
            "status": "running",
            "phase": "measurements",
            "completed": 20,
            "total": 100,
            "terminal": False,
        },
    }
    observed = _evaluate_callback(
        source,
        invocation=f"callback(1, {json.dumps(active)})",
        responses=[
            {
                "accepted_job_id": "possible-superseding-job",
                "throw_error": "network response lost",
            }
        ],
    )

    assert observed["acceptedJobs"] == ["possible-superseding-job"]
    assert observed["result"]["job"] == active["job"]
    assert observed["result"]["poll_path"] == active["poll_path"]
    assert observed["result"]["submission_outcome"] == "unknown"
    assert observed["result"]["submission_error"] == (
        "TypeError: network response lost"
    )
    assert observed["result"]["job"]["terminal"] is False
    assert "superseded_job_id" not in observed["result"]


def test_proxy_rejection_does_not_assert_submission_was_not_accepted() -> None:
    """A gateway response cannot authoritatively terminalize the POST."""
    source = async_binding_callback_source(
        api_url="/sandbox/api/viewer/output-root",
        redirect_url="/results/",
        selection_required=False,
    )
    observed = _evaluate_callback(
        source,
        invocation="callback(1, null)",
        responses=[
            {
                "ok": False,
                "status": 502,
                "body": {"error": "upstream response unavailable"},
            }
        ],
    )

    assert observed["result"]["status"] == "unknown"
    assert observed["result"]["submission_outcome"] == "unknown"
    assert observed["result"]["job"]["terminal"] is False
    assert observed["result"]["job"]["authoritative"] is False
    assert "unchanged" not in observed["result"]["job"]["detail"]


@pytest.mark.parametrize(
    "response",
    [
        {"ok": True, "status": 200, "body": {}},
        {
            "ok": True,
            "status": 200,
            "json_error": "unexpected end of JSON input",
        },
    ],
)
def test_incomplete_200_does_not_redirect_as_success(
    response: dict[str, object],
) -> None:
    """Only a complete synchronous-success contract permits navigation."""
    source = async_binding_callback_source(
        api_url="/sandbox/api/viewer/output-root",
        redirect_url="/results/",
        selection_required=False,
    )
    observed = _evaluate_callback(
        source,
        invocation="callback(1, null)",
        responses=[response],
    )

    assert observed["result"]["status"] == "unknown"
    assert observed["result"]["submission_outcome"] == "unknown"
    assert observed["result"]["job"]["terminal"] is False
    assert observed["navigations"] == []


def test_complete_synchronous_success_contract_redirects() -> None:
    """The backward-compatible 200 contract remains explicitly supported."""
    source = async_binding_callback_source(
        api_url="/sandbox/api/viewer/output-root",
        redirect_url="/results/",
        selection_required=False,
    )
    observed = _evaluate_callback(
        source,
        invocation="callback(1, null)",
        responses=[
            {
                "ok": True,
                "status": 200,
                "body": {
                    "status": "ok",
                    "abs_path": "/sandbox/results/output",
                    "snapshot": {
                        "processing_fingerprint": "abcdef012345",
                    },
                },
            }
        ],
    )

    assert observed["result"]["job"]["status"] == "succeeded"
    assert observed["result"]["job"]["terminal"] is True
    assert observed["navigations"] == ["/results/"]


@pytest.mark.parametrize(
    "body",
    [
        {
            "poll_path": "/sandbox/api/viewer/output-root/jobs/job-1",
            "cancel_path": "/sandbox/api/viewer/output-root/jobs/job-1",
            "job": {
                "job_id": "job-1",
                "status": "running",
                "terminal": False,
            },
        },
        {
            "job_id": "job-1",
            "poll_path": "/sandbox/api/viewer/output-root/jobs/job-1",
            "cancel_path": "/sandbox/api/viewer/output-root/jobs/job-1",
            "job": {
                "job_id": "job-1",
                "terminal": False,
            },
        },
        {
            "job_id": "job-1",
            "poll_path": "/sandbox/api/viewer/output-root/jobs/job-1",
            "cancel_path": "/sandbox/api/viewer/output-root/jobs/job-2",
            "job": {
                "job_id": "job-1",
                "status": "queued",
                "terminal": False,
            },
        },
        {
            "job_id": "job-1",
            "poll_path": "/sandbox/api/viewer/output-root/jobs/job-1",
            "cancel_path": "/sandbox/api/viewer/output-root/jobs/job-1",
            "job": {
                "job_id": "different-job",
                "status": "queued",
                "terminal": False,
            },
        },
    ],
)
def test_incomplete_202_contract_retains_unknown_outcome(
    body: dict[str, object],
) -> None:
    """A 202 must identify one active job and its consistent control path."""
    source = async_binding_callback_source(
        api_url="/sandbox/api/viewer/output-root",
        redirect_url="/results/",
        selection_required=False,
    )
    observed = _evaluate_callback(
        source,
        invocation="callback(1, null)",
        responses=[{"ok": True, "status": 202, "body": body}],
    )

    assert observed["result"]["status"] == "unknown"
    assert observed["result"]["submission_outcome"] == "unknown"
    assert observed["result"]["job"]["terminal"] is False
    assert observed["navigations"] == []


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (200, {}),
        (
            202,
            {
                "job_id": "possible-new-job",
                "poll_path": (
                    "/sandbox/api/viewer/output-root/jobs/possible-new-job"
                ),
                "cancel_path": (
                    "/sandbox/api/viewer/output-root/jobs/possible-new-job"
                ),
                "job": {
                    "job_id": "possible-new-job",
                    "terminal": False,
                },
            },
        ),
    ],
)
def test_incomplete_ok_response_does_not_replace_prior_authority(
    status: int,
    body: dict[str, object],
) -> None:
    """Malformed success-shaped responses leave the known monitor intact."""
    source = async_binding_callback_source(
        api_url="/sandbox/api/viewer/output-root",
        redirect_url="/analysis/",
        selection_required=False,
    )
    active = {
        "job_id": "known-job",
        "poll_path": "/sandbox/api/viewer/output-root/jobs/known-job",
        "cancel_path": "/sandbox/api/viewer/output-root/jobs/known-job",
        "job": {
            "job_id": "known-job",
            "status": "running",
            "phase": "inventory",
            "terminal": False,
        },
    }
    observed = _evaluate_callback(
        source,
        invocation=f"callback(1, {json.dumps(active)})",
        responses=[{"ok": True, "status": status, "body": body}],
    )

    assert observed["result"]["job"] == active["job"]
    assert observed["result"]["job_id"] == "known-job"
    assert observed["result"]["poll_path"] == active["poll_path"]
    assert observed["result"]["submission_outcome"] == "unknown"
    assert observed["navigations"] == []


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


@pytest.mark.parametrize("status_code", [502, 503, 504])
def test_transient_poll_failure_retains_active_job_then_reaches_success(
    status_code: int,
) -> None:
    """A proxy failure is not authoritative terminal binding state."""
    source = binding_poll_callback_source()
    active = {
        "job_id": "job-transient",
        "poll_path": "/sandbox/api/viewer/output-root/jobs/job-transient",
        "cancel_path": "/sandbox/api/viewer/output-root/jobs/job-transient",
        "redirect_url": "/results/",
        "job": {
            "job_id": "job-transient",
            "status": "running",
            "phase": "measurements",
            "detail": "Loading measurements.",
            "completed": 10,
            "total": 100,
            "terminal": False,
        },
    }
    unavailable = _evaluate_callback(
        source,
        invocation=f"callback(1, {json.dumps(active)})",
        responses=[
            {
                "ok": False,
                "status": status_code,
                "body": {"error": "proxy temporarily unavailable"},
            }
        ],
    )

    assert unavailable["result"]["job"] == active["job"]
    assert unavailable["result"]["job"]["terminal"] is False
    assert unavailable["result"]["poll_error"] == (
        "proxy temporarily unavailable"
    )
    assert unavailable["navigations"] == []

    succeeded = _evaluate_callback(
        source,
        invocation=f"callback(2, {json.dumps(unavailable['result'])})",
        responses=[
            {
                "ok": True,
                "status": 200,
                "body": {
                    "status": "succeeded",
                    "job": {
                        "job_id": "job-transient",
                        "status": "succeeded",
                        "phase": "complete",
                        "detail": "Results and Analysis binding published.",
                        "terminal": True,
                    },
                },
            }
        ],
    )
    assert succeeded["result"]["job"]["status"] == "succeeded"
    assert succeeded["result"]["poll_error"] is None
    assert succeeded["navigations"] == ["/results/"]


def test_concurrent_poll_ticks_issue_only_one_get() -> None:
    """The browser-side in-flight fence bounds polling to one GET per job."""
    if _NODE is None:
        pytest.skip("Node.js is required for JavaScript callback behavior tests")
    source = binding_poll_callback_source()
    active = {
        "job_id": "job-overlap",
        "poll_path": "/sandbox/api/viewer/output-root/jobs/job-overlap",
        "cancel_path": "/sandbox/api/viewer/output-root/jobs/job-overlap",
        "redirect_url": "/results/",
        "job": {
            "job_id": "job-overlap",
            "status": "running",
            "phase": "inventory",
            "terminal": False,
        },
    }
    script = f"""
    const callback = ({source});
    const noUpdate = Symbol("no_update");
    let calls = 0;
    let activeFetches = 0;
    let maxActiveFetches = 0;
    global.window = {{
        dash_clientside: {{no_update: noUpdate}},
        location: {{assign: () => {{}}}},
        setTimeout: setTimeout,
    }};
    global.fetch = async () => {{
        calls += 1;
        activeFetches += 1;
        maxActiveFetches = Math.max(maxActiveFetches, activeFetches);
        await new Promise((resolve) => setTimeout(resolve, 25));
        activeFetches -= 1;
        return {{
            ok: true,
            status: 200,
            json: async () => ({{
                job: {{
                    job_id: "job-overlap",
                    status: "running",
                    phase: "inventory",
                    terminal: false,
                }},
            }}),
        }};
    }};
    (async () => {{
        const state = {json.dumps(active)};
        const results = await Promise.all([
            callback(1, state),
            callback(2, state),
        ]);
        process.stdout.write(JSON.stringify({{
            calls,
            maxActiveFetches,
            noUpdateCount: results.filter((value) => value === noUpdate).length,
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
    observed = json.loads(completed.stdout)

    assert observed == {
        "calls": 1,
        "maxActiveFetches": 1,
        "noUpdateCount": 1,
    }


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
