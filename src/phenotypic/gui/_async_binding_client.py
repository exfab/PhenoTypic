"""Shared clientside callback sources for asynchronous Results binding."""

from __future__ import annotations

import json

__all__ = [
    "async_binding_callback_source",
    "binding_cancel_callback_source",
    "binding_poll_callback_source",
]


def async_binding_callback_source(
    *,
    api_url: str,
    redirect_url: str,
    selection_required: bool,
) -> str:
    """Build the proxy-safe job-submission callback.

    The returned callback performs only the short ``POST``. Polling is driven
    by :func:`binding_poll_callback_source` so each response can update the
    shared session store and render visible progress before the job finishes.

    Args:
        api_url: Browser-visible Results binding submission endpoint.
        redirect_url: Mount reloaded after terminal success.
        selection_required: Whether the callback receives a sidebar selection.

    Returns:
        JavaScript source for :meth:`dash.Dash.clientside_callback`.
    """
    api_literal = json.dumps(api_url)
    redirect_literal = json.dumps(redirect_url)
    signature = (
        "n_clicks, selection, current"
        if selection_required
        else "n_clicks, current"
    )
    guard = (
        "if (!n_clicks || !selection) {"
        if selection_required
        else "if (!n_clicks) {"
    )
    request_body = (
        """
        const path = selection.path;
        if (!path) {
            return window.dash_clientside.no_update;
        }
        const requestBody = {path: path};
        """
        if selection_required
        else "const requestBody = {refresh: true};"
    )
    return f"""
    async function({signature}) {{
        {guard}
            return window.dash_clientside.no_update;
        }}
        {request_body}
        const shared = window.__phenotypicResultsBinding ||
            (window.__phenotypicResultsBinding = {{
                epoch: 0,
                jobId: null,
                pollingJobId: null,
                cancelledJobId: null,
            }});
        const epoch = ++shared.epoch;
        shared.cancelledJobId = null;
        const previousJob = current && current.job ? current.job : current;
        const previousActive = previousJob &&
            (previousJob.status === "queued" ||
             previousJob.status === "running");

        const failureState = function(message, kind) {{
            return {{
                status: "failed",
                redirect_url: {redirect_literal},
                job: {{
                    status: "failed",
                    phase: "failed",
                    detail: "Results binding failed.",
                    terminal: true,
                    error_kind: kind || "unavailable",
                    error: message,
                }},
            }};
        }};
        try {{
            const response = await fetch(
                {api_literal},
                {{
                    method: "POST",
                    headers: {{"Content-Type": "application/json"}},
                    body: JSON.stringify(requestBody),
                }}
            );
            const data = await response.json().catch(() => ({{}}));
            if (shared.epoch !== epoch) {{
                return window.dash_clientside.no_update;
            }}
            if (!response.ok) {{
                const message = data.error || ("HTTP " + response.status);
                return failureState(message, data.error_kind);
            }}
            const job = data && data.job ? data.job : data;
            if (response.status !== 202) {{
                const complete = Object.assign({{}}, data, {{
                    status: "succeeded",
                    redirect_url: {redirect_literal},
                    job: Object.assign({{}}, job, {{
                        status: "succeeded",
                        phase: "complete",
                        terminal: true,
                    }}),
                }});
                window.setTimeout(
                    () => window.location.assign({redirect_literal}),
                    50
                );
                return complete;
            }}
            if (!data.poll_path || !data.cancel_path) {{
                return failureState(
                    "Binding job response omitted its polling contract.",
                    "unavailable"
                );
            }}
            shared.jobId = data.job_id || job.job_id || null;
            const state = Object.assign({{}}, data, {{
                job: job,
                redirect_url: {redirect_literal},
            }});
            if (
                previousActive &&
                previousJob.job_id &&
                previousJob.job_id !== shared.jobId
            ) {{
                state.superseded_job_id = previousJob.job_id;
            }}
            return state;
        }} catch (error) {{
            if (shared.epoch !== epoch) {{
                return window.dash_clientside.no_update;
            }}
            return failureState(String(error), "unavailable");
        }}
    }}
    """


def binding_poll_callback_source() -> str:
    """Build one bounded GET-per-tick binding monitor callback."""
    return """
    async function(_n_intervals, current) {
        const noUpdate = window.dash_clientside.no_update;
        if (!current) { return noUpdate; }
        const job = current.job || current;
        if (!job || (job.status !== "queued" && job.status !== "running")) {
            return noUpdate;
        }
        const pollPath = current.poll_path;
        const jobId = current.job_id || job.job_id;
        if (!pollPath || !jobId) { return noUpdate; }

        const shared = window.__phenotypicResultsBinding ||
            (window.__phenotypicResultsBinding = {
                epoch: 0,
                jobId: jobId,
                pollingJobId: null,
                cancelledJobId: null,
            });
        if (shared.jobId && shared.jobId !== jobId) { return noUpdate; }
        shared.jobId = jobId;
        if (shared.pollingJobId === jobId) { return noUpdate; }
        shared.pollingJobId = jobId;
        try {
            const response = await fetch(
                pollPath,
                {method: "GET", cache: "no-store"}
            );
            const data = await response.json().catch(() => ({}));
            if (shared.jobId !== jobId) { return noUpdate; }
            if (
                shared.cancelledJobId === jobId &&
                data && data.job &&
                (data.job.status === "queued" ||
                 data.job.status === "running")
            ) {
                return noUpdate;
            }
            if (!response.ok) {
                const message = data.error || ("HTTP " + response.status);
                return Object.assign({}, current, {
                    status: "failed",
                    job: Object.assign({}, job, {
                        status: "failed",
                        phase: "failed",
                        detail: "Binding progress is unavailable.",
                        terminal: true,
                        error_kind: "unavailable",
                        error: message,
                    }),
                });
            }
            const updated = Object.assign({}, current, data, {
                job: data && data.job ? data.job : data,
                poll_error: null,
            });
            if (updated.job && updated.job.status === "succeeded") {
                const redirectUrl = current.redirect_url;
                if (redirectUrl) {
                    window.setTimeout(
                        () => window.location.assign(redirectUrl),
                        50
                    );
                }
            }
            return updated;
        } catch (error) {
            if (shared.jobId !== jobId) { return noUpdate; }
            return Object.assign({}, current, {poll_error: String(error)});
        } finally {
            if (shared.pollingJobId === jobId) {
                shared.pollingJobId = null;
            }
        }
    }
    """


def binding_cancel_callback_source() -> str:
    """Build the cooperative ``DELETE`` cancellation callback."""
    return """
    async function(n_clicks, current) {
        const noUpdate = window.dash_clientside.no_update;
        if (!n_clicks || !current) { return noUpdate; }
        const job = current.job || current;
        if (!job || (job.status !== "queued" && job.status !== "running")) {
            return noUpdate;
        }
        const cancelPath = current.cancel_path;
        const jobId = current.job_id || job.job_id;
        if (!cancelPath || !jobId) { return noUpdate; }

        const shared = window.__phenotypicResultsBinding ||
            (window.__phenotypicResultsBinding = {
                epoch: 0,
                jobId: jobId,
                pollingJobId: null,
                cancelledJobId: null,
            });
        shared.jobId = jobId;
        shared.cancelledJobId = jobId;
        try {
            const response = await fetch(
                cancelPath,
                {method: "DELETE", cache: "no-store"}
            );
            const data = await response.json().catch(() => ({}));
            if (shared.jobId !== jobId) { return noUpdate; }
            if (!response.ok) {
                shared.cancelledJobId = null;
                return Object.assign({}, current, {
                    cancel_error: data.error || ("HTTP " + response.status),
                });
            }
            const updated = Object.assign({}, current, data, {
                job: data && data.job ? data.job : data,
                cancel_error: null,
                poll_error: null,
            });
            if (updated.job && updated.job.status === "succeeded") {
                const redirectUrl = current.redirect_url;
                if (redirectUrl) {
                    window.setTimeout(
                        () => window.location.assign(redirectUrl),
                        50
                    );
                }
            }
            return updated;
        } catch (error) {
            if (shared.jobId !== jobId) { return noUpdate; }
            shared.cancelledJobId = null;
            return Object.assign({}, current, {
                cancel_error: String(error),
            });
        }
    }
    """
