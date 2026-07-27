"""Shared clientside callback source for asynchronous Results binding."""

from __future__ import annotations

import json

__all__ = ["async_binding_callback_source"]

_POLL_INTERVAL_MS = 250


def async_binding_callback_source(
    *,
    api_url: str,
    redirect_url: str,
    selection_required: bool,
) -> str:
    """Build a Dash clientside callback that waits for binding publication.

    Args:
        api_url: Browser-visible Results binding submission endpoint.
        redirect_url: Page reloaded after terminal success.
        selection_required: Whether the callback receives a second
            ``selection`` argument and submits its ``path``. When ``False``,
            the callback submits ``{"refresh": true}``.

    Returns:
        JavaScript source for :meth:`dash.Dash.clientside_callback`.
    """
    api_literal = json.dumps(api_url)
    redirect_literal = json.dumps(redirect_url)
    signature = "n_clicks, selection" if selection_required else "n_clicks"
    guard = (
        "if (!n_clicks || !selection) {"
        if selection_required
        else "if (!n_clicks) {"
    )
    request_body = (
        """
            const path = selection.path;
            if (!path) { return "No sidebar selection."; }
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
        const errorMessage = function(data, status) {{
            const job = data && data.job;
            return (job && (job.error || job.detail)) ||
                (data && data.error) ||
                (job && job.status && ("Binding job " + job.status + ".")) ||
                ("HTTP " + status);
        }};
        try {{
            const acceptedResponse = await fetch(
                {api_literal},
                {{
                    method: "POST",
                    headers: {{"Content-Type": "application/json"}},
                    body: JSON.stringify(requestBody),
                }}
            );
            let data = await acceptedResponse.json().catch(() => ({{}}));
            if (!acceptedResponse.ok) {{
                return errorMessage(data, acceptedResponse.status);
            }}

            if (acceptedResponse.status === 202) {{
                const pollPath = data && data.poll_path;
                if (!pollPath) {{
                    return "Binding job response did not include poll_path.";
                }}
                while (true) {{
                    const pollResponse = await fetch(
                        pollPath,
                        {{method: "GET", cache: "no-store"}}
                    );
                    data = await pollResponse.json().catch(() => ({{}}));
                    if (!pollResponse.ok) {{
                        return errorMessage(data, pollResponse.status);
                    }}
                    const job = data && data.job ? data.job : data;
                    const status = job && job.status;
                    if (status === "succeeded") {{
                        break;
                    }}
                    if (
                        (job && job.terminal) ||
                        status === "failed" ||
                        status === "cancelled" ||
                        status === "superseded"
                    ) {{
                        return errorMessage(data, pollResponse.status);
                    }}
                    await new Promise(
                        (resolve) => window.setTimeout(
                            resolve,
                            {_POLL_INTERVAL_MS}
                        )
                    );
                }}
            }}

            window.location.assign({redirect_literal});
            return "";
        }} catch (err) {{
            return String(err);
        }}
    }}
    """
