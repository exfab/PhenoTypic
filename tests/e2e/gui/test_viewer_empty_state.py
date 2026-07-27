"""Results viewer empty-state pathway + JS prefix injection (E2E).

When ``output_root=None`` (the default after compose_hub builds the
viewer ``ToolSession``), the viewer renders the ``results-viewer-empty-state``
placeholder. ``window.__phenotypicAppPrefix`` must be injected so the
viewer's JS can construct hub-aware DZI tile URLs once a real
``output_root`` is loaded.
"""
from __future__ import annotations

from playwright.sync_api import Page, expect


def test_empty_state_layout_renders(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/results/")
    page.wait_for_selector("#results-viewer-empty-state", timeout=10_000)
    expect(page.locator("#results-viewer-empty-state")).to_be_visible()


def test_phenotypic_app_prefix_is_injected(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/results/")
    page.wait_for_selector("#results-viewer-empty-state")
    prefix = page.evaluate("() => window.__phenotypicAppPrefix")
    assert prefix == "/results/"


def test_each_mount_injects_its_own_prefix(page: Page, hub_url: str) -> None:
    """Each mount that has prefix-dependent JS assets injects its OWN prefix.

    The viewer needs it for hub-aware DZI tile URLs; the builder needs it
    so ``point_picker.js`` can resolve the vendored OpenSeadragon icon
    assets (see ``builder/_app.py`` ``_index_string_with_prefix``). The run
    console ships no prefix-dependent JS, so it must NOT inject the global.
    """
    page.goto(hub_url + "/builder/")
    page.wait_for_selector("#shell-top-bar")
    builder_prefix = page.evaluate("() => window.__phenotypicAppPrefix || null")
    assert builder_prefix == "/builder/"

    page.goto(hub_url + "/run/")
    page.wait_for_selector("#shell-top-bar")
    run_prefix = page.evaluate("() => window.__phenotypicAppPrefix || null")
    assert run_prefix is None


def test_open_click_shows_submission_before_post_acknowledgement(
    page: Page,
    hub_url: str,
) -> None:
    """A real click renders non-authoritative progress while POST is pending."""
    job_path = "/sandbox/api/viewer/output-root/jobs/delayed-ack"

    page.add_init_script(
        """
        (() => {
            const originalFetch = window.fetch.bind(window);
            const gate = {
                intercepted: false,
                released: false,
                release: null,
            };
            window.__bindingPostGate = gate;
            window.fetch = (...args) => {
                const input = args[0];
                const options = args[1] || {};
                const url = typeof input === "string" ? input : input.url;
                const method = String(
                    options.method ||
                    (typeof input === "string" ? "GET" : input.method) ||
                    "GET"
                ).toUpperCase();
                if (
                    method !== "POST" ||
                    !url.endsWith("/sandbox/api/viewer/output-root") ||
                    gate.intercepted
                ) {
                    return originalFetch(...args);
                }
                gate.intercepted = true;
                const response = originalFetch(...args);
                return new Promise((resolve, reject) => {
                    gate.release = () => {
                        gate.released = true;
                        response.then(resolve, reject);
                    };
                });
            };
        })();
        """
    )

    poll_gets = {"count": 0}

    def _binding_route(route) -> None:
        if route.request.method == "POST":
            route.fulfill(
                status=202,
                content_type="application/json",
                body=(
                    '{"status":"queued","job_id":"delayed-ack",'
                    f'"poll_path":"{job_path}","cancel_path":"{job_path}",'
                    '"deduplicated":false,"job":{'
                    '"job_id":"delayed-ack","status":"queued",'
                    '"phase":"queued","detail":"Waiting for a worker.",'
                    '"terminal":false,'
                    '"target":"/sandbox/results/CliOutputExample"}}'
                ),
            )
        else:
            poll_gets["count"] += 1
            route.fulfill(
                status=200,
                content_type="application/json",
                body=(
                    '{"status":"running","job_id":"delayed-ack",'
                    f'"poll_path":"{job_path}","cancel_path":"{job_path}",'
                    '"job":{"job_id":"delayed-ack","status":"running",'
                    '"phase":"inventory","detail":"Scanning files.",'
                    '"completed":1,"total":100,"terminal":false,'
                    '"target":"/sandbox/results/CliOutputExample"}}'
                ),
            )

    page.route("**/sandbox/api/viewer/output-root**", _binding_route)
    page.goto(hub_url + "/results/")
    page.wait_for_selector("#results-viewer-empty-state", timeout=10_000)
    page.click(
        'button[id*="\\"path\\":\\"results\\""][id*="shell-sidebar-entry"]'
    )
    page.wait_for_selector(
        'button[id*="\\"path\\":\\"results/CliOutputExample\\""'
        '][id*="shell-sidebar-entry"]',
        timeout=5_000,
    )
    page.click(
        'button[id*="\\"path\\":\\"results/CliOutputExample\\""'
        '][id*="shell-sidebar-entry"]'
    )
    expect(page.locator("#results-viewer-empty-handoff-open")).to_be_enabled()

    page.click("#results-viewer-empty-handoff-open")
    page.wait_for_function("() => window.__bindingPostGate.intercepted")

    expect(page.locator("#shell-results-binding-panel")).to_be_visible(
        timeout=1_000
    )
    expect(page.locator("#shell-results-binding-status")).to_have_text(
        "Submitting"
    )
    expect(page.locator("#shell-results-binding-phase")).to_have_text(
        "Submitting request"
    )
    expect(page.locator("#shell-results-binding-progress-label")).to_have_text(
        "Working"
    )
    expect(page.locator("#shell-results-binding-cancel")).to_be_disabled()
    assert poll_gets["count"] == 0

    page.evaluate("() => window.__bindingPostGate.release()")
    expect(page.locator("#shell-results-binding-status")).to_have_text(
        "Active",
        timeout=5_000,
    )
    expect(page.locator("#shell-results-binding-phase")).to_have_text(
        "Scanning processing inventory"
    )
    expect(page.locator("#shell-results-binding-cancel")).to_be_enabled()
    assert poll_gets["count"] >= 1


def test_binding_progress_and_cancel_are_visible_in_shared_sidebar(
    page: Page,
    hub_url: str,
) -> None:
    """The browser renders one running poll and cancels through DELETE."""
    job_path = "/sandbox/api/viewer/output-root/jobs/synthetic-large"
    progress_gets = {"results": 0, "analysis": 0}

    # Hold the first GET in each document beyond two 400 ms interval ticks.
    # The in-page probe observes real fetch promise lifetimes, so another GET
    # would raise maxActive above one if the client-side fence were ineffective.
    page.add_init_script(
        """
        (() => {
            const originalFetch = window.fetch.bind(window);
            const probe = {
                calls: 0,
                active: 0,
                maxActive: 0,
                completed: 0,
            };
            window.__bindingFetchProbe = probe;
            window.fetch = async (...args) => {
                const input = args[0];
                const options = args[1] || {};
                const url = typeof input === "string" ? input : input.url;
                const method = String(
                    options.method ||
                    (typeof input === "string" ? "GET" : input.method) ||
                    "GET"
                ).toUpperCase();
                if (
                    method !== "GET" ||
                    !url.includes(
                        "/sandbox/api/viewer/output-root/jobs/"
                    )
                ) {
                    return originalFetch(...args);
                }
                probe.calls += 1;
                probe.active += 1;
                probe.maxActive = Math.max(
                    probe.maxActive,
                    probe.active
                );
                try {
                    if (probe.calls === 1) {
                        await new Promise(
                            (resolve) => window.setTimeout(resolve, 950)
                        );
                    }
                    return await originalFetch(...args);
                } finally {
                    probe.active -= 1;
                    probe.completed += 1;
                }
            };
        })();
        """
    )

    def _binding_route(route) -> None:
        method = route.request.method
        if method == "POST":
            route.fulfill(
                status=202,
                content_type="application/json",
                body=(
                    '{"status":"running","job_id":"synthetic-large",'
                    f'"poll_path":"{job_path}","cancel_path":"{job_path}",'
                    '"deduplicated":false,"job":{'
                    '"job_id":"synthetic-large","status":"queued",'
                    '"phase":"queued","detail":"Waiting for a worker.",'
                    '"completed":0,"total":100,"terminal":false,'
                    '"target":"/sandbox/results/CliOutputExample"}}'
                ),
            )
        elif method == "DELETE":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=(
                    '{"status":"cancelled","job_id":"synthetic-large",'
                    f'"poll_path":"{job_path}","cancel_path":"{job_path}",'
                    '"job":{"job_id":"synthetic-large",'
                    '"status":"cancelled","phase":"cancelled",'
                    '"detail":"Results binding cancelled.",'
                    '"terminal":true,'
                    '"target":"/sandbox/results/CliOutputExample"}}'
                ),
            )
        else:
            referer = route.request.headers.get("referer", "")
            on_analysis = "/analysis/" in referer
            mount = "analysis" if on_analysis else "results"
            progress_gets[mount] += 1
            phase = "indexing" if on_analysis else "inventory"
            detail = (
                "Indexing viewer rows."
                if on_analysis
                else "Scanning synthetic files."
            )
            completed = 75 if on_analysis else 25
            route.fulfill(
                status=200,
                content_type="application/json",
                body=(
                    '{"status":"running","job_id":"synthetic-large",'
                    f'"poll_path":"{job_path}",'
                    f'"cancel_path":"{job_path}",'
                    '"job":{"job_id":"synthetic-large",'
                    '"status":"running",'
                    f'"phase":"{phase}","detail":"{detail}",'
                    f'"completed":{completed},"total":100,'
                    '"terminal":false,'
                    '"target":"/sandbox/results/CliOutputExample"}}'
                ),
            )

    page.route("**/sandbox/api/viewer/output-root**", _binding_route)
    page.goto(hub_url + "/results/")
    page.wait_for_selector("#results-viewer-empty-state", timeout=10_000)
    page.click(
        'button[id*="\\"path\\":\\"results\\""][id*="shell-sidebar-entry"]'
    )
    page.wait_for_selector(
        'button[id*="\\"path\\":\\"results/CliOutputExample\\""'
        '][id*="shell-sidebar-entry"]',
        timeout=5_000,
    )
    page.click(
        'button[id*="\\"path\\":\\"results/CliOutputExample\\""'
        '][id*="shell-sidebar-entry"]'
    )
    expect(page.locator("#results-viewer-empty-handoff-open")).to_be_enabled()
    page.click("#results-viewer-empty-handoff-open")

    panel = page.locator("#shell-results-binding-panel")
    expect(panel).to_be_visible(timeout=5_000)
    expect(page.locator("#shell-results-binding-status")).to_have_text("Active")
    expect(page.locator("#shell-results-binding-phase")).to_have_text(
        "Scanning processing inventory"
    )
    expect(page.locator("#shell-results-binding-progress-label")).to_have_text(
        "25 of 100"
    )
    assert progress_gets["results"] >= 1
    page.wait_for_function(
        """() => window.__bindingFetchProbe.calls >= 2 &&
            window.__bindingFetchProbe.completed >= 2"""
    )
    results_probe = page.evaluate("() => window.__bindingFetchProbe")
    assert results_probe["maxActive"] == 1
    expect(page.locator("#shell-results-binding-cancel")).to_be_enabled()

    # The session-backed shell state survives a cross-mount hard navigation,
    # and the Analysis mount resumes the same bounded polling contract.
    page.goto(hub_url + "/analysis/")
    expect(page.locator("#shell-results-binding-panel")).to_be_visible(
        timeout=5_000
    )
    expect(page.locator("#shell-results-binding-status")).to_have_text("Active")
    expect(page.locator("#shell-results-binding-phase")).to_have_text(
        "Indexing viewer data"
    )
    expect(page.locator("#shell-results-binding-progress-label")).to_have_text(
        "75 of 100"
    )
    assert progress_gets["analysis"] >= 1
    page.wait_for_function(
        """() => window.__bindingFetchProbe.calls >= 2 &&
            window.__bindingFetchProbe.completed >= 2"""
    )
    analysis_probe = page.evaluate("() => window.__bindingFetchProbe")
    assert analysis_probe["maxActive"] == 1

    with page.expect_request(
        lambda request: request.method == "DELETE"
        and request.url.endswith(job_path)
    ):
        page.click("#shell-results-binding-cancel")
    expect(page.locator("#shell-results-binding-status")).to_have_text(
        "Cancelled",
        timeout=5_000,
    )
    expect(page.locator("#shell-results-binding-diagnostic")).to_contain_text(
        "previous Results + Analysis publication is unchanged"
    )
    expect(page.locator("#shell-results-binding-cancel")).to_be_disabled()
