"""Playwright E2E coverage for the default linear builder port map."""

from __future__ import annotations

import threading
import time

from playwright.sync_api import Page, expect

from tests.e2e.gui.builder.conftest import (
    _click_palette_button,
    _linear_node_titles,
    _open_builder,
    _publish_retired_store,
)


def test_linear_palette_click_adds_operation_and_port_menu_closes(
    page: Page, hub_url: str
) -> None:
    """Palette clicks add at the selected linear target; ports open menus."""

    _open_builder(page, hub_url)

    assert page.locator("#canvas-cytoscape").count() == 0
    _click_palette_button(page, "BayesShrinkCorrector")

    expect(page.locator(".linear-side-title")).to_have_text("BayesShrinkCorrector")
    assert _linear_node_titles(page) == ["InputImage", "BayesShrinkCorrector"]

    page.locator(
        'button.linear-port-image-out[aria-label="Insert after BayesShrinkCorrector"]'
    ).click()
    expect(page.locator(".linear-port-menu")).to_be_visible()
    expect(page.get_by_role("button", name="Preview here")).to_be_visible()

    page.get_by_role("button", name="Close").click()
    expect(page.locator(".linear-port-menu")).to_be_hidden()


def test_preview_here_selects_the_previewed_output_block(
    page: Page, hub_url: str
) -> None:
    """Previewing an upstream output keeps the side loader on that block."""

    _open_builder(page, hub_url)
    _click_palette_button(page, "BayesShrinkCorrector")
    _click_palette_button(page, "GaussianBlur")
    expect(page.locator(".linear-side-title")).to_have_text("GaussianBlur")

    page.locator(
        'button.linear-port-image-out[aria-label="Insert after BayesShrinkCorrector"]'
    ).click()
    expect(page.locator(".linear-port-menu")).to_be_visible()
    page.get_by_role("button", name="Preview here").click()

    expect(page.locator(".linear-port-menu")).to_be_hidden()
    expect(page.locator(".linear-side-title")).to_have_text("BayesShrinkCorrector")


def test_preview_mount_survives_reselection_and_marks_param_edit_stale(
    page: Page, hub_url: str
) -> None:
    """Inspector updates preserve the mount and invalidate semantic edits."""

    _open_builder(page, hub_url)
    _click_palette_button(page, "GaussianBlur")
    page.evaluate(
        """() => {
            const status = document.querySelector("#preview-status");
            if (!status) throw new Error("preview status is not mounted");
            window.__previewStatusHistory = [status.textContent];
            new MutationObserver(() => {
                window.__previewStatusHistory.push(status.textContent);
            }).observe(status, {
                childList: true,
                characterData: true,
                subtree: true,
            });
        }"""
    )
    page.locator("#btn-run-preview").click()
    expect(page.locator("#preview-status")).to_contain_text(
        "Preview complete",
        timeout=20_000,
    )
    lifecycle = page.evaluate("() => window.__previewStatusHistory")
    running_index = next(
        i for i, message in enumerate(lifecycle) if "Preview running" in message
    )
    complete_index = next(
        i for i, message in enumerate(lifecycle) if "Preview complete" in message
    )
    assert running_index < complete_index
    expect(page.locator("#inspector-preview img")).to_have_count(
        1,
        timeout=20_000,
    )
    page.evaluate(
        "() => { window.__previewMount = document.querySelector('#inspector-preview'); }"
    )

    page.locator(
        "button.linear-node-title-button", has_text="GaussianBlur"
    ).click()
    expect(page.locator("#inspector-preview img")).to_have_count(1)
    assert page.evaluate(
        "() => window.__previewMount === document.querySelector('#inspector-preview')"
    )

    page.evaluate(
        """() => {
            const sigma = document.querySelector(
                "#inspector-param-form input[type='number']"
            );
            if (!sigma) throw new Error("sigma input not mounted");
            window.dash_clientside.set_props(
                JSON.parse(sigma.id),
                { value: 3, n_blur: 1 }
            );
        }"""
    )
    expect(page.locator("#inspector-preview")).to_have_text(
        "Preview stale - run again"
    )
    expect(page.locator("#preview-status")).to_have_text(
        "Preview stale - run again"
    )
    assert page.evaluate(
        "() => window.__previewMount === document.querySelector('#inspector-preview')"
    )

    page.locator("#btn-run-preview").click()
    expect(page.locator("#preview-status")).to_contain_text(
        "Preview complete",
        timeout=20_000,
    )
    expect(page.locator("#inspector-preview img")).to_have_count(
        1,
        timeout=20_000,
    )


def test_preview_running_renders_before_blocked_compute(
    page: Page, hub_url: str
) -> None:
    """The launch response reaches the DOM before preview computation starts."""

    compute_request_seen = threading.Event()

    def delay_compute(route, request) -> None:  # noqa: ANN001
        payload = request.post_data_json
        if (
            isinstance(payload, dict)
            and payload.get("output") == "store-preview-result.data"
        ):
            compute_request_seen.set()
            time.sleep(1.2)
        route.continue_()

    page.route("**/_dash-update-component", delay_compute)
    _open_builder(page, hub_url)
    _click_palette_button(page, "GaussianBlur")
    page.evaluate(
        """() => {
            const status = document.querySelector("#preview-status");
            window.__blockedPreviewHistory = [{
                text: status.textContent,
                at: performance.now(),
            }];
            new MutationObserver(() => {
                window.__blockedPreviewHistory.push({
                    text: status.textContent,
                    at: performance.now(),
                });
            }).observe(status, {
                childList: true,
                characterData: true,
                subtree: true,
            });
        }"""
    )
    page.locator("#btn-run-preview").click()

    expect(page.locator("#preview-status")).to_contain_text(
        "Preview complete",
        timeout=20_000,
    )
    assert compute_request_seen.is_set()
    history = page.evaluate("() => window.__blockedPreviewHistory")
    running = next(item for item in history if "Preview running" in item["text"])
    complete = next(item for item in history if "Preview complete" in item["text"])
    assert complete["at"] - running["at"] >= 1_000


def test_late_terminal_result_cannot_overwrite_newer_running_request(
    page: Page, hub_url: str
) -> None:
    """The browser gate rejects A after B is already the current request."""

    captured_requests: list[dict] = []

    def capture_and_abort_compute(route, request) -> None:  # noqa: ANN001
        payload = request.post_data_json
        if (
            isinstance(payload, dict)
            and payload.get("output") == "store-preview-result.data"
        ):
            inputs = payload.get("inputs")
            if isinstance(inputs, list):
                work = next(
                    (
                        item.get("value")
                        for item in inputs
                        if item.get("id") == "store-preview-work"
                    ),
                    None,
                )
                if isinstance(work, dict):
                    captured_requests.append(work)
            route.abort()
            return
        route.continue_()

    page.route("**/_dash-update-component", capture_and_abort_compute)
    _open_builder(page, hub_url)
    _click_palette_button(page, "GaussianBlur")

    page.locator("#btn-run-preview").click()
    for _ in range(20):
        if len(captured_requests) >= 1:
            break
        page.wait_for_timeout(50)
    assert len(captured_requests) == 1
    page.locator("#btn-run-preview").click()
    for _ in range(20):
        if len(captured_requests) >= 2:
            break
        page.wait_for_timeout(50)
    assert len(captured_requests) == 2
    request_a, request_b = captured_requests
    assert request_a["request_id"] != request_b["request_id"]

    page.evaluate(
        """(request) => {
            window.dash_clientside.set_props("store-preview-status", {
                data: {
                    state: "running",
                    message: "B remains authoritative",
                    request_id: request.request_id,
                    session_id: request.session_id,
                    pipeline_revision: request.pipeline_revision,
                },
            });
        }""",
        request_b,
    )
    expect(page.locator("#preview-status")).to_have_text(
        "B remains authoritative"
    )

    def terminal(request: dict, message: str) -> dict:
        return {
            "schema_version": 1,
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "pipeline_revision": request["pipeline_revision"],
            "state": "error",
            "message": message,
            "intermediate_keys": None,
            "preview_snapshot": None,
        }

    page.evaluate(
        """(result) => {
            window.dash_clientside.set_props(
                "store-preview-result",
                {data: result}
            );
        }""",
        terminal(request_a, "STALE A MUST NOT RENDER"),
    )
    page.wait_for_timeout(250)
    expect(page.locator("#preview-status")).to_have_text(
        "B remains authoritative"
    )

    page.evaluate(
        """(result) => {
            window.dash_clientside.set_props(
                "store-preview-result",
                {data: result}
            );
        }""",
        terminal(request_b, "B terminal accepted"),
    )
    expect(page.locator("#preview-status")).to_have_text("B terminal accepted")


def test_same_preview_work_is_consumed_once_with_reversed_responses(
    page: Page, hub_url: str
) -> None:
    """Duplicate WORK returns first but cannot publish a second generation."""

    captured_payloads: list[dict] = []

    def capture_initial_work(route, request) -> None:  # noqa: ANN001
        payload = request.post_data_json
        if (
            isinstance(payload, dict)
            and payload.get("output") == "store-preview-result.data"
            and any(
                isinstance(item.get("value"), dict)
                for item in payload.get("inputs", [])
            )
        ):
            captured_payloads.append(payload)
            route.abort()
            return
        route.continue_()

    page.route("**/_dash-update-component", capture_initial_work)
    _open_builder(page, hub_url)
    _click_palette_button(page, "GaussianBlur")
    page.locator("#btn-run-preview").click()
    for _ in range(20):
        if captured_payloads:
            break
        page.wait_for_timeout(50)
    assert len(captured_payloads) == 1
    request_payload = captured_payloads[0]
    work_request = request_payload["inputs"][0]["value"]
    page.unroute("**/_dash-update-component", capture_initial_work)

    responses = page.evaluate(
        """async (payload) => {
            const endpoint = new URL(
                "_dash-update-component",
                window.location.href
            ).toString();
            const send = async () => {
                const response = await fetch(endpoint, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify(payload),
                });
                return {
                    at: performance.now(),
                    body: await response.json(),
                };
            };
            const first = send();
            await new Promise((resolve) => setTimeout(resolve, 10));
            const duplicate = send();
            return Promise.all([first, duplicate]);
        }""",
        request_payload,
    )
    first, duplicate = responses
    assert duplicate["at"] < first["at"]
    assert duplicate["body"] == {"multi": True, "response": {}}
    result = first["body"]["response"]["store-preview-result"]["data"]
    assert result["state"] == "complete"
    assert result["preview_snapshot"]["preview_generation"] == 1

    page.evaluate(
        """(result) => {
            window.dash_clientside.set_props(
                "store-preview-result",
                {data: result}
            );
        }""",
        result,
    )
    expect(page.locator("#preview-status")).to_contain_text("Preview complete")
    expect(page.locator("#inspector-preview img")).to_have_count(1)
    terminal_text = page.locator("#preview-status").inner_text()

    page.evaluate(
        """() => {
            window.dash_clientside.set_props(
                "store-preview-work",
                {data: null}
            );
        }"""
    )
    page.wait_for_timeout(100)
    page.evaluate(
        """(request) => {
            window.dash_clientside.set_props(
                "store-preview-work",
                {data: request}
            );
        }""",
        work_request,
    )
    page.wait_for_timeout(300)
    expect(page.locator("#preview-status")).to_have_text(terminal_text)

    page.locator(
        "button.linear-node-title-button", has_text="InputImage"
    ).click()
    page.locator(
        "button.linear-node-title-button", has_text="GaussianBlur"
    ).click()
    expect(page.locator("#inspector-preview img")).to_have_count(1)


def test_preview_running_transitions_to_error(
    page: Page, hub_url: str
) -> None:
    """A failed computation retains its visible running-to-error lifecycle."""

    _open_builder(page, hub_url)
    _click_palette_button(page, "GaussianBlur")
    page.locator("#btn-run-preview").click()
    expect(page.locator("#preview-status")).to_contain_text(
        "Preview complete",
        timeout=20_000,
    )
    preview_image = page.locator("#inspector-preview img")
    expect(preview_image).to_have_count(1)
    published_src = preview_image.get_attribute("src")
    assert published_src
    page.evaluate(
        """() => {
            const status = document.querySelector("#preview-status");
            if (!status) throw new Error("preview status is not mounted");
            window.__previewStatusHistory = [status.textContent];
            new MutationObserver(() => {
                window.__previewStatusHistory.push(status.textContent);
            }).observe(status, {
                childList: true,
                characterData: true,
                subtree: true,
            });
            window.dash_clientside.set_props(
                "store-image-path",
                {data: "/definitely/missing/phenotypic-preview.tiff"}
            );
        }"""
    )
    page.locator("#btn-run-preview").click()

    expect(page.locator("#preview-status")).to_have_class(
        "small text-danger mt-2",
        timeout=20_000,
    )
    lifecycle = page.evaluate("() => window.__previewStatusHistory")
    running_index = next(
        i for i, message in enumerate(lifecycle) if "Preview running" in message
    )
    error_index = next(
        i
        for i, message in enumerate(lifecycle)
        if message
        and "Preview running" not in message
        and i > running_index
    )
    assert running_index < error_index
    expect(preview_image).to_have_attribute("src", published_src)


def test_linear_map_source_and_connectors_align(page: Page, hub_url: str) -> None:
    """The source node body and connector lines stay on the port grid."""

    _open_builder(page, hub_url)
    _click_palette_button(page, "GaussianBlur")
    _click_palette_button(page, "FilamentousFungiDetector")
    expect(page.locator(".linear-node-card")).to_have_count(3)

    geometry = page.evaluate(
        """() => {
            const rect = (element) => {
                if (!element) return null;
                const box = element.getBoundingClientRect();
                return {
                    x: box.x,
                    y: box.y,
                    width: box.width,
                    height: box.height,
                };
            };
            const cards = Array.from(document.querySelectorAll('.linear-node-card'));
            const connectorLines = Array.from(
                document.querySelectorAll('.linear-connector-line')
            );
            const terminalLine = document.querySelector('.linear-terminal-line');
            const floating = document.querySelector('.linear-floating-port');
            const firstOut = rect(cards[0].querySelector('.linear-port-image-out'));
            const secondOut = rect(cards[1].querySelector('.linear-port-image-out'));
            const thirdOut = rect(cards[2].querySelector('.linear-port-image-out'));
            const connectorA = rect(connectorLines[0]);
            const connectorB = rect(connectorLines[1]);
            const terminal = rect(terminalLine);
            const floatingRect = rect(floating);
            return {
                inputChildClasses: Array.from(cards[0].children).map(
                    (element) => element.className
                ),
                inputBodyWidth: cards[0].querySelector(
                    '.linear-node-body'
                ).getBoundingClientRect().width,
                trackGap: getComputedStyle(
                    document.querySelector('.linear-map-track')
                ).gap,
                firstGap: connectorA.x - (firstOut.x + firstOut.width),
                secondGap: connectorB.x - (secondOut.x + secondOut.width),
                terminalGap: terminal.x - (thirdOut.x + thirdOut.width),
                floatingGap: floatingRect.x - (terminal.x + terminal.width),
            };
        }"""
    )

    assert geometry["inputChildClasses"][0].endswith("linear-node-port-placeholder")
    assert geometry["inputBodyWidth"] > 100
    assert geometry["trackGap"] == "0px"
    assert abs(geometry["firstGap"]) <= 1
    assert abs(geometry["secondGap"]) <= 1
    assert abs(geometry["terminalGap"]) <= 1
    assert abs(geometry["floatingGap"]) <= 1


def test_mobile_limited_mode_disables_edit_controls(page: Page, hub_url: str) -> None:
    """Mobile builder keeps inspection/help available while disabling edits."""

    page.set_viewport_size({"width": 375, "height": 760})
    _open_builder(page, hub_url)

    state = page.evaluate(
        """() => {
            const palette = document.querySelector('.palette-button');
            const port = document.querySelector('.linear-port-button');
            const save = document.querySelector('#btn-save');
            const help = document.querySelector('.linear-help-button');
            const zoom = document.querySelector('.linear-map-zoom-control');
            const label = document.querySelector('#input-node-label');
            return {
                paletteDisabled: palette ? palette.disabled : null,
                paletteAria: palette ? palette.getAttribute('aria-disabled') : null,
                portDisabled: port ? port.disabled : null,
                portAria: port ? port.getAttribute('aria-disabled') : null,
                saveDisabled: save ? save.disabled : null,
                helpDisabled: help ? help.disabled : null,
                helpAria: help ? help.getAttribute('aria-disabled') : null,
                zoomDisabled: zoom ? zoom.disabled : null,
                zoomAria: zoom ? zoom.getAttribute('aria-disabled') : null,
                labelReadOnly: label ? label.readOnly : null,
            };
        }"""
    )

    assert state == {
        "paletteDisabled": True,
        "paletteAria": "true",
        "portDisabled": False,
        "portAria": None,
        "saveDisabled": True,
        "helpDisabled": False,
        "helpAria": None,
        "zoomDisabled": False,
        "zoomAria": None,
        "labelReadOnly": True,
    }


def test_mobile_limited_mode_allows_pipeline_inspection_not_editing(
    page: Page, hub_url: str
) -> None:
    """Mobile can drill into embedded pipelines while edit controls stay disabled."""

    page.set_viewport_size({"width": 1280, "height": 720})
    _open_builder(page, hub_url)
    _click_palette_button(page, "FilamentousFungiDetector")
    # Open the inspector slide-over (closed by default) to reach its ports.
    page.locator("#btn-inspector-slideover-toggle").click()
    page.locator(
        'button.linear-side-param-port[aria-label="Fill inoculum_detector"]'
    ).click()
    page.locator("#btn-new-pipeline-node").click()
    expect(page.locator("#breadcrumb button")).to_have_text("Pipeline")

    page.locator("#breadcrumb button").click()
    # Close the inspector so it doesn't overlay the canvas node we select.
    page.locator("#btn-inspector-slideover-toggle").click()
    page.locator(
        "button.linear-node-title-button", has_text="FilamentousFungiDetector"
    ).click()
    expect(page.locator(".linear-side-value-label")).to_have_text("ImagePipeline")
    expect(page.locator(".linear-side-drill-action")).to_have_text("Open")

    page.set_viewport_size({"width": 375, "height": 760})
    page.wait_for_function(
        """() => {
            const drill = document.querySelector('.linear-side-drill-action');
            const save = document.querySelector('#btn-save');
            const port = document.querySelector('.linear-port-button');
            return drill && save && port && !drill.disabled && save.disabled
                && !port.disabled;
        }"""
    )
    state = page.evaluate(
        """() => {
            const drill = document.querySelector('.linear-side-drill-action');
            const palette = document.querySelector('.palette-button');
            const port = document.querySelector('.linear-port-button');
            const save = document.querySelector('#btn-save');
            const label = document.querySelector('#input-node-label');
            return {
                drillDisabled: drill.disabled,
                drillAria: drill.getAttribute('aria-disabled'),
                paletteDisabled: palette.disabled,
                portDisabled: port.disabled,
                portAria: port.getAttribute('aria-disabled'),
                saveDisabled: save.disabled,
                labelReadOnly: label.readOnly,
            };
        }"""
    )

    assert state == {
        "drillDisabled": False,
        "drillAria": None,
        "paletteDisabled": True,
        "portDisabled": False,
        "portAria": None,
        "saveDisabled": True,
        "labelReadOnly": True,
    }

    # The drill action lives in the inspector (closed by default at this
    # mobile width); dispatch the click straight to it — this test verifies
    # drilling works in limited mode, not the slide-over's hit-testing.
    page.locator(".linear-side-drill-action").dispatch_event("click")
    expect(page.locator("#breadcrumb button")).to_have_text("Pipeline")


def test_linear_zoom_controls_are_view_only_and_keep_ports_clickable(
    page: Page, hub_url: str
) -> None:
    """Zoom controls scale only the map view and keep port buttons usable."""

    _open_builder(page, hub_url)
    for _ in range(5):
        _click_palette_button(page, "GaussianBlur")
    expect(page.locator(".linear-node-card")).to_have_count(6)
    expect(page.locator(".linear-side-title")).to_have_text("GaussianBlur")
    titles_before = _linear_node_titles(page)

    page.locator("#linear-zoom-in").click()
    zoomed = page.evaluate(
        """() => {
            const viewport = document.querySelector('#linear-map-container');
            const content = document.querySelector('.linear-map-zoom-content');
            return {
                zoom: Number(viewport.dataset.linearZoom),
                transform: content.style.transform,
                titles: Array.from(
                    document.querySelectorAll('.linear-node-title-button')
                ).map((el) => el.textContent),
            };
        }"""
    )
    assert zoomed["zoom"] > 1
    assert "scale(" in zoomed["transform"]
    assert zoomed["titles"] == titles_before

    # The floating "+" port sits at the far right of a long chain, where the
    # closed inspector tab's edge sliver overlaps it. Dispatch the click
    # straight to the element so it isn't routed to the tab on top; the port
    # is present and functional (the overlap is intended chrome).
    page.locator("button.linear-floating-port").dispatch_event("click")
    expect(page.locator(".linear-port-menu")).to_be_visible()
    page.get_by_role("button", name="Close").click()
    expect(page.locator(".linear-port-menu")).to_be_hidden()

    page.locator("#linear-zoom-fit").click()
    fit_zoom = page.evaluate(
        "() => Number(document.querySelector('#linear-map-container').dataset.linearZoom)"
    )
    assert fit_zoom <= 1

    page.locator("#linear-zoom-reset").click()
    reset = page.evaluate(
        """() => ({
            zoom: Number(document.querySelector('#linear-map-container').dataset.linearZoom),
            titles: Array.from(
                document.querySelectorAll('.linear-node-title-button')
            ).map((el) => el.textContent),
        })"""
    )
    assert reset["zoom"] == 1
    assert reset["titles"] == titles_before


def test_new_pipeline_side_target_drills_and_breadcrumb_returns(
    page: Page, hub_url: str
) -> None:
    """A side target filled with + New Pipeline drills into a DAG scope."""

    _open_builder(page, hub_url)
    _click_palette_button(page, "FilamentousFungiDetector")
    expect(page.locator(".linear-side-title")).to_have_text(
        "FilamentousFungiDetector"
    )

    # Open the inspector slide-over (closed by default) to reach its ports.
    page.locator("#btn-inspector-slideover-toggle").click()
    page.locator(
        'button.linear-side-param-port[aria-label="Fill inoculum_detector"]'
    ).click()
    page.locator("#btn-new-pipeline-node").click()

    expect(page.locator("#breadcrumb button")).to_have_text("Pipeline")
    expect(page.locator(".linear-node-card")).to_have_count(1)
    assert _linear_node_titles(page) == ["InputImage"]

    _click_palette_button(page, "GaussianBlur")
    expect(page.locator(".linear-node-card")).to_have_count(2)
    assert _linear_node_titles(page) == ["InputImage", "GaussianBlur"]

    page.locator("#breadcrumb button").click()
    # Close the inspector slide-over so it doesn't overlay (and intercept
    # clicks on) the canvas node we're about to select.
    page.locator("#btn-inspector-slideover-toggle").click()
    page.locator(
        "button.linear-node-title-button", has_text="FilamentousFungiDetector"
    ).click()
    expect(page.locator(".linear-side-title")).to_have_text(
        "FilamentousFungiDetector"
    )
    expect(page.locator(".linear-side-value-label")).to_have_text("ImagePipeline")


def test_scalar_aux_replace_keeps_operation_off_main_spine(
    page: Page,
    hub_url: str,
) -> None:
    """Replace consumes the exact scalar side target across a palette click."""

    _open_builder(page, hub_url)
    _click_palette_button(page, "FilamentousFungiDetector")
    page.locator("#btn-inspector-slideover-toggle").click()
    page.locator(
        'button.linear-side-param-port[aria-label="Fill inoculum_detector"]'
    ).click()
    page.get_by_role("button", name="OtsuDetector", exact=True).click()
    expect(page.locator(".linear-side-value-label")).to_have_text("OtsuDetector")
    assert _linear_node_titles(page) == [
        "InputImage",
        "FilamentousFungiDetector",
    ]

    page.get_by_role("button", name="Replace", exact=True).click()
    page.get_by_role("button", name="MeanDetector", exact=True).click()

    expect(page.locator(".linear-side-value-label")).to_have_text("MeanDetector")
    assert _linear_node_titles(page) == [
        "InputImage",
        "FilamentousFungiDetector",
    ]


def test_retired_drag_and_wire_stores_are_inert(page: Page, hub_url: str) -> None:
    """Old drag/drop and wire stores remain mounted but do not mutate state."""

    _open_builder(page, hub_url)
    assert _linear_node_titles(page) == ["InputImage"]

    _publish_retired_store(
        page,
        "store-palette-drop",
        {
            "kind": "block_create",
            "class_name": "GaussianBlur",
            "x": 0,
            "y": 0,
            "ts": 1,
        },
    )
    _publish_retired_store(
        page,
        "store-edge-event",
        {
            "kind": "edge_create",
            "source_block_id": "stale-a",
            "target_block_id": "stale-b",
            "target_port": "in",
            "edge_kind": "image",
            "ts": 2,
        },
    )
    page.wait_for_timeout(750)

    assert _linear_node_titles(page) == ["InputImage"]
    assert page.locator(".linear-port-menu").count() == 0
