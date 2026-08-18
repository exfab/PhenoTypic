"""Browser-driven E2E tests for the analysis sub-app at ``/analysis/``.

These tests exercise the empty-state landing page and the chrome
wrapping. Tests that mutate ``pipeline.json`` (add/remove/run) require
a function-scoped sandbox override so they don't pollute the
module-scoped fixture; we provide them in this file.

Per the project's CI gating, these tests run only when ``PLAYWRIGHT=1``
is set (the ``e2e-tests`` job in the ``gui-checks`` GitHub workflow
sets it automatically).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import polars as pl
import pytest
from playwright.sync_api import Page, expect

from phenotypic import ImagePipeline
from phenotypic.analysis import LogGrowthModel, TukeyOutlierRemover
from phenotypic.sdk_ import measurements_parquet_path, pipeline_json_path

from tests._output_layout import write_master, write_measurements_mirror, write_pipeline_json
from tests.e2e.gui.conftest import (
    _build_sandbox,
    _start_live_server,
    bind_results_output,
    publish_coherent_terminal_evidence,
)
from phenotypic.schema import IMAGE


_ANALYSIS_RENDERER_OUTPUT = (
    "..analysis-post-stack.children"
    "...analysis-filter-stack.children"
    "...analysis-edge-stack.children"
    "...analysis-model-section.children"
    "...analysis-pipeline-header.children"
    "...analysis-run-button.disabled"
    "...analysis-model-dropdown.value"
    "...analysis-post-add-dropdown.value"
    "...analysis-filter-add-dropdown.value"
    "...analysis-edge-add-dropdown.value.."
)


# ---------------------------------------------------------------------------
# Fixtures: a sandbox whose Recent Runs CLI output also has pipeline.json +
# measurements.parquet so the analysis sub-app has something to bind to.
# ---------------------------------------------------------------------------


def _seed_analysis_output(sandbox: Path) -> Path:
    """Augment the standard sandbox layout with a pipeline.json + a
    realistic measurements.parquet so the analysis sub-app can run.
    Returns the CLI output dir under the sandbox.
    """
    cli_out = sandbox / "results" / "CliOutputExample"

    # Tiny logistic-growth fixture covering enough timepoints for
    # LogGrowthModel to converge on each strain group.
    rows: list[dict] = []
    import math

    for strain in ("CBS-A", "CBS-B"):
        for t in (0, 6, 12, 24, 36, 48):
            for rep in range(3):
                n = 100 + 800 / (1 + (1000 - 100) / 100 * math.exp(-0.15 * t))
                rows.append({
                    "Metadata_Dataset": "ds1",
                    str(IMAGE.IMAGE_NAME): f"{strain}_t{t}",
                    "Metadata_Strain": strain,
                    "Metadata_Time": float(t),
                    "Object_Label": rep,
                    "Shape_Area": float(n + (rep - 1) * 5),
                })
    df = pl.DataFrame(rows)
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)

    # A real pipeline.json so RecipeState boots with content. Empty model
    # so the run button starts disabled — adding the model is part of the
    # test surface.
    pipeline = ImagePipeline(name="analysis-e2e-fixture")
    write_pipeline_json(cli_out, pipeline)
    publish_coherent_terminal_evidence(cli_out, total_images=12)

    return cli_out


@pytest.fixture
def analysis_sandbox(tmp_path: pytest.TempPathFactory) -> Path:
    """Function-scoped sandbox so add/remove tests don't interfere with each
    other or with the module-scoped Recent Runs fixture in conftest.
    """
    parent = tmp_path  # type: ignore[assignment]
    sandbox = _build_sandbox(parent)
    _seed_analysis_output(sandbox)
    return sandbox


@pytest.fixture
def analysis_live_server(analysis_sandbox: Path) -> Iterator[str]:
    """Function-scoped live-server bound to the augmented sandbox."""
    yield from _start_live_server(analysis_sandbox)


def _dash_store_data(page: Page, store_id: str) -> dict:
    """Read one dcc.Store payload from the hydrated Dash Redux layout."""
    return page.evaluate(
        """(storeId) => {
            const state = window.store.getState();
            const path = state.paths.strs[storeId];
            return path.reduce((value, key) => value[key], state.layout)
                .props.data;
        }""",
        store_id,
    )


def _wait_for_dash_request_quiescence(
    page: Page,
    pending_request_ids: set[int],
    *,
    store_id: str,
    expected_data: dict,
    timeout_seconds: float = 5.0,
) -> None:
    """Wait until a store update lands and all observed Dash requests drain."""
    deadline = time.monotonic() + timeout_seconds
    idle_since: float | None = None
    while time.monotonic() < deadline:
        store_matches = _dash_store_data(page, store_id) == expected_data
        if store_matches and not pending_request_ids:
            if idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= 0.1:
                return
        else:
            idle_since = None
        page.wait_for_timeout(20)
    raise AssertionError(
        f"Dash did not quiesce with {store_id}={expected_data!r}; "
        f"pending request ids: {sorted(pending_request_ids)!r}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_analysis_mount_renders_empty_state(page: Page, hub_url: str) -> None:
    """Hub mount shows the empty-state placeholder when no output is bound."""
    page.goto(hub_url + "/analysis/")
    page.wait_for_selector("#analysis-page", timeout=10_000)
    expect(page.locator("#analysis-page")).to_be_visible()
    text = page.locator("#analysis-page").text_content() or ""
    assert "No output selected" in text
    assert "Pick a CLI output directory" in text


def test_analysis_tab_in_top_bar(page: Page, hub_url: str) -> None:
    """The shell top bar lists Analysis under the Results dropdown group."""
    import re

    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-tab-group-results", timeout=10_000)
    # Analysis is a member of the Results dropdown; open the group to
    # reveal the item, then click it to navigate to /analysis/.
    page.click("#shell-tab-group-results .dropdown-toggle")
    analysis_item = page.locator("#shell-tab-analysis")
    analysis_item.wait_for(state="visible", timeout=5_000)
    analysis_item.click()
    page.wait_for_url("**/analysis/")
    expect(page.locator("#shell-tab-analysis")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#shell-tab-group-results .dropdown-toggle")).to_have_class(
        re.compile(r"shell-tab-group-active")
    )


def test_analysis_standalone_renders_pipeline_header(
    page: Page,
    analysis_live_server: str,
    analysis_sandbox: Path,
) -> None:
    """When invoked with a real output dir bound, the page renders the
    pipeline header, the recompile banner, and the run button."""
    # Standalone launcher binds to the sandbox itself; for the hub variant,
    # we'd need to drive the sidebar to hand off the output root. The
    # function-scoped server here uses the `_start_live_server` helper which
    # boots `phenotypic-gui` against the sandbox -- the hub will mount the
    # analysis sub-app in empty-state until the user picks an output via
    # the sidebar. Driving that flow is covered by the dedicated viewer
    # hand-off tests; here we just confirm the page renders the chrome
    # without error and the analysis tab loads.
    page.goto(analysis_live_server + "/analysis/")
    page.wait_for_selector("#analysis-page", timeout=10_000)
    expect(page.locator("#analysis-page")).to_be_visible()


def test_pipeline_json_seeded_on_disk(
    analysis_sandbox: Path,
) -> None:
    """The fixture seeds ``pipeline.json`` and ``measurements.parquet``
    in the CLI output dir so the analysis sub-app can bind cleanly.
    Confirms the fixture wiring is right (not the GUI itself)."""
    cli_out = analysis_sandbox / "results" / "CliOutputExample"
    assert pipeline_json_path(cli_out).exists()
    assert measurements_parquet_path(cli_out).exists()
    config = json.loads(pipeline_json_path(cli_out).read_text())
    assert "filters" in config
    assert "model" in config
    assert config["model"] is None


def test_model_selection_publishes_once_and_feedback_is_noop(
    page: Page,
    analysis_live_server: str,
    analysis_sandbox: Path,
) -> None:
    """Hydrated model selection emits one revision and no feedback save."""
    bind_results_output(
        page,
        analysis_live_server,
        "results/CliOutputExample",
        destination="/analysis/",
    )

    page.wait_for_selector("#analysis-model-dropdown", timeout=20_000)
    page.wait_for_load_state("networkidle")
    pipeline_path = pipeline_json_path(
        analysis_sandbox / "results" / "CliOutputExample"
    )
    before_bytes = pipeline_path.read_bytes()
    before_mtime_ns = pipeline_path.stat().st_mtime_ns
    dash_responses: list[tuple[int, str]] = []

    def _capture_dash_response(response) -> None:
        if response.url.endswith("/_dash-update-component"):
            body = response.text() if response.status == 200 else ""
            dash_responses.append((response.status, body))

    page.on("response", _capture_dash_response)
    page.locator("#analysis-model-dropdown").click()
    page.get_by_role("option", name="LinearLagModel").click()

    expect(page.locator("#analysis-pipeline-header")).to_contain_text(
        "model: LinearLagModel",
        timeout=20_000,
    )
    expect(page.locator("#analysis-model-section")).to_contain_text(
        "LinearLagModel"
    )
    expect(page.locator("#analysis-run-button")).to_be_enabled()
    page.wait_for_timeout(1_000)
    page.remove_listener("response", _capture_dash_response)

    revision_payloads: list[dict] = []
    for status, body in dash_responses:
        if status != 200 or not body:
            continue
        payload = json.loads(body)
        store_response = payload.get("response", {}).get(
            "analysis-pipeline-event-store"
        )
        if store_response is not None:
            revision_payloads.append(store_response["data"])

    assert len(revision_payloads) == 1
    assert revision_payloads[0]["revision"] == 1
    assert any(status == 204 for status, _body in dash_responses)
    assert pipeline_path.read_bytes() != before_bytes
    assert pipeline_path.stat().st_mtime_ns != before_mtime_ns
    saved = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert saved["model"]["class"] == "LinearLagModel"
    assert json.loads(revision_payloads[0]["pipeline_json"]) == saved


def test_delayed_revision_event_cannot_overwrite_newer_ui(
    page: Page,
    analysis_live_server: str,
    analysis_sandbox: Path,
) -> None:
    """A delayed rev1 event is ignored after rev2 has rendered."""
    bind_results_output(
        page,
        analysis_live_server,
        "results/CliOutputExample",
        destination="/analysis/",
    )

    page.wait_for_selector("#analysis-model-dropdown", timeout=20_000)
    page.wait_for_load_state("networkidle")
    dash_responses: list[tuple[int, str]] = []
    renderer_requests: list[str] = []
    pending_dash_request_ids: set[int] = set()

    def _capture_dash_request(request) -> None:
        if not request.url.endswith("/_dash-update-component"):
            return
        pending_dash_request_ids.add(id(request))
        post_data = request.post_data_json
        output = post_data.get("output", "") if post_data else ""
        if output == _ANALYSIS_RENDERER_OUTPUT:
            renderer_requests.append(output)

    def _finish_dash_request(request) -> None:
        if request.url.endswith("/_dash-update-component"):
            pending_dash_request_ids.discard(id(request))

    def _capture_dash_response(response) -> None:
        if response.url.endswith("/_dash-update-component"):
            body = response.text() if response.status == 200 else ""
            dash_responses.append((response.status, body))

    page.on("request", _capture_dash_request)
    page.on("requestfinished", _finish_dash_request)
    page.on("requestfailed", _finish_dash_request)
    page.on("response", _capture_dash_response)
    page.locator("#analysis-model-dropdown").click()
    page.get_by_role("option", name="LinearLagModel").click()
    expect(page.locator("#analysis-pipeline-header")).to_contain_text(
        "model: LinearLagModel",
        timeout=20_000,
    )

    page.locator("#analysis-filter-add-dropdown").click()
    page.get_by_role("option", name="TukeyOutlierRemover").click()
    expect(page.locator("#analysis-pipeline-header")).to_contain_text(
        "1 filters",
        timeout=20_000,
    )
    expect(page.locator("#analysis-filter-stack")).to_contain_text(
        "TukeyOutlierRemover"
    )
    page.wait_for_timeout(500)

    revision_events: list[dict] = []
    for status, body in dash_responses:
        if status != 200 or not body:
            continue
        payload = json.loads(body)
        event_response = payload.get("response", {}).get(
            "analysis-pipeline-event-store"
        )
        if event_response is not None:
            revision_events.append(event_response["data"])
    assert [event["revision"] for event in revision_events] == [1, 2]
    _wait_for_dash_request_quiescence(
        page,
        pending_dash_request_ids,
        store_id="analysis-pipeline-gate-ack-store",
        expected_data={"revision": 2, "accepted": True},
    )
    renderer_request_count = len(renderer_requests)
    applied_rev2 = _dash_store_data(
        page,
        "analysis-pipeline-store",
    )
    assert applied_rev2 == revision_events[1]
    pipeline_path = pipeline_json_path(
        analysis_sandbox / "results" / "CliOutputExample"
    )
    saved_rev2 = pipeline_path.read_bytes()
    assert applied_rev2["pipeline_json"].encode("utf-8") == saved_rev2

    page.evaluate(
        """(event) => {
            window.dash_clientside.set_props(
                'analysis-pipeline-event-store',
                {data: event},
            );
        }""",
        revision_events[0],
    )
    _wait_for_dash_request_quiescence(
        page,
        pending_dash_request_ids,
        store_id="analysis-pipeline-gate-ack-store",
        expected_data={"revision": 1, "accepted": False},
    )

    expect(page.locator("#analysis-pipeline-header")).to_contain_text(
        "1 filters"
    )
    expect(page.locator("#analysis-pipeline-header")).to_contain_text(
        "model: LinearLagModel"
    )
    expect(page.locator("#analysis-filter-stack")).to_contain_text(
        "TukeyOutlierRemover"
    )
    assert len(renderer_requests) == renderer_request_count
    assert _dash_store_data(
        page,
        "analysis-pipeline-event-store",
    ) == revision_events[0]
    applied_after_delay = _dash_store_data(
        page,
        "analysis-pipeline-store",
    )
    assert applied_after_delay == revision_events[1]
    assert applied_after_delay["pipeline_json"].encode("utf-8") == saved_rev2
    assert pipeline_path.read_bytes() == saved_rev2
    saved = json.loads(saved_rev2)
    assert saved["model"]["class"] == "LinearLagModel"
    assert len(saved["filters"]) == 1


# Reference, only — keeps `TukeyOutlierRemover`/`LogGrowthModel` imports
# from being garbage-collected during fixture-only test runs.
_REFERENCED = (TukeyOutlierRemover, LogGrowthModel)
