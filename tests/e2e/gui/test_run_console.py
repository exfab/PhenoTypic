"""Run console layout + Recent Runs + picker modals (E2E).

Confirms:

* The form's three picker buttons are mounted.
* The pipeline picker modal opens on button click.
* Recent Runs panel rehydrates the ``CliOutputExample`` row from
  ``fake_sandbox`` at boot time (no manual refresh needed).
* Clicking a Recent Runs row sets the iframe ``src`` to the dashboard.
* The iframe ``src`` uses the absolute ``/runs/...`` path (mounted on
  the shell Flask, not under any Dash sub-app's prefix).
"""
from __future__ import annotations

from playwright.sync_api import Page, expect


def test_run_console_form_has_pickers(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-pick-pipeline", timeout=10_000)
    expect(page.locator("#rc-btn-pick-pipeline")).to_be_visible()
    expect(page.locator("#rc-btn-pick-input")).to_be_visible()
    expect(page.locator("#rc-btn-pick-output")).to_be_visible()


def test_pipeline_picker_modal_opens(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-pick-pipeline")
    page.click("#rc-btn-pick-pipeline")
    # dbc.Modal becomes visible (display !== none) when ``is_open`` flips.
    page.wait_for_function(
        "() => {"
        "  const m = document.getElementById('rc-modal-pipeline');"
        "  return m && getComputedStyle(m).display !== 'none';"
        "}",
        timeout=5_000,
    )


def test_recent_runs_rehydrates_cli_output_example(
    page: Page, hub_url: str,
) -> None:
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-recents-body", timeout=10_000)
    text = page.locator("#rc-recents-body").text_content() or ""
    assert "CliOutputExample" in text


def test_recent_runs_row_click_sets_iframe_src(
    page: Page, hub_url: str,
) -> None:
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-recents-body")
    page.wait_for_selector('[id*="rc-recents-row"]')
    # Click the first row (pattern-matching id).
    page.locator('[id*="rc-recents-row"]').first.click()
    # The iframe src callback writes via Dash output → wait for it.
    page.wait_for_function(
        "() => {"
        "  const f = document.getElementById('rc-iframe');"
        "  return f && /\\/runs\\/.*dashboard\\.html/.test(f.src || '');"
        "}",
        timeout=5_000,
    )
    src = page.locator("#rc-iframe").get_attribute("src") or ""
    assert "/runs/results/CliOutputExample/dashboard.html" in src
    # Iframe display should be ``block`` after the row click toggles
    # placeholder/iframe visibility.
    iframe_display = page.evaluate(
        "() => getComputedStyle(document.getElementById('rc-iframe')).display"
    )
    assert iframe_display == "block"
