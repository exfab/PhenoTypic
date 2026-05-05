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


def test_input_picker_modal_opens(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-pick-input")
    page.click("#rc-btn-pick-input")
    page.wait_for_function(
        "() => {"
        "  const m = document.getElementById('rc-modal-input');"
        "  return m && getComputedStyle(m).display !== 'none';"
        "}",
        timeout=5_000,
    )


def test_output_picker_modal_opens(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-pick-output")
    page.click("#rc-btn-pick-output")
    page.wait_for_function(
        "() => {"
        "  const m = document.getElementById('rc-modal-output');"
        "  return m && getComputedStyle(m).display !== 'none';"
        "}",
        timeout=5_000,
    )


def test_mode_toggle_switches_state(page: Page, hub_url: str) -> None:
    """The Local/SLURM radio drives the form state; switching to SLURM
    must reveal the SLURM-config collapse and update the underlying value."""
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-radio-mode")
    # Find the SLURM radio input and click it.
    slurm_input = page.locator('#rc-radio-mode input[value="slurm"]')
    slurm_input.click()
    # Read back the radio group value via Dash component id.
    value = page.evaluate(
        "() => {"
        "  const checked = document.querySelector('#rc-radio-mode input:checked');"
        "  return checked ? checked.value : null;"
        "}"
    )
    assert value == "slurm"


def test_validate_button_is_present_and_enabled(page: Page, hub_url: str) -> None:
    """The Validate button mounts and is clickable. Spawning a real
    dry-run subprocess needs a real CLI fixture (deferred); this test
    pins the affordance at the layout level."""
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-validate")
    expect(page.locator("#rc-btn-validate")).to_be_visible()
    expect(page.locator("#rc-btn-validate")).to_be_enabled()


def test_save_preset_writes_file(
    page: Page, hub_url: str, fake_sandbox,
) -> None:
    """Clicking Save preset with a name writes ``<sandbox>/.phenotypic-gui/presets/<name>.json``.

    The form state is empty here; that's fine — the preset just round-trips
    whatever the form's `RC_STORE_FORM_STATE` currently holds (default
    ``RunConsoleState`` round-tripped to JSON).
    """
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-input-preset-name")
    page.fill("#rc-input-preset-name", "smoke_preset")
    page.click("#rc-btn-save-preset")
    # The save callback returns a toast; wait for any toast text.
    page.wait_for_function(
        "() => {"
        "  const t = document.getElementById('rc-toast');"
        "  return t && (t.textContent || '').toLowerCase().includes('saved');"
        "}",
        timeout=5_000,
    )
    target = fake_sandbox / ".phenotypic-gui" / "presets" / "smoke_preset.json"
    assert target.is_file()


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
