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

import json
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from phenotypic.schema import IMAGE
from phenotypic.sdk_ import gui_launch_owner_path


def _prepare_action_paths(
    sandbox: Path,
    *,
    name: str,
) -> tuple[Path, Path, Path]:
    """Create isolated controls for one intentionally short CLI launch."""
    pipeline = sandbox / f"{name}-pipeline.json"
    pipeline.write_text('{"operations": "invalid"}', encoding="utf-8")
    input_dir = sandbox / "plate1"
    output_dir = sandbox / "results" / name
    output_dir.mkdir(parents=True)
    return pipeline, input_dir, output_dir


def _set_action_controls(
    page: Page,
    *,
    pipeline: Path,
    input_dir: Path,
    output_dir: Path,
    modes: list[str],
) -> None:
    """Set source controls, then confirm the exact typed output in the UI."""
    page.evaluate(
        """(values) => {
            window.dash_clientside.set_props(
                "rc-store-pipeline-path", {data: values.pipeline}
            );
            window.dash_clientside.set_props(
                "rc-store-input-dir", {data: values.inputDir}
            );
            window.dash_clientside.set_props(
                "rc-input-slurm-partition", {value: "compute"}
            );
        }""",
        {
            "pipeline": str(pipeline),
            "inputDir": str(input_dir),
        },
    )
    page.locator("#rc-btn-pick-output").click()
    output_input = page.locator("#rc-input-output-path")
    output_input.wait_for(state="visible")
    output_input.fill(str(output_dir))
    page.locator("#rc-btn-output-confirm").click()
    expect(page.locator("#rc-modal-output")).not_to_be_visible()
    expect(page.locator("#rc-label-output")).to_contain_text(output_dir.name)
    expect(page.locator("#rc-metadata-preflight")).to_contain_text("1 image(s)")
    page.evaluate(
        """(modes) => {
            for (const mode of modes) {
                window.dash_clientside.set_props(
                    "rc-radio-mode", {value: mode}
                );
            }
        }""",
        modes,
    )


def _wait_for_owner_status(
    output_dir: Path,
    *,
    terminal: bool,
    timeout: float = 15.0,
) -> dict[str, object]:
    """Read the durable owner after the server-side action callback commits."""
    owner = gui_launch_owner_path(output_dir)
    deadline = time.monotonic() + timeout
    last: dict[str, object] | None = None
    while time.monotonic() < deadline:
        if owner.is_file():
            last = json.loads(owner.read_text(encoding="utf-8"))
            if not terminal or last.get("status") in {
                "complete",
                "failed",
                "cancelled",
            }:
                return last
        time.sleep(0.05)
    raise AssertionError(f"owner did not reach expected state: {last!r}")


def test_run_console_form_has_pickers(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-pick-pipeline", timeout=10_000)
    expect(page.locator("#rc-btn-pick-pipeline")).to_be_visible()
    expect(page.locator("#rc-btn-pick-input")).to_be_visible()
    expect(page.locator("#rc-btn-pick-output")).to_be_visible()


@pytest.mark.parametrize("btn_id,modal_id", [
    ("#rc-btn-pick-pipeline", "#rc-modal-pipeline"),
    ("#rc-btn-pick-input",    "#rc-modal-input"),
    ("#rc-btn-pick-output",   "#rc-modal-output"),
])
def test_picker_modal_opens(
    page: Page, hub_url: str, btn_id: str, modal_id: str,
) -> None:
    """Each picker button opens its corresponding dbc.Modal."""
    page.goto(hub_url + "/run/")
    page.wait_for_selector(btn_id)
    page.click(btn_id)
    # dbc.Modal becomes visible (display !== none) when ``is_open`` flips.
    page.wait_for_function(
        "(modalId) => {"
        "  const m = document.querySelector(modalId);"
        "  return m && getComputedStyle(m).display !== 'none';"
        "}",
        arg=modal_id,
        timeout=5_000,
    )


def test_typed_nonexistent_output_keeps_exact_target(
    page: Page,
    hub_url: str,
    fake_sandbox: Path,
) -> None:
    """Confirming a typed new path never substitutes the browsed sandbox root."""
    target = fake_sandbox / "typed-output-new" / "nested"
    assert not target.exists()
    page.goto(hub_url + "/run/")
    page.locator("#rc-btn-pick-output").click()
    output_input = page.locator("#rc-input-output-path")
    output_input.fill("typed-output-new/nested")
    page.locator("#rc-btn-output-confirm").click()

    expect(page.locator("#rc-modal-output")).not_to_be_visible()
    expect(page.locator("#rc-label-output")).to_have_text(
        "typed-output-new/nested"
    )
    assert not target.exists()


def test_output_picker_refuses_sandbox_root(
    page: Page,
    hub_url: str,
) -> None:
    page.goto(hub_url + "/run/")
    page.locator("#rc-btn-pick-output").click()
    page.locator("#rc-input-output-path").fill(".")
    page.locator("#rc-btn-output-confirm").click()

    expect(page.locator("#rc-modal-output")).to_be_visible()
    expect(page.locator("#rc-toast")).to_contain_text(
        "sandbox root cannot be used"
    )
    expect(page.locator("#rc-label-output")).to_have_text("(none)")


def test_metadata_preflight_shows_ambient_descriptor_but_defaults_to_omit(
    page: Page,
    hub_url: str,
    fake_sandbox: Path,
) -> None:
    """One-image metadata is visible and compatible without implicit authority."""
    metadata = fake_sandbox / "one-image-metadata.csv"
    metadata.write_text(
        f"{IMAGE.IMAGE_NAME},Treatment\nimage,control\n",
        encoding="utf-8",
    )
    page.goto(hub_url + "/run/")
    page.locator("#shell-settings-button").click()
    page.locator("#shell-settings-metadata-csv-pick").click()
    metadata_modal = page.locator("#shell-metadata-csv-modal")
    expect(metadata_modal).to_be_visible()
    metadata_entry = page.locator(
        '[id*="shell-metadata-csv-entry"]'
        '[id*="one-image-metadata.csv"]'
    )
    metadata_entry.click()
    expect(page.locator("#shell-metadata-csv-modal-body")).to_contain_text(
        "Selected CSV: one-image-metadata.csv"
    )
    page.locator("#shell-metadata-csv-confirm").click()
    expect(metadata_modal).not_to_be_visible()
    page.evaluate(
        """(source) => window.dash_clientside.set_props(
            "rc-store-input-dir", {data: source}
        )""",
        str(fake_sandbox / "plate1"),
    )

    preflight = page.locator("#rc-metadata-preflight")
    expect(preflight).to_contain_text(str(metadata))
    expect(preflight).to_contain_text("Status: compatible")
    expect(preflight).to_contain_text("1/1 input image(s) matched")
    expect(
        page.locator('#rc-metadata-choice input[value="omit"]')
    ).to_be_checked()


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


def test_validate_exit_reaches_terminal_registry_state(
    page: Page,
    hub_url: str,
    fake_sandbox: Path,
) -> None:
    """Validate allocates before spawn and records its immediate failure."""
    pipeline, input_dir, output_dir = _prepare_action_paths(
        fake_sandbox, name="ValidateTerminal"
    )
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-validate")
    _set_action_controls(
        page,
        pipeline=pipeline,
        input_dir=input_dir,
        output_dir=output_dir,
        modes=["local"],
    )
    page.locator("#rc-btn-validate").click()

    owner = _wait_for_owner_status(output_dir, terminal=True)
    assert owner["mode"] == "validate"
    assert owner["status"] == "failed"
    assert isinstance(owner["generation"], str)


def test_run_local_exit_reaches_terminal_registry_state(
    page: Page,
    hub_url: str,
    fake_sandbox: Path,
) -> None:
    """Run Local observes process exit without requiring another click."""
    pipeline, input_dir, output_dir = _prepare_action_paths(
        fake_sandbox, name="LocalTerminal"
    )
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-run")
    _set_action_controls(
        page,
        pipeline=pipeline,
        input_dir=input_dir,
        output_dir=output_dir,
        modes=["local"],
    )
    page.locator("#rc-btn-run").click()

    owner = _wait_for_owner_status(output_dir, terminal=True)
    assert owner["mode"] == "local"
    assert owner["status"] == "failed"


@pytest.mark.parametrize(
    ("modes", "expected_mode"),
    [
        (["slurm", "local"], "local"),
        (["local", "slurm"], "slurm"),
    ],
)
def test_run_uses_final_visible_mode_during_rapid_toggle(
    page: Page,
    hub_url: str,
    fake_sandbox: Path,
    modes: list[str],
    expected_mode: str,
) -> None:
    """The action request uses raw radio state, not the lagging form store."""
    name = f"ModeRace-{expected_mode}"
    pipeline, input_dir, output_dir = _prepare_action_paths(
        fake_sandbox, name=name
    )
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-run")
    _set_action_controls(
        page,
        pipeline=pipeline,
        input_dir=input_dir,
        output_dir=output_dir,
        modes=modes,
    )
    page.locator("#rc-btn-run").click()

    owner = _wait_for_owner_status(output_dir, terminal=False)
    assert owner["mode"] == expected_mode


# Moved to test_save_preset.py — that test mutates fake_sandbox so it needs
# function-scoped fixture overrides; the rest of this file shares the
# module-scoped fake_sandbox/live_server from conftest.py.


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
    # The dashboard now lives under ``deliverables/`` — the iframe src points
    # at ``/runs/<rel>/deliverables/dashboard.html``.
    assert "/runs/results/CliOutputExample/deliverables/dashboard.html" in src
    # Iframe display should be ``block`` after the row click toggles
    # placeholder/iframe visibility.
    iframe_display = page.evaluate(
        "() => getComputedStyle(document.getElementById('rc-iframe')).display"
    )
    assert iframe_display == "block"
