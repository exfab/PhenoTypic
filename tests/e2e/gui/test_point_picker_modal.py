"""End-to-end tests for the in-builder point-picker modal.

These tests exercise the modal's stable contract from the browser side:

* Dropping ``ManualPointDetector`` on the builder canvas mounts a
  ``Pick on image…`` button in the param form.
* Clicking that button opens ``#modal-point-picker``.
* Pushing staged points into ``#picker-staged-store`` (the same path
  the OSD click handler uses via ``dash_clientside.set_props``) updates
  the in-modal count label.
* Re-clicking the always-enabled ``rgb`` channel radio doesn't lose
  staged points.
* Confirm closes the modal and round-trips the staged points into the
  node-specific ``param-point-picker-count`` Span outside the modal.

We drive ``picker-staged-store.data`` directly via ``page.evaluate``
because OSD's canvas intercepts mouse events; synthesising a real
click on the OSD viewer is fragile, but the count-label /
channel-toggle / Confirm wiring is identical regardless of who
populated the store.
"""
from __future__ import annotations

import json

from playwright.sync_api import Page, expect


def _open_picker_modal_for_manual_point_detector(
    page: Page, hub_url: str
) -> str:
    """Boot the builder, drop ``ManualPointDetector``, open the picker modal.

    Returns the ``node_id`` Dash assigned to the new node so the caller
    can target the node-specific count label outside the modal.
    """
    page.goto(hub_url + "/builder/")
    page.wait_for_selector("#palette", timeout=10_000)

    # Each operations-palette accordion section is collapsed by default
    # except the first (``Corrector``). ``ManualPointDetector`` lives
    # under ``Detector``, so expand that header before clicking the
    # operation button. ``dbc.Accordion`` uses ``aria-controls`` on the
    # header button — match by visible text instead.
    detector_header = page.locator(
        "#palette button.accordion-button"
    ).filter(has_text="Detector")
    if detector_header.count() > 0:
        klass = detector_header.first.get_attribute("class") or ""
        if "collapsed" in klass:
            detector_header.first.click()

    # ``has_text`` matches the inner Span's text even when the button
    # also contains a "PICK" badge.
    page.locator("button", has_text="ManualPointDetector").first.click(
        timeout=10_000
    )

    # The "Pick on image…" button is rendered in the param form once the
    # newly added node becomes the inspector selection.
    page.wait_for_selector("text=Pick on image…", timeout=10_000)
    page.click("text=Pick on image…")

    # ``dbc.Modal`` flips ``display: block`` when ``is_open`` becomes True.
    page.wait_for_function(
        "() => {"
        "  const m = document.getElementById('modal-point-picker');"
        "  return m && getComputedStyle(m).display !== 'none';"
        "}",
        timeout=10_000,
    )

    # Recover ``node_id`` from the picker button's pattern-matching id.
    # Dash serialises pattern-matching ids as a JSON string in the DOM.
    node_id = page.evaluate(
        "() => {"
        "  const btn = document.querySelector(\"button[id*='param-point-picker-btn']\");"
        "  if (!btn) return null;"
        "  try { return JSON.parse(btn.id).prefix; } catch (e) { return null; }"
        "}"
    )
    return node_id or ""


def _push_staged_points(page: Page, points: list[list[float]]) -> None:
    """Mimic the OSD click handler by pushing into ``picker-staged-store``.

    The clientside JS calls
    ``dash_clientside.set_props('picker-staged-store', {data: nextPoints})``
    on every accepted click; doing the same from Playwright exercises
    every server-side observer of that store.
    """
    page.evaluate(
        f"window.dash_clientside.set_props('picker-staged-store', "
        f"{{data: {json.dumps(points)}}})"
    )


def test_pick_three_points_updates_count(page: Page, hub_url: str) -> None:
    """Pushing three points updates the in-modal count label."""
    _open_picker_modal_for_manual_point_detector(page, hub_url)

    # OSD host div mounts as soon as the modal opens.
    expect(
        page.locator('[data-testid="point-picker-osd-canvas"]')
    ).to_be_visible(timeout=10_000)

    _push_staged_points(
        page, [[10.0, 20.0], [50.0, 60.0], [100.0, 120.0]]
    )

    expect(page.locator("#picker-count-label")).to_have_text(
        "3 points", timeout=5_000
    )


def test_channel_toggle_preserves_points(
    page: Page, hub_url: str
) -> None:
    """Toggling the channel radio leaves staged points intact.

    The fake_sandbox image is an empty placeholder, so the open-modal
    callback can't actually stage a PNG and the ``rgb`` radio mounts
    disabled. Stub the avail store so the radio is enabled, then click
    it — the staged-store data must survive the channel change.
    """
    _open_picker_modal_for_manual_point_detector(page, hub_url)
    _push_staged_points(page, [[10.0, 20.0], [50.0, 60.0]])
    expect(page.locator("#picker-count-label")).to_have_text(
        "2 points", timeout=5_000
    )

    # Mark rgb as available so the radio's ``disabled`` flips off.
    page.evaluate(
        "() => window.dash_clientside.set_props("
        "'picker-channel-avail-store', "
        "{data: {rgb: true, intermediate: false}})"
    )
    # Wait for the toggle_radio_options callback to re-render the input.
    page.wait_for_function(
        "() => {"
        "  const inp = document.querySelector(\"input[value='rgb']\");"
        "  return inp && !inp.disabled;"
        "}",
        timeout=5_000,
    )

    page.click("input[value='rgb']")
    expect(page.locator("#picker-count-label")).to_have_text("2 points")


def test_confirm_writes_centers_to_node(
    page: Page, hub_url: str
) -> None:
    """Confirm closes the modal and writes points to the node's params."""
    node_id = _open_picker_modal_for_manual_point_detector(page, hub_url)
    assert node_id, "could not recover node_id from picker button id"

    _push_staged_points(page, [[5.0, 5.0], [25.0, 25.0]])
    expect(page.locator("#picker-count-label")).to_have_text(
        "2 points", timeout=5_000
    )

    page.click("#btn-picker-confirm")

    # Modal closes once Confirm fires.
    page.wait_for_function(
        "() => {"
        "  const m = document.getElementById('modal-point-picker');"
        "  return !m || getComputedStyle(m).display === 'none';"
        "}",
        timeout=5_000,
    )

    # Dash renders pattern-matching ids as a JSON string with keys
    # sorted alphabetically and no whitespace between separators.
    outside_id = json.dumps(
        {
            "name": "centers",
            "prefix": node_id,
            "type": "param-point-picker-count",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    expect(page.locator(f"[id='{outside_id}']")).to_have_text(
        "2 points", timeout=10_000
    )
