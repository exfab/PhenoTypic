"""Lazy tree expand + sidebar hand-off (E2E).

These two features were the only remaining v1 limitations after Phase 9
landed. This file pins their behaviour:

* **Lazy expand:** clicking a folder row in the sidebar appends a nested
  ``<ul>`` of that folder's children. Clicking again collapses it.
* **Hand-off banner (run console):** clicking a sidebar entry stamps
  ``SHELL_SIDEBAR_SELECTION_STORE`` and the run console banner activates
  with contextual ``Set as pipeline`` / ``Set as input dir`` / ``Set as
  output dir`` buttons. Clicking ``Set as input dir`` writes the path
  into the run console's input field.
"""
from __future__ import annotations

from playwright.sync_api import Page, expect


# ---------------------------------------------------------------------------
# Lazy tree expand
# ---------------------------------------------------------------------------

def test_lazy_expand_collapse_state_machine(page: Page, hub_url: str) -> None:
    """One page load that walks the full expand/collapse state machine:

    1. Initial state — ``plate1`` row visible, ``image.tif`` not.
    2. Click ``plate1`` — children appear, icon flips 📁 → 📂.
    3. Click again — children hidden.

    Replaces three separate tests
    (``test_lazy_expand_reveals_children``, ``test_lazy_collapse_hides_children``,
    ``test_expanded_folder_uses_open_icon``) that each paid the cost of a fresh
    page load to assert one of these three transitions in isolation.
    """
    selector = (
        'button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]'
    )

    # 1. Initial state.
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
    initial_text = page.locator("#shell-sidebar-tree").text_content() or ""
    assert "plate1" in initial_text
    assert "image.tif" not in initial_text  # not yet visible
    icon_before = page.evaluate(
        "(s) => document.querySelector(s)"
        "       ?.querySelector('.shell-sidebar-icon')?.textContent",
        selector,
    )
    assert icon_before == "📁"

    # 2. Expand.
    page.click(selector)
    page.wait_for_function(
        "() => {"
        "  const t = document.getElementById('shell-sidebar-tree');"
        "  return t && (t.textContent || '').includes('image.tif');"
        "}",
        timeout=5_000,
    )
    page.wait_for_function(
        "(s) => document.querySelector(s)"
        "       ?.querySelector('.shell-sidebar-icon')?.textContent === '📂'",
        arg=selector,
        timeout=5_000,
    )

    # 3. Collapse.
    page.click(selector)
    page.wait_for_function(
        "() => !document.getElementById('shell-sidebar-tree').textContent.includes('image.tif')",
        timeout=5_000,
    )


# ---------------------------------------------------------------------------
# Sidebar hand-off banner
# ---------------------------------------------------------------------------

def test_handoff_banner_hidden_by_default(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/run/")
    # ``state="attached"`` — the banner element is in the DOM but its
    # initial style sets ``display: none`` so it would never become
    # ``visible`` until the consumer callback flips it.
    page.wait_for_selector("#rc-handoff-banner", state="attached", timeout=10_000)
    display = page.evaluate(
        "() => getComputedStyle(document.getElementById('rc-handoff-banner')).display"
    )
    assert display == "none"


def test_handoff_banner_activates_on_dir_click(
    page: Page, hub_url: str,
) -> None:
    """Clicking a sidebar directory makes the banner appear with the
    input + output buttons enabled, pipeline disabled."""
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
    page.click(
        'button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]'
    )
    page.wait_for_function(
        "() => getComputedStyle(document.getElementById('rc-handoff-banner')).display !== 'none'",
        timeout=5_000,
    )
    # Banner shows the selected path.
    label = page.locator("#rc-handoff-label").text_content()
    assert label == "plate1"
    # Pipeline button disabled (plate1 is a dir, not a JSON).
    expect(page.locator("#rc-handoff-use-pipeline")).to_be_disabled()
    # Input + output buttons enabled.
    expect(page.locator("#rc-handoff-use-input")).to_be_enabled()
    expect(page.locator("#rc-handoff-use-output")).to_be_enabled()


def test_handoff_use_input_writes_to_input_store(
    page: Page, hub_url: str, fake_sandbox,
) -> None:
    """Clicking ``Set as input dir`` writes the absolute path to the
    run console's input-dir store and surfaces a confirmation toast."""
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
    page.click(
        'button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]'
    )
    page.wait_for_function(
        "() => !document.getElementById('rc-handoff-use-input').disabled",
        timeout=5_000,
    )
    page.click("#rc-handoff-use-input")
    # The input-dir picker label updates from "(none)" to the path.
    page.wait_for_function(
        "() => {"
        "  const lbl = document.getElementById('rc-label-input');"
        "  return lbl && (lbl.textContent || '').includes('plate1');"
        "}",
        timeout=5_000,
    )


def test_handoff_dismiss_hides_banner(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
    page.click(
        'button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]'
    )
    page.wait_for_function(
        "() => getComputedStyle(document.getElementById('rc-handoff-banner')).display !== 'none'",
        timeout=5_000,
    )
    page.click("#rc-handoff-dismiss")
    page.wait_for_function(
        "() => getComputedStyle(document.getElementById('rc-handoff-banner')).display === 'none'",
        timeout=5_000,
    )
