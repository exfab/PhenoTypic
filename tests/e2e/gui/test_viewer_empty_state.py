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


def test_other_mounts_do_not_inject_prefix(page: Page, hub_url: str) -> None:
    """Builder + Run console index pages should NOT inject the prefix —
    only the viewer has the JS-pad assets that need it."""
    page.goto(hub_url + "/builder/")
    page.wait_for_selector("#shell-top-bar")
    builder_prefix = page.evaluate("() => window.__phenotypicAppPrefix || null")
    assert builder_prefix is None

    page.goto(hub_url + "/run/")
    page.wait_for_selector("#shell-top-bar")
    run_prefix = page.evaluate("() => window.__phenotypicAppPrefix || null")
    assert run_prefix is None
