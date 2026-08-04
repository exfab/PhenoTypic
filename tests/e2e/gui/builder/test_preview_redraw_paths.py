"""Playwright coverage for Builder preview stability across redraw callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.gui.builder.conftest import (
    _click_palette_button,
    _open_builder,
)
from tests.e2e.gui.conftest import _build_sandbox, _start_live_server


@pytest.fixture(scope="module")
def preview_redraw_sandbox(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Build a sandbox containing one loadable pipeline."""
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.enhance import BlurGauss

    sandbox = _build_sandbox(tmp_path_factory.mktemp("preview_redraw"))
    pipeline = ImagePipeline(
        ops=[BlurGauss(sigma=1.0), OtsuDetector()],
        name="preview-redraw-load",
    )
    (sandbox / "preview-redraw.json.pht-pipe").write_text(
        pipeline.to_json(),
        encoding="utf-8",
    )
    return sandbox


@pytest.fixture(scope="module")
def live_server(preview_redraw_sandbox: Path) -> Iterator[str]:
    """Run the GUI against the preview-redraw sandbox."""
    yield from _start_live_server(preview_redraw_sandbox)


@pytest.fixture(scope="module")
def hub_url(live_server: str) -> str:
    """Return the module's GUI base URL."""
    return live_server


def _remember_preview_mount(page: Page) -> None:
    page.evaluate(
        "() => { window.__previewMount = document.querySelector('#inspector-preview'); }"
    )


def _assert_same_preview_mount(page: Page) -> None:
    assert page.evaluate(
        "() => window.__previewMount === document.querySelector('#inspector-preview')"
    )


def _load_saved_pipeline(page: Page) -> None:
    page.locator("#btn-load").click()
    expect(page.locator("#modal-load-picker")).to_be_visible()
    page.locator("#btn-load-json-choice").click()
    page.locator(".list-group-item").filter(
        has_text="preview-redraw.json.pht-pipe"
    ).click()
    expect(page.locator("#modal-load-picker")).to_be_hidden()


def _load_prefab(page: Page) -> None:
    page.locator("#btn-load").click()
    expect(page.locator("#modal-load-picker")).to_be_visible()
    page.locator("#btn-load-prefab-choice").click()
    page.locator(".list-group-item").filter(
        has_text="HeavyOtsuPipeline"
    ).first.click()
    expect(page.locator("#modal-load-picker")).to_be_hidden()


def _expect_stale_same_mount(page: Page) -> None:
    expect(page.locator("#inspector-preview")).to_have_text(
        "Preview stale - run again"
    )
    _assert_same_preview_mount(page)


def test_load_prefab_and_delete_keep_preview_mount(
    page: Page,
    hub_url: str,
) -> None:
    """Every redraw path leaves preview ownership with the stable mount."""
    _open_builder(page, hub_url)
    _click_palette_button(page, "BlurGauss")
    page.locator("#btn-run-preview").click()
    expect(page.locator("#inspector-preview img")).to_have_count(
        1,
        timeout=20_000,
    )
    _remember_preview_mount(page)

    _load_saved_pipeline(page)
    _expect_stale_same_mount(page)

    page.locator(
        "button.linear-node-title-button",
        has_text="BlurGauss",
    ).click()
    page.locator("#btn-run-preview").click()
    expect(page.locator("#inspector-preview img")).to_have_count(
        1,
        timeout=20_000,
    )

    _load_prefab(page)
    _expect_stale_same_mount(page)

    _load_saved_pipeline(page)
    page.locator(
        "button.linear-node-title-button",
        has_text="BlurGauss",
    ).click()
    page.locator("#btn-run-preview").click()
    expect(page.locator("#inspector-preview img")).to_have_count(
        1,
        timeout=20_000,
    )
    page.locator(".linear-side-action-danger").dispatch_event("click")
    expect(page.locator("#confirm-delete-modal")).to_be_visible()
    page.locator("#btn-confirm-delete").click()
    expect(page.locator("#confirm-delete-modal")).to_be_hidden()
    _expect_stale_same_mount(page)
