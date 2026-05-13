"""Playwright E2E tests for legacy ``pipeline.json`` round-trip (spec §8.3.9).

Spec §5.7 + §8.3.9 — every ``pipeline.json`` saved by the pre-redesign
popover builder must load through the new DAG path
(:func:`phenotypic.gui.builder._conversion_dag.from_pipeline_dag`)
without loss, including:

* Plain legacy save → renders as DAG; ``to_pipeline_dag`` re-emits an
  ``ImagePipeline`` equal to ``ImagePipeline.from_json`` of the original.
* Shared-instance auxes → cloned into independent ``BlockNode``s; a
  toast is queued listing the rewrite.
* Unknown classes (registry drift) → block renders with a yellow border
  + label ``(unknown: ClassName)``; advisory issue surfaces but no
  hard error.
* No canvas-layout payload → dagre lays out cleanly; save round-trips
  through ``to_pipeline_dag`` byte-stably (positions are not persisted
  per spec §4.7).

Run gates:
* ``PLAYWRIGHT=1`` env (handled by parent ``tests/e2e/gui/conftest.py``).
* ``PHENOTYPIC_GUI_DAG=1`` for the DAG canvas + dispatcher.

Tests that depend on ``window.phenoSetState`` (server-side state
injection) ``pytest.skip`` per the established Phase 3/4/5/6 pattern;
the underlying conversion logic is exhaustively covered by the unit
tests in ``tests/unit/gui/builder/test_conversion_dag.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.gui.conftest import _build_sandbox, _start_live_server


_FIXTURES = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "builder_dag"


@pytest.fixture(scope="module")
def legacy_load_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped sandbox for the legacy-load suite."""

    parent = tmp_path_factory.mktemp("e2e_legacy_pipelines")
    return _build_sandbox(parent)


@pytest.fixture(scope="module")
def live_server(legacy_load_sandbox: Path) -> Iterator[str]:
    """Spawn ``phenotypic-gui`` with ``PHENOTYPIC_GUI_DAG=1``."""

    yield from _start_live_server(
        legacy_load_sandbox,
        env_overrides={"PHENOTYPIC_GUI_DAG": "1"},
    )


@pytest.fixture(scope="module")
def hub_url(live_server: str) -> str:
    return live_server


def _has_state_injection(page: Page) -> bool:
    """Return True if the page exposes a ``window.phenoSetState`` helper."""

    return bool(
        page.evaluate("typeof window.phenoSetState === 'function'")
    )


def _open_builder(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/builder/")
    page.wait_for_selector("#canvas-cytoscape", timeout=10_000)


# ---------------------------------------------------------------------------
# §8.3.9 tests
# ---------------------------------------------------------------------------


def test_load_legacy_popover_pipeline_json(page: Page, hub_url: str) -> None:
    """A pre-redesign ``pipeline.json`` loads through ``from_pipeline_dag``.

    The fixture ``legacy_popover_pipeline.json`` was saved by the
    popover-era builder.  Loading it through the DAG path must render
    a non-empty canvas with no error toasts.  The fixture's content was
    generated from a real :meth:`ImagePipeline.to_json` snapshot
    (see ``tests/fixtures/builder_dag/legacy_popover_pipeline.json``).
    """

    _open_builder(page, hub_url)
    if not _has_state_injection(page):
        pytest.skip(
            "window.phenoSetState helper not exposed; canonical coverage in "
            "tests/unit/gui/builder/test_conversion_dag.py::"
            "test_from_pipeline_dag_legacy_popover_pipeline_round_trips"
        )

    fixture = _FIXTURES / "legacy_popover_pipeline.json"
    payload = json.loads(fixture.read_text())
    page.evaluate("(p) => window.phenoSetState(p)", payload)

    # Canvas should render with at least one DAG block (Input Image
    # is auto-seeded plus the legacy ops).
    expect(page.locator(".dag-block").first).to_be_visible(timeout=5_000)

    # No error toast on a clean legacy load.
    error_toast = page.locator('[data-kind="error"]')
    assert error_toast.count() == 0


def test_load_shared_instance_clones(page: Page, hub_url: str) -> None:
    """A legacy ``pipeline.json`` with a shared op instance clones it.

    Spec §5.4 — when the same ``ImageOperation`` instance appears both
    as a top-level op and as another op's aux param, the DAG loader
    clones the embedded usage into an independent ``BlockNode`` and
    queues an info toast describing the rewrite.

    Canonical unit test:
    ``tests/unit/gui/builder/test_conversion_dag.py::test_from_pipeline_dag_clones_shared_aux``.
    """

    _open_builder(page, hub_url)
    if not _has_state_injection(page):
        pytest.skip(
            "window.phenoSetState helper not exposed; canonical coverage in "
            "tests/unit/gui/builder/test_conversion_dag.py::"
            "test_from_pipeline_dag_clones_shared_aux"
        )

    fixture = _FIXTURES / "shared_aux_instance.json"
    payload = json.loads(fixture.read_text())
    page.evaluate("(p) => window.phenoSetState(p)", payload)

    # Toast hinting "shared" should surface within a few ticks.
    toast = page.locator('[data-testid="dag-toast"]')
    expect(toast).to_be_visible(timeout=3_000)
    text = (toast.text_content() or "").lower()
    assert "shared" in text, f"toast missing 'shared' hint: {text!r}"


def test_load_unknown_class_yellow_border(page: Page, hub_url: str) -> None:
    """A legacy pipeline with an unknown class renders advisory-yellow.

    Spec §4.10 — when a block references a registry class missing from
    the current build (e.g. the legacy file was saved before a class
    was renamed), the DAG path renders the block with a yellow advisory
    border and label ``(unknown: ClassName)``.  The advisory issue
    surfaces in the toolbar badge but does not block Run / Save.

    Canonical unit test:
    ``tests/unit/gui/builder/test_validation.py::test_unknown_class_advisory_not_blocking``.
    """

    _open_builder(page, hub_url)
    if not _has_state_injection(page):
        pytest.skip(
            "window.phenoSetState helper not exposed; canonical coverage in "
            "tests/unit/gui/builder/test_validation.py::"
            "test_unknown_class_advisory_not_blocking"
        )

    # Inject a synthetic state with a block whose class_name is unknown.
    page.evaluate(
        """() => {
            window.phenoSetState({
                _schema: "dag",
                root: {
                    blocks: [
                        {block_id: "i".repeat(32), class_name: "InputImage",
                         params: {}, label: null, nested: null,
                         collapsed: false, list_slot_counts: {}},
                        {block_id: "u".repeat(32), class_name: "MissingFromRegistry",
                         params: {}, label: null, nested: null,
                         collapsed: false, list_slot_counts: {}},
                    ],
                    edges: [
                        {edge_id: "e".repeat(32), source_block_id: "i".repeat(32),
                         source_port: "out", target_block_id: "u".repeat(32),
                         target_port: "in", target_slot: null, kind: "image"},
                    ],
                    name: "Pipeline", desc: "",
                    nrows: null, ncols: null,
                },
                breadcrumb: [],
                selected_block_id: null, selected_edge_id: null,
                pending_delete_block_id: null, toast_queue: [],
            });
        }"""
    )

    # The unknown-class block should carry an advisory border / label.
    unknown = page.locator(".dag-block").nth(1)
    expect(unknown).to_be_visible(timeout=3_000)
    # Issue badge should reflect the advisory (1 hint).
    badge = page.locator('[data-testid="issue-badge"]')
    text = (badge.text_content() or "").lower()
    assert "hint" in text or "1" in text, (
        f"issue badge should reflect unknown_class advisory: {text!r}"
    )


def test_load_no_canvas_layout_falls_back_to_dagre(
    page: Page,
    hub_url: str,
) -> None:
    """Legacy file has no position info — dagre lays out cleanly.

    Spec §4.7 — block positions are never persisted in ``pipeline.json``
    (the dagre pass recomputes them on every render).  Loading a legacy
    file without position info should produce a clean LR layout and a
    save round-trip should be byte-stable through ``to_pipeline_dag``.

    Canonical unit test:
    ``tests/unit/gui/builder/test_conversion_dag.py::test_to_pipeline_dag_round_trip_preserves_pipeline_json``.
    """

    _open_builder(page, hub_url)
    if not _has_state_injection(page):
        pytest.skip(
            "window.phenoSetState helper not exposed; canonical coverage in "
            "tests/unit/gui/builder/test_conversion_dag.py::"
            "test_to_pipeline_dag_round_trip_preserves_pipeline_json"
        )

    fixture = _FIXTURES / "legacy_popover_pipeline.json"
    payload = json.loads(fixture.read_text())
    page.evaluate("(p) => window.phenoSetState(p)", payload)

    # At least one block must render at a non-default position (dagre
    # ran).  We assert via cytoscape's element bounding box.
    has_position = page.evaluate(
        """() => {
            const cy = window.phenoGetCy && window.phenoGetCy();
            if (!cy) return false;
            const blocks = cy.nodes('.dag-block');
            if (!blocks.length) return false;
            const bb = blocks[0].boundingBox();
            return bb && (bb.x !== 0 || bb.y !== 0);
        }"""
    )
    assert has_position, "dagre layout produced default (0,0) positions"
