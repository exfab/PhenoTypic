"""Browser-driven E2E proving results-viewer parity on a standalone bundle.

A portable, deliverables-only bundle (``deliverables/master_measurements.parquet``
+ mirror + overlays + ``deliverables/qc/`` summary/members, **no** per-image
``results/``) must open in the results viewer with the same curation + QC-review
affordances as a full run — minus the per-image pixel-layer toggle, which has no
HDF to source.

This module seeds one sandbox carrying **both** a standalone bundle and a full
run, boots one Werkzeug hub over it, and drives:

1. The viewer boots in **Standalone bundle** mode (header mode badge) and the
   colony grid renders tiles from the bundle's measurements.
2. Marking an error category (Debris) persists into
   ``<bundle>/deliverables/qc/curation_labels.parquet`` and survives a reload.
3. The QC review tab renders worklist rows + gallery tiles sourced from
   ``deliverables/qc/qc_members.parquet``.
4. The pixel-layer toggle is **absent** in the standalone bundle (no
   ``results/``) but **present** in the full-run fixture.

Mirrors ``test_radial_triage.py`` / ``test_qc_review_splitter.py`` for the
function-scoped sandbox + viewer-handoff harness. Gated by ``PLAYWRIGHT=1`` via
the module-level skip in ``conftest.py``.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Iterator

import polars as pl
import pytest
from PIL import Image as PILImage
from playwright.sync_api import Page, expect

from phenotypic import ImagePipeline
from phenotypic.analysis import ReplicateAgreement
from phenotypic.schema import CULTURE, EXPERIMENT, IMAGE
from phenotypic.sdk_ import curation_labels_parquet_path
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import run_qc
from tests._output_layout import (
    write_master,
    write_measurements_mirror,
    write_pipeline_json,
)
from tests.e2e.gui.conftest import (
    _build_sandbox,
    _start_live_server,
    bind_results_output,
    publish_coherent_terminal_evidence,
)

# Single-threaded Werkzeug dev server + Dash callback-chain timing flakes on
# GHA shared runners (skipped on CI via ``-m "not ci_flaky"``); the SUT is
# correct locally. See tests/CLAUDE.md.
pytestmark = pytest.mark.ci_flaky

_BUNDLE_DIRNAME = "standalone_bundle"
_FULLRUN_DIRNAME = "full_run"
_DATASET = "ds1"
_IMAGES = ("plate_001.tif", "plate_002.tif")
_NROWS, _NCOLS = 3, 4
_INSTANCE_ID = "qc-SE-standalone01"
_DATASET_COLUMN = str(EXPERIMENT.DATASET)
_TIME_COLUMN = str(CULTURE.TIME)


def _build_master() -> pl.DataFrame:
    """Two-image grid frame with bbox + centroid columns the viewer can load."""
    rows: list[dict[str, object]] = []
    label = 0
    for image in _IMAGES:
        for r in range(1, _NROWS + 1):
            for c in range(1, _NCOLS + 1):
                label += 1
                rows.append(
                    {
                        _DATASET_COLUMN: _DATASET,
                        str(IMAGE.IMAGE_NAME): image,
                        _TIME_COLUMN: 0.0,
                        "Object_Label": label,
                        "Grid_RowNum": r,
                        "Grid_ColNum": c,
                        "Bbox_MinRR": 40,
                        "Bbox_MaxRR": 60,
                        "Bbox_MinCC": 40,
                        "Bbox_MaxCC": 60,
                        "Bbox_CenterRR": 50.0,
                        "Bbox_CenterCC": 50.0,
                        "Size_Area": float(100 + r * 10 + c),
                    }
                )
    return pl.DataFrame(rows)


def _pipeline() -> ImagePipeline:
    """A pipeline carrying one ReplicateAgreement QC entry grouped by row."""
    pipeline = ImagePipeline(name="standalone-bundle-e2e")
    pipeline.set_qc(
        [
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={
                    "on": "Size_Area",
                    "groupby": ["Grid_RowNum"],
                    "min_replicates": 2,
                },
                instance_id=_INSTANCE_ID,
                enabled=True,
            )
        ]
    )
    return pipeline


def _write_overlays(deliverables_base: Path) -> None:
    """Write a real overlay PNG per image so the colony grid emits ``<img>`` tiles."""
    overlays = deliverables_base / "overlays" / _DATASET
    overlays.mkdir(parents=True, exist_ok=True)
    for image in _IMAGES:
        stem = Path(image).stem
        PILImage.new("RGB", (96, 96), (180, 120, 60)).save(overlays / f"{stem}.png")


def _seed_standalone_bundle(sandbox: Path) -> Path:
    """Seed a deliverables-only bundle (no ``results/``); return its deliverables dir.

    ``run_qc`` + the output-layout helpers resolve every artefact through
    ``deliverables_dir(out)``, so writing against ``out`` lands the master,
    mirror, overlays, pipeline, and ``qc/`` under ``out/deliverables/`` with no
    sibling ``results/`` — exactly the portable bundle shape. The viewer is then
    pointed at ``out/deliverables`` itself.
    """
    out = sandbox / _BUNDLE_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    master = _build_master()
    write_master(out, master)
    write_measurements_mirror(out, master)
    deliverables_base = out / "deliverables"
    _write_overlays(deliverables_base)
    pipeline = _pipeline()
    write_pipeline_json(out, pipeline)
    # qc_summary.parquet + qc_members.parquet under out/deliverables/qc/.
    run_qc(master.to_pandas(), pipeline, out)
    # Deliberately NO ``out/results/`` — this is what makes it a bundle.
    return deliverables_base


def _seed_full_run(sandbox: Path) -> Path:
    """Seed a full run (with ``results/`` + overlays); return its output dir.

    Used only to prove the pixel-layer toggle renders when per-image
    ``results/`` are present (the toggle is gated on ``has_results``).
    """
    out = sandbox / _FULLRUN_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    master = _build_master()
    write_master(out, master)
    write_measurements_mirror(out, master)
    _write_overlays(out / "deliverables")
    # The per-image results/ tree drives ``layout.has_results`` -> layer toggle.
    (out / "results" / _DATASET / "measurements").mkdir(parents=True, exist_ok=True)
    write_pipeline_json(out, _pipeline())
    publish_coherent_terminal_evidence(out, total_images=len(_IMAGES))
    return out


@pytest.fixture
def fake_sandbox(tmp_path: Path) -> Path:
    """Function-scoped sandbox carrying both a standalone bundle and a full run."""
    sandbox = _build_sandbox(tmp_path)
    _seed_standalone_bundle(sandbox)
    _seed_full_run(sandbox)
    return sandbox


@pytest.fixture
def live_server(fake_sandbox: Path) -> Iterator[str]:
    """Function-scoped live server over the dual-output sandbox."""
    yield from _start_live_server(fake_sandbox)


@pytest.fixture
def hub_url(live_server: str) -> str:
    """Alias for the live-server base URL."""
    return live_server


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------


def _hand_off_viewer(page: Page, hub_url: str, output_rel: str) -> None:
    """POST a sandbox-relative output path to the viewer-handoff endpoint."""
    bind_results_output(page, hub_url, output_rel)


def _open_viewer_colony(page: Page, hub_url: str) -> None:
    """Load the viewer and switch to the Colony tab."""
    page.goto(hub_url + "/results/")
    page.wait_for_selector("#colony-grid-container", state="attached", timeout=20_000)
    colony_tab = page.locator("a.nav-link", has_text="Colony").first
    if colony_tab.count():
        colony_tab.click()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_standalone_bundle_boots_in_bundle_mode_with_tiles(
    page: Page, hub_url: str
) -> None:
    """The viewer opens a deliverables-only bundle in Standalone-bundle mode.

    Covers parity item (a): the colony grid renders tiles from the bundle's
    measurements, and the header mode badge reads "Standalone bundle".
    """
    _hand_off_viewer(page, hub_url, f"{_BUNDLE_DIRNAME}/deliverables")
    _open_viewer_colony(page, hub_url)

    badge = page.locator("#header-mode-badge")
    expect(badge).to_have_text("Standalone bundle", timeout=15_000)

    # Tiles render -> the measurements frame loaded and produced rows.
    tile = page.locator("#colony-grid-container .colony-cell").first
    expect(tile).to_be_visible(timeout=20_000)


def test_standalone_bundle_curation_persists_and_reloads(
    page: Page, hub_url: str, fake_sandbox: Path
) -> None:
    """Marking Debris on a bundle writes the durable labels parquet + survives reload.

    Covers parity item (b): the curated label persists into
    ``<bundle>/deliverables/qc/curation_labels.parquet`` and the tile renders a
    debris badge again after a full page reload.
    """
    _hand_off_viewer(page, hub_url, f"{_BUNDLE_DIRNAME}/deliverables")
    _open_viewer_colony(page, hub_url)

    trigger = page.locator("button.radial-badge--neutral").first
    expect(trigger).to_be_visible(timeout=20_000)
    trigger.click()

    debris_wedge = page.locator("button.radial-wedge", has_text="debris").first
    expect(debris_wedge).to_be_visible(timeout=10_000)
    debris_wedge.click()

    # The durable labels store lands under <bundle>/deliverables/qc/.
    labels_path = curation_labels_parquet_path(fake_sandbox / _BUNDLE_DIRNAME)
    deadline = _time.monotonic() + 15.0
    while _time.monotonic() < deadline and not labels_path.exists():
        page.wait_for_timeout(250)
    assert labels_path.exists(), (
        f"curation_labels.parquet was not written within 15s (expected at "
        f"{labels_path})"
    )
    labels = pl.read_parquet(labels_path)
    assert labels.height >= 1

    # The marked tile flips to a debris category badge.
    expect(
        page.locator("button.radial-badge", has_text="debris").first
    ).to_be_visible(timeout=15_000)

    # Reload the viewer: the durable parquet rehydrates the label at boot.
    _open_viewer_colony(page, hub_url)
    expect(
        page.locator("button.radial-badge", has_text="debris").first
    ).to_be_visible(timeout=20_000)


@pytest.mark.skip(
    reason=(
        "The QC tab is unmounted by "
        "docs/superpowers/specs/2026-08-26-gui-simplification-removals "
        "(spec section 3). Only this test in the module drives it; the "
        "bundle-mode, curation and layer-toggle tests keep running. Delete "
        "this marker when the surface returns."
    )
)
def test_standalone_bundle_qc_review_renders_members(
    page: Page, hub_url: str
) -> None:
    """The QC review tab renders worklist rows + gallery tiles from qc_members.

    Covers parity item (c): the bundle's ``deliverables/qc/qc_members.parquet``
    drives the Review worklist + detail gallery just like a full run.
    """
    _hand_off_viewer(page, hub_url, f"{_BUNDLE_DIRNAME}/deliverables")

    page.goto(hub_url + "/results/")
    page.wait_for_selector("#qc-cards-container", state="attached", timeout=15_000)
    page.locator("a.nav-link", has_text="QC").first.click()
    page.wait_for_selector(
        'label[for$="qc-subview-toggle_input_review"]', timeout=10_000
    )
    page.locator('label[for$="qc-subview-toggle_input_review"]').click()

    # Worklist rows (from qc_summary) + at least one gallery tile (from qc_members).
    page.wait_for_selector(".qc-worklist-row", timeout=15_000)
    gallery_tile = page.locator("#qc-review-gallery .colony-cell").first
    expect(gallery_tile).to_be_visible(timeout=15_000)


def test_layer_toggle_absent_in_bundle_present_in_full_run(
    page: Page, hub_url: str
) -> None:
    """Covers parity item (d): the layer toggle tracks ``has_results``.

    Absent for the standalone bundle (no per-image HDFs to source), present for
    a full run.
    """
    # Standalone bundle: no per-image results/ -> no layer toggle.
    _hand_off_viewer(page, hub_url, f"{_BUNDLE_DIRNAME}/deliverables")
    _open_viewer_colony(page, hub_url)
    expect(page.locator("#header-mode-badge")).to_have_text(
        "Standalone bundle", timeout=15_000
    )
    assert page.locator("#colony-layer-toggle").count() == 0, (
        "the pixel-layer toggle must be hidden for a standalone bundle"
    )

    # Full run: results/ present -> layer toggle renders.
    _hand_off_viewer(page, hub_url, _FULLRUN_DIRNAME)
    _open_viewer_colony(page, hub_url)
    expect(page.locator("#header-mode-badge")).to_have_text(
        "Full run", timeout=15_000
    )
    expect(page.locator("#colony-layer-toggle")).to_be_visible(timeout=15_000)
