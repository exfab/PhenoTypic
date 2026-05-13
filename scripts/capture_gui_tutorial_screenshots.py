"""Regenerate the GUI walkthrough tutorial screenshots.

Run from the repo root::

    uv run python scripts/capture_gui_tutorial_screenshots.py

The script:

1. Builds a synthetic 3-plate yeast dataset under
   ``docs/source/_static/gui_images/_dataset/``.
2. Runs the CLI once against that dataset to produce real CLI output
   (master parquet + ``results/`` + ``dashboard.html``).
3. Boots ``phenotypic-gui --root <dataset_parent>`` on a free port.
4. Drives a headless Chromium browser through every tutorial workflow
   and saves PNGs into ``docs/source/_static/gui_images/<workflow>/``.
5. Tears down the GUI subprocess.

Re-running this script overwrites the existing screenshots — that is the
point. Pass ``--force`` to also regenerate the synthetic dataset.

Requires Playwright + Chromium. Install on first run::

    uv run playwright install chromium
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_ROOT = REPO_ROOT / "docs" / "source" / "_static" / "gui_images"
DATASET_DIR = ASSETS_ROOT / "_dataset"
PLATES_DIR = DATASET_DIR / "plates"
METADATA_CSV = DATASET_DIR / "metadata.csv"
PIPELINE_JSON = DATASET_DIR / "pipeline.json"
OUTPUT_DIR = DATASET_DIR / "results"

VIEWPORT = {"width": 1280, "height": 900}


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

PIPELINE_DOC = {
    "version": "0.1.0",
    "name": "gui_tutorial",
    "desc": "Synthetic yeast tutorial pipeline",
    "reset": False,
    "pipe_cfgs": {
        "GaussianBlur": {
            "class": "GaussianBlur",
            "params": {"sigma": 2},
        },
        "OtsuDetector": {
            "class": "OtsuDetector",
            "params": {"ignore_zeros": True},
        },
    },
    "meas": {
        "MeasureShape": {"class": "MeasureShape", "params": {}},
        "MeasureSize": {"class": "MeasureSize", "params": {}},
    },
    "post": {},
    # The analysis sub-app's tutorial walkthrough renders better when the
    # synthetic CLI run produces a real ``analysis.parquet`` to demo the
    # populated state. ``EdgeCorrector`` correction needs at least one
    # interior + one edge colony per group to compute a threshold, and
    # ``LogGrowthModel`` fits expect ``Metadata_Time``-keyed measurements
    # — neither holds for the single-timepoint synthetic dataset, so the
    # filter/model are configured here primarily as recipe metadata; the
    # CLI's ``_emit_analysis_outputs`` swallows the resulting fit failure
    # at WARNING and the master output is unaffected.
    "filters": {
        "TukeyOutlierRemover": {
            "class": "TukeyOutlierRemover",
            "params": {"on": "Shape_Area", "groupby": ["Metadata_StrainID"], "k": 3.0},
        },
    },
    "model": {
        "class": "LogGrowthModel",
        "params": {
            "on": "Shape_Area",
            "groupby": ["Metadata_StrainID"],
            "time_label": "Metadata_RunDate",
            "n_jobs": 1,
        },
    },
    "nrows": 8,
    "ncols": 12,
}

METADATA_ROWS = [
    "Metadata_ImageName,Metadata_StrainID,Metadata_MatingType,"
    "Metadata_Media,Metadata_RunDate,Metadata_PlateNum,"
    "Metadata_Replicate,Grid_RowNum,Grid_ColNum",
    "plate_001.tif,SYN_001,a,YPD,2026-05-01,1,1,8,12",
    "plate_002.tif,SYN_002,A,YPD,2026-05-01,2,1,8,12",
    "plate_003.tif,SYN_003,a,SGAL,2026-05-01,3,1,8,12",
]


def build_tutorial_dataset(force: bool = False) -> None:
    """Populate the dataset directory with synthetic plates + metadata + pipeline.

    Layout::

        _dataset/
            plates/
                plate_001.tif        # 8x12 grid, seed 1
                plate_002.tif        # 8x12 grid, seed 2
                plate_003.tif        # 8x12 grid, seed 3
            metadata.csv             # synthetic plate metadata (Neurospora-style schema)
            pipeline.json            # GaussianBlur + OtsuDetector + MeasureShape + MeasureSize
    """
    if DATASET_DIR.exists() and not force:
        print(f"[dataset] reusing existing {DATASET_DIR.relative_to(REPO_ROOT)}")
        return

    from phenotypic.data import make_synthetic_plate
    import imageio.v3 as iio

    print(f"[dataset] generating fresh dataset under {DATASET_DIR.relative_to(REPO_ROOT)}")
    PLATES_DIR.mkdir(parents=True, exist_ok=True)

    for i, seed in enumerate((1, 2, 3), start=1):
        arr = make_synthetic_plate(
            nrows=8,
            ncols=12,
            plate_h=1024,
            plate_w=1536,
            seed=seed,
            spacing_factor=0.85,
            colony_size_variation=0.15,
        )
        out = PLATES_DIR / f"plate_{i:03d}.tif"
        iio.imwrite(out, arr)
        print(f"[dataset]   wrote {out.name} ({arr.shape}, {arr.dtype})")

    METADATA_CSV.write_text("\n".join(METADATA_ROWS) + "\n", encoding="utf-8")
    print(f"[dataset]   wrote metadata.csv ({len(METADATA_ROWS) - 1} rows)")

    PIPELINE_JSON.write_text(json.dumps(PIPELINE_DOC, indent=2), encoding="utf-8")
    print("[dataset]   wrote pipeline.json")


def run_cli_once() -> None:
    """Invoke ``python -m phenotypic`` to produce real CLI output.

    The output (parquet + per-image overlays + dashboard.html) is what the
    "Viewing Results" walkthrough screenshots capture. Skip if the output
    already exists — re-running this is expensive (~minutes on a real
    pipeline; ~seconds on the synthetic dataset but still avoidable).
    """
    if (OUTPUT_DIR / "master_measurements.parquet").exists():
        print(f"[cli] reusing existing CLI output at {OUTPUT_DIR.relative_to(REPO_ROOT)}")
        return

    print("[cli] running pipeline against synthetic dataset")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "phenotypic",
        str(PIPELINE_JSON),
        str(PLATES_DIR),
        "-o",
        str(OUTPUT_DIR),
        "--n-jobs",
        "1",
    ]
    print(f"[cli]   {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    print("[cli]   done")


# ---------------------------------------------------------------------------
# GUI server lifecycle
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_http_200(url: str, *, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, ConnectionRefusedError, OSError) as err:
            last_err = err
        time.sleep(0.2)
    raise RuntimeError(
        f"GUI did not respond at {url} within {timeout}s (last error: {last_err!r})"
    )


def boot_gui(root: Path) -> tuple[subprocess.Popen[str], str]:
    """Boot ``phenotypic-gui`` on a free port. Returns (process, base_url)."""
    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "phenotypic.gui",
        "--root",
        str(root),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    print(f"[gui] booting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    _wait_for_http_200(base_url + "/", timeout=30.0)
    print(f"[gui]   ready at {base_url}")
    return proc, base_url


def shutdown_gui(proc: subprocess.Popen[str]) -> None:
    print("[gui] shutting down")
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------

def capture_workflow_screenshots(base_url: str, headed: bool = False) -> None:
    """Drive the GUI through every workflow and save screenshots.

    Each workflow gets its own fresh page so state from one section
    cannot leak into another (e.g. expanded sidebar tree).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Playwright is not installed. Install it with:\n"
            "  uv add --group dev playwright\n"
            "  uv run playwright install chromium"
        ) from exc

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        try:
            context = browser.new_context(viewport=VIEWPORT)
            _capture_setup(context, base_url)
            _capture_file_explorer(context, base_url)
            _capture_build_pipeline(context, base_url)
            _capture_run_local(context, base_url)
            _capture_run_slurm(context, base_url)
            _capture_view_results(context, base_url)
            _capture_pick_points(context, base_url)
            _capture_analysis(context, base_url)
            _capture_aux_ports(context, base_url)
            _capture_qc_curation_loop(context, base_url)
            _capture_heatmap_exploration(context, base_url)
        finally:
            browser.close()


def _save(page, workflow: str, name: str) -> None:
    target_dir = ASSETS_ROOT / workflow
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / name
    page.screenshot(path=str(out), full_page=False)
    print(f"[shot]   {workflow}/{name}")


def _new_page(context, base_url: str, path: str = "/"):
    page = context.new_page()
    page.goto(base_url + path)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(500)  # let Dash callbacks settle
    return page


# --- setup --------------------------------------------------------------

def _capture_setup(context, base_url: str) -> None:
    print("[shot] workflow=setup")
    page = _new_page(context, base_url, "/")
    _save(page, "setup", "01_landing_page.png")
    page.close()


# --- file explorer ------------------------------------------------------

def _capture_file_explorer(context, base_url: str) -> None:
    print("[shot] workflow=file_explorer")
    page = _new_page(context, base_url, "/")

    page.wait_for_selector("#shell-sidebar-tree", timeout=15_000)
    _save(page, "file_explorer", "01_sidebar_collapsed.png")

    # Expand the plates folder by clicking its row in the sidebar.
    plates_selector = (
        'button[id*="\\"path\\":\\"plates\\""][id*="shell-sidebar-entry"]'
    )
    if page.locator(plates_selector).count() > 0:
        page.click(plates_selector)
        page.wait_for_timeout(800)
        _save(page, "file_explorer", "02_sidebar_expanded.png")

    # Capability badges shot — same view, but cropped tighter via
    # an element screenshot rather than the full page.
    sidebar = page.locator("#shell-sidebar-tree")
    if sidebar.count() > 0:
        target_dir = ASSETS_ROOT / "file_explorer"
        target_dir.mkdir(parents=True, exist_ok=True)
        sidebar.screenshot(path=str(target_dir / "03_capability_badges.png"))
        print("[shot]   file_explorer/03_capability_badges.png")

    # Collapsed-sidebar shot — click the chrome's collapse button so the
    # whole file explorer slides off, then capture the page so readers
    # see the wide-canvas mode the toggle unlocks.
    collapse_btn = page.locator("#shell-sidebar-collapse-button")
    if collapse_btn.count() > 0:
        collapse_btn.click()
        page.wait_for_timeout(400)  # CSS transition is 200ms
        _save(page, "file_explorer", "04_sidebar_hidden.png")
        # Restore the expanded state so the next workflow's screenshots
        # still show the sidebar (localStorage persists otherwise).
        collapse_btn.click()
        page.wait_for_timeout(400)

    page.close()


# --- build pipeline -----------------------------------------------------

def _capture_build_pipeline(context, base_url: str) -> None:
    print("[shot] workflow=build_pipeline")
    page = _new_page(context, base_url, "/builder/")
    _save(page, "build_pipeline", "01_builder_empty.png")
    page.close()


# --- run console (local) ------------------------------------------------

def _capture_run_local(context, base_url: str) -> None:
    print("[shot] workflow=run_local")
    page = _new_page(context, base_url, "/run/")
    _save(page, "run_local", "01_run_console_form.png")

    # Open the modal browser for the input dir picker so readers can see
    # what file selection looks like.
    btn = page.locator("#rc-btn-pick-input")
    if btn.count() > 0:
        btn.click()
        page.wait_for_timeout(500)
        _save(page, "run_local", "02_input_picker_modal.png")
        # Close modal
        cancel = page.locator("#rc-btn-input-cancel")
        if cancel.count() > 0:
            cancel.click()
            page.wait_for_timeout(300)

    # Recent Runs row + dashboard iframe (real CLI output produced earlier).
    page.wait_for_timeout(500)
    _save(page, "run_local", "03_recent_runs_panel.png")

    page.close()


# --- run console (slurm) -------------------------------------------------

def _capture_run_slurm(context, base_url: str) -> None:
    print("[shot] workflow=run_slurm")
    page = _new_page(context, base_url, "/run/")

    # Toggle SLURM mode in the radio. dbc.RadioItems renders as labels
    # with hidden inputs; clicking the visible label is the reliable path.
    slurm_label = page.locator('label:has-text("SLURM")').first
    if slurm_label.count() > 0:
        slurm_label.click()
        page.wait_for_timeout(400)

    # Open SLURM collapse if it isn't already.
    toggle = page.locator("#rc-btn-toggle-slurm")
    if toggle.count() > 0:
        toggle.click()
        page.wait_for_timeout(400)

    _save(page, "run_slurm", "01_slurm_mode.png")
    page.close()


# --- view results -------------------------------------------------------

def _capture_view_results(context, base_url: str) -> None:
    """Empty-state hub viewer screenshot only.

    Loaded-viewer screenshots are captured separately by
    :func:`capture_standalone_viewer_screenshots` after the hub is torn
    down — the hub's viewer mount renders empty in v1 (rebuild-on-select
    is not wired), so we boot the standalone viewer pointed at the real
    CLI output for the populated screenshots.
    """
    print("[shot] workflow=view_results (empty state via hub)")
    page = _new_page(context, base_url, "/results/")
    _save(page, "view_results", "01_viewer_empty.png")
    page.close()


def _capture_pick_points(context, base_url: str) -> None:
    """Drive the in-builder point-picker workflow and capture eight PNGs.

    The shots demonstrate the manual-curation flow described in
    ``docs/source/tutorials/gui/07_pick_points.md``:

    1. Palette with the PICK badge visible on the two pickable ops.
    2. Canvas with ``GaussianBlur → OtsuDetector → ManualSelector``.
    3. Inspector param form for ``ManualSelector`` (count = "0 points").
    4. Picker modal open on the original RGB plate.
    5. Picker modal toggled to the predecessor's intermediate.
    6. RGB plate after staging three picks via ``set_props``.
    7. Param form outside the modal showing ``"3 points"`` after Confirm.
    8. Inspector preview after re-running preview (curated objmap).

    The helper does NOT issue real OSD canvas clicks — the synthetic
    plate's pixel coordinates are embedded directly into
    ``picker-staged-store`` via ``window.dash_clientside.set_props`` so
    the screenshots are deterministic regardless of viewport / DPR.
    """
    print("[shot] workflow=pick_points")
    page = _new_page(context, base_url, "/builder/")

    # Wait for the palette to populate.
    page.wait_for_selector("#palette", timeout=15_000)
    page.wait_for_timeout(500)

    # 1) Palette with PICK badge.
    # Open every Operations accordion section so the PICK-badged ops are
    # visible in a single shot AND so subsequent _add_op clicks can reach
    # ops in any category (Bootstrap collapsed items have ``display:
    # none``, so Playwright considers them invisible). Builder.css renders
    # the accordion with ``always_open=True``, but the *active* item is the
    # first one only — clicking the headers opens the rest.
    for header_text in ("Corrector", "Detector", "Enhancer", "Refiner"):
        header = page.locator(
            f'button.accordion-button:has-text("{header_text}")'
        ).first
        if header.count() > 0:
            try:
                cls = header.get_attribute("class") or ""
                if "collapsed" in cls:
                    header.click()
                    page.wait_for_timeout(250)
            except Exception:  # pragma: no cover - best-effort
                pass
    page.wait_for_timeout(300)
    _save(page, "pick_points", "01_palette_with_badge.png")

    # 2) Pipeline with GaussianBlur → OtsuDetector → ManualSelector.
    # Each palette button is a pattern-matching id of the form
    # {"type": "palette-add", "class_name": "<name>"}; Dash hashes the dict
    # to a JSON string when it renders the DOM, so the selector below
    # matches the rendered id substring.
    def _add_op(class_name: str) -> None:
        sel = (
            f'button[id*="\\"type\\":\\"palette-add\\""]'
            f'[id*="\\"class_name\\":\\"{class_name}\\""]'
        )
        page.click(sel)
        page.wait_for_timeout(400)

    for cls in ("GaussianBlur", "OtsuDetector", "ManualSelector"):
        _add_op(cls)
    page.wait_for_timeout(600)
    _save(page, "pick_points", "02_pipeline_with_selector.png")

    # 3) Inspector param form for ManualSelector. Click the node on the
    # canvas so the inspector opens against it. Cytoscape renders nodes as
    # SVG groups; the simplest reliable handle is the node label
    # rendered inside Cytoscape — but Cytoscape uses a canvas backend
    # without per-node DOM nodes, so we trigger selection programmatically
    # via the cy instance.
    page.evaluate(
        """
        () => {
            const cy = window.cy || (window._cy_instances && window._cy_instances[0]);
            if (!cy) return;
            const nodes = cy.nodes(`[label *= "ManualSelector"], [class_name = "ManualSelector"]`);
            if (nodes.length > 0) {
                cy.elements().unselect();
                nodes[0].select();
            }
        }
        """
    )
    page.wait_for_timeout(800)
    _save(page, "pick_points", "03_param_form.png")

    # Run preview once so OtsuDetector caches its intermediate. The
    # picker modal needs the predecessor cache to enable the
    # "Input to this op" radio option.
    preview_btn = page.locator("#btn-run-preview")
    if preview_btn.count() > 0:
        preview_btn.click()
        page.wait_for_timeout(2500)

    # 4) Open the picker modal. The picker button is rendered with a
    # pattern-matching id of the form
    # {"type": "param-point-picker-btn", "prefix": <node_id>, "name": "centers"}.
    picker_btn_sel = (
        'button[id*="\\"type\\":\\"param-point-picker-btn\\""]'
    )
    page.click(picker_btn_sel)
    # Wait for OSD canvas + the modal to settle (aria-busy flips off when
    # the tile pyramid finishes rendering the first level).
    page.wait_for_selector(
        '[data-testid="point-picker-osd-canvas"]', timeout=15_000
    )
    osd = page.locator('[data-testid="point-picker-osd-canvas"]')
    try:
        osd.wait_for(state="visible", timeout=10_000)
    except Exception:  # pragma: no cover - best-effort
        pass
    # Wait for aria-busy to flip false on the parent or a short fallback.
    try:
        page.wait_for_function(
            """
            () => {
                const el = document.querySelector('[data-testid="point-picker-osd-canvas"]');
                if (!el) return false;
                const busy = el.getAttribute('aria-busy');
                return busy === null || busy === 'false';
            }
            """,
            timeout=10_000,
        )
    except Exception:  # pragma: no cover - best-effort
        page.wait_for_timeout(1500)
    page.wait_for_timeout(800)
    _save(page, "pick_points", "04_modal_rgb.png")

    # 5) Toggle the channel radio to "Input to this op".
    intermediate_label = page.locator(
        'label:has-text("Input to this op")'
    ).first
    if intermediate_label.count() > 0:
        intermediate_label.click()
        # First toggle dumps + tiles the intermediate PNG; allow time.
        page.wait_for_timeout(2000)
        try:
            page.wait_for_function(
                """
                () => {
                    const el = document.querySelector('[data-testid="point-picker-osd-canvas"]');
                    if (!el) return false;
                    const busy = el.getAttribute('aria-busy');
                    return busy === null || busy === 'false';
                }
                """,
                timeout=10_000,
            )
        except Exception:  # pragma: no cover - best-effort
            page.wait_for_timeout(1500)
        page.wait_for_timeout(500)
    _save(page, "pick_points", "05_modal_intermediate.png")

    # Toggle back to RGB before staging the points so the final shot
    # shows red markers on the RGB plate (more visually intuitive).
    rgb_label = page.locator('label:has-text("Original RGB")').first
    if rgb_label.count() > 0:
        rgb_label.click()
        page.wait_for_timeout(800)

    # 6) Push three test points into the staged store. Coordinates target
    # well-separated colonies on the synthetic 8x12 plate (1024x1536).
    # The clientside redraw subscribes to picker-staged-store changes and
    # paints red OSD overlay markers.
    page.evaluate(
        """
        () => {
            const points = [[160, 240], [400, 700], [720, 1180]];
            window.dash_clientside.set_props(
                'picker-staged-store', {data: points}
            );
        }
        """
    )
    page.wait_for_timeout(800)
    _save(page, "pick_points", "06_three_picks.png")

    # 7) Confirm and capture the param form's count label.
    confirm_btn = page.locator("#btn-picker-confirm")
    if confirm_btn.count() > 0:
        confirm_btn.click()
        # Modal closes; param form re-renders with the new count.
        page.wait_for_timeout(1000)
    _save(page, "pick_points", "07_param_form_after_confirm.png")

    # 8) Re-run preview against the curated pipeline and snapshot the
    # inspector's preview pane.
    if preview_btn.count() > 0:
        preview_btn.click()
        page.wait_for_timeout(2500)
    _save(page, "pick_points", "08_preview_after_curation.png")

    page.close()


def _capture_aux_ports(context, base_url: str) -> None:
    """Drive the aux-port popover workflow and capture nine PNGs.

    The shots demonstrate the canvas-anchored aux popover flow described
    in ``docs/source/tutorials/gui/09_aux_ports.md``. Eight PNGs total
    cover the scalar-aux flow plus the list-typed multi-slot scenario:

    1. ``01_initial.png`` — empty builder canvas with the palette visible.
    2. ``02_main_pipeline.png`` — four ribbon ops (``GaussianBlur`` →
       ``ContrastStretching`` → ``FilamentousFungiDetector`` →
       ``MeasureSize``) wired left-to-right; the FFD node carries a
       hollow purple ``aux-port`` square on its bottom edge.
    3. ``03_popover_empty.png`` — the FFD aux port has been tapped; the
       canvas-anchored popover is open in palette mode listing every
       compatible ``ObjectDetector`` / ``ImagePipeline`` class.
    4. ``04_popover_wired.png`` — ``OtsuDetector`` picked from the
       palette; popover transitions to its wired-row state (class label
       plus ``Edit`` / ``Drill in`` / ``Disconnect`` actions). Aux port
       marker flips to the filled ``aux-port--wired`` variant.
    5. ``05_drill_in.png`` — ``Drill in →`` clicked; canvas swaps to the
       drilled aux scope (single-op ribbon with just ``OtsuDetector``)
       and the breadcrumb shows the drill path. To extend a scalar aux
       into a multi-step inline pipeline, the user picks ``ImagePipeline``
       from the popover palette instead of a concrete detector class —
       drilling into an ``ImagePipeline`` aux opens a writable nested
       scope that accepts palette adds. (Single-op auxes are surfaced
       as a 1-step wrapper for visual continuity but are not editable
       in place.)
    6. ``06_drill_out.png`` — first breadcrumb crumb clicked; canvas
       restores to the original 4-step main ribbon with the aux port
       still in its wired (filled-purple) state.
    7. ``07_list_port_popover.png`` — a ``CompositeDetector`` is added
       to the ribbon and its ``detectors`` aux port tapped; the popover
       opens in list mode with a ``+ Add slot`` button (the empty
       placeholder is the bootstrap affordance for a list-typed param).
    8. ``08_list_port_two_wired.png`` — two slots wired via ``+ Add slot``
       + class pick (``OtsuDetector`` at slot 0, ``WatershedDetector``
       at slot 1); per-slot ``✎ / → / ⨯`` actions are visible on every
       wired row.
    9. ``09_per_slot_disconnect.png`` — slot 0's per-slot ⨯ Disconnect
       clicked; that slot's row reverts to the class palette while
       slot 1 stays wired (per-slot independence inside one popover).

    Implementation notes
    --------------------
    Aux ports are rendered as cytoscape *nodes* (not DOM elements) with
    flat ids of the form ``"aux-port__<target_node_id>__<param>"`` (see
    :func:`phenotypic.gui.builder._ids._encode_aux_port_id`). Because
    cytoscape paints to a single ``<canvas>`` element, the only reliable
    way to click an aux port from Playwright is to emit a ``tap`` event
    on the node via the live cy instance — exposed by
    ``window.phenoGetCy()`` from ``assets/builder.js`` and bound by
    ``assets/aux_popover.js`` on every cy refresh.

    The popover container itself (id ``cy-popover-container``, class
    ``cy-popover``) IS a DOM element, so its action buttons can be
    located with normal CSS selectors. Each action button's id is a
    Dash pattern-matching dict serialised to JSON:
    ``{"type": "popover-action", "action": "pick_class" | "edit" |
    "drill" | "disconnect" | "add_slot", "target_node_id": ...,
    "param": ..., "slot": ..., "class_name": ...}``.

    Wait-target selectors used below:
      * ``.cy-popover-palette`` — palette-mode popover body.
      * ``.cy-popover-wired-row`` — wired-mode popover body.
      * ``.pheno-breadcrumb`` — scope breadcrumb nav (always present).
    """
    print("[shot] workflow=aux_ports")
    page = _new_page(context, base_url, "/builder/")

    # Wait for the palette to populate.
    page.wait_for_selector("#palette", timeout=15_000)
    page.wait_for_timeout(500)

    # 1) Empty canvas — same starting point as the Build Pipeline tutorial.
    _save(page, "aux_ports", "01_initial.png")

    # Open every Operations / Measurements accordion section so palette
    # buttons for ops in any category are reachable.
    # ``always_open=True`` only auto-expands the first item — clicking
    # the headers expands the rest. The Measure palette ("Measurements")
    # lives in a separate accordion (``palette-meas``) from the image-
    # ops palette (``palette``); a single accordion-button header
    # selector covers both.
    for header_text in ("Corrector", "Detector", "Enhancer", "Refiner", "Measure"):
        header = page.locator(
            f'button.accordion-button:has-text("{header_text}")'
        ).first
        if header.count() > 0:
            try:
                cls = header.get_attribute("class") or ""
                if "collapsed" in cls:
                    header.click()
                    page.wait_for_timeout(250)
            except Exception:  # pragma: no cover - best-effort
                pass
    page.wait_for_timeout(300)

    # Helper: add a palette op by class_name. Mirrors _capture_pick_points.
    def _add_op(class_name: str) -> None:
        sel = (
            f'button[id*="\\"type\\":\\"palette-add\\""]'
            f'[id*="\\"class_name\\":\\"{class_name}\\""]'
        )
        loc = page.locator(sel)
        if loc.count() > 0:
            loc.first.click()
            page.wait_for_timeout(600)

    # Helper: tap an aux port via the live cytoscape instance.
    # Aux port nodes have ids of the form
    # ``aux-port__<target_node_id>__<param>`` and the popover's
    # clientside glue (``aux_popover.js``) binds ``cy.on('tap', 'node[id
    # ^= "aux-port__"]', ...)``. Emitting a ``tap`` event programmatically
    # exercises the same code path as a real click.
    def _click_aux_port(target_node_id: str, param: str) -> None:
        page.evaluate(
            f"""
            () => {{
                const cy = window.phenoGetCy && window.phenoGetCy();
                if (!cy) return;
                const port = cy.getElementById(
                    'aux-port__{target_node_id}__{param}'
                );
                if (port && port.length > 0) {{
                    port.emit('tap');
                }}
            }}
            """
        )
        page.wait_for_timeout(500)

    # Helper: resolve the node id of the most recently-added ribbon op
    # for *class_name*. Dash assigns each StepNode a fresh 8-char hex id
    # at construction time, so screenshots can't hardcode it.
    def _last_main_node_id(class_name: str) -> str:
        return page.evaluate(
            f"""
            () => {{
                const cy = window.phenoGetCy && window.phenoGetCy();
                if (!cy) return '';
                const nodes = cy.nodes('[class_name = "{class_name}"]');
                if (!nodes || nodes.length === 0) return '';
                return nodes.last().id();
            }}
            """
        )

    # 2) Build the 4-step main pipeline. Order matters — the screenshot
    #    is taken after all four are wired left-to-right so the image-
    #    flow edges are visible.
    for cls in (
        "GaussianBlur",
        "ContrastStretching",
        "FilamentousFungiDetector",
        "MeasureSize",
    ):
        _add_op(cls)
    page.wait_for_timeout(800)
    _save(page, "aux_ports", "02_main_pipeline.png")

    # 3) Resolve the FFD consumer node id, tap its aux port, and wait
    #    for the popover palette to mount.
    ffd_node_id = _last_main_node_id("FilamentousFungiDetector")
    if not ffd_node_id:
        # Defensive: if cy isn't ready or class_name lookup misses,
        # skip the popover-driven shots rather than emit duplicates.
        print(
            "[shot]   aux_ports: could not resolve FFD node id — "
            "popover screenshots skipped"
        )
        page.close()
        return
    _click_aux_port(ffd_node_id, "inoculum_detector")
    try:
        page.wait_for_selector(
            "#cy-popover-container .cy-popover-palette",
            timeout=5_000,
        )
    except Exception:  # pragma: no cover - best-effort
        page.wait_for_timeout(800)
    _save(page, "aux_ports", "03_popover_empty.png")

    # 4) Click the ``OtsuDetector`` palette button in the popover.
    #    The popover action button ids are pattern-matching dicts of the
    #    form {"type": "popover-action", "action": "pick_class",
    #    "target_node_id": ..., "param": ..., "slot": 0,
    #    "class_name": "OtsuDetector"}. Match by the two id segments
    #    most likely to be unique on this popover.
    pick_btn_sel = (
        '#cy-popover-container '
        'button[id*="\\"type\\":\\"popover-action\\""]'
        '[id*="\\"action\\":\\"pick_class\\""]'
        '[id*="\\"class_name\\":\\"OtsuDetector\\""]'
    )
    pick_btn = page.locator(pick_btn_sel)
    if pick_btn.count() > 0:
        pick_btn.first.click()
        try:
            page.wait_for_selector(
                "#cy-popover-container .cy-popover-wired-row",
                timeout=8_000,
            )
        except Exception:  # pragma: no cover - best-effort
            page.wait_for_timeout(1000)
    else:
        # Popover container might have been wiped by an earlier callback
        # (see ``test_aux_port_e2e`` Wave 3 bug note). The screenshot
        # will document whatever state the GUI is in.
        page.wait_for_timeout(800)
    _save(page, "aux_ports", "04_popover_wired.png")

    # 5) Drill in via the popover's drill action button. The canvas
    #    re-renders to the drilled aux scope and the popover dismisses.
    drill_btn_sel = (
        '#cy-popover-container '
        'button[id*="\\"type\\":\\"popover-action\\""]'
        '[id*="\\"action\\":\\"drill\\""]'
    )
    drill_btn = page.locator(drill_btn_sel)
    if drill_btn.count() > 0:
        drill_btn.first.click()
        # Drill-in fires a scope swap; the breadcrumb is always rendered
        # but the canvas takes a tick to re-layout. Wait for the
        # popover to dismiss as the scope-swap signal.
        try:
            page.wait_for_function(
                """
                () => {
                    const el = document.getElementById('cy-popover-container');
                    if (!el) return true;
                    return getComputedStyle(el).display === 'none';
                }
                """,
                timeout=5_000,
            )
        except Exception:  # pragma: no cover - best-effort
            page.wait_for_timeout(800)
        page.wait_for_timeout(600)
    _save(page, "aux_ports", "05_drill_in.png")

    # 6) Drill back out by clicking the first breadcrumb crumb. The
    #    breadcrumb's non-leaf segments are rendered as ``dbc.Button``s
    #    with a ``{"type": "breadcrumb-link", "depth": N}`` pattern-
    #    matching id; ``.pheno-breadcrumb button`` is a deliberately
    #    schema-agnostic selector that survives future id renames.
    crumbs = page.locator(".pheno-breadcrumb button")
    if crumbs.count() > 0:
        crumbs.first.click()
        page.wait_for_timeout(800)
    _save(page, "aux_ports", "06_drill_out.png")

    # 7-9) Multi-slot list-typed aux ports. ``CompositeDetector.detectors``
    # is a ``list[ObjectDetector]`` aux — the popover renders one row per
    # slot plus a ``+ Add slot`` button so the user can wire any number
    # of detectors and per-slot disconnect / drill in independently.
    _aux_ports_list_scenarios(page, _add_op, _click_aux_port, _last_main_node_id)

    page.close()


def _aux_ports_list_scenarios(
    page,
    add_op,
    click_aux_port,
    last_main_node_id,
) -> None:
    """Capture the list-typed aux port (multi-slot) screenshots 07-09.

    Continues from a builder canvas that already has the 4-step main
    pipeline + a wired ``FilamentousFungiDetector.inoculum_detector``.
    Adds ``CompositeDetector`` as a 5th ribbon node so its
    ``detectors: list[ObjectDetector | ImagePipeline]`` aux port can be
    exercised:

    * Empty list port → popover renders the ``+ Add slot`` button plus a
      "No slots yet" placeholder (one-step bootstrap affordance).
    * Two ``+ Add slot`` + pick-class cycles wire ``OtsuDetector`` at
      slot 0 and ``WatershedDetector`` at slot 1; both wired-rows are
      visible inside one popover.
    * Per-slot ⨯ Disconnect on slot 0 reverts that slot to the palette
      while leaving slot 1 wired — demonstrates slot independence.

    The selectors mirror those used in ``_capture_aux_ports`` above and
    in the e2e suite. ``CompositeDetector`` lives in the ``Detector``
    palette accordion, which the parent function already expanded, so
    no extra accordion-open is needed here.
    """
    add_op("CompositeDetector")
    page.wait_for_timeout(800)

    composite_node_id = last_main_node_id("CompositeDetector")
    if not composite_node_id:
        print(
            "[shot]   aux_ports: could not resolve CompositeDetector node id — "
            "list-port screenshots skipped"
        )
        return

    # 7) Tap the list-typed ``detectors`` aux port. With no slots yet,
    #    the popover surfaces the ``+ Add slot`` affordance + a muted
    #    "No slots yet" placeholder so the user has somewhere to start.
    click_aux_port(composite_node_id, "detectors")
    try:
        page.wait_for_selector(
            "#cy-popover-container .cy-popover-add-slot",
            timeout=5_000,
        )
    except Exception:  # pragma: no cover - best-effort
        page.wait_for_timeout(800)
    _save(page, "aux_ports", "07_list_port_popover.png")

    # Helper: click ``+ Add slot`` once and wait for a fresh empty-slot
    # palette to mount.
    def _add_slot_and_wait_for_palette(prev_slot_count: int) -> None:
        add_slot_btn = page.locator(
            "#cy-popover-container .cy-popover-add-slot"
        )
        if add_slot_btn.count() == 0:
            return
        add_slot_btn.first.click()
        try:
            page.wait_for_function(
                f"""
                () => {{
                    const rows = document.querySelectorAll(
                        '#cy-popover-container .cy-popover-slot-row'
                    );
                    return rows.length > {prev_slot_count};
                }}
                """,
                timeout=5_000,
            )
        except Exception:  # pragma: no cover - best-effort
            page.wait_for_timeout(800)

    # Helper: pick *class_name* in the *slot_idx* slot's palette. Each
    # slot row carries its own palette while empty, so we scope the
    # button match to the row matching ``data-slot=<slot_idx>`` when
    # available, falling back to the first matching pick_class button.
    def _pick_class_for_slot(slot_idx: int, class_name: str) -> None:
        slot_pick_sel = (
            "#cy-popover-container "
            'button[id*="\\"type\\":\\"popover-action\\""]'
            '[id*="\\"action\\":\\"pick_class\\""]'
            f'[id*="\\"slot\\":{slot_idx}"]'
            f'[id*="\\"class_name\\":\\"{class_name}\\""]'
        )
        loc = page.locator(slot_pick_sel)
        if loc.count() == 0:
            return
        loc.first.click()
        # Wait for the popover to re-render with the wired-row for this
        # slot (the row's row-class flips from palette-mode to wired).
        try:
            page.wait_for_function(
                f"""
                () => {{
                    const wired = document.querySelectorAll(
                        '#cy-popover-container .cy-popover-wired-row'
                    );
                    return wired.length > {slot_idx};
                }}
                """,
                timeout=6_000,
            )
        except Exception:  # pragma: no cover - best-effort
            page.wait_for_timeout(800)

    # 8) Wire two slots: OtsuDetector at slot 0, WatershedDetector at
    #    slot 1. The popover stays open between picks; only the per-slot
    #    rows transition from palette to wired state.
    _add_slot_and_wait_for_palette(prev_slot_count=0)
    _pick_class_for_slot(0, "OtsuDetector")
    _add_slot_and_wait_for_palette(prev_slot_count=1)
    _pick_class_for_slot(1, "WatershedDetector")
    page.wait_for_timeout(400)
    _save(page, "aux_ports", "08_list_port_two_wired.png")

    # Re-tap the aux port so the popover's inspector focus jumps to
    # the FIRST wired slot (slot 0). Without this, the focus is still
    # on slot 1 (it was set when we picked WatershedDetector there),
    # and ``wire_delete`` clears the focus when it disconnects the
    # focused slot — dismissing the popover entirely. Tapping again
    # re-runs ``open_popover_from_port_click`` which sets focus to the
    # first wired slot (slot 0), so the subsequent slot-1 disconnect
    # leaves the focus unchanged and the popover stays open.
    click_aux_port(composite_node_id, "detectors")
    page.wait_for_timeout(400)

    # 9) Disconnect slot 1 via its per-slot ⨯ button. Slot 1's row
    #    reverts to the class palette while slot 0 stays wired —
    #    showcases slot-level independence inside one popover.
    #    (We disconnect the LAST slot rather than the first so that
    #    the compact wired-row for slot 0 stays visible at the top of
    #    the popover; the ~60-button palette that mounts in the
    #    disconnected slot is large enough to push lower-numbered
    #    wired-rows off-screen otherwise.)
    disconnect_sel = (
        "#cy-popover-container "
        'button[id*="\\"type\\":\\"popover-action\\""]'
        '[id*="\\"action\\":\\"disconnect\\""]'
        '[id*="\\"slot\\":1"]'
    )
    disconnect_btn = page.locator(disconnect_sel)
    if disconnect_btn.count() > 0:
        disconnect_btn.first.click()
        try:
            page.wait_for_function(
                """
                () => {
                    // Slot 1 should drop back to palette mode; slot 0
                    // stays wired. We expect exactly one wired row
                    // remaining.
                    const wired = document.querySelectorAll(
                        '#cy-popover-container .cy-popover-wired-row'
                    );
                    return wired.length === 1;
                }
                """,
                timeout=5_000,
            )
        except Exception:  # pragma: no cover - best-effort
            page.wait_for_timeout(600)
    _save(page, "aux_ports", "09_per_slot_disconnect.png")


def _capture_qc_curation_loop(context, base_url: str) -> None:
    """Capture the QC curation loop workflow.

    Spec §1232. Drives the Results Viewer's QC tab through the
    add-check / curate / re-render loop. The tab body composes one
    Plotly card per configured :class:`QualityCheck`; each card
    subscribes to ``STORE_REMOVED_KEYS`` so removing a colony from
    the measurements table triggers an automatic figure refresh
    without a manual reload.

    Three screenshots illustrate the workflow:

    1. ``01_empty_state.png`` — the Viewer mounted in empty state
       with the QC tab selected so the reader sees what "no checks
       configured" looks like (the hub-mounted viewer has no
       ``output_root`` until the sidebar binds one).
    2. ``02_add_check_modal.png`` — the ``+ Add check`` modal open
       in add mode, showing the class picker dropdown that lists
       every concrete ``QualityCheck`` subclass.
    3. ``03_card_after_add.png`` — the QC tab body after the modal
       has been dismissed, illustrating the "no cards" placeholder
       remains when the user closes the modal without picking a
       class (the empty hub-mounted viewer cannot persist a real
       recipe entry — see the tutorial page for the full add-and-
       curate flow).
    """
    print("[shot] workflow=qc_curation_loop")
    page = _new_page(context, base_url, "/results/")
    # The hub viewer mounts in empty state until the sidebar binds an
    # ``output_root``. The QC tab is still selectable; it renders the
    # "no checks configured" placeholder which is the natural empty
    # state shot for the tutorial.
    page.wait_for_timeout(800)
    _save(page, "qc_curation_loop", "01_empty_state.png")

    # Try to click the QC tab if it's mounted (the hub viewer's empty
    # state may render the tab strip even without an ``output_root``).
    # dbc.Tab anchors don't carry a stable id, so match by visible text.
    qc_tab = page.locator('a[role="tab"]:has-text("QC")').first
    if qc_tab.count() > 0:
        try:
            qc_tab.click(timeout=2000)
            page.wait_for_timeout(600)
            _save(page, "qc_curation_loop", "02_qc_tab_selected.png")
        except Exception:  # pragma: no cover - best-effort
            pass

    # Try to open the Add check modal so the reader sees the picker.
    add_btn = page.locator("#qc-add-check-btn")
    if add_btn.count() > 0:
        try:
            add_btn.click(timeout=2000)
            page.wait_for_timeout(700)
            _save(page, "qc_curation_loop", "03_add_check_modal.png")
        except Exception:  # pragma: no cover - best-effort
            pass

    page.close()


def _capture_heatmap_exploration(context, base_url: str) -> None:
    """Capture the Heatmap exploration workflow.

    Spec §1233. Drives the Results Viewer's Heatmap tab so the
    reader sees the color / image / aggregator pickers and the
    plate-shaped heatmap output. Removed cells overlay as
    ``COLOR_MUTED`` × markers so the visual distinction between
    "curated" and "low value" cells is clear.

    Two screenshots illustrate the workflow:

    1. ``01_default_view.png`` — the Heatmap tab on first open,
       showing the empty-state explanation when ``Grid_RowNum`` /
       ``Grid_ColNum`` are missing (the synthetic tutorial dataset
       does not run ``GridMeasureFeatures``).
    2. ``02_color_picker_open.png`` — the color column dropdown
       open so the reader can see the measurement / QC severity
       union the dropdown surfaces.
    """
    print("[shot] workflow=heatmap_exploration")
    page = _new_page(context, base_url, "/results/")
    page.wait_for_timeout(800)

    # Switch to the Heatmap tab.
    heatmap_tab = page.locator('a[role="tab"]:has-text("Heatmap")').first
    if heatmap_tab.count() > 0:
        try:
            heatmap_tab.click(timeout=2000)
            page.wait_for_timeout(600)
        except Exception:  # pragma: no cover - best-effort
            pass
    _save(page, "heatmap_exploration", "01_default_view.png")

    # Open the color picker dropdown so the reader sees the union of
    # measurement columns + QC severity columns.
    color_picker = page.locator("#heatmap-color-picker")
    if color_picker.count() > 0:
        try:
            color_picker.click(timeout=2000)
            page.wait_for_timeout(500)
            _save(page, "heatmap_exploration", "02_color_picker_open.png")
        except Exception:  # pragma: no cover - best-effort
            pass

    page.close()


def _capture_analysis(context, base_url: str) -> None:
    """Capture the analysis sub-app's empty-state hand-off banner.

    The hub-mounted analysis sub-app renders the empty-state placeholder
    until the user clicks a CLI output in the sidebar. The hand-off
    banner picks up the selection and exposes the "↩ Open in analysis"
    button. Loaded-state screenshots come from
    :func:`capture_standalone_analysis_screenshots` because they need
    a bound ``output_root`` and the unified hub does not preconfigure
    the sidebar selection at startup.
    """
    print("[shot] workflow=analysis (empty state via hub)")
    page = _new_page(context, base_url, "/analysis/")
    _save(page, "analysis", "01_analysis_empty.png")
    page.close()


def capture_standalone_analysis_screenshots(headed: bool = False) -> None:
    """Boot ``python -m phenotypic.gui.analysis --root <real>`` and capture.

    Mirrors :func:`capture_standalone_viewer_screenshots`: spawns the
    standalone analysis launcher against the synthetic CLI output dir,
    drives a headless Chromium through the loaded-state UX (pipeline
    header + section param forms + run console), and writes the PNGs
    into ``docs/source/_static/gui_images/analysis/``. Required so the
    WORKFLOWS row can flip from ``🔭 planned`` to ``✅ shipping`` once
    a developer regenerates the screenshots on their workstation.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover
        return

    print("[shot] workflow=analysis (loaded state via standalone)")
    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "phenotypic.gui.analysis",
        "--root",
        str(OUTPUT_DIR),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"

    try:
        # Wait for the standalone server to come up.
        deadline = time.time() + 20.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(base_url + "/", timeout=1.0) as r:
                    if 200 <= r.status < 300:
                        break
            except Exception:
                time.sleep(0.2)
        else:
            print("[shot] standalone analysis did not respond — skipping")
            return

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not headed)
            try:
                context = browser.new_context(viewport=VIEWPORT)
                page = _new_page(context, base_url, "/")
                # 02: full-page hero shot — pipeline header + post + filter
                # stacks visible at the top.
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(150)
                _save(page, "analysis", "02_pipeline_loaded.png")

                # 03: element-only screenshot of a filter section's param
                # form. Avoids re-capturing the same visible viewport — the
                # bounding box clips the screenshot to just the section card.
                target_dir = ASSETS_ROOT / "analysis"
                section = page.locator(".analysis-filter-section").first
                if section.count() > 0:
                    section.screenshot(
                        path=str(target_dir / "03_filter_section_with_form.png"),
                    )
                    print("[shot]   analysis/03_filter_section_with_form.png")

                # 04: scroll to the model section + run console so the
                # tutorial shows the "configured and ready" state with the
                # Run button enabled. We deliberately don't click Run —
                # the synthetic single-timepoint dataset cannot fit a real
                # ``LogGrowthModel``, so a screenshotted post-Run state
                # would document a fit failure rather than the success
                # path most users hit on real time-course data.
                model_section = page.locator("#analysis-model-section")
                if model_section.count() > 0:
                    model_section.scroll_into_view_if_needed()
                    page.wait_for_timeout(300)
                _save(page, "analysis", "04_model_section.png")

                page.close()
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


def capture_standalone_viewer_screenshots(headed: bool = False) -> None:
    """Boot ``python -m phenotypic.gui.results_viewer --output-root <real>``
    on a fresh port and capture the populated viewer.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover
        return

    print("[shot] workflow=view_results (loaded state via standalone viewer)")
    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "phenotypic.gui.results_viewer",
        "--output-root",
        str(OUTPUT_DIR),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_http_200(base_url + "/", timeout=30.0)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not headed)
            try:
                context = browser.new_context(viewport=VIEWPORT)
                page = context.new_page()
                page.goto(base_url + "/")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(2500)

                # Select the first image from the dropdown so the
                # OpenSeadragon canvas actually renders a colony plate.
                dropdown = page.locator(
                    ".Select-control, div[class*='css-'][class*='control']"
                ).first
                if dropdown.count() > 0:
                    try:
                        dropdown.click(timeout=2000)
                        page.wait_for_timeout(400)
                        # Pick the first option that appears.
                        first_option = page.locator(
                            ".Select-option, div[class*='option']"
                        ).first
                        if first_option.count() > 0:
                            first_option.click(timeout=2000)
                            page.wait_for_timeout(2500)
                    except Exception as exc:  # pragma: no cover - best-effort
                        print(
                            f"[shot]   image-dropdown selection skipped: {exc!r}"
                        )

                _save(page, "view_results", "02_viewer_loaded.png")

                page.mouse.wheel(0, 800)
                page.wait_for_timeout(500)
                _save(page, "view_results", "03_measurement_table.png")

                # While the standalone viewer is still up and the page is
                # populated, capture the loaded-state versions of the QC
                # and Heatmap tab tutorials. The hub-mounted captures
                # above only see the empty state because the hub viewer
                # is unbound at startup; the standalone launcher has a
                # real ``output_root`` and renders the tab strip, so QC
                # / Heatmap controls are reachable from here.
                page.mouse.wheel(0, -800)
                page.wait_for_timeout(400)

                _qc_curation_loop_loaded_shots(page)
                _heatmap_exploration_loaded_shots(page)

                page.close()
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


def _qc_curation_loop_loaded_shots(page) -> None:
    """Capture loaded-state QC tab screenshots inside the standalone viewer.

    The hub-mounted ``_capture_qc_curation_loop`` only sees the empty
    state because the hub viewer is unbound at startup. The standalone
    viewer has a real ``output_root``, so the tab strip mounts and the
    QC top-strip buttons (``+ Add check`` / ``Export QC report``) are
    reachable. Three additional screenshots cover the loaded state:

    * ``02_qc_tab_selected.png`` — empty cards container with the top
      strip visible.
    * ``03_add_check_modal.png`` — the add-check modal open in add
      mode showing the class picker.
    """
    qc_tab = page.locator('a[role="tab"]:has-text("QC")').first
    if qc_tab.count() == 0:
        print("[shot]   qc_curation_loop: QC tab not found — loaded captures skipped")
        return
    try:
        qc_tab.click(timeout=3000)
        page.wait_for_timeout(800)
    except Exception:  # pragma: no cover - best-effort
        pass
    _save(page, "qc_curation_loop", "02_qc_tab_selected.png")

    add_btn = page.locator("#qc-add-check-btn")
    if add_btn.count() > 0:
        try:
            add_btn.click(timeout=2000)
            # Wait for the modal to mount.
            page.wait_for_selector("#qc-add-check-modal", timeout=4000)
            page.wait_for_timeout(500)
        except Exception:  # pragma: no cover - best-effort
            pass
    _save(page, "qc_curation_loop", "03_add_check_modal.png")

    # Close the modal so the next workflow's screenshots aren't polluted.
    cancel = page.locator("#qc-add-check-cancel")
    if cancel.count() > 0:
        try:
            cancel.click(timeout=2000)
            page.wait_for_timeout(400)
        except Exception:  # pragma: no cover - best-effort
            pass


def _heatmap_exploration_loaded_shots(page) -> None:
    """Capture loaded-state Heatmap tab screenshots inside the standalone viewer.

    Mirrors :func:`_qc_curation_loop_loaded_shots`: switches to the
    Heatmap tab in the standalone viewer (where the tab strip is
    mounted) and captures the controls + figure. The synthetic
    tutorial dataset does not run ``GridMeasureFeatures``, so the
    populated state still renders the empty-state explanation card
    rather than a real heatmap — but the captures reach more of the
    real DOM than the empty-hub fallback.
    """
    heatmap_tab = page.locator('a[role="tab"]:has-text("Heatmap")').first
    if heatmap_tab.count() == 0:
        print(
            "[shot]   heatmap_exploration: Heatmap tab not found — loaded captures skipped"
        )
        return
    try:
        heatmap_tab.click(timeout=3000)
        page.wait_for_timeout(800)
    except Exception:  # pragma: no cover - best-effort
        pass
    _save(page, "heatmap_exploration", "02_heatmap_tab_loaded.png")

    # Try opening the color picker so the dropdown options are visible.
    color_picker = page.locator("#heatmap-color-picker")
    if color_picker.count() > 0:
        try:
            color_picker.click(timeout=2000)
            page.wait_for_timeout(500)
            _save(page, "heatmap_exploration", "03_color_picker_open.png")
        except Exception:  # pragma: no cover - best-effort
            pass


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n")[0],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the synthetic dataset even if it already exists.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Chromium with a visible window (debugging).",
    )
    parser.add_argument(
        "--skip-cli",
        action="store_true",
        help="Skip the CLI run (useful when iterating on screenshots only).",
    )
    args = parser.parse_args(argv)

    build_tutorial_dataset(force=args.force)
    if not args.skip_cli:
        try:
            run_cli_once()
        except subprocess.CalledProcessError as exc:
            print(
                f"[cli] FAILED with exit {exc.returncode}; continuing without "
                "CLI output (Recent Runs / viewer screenshots will reflect "
                "an empty sandbox).",
                file=sys.stderr,
            )

    proc, base_url = boot_gui(DATASET_DIR)
    try:
        capture_workflow_screenshots(base_url, headed=args.headed)
    finally:
        shutdown_gui(proc)

    capture_standalone_viewer_screenshots(headed=args.headed)
    capture_standalone_analysis_screenshots(headed=args.headed)

    print("[done] all screenshots written to docs/source/_static/gui_images/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
