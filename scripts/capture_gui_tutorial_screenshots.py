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
import tempfile
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


def _gui_log_sink(port: int) -> Path:
    """Return the temp log path for a GUI subprocess booted on *port*.

    The GUI's stdout+stderr are redirected to this file rather than an
    in-process ``subprocess.PIPE``.  An undrained pipe deadlocks the GUI
    once Werkzeug's per-request logging fills the OS pipe buffer
    (~64 KB): the process blocks on its next ``stderr`` write and every
    subsequent ``page.goto`` times out (this is exactly what stalled the
    last capture workflow before this fix).  A file sink has no such
    bound and doubles as a triage artifact for failed runs.
    """
    return Path(tempfile.gettempdir()) / f"phenotypic-gui-capture-{port}.log"


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
    log_path = _gui_log_sink(port)
    print(f"[gui] booting: {' '.join(cmd)}")
    print(f"[gui]   logs -> {log_path}")
    # ``subprocess.Popen`` dups the file descriptor for the child, so the
    # parent handle can be closed immediately — the GUI keeps writing.
    with log_path.open("w", encoding="utf-8") as gui_log:
        proc = subprocess.Popen(
            cmd,
            stdout=gui_log,
            stderr=subprocess.STDOUT,
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
            _capture_aux_wire_in_dag(context, base_url)
            _capture_wire_pipeline_as_aux(context, base_url)
            _capture_fix_validation_issues(context, base_url)
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


def _emit_empty_state_shot(
    context, base_url: str, path: str, workflow: str, name: str, *, log: str
) -> None:
    """Open ``path``, save a single screenshot, close.  The shared shape of
    the empty-state hub captures (setup landing, results viewer, analysis
    hand-off banner) which only need one shot of the page as-mounted."""
    print(log)
    page = _new_page(context, base_url, path)
    _save(page, workflow, name)
    page.close()


# ---------------------------------------------------------------------------
# DAG builder interaction helpers
# ---------------------------------------------------------------------------
#
# The post-redesign builder canvas is a dash-cytoscape graph driven by three
# clientside ``dcc.Store`` components (spec §5.5 "Clientside event contract"):
#
#   * ``store-palette-drop``  — ``block_create`` payloads (palette → canvas).
#   * ``store-edge-event``    — ``edge_create`` / ``edge_delete`` (wire draw).
#   * ``store-builder-state`` — ``block_select`` / ``block_delete_request`` …
#
# ``palette_dnd.js`` / ``wire_drawing.js`` write to these stores via
# ``window.dash_clientside.set_props`` in response to native drag-and-drop
# and port-mousedown gestures.  Playwright cannot faithfully replay HTML5
# drag-and-drop against a ``<canvas>``-backed graph, so the capture helpers
# below dispatch the *same payloads* the JS would emit — exercising the real
# server-side dispatcher (``_dispatch_state_update``) and clientside layout
# path, just without synthesising raw pointer events.


def _new_builder_page(context, base_url: str):
    """Open ``/builder/`` and block until the palette has mounted.

    The shared opener for the DAG-builder capture helpers: a fresh page,
    a wait on ``#palette`` (15s — the operation registry scan is slow on
    a cold boot), then a short settle for the canvas's first dagre pass.
    """
    page = _new_page(context, base_url, "/builder/")
    page.wait_for_selector("#palette", timeout=15_000)
    page.wait_for_timeout(500)
    return page


def _relayout_canvas(page) -> None:
    """Re-run the leaf-first dagre layout and wait for it to settle.

    ``viewport_ops.js`` auto-relayouts on every mutation, but it debounces
    behind dash-cytoscape's own ``breadthfirst`` pass; calling
    ``phenotypicRelayout()`` explicitly before a screenshot guarantees the
    canvas shows the final dagre layout + port placement deterministically.
    """
    page.evaluate(
        "() => window.phenotypicRelayout && window.phenotypicRelayout()"
    )
    page.wait_for_timeout(900)


def _dispatch_block_create(
    page, class_name: str, *, container_block_id: str | None = None
) -> None:
    """Mint a DAG block via ``store-palette-drop`` (``block_create``).

    Mirrors the payload ``palette_dnd.js`` emits on a palette drop.  Unlike
    the ``palette-add`` keyboard-fallback click (which auto-wires the new
    block onto the current scope tail), a raw ``block_create`` lands the
    block free-floating — exactly what the aux-producer / stranded-block
    tutorials need.  ``container_block_id`` drops the block into that
    container's nested scope.
    """
    page.evaluate(
        """([cn, cbid]) => {
            window.dash_clientside.set_props('store-palette-drop', {
                data: {
                    kind: 'block_create',
                    class_name: cn,
                    x: 0, y: 0,
                    container_block_id: cbid,
                    ts: Date.now(),
                },
            });
        }""",
        [class_name, container_block_id],
    )
    page.wait_for_timeout(900)


def _dispatch_edge_create(
    page,
    source_block_id: str,
    target_block_id: str,
    target_port: str,
    edge_kind: str,
) -> None:
    """Draw a wire via ``store-edge-event`` (``edge_create``).

    ``edge_kind`` is ``"image"`` (blue image-flow wire into the ``"in"``
    port) or ``"aux"`` (purple wire into a named aux port).  Mirrors the
    payload ``wire_drawing.js`` emits on a compatible port drop.
    """
    page.evaluate(
        """([s, t, port, ek]) => {
            window.dash_clientside.set_props('store-edge-event', {
                data: {
                    kind: 'edge_create',
                    source_block_id: s,
                    target_block_id: t,
                    target_port: port,
                    edge_kind: ek,
                    ts: Date.now(),
                },
            });
        }""",
        [source_block_id, target_block_id, target_port, edge_kind],
    )
    page.wait_for_timeout(900)


def _block_id(page, class_name: str, which: str = "last") -> str:
    """Resolve a block's cytoscape id by ``class_name`` (``"first"`` / ``"last"``).

    DAG blocks get a fresh 8-char hex id at creation time, so capture
    helpers resolve ids from the live cy instance rather than hardcoding.
    """
    return page.evaluate(
        """([cn, which]) => {
            const cy = window.phenoGetCy && window.phenoGetCy();
            if (!cy) return '';
            const ns = cy.nodes('[class_name = "' + cn + '"]');
            if (!ns || ns.length === 0) return '';
            return (which === 'first' ? ns.first() : ns.last()).id();
        }""",
        [class_name, which],
    )


def _tap_block(page, block_id: str) -> None:
    """Select a block by emitting a ``tap`` on its cytoscape node.

    dash-cytoscape mirrors cytoscape ``tap`` events onto its ``tapNodeData``
    prop, which the builder's canvas-tap callback routes to a
    ``block_select`` dispatch — the same path a real click takes.
    """
    page.evaluate(
        """(bid) => {
            const cy = window.phenoGetCy && window.phenoGetCy();
            if (!cy) return;
            const n = cy.getElementById(bid);
            if (n && n.length) { cy.elements().unselect(); n.emit('tap'); }
        }""",
        block_id,
    )
    page.wait_for_timeout(700)


# --- setup --------------------------------------------------------------

def _capture_setup(context, base_url: str) -> None:
    _emit_empty_state_shot(
        context, base_url, "/", "setup", "01_landing_page.png",
        log="[shot] workflow=setup",
    )


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
    page.wait_for_selector("#palette", timeout=15_000)
    # Settle the dagre layout so the auto-seeded Input Image block sits
    # cleanly centred rather than at the breadthfirst fallback origin.
    _relayout_canvas(page)
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
    _emit_empty_state_shot(
        context, base_url, "/results/", "view_results", "01_viewer_empty.png",
        log="[shot] workflow=view_results (empty state via hub)",
    )


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
    # Re-run the leaf-first dagre layout so the ribbon shows the settled
    # left-to-right chain rather than dash-cytoscape's breadthfirst pass.
    _relayout_canvas(page)
    _save(page, "pick_points", "02_pipeline_with_selector.png")

    # 3) Inspector param form for ManualSelector. Tap the block so the
    # canvas-tap callback dispatches ``block_select`` and the inspector
    # re-renders against it (a clientside-only ``cy ... .select()`` would
    # not update the server-side ``selected_block_id``).
    selector_id = _block_id(page, "ManualSelector")
    if selector_id:
        _tap_block(page, selector_id)
    page.wait_for_timeout(600)
    _save(page, "pick_points", "03_param_form.png")

    # Load the synthetic plate so Run preview has an image to apply the
    # pipeline against. Without an image, OtsuDetector's intermediate is
    # never cached and the picker modal opens against an empty OSD
    # canvas (timeout on ``[data-testid="point-picker-osd-canvas"]``).
    synth_btn = page.locator("#btn-use-synthetic")
    if synth_btn.count() > 0:
        synth_btn.click()
        page.wait_for_timeout(800)

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
    """Drive the aux-port wiring + inspector workflow and capture 4 PNGs.

    Spec §4.2 / §4.5.  Phase 7 retired the canvas-anchored popover; in the
    post-redesign builder an operation-typed parameter (e.g.
    ``FilamentousFungiDetector.inoculum_detector``) is a bottom-edge aux
    port, the aux producer is a first-class canvas block, and the wired
    aux surfaces in the inspector's **Aux ports section** (where the
    ``Disconnect`` action now lives — it moved off the popover).

    Steps exercised against the real dispatcher:

    1. ``01_initial.png`` — empty builder canvas + palette.
    2. ``02_main_pipeline.png`` — palette clicks build the ribbon
       ``Input Image → GaussianBlur → FilamentousFungiDetector``; FFD's
       required ``inoculum_detector`` aux port renders as an empty
       red-ringed square (Rule 3) on its bottom edge.
    3. ``03_aux_wired.png`` — an ``OtsuDetector`` block is minted
       free-floating and an ``edge_create`` of kind ``aux`` wires it into
       ``FilamentousFungiDetector.inoculum_detector``; the aux port flips
       to filled purple and the producer block's border turns purple.
    4. ``04_inspector_aux.png`` — the consumer block is selected; the
       inspector's Aux ports section shows the wired ``OtsuDetector`` row
       with its ``Disconnect`` action.

    The capture dispatches the same ``store-palette-drop`` /
    ``store-edge-event`` payloads ``palette_dnd.js`` / ``wire_drawing.js``
    emit — see the "DAG builder interaction helpers" section above.
    """
    print("[shot] workflow=aux_ports")
    page = _new_builder_page(context, base_url)

    # 1) Empty canvas — same starting point as the Build Pipeline tutorial.
    _relayout_canvas(page)
    _save(page, "aux_ports", "01_initial.png")

    _expand_palette_accordions(page)

    # 2) Main ribbon with the aux-consuming op (FFD) on the tail.  FFD's
    #    ``inoculum_detector`` is a required aux port — it renders empty
    #    (red ring) until wired.
    for cls in ("GaussianBlur", "FilamentousFungiDetector"):
        _add_palette_op(page, cls)
    _relayout_canvas(page)
    _save(page, "aux_ports", "02_main_pipeline.png")

    # 3) Free-floating aux producer + the purple aux wire into the
    #    consumer's bottom-edge port.
    _dispatch_block_create(page, "OtsuDetector")
    otsu_id = _block_id(page, "OtsuDetector")
    ffd_id = _block_id(page, "FilamentousFungiDetector")
    if otsu_id and ffd_id:
        _dispatch_edge_create(page, otsu_id, ffd_id, "inoculum_detector", "aux")
        _relayout_canvas(page)
    else:  # pragma: no cover - best-effort
        print("[shot]   aux_ports: could not resolve block ids")
    _save(page, "aux_ports", "03_aux_wired.png")

    # 4) Select the consumer so the inspector's Aux ports section renders
    #    the wired-row + Disconnect action (spec §4.5 — the popover's
    #    wired-row moved here).
    if ffd_id:
        _tap_block(page, ffd_id)
        page.wait_for_timeout(600)
    _save(page, "aux_ports", "04_inspector_aux.png")
    page.close()


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


def _expand_palette_accordions(page) -> None:
    """Open every collapsed palette accordion section.

    ``dbc.Accordion(always_open=True)`` auto-expands only the first item;
    the rest start ``collapsed`` (``display: none``), so palette-add
    clicks against ops in those categories would miss.  Clicking each
    collapsed header opens it.
    """
    for header_text in ("Corrector", "Detector", "Enhancer", "Refiner", "Measure"):
        header = page.locator(
            f'button.accordion-button:has-text("{header_text}")'
        ).first
        if header.count() > 0:
            try:
                cls = header.get_attribute("class") or ""
                if "collapsed" in cls:
                    header.click()
                    page.wait_for_timeout(200)
            except Exception:  # pragma: no cover - best-effort
                pass
    page.wait_for_timeout(250)


def _add_palette_op(page, class_name: str) -> None:
    """Add an op via its palette button (keyboard-fallback click path).

    The ``palette-add`` click dispatches ``block_create`` *and* auto-wires
    the new block onto the current scope's tail (handoff fix) — so a
    sequence of ``_add_palette_op`` calls builds a connected main ribbon
    ``Input Image → … → tail``.  For a *free-floating* block (an aux
    producer, a stranded orphan) use :func:`_dispatch_block_create`
    instead.
    """
    sel = (
        f'button[id*="\\"type\\":\\"palette-add\\""]'
        f'[id*="\\"class_name\\":\\"{class_name}\\""]'
    )
    loc = page.locator(sel)
    if loc.count() > 0:
        loc.first.click()
        page.wait_for_timeout(700)
    else:  # pragma: no cover - best-effort
        print(f"[shot]   palette button for {class_name} not found")


def _capture_aux_wire_in_dag(context, base_url: str) -> None:
    """Drive the post-redesign aux-wiring DAG workflow and capture 3 PNGs.

    Mirrors the post-popover aux flow in
    ``docs/source/tutorials/gui/12_aux_wire_in_dag.md`` (spec §4.2-§4.3):
    every operation — including the aux producer — is a first-class block
    on the canvas, and the aux assignment is a purple wire drawn from the
    producer's output port to the consumer's bottom-edge aux port.  There
    is no popover class palette in v2.

    Steps exercised against the real dispatcher:

    1. ``01_main_with_consumer.png`` — palette clicks build the main
       ribbon ``Input Image → GaussianBlur → ContrastStretching →
       FilamentousFungiDetector``.  FFD's required ``inoculum_detector``
       aux port renders as an empty red-ringed square and the toolbar
       issue badge lights up (Rule 3).
    2. ``02_detector_dropped.png`` — an ``OtsuDetector`` block is minted
       free-floating via a raw ``block_create`` (no auto-wire), ready to
       feed the aux port.
    3. ``03_aux_wired.png`` — an ``edge_create`` of kind ``aux`` draws
       the purple wire ``OtsuDetector → FilamentousFungiDetector
       .inoculum_detector``; the aux port flips to filled purple, the
       producer's border turns purple (aux-consumed), and the issue
       badge clears.
    """
    print("[shot] workflow=aux-wire-in-dag")
    page = _new_builder_page(context, base_url)
    _expand_palette_accordions(page)

    # 1) Main ribbon + the consumer whose aux port we will feed.
    for cls in ("GaussianBlur", "ContrastStretching", "FilamentousFungiDetector"):
        _add_palette_op(page, cls)
    _relayout_canvas(page)
    _save(page, "aux-wire-in-dag", "01_main_with_consumer.png")

    # 2) Free-floating aux producer — a raw block_create, NOT a palette
    #    click, so it is not auto-wired into the main ribbon.
    _dispatch_block_create(page, "OtsuDetector")
    _relayout_canvas(page)
    _save(page, "aux-wire-in-dag", "02_detector_dropped.png")

    # 3) Draw the purple aux wire: OtsuDetector.output →
    #    FilamentousFungiDetector.inoculum_detector.
    otsu_id = _block_id(page, "OtsuDetector")
    ffd_id = _block_id(page, "FilamentousFungiDetector")
    if otsu_id and ffd_id:
        _dispatch_edge_create(page, otsu_id, ffd_id, "inoculum_detector", "aux")
        _relayout_canvas(page)
    else:  # pragma: no cover - best-effort
        print("[shot]   aux-wire-in-dag: could not resolve block ids")
    _save(page, "aux-wire-in-dag", "03_aux_wired.png")
    page.close()


def _capture_wire_pipeline_as_aux(context, base_url: str) -> None:
    """Drive the Pipeline-container-as-aux workflow and capture 3 PNGs.

    Mirrors ``docs/source/tutorials/gui/13_wire_pipeline_as_aux.md``
    (spec §4.4): a ``Pipeline`` container holds a multi-step chain in its
    nested scope, and the whole container is wired as a single aux
    producer into a consumer outside it.

    Steps exercised against the real dispatcher:

    1. ``01_empty_container.png`` — a ``block_create`` for the
       ``ImagePipeline`` sentinel mints an empty container; its nested
       scope is auto-seeded with a consumer-fed ``Input Image`` dot and
       it renders the ``+ drop ops here`` placeholder.
    2. ``02_chain_in_container.png`` — two ops are dropped into the
       container's nested scope (``container_block_id`` set) and wired
       ``Input Image → GaussianBlur → OtsuDetector`` inside it.
    3. ``03_pipeline_wired_as_aux.png`` — a free-floating
       ``FilamentousFungiDetector`` consumer is added at the root scope
       and an ``edge_create`` of kind ``aux`` wires the *container* into
       its ``inoculum_detector`` port.
    """
    print("[shot] workflow=wire-pipeline-as-aux")
    page = _new_builder_page(context, base_url)

    # 1) Empty Pipeline container.  ``ImagePipeline`` is the container
    #    sentinel class (builder/_state.PIPELINE_CLASS_NAME).
    _dispatch_block_create(page, "ImagePipeline")
    _relayout_canvas(page)
    _save(page, "wire-pipeline-as-aux", "01_empty_container.png")

    container_id = _block_id(page, "ImagePipeline")
    # The container's nested scope auto-seeds its own Input Image; it is
    # the most-recently-created InputImage block (the root one predates it).
    nested_input = _block_id(page, "InputImage", which="last")

    # 2) Drop a 2-step chain into the container's nested scope and wire
    #    it left-to-right inside the container.
    if container_id:
        _dispatch_block_create(page, "GaussianBlur", container_block_id=container_id)
        _dispatch_block_create(page, "OtsuDetector", container_block_id=container_id)
        gb_id = _block_id(page, "GaussianBlur")
        od_id = _block_id(page, "OtsuDetector")
        if nested_input and gb_id:
            _dispatch_edge_create(page, nested_input, gb_id, "in", "image")
        if gb_id and od_id:
            _dispatch_edge_create(page, gb_id, od_id, "in", "image")
        _relayout_canvas(page)
    else:  # pragma: no cover - best-effort
        print("[shot]   wire-pipeline-as-aux: container id unresolved")
    _save(page, "wire-pipeline-as-aux", "02_chain_in_container.png")

    # 3) Add the consumer at root scope and wire the whole container into
    #    its aux port.
    _dispatch_block_create(page, "FilamentousFungiDetector")
    ffd_id = _block_id(page, "FilamentousFungiDetector")
    if container_id and ffd_id:
        _dispatch_edge_create(
            page, container_id, ffd_id, "inoculum_detector", "aux"
        )
        _relayout_canvas(page)
    _save(page, "wire-pipeline-as-aux", "03_pipeline_wired_as_aux.png")
    page.close()


def _capture_fix_validation_issues(context, base_url: str) -> None:
    """Drive the validation-issue triage workflow and capture 3 PNGs.

    Mirrors ``docs/source/tutorials/gui/14_fix_validation_issues.md``
    (spec §4.6): a stranded block trips a blocking validation rule, the
    toolbar issue badge surfaces the count, and deleting the orphan
    clears it and re-enables ``Run preview``.

    Steps exercised against the real dispatcher:

    1. ``01_issue_introduced.png`` — palette clicks build a clean ribbon
       ``Input Image → GaussianBlur → OtsuDetector``; then a
       ``SmallObjectRemover`` block is minted free-floating via a raw
       ``block_create``.  With no incoming image wire it is unreachable
       from ``Input Image`` (Rule 2) — it renders with a dashed red
       border + ``!`` badge and the toolbar issue badge shows the count.
    2. ``02_issue_focused.png`` — the toolbar issue badge is clicked,
       surfacing the issue-row tooltip listing the offender.
    3. ``03_issue_resolved.png`` — the stranded ``SmallObjectRemover``
       block is selected and deleted via the toolbar ``Delete selected``
       button; the issue badge returns to ``0 issues`` and the ribbon is
       clean.
    """
    print("[shot] workflow=fix-validation-issues")
    page = _new_builder_page(context, base_url)
    _expand_palette_accordions(page)

    # 1) Clean ribbon, then a stranded orphan block (raw block_create,
    #    so it is NOT auto-wired into the ribbon).
    for cls in ("GaussianBlur", "OtsuDetector"):
        _add_palette_op(page, cls)
    _dispatch_block_create(page, "SmallObjectRemover")
    _relayout_canvas(page)
    _save(page, "fix-validation-issues", "01_issue_introduced.png")

    # 2) Click the toolbar issue badge to surface the issue list.
    issue_badge = page.locator("#issue-badge")
    if issue_badge.count() > 0:
        try:
            issue_badge.first.click()
            page.wait_for_timeout(700)
        except Exception:  # pragma: no cover - best-effort
            pass
    _save(page, "fix-validation-issues", "02_issue_focused.png")

    # 3) Select the orphan and delete it via the toolbar button; the
    #    validator re-runs and the badge clears.
    orphan_id = _block_id(page, "SmallObjectRemover")
    if orphan_id:
        _tap_block(page, orphan_id)
        delete_btn = page.locator("#btn-delete-node")
        if delete_btn.count() > 0:
            try:
                delete_btn.first.click()
                page.wait_for_timeout(700)
            except Exception:  # pragma: no cover - best-effort
                pass
    _relayout_canvas(page)
    _save(page, "fix-validation-issues", "03_issue_resolved.png")
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
    _emit_empty_state_shot(
        context, base_url, "/analysis/", "analysis", "01_analysis_empty.png",
        log="[shot] workflow=analysis (empty state via hub)",
    )


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
    # Redirect to a file sink, not an undrained PIPE — see _gui_log_sink.
    log_path = _gui_log_sink(port)
    print(f"[shot]   standalone viewer logs -> {log_path}")
    with log_path.open("w", encoding="utf-8") as gui_log:
        proc = subprocess.Popen(
            cmd,
            stdout=gui_log,
            stderr=subprocess.STDOUT,
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
