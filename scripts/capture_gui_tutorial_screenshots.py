"""Regenerate the GUI walkthrough tutorial screenshots.

Run from the repo root::

    uv run python scripts/capture_gui_tutorial_screenshots.py

The script:

1. Builds a synthetic 3-plate yeast dataset under
   ``docs/source/_static/gui_images/_dataset/``.
2. Runs the CLI once against that dataset to produce real CLI output
   (``deliverables/`` + ``results/``).
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
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_ROOT = REPO_ROOT / "docs" / "source" / "_static" / "gui_images"
DATASET_DIR = ASSETS_ROOT / "_dataset"
PLATES_DIR = DATASET_DIR / "plates"
METADATA_CSV = DATASET_DIR / "metadata.csv"
PIPELINE_JSON = DATASET_DIR / "pipeline.json.pht-pipe"
OUTPUT_DIR = DATASET_DIR / "results"

VIEWPORT = {"width": 1280, "height": 900}

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

PIPELINE_DOC = {
    "version"  : "0.1.0",
    "name"     : "gui_tutorial",
    "desc"     : "Synthetic yeast tutorial pipeline",
    "reset"    : False,
    "pipe_cfgs": {
        "BlurGauss": {
            "class" : "BlurGauss",
            "params": {"sigma": 2},
        },
        "OtsuDetector": {
            "class" : "OtsuDetector",
            "params": {"ignore_zeros": True},
        },
    },
    "meas"     : {
        "MeasureShape": {"class": "MeasureShape", "params": {}},
        "MeasureSize" : {"class": "MeasureSize", "params": {}},
    },
    "post"     : {},
    # The analysis sub-app's tutorial walkthrough renders better when the
    # synthetic CLI run produces a real ``analysis.parquet`` to demo the
    # populated state. ``EdgeCorrector`` correction needs at least one
    # interior + one edge colony per group to compute a threshold, and
    # ``LogGrowthModel`` fits expect ``Metadata_Time``-keyed measurements
    # — neither holds for the single-timepoint synthetic dataset, so the
    # filter/model are configured here primarily as recipe metadata; the
    # CLI's ``_emit_analysis_outputs`` swallows the resulting fit failure
    # at WARNING and the master output is unaffected.
    "filters"  : {
        "TukeyOutlierRemover": {
            "class" : "TukeyOutlierRemover",
            "params": {"on": "Shape_Area", "groupby": ["Metadata_StrainID"], "k": 3.0},
        },
    },
    "model"    : {
        "class" : "LogGrowthModel",
        "params": {
            "on"        : "Shape_Area",
            "groupby"   : ["Metadata_StrainID"],
            "time_label": "Metadata_RunDate",
            "n_jobs"    : 1,
        },
    },
    "nrows"    : 8,
    "ncols"    : 12,
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
            pipeline.json.pht-pipe   # BlurGauss + OtsuDetector + MeasureShape + MeasureSize
    """
    # Current captures should advertise only the canonical typed config. These
    # are capture-owned legacy residues from pre-suffix fixture generations.
    for legacy_pipeline in (DATASET_DIR / "pipeline.json", OUTPUT_DIR / "pipeline.json"):
        if legacy_pipeline.is_file():
            legacy_pipeline.unlink()

    if DATASET_DIR.exists() and not force:
        # Keep reusable fixtures on the current public suffix/schema even when
        # the large synthetic images and prior CLI output are retained.
        METADATA_CSV.write_text("\n".join(METADATA_ROWS) + "\n", encoding="utf-8")
        PIPELINE_JSON.write_text(json.dumps(PIPELINE_DOC, indent=2), encoding="utf-8")
        print(f"[dataset] reusing existing {DATASET_DIR.relative_to(REPO_ROOT)}")
        return

    from phenotypic.data import make_synthetic_plate
    import imageio.v3 as iio

    print(
        f"[dataset] generating fresh dataset under {DATASET_DIR.relative_to(REPO_ROOT)}")
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
    print(f"[dataset]   wrote {PIPELINE_JSON.name}")


def run_cli_once() -> None:
    """Invoke ``python -m phenotypic`` to produce real CLI output.

    The output (deliverables + per-image overlays) is what the "Viewing
    Results" walkthrough screenshots capture. Skip if the output already
    exists — re-running this is expensive (~minutes on a real pipeline;
    ~seconds on the synthetic dataset but still avoidable).
    """
    if (OUTPUT_DIR / "deliverables" / "master_measurements.parquet").exists():
        print(
            f"[cli] reusing existing CLI output at {OUTPUT_DIR.relative_to(REPO_ROOT)}")
        return

    print("[cli] running pipeline against synthetic dataset")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "phenotypic",
        "--mode",
        "full",
        "--pipeline",
        str(PIPELINE_JSON),
        "--input",
        str(PLATES_DIR),
        "-o",
        str(OUTPUT_DIR),
        "--njobs",
        "1",
    ]
    print(f"[cli]   {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    print("[cli]   done")


def _repair_cached_legacy_master(
    full: Any, master_path: Path
) -> tuple[Any, str]:
    """Canonicalize a cached pre-flat-namespace tutorial master in place."""
    from phenotypic.schema import IMAGE

    canonical_image_name = str(IMAGE.IMAGE_NAME)
    legacy_image_name = "MetadataImage_ImageName"
    if (
        canonical_image_name not in full.columns
        and legacy_image_name in full.columns
    ):
        # This changes only the reusable capture fixture's serialized schema.
        full = full.rename({legacy_image_name: canonical_image_name})
        full.write_parquet(master_path)
    return full, canonical_image_name


def _seed_error_triage_labels() -> None:
    """Label the smallest-``Size_Area`` synthetic objects so the Error tab renders.

    The Error-analysis tab only ranks measurements once a category carries
    ``>= ErrorCutoffFinder().min_error_n`` labels (8); the synthetic run carries
    none, so seed 12 ``background_noise`` labels (the smallest colonies —
    plausible debris/noise) for a populated, meaningful screenshot. The seeded
    labels give the cutoff finder real separation → a ranked table + a
    distribution plot with a cutoff line.

    **The labeled rows must stay PRESENT in the frame the viewer reloads.** The
    results viewer (and the Error tab) reads good/error frames from
    ``OutputRoot.master_df``, which prefers the post-applied
    ``deliverables/measurements.parquet`` mirror. But ``CurationLabels`` treats a
    label as a *removal*: ``mark_many`` rewrites the mirror with the labeled rows
    anti-joined OUT. A freshly-booted standalone viewer would then reload that
    shrunken mirror, fail to find the 12 labeled keys, drop every label in
    re-keying, and render the empty state — exactly what a naive seed produces.

    In a *live* GUI session this never bites because ``master_df`` is the full
    frame held in memory from before any curation; only the on-disk mirror
    shrinks. To reproduce that for a freshly-booted capture viewer, this helper:

    1. restores a FULL mirror from the clean ``master_measurements.parquet`` (so
       the chosen objects + their fingerprints are present and stable),
    2. clears any stale labels parquet from a prior run and marks exactly the 12
       smallest-``Size_Area`` objects as ``background_noise`` (this writes the
       durable labels store + ``errors/background_noise.parquet`` and curates the
       mirror),
    3. restores the FULL mirror again so the reloaded viewer's ``master_df``
       still carries the 12 labeled (error-class) rows alongside the good rows.

    Must run AFTER ``run_cli_once`` (it needs the master parquet) and BEFORE the
    standalone viewer boots.
    """
    import polars as pl

    from phenotypic.gui.results_viewer._curation_labels import CurationLabels
    from phenotypic.gui.results_viewer._qc_tab.review._review_state import ReviewState
    from phenotypic import ImagePipeline
    from phenotypic.analysis.qc import MaxModifiedZScore
    from phenotypic.sdk_ import (
        BundleLayout,
        curation_labels_parquet_path,
        master_measurements_parquet_path,
        measurements_csv_path,
        measurements_parquet_path,
    )
    from phenotypic.sdk_._qc_recipe import QcRecipeEntry
    from phenotypic.sdk_._qc_recipe._runner import run_qc

    master_path = master_measurements_parquet_path(OUTPUT_DIR)
    if not master_path.is_file():
        print("[seed] no master parquet — Error-tab label seeding skipped")
        return

    full, canonical_image_name = _repair_cached_legacy_master(
        pl.read_parquet(master_path), master_path
    )
    mirror_path = measurements_parquet_path(OUTPUT_DIR)
    mirror_path.parent.mkdir(parents=True, exist_ok=True)

    # (1) Restore the full mirror so the labeled rows are present + fingerprintable.
    full.write_parquet(mirror_path)

    # (2) Start from a clean label set (a prior run may have left stale keys),
    #     then mark exactly the 12 smallest-Size_Area objects.
    labels_path = curation_labels_parquet_path(OUTPUT_DIR)
    if labels_path.exists():
        labels_path.unlink()
    smallest = (
        full.sort("Size_Area")
        .head(12)
        .select([canonical_image_name, "Object_Label"])
    )
    keys = [(str(f), int(label)) for f, label in smallest.iter_rows()]
    layout = BundleLayout.detect(OUTPUT_DIR)
    store = CurationLabels.load(layout, full)
    store.mark_many(keys, "background_noise")

    # Seed a real curation-supporting QC module and mark every image group
    # reviewed. The Verified-only Error preview then has a non-empty, explicit
    # good baseline rather than falling back to its "review more groups" state.
    instance_id = "qc-MaxModifiedZScore-gui-capture"
    qc_pipeline = ImagePipeline()
    qc_pipeline.set_qc(
        [
            QcRecipeEntry(
                cls=MaxModifiedZScore,
                params={
                    "on": "Size_Area",
                    "groupby": [canonical_image_name],
                },
                instance_id=instance_id,
                enabled=True,
            )
        ]
    )
    run_qc(full.to_pandas(), qc_pipeline, OUTPUT_DIR)
    review = ReviewState.load(layout)
    for image_name in full.get_column(canonical_image_name).unique().to_list():
        if not review.mark_reviewed(instance_id, (str(image_name),)):
            raise RuntimeError(
                f"could not seed reviewed QC group for {image_name!r}"
            )

    # (3) Restore the full mirror again — mark_many curated the labeled rows OUT,
    #     but the Error tab needs them present in OutputRoot.master_df to rank the
    #     error class. The durable labels parquet (written in step 2) is what the
    #     reloaded viewer re-keys onto this full frame.
    full.write_parquet(mirror_path)
    full.write_csv(measurements_csv_path(OUTPUT_DIR))

    # The fixture mutates marker-authorized core files outside the live GUI
    # mutation path. Refresh both marker-last publications so the Results
    # viewer never observes a deliberately mixed tutorial snapshot.
    from phenotypic._cli._cli_completion import (
        publish_aggregate_snapshot,
        publish_run_completion_evidence,
    )

    publish_aggregate_snapshot(OUTPUT_DIR)
    publish_run_completion_evidence(
        OUTPUT_DIR,
        execution_epoch="gui-tutorial-capture",
    )
    print(
        f"[seed] labeled {len(keys)} smallest-Size_Area objects as "
        f"'background_noise' and reviewed all QC image groups"
    )


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


def _stop_process(proc: subprocess.Popen[Any]) -> None:
    """Terminate *proc*, escalating to ``kill`` after five seconds."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


def _read_log_tail(log_path: Path, *, max_lines: int = 80) -> str:
    """Return the final lines of a child-process log for CI diagnostics."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"<unable to read {log_path}: {exc}>"
    return "\n".join(lines[-max_lines:]) or "<log is empty>"


def _wait_for_process_http_200(
        url: str,
        proc: subprocess.Popen[str],
        log_path: Path,
        *,
        timeout: float = 30.0,
) -> None:
    """Wait for one GUI child and surface its log if readiness fails."""
    try:
        _wait_for_http_200(url, timeout=timeout)
    except RuntimeError as exc:
        _stop_process(proc)
        raise RuntimeError(
                f"{exc}\nChild exit code: {proc.poll()}\n"
                f"Child log tail ({log_path}):\n{_read_log_tail(log_path)}"
        ) from exc


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
    _wait_for_process_http_200(
            base_url + "/", proc, log_path, timeout=30.0,
    )
    print(f"[gui]   ready at {base_url}")
    return proc, base_url


def shutdown_gui(proc: subprocess.Popen[str]) -> None:
    print("[gui] shutting down")
    _stop_process(proc)


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
            _capture_browse(context, base_url)
            _capture_file_explorer(context, base_url)
            _capture_build_pipeline(context, base_url)
            _capture_run_local(context, base_url)
            _capture_run_slurm(context, base_url)
            _capture_view_results(context, base_url)
            _capture_pick_points(context, base_url)
            _capture_analysis(context, base_url)
            _capture_aux_ports(context, base_url)
            _capture_qc_curation_loop(context, base_url)
            _capture_qc_review(context, base_url)
            _capture_heatmap_exploration(context, base_url)
            _capture_error_analysis(context, base_url)
            _capture_aux_wire_in_dag(context, base_url)
            _capture_wire_pipeline_as_aux(context, base_url)
            _capture_fix_validation_issues(context, base_url)
        finally:
            browser.close()


def _save(
    page,
    workflow: str,
    name: str,
    *,
    full_page: bool = False,
) -> None:
    target_dir = ASSETS_ROOT / workflow
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / name
    page.screenshot(path=str(out), full_page=full_page)
    print(f"[shot]   {workflow}/{name}")


def _assert_distinct_stage_images(workflow: str, names: tuple[str, ...]) -> None:
    """Fail when two named tutorial stages accidentally capture one UI state."""
    digests: dict[str, str] = {}
    for name in names:
        path = ASSETS_ROOT / workflow / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in digests:
            raise RuntimeError(
                f"{workflow}/{name} duplicates {workflow}/{digests[digest]} "
                f"(sha256={digest})"
            )
        digests[digest] = name
        print(f"[verify] {workflow}/{name} sha256={digest}")


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


def _bind_hub_results_asynchronously(page, base_url: str) -> None:
    """Bind the tutorial output through the hub's asynchronous API.

    This intentionally exercises the production bind-job contract instead of
    relying on a standalone viewer for every loaded-state capture.  The output
    was produced by :func:`run_cli_once`, so it has coherent terminal evidence;
    the helper waits for the server's terminal job response before reloading the
    Results mount that serves the atomically published snapshot.
    """
    submission = page.evaluate(
        """async () => {
            const response = await fetch('/sandbox/api/viewer/output-root', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: 'results'}),
            });
            return {status: response.status, payload: await response.json()};
        }"""
    )
    if submission["status"] != 202:
        raise RuntimeError(f"hub Results bind was rejected: {submission!r}")
    poll_path = submission["payload"].get("poll_path")
    if not isinstance(poll_path, str) or not poll_path:
        raise RuntimeError(f"hub Results bind omitted poll_path: {submission!r}")

    terminal: dict[str, Any] | None = None
    for _ in range(120):
        poll = page.evaluate(
            """async path => {
                const response = await fetch(path);
                return {status: response.status, payload: await response.json()};
            }""",
            poll_path,
        )
        if poll["status"] != 200:
            raise RuntimeError(f"hub Results bind poll failed: {poll!r}")
        payload = poll["payload"]
        job = payload.get("job", {}) if isinstance(payload, dict) else {}
        if job.get("terminal") is True:
            terminal = payload
            break
        page.wait_for_timeout(250)
    if terminal is None:
        raise RuntimeError("hub Results bind did not reach a terminal state")
    if terminal.get("status") != "succeeded":
        raise RuntimeError(f"hub Results bind did not succeed: {terminal!r}")

    page.goto(base_url + "/results/")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)


# ---------------------------------------------------------------------------
# Linear builder interaction helpers
# ---------------------------------------------------------------------------
#
# The default builder surface is a fixed HTML port map. Tutorial capture drives
# the same click targets a user sees: palette buttons append/fill the selected
# green target, side-loader ports select parameter targets, and breadcrumbs
# drill back out of embedded ImagePipeline scopes. Retired Cytoscape drag/drop
# stores are intentionally not used here.


def _new_builder_page(context, base_url: str):
    """Open ``/builder/`` and block until the linear map has mounted."""
    page = _new_page(context, base_url, "/builder/")
    page.wait_for_selector("#palette", timeout=15_000)
    page.wait_for_selector("#linear-map-container", timeout=15_000)
    # The inspector slide-over and palette width/collapse persist their UI
    # state to localStorage, shared across pages in this context. Clear those
    # keys AND normalize the DOM (builder.js already restored from storage on
    # bind, so clearing alone won't re-close an already-open inspector) so
    # every builder capture starts from the server defaults — inspector
    # closed, palette expanded at the default width — regardless of the order
    # captures run in. Helpers re-open the inspector where a capture needs it.
    page.evaluate(
            """() => {
                try {
                    window.localStorage.removeItem('phenotypicBuilderInspectorClosed');
                    window.localStorage.removeItem('phenotypicBuilderPaletteCollapsed');
                    window.localStorage.removeItem('phenotypicBuilderPaletteWidth');
                } catch (err) { /* private mode — ignore */ }
                const slideover = document.getElementById('inspector-slideover');
                if (slideover) slideover.classList.add('is-closed');
                const columns = document.getElementById('builder-columns');
                if (columns) {
                    columns.classList.remove('palette-collapsed');
                    columns.style.removeProperty('--builder-palette-width');
                }
            }"""
    )
    page.wait_for_timeout(500)
    return page


def _relayout_canvas(page) -> None:
    """Settle the builder map before taking a screenshot.

    Older tutorial captures called Cytoscape's dagre relayout here. The fixed
    map has no graph layout pass, but keeping this helper as a short settle
    makes the capture flow deterministic after Dash callback redraws.
    """
    page.evaluate(
            "() => window.phenotypicRelayout && window.phenotypicRelayout()"
    )
    page.wait_for_timeout(700)


def _wait_linear_node_count(page, count: int) -> None:
    page.wait_for_function(
            "expected => document.querySelectorAll('.linear-node-card').length === expected",
            arg=count,
            timeout=15_000,
    )
    page.wait_for_timeout(250)


def _select_linear_node(page, class_name: str, which: str = "last") -> None:
    locator = page.locator(
            f'button.linear-node-title-button:has-text("{class_name}")'
    )
    if locator.count() == 0:
        print(f"[shot]   linear node {class_name} not found")
        return
    target = locator.first if which == "first" else locator.last
    # Dispatch the click rather than a real pointer click: once the inspector
    # slide-over is open it overlays the canvas's right edge, so a node there
    # is "obstructed" for a real click. Selection only needs the DOM click
    # event to reach Dash's delegated listener, which a dispatched event does.
    target.dispatch_event("click")
    page.wait_for_timeout(500)


def _open_inspector(page) -> None:
    """Open the inspector slide-over if it is closed.

    The inspector docks as a right-edge slide-over that is collapsed by
    default, so its contents (param form, point-picker buttons) sit off the
    right of the viewport until the tab handle is clicked. Captures that need
    the inspector visible call this after selecting a node.
    """

    slideover = page.locator("#inspector-slideover")
    if slideover.count() == 0:
        return
    classes = slideover.first.get_attribute("class") or ""
    if "is-closed" in classes:
        page.locator("#btn-inspector-slideover-toggle").first.click()
        page.wait_for_timeout(350)


def _select_side_param_target(page, param_name: str) -> None:
    """Select a side-loader parameter port as the active fill target."""

    # Side-loader ports live inside the closed-by-default inspector slide-over.
    _open_inspector(page)
    locator = page.locator(
            f'button.linear-side-param-port[aria-label="Fill {param_name}"]'
    )
    if locator.count() == 0:
        locator = page.locator(
                f'button.linear-port-param[aria-label="Fill {param_name}"]'
        )
    if locator.count() == 0:
        print(f"[shot]   side param port {param_name} not found")
        return
    locator.first.click()
    page.wait_for_timeout(600)


def _click_new_pipeline_button(page) -> None:
    button = page.locator("#btn-new-pipeline-node")
    if button.count() == 0:
        print("[shot]   + New Pipeline button not found")
        return
    button.first.click()
    page.wait_for_timeout(900)


def _click_root_breadcrumb(page) -> None:
    root = page.locator("#breadcrumb button").first
    if root.count() == 0:
        print("[shot]   root breadcrumb button not found")
        return
    root.click()
    page.wait_for_timeout(700)


# --- setup --------------------------------------------------------------

def _capture_setup(context, base_url: str) -> None:
    _emit_empty_state_shot(
            context, base_url, "/", "setup", "01_landing_page.png",
            log="[shot] workflow=setup",
    )


# --- browse (source image viewer) ---------------------------------------

def _browse_source_payload() -> dict | None:
    """Build the shared-source-root store payload for the Browse capture.

    The hub is booted rooted at :data:`DATASET_DIR`, so a ``SandboxRoot`` built
    here over the same root produces the exact versioned payload the shell's
    ``SHELL_SOURCE_IMAGE_ROOT_STORE`` expects. Pointing the source root at the
    dataset (rather than ``PLATES_DIR``) keeps ``plates/`` a nested dataset, so
    both cascading dropdowns are exercised. Returns ``None`` if the helper
    cannot resolve the path (e.g. the dataset was not built).
    """
    try:
        from phenotypic.gui.shell._sandbox import SandboxRoot
        from phenotypic.gui.shell._source_context import source_payload_from_path
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"[shot]   browse: source payload import failed: {exc!r}")
        return None
    sandbox = SandboxRoot.from_path(DATASET_DIR)
    payload = source_payload_from_path(sandbox, DATASET_DIR, source="manual")
    if payload is None:
        print("[shot]   browse: source payload could not be resolved")
    return payload


def _capture_browse(context, base_url: str) -> None:
    """Drive Browse and capture its optimized single-image workflow.

    Mirrors ``docs/source/tutorials/gui/18_browse.md``:

    1. ``01_empty_state.png`` — the Browse tab with no source root selected
       (the empty hint prompts the user to pick a source from the top bar).
    2. ``02_viewer.png`` — after setting the source root (pushed into the
       shared ``SHELL_SOURCE_IMAGE_ROOT_STORE`` via ``set_props``, exactly as
       the top-bar source picker would), the dataset + image dropdowns
       populate, the first image auto-selects, and the progressive preview,
       preparation controls, position readout, filmstrip, OpenSeadragon
       viewport, and metadata panel are visible together.

    The source root is set by writing the real versioned store payload rather
    than driving the modal picker — the same deterministic store-injection the
    point-picker capture uses for ``picker-staged-store``.
    """
    print("[shot] workflow=browse")
    page = _new_page(context, base_url, "/browse/")

    # 1) Empty state — no source root bound yet, so the empty hint shows.
    page.wait_for_timeout(500)
    _save(page, "browse", "01_empty_state.png")

    # 2) Set the shared source root, then let the dataset/image dropdowns
    #    populate and the OSD viewport render the first image.
    payload = _browse_source_payload()
    if payload is not None:
        page.evaluate(
                """payload => {
                    window.dash_clientside.set_props(
                        'shell-source-image-root-store', {data: payload}
                    );
                }""",
                payload,
        )
        # The dataset callback repopulates the pickers; the image callback then
        # auto-selects the first image, whose token feeds the clientside OSD
        # mount. Wait for the filmstrip and OSD canvas so the screenshot records
        # the optimized Single-view chrome instead of a transient loading state.
        page.wait_for_timeout(1500)
        try:
            page.wait_for_selector(
                "#browse-filmstrip .browse-filmstrip-item", timeout=10_000
            )
        except Exception:  # pragma: no cover - best-effort
            pass
        try:
            page.wait_for_selector(
                '#browse-filmstrip .browse-filmstrip-thumb[data-loaded="true"]',
                timeout=10_000,
            )
        except Exception:  # pragma: no cover - best-effort
            pass
        try:
            page.wait_for_selector("#browse-osd-div canvas", timeout=10_000)
        except Exception:  # pragma: no cover - best-effort
            pass
        page.wait_for_timeout(1500)
        page.locator("#browse-meta-image-name").scroll_into_view_if_needed()
    _save(page, "browse", "02_viewer.png", full_page=True)
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
    page = _new_builder_page(context, base_url)
    _relayout_canvas(page)
    _save(page, "build_pipeline", "01_builder_empty.png")

    _expand_palette_accordions(page)
    for cls in ("BlurGauss", "OtsuDetector", "MeasureShape", "MeasureSize"):
        _add_palette_op(page, cls)
    _wait_linear_node_count(page, 5)
    fit = page.locator("#linear-zoom-fit")
    if fit.count() > 0:
        fit.first.click()
        page.wait_for_timeout(400)
    _save(page, "build_pipeline", "02_builder_chain.png")
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

    # Bind a complete, sandbox-valid form. Store injection uses the same Dash
    # stores the three picker-confirm callbacks publish.
    slurm_output = DATASET_DIR / "slurm_output"
    if slurm_output.exists():
        shutil.rmtree(slurm_output)
    selections = {
        "rc-store-pipeline-path": str(PIPELINE_JSON.resolve()),
        "rc-store-input-dir": str(PLATES_DIR.resolve()),
        "rc-store-output-dir": str(slurm_output.resolve()),
    }
    page.evaluate(
        """items => {
            for (const [id, data] of Object.entries(items)) {
                window.dash_clientside.set_props(id, {data});
            }
        }""",
        selections,
    )
    page.wait_for_function(
        """() => {
            const pipeline = document.querySelector('#rc-label-pipeline');
            const input = document.querySelector('#rc-label-input');
            const output = document.querySelector('#rc-label-output');
            return pipeline?.textContent?.includes('pipeline.json.pht-pipe')
                && input?.textContent?.includes('plates')
                && output?.textContent?.includes('slurm_output');
        }""",
        timeout=8_000,
    )

    # Toggle SLURM mode in the radio. dbc.RadioItems renders as labels
    # with hidden inputs; clicking the visible label is the reliable path.
    slurm_label = page.locator('label:has-text("SLURM")').first
    slurm_label.wait_for(state="visible", timeout=8_000)
    slurm_label.click()
    page.wait_for_timeout(400)

    # Open SLURM collapse if it isn't already.
    toggle = page.locator("#rc-btn-toggle-slurm")
    toggle.wait_for(state="visible", timeout=8_000)
    collapse = page.locator("#rc-collapse-slurm")
    if not collapse.is_visible():
        toggle.click()
        page.wait_for_timeout(400)

    values = {
        "#rc-input-slurm-partition": "general",
        "#rc-input-slurm-time": "04:00:00",
        "#rc-input-slurm-mem": "16G",
        "#rc-input-slurm-cpus": "4",
        "#rc-input-slurm-gpus": "0",
        "#rc-input-slurm-extra": "account=lab\nqos=normal",
    }
    for selector, value in values.items():
        field = page.locator(selector)
        field.wait_for(state="visible", timeout=8_000)
        field.fill(value)
        field.press("Enter")
    page.wait_for_timeout(700)

    # Require the form-level validator to accept every selected path/resource.
    # The capture does not launch even a dry-run generation because the durable
    # registry intentionally rejects reusing an output with a nonterminal run.
    validate = page.locator("#rc-btn-validate")
    validate.wait_for(state="visible", timeout=8_000)
    if not validate.is_enabled():
        raise RuntimeError("SLURM tutorial form is not valid after path selection")
    for selector, expected in values.items():
        actual = page.locator(selector).input_value().rstrip("\n")
        if actual != expected:
            raise RuntimeError(
                f"SLURM tutorial field {selector} did not retain {expected!r}"
            )
    page.set_viewport_size({"width": VIEWPORT["width"], "height": 1400})
    page.evaluate(
        """() => {
            window.scrollTo(0, 0);
            for (const element of document.querySelectorAll('*')) {
                if (element.scrollTop) element.scrollTop = 0;
            }
        }"""
    )
    _save(page, "run_slurm", "01_slurm_mode.png", full_page=True)
    page.close()


# --- view results -------------------------------------------------------

def _capture_view_results(context, base_url: str) -> None:
    """Capture the empty, asynchronous-bind, and loaded hub Results states."""
    print("[shot] workflow=view_results (hub async binding)")
    page = _new_page(context, base_url, "/results/")
    _save(page, "view_results", "01_viewer_empty.png")
    _bind_hub_results_asynchronously(page, base_url)
    page.wait_for_selector("#results-viewer-empty-state", state="detached", timeout=15_000)
    snapshot_status = page.locator("#header-snapshot-status")
    snapshot_status.wait_for(state="visible", timeout=15_000)
    if snapshot_status.inner_text().strip() != "Current":
        raise RuntimeError(
            "hub Results bind did not publish a coherent mutation-enabled "
            f"snapshot: {snapshot_status.inner_text()!r}"
        )
    _save(page, "view_results", "05_hub_bound_snapshot.png")
    page.close()


def _capture_pick_points(context, base_url: str) -> None:
    """Drive the in-builder point-picker workflow and capture eight PNGs.

    The shots demonstrate the manual-curation flow described in
    ``docs/source/tutorials/gui/07_pick_points.md``:

    1. Palette with the PICK badge visible on the two pickable ops.
    2. Canvas with ``BlurGauss → OtsuDetector → ManualRefine``.
    3. Inspector param form for ``ManualRefine`` (count = "0 points").
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

    # 2) Pipeline with BlurGauss → OtsuDetector → ManualRefine.
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

    for cls in ("BlurGauss", "OtsuDetector", "ManualRefine"):
        _add_op(cls)
    _wait_linear_node_count(page, 4)
    _relayout_canvas(page)
    _save(page, "pick_points", "02_pipeline_with_selector.png")

    # 3) Inspector param form for ManualRefine. The inspector docks as a
    # closed-by-default slide-over, so open it before screenshotting the
    # param form (and before reaching the picker button inside it).
    _select_linear_node(page, "ManualRefine")
    _open_inspector(page)
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
    """Drive the linear side-loader aux workflow and capture 4 PNGs.

    Operation-typed parameters render as visible ports on the map and in the
    side loader. Selecting the side-loader port marks it green; the next
    compatible palette click fills that slot with a hidden aux source block.

    Steps exercised against the real dispatcher:

    1. ``01_initial.png`` — empty builder canvas + palette.
    2. ``02_main_pipeline.png`` — palette clicks build the ribbon
       ``Input Image -> BlurGauss -> FilamentousFungiDetector``; the
       detector shows a required empty ``inoculum_detector`` side value.
    3. ``03_aux_wired.png`` — selecting that side port and clicking
       ``OtsuDetector`` fills the value row.
    4. ``04_inspector_aux.png`` — the consumer's side loader shows Replace,
       Clear, and docstring actions for the filled value.
    """
    print("[shot] workflow=aux_ports")
    page = _new_builder_page(context, base_url)

    # 1) Empty canvas — same starting point as the Build Pipeline tutorial.
    _relayout_canvas(page)
    _save(page, "aux_ports", "01_initial.png")

    _expand_palette_accordions(page)

    # 2) Main ribbon with the aux-consuming op (FFD) on the tail.
    for cls in ("BlurGauss", "FilamentousFungiDetector"):
        _add_palette_op(page, cls)
    _wait_linear_node_count(page, 3)
    _relayout_canvas(page)
    _save(page, "aux_ports", "02_main_pipeline.png")

    # 3) Select the side-loader target, then fill it with a compatible
    #    detector from the palette.
    _select_linear_node(page, "FilamentousFungiDetector")
    _select_side_param_target(page, "inoculum_detector")
    _add_palette_op(page, "OtsuDetector")
    _relayout_canvas(page)
    _save(page, "aux_ports", "03_aux_wired.png")

    # 4) Keep the consumer selected so the filled side value row is visible.
    _select_linear_node(page, "FilamentousFungiDetector")
    _open_inspector(page)
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


def _capture_qc_review(context, base_url: str) -> None:
    """Capture the QC review walkthrough workflow.

    Spec §D. Drives the Results Viewer's QC tab into its **Review**
    sub-view — the master–detail walkthrough that walks the
    worst-agreeing groups for a selected QC module, reuses the
    colony-view tile gallery for per-colony curation, and recomputes the
    module after each curated group.

    The hub-mounted viewer has no ``output_root`` until the sidebar binds
    one, so the populated Review state (worklist + detail gallery) is
    captured from the standalone viewer in :func:`_qc_review_loaded_shots`.
    This hub capture records the empty-state entry point:

    1. ``01_empty_state.png`` — the QC tab in the hub viewer's empty
       state, before a module/output is bound.
    """
    print("[shot] workflow=qc_review")
    page = _new_page(context, base_url, "/results/")
    page.wait_for_timeout(800)
    qc_tab = page.locator('a[role="tab"]:has-text("QC")').first
    if qc_tab.count() > 0:
        try:
            qc_tab.click(timeout=2000)
            page.wait_for_timeout(600)
        except Exception:  # pragma: no cover - best-effort
            pass
    _save(page, "qc_review", "01_empty_state.png")
    page.close()


def _qc_review_loaded_shots(page) -> None:
    """Capture loaded-state QC Review screenshots in the standalone viewer.

    The standalone launcher has a real ``output_root`` carrying
    ``deliverables/qc/qc.duckdb``, so flipping the QC tab's Configure | Review toggle renders a
    populated worklist + detail gallery. Captures:

    * ``02_review_worklist.png`` — the Review sub-view with the module
      picker, summary header, and worst-first worklist.
    * ``03_detail_gallery.png`` — the detail pane for the first
      (worst) group: header + tile gallery + action bar.
    """
    qc_tab = page.locator('a[role="tab"]:has-text("QC")').first
    if qc_tab.count() == 0:
        print("[shot]   qc_review: QC tab not found — loaded captures skipped")
        return
    try:
        qc_tab.click(timeout=3000)
        page.wait_for_timeout(800)
    except Exception:  # pragma: no cover - best-effort
        pass

    # Flip the Configure | Review segmented toggle to Review. The dbc
    # RadioItems renders the option as a Bootstrap ``btn-check`` radio
    # whose <input> is ``display:none``; click its <label> (matched by the
    # input's id suffix) with ``force=True``, then wait for the Review
    # container to actually become visible + its worklist to render before
    # shooting — a plain click+sleep raced the callback and captured the
    # Configure view.
    review_label = page.locator(
        'label[for$="qc-subview-toggle_input_review"]'
    ).first
    if review_label.count() > 0:
        try:
            review_label.click(timeout=2000, force=True)
            page.wait_for_function(
                "() => {"
                "  const v = document.querySelector('#qc-review-view');"
                "  return v && getComputedStyle(v).display !== 'none';"
                "}",
                timeout=5000,
            )
            page.wait_for_selector(".qc-worklist-row", timeout=5000)
            page.wait_for_timeout(500)
        except Exception:  # pragma: no cover - best-effort
            pass
    _save(page, "qc_review", "02_review_worklist.png")

    # Click the first (worst) worklist row to populate the detail pane.
    first_row = page.locator(".qc-worklist-row").first
    if first_row.count() > 0:
        try:
            first_row.click(timeout=2000)
            page.wait_for_timeout(1500)
        except Exception:  # pragma: no cover - best-effort
            pass
    _save(page, "qc_review", "03_detail_gallery.png")


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
       open so the reader can see the available measurement columns.
       Outputs with QC data also contribute QC-derived options.
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

    # Open the color picker dropdown so the available columns are visible.
    # This fixture need not contain a QC-derived option for every capture.
    color_picker = page.locator("#heatmap-color-picker")
    if color_picker.count() > 0:
        try:
            color_picker.click(timeout=2000)
            page.wait_for_timeout(500)
            _save(page, "heatmap_exploration", "02_color_picker_open.png")
        except Exception:  # pragma: no cover - best-effort
            pass

    page.close()


def _capture_error_analysis(context, base_url: str) -> None:
    """Capture the Error-analysis walkthrough (empty-state via the hub viewer).

    The hub-mounted viewer has no ``output_root`` until the sidebar binds
    one, so the Error tab renders its "need more labels" empty state — the
    natural entry-point shot for the tutorial. The populated ranked table +
    distribution plot + good-baseline toggle are captured from the standalone
    viewer in :func:`_error_analysis_loaded_shots` (it has a real
    ``output_root`` carrying the seeded
    ``deliverables/qc/curation_labels.parquet``).
    """
    print("[shot] workflow=error_analysis")
    page = _new_page(context, base_url, "/results/")
    page.wait_for_timeout(800)
    tab = page.locator('a[role="tab"]:has-text("Error")').first
    if tab.count() > 0:
        try:
            tab.click(timeout=2000)
            page.wait_for_timeout(600)
        except Exception:  # pragma: no cover - best-effort
            pass
    _save(page, "error_analysis", "01_empty_state.png")
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

    The fixed linear builder routes palette clicks through the selected
    green target. The default target is the floating continuation, so a
    sequence of calls builds ``Input Image -> ... -> tail``. If a side-loader
    parameter port is selected first, the same click fills that parameter.
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
    """Drive the linear aux-fill workflow and capture 3 PNGs.

    The folder name is kept for existing tutorial links, but the default
    builder no longer exposes DAG wire drawing. The visible flow is: select a
    gold parameter port, click a compatible palette operation, and manage the
    filled value from the side loader.

    Steps exercised against the real dispatcher:

    1. ``01_main_with_consumer.png`` — palette clicks build the main
       ribbon ``Input Image -> BlurGauss -> ContrastStretching ->
       FilamentousFungiDetector``.
    2. ``02_detector_dropped.png`` — the ``inoculum_detector`` side port is
       selected green as the active fill target.
    3. ``03_aux_wired.png`` — clicking ``OtsuDetector`` fills the side value
       and returns the active target to the floating continuation.
    """
    print("[shot] workflow=aux-wire-in-dag")
    page = _new_builder_page(context, base_url)
    _expand_palette_accordions(page)

    # 1) Main ribbon + the consumer whose aux port we will feed.
    for cls in ("BlurGauss", "ContrastStretching", "FilamentousFungiDetector"):
        _add_palette_op(page, cls)
    _wait_linear_node_count(page, 4)
    _relayout_canvas(page)
    _save(page, "aux-wire-in-dag", "01_main_with_consumer.png")

    # 2) Select the side parameter port so readers can see the green target.
    _select_linear_node(page, "FilamentousFungiDetector")
    _select_side_param_target(page, "inoculum_detector")
    _relayout_canvas(page)
    _save(page, "aux-wire-in-dag", "02_detector_dropped.png")

    # 3) Fill that target with a compatible detector.
    _add_palette_op(page, "OtsuDetector")
    _relayout_canvas(page)
    _save(page, "aux-wire-in-dag", "03_aux_wired.png")
    page.close()


def _capture_wire_pipeline_as_aux(context, base_url: str) -> None:
    """Drive the embedded ImagePipeline side-value workflow and capture 3 PNGs.

    A side parameter target can be filled with ``+ New Pipeline``. The builder
    immediately drills into that embedded scope; breadcrumbs return to the
    consumer that owns the side value.

    Steps exercised against the real dispatcher:

    1. ``01_empty_container.png`` — the embedded pipeline has just been
       created and the active breadcrumb is inside its empty linear scope.
    2. ``02_chain_in_container.png`` — palette clicks build
       ``Input Image -> BlurGauss -> OtsuDetector`` inside that scope.
    3. ``03_pipeline_wired_as_aux.png`` — breadcrumb returns to root and the
       consumer side loader shows the embedded ``ImagePipeline`` value.
    """
    print("[shot] workflow=wire-pipeline-as-aux")
    page = _new_builder_page(context, base_url)
    _expand_palette_accordions(page)

    _add_palette_op(page, "FilamentousFungiDetector")
    _wait_linear_node_count(page, 2)
    _select_linear_node(page, "FilamentousFungiDetector")
    _select_side_param_target(page, "inoculum_detector")
    _click_new_pipeline_button(page)
    _relayout_canvas(page)
    _save(page, "wire-pipeline-as-aux", "01_empty_container.png")

    # 2) Build the embedded linear chain.
    for cls in ("BlurGauss", "OtsuDetector"):
        _add_palette_op(page, cls)
    _wait_linear_node_count(page, 3)
    _relayout_canvas(page)
    _save(page, "wire-pipeline-as-aux", "02_chain_in_container.png")

    # 3) Drill back out and show the embedded pipeline value on the consumer.
    _click_root_breadcrumb(page)
    _select_linear_node(page, "FilamentousFungiDetector")
    _relayout_canvas(page)
    _save(page, "wire-pipeline-as-aux", "03_pipeline_wired_as_aux.png")
    page.close()


def _capture_fix_validation_issues(context, base_url: str) -> None:
    """Drive the validation-issue triage workflow and capture 3 PNGs.

    Mirrors ``docs/source/tutorials/gui/14_fix_validation_issues.md``:
    a missing required side value trips a blocking validation rule, the
    toolbar issue badge surfaces the count, and filling the side value clears
    the issue.

    Steps exercised against the real dispatcher:

    1. ``01_issue_introduced.png`` — ``FilamentousFungiDetector`` is on the
       main spine with an empty required ``inoculum_detector`` value.
    2. ``02_issue_focused.png`` — the issue badge popover lists the missing
       required parameter.
    3. ``03_issue_resolved.png`` — selecting that side target and clicking
       ``OtsuDetector`` clears the issue.
    """
    print("[shot] workflow=fix-validation-issues")
    page = _new_builder_page(context, base_url)
    _expand_palette_accordions(page)

    # 1) Build a main spine with a consumer missing a required side value.
    for cls in ("BlurGauss", "FilamentousFungiDetector"):
        _add_palette_op(page, cls)
    _wait_linear_node_count(page, 3)
    _select_linear_node(page, "FilamentousFungiDetector")
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

    # 3) Fill the required side value.
    _select_linear_node(page, "FilamentousFungiDetector")
    _select_side_param_target(page, "inoculum_detector")
    _add_palette_op(page, "OtsuDetector")
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
        _stop_process(proc)


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
        _wait_for_process_http_200(
                base_url + "/", proc, log_path, timeout=30.0,
        )
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

                # Open the right-docked filter offcanvas from the top-bar
                # toggle and capture the slide-in filter panel, then close
                # it again so the later full-page shots are unobstructed.
                try:
                    page.click("#btn-filters-toggle", timeout=4000)
                    # Wait on the always-sized "+ Add filter row" button —
                    # the rows container is empty (zero-height) until a row
                    # is added, so it never reports "visible".
                    page.wait_for_selector(
                            "#btn-add-filter-row",
                            state="visible",
                            timeout=10_000,
                    )
                    page.wait_for_timeout(500)
                    _save(page, "view_results", "04_filter_offcanvas.png")
                    backdrop = page.locator(".offcanvas-backdrop")
                    if backdrop.count() > 0:
                        backdrop.first.click(timeout=2000)
                    else:
                        page.keyboard.press("Escape")
                    page.wait_for_timeout(400)
                except Exception as exc:  # pragma: no cover - best-effort
                    print(f"[shot]   filter-offcanvas shot skipped: {exc!r}")

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
                _qc_review_loaded_shots(page)
                _heatmap_exploration_loaded_shots(page)
                _error_analysis_loaded_shots(page)

                page.close()
            finally:
                browser.close()
    finally:
        _stop_process(proc)


def _qc_curation_loop_loaded_shots(page) -> None:
    """Capture loaded-state QC tab screenshots inside the standalone viewer.

    The hub-mounted ``_capture_qc_curation_loop`` only sees the empty
    state because the hub viewer is unbound at startup. The standalone
    viewer has a real ``output_root``, so the tab strip mounts and the
    QC top-strip controls (``+ Add check`` / ``Rebuild QC database``) are
    reachable. Two additional screenshots cover the loaded state:

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


def _error_analysis_loaded_shots(page) -> None:
    """Capture the populated Error-analysis tab in the standalone viewer.

    The standalone launcher has a real ``output_root`` carrying the seeded
    ``deliverables/qc/curation_labels.parquet`` (12 ``background_noise`` labels written by
    :func:`_seed_error_triage_labels`), so flipping to the Error tab ranks the
    measurements that separate that error category from the all-unlabeled good
    baseline. The recompute only fires when the tab is active (off-tab
    ``PreventUpdate``), so click the tab and wait for the ranked
    ``dash_table.DataTable`` cells before shooting. Captures:

    * ``02_ranked_table.png`` — category chips + ranked cutoff table
      (measurement, AUC, suggested cutoff, recall/specificity, BH-p).
    * ``03_distribution_cutoff.png`` — the good-vs-error distribution figure
      with the draggable cutoff line + recall/specificity readout.
    * ``04_good_baseline_toggle.png`` — the All-unlabeled / Verified-only
      good-baseline toggle + the verified-good count badge.
    """
    tab = page.locator('a[role="tab"]:has-text("Error")').first
    tab.wait_for(state="visible", timeout=8_000)
    tab.click(timeout=3_000)
    # Fatal assertion: never save the empty/fallback view as a ranked-table
    # tutorial stage.
    page.wait_for_selector(
        "#error-cutoff-table .dash-cell",
        state="visible",
        timeout=12_000,
    )
    if page.locator("#error-empty-state").is_visible():
        raise RuntimeError("Error analysis remained in Need more labels state")
    page.wait_for_timeout(800)
    _save(page, "error_analysis", "02_ranked_table.png")

    # The good-vs-error distribution figure (dcc.Graph) with the cutoff line.
    figure = page.locator("#error-distribution-figure")
    figure.wait_for(state="visible", timeout=8_000)
    page.wait_for_selector(
        "#error-distribution-figure .js-plotly-plot",
        state="visible",
        timeout=8_000,
    )
    readout = page.locator("#error-readout")
    readout.wait_for(state="visible", timeout=8_000)
    readout.evaluate(
        "(el) => el.scrollIntoView({block: 'center', inline: 'nearest'})"
    )
    page.wait_for_timeout(400)
    _save(page, "error_analysis", "03_distribution_cutoff.png")

    # Good-baseline toggle (All unlabeled / Verified only) + verified count.
    verified = page.locator(
        'label[for$="error-good-mode-toggle_input_verified"]'
    ).first
    verified.wait_for(state="visible", timeout=8_000)
    verified.click(timeout=3_000, force=True)
    badge = page.locator("#error-verified-count")
    badge.wait_for(state="visible", timeout=8_000)
    page.wait_for_function(
        "() => {"
        " const b = document.querySelector('#error-verified-count');"
        " return b && /verified good:\\s*[1-9]/.test(b.textContent || '');"
        "}",
        timeout=8_000,
    )
    if page.locator("#error-empty-state").is_visible():
        raise RuntimeError("Verified-only Error preview lacks a seeded good baseline")
    page.locator("#error-good-mode-toggle").scroll_into_view_if_needed(timeout=2_000)
    page.wait_for_timeout(400)
    _save(page, "error_analysis", "04_good_baseline_toggle.png")


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
        run_cli_once()

    # Seed synthetic error-category labels so the Error-analysis tab renders a
    # populated ranked table in the standalone-viewer capture. Runs after the
    # CLI master exists (or after a ``--skip-cli`` reuse of a prior master) and
    # before any GUI boots. This is required evidence and therefore fatal.
    _seed_error_triage_labels()

    proc, base_url = boot_gui(DATASET_DIR)
    try:
        capture_workflow_screenshots(base_url, headed=args.headed)
    finally:
        shutdown_gui(proc)

    capture_standalone_viewer_screenshots(headed=args.headed)
    capture_standalone_analysis_screenshots(headed=args.headed)
    _assert_distinct_stage_images(
        "error_analysis",
        (
            "02_ranked_table.png",
            "03_distribution_cutoff.png",
            "04_good_baseline_toggle.png",
        ),
    )

    print("[done] all screenshots written to docs/source/_static/gui_images/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
