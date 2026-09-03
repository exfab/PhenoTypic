"""
Dashboard HTML generator for PhenoTypic CLI processing progress.

Generates a single self-contained HTML+CSS+JS dashboard file that monitors
processing progress by polling ``manifest.json`` and ``failures.jsonl`` in
the ``.phenotypic/progress/`` machine-state subdirectory.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Collection, Mapping
from pathlib import Path

from phenotypic._assets import asset_bytes
from phenotypic.sdk_ import (
    DashboardManifestKey,
    DIR_PHENOTYPIC,
    DIR_PROGRESS,
    atomic_write_json,
    dashboard_html_path,
    manifest_json_path,
)
from phenotypic.sdk_.typing_ import ExecutionMode

logger = logging.getLogger(__name__)


def generate_dashboard(output_dir: Path, *, execution_mode: ExecutionMode = "local") -> None:
    """Write the live progress ``dashboard.html`` into ``deliverables/``.

    The dashboard is a user-facing deliverable, while the manifest and failure
    records it polls stay in ``<output>/.phenotypic/progress/``.

    Args:
        output_dir: Root output directory.
        execution_mode: ``"local"`` or ``"slurm"``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = dashboard_html_path(output_dir)
    # This HTML writer does not go through the atomic writer, so create its
    # deliverables parent explicitly.
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(_build_html(execution_mode), encoding="utf-8")
    logger.info("Dashboard written to %s", dashboard_path)


def regenerate_dashboard_artifacts(
    output_dir: Path,
    job_meta: "dict | None",
    datasets: "dict[str, int]",
    *,
    dataset_inventory: "Mapping[str, Collection[str]] | None" = None,
) -> None:
    """Rebuild manifest.json and regenerate dashboard HTML in one call.

    Consolidates the ``build_manifest`` + ``generate_dashboard`` pattern used
    by checkpoint handlers, recompile workers, and local-run finalisers.

    Args:
        output_dir: Root output directory.
        job_meta: Parsed ``job_metadata.json`` dict (may be ``None`` for
            runs that pre-date metadata persistence).
        datasets: Mapping of dataset name to total image count.
        dataset_inventory: The authorized image names per dataset. Given
            explicitly it wins over the one derived from *job_meta*, which
            is what a caller holding a freshly-walked inventory should do
            -- and what a run with no ``job_metadata.json`` at all must
            do, since there is nothing to derive from.

            Supplying it is not optional detail. ``build_manifest``
            reconciles ``datasets`` against it and raises on a mismatch,
            and falls back to counting completions from it when the event
            log carries nothing for a dataset. Passing ``None`` disables
            both: a recompile that did so wrote a manifest declaring
            ``completed: 0`` for a fully completed run, and no guard
            noticed.
    """
    from phenotypic.sdk_ import JobMetadataKey, progress_dir, resolve_execution_mode
    from ._manifest_builder import (
        build_manifest,
        dataset_inventory_from_metadata,
    )

    prog_dir = progress_dir(output_dir)
    execution_mode = resolve_execution_mode(job_meta)
    gui_record_generation = (
        (job_meta or {}).get(JobMetadataKey.GUI_RECORD_GENERATION)
        if execution_mode == "local"
        else None
    )
    if dataset_inventory is None:
        dataset_inventory = dataset_inventory_from_metadata(
            (job_meta or {}).get(JobMetadataKey.DATASETS)
        )

    build_manifest(
        output_dir=output_dir,
        progress_dir=prog_dir,
        datasets=datasets,
        execution_mode=execution_mode,
        start_time=(job_meta or {}).get(JobMetadataKey.START_TIME, ""),
        slurm_job_ids=(job_meta or {}).get(JobMetadataKey.CHUNK_JOB_IDS),
        chunk_scripts=(job_meta or {}).get(JobMetadataKey.CHUNK_SCRIPTS),
        input_path=(job_meta or {}).get(JobMetadataKey.INPUT_PATH),
        dataset_inventory=dataset_inventory,
        processing_generation=(job_meta or {}).get(
            JobMetadataKey.PROCESSING_GENERATION
        ),
    )
    generate_dashboard(output_dir, execution_mode=execution_mode)
    if isinstance(gui_record_generation, str) and gui_record_generation:
        # Authenticate the manifest only after both canonical manifest and
        # dashboard publication succeed. A dashboard-only failure therefore
        # cannot leave current-generation evidence for the GUI exit observer.
        path = manifest_json_path(output_dir)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Canonical dashboard manifest is not an object")
        payload[DashboardManifestKey.GUI_RECORD_GENERATION] = (
            gui_record_generation
        )
        atomic_write_json(path, payload, sort_keys=False)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_html(execution_mode: str) -> str:
    """Assemble the complete self-contained HTML document."""
    logo_data_uri = _load_logo_data_uri()
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "  <title>PhenoTypic Dashboard</title>\n"
        "  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n"
        "  <link href=\"https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap\" rel=\"stylesheet\">\n"
        f"  <style>\n{_build_css()}\n  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{_build_body(execution_mode, logo_data_uri)}\n"
        f"  <script>\n{_build_js(execution_mode)}\n  </script>\n"
        "</body>\n"
        "</html>\n"
    )


def _load_logo_data_uri() -> str:
    """Read the logo PNG and return a ``data:`` URI, or empty string on failure."""
    try:
        raw = asset_bytes("logos/LogoArtOnly.png")
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except (OSError, ModuleNotFoundError, TypeError):
        logger.debug("Logo asset not found — skipping")
        return ""


def _build_css() -> str:
    """Return the inline CSS block for the dashboard."""
    base_css = """\
    /* ── Reset & Base ─────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --color-navy:    #003660;
      --color-blue:    #1b75bc;
      --color-gold:    #febc11;
      --color-white:   #ffffff;
      --color-bg:      #f5f7fa;
      --color-surface: #ffffff;
      --color-border:  #dde3ed;
      --color-rule:    #e8ecf2;
      --color-muted:   #8892a4;
      --color-body:    #2e3a4e;
      --color-heading: #003660;

      --oi-orange:    #E69F00;
      --oi-sky:       #56B4E9;
      --oi-green:     #009E73;
      --oi-vermilion: #D55E00;
      --oi-blue:      #0072B2;
      --oi-purple:    #CC79A7;
      --oi-yellow:    #F0E442;
      --oi-grey:      #BBBBBB;

      --color-success: #009E73;
      --color-info:    #56B4E9;
      --color-warning: #E69F00;
      --color-danger:  #D55E00;

      --font-display: 'DM Serif Display', Georgia, serif;
      --font-body:    'DM Sans', system-ui, sans-serif;
      --font-mono:    'DM Mono', 'Courier New', monospace;

      --text-xs:   0.6875rem;
      --text-sm:   0.8125rem;
      --text-base: 0.9375rem;
      --text-md:   1.0625rem;
      --text-lg:   1.25rem;
      --text-xl:   1.5rem;
      --text-2xl:  1.875rem;
      --text-3xl:  2.5rem;
      --text-4xl:  3.25rem;

      --sp-1: 0.25rem;
      --sp-2: 0.5rem;
      --sp-3: 0.75rem;
      --sp-4: 1rem;
      --sp-5: 1.25rem;
      --sp-6: 1.5rem;
      --sp-8: 2rem;
      --sp-10: 2.5rem;
      --sp-12: 3rem;
      --sp-16: 4rem;

      --radius-sm: 3px;
      --radius:    6px;
      --radius-md: 10px;
      --radius-lg: 16px;

      --shadow-sm: 0 1px 3px rgba(0,54,96,0.07), 0 1px 2px rgba(0,54,96,0.04);
      --shadow:    0 4px 12px rgba(0,54,96,0.08), 0 1px 3px rgba(0,54,96,0.05);
      --shadow-md: 0 8px 24px rgba(0,54,96,0.10), 0 2px 6px rgba(0,54,96,0.06);
      --shadow-lg: 0 16px 40px rgba(0,54,96,0.12), 0 4px 12px rgba(0,54,96,0.07);

      --ease-out: cubic-bezier(0.22, 1, 0.36, 1);
      --transition: 180ms var(--ease-out);
    }

    body {
      font-family: var(--font-body);
      background: var(--color-bg);
      color: var(--color-body);
      line-height: 1.5;
      min-height: 100vh;
      font-size: var(--text-base);
    }

    /* ── Layout ───────────────────────────────────────────────── */
    .container {
      max-width: 1280px;
      margin: 0 auto;
      padding: var(--sp-6) var(--sp-5) var(--sp-16);
    }

    /* ── Header ───────────────────────────────────────────────── */
    .header {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: var(--sp-3);
      margin-bottom: var(--sp-8);
      padding-bottom: var(--sp-5);
      border-bottom: 1px solid var(--color-rule);
    }
    .header h1 {
      font-family: var(--font-display);
      font-size: var(--text-2xl);
      font-weight: 400;
      color: var(--color-heading);
    }
    .header-title-group {
      display: flex;
      align-items: center;
      gap: var(--sp-3);
    }
    .header-logo img {
      max-height: 180px;
      width: auto;
      background: transparent;
    }
    .header-right {
      display: flex;
      align-items: center;
      gap: var(--sp-4);
      font-size: var(--text-sm);
      color: var(--color-muted);
    }
    .input-path {
      font-family: var(--font-mono);
      font-size: var(--text-xs);
      color: var(--color-muted);
      background: var(--color-bg);
      border: 1px solid var(--color-border);
      border-radius: 9999px;
      padding: 0.15rem 0.6rem;
      letter-spacing: 0.02em;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 0.2rem 0.55rem;
      border-radius: 9999px;
      font-family: var(--font-mono);
      font-weight: 500;
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border: 1px solid;
    }
    .status-live {
      background: rgba(0,158,115,0.08);
      color: #006B4F;
      border-color: rgba(0,158,115,0.20);
    }
    .status-complete {
      background: rgba(0,54,96,0.08);
      color: #003660;
      border-color: rgba(0,54,96,0.15);
    }
    .status-error {
      background: rgba(213,94,0,0.08);
      color: #D55E00;
      border-color: rgba(213,94,0,0.20);
    }
    .pulse-dot {
      width: 5px; height: 5px;
      border-radius: 50%;
      background: #009E73;
      animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50%      { opacity: 0.4; transform: scale(0.7); }
    }

    /* ── Summary Cards ────────────────────────────────────────── */
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: var(--sp-4);
      margin-bottom: var(--sp-8);
    }
    .card {
      position: relative;
      overflow: hidden;
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--sp-5) var(--sp-6);
      text-align: center;
      box-shadow: var(--shadow-sm);
      transition: border-color var(--transition);
    }
    .card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: var(--color-navy);
    }
    .card:hover { border-color: var(--color-blue); }
    .card-value {
      font-family: var(--font-display);
      font-size: var(--text-3xl);
      font-weight: 400;
      color: var(--color-heading);
    }
    .card-label {
      font-family: var(--font-mono);
      font-size: var(--text-xs);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--color-muted);
      margin-top: var(--sp-1);
    }
    .card.completed::before { background: #009E73; }
    .card.failed::before    { background: #D55E00; }
    .card.running::before   { background: #56B4E9; }
    .card.pending::before   { background: var(--color-muted); }

    /* ── Overall Progress Bar ─────────────────────────────────── */
    .progress-section {
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--sp-5) var(--sp-6);
      margin-bottom: var(--sp-8);
      box-shadow: var(--shadow-sm);
    }
    .progress-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: var(--sp-3);
    }
    .progress-title {
      font-family: var(--font-body);
      font-weight: 500;
      color: var(--color-heading);
      font-size: var(--text-sm);
    }
    .progress-pct {
      font-family: var(--font-mono);
      font-weight: 500;
      font-size: var(--text-md);
      color: var(--color-heading);
    }
    .progress-track {
      width: 100%;
      height: 10px;
      background: var(--color-rule);
      border-radius: 9999px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      border-radius: 9999px;
      background: var(--color-navy);
      background-image: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.25));
      transition: width 0.6s ease;
    }
    .progress-fill.complete {
      background: var(--oi-green);
      background-image: none;
    }

    /* ── Active Batch / SLURM Info ────────────────────────────── */
    .slurm-section {
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--sp-5) var(--sp-6);
      margin-bottom: var(--sp-8);
      box-shadow: var(--shadow-sm);
    }
    .slurm-section h2 {
      font-family: var(--font-display);
      font-size: var(--text-lg);
      font-weight: 400;
      color: var(--color-heading);
      margin-bottom: var(--sp-4);
    }
    .chunk-grid {
      display: flex;
      flex-wrap: wrap;
      gap: var(--sp-2);
    }
    .chunk-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 0.2rem 0.55rem;
      border-radius: 9999px;
      font-family: var(--font-mono);
      font-size: 0.65rem;
      font-weight: 500;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      border: 1px solid;
    }
    .chunk-active {
      background: rgba(86,180,233,0.10);
      color: #0B6E9E;
      border-color: rgba(86,180,233,0.25);
    }
    .chunk-completed {
      background: rgba(0,158,115,0.08);
      color: #006B4F;
      border-color: rgba(0,158,115,0.20);
    }
    .chunk-pending {
      background: rgba(0,54,96,0.08);
      color: var(--color-muted);
      border-color: rgba(0,54,96,0.15);
    }
    .chunk-label {
      font-size: 0.6rem;
      font-weight: 400;
      opacity: 0.8;
    }

    /* ── Per-Dataset Breakdown ────────────────────────────────── */
    .datasets-section {
      margin-bottom: var(--sp-8);
    }
    .datasets-section > h2 {
      font-family: var(--font-display);
      font-size: var(--text-lg);
      font-weight: 400;
      color: var(--color-heading);
      margin-bottom: var(--sp-4);
    }
    .dataset-item {
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      margin-bottom: var(--sp-3);
      overflow: hidden;
      box-shadow: var(--shadow-sm);
    }
    .dataset-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: var(--sp-4) var(--sp-5);
      cursor: pointer;
      user-select: none;
      transition: background var(--transition);
    }
    .dataset-header:hover { background: var(--color-bg); }
    .dataset-name {
      font-family: var(--font-body);
      font-weight: 600;
      color: var(--color-heading);
      font-size: var(--text-sm);
    }
    .dataset-stats {
      display: flex;
      gap: var(--sp-4);
      font-family: var(--font-mono);
      font-size: var(--text-xs);
    }
    .dataset-stats .ds-completed { color: #006B4F; }
    .dataset-stats .ds-failed    { color: #D55E00; }
    .dataset-stats .ds-running   { color: #0B6E9E; }
    .dataset-stats .ds-pending   { color: var(--color-muted); }
    .dataset-expand {
      transition: transform 0.2s;
      color: var(--color-muted);
      font-size: var(--text-xs);
    }
    .dataset-expand.open { transform: rotate(90deg); }
    .dataset-body {
      display: none;
      padding: 0 var(--sp-5) var(--sp-4);
    }
    .dataset-body.open { display: block; }
    .dataset-progress-track {
      width: 100%;
      height: 6px;
      background: var(--color-rule);
      border-radius: 9999px;
      overflow: hidden;
      margin-top: var(--sp-2);
    }
    .dataset-progress-fill {
      height: 100%;
      border-radius: 9999px;
      background: var(--color-navy);
      transition: width 0.5s ease;
    }

    /* ── Failure Category Chart ───────────────────────────────── */
    .chart-section {
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--sp-5) var(--sp-6);
      margin-bottom: var(--sp-8);
      box-shadow: var(--shadow-sm);
    }
    .chart-section h2 {
      font-family: var(--font-display);
      font-size: var(--text-lg);
      font-weight: 400;
      color: var(--color-heading);
      margin-bottom: var(--sp-4);
    }
    .chart-row {
      display: flex;
      align-items: center;
      gap: var(--sp-3);
      margin-bottom: var(--sp-2);
    }
    .chart-label {
      flex: 0 0 180px;
      text-align: right;
      font-family: var(--font-mono);
      font-size: var(--text-xs);
      color: var(--color-body);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .chart-bar-track {
      flex: 1;
      height: 16px;
      background: var(--color-rule);
      border-radius: 9999px;
      overflow: hidden;
    }
    .chart-bar-fill {
      height: 100%;
      background: #D55E00;
      border-radius: 9999px;
      transition: width 0.5s ease;
      min-width: 2px;
    }
    .chart-count {
      flex: 0 0 48px;
      font-family: var(--font-mono);
      font-size: var(--text-xs);
      font-weight: 500;
      color: #D55E00;
      text-align: right;
    }
    .chart-empty {
      color: var(--color-muted);
      font-size: var(--text-sm);
      padding: var(--sp-3) 0;
    }

    /* ── Recent Failures Table ────────────────────────────────── */
    .failures-section {
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--sp-5) var(--sp-6);
      margin-bottom: var(--sp-8);
      box-shadow: var(--shadow-sm);
    }
    .failures-section h2 {
      font-family: var(--font-display);
      font-size: var(--text-lg);
      font-weight: 400;
      color: var(--color-heading);
      margin-bottom: var(--sp-4);
    }
    .failures-table {
      width: 100%;
      border-collapse: collapse;
    }
    .failures-table th {
      text-align: left;
      padding: 12px 16px;
      font-family: var(--font-mono);
      font-size: 0.6875rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--color-muted);
      border-bottom: 2px solid var(--color-navy);
      white-space: nowrap;
    }
    .failures-table td {
      padding: 12px 16px;
      font-size: var(--text-sm);
      border-bottom: 1px solid var(--color-rule);
      vertical-align: top;
      color: var(--color-body);
    }
    .failures-table tr:last-child td { border-bottom: none; }
    .failures-table tr:hover td { background: rgba(27,117,188,0.03); }
    .failures-table .col-ts    { color: var(--color-muted); font-family: var(--font-mono); font-size: var(--text-xs); white-space: nowrap; }
    .failures-table .col-ds    { font-weight: 500; }
    .failures-table .col-img   { font-family: var(--font-mono); font-size: var(--text-xs); }
    .failures-table .col-type  { font-family: var(--font-mono); font-size: var(--text-xs); color: #D55E00; }
    .failure-msg-toggle {
      cursor: pointer;
      color: var(--color-blue);
      font-size: var(--text-xs);
      user-select: none;
    }
    .failure-msg-toggle:hover { text-decoration: underline; }
    .failure-msg-body {
      display: none;
      margin-top: var(--sp-2);
      padding: var(--sp-3);
      background: var(--color-bg);
      border: 1px solid var(--color-border);
      border-radius: var(--radius);
      font-family: var(--font-mono);
      font-size: var(--text-xs);
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--color-body);
      max-height: 280px;
      overflow-y: auto;
    }
    .failure-msg-body.open { display: block; }
    .failures-empty {
      color: var(--color-muted);
      font-size: var(--text-sm);
      padding: var(--sp-3) 0;
    }

    /* ── Tabs ────────────────────────────────────────────────── */
    .tab-bar {
      display: flex;
      gap: var(--sp-1);
      margin-bottom: var(--sp-6);
      border-bottom: 2px solid var(--color-rule);
      padding-bottom: 0;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--color-muted);
      font-family: var(--font-body);
      font-size: var(--text-sm);
      font-weight: 500;
      padding: 12px 20px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
      transition: color var(--transition), border-color var(--transition);
    }
    .tab-btn:hover { color: var(--color-body); border-color: var(--color-border); }
    .tab-btn.active {
      color: var(--color-navy);
      border-bottom-color: var(--color-navy);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* ── Download ────────────────────────────────────────────── */
    .download-container {
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--sp-8) var(--sp-10);
      box-shadow: var(--shadow-sm);
    }
    .download-container h2 { font-family: var(--font-display); font-size: var(--text-xl); font-weight: 400; color: var(--color-heading); margin-bottom: var(--sp-4); }
    .download-note {
      display: flex;
      gap: var(--sp-4);
      background: rgba(86,180,233,0.08);
      color: #0B5E87;
      border: none;
      border-left: 4px solid #56B4E9;
      border-radius: var(--radius);
      padding: var(--sp-4) var(--sp-5);
      margin-bottom: var(--sp-6);
      font-size: var(--text-sm);
      line-height: 1.6;
    }
    .download-section-title { font-family: var(--font-body); font-weight: 600; color: var(--color-heading); margin: var(--sp-5) 0 var(--sp-2); font-size: var(--text-sm); }
    .download-cmd {
      background: var(--color-bg);
      border: 1px solid var(--color-border);
      padding: var(--sp-4) var(--sp-5);
      border-radius: var(--radius);
      font-family: var(--font-mono);
      font-size: var(--text-xs);
      color: var(--color-body);
      user-select: all;
      margin: var(--sp-2) 0 var(--sp-4);
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }
    .download-input {
      width: 100%;
      background: var(--color-white);
      border: 1.5px solid var(--color-border);
      border-radius: var(--radius);
      padding: 0.5rem 0.875rem;
      color: var(--color-body);
      font-family: var(--font-mono);
      font-size: var(--text-sm);
      margin: var(--sp-1) 0 var(--sp-3);
      transition: border-color var(--transition), box-shadow var(--transition);
    }
    .download-input:focus { outline: none; border-color: var(--color-blue); box-shadow: 0 0 0 3px rgba(27,117,188,0.12); }

    /* ── Responsive ───────────────────────────────────────────── */
    @media (max-width: 768px) {
      .cards { grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); }
      .chart-label { flex: 0 0 100px; font-size: var(--text-xs); }
      .header { flex-direction: column; align-items: flex-start; }
      .dataset-stats { gap: var(--sp-2); font-size: var(--text-xs); }
      .failures-table { font-size: var(--text-xs); }
    }

    @media (max-width: 480px) {
      .container { padding: var(--sp-3) var(--sp-3) var(--sp-10); }
      .cards { grid-template-columns: repeat(2, 1fr); gap: var(--sp-2); }
      .card { padding: var(--sp-3) var(--sp-3); }
      .card-value { font-size: var(--text-xl); }
    }

    """
    return base_css


def _build_body(execution_mode: str, logo_data_uri: str = "") -> str:
    """Return the HTML body content (no <body> tags)."""
    logo_html = (
        f'<div class="header-logo"><img src="{logo_data_uri}" alt="PhenoTypic"></div>'
        if logo_data_uri
        else ""
    )
    tab_bar = ""
    download_panel = ""
    progress_attrs = ' id="progress-panel"'
    if execution_mode == "slurm":
        progress_attrs = ' class="tab-content active" id="tab-progress"'
        tab_bar = """\
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('progress')">Progress</button>
      <button class="tab-btn" onclick="switchTab('download')">Download</button>
    </div>"""
        download_panel = """\
    <div class="tab-content" id="tab-download">
      <div class="download-container">
        <h2>Download Results</h2>
        <div class="download-note">
          Direct browser downloads are not available due to HPCC security
          policies. Use the <code>wget</code> commands below from your
          local terminal.
        </div>

        <p class="download-section-title">Server URL</p>
        <input type="text" id="dl-url" class="download-input"
               placeholder="https://your-hpcc.edu/path/to/output/"
               oninput="updateCommands()">

        <p class="download-section-title">Authentication (if required)</p>
        <div style="display:flex;gap:12px">
          <input type="text" id="dl-user" class="download-input"
                 placeholder="Username" oninput="updateCommands()">
          <input type="password" id="dl-pass" class="download-input"
                 placeholder="Password" oninput="updateCommands()">
        </div>

        <input type="hidden" id="dl-cutdirs" value="N">

        <p class="download-section-title">1. Measurements (CSV)</p>
        <p style="font-size:var(--text-sm);color:var(--color-muted)">
          Post-applied measurements with external metadata joined,
          across all datasets:</p>
        <div class="download-cmd" id="cmd-csv"></div>

        <p class="download-section-title">2. Overlay images only</p>
        <p style="font-size:var(--text-sm);color:var(--color-muted)">
          Annotated plate images from every dataset:</p>
        <div class="download-cmd" id="cmd-png"></div>

        <p class="download-section-title">3. Full output directory</p>
        <p style="font-size:var(--text-sm);color:var(--color-muted)">
          Everything (measurements, overlays, logs, checkpoints):</p>
        <div class="download-cmd" id="cmd-full"></div>
      </div>
    </div>"""

    return f"""\
  <div class="container">
    <!-- Header -->
    <div class="header">
      <div class="header-title-group">
        {logo_html}
        <h1>PhenoTypic Processing Dashboard</h1>
      </div>
      <div class="header-right">
        <span class="input-path" id="input-path" style="display:none"></span>
        <span id="last-updated"></span>
        <span id="status-badge" class="status-badge status-live">
          <span class="pulse-dot"></span> Live
        </span>
      </div>
    </div>

{tab_bar}

    <div{progress_attrs}>
      <!-- Summary cards -->
      <div class="cards" id="cards"></div>

      <!-- Overall progress bar -->
      <div class="progress-section">
        <div class="progress-header">
          <span class="progress-title">Overall Progress</span>
          <span class="progress-pct" id="progress-pct">0%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" id="progress-fill" style="width:0%"></div>
        </div>
      </div>

      <!-- Active batch / SLURM -->
      <div class="slurm-section" id="slurm-section" style="display:none">
        <h2>SLURM Batch Status</h2>
        <div class="chunk-grid" id="chunk-grid"></div>
      </div>

      <!-- Per-dataset breakdown -->
      <div class="datasets-section">
        <h2>Datasets</h2>
        <div id="datasets-list"></div>
      </div>

      <!-- Failure category chart -->
      <div class="chart-section">
        <h2>Failure Categories</h2>
        <div id="failure-chart"></div>
      </div>

      <!-- Recent failures table -->
      <div class="failures-section">
        <h2>Recent Failures</h2>
        <div id="failures-table-container"></div>
      </div>
    </div>

{download_panel}

  </div>"""


def _build_js(
    execution_mode: str,
    root_prefix: str = "../",
) -> str:
    """Return the inline JavaScript for the dashboard.

    Args:
        execution_mode: ``"local"`` or ``"slurm"``.
        root_prefix: Relative path from the generated HTML back to the
            output root. The dashboard lives in ``deliverables/`` while
            machine state lives in ``.phenotypic/progress/`` and results
            stay at the output root, so the default ``"../"`` re-roots both.
    """
    framework_js = f"""\
    // ── State ──────────────────────────────────────────────────
    const EXECUTION_MODE = "{execution_mode}";
    // Path from this HTML (in deliverables/) back to the output root, where
    // results/ lives. Machine-state sidecars use the canonical hidden cache.
    const ROOT_PREFIX = "{root_prefix}";
    const PROGRESS_PREFIX = ROOT_PREFIX + "{DIR_PHENOTYPIC}/{DIR_PROGRESS}/";
    let refreshTimer = null;
    const REFRESH_MS = 10000;
    let fetchErrors = 0;

    // ── Parent-frame messaging (PhenoTypic GUI shell) ──────────
    // The unified GUI iframes this dashboard at /runs/<rel>/dashboard.html.
    // When iframed (window.parent !== window), post a small JSON message
    // on key events so the parent shell can update its run-status badges
    // without re-fetching the manifest itself. When opened standalone
    // (file://, double-click), the guard keeps this path silent so we
    // don't spam ``console`` with cross-origin warnings.
    function postShellEvent(kind, payload) {{
      try {{
        if (window.parent === window) return;
        window.parent.postMessage(
          {{
            source: 'phenotypic-dashboard',
            kind: kind,
            payload: payload || null,
          }},
          '*'  // origin filtering is applied at the receiver
        );
      }} catch (_) {{
        // Cross-origin lockdown / other failure; silent.
      }}
    }}

    // ── Helpers ────────────────────────────────────────────────
    function esc(s) {{
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }}

    function fmtTs(iso) {{
      if (!iso) return '';
      const d = new Date(iso);
      if (isNaN(d)) return esc(iso);
      return d.toLocaleString();
    }}

    function shortTs(iso) {{
      if (!iso) return '';
      const d = new Date(iso);
      if (isNaN(d)) return esc(iso);
      const hh = String(d.getHours()).padStart(2, '0');
      const mm = String(d.getMinutes()).padStart(2, '0');
      const ss = String(d.getSeconds()).padStart(2, '0');
      return hh + ':' + mm + ':' + ss;
    }}

    // ── Tab Switching ──────────────────────────────────────────
    function switchTab(tabId) {{
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      document.getElementById('tab-' + tabId).classList.add('active');
      document.querySelector('[onclick*="' + tabId + '"]').classList.add('active');
    }}

    // ── Download URL Helpers ───────────────────────────────────
    // `base` is the OUTPUT ROOT, even though this dashboard now lives in
    // deliverables/. The HTML's own directory is <...>/deliverables/, but the
    // wget commands target results/ (at the root) and the whole output tree,
    // so we strip a trailing "deliverables/" segment to recover the root.
    function stripDeliverables(dir) {{
      // dir always ends with a trailing slash; drop a trailing
      // "deliverables/" segment if present so base = output root.
      return dir.replace(/deliverables\\/$/, '');
    }}

    function getBaseUrl() {{
      if (window.location.protocol === 'file:') return '';
      const path = window.location.pathname;
      const dir = path.substring(0, path.lastIndexOf('/') + 1);
      return window.location.origin + stripDeliverables(dir);
    }}

    function getCutDirs() {{
      if (window.location.protocol === 'file:') return 'N';
      const path = window.location.pathname;
      const dir = stripDeliverables(path.substring(0, path.lastIndexOf('/') + 1));
      return String(dir.split('/').filter(Boolean).length);
    }}

    function updateCommands() {{
      const base = document.getElementById('dl-url').value || '<YOUR_SERVER_URL>/path/to/output/';
      const user = document.getElementById('dl-user').value;
      const pass = document.getElementById('dl-pass').value;
      const cutDirs = document.getElementById('dl-cutdirs').value || 'N';

      let auth = '';
      if (user) auth = ' --user=' + user + (pass ? " --password='" + pass + "'" : '');

      // base = output root; measurements.csv lives in deliverables/ now.
      document.getElementById('cmd-csv').textContent =
        'wget' + auth + ' ' + base + 'deliverables/measurements.csv';
      document.getElementById('cmd-png').textContent =
        'wget -r -np -nH -e robots=off --cut-dirs=' + cutDirs + ' -A "*.png"' + auth + ' ' + base + 'deliverables/overlays/';
      document.getElementById('cmd-full').textContent =
        'wget -r -np -nH -e robots=off --cut-dirs=' + cutDirs + auth + ' ' + base;
    }}

    // ── Render: Summary Cards ──────────────────────────────────
    function renderSummary(data) {{
      const cards = [
        {{ label: 'Total',     value: data.total_images ?? 0, cls: '' }},
        {{ label: 'Completed', value: data.completed ?? 0,    cls: 'completed' }},
        {{ label: 'Failed',    value: data.failed ?? 0,       cls: 'failed' }},
        {{ label: 'In Progress', value: data.started ?? 0,    cls: 'running' }},
        {{ label: 'Pending',   value: data.pending ?? 0,      cls: 'pending' }},
      ];
      const el = document.getElementById('cards');
      el.innerHTML = cards.map(c =>
        '<div class="card ' + c.cls + '">' +
          '<div class="card-value">' + c.value.toLocaleString() + '</div>' +
          '<div class="card-label">' + c.label + '</div>' +
        '</div>'
      ).join('');
    }}

    // ── Render: Overall Progress ───────────────────────────────
    function renderProgress(data) {{
      const total = data.total_images || 1;
      const done  = (data.completed ?? 0) + (data.failed ?? 0);
      const pct   = Math.min(100, (done / total) * 100);
      const fill  = document.getElementById('progress-fill');
      fill.style.width = pct.toFixed(1) + '%';
      if (data.is_complete) {{
        fill.classList.add('complete');
      }} else {{
        fill.classList.remove('complete');
      }}
      document.getElementById('progress-pct').textContent = pct.toFixed(1) + '%';
    }}

    // ── Render: SLURM Chunks ───────────────────────────────────
    function renderSlurm(data) {{
      const info = data.slurm_info;
      const section = document.getElementById('slurm-section');
      if (!info || data.execution_mode !== 'slurm') {{
        section.style.display = 'none';
        return;
      }}
      section.style.display = '';

      const totalChunks = info.total_chunks || 0;
      const active    = new Set(info.active_chunks    || []);
      const completed = new Set(info.completed_chunks || []);
      const pending   = new Set(info.pending_chunks   || []);
      const jobIds    = info.chunk_job_ids || {{}};

      let html = '';
      for (let i = 0; i < totalChunks; i++) {{
        let cls, statusText;
        if (active.has(i))         {{ cls = 'chunk-active';    statusText = 'running'; }}
        else if (completed.has(i)) {{ cls = 'chunk-completed'; statusText = 'done'; }}
        else if (pending.has(i))   {{ cls = 'chunk-pending';   statusText = 'pending'; }}
        else                       {{ cls = 'chunk-pending';   statusText = 'unknown'; }}

        const jid = jobIds[String(i)];
        const idLabel = jid ? ' <span class="chunk-label">(' + esc(String(jid)) + ')</span>' : '';
        html += '<span class="chunk-badge ' + cls + '">' +
                  'Chunk ' + i + idLabel +
                '</span>';
      }}
      document.getElementById('chunk-grid').innerHTML = html;
    }}

    // ── Render: Datasets ───────────────────────────────────────
    function renderDatasets(datasets) {{
      const container = document.getElementById('datasets-list');
      if (!datasets || Object.keys(datasets).length === 0) {{
        container.innerHTML = '<div style="color:var(--color-muted);font-size:var(--text-sm)">No dataset information available.</div>';
        return;
      }}
      let html = '';
      for (const [name, ds] of Object.entries(datasets)) {{
        const total     = ds.total || 1;
        const completed = ds.completed || 0;
        const failed    = ds.failed || 0;
        const started   = ds.started || 0;
        const pending   = ds.pending || 0;
        const pct       = ((completed / total) * 100).toFixed(1);
        const uid       = 'ds-' + name.replace(/[^a-zA-Z0-9_-]/g, '_');

        html += '<div class="dataset-item">' +
          '<div class="dataset-header" onclick="toggleDataset(\\'' + uid + '\\')">' +
            '<div style="display:flex;align-items:center;gap:10px">' +
              '<span class="dataset-expand" id="' + uid + '-arrow">&#9654;</span>' +
              '<span class="dataset-name">' + esc(name) + '</span>' +
            '</div>' +
            '<div class="dataset-stats">' +
              '<span class="ds-completed">' + completed + ' done</span>' +
              '<span class="ds-failed">' + failed + ' fail</span>' +
              '<span class="ds-running">' + started + ' run</span>' +
              '<span class="ds-pending">' + pending + ' wait</span>' +
            '</div>' +
          '</div>' +
          '<div class="dataset-body" id="' + uid + '-body">' +
            '<div style="font-size:var(--text-sm);color:var(--color-muted);margin-bottom:4px">' +
              pct + '% complete (' + completed + ' / ' + total + ')' +
            '</div>' +
            '<div class="dataset-progress-track">' +
              '<div class="dataset-progress-fill" style="width:' + pct + '%"></div>' +
            '</div>' +
          '</div>' +
        '</div>';
      }}
      container.innerHTML = html;
    }}

    function toggleDataset(uid) {{
      const body  = document.getElementById(uid + '-body');
      const arrow = document.getElementById(uid + '-arrow');
      if (body.classList.contains('open')) {{
        body.classList.remove('open');
        arrow.classList.remove('open');
      }} else {{
        body.classList.add('open');
        arrow.classList.add('open');
      }}
    }}

    // ── Render: Failure Category Chart ─────────────────────────
    function renderFailureChart(categories) {{
      const el = document.getElementById('failure-chart');
      if (!categories || Object.keys(categories).length === 0) {{
        el.innerHTML = '<div class="chart-empty">No failures recorded.</div>';
        return;
      }}
      const entries = Object.entries(categories).sort((a, b) => b[1] - a[1]);
      const maxVal  = entries[0][1] || 1;

      let html = '';
      for (const [cat, count] of entries) {{
        const widthPct = ((count / maxVal) * 100).toFixed(1);
        html += '<div class="chart-row">' +
          '<span class="chart-label" title="' + esc(cat) + '">' + esc(cat) + '</span>' +
          '<div class="chart-bar-track">' +
            '<div class="chart-bar-fill" style="width:' + widthPct + '%"></div>' +
          '</div>' +
          '<span class="chart-count">' + count + '</span>' +
        '</div>';
      }}
      el.innerHTML = html;
    }}

    // ── Render: Recent Failures Table ──────────────────────────
    async function renderRecentFailures(data) {{
      const container = document.getElementById('failures-table-container');
      let records = [];
      try {{
        const resp = await fetch(PROGRESS_PREFIX + 'failures.jsonl?' + Date.now());
        if (resp.ok) {{
          const text = await resp.text();
          const lines = text.trim().split('\\n').filter(Boolean);
          for (const line of lines) {{
            try {{ records.push(JSON.parse(line)); }} catch(e) {{ /* skip malformed */ }}
          }}
        }}
      }} catch(e) {{ /* no failures file yet */ }}

      // Show only last 50
      const recent = records.slice(-50).reverse();

      if (recent.length === 0) {{
        container.innerHTML = '<div class="failures-empty">No failures recorded.</div>';
        return;
      }}

      let html = '<table class="failures-table"><thead><tr>' +
        '<th>Time</th><th>Dataset</th><th>Image</th><th>Error Type</th><th>Details</th>' +
        '</tr></thead><tbody>';

      for (let i = 0; i < recent.length; i++) {{
        const r = recent[i];
        const msgId = 'fail-msg-' + i;
        const hasMsg = r.error_message || r.traceback;
        const msgContent = (r.error_message || '') +
          (r.traceback ? '\\n\\n' + r.traceback : '');

        html += '<tr>' +
          '<td class="col-ts">' + shortTs(r.timestamp) + '</td>' +
          '<td class="col-ds">' + esc(r.dataset || '') + '</td>' +
          '<td class="col-img">' + esc(r.image || '') + '</td>' +
          '<td class="col-type">' + esc(r.error_type || '') + '</td>' +
          '<td>' +
            (hasMsg
              ? '<span class="failure-msg-toggle" onclick="toggleFailMsg(\\'' + msgId + '\\')">show details</span>' +
                '<div class="failure-msg-body" id="' + msgId + '">' + esc(msgContent) + '</div>'
              : '<span style="color:var(--color-muted)">-</span>') +
          '</td>' +
        '</tr>';
      }}
      html += '</tbody></table>';
      container.innerHTML = html;
    }}

    function toggleFailMsg(id) {{
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.toggle('open');
      const toggle = el.previousElementSibling;
      if (toggle) {{
        toggle.textContent = el.classList.contains('open') ? 'hide details' : 'show details';
      }}
    }}

    // ── Fetch Error Display ────────────────────────────────────
    function showFetchError(reason) {{
      const badge = document.getElementById('status-badge');
      badge.className = 'status-badge status-error';
      badge.innerHTML = '&#9888; Data unavailable';
      let hint = document.getElementById('fetch-error-hint');
      if (!hint) {{
        hint = document.createElement('div');
        hint.id = 'fetch-error-hint';
        hint.style.cssText = 'padding:12px 16px;background:rgba(213,94,0,0.08);color:#D55E00;' +
          'border-radius:6px;margin-bottom:16px;font-size:var(--text-sm)';
        const container = document.getElementById('tab-progress') ||
          document.getElementById('progress-panel');
        if (container) container.prepend(hint);
      }}
      hint.innerHTML = 'Cannot load <code>' + PROGRESS_PREFIX + 'manifest.json</code> (' + esc(String(reason)) +
        '). Verify the progress directory exists and is accessible via this web server.' +
        '<br><small style="opacity:0.7">Retrying every ' + (REFRESH_MS/1000) + 's...</small>';
    }}

    function clearFetchError() {{
      const badge = document.getElementById('status-badge');
      if (badge.classList.contains('status-error')) {{
        badge.className = 'status-badge status-live';
        badge.innerHTML = '<span class="pulse-dot"></span> Live';
      }}
      const hint = document.getElementById('fetch-error-hint');
      if (hint) hint.remove();
    }}

    // ── Main Refresh Loop ──────────────────────────────────────
    async function refresh() {{
      try {{
        const resp = await fetch(PROGRESS_PREFIX + 'manifest.json?' + Date.now());
        if (!resp.ok) {{
          fetchErrors++;
          console.warn('Dashboard: manifest fetch HTTP ' + resp.status);
          if (fetchErrors >= 3) showFetchError('HTTP ' + resp.status);
          return;
        }}
        fetchErrors = 0;
        clearFetchError();
        const data = await resp.json();

        renderSummary(data);
        renderProgress(data);
        renderSlurm(data);
        renderDatasets(data.datasets);
        renderFailureChart(data.failure_categories);
        await renderRecentFailures(data);

        const inputEl = document.getElementById('input-path');
        if (data.input_path) {{
          inputEl.textContent = data.input_path;
          inputEl.style.display = '';
        }}

        document.getElementById('last-updated').textContent =
          'Updated ' + new Date().toLocaleTimeString();

        // Tell the parent shell about the latest manifest snapshot. The
        // shell can use this to live-update badges in the Recent Runs
        // panel without re-fetching the manifest itself.
        postShellEvent('manifest', {{
          completed: data.completed,
          failed: data.failed,
          total: data.total_images,
          is_complete: !!data.is_complete,
          last_updated: data.last_updated,
        }});

        if (data.is_complete) {{
          stopRefresh();
          showComplete();
          postShellEvent('complete', {{
            failed: data.failed,
          }});
        }}
      }} catch(e) {{
        fetchErrors++;
        console.warn('Dashboard: fetch error:', e.message || e);
        if (fetchErrors >= 3) showFetchError(e.message || 'network error');
      }}
    }}

    function stopRefresh() {{
      if (refreshTimer) {{
        clearInterval(refreshTimer);
        refreshTimer = null;
      }}
    }}

    function showComplete() {{
      const badge = document.getElementById('status-badge');
      badge.className = 'status-badge status-complete';
      badge.innerHTML = '&#10003; Complete';
      const fill = document.getElementById('progress-fill');
      fill.classList.add('complete');
    }}

    // ── Boot ───────────────────────────────────────────────────
    refresh();
    refreshTimer = setInterval(refresh, REFRESH_MS);

    if (EXECUTION_MODE === 'slurm') {{
      // Auto-populate the SLURM download helper from window.location.
      const detected = getBaseUrl();
      if (detected) {{
        document.getElementById('dl-url').value = detected;
        document.getElementById('dl-cutdirs').value = getCutDirs();
      }}
      updateCommands();
    }}

"""
    if execution_mode != "slurm":
        helpers_start = framework_js.index("    // ── Tab Switching")
        helpers_end = framework_js.index("    // ── Render: Summary Cards")
        framework_js = (
            framework_js[:helpers_start] + framework_js[helpers_end:]
        )
        boot_start = framework_js.index(
            "    if (EXECUTION_MODE === 'slurm')"
        )
        framework_js = framework_js[:boot_start]
    return framework_js
