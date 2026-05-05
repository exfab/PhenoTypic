"""
Dashboard HTML generator for PhenoTypic CLI processing progress.

Generates a single self-contained HTML+CSS+JS dashboard file that monitors
processing progress by polling ``manifest.json`` and ``failures.jsonl`` in
the ``progress/`` subdirectory.
"""

from __future__ import annotations

import base64
import logging
from importlib.resources import files
from pathlib import Path

from phenotypic.tools_.register import AnalysisPluginRegistry

from ._vendor_js import MARKED_MIN_JS

logger = logging.getLogger(__name__)


def _get_analysis_plugins() -> list:
    """Return registered analysis plugins sorted by sort_order."""
    # Import triggers plugin registration
    from . import _analysis  # noqa: F401

    plugins = []
    for name in AnalysisPluginRegistry.available():
        try:
            cls = AnalysisPluginRegistry.get(name)
            plugins.append(cls())
        except Exception:
            logger.warning("Failed to instantiate analysis plugin %r", name)
    plugins.sort(key=lambda p: p.sort_order)
    return plugins


def _build_analysis_subtabs(plugins: list) -> str:
    """Build analysis sub-tab HTML from registered plugins."""
    if not plugins:
        return '<div class="analysis-empty">No analysis plugins available.</div>'

    # Sub-tab buttons
    buttons = []
    for i, p in enumerate(plugins):
        active = " active" if i == 0 else ""
        buttons.append(
                f'<button class="sub-tab-btn{active}" '
                f"onclick=\"switchSubTab('{p.call_name}')\">{p.display_name}</button>"
        )

    # Sub-tab content panels
    panels = []
    for i, p in enumerate(plugins):
        active = " active" if i == 0 else ""
        panels.append(
                f'<div class="sub-tab-content{active}" id="subtab-{p.call_name}">'
                f"{p.html()}</div>"
        )

    return (
            '<div class="analysis-sub-tabs">\n          '
            + "\n          ".join(buttons)
            + "\n        </div>\n        "
            + "\n        ".join(panels)
    )


def generate_dashboard(output_dir: Path, *, execution_mode: str = "local") -> None:
    """Write ``dashboard.html`` and ``analysis.html`` to the output directory root.

    Args:
        output_dir: Root output directory.
        execution_mode: ``"local"`` or ``"slurm"``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(_build_html(execution_mode), encoding="utf-8")
    logger.info("Dashboard written to %s", dashboard_path)

    analysis_path = output_dir / "analysis.html"
    analysis_path.write_text(_build_analysis_html(), encoding="utf-8")
    logger.info("Analysis page written to %s", analysis_path)

    # Write JS sidecars for lazy loading by the Analysis page
    _write_js_sidecar(output_dir, "plotly.min.js", "Plotly.js")
    _write_js_sidecar(output_dir, "hyparquet.min.js", "hyparquet.js")


def _write_js_sidecar(output_dir: Path, filename: str, label: str) -> None:
    """Copy a vendored JS asset to the progress directory as a sidecar file.

    Args:
        output_dir: Root output directory.
        filename: Asset filename (e.g. ``"plotly.min.js"``).
        label: Human-readable name for log messages (e.g. ``"Plotly.js"``).
    """
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    dest = progress_dir / filename
    try:
        src = files("phenotypic._cli._dashboard").joinpath("_assets", filename)
        src_data = src.read_bytes()
        if dest.exists() and dest.stat().st_size == len(src_data):
            return
        dest.write_bytes(src_data)
        logger.debug("%s sidecar written to %s", label, dest)
    except (OSError, ModuleNotFoundError, TypeError):
        logger.debug("%s asset not found -- feature will not be available", label)


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
        f"  <style>\n{_build_css(None)}\n  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{_build_body(execution_mode, logo_data_uri, None)}\n"
        f"  <script>\n{MARKED_MIN_JS}\n  </script>\n"
        f"  <script>\n{_build_js(execution_mode, None)}\n  </script>\n"
        "</body>\n"
        "</html>\n"
    )


def _build_analysis_html() -> str:
    """Assemble a self-contained analysis HTML page."""
    logo_data_uri = _load_logo_data_uri()
    plugins = _get_analysis_plugins()
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "  <title>PhenoTypic Analysis</title>\n"
        "  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n"
        "  <link href=\"https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap\" rel=\"stylesheet\">\n"
        f"  <style>\n{_build_css(plugins)}\n  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{_build_analysis_body(logo_data_uri, plugins)}\n"
        f"  <script>\n{_build_analysis_js(plugins)}\n  </script>\n"
        "</body>\n"
        "</html>\n"
    )


def _load_logo_data_uri() -> str:
    """Read the logo PNG and return a ``data:`` URI, or empty string on failure."""
    try:
        raw = (
            files("phenotypic._cli._dashboard")
            .joinpath("_assets", "LogoArtOnly.png")
            .read_bytes()
        )
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except (OSError, ModuleNotFoundError, TypeError):
        logger.debug("Logo asset not found — skipping")
        return ""


def _build_css(plugins: list | None = None) -> str:
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

    /* ── README ──────────────────────────────────────────────── */
    .readme-container {
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--sp-8) var(--sp-10);
      line-height: 1.7;
      color: var(--color-body);
      box-shadow: var(--shadow-sm);
    }
    .readme-container h1 { font-family: var(--font-display); font-size: var(--text-2xl); font-weight: 400; color: var(--color-heading); margin: var(--sp-6) 0 var(--sp-3); border-bottom: 1px solid var(--color-rule); padding-bottom: var(--sp-2); }
    .readme-container h2 { font-family: var(--font-display); font-size: var(--text-xl); font-weight: 400; color: var(--color-heading); margin: var(--sp-5) 0 var(--sp-3); }
    .readme-container h3 { font-family: var(--font-display); font-size: var(--text-md); font-weight: 400; color: var(--color-heading); margin: var(--sp-4) 0 var(--sp-2); }
    .readme-container code { background: #edf2f7; color: var(--color-navy); padding: 1px 5px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.88em; }
    .readme-container pre { background: var(--color-bg); border: 1px solid var(--color-border); padding: var(--sp-4); border-radius: var(--radius); overflow-x: auto; margin: var(--sp-3) 0; }
    .readme-container pre code { background: none; padding: 0; }
    .readme-container table { border-collapse: collapse; width: 100%; margin: var(--sp-3) 0; }
    .readme-container th, .readme-container td { border: 1px solid var(--color-rule); padding: var(--sp-2) var(--sp-3); text-align: left; }
    .readme-container th { background: var(--color-bg); font-family: var(--font-mono); font-size: var(--text-xs); font-weight: 500; color: var(--color-heading); text-transform: uppercase; letter-spacing: 0.08em; }
    .readme-loading { color: var(--color-muted); font-size: var(--text-sm); }

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

    /* ── Analysis Tab ─────────────────────────────────────────── */
    .analysis-container {
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--sp-5) var(--sp-6);
      box-shadow: var(--shadow-sm);
    }
    .analysis-banner {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: var(--sp-3) var(--sp-5);
      background: rgba(86,180,233,0.08);
      border-left: 4px solid #56B4E9;
      border-radius: var(--radius);
      margin-bottom: var(--sp-4);
      font-size: var(--text-sm);
      color: #0B5E87;
    }
    .analysis-banner-btn {
      background: var(--color-navy);
      color: #fff;
      border: none;
      border-radius: var(--radius);
      padding: 0.3rem 0.75rem;
      font-family: var(--font-body);
      font-size: 11px;
      font-weight: 500;
      cursor: pointer;
      transition: background var(--transition);
    }
    .analysis-banner-btn:hover { background: #004a8a; }
    .analysis-sub-tabs {
      display: flex;
      gap: var(--sp-1);
      margin-bottom: var(--sp-5);
      border-bottom: 2px solid var(--color-rule);
    }
    .sub-tab-btn {
      background: none;
      border: none;
      color: var(--color-muted);
      font-family: var(--font-body);
      font-size: var(--text-sm);
      font-weight: 500;
      padding: 10px 16px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
      transition: color var(--transition), border-color var(--transition);
    }
    .sub-tab-btn:hover { color: var(--color-body); border-color: var(--color-border); }
    .sub-tab-btn.active { color: var(--color-navy); border-bottom-color: var(--color-navy); }
    .sub-tab-content { display: none; }
    .sub-tab-content.active { display: block; }
    .analysis-empty {
      color: var(--color-muted);
      font-size: var(--text-sm);
      padding: var(--sp-8) 0;
      text-align: center;
    }
    .analysis-sample-label {
      font-family: var(--font-mono);
      font-size: var(--text-xs);
      color: var(--color-muted);
      margin-bottom: var(--sp-3);
    }"""

    # Append plugin CSS
    plugin_css = "\n".join(p.css() for p in (plugins or []))
    return base_css + "\n" + plugin_css


def _build_body(execution_mode: str, logo_data_uri: str = "",
                plugins: list | None = None) -> str:
    """Return the HTML body content (no <body> tags)."""
    logo_html = (
        f'<div class="header-logo"><img src="{logo_data_uri}" alt="PhenoTypic"></div>'
        if logo_data_uri
        else ""
    )
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

    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('progress')">Progress</button>
      <a href="analysis.html" class="tab-btn" style="text-decoration:none">Analysis</a>
      <button class="tab-btn" onclick="switchTab('readme')">README</button>
      <button class="tab-btn" id="download-tab-btn" onclick="switchTab('download')"
              style="display:none">Download</button>
    </div>

    <div class="tab-content active" id="tab-progress">
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

    <div class="tab-content" id="tab-readme">
      <div class="readme-container" id="readme-content">
        <div class="readme-loading">Loading README...</div>
      </div>
    </div>

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

        <p class="download-section-title">1. Master measurements (CSV)</p>
        <p style="font-size:var(--text-sm);color:var(--color-muted)">
          Single file with all colony measurements across datasets:</p>
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
    </div>

  </div>"""


def _build_js(execution_mode: str, plugins: list | None = None) -> str:
    """Return the inline JavaScript for the dashboard."""
    framework_js = f"""\
    // ── State ──────────────────────────────────────────────────
    const EXECUTION_MODE = "{execution_mode}";
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
    let readmeLoaded = false;
    function switchTab(tabId) {{
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      document.getElementById('tab-' + tabId).classList.add('active');
      document.querySelector('[onclick*="' + tabId + '"]').classList.add('active');
      if (tabId === 'readme' && !readmeLoaded) loadReadme();
    }}

    // ── README Loading ─────────────────────────────────────────
    async function loadReadme() {{
      try {{
        const resp = await fetch('README.md?' + Date.now());
        if (!resp.ok) {{
          document.getElementById('readme-content').innerHTML =
            '<div class="readme-loading">README.md not found.</div>';
          return;
        }}
        const md = await resp.text();
        document.getElementById('readme-content').innerHTML = marked.parse(md);
        readmeLoaded = true;
      }} catch(e) {{
        document.getElementById('readme-content').innerHTML =
          '<div class="readme-loading">Could not load README.md.</div>';
      }}
    }}

    // ── Download URL Helpers ───────────────────────────────────
    function getBaseUrl() {{
      if (window.location.protocol === 'file:') return '';
      const path = window.location.pathname;
      const dir = path.substring(0, path.lastIndexOf('/') + 1);
      return window.location.origin + dir;
    }}

    function getCutDirs() {{
      if (window.location.protocol === 'file:') return 'N';
      const path = window.location.pathname;
      const dir = path.substring(0, path.lastIndexOf('/') + 1);
      return String(dir.split('/').filter(Boolean).length);
    }}

    function updateCommands() {{
      const base = document.getElementById('dl-url').value || '<YOUR_SERVER_URL>/path/to/output/';
      const user = document.getElementById('dl-user').value;
      const pass = document.getElementById('dl-pass').value;
      const cutDirs = document.getElementById('dl-cutdirs').value || 'N';

      let auth = '';
      if (user) auth = ' --user=' + user + (pass ? " --password='" + pass + "'" : '');

      document.getElementById('cmd-csv').textContent =
        'wget' + auth + ' ' + base + 'master_measurements.csv';
      document.getElementById('cmd-png').textContent =
        'wget -r -np -nH -e robots=off --cut-dirs=' + cutDirs + ' -A "*.png"' + auth + ' ' + base + 'results/';
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
        const resp = await fetch('progress/failures.jsonl?' + Date.now());
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
        const container = document.querySelector('.tab-content.active');
        if (container) container.prepend(hint);
      }}
      hint.innerHTML = 'Cannot load <code>progress/manifest.json</code> (' + esc(String(reason)) +
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
        const resp = await fetch('progress/manifest.json?' + Date.now());
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

    // Show Download tab if SLURM mode
    if (EXECUTION_MODE === 'slurm') {{
      const dlBtn = document.getElementById('download-tab-btn');
      if (dlBtn) dlBtn.style.display = '';
    }}

    // Auto-populate URL from window.location
    const detected = getBaseUrl();
    if (detected) {{
      document.getElementById('dl-url').value = detected;
      document.getElementById('dl-cutdirs').value = getCutDirs();
    }}
    updateCommands();

"""
    # Append plugin JS
    plugin_js = "\n".join(p.js() for p in (plugins or []))
    return framework_js + "\n" + plugin_js


def _build_analysis_body(logo_data_uri: str, plugins: list) -> str:
    """Return the HTML body content for the analysis page."""
    logo_html = (
        f'<div class="header-logo"><img src="{logo_data_uri}" alt="PhenoTypic"></div>'
        if logo_data_uri
        else ""
    )
    return f"""\
  <div class="container">
    <div class="header">
      <div class="header-title-group">
        {logo_html}
        <h1>PhenoTypic Analysis</h1>
      </div>
      <div class="header-right">
        <a href="dashboard.html" class="tab-btn" style="text-decoration:none">&larr; Dashboard</a>
        <span id="last-updated"></span>
      </div>
    </div>
    <div class="analysis-container">
      <div class="analysis-banner" id="analysis-banner" style="display:none">
        <span>New data available</span>
        <button class="analysis-banner-btn" onclick="refreshAnalysisData()">Refresh</button>
      </div>
      {_build_analysis_subtabs(plugins)}
    </div>
  </div>"""


def _build_analysis_js(plugins: list) -> str:
    """Return the inline JavaScript for the analysis page."""
    framework_js = """\
    // ── Helpers ────────────────────────────────────────────────
    function esc(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    // ── Analysis State ────────────────────────────────────────
    let analysisData = {};
    let analysisDataVersion = null;
    let analysisInitialized = {};
    let sharedParquetState = {
      allData: {},
      allColumns: [],
      numericCols: [],
      catCols: [],
      nRows: 0,
      loaded: false,
      loading: null
    };
    const _scriptCache = {};

    function loadScript(src, globalName) {
      if (_scriptCache[src]) return _scriptCache[src];
      _scriptCache[src] = new Promise((resolve, reject) => {
        if (window[globalName]) { resolve(); return; }
        const s = document.createElement('script');
        s.src = src;
        s.onload = resolve;
        s.onerror = () => reject(new Error('Failed to load ' + src));
        document.head.appendChild(s);
      });
      return _scriptCache[src];
    }
    function loadPlotly() { return loadScript('progress/plotly.min.js', 'Plotly'); }
    function loadHyparquet() { return loadScript('progress/hyparquet.min.js', 'hyparquet'); }

    function _appendParquetRows(rows) {
      if (rows.length === 0) return;
      var cols = Object.keys(rows[0]);
      for (var ci = 0; ci < cols.length; ci++) {
        var col = cols[ci];
        if (sharedParquetState.allColumns.indexOf(col) < 0) {
          sharedParquetState.allColumns.push(col);
        }
        var existing = sharedParquetState.allData[col] || [];
        var newVals = [];
        for (var ri = 0; ri < rows.length; ri++) {
          newVals.push(rows[ri][col]);
        }
        sharedParquetState.allData[col] = existing.concat(newVals);
      }
      sharedParquetState.nRows += rows.length;
      sharedParquetState.numericCols = [];
      sharedParquetState.catCols = [];
      for (var ni = 0; ni < sharedParquetState.allColumns.length; ni++) {
        var c = sharedParquetState.allColumns[ni];
        var sample = sharedParquetState.allData[c];
        if (!sample || sample.length === 0) continue;
        var isNumeric = false;
        for (var si = 0; si < Math.min(sample.length, 10); si++) {
          if (sample[si] !== null && sample[si] !== undefined && !isNaN(sample[si]) && typeof sample[si] === 'number') {
            isNumeric = true;
            break;
          }
        }
        if (isNumeric) {
          sharedParquetState.numericCols.push(c);
        } else {
          sharedParquetState.catCols.push(c);
        }
      }
    }

    function _loadParquetFile(url) {
      return fetch(url + '?' + Date.now()).then(function(resp) {
        if (!resp.ok) throw new Error('Failed to fetch ' + url);
        return resp.arrayBuffer();
      }).then(function(buf) {
        return new Promise(function(resolve, reject) {
          try {
            hyparquet.parquetRead({
              file: {
                byteLength: buf.byteLength,
                slice: function(start, end) { return buf.slice(start, end); }
              },
              onComplete: function(rows) {
                _appendParquetRows(rows);
                resolve();
              }
            });
          } catch(ex) { reject(ex); }
        });
      });
    }

    function loadSharedParquet() {
      if (sharedParquetState.loaded) return Promise.resolve();
      if (sharedParquetState.loading) return sharedParquetState.loading;
      sharedParquetState.loading = _loadParquetFile('master_measurements.parquet')
        .then(function() {
          sharedParquetState.loaded = true;
          sharedParquetState.loading = null;
        })
        .catch(function() {
          return _loadParquetFile('progress/analysis_full.parquet')
            .then(function() {
              sharedParquetState.loaded = true;
              sharedParquetState.loading = null;
            });
        })
        .catch(function(e) {
          console.warn('Parquet loading failed:', e);
          sharedParquetState.loading = null;
          throw e;
        });
      return sharedParquetState.loading;
    }

    function switchSubTab(tabId) {
      document.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.sub-tab-content').forEach(c => c.classList.remove('active'));
      document.getElementById('subtab-' + tabId).classList.add('active');
      document.querySelectorAll('.sub-tab-btn').forEach(b => {
        if (b.getAttribute('onclick') && b.getAttribute('onclick').includes("'" + tabId + "'")) b.classList.add('active');
      });
      if (!analysisInitialized[tabId]) {
        initSubTab(tabId);
      }
    }

    async function initSubTab(tabId) {
      await fetchAnalysisData();
      await Promise.allSettled([loadPlotly(), loadHyparquet()]);
      await loadSharedParquet();
      analysisInitialized[tabId] = true;
      renderSubTab(tabId);
    }

    async function fetchAnalysisData() {
      const files = ['analysis_stats.json', 'overlay_manifest.json'];
      const keys = ['stats', 'overlay'];
      for (let i = 0; i < files.length; i++) {
        try {
          const resp = await fetch('progress/' + files[i] + '?' + Date.now());
          if (resp.ok) analysisData[keys[i]] = await resp.json();
        } catch(e) { /* file not ready yet */ }
      }
    }

    async function refreshAnalysisData() {
      sharedParquetState = {allData:{}, allColumns:[], numericCols:[], catCols:[], nRows:0, loaded:false, loading:null};
      await fetchAnalysisData();
      analysisInitialized = {};
      const active = document.querySelector('.sub-tab-content.active');
      if (active) {
        const tabId = active.id.replace('subtab-', '');
        await initSubTab(tabId);
      }
      document.getElementById('analysis-banner').style.display = 'none';
    }

    function renderSubTab(tabId) {
      var fn = window['initAnalysis_' + tabId];
      if (typeof fn === 'function') fn();
    }

    // ── Auto-refresh polling ──────────────────────────────────
    let _refreshTimer = null;
    async function _pollManifest() {
      try {
        const resp = await fetch('progress/manifest.json?' + Date.now());
        if (!resp.ok) return;
        const data = await resp.json();
        const newVersion = data.analysis_data_version || 0;
        if (analysisDataVersion !== null && newVersion > analysisDataVersion) {
          document.getElementById('analysis-banner').style.display = '';
        }
        analysisDataVersion = newVersion;
        document.getElementById('last-updated').textContent =
          'Updated ' + new Date().toLocaleTimeString();
      } catch(e) { /* ignore */ }
    }

    // ── Boot ──────────────────────────────────────────────────
    (async function() {
      await _pollManifest();
      _refreshTimer = setInterval(_pollManifest, 10000);
      const activeSubTab = document.querySelector('.sub-tab-content.active');
      if (activeSubTab) {
        const subId = activeSubTab.id.replace('subtab-', '');
        initSubTab(subId);
      }
    })();
"""
    # Shared transform helpers available to all analysis plugins
    transform_js = """\

    // ── Stat Helpers ────────────────────────────────────────────
    // Basic statistical functions used by transform helpers and plugins.

    function isNum(v) {
      return v !== null && v !== undefined && !isNaN(v);
    }

    function statCount(arr) {
      var c = 0;
      for (var i = 0; i < arr.length; i++) {
        if (isNum(arr[i])) c++;
      }
      return c;
    }

    function statMean(arr) {
      var s = 0, c = 0;
      for (var i = 0; i < arr.length; i++) {
        if (isNum(arr[i])) { s += arr[i]; c++; }
      }
      return c > 0 ? s / c : null;
    }

    function statStd(arr, mean) {
      if (mean === null) return null;
      var s = 0, c = 0;
      for (var i = 0; i < arr.length; i++) {
        if (isNum(arr[i])) { s += (arr[i] - mean) * (arr[i] - mean); c++; }
      }
      return c > 1 ? Math.sqrt(s / (c - 1)) : null;
    }

    function statMedian(arr) {
      var clean = [];
      for (var i = 0; i < arr.length; i++) {
        if (isNum(arr[i])) clean.push(arr[i]);
      }
      if (clean.length === 0) return null;
      clean.sort(function(a, b) { return a - b; });
      var mid = Math.floor(clean.length / 2);
      if (clean.length % 2 !== 0) return clean[mid];
      return (clean[mid - 1] + clean[mid]) / 2;
    }

    function statMin(arr) {
      var m = Infinity;
      for (var i = 0; i < arr.length; i++) {
        if (isNum(arr[i]) && arr[i] < m) m = arr[i];
      }
      return m === Infinity ? null : m;
    }

    function statMax(arr) {
      var m = -Infinity;
      for (var i = 0; i < arr.length; i++) {
        if (isNum(arr[i]) && arr[i] > m) m = arr[i];
      }
      return m === -Infinity ? null : m;
    }

    // ── Transform Helpers ────────────────────────────────────────
    // Shared data transform functions for analysis plugins.
    // All accept `data` (column-name → value-array, e.g. statsState.allData)
    // and `indices` (filtered row indices from getFilteredIndices()).

    /**
     * Filter indices to rows where data[field][i] === value.
     */
    function filterByCategory(data, indices, field, value) {
      var result = [];
      var col = data[field];
      if (!col) return result;
      for (var i = 0; i < indices.length; i++) {
        var idx = indices[i];
        if (col[idx] === value) {
          result.push(idx);
        }
      }
      return result;
    }

    /**
     * Filter indices to rows where min <= data[field][i] <= max.
     * Skips null/undefined/NaN values.
     */
    function filterByRange(data, indices, field, min, max) {
      var result = [];
      var col = data[field];
      if (!col) return result;
      for (var i = 0; i < indices.length; i++) {
        var idx = indices[i];
        var v = col[idx];
        if (isNum(v) && v >= min && v <= max) {
          result.push(idx);
        }
      }
      return result;
    }

    /**
     * Group-by summary statistics for bar charts with error bars.
     * Returns {groups: string[], means: number[], stdevs: number[], ns: number[]}.
     */
    function groupSummary(data, indices, groupField, valueField) {
      var empty = { groups: [], means: [], stdevs: [], ns: [] };
      if (!data[groupField] || !data[valueField]) return empty;

      var groupCol = data[groupField];
      var valueCol = data[valueField];
      var buckets = {};
      var seenGroups = {};

      for (var i = 0; i < indices.length; i++) {
        var idx = indices[i];
        var g = groupCol[idx];
        var v = valueCol[idx];
        if (g === null || g === undefined) continue;
        var key = String(g);
        if (!seenGroups[key]) {
          seenGroups[key] = true;
          buckets[key] = [];
        }
        if (isNum(v)) buckets[key].push(v);
      }

      var groups = [];
      for (var k in seenGroups) {
        if (seenGroups.hasOwnProperty(k)) groups.push(k);
      }
      groups.sort();

      var means = [], stdevs = [], ns = [];
      for (var j = 0; j < groups.length; j++) {
        var vals = buckets[groups[j]];
        var n = vals.length;
        ns.push(n);
        if (n === 0) {
          means.push(null);
          stdevs.push(null);
        } else {
          var m = statMean(vals);
          means.push(m);
          stdevs.push(n >= 2 ? statStd(vals, m) : null);
        }
      }
      return { groups: groups, means: means, stdevs: stdevs, ns: ns };
    }

    /**
     * Collect raw numeric values per group for box/strip plots.
     * Returns plain object: {groupName: number[], ...} with sorted keys.
     */
    function groupedArrays(data, indices, groupField, valueField) {
      if (!data[groupField] || !data[valueField]) return {};

      var groupCol = data[groupField];
      var valueCol = data[valueField];
      var buckets = {};
      var seenGroups = {};

      for (var i = 0; i < indices.length; i++) {
        var idx = indices[i];
        var g = groupCol[idx];
        var v = valueCol[idx];
        if (g === null || g === undefined) continue;
        var key = String(g);
        if (!seenGroups[key]) {
          seenGroups[key] = true;
          buckets[key] = [];
        }
        if (isNum(v)) buckets[key].push(v);
      }

      var sortedKeys = [];
      for (var k in seenGroups) {
        if (seenGroups.hasOwnProperty(k)) sortedKeys.push(k);
      }
      sortedKeys.sort();

      var result = {};
      for (var j = 0; j < sortedKeys.length; j++) {
        result[sortedKeys[j]] = buckets[sortedKeys[j]];
      }
      return result;
    }

    /**
     * Build scatter-plot arrays, optionally grouped.
     * Returns {groupName: {x: number[], y: number[]}, ...}.
     * If no groupField, uses single key "all".
     */
    function scatterArrays(data, indices, xField, yField, groupField) {
      if (!data[xField] || !data[yField]) return {};

      var xCol = data[xField];
      var yCol = data[yField];
      var hasGroup = groupField && groupField !== '' && data[groupField];
      var gCol = hasGroup ? data[groupField] : null;

      var groups = {};
      for (var i = 0; i < indices.length; i++) {
        var idx = indices[i];
        var xi = xCol[idx];
        var yi = yCol[idx];
        if (!isNum(xi) || !isNum(yi)) continue;
        var g = hasGroup ? String(gCol[idx]) : 'all';
        if (!groups[g]) groups[g] = { x: [], y: [] };
        groups[g].x.push(xi);
        groups[g].y.push(yi);
      }

      if (!hasGroup) return groups;

      var keys = [];
      for (var k in groups) {
        if (groups.hasOwnProperty(k)) keys.push(k);
      }
      keys.sort();

      var result = {};
      for (var j = 0; j < keys.length; j++) {
        result[keys[j]] = groups[keys[j]];
      }
      return result;
    }

    /**
     * Z-score normalization within group.
     * Returns number[] aligned 1:1 with indices (NaN for non-numeric values).
     */
    function zscoreWithinGroup(data, indices, valueField, groupField) {
      var n = indices.length;
      if (!data[valueField] || !data[groupField]) {
        var nanArr = [];
        for (var a = 0; a < n; a++) nanArr.push(NaN);
        return nanArr;
      }

      var valCol = data[valueField];
      var grpCol = data[groupField];

      // Partition indices by group (store position in indices array)
      var groupMap = {};
      for (var i = 0; i < n; i++) {
        var g = String(grpCol[indices[i]]);
        if (!groupMap[g]) groupMap[g] = [];
        groupMap[g].push(i);
      }

      // Per-group mean and stdev
      var groupStats = {};
      for (var gk in groupMap) {
        if (!groupMap.hasOwnProperty(gk)) continue;
        var positions = groupMap[gk];
        var sum = 0, count = 0;
        for (var p = 0; p < positions.length; p++) {
          var v = valCol[indices[positions[p]]];
          if (isNum(v)) { sum += v; count++; }
        }
        var mean = count > 0 ? sum / count : 0;
        var sqSum = 0;
        for (var q = 0; q < positions.length; q++) {
          var v2 = valCol[indices[positions[q]]];
          if (isNum(v2)) { sqSum += (v2 - mean) * (v2 - mean); }
        }
        var stdev = count > 0 ? Math.sqrt(sqSum / count) : 0;
        groupStats[gk] = { mean: mean, stdev: stdev };
      }

      // Map each index to its z-score
      var result = new Array(n);
      for (var r = 0; r < n; r++) {
        var rv = valCol[indices[r]];
        var rg = String(grpCol[indices[r]]);
        var stats = groupStats[rg];
        if (!isNum(rv)) {
          result[r] = NaN;
        } else if (stats.stdev === 0) {
          result[r] = 0;
        } else {
          result[r] = (rv - stats.mean) / stats.stdev;
        }
      }
      return result;
    }

    /**
     * Pivot to dense matrix for heatmaps.
     * Returns {rowLabels: string[], colLabels: string[], z: number[][]}.
     * Multiple values per cell are averaged. Missing cells are NaN.
     */
    function pivotToMatrix(data, indices, rowField, colField, valueField) {
      if (!data[rowField] || !data[colField] || !data[valueField]) {
        return { rowLabels: [], colLabels: [], z: [] };
      }

      var rowCol = data[rowField];
      var colCol = data[colField];
      var valCol = data[valueField];

      // Collect unique labels
      var rowSet = {}, colSet = {};
      for (var i = 0; i < indices.length; i++) {
        var r = rowCol[indices[i]];
        var c = colCol[indices[i]];
        if (r !== null && r !== undefined) rowSet[r] = true;
        if (c !== null && c !== undefined) colSet[c] = true;
      }

      var rowLabels = Object.keys(rowSet).sort();
      var colLabels = Object.keys(colSet).sort();

      // Build lookup maps
      var rowIdx = {}, colIdx = {};
      for (i = 0; i < rowLabels.length; i++) rowIdx[rowLabels[i]] = i;
      for (i = 0; i < colLabels.length; i++) colIdx[colLabels[i]] = i;

      // Initialize accumulators
      var nRows = rowLabels.length, nCols = colLabels.length;
      var sum = [], count = [], z = [];
      for (i = 0; i < nRows; i++) {
        sum[i] = []; count[i] = []; z[i] = [];
        for (var j = 0; j < nCols; j++) {
          sum[i][j] = 0; count[i][j] = 0; z[i][j] = NaN;
        }
      }

      // Accumulate values
      for (i = 0; i < indices.length; i++) {
        var rv = rowCol[indices[i]];
        var cv = colCol[indices[i]];
        var v = valCol[indices[i]];
        if (!isNum(v)) continue;
        if (rv === null || rv === undefined || cv === null || cv === undefined) continue;
        var ri = rowIdx[rv], ci = colIdx[cv];
        sum[ri][ci] += v;
        count[ri][ci] += 1;
      }

      // Compute means
      for (i = 0; i < nRows; i++) {
        for (j = 0; j < nCols; j++) {
          if (count[i][j] > 0) z[i][j] = sum[i][j] / count[i][j];
        }
      }

      return { rowLabels: rowLabels, colLabels: colLabels, z: z };
    }
"""
    # Append plugin JS
    plugin_js = "\n".join(p.js() for p in (plugins or []))
    return framework_js + "\n" + transform_js + "\n" + plugin_js
