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

from ._cli_vendor_js import MARKED_MIN_JS

logger = logging.getLogger(__name__)


def generate_dashboard(output_dir: Path, *, execution_mode: str = "local") -> None:
    """Write ``dashboard.html`` to the output directory root.

    Args:
        output_dir: Root output directory.
        execution_mode: ``"local"`` or ``"slurm"``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(_build_html(execution_mode), encoding="utf-8")
    logger.info("Dashboard written to %s", dashboard_path)


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
        "  <title>PhenoTypic Processing Dashboard</title>\n"
        "  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n"
        "  <link href=\"https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap\" rel=\"stylesheet\">\n"
        f"  <style>\n{_build_css()}\n  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{_build_body(execution_mode, logo_data_uri)}\n"
        f"  <script>\n{MARKED_MIN_JS}\n  </script>\n"
        f"  <script>\n{_build_js(execution_mode)}\n  </script>\n"
        "</body>\n"
        "</html>\n"
    )


def _load_logo_data_uri() -> str:
    """Read the logo PNG and return a ``data:`` URI, or empty string on failure."""
    try:
        raw = (
            files("phenotypic._cli")
            .joinpath("_assets", "light_logo.png")
            .read_bytes()
        )
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except (OSError, ModuleNotFoundError, TypeError):
        logger.debug("Logo asset not found — skipping")
        return ""


def _build_css() -> str:
    """Return the inline CSS block for the dashboard."""
    return """\
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
    .header-logo img {
      max-height: 48px;
      width: auto;
      margin-top: var(--sp-2);
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
    }"""


def _build_body(execution_mode: str, logo_data_uri: str = "") -> str:
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
      <div>
        <h1>PhenoTypic Processing Dashboard</h1>
        {logo_html}
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
          Direct browser downloads are not available due to Apache directory
          permissions and HPCC security policies. Use the command-line tools
          below to download results to your local machine.
        </div>

        <p class="download-section-title">Server URL</p>
        <p>Auto-detected from your browser. Edit if needed:</p>
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

        <p class="download-section-title">Full recursive download</p>
        <p>Downloads the entire output directory:</p>
        <div class="download-cmd" id="cmd-full"></div>

        <p class="download-section-title">Download measurements only</p>
        <div class="download-cmd" id="cmd-csv"></div>

        <p class="download-section-title">Download overlays only</p>
        <div class="download-cmd" id="cmd-png"></div>

        <p class="download-section-title">Using rsync (if SSH access available)</p>
        <div class="download-cmd">rsync -avz user@hpcc:/path/to/output/ ./local_output/</div>

        <p style="color:var(--color-muted);font-size:var(--text-sm);margin-top:var(--sp-6)">
          Adjust <code>--cut-dirs</code> to control local directory nesting.
        </p>
      </div>
    </div>
  </div>"""


def _build_js(execution_mode: str) -> str:
    """Return the inline JavaScript for the dashboard."""
    return f"""\
    // ── State ──────────────────────────────────────────────────
    const EXECUTION_MODE = "{execution_mode}";
    let refreshTimer = null;
    const REFRESH_MS = 10000;
    let fetchErrors = 0;

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
      if (user) auth = ' --user=' + user + (pass ? ' --password=' + pass : '');

      document.getElementById('cmd-full').textContent =
        'wget -r -np -nH --cut-dirs=' + cutDirs + auth + ' ' + base;
      document.getElementById('cmd-csv').textContent =
        'wget -r -np -nH --cut-dirs=' + cutDirs + ' -A "*.csv"' + auth + ' ' + base;
      document.getElementById('cmd-png').textContent =
        'wget -r -np -nH --cut-dirs=' + cutDirs + ' -A "*.png"' + auth + ' ' + base + 'results/';
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

        if (data.is_complete) {{
          stopRefresh();
          showComplete();
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
    updateCommands();"""
