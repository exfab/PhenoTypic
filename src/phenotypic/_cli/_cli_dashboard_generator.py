"""
Dashboard HTML generator for PhenoTypic CLI processing progress.

Generates a single self-contained HTML+CSS+JS dashboard file that monitors
processing progress by polling ``manifest.json`` and ``failures.jsonl`` in
the ``progress/`` subdirectory.
"""

from __future__ import annotations

import logging
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
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "  <title>PhenoTypic Processing Dashboard</title>\n"
        f"  <style>\n{_build_css()}\n  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{_build_body(execution_mode)}\n"
        f"  <script>\n{MARKED_MIN_JS}\n  </script>\n"
        f"  <script>\n{_build_js(execution_mode)}\n  </script>\n"
        "</body>\n"
        "</html>\n"
    )


def _build_css() -> str:
    """Return the inline CSS block for the dashboard."""
    return """\
    /* ── Reset & Base ─────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:          #1a1d23;
      --bg-card:     #23272e;
      --bg-hover:    #2b3038;
      --border:      #333842;
      --text:        #cdd6e0;
      --text-muted:  #8b95a5;
      --text-bright: #e8ecf1;
      --accent:      #4c9aff;
      --green:       #36b37e;
      --green-bg:    rgba(54,179,126,0.12);
      --red:         #ff5630;
      --red-bg:      rgba(255,86,48,0.12);
      --blue:        #4c9aff;
      --blue-bg:     rgba(76,154,255,0.12);
      --gray:        #8b95a5;
      --gray-bg:     rgba(139,149,165,0.10);
      --yellow:      #ffab00;
      --mono:        'Menlo', 'Consolas', 'Liberation Mono', 'Courier New', monospace;
      --sans:        -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     Helvetica, Arial, sans-serif;
    }

    body {
      font-family: var(--sans);
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
    }

    /* ── Layout ───────────────────────────────────────────────── */
    .container {
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px 20px 60px;
    }

    /* ── Header ───────────────────────────────────────────────── */
    .header {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }
    .header h1 {
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text-bright);
    }
    .header-right {
      display: flex;
      align-items: center;
      gap: 16px;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: 20px;
      font-weight: 600;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .status-live {
      background: var(--green-bg);
      color: var(--green);
    }
    .status-complete {
      background: var(--blue-bg);
      color: var(--blue);
    }
    .pulse-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--green);
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
      gap: 14px;
      margin-bottom: 28px;
    }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px 20px;
      text-align: center;
      transition: border-color 0.2s;
    }
    .card:hover { border-color: var(--accent); }
    .card-value {
      font-family: var(--mono);
      font-size: 2rem;
      font-weight: 700;
      color: var(--text-bright);
    }
    .card-label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .card.completed .card-value { color: var(--green); }
    .card.failed    .card-value { color: var(--red);   }
    .card.running   .card-value { color: var(--blue);  }
    .card.pending   .card-value { color: var(--gray);  }

    /* ── Overall Progress Bar ─────────────────────────────────── */
    .progress-section {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 28px;
    }
    .progress-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 10px;
    }
    .progress-title {
      font-weight: 600;
      color: var(--text-bright);
      font-size: 0.95rem;
    }
    .progress-pct {
      font-family: var(--mono);
      font-weight: 700;
      font-size: 1.1rem;
      color: var(--accent);
    }
    .progress-track {
      width: 100%;
      height: 14px;
      background: var(--bg);
      border-radius: 7px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      border-radius: 7px;
      background: linear-gradient(90deg, var(--accent), var(--green));
      transition: width 0.6s ease;
      animation: shimmer 2s linear infinite;
      background-size: 200% 100%;
    }
    @keyframes shimmer {
      0%   { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }
    .progress-fill.complete {
      animation: none;
      background: var(--green);
    }

    /* ── Active Batch / SLURM Info ────────────────────────────── */
    .slurm-section {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 28px;
    }
    .slurm-section h2 {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-bright);
      margin-bottom: 14px;
    }
    .chunk-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chunk-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 5px 12px;
      border-radius: 6px;
      font-family: var(--mono);
      font-size: 0.8rem;
      font-weight: 600;
    }
    .chunk-active    { background: var(--blue-bg);  color: var(--blue);  }
    .chunk-completed { background: var(--green-bg); color: var(--green); }
    .chunk-pending   { background: var(--gray-bg);  color: var(--gray);  }
    .chunk-label {
      font-size: 0.7rem;
      font-weight: 400;
      opacity: 0.8;
    }

    /* ── Per-Dataset Breakdown ────────────────────────────────── */
    .datasets-section {
      margin-bottom: 28px;
    }
    .datasets-section > h2 {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-bright);
      margin-bottom: 14px;
    }
    .dataset-item {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 10px;
      overflow: hidden;
    }
    .dataset-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 20px;
      cursor: pointer;
      user-select: none;
      transition: background 0.15s;
    }
    .dataset-header:hover { background: var(--bg-hover); }
    .dataset-name {
      font-weight: 600;
      color: var(--text-bright);
      font-size: 0.9rem;
    }
    .dataset-stats {
      display: flex;
      gap: 14px;
      font-family: var(--mono);
      font-size: 0.8rem;
    }
    .dataset-stats .ds-completed { color: var(--green); }
    .dataset-stats .ds-failed    { color: var(--red); }
    .dataset-stats .ds-running   { color: var(--blue); }
    .dataset-stats .ds-pending   { color: var(--gray); }
    .dataset-expand {
      transition: transform 0.2s;
      color: var(--text-muted);
      font-size: 0.8rem;
    }
    .dataset-expand.open { transform: rotate(90deg); }
    .dataset-body {
      display: none;
      padding: 0 20px 16px;
    }
    .dataset-body.open { display: block; }
    .dataset-progress-track {
      width: 100%;
      height: 8px;
      background: var(--bg);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 8px;
    }
    .dataset-progress-fill {
      height: 100%;
      border-radius: 4px;
      background: var(--green);
      transition: width 0.5s ease;
    }

    /* ── Failure Category Chart ───────────────────────────────── */
    .chart-section {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 28px;
    }
    .chart-section h2 {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-bright);
      margin-bottom: 14px;
    }
    .chart-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .chart-label {
      flex: 0 0 180px;
      text-align: right;
      font-family: var(--mono);
      font-size: 0.82rem;
      color: var(--text);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .chart-bar-track {
      flex: 1;
      height: 20px;
      background: var(--bg);
      border-radius: 4px;
      overflow: hidden;
    }
    .chart-bar-fill {
      height: 100%;
      background: var(--red);
      border-radius: 4px;
      transition: width 0.5s ease;
      min-width: 2px;
    }
    .chart-count {
      flex: 0 0 48px;
      font-family: var(--mono);
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--red);
      text-align: right;
    }
    .chart-empty {
      color: var(--text-muted);
      font-size: 0.85rem;
      padding: 12px 0;
    }

    /* ── Recent Failures Table ────────────────────────────────── */
    .failures-section {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 28px;
    }
    .failures-section h2 {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-bright);
      margin-bottom: 14px;
    }
    .failures-table {
      width: 100%;
      border-collapse: collapse;
    }
    .failures-table th {
      text-align: left;
      padding: 10px 12px;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }
    .failures-table td {
      padding: 10px 12px;
      font-size: 0.85rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    .failures-table tr:last-child td { border-bottom: none; }
    .failures-table tr:hover td { background: var(--bg-hover); }
    .failures-table .col-ts    { color: var(--text-muted); font-family: var(--mono); font-size: 0.78rem; white-space: nowrap; }
    .failures-table .col-ds    { font-weight: 500; }
    .failures-table .col-img   { font-family: var(--mono); font-size: 0.8rem; }
    .failures-table .col-type  { font-family: var(--mono); font-size: 0.8rem; color: var(--red); }
    .failure-msg-toggle {
      cursor: pointer;
      color: var(--accent);
      font-size: 0.8rem;
      user-select: none;
    }
    .failure-msg-toggle:hover { text-decoration: underline; }
    .failure-msg-body {
      display: none;
      margin-top: 6px;
      padding: 10px;
      background: var(--bg);
      border-radius: 6px;
      font-family: var(--mono);
      font-size: 0.78rem;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
      max-height: 280px;
      overflow-y: auto;
    }
    .failure-msg-body.open { display: block; }
    .failures-empty {
      color: var(--text-muted);
      font-size: 0.85rem;
      padding: 12px 0;
    }

    /* ── Responsive ───────────────────────────────────────────── */
    @media (max-width: 768px) {
      .cards { grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); }
      .chart-label { flex: 0 0 100px; font-size: 0.72rem; }
      .header { flex-direction: column; align-items: flex-start; }
      .dataset-stats { gap: 8px; font-size: 0.72rem; }
      .failures-table { font-size: 0.78rem; }
    }

    @media (max-width: 480px) {
      .container { padding: 12px 10px 40px; }
      .cards { grid-template-columns: repeat(2, 1fr); gap: 8px; }
      .card { padding: 12px 10px; }
      .card-value { font-size: 1.5rem; }
    }

    /* ── Tabs ────────────────────────────────────────────────── */
    .tab-bar {
      display: flex;
      gap: 4px;
      margin-bottom: 24px;
      border-bottom: 2px solid var(--border);
      padding-bottom: 0;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      font-family: var(--sans);
      font-size: 0.9rem;
      font-weight: 600;
      padding: 10px 20px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
      transition: color 0.2s, border-color 0.2s;
    }
    .tab-btn:hover { color: var(--text-bright); }
    .tab-btn.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* ── README ──────────────────────────────────────────────── */
    .readme-container {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 32px 40px;
      line-height: 1.7;
      color: var(--text);
    }
    .readme-container h1 { font-size: 1.6rem; color: var(--text-bright); margin: 24px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    .readme-container h2 { font-size: 1.25rem; color: var(--text-bright); margin: 20px 0 10px; }
    .readme-container h3 { font-size: 1.05rem; color: var(--text-bright); margin: 16px 0 8px; }
    .readme-container code { background: var(--bg); padding: 2px 6px; border-radius: 4px; font-family: var(--mono); font-size: 0.88em; }
    .readme-container pre { background: var(--bg); padding: 16px; border-radius: 8px; overflow-x: auto; margin: 12px 0; }
    .readme-container pre code { background: none; padding: 0; }
    .readme-container table { border-collapse: collapse; width: 100%; margin: 12px 0; }
    .readme-container th, .readme-container td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; }
    .readme-container th { background: var(--bg); font-weight: 600; color: var(--text-bright); }
    .readme-loading { color: var(--text-muted); font-size: 0.9rem; }

    /* ── Download ────────────────────────────────────────────── */
    .download-container {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 32px 40px;
    }
    .download-container h2 { font-size: 1.25rem; color: var(--text-bright); margin-bottom: 16px; }
    .download-note {
      background: var(--blue-bg);
      color: var(--text);
      border: 1px solid rgba(76,154,255,0.25);
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 24px;
      font-size: 0.88rem;
      line-height: 1.6;
    }
    .download-section-title { font-weight: 600; color: var(--text-bright); margin: 20px 0 6px; font-size: 0.95rem; }
    .download-cmd {
      background: var(--bg);
      padding: 14px 18px;
      border-radius: 8px;
      font-family: var(--mono);
      font-size: 0.85rem;
      color: var(--text);
      user-select: all;
      margin: 8px 0 16px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }
    .download-input {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 14px;
      color: var(--text);
      font-family: var(--mono);
      font-size: 0.85rem;
      margin: 4px 0 12px;
    }
    .download-input:focus { outline: none; border-color: var(--accent); }"""


def _build_body(execution_mode: str) -> str:
    """Return the HTML body content (no <body> tags)."""
    return """\
  <div class="container">
    <!-- Header -->
    <div class="header">
      <h1>PhenoTypic Processing Dashboard</h1>
      <div class="header-right">
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

        <p style="color:var(--text-muted);font-size:0.82rem;margin-top:24px">
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
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">No dataset information available.</div>';
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
            '<div style="font-size:0.82rem;color:var(--text-muted);margin-bottom:4px">' +
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
              : '<span style="color:var(--text-muted)">-</span>') +
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

    // ── Main Refresh Loop ──────────────────────────────────────
    async function refresh() {{
      try {{
        const resp = await fetch('progress/manifest.json?' + Date.now());
        if (!resp.ok) return;
        const data = await resp.json();

        renderSummary(data);
        renderProgress(data);
        renderSlurm(data);
        renderDatasets(data.datasets);
        renderFailureChart(data.failure_categories);
        await renderRecentFailures(data);

        document.getElementById('last-updated').textContent =
          'Updated ' + new Date().toLocaleTimeString();

        if (data.is_complete) {{
          stopRefresh();
          showComplete();
        }}
      }} catch(e) {{
        // manifest not available yet — keep trying
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
