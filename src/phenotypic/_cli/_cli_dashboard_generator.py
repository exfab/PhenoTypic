"""
Dashboard HTML generator for PhenoTypic CLI processing progress.

Generates a single self-contained HTML+CSS+JS dashboard file that monitors
processing progress by polling ``manifest.json`` and ``failures.jsonl`` in
the same ``progress/`` directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_dashboard(progress_dir: Path) -> None:
    """Write ``dashboard.html`` to the progress directory.

    Args:
        progress_dir: Directory containing ``manifest.json`` and
            ``failures.jsonl``.  The generated dashboard is written as
            ``dashboard.html`` in this same directory so that relative
            ``fetch()`` calls resolve correctly.
    """
    progress_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = progress_dir / "dashboard.html"
    dashboard_path.write_text(_build_html(), encoding="utf-8")
    logger.info("Dashboard written to %s", dashboard_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_html() -> str:
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
        f"{_build_body()}\n"
        f"  <script>\n{_build_js()}\n  </script>\n"
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
    }"""


def _build_body() -> str:
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
  </div>"""


def _build_js() -> str:
    """Return the inline JavaScript for the dashboard."""
    return """\
    // ── State ──────────────────────────────────────────────────
    let refreshTimer = null;
    const REFRESH_MS = 10000;

    // ── Helpers ────────────────────────────────────────────────
    function esc(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    function fmtTs(iso) {
      if (!iso) return '';
      const d = new Date(iso);
      if (isNaN(d)) return esc(iso);
      return d.toLocaleString();
    }

    function shortTs(iso) {
      if (!iso) return '';
      const d = new Date(iso);
      if (isNaN(d)) return esc(iso);
      const hh = String(d.getHours()).padStart(2, '0');
      const mm = String(d.getMinutes()).padStart(2, '0');
      const ss = String(d.getSeconds()).padStart(2, '0');
      return hh + ':' + mm + ':' + ss;
    }

    // ── Render: Summary Cards ──────────────────────────────────
    function renderSummary(data) {
      const cards = [
        { label: 'Total',     value: data.total_images ?? 0, cls: '' },
        { label: 'Completed', value: data.completed ?? 0,    cls: 'completed' },
        { label: 'Failed',    value: data.failed ?? 0,       cls: 'failed' },
        { label: 'In Progress', value: data.started ?? 0,    cls: 'running' },
        { label: 'Pending',   value: data.pending ?? 0,      cls: 'pending' },
      ];
      const el = document.getElementById('cards');
      el.innerHTML = cards.map(c =>
        '<div class="card ' + c.cls + '">' +
          '<div class="card-value">' + c.value.toLocaleString() + '</div>' +
          '<div class="card-label">' + c.label + '</div>' +
        '</div>'
      ).join('');
    }

    // ── Render: Overall Progress ───────────────────────────────
    function renderProgress(data) {
      const total = data.total_images || 1;
      const done  = (data.completed ?? 0) + (data.failed ?? 0);
      const pct   = Math.min(100, (done / total) * 100);
      const fill  = document.getElementById('progress-fill');
      fill.style.width = pct.toFixed(1) + '%';
      if (data.is_complete) {
        fill.classList.add('complete');
      } else {
        fill.classList.remove('complete');
      }
      document.getElementById('progress-pct').textContent = pct.toFixed(1) + '%';
    }

    // ── Render: SLURM Chunks ───────────────────────────────────
    function renderSlurm(data) {
      const info = data.slurm_info;
      const section = document.getElementById('slurm-section');
      if (!info || data.execution_mode !== 'slurm') {
        section.style.display = 'none';
        return;
      }
      section.style.display = '';

      const totalChunks = info.total_chunks || 0;
      const active    = new Set(info.active_chunks    || []);
      const completed = new Set(info.completed_chunks || []);
      const pending   = new Set(info.pending_chunks   || []);
      const jobIds    = info.chunk_job_ids || {};

      let html = '';
      for (let i = 0; i < totalChunks; i++) {
        let cls, statusText;
        if (active.has(i))         { cls = 'chunk-active';    statusText = 'running'; }
        else if (completed.has(i)) { cls = 'chunk-completed'; statusText = 'done'; }
        else if (pending.has(i))   { cls = 'chunk-pending';   statusText = 'pending'; }
        else                       { cls = 'chunk-pending';   statusText = 'unknown'; }

        const jid = jobIds[String(i)];
        const idLabel = jid ? ' <span class="chunk-label">(' + esc(String(jid)) + ')</span>' : '';
        html += '<span class="chunk-badge ' + cls + '">' +
                  'Chunk ' + i + idLabel +
                '</span>';
      }
      document.getElementById('chunk-grid').innerHTML = html;
    }

    // ── Render: Datasets ───────────────────────────────────────
    function renderDatasets(datasets) {
      const container = document.getElementById('datasets-list');
      if (!datasets || Object.keys(datasets).length === 0) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem">No dataset information available.</div>';
        return;
      }
      let html = '';
      for (const [name, ds] of Object.entries(datasets)) {
        const total     = ds.total || 1;
        const completed = ds.completed || 0;
        const failed    = ds.failed || 0;
        const started   = ds.started || 0;
        const pending   = ds.pending || 0;
        const pct       = ((completed / total) * 100).toFixed(1);
        const uid       = 'ds-' + name.replace(/[^a-zA-Z0-9_-]/g, '_');

        html += '<div class="dataset-item">' +
          '<div class="dataset-header" onclick="toggleDataset(\'' + uid + '\')">' +
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
      }
      container.innerHTML = html;
    }

    function toggleDataset(uid) {
      const body  = document.getElementById(uid + '-body');
      const arrow = document.getElementById(uid + '-arrow');
      if (body.classList.contains('open')) {
        body.classList.remove('open');
        arrow.classList.remove('open');
      } else {
        body.classList.add('open');
        arrow.classList.add('open');
      }
    }

    // ── Render: Failure Category Chart ─────────────────────────
    function renderFailureChart(categories) {
      const el = document.getElementById('failure-chart');
      if (!categories || Object.keys(categories).length === 0) {
        el.innerHTML = '<div class="chart-empty">No failures recorded.</div>';
        return;
      }
      const entries = Object.entries(categories).sort((a, b) => b[1] - a[1]);
      const maxVal  = entries[0][1] || 1;

      let html = '';
      for (const [cat, count] of entries) {
        const widthPct = ((count / maxVal) * 100).toFixed(1);
        html += '<div class="chart-row">' +
          '<span class="chart-label" title="' + esc(cat) + '">' + esc(cat) + '</span>' +
          '<div class="chart-bar-track">' +
            '<div class="chart-bar-fill" style="width:' + widthPct + '%"></div>' +
          '</div>' +
          '<span class="chart-count">' + count + '</span>' +
        '</div>';
      }
      el.innerHTML = html;
    }

    // ── Render: Recent Failures Table ──────────────────────────
    async function renderRecentFailures(data) {
      const container = document.getElementById('failures-table-container');
      let records = [];
      try {
        const resp = await fetch('failures.jsonl?' + Date.now());
        if (resp.ok) {
          const text = await resp.text();
          const lines = text.trim().split('\\n').filter(Boolean);
          for (const line of lines) {
            try { records.push(JSON.parse(line)); } catch(e) { /* skip malformed */ }
          }
        }
      } catch(e) { /* no failures file yet */ }

      // Show only last 50
      const recent = records.slice(-50).reverse();

      if (recent.length === 0) {
        container.innerHTML = '<div class="failures-empty">No failures recorded.</div>';
        return;
      }

      let html = '<table class="failures-table"><thead><tr>' +
        '<th>Time</th><th>Dataset</th><th>Image</th><th>Error Type</th><th>Details</th>' +
        '</tr></thead><tbody>';

      for (let i = 0; i < recent.length; i++) {
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
              ? '<span class="failure-msg-toggle" onclick="toggleFailMsg(\'' + msgId + '\')">show details</span>' +
                '<div class="failure-msg-body" id="' + msgId + '">' + esc(msgContent) + '</div>'
              : '<span style="color:var(--text-muted)">-</span>') +
          '</td>' +
        '</tr>';
      }
      html += '</tbody></table>';
      container.innerHTML = html;
    }

    function toggleFailMsg(id) {
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.toggle('open');
      const toggle = el.previousElementSibling;
      if (toggle) {
        toggle.textContent = el.classList.contains('open') ? 'hide details' : 'show details';
      }
    }

    // ── Main Refresh Loop ──────────────────────────────────────
    async function refresh() {
      try {
        const resp = await fetch('manifest.json?' + Date.now());
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

        if (data.is_complete) {
          stopRefresh();
          showComplete();
        }
      } catch(e) {
        // manifest not available yet — keep trying
      }
    }

    function stopRefresh() {
      if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
      }
    }

    function showComplete() {
      const badge = document.getElementById('status-badge');
      badge.className = 'status-badge status-complete';
      badge.innerHTML = '&#10003; Complete';
      const fill = document.getElementById('progress-fill');
      fill.classList.add('complete');
    }

    // ── Boot ───────────────────────────────────────────────────
    refresh();
    refreshTimer = setInterval(refresh, REFRESH_MS);"""
