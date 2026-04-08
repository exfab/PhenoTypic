"""Summary statistics plugin for the Analysis tab.

Supports live filtered stats from Parquet data via hyparquet in the browser,
with fallback to pre-computed JSON (``analysisData.stats``) when hyparquet
is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from phenotypic.tools_.register import register_analysis

from ._base_plugin import BaseAnalysisPlugin

if TYPE_CHECKING:
    from ._prepare_context import AnalysisPrepareContext


@register_analysis
class SummaryStatsPlugin(BaseAnalysisPlugin):
    """Live-filterable descriptive statistics from Parquet data.

    When hyparquet is available in the browser, loads Parquet files
    (chunked or single) client-side, provides dataset/column filtering,
    and computes descriptive statistics in JavaScript.  Falls back to
    pre-computed JSON stats when hyparquet is not loaded.
    """

    call_name = "stats"
    display_name = "Statistics"
    sort_order = 20

    def prepare_data(self, ctx: AnalysisPrepareContext) -> None:
        """Write ``analysis_stats.json`` (JSON fallback for stats)."""
        if ctx.merged_df is None:
            return

        from .._analysis_helpers import (
            partition_by_dataset,
            sanitize_for_json,
            write_json_atomic,
        )

        df = ctx.merged_df
        numeric_cols = [
            c for c in df.columns if df[c].dtype.is_numeric()
        ]

        # Build column groups by splitting on the first underscore.
        column_groups: Dict[str, List[str]] = {}
        for col in numeric_cols:
            if "_" in col:
                group = col.split("_", 1)[0]
            else:
                group = col
            column_groups.setdefault(group, []).append(col)

        groups = partition_by_dataset(df)

        datasets: Dict[str, dict] = {}
        for ds_name, group_df in sorted(groups.items()):
            col_stats: Dict[str, dict] = {}
            for col in numeric_cols:
                series = group_df[col]
                count = int(series.drop_nulls().len())
                mean = series.mean() if count else None
                std = series.std() if count else None
                col_min = series.min() if count else None
                col_max = series.max() if count else None
                median = series.median() if count else None

                if mean is not None and std is not None and mean != 0:
                    cv = float(std) / abs(float(mean)) * 100  # type: ignore[arg-type]
                else:
                    cv = None

                col_stats[col] = {
                    "count": count,
                    "mean": sanitize_for_json(mean),
                    "std": sanitize_for_json(std),
                    "min": sanitize_for_json(col_min),
                    "max": sanitize_for_json(col_max),
                    "median": sanitize_for_json(median),
                    "cv": sanitize_for_json(cv),
                }

            datasets[str(ds_name)] = {"columns": col_stats}

        summary_stats = {
            "datasets": datasets,
            "column_groups": column_groups,
        }

        write_json_atomic(summary_stats, ctx.progress_dir / "analysis_stats.json")

    def css(self) -> str:
        """Return CSS scoped with the plugin's call_name prefix."""
        return """\
/* Scoped: analysis-stats plugin */
.stats-controls {
  margin-bottom: var(--sp-4);
}
.stats-controls select {
  background: var(--color-white);
  border: 1.5px solid var(--color-border);
  border-radius: var(--radius);
  padding: 0.4rem 0.6rem;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--color-body);
  min-width: 200px;
}
.stats-table-wrapper { overflow-x: auto; }
.stats-table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--text-sm);
}
.stats-table th {
  text-align: left;
  padding: 10px 12px;
  font-family: var(--font-mono);
  font-size: 0.6875rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--color-muted);
  border-bottom: 2px solid var(--color-navy);
  white-space: nowrap;
}
.stats-table td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--color-rule);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--color-body);
  white-space: nowrap;
}
.stats-table tbody tr:hover td { background: rgba(27,117,188,0.03); }
.stats-table td.cv-high { color: #D55E00; font-weight: 500; }
.stats-filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: var(--sp-3);
  align-items: flex-end;
  margin-bottom: var(--sp-4);
}
.stats-dataset-panel {
  display: flex;
  flex-wrap: wrap;
  gap: var(--sp-2);
  margin-bottom: var(--sp-3);
}
.stats-dataset-panel label {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: var(--text-xs);
  font-family: var(--font-mono);
  cursor: pointer;
}
.stats-filter-row {
  display: flex;
  gap: var(--sp-2);
  align-items: center;
  margin-bottom: var(--sp-2);
}
.stats-filter-row select,
.stats-filter-row input {
  background: var(--color-white);
  border: 1.5px solid var(--color-border);
  border-radius: var(--radius);
  padding: 0.3rem 0.5rem;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--color-body);
}
.stats-filter-row input { width: 100px; }
.stats-filter-row button {
  background: none;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: 0.2rem 0.5rem;
  cursor: pointer;
  font-size: var(--text-xs);
  color: var(--color-muted);
}
.stats-row-count {
  font-size: var(--text-xs);
  font-family: var(--font-mono);
  color: var(--color-muted);
  margin-bottom: var(--sp-3);
}
.stats-loading {
  font-size: var(--text-sm);
  color: var(--color-muted);
  padding: var(--sp-4);
}
"""

    def html(self) -> str:
        """Return HTML for the sub-tab body."""
        return (
            '<div id="analysis-stats-container">'
            '<div class="analysis-empty">'
            "No data available yet."
            "</div>"
            "</div>"
        )

    def js(self) -> str:
        """Return JS including an ``initAnalysis_stats()`` function."""
        return """\
var statsState = {
  selectedDatasets: null,
  filters: []
};

var STAT_KEYS = ['count', 'mean', 'std', 'min', 'median', 'max', 'cv'];

/* ---- Shared helpers ---- */

function columnGroup(name) {
  var idx = name.indexOf('_');
  return idx > 0 ? name.substring(0, idx) : 'Other';
}

function formatStatValue(v) {
  if (v === null || v === undefined) return '-';
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return v.toLocaleString();
  return v.toFixed(4);
}

function buildStatCell(key, v) {
  var cls = (key === 'cv' && typeof v === 'number' && v > 25) ? ' class="cv-high"' : '';
  return '<td' + cls + '>' + esc(formatStatValue(v)) + '</td>';
}

function buildStatsTableHeader() {
  var html = '<div class="stats-table-wrapper"><table class="stats-table"><thead><tr><th>Column</th>';
  for (var ki = 0; ki < STAT_KEYS.length; ki++) {
    html += '<th>' + STAT_KEYS[ki].toUpperCase() + '</th>';
  }
  html += '</tr></thead><tbody>';
  return html;
}

function buildStatsTableRow(col, stats) {
  var html = '<tr><td style="font-weight:500;color:var(--color-heading)">' + esc(col) + '</td>';
  for (var si = 0; si < STAT_KEYS.length; si++) {
    html += buildStatCell(STAT_KEYS[si], stats[STAT_KEYS[si]]);
  }
  html += '</tr>';
  return html;
}

/* ---- Init ---- */

function initAnalysis_stats() {
  if (sharedParquetState.loaded && sharedParquetState.nRows > 0) {
    buildStatsFilterUI();
    updateLiveStats();
  } else if (analysisData.stats) {
    renderFallbackStats();
  } else {
    var container = document.getElementById('analysis-stats-container');
    container.innerHTML = '<div class="analysis-empty">No data available yet.</div>';
  }
}

/* ---- Filter UI ---- */

function buildStatsFilterUI() {
  var container = document.getElementById('analysis-stats-container');
  var html = '';

  var datasets = [];
  var dsCol = sharedParquetState.allData['Metadata_Dataset'];
  if (dsCol) {
    var seen = {};
    for (var di = 0; di < dsCol.length; di++) {
      var dv = dsCol[di];
      if (dv && !seen[dv]) { seen[dv] = true; datasets.push(String(dv)); }
    }
    datasets.sort();
  }

  if (datasets.length > 0) {
    html += '<div class="stats-dataset-panel">';
    for (var dsi = 0; dsi < datasets.length; dsi++) {
      html += '<label><input type="checkbox" checked onchange="onStatsDatasetChange()" value="' + esc(datasets[dsi]) + '"> ' + esc(datasets[dsi]) + '</label>';
    }
    html += '</div>';
  }

  var groups = {};
  for (var gi = 0; gi < sharedParquetState.numericCols.length; gi++) {
    var nc = sharedParquetState.numericCols[gi];
    var g = columnGroup(nc);
    if (!groups[g]) groups[g] = [];
    groups[g].push(nc);
  }
  var groupNames = Object.keys(groups).sort();

  html += '<div class="stats-filter-bar">';
  html += '<div class="scatter-control-group"><label>Column Group</label>';
  html += '<select id="stats-group" onchange="updateLiveStats()">';
  html += '<option value="">All columns</option>';
  for (var gni = 0; gni < groupNames.length; gni++) {
    html += '<option value="' + esc(groupNames[gni]) + '">' + esc(groupNames[gni]) + ' (' + groups[groupNames[gni]].length + ')</option>';
  }
  html += '</select></div>';

  html += '<button onclick="addStatsFilter()" style="align-self:flex-end;padding:0.4rem 0.8rem;background:var(--color-white);border:1.5px solid var(--color-border);border-radius:var(--radius);font-size:var(--text-xs);cursor:pointer;">+ Add Filter</button>';
  html += '</div>';

  html += '<div id="stats-filters"></div>';
  html += '<div class="stats-row-count" id="stats-row-count"></div>';
  html += '<div id="stats-table-wrap"></div>';

  container.innerHTML = html;
}

function addStatsFilter() {
  statsState.filters.push({col: sharedParquetState.numericCols[0] || '', op: '>', val: ''});
  renderStatsFilters();
}

function removeStatsFilter(idx) {
  statsState.filters.splice(idx, 1);
  renderStatsFilters();
  updateLiveStats();
}

function renderStatsFilters() {
  var container = document.getElementById('stats-filters');
  if (!container) return;
  var ops = ['>', '<', '>=', '<=', '==', '!='];
  var html = '';
  for (var fi = 0; fi < statsState.filters.length; fi++) {
    var f = statsState.filters[fi];
    html += '<div class="stats-filter-row">';
    html += '<select onchange="statsState.filters[' + fi + '].col=this.value;updateLiveStats()">';
    for (var ci = 0; ci < sharedParquetState.numericCols.length; ci++) {
      var c = sharedParquetState.numericCols[ci];
      html += '<option value="' + esc(c) + '"' + (c === f.col ? ' selected' : '') + '>' + esc(c) + '</option>';
    }
    html += '</select>';
    html += '<select onchange="statsState.filters[' + fi + '].op=this.value;updateLiveStats()">';
    for (var oi = 0; oi < ops.length; oi++) {
      html += '<option value="' + ops[oi] + '"' + (ops[oi] === f.op ? ' selected' : '') + '>' + ops[oi] + '</option>';
    }
    html += '</select>';
    html += '<input type="number" step="any" value="' + esc(String(f.val)) + '" onchange="statsState.filters[' + fi + '].val=parseFloat(this.value);updateLiveStats()">';
    html += '<button onclick="removeStatsFilter(' + fi + ')">\\u00d7</button>';
    html += '</div>';
  }
  container.innerHTML = html;
}

function onStatsDatasetChange() {
  var checks = document.querySelectorAll('.stats-dataset-panel input[type=checkbox]');
  var selected = {};
  var selectedCount = 0;
  for (var ci = 0; ci < checks.length; ci++) {
    if (checks[ci].checked) { selected[checks[ci].value] = true; selectedCount++; }
  }
  statsState.selectedDatasets = selectedCount === checks.length ? null : selected;
  updateLiveStats();
}

/* ---- Row filtering ---- */

function getFilteredIndices() {
  var n = sharedParquetState.nRows;
  var indices = [];
  var dsCol = sharedParquetState.allData['Metadata_Dataset'];

  for (var i = 0; i < n; i++) {
    if (statsState.selectedDatasets && dsCol) {
      if (!statsState.selectedDatasets[String(dsCol[i])]) continue;
    }
    var pass = true;
    for (var fi = 0; fi < statsState.filters.length; fi++) {
      var f = statsState.filters[fi];
      if (!f.col || f.val === '' || isNaN(f.val)) continue;
      var v = sharedParquetState.allData[f.col] ? sharedParquetState.allData[f.col][i] : null;
      if (v === null || v === undefined) { pass = false; break; }
      var fv = parseFloat(f.val);
      switch(f.op) {
        case '>':  if (!(v > fv))  pass = false; break;
        case '<':  if (!(v < fv))  pass = false; break;
        case '>=': if (!(v >= fv)) pass = false; break;
        case '<=': if (!(v <= fv)) pass = false; break;
        case '==': if (!(v == fv)) pass = false; break;
        case '!=': if (!(v != fv)) pass = false; break;
      }
      if (!pass) break;
    }
    if (pass) indices.push(i);
  }
  return indices;
}

/* ---- Descriptive statistics ---- */

function computeColumnStats(values) {
  var count = statCount(values);
  var mean = statMean(values);
  var std = statStd(values, mean);
  var cv = (mean !== null && std !== null && mean !== 0) ? (std / Math.abs(mean)) * 100 : null;
  return {
    count: count,
    mean: mean,
    std: std,
    min: statMin(values),
    median: statMedian(values),
    max: statMax(values),
    cv: cv
  };
}

/* ---- Live stats table (Parquet path) ---- */

function updateLiveStats() {
  var indices = getFilteredIndices();
  var countEl = document.getElementById('stats-row-count');
  if (countEl) countEl.textContent = 'Showing ' + indices.length + ' of ' + sharedParquetState.nRows + ' rows';

  var groupSel = document.getElementById('stats-group');
  var groupFilter = groupSel ? groupSel.value : '';

  var cols = sharedParquetState.numericCols;
  if (groupFilter) {
    cols = cols.filter(function(c) { return columnGroup(c) === groupFilter; });
  }

  var html = buildStatsTableHeader();
  for (var cIdx = 0; cIdx < cols.length; cIdx++) {
    var col = cols[cIdx];
    var fullArr = sharedParquetState.allData[col] || [];
    var filtered = [];
    for (var ii = 0; ii < indices.length; ii++) {
      filtered.push(fullArr[indices[ii]]);
    }
    html += buildStatsTableRow(col, computeColumnStats(filtered));
  }
  html += '</tbody></table></div>';
  document.getElementById('stats-table-wrap').innerHTML = html;
}

/* ---- Fallback: pre-computed JSON stats ---- */

function renderFallbackStats() {
  var container = document.getElementById('analysis-stats-container');
  var d = analysisData.stats;
  if (!d || !d.datasets || Object.keys(d.datasets).length === 0) {
    container.innerHTML = '<div class="analysis-empty">No data available yet.</div>';
    return;
  }
  var dsNames = Object.keys(d.datasets);
  var html = '<div class="stats-controls"><select id="stats-dataset" onchange="updateFallbackStatsTable()">';
  for (var i = 0; i < dsNames.length; i++) {
    html += '<option value="' + esc(dsNames[i]) + '">' + esc(dsNames[i]) + '</option>';
  }
  html += '</select></div>';
  html += '<div id="stats-table-wrap"></div>';
  container.innerHTML = html;
  updateFallbackStatsTable();
}

function updateFallbackStatsTable() {
  var d = analysisData.stats;
  if (!d) return;
  var ds = document.getElementById('stats-dataset').value;
  var dsData = d.datasets[ds];
  if (!dsData || !dsData.columns) {
    document.getElementById('stats-table-wrap').innerHTML = '<div class="analysis-empty">No data for this dataset.</div>';
    return;
  }
  var sortedCols = Object.keys(dsData.columns).sort(function(a, b) {
    var ga = columnGroup(a).replace('Other', 'ZZZ');
    var gb = columnGroup(b).replace('Other', 'ZZZ');
    if (ga !== gb) return ga.localeCompare(gb);
    return a.localeCompare(b);
  });
  var html = buildStatsTableHeader();
  for (var ci = 0; ci < sortedCols.length; ci++) {
    html += buildStatsTableRow(sortedCols[ci], dsData.columns[sortedCols[ci]]);
  }
  html += '</tbody></table></div>';
  document.getElementById('stats-table-wrap').innerHTML = html;
}
"""
