"""Scatter plot plugin for the Analysis tab."""

from __future__ import annotations

from typing import TYPE_CHECKING

from phenotypic.tools_.register import register_analysis

from ._base_plugin import BaseAnalysisPlugin

if TYPE_CHECKING:
    from ._prepare_context import AnalysisPrepareContext


@register_analysis
class ScatterPlotPlugin(BaseAnalysisPlugin):
    """Interactive scatter plot with column pickers and color-by selector."""

    call_name = "scatter"
    display_name = "Scatter Plot"
    sort_order = 10

    def prepare_data(self, ctx: AnalysisPrepareContext) -> None:
        """No-op — scatter data is loaded client-side from Parquet via hyparquet."""
        return

    def css(self) -> str:
        """Return CSS scoped with the plugin's call_name prefix."""
        return """\
/* Scoped: analysis-scatter plugin */
.scatter-controls {
  display: flex;
  flex-wrap: wrap;
  gap: var(--sp-3);
  margin-bottom: var(--sp-4);
  align-items: flex-end;
}
.scatter-control-group {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.scatter-control-group label {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--color-muted);
}
.scatter-control-group select {
  background: var(--color-white);
  border: 1.5px solid var(--color-border);
  border-radius: var(--radius);
  padding: 0.4rem 0.6rem;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--color-body);
  min-width: 180px;
  transition: border-color var(--transition);
}
.scatter-control-group select:focus { outline: none; border-color: var(--color-blue); box-shadow: 0 0 0 3px rgba(27,117,188,0.12); }
#scatter-plot { min-height: 450px; }
"""

    def html(self) -> str:
        """Return HTML for the sub-tab body."""
        return (
            '<div id="analysis-scatter-container">'
            '<div class="analysis-empty">No data available yet.</div>'
            "</div>"
        )

    def js(self) -> str:
        """Return JS including an ``initAnalysis_scatter()`` function."""
        return """\
var OKABE_ITO_SCATTER = ['#E69F00','#56B4E9','#009E73','#D55E00','#0072B2','#CC79A7'];
var scatterIndices = null;

function initAnalysis_scatter() {
  var container = document.getElementById('analysis-scatter-container');
  if (!sharedParquetState.loaded || sharedParquetState.nRows === 0) {
    container.innerHTML = '<div class="analysis-empty">No data available yet.</div>';
    return;
  }
  renderScatterPlot();
}

function selectScatterColumns(allNumericCols) {
  var prefixes = ['Metadata_', 'Grid_', 'Shape_', 'Intensity_', 'Color_'];
  var selected = [];
  for (var pi = 0; pi < prefixes.length; pi++) {
    var prefix = prefixes[pi];
    for (var ci = 0; ci < allNumericCols.length; ci++) {
      var c = allNumericCols[ci];
      if (c.indexOf('Texture') === 0) continue;
      if (c.indexOf(prefix) === 0 && selected.indexOf(c) < 0) {
        selected.push(c);
      }
    }
  }
  for (var ri = 0; ri < allNumericCols.length; ri++) {
    var rc = allNumericCols[ri];
    if (rc.indexOf('Texture') === 0) continue;
    if (selected.indexOf(rc) < 0) selected.push(rc);
  }
  return selected.slice(0, 25);
}

function buildScatterOptions(cols, selectedVal) {
  var grouped = {};
  cols.forEach(function(c) {
    var g = c.indexOf('_') > 0 ? c.substring(0, c.indexOf('_')) : 'Other';
    if (!grouped[g]) grouped[g] = [];
    grouped[g].push(c);
  });
  var h = '';
  for (var g in grouped) {
    h += '<optgroup label="' + esc(g) + '">';
    grouped[g].forEach(function(c) {
      h += '<option value="' + esc(c) + '"' + (c === selectedVal ? ' selected' : '') + '>' + esc(c) + '</option>';
    });
    h += '</optgroup>';
  }
  return h;
}

function renderScatterPlot() {
  var container = document.getElementById('analysis-scatter-container');
  var numCols = selectScatterColumns(sharedParquetState.numericCols);
  var catCols = sharedParquetState.catCols;

  if (numCols.length < 2) {
    container.innerHTML = '<div class="analysis-empty">Not enough numeric columns for scatter plot.</div>';
    return;
  }

  var totalRows = sharedParquetState.nRows;
  var maxSample = 10000;
  var sampled = totalRows > maxSample;
  if (sampled) {
    scatterIndices = [];
    var seen = {};
    while (scatterIndices.length < maxSample) {
      var ri = Math.floor(Math.random() * totalRows);
      if (!seen[ri]) { seen[ri] = true; scatterIndices.push(ri); }
    }
    scatterIndices.sort(function(a, b) { return a - b; });
  } else {
    scatterIndices = [];
    for (var i = 0; i < totalRows; i++) scatterIndices.push(i);
  }

  var defaultX = numCols.find(function(c) { return c.indexOf('Area') >= 0; }) || numCols[0];
  var defaultY = numCols.find(function(c) { return c.indexOf('MeanIntensity') >= 0 || c.indexOf('Intensity') >= 0; }) || numCols[1];
  var defaultColor = catCols.find(function(c) { return c === 'Metadata_Dataset'; }) || '';

  var html = '';
  if (sampled) {
    html += '<div class="analysis-sample-label">Showing 10,000 of ' + totalRows.toLocaleString() + ' rows (sampled for performance)</div>';
  }
  html += '<div class="scatter-controls">';
  html += '<div class="scatter-control-group"><label>X Axis</label><select id="scatter-x" onchange="updateScatter()">' + buildScatterOptions(numCols, defaultX) + '</select></div>';
  html += '<div class="scatter-control-group"><label>Y Axis</label><select id="scatter-y" onchange="updateScatter()">' + buildScatterOptions(numCols, defaultY) + '</select></div>';
  html += '<div class="scatter-control-group"><label>Color By</label><select id="scatter-color" onchange="updateScatter()"><option value="">None</option>' + buildScatterOptions(catCols, defaultColor) + '</select></div>';
  html += '</div>';
  html += '<div id="scatter-plot"></div>';
  container.innerHTML = html;
  updateScatter();
}

function updateScatter() {
  if (!window.Plotly) return;
  if (!scatterIndices) return;
  var xCol = document.getElementById('scatter-x').value;
  var yCol = document.getElementById('scatter-y').value;
  var colorCol = document.getElementById('scatter-color').value;

  var groups = scatterArrays(sharedParquetState.allData, scatterIndices, xCol, yCol, colorCol);
  var keys = Object.keys(groups);
  var traces;

  if (colorCol && keys.length > 0 && keys[0] !== 'all') {
    var namedKeys = keys.slice(0, 6);
    traces = namedKeys.map(function(gName, i) {
      return {
        x: groups[gName].x,
        y: groups[gName].y,
        mode: 'markers', type: 'scattergl', name: gName,
        marker: { color: OKABE_ITO_SCATTER[i % OKABE_ITO_SCATTER.length], size: 4, opacity: 0.7 }
      };
    });
    if (keys.length > 6) {
      var otherX = [], otherY = [];
      for (var oi = 6; oi < keys.length; oi++) {
        otherX = otherX.concat(groups[keys[oi]].x);
        otherY = otherY.concat(groups[keys[oi]].y);
      }
      traces.push({
        x: otherX, y: otherY,
        mode: 'markers', type: 'scattergl', name: 'Other',
        marker: { color: '#BBBBBB', size: 4, opacity: 0.5 }
      });
    }
  } else if (groups['all']) {
    traces = [{ x: groups['all'].x, y: groups['all'].y, mode: 'markers', type: 'scattergl', marker: { color: '#003660', size: 4, opacity: 0.7 } }];
  } else {
    traces = [];
  }

  var layout = {
    xaxis: { title: xCol },
    yaxis: { title: yCol },
    margin: { t: 10 },
    legend: { orientation: 'h', y: -0.15 },
    dragmode: 'pan',
    plot_bgcolor: '#f5f7fa'
  };
  Plotly.newPlot('scatter-plot', traces, layout, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d','select2d'] });
}
"""
