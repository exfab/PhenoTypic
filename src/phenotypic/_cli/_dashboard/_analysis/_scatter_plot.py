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
        """Write ``analysis_scatter.json`` to *ctx.progress_dir*."""
        if ctx.merged_df is None:
            return

        from .._analysis_helpers import (
            select_scatter_columns,
            stratified_sample,
            to_columnar,
            write_json_atomic,
        )

        df = ctx.merged_df
        columns = select_scatter_columns(df.columns)
        sub = df.select(columns)

        max_rows = 10_000
        total_rows = sub.height
        sampled = total_rows > max_rows
        if sampled:
            sub = stratified_sample(sub, max_rows)

        payload = to_columnar(sub, total_rows, sampled)
        write_json_atomic(payload, ctx.progress_dir / "analysis_scatter.json")

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
var OKABE_ITO_SCATTER = ['#003660','#E69F00','#56B4E9','#009E73','#D55E00','#0072B2','#CC79A7'];

function initAnalysis_scatter() {
  renderScatterPlot();
}

function renderScatterPlot() {
  var container = document.getElementById('analysis-scatter-container');
  var d = analysisData.scatter;
  if (!d || !d.columns || d.columns.length < 2) {
    container.innerHTML = '<div class="analysis-empty">No data available yet.</div>';
    return;
  }
  var numCols = d.columns.filter(function(c) {
    var vals = d.data[c];
    return vals && vals.some(function(v) { return typeof v === 'number'; });
  });
  var catCols = d.columns.filter(function(c) {
    var vals = d.data[c];
    return vals && vals.some(function(v) { return typeof v === 'string'; });
  });

  function buildScatterOptions(cols, selected) {
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
        h += '<option value="' + esc(c) + '"' + (c === selected ? ' selected' : '') + '>' + esc(c) + '</option>';
      });
      h += '</optgroup>';
    }
    return h;
  }

  var defaultX = numCols.find(function(c) { return c.includes('Area'); }) || numCols[0] || d.columns[0];
  var defaultY = numCols.find(function(c) { return c.includes('MeanIntensity') || c.includes('Intensity'); }) || numCols[1] || d.columns[1];
  var defaultColor = catCols.find(function(c) { return c === 'Metadata_Dataset'; }) || catCols[0] || '';

  var html = '';
  if (d.sampled) {
    html += '<div class="analysis-sample-label">Showing ' + d.data[d.columns[0]].length.toLocaleString() + ' of ' + d.total_rows.toLocaleString() + ' objects (sampled)</div>';
  }
  html += '<div class="scatter-controls">';
  html += '<div class="scatter-control-group"><label>X Axis</label><select id="scatter-x" onchange="updateScatter()">' + buildScatterOptions(numCols.length > 0 ? numCols : d.columns, defaultX) + '</select></div>';
  html += '<div class="scatter-control-group"><label>Y Axis</label><select id="scatter-y" onchange="updateScatter()">' + buildScatterOptions(numCols.length > 0 ? numCols : d.columns, defaultY) + '</select></div>';
  html += '<div class="scatter-control-group"><label>Color By</label><select id="scatter-color" onchange="updateScatter()"><option value="">None</option>' + buildScatterOptions(catCols.length > 0 ? catCols : d.columns, defaultColor) + '</select></div>';
  html += '</div>';
  html += '<div id="scatter-plot"></div>';
  container.innerHTML = html;
  updateScatter();
}

function updateScatter() {
  if (!window.Plotly) return;
  var d = analysisData.scatter;
  if (!d) return;
  var xCol = document.getElementById('scatter-x').value;
  var yCol = document.getElementById('scatter-y').value;
  var colorCol = document.getElementById('scatter-color').value;
  var xVals = d.data[xCol] || [];
  var yVals = d.data[yCol] || [];
  var traces;
  if (colorCol && d.data[colorCol]) {
    var colorVals = d.data[colorCol];
    var seen = {};
    var categories = [];
    colorVals.forEach(function(v) {
      if (v !== null && v !== undefined && !seen[v]) { seen[v] = true; categories.push(v); }
    });
    traces = categories.slice(0, 6).map(function(cat, i) {
      var mask = colorVals.map(function(v) { return v === cat; });
      return {
        x: xVals.filter(function(_, j) { return mask[j]; }),
        y: yVals.filter(function(_, j) { return mask[j]; }),
        mode: 'markers', type: 'scattergl', name: String(cat),
        marker: { color: OKABE_ITO_SCATTER[i % OKABE_ITO_SCATTER.length], size: 4, opacity: 0.7 }
      };
    });
    if (categories.length > 6) {
      var topCats = categories.slice(0, 6);
      var otherMask = colorVals.map(function(v) { return topCats.indexOf(v) === -1; });
      traces.push({
        x: xVals.filter(function(_, j) { return otherMask[j]; }),
        y: yVals.filter(function(_, j) { return otherMask[j]; }),
        mode: 'markers', type: 'scattergl', name: 'Other',
        marker: { color: '#BBBBBB', size: 4, opacity: 0.5 }
      });
    }
  } else {
    traces = [{ x: xVals, y: yVals, mode: 'markers', type: 'scattergl', marker: { color: '#003660', size: 4, opacity: 0.7 } }];
  }
  var layout = {
    xaxis: { title: { text: xCol, font: { family: "'DM Mono', monospace", size: 11, color: '#8892a4' } }, gridcolor: '#e8ecf2', linecolor: '#dde3ed', tickfont: { family: "'DM Mono', monospace", size: 9, color: '#8892a4' } },
    yaxis: { title: { text: yCol, font: { family: "'DM Mono', monospace", size: 11, color: '#8892a4' } }, gridcolor: '#e8ecf2', linecolor: '#dde3ed', tickfont: { family: "'DM Mono', monospace", size: 9, color: '#8892a4' } },
    plot_bgcolor: '#ffffff', paper_bgcolor: '#ffffff',
    font: { family: "'DM Sans', system-ui, sans-serif" },
    margin: { t: 30, r: 30, b: 60, l: 70 },
    legend: { font: { family: "'DM Sans', system-ui, sans-serif", size: 11 } },
    hovermode: 'closest'
  };
  Plotly.newPlot('scatter-plot', traces, layout, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d','select2d'] });
}
"""
