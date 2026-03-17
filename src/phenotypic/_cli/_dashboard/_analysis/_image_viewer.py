"""Overlay image viewer plugin for the Analysis tab."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from phenotypic.tools_.register import register_analysis

from ._base_plugin import BaseAnalysisPlugin

if TYPE_CHECKING:
    from ._prepare_context import AnalysisPrepareContext


@register_analysis
class ImageViewerPlugin(BaseAnalysisPlugin):
    """Overlay PNG viewer with per-image measurement data."""

    call_name = "images"
    display_name = "Image Viewer"
    sort_order = 30

    def prepare_data(self, ctx: AnalysisPrepareContext) -> None:
        """Write ``overlay_manifest.json`` by scanning overlay PNGs."""
        from .._analysis_helpers import write_json_atomic

        results_dir = ctx.output_dir / "results"
        datasets: Dict[str, List[str]] = {}

        if results_dir.is_dir():
            for dataset_dir in sorted(results_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                overlay_dir = dataset_dir / "overlays"
                if not overlay_dir.is_dir():
                    continue
                png_files = sorted(f.name for f in overlay_dir.glob("*.png"))
                if png_files:
                    datasets[dataset_dir.name] = png_files

        write_json_atomic(
            {"datasets": datasets},
            ctx.progress_dir / "overlay_manifest.json",
        )

    def css(self) -> str:
        """Return CSS scoped with the plugin's call_name prefix."""
        return (
            "/* Scoped: analysis-images plugin */\n"
            ".image-viewer-controls {\n"
            "  display: flex;\n"
            "  flex-wrap: wrap;\n"
            "  gap: var(--sp-3);\n"
            "  margin-bottom: var(--sp-4);\n"
            "  align-items: flex-end;\n"
            "}\n"
            ".image-viewer-controls select {\n"
            "  background: var(--color-white);\n"
            "  border: 1.5px solid var(--color-border);\n"
            "  border-radius: var(--radius);\n"
            "  padding: 0.4rem 0.6rem;\n"
            "  font-family: var(--font-mono);\n"
            "  font-size: var(--text-xs);\n"
            "  color: var(--color-body);\n"
            "  min-width: 200px;\n"
            "}\n"
            ".image-viewer-layout {\n"
            "  display: grid;\n"
            "  grid-template-columns: 1fr 1fr;\n"
            "  gap: var(--sp-4);\n"
            "}\n"
            "@media (max-width: 900px) {\n"
            "  .image-viewer-layout { grid-template-columns: 1fr; }\n"
            "}\n"
            ".image-viewer-img {\n"
            "  width: 100%;\n"
            "  border: 1px solid var(--color-border);\n"
            "  border-radius: var(--radius);\n"
            "}\n"
            ".image-viewer-img img {\n"
            "  width: 100%;\n"
            "  height: auto;\n"
            "  display: block;\n"
            "  border-radius: var(--radius);\n"
            "}\n"
            ".image-viewer-data {\n"
            "  max-height: 500px;\n"
            "  overflow-y: auto;\n"
            "}\n"
        )

    def html(self) -> str:
        """Return HTML for the sub-tab body."""
        return (
            '<div id="analysis-images-container">'
            '<div class="analysis-empty">'
            "No data available yet."
            "</div>"
            "</div>"
        )

    def _js_empty_msg(self, text: str) -> str:
        """Return an analysis-empty div string for use in JS literals."""
        return '<div class=\\"analysis-empty\\">' + text + "</div>"

    def js(self) -> str:
        """Return JS including an ``initAnalysis_images()`` function."""
        no_data = self._js_empty_msg("No measurement data for this image.")
        empty_file = self._js_empty_msg("Empty measurement file.")
        load_error = self._js_empty_msg("Could not load measurement data.")

        return (
            "function initAnalysis_images() {\n"
            "  renderImageViewer();\n"
            "}\n"
            "\n"
            "function buildMeasurementTable(headers, rowValues) {\n"
            "  var html = '<table class=\"analysis-table\">"
            "<thead><tr>';\n"
            "  headers.forEach(function(h) {"
            " html += '<th>' + esc(h) + '</th>'; });\n"
            "  html += '</tr></thead><tbody>';\n"
            "  rowValues.forEach(function(cells) {\n"
            "    html += '<tr>';\n"
            "    cells.forEach(function(v) {\n"
            "      var display = v === null || v === undefined"
            " ? '' : String(v);\n"
            "      html += '<td>' + esc(display) + '</td>';\n"
            "    });\n"
            "    html += '</tr>';\n"
            "  });\n"
            "  html += '</tbody></table>';\n"
            "  return html;\n"
            "}\n"
            "\n"
            "function renderImageViewer() {\n"
            "  var container = document.getElementById("
            "'analysis-images-container');\n"
            "  var d = analysisData.overlay;\n"
            "  if (!d || !d.datasets"
            " || Object.keys(d.datasets).length === 0) {\n"
            "    container.innerHTML = '<div class=\"analysis-empty\">"
            "No overlay images available yet.</div>';\n"
            "    return;\n"
            "  }\n"
            "  var dsNames = Object.keys(d.datasets);\n"
            "  var html = '<div class=\"image-viewer-controls\">';\n"
            "  html += '<div class=\"scatter-control-group\">"
            "<label>Dataset</label>"
            "<select id=\"iv-dataset\" onchange=\"updateImageList()\">';\n"
            "  dsNames.forEach(function(ds) {"
            " html += '<option value=\"' + esc(ds) + '\">' + esc(ds)"
            " + '</option>'; });\n"
            "  html += '</select></div>';\n"
            "  html += '<div class=\"scatter-control-group\">"
            "<label>Image</label>"
            "<select id=\"iv-image\" onchange=\"loadOverlayImage()\">"
            "</select></div>';\n"
            "  html += '</div>';\n"
            "  html += '<div class=\"image-viewer-layout\">';\n"
            "  html += '<div class=\"image-viewer-img\" id=\"iv-img-wrap\">"
            "<div class=\"analysis-empty\">Select an image above.</div>"
            "</div>';\n"
            "  html += '<div class=\"image-viewer-data\" id=\"iv-data-wrap\">"
            "<div class=\"analysis-empty\">"
            "Select an image to view measurements.</div>"
            "</div>';\n"
            "  html += '</div>';\n"
            "  container.innerHTML = html;\n"
            "  updateImageList();\n"
            "}\n"
            "\n"
            "function updateImageList() {\n"
            "  var d = analysisData.overlay;\n"
            "  if (!d) return;\n"
            "  var ds = document.getElementById('iv-dataset').value;\n"
            "  var images = d.datasets[ds] || [];\n"
            "  var sel = document.getElementById('iv-image');\n"
            "  sel.innerHTML = images.map(function(img) {"
            " return '<option value=\"' + esc(img) + '\">' + esc(img)"
            " + '</option>'; }).join('');\n"
            "  if (images.length > 0) loadOverlayImage();\n"
            "}\n"
            "\n"
            "function measurementPath(ds, filename) {\n"
            "  return 'results/' + encodeURIComponent(ds)"
            " + '/measurements/' + encodeURIComponent(filename);\n"
            "}\n"
            "\n"
            "function loadOverlayImage() {\n"
            "  var ds = document.getElementById('iv-dataset').value;\n"
            "  var img = document.getElementById('iv-image').value;\n"
            "  if (!ds || !img) return;\n"
            "  var imgWrap = document.getElementById('iv-img-wrap');\n"
            "  var overlayPath = 'results/' + encodeURIComponent(ds)"
            " + '/overlays/' + encodeURIComponent(img);\n"
            "  imgWrap.innerHTML = '<img src=\"' + overlayPath"
            " + '\" alt=\"' + esc(img)"
            " + '\" onerror=\"this.parentElement.innerHTML="
            "\\'<div class=analysis-empty>Image not found.</div>\\'\">';\n"
            "  var dataWrap = document.getElementById('iv-data-wrap');\n"
            "  if (window.hyparquet) {\n"
            "    loadMeasurementsParquet(dataWrap, ds, img);\n"
            "  } else {\n"
            "    loadMeasurementsCsv(dataWrap, ds, img);\n"
            "  }\n"
            "}\n"
            "\n"
            "function loadMeasurementsParquet(dataWrap, ds, img) {\n"
            "  var pqName = img.replace(/\\.png$/i, '.parquet');\n"
            "  var pqPath = measurementPath(ds, pqName);\n"
            "  fetch(pqPath + '?' + Date.now()).then(function(resp) {\n"
            "    if (!resp.ok) {\n"
            "      dataWrap.innerHTML = '" + no_data + "';\n"
            "      return;\n"
            "    }\n"
            "    return resp.arrayBuffer();\n"
            "  }).then(function(buf) {\n"
            "    if (!buf) return;\n"
            "    hyparquet.parquetRead({\n"
            "      file: {\n"
            "        byteLength: buf.byteLength,\n"
            "        slice: function(start, end) {"
            " return buf.slice(start, end); }\n"
            "      },\n"
            "      onComplete: function(rows) {\n"
            "        if (rows.length === 0) {\n"
            "          dataWrap.innerHTML = '" + empty_file + "';\n"
            "          return;\n"
            "        }\n"
            "        var headers = Object.keys(rows[0]);\n"
            "        var rowValues = rows.map(function(row) {\n"
            "          return headers.map(function(h) {"
            " return row[h]; });\n"
            "        });\n"
            "        dataWrap.innerHTML ="
            " buildMeasurementTable(headers, rowValues);\n"
            "      }\n"
            "    });\n"
            "  }).catch(function() {\n"
            "    dataWrap.innerHTML = '" + load_error + "';\n"
            "  });\n"
            "}\n"
            "\n"
            "function loadMeasurementsCsv(dataWrap, ds, img) {\n"
            "  var csvName = img.replace(/\\.png$/i, '.csv');\n"
            "  var csvPath = measurementPath(ds, csvName);\n"
            "  fetch(csvPath + '?' + Date.now()).then(function(resp) {\n"
            "    if (!resp.ok) {\n"
            "      dataWrap.innerHTML = '" + no_data + "';\n"
            "      return;\n"
            "    }\n"
            "    return resp.text();\n"
            "  }).then(function(text) {\n"
            "    if (!text) return;\n"
            "    var lines = text.trim().split('\\n');\n"
            "    if (lines.length < 2) {\n"
            "      dataWrap.innerHTML = '" + empty_file + "';\n"
            "      return;\n"
            "    }\n"
            "    var headers = lines[0].split(',').map(function(h) {"
            " return h.trim(); });\n"
            "    var rowValues = [];\n"
            "    for (var r = 1; r < lines.length; r++) {\n"
            "      rowValues.push(lines[r].split(',').map(function(c) {"
            " return c.trim(); }));\n"
            "    }\n"
            "    dataWrap.innerHTML ="
            " buildMeasurementTable(headers, rowValues);\n"
            "  }).catch(function() {\n"
            "    dataWrap.innerHTML = '" + load_error + "';\n"
            "  });\n"
            "}\n"
        )
