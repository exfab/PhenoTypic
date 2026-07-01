"""Overlay image viewer plugin for the Analysis tab."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from phenotypic.schema import EXPERIMENT_METADATA, METADATA
from phenotypic.sdk_.register import register_analysis
from phenotypic.sdk_ import overlays_dir, overlay_manifest_path

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
        """Write ``overlay_manifest.json`` by scanning the overlay package."""
        from .._analysis_helpers import write_json_atomic

        package_dir = overlays_dir(ctx.output_dir)
        datasets: Dict[str, List[str]] = {}

        if package_dir.is_dir():
            for dataset_dir in sorted(package_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                png_files = sorted(f.name for f in dataset_dir.glob("*.png"))
                if png_files:
                    datasets[dataset_dir.name] = png_files

        write_json_atomic(
            {"datasets": datasets},
            overlay_manifest_path(ctx.output_dir),
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
            "function loadOverlayImage() {\n"
            "  var ds = document.getElementById('iv-dataset').value;\n"
            "  var img = document.getElementById('iv-image').value;\n"
            "  if (!ds || !img) return;\n"
            "  var imgWrap = document.getElementById('iv-img-wrap');\n"
            "  var overlayPath = 'overlays/' + encodeURIComponent(ds)"
            " + '/' + encodeURIComponent(img);\n"
            "  imgWrap.innerHTML = '<img src=\"' + overlayPath"
            " + '\" alt=\"' + esc(img)"
            " + '\" onerror=\"this.parentElement.innerHTML="
            "\\'<div class=analysis-empty>Image not found.</div>\\'\">';\n"
            "  var dataWrap = document.getElementById('iv-data-wrap');\n"
            "  loadMeasurementsParquet(dataWrap, ds, img);\n"
            "}\n"
            "\n"
            "function loadMeasurementsParquet(dataWrap, ds, img) {\n"
            "  var imageStem = img.replace(/\\.[^.]+$/, '');\n"
            "  loadSharedParquet().then(function() {\n"
            "    var d = sharedParquetState.allData;\n"
            "    var cols = sharedParquetState.allColumns;\n"
            "    if (!d || cols.length === 0) {\n"
            "      dataWrap.innerHTML = '" + no_data + "';\n"
            "      return;\n"
            "    }\n"
            "    var dsCol = d['" + str(EXPERIMENT_METADATA.DATASET) + "'] || [];\n"
            "    var imgCol = d['" + str(METADATA.IMAGE_NAME) + "'] || [];\n"
            "    var matchIdx = [];\n"
            "    for (var i = 0; i < dsCol.length; i++) {\n"
            "      if (dsCol[i] === ds && imgCol[i] === imageStem)"
            " matchIdx.push(i);\n"
            "    }\n"
            "    if (matchIdx.length === 0) {\n"
            "      dataWrap.innerHTML = '" + no_data + "';\n"
            "      return;\n"
            "    }\n"
            "    var displayCols = cols.filter(function(c) {\n"
            "      return c !== '" + str(EXPERIMENT_METADATA.DATASET) + "'"
            " && c !== '" + str(METADATA.IMAGE_NAME) + "'"
            " && c !== 'filename';\n"
            "    });\n"
            "    var headers = displayCols;\n"
            "    var rowValues = matchIdx.map(function(idx) {\n"
            "      return displayCols.map(function(c) {\n"
            "        var val = d[c] ? d[c][idx] : null;\n"
            "        if (typeof val === 'number' && val.toFixed)"
            " val = val.toFixed(4);\n"
            "        return val != null ? val : '';\n"
            "      });\n"
            "    });\n"
            "    dataWrap.innerHTML ="
            " buildMeasurementTable(headers, rowValues);\n"
            "  }).catch(function() {\n"
            "    dataWrap.innerHTML = '" + load_error + "';\n"
            "  });\n"
            "}\n"
        )
