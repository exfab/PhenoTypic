"""Static HTML export for sweep results.

Generates a standalone HTML file for viewing and comparing sweep results
without requiring Python or Panel.
"""

from __future__ import annotations

from base64 import b64encode
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import json
import logging

try:
    from jinja2 import Template

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Template = None

try:
    from skimage import io as skio
    import numpy as np
    from PIL import Image
    import io

    IMAGE_LIBS_AVAILABLE = True
except ImportError:
    IMAGE_LIBS_AVAILABLE = False

if TYPE_CHECKING:
    from ..explorer._sweep_results import SweepResults, SweepResult

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def image_to_base64(path: Path) -> Optional[str]:
    """Convert image file to base64 data URI.

    Args:
        path: Path to image file.

    Returns:
        Base64 data URI string or None if loading fails.
    """
    if not IMAGE_LIBS_AVAILABLE:
        return None

    try:
        if not path.exists():
            return None

        # Read image
        img_array = skio.imread(str(path))

        # Convert to PIL Image
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        pil_img = Image.fromarray(img_array)

        # Convert to PNG bytes
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode to base64
        b64_data = b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"

    except Exception as e:
        logger.warning(f"Failed to convert image to base64: {path} - {e}")
        return None


def prepare_variant_data(
    result: "SweepResult",
    embed_images: bool = True,
) -> Dict[str, Any]:
    """Prepare variant data for template.

    Args:
        result: SweepResult to convert.
        embed_images: Whether to embed images as base64.

    Returns:
        Dictionary for template rendering.
    """
    data = {
        "variant_id": result.variant_id,
        "image_name": result.image_name,
        "success": result.success,
        "error": result.error,
        "execution_time": result.execution_time,
        "metrics": result.metrics,
        "config": result.pipeline_config,
        "outputs": {},
    }

    # Process outputs
    for view_name, path in result.outputs.items():
        if embed_images:
            data_uri = image_to_base64(Path(path))
            if data_uri:
                data["outputs"][view_name] = data_uri
            else:
                data["outputs"][view_name] = str(path)
        else:
            # Use relative path
            data["outputs"][view_name] = str(path)

    return data


def group_results_by_image(
    results: List["SweepResult"],
) -> Dict[str, List["SweepResult"]]:
    """Group results by image name.

    Args:
        results: List of results to group.

    Returns:
        Dictionary mapping image name to list of results.
    """
    groups = {}
    for result in results:
        groups.setdefault(result.image_name, []).append(result)
    return groups


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sweep Results - {{ title }}</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .header .meta {
            color: #666;
            font-size: 14px;
        }
        .controls {
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        .controls label {
            font-weight: 500;
        }
        .controls select {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .controls .keyboard-hint {
            margin-left: auto;
            color: #888;
            font-size: 13px;
        }
        .keyboard-hint kbd {
            background: #eee;
            padding: 2px 6px;
            border-radius: 3px;
            border: 1px solid #ddd;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .variant-card {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }
        .variant-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .variant-card.selected {
            outline: 3px solid #1976D2;
        }
        .variant-card .image-container {
            position: relative;
            padding-top: 75%;
            background: #f0f0f0;
        }
        .variant-card img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .variant-card .info {
            padding: 15px;
        }
        .variant-card .variant-id {
            font-weight: 600;
            margin-bottom: 5px;
        }
        .variant-card .metrics {
            font-size: 13px;
            color: #666;
        }
        .variant-card .error {
            color: #d32f2f;
            font-size: 13px;
        }
        .comparison {
            display: none;
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .comparison.active {
            display: block;
        }
        .comparison h2 {
            margin: 0 0 15px 0;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .comparison-panel {
            text-align: center;
        }
        .comparison-panel img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .comparison-panel .title {
            font-weight: 600;
            margin-bottom: 10px;
        }
        .comparison-panel .details {
            text-align: left;
            font-size: 13px;
            margin-top: 10px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .metrics-table {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-table h2 {
            margin: 0;
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f9f9f9;
            font-weight: 600;
            cursor: pointer;
        }
        th:hover {
            background: #f0f0f0;
        }
        tr:hover {
            background: #f9f9f9;
        }
        .success { color: #4CAF50; }
        .failure { color: #d32f2f; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <div class="meta">
            Created: {{ created }} |
            Total variants: {{ total_variants }} |
            Successful: {{ successful_count }} |
            Failed: {{ failed_count }}
        </div>
    </div>

    <div class="controls">
        <label>Image:</label>
        <select id="imageSelect" onchange="filterByImage()">
            {% for image_name in image_names %}
            <option value="{{ image_name }}">{{ image_name }}</option>
            {% endfor %}
        </select>

        <label>View:</label>
        <select id="viewSelect" onchange="changeView()">
            {% for view in views %}
            <option value="{{ view }}" {% if view == default_view %}selected{% endif %}>{{ view }}</option>
            {% endfor %}
        </select>

        <label>
            <input type="checkbox" id="showComparison" onchange="toggleComparison()">
            Compare Mode
        </label>

        <div class="keyboard-hint">
            <kbd>←</kbd><kbd>→</kbd> Navigate |
            <kbd>1</kbd>-<kbd>9</kbd> Quick select
        </div>
    </div>

    <div class="comparison" id="comparisonPanel">
        <h2>Comparison</h2>
        <div class="comparison-grid">
            <div class="comparison-panel" id="compareA">
                <div class="title">Variant A</div>
                <img id="imgA" src="" alt="Variant A">
                <div class="details" id="detailsA"></div>
            </div>
            <div class="comparison-panel" id="compareB">
                <div class="title">Variant B</div>
                <img id="imgB" src="" alt="Variant B">
                <div class="details" id="detailsB"></div>
            </div>
        </div>
    </div>

    <div class="grid" id="variantGrid">
        {% for variant in variants %}
        <div class="variant-card"
             data-variant-id="{{ variant.variant_id }}"
             data-image="{{ variant.image_name }}"
             data-success="{{ variant.success|lower }}"
             onclick="selectVariant('{{ variant.variant_id }}', '{{ variant.image_name }}')">
            <div class="image-container">
                {% if variant.outputs %}
                <img class="variant-image"
                     {% for view, src in variant.outputs.items() %}
                     data-{{ view }}="{{ src }}"
                     {% endfor %}
                     src="{{ variant.outputs.values()|first }}"
                     alt="{{ variant.variant_id }}">
                {% else %}
                <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#999;">
                    No output
                </div>
                {% endif %}
            </div>
            <div class="info">
                <div class="variant-id">{{ variant.variant_id }}</div>
                {% if variant.success %}
                <div class="metrics">
                    {% for key, value in variant.metrics.items() %}
                    {{ key }}: {{ "%.2f"|format(value) if value is number else value }}{% if not loop.last %}, {% endif %}
                    {% endfor %}
                </div>
                {% else %}
                <div class="error">Failed: {{ variant.error }}</div>
                {% endif %}
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="metrics-table">
        <h2>Metrics Summary</h2>
        <table id="metricsTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Variant</th>
                    <th onclick="sortTable(1)">Image</th>
                    <th onclick="sortTable(2)">Status</th>
                    <th onclick="sortTable(3)">Time (s)</th>
                    {% for metric_name in metric_names %}
                    <th onclick="sortTable({{ loop.index + 3 }})">{{ metric_name }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for variant in variants %}
                <tr data-image="{{ variant.image_name }}">
                    <td>{{ variant.variant_id }}</td>
                    <td>{{ variant.image_name }}</td>
                    <td class="{{ 'success' if variant.success else 'failure' }}">
                        {{ 'Success' if variant.success else 'Failed' }}
                    </td>
                    <td>{{ "%.3f"|format(variant.execution_time) }}</td>
                    {% for metric_name in metric_names %}
                    <td>
                        {% if metric_name in variant.metrics %}
                        {{ "%.4f"|format(variant.metrics[metric_name]) if variant.metrics[metric_name] is number else variant.metrics[metric_name] }}
                        {% else %}
                        -
                        {% endif %}
                    </td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <script>
        // Data from Python
        const variantsData = {{ variants_json|safe }};
        let selectedVariants = [];
        let currentView = '{{ default_view }}';
        let currentImage = '{{ image_names[0] if image_names else "" }}';

        function filterByImage() {
            currentImage = document.getElementById('imageSelect').value;
            const cards = document.querySelectorAll('.variant-card');
            cards.forEach(card => {
                if (card.dataset.image === currentImage) {
                    card.style.display = '';
                } else {
                    card.style.display = 'none';
                }
            });

            // Filter table rows too
            const rows = document.querySelectorAll('#metricsTable tbody tr');
            rows.forEach(row => {
                if (row.dataset.image === currentImage) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        }

        function changeView() {
            currentView = document.getElementById('viewSelect').value;
            const images = document.querySelectorAll('.variant-image');
            images.forEach(img => {
                const src = img.dataset[currentView];
                if (src) {
                    img.src = src;
                }
            });
            updateComparison();
        }

        function selectVariant(variantId, imageName) {
            const card = document.querySelector(
                `.variant-card[data-variant-id="${variantId}"][data-image="${imageName}"]`
            );

            if (document.getElementById('showComparison').checked) {
                // Comparison mode - select up to 2
                if (selectedVariants.length >= 2) {
                    // Deselect first
                    const firstCard = document.querySelector(
                        `.variant-card[data-variant-id="${selectedVariants[0].id}"][data-image="${selectedVariants[0].image}"]`
                    );
                    if (firstCard) firstCard.classList.remove('selected');
                    selectedVariants.shift();
                }
                selectedVariants.push({id: variantId, image: imageName});
                card.classList.add('selected');
                updateComparison();
            } else {
                // Single selection mode
                document.querySelectorAll('.variant-card.selected').forEach(c => {
                    c.classList.remove('selected');
                });
                card.classList.add('selected');
                selectedVariants = [{id: variantId, image: imageName}];
            }
        }

        function toggleComparison() {
            const panel = document.getElementById('comparisonPanel');
            if (document.getElementById('showComparison').checked) {
                panel.classList.add('active');
            } else {
                panel.classList.remove('active');
            }
        }

        function updateComparison() {
            if (selectedVariants.length >= 1) {
                const v = selectedVariants[0];
                const data = variantsData.find(
                    d => d.variant_id === v.id && d.image_name === v.image
                );
                if (data && data.outputs && data.outputs[currentView]) {
                    document.getElementById('imgA').src = data.outputs[currentView];
                    document.getElementById('detailsA').innerHTML = formatDetails(data);
                }
            }
            if (selectedVariants.length >= 2) {
                const v = selectedVariants[1];
                const data = variantsData.find(
                    d => d.variant_id === v.id && d.image_name === v.image
                );
                if (data && data.outputs && data.outputs[currentView]) {
                    document.getElementById('imgB').src = data.outputs[currentView];
                    document.getElementById('detailsB').innerHTML = formatDetails(data);
                }
            }
        }

        function formatDetails(data) {
            let html = '<strong>Config:</strong><br>';
            for (const [key, value] of Object.entries(data.config || {})) {
                html += `${key}: ${value}<br>`;
            }
            html += '<br><strong>Metrics:</strong><br>';
            for (const [key, value] of Object.entries(data.metrics || {})) {
                const formatted = typeof value === 'number' ? value.toFixed(4) : value;
                html += `${key}: ${formatted}<br>`;
            }
            html += `<br>Time: ${data.execution_time.toFixed(3)}s`;
            return html;
        }

        function sortTable(columnIndex) {
            const table = document.getElementById('metricsTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            rows.sort((a, b) => {
                const aVal = a.cells[columnIndex].textContent.trim();
                const bVal = b.cells[columnIndex].textContent.trim();

                // Try numeric comparison
                const aNum = parseFloat(aVal);
                const bNum = parseFloat(bVal);
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return bNum - aNum;  // Descending
                }
                return aVal.localeCompare(bVal);
            });

            rows.forEach(row => tbody.appendChild(row));
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            const visibleCards = Array.from(
                document.querySelectorAll('.variant-card')
            ).filter(c => c.style.display !== 'none');

            if (visibleCards.length === 0) return;

            const selectedCard = document.querySelector('.variant-card.selected');
            let currentIndex = selectedCard ? visibleCards.indexOf(selectedCard) : -1;

            if (e.key === 'ArrowRight') {
                currentIndex = (currentIndex + 1) % visibleCards.length;
                const card = visibleCards[currentIndex];
                selectVariant(card.dataset.variantId, card.dataset.image);
            } else if (e.key === 'ArrowLeft') {
                currentIndex = currentIndex <= 0 ? visibleCards.length - 1 : currentIndex - 1;
                const card = visibleCards[currentIndex];
                selectVariant(card.dataset.variantId, card.dataset.image);
            } else if (e.key >= '1' && e.key <= '9') {
                const idx = parseInt(e.key) - 1;
                if (idx < visibleCards.length) {
                    const card = visibleCards[idx];
                    selectVariant(card.dataset.variantId, card.dataset.image);
                }
            }
        });

        // Initial filter
        filterByImage();
    </script>
</body>
</html>
"""


# =============================================================================
# SweepHTMLExporter
# =============================================================================


class SweepHTMLExporter:
    """Generate static HTML viewer for sweep results.

    Creates a standalone HTML file with:
    - Grid view of all variants
    - Side-by-side comparison mode
    - Keyboard navigation (arrow keys, number keys)
    - Sortable metrics table
    - Image/view filtering

    Args:
        results: SweepResults to export.
        embed_images: Whether to embed images as base64 (default True).
            Set to False to keep images as file references.

    Examples:
        Basic export:

        >>> from phenotypic.gui.viewer import SweepHTMLExporter
        >>> from phenotypic.gui.explorer import SweepResults
        >>> results = SweepResults.load_manifest('./results/manifest.json')
        >>> exporter = SweepHTMLExporter(results)
        >>> output_path = exporter.export('./results/viewer.html')

        Without embedding images:

        >>> exporter = SweepHTMLExporter(results, embed_images=False)
        >>> exporter.export()  # Saves to results directory
    """

    def __init__(
        self,
        results: "SweepResults",
        embed_images: bool = True,
    ):
        """Initialize the exporter.

        Args:
            results: SweepResults to export.
            embed_images: Whether to embed images as base64.
        """
        if not JINJA2_AVAILABLE:
            raise ImportError(
                "SweepHTMLExporter requires Jinja2. "
                "Install with: pip install jinja2"
            )

        self.results = results
        self.embed_images = embed_images

    def export(self, output_path: Optional[Path] = None) -> Path:
        """Generate HTML viewer.

        Args:
            output_path: Path for output HTML file. Defaults to
                results_dir/viewer.html.

        Returns:
            Path to generated HTML file.
        """
        if output_path is None:
            output_path = self.results.sweep_dir / "viewer.html"
        else:
            output_path = Path(output_path)

        # Prepare variant data
        variants = []
        all_metrics = set()
        all_views = set()

        for result in self.results.results:
            variant_data = prepare_variant_data(result, self.embed_images)
            variants.append(variant_data)
            all_metrics.update(result.metrics.keys())
            all_views.update(result.outputs.keys())

        # Get unique image names
        image_names = sorted(set(r.image_name for r in self.results.results))

        # Determine default view
        default_view = "overlay" if "overlay" in all_views else (
            sorted(all_views)[0] if all_views else ""
        )

        # Build template context
        context = {
            "title": self.results.sweep_dir.name,
            "created": self.results.created.strftime("%Y-%m-%d %H:%M:%S"),
            "total_variants": len(self.results.results),
            "successful_count": len(self.results.successful),
            "failed_count": len(self.results.failed),
            "variants": variants,
            "variants_json": json.dumps(variants),
            "image_names": image_names,
            "views": sorted(all_views),
            "default_view": default_view,
            "metric_names": sorted(all_metrics),
        }

        # Render template
        template = Template(HTML_TEMPLATE)
        html_content = template.render(**context)

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")

        logger.info(f"Exported HTML viewer to {output_path}")
        return output_path

    def get_html(self) -> str:
        """Get HTML content without writing to file.

        Returns:
            Complete HTML string.
        """
        # Prepare data (same as export)
        variants = []
        all_metrics = set()
        all_views = set()

        for result in self.results.results:
            variant_data = prepare_variant_data(result, self.embed_images)
            variants.append(variant_data)
            all_metrics.update(result.metrics.keys())
            all_views.update(result.outputs.keys())

        image_names = sorted(set(r.image_name for r in self.results.results))
        default_view = "overlay" if "overlay" in all_views else (
            sorted(all_views)[0] if all_views else ""
        )

        context = {
            "title": self.results.sweep_dir.name,
            "created": self.results.created.strftime("%Y-%m-%d %H:%M:%S"),
            "total_variants": len(self.results.results),
            "successful_count": len(self.results.successful),
            "failed_count": len(self.results.failed),
            "variants": variants,
            "variants_json": json.dumps(variants),
            "image_names": image_names,
            "views": sorted(all_views),
            "default_view": default_view,
            "metric_names": sorted(all_metrics),
        }

        template = Template(HTML_TEMPLATE)
        return template.render(**context)
