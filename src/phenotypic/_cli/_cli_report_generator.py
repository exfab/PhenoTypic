"""
HTML report generator for the PhenoTypic CLI.

Generates visual HTML reports showing processing results and failures
with collapsible tracebacks and dataset summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ._cli_types import DatasetResults, ExecutionResults


class HTMLReportGenerator:
    """Generate HTML failure reports with collapsible tracebacks."""

    def generate_report(
        self, results: ExecutionResults, output_path: Path
    ) -> None:
        """
        Generate complete HTML report.

        Args:
            results: Execution results to report
            output_path: Path to save HTML file
        """
        html = self._generate_html(results)
        output_path.write_text(html)

    def _generate_html(self, results: ExecutionResults) -> str:
        """Generate HTML content."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>PhenoTypic Processing Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {self._get_styles()}
</head>
<body>
    <div class="container">
        <h1>PhenoTypic Processing Report</h1>
        {self._generate_summary(results)}
        {self._generate_dataset_details(results)}
    </div>
</body>
</html>"""

    def _get_styles(self) -> str:
        """Inline CSS for self-contained report."""
        return """<style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 
                         'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }

        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }

        h3 {
            color: #555;
            margin: 25px 0 15px 0;
            font-size: 1.4em;
        }

        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-card.success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }

        .stat-card.failure {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }

        .stat-card.info {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }

        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .success-text { color: #27ae60; font-weight: bold; }
        .failure-text { color: #e74c3c; font-weight: bold; }
        .warning-text { color: #f39c12; font-weight: bold; }

        progress {
            width: 100%;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            border: none;
            background: #ecf0f1;
        }

        progress::-webkit-progress-bar {
            background: #ecf0f1;
            border-radius: 15px;
        }

        progress::-webkit-progress-value {
            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
            border-radius: 15px;
        }

        progress::-moz-progress-bar {
            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
            border-radius: 15px;
        }

        .dataset {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 25px;
            border: 1px solid #dee2e6;
        }

        .progress-container {
            margin: 20px 0;
        }

        .progress-label {
            margin-bottom: 10px;
            font-size: 1.1em;
            color: #555;
        }

        details {
            margin: 15px 0;
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }

        summary {
            cursor: pointer;
            padding: 15px;
            background: #ecf0f1;
            font-weight: 600;
            user-select: none;
            transition: background 0.2s;
        }

        summary:hover {
            background: #d5dbdb;
        }

        summary::-webkit-details-marker {
            display: none;
        }

        summary::before {
            content: '▶';
            display: inline-block;
            margin-right: 10px;
            transition: transform 0.2s;
        }

        details[open] summary::before {
            transform: rotate(90deg);
        }

        .traceback {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 10px;
            line-height: 1.4;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }

        th {
            background: #34495e;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }

        tr:hover {
            background: #f8f9fa;
        }

        tr:last-child td {
            border-bottom: none;
        }

        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.9em;
        }

        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
        }

        .metadata {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }

        .metadata p {
            margin: 5px 0;
        }

        @media print {
            body {
                background: white;
            }
            .container {
                box-shadow: none;
            }
        }
    </style>"""

    def _generate_summary(self, results: ExecutionResults) -> str:
        """Generate summary dashboard."""
        success_rate = results.success_rate * 100
        duration_str = self._format_duration(results.duration)

        return f"""
        <div class="metadata">
            <p><strong>Execution Mode:</strong> {results.execution_mode.upper()}</p>
            <p><strong>Start Time:</strong> {results.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>End Time:</strong> {results.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Duration:</strong> {duration_str}</p>
        </div>
        
        <div class="summary">
            <div class="stat-card info">
                <div class="stat-value">{results.total_images}</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value">{results.total_completed}</div>
                <div class="stat-label">Completed</div>
            </div>
            <div class="stat-card failure">
                <div class="stat-value">{results.total_failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
        """

    def _generate_dataset_details(self, results: ExecutionResults) -> str:
        """Generate per-dataset breakdown."""
        if not results.datasets:
            return "<h2>Datasets</h2><p>No dataset information available (jobs may still be running).</p>"

        html_parts = ["<h2>Datasets</h2>"]

        for dataset_name, ds_result in results.datasets.items():
            html_parts.append(
                self._generate_dataset_section(dataset_name, ds_result)
            )

        return "\n".join(html_parts)

    def _generate_dataset_section(
        self, name: str, ds: DatasetResults
    ) -> str:
        """Generate section for one dataset."""
        success_rate = (
            (ds.completed / ds.total * 100) if ds.total > 0 else 0
        )

        display_name = name

        html = f"""
        <div class="dataset">
            <h3>{display_name}</h3>
            <div class="progress-container">
                <div class="progress-label">
                    {ds.completed}/{ds.total} successful 
                    <span class="{'success-text' if success_rate == 100 else 'warning-text' if success_rate > 0 else 'failure-text'}">
                        ({success_rate:.1f}%)
                    </span>
                </div>
                <progress value="{ds.completed}" max="{ds.total}"></progress>
            </div>
        """

        if ds.failures:
            html += f"""
            <details>
                <summary>Failed Images ({len(ds.failures)})</summary>
                <table>
                    <thead>
                        <tr>
                            <th>Image</th>
                            <th>Error Type</th>
                            <th>Error Message</th>
                            <th>Traceback</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for failure in ds.failures:
                html += f"""
                        <tr>
                            <td><strong>{failure.image_filename}</strong></td>
                            <td><code>{failure.error_type}</code></td>
                            <td>{failure.error_message}</td>
                            <td>
                                <details>
                                    <summary>View Traceback</summary>
                                    <pre class="traceback">{self._escape_html(failure.traceback)}</pre>
                                </details>
                            </td>
                        </tr>
                """

            html += """
                    </tbody>
                </table>
            </details>
            """
        else:
            html += '<p class="success-text">✓ All images processed successfully</p>'

        html += "</div>"
        return html

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hr"

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
