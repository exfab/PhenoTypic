"""
HTML report generator for the PhenoTypic CLI.

Generates visual HTML reports showing processing results and failures
with collapsible tracebacks and dataset summaries.
"""

from __future__ import annotations

from pathlib import Path

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
<html lang="en">
<head>
    <title>PhenoTypic Processing Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap" rel="stylesheet">
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
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        :root {
            --color-navy: #003660;
            --color-blue: #1b75bc;
            --color-gold: #febc11;
            --color-white: #ffffff;
            --color-bg: #f5f7fa;
            --color-surface: #ffffff;
            --color-border: #dde3ed;
            --color-rule: #e8ecf2;
            --color-muted: #8892a4;
            --color-body: #2e3a4e;
            --color-heading: #003660;

            --oi-green: #009E73;
            --oi-vermilion: #D55E00;
            --oi-sky: #56B4E9;

            --font-display: 'DM Serif Display', Georgia, serif;
            --font-body: 'DM Sans', system-ui, sans-serif;
            --font-mono: 'DM Mono', 'Courier New', monospace;

            --text-xs: 0.6875rem;
            --text-sm: 0.8125rem;
            --text-base: 0.9375rem;
            --text-lg: 1.25rem;
            --text-xl: 1.5rem;
            --text-2xl: 1.875rem;
            --text-3xl: 2.5rem;

            --sp-1: 0.25rem;
            --sp-2: 0.5rem;
            --sp-3: 0.75rem;
            --sp-4: 1rem;
            --sp-5: 1.25rem;
            --sp-6: 1.5rem;
            --sp-8: 2rem;
            --sp-10: 2.5rem;

            --radius-sm: 3px;
            --radius: 6px;
            --radius-md: 10px;

            --shadow-sm: 0 1px 3px rgba(0,54,96,0.07), 0 1px 2px rgba(0,54,96,0.04);
            --shadow: 0 4px 12px rgba(0,54,96,0.08), 0 1px 3px rgba(0,54,96,0.05);
        }

        body {
            font-family: var(--font-body);
            line-height: 1.6;
            color: var(--color-body);
            background: var(--color-bg);
            padding: var(--sp-5);
            font-size: var(--text-base);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: var(--color-surface);
            padding: var(--sp-10);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-sm);
        }

        h1 {
            font-family: var(--font-display);
            color: var(--color-heading);
            border-bottom: 3px solid var(--color-navy);
            padding-bottom: var(--sp-4);
            margin-bottom: var(--sp-8);
            font-size: var(--text-3xl);
            font-weight: 400;
        }

        h2 {
            font-family: var(--font-display);
            color: var(--color-heading);
            margin-top: var(--sp-10);
            margin-bottom: var(--sp-5);
            font-size: var(--text-2xl);
            font-weight: 400;
            border-left: 4px solid var(--color-navy);
            padding-left: var(--sp-4);
        }

        h3 {
            font-family: var(--font-display);
            color: var(--color-heading);
            margin: var(--sp-6) 0 var(--sp-4) 0;
            font-size: var(--text-xl);
            font-weight: 400;
        }

        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: var(--sp-5);
            margin: var(--sp-8) 0;
        }

        .stat-card {
            position: relative;
            overflow: hidden;
            background: var(--color-surface);
            border: 1px solid var(--color-border);
            padding: var(--sp-6);
            border-radius: var(--radius-md);
            text-align: center;
            box-shadow: var(--shadow-sm);
            transition: border-color 180ms cubic-bezier(0.22, 1, 0.36, 1);
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: var(--color-navy);
        }

        .stat-card:hover {
            border-color: var(--color-blue);
        }

        .stat-card.success::before {
            background: var(--oi-green);
        }

        .stat-card.failure::before {
            background: var(--oi-vermilion);
        }

        .stat-card.info::before {
            background: var(--oi-sky);
        }

        .stat-card.rate::before {
            background: var(--color-gold);
        }

        .stat-value {
            font-family: var(--font-display);
            font-size: var(--text-3xl);
            font-weight: 400;
            color: var(--color-heading);
            margin-bottom: var(--sp-1);
        }

        .stat-label {
            font-family: var(--font-mono);
            font-size: var(--text-xs);
            color: var(--color-muted);
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .success-text { color: #006B4F; font-weight: 600; }
        .failure-text { color: #D55E00; font-weight: 600; }
        .warning-text { color: #9A6B00; font-weight: 600; }

        progress {
            width: 100%;
            height: 6px;
            border-radius: 9999px;
            overflow: hidden;
            border: none;
            background: var(--color-rule);
        }

        progress::-webkit-progress-bar {
            background: var(--color-rule);
            border-radius: 9999px;
        }

        progress::-webkit-progress-value {
            background: var(--color-navy);
            border-radius: 9999px;
        }

        progress::-moz-progress-bar {
            background: var(--color-navy);
            border-radius: 9999px;
        }

        .dataset {
            background: var(--color-bg);
            padding: var(--sp-6);
            border-radius: var(--radius-md);
            margin-bottom: var(--sp-6);
            border: 1px solid var(--color-border);
        }

        .progress-container {
            margin: var(--sp-5) 0;
        }

        .progress-label {
            margin-bottom: var(--sp-3);
            font-size: var(--text-sm);
            color: var(--color-body);
        }

        details {
            margin: var(--sp-4) 0;
            border: 1px solid var(--color-border);
            border-radius: var(--radius);
            overflow: hidden;
        }

        summary {
            cursor: pointer;
            padding: var(--sp-4);
            background: var(--color-bg);
            font-weight: 600;
            font-family: var(--font-body);
            color: var(--color-heading);
            user-select: none;
            transition: background 180ms;
        }

        summary:hover {
            background: var(--color-rule);
        }

        summary::-webkit-details-marker {
            display: none;
        }

        summary::before {
            content: '\\25b6';
            display: inline-block;
            margin-right: var(--sp-3);
            transition: transform 0.2s;
        }

        details[open] summary::before {
            transform: rotate(90deg);
        }

        .traceback {
            background: var(--color-bg);
            color: var(--color-body);
            padding: var(--sp-5);
            border-radius: var(--radius);
            border: 1px solid var(--color-border);
            overflow-x: auto;
            font-family: var(--font-mono);
            font-size: var(--text-xs);
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: var(--sp-3);
            line-height: 1.4;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: var(--sp-5) 0;
            background: var(--color-surface);
        }

        th, td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--color-rule);
        }

        th {
            background: transparent;
            color: var(--color-muted);
            font-family: var(--font-mono);
            font-size: 0.6875rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border-bottom: 2px solid var(--color-navy);
        }

        tr:hover {
            background: rgba(27,117,188,0.03);
        }

        tr:last-child td {
            border-bottom: none;
        }

        code {
            background: #edf2f7;
            color: var(--color-navy);
            padding: 1px 5px;
            border-radius: var(--radius-sm);
            font-family: var(--font-mono);
            font-size: 0.9em;
        }

        .timestamp {
            color: var(--color-muted);
            font-size: var(--text-sm);
        }

        .metadata {
            background: rgba(86,180,233,0.08);
            padding: var(--sp-4) var(--sp-5);
            border-radius: var(--radius);
            margin: var(--sp-5) 0;
            border-left: 4px solid var(--oi-sky);
            color: #0B5E87;
        }

        .metadata p {
            margin: var(--sp-1) 0;
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
            <div class="stat-card rate">
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
