"""
Tests for the analysis plugin registry and dashboard analysis tab integration.

Covers:
- Plugin registration and discovery via AnalysisPluginRegistry
- Plugin sort order and metadata correctness
- Plugin CSS/HTML/JS output
- Dynamic sub-tab generation in dashboard HTML
- Analysis data sidecar file writing
- Stratified sampling
- Overlay manifest preparation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that cleans up after each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ──────────────────────────────────────────────────────────────────────
# Plugin Registry Tests
# ──────────────────────────────────────────────────────────────────────


class TestAnalysisPluginRegistry:

    def test_all_plugins_registered(self):
        """All four plugins should be registered."""
        from phenotypic.tools_.register import available_analysis_plugins

        # Trigger registration
        from phenotypic._cli._dashboard import _analysis  # noqa: F401

        names = available_analysis_plugins()
        assert "table" in names
        assert "scatter" in names
        assert "stats" in names
        assert "images" in names

    def test_plugin_sort_order(self):
        """Plugins should sort in the correct order by sort_order."""
        from phenotypic.tools_.register import AnalysisPluginRegistry

        # Trigger registration
        from phenotypic._cli._dashboard import _analysis  # noqa: F401

        plugins = []
        for name in AnalysisPluginRegistry.available():
            plugins.append(AnalysisPluginRegistry.get(name)())
        plugins.sort(key=lambda p: p.sort_order)
        assert [p.call_name for p in plugins] == [
            "table",
            "scatter",
            "stats",
            "images",
        ]

    def test_plugin_returns_strings(self):
        """Each plugin's css(), html(), js() should return non-empty strings."""
        from phenotypic.tools_.register import AnalysisPluginRegistry

        # Trigger registration
        from phenotypic._cli._dashboard import _analysis  # noqa: F401

        for name in AnalysisPluginRegistry.available():
            plugin = AnalysisPluginRegistry.get(name)()
            assert isinstance(plugin.css(), str), f"{name}.css() not a string"
            assert isinstance(plugin.html(), str), f"{name}.html() not a string"
            assert isinstance(plugin.js(), str), f"{name}.js() not a string"
            assert len(plugin.css()) > 0, f"{name}.css() is empty"
            assert len(plugin.html()) > 0, f"{name}.html() is empty"
            assert len(plugin.js()) > 0, f"{name}.js() is empty"

    def test_plugin_js_has_init_function(self):
        """Each plugin's JS should define an initAnalysis_{call_name} function."""
        from phenotypic.tools_.register import AnalysisPluginRegistry

        # Trigger registration
        from phenotypic._cli._dashboard import _analysis  # noqa: F401

        for name in AnalysisPluginRegistry.available():
            plugin = AnalysisPluginRegistry.get(name)()
            assert f"initAnalysis_{plugin.call_name}" in plugin.js(), (
                f"Plugin {name} missing initAnalysis_{plugin.call_name} in js()"
            )

    def test_plugin_display_names(self):
        """Plugins should have non-empty display names."""
        from phenotypic.tools_.register import AnalysisPluginRegistry

        # Trigger registration
        from phenotypic._cli._dashboard import _analysis  # noqa: F401

        for name in AnalysisPluginRegistry.available():
            plugin = AnalysisPluginRegistry.get(name)()
            assert plugin.display_name, f"Plugin {name} has empty display_name"
            assert isinstance(
                plugin.display_name, str
            ), f"Plugin {name} display_name not a string"


# ──────────────────────────────────────────────────────────────────────
# Dashboard HTML Tests
# ──────────────────────────────────────────────────────────────────────


class TestDashboardAnalysisTab:

    def test_analysis_tab_in_html(self, tmp_dir):
        """Dashboard should contain the analysis tab."""
        from phenotypic._cli._dashboard import generate_dashboard

        generate_dashboard(tmp_dir)
        html = (tmp_dir / "dashboard.html").read_text()
        assert "tab-analysis" in html
        assert "switchTab" in html

    def test_sub_tabs_present(self, tmp_dir):
        """All four plugin sub-tabs should be present in the HTML."""
        from phenotypic._cli._dashboard import generate_dashboard

        generate_dashboard(tmp_dir)
        html = (tmp_dir / "dashboard.html").read_text()
        assert "subtab-table" in html
        assert "subtab-scatter" in html
        assert "subtab-stats" in html
        assert "subtab-images" in html

    def test_sub_tab_buttons_present(self, tmp_dir):
        """Sub-tab buttons should be generated with correct display names."""
        from phenotypic._cli._dashboard import generate_dashboard

        generate_dashboard(tmp_dir)
        html = (tmp_dir / "dashboard.html").read_text()
        assert "Raw Table" in html
        assert "Scatter Plot" in html
        assert "Statistics" in html
        assert "Image Viewer" in html

    def test_first_sub_tab_active(self, tmp_dir):
        """The first sub-tab (table) should have the active class."""
        from phenotypic._cli._dashboard import generate_dashboard

        generate_dashboard(tmp_dir)
        html = (tmp_dir / "dashboard.html").read_text()
        # The first sub-tab button should be active
        assert 'sub-tab-btn active' in html
        # The first sub-tab content should be active
        assert 'sub-tab-content active' in html

    def test_dynamic_dispatch_in_js(self, tmp_dir):
        """renderSubTab should use dynamic dispatch, not hardcoded if/else."""
        from phenotypic._cli._dashboard import generate_dashboard

        generate_dashboard(tmp_dir)
        html = (tmp_dir / "dashboard.html").read_text()
        assert "window['initAnalysis_' + tabId]" in html
        # Old hardcoded dispatch should be gone
        assert "if (tabId === 'table') renderRawTable" not in html

    def test_plugin_css_in_html(self, tmp_dir):
        """Plugin-specific CSS should be included in the dashboard."""
        from phenotypic._cli._dashboard import generate_dashboard

        generate_dashboard(tmp_dir)
        html = (tmp_dir / "dashboard.html").read_text()
        # CSS from plugins
        assert "analysis-table" in html
        assert "scatter-controls" in html
        assert "stats-table" in html
        assert "image-viewer-controls" in html

    def test_framework_css_in_html(self, tmp_dir):
        """Framework-level CSS should be included in the dashboard."""
        from phenotypic._cli._dashboard import generate_dashboard

        generate_dashboard(tmp_dir)
        html = (tmp_dir / "dashboard.html").read_text()
        assert "analysis-container" in html
        assert "analysis-banner" in html
        assert "sub-tab-btn" in html
        assert "analysis-empty" in html
        assert "analysis-sample-label" in html

    def test_plugin_js_functions_in_html(self, tmp_dir):
        """Plugin JS init functions should be present in the dashboard."""
        from phenotypic._cli._dashboard import generate_dashboard

        generate_dashboard(tmp_dir)
        html = (tmp_dir / "dashboard.html").read_text()
        assert "initAnalysis_table" in html
        assert "initAnalysis_scatter" in html
        assert "initAnalysis_stats" in html
        assert "initAnalysis_images" in html

    def test_plotly_sidecar_written(self, tmp_dir):
        """Plotly.js sidecar should be written to progress/ dir."""
        from phenotypic._cli._dashboard import generate_dashboard

        generate_dashboard(tmp_dir)
        plotly_path = tmp_dir / "progress" / "plotly.min.js"
        assert plotly_path.exists()
        assert plotly_path.stat().st_size > 1000  # Should be >3MB

    def test_existing_tabs_preserved(self, tmp_dir):
        """Progress, README, and Download tabs should not be broken."""
        from phenotypic._cli._dashboard import generate_dashboard

        generate_dashboard(tmp_dir)
        html = (tmp_dir / "dashboard.html").read_text()
        assert "tab-progress" in html
        assert "tab-readme" in html
        assert "tab-download" in html
        assert "progress/manifest.json" in html
        assert "progress/failures.jsonl" in html

    def test_analysis_data_version_in_manifest(self, tmp_dir):
        """Manifest should contain analysis_data_version field."""
        from datetime import datetime

        from phenotypic._cli._cli_update_state import append_event
        from phenotypic._cli._dashboard._manifest_builder import build_manifest

        progress_dir = tmp_dir / "progress"
        progress_dir.mkdir()
        event_log = tmp_dir / "processing_events.log"
        append_event(event_log, "plate1", "img001.tif", "started")
        append_event(event_log, "plate1", "img001.tif", "completed")

        build_manifest(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            datasets={"plate1": 1},
            execution_mode="local",
            start_time=datetime.now().isoformat(timespec="milliseconds"),
        )
        manifest = json.loads((progress_dir / "manifest.json").read_text())
        assert "analysis_data_version" in manifest


# ──────────────────────────────────────────────────────────────────────
# Data Layer Tests
# ──────────────────────────────────────────────────────────────────────


class TestAnalysisData:

    def test_write_sidecar_no_data(self, tmp_dir):
        """write_analysis_sidecar should not crash when no data is present."""
        from phenotypic._cli._dashboard._analysis_data import (
            write_analysis_sidecar,
        )

        write_analysis_sidecar(tmp_dir)
        # No measurement data => no scatter/table/stats files
        assert not (tmp_dir / "progress" / "analysis_scatter.json").exists()

    def test_write_sidecar_with_data(self, tmp_dir):
        """write_analysis_sidecar should produce JSON files from measurement Parquets."""
        from phenotypic._cli._dashboard._analysis_data import (
            write_analysis_sidecar,
        )

        # Create fake measurement Parquet
        results_dir = tmp_dir / "results" / "plate1" / "measurements"
        results_dir.mkdir(parents=True)
        df = pd.DataFrame(
            {
                "Metadata_Dataset": ["plate1"] * 10,
                "Shape_Area": range(10),
                "Intensity_MeanIntensity": [float(x) * 0.1 for x in range(10)],
            }
        )
        pl.from_pandas(df).write_parquet(
            results_dir / "img001.parquet", compression="snappy"
        )

        write_analysis_sidecar(tmp_dir)

        scatter = json.loads(
            (tmp_dir / "progress" / "analysis_scatter.json").read_text()
        )
        assert "columns" in scatter
        assert "data" in scatter
        assert scatter["total_rows"] == 10

        stats = json.loads(
            (tmp_dir / "progress" / "analysis_stats.json").read_text()
        )
        assert "datasets" in stats
        assert "plate1" in stats["datasets"]

    def test_write_sidecar_with_metadata_csv(self, tmp_dir):
        """Metadata CSV columns should appear in sidecar data after join."""
        from phenotypic._cli._dashboard._analysis_data import (
            write_analysis_sidecar,
        )

        # Create fake measurement Parquet
        results_dir = tmp_dir / "results" / "plate1" / "measurements"
        results_dir.mkdir(parents=True)
        df = pd.DataFrame(
            {
                "Metadata_Dataset": ["plate1"] * 5,
                "Metadata_ImageFile": [f"img{i:03d}" for i in range(5)],
                "Shape_Area": range(5),
            }
        )
        pl.from_pandas(df).write_parquet(
            results_dir / "img000.parquet", compression="snappy"
        )

        # Create metadata CSV with a shared key and a new column
        metadata_csv = tmp_dir / "metadata.csv"
        meta_df = pd.DataFrame(
            {
                "Metadata_Dataset": ["plate1"] * 5,
                "Metadata_ImageFile": [f"img{i:03d}" for i in range(5)],
                "Metadata_Treatment": ["ctrl", "drugA", "drugB", "drugA", "ctrl"],
            }
        )
        meta_df.to_csv(metadata_csv, index=False)

        write_analysis_sidecar(tmp_dir, metadata_csv=metadata_csv)

        scatter = json.loads(
            (tmp_dir / "progress" / "analysis_scatter.json").read_text()
        )
        assert "Metadata_Treatment" in scatter["columns"]
        assert "Metadata_Treatment" in scatter["data"]
        assert scatter["data"]["Metadata_Treatment"] == [
            "ctrl", "drugA", "drugB", "drugA", "ctrl"
        ]

    def test_stratified_sampling(self, tmp_dir):
        """Stratified sampling should maintain proportional representation."""
        from phenotypic._cli._dashboard._analysis_data import (
            _stratified_sample,
        )

        df = pl.DataFrame(
            {
                "Metadata_Dataset": ["A"] * 100 + ["B"] * 100,
                "value": range(200),
            }
        )
        sampled = _stratified_sample(df, max_rows=50)
        assert sampled.height == 50
        # Should have roughly proportional representation
        counts = sampled["Metadata_Dataset"].value_counts()
        count_a = counts.filter(pl.col("Metadata_Dataset") == "A")["count"][0]
        count_b = counts.filter(pl.col("Metadata_Dataset") == "B")["count"][0]
        assert count_a > 10
        assert count_b > 10

    def test_overlay_manifest(self, tmp_dir):
        """Overlay manifest should discover PNG files grouped by dataset."""
        from phenotypic._cli._dashboard._analysis_data import (
            _prepare_overlay_manifest,
        )

        overlay_dir = tmp_dir / "results" / "plate1" / "overlays"
        overlay_dir.mkdir(parents=True)
        (overlay_dir / "img001.png").touch()
        (overlay_dir / "img002.png").touch()

        manifest = _prepare_overlay_manifest(tmp_dir)
        assert "datasets" in manifest
        assert "plate1" in manifest["datasets"]
        assert len(manifest["datasets"]["plate1"]) == 2
