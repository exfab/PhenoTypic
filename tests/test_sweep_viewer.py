"""Tests for the napari sweep viewer data model, utilities, and widgets."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_MANIFEST = {
    "version": "0.13.0",
    "description": "test sweep",
    "total_pipelines": 2,
    "configs": {
        "Pipeline": {
            "n_combinations": 2,
            "pipelines": {
                "Pipeline_0": {
                    "pipe_cfgs": {
                        "GaussianBlur_0": {
                            "class": "GaussianBlur",
                            "params": {"sigma": 1.0},
                        },
                        "OtsuDetector_0": {
                            "class": "OtsuDetector",
                            "params": {"ignore_zeros": True},
                        },
                    },
                    "meas_cfgs": {
                        "MeasureColor_0": {
                            "class": "MeasureColor",
                            "params": {},
                        },
                    },
                },
                "Pipeline_1": {
                    "pipe_cfgs": {
                        "GaussianBlur_0": {
                            "class": "GaussianBlur",
                            "params": {"sigma": 2.0},
                        },
                        "OtsuDetector_0": {
                            "class": "OtsuDetector",
                            "params": {"ignore_zeros": False},
                        },
                    },
                    "meas_cfgs": {},
                },
            },
        }
    },
}


@pytest.fixture()
def mock_sweep_dir(tmp_path: Path) -> Path:
    """Create a minimal mock sweep output directory."""
    # Write manifest
    (tmp_path / "sweep_manifest.json").write_text(
        json.dumps(MOCK_MANIFEST, indent=2)
    )

    # Create result images (1x1 white PNG via raw bytes is fragile,
    # so just create empty files with image extensions).
    results = tmp_path / "results"
    for pipe in ("Pipeline_0", "Pipeline_1"):
        for comp in ("overlays", "rgb", "objmask"):
            d = results / pipe / comp
            d.mkdir(parents=True, exist_ok=True)
            for stem in ("plate_001", "plate_002"):
                (d / f"{stem}.png").write_bytes(b"")

        # measurements CSVs
        meas_dir = results / pipe / "measurements"
        meas_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {"col_area": [100, 200], "col_roundness": [0.9, 0.8]}
        )
        df.to_csv(meas_dir / "plate_001.csv", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestSweepOutputScanner:
    """Tests for SweepOutputScanner."""

    def test_detect_sweep_dir_valid(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        result = SweepOutputScanner.detect_sweep_dir(mock_sweep_dir)
        assert result == mock_sweep_dir.resolve()

    def test_detect_sweep_dir_missing(self, tmp_path: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        with pytest.raises(FileNotFoundError, match="sweep_manifest.json"):
            SweepOutputScanner.detect_sweep_dir(tmp_path)

    def test_parse_manifest(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep._sweep_data_model import SweepOutputScanner

        manifest_path = mock_sweep_dir / "sweep_manifest.json"
        raw, configs = SweepOutputScanner._parse_manifest(manifest_path)

        assert raw["total_pipelines"] == 2
        assert "Pipeline_0" in configs
        assert "Pipeline_1" in configs

        cfg0 = configs["Pipeline_0"]
        assert cfg0.config_group == "Pipeline"
        assert len(cfg0.operations) == 2
        assert cfg0.operations[0]["class"] == "GaussianBlur"
        assert cfg0.operations[0]["params"]["sigma"] == 1.0
        assert len(cfg0.measurements) == 1
        assert cfg0.measurements[0]["class"] == "MeasureColor"

        cfg1 = configs["Pipeline_1"]
        assert len(cfg1.measurements) == 0

    def test_scan_results(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep._sweep_data_model import SweepOutputScanner

        results_dir = mock_sweep_dir / "results"
        files = SweepOutputScanner._scan_results(results_dir)

        # 2 pipelines x 3 components x 2 images = 12
        assert len(files) == 12

        # Verify structure
        components = {f.component for f in files}
        assert components == {"overlays", "rgb", "objmask"}

        pipelines = {f.pipeline_name for f in files}
        assert pipelines == {"Pipeline_0", "Pipeline_1"}

        stems = {f.image_stem for f in files}
        assert stems == {"plate_001", "plate_002"}

    def test_full_scan(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        data = SweepOutputScanner.scan(mock_sweep_dir)

        assert data.root_dir == mock_sweep_dir.resolve()
        assert data.pipeline_names == ["Pipeline_0", "Pipeline_1"]
        assert data.image_stems == ["plate_001", "plate_002"]
        assert sorted(data.components) == ["objmask", "overlays", "rgb"]

    def test_by_pipeline_index(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        data = SweepOutputScanner.scan(mock_sweep_dir)

        # Lookup: Pipeline_0 -> plate_001 -> rgb
        sf = data.by_pipeline["Pipeline_0"]["plate_001"]["rgb"]
        assert sf.pipeline_name == "Pipeline_0"
        assert sf.image_stem == "plate_001"
        assert sf.component == "rgb"
        assert sf.path.name == "plate_001.png"

    def test_by_image_index(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        data = SweepOutputScanner.scan(mock_sweep_dir)

        # Lookup: plate_002 -> objmask -> Pipeline_1
        sf = data.by_image["plate_002"]["objmask"]["Pipeline_1"]
        assert sf.pipeline_name == "Pipeline_1"
        assert sf.image_stem == "plate_002"
        assert sf.component == "objmask"


# ---------------------------------------------------------------------------
# Remote display tests
# ---------------------------------------------------------------------------


class TestRemoteDisplay:
    """Tests for remote display detection and configuration."""

    def test_detect_remote_session_false(self, monkeypatch: pytest.MonkeyPatch):
        from phenotypic.gui.sweep._remote_display import detect_remote_session

        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        assert detect_remote_session() is False

    def test_detect_remote_session_true_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from phenotypic.gui.sweep._remote_display import detect_remote_session

        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 56789 10.0.0.1 22")
        assert detect_remote_session() is True

    def test_detect_remote_session_true_client(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from phenotypic.gui.sweep._remote_display import detect_remote_session

        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 56789 22")
        assert detect_remote_session() is True

    def test_configure_remote_display(self, monkeypatch: pytest.MonkeyPatch):
        from phenotypic.gui.sweep._remote_display import configure_remote_display

        monkeypatch.delenv("QT_OPENGL", raising=False)
        monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)
        monkeypatch.setenv("LIBGL_ALWAYS_INDIRECT", "1")

        configure_remote_display()

        assert os.environ["QT_OPENGL"] == "software"
        assert os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1"
        assert "LIBGL_ALWAYS_INDIRECT" not in os.environ


# ---------------------------------------------------------------------------
# Measurements CSV loading test
# ---------------------------------------------------------------------------


class TestMeasurementsCSV:
    """Test that measurement CSVs are found and parsed correctly."""

    def test_measurements_csv_loading(self, mock_sweep_dir: Path):
        csv_path = (
            mock_sweep_dir
            / "results"
            / "Pipeline_0"
            / "measurements"
            / "plate_001.csv"
        )
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["col_area", "col_roundness"]
        assert len(df) == 2
        assert df["col_area"].iloc[0] == 100


# ---------------------------------------------------------------------------
# Label component detection tests
# ---------------------------------------------------------------------------


class TestLabelComponentDetection:
    """Tests for _LABEL_COMPONENTS and _is_label_component."""

    @pytest.mark.parametrize("comp", ["objmap", "objmask"])
    def test_label_components(self, comp: str):
        from phenotypic.gui.sweep._napari_sweep_viewer import (
            NapariSweepViewer,
        )

        assert NapariSweepViewer._is_label_component(comp)

    @pytest.mark.parametrize("comp", ["rgb", "overlays", "detect_mat"])
    def test_non_label_components(self, comp: str):
        from phenotypic.gui.sweep._napari_sweep_viewer import (
            NapariSweepViewer,
        )

        assert not NapariSweepViewer._is_label_component(comp)


# ---------------------------------------------------------------------------
# Stem data lookup tests
# ---------------------------------------------------------------------------


class TestStemDataLookups:
    """Tests for stem-level data lookups used by the tree widget."""

    def test_pipeline_first_stem_lookup(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        data = SweepOutputScanner.scan(mock_sweep_dir)

        stem_comps = data.by_pipeline["Pipeline_0"]["plate_001"]
        assert sorted(stem_comps.keys()) == [
            "objmask", "overlays", "rgb",
        ]
        for comp, sf in stem_comps.items():
            assert sf.pipeline_name == "Pipeline_0"
            assert sf.image_stem == "plate_001"
            assert sf.component == comp

    def test_image_first_stem_lookup(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        data = SweepOutputScanner.scan(mock_sweep_dir)

        comps = data.by_image["plate_001"]
        assert sorted(comps.keys()) == ["objmask", "overlays", "rgb"]

        # First pipeline alphabetically is deterministic
        for comp_name, pipes in comps.items():
            first_pipe = sorted(pipes.keys())[0]
            assert first_pipe == "Pipeline_0"
            sf = pipes[first_pipe]
            assert sf.image_stem == "plate_001"
            assert sf.component == comp_name


# ---------------------------------------------------------------------------
# File tree widget tests (require Qt via pytest-qt)
# ---------------------------------------------------------------------------

pytest_qt = pytest.importorskip("pytestqt")


@pytest.fixture()
def sweep_data(mock_sweep_dir: Path):
    """Return a scanned SweepOutputData from the mock directory."""
    from phenotypic.gui.sweep import SweepOutputScanner

    return SweepOutputScanner.scan(mock_sweep_dir)


@pytest.fixture()
def tree_widget(qtbot, sweep_data):
    """Create a SweepFileTreeWidget and register it with qtbot."""
    from phenotypic.gui.sweep._file_tree_widget import (
        SweepFileTreeWidget,
    )

    widget = SweepFileTreeWidget(sweep_data)
    qtbot.addWidget(widget)
    return widget


class TestFileTreeStructure:
    """Verify that the tree has exactly two levels in each mode."""

    def test_pipeline_first_two_levels(self, tree_widget):
        """Pipeline-first: top=pipelines, children=image stems."""
        tree = tree_widget._tree
        assert tree.topLevelItemCount() == 2  # Pipeline_0, Pipeline_1

        for i in range(tree.topLevelItemCount()):
            pipe_item = tree.topLevelItem(i)
            # Each pipeline has 2 image stems
            assert pipe_item.childCount() == 2
            for j in range(pipe_item.childCount()):
                leaf = pipe_item.child(j)
                # Leaves have no children (flat)
                assert leaf.childCount() == 0

    def test_image_first_two_levels(self, tree_widget):
        """Image-first: top=image stems, children=pipelines."""
        tree_widget._mode_combo.setCurrentIndex(
            tree_widget._IMAGE_FIRST,
        )

        tree = tree_widget._tree
        assert tree.topLevelItemCount() == 2  # plate_001, plate_002

        for i in range(tree.topLevelItemCount()):
            stem_item = tree.topLevelItem(i)
            # Each stem has 2 pipelines
            assert stem_item.childCount() == 2
            for j in range(stem_item.childCount()):
                leaf = stem_item.child(j)
                assert leaf.childCount() == 0

    def test_mode_switch_rebuilds_tree(self, tree_widget):
        """Switching modes rebuilds the tree with different top-level items."""
        tree = tree_widget._tree

        # Pipeline-first: top items are pipeline names
        top_names_pf = [
            tree.topLevelItem(i).text(0)
            for i in range(tree.topLevelItemCount())
        ]
        assert top_names_pf == ["Pipeline_0", "Pipeline_1"]

        # Switch to image-first
        tree_widget._mode_combo.setCurrentIndex(
            tree_widget._IMAGE_FIRST,
        )
        top_names_if = [
            tree.topLevelItem(i).text(0)
            for i in range(tree.topLevelItemCount())
        ]
        assert top_names_if == ["plate_001", "plate_002"]


class TestFileTreeSignals:
    """Verify signal emission from click routing."""

    def test_pipeline_node_emits_pipeline_selected(
        self, qtbot, tree_widget,
    ):
        """Clicking a top-level pipeline node emits pipeline_selected."""
        tree = tree_widget._tree
        pipe_item = tree.topLevelItem(0)  # Pipeline_0

        with qtbot.waitSignal(
            tree_widget.pipeline_selected, timeout=1000,
        ) as blocker:
            tree_widget._on_item_clicked(pipe_item, 0)

        assert blocker.args == ["Pipeline_0"]

    def test_leaf_emits_stem_selected_pipeline_first(
        self, qtbot, tree_widget,
    ):
        """Clicking a leaf in pipeline-first emits stem_selected."""
        tree = tree_widget._tree
        pipe_item = tree.topLevelItem(0)  # Pipeline_0
        leaf = pipe_item.child(0)  # plate_001

        with qtbot.waitSignal(
            tree_widget.stem_selected, timeout=1000,
        ) as blocker:
            tree_widget._on_item_clicked(leaf, 0)

        entries = blocker.args[0]
        assert len(entries) == 3  # objmask, overlays, rgb
        assert all(e["pipeline"] == "Pipeline_0" for e in entries)
        assert all(e["image_stem"] == "plate_001" for e in entries)
        components = {e["component"] for e in entries}
        assert components == {"objmask", "overlays", "rgb"}

    def test_leaf_emits_stem_selected_image_first(
        self, qtbot, tree_widget,
    ):
        """Clicking a leaf in image-first emits stem_selected."""
        tree_widget._mode_combo.setCurrentIndex(
            tree_widget._IMAGE_FIRST,
        )
        tree = tree_widget._tree
        stem_item = tree.topLevelItem(0)  # plate_001
        leaf = stem_item.child(0)  # Pipeline_0

        with qtbot.waitSignal(
            tree_widget.stem_selected, timeout=1000,
        ) as blocker:
            tree_widget._on_item_clicked(leaf, 0)

        entries = blocker.args[0]
        assert len(entries) == 3
        assert all(e["pipeline"] == "Pipeline_0" for e in entries)
        assert all(e["image_stem"] == "plate_001" for e in entries)

    def test_image_node_emits_stem_selected(
        self, qtbot, tree_widget,
    ):
        """Clicking a top-level image node emits stem_selected."""
        tree_widget._mode_combo.setCurrentIndex(
            tree_widget._IMAGE_FIRST,
        )
        tree = tree_widget._tree
        stem_item = tree.topLevelItem(0)  # plate_001

        with qtbot.waitSignal(
            tree_widget.stem_selected, timeout=1000,
        ) as blocker:
            tree_widget._on_item_clicked(stem_item, 0)

        entries = blocker.args[0]
        assert len(entries) == 3
        # Auto-selects best pipeline (Pipeline_0, alphabetical tiebreak)
        assert all(e["pipeline"] == "Pipeline_0" for e in entries)
        assert all(e["image_stem"] == "plate_001" for e in entries)


class TestFileTreeCompareMode:
    """Verify compare-mode signal routing."""

    def test_compare_leaf_emits_stem_compare_requested(
        self, qtbot, tree_widget,
    ):
        """With compare checked, leaf click emits stem_compare_requested."""
        tree_widget._compare_cb.setChecked(True)

        tree = tree_widget._tree
        pipe_item = tree.topLevelItem(0)  # Pipeline_0
        leaf = pipe_item.child(0)  # plate_001

        with qtbot.waitSignal(
            tree_widget.stem_compare_requested, timeout=1000,
        ) as blocker:
            tree_widget._on_item_clicked(leaf, 0)

        entries = blocker.args[0]
        assert len(entries) == 3
        assert all(e["pipeline"] == "Pipeline_0" for e in entries)
        assert all(e["image_stem"] == "plate_001" for e in entries)

    def test_compare_leaf_does_not_emit_stem_selected(
        self, qtbot, tree_widget,
    ):
        """With compare checked, leaf click does NOT emit stem_selected."""
        tree_widget._compare_cb.setChecked(True)

        tree = tree_widget._tree
        pipe_item = tree.topLevelItem(0)
        leaf = pipe_item.child(0)

        with qtbot.assertNotEmitted(tree_widget.stem_selected):
            tree_widget._on_item_clicked(leaf, 0)

    def test_compare_image_node_still_emits_stem_selected(
        self, qtbot, tree_widget,
    ):
        """Top-level image node always emits stem_selected, even in
        compare mode (no specific pipeline chosen).
        """
        tree_widget._mode_combo.setCurrentIndex(
            tree_widget._IMAGE_FIRST,
        )
        tree_widget._compare_cb.setChecked(True)

        tree = tree_widget._tree
        stem_item = tree.topLevelItem(0)  # plate_001

        with qtbot.waitSignal(
            tree_widget.stem_selected, timeout=1000,
        ):
            tree_widget._on_item_clicked(stem_item, 0)

    def test_compare_unchecked_leaf_emits_stem_selected(
        self, qtbot, tree_widget,
    ):
        """With compare unchecked, leaf emits stem_selected (not compare)."""
        tree_widget._compare_cb.setChecked(False)

        tree = tree_widget._tree
        pipe_item = tree.topLevelItem(0)
        leaf = pipe_item.child(0)

        with qtbot.waitSignal(
            tree_widget.stem_selected, timeout=1000,
        ):
            tree_widget._on_item_clicked(leaf, 0)

        with qtbot.assertNotEmitted(
            tree_widget.stem_compare_requested,
        ):
            tree_widget._on_item_clicked(leaf, 0)


class TestFileTreeEntryPayloads:
    """Verify that emitted entries contain valid paths and all keys."""

    def test_entries_have_required_keys(
        self, qtbot, tree_widget,
    ):
        """Every entry dict has path, pipeline, component, image_stem."""
        tree = tree_widget._tree
        leaf = tree.topLevelItem(0).child(0)

        with qtbot.waitSignal(
            tree_widget.stem_selected, timeout=1000,
        ) as blocker:
            tree_widget._on_item_clicked(leaf, 0)

        required = {"path", "pipeline", "component", "image_stem"}
        for entry in blocker.args[0]:
            assert required <= set(entry.keys())

    def test_entries_paths_are_absolute(
        self, qtbot, tree_widget,
    ):
        """All emitted paths are absolute path strings."""
        tree = tree_widget._tree
        leaf = tree.topLevelItem(0).child(0)

        with qtbot.waitSignal(
            tree_widget.stem_selected, timeout=1000,
        ) as blocker:
            tree_widget._on_item_clicked(leaf, 0)

        for entry in blocker.args[0]:
            assert Path(entry["path"]).is_absolute()


class TestSignalChainViaItemClicked:
    """Tests that use itemClicked.emit() to verify the full Qt signal chain.

    Previous tests call ``_on_item_clicked()`` directly, which bypasses
    the ``itemClicked`` signal connection.  These tests verify that the
    signal wiring in ``__init__`` actually connects to the slot.
    """

    def test_leaf_click_signal_emits_stem_selected(
        self, qtbot, tree_widget,
    ):
        """itemClicked signal on a leaf fires stem_selected."""
        tree = tree_widget._tree
        leaf = tree.topLevelItem(0).child(0)

        with qtbot.waitSignal(
            tree_widget.stem_selected, timeout=1000,
        ) as blocker:
            tree.itemClicked.emit(leaf, 0)

        entries = blocker.args[0]
        assert len(entries) == 3
        assert all(e["pipeline"] == "Pipeline_0" for e in entries)

    def test_pipeline_click_signal_emits_pipeline_selected(
        self, qtbot, tree_widget,
    ):
        """itemClicked signal on a pipeline node fires pipeline_selected."""
        tree = tree_widget._tree
        pipe_item = tree.topLevelItem(0)

        with qtbot.waitSignal(
            tree_widget.pipeline_selected, timeout=1000,
        ) as blocker:
            tree.itemClicked.emit(pipe_item, 0)

        assert blocker.args == ["Pipeline_0"]

    def test_image_first_leaf_signal_emits_stem_selected(
        self, qtbot, tree_widget,
    ):
        """itemClicked signal on an image-first leaf fires stem_selected."""
        tree_widget._mode_combo.setCurrentIndex(
            tree_widget._IMAGE_FIRST,
        )
        tree = tree_widget._tree
        leaf = tree.topLevelItem(0).child(0)

        with qtbot.waitSignal(
            tree_widget.stem_selected, timeout=1000,
        ) as blocker:
            tree.itemClicked.emit(leaf, 0)

        entries = blocker.args[0]
        assert len(entries) == 3
        assert all(e["image_stem"] == "plate_001" for e in entries)

    def test_compare_leaf_signal_emits_compare_requested(
        self, qtbot, tree_widget,
    ):
        """itemClicked signal with compare mode fires stem_compare_requested."""
        tree_widget._compare_cb.setChecked(True)
        tree = tree_widget._tree
        leaf = tree.topLevelItem(0).child(0)

        with qtbot.waitSignal(
            tree_widget.stem_compare_requested, timeout=1000,
        ) as blocker:
            tree.itemClicked.emit(leaf, 0)

        entries = blocker.args[0]
        assert len(entries) == 3
        assert all(e["pipeline"] == "Pipeline_0" for e in entries)

    def test_signal_chain_with_mock_receiver(
        self, qtbot, tree_widget,
    ):
        """Full chain: itemClicked → stem_selected → receiver callback."""
        received = []
        tree_widget.stem_selected.connect(
            lambda entries: received.extend(entries),
        )

        tree = tree_widget._tree
        leaf = tree.topLevelItem(0).child(0)
        tree.itemClicked.emit(leaf, 0)

        assert len(received) == 3
        components = {e["component"] for e in received}
        assert components == {"objmask", "overlays", "rgb"}
