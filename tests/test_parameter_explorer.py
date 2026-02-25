"""Tests for the Parameter Explorer widget and swept parameter analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic.gui.sweep._sweep_data_model import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    name: str,
    operations: list[dict],
    measurements: list[dict] | None = None,
) -> PipelineConfig:
    """Build a minimal PipelineConfig for testing."""
    return PipelineConfig(
        name=name,
        config_group="TestGroup",
        operations=operations,
        measurements=measurements or [],
        raw_json={},
    )


CONFIGS_TWO_PIPELINES = {
    "Pipeline_0": _make_config(
        "Pipeline_0",
        [
            {
                "name": "GaussianBlur_0",
                "class": "GaussianBlur",
                "params": {"sigma": 1.0, "truncate": 4.0},
            },
            {
                "name": "OtsuDetector_0",
                "class": "OtsuDetector",
                "params": {"ignore_zeros": True},
            },
        ],
    ),
    "Pipeline_1": _make_config(
        "Pipeline_1",
        [
            {
                "name": "GaussianBlur_0",
                "class": "GaussianBlur",
                "params": {"sigma": 2.0, "truncate": 4.0},
            },
            {
                "name": "OtsuDetector_0",
                "class": "OtsuDetector",
                "params": {"ignore_zeros": False},
            },
        ],
    ),
}

CONFIGS_MIXED_STRUCTURE = {
    "Blur_sigma1": _make_config(
        "Blur_sigma1",
        [
            {
                "name": "GaussianBlur_0",
                "class": "GaussianBlur",
                "params": {"sigma": 1.0},
            },
            {
                "name": "OtsuDetector_0",
                "class": "OtsuDetector",
                "params": {"ignore_zeros": True},
            },
        ],
    ),
    "Blur_sigma2": _make_config(
        "Blur_sigma2",
        [
            {
                "name": "GaussianBlur_0",
                "class": "GaussianBlur",
                "params": {"sigma": 2.0},
            },
            {
                "name": "OtsuDetector_0",
                "class": "OtsuDetector",
                "params": {"ignore_zeros": True},
            },
        ],
    ),
    "Median_k3": _make_config(
        "Median_k3",
        [
            {
                "name": "MedianFilter_0",
                "class": "MedianFilter",
                "params": {"kernel_size": 3},
            },
            {
                "name": "AdaptiveDetector_0",
                "class": "AdaptiveDetector",
                "params": {"block_size": 11},
            },
        ],
    ),
    "Median_k5": _make_config(
        "Median_k5",
        [
            {
                "name": "MedianFilter_0",
                "class": "MedianFilter",
                "params": {"kernel_size": 5},
            },
            {
                "name": "AdaptiveDetector_0",
                "class": "AdaptiveDetector",
                "params": {"block_size": 11},
            },
        ],
    ),
}

CONFIGS_THREE_PIPELINES = {
    "Pipeline_0": _make_config(
        "Pipeline_0",
        [
            {
                "name": "GaussianBlur_0",
                "class": "GaussianBlur",
                "params": {"sigma": 1.0},
            },
        ],
    ),
    "Pipeline_1": _make_config(
        "Pipeline_1",
        [
            {
                "name": "GaussianBlur_0",
                "class": "GaussianBlur",
                "params": {"sigma": 2.0},
            },
        ],
    ),
    "Pipeline_2": _make_config(
        "Pipeline_2",
        [
            {
                "name": "GaussianBlur_0",
                "class": "GaussianBlur",
                "params": {"sigma": 3.0},
            },
        ],
    ),
}


# ---------------------------------------------------------------------------
# Pure data tests — detect_swept_parameters
# ---------------------------------------------------------------------------


class TestDetectSweptParameters:
    """Tests for detect_swept_parameters()."""

    def test_finds_varied_params(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
        )

        swept = detect_swept_parameters(CONFIGS_TWO_PIPELINES)

        names = {(sp.operation_name, sp.param_name) for sp in swept}
        assert ("GaussianBlur_0", "sigma") in names
        assert ("OtsuDetector_0", "ignore_zeros") in names

    def test_excludes_fixed_params(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
        )

        swept = detect_swept_parameters(CONFIGS_TWO_PIPELINES)

        names = {(sp.operation_name, sp.param_name) for sp in swept}
        # truncate=4.0 is the same in both configs
        assert ("GaussianBlur_0", "truncate") not in names

    def test_numeric_sorted(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
        )

        swept = detect_swept_parameters(CONFIGS_TWO_PIPELINES)

        sigma_sp = next(
            sp for sp in swept
            if sp.param_name == "sigma"
        )
        assert sigma_sp.values == (1.0, 2.0)
        assert sigma_sp.is_numeric_ordered is True

    def test_bool_classified_as_dropdown(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
        )

        swept = detect_swept_parameters(CONFIGS_TWO_PIPELINES)

        ignore_sp = next(
            sp for sp in swept
            if sp.param_name == "ignore_zeros"
        )
        assert ignore_sp.is_numeric_ordered is False

    def test_single_pipeline_no_swept_params(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
        )

        single = {"Pipeline_0": CONFIGS_TWO_PIPELINES["Pipeline_0"]}
        swept = detect_swept_parameters(single)

        assert swept == []

    def test_empty_configs(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
        )

        assert detect_swept_parameters({}) == []

    def test_three_values_sorted(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
        )

        swept = detect_swept_parameters(CONFIGS_THREE_PIPELINES)

        sigma_sp = next(
            sp for sp in swept
            if sp.param_name == "sigma"
        )
        assert sigma_sp.values == (1.0, 2.0, 3.0)

    def test_tuple_classified_as_dropdown(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
        )

        configs = {
            "P0": _make_config(
                "P0",
                [
                    {
                        "name": "Op_0",
                        "class": "Op",
                        "params": {"size": [3, 3]},
                    },
                ],
            ),
            "P1": _make_config(
                "P1",
                [
                    {
                        "name": "Op_0",
                        "class": "Op",
                        "params": {"size": [5, 5]},
                    },
                ],
            ),
        }
        swept = detect_swept_parameters(configs)

        assert len(swept) == 1
        assert swept[0].is_numeric_ordered is False

    def test_results_sorted_by_op_and_param(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
        )

        swept = detect_swept_parameters(CONFIGS_TWO_PIPELINES)

        keys = [(sp.operation_name, sp.param_name) for sp in swept]
        assert keys == sorted(keys)

    def test_measurement_swept_params(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
        )

        configs = {
            "P0": _make_config(
                "P0",
                [
                    {
                        "name": "GaussianBlur_0",
                        "class": "GaussianBlur",
                        "params": {"sigma": 1.0},
                    },
                ],
                measurements=[
                    {
                        "name": "MeasureColor_0",
                        "class": "MeasureColor",
                        "params": {"color_space": "Lab"},
                    },
                ],
            ),
            "P1": _make_config(
                "P1",
                [
                    {
                        "name": "GaussianBlur_0",
                        "class": "GaussianBlur",
                        "params": {"sigma": 1.0},
                    },
                ],
                measurements=[
                    {
                        "name": "MeasureColor_0",
                        "class": "MeasureColor",
                        "params": {"color_space": "HSV"},
                    },
                ],
            ),
        }
        swept = detect_swept_parameters(configs)

        assert len(swept) == 1
        assert swept[0].operation_name == "MeasureColor_0"
        assert swept[0].param_name == "color_space"
        assert swept[0].is_numeric_ordered is False
        assert swept[0].values == ("Lab", "HSV")


# ---------------------------------------------------------------------------
# Pure data tests — build_param_to_pipeline_map and resolve
# ---------------------------------------------------------------------------


class TestPipelineLookup:
    """Tests for build_param_to_pipeline_map and resolve_pipeline_name."""

    def test_all_pipelines_mapped(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            build_param_to_pipeline_map,
            detect_swept_parameters,
        )

        swept = detect_swept_parameters(CONFIGS_TWO_PIPELINES)
        lookup = build_param_to_pipeline_map(
            CONFIGS_TWO_PIPELINES, swept,
        )

        assert set(lookup.values()) == {"Pipeline_0", "Pipeline_1"}

    def test_reverse_lookup_works(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            build_param_to_pipeline_map,
            detect_swept_parameters,
            resolve_pipeline_name,
        )

        swept = detect_swept_parameters(CONFIGS_TWO_PIPELINES)
        lookup = build_param_to_pipeline_map(
            CONFIGS_TWO_PIPELINES, swept,
        )

        # Build selections matching Pipeline_0
        selections = {
            ("GaussianBlur_0", "sigma"): 1.0,
            ("OtsuDetector_0", "ignore_zeros"): True,
        }
        result = resolve_pipeline_name(selections, lookup, swept)
        assert result == "Pipeline_0"

    def test_no_match_returns_none(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            build_param_to_pipeline_map,
            detect_swept_parameters,
            resolve_pipeline_name,
        )

        swept = detect_swept_parameters(CONFIGS_TWO_PIPELINES)
        lookup = build_param_to_pipeline_map(
            CONFIGS_TWO_PIPELINES, swept,
        )

        # Mix values that don't exist together in any pipeline
        selections = {
            ("GaussianBlur_0", "sigma"): 1.0,
            ("OtsuDetector_0", "ignore_zeros"): False,
        }
        result = resolve_pipeline_name(selections, lookup, swept)
        assert result is None

    def test_empty_swept_params(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            build_param_to_pipeline_map,
        )

        lookup = build_param_to_pipeline_map(
            CONFIGS_TWO_PIPELINES, [],
        )
        assert lookup == {}


# ---------------------------------------------------------------------------
# Pure data tests — get_swept_param_names
# ---------------------------------------------------------------------------


class TestGetSweptParamNames:
    """Tests for get_swept_param_names()."""

    def test_returns_correct_set(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            detect_swept_parameters,
            get_swept_param_names,
        )

        swept = detect_swept_parameters(CONFIGS_TWO_PIPELINES)
        names = get_swept_param_names(swept)

        assert ("GaussianBlur_0", "sigma") in names
        assert ("OtsuDetector_0", "ignore_zeros") in names
        assert ("GaussianBlur_0", "truncate") not in names


# ---------------------------------------------------------------------------
# Pure data tests — structural grouping
# ---------------------------------------------------------------------------


class TestStructuralGrouping:
    """Tests for compute_structural_signature and group_configs_by_structure."""

    def test_same_structure_same_signature(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            compute_structural_signature,
        )

        sig1 = compute_structural_signature(
            CONFIGS_MIXED_STRUCTURE["Blur_sigma1"],
        )
        sig2 = compute_structural_signature(
            CONFIGS_MIXED_STRUCTURE["Blur_sigma2"],
        )
        assert sig1 == sig2

    def test_different_structure_different_signature(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            compute_structural_signature,
        )

        sig_blur = compute_structural_signature(
            CONFIGS_MIXED_STRUCTURE["Blur_sigma1"],
        )
        sig_median = compute_structural_signature(
            CONFIGS_MIXED_STRUCTURE["Median_k3"],
        )
        assert sig_blur != sig_median

    def test_group_configs_by_structure_two_groups(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            group_configs_by_structure,
        )

        groups = group_configs_by_structure(CONFIGS_MIXED_STRUCTURE)

        assert len(groups) == 2

        # Each group should have 2 configs
        sizes = sorted(len(g) for g in groups.values())
        assert sizes == [2, 2]

        # Blur configs should be together
        all_names = [
            set(g.keys()) for g in groups.values()
        ]
        assert {"Blur_sigma1", "Blur_sigma2"} in all_names
        assert {"Median_k3", "Median_k5"} in all_names

    def test_group_configs_uniform_structure(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            group_configs_by_structure,
        )

        groups = group_configs_by_structure(CONFIGS_TWO_PIPELINES)

        assert len(groups) == 1
        group = next(iter(groups.values()))
        assert set(group.keys()) == {"Pipeline_0", "Pipeline_1"}

    def test_group_configs_empty(self):
        from phenotypic.gui.sweep._swept_param_analysis import (
            group_configs_by_structure,
        )

        assert group_configs_by_structure({}) == {}


# ---------------------------------------------------------------------------
# Qt widget tests (guarded by pytest-qt)
# ---------------------------------------------------------------------------


pytestqt = pytest.importorskip("pytestqt")


class TestParameterExplorerWidget:
    """Tests for the ParameterExplorerWidget."""

    def test_dropdown_for_numeric(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )
        from qtpy.QtWidgets import QComboBox

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_TWO_PIPELINES)

        # sigma is numeric — should have a combo box
        ctrl = widget._controls.get(("GaussianBlur_0", "sigma"))
        assert ctrl is not None
        assert isinstance(ctrl, QComboBox)

    def test_dropdown_for_bool(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )
        from qtpy.QtWidgets import QComboBox

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_TWO_PIPELINES)

        # ignore_zeros is bool — should have a combo box
        ctrl = widget._controls.get(
            ("OtsuDetector_0", "ignore_zeros"),
        )
        assert ctrl is not None
        assert isinstance(ctrl, QComboBox)

    def test_view_signal_emitted(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_TWO_PIPELINES)

        # Set to Pipeline_0
        widget.set_pipeline("Pipeline_0")

        with qtbot.waitSignal(
            widget.view_requested, timeout=1000,
        ) as blocker:
            widget._view_btn.click()

        assert blocker.args == ["Pipeline_0"]

    def test_view_split_signal_emitted(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_TWO_PIPELINES)

        widget.set_pipeline("Pipeline_1")

        with qtbot.waitSignal(
            widget.view_split_requested, timeout=1000,
        ) as blocker:
            widget._view_split_btn.click()

        assert blocker.args == ["Pipeline_1"]

    def test_set_pipeline_syncs_controls(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_TWO_PIPELINES)

        widget.set_pipeline("Pipeline_1")

        # Verify sigma combo has 2.0 selected
        sigma_combo = widget._controls[("GaussianBlur_0", "sigma")]
        assert json.loads(sigma_combo.currentData()) == 2.0

        # Verify ignore_zeros combo
        combo = widget._controls[
            ("OtsuDetector_0", "ignore_zeros")
        ]
        assert json.loads(combo.currentData()) is False

    def test_status_label_shows_pipeline_name(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_TWO_PIPELINES)

        widget.set_pipeline("Pipeline_0")
        assert widget._status_label.text() == "Pipeline_0"

    def test_empty_configs(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs({})

        assert not widget._view_btn.isEnabled()
        assert not widget._view_split_btn.isEnabled()
        assert widget._status_label.text() == ""

    def test_buttons_disabled_when_no_match(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_TWO_PIPELINES)

        # Set sigma=1.0 (Pipeline_0) but ignore_zeros=False (Pipeline_1)
        # — this combination doesn't match any pipeline
        sigma_combo = widget._controls[("GaussianBlur_0", "sigma")]
        sigma_combo.setCurrentIndex(0)  # sigma=1.0
        combo = widget._controls[("OtsuDetector_0", "ignore_zeros")]
        combo.setCurrentIndex(1)  # ignore_zeros=False

        assert not widget._view_btn.isEnabled()
        assert not widget._view_split_btn.isEnabled()
        assert widget._status_label.text() == "No match"

    def test_buttons_enabled_when_match_exists(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_TWO_PIPELINES)

        widget.set_pipeline("Pipeline_0")

        assert widget._view_btn.isEnabled()
        assert widget._view_split_btn.isEnabled()
        assert widget._status_label.text() == "Pipeline_0"

    def test_buttons_toggle_on_param_change(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_TWO_PIPELINES)

        # Start with a valid match
        widget.set_pipeline("Pipeline_0")
        assert widget._view_btn.isEnabled()

        # Change combo to produce no match (sigma=1.0 + ignore_zeros=False)
        combo = widget._controls[("OtsuDetector_0", "ignore_zeros")]
        combo.setCurrentIndex(1)  # ignore_zeros=False
        assert not widget._view_btn.isEnabled()

        # Change back to valid match (sigma=1.0 + ignore_zeros=True)
        combo.setCurrentIndex(0)  # ignore_zeros=True
        assert widget._view_btn.isEnabled()


class TestParameterExplorerGrouping:
    """Tests for structural group scoping in ParameterExplorerWidget."""

    def test_mixed_configs_shows_only_first_group(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_MIXED_STRUCTURE)

        # First group alphabetically is Blur (Blur_sigma1 < Median_k3)
        param_names = {
            sp.param_name for sp in widget.swept_params
        }
        assert "sigma" in param_names
        assert "kernel_size" not in param_names

    def test_set_pipeline_switches_group(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_MIXED_STRUCTURE)

        # Switch to a Median pipeline
        widget.set_pipeline("Median_k5")

        param_names = {
            sp.param_name for sp in widget.swept_params
        }
        assert "kernel_size" in param_names
        assert "sigma" not in param_names

    def test_set_pipeline_same_group_no_rebuild(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_MIXED_STRUCTURE)

        # Both are in the Blur group — group key should not change
        initial_key = widget._active_group_key
        widget.set_pipeline("Blur_sigma2")
        assert widget._active_group_key is initial_key

    def test_single_config_group_no_swept_params(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        # Create configs where one structural group has only 1 pipeline
        lone = {
            "Lone": _make_config(
                "Lone",
                [
                    {
                        "name": "SpecialOp_0",
                        "class": "SpecialOp",
                        "params": {"alpha": 0.5},
                    },
                ],
            ),
        }
        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(lone)

        assert widget.swept_params == []
        assert len(widget._controls) == 0

    def test_swept_params_scoped_to_group(self, qtbot):
        from phenotypic.gui.sweep._parameter_explorer_widget import (
            ParameterExplorerWidget,
        )

        widget = ParameterExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_configs(CONFIGS_MIXED_STRUCTURE)

        # Initial group is Blur — only sigma should be swept
        swept_keys = {
            (sp.operation_name, sp.param_name)
            for sp in widget.swept_params
        }
        assert ("GaussianBlur_0", "sigma") in swept_keys
        assert ("MedianFilter_0", "kernel_size") not in swept_keys

        # Switch to Median group
        widget.set_pipeline("Median_k3")
        swept_keys = {
            (sp.operation_name, sp.param_name)
            for sp in widget.swept_params
        }
        assert ("MedianFilter_0", "kernel_size") in swept_keys
        assert ("GaussianBlur_0", "sigma") not in swept_keys


class TestPipelineConfigBar:
    """Tests for the PipelineConfigBar."""

    def test_set_main_renders_html(self, qtbot):
        from phenotypic.gui.sweep._pipeline_config_bar import (
            PipelineConfigBar,
        )

        bar = PipelineConfigBar()
        qtbot.addWidget(bar)

        config = CONFIGS_TWO_PIPELINES["Pipeline_0"]
        bar.set_main_pipeline(config, set())

        html_text = bar._main_browser.toHtml()
        assert "GaussianBlur" in html_text
        assert not bar._main_group.isHidden()

    def test_swept_params_bolded(self, qtbot):
        from phenotypic.gui.sweep._pipeline_config_bar import (
            PipelineConfigBar,
        )

        bar = PipelineConfigBar()
        qtbot.addWidget(bar)

        config = CONFIGS_TWO_PIPELINES["Pipeline_0"]
        swept_names = {("GaussianBlur_0", "sigma")}
        bar.set_main_pipeline(config, swept_names)

        # Qt renders <b> as font-weight:600 in toHtml() output
        html_text = bar._main_browser.toHtml()
        assert "font-weight:600" in html_text
        assert "sigma = 1.0" in html_text

        # Verify the source HTML uses <b> tags
        source_html = PipelineConfigBar._format_config(
            config, swept_names,
        )
        assert "<b>sigma = 1.0</b>" in source_html

    def test_set_split_shows_right_panel(self, qtbot):
        from phenotypic.gui.sweep._pipeline_config_bar import (
            PipelineConfigBar,
        )

        bar = PipelineConfigBar()
        qtbot.addWidget(bar)

        config = CONFIGS_TWO_PIPELINES["Pipeline_1"]
        bar.set_split_pipeline(config, set())

        assert not bar._split_group.isHidden()

    def test_clear_split_hides_right_panel(self, qtbot):
        from phenotypic.gui.sweep._pipeline_config_bar import (
            PipelineConfigBar,
        )

        bar = PipelineConfigBar()
        qtbot.addWidget(bar)

        config = CONFIGS_TWO_PIPELINES["Pipeline_1"]
        bar.set_split_pipeline(config, set())
        assert not bar._split_group.isHidden()

        bar.clear_split()
        assert bar._split_group.isHidden()

    def test_clear_hides_both(self, qtbot):
        from phenotypic.gui.sweep._pipeline_config_bar import (
            PipelineConfigBar,
        )

        bar = PipelineConfigBar()
        qtbot.addWidget(bar)

        config = CONFIGS_TWO_PIPELINES["Pipeline_0"]
        bar.set_main_pipeline(config, set())
        bar.set_split_pipeline(config, set())

        bar.clear()
        assert bar._main_group.isHidden()
        assert bar._split_group.isHidden()


class TestSplitStepSliderWidget:
    """Tests for the SplitStepSliderWidget."""

    def test_main_signal_forwarded(self, qtbot):
        from phenotypic.gui.sweep._split_step_slider_widget import (
            SplitStepSliderWidget,
        )
        from phenotypic.gui.sweep._sweep_data_model import (
            IntermediateStep,
        )

        widget = SplitStepSliderWidget()
        qtbot.addWidget(widget)

        steps = [
            IntermediateStep(
                index=0,
                operation_name="GaussianBlur",
                h5_path=Path("/tmp/00_GaussianBlur.h5"),
            ),
        ]
        widget.set_main_steps(steps)

        with qtbot.waitSignal(
            widget.main_step_changed, timeout=1000,
        ):
            widget._main_slider._slider.setValue(0)

    def test_split_signal_forwarded(self, qtbot):
        from phenotypic.gui.sweep._split_step_slider_widget import (
            SplitStepSliderWidget,
        )
        from phenotypic.gui.sweep._sweep_data_model import (
            IntermediateStep,
        )

        widget = SplitStepSliderWidget()
        qtbot.addWidget(widget)

        steps = [
            IntermediateStep(
                index=0,
                operation_name="OtsuDetector",
                h5_path=Path("/tmp/00_OtsuDetector.h5"),
            ),
        ]
        widget.set_split_steps(steps)

        with qtbot.waitSignal(
            widget.split_step_changed, timeout=1000,
        ):
            widget._split_slider._slider.setValue(0)

    def test_clear_split_hides_row(self, qtbot):
        from phenotypic.gui.sweep._split_step_slider_widget import (
            SplitStepSliderWidget,
        )
        from phenotypic.gui.sweep._sweep_data_model import (
            IntermediateStep,
        )

        widget = SplitStepSliderWidget()
        qtbot.addWidget(widget)

        steps = [
            IntermediateStep(
                index=0,
                operation_name="Op",
                h5_path=Path("/tmp/00_Op.h5"),
            ),
        ]
        widget.set_split_steps(steps)
        assert not widget._split_row.isHidden()

        widget.clear_split()
        assert widget._split_row.isHidden()

    def test_both_cleared_hides_widget(self, qtbot):
        from phenotypic.gui.sweep._split_step_slider_widget import (
            SplitStepSliderWidget,
        )

        widget = SplitStepSliderWidget()
        qtbot.addWidget(widget)

        widget.clear_main()
        widget.clear_split()
        assert widget.isHidden()


class TestGroupedLayerWidget:
    """Tests for the updated GroupedLayerWidget."""

    def test_checkbox_created_for_components(self, qtbot):
        from phenotypic.gui.sweep._grouped_layer_widget import (
            GroupedLayerWidget,
        )

        widget = GroupedLayerWidget(viewer=None)
        qtbot.addWidget(widget)

        entries = [
            {"pipeline": "P0", "component": "rgb", "image_stem": "s1"},
            {
                "pipeline": "P0",
                "component": "detect_mat",
                "image_stem": "s1",
            },
        ]
        widget.set_layers(entries)

        assert "rgb" in widget._checkboxes
        assert "detect_mat" in widget._checkboxes
        assert len(widget._checkboxes) == 2

    def test_checkbox_toggles_visibility(self, qtbot):
        from phenotypic.gui.sweep._grouped_layer_widget import (
            GroupedLayerWidget,
        )

        # Create a mock viewer with layers
        class MockLayer:
            def __init__(self, name):
                self.name = name
                self.visible = True

        class MockLayers:
            def __init__(self):
                self._layers = [
                    MockLayer("main/P0/rgb/s1"),
                    MockLayer("split/P1/rgb/s1"),
                    MockLayer("main/P0/detect_mat/s1"),
                ]

            def __iter__(self):
                return iter(self._layers)

        class MockViewer:
            def __init__(self):
                self.layers = MockLayers()

        viewer = MockViewer()
        widget = GroupedLayerWidget(viewer=viewer)
        qtbot.addWidget(widget)

        entries = [
            {"pipeline": "P0", "component": "rgb", "image_stem": "s1"},
            {
                "pipeline": "P0",
                "component": "detect_mat",
                "image_stem": "s1",
            },
        ]
        widget.set_layers(entries)

        # Uncheck rgb
        widget._checkboxes["rgb"].setChecked(False)

        # Both rgb layers should be hidden
        assert viewer.layers._layers[0].visible is False
        assert viewer.layers._layers[1].visible is False
        # detect_mat should be unchanged
        assert viewer.layers._layers[2].visible is True

    def test_synced_across_main_split(self, qtbot):
        from phenotypic.gui.sweep._grouped_layer_widget import (
            GroupedLayerWidget,
        )

        class MockLayer:
            def __init__(self, name):
                self.name = name
                self.visible = True

        class MockLayers:
            def __init__(self):
                self._layers = [
                    MockLayer("main/P0/objmap/s1"),
                    MockLayer("split/P1/objmap/s1"),
                ]

            def __iter__(self):
                return iter(self._layers)

        class MockViewer:
            def __init__(self):
                self.layers = MockLayers()

        viewer = MockViewer()
        widget = GroupedLayerWidget(viewer=viewer)
        qtbot.addWidget(widget)

        entries = [
            {
                "pipeline": "P0",
                "component": "objmap",
                "image_stem": "s1",
            },
        ]
        widget.set_layers(entries)

        widget._checkboxes["objmap"].setChecked(False)

        # Both main and split should be hidden
        assert viewer.layers._layers[0].visible is False
        assert viewer.layers._layers[1].visible is False

    def test_add_layers_merges_components(self, qtbot):
        from phenotypic.gui.sweep._grouped_layer_widget import (
            GroupedLayerWidget,
        )

        widget = GroupedLayerWidget(viewer=None)
        qtbot.addWidget(widget)

        widget.set_layers(
            [
                {
                    "pipeline": "P0",
                    "component": "rgb",
                    "image_stem": "s1",
                },
            ],
        )
        assert len(widget._checkboxes) == 1

        widget.add_layers(
            [
                {
                    "pipeline": "P0",
                    "component": "gray",
                    "image_stem": "s1",
                },
            ],
        )
        assert len(widget._checkboxes) == 2
        assert "rgb" in widget._checkboxes
        assert "gray" in widget._checkboxes

    def test_clear_preserves_visibility_state(self, qtbot):
        from phenotypic.gui.sweep._grouped_layer_widget import (
            GroupedLayerWidget,
        )

        widget = GroupedLayerWidget(viewer=None)
        qtbot.addWidget(widget)

        widget.set_layers(
            [
                {
                    "pipeline": "P0",
                    "component": "rgb",
                    "image_stem": "s1",
                },
            ],
        )
        widget._checkboxes["rgb"].setChecked(False)

        widget.clear()
        assert len(widget._checkboxes) == 0

        # Re-add — rgb should remember being unchecked
        widget.set_layers(
            [
                {
                    "pipeline": "P0",
                    "component": "rgb",
                    "image_stem": "s1",
                },
            ],
        )
        assert not widget._checkboxes["rgb"].isChecked()
