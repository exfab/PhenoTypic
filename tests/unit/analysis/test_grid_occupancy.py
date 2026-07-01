"""Tests for the :class:`GridOccupancy` quality check.

``GridOccupancy`` subclasses :class:`ExpectedVsDetectedCount` but counts
*distinct filled grid cells* (``nunique`` over ``cell_label``) rather than raw
detections, so doublets collapse to one. It is lower-is-bad: occupancy below
the thresholds warns/fails.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import pytest

from phenotypic.analysis import GridOccupancy as _PublicGridOccupancy
from phenotypic.analysis.qc import ExpectedVsDetectedCount, GridOccupancy
from phenotypic.schema import METADATA


def _layout(image_file: str = "plate1.png", n_cells: int = 96) -> pd.DataFrame:
    """Return a layout frame with one row per expected grid cell."""
    return pd.DataFrame({
        str(METADATA.IMAGE_NAME): [image_file] * n_cells,
        "Object_Label": list(range(1, n_cells + 1)),
    })


def _measurements(
    image_file: str,
    cell_ids: list[int],
) -> pd.DataFrame:
    """Build a measurement frame; one row per colony.

    ``cell_ids`` lists the ``Grid_RowMajorIdx`` of every detected colony, so
    repeats encode doublets. ``Object_Label`` is unique per row.
    """
    return pd.DataFrame({
        str(METADATA.IMAGE_NAME): [image_file] * len(cell_ids),
        "Object_Label": list(range(1, len(cell_ids) + 1)),
        "Grid_RowMajorIdx": list(cell_ids),
    })


class TestPublicSurface:
    """Import + class-metadata wiring."""

    def test_exported_from_analysis_namespace(self) -> None:
        assert _PublicGridOccupancy is GridOccupancy

    def test_is_lower_is_bad_subclass(self) -> None:
        assert issubclass(GridOccupancy, ExpectedVsDetectedCount)
        assert GridOccupancy._HIGHER_IS_BAD is False
        assert GridOccupancy.name == "Occupancy"

    def test_column_helpers(self) -> None:
        assert GridOccupancy.metric_col() == "QC_Occupancy_Metric"
        assert GridOccupancy.flag_col() == "QC_Occupancy_Flag"
        assert GridOccupancy.status_col() == "QC_Occupancy_Status"

    def test_on_defaults_to_label(self) -> None:
        """``on`` is the label guard column; cells are counted via cell_label."""
        chk = GridOccupancy(metadata=_layout(), groupby=[str(METADATA.IMAGE_NAME)])
        assert chk.on == "Object_Label"
        assert chk.cell_label == "Grid_RowMajorIdx"


class TestOccupancyMetric:
    """Filled / expected metric across the pass/warn/fail bands."""

    def test_fully_filled_is_pass(self) -> None:
        chk = GridOccupancy(metadata=_layout(), groupby=[str(METADATA.IMAGE_NAME)])
        result = chk.analyze(_measurements("plate1.png", list(range(96))))

        assert (result["QC_Occupancy_Filled"] == 96).all()
        assert (result["QC_Occupancy_Expected"] == 96).all()
        assert (result["QC_Occupancy_Vacant"] == 0).all()
        assert (result["QC_Occupancy_Metric"] == 1.0).all()
        assert (result["QC_Occupancy_Status"] == "pass").all()
        assert (~result["QC_Occupancy_Flag"]).all()

    def test_doublets_do_not_inflate_occupancy(self) -> None:
        """92 colonies but only 90 distinct cells → occupancy reads 90/96.

        Two cells (5 and 17) carry doublets; the raw colony count (92) would
        mask the six empty cells, but the distinct-cell count exposes them.
        """
        cell_ids = list(range(90)) + [5, 17]  # 92 colonies, 90 distinct cells
        chk = GridOccupancy(metadata=_layout(), groupby=[str(METADATA.IMAGE_NAME)])
        result = chk.analyze(_measurements("plate1.png", cell_ids))

        assert (result["QC_Occupancy_Filled"] == 90).all()
        assert (result["QC_Occupancy_Vacant"] == 6).all()
        assert math.isclose(
            result["QC_Occupancy_Metric"].iloc[0], 90 / 96, rel_tol=1e-9
        )

    def test_low_occupancy_warns(self) -> None:
        # 93/96 = 0.96875 ... below the 0.95 warn line? No — choose 91/96.
        chk = GridOccupancy(metadata=_layout(), groupby=[str(METADATA.IMAGE_NAME)])
        result = chk.analyze(_measurements("plate1.png", list(range(91))))

        metric = result["QC_Occupancy_Metric"].iloc[0]
        assert math.isclose(metric, 91 / 96, rel_tol=1e-9)
        assert metric <= chk.warn_threshold  # 0.9479 <= 0.95
        assert metric > chk.fail_threshold  # 0.9479 > 0.90
        assert (result["QC_Occupancy_Status"] == "warn").all()
        assert (~result["QC_Occupancy_Flag"]).all()

    def test_very_low_occupancy_fails_and_flags(self) -> None:
        chk = GridOccupancy(metadata=_layout(), groupby=[str(METADATA.IMAGE_NAME)])
        result = chk.analyze(_measurements("plate1.png", list(range(80))))

        metric = result["QC_Occupancy_Metric"].iloc[0]
        assert math.isclose(metric, 80 / 96, rel_tol=1e-9)
        assert metric <= chk.fail_threshold
        assert (result["QC_Occupancy_Status"] == "fail").all()
        assert result["QC_Occupancy_Flag"].all()


class TestThresholdValidation:
    """Lower-is-bad threshold ordering is enforced by the base validator."""

    def test_inverted_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError):
            GridOccupancy(
                metadata=_layout(),
                groupby=[str(METADATA.IMAGE_NAME)],
                warn_threshold=0.90,
                fail_threshold=0.95,  # fail above warn → invalid for lower-is-bad
            )


class TestUnmatchedGroup:
    """A group with no metadata counterpart is forced to fail and recorded."""

    def test_unmatched_group_fails_and_is_recorded(self) -> None:
        chk = GridOccupancy(
            metadata=_layout("plate1.png"), groupby=[str(METADATA.IMAGE_NAME)]
        )
        result = chk.analyze(_measurements("plate2.png", list(range(10))))

        assert (result["QC_Occupancy_Expected"] == 0).all()
        assert (result["QC_Occupancy_Metric"] == 0.0).all()
        assert (result["QC_Occupancy_Status"] == "fail").all()
        assert result["QC_Occupancy_Flag"].all()
        assert chk.unmatched_groups == [("plate2.png",)]

    def test_unmatched_groups_reset_between_runs(self) -> None:
        chk = GridOccupancy(
            metadata=_layout("plate1.png"), groupby=[str(METADATA.IMAGE_NAME)]
        )
        chk.analyze(_measurements("plate2.png", list(range(10))))
        chk.analyze(_measurements("plate1.png", list(range(96))))
        assert chk.unmatched_groups == []


class TestMissingCellColumn:
    """The cell-id column is guarded before the per-group loop runs."""

    def test_missing_cell_label_raises_keyerror(self) -> None:
        chk = GridOccupancy(metadata=_layout(), groupby=[str(METADATA.IMAGE_NAME)])
        bad = pd.DataFrame({
            str(METADATA.IMAGE_NAME): ["plate1.png"] * 3,
            "Object_Label": [1, 2, 3],
        })
        with pytest.raises(KeyError, match="Grid_RowMajorIdx"):
            chk.analyze(bad)


class TestSerializationRoundTrip:
    """Inherits the unified ``metadata`` path round-trip from the parent."""

    def test_round_trips_through_json(self, tmp_path) -> None:
        layout_path = tmp_path / "layout.csv"
        _layout().to_csv(layout_path, index=False)

        chk = GridOccupancy(
            metadata=str(layout_path), groupby=[str(METADATA.IMAGE_NAME)]
        )
        dumped = chk.model_dump(mode="json")
        # The unified field persists the source *path*; the legacy split
        # ``metadata_source`` field is gone (hard cutover).
        assert dumped["metadata"] == str(layout_path)
        assert "metadata_source" not in dumped

        rebuilt = GridOccupancy(**dumped)
        result = rebuilt.analyze(_measurements("plate1.png", list(range(96))))
        assert (result["QC_Occupancy_Metric"] == 1.0).all()


class TestDash:
    """Plotly output."""

    def test_dash_returns_figure_after_analyze(self) -> None:
        chk = GridOccupancy(metadata=_layout(), groupby=[str(METADATA.IMAGE_NAME)])
        chk.analyze(_measurements("plate1.png", list(range(80))))
        fig = chk.dash()
        assert isinstance(fig, go.Figure)

    def test_dash_before_analyze_raises(self) -> None:
        chk = GridOccupancy(metadata=_layout(), groupby=[str(METADATA.IMAGE_NAME)])
        with pytest.raises(RuntimeError, match="analyze"):
            chk.dash()
