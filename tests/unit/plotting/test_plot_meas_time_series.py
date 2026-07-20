"""Multi-page measurement time-series behavior."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from phenotypic.plotting import (
    PlotColonyMetricOverTime,
    PlotMeasTimeSeries,
    publish_plot_output,
)
from phenotypic.plotting._writer import FigureAdapter


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strain": ["B", "A", "A", "A", "A", "B", "B", "B"],
            "environment": ["warm", "warm", "warm", "cold", "cold", "warm", "cold", "cold"],
            "replicate": ["r1", "r1", "r1", "r2", "r2", "r1", "r2", "r2"],
            "time": [0, 1, 0, 1, 0, 1, 0, 1],
            "Size_Area": [10.0, 12.0, 11.0, 21.0, 20.0, 13.0, 19.0, 22.0],
            "custom_numeric": [1.0, 2.0, 1.5, 4.0, 3.0, 2.5, 3.5, 4.5],
            "Metadata_note": ["x"] * 8,
        }
    )


def _plot(**kwargs: object) -> PlotMeasTimeSeries:
    return PlotMeasTimeSeries(
        page_by=["strain"],
        environment_by=["environment"],
        replicate_by=["replicate"],
        time="time",
        **kwargs,
    )


def test_one_page_per_strain_and_no_aggregation() -> None:
    frame = _frame()
    original = frame.copy(deep=True)
    output = _plot(measurements=["Size_Area"]).inspect(frame)

    assert [page.label for page in output.pages] == ["A", "B"]
    assert sum(len(trace.x) for trace in output.pages[0].figure.data) == 4
    pd.testing.assert_frame_equal(frame, original)


def test_auto_measurements_include_custom_numeric_in_column_order() -> None:
    output = _plot().inspect(_frame())
    figure = output.pages[0].figure
    y_titles = [axis.title.text for axis in figure.select_yaxes() if axis.title.text]
    assert y_titles[:2] == ["Size_Area", "custom_numeric"]


def test_duplicate_timepoints_remain_distinct_points() -> None:
    frame = _frame()
    duplicate = frame.iloc[[1]].copy()
    duplicate["Size_Area"] = 99.0
    frame = pd.concat([frame, duplicate], ignore_index=True)
    output = _plot(measurements=["Size_Area"]).inspect(frame)
    assert sum(len(trace.x) for trace in output.pages[0].figure.data) == 5


def test_null_page_group_is_preserved() -> None:
    frame = _frame()
    frame.loc[0, "strain"] = None
    output = _plot(measurements=["Size_Area"]).inspect(frame)
    assert any("<null>" in (page.label or "") for page in output.pages)


@pytest.mark.parametrize(
    "page_values",
    [
        [date(2026, 7, 16), date(2026, 7, 17)],
        [
            datetime(2026, 7, 16, 12, 30),
            datetime(2026, 7, 16, 12, 30, 0, 1),
        ],
        [timedelta(seconds=1), timedelta(seconds=1, microseconds=1)],
    ],
    ids=["date", "datetime", "timedelta"],
)
def test_temporal_page_values_have_distinct_json_safe_metadata(
    page_values: list[object],
) -> None:
    frame = pd.DataFrame(
        {
            "strain": page_values,
            "environment": ["warm", "warm"],
            "replicate": ["r1", "r1"],
            "time": [0, 1],
            "Size_Area": [10.0, 11.0],
        }
    )

    output = _plot(measurements=["Size_Area"]).inspect(frame)

    assert len(output.pages) == 2
    assert len({page.key for page in output.pages}) == 2
    metadata = [dict(page.metadata) for page in output.pages]
    assert json.loads(json.dumps(metadata)) == metadata


def test_nanosecond_timedeltas_publish_distinct_pages_manifest_last(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "strain": [pd.Timedelta(1, unit="ns"), pd.Timedelta(2, unit="ns")],
            "environment": ["warm", "warm"],
            "replicate": ["r1", "r1"],
            "time": [0, 1],
            "Size_Area": [10.0, 11.0],
        }
    )
    output = _plot(measurements=["Size_Area"]).inspect(frame)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"sentinel": "old"}', encoding="utf-8")

    def save_page(_figure: object, path) -> None:
        assert json.loads(manifest_path.read_text()) == {"sentinel": "old"}
        path.write_bytes(b"png")

    monkeypatch.setattr(FigureAdapter, "save_png", save_page)

    manifest = publish_plot_output(output, tmp_path, plot_id="temporal")

    assert len({page.key for page in output.pages}) == 2
    assert [page.metadata["strain"] for page in output.pages] == ["1 ns", "2 ns"]
    assert len(manifest["pages"]) == 2
    assert json.loads(manifest_path.read_text()) == manifest


def test_empty_input_returns_no_pages() -> None:
    assert _plot().inspect(_frame().iloc[:0]).pages == ()


def test_explicit_nonnumeric_measurement_rejected() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        _plot(measurements=["Metadata_note"]).inspect(_frame())


def test_overlapping_roles_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be used for both"):
        PlotMeasTimeSeries(
            page_by=["strain"],
            environment_by=["strain"],
            replicate_by=["replicate"],
            time="time",
        )


def _radius_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strain_index, strain in enumerate(("BY4741", "RM11-1a")):
        for medium_index, medium in enumerate(("SC", "YPD")):
            for treatment_index, treatment in enumerate(("control", "salt")):
                for replicate_index, replicate in enumerate(("bio-1", "bio-2")):
                    for time in (0, 12, 24):
                        rows.append(
                            {
                                "MetadataGenetic_Strain": strain,
                                "MetadataCondition_Media": medium,
                                "MetadataCondition_Treatment": treatment,
                                "MetadataSample_BioReplicate": replicate,
                                "MetadataCulture_Time": time,
                                "Shape_MeanRadius": float(
                                    10 * strain_index
                                    + 3 * medium_index
                                    + 2 * treatment_index
                                    + replicate_index
                                    + time
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def test_colony_radius_pages_group_conditions_and_preserve_replicates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plot = PlotColonyMetricOverTime(
        on="Shape_MeanRadius",
        environment_by=[
            "MetadataCondition_Media",
            "MetadataCondition_Treatment",
        ]
    )
    frame = _radius_frame()
    duplicate = frame[
        (frame["MetadataGenetic_Strain"] == "BY4741")
        & (frame["MetadataCondition_Media"] == "SC")
        & (frame["MetadataCondition_Treatment"] == "control")
        & (frame["MetadataSample_BioReplicate"] == "bio-1")
        & (frame["MetadataCulture_Time"] == 12)
    ].copy()
    duplicate["Shape_MeanRadius"] = 99.0
    frame = pd.concat([frame, duplicate], ignore_index=True)

    output = plot.inspect(frame)

    assert [page.label for page in output.pages] == ["BY4741", "RM11-1a"]
    for page in output.pages:
        figure = page.figure
        assert len(figure.data) == 8  # four conditions, two replicates each
        assert {trace.name for trace in figure.data} == {
            "MetadataSample_BioReplicate=bio-1",
            "MetadataSample_BioReplicate=bio-2",
        }
        assert len(figure.layout.annotations) == 4

    by4741_traces = output.pages[0].figure.data
    assert [trace.xaxis for trace in by4741_traces] == [
        "x",
        "x",
        "x2",
        "x2",
        "x3",
        "x3",
        "x4",
        "x4",
    ]
    assert list(by4741_traces[0].x) == [0, 12, 12, 24]
    assert list(by4741_traces[0].y) == [0.0, 12.0, 99.0, 24.0]
    assert list(by4741_traces[1].y) == [1.0, 13.0, 25.0]
    assert list(by4741_traces[2].y) == [2.0, 14.0, 26.0]
    assert list(by4741_traces[4].y) == [3.0, 15.0, 27.0]
    assert list(by4741_traces[6].y) == [5.0, 17.0, 29.0]

    rm11_traces = output.pages[1].figure.data
    assert all(list(trace.x) == [0, 12, 24] for trace in rm11_traces)
    assert list(rm11_traces[0].y) == [10.0, 22.0, 34.0]

    monkeypatch.setattr(
        FigureAdapter,
        "save_png",
        lambda _figure, path: path.write_bytes(b"png"),
    )
    destination = (
        tmp_path
        / "deliverables"
        / "plots"
        / type(plot).__name__
    )

    manifest = publish_plot_output(
        output,
        destination,
        plot_id=type(plot).__name__,
    )

    assert [page["file"] for page in manifest["pages"]] == [
        "BY4741.png",
        "RM11-1a.png",
    ]
    assert (destination / "BY4741.png").read_bytes() == b"png"
    assert (destination / "RM11-1a.png").read_bytes() == b"png"


def test_colony_metric_defaults_use_public_schema_columns() -> None:
    plot = PlotColonyMetricOverTime(on="Intensity_MeanIntensity")
    schema = type(plot).model_json_schema()

    assert plot.on == "Intensity_MeanIntensity"
    assert plot.page_by == ["MetadataGenetic_Strain"]
    assert plot.environment_by == ["MetadataCondition_Media"]
    assert plot.replicate_by == ["MetadataSample_BioReplicate"]
    assert plot.time == "MetadataCulture_Time"
    assert "on" in schema["required"]
    assert "measurements" not in schema["properties"]


def test_colony_metric_on_accepts_any_numeric_measurement() -> None:
    plot = PlotColonyMetricOverTime(
        on="custom_numeric",
        page_by=["strain"],
        environment_by=["environment"],
        replicate_by=["replicate"],
        time="time",
    )

    output = plot.inspect(_frame())

    figure = output.pages[0].figure
    y_titles = [axis.title.text for axis in figure.select_yaxes() if axis.title.text]
    assert y_titles == ["custom_numeric"]
    assert list(figure.data[0].y) == [3.0, 4.0]


def test_colony_metric_rejects_removed_measurements_override() -> None:
    plot = PlotColonyMetricOverTime(
        on="custom_numeric",
        page_by=["strain"],
        environment_by=["environment"],
        replicate_by=["replicate"],
        time="time",
    )

    with pytest.raises(ValidationError, match="measurements"):
        plot.inspect(_frame(), measurements=["Size_Area"])


def test_colony_metric_report_rejects_unknown_override() -> None:
    plot = PlotColonyMetricOverTime(
        on="custom_numeric",
        page_by=["strain"],
        environment_by=["environment"],
        replicate_by=["replicate"],
        time="time",
    )

    with pytest.raises(ValidationError, match="typo"):
        plot.report(_frame(), typo=True)
