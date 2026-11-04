"""``ExpectedVsDetectedCount`` without a layout.

**Derive mode** — ``metadata`` unset, so the expected colony count comes from
the distinct positions of the frame being analyzed. Kept out of
``test_expected_vs_detected.py`` so every test there still runs unedited. The
whole point of the guards around derive mode is that the *silent* failure
(expected == detected, every group passes) is impossible, so most of these
tests assert a refusal.

Every frame here is hand-built in the shape the CLI's ``--metadata`` left join
produces: one row per expected colony position per image, phantoms included.
One class reads that frame back from a CSV, as a reader with only the
delivered ``measurements.csv`` would.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from phenotypic import ImagePipeline
from phenotypic.analysis.qc import ExpectedVsDetectedCount, GridOccupancy
from phenotypic.schema import (
    GRID,
    IMAGE,
    METADATA_MATCH,
    OBJECT,
    QUALITY_COUNT,
    SIZE,
)
from phenotypic.analysis.abc_._quality_check import IN_MEMORY_PARAM
from phenotypic.sdk_._qc_recipe import QcRecipeEntry

IMAGE_NAME = str(IMAGE.IMAGE_NAME)
ROW = str(GRID.ROW_NUM)
COL = str(GRID.COL_NUM)
LABEL = str(OBJECT.LABEL)
FLAG = str(METADATA_MATCH.METADATA_ONLY)
EXPECTED = str(QUALITY_COUNT.EXPECTED)
DETECTED = str(QUALITY_COUNT.DETECTED)
DELTA = str(QUALITY_COUNT.DELTA)

#: An arbitrary user column, spelled the way the join's namespace
#: normalization emits it.
STRAIN = "Metadata_Strain"


def _joined_row(
    image: str,
    row: int,
    col: int,
    *,
    label: float | None,
    strain: str = "SYN-01",
) -> dict:
    """One row of a CLI metadata left join.

    ``label=None`` is a phantom: its position is expected, and nothing was
    detected there.
    """
    return {
        IMAGE_NAME: image,
        ROW: row,
        COL: col,
        LABEL: label,
        FLAG: label is None,
        STRAIN: strain,
        str(SIZE.AREA): None if label is None else 100.0 + float(label),
    }


def _joined_frame() -> pd.DataFrame:
    """Two images, four expected positions per image.

    ``p2.png`` misses one of them, so detected and expected differ without any
    doublet in play.
    """
    rows = [
        _joined_row("p1.png", r, c, label=float(2 * r + c))
        for r in (1, 2)
        for c in (1, 2)
    ]
    rows += [
        _joined_row("p2.png", 1, 1, label=11.0),
        _joined_row("p2.png", 1, 2, label=12.0),
        _joined_row("p2.png", 2, 1, label=13.0),
        _joined_row("p2.png", 2, 2, label=None),
    ]
    return pd.DataFrame(rows)


def _distinct_positions(frame: pd.DataFrame, image: str) -> int:
    """Count one image's distinct grid cells, independently of the check."""
    subset = frame.loc[frame[IMAGE_NAME] == image, [IMAGE_NAME, ROW, COL]]
    return int(len(subset.drop_duplicates()))


def _per_image(result: pd.DataFrame, column: str) -> dict[str, int]:
    """Collapse a broadcast QC column to one value per image."""
    return {
        str(image): int(group[column].iloc[0])
        for image, group in result.groupby(IMAGE_NAME, dropna=False)
    }


class TestDeriveModeExpectedCount:
    """``metadata`` unset: expected comes from the frame's own positions."""

    def test_expected_matches_the_frames_own_expected_positions(self) -> None:
        frame = _joined_frame()
        check = ExpectedVsDetectedCount(groupby=[IMAGE_NAME])

        result = check.analyze(frame)

        assert _per_image(result, EXPECTED) == {
            "p1.png": _distinct_positions(frame, "p1.png"),
            "p2.png": _distinct_positions(frame, "p2.png"),
        }

    def test_a_phantom_is_expected_but_not_detected(self) -> None:
        """The divergence the check exists to show, on the derived path."""
        check = ExpectedVsDetectedCount(groupby=[IMAGE_NAME])

        result = check.analyze(_joined_frame())

        # p2.png has four pinned positions and three colonies.
        assert _per_image(result, EXPECTED)["p2.png"] == 4
        assert _per_image(result, DETECTED)["p2.png"] == 3
        assert _per_image(result, DELTA)["p2.png"] == -1
        assert result.loc[result[IMAGE_NAME] == "p2.png", check.metric_col()].iloc[
            0
        ] == pytest.approx(0.25)

    def test_two_objects_in_one_well_count_once_toward_expected(self) -> None:
        """A doublet is one expected position and two detections."""
        frame = pd.DataFrame([
            _joined_row("p1.png", 1, 1, label=1.0),
            _joined_row("p1.png", 1, 1, label=2.0),
            _joined_row("p1.png", 1, 2, label=3.0),
        ])
        check = ExpectedVsDetectedCount(groupby=[IMAGE_NAME])

        result = check.analyze(frame)

        assert _per_image(result, EXPECTED) == {"p1.png": 2}
        assert _per_image(result, DETECTED) == {"p1.png": 3}

    def test_a_custom_position_by_is_honoured(self) -> None:
        """Coarser positions collapse the doublet's neighbour too."""
        frame = pd.DataFrame([
            _joined_row("p1.png", 1, 1, label=1.0),
            _joined_row("p1.png", 1, 2, label=2.0),
        ])
        check = ExpectedVsDetectedCount(
            groupby=[IMAGE_NAME], position_by=[IMAGE_NAME, ROW]
        )

        result = check.analyze(frame)

        assert _per_image(result, EXPECTED) == {"p1.png": 1}

    def test_a_bare_unknown_position_label_is_found_in_the_frame(self) -> None:
        """A position column the user spelled bare, as their metadata CSV does.

        ``Well`` is not a schema member, so the configured ``position_by``
        normalizes it to ``Metadata_Well`` — and the frame's own ``Well``
        column must be normalized to match, which happens only because the
        position columns are among the references the join normalizer serves.
        """
        frame = pd.DataFrame({
            IMAGE_NAME: ["p1.png"] * 3 + ["p2.png"] * 3,
            "Well": ["A1", "A1", "A2", "A1", "A2", "A3"],
            LABEL: [1.0, 2.0, None, 1.0, 2.0, 3.0],
            FLAG: [False, False, True, False, False, False],
        })
        check = ExpectedVsDetectedCount(
            groupby=[IMAGE_NAME], position_by=[IMAGE_NAME, "Well"]
        )
        assert check.position_by == [IMAGE_NAME, "Metadata_Well"]

        result = check.analyze(frame)

        assert _per_image(result, EXPECTED) == {"p1.png": 2, "p2.png": 3}
        assert _per_image(result, DETECTED) == {"p1.png": 2, "p2.png": 3}

    def test_a_bare_position_by_spelling_resolves_to_the_joined_column(self) -> None:
        """``position_by`` gets the same layout namespace rule ``groupby`` does."""
        check = ExpectedVsDetectedCount(
            groupby=["ImageName"], position_by=["ImageName", ROW, COL]
        )

        assert check.position_by == [IMAGE_NAME, ROW, COL]
        assert check.groupby == [IMAGE_NAME]

    def test_grouping_by_a_metadata_column_counts_that_groups_positions(
        self,
    ) -> None:
        """``groupby`` need not be a position column.

        The expectation per strain is the strain's own distinct positions, and
        every group the frame holds has positions, so nothing is unmatched.
        """
        frame = _joined_frame()
        frame.loc[frame[IMAGE_NAME] == "p2.png", STRAIN] = "SYN-02"
        check = ExpectedVsDetectedCount(groupby=[STRAIN])

        result = check.analyze(frame)

        per_strain = result.groupby(STRAIN)[[EXPECTED, DETECTED]].first()
        assert per_strain.to_dict("index") == {
            "SYN-01": {EXPECTED: 4, DETECTED: 4},
            "SYN-02": {EXPECTED: 4, DETECTED: 3},
        }
        assert check.unmatched_groups == []

    def test_reanalyzing_recomputes_the_expectation(self) -> None:
        """The expectation is a property of the frame, not of the instance."""
        check = ExpectedVsDetectedCount(groupby=[IMAGE_NAME])
        check.analyze(_joined_frame())

        smaller = _joined_frame().loc[lambda f: f[COL] == 1]
        result = check.analyze(smaller)

        assert _per_image(result, EXPECTED) == {"p1.png": 2, "p2.png": 2}


class TestDeriveModeOnABareMeasurementsCsv:
    """Review focus: a delivered ``measurements.csv``, read back, no layout.

    A reader who receives only the CSV has no layout frame to pass. The CSV
    round trip is what changes the dtypes: the label comes back as a float
    with NaN on phantoms, the metadata-only flag as a bool column, and the
    grid indices as integers.
    """

    @staticmethod
    def _round_trip(frame: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
        path = tmp_path / "measurements.csv"
        frame.to_csv(path, index=False)
        return pd.read_csv(path)

    def test_the_check_runs_on_the_csv_alone(self, tmp_path: Path) -> None:
        table = self._round_trip(_joined_frame(), tmp_path)
        assert table[LABEL].isna().sum() == 1
        assert table[FLAG].dtype == bool

        check = ExpectedVsDetectedCount(groupby=[IMAGE_NAME])
        result = check.analyze(table)

        assert _per_image(result, EXPECTED) == {"p1.png": 4, "p2.png": 4}
        assert _per_image(result, DETECTED) == {"p1.png": 4, "p2.png": 3}
        statuses = result.groupby(IMAGE_NAME)[check.status_col()].first()
        assert statuses.to_dict() == {"p1.png": "pass", "p2.png": "fail"}

    def test_the_csv_gives_the_same_answer_as_the_frame(self, tmp_path: Path) -> None:
        """Reading through a CSV must not move any count."""
        columns = [IMAGE_NAME, EXPECTED, DETECTED, DELTA]
        direct = ExpectedVsDetectedCount(groupby=[IMAGE_NAME]).analyze(
            _joined_frame()
        )
        via_csv = ExpectedVsDetectedCount(groupby=[IMAGE_NAME]).analyze(
            self._round_trip(_joined_frame(), tmp_path)
        )

        pd.testing.assert_frame_equal(
            direct[columns].reset_index(drop=True),
            via_csv[columns].reset_index(drop=True),
            check_dtype=False,
        )

    def test_an_off_grid_detection_is_named_as_such(self, tmp_path: Path) -> None:
        """A detected object with no grid cell is not metadata at the wrong grain.

        On a real table an object measured outside every cell carries a label
        and no ``Grid_RowNum``/``Grid_ColNum``. The check still refuses — such
        an object has no expected position — but the message must say what it
        is, since the fix is to drop it, not to change the metadata.
        """
        frame = pd.concat(
            [_joined_frame(), pd.DataFrame([_joined_row("p1.png", 1, 1, label=99.0)])],
            ignore_index=True,
        )
        frame.loc[len(frame) - 1, [ROW, COL]] = None
        table = self._round_trip(frame, tmp_path)

        with pytest.raises(ValueError, match="blank value") as excinfo:
            ExpectedVsDetectedCount(groupby=[IMAGE_NAME]).analyze(table)

        message = str(excinfo.value)
        assert "1 detected object(s)" in message
        assert "off-grid" in message
        assert "wrong grain" not in message

    def test_a_csv_without_the_flag_column_is_refused(self, tmp_path: Path) -> None:
        """A table that never went through the metadata join cannot say."""
        table = self._round_trip(_joined_frame().drop(columns=[FLAG]), tmp_path)

        with pytest.raises(ValueError, match="metadata="):
            ExpectedVsDetectedCount(groupby=[IMAGE_NAME]).analyze(table)


class TestDeriveModeRefusals:
    """A frame that cannot say must stop the check, never pass every group."""

    def test_refuses_a_frame_without_the_metadata_only_flag(self) -> None:
        frame = _joined_frame().drop(columns=[FLAG])
        check = ExpectedVsDetectedCount(groupby=[IMAGE_NAME])

        with pytest.raises(ValueError, match=r"metadata="):
            check.analyze(frame)

    def test_the_refusal_names_the_flag_column_and_the_cli(self) -> None:
        frame = _joined_frame().drop(columns=[FLAG])
        check = ExpectedVsDetectedCount(groupby=[IMAGE_NAME])

        with pytest.raises(ValueError) as excinfo:
            check.analyze(frame)

        message = str(excinfo.value)
        assert FLAG in message
        assert "--metadata" in message

    def test_a_frame_without_the_flag_would_otherwise_pass_every_group(
        self,
    ) -> None:
        """Why the refusal exists, stated as a test.

        With no phantoms every expected position is one that was found, so a
        check that fell back to counting rows would report metric 0.0 for a
        plate that missed a colony. The refusal above is what stops it; this
        pins the arithmetic that would otherwise be reported.
        """
        frame = _joined_frame().drop(columns=[FLAG])
        real = frame.loc[frame[LABEL].notna()]

        assert _distinct_positions(real, "p2.png") == int(
            real[real[IMAGE_NAME] == "p2.png"][LABEL].notna().sum()
        )

    def test_refuses_a_blank_position(self) -> None:
        frame = _joined_frame()
        frame.loc[0, COL] = None
        check = ExpectedVsDetectedCount(groupby=[IMAGE_NAME])

        with pytest.raises(ValueError, match="blank value"):
            check.analyze(frame)

    def test_refuses_an_absent_position_column(self) -> None:
        frame = _joined_frame().drop(columns=[COL])
        check = ExpectedVsDetectedCount(groupby=[IMAGE_NAME])

        with pytest.raises(KeyError, match="position columns absent"):
            check.analyze(frame)


class TestLayoutModeIsUnchanged:
    """Supplying ``metadata`` keeps the old behaviour, frame guards included."""

    def test_a_layout_never_consults_the_frames_flag(self) -> None:
        layout = pd.DataFrame({
            IMAGE_NAME: ["p1.png"] * 4,
            "Object_Label": [1, 2, 3, 4],
        })
        # No flag column at all: a layout-configured check must not care.
        frame = _joined_frame().drop(columns=[FLAG])
        check = ExpectedVsDetectedCount(metadata=layout, groupby=[IMAGE_NAME])

        result = check.analyze(frame)

        assert _per_image(result, EXPECTED)["p1.png"] == 4

    def test_a_layout_with_blank_positions_is_not_refused(self) -> None:
        """The position guards belong to derive mode only."""
        layout = pd.DataFrame({IMAGE_NAME: ["p1.png"] * 4, "Object_Label": [1, 2, 3, 4]})
        frame = _joined_frame()
        frame.loc[0, COL] = None
        check = ExpectedVsDetectedCount(metadata=layout, groupby=[IMAGE_NAME])

        result = check.analyze(frame)

        assert _per_image(result, EXPECTED)["p1.png"] == 4


class TestGridOccupancyStillNeedsALayout:
    """The no-layout mode is the count check's, not its occupancy subclass's."""

    def test_no_metadata_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="metadata"):
            GridOccupancy(groupby=[IMAGE_NAME])

    def test_metadata_none_is_refused_with_the_old_message(self) -> None:
        with pytest.raises(ValidationError, match="requires 'metadata'"):
            GridOccupancy(metadata=None, groupby=[IMAGE_NAME])

    def test_a_layout_still_works(self) -> None:
        layout = pd.DataFrame({IMAGE_NAME: ["p1.png"] * 4, "Object_Label": [1, 2, 3, 4]})
        check = GridOccupancy(metadata=layout, groupby=[IMAGE_NAME])

        assert check.metadata is layout

    def test_position_by_is_refused_not_ignored(self) -> None:
        """It only applies to the count check's no-layout mode."""
        layout = pd.DataFrame({IMAGE_NAME: ["p1.png"] * 4, "Object_Label": [1, 2, 3, 4]})

        with pytest.raises(ValidationError, match="takes no position_by"):
            GridOccupancy(
                metadata=layout, groupby=[IMAGE_NAME], position_by=[IMAGE_NAME, ROW]
            )

    def test_position_by_is_not_in_its_schema_or_its_dump(self, tmp_path: Path) -> None:
        layout_csv = tmp_path / "layout.csv"
        pd.DataFrame({IMAGE_NAME: ["p1.png"] * 4, "Object_Label": [1, 2, 3, 4]}).to_csv(
            layout_csv, index=False
        )
        check = GridOccupancy(metadata=str(layout_csv), groupby=[IMAGE_NAME])

        assert "position_by" not in GridOccupancy.model_json_schema()["properties"]
        assert "position_by" not in check.model_dump(mode="json")
        assert "position_by" in ExpectedVsDetectedCount.model_json_schema()["properties"]
        rebuilt = GridOccupancy(**check.model_dump(mode="json"))
        assert rebuilt.metadata == str(layout_csv)


class TestSerializationRoundTrip:
    """Derive mode survives ``pipeline.json``; a frame says it will not."""

    @staticmethod
    def _pipeline(params: dict) -> ImagePipeline:
        return ImagePipeline(
            qc=[
                QcRecipeEntry(
                    cls=ExpectedVsDetectedCount,
                    params=params,
                    instance_id="qc-Count-derive01",
                    enabled=True,
                )
            ]
        )

    def test_defaults_are_absent(self) -> None:
        check = ExpectedVsDetectedCount(groupby=[IMAGE_NAME])

        assert check.metadata is None
        assert check.position_by is None

    def test_derive_mode_serializes_metadata_as_null(self) -> None:
        config = json.loads(self._pipeline({"groupby": [IMAGE_NAME]}).to_json())
        params = next(
            entry["params"]
            for entry in config["qc"]
            if entry["class"] == "ExpectedVsDetectedCount"
        )

        assert params["metadata"] is None
        assert params["position_by"] is None

    def test_derive_mode_round_trips_through_the_pipeline(self) -> None:
        pipeline = self._pipeline(
            {"groupby": [IMAGE_NAME], "position_by": [IMAGE_NAME, ROW, COL]}
        )

        reloaded = ImagePipeline.from_json(pipeline.to_json())
        entry = next(
            e for e in reloaded.get_qc() if e.cls is ExpectedVsDetectedCount
        )
        rebuilt = entry.instantiate()

        assert rebuilt.metadata is None
        assert rebuilt.position_by == [IMAGE_NAME, ROW, COL]
        result = rebuilt.analyze(_joined_frame())
        assert _per_image(result, EXPECTED) == {"p1.png": 4, "p2.png": 4}

    def test_an_in_memory_frame_warns_that_it_will_not_survive_a_reload(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A frame dumps to the placeholder, and writing it says so."""
        layout = pd.DataFrame({IMAGE_NAME: ["p1.png"] * 4, "Object_Label": [1, 2, 3, 4]})
        entry = QcRecipeEntry(
            cls=ExpectedVsDetectedCount,
            params={"metadata": layout, "groupby": [IMAGE_NAME]},
            instance_id="qc-Count-frame01",
            enabled=True,
        )

        with caplog.at_level(logging.WARNING):
            dumped = entry.to_dict()

        assert dumped["params"]["metadata"] == IN_MEMORY_PARAM
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "qc-Count-frame01" in m and "metadata" in m and "reload" in m
            for m in warnings
        ), warnings

    def test_a_path_or_derive_mode_does_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        layout_csv = tmp_path / "layout.csv"
        pd.DataFrame(
            {IMAGE_NAME: ["p1.png"] * 4, "Object_Label": [1, 2, 3, 4]}
        ).to_csv(layout_csv, index=False)
        entries = [
            QcRecipeEntry(
                cls=ExpectedVsDetectedCount,
                params=params,
                instance_id=f"qc-Count-quiet0{index}",
                enabled=True,
            )
            for index, params in enumerate(
                (
                    {"metadata": str(layout_csv), "groupby": [IMAGE_NAME]},
                    {"groupby": [IMAGE_NAME]},
                )
            )
        ]

        with caplog.at_level(logging.WARNING):
            for entry in entries:
                entry.to_dict()

        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestAnInMemoryLayoutDoesNotReloadAsADifferentCheck:
    """A layout given as a DataFrame cannot round-trip, and a reload says so.

    Its JSON form is a placeholder, not ``None``: ``None`` is the count check's
    no-layout configuration, so dumping a frame to ``None`` would reload as a
    check that counts against a different expectation under the same
    ``instance_id``. The placeholder is refused instead, as a missing layout
    was before the no-layout mode existed.
    """

    @staticmethod
    def _layout() -> pd.DataFrame:
        return pd.DataFrame({IMAGE_NAME: ["p1.png"] * 4, "Object_Label": [1, 2, 3, 4]})

    @pytest.mark.parametrize("cls", [ExpectedVsDetectedCount, GridOccupancy])
    def test_the_json_dump_is_the_placeholder_and_a_rebuild_refuses_it(
        self, cls: type
    ) -> None:
        check = cls(metadata=self._layout(), groupby=[IMAGE_NAME])
        dumped = check.model_dump(mode="json")

        assert dumped["metadata"] == IN_MEMORY_PARAM
        with pytest.raises(ValidationError, match="in-memory layout DataFrame"):
            cls(**dumped)

    @pytest.mark.parametrize("cls", [ExpectedVsDetectedCount, GridOccupancy])
    def test_a_pipeline_round_trip_keeps_the_entry_but_refuses_to_build_it(
        self, cls: type, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The entry survives ``pipeline.json``; building it fails loudly."""
        pipeline = ImagePipeline(
            qc=[
                QcRecipeEntry(
                    cls=cls,
                    params={"metadata": self._layout(), "groupby": [IMAGE_NAME]},
                    instance_id="qc-frame-0001",
                    enabled=True,
                )
            ]
        )
        with caplog.at_level(logging.WARNING):
            config = pipeline.to_json()
        assert any("qc-frame-0001" in r.getMessage() for r in caplog.records)

        reloaded = ImagePipeline.from_json(config)
        (entry,) = [e for e in reloaded.get_qc() if e.cls is cls]
        assert entry.params["metadata"] == IN_MEMORY_PARAM
        with pytest.raises(ValidationError, match="in-memory layout DataFrame"):
            entry.instantiate()

    def test_a_real_none_still_means_no_layout(self) -> None:
        """The placeholder is the only refusal; ``None`` is the no-layout mode."""
        dumped = ExpectedVsDetectedCount(groupby=[IMAGE_NAME]).model_dump(mode="json")
        assert dumped["metadata"] is None
        assert ExpectedVsDetectedCount(**dumped).metadata is None
