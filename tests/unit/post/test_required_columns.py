"""Post ops report missing and produced columns with their own resolution rules.

Spec ``2026-09-24-cli-preflight`` §8 (F23; review R6). ``preflight_columns``
must agree with ``_operate``: a column it calls present must not raise at run
time, and one it calls missing must.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from phenotypic.post import (
    AppendString,
    ExpandMetadata,
    JoinMetadata,
    MergeMetadata,
    PrependString,
)


def test_append_string_needs_its_column() -> None:
    op = AppendString(column="Strain", value="_x")

    assert op.preflight_columns(["Metadata_Strain"]) == ((), ())
    assert op.preflight_columns(["Metadata_Other"]) == (("Metadata_Strain",), ())


def test_prepend_string_needs_its_column() -> None:
    op = PrependString(column="Strain", value="x_")

    assert op.preflight_columns(["Metadata_Other"])[0] == ("Metadata_Strain",)


def test_expand_metadata_needs_its_source_and_produces_its_labels() -> None:
    op = ExpandMetadata(column="Tag", labels=["Part1", "Part2"], delimiter="_")

    missing, produced = op.preflight_columns(["Metadata_Tag"])

    assert missing == ()
    assert set(produced) == {"Metadata_Part1", "Metadata_Part2"}


def test_merge_metadata_needs_every_source_and_produces_its_label() -> None:
    op = MergeMetadata(columns=["A", "B"], label="AB")

    assert op.preflight_columns(["Metadata_A"]) == (("Metadata_B",), ("Metadata_AB",))


def test_join_metadata_reports_keys_in_the_frames_spelling(tmp_path: Path) -> None:
    """Review R6: ``on`` holds the table's spelling after validation."""
    table = tmp_path / "strains.csv"
    table.write_text("Strain,Media\nWT,YPD\n", encoding="utf-8")
    op = JoinMetadata(metadata=table, on=["Strain"])

    assert op.preflight_columns(["Metadata_Strain", "Size_Area"]) == ((), ("Metadata_Media",))
    missing, _ = op.preflight_columns(["Size_Area"])
    assert missing == ("Metadata_Strain",)


@pytest.mark.parametrize(
    "op",
    [
        AppendString(column="Strain", value="_x"),
        ExpandMetadata(column="Strain", labels=["A", "B"]),
        MergeMetadata(columns=["Strain", "Media"], label="SM"),
    ],
)
def test_the_prediction_agrees_with_the_operation(op) -> None:
    """Present means ``apply`` succeeds; missing means it raises ``KeyError``."""
    present = pd.DataFrame({"Metadata_Strain": ["WT_A"], "Metadata_Media": ["YPD"]})
    absent = pd.DataFrame({"Metadata_Other": ["x"]})

    assert op.preflight_columns(list(present.columns))[0] == ()
    op.apply(present)
    assert op.preflight_columns(list(absent.columns))[0]
    with pytest.raises(KeyError):
        op.apply(absent)
