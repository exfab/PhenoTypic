"""Every header the Measurements reference documents is one the producer writes.

The reference renders each column through ``MeasurementProducer.output_header``;
these tests run the producers and check the documented names, with their
placeholders filled in, against the columns actually returned. Together with
``tests/unit/docs/test_measurements_ref_extension.py`` (pages show
``output_header``) they close the loop from code to docs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis import EdgeCorrector, ICC, LogGrowthModel
from phenotypic.analysis.abc_ import QualityCheck
from phenotypic.schema import QUALITY_CHECK, TEXTURE
from phenotypic.util import measurement_producers

_ON = "Size_Area"


def _producer(name: str):
    return next(p for p in measurement_producers() if p.output_key == name)


def _documented(name: str, on: str | None = None) -> list[str]:
    producer = _producer(name)
    return [
        producer.output_header(member, on)
        for info in (*producer.primary_infos, *producer.shared_infos)
        for member in info
    ]


@pytest.fixture(scope="module")
def growth_frame() -> pd.DataFrame:
    """Two plates of twelve wells over six timepoints, logistic colony area."""
    rng = np.random.default_rng(0)
    rows = []
    for plate in ("P1", "P2"):
        for well in range(12):
            for time in range(6):
                area = 400 / (1 + np.exp(-(time - 2.5))) + rng.normal(0, 3)
                rows.append(
                    {
                        "Metadata_SourcePlate": plate,
                        "Grid_RowMajorIdx": well,
                        "Object_Label": well + 1,
                        "Metadata_Time": time,
                        "Metadata_BioReplicate": well % 2,
                        "Metadata_Clone": f"S{well // 4}",
                        _ON: area,
                    }
                )
    return pd.DataFrame(rows)


def test_quality_check_documents_the_named_qc_trio(growth_frame: pd.DataFrame) -> None:
    out = ICC(on=_ON, groupby=["Metadata_SourcePlate", "Metadata_Clone"]).analyze(growth_frame)
    assert set(_documented("ICC")) <= set(out.columns)
    assert str(QUALITY_CHECK.METRIC) not in out.columns


@pytest.mark.parametrize(
    "check", [p.producer for p in measurement_producers() if issubclass(p.producer, QualityCheck)]
)
def test_every_quality_checks_trio_matches_its_column_accessors(check: type) -> None:
    assert check.output_header(QUALITY_CHECK.METRIC) == check.metric_col()
    assert check.output_header(QUALITY_CHECK.FLAG) == check.flag_col()
    assert check.output_header(QUALITY_CHECK.STATUS) == check.status_col()


def test_growth_model_documents_metric_qualified_headers(growth_frame: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = LogGrowthModel(
            on=_ON, groupby=["Metadata_SourcePlate", "Grid_RowMajorIdx"]
        ).analyze(growth_frame)
    documented = _documented("LogGrowthModel", _ON)
    assert set(documented) <= set(out.columns)
    assert all("_Area_" in header for header in documented)
    placeholders = LogGrowthModel.output_header_placeholders()
    assert "LogGrowthModel_Area_r" in placeholders["<metric>"]


def test_edge_corrector_documents_hyphenated_headers(growth_frame: pd.DataFrame) -> None:
    out = EdgeCorrector(
        on=_ON, groupby=["Metadata_SourcePlate"], nrows=3, ncols=4, pvalue=0.0
    ).analyze(growth_frame)
    assert set(_documented("EdgeCorrector", _ON)) == {
        f"EdgeCorrection_NewVal-{_ON}",
        f"EdgeCorrection_Cap-{_ON}",
    }
    assert set(_documented("EdgeCorrector", _ON)) <= set(out.columns)


@pytest.fixture(scope="module")
def detected_plate():
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OtsuDetector().apply(load_synth_yeast_plate())


def test_texture_pattern_expands_to_the_columns_written(detected_plate) -> None:
    from phenotypic.measure import MeasureTexture

    out = MeasureTexture(scale=5).measure(detected_plate)
    directions = ("deg000", "deg045", "deg090", "deg135", "avg")
    expanded = {
        pattern.replace("<x>", "05").replace("<direction>", direction)
        for pattern in _documented("MeasureTexture")
        for direction in directions
    }
    written = {str(column) for column in out.columns} - {"Object_Label"}
    assert expanded == written
    assert set(MeasureTexture.output_header_placeholders()) == {"<x>", "<direction>"}
    assert TEXTURE.header(TEXTURE.CONTRAST, "deg045", 5) == "Texture_Contrast-deg045-scale05"


@pytest.mark.parametrize(
    "name",
    [
        p.output_key
        for p in measurement_producers()
        if hasattr(p.producer, "get_measurement_infoclasses")
        and not p.producer.output_header_placeholders()
    ],
)
def test_fixed_name_measurers_write_exactly_their_documented_headers(
    name: str, detected_plate
) -> None:
    producer = _producer(name)
    operation = producer.producer()
    out = operation.measure(detected_plate)
    active = operation.get_measurement_infoclasses()
    documented = {
        producer.output_header(member) for info in active for member in info
    }
    written = {str(column) for column in out.columns}
    assert documented <= written, sorted(documented - written)


def test_a_quality_check_without_a_name_raises_instead_of_writing_placeholders() -> None:
    class NamelessCheck(QualityCheck):
        """A check that forgot to set ``name``."""

    with pytest.raises(AttributeError, match="NamelessCheck defines no `name`"):
        NamelessCheck.metric_col()
    with pytest.raises(AttributeError, match="defines no `name`"):
        NamelessCheck.output_header(QUALITY_CHECK.FLAG)
    # Only the abstract base renders the documented placeholder.
    assert QualityCheck.output_header(QUALITY_CHECK.METRIC) == "QC_<name>_Metric"
    assert ICC.output_header(QUALITY_CHECK.METRIC) == "QC_ICC_Metric"


def test_quality_check_docstring_and_emitter_share_one_formatter() -> None:
    for member in QUALITY_CHECK:
        assert ICC.output_header(member) == QUALITY_CHECK.header(member, ICC.name)
        assert f"``{QUALITY_CHECK.header(member, ICC.name)}``" in (ICC.__doc__ or "")
        assert QualityCheck.output_header(member) == QUALITY_CHECK.header(member)


def test_edge_corrector_worker_names_columns_through_its_own_class(
    growth_frame: pd.DataFrame,
) -> None:
    class RenamedEdge(EdgeCorrector):
        @classmethod
        def output_header(cls, member, on=None):
            return "Renamed_" + super().output_header(member, on)

    out = RenamedEdge(
        on=_ON, groupby=["Metadata_SourcePlate"], nrows=3, ncols=4, pvalue=0.0
    ).analyze(growth_frame)
    assert f"Renamed_EdgeCorrection_NewVal-{_ON}" in out.columns
    assert f"EdgeCorrection_NewVal-{_ON}" not in out.columns
