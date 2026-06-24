"""Round-trip tests for the pipeline ``qc`` section (Phase B).

Covers the ``pipeline.json`` ``qc`` array contract:

* a LIST of ``{instance_id, class, enabled, params}`` entries (not the bare
  ``{class, params}`` analyzer shape), preserving stable ``instance_id`` +
  ``enabled`` across ``to_json``/``from_json``;
* **duplicate-class** entries are sliced by ``instance_id`` (not class);
* an :class:`ExpectedVsDetectedCount` configured from a metadata **path**
  round-trips that path under the unified ``metadata`` key (never a null
  DataFrame), and a rebuilt instance re-reads the layout file;
* an unknown QC class follows the **analyzer** path: skip+warn when
  ``skip_unknown_analyzers=True``, hard error when ``False`` — one stale
  entry must never brick pipeline load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount, ReplicateAgreement
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape
from phenotypic.sdk_._qc_recipe import QcRecipeEntry


@pytest.fixture
def layout_csv(tmp_path: Path) -> Path:
    """Write a 96-well layout CSV and return its path."""
    md = pd.DataFrame({
        "Metadata_ImageFile": ["plate1.png"] * 96,
        "Object_Label": list(range(1, 97)),
    })
    path = tmp_path / "layout.csv"
    md.to_csv(path, index=False)
    return path


def _mixed_qc_pipeline(layout_csv: Path) -> ImagePipeline:
    """A pipeline with a duplicate-class pair + a path-based Count check."""
    return ImagePipeline(
        ops=[OtsuDetector()],
        meas=[MeasureShape()],
        qc=[
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={"on": "Size_Area", "groupby": ["Metadata_ImageFile"]},
                instance_id="qc-SE-area0001",
                enabled=True,
            ),
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={
                    "on": "Size_Perimeter",
                    "groupby": ["Metadata_ImageFile"],
                },
                instance_id="qc-SE-perim002",
                enabled=False,
            ),
            QcRecipeEntry(
                cls=ExpectedVsDetectedCount,
                params={
                    "metadata": str(layout_csv),
                    "groupby": ["Metadata_ImageFile"],
                },
                instance_id="qc-Count-cnt003",
                enabled=True,
            ),
        ],
    )


class TestQcArrayShape:
    """The serialized ``qc`` array uses the dedicated entry shape."""

    def test_qc_is_a_list_of_entry_dicts(self, layout_csv: Path) -> None:
        cfg = json.loads(_mixed_qc_pipeline(layout_csv).to_json())

        assert isinstance(cfg["qc"], list)
        for entry in cfg["qc"]:
            assert set(entry) == {"instance_id", "class", "enabled", "params"}

    def test_qc_key_omitted_when_empty(self) -> None:
        pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])
        cfg = json.loads(pipe.to_json())
        assert "qc" not in cfg

    def test_entry_ids_classes_and_enabled_persisted(
        self, layout_csv: Path
    ) -> None:
        cfg = json.loads(_mixed_qc_pipeline(layout_csv).to_json())
        by_id = {e["instance_id"]: e for e in cfg["qc"]}

        assert by_id["qc-SE-area0001"]["class"] == "ReplicateAgreement"
        assert by_id["qc-SE-area0001"]["enabled"] is True
        assert by_id["qc-SE-perim002"]["enabled"] is False


class TestDuplicateClassRoundTrip:
    """Duplicate-class entries are distinguished by ``instance_id``."""

    def test_round_trip_preserves_both_duplicate_class_entries(
        self, layout_csv: Path
    ) -> None:
        pipe2 = ImagePipeline.from_json(_mixed_qc_pipeline(layout_csv).to_json())
        # Slice by instance_id, NOT class — both are ReplicateAgreement.
        by_id = {e.instance_id: e for e in pipe2.get_qc()}

        assert by_id["qc-SE-area0001"].cls is ReplicateAgreement
        assert by_id["qc-SE-perim002"].cls is ReplicateAgreement
        assert by_id["qc-SE-area0001"].params["on"] == "Size_Area"
        assert by_id["qc-SE-perim002"].params["on"] == "Size_Perimeter"
        assert by_id["qc-SE-area0001"].enabled is True
        assert by_id["qc-SE-perim002"].enabled is False


class TestExpectedVsDetectedMetadataPath:
    """``ExpectedVsDetectedCount`` round-trips its metadata as a path."""

    def test_metadata_path_preserved_no_frame_leak(
        self, layout_csv: Path
    ) -> None:
        cfg = json.loads(_mixed_qc_pipeline(layout_csv).to_json())
        count_params = next(
            e["params"] for e in cfg["qc"] if e["class"] == "ExpectedVsDetectedCount"
        )

        # The unified ``metadata`` field persists the layout *path*; a
        # resolved DataFrame must never leak into the JSON params.
        assert count_params["metadata"] == str(layout_csv)
        # The legacy split field is gone (hard cutover).
        assert "metadata_source" not in count_params

    def test_rebuilt_count_reads_layout_from_path(
        self, layout_csv: Path
    ) -> None:
        pipe2 = ImagePipeline.from_json(_mixed_qc_pipeline(layout_csv).to_json())
        count_entry = next(
            e for e in pipe2.get_qc() if e.cls is ExpectedVsDetectedCount
        )

        rebuilt = count_entry.instantiate()
        # ``metadata`` echoes the path it round-tripped under; the resolved
        # layout frame lives on the private ``_metadata`` slot.
        assert rebuilt.metadata == str(layout_csv)
        assert len(rebuilt._metadata) == 96


class TestUnknownClassTolerance:
    """Unknown QC class → skip+warn (skip=True) or raise (skip=False)."""

    _BAD = {
        "pipe_cfgs": {},
        "meas": {},
        "qc": [
            {
                "instance_id": "qc-Nope-zzz999",
                "class": "NotARealCheck",
                "enabled": True,
                "params": {},
            }
        ],
    }

    def test_skip_true_drops_entry_and_records_warning(self) -> None:
        warnings: list = []
        pipe = ImagePipeline.from_json(
            self._BAD, skip_unknown_analyzers=True, load_warnings=warnings
        )

        assert pipe.get_qc() == []
        assert len(warnings) == 1
        assert warnings[0].slot == "qc"
        assert warnings[0].class_name == "NotARealCheck"

    def test_skip_false_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="NotARealCheck"):
            ImagePipeline.from_json(self._BAD, skip_unknown_analyzers=False)

    def test_one_stale_entry_does_not_drop_valid_siblings(
        self, layout_csv: Path
    ) -> None:
        # A pipeline JSON that mixes a valid entry with a stale one must not
        # lose the valid entry under skip mode.
        good = _mixed_qc_pipeline(layout_csv).to_json()
        cfg = json.loads(good)
        cfg["qc"].append({
            "instance_id": "qc-Nope-zzz999",
            "class": "NotARealCheck",
            "enabled": True,
            "params": {},
        })

        warnings: list = []
        pipe = ImagePipeline.from_json(
            cfg, skip_unknown_analyzers=True, load_warnings=warnings
        )
        ids = {e.instance_id for e in pipe.get_qc()}

        assert ids == {"qc-SE-area0001", "qc-SE-perim002", "qc-Count-cnt003"}
        assert any(w.class_name == "NotARealCheck" for w in warnings)


class TestSetGetQc:
    """``set_qc`` / ``get_qc`` mirror the post/filters accessor contract."""

    def test_set_qc_none_clears(self, layout_csv: Path) -> None:
        pipe = _mixed_qc_pipeline(layout_csv)
        pipe.set_qc(None)
        assert pipe.get_qc() == []

    def test_get_qc_returns_a_copy(self, layout_csv: Path) -> None:
        pipe = _mixed_qc_pipeline(layout_csv)
        got = pipe.get_qc()
        got.clear()
        # Mutating the returned list must not affect the pipeline.
        assert len(pipe.get_qc()) == 3
