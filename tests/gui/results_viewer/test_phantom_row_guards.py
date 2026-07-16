"""Null-``Object_Label`` ("phantom row") guards in the viewer key extractors.

The CLI's ``--metadata`` join is a **left** join, so the post-applied
``deliverables/measurements.parquet`` mirror the results viewer reads carries
phantom rows: metadata for strains that were never detected, with a null
``Object_Label`` and null measurements. Those rows must stay in the mirror (the
user needs to see which strains went undetected), so the key extractors — which
turn frame rows into ``(image_file, object_label)`` curation keys — are what has
to tolerate them. Before the guards, both did ``int(None)`` and raised
``TypeError``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._filtered_state import (
    FilteredMeasurements,
    _extract_keys,
)
from phenotypic.schema import METADATA
from phenotypic.sdk_ import BundleLayout

from tests._output_layout import write_master


def _layout(tmp_path: Path) -> BundleLayout:
    """Full-run-style layout rooted at ``tmp_path`` (deliverables under it)."""
    return BundleLayout(deliverables_base=tmp_path / "deliverables", output_root=tmp_path)


def _master() -> pl.DataFrame:
    """Clean master: two real, detected objects in one image."""
    return pl.DataFrame(
        {
            str(METADATA.IMAGE_NAME): ["plateA", "plateA"],
            "Object_Label": [1, 2],
            "Bbox_CenterRR": [10.0, 20.0],
            "Bbox_CenterCC": [30.0, 40.0],
            "Size_Area": [100.0, 200.0],
        }
    )


def _mirror_with_phantom() -> pl.DataFrame:
    """The left-joined mirror: the two real rows plus one phantom.

    The phantom is an undetected strain — metadata present, every
    detection-derived value null.
    """
    return pl.DataFrame(
        {
            str(METADATA.IMAGE_NAME): ["plateA", "plateA", None],
            "Object_Label": [1, 2, None],
            "Bbox_CenterRR": [10.0, 20.0, None],
            "Bbox_CenterCC": [30.0, 40.0, None],
            "Size_Area": [100.0, 200.0, None],
        },
        schema={
            str(METADATA.IMAGE_NAME): pl.String,
            "Object_Label": pl.Int64,
            "Bbox_CenterRR": pl.Float64,
            "Bbox_CenterCC": pl.Float64,
            "Size_Area": pl.Float64,
        },
    )


def test_removed_count_in_ignores_phantom_rows(tmp_path: Path) -> None:
    """``CurationLabels.removed_count_in`` counts real rows past a phantom.

    Reproduces the live crash: the filter panel calls ``removed_count_in``
    with a frame derived from the mirror on the user's first curation click.
    A null ``Object_Label`` used to reach ``int(None)`` inside ``_keys_of``.
    """
    master = _master()
    write_master(tmp_path, master, csv=False)
    store = CurationLabels.load(_layout(tmp_path), master)
    store.remove("plateA", 1)

    assert store.removed_count_in(_mirror_with_phantom()) == 1


def test_curation_labels_rekey_survives_mirror_fallback(tmp_path: Path) -> None:
    """``_master_index`` tolerates phantoms on the clean-master fallback.

    ``_read_clean_master`` falls back to the mirror when
    ``master_measurements.parquet`` is absent, so a stored labels parquet is
    then re-keyed against a frame that contains phantom rows.
    """
    layout = _layout(tmp_path)
    mirror = _mirror_with_phantom()

    # Seed a durable labels parquet (no clean master on disk → fallback path).
    seed = CurationLabels.load(layout, mirror)
    seed.remove("plateA", 2)
    assert layout.curation_labels_parquet.exists()
    assert not layout.master_parquet.exists()

    reloaded = CurationLabels.load(layout, mirror)

    assert reloaded.labels == {("plateA", 2): "other"}
    assert reloaded.rekey_report.kept == 1


def test_extract_keys_ignores_phantom_rows() -> None:
    """``_extract_keys`` mirrors the ``_keys_of`` guard (parity path)."""
    assert _extract_keys(_mirror_with_phantom()) == {("plateA", 1), ("plateA", 2)}


def test_filtered_measurements_removed_count_in_ignores_phantom_rows(
    tmp_path: Path,
) -> None:
    """``FilteredMeasurements.removed_count_in`` counts real rows past a phantom."""
    master = _master()
    write_master(tmp_path, master, csv=False)
    state = FilteredMeasurements.load(tmp_path, master)
    state.remove("plateA", 1)

    assert state.removed_count_in(_mirror_with_phantom()) == 1
