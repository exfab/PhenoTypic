"""The join, checked against a real migrated run rather than a fixture.

A value shown against the wrong colony is the one failure this feature must
not have, and it is exactly the failure a synthetic fixture cannot catch: a
fixture built by the same code under test will agree with itself about label
ordering, about which store a stem resolves to, and about whether the join
key is positional. Only a run written by the CLI months earlier can disagree.

The run is the OME-Zarr migration test at
``ucr_029_e_d_Maresca/.../2026-08-11-migration-test`` -- 36 stores, four of
them carrying no ``tables`` descriptor at all. It lives outside the repo, so
this module skips where it is absent; where it is present it is the only
evidence that the join holds against data nobody wrote for it.

The expected values below are read from the Parquet **by a second, independent
path** (pandas over ``pyarrow``, no ``phenotypic`` import), so an assertion
failure means the reader and the file disagree, not that the reader disagrees
with itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

#: A real migrated run. Not a repo fixture -- see the module docstring.
MIGRATION_RUN = Path(
    "/rhome/anguy344/bigdata_exfab/projects/ucr_029_e_d_Maresca/data/"
    "results/2026-08-11-migration-test"
)
DATASET = "7-24-26_redo_full"

#: One store out of the 36. 5086x3132, 5 pyramid levels, an 8x12 grid, and a
#: 71 KB embedded table whose ``target`` names ``Object_Label`` against
#: ``rgb/labels/objmap``.
STEM = "d000475_280_079_2026-08-01_23-40-28"

pytestmark = pytest.mark.skipif(
    not MIGRATION_RUN.is_dir(),
    reason=f"migration-test run not present at {MIGRATION_RUN}",
)


@pytest.fixture(scope="module")
def store() -> Path:
    return MIGRATION_RUN / "results" / DATASET / "zarr" / f"{STEM}.ome.zarr"


@pytest.fixture(scope="module")
def output_root(tmp_path_factory):
    """The real run, discovered read-only with its caches sent elsewhere.

    ``cache_root`` is a tmp directory precisely because it must not be
    inside the selected output: this is somebody's real results tree and
    discovery must leave it byte-unchanged.
    """
    from phenotypic.gui.results_viewer._output_root import OutputRoot

    return OutputRoot.discover(
        MIGRATION_RUN,
        cache_root=tmp_path_factory.mktemp("migration-run-viewer-cache"),
    )


@pytest.fixture(scope="module")
def expected_areas(store: Path) -> dict[int, float]:
    """``{Object_Label: Shape_Area}`` read straight off the Parquet.

    Deliberately not through ``phenotypic``: this is the independent side of
    the comparison.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(store / MEASUREMENT_TABLE_RELATIVE_PATH).to_pandas()
    return {
        int(label): float(area)
        for label, area in zip(
            table["Object_Label"], table["Shape_Area"], strict=True
        )
    }


def test_the_reader_reproduces_the_parquet_exactly(
    store: Path, expected_areas: dict[int, float]
) -> None:
    """Every label maps to the area the file records for that label."""
    from phenotypic.sdk_ import read_embedded_measurement_column

    assert read_embedded_measurement_column(store, "Shape_Area") == (
        expected_areas
    )


def test_the_reader_is_keyed_by_label_not_by_row_order(
    store: Path, expected_areas: dict[int, float]
) -> None:
    """A positional join would still pass the equality above on sorted data.

    ``Object_Label`` runs 1..7 in row order here, so ``dict(enumerate(...))``
    shifted by one reproduces the same mapping. Naming a specific label and
    its value is what distinguishes reading the join key from reading the
    index -- and it is the failure mode that puts a number on the wrong card.
    """
    from phenotypic.sdk_ import read_embedded_measurement_column

    values = read_embedded_measurement_column(store, "Shape_Area")
    # Label 4 is the small colony in this plate: 770 px against its
    # neighbours' 77k-106k. Off-by-one on the join key and this label wears
    # a neighbour's area.
    assert values[4] == pytest.approx(770.0)
    assert values[4] == pytest.approx(expected_areas[4])
    assert min(values, key=lambda label: values[label]) == 4


def test_the_value_reaches_the_card_it_was_measured_for(
    output_root, store: Path, expected_areas: dict[int, float]
) -> None:
    """End to end: store -> source helper -> the grid's own cell key.

    ``measurement_values_for`` is what the Colony render callback calls, and
    it keys on ``(image_file, label)`` -- the same key ``build_grid`` uses to
    look a value up per cell. If the two ever disagree, this is where it
    shows.
    """
    from phenotypic.gui.results_viewer._measurement_source import (
        measurement_values_for,
    )

    values = measurement_values_for(
        output_root, [(DATASET, STEM)], "Shape_Area"
    )
    assert values == {
        (STEM, label): area for label, area in expected_areas.items()
    }
    assert values[(STEM, 4)] == pytest.approx(770.0)


def test_the_picker_offers_numeric_columns_and_not_the_hex_one(
    output_root,
) -> None:
    """``ColorLab_MedoidColorHex`` is declared and still has no scale.

    The store declares 136 columns including that one. The picker is the
    place a string column is filtered out; the route refuses it as a second
    line of defence.
    """
    from phenotypic.gui.results_viewer._measurement_source import (
        displayable_measurement_columns,
    )

    columns = displayable_measurement_columns(
        output_root, output_root.image_pairs(output_root.master_df)
    )
    assert "Shape_Area" in columns
    assert "Intensity_MeanIntensity" in columns
    assert "ColorLab_MedoidColorHex" not in columns
    assert "Metadata_ImageName" not in columns


def test_the_four_table_less_stores_contribute_nothing_and_raise_nothing(
    output_root,
) -> None:
    """Four of the 36 stores carry no ``tables`` descriptor.

    That is a normal state, not a partial migration: the picker still fills
    from the other 32, and reading a column simply yields no keys for those
    four. Nothing here may raise, and nothing may report "pending".
    """
    from phenotypic.gui.results_viewer._measurement_source import (
        measurement_values_for,
    )

    zarr_dir = MIGRATION_RUN / "results" / DATASET / "zarr"
    table_less = [
        candidate.name.removesuffix(".ome.zarr")
        for candidate in sorted(zarr_dir.glob("*.ome.zarr"))
        if not (candidate / MEASUREMENT_TABLE_RELATIVE_PATH).is_file()
    ]
    assert table_less, "expected at least one store with no embedded table"

    values = measurement_values_for(
        output_root,
        [(DATASET, stem) for stem in table_less],
        "Shape_Area",
    )
    assert values == {}
