"""Tests for the grid-quality measurers MeasureGridSpread and MeasureGridLinRegStats.

Both measurers used to leak columns that belong to no ``phenotypic.schema``
enum: MeasureGridSpread returned a section-level frame (indexed by
``Grid_RowMajorIdx``, with a ``count`` column and no ``Object_Label``), which
made ``ImagePipeline.measure()`` raise ``KeyError: Object_Label``; and
MeasureGridLinRegStats returned an ``index`` column plus copies of the
``grid.info()`` block, so ``index`` reached ``deliverables/measurements.csv``.
"""

import numpy as np
import pytest

from phenotypic import GridImage, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.grid import ManualGridFinder
from phenotypic.measure import MeasureGridLinRegStats, MeasureGridSpread
from phenotypic.schema import GRID_LINREG_STATS, GRID_SPREAD, OBJECT


def _disc(objmap: np.ndarray, label: int, rr: int, cc: int, radius: int) -> None:
    """Stamp a filled disc with ``label`` centred on (rr, cc) into ``objmap``."""
    rr_grid, cc_grid = np.ogrid[:objmap.shape[0], :objmap.shape[1]]
    objmap[(rr_grid - rr) ** 2 + (cc_grid - cc) ** 2 <= radius ** 2] = label


@pytest.fixture
def three_section_image() -> GridImage:
    """A 1x3 grid whose sections hold 3, 1 and 2 discs respectively.

    Section 0's three centroids form a right isosceles triangle, so two distinct
    pairs share the same distance (20 px). Section counts differ, so the
    count-sorted section order (0, 2, 1) differs from the section-label order.
    """
    height, width = 100, 300
    image = GridImage(
            arr=np.zeros((height, width, 3), dtype=np.uint8),
            grid_finder=ManualGridFinder(
                    row_edges=np.array([0, height]),
                    col_edges=np.array([0, 100, 200, 300]),
            ),
    )
    objmap = np.zeros((height, width), dtype=np.uint16)
    # Section 0: (rr, cc) = (20, 20), (20, 40), (40, 20)
    _disc(objmap, 1, 20, 20, 4)
    _disc(objmap, 2, 20, 40, 4)
    _disc(objmap, 3, 40, 20, 4)
    # Section 1: a single colony
    _disc(objmap, 4, 50, 150, 4)
    # Section 2: two colonies 30 px apart along a row
    _disc(objmap, 5, 50, 220, 4)
    _disc(objmap, 6, 50, 250, 4)
    image.objmap[:] = objmap
    return image


class TestMeasureGridSpread:
    def test_output_is_per_object_with_schema_columns_only(self, three_section_image):
        df = MeasureGridSpread().measure(three_section_image)

        assert list(df.columns) == [OBJECT.LABEL, *GRID_SPREAD.get_headers()]
        assert sorted(df[OBJECT.LABEL]) == [1, 2, 3, 4, 5, 6]

    def test_values_follow_each_object_section(self, three_section_image):
        df = MeasureGridSpread().measure(three_section_image).set_index(OBJECT.LABEL)

        # Section 0: pairs 20, 20 and sqrt(800) px -> 400 + 400 + 800. The two
        # equal 20 px pairs are distinct pairs and must both be counted.
        expected_spread = {1: 1600.0, 2: 1600.0, 3: 1600.0, 4: 0.0, 5: 900.0, 6: 900.0}
        expected_count = {1: 3, 2: 3, 3: 3, 4: 1, 5: 2, 6: 2}
        for label in expected_spread:
            assert df.loc[label, str(GRID_SPREAD.OBJECT_SPREAD)] == pytest.approx(
                    expected_spread[label], rel=1e-12)
            assert df.loc[label, str(GRID_SPREAD.OBJECT_COUNT)] == expected_count[label]

    def test_empty_image_returns_no_rows(self):
        image = GridImage(
                arr=np.zeros((50, 50, 3), dtype=np.uint8),
                grid_finder=ManualGridFinder(
                        row_edges=np.array([0, 50]), col_edges=np.array([0, 50])),
        )
        df = MeasureGridSpread().measure(image)

        assert len(df) == 0
        assert list(df.columns) == [OBJECT.LABEL, *GRID_SPREAD.get_headers()]


class TestMeasureGridLinRegStats:
    def test_output_carries_schema_columns_only(self, synth_plate_detected):
        df = MeasureGridLinRegStats().measure(synth_plate_detected)

        assert df.index.name == OBJECT.LABEL
        assert list(df.columns) == GRID_LINREG_STATS.get_headers()
        assert len(df) == synth_plate_detected.num_objects


class TestGridQualityInPipeline:
    @pytest.mark.parametrize(
            "meas",
            [
                [MeasureGridSpread()],
                [MeasureGridSpread(), MeasureGridLinRegStats()],
                [MeasureGridLinRegStats(), MeasureGridSpread()],
            ],
            ids=["spread-only", "spread-first", "linreg-first"],
    )
    def test_pipeline_emits_only_schema_and_info_columns(self, meas):
        pipe = ImagePipeline(ops=[OtsuDetector()], meas=meas)
        image = pipe.apply(load_synth_yeast_plate())
        df = pipe.measure(image)

        measurer_headers = set()
        for m in meas:
            measurer_headers |= set(m._measurement_infoclass.get_headers())
        info_columns = set(image.grid.info(include_metadata=True).columns)

        assert len(df) == image.num_objects
        assert df[OBJECT.LABEL].is_unique
        assert measurer_headers <= set(df.columns)
        assert set(df.columns) - info_columns == measurer_headers
