"""Tests for MeasureBounds measurement operation."""

import numpy as np
import pytest

from phenotypic.measure import MeasureBounds
from phenotypic.schema import OBJECT
from phenotypic.schema import BBOX


class TestMeasureBounds:
    """Tests for MeasureBounds measurement operation."""

    @pytest.fixture
    def sample_image(self, synth_plate):
        # Reuse session-scoped synth_plate from tests/unit/conftest.py.
        # Tests that mutate the image do so via .copy() first.
        return synth_plate

    @pytest.fixture
    def measurer(self):
        return MeasureBounds()

    def test_output_has_required_columns(self, sample_image, measurer):
        df = measurer.measure(sample_image)

        assert df.columns[0] == OBJECT.LABEL
        for col in (
            BBOX.CENTER_RR,
            BBOX.CENTER_CC,
            BBOX.INTENSITY_WEIGHTED_CENTER_RR,
            BBOX.INTENSITY_WEIGHTED_CENTER_CC,
            BBOX.DIST_WEIGHTED_CENTER_RR,
            BBOX.DIST_WEIGHTED_CENTER_CC,
            BBOX.MIN_RR,
            BBOX.MIN_CC,
            BBOX.MAX_RR,
            BBOX.MAX_CC,
        ):
            assert str(col) in df.columns, f"Missing column: {col}"

    def test_row_count_matches_objects(self, sample_image, measurer):
        df = measurer.measure(sample_image)
        objmap = sample_image.objmap[:]
        n_objects = len(np.unique(objmap[objmap > 0]))
        assert len(df) == n_objects

    def test_single_object_does_not_crash(self, sample_image, measurer):
        """Regression: maximum_position returns [(r, c)] for length-1 index;
        the previous conditional `if len(labels) == 1: positions = [positions]`
        double-wrapped to shape (1, 1, 2) and broke positions[:, 1] indexing.
        """
        image = sample_image.copy()
        objmap = np.zeros_like(image.objmap[:])
        objmap[50:100, 50:100] = 1
        image.objmap[:] = objmap

        df = measurer.measure(image)

        assert len(df) == 1
        assert df[OBJECT.LABEL].iloc[0] == 1

        rr = df[str(BBOX.DIST_WEIGHTED_CENTER_RR)].iloc[0]
        cc = df[str(BBOX.DIST_WEIGHTED_CENTER_CC)].iloc[0]
        assert 50 <= rr < 100
        assert 50 <= cc < 100

    def test_empty_objmap_returns_empty_frame(self, sample_image, measurer):
        image = sample_image.copy()
        image.objmap[:] = np.zeros_like(image.objmap[:])

        df = measurer.measure(image)

        assert len(df) == 0
        assert str(BBOX.DIST_WEIGHTED_CENTER_RR) in df.columns
        assert str(BBOX.DIST_WEIGHTED_CENTER_CC) in df.columns

    def test_multiple_objects_distance_centers_inside_bbox(self, sample_image, measurer):
        image = sample_image.copy()
        objmap = np.zeros_like(image.objmap[:])
        objmap[20:60, 20:60] = 1
        objmap[120:180, 120:180] = 2
        objmap[200:240, 40:90] = 3
        image.objmap[:] = objmap

        df = measurer.measure(image)

        assert len(df) == 3
        for _, row in df.iterrows():
            rr = row[str(BBOX.DIST_WEIGHTED_CENTER_RR)]
            cc = row[str(BBOX.DIST_WEIGHTED_CENTER_CC)]
            assert row[str(BBOX.MIN_RR)] <= rr <= row[str(BBOX.MAX_RR)]
            assert row[str(BBOX.MIN_CC)] <= cc <= row[str(BBOX.MAX_CC)]

    def test_dist_center_is_weighted_centroid_not_argmax_on_dumbbell(self, sample_image, measurer):
        """A two-lobe (budding) colony: the DT-weighted centroid sits at the neck
        (~midway), NOT on one lobe as DT-argmax (maximum_position) would."""
        image = sample_image.copy()
        objmap = np.zeros_like(image.objmap[:])
        objmap[100:140, 100:140] = 1     # lobe A, center cc=120
        objmap[100:140, 200:240] = 1     # lobe B, center cc=220
        objmap[118:122, 140:200] = 1     # thin neck joining them
        image.objmap[:] = objmap
        df = measurer.measure(image)
        cc = float(df[str(BBOX.DIST_WEIGHTED_CENTER_CC)].iloc[0])
        rr = float(df[str(BBOX.DIST_WEIGHTED_CENTER_RR)].iloc[0])
        assert 155 < cc < 185      # between the lobes (~170), not ~120 or ~220
        assert 110 < rr < 130
