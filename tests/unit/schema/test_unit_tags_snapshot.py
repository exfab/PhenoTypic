"""Every public non-metadata member's unit, pinned as a literal snapshot.

The scaling test (``tests/unit/measure/test_unit_scaling.py``) catches a tag
whose length power is wrong, but only on the families a synthetic scene can
measure, and never a swap between two units of the same power: ``RADIAN``
for ``DEGREE`` misstates an angle by a factor of 57.3 at length power 0. This
snapshot pins every tag, so any change to one must edit this file on purpose
and be read in review. Keys are ``<schema class>.<member name>``; values are
``UNITS`` keys (``member.unit.value``).

To change a tag: edit the schema, then the matching line here. Do not
regenerate the mapping wholesale; that turns a deliberate edit back into an
unread one.
"""

from __future__ import annotations

import phenotypic.schema as S
from phenotypic.schema import UNITS, MeasurementInfo, MetadataInfo


def _actual_tags() -> dict[str, str]:
    """``<class>.<member>`` to unit key for every public non-metadata member."""
    tags: dict[str, str] = {}
    seen: list[type] = []
    for name in S.__all__:
        cls = getattr(S, name)
        if not (isinstance(cls, type) and issubclass(cls, MeasurementInfo)):
            continue
        # ORIENTATION_ZONES is an alias of ORIENTATION_ZONE_DIAGNOSTIC.
        if cls in seen or not list(cls):
            continue
        seen.append(cls)
        for member in cls:
            if isinstance(member, MetadataInfo):
                continue
            key = f"{cls.__name__}.{member.name}"
            assert key not in tags, f"two schema classes are named {cls.__name__}"
            unit = member.unit
            tags[key] = unit.value if isinstance(unit, UNITS) else repr(unit)
    return tags


def test_every_unit_tag_matches_the_snapshot():
    actual = _actual_tags()
    added = sorted(set(actual) - set(_SNAPSHOT))
    removed = sorted(set(_SNAPSHOT) - set(actual))
    changed = sorted(
        f"{key}: {_SNAPSHOT[key]} -> {actual[key]}"
        for key in set(actual) & set(_SNAPSHOT)
        if actual[key] != _SNAPSHOT[key]
    )
    assert (added, removed, changed) == ([], [], []), (
        f"unit tags differ from the snapshot: added {added}, removed {removed}, "
        f"changed {changed}"
    )


def test_the_snapshot_names_only_real_units():
    keys = {unit.value for unit in UNITS}
    assert sorted(set(_SNAPSHOT.values()) - keys) == []


_SNAPSHOT: dict[str, str] = {
    # METADATA_MATCH
    "METADATA_MATCH.METADATA_ONLY": "not_a_quantity",
    # BBOX
    "BBOX.CENTER_RR": "px",
    "BBOX.MIN_RR": "px",
    "BBOX.MAX_RR": "px",
    "BBOX.CENTER_CC": "px",
    "BBOX.MIN_CC": "px",
    "BBOX.MAX_CC": "px",
    "BBOX.INTENSITY_WEIGHTED_CENTER_RR": "px",
    "BBOX.INTENSITY_WEIGHTED_CENTER_CC": "px",
    "BBOX.DIST_WEIGHTED_CENTER_RR": "px",
    "BBOX.DIST_WEIGHTED_CENTER_CC": "px",
    # ColorComposition
    "ColorComposition.BLACK_PCT": "unitless",
    "ColorComposition.WHITE_PCT": "unitless",
    "ColorComposition.GRAY_PCT": "unitless",
    "ColorComposition.PINK_PCT": "unitless",
    "ColorComposition.BROWN_PCT": "unitless",
    "ColorComposition.RED_PCT": "unitless",
    "ColorComposition.ORANGE_PCT": "unitless",
    "ColorComposition.YELLOW_PCT": "unitless",
    "ColorComposition.GREEN_PCT": "unitless",
    "ColorComposition.CYAN_PCT": "unitless",
    "ColorComposition.BLUE_PCT": "unitless",
    "ColorComposition.PURPLE_PCT": "unitless",
    # ColorHSV
    "ColorHSV.HUE_ROBUST_MEAN": "turn",
    "ColorHSV.SATURATION_ROBUST_MEAN": "unitless",
    "ColorHSV.VALUE_ROBUST_MEAN": "unitless",
    "ColorHSV.HSV_CONE_VARIANCE": "unitless",
    # ColorLab
    "ColorLab.L_STAR_GEOMEDIAN": "cielab",
    "ColorLab.A_STAR_GEOMEDIAN": "cielab",
    "ColorLab.B_STAR_GEOMEDIAN": "cielab",
    "ColorLab.L_STAR_MEDOID": "cielab",
    "ColorLab.A_STAR_MEDOID": "cielab",
    "ColorLab.B_STAR_MEDOID": "cielab",
    "ColorLab.DELTA_E2000_MEDIAN": "cielab",
    "ColorLab.DELTA_E2000_MEAN": "cielab",
    "ColorLab.DELTA_E2000_P95": "cielab",
    "ColorLab.LAB_TOTAL_VARIANCE": "cielab2",
    "ColorLab.MEDOID_COLOR_HEX": "not_a_quantity",
    # Colorxy
    "Colorxy.x_MINIMUM": "unitless",
    "Colorxy.x_Q1": "unitless",
    "Colorxy.x_MEAN": "unitless",
    "Colorxy.x_MEDIAN": "unitless",
    "Colorxy.x_Q3": "unitless",
    "Colorxy.x_MAXIMUM": "unitless",
    "Colorxy.x_STDDEV": "unitless",
    "Colorxy.x_COEFF_VARIANCE": "unitless",
    "Colorxy.y_MINIMUM": "unitless",
    "Colorxy.y_Q1": "unitless",
    "Colorxy.y_MEAN": "unitless",
    "Colorxy.y_MEDIAN": "unitless",
    "Colorxy.y_Q3": "unitless",
    "Colorxy.y_MAXIMUM": "unitless",
    "Colorxy.y_STDDEV": "unitless",
    "Colorxy.y_COEFF_VARIANCE": "unitless",
    # ColorXYZ
    "ColorXYZ.X_MINIMUM": "unitless",
    "ColorXYZ.X_Q1": "unitless",
    "ColorXYZ.X_MEAN": "unitless",
    "ColorXYZ.X_MEDIAN": "unitless",
    "ColorXYZ.X_Q3": "unitless",
    "ColorXYZ.X_MAXIMUM": "unitless",
    "ColorXYZ.X_STDDEV": "unitless",
    "ColorXYZ.X_COEFF_VARIANCE": "unitless",
    "ColorXYZ.Y_MINIMUM": "unitless",
    "ColorXYZ.Y_Q1": "unitless",
    "ColorXYZ.Y_MEAN": "unitless",
    "ColorXYZ.Y_MEDIAN": "unitless",
    "ColorXYZ.Y_Q3": "unitless",
    "ColorXYZ.Y_MAXIMUM": "unitless",
    "ColorXYZ.Y_STDDEV": "unitless",
    "ColorXYZ.Y_COEFF_VARIANCE": "unitless",
    "ColorXYZ.Z_MINIMUM": "unitless",
    "ColorXYZ.Z_Q1": "unitless",
    "ColorXYZ.Z_MEAN": "unitless",
    "ColorXYZ.Z_MEDIAN": "unitless",
    "ColorXYZ.Z_Q3": "unitless",
    "ColorXYZ.Z_MAXIMUM": "unitless",
    "ColorXYZ.Z_STDDEV": "unitless",
    "ColorXYZ.Z_COEFF_VARIANCE": "unitless",
    # CURATION
    "CURATION.ERROR_CATEGORY": "not_a_quantity",
    # LINEAR_CAP_AND_LAG_MODEL
    "LINEAR_CAP_AND_LAG_MODEL.v": "of_input_per_time",
    "LINEAR_CAP_AND_LAG_MODEL.s0": "of_input",
    "LINEAR_CAP_AND_LAG_MODEL.lam": "time",
    "LINEAR_CAP_AND_LAG_MODEL.alpha": "per_time",
    "LINEAR_CAP_AND_LAG_MODEL.smax": "of_input",
    "LINEAR_CAP_AND_LAG_MODEL.beta": "per_input",
    "LINEAR_CAP_AND_LAG_MODEL.mode": "not_a_quantity",
    # EDGE_CORRECTION
    "EDGE_CORRECTION.CORRECTED_CAP": "of_input",
    "EDGE_CORRECTION.NEW_VAL": "of_input",
    # ErrorCategory
    "ErrorCategory.OVERSEGMENTED": "not_a_quantity",
    "ErrorCategory.UNDERSEGMENTED": "not_a_quantity",
    "ErrorCategory.MERGED": "not_a_quantity",
    "ErrorCategory.BACKGROUND_NOISE": "not_a_quantity",
    "ErrorCategory.DEBRIS": "not_a_quantity",
    "ErrorCategory.OTHER": "not_a_quantity",
    # GRID
    "GRID.ROW_NUM": "not_a_quantity",
    "GRID.ROW_INTERVAL_START": "px",
    "GRID.ROW_INTERVAL_END": "px",
    "GRID.COL_NUM": "not_a_quantity",
    "GRID.COL_INTERVAL_START": "px",
    "GRID.COL_INTERVAL_END": "px",
    "GRID.ROW_MAJOR_IDX": "not_a_quantity",
    "GRID.COL_MAJOR_IDX": "not_a_quantity",
    # GRID_LINREG_STATS
    "GRID_LINREG_STATS.ROW_LINREG_M": "unitless",
    "GRID_LINREG_STATS.ROW_LINREG_B": "px",
    "GRID_LINREG_STATS.COL_LINREG_M": "unitless",
    "GRID_LINREG_STATS.COL_LINREG_B": "px",
    "GRID_LINREG_STATS.PRED_RR": "px",
    "GRID_LINREG_STATS.PRED_CC": "px",
    "GRID_LINREG_STATS.RESIDUAL_ERR": "px",
    # NEIGHBOR_DIST
    "NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL": "not_a_quantity",
    "NEIGHBOR_DIST.LEFT_DISTANCE": "px",
    "NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL": "not_a_quantity",
    "NEIGHBOR_DIST.RIGHT_DISTANCE": "px",
    "NEIGHBOR_DIST.ABOVE_NEIGHBOR_OBJ_LABEL": "not_a_quantity",
    "NEIGHBOR_DIST.ABOVE_DISTANCE": "px",
    "NEIGHBOR_DIST.UNDER_NEIGHBOR_OBJ_LABEL": "not_a_quantity",
    "NEIGHBOR_DIST.UNDER_DISTANCE": "px",
    "NEIGHBOR_DIST.NEAREST_OBJ_LABEL": "not_a_quantity",
    "NEIGHBOR_DIST.NEAREST_DISTANCE": "px",
    "NEIGHBOR_DIST.NEAREST_RELATION": "not_a_quantity",
    # GRID_SPREAD
    "GRID_SPREAD.OBJECT_SPREAD": "px2",
    "GRID_SPREAD.OBJECT_COUNT": "unitless",
    # INTENSITY
    "INTENSITY.INTEGRATED_INTENSITY": "intensity_px2",
    "INTENSITY.DENSITY": "intensity",
    "INTENSITY.CONVEX_DENSITY": "intensity",
    "INTENSITY.MINIMUM_INTENSITY": "intensity",
    "INTENSITY.MAXIMUM_INTENSITY": "intensity",
    "INTENSITY.MEAN_INTENSITY": "intensity",
    "INTENSITY.MEDIAN_INTENSITY": "intensity",
    "INTENSITY.STANDARD_DEVIATION_INTENSITY": "intensity",
    "INTENSITY.COEFFICIENT_VARIANCE_INTENSITY": "unitless",
    "INTENSITY.Q1_INTENSITY": "intensity",
    "INTENSITY.Q3_INTENSITY": "intensity",
    "INTENSITY.IQR_INTENSITY": "intensity",
    # LINEAR_LAG_MODEL
    "LINEAR_LAG_MODEL.v": "of_input_per_time",
    "LINEAR_LAG_MODEL.s0": "of_input",
    "LINEAR_LAG_MODEL.lam": "time",
    "LINEAR_LAG_MODEL.alpha": "per_time",
    # LOG_GROWTH_MODEL
    "LOG_GROWTH_MODEL.R_FIT": "per_time",
    "LOG_GROWTH_MODEL.K_FIT": "of_input",
    "LOG_GROWTH_MODEL.N0_FIT": "of_input",
    "LOG_GROWTH_MODEL.LAM": "unitless",
    "LOG_GROWTH_MODEL.BETA": "unitless",
    "LOG_GROWTH_MODEL.GROWTH_RATE": "of_input_per_time",
    "LOG_GROWTH_MODEL.K_MAX": "of_input",
    # MODEL_METRICS
    "MODEL_METRICS.MAE": "of_input",
    "MODEL_METRICS.MSE": "of_input_squared",
    "MODEL_METRICS.RMSE": "of_input",
    "MODEL_METRICS.R2": "unitless",
    "MODEL_METRICS.NUM_SAMPLES": "unitless",
    "MODEL_METRICS.LOSS": "not_a_quantity",
    "MODEL_METRICS.STATUS": "not_a_quantity",
    # OBJECT
    "OBJECT.LABEL": "not_a_quantity",
    # QUALITY_CHECK
    "QUALITY_CHECK.FLAG": "not_a_quantity",
    "QUALITY_CHECK.METRIC": "unitless",
    "QUALITY_CHECK.STATUS": "not_a_quantity",
    # QUALITY_COUNT
    "QUALITY_COUNT.DETECTED": "unitless",
    "QUALITY_COUNT.EXPECTED": "unitless",
    "QUALITY_COUNT.DELTA": "unitless",
    # QUALITY_ICC
    "QUALITY_ICC.NUM_SUBJECTS": "unitless",
    "QUALITY_ICC.NUM_RATERS": "unitless",
    "QUALITY_ICC.NUM_MEMBERS": "unitless",
    # QUALITY_MAD
    "QUALITY_MAD.MEDIAN": "of_input",
    "QUALITY_MAD.MAD": "of_input",
    "QUALITY_MAD.NUM_MEMBERS": "unitless",
    # QUALITY_OCCUPANCY
    "QUALITY_OCCUPANCY.FILLED": "unitless",
    "QUALITY_OCCUPANCY.EXPECTED": "unitless",
    "QUALITY_OCCUPANCY.VACANT": "unitless",
    # QUALITY_SE
    "QUALITY_SE.VALUE": "of_input",
    "QUALITY_SE.MEAN": "of_input",
    "QUALITY_SE.CV": "unitless",
    "QUALITY_SE.NUM_REPLICATES": "unitless",
    # QUALITY_TUKEY
    "QUALITY_TUKEY.LOWER_FENCE": "of_input",
    "QUALITY_TUKEY.UPPER_FENCE": "of_input",
    "QUALITY_TUKEY.NUM_OUTLIERS": "unitless",
    "QUALITY_TUKEY.NUM_MEMBERS": "unitless",
    # QUALITY_ZMAX
    "QUALITY_ZMAX.MEDIAN": "of_input",
    "QUALITY_ZMAX.MAD": "of_input",
    "QUALITY_ZMAX.NUM_MEMBERS": "unitless",
    # RADIAL_EXPANSION
    "RADIAL_EXPANSION.ROBUST_MEAN_RADIUS": "px",
    "RADIAL_EXPANSION.MEAN_RADIUS": "px",
    "RADIAL_EXPANSION.MEDIAN_RADIUS": "px",
    "RADIAL_EXPANSION.NUM_BRANCHES": "unitless",
    "RADIAL_EXPANSION.MAX_BRANCH_LENGTH": "px",
    "RADIAL_EXPANSION.RUNNER_LENGTH": "px",
    "RADIAL_EXPANSION.RUNNER_DETECTED": "not_a_quantity",
    "RADIAL_EXPANSION.CORE_RADIUS": "px",
    # SHAPE
    "SHAPE.CIRCULARITY": "unitless",
    "SHAPE.ECCENTRICITY": "unitless",
    "SHAPE.SOLIDITY": "unitless",
    "SHAPE.EXTENT": "unitless",
    "SHAPE.COMPACTNESS": "unitless",
    "SHAPE.ORIENTATION": "radian",
    "SHAPE.MEAN_BOUNDARY_DIST": "px",
    "SHAPE.MEDIAN_BOUNDARY_DIST": "px",
    # SIZE
    "SIZE.AREA": "px2",
    "SIZE.INTEGRATED_INTENSITY": "intensity_px2",
    "SIZE.PERIMETER": "px",
    "SIZE.CONVEX_AREA": "px2",
    "SIZE.BBOX_AREA": "px2",
    "SIZE.MAJOR_AXIS_LENGTH": "px",
    "SIZE.MINOR_AXIS_LENGTH": "px",
    "SIZE.MIN_FERET_DIAMETER": "px",
    "SIZE.MAX_FERET_DIAMETER": "px",
    "SIZE.INSCRIBED_RADIUS": "px",
    "SIZE.MEDIAN_RADIUS": "px",
    "SIZE.MEAN_RADIUS": "px",
    "SIZE.ROBUST_MEAN_RADIUS": "px",
    "SIZE.MAX_RADIUS": "px",
    # ORIENTATION_ZONE_DIAGNOSTIC
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RAW_PEAK_OVERALL": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RAW_PEAK_DENSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RAW_PEAK_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_P90_OVERALL": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_P90_DENSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_P90_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_P95_OVERALL": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_P95_DENSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_P95_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_MEDIAN_MAGNITUDE_OVERALL": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_MEDIAN_MAGNITUDE_DENSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_MEDIAN_MAGNITUDE_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_ABSOLUTE_AREA_OVERALL": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_ABSOLUTE_AREA_DENSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_ABSOLUTE_AREA_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_TOTAL_VARIATION_OVERALL": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_TOTAL_VARIATION_DENSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_TOTAL_VARIATION_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RATE_GRADIENT_OVERALL": "degree_per_px2",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RATE_GRADIENT_DENSE": "degree_per_px2",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RATE_GRADIENT_SPARSE": "degree_per_px2",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RING_SUPPORT_OVERALL": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RING_SUPPORT_DENSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RING_SUPPORT_SPARSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RUN_SPAN_SUPPORT_OVERALL": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RUN_SPAN_SUPPORT_DENSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_RUN_SPAN_SUPPORT_SPARSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_MEDIAN_RESULTANT_OVERALL": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_MEDIAN_RESULTANT_DENSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_ROTATION_MEDIAN_RESULTANT_SPARSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.CONCENTRATION_RADIAL_OVERALL": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.CONCENTRATION_RADIAL_DENSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.CONCENTRATION_RADIAL_SPARSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.CONCENTRATION_MASK_OVERALL": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.CONCENTRATION_MASK_DENSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.CONCENTRATION_MASK_SPARSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.TURNING_RADIAL_OVERALL": "degree_per_px",
    "ORIENTATION_ZONE_DIAGNOSTIC.TURNING_RADIAL_DENSE": "degree_per_px",
    "ORIENTATION_ZONE_DIAGNOSTIC.TURNING_RADIAL_SPARSE": "degree_per_px",
    "ORIENTATION_ZONE_DIAGNOSTIC.TURNING_MASK_OVERALL": "degree_per_px",
    "ORIENTATION_ZONE_DIAGNOSTIC.TURNING_MASK_DENSE": "degree_per_px",
    "ORIENTATION_ZONE_DIAGNOSTIC.TURNING_MASK_SPARSE": "degree_per_px",
    "ORIENTATION_ZONE_DIAGNOSTIC.COHERENCE_RADIAL_OVERALL": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.COHERENCE_RADIAL_DENSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.COHERENCE_RADIAL_SPARSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.COHERENCE_MASK_OVERALL": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.COHERENCE_MASK_DENSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.COHERENCE_MASK_SPARSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.RADIAL_TILT_MASK_OVERALL": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.RADIAL_TILT_MASK_DENSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.RADIAL_TILT_MASK_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_TURNING_MASK_OVERALL": "degree_per_px",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_TURNING_MASK_DENSE": "degree_per_px",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTWARD_TURNING_MASK_SPARSE": "degree_per_px",
    "ORIENTATION_ZONE_DIAGNOSTIC.RADIAL_SECTOR_SUPPORT_MASK_OVERALL": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.RADIAL_SECTOR_SUPPORT_MASK_DENSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.RADIAL_SECTOR_SUPPORT_MASK_SPARSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.LONG_RANGE_ROTATION_MASK_OVERALL": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.LONG_RANGE_ROTATION_MASK_DENSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.LONG_RANGE_ROTATION_MASK_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.SIGNED_LONG_RANGE_ROTATION_MASK_OVERALL": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.SIGNED_LONG_RANGE_ROTATION_MASK_DENSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.SIGNED_LONG_RANGE_ROTATION_MASK_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.LONG_RANGE_ROTATION_SUPPORT_MASK_OVERALL": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.LONG_RANGE_ROTATION_SUPPORT_MASK_DENSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.LONG_RANGE_ROTATION_SUPPORT_MASK_SPARSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.LONG_RANGE_ROTATION_MASK_DENSE_TO_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.SIGNED_LONG_RANGE_ROTATION_MASK_DENSE_TO_SPARSE": "degree",
    "ORIENTATION_ZONE_DIAGNOSTIC.LONG_RANGE_ROTATION_SUPPORT_MASK_DENSE_TO_SPARSE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.ZONE_SEGMENTATION_METHOD_CODE": "not_a_quantity",
    "ORIENTATION_ZONE_DIAGNOSTIC.CORE_ZONE_END_RADIUS": "px",
    "ORIENTATION_ZONE_DIAGNOSTIC.DENSE_RADIUS": "px",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTER_RADIUS": "px",
    "ORIENTATION_ZONE_DIAGNOSTIC.FULL_EXTENT_RADIUS": "px",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTER_ZONE_PERCENTILE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.OUTER_ZONE_RETAINED_MASK_FRACTION": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.ZONE_SUPPORTED_RING_FRACTION": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.ZONE_CHANGE_POINT_OBJECTIVE": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.ZONE_CHANGE_POINT_RING_COUNT": "unitless",
    "ORIENTATION_ZONE_DIAGNOSTIC.ZONE_CHANGE_POINT_MINIMUM_SEGMENT": "unitless",
    # ORIENTATION_ZONE_PRIMARY
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_SUSTAINED_PEAK_OVERALL": "degree",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_SUSTAINED_PEAK_DENSE": "degree",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_SUSTAINED_PEAK_SPARSE": "degree",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_NET_OVERALL": "degree",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_NET_DENSE": "degree",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_NET_SPARSE": "degree",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_RATE_OVERALL": "degree_per_px",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_RATE_DENSE": "degree_per_px",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_RATE_SPARSE": "degree_per_px",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_CONSISTENCY_OVERALL": "unitless",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_CONSISTENCY_DENSE": "unitless",
    "ORIENTATION_ZONE_PRIMARY.OUTWARD_ROTATION_CONSISTENCY_SPARSE": "unitless",
    # SYMMETRIC_ZONES
    "SYMMETRIC_ZONES.CORE_RADIUS": "px",
    "SYMMETRIC_ZONES.SYMMETRIC_RADIUS": "px",
    "SYMMETRIC_ZONES.MEAN_EXPANSION": "px",
    "SYMMETRIC_ZONES.MAX_EXPANSION": "px",
    "SYMMETRIC_ZONES.CORE_END_RADIUS": "px",
    "SYMMETRIC_ZONES.DENSE_END_RADIUS": "px",
    "SYMMETRIC_ZONES.SPARSE_END_RADIUS": "px",
    "SYMMETRIC_ZONES.CORE_AREA": "px2",
    "SYMMETRIC_ZONES.DENSE_AREA": "px2",
    "SYMMETRIC_ZONES.SPARSE_AREA": "px2",
    # TEXTURE
    "TEXTURE.ANGULAR_SECOND_MOMENT": "unitless",
    "TEXTURE.CONTRAST": "unitless",
    "TEXTURE.CORRELATION": "unitless",
    "TEXTURE.VARIANCE": "unitless",
    "TEXTURE.INVERSE_DIFFERENCE_MOMENT": "unitless",
    "TEXTURE.SUM_AVERAGE": "unitless",
    "TEXTURE.SUM_VARIANCE": "unitless",
    "TEXTURE.SUM_ENTROPY": "unitless",
    "TEXTURE.ENTROPY": "unitless",
    "TEXTURE.DIFFERENCE_VARIANCE": "unitless",
    "TEXTURE.DIFFERENCE_ENTROPY": "unitless",
    "TEXTURE.IMC1": "unitless",
    "TEXTURE.IMC2": "unitless",
}
