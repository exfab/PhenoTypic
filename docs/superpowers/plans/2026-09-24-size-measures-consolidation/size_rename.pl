#!/usr/bin/perl -pi
# Rename retired Shape_* columns and SHAPE members to their 0.20.0 successors (plan Task 7).
# Word-bounded. NOT a prefix swap: Shape_MaxRadius -> Size_InscribedRadius and the
# Mean/MedianRadius -> BoundaryDist rows are the same-name trap (spec §6).
# Never run on: schema/_change_notes.py, _size.py, _shape.py, _measurement_info.py (toy doctest),
# the equivalence/shape/change-note tests, test_measurement_join_migration_run.py (reads a real
# run's old stores), any _golden* fixture, or docs/superpowers/** (plan amendment A7).
# Blind spot: `_` is a word character, so a retired name followed by `_suffix` (Shape_Area_stderr)
# is NOT rewritten. After a run, grep for `Shape_(Area|Perimeter|...|MedianRadius)_[A-Za-z0-9]`
# and fix those by hand (phase-2 review HIGH-1).
s/\bShape_Area\b/Size_Area/g;
s/\bShape_Perimeter\b/Size_Perimeter/g;
s/\bShape_ConvexArea\b/Size_ConvexArea/g;
s/\bShape_BboxArea\b/Size_BboxArea/g;
s/\bShape_MajorAxisLength\b/Size_MajorAxisLength/g;
s/\bShape_MinorAxisLength\b/Size_MinorAxisLength/g;
s/\bShape_MaxRadius\b/Size_InscribedRadius/g;
s/\bShape_MeanRadius\b/Shape_MeanBoundaryDist/g;
s/\bShape_MedianRadius\b/Shape_MedianBoundaryDist/g;
s/\bSHAPE\.AREA\b/SIZE.AREA/g;
s/\bSHAPE\.PERIMETER\b/SIZE.PERIMETER/g;
s/\bSHAPE\.CONVEX_AREA\b/SIZE.CONVEX_AREA/g;
s/\bSHAPE\.BBOX_AREA\b/SIZE.BBOX_AREA/g;
s/\bSHAPE\.MAJOR_AXIS_LENGTH\b/SIZE.MAJOR_AXIS_LENGTH/g;
s/\bSHAPE\.MINOR_AXIS_LENGTH\b/SIZE.MINOR_AXIS_LENGTH/g;
s/\bSHAPE\.MAX_RADIUS\b/SIZE.INSCRIBED_RADIUS/g;
s/\bSHAPE\.MEAN_RADIUS\b/SHAPE.MEAN_BOUNDARY_DIST/g;
s/\bSHAPE\.MEDIAN_RADIUS\b/SHAPE.MEDIAN_BOUNDARY_DIST/g;
