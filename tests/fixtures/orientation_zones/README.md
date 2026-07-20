# Orientation-zone real-image fixture

`r3c4_twok_literal_crossing.npz` is a 512 x 512 crop centered around the R3C4
colony used to develop the literal skeleton-ring crossing calculation.

The fixture contains:

- `detect_mat`: the float32 notebook composite actually used to calculate the
  orientation field;
- `objmap`: a uint16 isolated-object map with the source TwoK label 24 remapped
  to fixture label 1;
- `crop_origin_rc`: the crop origin in the cached 2812 x 4716 plate array;
- `source_label`: the original TwoK label, 24;
- `colony`: the plate position, `R3C4`; and
- SHA-256 hashes of the full cached composite and TwoK object map.

The crop is rows 992:1504 and columns 1491:2003. The detected-object bounding
box is rows 1051:1463 and columns 1541:1954 in the full cached plate, leaving
approximately 50 pixels of real background around its furthest branches. Raw
RGB is intentionally omitted because `MeasureOrientationZones` used the cached
notebook `detect_mat` as its configured orientation-field source.
