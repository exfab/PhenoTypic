"""``_load_v2_grouped`` must accept the pre-rename ``enh_gray`` layer.

``valid_staged_hdf`` accepted ``enh_gray`` at ``schema_version >= 2``
(``_cli_staged_resume.py``), so the code believes schema-2 files carrying it
exist in the wild -- but the v2 loader did a bare ``layers["detect_mat"]``
with no fallback, while only the v1-flat loader had one. Recorded as
OPEN-QUESTIONS D8; the fallback lands here because ``--mode migrate`` reads
these files and Phase 6 keeps ``_load_v2_grouped`` as the migration reader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phenotypic import Image

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_hdf"


def test_a_v2_grouped_hdf_with_enh_gray_loads_a_populated_detect_mat() -> None:
    loaded = Image._load_hdf5_for_migration(FIXTURES / "v2_enh_gray" / "img.h5")
    assert loaded.detect_mat[:].any(), (
        "enh_gray was dropped: the detection matrix came back all zeros"
    )


def test_the_enh_gray_fallback_matches_the_renamed_layer() -> None:
    """The two fixtures differ only in the layer's NAME, so they must agree."""
    renamed = Image._load_hdf5_for_migration(FIXTURES / "v2_grouped" / "img.h5")
    legacy = Image._load_hdf5_for_migration(FIXTURES / "v2_enh_gray" / "img.h5")
    np.testing.assert_array_equal(legacy.detect_mat[:], renamed.detect_mat[:])


def test_the_fallback_assumes_gray_detect_mode() -> None:
    """``enh_gray`` datasets carried no ``detect_mode`` attr to read."""
    legacy = Image._load_hdf5_for_migration(FIXTURES / "v2_enh_gray" / "img.h5")
    assert legacy._data.detect_mode == "gray"
