"""The ``metadata_only_mask`` fallback contract.

``QC_MetadataOnly`` is emitted only by the CLI's ``--metadata`` left join, but
every consumer of the mask is public API a user may call on a notebook frame
that has no such column. The mask must degrade to exactly the pre-left-join
behavior (all-``False``) in every case where the flag is not a real boolean
column — never guess, never coerce.
"""

from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.schema import METADATA_MATCH
from phenotypic.sdk_ import metadata_only_mask

FLAG = str(METADATA_MATCH.METADATA_ONLY)


def test_flag_column_name_is_qc_metadata_only():
    """The column string is a cross-module contract — pin it."""
    assert FLAG == "QC_MetadataOnly"


def test_real_bool_column_is_honored():
    df = pd.DataFrame({FLAG: [False, True, False], "Shape_Area": [1.0, None, 3.0]})

    assert metadata_only_mask(df).tolist() == [False, True, False]


def test_absent_column_degrades_to_all_false():
    """A notebook frame from ``image.measure()`` carries no flag."""
    df = pd.DataFrame({"Shape_Area": [1.0, 2.0]})

    mask = metadata_only_mask(df)

    assert mask.tolist() == [False, False]
    assert mask.dtype == bool


def test_string_column_is_rejected_not_coerced():
    """``pd.Series(["False", "True"]).astype(bool)`` is ``[True, True]``.

    The string ``"False"`` is truthy, so a lenient coercion would mark EVERY row
    a phantom — reporting zero detections everywhere. A non-bool column must
    fall back to today's behavior instead.
    """
    df = pd.DataFrame({FLAG: ["False", "True"]})
    assert df[FLAG].astype(bool).tolist() == [True, True]  # the trap, demonstrated

    assert metadata_only_mask(df).tolist() == [False, False]


def test_numeric_column_is_rejected():
    df = pd.DataFrame({FLAG: [0, 1]})

    assert metadata_only_mask(df).tolist() == [False, False]


def test_all_null_column_degrades_to_all_false():
    """An all-null column is object dtype, not bool — rejected."""
    df = pd.DataFrame({FLAG: [None, None]})

    assert metadata_only_mask(df).tolist() == [False, False]


def test_nullable_boolean_nulls_are_false():
    """A real (nullable) boolean column with a null: null is not a phantom."""
    df = pd.DataFrame({FLAG: pd.array([True, None, False], dtype="boolean")})

    mask = metadata_only_mask(df)

    assert mask.tolist() == [True, False, False]
    assert mask.dtype == bool


@pytest.mark.parametrize("index", [[0, 1, 2], [7, 8, 9], ["a", "b", "c"]])
def test_mask_is_index_aligned(index):
    """Consumers use the mask to slice the frame; it must share its index."""
    df = pd.DataFrame({"Shape_Area": [1.0, 2.0, 3.0]}, index=index)

    assert metadata_only_mask(df).index.tolist() == index
    assert df[~metadata_only_mask(df)].equals(df)
