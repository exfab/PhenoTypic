"""A click index resolves against master_df, and a stale one is refused.

The producer (``index_frame``) and the consumer (``resolve_click``) are
tested together on purpose. Gate 0 found the index being built against the
*filtered* frame and resolved against ``master_df``: every click opened a
real but wrong colony, and nothing raised. Testing either half alone
reproduces exactly the review that missed it.
"""

from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer._filter_state import (
    METHOD_IS_ANY_OF,
    FilterSpec,
)
from phenotypic.gui.results_viewer._scatter_tab._figure import CUSTOMDATA_COL
from phenotypic.gui.results_viewer._scatter_tab._inspector import (
    ColonyRef,
    index_frame,
    resolve_click,
)


def _master() -> pl.DataFrame:
    """Six colonies over three images, so a filter can drop rows.

    Deliberately not three rows: a filter that leaves one row makes the
    carried index equal the row position by coincidence, which is the
    coincidence B1 hid behind.
    """
    return pl.DataFrame(
        {
            "Metadata_Dataset": ["ds"] * 6,
            "Metadata_ImageName": ["a", "a", "b", "b", "c", "c"],
            "Object_Label": [1, 2, 1, 2, 1, 2],
        }
    )


# --------------------------------------------------------------------
# The consumer
# --------------------------------------------------------------------


def test_an_index_resolves_to_its_colony() -> None:
    assert resolve_click(_master(), 1, "fp", "fp") == ColonyRef("ds", "a", 2)


def test_a_stale_fingerprint_is_refused_not_resolved() -> None:
    """The race this prevents: the user changes a filter, clicks the
    still-rendered old figure, and the index resolves against a new frame.
    It would open a real colony -- the wrong one -- silently."""
    assert resolve_click(_master(), 1, "old", "new") is None


def test_a_phantom_row_is_refused_not_crashed_on() -> None:
    """master_df is the mirror, so it carries metadata-only phantoms.

    121 of the fixture's 844 rows and 117,415 of the full run's 231,229.
    ``Object_Label`` is ``Int64`` there (measured), so a phantom is a true
    null rather than a NaN -- ``int(None)`` raises ``TypeError`` and 500s
    the callback. The point is that a phantom has no colony to open, so
    the answer is None.
    """
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["ds"],
            "Metadata_ImageName": ["a"],
            "Object_Label": [None],
        }
    )
    assert resolve_click(master, 0, "fp", "fp") is None


def test_a_label_that_cannot_become_an_int_is_refused() -> None:
    """The phantom guard again, for the dtype it does not cover.

    ``Object_Label`` is ``Int64`` in the mirror, so a phantom is a true
    null. That is a property of this run, not of the schema: in a Float64
    column a phantom would be ``NaN``, which is not ``None`` and which
    ``int()`` rejects with ``ValueError`` -- the same 500 the phantom
    guard was written to stop, wearing a different exception.
    """
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["ds"],
            "Metadata_ImageName": ["a"],
            "Object_Label": [float("nan")],
        }
    )
    assert master.schema["Object_Label"].is_float()
    assert resolve_click(master, 0, "fp", "fp") is None


def test_a_null_dataset_or_image_is_refused_not_stringified() -> None:
    """``str(None)`` is ``"None"`` -- a plausible ref to a real-looking
    dataset that does not exist. Refusing is the same failure family as
    the phantom guard: never hand back a ColonyRef that cannot be opened.

    Both columns are non-null across the verification fixture, so this
    pins a property of the resolver rather than of that run.
    """
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["ds", None],
            "Metadata_ImageName": [None, "a"],
            "Object_Label": [1, 2],
        }
    )
    assert resolve_click(master, 0, "fp", "fp") is None
    assert resolve_click(master, 1, "fp", "fp") is None


def test_an_out_of_range_index_is_refused() -> None:
    """The negative case is load-bearing, not decorative.

    ``DataFrame.row(-1)`` indexes Python-style and returns the LAST row
    (measured), so without the guard a negative index resolves silently to
    a real colony from the wrong end of the frame.
    """
    assert resolve_click(_master(), 99, "fp", "fp") is None
    assert resolve_click(_master(), 6, "fp", "fp") is None
    assert resolve_click(_master(), -1, "fp", "fp") is None


def test_a_non_integer_index_is_refused_rather_than_raising() -> None:
    """``row(1.0)`` raises TypeError and ``row(True)`` resolves as row 1.

    Both measured. ``bool`` is an ``int`` subclass and implements
    ``__index__``, so a truthy flag arriving where an index is expected
    passes every arithmetic guard and opens colony 1.
    """
    master = _master()
    assert resolve_click(master, True, "fp", "fp") is None  # type: ignore[arg-type]
    assert resolve_click(master, 1.0, "fp", "fp") is None  # type: ignore[arg-type]
    assert resolve_click(master, "1", "fp", "fp") is None  # type: ignore[arg-type]
    assert resolve_click(master, None, "fp", "fp") is None  # type: ignore[arg-type]


def test_a_numpy_integer_still_resolves() -> None:
    """A stricter ``isinstance(index, int)`` guard would refuse this.

    ``numpy.int64`` is not a subclass of Python ``int``, but polars
    accepts it as a row index. Refusing it would turn a working click
    into a dead one, so the guard is ``__index__``-based rather than a
    type whitelist -- and this test is what says so.
    """
    import numpy as np

    assert resolve_click(_master(), np.int64(1), "fp", "fp") == ColonyRef(
        "ds", "a", 2
    )


# --------------------------------------------------------------------
# The producer, and the round trip between the two
# --------------------------------------------------------------------


def test_index_frame_stamps_the_position_in_the_frame_it_is_given() -> None:
    indexed = index_frame(_master())
    assert indexed[CUSTOMDATA_COL].to_list() == [0, 1, 2, 3, 4, 5]
    # The stamp must not disturb the frame it indexes.
    assert indexed.drop(CUSTOMDATA_COL).equals(_master())


def test_index_frame_is_idempotent_rather_than_re_stamping() -> None:
    """A second call keeps the first index instead of renumbering.

    ``with_row_index`` raises ``DuplicateError`` on a re-call (measured),
    and re-stamping would be worse still: on an already-filtered frame it
    is precisely the B1 defect. Cluster G calls this from a callback,
    where a defensive re-call is likelier than a bug, so the safe answer
    is to hand back the master-anchored index already present.
    """
    once = index_frame(_master())
    scrambled = once.filter(pl.col("Metadata_ImageName") != "a")
    assert index_frame(scrambled).equals(scrambled)
    assert index_frame(once).equals(once)


def test_an_index_survives_the_real_filter_path_and_a_re_sort() -> None:
    """The round trip Gate 0 found missing, and the bug it would catch.

    The index must be stamped on ``master_df`` BEFORE any filtering, so a
    point drawn from a filtered, re-sorted frame still addresses the right
    row of the unfiltered one. Indexing the filtered frame instead passes
    every other test in this module -- none of them exercise the producer
    -- and opens the wrong colony in production.

    The filter runs through ``FilterSpec.apply_to``, which is what the
    viewer actually calls and is NOT a bare ``.filter()``: it runs
    ``normalize_viewer_frame`` first, which renames every non-metadata
    column to a shield name, normalizes, and renames back. ``CUSTOMDATA_COL``
    makes that round trip on every filter application, so a bare
    ``.filter()`` here would test a path production never takes.
    """
    master = _master()
    indexed = index_frame(master)

    spec = FilterSpec.from_store(
        [
            {
                "column": "Metadata_ImageName",
                "method": METHOD_IS_ANY_OF,
                "values": ["b", "c"],
            }
        ]
    )
    scrambled = spec.apply_to(indexed).sort("Object_Label", descending=True)

    # The filter must actually drop rows and the sort must actually
    # reorder, or the carried index equals the row position and the test
    # passes against the defect it exists to catch.
    assert scrambled.height == 4
    assert scrambled[CUSTOMDATA_COL].to_list() != list(range(scrambled.height))

    for row in scrambled.iter_rows(named=True):
        idx = int(row[CUSTOMDATA_COL])
        assert resolve_click(master, idx, "fp", "fp") == ColonyRef(
            row["Metadata_Dataset"],
            row["Metadata_ImageName"],
            int(row["Object_Label"]),
        )
