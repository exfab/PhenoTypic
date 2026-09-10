"""The one home of the v1/v2 master discrimination.

Spec §7.3 moved the metadata join out of the per-image embedded tables and
into finalization, which changes what ``deliverables/master_measurements.parquet``
contains:

* **v1** (pre-inversion) -- every measured row already carried its
  publication-time user metadata, because the join happened per image.
* **v2** (post-inversion) -- the master is the un-joined archival set:
  intrinsic identity plus measurements. The join lives in the
  ``deliverables/measurements.*`` mirror.

**Nothing stamps the file** (user ruling, 2026-09-06): the master already
self-describes, and a stamp would be a second on-disk home for a fact the
columns already state. The discrimination therefore lives here, in one
module, so its *retirement condition* has one home too -- spreading a
two-line column check across seven reader modules would give the condition
seven homes and, in practice, none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl

__all__ = ["master_carries_user_metadata", "user_metadata_headers"]


def user_metadata_headers(columns: Iterable[str]) -> tuple[str, ...]:
    """Return the headers that could only have come from ``--metadata``.

    A master's *intrinsic* metadata is the identity the image carries about
    itself: the ``IMAGE``-owned per-image provenance block, plus
    ``EXPERIMENT.DATASET``, which the CLI inserts from the dataset directory
    name rather than from any CSV. Everything else in the metadata namespace
    was joined in from the run's ``metadata.csv``.

    **Ownership, not the prefix.** ``Metadata_Strain`` is a real schema
    member (``GENETIC.STRAIN``), so "carries a ``Metadata_*`` column" does
    *not* separate the two shapes -- a v2 master carries
    ``Metadata_Dataset`` and ``Metadata_ImageName`` and would be misread as
    v1 by that test. Namespace detection goes through
    :func:`is_metadata_header` and routing through
    :func:`metadata_owner_for_header`, never through prefix parsing.

    Args:
        columns: Column names of a master frame.

    Returns:
        The user-metadata headers present, in the order given.
    """
    from phenotypic.schema import EXPERIMENT, IMAGE

    from ._metadata_helpers import is_metadata_header, metadata_owner_for_header

    intrinsic = {str(EXPERIMENT.DATASET)}
    return tuple(
        column
        for column in columns
        if is_metadata_header(column)
        and metadata_owner_for_header(column) is not IMAGE
        and column not in intrinsic
    )


# V1/V2 MASTER DISCRIMINATION -- DELETE WHEN: no run predating the §7.3
# inversion is still readable, i.e. every master in the wild was written by
# finalize_run's post-inversion path. A v1 master carries user-metadata
# columns because the join happened per-image; a v2 master does not, because
# the join moved to finalization. Nothing else distinguishes them, and
# nothing stamps them. When that condition holds, this function and every
# branch on it are dead code and should go together.
def master_carries_user_metadata(frame: "pl.DataFrame") -> bool:
    """Return whether this master predates the §7.3 inversion.

    The one genuinely dangerous failure mode in §7 is a reader that filters
    or groups a master on a user-metadata column: against a v2 master that
    returns **empty** rather than raising. This predicate is what such a
    reader branches on.

    A v1 run that was given no ``--metadata`` is indistinguishable from a v2
    run by this test, and that is expected to be harmless -- neither has
    anything to join. ``test_a_v1_metadata_free_master_is_indistinguishable_from_v2``
    is the designated falsifier for that expectation.

    Args:
        frame: A master measurements frame.

    Returns:
        ``True`` when the frame carries at least one user-metadata column.
    """
    return bool(user_metadata_headers(frame.columns))
