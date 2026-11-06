"""Post-measurement op that numbers each plate's images in capture order."""

from __future__ import annotations

from typing import List, cast

import pandas as pd
from pydantic import field_validator

from phenotypic.abc_._post_measurement import PostMeasurement
from phenotypic.schema import CULTURE, IMAGE
from ._utils import (
    coalesce_metadata_aliases,
    ensure_metadata_prefix,
    resolve_metadata_column,
)


class AddFrameIndex(PostMeasurement):
    """Add a per-plate frame index that lines a plate's images up in capture order.

    The CLI's measurement table carries no time column at all: no capture
    timestamp, no elapsed time, no frame index. ``AddFrameIndex`` fills that
    gap by ranking each plate's *distinct* ``order_by`` values and stamping
    every row of a given image with that rank, **1-based** (the first image
    of a plate is frame 1), matching the schema's own documented
    :attr:`~phenotypic.schema.CULTURE.FRAME_INDEX` semantics. It is meant to
    run once, on the whole aggregated measurement table.

    Args:
        plate_by: Column(s) identifying one plate — every row sharing the
            same ``plate_by`` values is one capture series to rank within.
            The schema category prefix is added automatically to each column
            (e.g. ``PlateID`` -> ``Metadata_PlateID``; unknown labels get a
            generic ``Metadata_`` prefix).
        order_by: Column whose distinct values, sorted ascending, give a
            plate's capture order. Defaults to the image name column
            (:attr:`phenotypic.schema.IMAGE.IMAGE_NAME`), following the same
            prefixing rule as ``plate_by``.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with
        ``Metadata_FrameIndex`` (:attr:`phenotypic.schema.CULTURE.FRAME_INDEX`)
        added (nullable integer dtype) — or, when that column already carries
        a user-supplied index, an unmodified copy (see Notes).

    Notes:
        **A pre-existing, partially-or-fully populated `Metadata_FrameIndex`
        is never touched.** ``Metadata_FrameIndex`` can arrive already
        populated from a ``--metadata`` CSV. If the column is present and has
        *any* non-null value, this op leaves it completely alone (logged at
        INFO) rather than deriving anything — including rows this op could
        have filled. Only an **absent** column, or one that is present but
        **entirely null**, gets derived. This op never partially fills a
        user's column.

        **The default `order_by` only works when image names sort lexically
        into capture order.** A name that ends in an ISO capture timestamp
        behind a per-plate constant prefix (``plate07_2026-01-05_08-00-00``)
        sorts that way; so does a zero-padded counter (``plate07_006h``). Pass
        a different ``order_by`` (an actual timestamp column, say) when image
        names don't have that shape.

        **This is a cross-image, whole-table operation** — unlike every other
        post op, which is row-local. It is only correct applied once, to the
        complete aggregated frame. On the CLI, per-image paths call
        ``measure(..., apply_post=False)`` and post ops run exactly once, over
        the full aggregate, in ``_apply_post_to_master``
        (``_cli_output_manager.py``). A notebook call
        ``pipeline.measure(image)`` applies it to a single image, where every
        row gets frame 1 — correct for that one image, but not what a
        multi-image run wants; run it over the assembled table instead.

        **A skipped image shifts every later frame.** If a plate is missing
        one capture in the middle of a run, every subsequent image on that
        plate ranks one lower than its sister plates' equivalent frame. This
        mirrors the reference this op was derived from and is a known
        limitation, not a bug: whether elapsed time should replace rank is an
        open question, not something this op resolves. (The reference's own
        frame index is 0-based; this op's ``Metadata_FrameIndex`` is exactly
        one more than it, per plate-image.)

        Every row of one image shares its image's index — this includes
        per-position phantom rows from a ``--metadata`` left join (a phantom
        still carries its image's name). A row with a missing ``plate_by`` or
        ``order_by`` value gets a missing (``pd.NA``) index instead of being
        dropped or grouped under a false key. Applying the op a second time is
        a no-op: the second call sees its own output already populated and
        leaves it unchanged (see the first Note above).

    Raises:
        KeyError: If ``plate_by`` or ``order_by`` names a column absent from
            the DataFrame (only when a frame index must actually be derived).
        ValueError: If ``plate_by`` is empty when a frame index must be
            derived.

    Examples:
        Two plates, four capture times each, ranked independently, 1-based:

        >>> import pandas as pd
        >>> from phenotypic.post import AddFrameIndex
        >>> from phenotypic.schema import CULTURE, IMAGE, PLATE
        >>> plate = str(PLATE.PLATE_ID)
        >>> image_name = str(IMAGE.IMAGE_NAME)
        >>> frame_index = str(CULTURE.FRAME_INDEX)
        >>> df = pd.DataFrame({
        ...     plate: ["P1", "P1", "P2", "P2"],
        ...     image_name: ["p1_t1", "p1_t0", "p2_t0", "p2_t1"],
        ...     "Object_Label": [1, 1, 1, 1],
        ... })
        >>> result = AddFrameIndex(plate_by=["PlateID"]).apply(df)
        >>> result[frame_index].tolist()
        [2, 1, 1, 2]

        A blank plate value gets a blank index rather than a false rank:

        >>> df.loc[0, plate] = None
        >>> result = AddFrameIndex(plate_by=["PlateID"]).apply(df)
        >>> result[frame_index].isna().tolist()
        [True, False, False, False]
    """

    plate_by: List[str]
    order_by: str = str(IMAGE.IMAGE_NAME)

    @field_validator("plate_by", mode="before")
    @classmethod
    def _prefix_plate_by(cls, plate_by: List[str]) -> List[str]:
        """Apply the schema category prefix to each plate-identifying column."""
        return [ensure_metadata_prefix(col) for col in plate_by] if plate_by else []

    @field_validator("order_by")
    @classmethod
    def _prefix_order_by(cls, order_by: str) -> str:
        """Apply the schema category prefix (generic ``Metadata_`` fallback) to `order_by`."""
        return ensure_metadata_prefix(order_by) if order_by else ""

    def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank each plate's distinct ``order_by`` values and stamp every row.

        A pre-existing, non-empty ``Metadata_FrameIndex`` short-circuits
        derivation entirely (see the class docstring's Notes).

        Args:
            df: Measurement DataFrame, optionally already carrying
                ``plate_by`` and ``order_by``.

        Returns:
            DataFrame with ``Metadata_FrameIndex`` derived, or an unmodified
            copy when a user-supplied index is already present.
        """
        frame_index_column = str(CULTURE.FRAME_INDEX)

        try:
            existing_column = resolve_metadata_column(df.columns, frame_index_column)
        except KeyError:
            existing_column = None

        if existing_column is not None and df[existing_column].notna().any():
            self._logger.info(
                "%s: '%s' already has a user-supplied frame index; leaving it "
                "unchanged.",
                type(self).__name__,
                existing_column,
            )
            return df.copy()

        if not self.plate_by:
            raise ValueError(
                "AddFrameIndex requires at least one plate_by column identifying "
                "a plate; got an empty list."
            )

        try:
            result = coalesce_metadata_aliases(df, [*self.plate_by, self.order_by])
            plate_columns = [
                resolve_metadata_column(result.columns, col) for col in self.plate_by
            ]
            order_column = resolve_metadata_column(result.columns, self.order_by)
        except KeyError as exc:
            raise KeyError(
                f"Column {exc.args[0]!r} not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            ) from None

        valid = result[order_column].notna()
        for plate_column in plate_columns:
            valid &= result[plate_column].notna()

        frame_index = pd.Series(pd.NA, index=result.index, dtype="Int64")
        if valid.any():
            # Rows excluded here never enter a group, so they cannot be
            # silently pooled under a false (NA) plate key. pandas' dense
            # rank is already 1-based (the smallest value ranks 1).
            dense_rank = (
                result.loc[valid]
                .groupby(plate_columns, sort=False)[order_column]
                .rank(method="dense")
                .astype("Int64")
            )
            frame_index.loc[valid] = dense_rank

        if existing_column is not None:
            result[existing_column] = frame_index
        else:
            insert_pos = cast(int, result.columns.get_loc(order_column)) + 1
            result.insert(insert_pos, frame_index_column, frame_index)

        return result
