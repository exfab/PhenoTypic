"""Rectangle regions of interest and tile lattices for in-frame colour checkers.

A :class:`CheckerRoi` says **only where to look** for a colour-checker card in a
plate image: a rectangle, and nothing about what is inside it.  How many tiles
the region holds and which chart patch each one is are recovered from the
pixels by the detection and identity stages, never declared by the caller.  A
declared layout is a claim about the image that the image can contradict, and
the failure is silent -- a card mounted upside down after a rig service still
produces twelve plausible tiles, each then fitted against the wrong reference
colour.

A :class:`CheckerLattice` is the geometric result of detection: one
:class:`ColumnLattice` per column of patches, plus the rigid displacement that
placed it on this frame.  It converts to per-tile boxes for measurement.

See :doc:`the design spec
</superpowers/specs/2026-09-21-in-frame-checker-color-correction/README>` for
the surrounding algorithm.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Iterable, Sequence

import numpy as np

from pydantic import BaseModel, ConfigDict, model_validator

from ...sdk_.typing_ import TuneSpec

#: Coordinate order of the bounding-box shorthand, quoted in every error
#: raised by :meth:`CheckerRoi.from_bbox`.  It matches
#: ``skimage.measure.regionprops``' ``bbox``, so a region found by an upstream
#: segmentation drops straight in; it is *not* ``(x, y, width, height)`` and
#: not the ``(row_slice, col_slice)`` pair :class:`ColorCheckerProfile` takes.
BBOX_ORDER = "row_min, col_min, row_max, col_max"


class CheckerRoi(BaseModel):
    """A rectangle in which to look for one colour-checker card.

    Args:
        row: ``(row_min, row_max)`` bounds, half-open, in pixels.
        col: ``(col_min, col_max)`` bounds, half-open, in pixels.
        label: Optional name carried into diagnostics (e.g. ``"left band"``).
        expect_tiles: Optional assertion.  When set and detection finds a
            different number of tiles, that is a QC failure rather than a
            silent adjustment.  It never steers detection.
        anchor_col: Optional index of the column to measure displacement from
            during rigid refinement.  Use the column that is *not* clipped by
            the frame border: a clipped column's visible extent changes with
            the shift, so it tracks displacement at roughly half rate.

    Raises:
        ValueError: If the rectangle is empty, inverted, or negative.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Numeric fields here describe the rig, not a knob: never tuned.
    row: tuple[int, int]
    col: tuple[int, int]
    label: str | None = None
    expect_tiles: Annotated[int | None, TuneSpec(tunable=False)] = None
    anchor_col: Annotated[int | None, TuneSpec(tunable=False)] = None

    @model_validator(mode="after")
    def _validate_rectangle(self) -> CheckerRoi:
        """Reject an empty, inverted, or negative rectangle."""
        for axis, (lo, hi) in (("row", self.row), ("col", self.col)):
            if lo < 0 or hi < 0:
                raise ValueError(
                        f"CheckerRoi {axis} bounds must be non-negative; got ({lo}, {hi})."
                )
            if lo >= hi:
                raise ValueError(
                        f"CheckerRoi {axis} bounds are empty or inverted: ({lo}, {hi}). "
                        f"Expected ({axis}_min, {axis}_max)."
                )
        if self.expect_tiles is not None and self.expect_tiles < 1:
            raise ValueError(
                    f"expect_tiles must be a positive count; got {self.expect_tiles}."
            )
        return self

    @classmethod
    def from_bbox(cls, bbox: Sequence[Any], **kwargs: Any) -> CheckerRoi:
        """Build an ROI from a ``[row_min, col_min, row_max, col_max]`` box.

        This is ``skimage.measure.regionprops``' ``bbox`` order, so
        ``[CheckerRoi.from_bbox(r.bbox) for r in regionprops(labels)]`` works
        without re-ordering.

        Args:
            bbox: Four integer pixel coordinates in :data:`BBOX_ORDER`.
            **kwargs: Passed through to the constructor (``label``,
                ``expect_tiles``, ``anchor_col``).

        Returns:
            The equivalent :class:`CheckerRoi`.

        Raises:
            ValueError: If *bbox* is not four integers, or describes an empty
                or inverted rectangle -- which is what a
                ``(row, col, height, width)`` or ``(x, y, width, height)`` box
                usually looks like once read in this order.
        """
        values = list(bbox)
        if len(values) != 4:
            raise ValueError(
                    f"A bounding box needs four coordinates ({BBOX_ORDER}); "
                    f"got {len(values)}."
            )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                        f"Bounding-box coordinates must be integer pixels "
                        f"({BBOX_ORDER}); got {value!r}."
                )
        row_min, col_min, row_max, col_max = (int(v) for v in values)
        if row_min >= row_max or col_min >= col_max:
            raise ValueError(
                    f"Bounding box {values} is empty or inverted. Expected "
                    f"{BBOX_ORDER} (scikit-image regionprops order) -- a "
                    f"(row, col, height, width) or (x, y, width, height) box "
                    f"will look like this."
            )
        return cls(row=(row_min, row_max), col=(col_min, col_max), **kwargs)

    @classmethod
    def coerce(cls, value: Any) -> CheckerRoi:
        """Accept a :class:`CheckerRoi`, a mapping, or a bounding-box sequence.

        Used by the ``rois`` field validator on the operation so a caller may
        write ``rois=[[1170, 0, 2840, 340], ...]``.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        if isinstance(value, (list, tuple)):
            return cls.from_bbox(value)
        raise ValueError(
                f"Cannot read {value!r} as a checker ROI. Pass a CheckerRoi or a "
                f"[{BBOX_ORDER}] sequence."
        )

    @property
    def row_slice(self) -> slice:
        """Row slice into an image array."""
        return slice(self.row[0], self.row[1])

    @property
    def col_slice(self) -> slice:
        """Column slice into an image array."""
        return slice(self.col[0], self.col[1])

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the rectangle, in pixels."""
        return self.row[1] - self.row[0], self.col[1] - self.col[0]

    def __repr__(self) -> str:
        name = f" {self.label!r}" if self.label else ""
        return (
            f"CheckerRoi{name} rows {self.row[0]}-{self.row[1]} "
            f"cols {self.col[0]}-{self.col[1]}"
        )


class ColumnLattice(BaseModel):
    """One column of patches within a detected checker lattice.

    Args:
        x0: Left edge of the column, in ROI-local pixels.
        x1: Right edge of the column, in ROI-local pixels.
        start: ROI-local row of the first tile's top edge.
        pitch: Tile-to-tile spacing down the column, in pixels.
        duty: Tile height as a fraction of *pitch* -- the rest is gutter.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Detected geometry, written by fit_lattice/refine: data, never tuned.
    x0: Annotated[int, TuneSpec(tunable=False)]
    x1: Annotated[int, TuneSpec(tunable=False)]
    start: Annotated[float, TuneSpec(tunable=False)]
    pitch: Annotated[float, TuneSpec(tunable=False)]
    duty: Annotated[float, TuneSpec(tunable=False)]


class CheckerLattice(BaseModel):
    """A detected tile lattice for one ROI, with its placement on this frame.

    Args:
        columns: One :class:`ColumnLattice` per column of patches.
        nrows: Number of tiles down each column.
        dy: Vertical displacement applied by refinement, in pixels.
        dx: Horizontal displacement applied by refinement, in pixels.
        rot: Rotation in radians about the lattice centroid; positive turns
            +x toward +y, as OpenCV's warps do.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Detected geometry and the refinement applied to it: never tuned.
    columns: list[ColumnLattice]
    nrows: Annotated[int, TuneSpec(tunable=False)]
    dy: Annotated[float, TuneSpec(tunable=False)] = 0.0
    dx: Annotated[float, TuneSpec(tunable=False)] = 0.0
    rot: Annotated[float, TuneSpec(tunable=False)] = 0.0

    @property
    def n_tiles(self) -> int:
        """Total number of tiles the lattice describes."""
        return len(self.columns) * self.nrows

    def boxes(
            self,
            dy: float = 0.0,
            dx: float = 0.0,
            core: float = 0.0,
            rot: float = 0.0,
    ) -> list[tuple[int, int, float, float, float, float]]:
        """Tile boxes as ``(row, col, y0, y1, x0, x1)`` in ROI-local pixels.

        Args:
            dy: Extra vertical translation applied to the whole lattice.
            dx: Extra horizontal translation applied to the whole lattice.
            core: Fraction of each box trimmed away, split evenly between the
                two edges of each axis.  ``0.4`` keeps the central 60 % of
                each axis, i.e. 36 % of the area, which is what keeps the
                measurement off the gutters when the lattice is a few pixels
                out.
            rot: Rotation in radians about the lattice centroid; positive
                turns +x toward +y, as OpenCV's warps do.

        Returns:
            One tuple per tile, column-major: all rows of column 0, then all
            rows of column 1, and so on.
        """
        if not 0.0 <= core < 1.0:
            raise ValueError(f"core must be in [0, 1); got {core}.")

        placed: list[list[float]] = []
        index: list[tuple[int, int]] = []
        for col_idx, column in enumerate(self.columns):
            height = column.duty * column.pitch
            for row_idx in range(self.nrows):
                y0 = column.start + row_idx * column.pitch
                placed.append([y0, y0 + height, float(column.x0), float(column.x1)])
                index.append((row_idx, col_idx))

        if rot:
            cy = sum((b[0] + b[1]) / 2 for b in placed) / len(placed)
            cx = sum((b[2] + b[3]) / 2 for b in placed) / len(placed)
            cos_r, sin_r = math.cos(rot), math.sin(rot)
            for box in placed:
                my, mx = (box[0] + box[1]) / 2 - cy, (box[2] + box[3]) / 2 - cx
                half_h, half_w = (box[1] - box[0]) / 2, (box[3] - box[2]) / 2
                # OpenCV's sense (image y points down): positive rot turns
                # +x toward +y, matching the angle refine_ecc recovers.
                ny = mx * sin_r + my * cos_r + cy
                nx = mx * cos_r - my * sin_r + cx
                box[0], box[1] = ny - half_h, ny + half_h
                box[2], box[3] = nx - half_w, nx + half_w

        out: list[tuple[int, int, float, float, float, float]] = []
        for (row_idx, col_idx), box in zip(index, placed):
            y0, y1, x0, x1 = box
            y0, y1 = y0 + dy, y1 + dy
            x0, x1 = x0 + dx, x1 + dx
            if core:
                margin_y = core * (y1 - y0) / 2
                margin_x = core * (x1 - x0) / 2
                y0, y1 = y0 + margin_y, y1 - margin_y
                x0, x1 = x0 + margin_x, x1 - margin_x
            out.append((row_idx, col_idx, y0, y1, x0, x1))
        return out

    def centers(self, **kwargs: Any) -> list[tuple[float, float]]:
        """Tile centres as ``(y, x)``, taking the same arguments as :meth:`boxes`."""
        return [
            ((y0 + y1) / 2, (x0 + x1) / 2)
            for _, _, y0, y1, x0, x1 in self.boxes(**kwargs)
        ]

    def translated(self, dy: float, dx: float) -> CheckerLattice:
        """A copy of this lattice moved by ``(dy, dx)``, recording the shift."""
        return CheckerLattice(
                columns=[
                    ColumnLattice(
                            x0=int(round(column.x0 + dx)),
                            x1=int(round(column.x1 + dx)),
                            start=column.start + dy,
                            pitch=column.pitch,
                            duty=column.duty,
                    )
                    for column in self.columns
                ],
                nrows=self.nrows,
                dy=self.dy + dy,
                dx=self.dx + dx,
                rot=self.rot,
        )


def coerce_rois(values: Iterable[Any]) -> list[CheckerRoi]:
    """Coerce every entry of a ``rois`` argument to a :class:`CheckerRoi`."""
    if isinstance(values, (CheckerRoi, dict)) or (
            isinstance(values, (list, tuple))
            and values
            and all(isinstance(v, (int, float)) for v in values)
    ):
        raise ValueError(
                "rois must be a list of ROIs, not a single ROI. Wrap it: "
                "rois=[...]."
        )
    return [CheckerRoi.coerce(value) for value in values]
