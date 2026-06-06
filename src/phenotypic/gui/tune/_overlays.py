"""Pure overlay backend for the ``/tune/`` Curate view (B-i).

The Curate Dash surface (built in B-ii) lets a user audit a tuning trial's
segmentation on a chosen plate: it renders the candidate's objmap over the
detect_mat, diffs two candidates' objects, and caches the rendered arrays on a
background thread. This module is the **pure**, Dash-free backend those
callbacks call:

* :func:`render_candidate_overlay` — ``build_pipeline(base, params)`` →
  ``pipeline.apply(plate)`` → an RGB ``label2rgb`` overlay **array** for a
  Plotly ``go.Image`` (NOT base64, NOT PNG bytes).

**Lazy-optuna lock.** Importing this module — like
:mod:`phenotypic.gui.tune` itself — must never drag ``optuna`` into
``sys.modules``. ``build_pipeline`` lives in the optuna-free
:mod:`phenotypic.tune._evaluation._builder`; the overlay core is the builder's
:func:`~phenotypic.gui.builder._image_renderer.to_overlay_rgb_array`. Neither
imports optuna, so the lock holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from phenotypic.gui._design import OI_GREY, OI_ORANGE, OI_SKY
from phenotypic.gui.builder._image_renderer import to_overlay_rgb_array
from phenotypic.tune._evaluation._builder import build_pipeline
from phenotypic.tune._scoring._matching import match_iou_greedy

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import numpy.typing as npt

    from phenotypic import Image, ImagePipeline


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a ``#RRGGBB`` design token to an ``(R, G, B)`` uint8 triple.

    Args:
        hex_color: A 6-digit hex color string (with or without a leading ``#``),
            e.g. an Okabe-Ito ``OI_*`` token from
            :mod:`phenotypic.gui._design`.

    Returns:
        The ``(R, G, B)`` channel values as plain ints in ``[0, 255]``.
    """
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def render_candidate_overlay(
    base_pipeline: "ImagePipeline",
    params: dict[str, Any],
    plate_image: "Image",
    *,
    max_dim: int = 640,
) -> np.ndarray:
    """Render a tuning candidate's segmentation as an RGB overlay array.

    Overlays ``params`` onto ``base_pipeline`` via
    :func:`~phenotypic.tune._evaluation._builder.build_pipeline` (the same
    flat ``{"<pos>.<field>": value}`` combo grammar the strategies use),
    applies the resulting pipeline to a fresh copy of ``plate_image``, and
    composites the detected objmap over the post-op detect_mat with
    ``skimage.color.label2rgb`` — the exact same core the builder's preview
    uses, so a tuned candidate and a hand-built pipeline look identical.

    The returned array is RGB and display-ready for a Plotly ``go.Image``
    trace; it is **not** PNG-encoded or base64-wrapped.

    Args:
        base_pipeline: The base :class:`~phenotypic.ImagePipeline` embedded in
            the tuning spec. Not mutated — ``build_pipeline`` deep-copies it.
        params: A flat combo (``{"<pos>.<field>": value}``, e.g.
            ``{"0.sigma": 2.0}``) addressing ops by position index, exactly as
            ``build_pipeline`` expects.
        plate_image: The plate :class:`~phenotypic.Image` (or
            :class:`~phenotypic.GridImage`) to segment. Copied before
            ``apply`` so the caller's image is untouched.
        max_dim: Maximum length of the longer spatial side of the overlay, in
            pixels. Defaults to ``640``.

    Returns:
        An ``(H, W, 3)`` uint8 RGB overlay array.

    Raises:
        IndexError / ValueError / pydantic.ValidationError: Propagated from
            ``build_pipeline`` when ``params`` carries a bad key or an
            out-of-bounds value (see its docstring).

    Examples:
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.enhance import GaussianBlur
        >>> base = ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()])
        >>> overlay = render_candidate_overlay(
        ...     base, {"0.sigma": 2.0}, load_synth_yeast_plate()
        ... )
        >>> overlay.ndim, overlay.shape[2]
        (3, 3)
    """
    pipeline = build_pipeline(base_pipeline, params)
    segmented = pipeline.apply(plate_image.copy())
    return to_overlay_rgb_array(segmented, max_dim=max_dim)


# ---------------------------------------------------------------------------
# A/B difference overlay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffResult:
    """The object-id partition of an A-vs-B segmentation comparison.

    Produced by :func:`difference_objects`. ``A`` is the predicted side and
    ``B`` the reference side of :func:`~phenotypic.tune._scoring._matching.\
match_iou_greedy`, so an object that both pipelines agree on lands in ``both``,
    an object only the A pipeline found lands in ``only_a``, and an object only
    the B pipeline found lands in ``only_b``.

    Attributes:
        both: Object labels (from A's objmap) matched one-to-one with a B
            object — the colonies both segmentations agree on.
        only_a: A-objmap labels with no B counterpart (A found a colony B
            missed, or A split one of B's colonies).
        only_b: B-objmap labels with no A counterpart (B found a colony A
            missed, or B split one of A's colonies).
    """

    both: list[int] = field(default_factory=list)
    only_a: list[int] = field(default_factory=list)
    only_b: list[int] = field(default_factory=list)


def difference_objects(
    objmap_a: "npt.ArrayLike",
    objmap_b: "npt.ArrayLike",
    *,
    tau: float = 0.5,
) -> DiffResult:
    """Partition two objmaps' objects into agreed / A-only / B-only sets.

    Pairs A's objects against B's via
    :func:`~phenotypic.tune._scoring._matching.match_iou_greedy` (A is the
    ``pred`` side, B the ``gt`` side). A returned pair ``(a, b)`` with both
    non-``None`` is an agreement; ``(a, None)`` is an A-only object; ``(None,
    b)`` is a B-only object.

    Args:
        objmap_a: The A-side label/objmap array (``0`` is background).
        objmap_b: The B-side label/objmap array, the same shape as ``objmap_a``.
        tau: The IoU acceptance threshold passed to ``match_iou_greedy``. At the
            default ``0.5`` the matching is provably one-to-one, so a merge or
            split surfaces as an only-A / only-B object. Defaults to ``0.5``.

    Returns:
        A :class:`DiffResult` whose ``both`` / ``only_a`` / ``only_b`` lists
        partition the union of A's and B's object labels.

    Examples:
        >>> import numpy as np
        >>> a = np.zeros((4, 8), dtype=int)
        >>> a[1:3, 1:3] = 1
        >>> a[1:3, 5:7] = 2
        >>> b = np.zeros((4, 8), dtype=int)
        >>> b[1:3, 1:3] = 1
        >>> diff = difference_objects(a, b)
        >>> diff.both, diff.only_a, diff.only_b
        ([1], [2], [])
    """
    both: list[int] = []
    only_a: list[int] = []
    only_b: list[int] = []
    for a_label, b_label in match_iou_greedy(objmap_a, objmap_b, tau=tau):
        if a_label is not None and b_label is not None:
            both.append(int(a_label))
        elif a_label is not None:
            only_a.append(int(a_label))
        elif b_label is not None:
            only_b.append(int(b_label))
    return DiffResult(both=both, only_a=only_a, only_b=only_b)


def _paint_outlines(
    canvas: np.ndarray,
    objmap: np.ndarray,
    labels: list[int],
    color: tuple[int, int, int],
) -> None:
    """Paint the boundary pixels of ``labels`` in ``objmap`` onto ``canvas``.

    Mutates ``canvas`` in place. Outline pixels are the object boundaries
    (``skimage.segmentation.find_boundaries``) restricted to the requested
    labels, so disjoint colonies each get a crisp colored ring.

    Args:
        canvas: The ``(H, W, 3)`` uint8 RGB image being drawn on.
        objmap: The integer label array the boundaries are computed from.
        labels: The object labels in ``objmap`` to outline.
        color: The ``(R, G, B)`` outline color.
    """
    if not labels:
        return
    from skimage.segmentation import find_boundaries

    selected = np.isin(objmap, labels)
    if not selected.any():
        return
    # Boundaries of the selected-label region only, so unrelated colonies that
    # touch don't bleed a ring between them.
    masked = np.where(selected, objmap, 0)
    edges = find_boundaries(masked, mode="inner") & selected
    canvas[edges] = color


def render_difference(
    plate: "npt.ArrayLike",
    objmap_a: "npt.ArrayLike",
    objmap_b: "npt.ArrayLike",
    *,
    tau: float = 0.5,
) -> np.ndarray:
    """Render A-vs-B object outlines colored by agreement over the plate.

    Colonies both pipelines agree on get a grey outline, A-only colonies a sky
    outline, and B-only colonies an orange outline — the Okabe-Ito data-palette
    tokens ``OI_GREY`` / ``OI_SKY`` / ``OI_ORANGE`` from
    :mod:`phenotypic.gui._design` (never hard-coded hex).

    Args:
        plate: The background plate image as an ``(H, W)`` or ``(H, W, 3)``
            array; grayscale input is broadcast to RGB. The background is shown
            dimmed under the colored outlines.
        objmap_a: The A-side objmap (``0`` is background).
        objmap_b: The B-side objmap, the same shape as ``objmap_a``.
        tau: The IoU threshold for the underlying matching (see
            :func:`difference_objects`). Defaults to ``0.5``.

    Returns:
        An ``(H, W, 3)`` uint8 RGB array with the difference outlines drawn,
        ready for a Plotly ``go.Image`` trace.
    """
    a = np.asarray(objmap_a)
    b = np.asarray(objmap_b)
    diff = difference_objects(a, b, tau=tau)

    base = np.asarray(plate)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    elif base.shape[-1] == 4:
        base = base[..., :3]
    canvas = np.ascontiguousarray(base[..., :3]).astype(np.uint8)

    # Order matters only where outlines overlap; both/only-A/only-B are disjoint
    # object sets so the draw order is cosmetic, but keep agreement on top.
    _paint_outlines(canvas, a, diff.only_a, _hex_to_rgb(OI_SKY))
    _paint_outlines(canvas, b, diff.only_b, _hex_to_rgb(OI_ORANGE))
    _paint_outlines(canvas, a, diff.both, _hex_to_rgb(OI_GREY))
    return canvas


def cell_disagreement(grid_a: Any, grid_b: Any) -> int:
    """Count grid cells whose per-cell colony counts differ between A and B.

    Reads each ``GridImage``'s per-cell colony counts via
    ``grid.get_section_counts()`` (a :class:`pandas.Series` keyed by section
    number — a cell with no colonies is **absent**, i.e. an implicit zero),
    aligns the two Series on the union of their cell labels filling missing
    cells with ``0``, and counts the cells whose counts differ.

    Args:
        grid_a: A ``GridImage`` (A side) exposing
            ``grid.get_section_counts()``.
        grid_b: A ``GridImage`` (B side) exposing
            ``grid.get_section_counts()``.

    Returns:
        The number of grid cells on which the two segmentations report a
        different colony count. ``0`` when the two count Series are identical.
    """
    counts_a = grid_a.grid.get_section_counts()
    counts_b = grid_b.grid.get_section_counts()
    cells = counts_a.index.union(counts_b.index)
    aligned_a = counts_a.reindex(cells, fill_value=0)
    aligned_b = counts_b.reindex(cells, fill_value=0)
    return int((aligned_a != aligned_b).sum())


__all__ = [
    "render_candidate_overlay",
    "DiffResult",
    "difference_objects",
    "render_difference",
    "cell_disagreement",
]
