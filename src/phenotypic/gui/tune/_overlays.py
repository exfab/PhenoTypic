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

import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from phenotypic.gui._config import THREAD_NAME_PREFIX
from phenotypic.gui._design import OI_GREY, OI_ORANGE, OI_SKY
from phenotypic.gui.builder._image_renderer import _downscale, to_overlay_rgb_array
from phenotypic.tune._evaluation._builder import build_pipeline
from phenotypic.tune._scoring._matching import match_iou_greedy

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import numpy.typing as npt

    from phenotypic import Image, ImagePipeline

#: An overlay-cache key: ``(trial_number, plate_name, mode)``. ``mode`` is the
#: overlay flavour (e.g. ``"candidate"`` / ``"difference"``).
OverlayKey = tuple[int, str, str]

#: The default longest-spatial-side clamp (px) for BOTH overlay flavours
#: (candidate + difference). Bounds the array serialized to the browser per
#: ``go.Image`` and cached in the LRU — a full-res plate is ~tens of MB each.
OVERLAY_MAX_DIM: int = 640


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
    max_dim: int = OVERLAY_MAX_DIM,
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


def _target_shape(h: int, w: int, max_dim: int) -> tuple[int, int]:
    """Return ``(new_h, new_w)`` shrinking the longer side to ``max_dim``.

    Returns the input shape unchanged when it already fits (never up-scales).
    """
    longer = max(h, w)
    if longer <= max_dim:
        return h, w
    scale = max_dim / float(longer)
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _downscale_labels_nn(objmap: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor (label-preserving) resize of ``objmap`` to ``target_hw``.

    Object/label maps must be resampled with ``order=0`` (nearest-neighbor) so
    integer label IDs survive the resize — an averaging interpolation would
    blend neighbouring labels at boundaries and invent spurious IDs. Returns
    ``objmap`` unchanged when it already matches ``target_hw``.

    Args:
        objmap: An integer ``(H, W)`` label array (``0`` is background).
        target_hw: The desired ``(height, width)``.

    Returns:
        The nearest-neighbor-resized label array, same dtype as ``objmap``.
    """
    new_h, new_w = target_hw
    if objmap.shape[:2] == (new_h, new_w):
        return objmap
    rows = (np.arange(new_h) * (objmap.shape[0] / new_h)).astype(np.intp)
    cols = (np.arange(new_w) * (objmap.shape[1] / new_w)).astype(np.intp)
    rows = np.clip(rows, 0, objmap.shape[0] - 1)
    cols = np.clip(cols, 0, objmap.shape[1] - 1)
    return objmap[np.ix_(rows, cols)]


def render_difference(
    plate: "npt.ArrayLike",
    objmap_a: "npt.ArrayLike",
    objmap_b: "npt.ArrayLike",
    *,
    tau: float = 0.5,
    max_dim: int | None = None,
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
        max_dim: When set, clamp the output's longer spatial side to this many
            pixels. The plate is anti-aliased-shrunk (``cv2.INTER_AREA``, the
            same path the candidate overlay uses) and **both** objmaps are
            downscaled label-aware (nearest-neighbor) to the **same** target
            shape, so the diff is computed consistently on the downscaled maps.
            This is the memory/perf guard the Curate render path needs: a full
            ``(4000, 6000, 3)`` difference is ~72 MB per ``go.Image`` and would
            be cached at full res. ``None`` (default) keeps the full-resolution
            behaviour (B-i's un-downscaled contract for outline correctness).

    Returns:
        An ``(H', W', 3)`` uint8 RGB array with the difference outlines drawn,
        ready for a Plotly ``go.Image`` trace. ``H'``/``W'`` are bounded by
        ``max_dim`` when set, else the input plate's dimensions.
    """
    a = np.asarray(objmap_a)
    b = np.asarray(objmap_b)

    base = np.asarray(plate)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    elif base.shape[-1] == 4:
        base = base[..., :3]
    canvas = np.ascontiguousarray(base[..., :3]).astype(np.uint8)

    if max_dim is not None:
        # Downscale the plate (anti-aliased) and BOTH objmaps (label-aware NN)
        # to the SAME target shape so the diff stays consistent. Anchor the
        # target on the plate's spatial size (what the user sees).
        target_hw = _target_shape(canvas.shape[0], canvas.shape[1], max_dim)
        if target_hw != canvas.shape[:2]:
            canvas = np.ascontiguousarray(_downscale(canvas, max_dim)).astype(
                np.uint8
            )
        a = _downscale_labels_nn(a, canvas.shape[:2])
        b = _downscale_labels_nn(b, canvas.shape[:2])

    diff = difference_objects(a, b, tau=tau)

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


# ---------------------------------------------------------------------------
# Background overlay worker + disk-LRU cache
# ---------------------------------------------------------------------------


class OverlayCache:
    """A bounded disk-LRU cache of overlay arrays rendered on a worker pool.

    The Curate Dash surface (B-ii) submits overlay renders here and polls for
    readiness rather than blocking a callback on a heavy ``apply``. Each render
    runs on a :class:`~concurrent.futures.ThreadPoolExecutor` worker (named
    ``f"{THREAD_NAME_PREFIX}-overlay"`` for log/trace attribution), and the
    resulting array is memoized **both** in a bounded in-memory LRU and on disk
    as a ``.npy`` file under ``cache_dir`` — so a fresh ``OverlayCache`` over
    the same directory (e.g. after a process restart) reuses prior renders
    without re-running the pipeline.

    The LRU ordering dict is guarded by a :class:`threading.RLock`: Werkzeug
    serves Dash callbacks from multiple threads, so concurrent submits and
    cache reads race on the ordering and eviction without it.

    Args:
        cache_dir: Directory for the persisted ``.npy`` overlay arrays. Created
            if absent.
        capacity: Maximum number of distinct keys retained in the in-memory
            LRU; the least-recently-used entry is evicted past this. The
            on-disk ``.npy`` for an evicted key is removed too, so the disk
            cache and the memory LRU stay bounded together. Defaults to ``64``.

    Examples:
        >>> import numpy as np, tempfile
        >>> d = tempfile.mkdtemp()
        >>> cache = OverlayCache(d, capacity=4)
        >>> calls = []
        >>> def render():
        ...     calls.append(1)
        ...     return np.zeros((2, 2, 3), dtype=np.uint8)
        >>> a = cache.get_or_render((0, "plate", "candidate"), render)
        >>> b = cache.get_or_render((0, "plate", "candidate"), render)  # cache hit
        >>> len(calls)  # rendered once
        1
        >>> np.array_equal(a, b)
        True
    """

    def __init__(self, cache_dir: str | Path, capacity: int = 64) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._capacity = capacity
        # MRU-ordered; value is the in-memory overlay array. Guarded by _lock.
        self._mem: "OrderedDict[OverlayKey, np.ndarray]" = OrderedDict()
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"{THREAD_NAME_PREFIX}-overlay",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _disk_path(self, key: OverlayKey) -> Path:
        """Return the ``.npy`` path for *key* (stable hash of the key tuple)."""
        digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.npy"

    def _remember(self, key: OverlayKey, array: np.ndarray) -> None:
        """Insert/refresh *key* in the LRU and evict the oldest past capacity.

        Caller must NOT hold ``self._lock`` (this method takes it).
        """
        with self._lock:
            if key in self._mem:
                self._mem.pop(key)
            self._mem[key] = array
            while len(self._mem) > self._capacity:
                old_key, _ = self._mem.popitem(last=False)
                self._disk_path(old_key).unlink(missing_ok=True)

    def _lookup(self, key: OverlayKey) -> np.ndarray | None:
        """Return the cached array for *key* (mem or disk), or ``None``.

        A memory hit bumps the key to MRU. A disk hit (memory miss) re-populates
        the in-memory LRU. ``render_fn`` is never called from here.
        """
        with self._lock:
            cached = self._mem.get(key)
            if cached is not None:
                self._mem.move_to_end(key)
                return cached
        # Memory miss — try the persisted .npy outside the lock (disk I/O).
        path = self._disk_path(key)
        if path.exists():
            array = np.load(path)
            self._remember(key, array)
            return array
        return None

    def _render_and_store(
        self, key: OverlayKey, render_fn: Callable[[], np.ndarray]
    ) -> np.ndarray:
        """Run *render_fn*, persist the array to disk, and seed the LRU.

        Re-checks the cache first so a concurrent submit of the same key renders
        only once. Executed on a pool worker.
        """
        existing = self._lookup(key)
        if existing is not None:
            return existing
        array = np.asarray(render_fn())
        np.save(self._disk_path(key), array)
        self._remember(key, array)
        return array

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self, key: OverlayKey, render_fn: Callable[[], np.ndarray]
    ) -> "Future[np.ndarray]":
        """Schedule (or short-circuit) the overlay render for *key*.

        On a cache hit (memory or disk) returns an already-resolved
        :class:`~concurrent.futures.Future` and never calls ``render_fn``.
        Otherwise the render runs on a background pool worker; poll the returned
        future, or use :meth:`is_ready` / :meth:`result`.

        Args:
            key: The ``(trial_number, plate_name, mode)`` cache key.
            render_fn: A zero-arg callable returning the overlay array. Invoked
                at most once per key (until eviction); not called on a cache hit.

        Returns:
            A :class:`~concurrent.futures.Future` resolving to the overlay array.
        """
        cached = self._lookup(key)
        if cached is not None:
            done: "Future[np.ndarray]" = Future()
            done.set_result(cached)
            return done
        return self._pool.submit(self._render_and_store, key, render_fn)

    def get_or_render(
        self, key: OverlayKey, render_fn: Callable[[], np.ndarray]
    ) -> np.ndarray:
        """Return the overlay for *key*, rendering on a worker thread if absent.

        Synchronous convenience over :meth:`submit` — blocks until the render
        (which runs on a background pool worker) completes. ``render_fn`` is
        invoked at most once per key; a second call for the same key is served
        from the cache.

        Args:
            key: The ``(trial_number, plate_name, mode)`` cache key.
            render_fn: A zero-arg callable returning the overlay array.

        Returns:
            The overlay array for *key*.
        """
        return self.submit(key, render_fn).result()

    def is_ready(self, key: OverlayKey) -> bool:
        """Return ``True`` when *key*'s overlay is cached (memory or disk).

        Args:
            key: The cache key to probe.

        Returns:
            Whether a rendered array is available without re-running
            ``render_fn``.
        """
        with self._lock:
            if key in self._mem:
                return True
        return self._disk_path(key).exists()

    def result(self, key: OverlayKey) -> np.ndarray | None:
        """Return *key*'s cached overlay array, or ``None`` if not ready.

        Does not render. Use after :meth:`is_ready` (or a resolved
        :meth:`submit` future) to fetch the array.

        Args:
            key: The cache key to fetch.

        Returns:
            The cached overlay array, or ``None`` when nothing is cached yet.
        """
        return self._lookup(key)

    def peek(self, key: OverlayKey) -> np.ndarray | None:
        """Return *key*'s cached overlay array WITHOUT consuming a future.

        The authoritative, non-destructive read the Curate readiness poll
        self-heals from (B4): the rendered array is memoized here (mem + disk
        LRU) independent of the per-tab ``_PENDING`` future registry, so when a
        re-submit drops or a consume-once ``take_overlay`` pops the future for a
        slot, the poll can still recover the figure by peeking the cache. Never
        renders (a miss returns ``None``) and never mutates the cache except to
        bump LRU recency on a hit (so a peeked key isn't evicted out from under
        the poll). Functionally an alias of :meth:`result`; named for the poll's
        "look, don't take" intent.

        Args:
            key: The cache key to probe.

        Returns:
            The cached overlay array, or ``None`` when nothing is cached yet.
        """
        return self._lookup(key)


# ---------------------------------------------------------------------------
# Process-wide OverlayCache singleton (one per run's machine-state tree)
# ---------------------------------------------------------------------------

#: The overlay-cache subdirectory under a run's ``.pht-tune-cache/`` tree.
_OVERLAY_CACHE_SUBDIR: str = "overlays"

#: Process-wide ``OverlayCache`` instances, keyed by their resolved cache dir.
#: Mirrors ``builder/_session.get_cache()`` — Werkzeug serves callbacks from
#: many threads, so the registry + its lazy init are guarded by a lock.
_OVERLAY_CACHES: "dict[str, OverlayCache]" = {}
_OVERLAY_CACHES_LOCK = threading.Lock()


def overlay_cache_dir(run_path: "str | Path") -> Path:
    """Return the overlay-cache directory under a run's machine-state tree.

    The ``.npy`` overlay memo lives under
    ``<output>/.pht-tune-cache/overlays/`` — the tune-side sibling of the
    builder's preview cache, scoped to the run's own sandbox tree (never the
    process temp dir) so a re-opened run reuses prior renders. Pure path
    expression; :class:`OverlayCache` ``mkdir``s it.

    Args:
        run_path: The tune run output directory (``TuneRunRoot.path``).

    Returns:
        ``<run_path>/.pht-tune-cache/overlays/``.
    """
    from phenotypic.tools_ import tune_cache_dir

    return tune_cache_dir(Path(run_path)) / _OVERLAY_CACHE_SUBDIR


def get_overlay_cache(run_path: "str | Path") -> "OverlayCache":
    """Return the process-wide :class:`OverlayCache` for a run.

    Lazily creates (and memoizes) one cache per run, keyed by its resolved
    overlay-cache directory (:func:`overlay_cache_dir`). Thread-safe — all
    Curate callbacks route through this rather than constructing their own
    instance, so concurrent renders share the LRU + disk memo.

    Args:
        run_path: The tune run output directory (``TuneRunRoot.path``).

    Returns:
        The shared :class:`OverlayCache` for that run.
    """
    cache_dir = overlay_cache_dir(run_path)
    key = str(cache_dir)
    with _OVERLAY_CACHES_LOCK:
        cache = _OVERLAY_CACHES.get(key)
        if cache is None:
            cache = OverlayCache(cache_dir)
            _OVERLAY_CACHES[key] = cache
        return cache


__all__ = [
    "render_candidate_overlay",
    "DiffResult",
    "difference_objects",
    "render_difference",
    "cell_disagreement",
    "OverlayKey",
    "OverlayCache",
    "OVERLAY_MAX_DIM",
    "overlay_cache_dir",
    "get_overlay_cache",
]
