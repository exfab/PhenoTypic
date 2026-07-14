"""Lazy FilFinder 1.8 raster-product adapter."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import re
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Generic,
    Literal,
    overload,
    Self,
    cast,
    TypeVar,
)
import warnings

import numpy as np
from pydantic import Field
from scipy import ndimage

from phenotypic.abc_ import ObjectDetector

if TYPE_CHECKING:
    from types import TracebackType

    from phenotypic._core._image import Image
    from phenotypic._core._grid_image import GridImage


EXPECTED_SUPPLIED_MASK_WARNING = (
    "Using inputted mask. Skipping creation of anew mask."
)
_TOPOLOGY_IMPORT_ERROR = (
    "FilFinderDetector requires FilFinder and Astropy. Install PhenoTypic "
    "with the `topology` extra before applying a nonempty detection."
)
_T = TypeVar("_T")
_WarningRecord = tuple[str, type[Warning], str, int]
_WorkerResult = tuple[int, str, _T, list[_WarningRecord]]


def _execute_with_warning_capture(
    task_index: int,
    function: Callable[..., _T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _WorkerResult[_T]:
    """Execute one process task and serialize its keyed warning records."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args, **kwargs)
    records: list[_WarningRecord] = [
        (str(item.message), item.category, item.filename, item.lineno)
        for item in caught
    ]
    return task_index, function.__qualname__, result, records


class _WarningForwardingFuture(Generic[_T]):
    """Expose a source-compatible future that retains and re-emits warnings."""

    def __init__(
        self,
        future: Future[_WorkerResult[_T]],
        task_index: int,
        warning_sink: dict[int, dict[str, object]],
    ) -> None:
        self._future = future
        self._task_index = task_index
        self._warning_sink = warning_sink

    def result(self, timeout: float | None = None) -> _T:
        """Return the child result after keyed parent warning forwarding."""
        if timeout is None:
            task_index, function_name, result, records = self._future.result()
        else:
            task_index, function_name, result, records = self._future.result(
                timeout=timeout
            )
        if task_index != self._task_index:
            raise RuntimeError("FilFinder worker task order changed")
        self._warning_sink[task_index] = {
            "task_index": task_index,
            "function": function_name,
            "warnings": records,
        }
        for message, category, filename, lineno in records:
            warnings.warn_explicit(message, category, filename, lineno)
        return result


class _WarningForwardingProcessPool:
    """Own one real process while preserving keyed worker-warning visibility."""

    def __init__(self, *, max_workers: int) -> None:
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._next_task_index = 0
        self.warning_records_by_task: dict[int, dict[str, object]] = {}

    def submit(
        self,
        function: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _WarningForwardingFuture:
        """Submit one indexed task through the warning-capture trampoline."""
        task_index = self._next_task_index
        self._next_task_index += 1
        future = self._executor.submit(
            _execute_with_warning_capture,
            task_index,
            function,
            args,
            kwargs,
        )
        return _WarningForwardingFuture(
            future, task_index, self.warning_records_by_task
        )

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        """Shut down the owned process executor."""
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.shutdown(wait=True)


def _create_warning_forwarding_pool() -> _WarningForwardingProcessPool:
    """Return the contract's fresh one-process warning-forwarding executor."""
    return _WarningForwardingProcessPool(max_workers=1)


def _load_filfinder_runtime() -> tuple[Any, Any]:
    """Import the optional runtime only for a nonempty application."""
    try:
        import astropy.units as units  # type: ignore[import-untyped]
        from fil_finder import FilFinder2D  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(_TOPOLOGY_IMPORT_ERROR) from error
    return FilFinder2D, units


def _copy_float32_source(detect_mat: np.ndarray) -> np.ndarray:
    """Apply the ImageData float32 seam before making the float64 source copy."""
    quantized = np.asarray(detect_mat, dtype=np.float32)
    return np.array(quantized, dtype=np.float64, copy=True)


def _create_mask_with_narrow_warning_policy(filfinder: Any) -> None:
    """Create the supplied mask while suppressing one exact source warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=f"^{re.escape(EXPECTED_SUPPLIED_MASK_WARNING)}$",
            category=UserWarning,
        )
        filfinder.create_mask(use_existing_mask=True)


class FilFinderDetector(ObjectDetector):
    """Detect filament rasters through the pinned FilFinder 1.8 workflow.

    Threshold ``detect_mat`` into a supplied FilFinder mask, then return the
    existing mask, pre-prune medial skeleton, or analyzed longest-path raster.
    The selected product becomes consecutive 8-connected object labels.

    Best For:
        - Connected linear growth for which a medial skeleton is informative.
        - Comparing an inclusive threshold mask with its skeleton.
        - Topology workflows that require FilFinder's pruned longest path.

    Consider Also:
        - :class:`OtsuDetector` for compact colonies separated by one threshold.
        - :class:`FilamentousFungiDetector` for the complete fungal workflow.

    Args:
        threshold: Inclusive ``detect_mat`` threshold in ``[0, 1]``. Equality
            is foreground. Default: 0.5.
        output: Raster product. Accepted values are ``"mask"``, ``"skeleton"``,
            and ``"longest_path"``. Default: ``"mask"``.
        beamwidth_px: Positive finite FilFinder beam width in pixels. Default:
            1.0.
        prune_criteria: Branch-pruning criterion. Accepted values are ``"all"``,
            ``"intensity"``, and ``"length"``. Default: ``"all"``.
        relative_intensity_threshold: Relative branch-intensity cutoff in
            ``(0, 1]``. Default: 0.2.
        branch_threshold_px: Positive finite branch-length cutoff in pixels, or
            ``None`` for FilFinder's three-beam-width default. Default: None.
        max_prune_iterations: Positive pruning-iteration cap. Default: 10.
        rng_seed: Nonnegative medial-axis tie-breaking seed. Default: 0.

    Returns:
        Image: A copy by default with ``objmask`` equal to the selected raster
        and ``objmap`` equal to deterministic 8-connected labels.

    Raises:
        ImportError: A nonempty application cannot import the ``topology`` extra.

    Examples:
        Construct the operation without importing optional dependencies:

        >>> from phenotypic.detect._filfinder_detector import FilFinderDetector
        >>> detector = FilFinderDetector(output="skeleton", rng_seed=7)
        >>> (detector.output, detector.rng_seed)
        ('skeleton', 7)

    References:
        Koch, E. W., and Rosolowsky, E. W. (2015), "Filament identification
        through mathematical morphology," MNRAS, 452(4), 3435-3450.
    """

    threshold: Annotated[
        float,
        Field(ge=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.5
    output: Literal["mask", "skeleton", "longest_path"] = "mask"
    beamwidth_px: Annotated[
        float,
        Field(gt=0.0, allow_inf_nan=False),
    ] = 1.0
    prune_criteria: Literal["all", "intensity", "length"] = "all"
    relative_intensity_threshold: Annotated[
        float,
        Field(gt=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.2
    branch_threshold_px: (
        Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None
    ) = None
    max_prune_iterations: Annotated[int, Field(ge=1, strict=True)] = 10
    rng_seed: Annotated[int, Field(ge=0, strict=True)] = 0

    @overload
    def apply(self, image: GridImage, inplace: bool = False) -> GridImage: ...

    @overload
    def apply(self, image: Image, inplace: bool = False) -> Image: ...

    def apply(
        self, image: Image | GridImage, inplace: bool = False
    ) -> Image | GridImage:
        """Apply detection while preserving the dependency error type."""
        try:
            return super().apply(image=image, inplace=inplace)
        except RuntimeError as error:
            cause: BaseException | None = error
            while cause is not None:
                if (
                    isinstance(cause, ImportError)
                    and str(cause) == _TOPOLOGY_IMPORT_ERROR
                ):
                    raise ImportError(_TOPOLOGY_IMPORT_ERROR) from cause
                cause = cause.__cause__
            raise

    def _operate(self, image: Image) -> Image:
        """Run the frozen FilFinder stage graph and label its selected raster."""
        source_image = _copy_float32_source(image.detect_mat[:])
        threshold_mask = source_image >= self.threshold
        if not threshold_mask.any():
            image.objmask[:] = np.zeros_like(threshold_mask, dtype=bool)
            image.objmap[:] = np.zeros_like(threshold_mask, dtype=np.int32)
            return image

        FilFinder2D, units = _load_filfinder_runtime()
        pool = _create_warning_forwarding_pool()
        try:
            filfinder = FilFinder2D(
                source_image.copy(),
                beamwidth=self.beamwidth_px * units.pix,
                mask=threshold_mask.copy(),
                pool=pool,
            )
            _create_mask_with_narrow_warning_policy(filfinder)

            if self.output == "mask":
                selected = np.asarray(filfinder.mask, dtype=bool).copy()
            else:
                filfinder.medskel(rng=self.rng_seed)
                if self.output == "skeleton":
                    selected = np.asarray(
                        filfinder.skeleton, dtype=bool
                    ).copy()
                else:
                    branch_threshold = (
                        None
                        if self.branch_threshold_px is None
                        else self.branch_threshold_px * units.pix
                    )
                    filfinder.analyze_skeletons(
                        prune_criteria=self.prune_criteria,
                        relintens_thresh=self.relative_intensity_threshold,
                        skel_thresh=1.0 * units.pix,
                        branch_thresh=branch_threshold,
                        max_prune_iter=self.max_prune_iterations,
                    )
                    selected = np.asarray(
                        filfinder.skeleton_longpath,
                        dtype=bool,
                    ).copy()
        finally:
            pool.shutdown(wait=True)

        objmap, _ = cast(
            tuple[np.ndarray, int],
            ndimage.label(
                selected,
                structure=np.ones((3, 3), dtype=np.uint8),
            ),
        )
        image.objmap[:] = objmap
        image.objmask[:] = objmap > 0
        return image


FilFinderDetector.apply.__doc__ = FilFinderDetector.__doc__
