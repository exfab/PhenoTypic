"""ImagePipelineBatch extends ImagePipelineCore to support batched / parallel
processing of an `ImageSet` in addition to single `Image` instances.

If an `Image` object is supplied the behaviour is identical to
`ImagePipelineCore.apply_and_measure`.  When an `ImageSet` instance is
supplied the following strategy is used:

1.  A **producer** thread iterates over the image-names that live in the
    `ImageSet` HDF5 file.  For every name it first estimates the size of
    the corresponding image-group on disk (without loading it into RAM) and
    waits until *1.25 × size* bytes of free RAM are available before enqueuing
    the name for processing.  This guards against multiple workers loading
    large images concurrently and exhausting memory on shared HPC nodes.

2.  A configurable pool of *N* worker **processes** (default: number of CPU
    cores) consumes these names.  Each worker:
        a. Opens the underlying HDF5 file in SWMR **read** mode.
        b. Loads the image via `ImageSet.get_image()`.
        c. Executes the regular `apply_and_measure` logic (in-memory) that we
           inherit from `ImagePipelineCore`.
        d. Places the processed `Image` object together with its measurement
           `DataFrame` onto a *results* queue.

3.  A dedicated single **writer** thread consumes the results queue and writes
    the processed image back to the HDF5 file (same dataset – overwrite) and
    stores the measurement table alongside it.  The writer is the *single
    writer* required for HDF5 SWMR; it keeps the file open with
    ``libver='latest'`` and periodically flushes/refreshes to allow readers to
    see the updates.

The public API returns a `pandas.DataFrame` that concatenates the individual
measurement tables (one per image) so that users can continue their
analyses in-memory once the batch job is finished.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict

from ...abstract import ImageOperation, MeasureFeatures

if TYPE_CHECKING: from phenotypic import Image, ImageSet

import multiprocessing as _mp
from multiprocessing import Queue, Event
from threading import Thread
import queue as _queue
import time
from typing import List, Tuple, Union, Optional
import logging
import os

import pandas as pd
import pickle
import warnings
import psutil
import h5py

from .._image_set import ImageSet
from phenotypic.util.constants_ import SET_STATUS
from ._image_pipeline_core import ImagePipelineCore

import threading

# Create module-level logger
logger = logging.getLogger(__name__)


class ImagePipelineBatch(ImagePipelineCore):
    """Run an `ImagePipeline` on many images concurrently."""

    def __init__(self,
                 ops: List[ImageOperation] | Dict[str, ImageOperation] | None = None,
                 meas: List[MeasureFeatures] | Dict[str, MeasureFeatures] | None = None,
                 num_workers: int = -1,
                 verbose: bool = True,
                 memblock_factor=1.25,
                 benchmark: bool = False
                 ):
        super().__init__(ops, meas, benchmark, verbose)
        # Fix: Set default num_workers to CPU count if -1, ensuring valid multiprocessing
        if num_workers == -1:
            self.num_workers = _mp.cpu_count() or 1
        else:
            self.num_workers = num_workers
        self.verbose = verbose
        self.memblock_factor = memblock_factor

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------
    def apply(  # type: ignore[override]
            self,
            subject: Union[Image, ImageSet],
            *,
            inplace: bool = False,
            reset: bool = True,
    ) -> Union[Image, None]:
        import phenotypic
        if isinstance(subject, phenotypic.Image):
            return super().apply(subject, inplace=inplace, reset=reset)
        if isinstance(subject, ImageSet):
            self._run_imageset(subject, mode="apply", num_workers=self.num_workers, verbose=self.verbose)
            return None
        raise TypeError("image must be Image or ImageSet")

    def measure(
            self,
            image: Union[Image, ImageSet],
            verbose: bool = False,
    ) -> pd.DataFrame:
        import phenotypic
        if isinstance(image, phenotypic.Image):
            return super().measure(image)
        if isinstance(image, ImageSet):
            return self._run_imageset(image, mode="measure",
                                      num_workers=self.num_workers,
                                      verbose=verbose if verbose else self.verbose)
        raise TypeError("image must be Image or ImageSet")

    def apply_and_measure(self, image: Image, inplace: bool = False, reset: bool = True, include_metadata: bool = True) -> pd.DataFrame:
        if isinstance(image, Image):
            super().apply_and_measure(image=image, inplace=inplace, reset=reset, include_metadata=include_metadata)
        elif isinstance(image, ImageSet):
            self._run_imageset(image, mode="apply_and_measure", num_workers=self.num_workers, verbose=self.verbose)

    # ----------------
    # Implementation
    # ----------------

    # TODO: Implement Pipeline apply on ImageSet metric
    def _run_imageset(self, image_set: ImageSet,
                      *,
                      mode: str,
                      num_workers: Optional[int] = None,
                      verbose: bool = False) -> Union[pd.DataFrame, None]:
        assert self.num_workers >= 3, 'Not enough cores to run image set in parallel'

        if mode == 'measure':   # Preallocate space
            pass

    # Sequential HDF5 access pattern - no concurrent access needed
    # Producer completes all file access before writer starts
    def _preallocate_measurement_datasets(self, imageset: ImageSet) -> None:
        """Pre-allocate measurement datasets for SWMR compatibility.

                    This method is called by the `ImagePipeline` class before the
                    `ImagePipeline` is run.  It creates HDF5 datasets for each measurement in image
                    in the `ImageSet` and stores them in the same HDF5 file.  This
                    ensures that the HDF5 file is not closed during the processing of
                    individual images, which would cause the file to be locked and
                    prevent any further processing.

                    Note:
                        - The image data is assumed to already be present
                    """
        pass

    def _get_measurements_dtypes_for_swmr(self):
        # needed for dtype detection
        from phenotypic import GridImage
        from phenotypic.data import make_synthetic_colony
        from phenotypic.detection import OtsuDetector
        test_image = GridImage(make_synthetic_colony(h=50, w=50, seed=0), name='dtype_test_plat', nrows=8, ncols=12)
        OtsuDetector().apply(image=test_image, inplace=True)
        try:
            meas = super().apply(test_image, inplace=False, reset=True)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            raise RuntimeError(f"Failed to run test image through pipeline: {e}") from e

        if meas is None:
            raise RuntimeError("Failed to run test image through pipeline")

        return meas
