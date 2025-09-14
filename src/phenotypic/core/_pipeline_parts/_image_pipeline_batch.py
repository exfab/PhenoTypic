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

import os
import pickle
import queue
import time
from typing import TYPE_CHECKING, Dict, Tuple, Literal

import psutil

from phenotypic.abstract import ImageOperation, MeasureFeatures
from phenotypic.util.constants_ import PIPE_STATUS

if TYPE_CHECKING: from phenotypic import Image, ImageSet, GridImage

import multiprocessing as _mp
import threading
from typing import List, Union, Optional
import logging

import pandas as pd

from .._image_set import ImageSet
from ._image_pipeline_core import ImagePipelineCore

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
                 benchmark: bool = False,
                 timeout: int | None = None,
                 ):
        super().__init__(ops, meas, benchmark, verbose)
        # Fix: Set default num_workers to CPU count if -1, ensuring valid multiprocessing
        if num_workers == -1:
            self.num_workers = _mp.cpu_count() or 1
        else:
            self.num_workers = num_workers
        self.verbose = verbose
        self.memblock_factor = memblock_factor
        self.timeout = timeout

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------
    def apply(  # type: ignore[override]
            self,
            image: Union[Image, ImageSet],
            inplace: bool = False,
            reset: bool = True,
    ) -> Union[Image, None]:
        import phenotypic
        if isinstance(image, phenotypic.Image):
            return super().apply(image, inplace=inplace, reset=reset)
        if isinstance(image, ImageSet):
            self._coordinator(image, mode="apply", num_workers=self.num_workers, verbose=self.verbose)
            return None
        raise TypeError("image must be Image or ImageSet")

    def measure(
            self,
            image: Union[Image, ImageSet],
            include_metadata: bool = True,
            verbose: bool = False,
    ) -> pd.DataFrame:
        import phenotypic
        if isinstance(image, phenotypic.Image):
            return super().measure(image, include_metadata=include_metadata)
        if isinstance(image, ImageSet):
            return self._coordinator(image, mode="measure",
                                     num_workers=self.num_workers,
                                     verbose=verbose if verbose else self.verbose,
                                     include_metadata=include_metadata)
        raise TypeError("image must be Image or ImageSet")

    def apply_and_measure(self, image: Image,
                          inplace: bool = False,
                          reset: bool = True,
                          include_metadata: bool = True) -> pd.DataFrame:
        if isinstance(image, Image):
            return super().apply_and_measure(
                image=image,
                inplace=inplace,
                reset=reset,
                include_metadata=include_metadata,
            )
        elif isinstance(image, ImageSet):
            return self._coordinator(image, mode="apply_and_measure", num_workers=self.num_workers)

    # ----------------
    # Implementation
    # ----------------

    # TODO: Implement Pipeline apply on ImageSet metric
    def _coordinator(self,
                     image_set: ImageSet,
                     *,
                     mode: str,
                     num_workers: Optional[int] = None,
                     ) -> Union[pd.DataFrame, None]:
        assert self.num_workers >= 3, 'Not enough cores to run image set in parallel'

        """
        Step 1: Allocate space for writing since SWMR mode only allows appending
        new data blocks.  This is required because SWMR does not allow concurrent writes.
        """
        if mode == 'measure':
            logger.info(f"allocating measurement datasets for {image_set.name}")
            self._allocate_measurement_datasets(image_set)
            logger.debug(f'allocation done. ready to process images.')
        """
        Step 2: spawn writer, producer, and worker processes.
            - single producer will wait till memory is available then enqueue image names to worker queue for processing
            - many worker processes will process each image, then enqueue results to writer queue
            - single writer will write data to hdf file as they are completed
            
            Queues:
            - work queue: holds names of image groups that need processing
            - results queue: holds processed images and their measurement tables for writing
            
            Events:
            - writer: file is open and ready for writing
            - producer_finished: when no more images are available to process, it sets stop condition
            - stop_event: event to signal when producer, workers, and writers are complete
            
        """
        parallel_logger = logging.getLogger(f'{__name__}.ImagePipeline.')
        parallel_logger.debug(f'_coordinator called with mode:{mode}, num_workers: {num_workers}')

        try:
            mp_context = _mp.get_context('spawn')
            parallel_logger.info("Using spawn multiprocessing context for cross-platform compatibility")
        except RuntimeError:
            # Fallback to the default context if spawn is not available
            mp_context = _mp
            parallel_logger.info("Using default multiprocessing context (spawn not available)")

        work_q: _mp.Queue[str] = mp_context.Queue(maxsize=self.num_workers * 2)
        results_q: _mp.Queue[Tuple[str, bytes, bytes]] = mp_context.Queue()
        thread_stop_event: threading.Event = threading.Event()

        image_names = image_set.get_image_names()

        """
        Step 2.1: Spawn Producer process to enqueue work items (image names)
        """
        producer = threading.Thread(
            target=self._producer,
            kwargs=dict(image_set=image_set, image_names=image_names, work_q=work_q, stop_event=thread_stop_event),
        )
        producer.start()

        """
        Step 2.2: Spawn writer to start processing results and writing them back to HDF5
        """
        writer = threading.Thread(
            target=self._writer,
            kwargs=dict(image_set=image_set, results_q=results_q, stop_event=thread_stop_event),
            daemon=False,
        )
        writer.start()

        """
        Step 2.3: Spawn worker processes to process images and generate measurement data
        """

        logger.debug("spawning %d workers", self.num_workers)
        workers = [
            mp_context.Process(
                target=self._worker,
                kwargs=dict(ops=self._ops, meas_ops=self._meas, work_q=work_q, results_q=results_q, mode=mode),
                daemon=False,
            )
            for _ in range(self.num_workers - 2)
        ]
        logger.debug("all workers spawned")

        for w in workers: w.start()
        logger.info('All worker processes started')

        for w in workers: w.join()
        logger.info(f'All worker processes completed, joining writer...')

        thread_stop_event.set()
        writer.join(timeout=self.timeout)

        """
        Step 3: Check file handles are closed and concatenate results into a single dataframe if in measure mode
        """
        if mode == 'measure':
            # TODO: Add measurement aggregator
            pass

    # Sequential HDF5 access pattern - no concurrent access needed
    # Producer completes all file access before writer starts
    def _allocate_measurement_datasets(self, imageset: ImageSet) -> None:
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
        sample_meas = self._get_measurements_dtypes_for_swmr()
        image_names = imageset.get_image_names()
        with imageset.hdf_.safe_writer() as writer:
            for image_name in image_names:
                status_group = imageset.hdf_.get_image_status_subgroup(handle=writer, image=image_name)
                status_group.attrs[PIPE_STATUS.PROCESSED] = False
                status_group.attrs[PIPE_STATUS.MEASURED] = False

                meas_group = imageset.hdf_.get_image_measurement_subgroup(handle=writer, image_name=image_name)
                imageset.hdf_.preallocate_frame_layout(
                    group=meas_group,
                    dataframe=sample_meas,
                    chunks=25,
                    compression='gzip',
                    preallocate=100,
                    string_fixed_length=100,
                    require_swmr=False,
                )

    def _get_measurements_dtypes_for_swmr(self) -> pd.DataFrame:
        # needed for dtype detection
        from phenotypic.data import load_synthetic_colony
        test_image = load_synthetic_colony(mode='Image') # This is a tiny image for fast execution of measurements
        try:
            meas = super().measure(test_image, include_metadata=False)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            raise RuntimeError(f"Failed to run test image through pipeline: {e}") from e

        if meas is None:
            raise RuntimeError("Failed to run test image through pipeline")

        return meas

    def _producer(self, image_set: ImageSet, image_names: List[str],
                  work_q: _mp.Queue[Image | GridImage | None], stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            with image_set.hdf_.swmr_reader() as reader:
                for name in image_names:
                    image_group = image_set.hdf_.get_image_group(handle=reader, image_name=name)
                    image_footprint = image_set.hdf_.get_uncompressed_sizes_for_group(image_group)

                    # protect from out-of-memory error and release GIL
                    while psutil.virtual_memory().available < image_footprint * self.memblock_factor:
                        time.sleep(0.1)

                    if image_set.grid_finder is None:
                        image = Image()
                    else:
                        image = GridImage(grid_finder=image_set.grid_finder)
                    work_q.put(
                        image._load_from_hdf5_group(image_group),
                    )

    def _writer(self, image_set: ImageSet, results_q: _mp.Queue[Tuple[str, bytes, bytes]], stop_event: threading.Event):
        while not stop_event.is_set():
            try:
                image_name, image_bytes, meas_bytes = results_q.get(timeout=1)
                with image_set.hdf_.swmr_writer() as writer:
                    status_group = image_set.hdf_.get_image_status_subgroup(handle=writer, image=image_name)

                    try:  # Save processed image if pipeline successfully executed
                        image = pickle.loads(image_bytes)
                        if image != b'':
                            image_group = image_set.hdf_.get_image_group(handle=writer, image_name=image_name)
                            image._save_image2hdfgroup(grp=image_group, )
                            status_group.attrs[PIPE_STATUS.PROCESSED] = True
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except Exception as e:
                        logger.error(f'Error saving image {image_name}: {e}')

                    try:  # save measurements if pipeline successfully executed
                        meas = pickle.loads(meas_bytes)
                        if meas != b'':
                            meas_group = image_set.hdf_.get_image_measurement_subgroup(handle=writer, image_name=image_name)
                            image_set.hdf_.save_frame_update(meas_group, meas, start=0, require_swmr=True)
                            status_group.attrs[PIPE_STATUS.MEASURED] = True
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except Exception as e:
                        logger.error(f'Error saving measurements for {image_name}: {e}')

            except queue.Empty: # release GIL if queue is empty
                time.sleep(0.1)

    @classmethod
    def _worker(cls, ops, meas_ops,
                work_q: _mp.Queue[Image | GridImage | None],
                results_q: _mp.Queue[Tuple[str, bytes, bytes]],
                mode: Literal['apply', 'measure', 'apply_and_measure']
                ) -> None:
        logger = logging.getLogger(f"{__name__}.worker")
        worker_pid = os.getpid()
        logger.info(f"Worker started - PID: {worker_pid}, Mode: {mode}")

        pipe = cls(ops, meas_ops, benchmark=False, verbose=False)
        while True:
            image = work_q.get()
            if image is None:  # Sentinel
                logger.debug("Termination signal received. Exiting worker.")
                break

            else:
                # default image name and meas value
                image_name, meas = image.name, b''

                logger.info(f'Starting processing of image {image.name} (PID: {worker_pid})')
                if mode == 'apply' or mode == 'apply_and_measure':
                    try:
                        pipe.apply(image, inplace=True)
                        logger.debug(f'Image {image_name} successfully processed.')
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except Exception as e:
                        logger.error(f"Exception occurred during apply phase on image {image.name}: {e}")
                        image = b''  # If processing error occurs we pass an empty byte string so that nothing is overwritten

                if (mode == 'measure' or mode == 'apply_and_measure') and image != b'':
                    try:
                        meas = pipe.measure(image)
                        logger.debug(f'Measurements saved for image {image_name}')
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except Exception as e:
                        logger.error(f"Exception occurred during measure phase on image {image.name}: {e}")
                        meas = b''

                results_q.put((image_name, pickle.dumps(image), pickle.dumps(meas)))
