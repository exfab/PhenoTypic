from __future__ import annotations

import os
import pickle
import queue
import time
from typing import Dict, Tuple, TYPE_CHECKING

import psutil

from phenotypic.abc_ import ImageOperation, MeasureFeatures
from phenotypic.tools.constants_ import PIPE_STATUS

if TYPE_CHECKING: from phenotypic import Image, ImageSet, GridImage

import multiprocessing as _mp
import threading
from typing import List, Union, Optional
import logging
from enum import Enum
import traceback

import pandas as pd

from .._image_set import ImageSet
from ._image_pipeline_core import ImagePipelineCore

# Create module-level logger
logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    APPLY = "apply"
    MEASURE = "measure"
    APPLY_MEASURE = "apply_measure"


class ImagePipelineBatch(ImagePipelineCore):
    """
    Handles batch processing of images using specified operations and measurement
    features while supporting multi-processing for enhanced performance.

    This class manages execution of various image processing tasks in parallel
    using a configurable number of workers. Depending on the input type, it applies
    image operations, performs measurements, or executes both in sequence. The
    class also provides features for producer-consumer-style multi-threading
    coordination and ensures compatibility with shared HDF5 resources.

    Attributes:
        num_workers (int): Number of worker processes for parallel execution. Default is
            equal to the system's CPU count or as specified.
        verbose (bool): Whether to enable verbose logging for debugging purposes.
        memblock_factor (float): Adjustment factor for memory allocation during operations.
        timeout (Optional[int]): Time limit (in seconds) for joining threads during
            multi-threaded execution.
    """

    def __init__(self,
                 ops: List[ImageOperation] | Dict[str, ImageOperation] | None = None,
                 meas: List[MeasureFeatures] | Dict[str, MeasureFeatures] | None = None,
                 njobs: int = -1,
                 verbose: bool = True,
                 memblock_factor=1.25,
                 benchmark: bool = False,
                 timeout: int | None = None,
                 ):
        super().__init__(ops, meas, benchmark, verbose)
        # Fix: Set default n_jobs to CPU count if -1, ensuring valid multiprocessing
        if njobs == -1:
            self.num_workers = _mp.cpu_count() or 1
        else:
            self.num_workers = njobs
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
    ) -> Union[GridImage, Image, None]:
        import phenotypic

        if isinstance(image, phenotypic.Image):
            return super().apply(image, inplace=inplace, reset=reset)
        if isinstance(image, ImageSet):
            self._coordinator(image, mode=PipelineMode.APPLY, num_workers=self.num_workers, verbose=self.verbose)
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
        if isinstance(image, phenotypic.ImageSet):
            return self._coordinator(image,
                                     mode=PipelineMode.MEASURE,
                                     num_workers=self.num_workers, )
        raise TypeError("image must be Image or ImageSet")

    def apply_and_measure(self, image: Image | ImageSet,
                          inplace: bool = False,
                          reset: bool = True,
                          include_metadata: bool = True) -> Union[pd.DataFrame, None]:
        import phenotypic as pt

        if isinstance(image, pt.Image):
            return super().apply_and_measure(
                    image=image,
                    inplace=inplace,
                    reset=reset,
                    include_metadata=include_metadata,
            )
        elif isinstance(image, pt.ImageSet):
            return self._coordinator(image,
                                     mode=PipelineMode.APPLY_MEASURE,
                                     num_workers=self.num_workers, reset=reset)
        else:
            raise TypeError("image must be Image or ImageSet")

    # ----------------
    # Implementation
    # ----------------

    # TODO: Implement Pipeline apply on ImageSet metric
    def _coordinator(self,
                     image_set: ImageSet,
                     *,
                     mode: PipelineMode,
                     num_workers: Optional[int] = None,
                     reset: bool = True,
                     ) -> Union[pd.DataFrame, None]:
        assert self.num_workers >= 2, 'Not enough cores to run image set in parallel'
        logger = logging.getLogger('ImagePipeline.coordinator')

        """
        Step 1: Allocate space for writing since SWMR mode only allows appending
        new data blocks.  This is required because SWMR does not allow concurrent writes.
        """
        if mode in {PipelineMode.MEASURE, PipelineMode.APPLY_MEASURE}:
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
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            parallel_logger = logging.getLogger(f'ImagePipeline.parallel')
            parallel_logger.debug(f'_coordinator called with mode:{mode}, njobs: {num_workers}')

            try:
                mp_context = _mp.get_context('spawn')
                parallel_logger.info("Using spawn multiprocessing context for cross-platform compatibility")
            except RuntimeError:
                # Fallback to the default context if spawn is not available
                mp_context = _mp
                parallel_logger.info("Using default multiprocessing context (spawn not available)")

            work_q: _mp.Queue[bytes | None] = mp_context.Queue(maxsize=self.num_workers*2)
            results_q: _mp.Queue[Tuple[str, bytes, bytes]] = mp_context.Queue()
            writer_access_event: threading.Event = threading.Event()
            thread_stop_event: threading.Event = threading.Event()

            image_names = image_set.get_image_names()

            """
            Step 2.1: Spawn writer to start processing results and writing them back to HDF5
            """
            writer = threading.Thread(
                    target=self._writer,
                    kwargs=dict(image_set=image_set,
                                results_q=results_q,
                                writer_access_event=writer_access_event,
                                stop_event=thread_stop_event),
                    daemon=False,
            )
            writer.start()

            """
            Step 2.2: Spawn Producer process to enqueue work items (image names)
            """
            producer = threading.Thread(
                    target=self._producer,
                    kwargs=dict(image_set=image_set,
                                image_names=image_names,
                                work_q=work_q,
                                writer_access_event=writer_access_event,
                                stop_event=thread_stop_event),
            )
            producer.start()

            """
            Step 2.3: Spawn worker processes to process images and generate measurement data
            """

            logger.debug("spawning %d workers", self.num_workers)
            workers = [
                mp_context.Process(
                        target=self._worker,
                        kwargs=dict(ops=self._ops,
                                    meas_ops=self._meas,
                                    work_q=work_q,
                                    results_q=results_q,
                                    mode=mode,
                                    reset=reset),
                        daemon=False,
                )
                for _ in range(self.num_workers - 1)
            ]
            logger.debug("all workers spawned")

            for w in workers: w.start()
            logger.info('All worker processes started')

            for w in workers: w.join()
            logger.info(f'All worker processes completed, joining writer...')

            thread_stop_event.set()
            logger.info(f'Stop event set. Waiting on writer to complete...')
            producer.join(timeout=self.timeout)
            writer.join(timeout=self.timeout)
            logger.info(f'Writer and producer joined successfully. Exiting coordinator.''')

        """
        Step 3: Check file handles are closed and concatenate results into a single dataframe if in measure mode
        """
        logger.info('Cleaning up after processing and aggregating measurements...')
        if mode in {PipelineMode.MEASURE, PipelineMode.APPLY_MEASURE}:
            return image_set.get_measurement()
        else:
            return None

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
        logger = logging.getLogger('ImagePipeline')
        sample_meas = self._get_measurements_dtypes_for_swmr(imageset.imtype)
        logger.debug(f'allocating measurements with columns: {sample_meas.columns}')
        image_names = imageset.get_image_names()
        with imageset.hdf_.safe_writer() as writer:
            for image_name in image_names:
                status_group = imageset.hdf_.get_status_subgroup(handle=writer, image_name=image_name)
                status_group.attrs.modify(name=PIPE_STATUS.PROCESSED.label, value=False)
                status_group.attrs.modify(name=PIPE_STATUS.MEASURED.label, value=False)
                assert PIPE_STATUS.PROCESSED.label in status_group.attrs, "processed flag missing from status group attrs"
                assert PIPE_STATUS.MEASURED.label in status_group.attrs, "measured flag missing from status group attrs"
                logger.debug(
                        f'Allocating statuses for {image_name}: {status_group.attrs.keys()} -> {status_group.attrs.values()}')
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
        return

    def _get_measurements_dtypes_for_swmr(self, imtype: str) -> pd.DataFrame:
        # needed for dtype detection
        from phenotypic.data import load_synthetic_colony
        from phenotypic import GridImage, Image
        from phenotypic.abc_ import ObjectDetector

        class DetectFull(ObjectDetector):
            def _operate(self, image):
                image.objmask[:] = 1
                return image

        # Create test image matching the ImageSet's image type
        array_data = load_synthetic_colony(mode='array')
        if imtype == "GridImage":
            test_image = GridImage(array_data)
        else:  # "Image" or default
            test_image = Image(array_data, name="test")

        DetectFull().apply(test_image, inplace=True)
        try:
            meas = super().measure(test_image, include_metadata=False)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            raise RuntimeError(f"Failed to run test image through pipeline: {e}") from e

        if meas is None:
            raise RuntimeError("Failed to run test image through pipeline")

        return meas

    def _producer(self,
                  image_set: ImageSet, image_names: List[str],
                  work_q: _mp.Queue[bytes | None],
                  writer_access_event: threading.Event,
                  stop_event: threading.Event) -> None:
        logging.getLogger('ImagePipeline.producer')

        import phenotypic as pt

        while not writer_access_event.is_set():
            logger.info(f"producer waiting for access to {writer_access_event}")
            time.sleep(0.1)
        logger.info('Writer completed access event, starting producer loop')
        with image_set.hdf_.swmr_reader() as reader:
            while not stop_event.is_set():
                logger.info(f'producer reading images...')
                for name in image_names:
                    image_group = image_set.hdf_.get_image_group(handle=reader, image_name=name)
                    image_footprint = image_set.hdf_.get_uncompressed_sizes_for_group(image_group)[1]

                    # protect from out-of-memory error and release GIL
                    while psutil.virtual_memory().available < image_footprint*self.memblock_factor:
                        time.sleep(0.1)

                    image = image_set._get_image(image_name=name, handle=image_group, **image_set.imparams)

                    assert isinstance(image, (pt.Image, pt.GridImage)), f"Invalid Image type: {type(image)}"
                    image_pkl = pickle.dumps(image)
                    work_q.put(
                            image_pkl,
                    )
                for _ in range(self.num_workers - 1):
                    work_q.put(None)
                break

    def _writer(self, image_set: ImageSet, results_q: _mp.Queue[Tuple[str, bytes, bytes]],
                writer_access_event: threading.Event, stop_event: threading.Event):
        import phenotypic as pt

        logger = logging.getLogger(f'ImagePipeline.writer')
        logger.info(f'Accessing hdf file')

        num_workers = self.num_workers - 1
        workers_done = 0

        with image_set.hdf_.swmr_writer() as writer:
            writer_access_event.set()
            logger.info('Writer set access event, starting writer loop')

            while workers_done < num_workers and not stop_event.is_set():
                try:
                    result = results_q.get(timeout=1)
                    if result == ("WORKER_DONE", None, None):
                        workers_done += 1
                        logger.info(f'Worker completed. {workers_done}/{num_workers} workers done.')
                        continue

                    image_name, image_bytes, meas_bytes = result
                    status_group = image_set.hdf_.get_status_subgroup(handle=writer, image_name=image_name)

                    logger.info(f'Saving image: {image_name}')
                    try:  # Save processed image if pipeline successfully executed
                        image = pickle.loads(image_bytes)
                        if isinstance(image, pt.Image) or isinstance(image, pt.GridImage):
                            logger.info('Got valid image from queue')
                            image_group = image_set.hdf_.get_data_group(handle=writer)
                            image._save_image2hdfgroup(grp=image_group, overwrite=False)
                            status_group.attrs.modify(PIPE_STATUS.PROCESSED.label, True)
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except Exception as e:
                        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))

                        logger.error(f'Error saving image {image_name}: {tb_str}')

                    logger.info(f'Starting measurement processing for: {image_name}')
                    try:  # save measurements if pipeline successfully executed
                        meas = pickle.loads(meas_bytes)
                        if isinstance(meas, pd.DataFrame):
                            logger.debug('Got valid DataFrame')
                            meas_group = image_set.hdf_.get_image_measurement_subgroup(handle=writer,
                                                                                       image_name=image_name)
                            image_set.hdf_.save_frame_update(meas_group, meas, start=0, require_swmr=True)
                            status_group.attrs.modify(PIPE_STATUS.MEASURED.label, True)
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except Exception as e:
                        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                        logger.error(f'Error saving measurements for {image_name}: {tb_str}')

                except queue.Empty:  # release GIL if queue is empty
                    continue

    @classmethod
    def _worker(cls, ops, meas_ops,
                work_q: _mp.Queue[bytes | None],
                results_q: _mp.Queue[Tuple[str, bytes, bytes]],
                mode: PipelineMode,
                reset: bool
                ) -> None:
        logger = logging.getLogger(f"ImagePipeline._worker()")
        worker_pid = os.getpid()
        logger.info(f"Worker started - PID: {worker_pid}, Mode: {mode}")

        pipe = cls(benchmark=False, verbose=False)
        pipe.set_ops(ops)
        pipe.set_measurements(meas_ops)
        # pipe._ops = ops
        # pipe._meas = meas_ops
        while True:
            image = work_q.get()
            if image is None:  # Sentinel
                logger.debug("Termination signal received. Exiting worker.")
                results_q.put((f'WORKER_DONE', None, None))
                break

            else:
                image = pickle.loads(image)

                # default image name and meas value
                image_name, meas = image.name, b''

                logger.info(f'Starting processing of image {image.name} (PID: {worker_pid})')
                if mode in {PipelineMode.APPLY, PipelineMode.APPLY_MEASURE}:
                    try:
                        image = pipe.apply(image, inplace=True, reset=reset)
                        logger.debug(f'Image {image_name} successfully processed.')
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except Exception as e:
                        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                        logger.error(f"Exception occurred during apply phase on image {image.name}: {tb_str}")
                        image = b''  # If processing error occurs we pass an empty byte string so that nothing is overwritten

                if mode in {PipelineMode.MEASURE, PipelineMode.APPLY_MEASURE} and not isinstance(image, bytes):
                    try:
                        meas = pipe.measure(image, include_metadata=False)
                        logger.debug(f'Measurements saved for image {image_name}')
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except Exception as e:
                        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))

                        logger.error(f"Exception occurred during measure phase on image {image.name}: {tb_str}")
                        meas = b''

                results_q.put((image_name, pickle.dumps(image), pickle.dumps(meas)))
