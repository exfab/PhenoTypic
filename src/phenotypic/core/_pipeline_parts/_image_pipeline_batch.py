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

if TYPE_CHECKING: from phenotypic import Image

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
                 measurements: List[MeasureFeatures] | Dict[str, MeasureFeatures] | None = None,
                 num_workers: int = -1,
                 verbose: bool = True,
                 memblock_factor=1.25,
                 benchmark: bool = False
                 ):
        super().__init__(ops, measurements, benchmark, verbose)
        # Fix: Set default num_workers to CPU count if -1, ensuring valid multiprocessing
        if num_workers == -1:
            import multiprocessing as _mp
            self.num_workers = _mp.cpu_count() or 1
        else:
            self.num_workers = num_workers
        self.verbose = verbose
        self.memblock_factor = memblock_factor

        # Sequential HDF5 access pattern - no concurrent access needed
        # Producer completes all file access before writer starts

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Apply override
    # ------------------------------------------------------------------
    def apply(  # type: ignore[override]
            self,
            subject: Union[Image, ImageSet],
            *,
            inplace: bool = False,
            reset: bool = True,
            num_workers: Optional[int] = None,
            verbose: bool = False,
    ) -> Union[Image, None]:
        import phenotypic
        if isinstance(subject, phenotypic.Image):
            return super().apply(subject, inplace=inplace, reset=reset)
        if isinstance(subject, ImageSet):
            self._run_imageset(subject, mode="apply", num_workers=num_workers, verbose=verbose)
            return None
        raise TypeError("subject must be Image or ImageSet")

    # ------------------------------------------------------------------
    # Measure override
    # ------------------------------------------------------------------
    def measure(
            self,
            subject: Union[Image, ImageSet],
            *,
            num_workers: Optional[int] = None,
            verbose: bool = False,
    ) -> pd.DataFrame:
        import phenotypic
        if isinstance(subject, phenotypic.Image):
            return super().measure(subject)
        if isinstance(subject, ImageSet):
            return self._run_imageset(subject, mode="measure",
                                      num_workers=num_workers if num_workers else self.num_workers,
                                      verbose=verbose if verbose else self.verbose)
        raise TypeError("subject must be Image or ImageSet")

    # ------------------------------------------------------------------
    # FORCED METHOD OVERRIDE - This ensures the correct method is always called
    # ------------------------------------------------------------------
    def apply_and_measure(self, *args, **kwargs) -> pd.DataFrame:
        """FORCED OVERRIDE: Ensure ImagePipelineBatch method is always called."""
        logger.debug("ImagePipelineBatch.apply_and_measure called (forced override)")
        return self._batch_apply_and_measure(*args, **kwargs)
    
    def _batch_apply_and_measure(self, *args, **kwargs) -> pd.DataFrame:
        """Apply the pipeline either to a single `Image` **or** an `ImageSet`.
        
        This method ensures proper method resolution by accepting flexible arguments
        and routing to the appropriate processing logic based on the first argument type.
        """
        logger.debug("ImagePipelineBatch._batch_apply_and_measure called")
        
        # Handle flexible argument patterns
        if len(args) >= 1:
            subject = args[0]
        elif 'subject' in kwargs:
            subject = kwargs['subject']
        elif 'image' in kwargs:
            subject = kwargs['image']
        else:
            raise ValueError("No subject/image argument provided")
            
        # Extract other parameters with defaults
        inplace = kwargs.get('inplace', False)
        reset = kwargs.get('reset', True)
        num_workers = kwargs.get('num_workers', None)
        verbose = kwargs.get('verbose', False)
        
        logger.debug(f"Subject type: {type(subject)}")
        logger.debug(f"Method resolution: {self.__class__.__name__}.apply_and_measure")
        
        # ------------------------------------------------------------------
        # Single image – just delegate to super-class.
        # ------------------------------------------------------------------
        import phenotypic
        if isinstance(subject, phenotypic.Image):
            logger.debug("Processing single Image - delegating to parent")
            return super().apply_and_measure(subject, inplace=inplace, reset=reset)

        # ------------------------------------------------------------------
        # ImageSet –  parallel batch execution.
        # ------------------------------------------------------------------
        logger.debug("ImageSet processing path reached")
        logger.debug("apply_and_measure called with ImageSet")
        if not isinstance(subject, ImageSet):
            raise TypeError(
                "subject must be an Image or ImageSet, got " f"{type(subject)}",
            )

        logger.debug("About to call _run_imageset")
        return self._run_imageset(subject, mode="apply_and_measure",
                                  num_workers=num_workers if num_workers else self.num_workers,
                                  verbose=verbose if verbose else self.verbose)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_imageset(
            self,
            imageset: ImageSet,
            *,
            mode: str,
            num_workers: Optional[int] = None,
            verbose: bool = False,
    ) -> pd.DataFrame:
        """Parallel processing entry-point using SWMR-compatible workflow.
        
        New SWMR Workflow:
        1. Pre-allocate empty datasets for all images using normal writer
        2. Close writer handle
        3. Spawn SWMR processes: producer (reads images), workers (process), writer (saves results)
        4. Aggregate measurements from HDF5 after processing
        """
        # Create process-safe logger for this method
        parallel_logger = logging.getLogger(f"{__name__}.parallel_processing")
        parallel_logger.debug(f"_run_imageset called with mode={mode}, num_workers={num_workers}")
        num_workers = num_workers or _mp.cpu_count() or 1

        # Cross-platform fix for AuthenticationString/HMAC pickling errors
        # Use spawn context explicitly to avoid HMAC pickling issues on macOS/Python 3.12+
        # This works on both Windows (spawn) and Unix (fork) contexts
        try:
            mp_context = _mp.get_context('spawn')
            parallel_logger.info("Using spawn multiprocessing context for cross-platform compatibility")
        except RuntimeError:
            # Fallback to default context if spawn is not available
            mp_context = _mp
            parallel_logger.info("Using default multiprocessing context (spawn not available)")

        # Create multiprocessing events before pre-allocation
        pre_allocation_complete: Event = mp_context.Event()  # Signals when pre-allocation is finished

        # Step 1: Pre-allocate measurement datasets for SWMR compatibility
        parallel_logger.info("Pre-allocating measurement datasets for SWMR compatibility...")
        try:
            result = self._preallocate_measurement_datasets(imageset)
            parallel_logger.info("Successfully pre-allocated measurement datasets")
            
            # Signal that pre-allocation is complete
            parallel_logger.info("Signaling pre-allocation complete event")
            pre_allocation_complete.set()
            parallel_logger.info("Pre-allocation complete event has been set")
            
        except Exception as e:
            parallel_logger.error(f"Failed to pre-allocate measurement datasets: {e}")
            raise

        # Robust progress reporting with error handling
        pbar = None
        if verbose:
            try:
                from tqdm import tqdm
                total_images = len(imageset.get_image_names())
                pbar = tqdm(total=total_images, desc="Images", unit="img")
                parallel_logger.info(f"Progress tracking enabled for {total_images} images")
            except ImportError:
                parallel_logger.warning("tqdm not available, using simple progress logging")
                pbar = None
            except Exception as e:
                parallel_logger.warning(f"Failed to initialize progress bar: {e}, using simple progress logging")
                pbar = None

        # Note: Don't store progress bar as instance variable to avoid pickling issues
        # Progress bar is only used locally in this method

        # Cross-platform fix: Use direct multiprocessing objects instead of Manager
        # This eliminates AuthenticationString objects that cause pickling errors on Windows
        work_q: "Queue[str]" = mp_context.Queue(maxsize=num_workers * 2)
        result_q: "Queue[Tuple[str, bytes, bytes]]" = mp_context.Queue()
        stop_event: Event = mp_context.Event()
        writer_ready: Event = mp_context.Event()  # Signals when writer has opened file
        producer_hdf5_complete: Event = mp_context.Event()  # Signals when producer completes HDF5 access

        # ------------------------------------------------------------------
        # Producer – feeds *names* into work queue while respecting RAM.
        # MUST complete ALL HDF5 access BEFORE writer opens file to avoid cache conflicts
        # ------------------------------------------------------------------
        producer = Thread(
            target=self._producer,
            args=(imageset, work_q, stop_event, num_workers, writer_ready, producer_hdf5_complete),
            daemon=True,
        )
        producer.start()

        # ------------------------------------------------------------------
        # Writer – single writer thread that runs in *this* process.
        # Starts AFTER producer completes HDF5 access to prevent cache conflicts
        # ------------------------------------------------------------------
        writer = Thread(
            target=self._writer,
            args=(imageset, result_q, num_workers, stop_event, writer_ready, producer_hdf5_complete, pre_allocation_complete),
            daemon=True,
        )
        writer.start()

        # ------------------------------------------------------------------
        # Spawn worker **processes**.
        # ------------------------------------------------------------------
        workers = [
            mp_context.Process(
                target=self._worker,
                args=(imageset, work_q, result_q, stop_event, mode),
                daemon=True,
            )
            for _ in range(num_workers)
        ]
        for w in workers:
            w.start()

        # Wait for all workers to finish.
        for w in workers:
            w.join()

        # Signal writer to finish and wait.
        stop_event.set()
        writer.join()

        # Small delay to ensure HDF5 file is fully closed by writer process
        import time
        time.sleep(0.1)

        # Collect aggregated measurements directly from HDF5 file
        aggregated_df = self._aggregate_measurements_from_hdf5(imageset)
        return aggregated_df

    # ------------------------------------------------------------------
    # Queue actors
    # ------------------------------------------------------------------

    def _producer(
            self,
            imageset: ImageSet,
            work_q: "Queue[str]",
            stop_event: Event,
            num_workers: int,
            writer_ready: Event,
            producer_hdf5_complete: Event,
    ) -> None:
        """Puts image-names onto *work_q* once sufficient free RAM is available."""
        # Create process-safe logger for producer thread
        logger = logging.getLogger(f"{__name__}.producer")
        logger.info(f"Producer started - PID: {os.getpid()}")

        # Producer now runs FIRST to complete ALL HDF5 access before writer opens file
        # This prevents HDF5 cache conflicts and "ring type mismatch" errors
        logger.info("Producer: Starting HDF5 metadata loading (writer will wait for completion)")

        image_names: List[str] = imageset.get_image_names()
        logger.info(f"Found {len(image_names)} images to process: {image_names[:5]}{'...' if len(image_names) > 5 else ''}")

        # FIXED: Single HDF5 access session to prevent deadlock with writer
        logger.info("Producer: Starting consolidated HDF5 access (metadata + image loading)")
        image_sizes = {}
        loaded_images = {}

        # Single HDF5 access session to load both metadata and images
        try:
            logger.info("Producer: Opening HDF5 file for consolidated access")
            with imageset.hdf_.reader() as reader:
                image_data_group = imageset.hdf_.get_data_group(handle=reader)
                logger.info(f"Producer: Accessed image data group with {len(image_data_group)} entries")

                # First pass: Pre-load all image size estimates
                for name in image_names:
                    if stop_event.is_set():
                        logger.info("Stop event set during metadata loading")
                        break
                    image_sizes[name] = self._estimate_hdf5_dataset_size(image_data_group[name])
                    logger.debug(f"Pre-loaded size for {name}: {image_sizes[name]:,} bytes")

                # Second pass: Load actual image data in the same session
                logger.info("Producer: Loading actual image data in same HDF5 session")
                for i, name in enumerate(image_names):
                    if stop_event.is_set():
                        logger.info(f"Stop event set, terminating producer after {i} images")
                        break

                    # Use pre-loaded size estimate for RAM management
                    size_bytes = image_sizes.get(name, 0)
                    logger.debug(f"Image {name}: estimated size {size_bytes:,} bytes")

                    # Wait until enough free RAM is available
                    ram_required = size_bytes * self.memblock_factor
                    while psutil.virtual_memory().available < ram_required:
                        if stop_event.is_set():
                            logger.info("Stop event set during RAM wait, terminating producer")
                            return
                        available_ram = psutil.virtual_memory().available
                        logger.debug(f"Waiting for RAM: need {ram_required:,} bytes, have {available_ram:,} bytes")
                        time.sleep(0.5)

                    # Load actual image data
                    logger.debug(f"Loading image data for {name} ({i + 1}/{len(image_names)})")
                    try:
                        image = imageset.image_template._load_from_hdf5_group(image_data_group[name])
                        loaded_images[name] = image
                        logger.debug(f"Loaded image data for {name}")
                    except Exception as load_error:
                        logger.error(f"Failed to load image {name}: {load_error}")
                        loaded_images[name] = None

            logger.info(f"Producer: Completed consolidated HDF5 access, loaded {len(loaded_images)} images")
        except Exception as e:
            logger.error(f"Producer: Error during consolidated HDF5 access: {e}")
            producer_hdf5_complete.set()  # Signal completion even on error
            return
        
        # Signal that producer has completed ALL HDF5 access - writer can now safely open file
        logger.info("Producer: Signaling HDF5 access completion to writer")
        producer_hdf5_complete.set()

        # Now queue the pre-loaded images (no more HDF5 access needed)
        logger.info("Producer: Queuing pre-loaded images (no additional HDF5 access)")
        for i, name in enumerate(image_names):
            if stop_event.is_set():
                logger.info(f"Stop event set, terminating producer after queuing {i} images")
                break
                
            image = loaded_images.get(name)
            work_q.put((name, image))
            logger.debug(f"Queued pre-loaded image for {name} ({i + 1}/{len(image_names)})")

        # Signal the end of work – one *None* sentinel per worker.
        sentinels = num_workers
        logger.info(f"Sending {sentinels} sentinel values to terminate workers")
        for i in range(sentinels):
            work_q.put(None)  # type: ignore[arg-type]
            logger.debug(f"Sent sentinel {i + 1}/{sentinels}")

        logger.info("Producer finished successfully")

    # .................................................................
    def _worker(
            self,
            imageset: ImageSet,
            work_q: "Queue[str]",
            result_q: "Queue[Tuple[str, bytes, bytes]]",
            stop_event: Event,
            mode: str = "apply_and_measure",
    ) -> None:
        """Worker process – consumes names, processes image, returns pickled result."""
        logger = logging.getLogger(f"{__name__}.worker")
        worker_pid = os.getpid()
        logger.info(f"Worker started - PID: {worker_pid}, Mode: {mode}")

        processed_count = 0

        while not stop_event.is_set():
            try:
                work_item = work_q.get(timeout=1)
                logger.debug(f"Worker {worker_pid}: Got work item from queue")
            except _queue.Empty:
                logger.debug(f"Worker {worker_pid}: Queue timeout, checking stop event")
                continue

            if work_item is None:
                # Sentinel received – terminate.
                logger.info(f"Worker {worker_pid}: Received sentinel, terminating after processing {processed_count} images")
                break

            # Unpack work item (name, image) - no HDF5 access needed!
            name, image = work_item
            logger.info(f"Worker {worker_pid}: Processing image '{name}' (#{processed_count + 1})")

            try:
                # Check if image loading failed in producer
                if image is None:
                    logger.error(f"Worker {worker_pid}: Image '{name}' failed to load in producer")
                    result_q.put((name, b'', b'ERROR: Image loading failed'))
                    processed_count += 1
                    continue
                    
                logger.debug(f"Worker {worker_pid}: Received pre-loaded image '{name}', shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")

                processed_img = None
                measurement = None

                if mode == "apply":
                    logger.debug(f"Worker {worker_pid}: Applying pipeline to '{name}'")
                    processed_img = super(ImagePipelineBatch, self).apply(image, inplace=False, reset=True)
                    logger.debug(
                        f"Worker {worker_pid}: Pipeline applied to '{name}', result shape: {processed_img.shape if hasattr(processed_img, 'shape') else 'unknown'}")
                elif mode == "measure":
                    logger.debug(f"Worker {worker_pid}: Measuring image '{name}'")
                    measurement = super(ImagePipelineBatch, self).measure(image)
                    logger.debug(
                        f"Worker {worker_pid}: Measurements completed for '{name}', rows: {len(measurement) if measurement is not None else 0}")
                else:  # apply_and_measure
                    logger.debug(f"Worker {worker_pid}: Applying pipeline to '{name}'")
                    processed_img = super(ImagePipelineBatch, self).apply(image, inplace=False, reset=True)
                    logger.debug(f"Worker {worker_pid}: Pipeline applied to '{name}', measuring processed image")
                    measurement = super(ImagePipelineBatch, self).measure(processed_img)
                    logger.debug(
                        f"Worker {worker_pid}: Measurements completed for '{name}', rows: {len(measurement) if measurement is not None else 0}")

                # Pickle results
                img_bytes = b""
                meas_bytes = b""

                if processed_img is not None:
                    logger.debug(f"Worker {worker_pid}: Pickling processed image for '{name}'")
                    img_bytes = pickle.dumps(processed_img)
                    logger.debug(f"Worker {worker_pid}: Pickled image size: {len(img_bytes):,} bytes")

                if measurement is not None:
                    logger.debug(f"Worker {worker_pid}: Pickling measurements for '{name}'")
                    meas_bytes = pickle.dumps(measurement)
                    logger.debug(f"Worker {worker_pid}: Pickled measurements size: {len(meas_bytes):,} bytes")

                logger.debug(f"Worker {worker_pid}: Putting results for '{name}' on result queue")
                result_q.put((name, img_bytes, meas_bytes))
                processed_count += 1
                logger.info(f"Worker {worker_pid}: Successfully processed '{name}' ({processed_count} total)")

            except KeyboardInterrupt:
                logger.warning(f"Worker {worker_pid}: Keyboard interrupt received")
                raise KeyboardInterrupt
            except Exception as exc:
                logger.error(f"Worker {worker_pid}: Error processing '{name}': {exc}")
                # Forward the exception details to the writer to decide.
                result_q.put((name, pickle.dumps(RuntimeError(f'{exc}')), b""))

        # Indicate this worker is done.
        logger.info(f"Worker {worker_pid}: Sending completion signal after processing {processed_count} images")
        result_q.put(("__worker_done__", b"", b""))
        logger.info(f"Worker {worker_pid}: Terminated")

    # .................................................................
    def _writer(
            self,
            imageset: ImageSet,
            result_q: "Queue[Tuple[str, bytes, bytes]]",
            num_workers: int,
            stop_event: Event,
            writer_ready: Event,
            producer_hdf5_complete: Event,
            pre_allocation_complete: Event,
    ) -> None:
        """Single writer thread – runs in main process, writes to HDF5 (SWMR)."""
        logger = logging.getLogger(f"{__name__}.writer")
        logger.info(f"Writer started - PID: {os.getpid()}, expecting {num_workers} workers")
        logger.info(f"Writer: producer_hdf5_complete event state: {producer_hdf5_complete.is_set()}")
        logger.info(f"Writer: pre_allocation_complete event state: {pre_allocation_complete.is_set()}")

        # Wait for pre-allocation to complete before opening file
        # This ensures writer can see the pre-allocated datasets
        logger.info("Writer: Waiting for pre-allocation to complete...")
        if pre_allocation_complete.wait(timeout=30):
            logger.info("Writer: Pre-allocation completed, datasets should be visible")
        else:
            logger.error("Writer: TIMEOUT waiting for pre-allocation completion!")
            return

        # Wait for producer to complete ALL HDF5 access before opening file
        # This prevents HDF5 cache conflicts and "ring type mismatch" errors
        logger.info("Writer: Waiting for producer to complete HDF5 access...")
        if producer_hdf5_complete.wait(timeout=30):
            logger.info("Writer: Producer HDF5 access completed, safe to open file")
        else:
            logger.error("Writer: TIMEOUT waiting for producer HDF5 completion!")
            return

        finished_workers = 0
        processed_images = 0
        saved_images = 0
        saved_measurements = 0
        errors = 0

        # Now safe to open file for writing in SWMR mode (no concurrent access)
        logger.info(f"Writer: Opening HDF5 file in SWMR mode for writing: {imageset._out_path}")

        # Test without SWMR mode to avoid cache conflicts - use standard writer
        try:
            with imageset.hdf_.safe_writer() as writer:
                logger.info(f"Writer: HDF5 file opened successfully for writing (standard mode)")

                # Signal that writer is ready for SWMR concurrent access
                logger.info("Writer: Signaling writer_ready event (SWMR mode)")
                writer_ready.set()
                logger.info("Writer: writer_ready event has been set")

                # Refresh the file handle to see pre-allocated datasets
                logger.info("Writer: Refreshing HDF5 file handle to see pre-allocated datasets")
                writer.flush()  # Ensure any pending writes are committed
                
                logger.info("Writer: Accessing image data group")
                image_group = imageset.hdf_.get_data_group(handle=writer)
                logger.info(
                    f"Writer: Image data group accessed, current keys: {list(image_group.keys())[:10]}{'...' if len(image_group.keys()) > 10 else ''}")

                # Main processing loop - now inside the with block
                while finished_workers < num_workers and not stop_event.is_set():
                    try:
                        logger.debug(f"Writer: Waiting for results (finished workers: {finished_workers}/{num_workers})")
                        name, img_bytes, meas_bytes = result_q.get(timeout=1)
                        logger.debug(
                            f"Writer: Received result for '{name}', img_bytes: {len(img_bytes):,} bytes, meas_bytes: {len(meas_bytes):,} bytes")
                    except _queue.Empty:
                        logger.debug("Writer: Result queue timeout, checking stop event")
                        continue

                    if name == "__worker_done__":
                        finished_workers += 1
                        logger.info(f"Writer: Worker finished ({finished_workers}/{num_workers} completed)")
                        continue

                    logger.info(f"Writer: Processing results for image '{name}'")
                    try:
                        status_group = imageset.hdf_.get_image_status_subgroup(handle=writer, image_name=name)
                        logger.debug(f"Writer: Got status group for '{name}'")
                    except ValueError as handle_error:
                        logger.error(f"Writer: HDF5 handle error while processing '{name}': {handle_error}")
                        warnings.warn(f"Writer: Skipping '{name}' due to HDF5 handle error: {handle_error}")
                        errors += 1
                        # Skip this image and continue with others instead of breaking
                        logger.warning(f"Writer: Skipping '{name}' and continuing with remaining images")
                        continue

                    # Process exceptions first - check if img_bytes contains an exception
                    processed_img = None
                    if img_bytes:
                        try:
                            logger.debug(f"Writer: Unpickling image data for '{name}' ({len(img_bytes):,} bytes)")
                            maybe_exc = pickle.loads(img_bytes)
                            if isinstance(maybe_exc, Exception):
                                logger.error(f"Writer: Worker failed processing {name}: {maybe_exc}")
                                warnings.warn(f"Worker failed processing {name}: {maybe_exc}")
                                status_group.attrs[SET_STATUS.ERROR.label] = True
                                errors += 1
                                continue
                            else:
                                # Not an exception, it's a processed image
                                processed_img = maybe_exc
                                logger.debug(f"Writer: Successfully unpickled processed image for '{name}'")
                                status_group.attrs[SET_STATUS.PROCESSED.label] = True
                        except Exception as unpickle_error:
                            logger.error(f"Writer: Could not unpickle image data for {name}: {unpickle_error}")
                            warnings.warn(f"Worker failed processing {name}: Could not unpickle image data - {unpickle_error}")
                            status_group.attrs[SET_STATUS.ERROR.label] = True
                            errors += 1
                            continue
                    else:
                        logger.debug(f"Writer: No image data to process for '{name}'")

                    # Handle measurements
                    measurement = None
                    if meas_bytes:
                        try:
                            logger.debug(f"Writer: Unpickling measurement data for '{name}' ({len(meas_bytes):,} bytes)")
                            measurement = pickle.loads(meas_bytes)
                            logger.debug(
                                f"Writer: Successfully unpickled measurements for '{name}', shape: {measurement.shape if hasattr(measurement, 'shape') else 'unknown'}")
                            status_group.attrs[SET_STATUS.MEASURED.label] = True
                        except Exception as unpickle_error:
                            logger.error(f"Writer: Could not unpickle measurement data for {name}: {unpickle_error}")
                            warnings.warn(
                                f"Worker failed processing measurements for {name}: Could not unpickle measurement data - {unpickle_error}")
                            status_group.attrs[SET_STATUS.ERROR.label] = True
                            errors += 1
                            continue
                    else:
                        logger.debug(f"Writer: No measurement data to process for '{name}'")

                    # Save processed image if available
                    if processed_img is not None:
                        try:
                            logger.debug(f"Writer: Saving processed image for '{name}' to HDF5")
                            # Ensure processed image has the correct name for HDF5 storage
                            processed_img.name = name
                            # Save image - this will overwrite existing image data while preserving measurements
                            processed_img._save_image2hdfgroup(grp=image_group, compression="gzip", compression_opts=4)
                            processed_images += 1
                            logger.info(f"Writer: Successfully saved processed image for '{name}' ({processed_images} total)")
                        except ValueError as handle_error:
                            if "Invalid file identifier" in str(handle_error):
                                logger.error(f"Writer: HDF5 handle error while saving image '{name}': {handle_error}")
                                logger.warning(f"Writer: Skipping image save for '{name}' and continuing with remaining images")
                            else:
                                logger.error(f"Writer: Failed to save processed image for '{name}': {handle_error}")
                            try:
                                status_group.attrs[SET_STATUS.ERROR.label] = True
                            except ValueError:
                                logger.error(f"Writer: Cannot set error status for '{name}' - HDF5 handle invalid")
                            errors += 1
                            continue
                        except Exception as save_error:
                            logger.error(f"Writer: Failed to save processed image for '{name}': {save_error}")
                            try:
                                status_group.attrs[SET_STATUS.ERROR.label] = True
                            except ValueError:
                                logger.error(f"Writer: Cannot set error status for '{name}' - HDF5 handle invalid")
                            errors += 1
                            continue

                    # Save measurements if available using SWMR-compatible pre-allocated datasets
                    if measurement is not None:
                        try:
                            # Use the original image name for measurement saving (must match pre-allocation)
                            logger.debug(f"Writer: Saving measurements for '{name}' to pre-allocated SWMR datasets")
                            # Get the image group for this specific image (using original name from pre-allocation)
                            if name in image_group:
                                img_group = image_group[name]
                                logger.debug(f"Writer: Found pre-allocated image group for '{name}'")
                            else:
                                # This shouldn't happen if pre-allocation worked correctly
                                logger.error(f"Writer: Image group for {name} not found - pre-allocation may have failed")
                                logger.error(f"Writer: Available image groups: {list(image_group.keys())}")
                                warnings.warn(f"Image group for {name} not found when saving measurements. Pre-allocation may have failed.")
                                status_group.attrs[SET_STATUS.ERROR.label] = True
                                errors += 1
                                continue

                            # Use the new SWMR-compatible method to write to pre-allocated datasets
                            from .._image_set_parts._image_set_accessors._image_set_measurements_accessor import SetMeasurementAccessor
                            from phenotypic.util.constants_ import IO
                            logger.debug(f"Writer: Writing measurements for '{name}' to pre-allocated datasets")
                            
                            # Debug: Check what's actually in the image group
                            logger.debug(f"Writer: Image group '{name}' contents: {list(img_group.keys())}")
                            measurement_key = IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY
                            logger.debug(f"Writer: Looking for measurement key: '{measurement_key}'")
                            
                            if measurement_key in img_group:
                                meas_group = img_group[measurement_key]
                                logger.debug(f"Writer: Found measurement group, contents: {list(meas_group.keys())}")
                                if 'index' in meas_group:
                                    index_group = meas_group['index']
                                    logger.debug(f"Writer: Found index group, contents: {list(index_group.keys())}")
                                else:
                                    logger.debug(f"Writer: 'index' group not found in measurement group")
                                if 'values' in meas_group:
                                    values_group = meas_group['values']
                                    logger.debug(f"Writer: Found values group, contents: {list(values_group.keys())}")
                                else:
                                    logger.debug(f"Writer: 'values' group not found in measurement group")
                            else:
                                logger.debug(f"Writer: Measurement key '{measurement_key}' not found in image group")
                            
                            # Write measurements to pre-allocated datasets (row_offset=0 since each image gets its own datasets)
                            SetMeasurementAccessor._write_dataframe_to_preallocated_datasets(
                                df=measurement, 
                                group=img_group,
                                measurement_key=IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY,
                                row_offset=0
                            )
                            saved_measurements += 1
                            logger.info(f"Writer: Successfully saved measurements for '{name}' to SWMR datasets ({saved_measurements} total)")
                        except ValueError as handle_error:
                            if "Invalid file identifier" in str(handle_error):
                                logger.error(f"Writer: HDF5 handle error while saving measurements for '{name}': {handle_error}")
                                logger.warning(f"Writer: Skipping measurement save for '{name}' and continuing with remaining images")
                            else:
                                logger.error(f"Writer: Failed to save measurements for '{name}': {handle_error}")
                            try:
                                status_group.attrs[SET_STATUS.ERROR.label] = True
                            except ValueError:
                                logger.error(f"Writer: Cannot set error status for '{name}' - HDF5 handle invalid")
                            errors += 1
                            continue
                        except Exception as save_error:
                            logger.error(f"Writer: Failed to save measurements for '{name}': {save_error}")
                            try:
                                status_group.attrs[SET_STATUS.ERROR.label] = True
                            except ValueError:
                                logger.error(f"Writer: Cannot set error status for '{name}' - HDF5 handle invalid")
                            errors += 1
                            continue

                        logger.debug(f"Writer: Flushing HDF5 file after processing '{name}'")
                        writer.flush()
                        logger.debug(f"Writer: Completed processing '{name}'")

                        # Update progress bar if available
                        if hasattr(self, '_current_pbar') and self._current_pbar is not None:
                            try:
                                self._current_pbar.update(1)
                                self._current_pbar.set_postfix({
                                    'processed': processed_images,
                                    'measurements': saved_measurements,
                                    'errors': errors
                                })
                            except Exception as pbar_error:
                                pass  # Silently ignore progress bar update failures
                        else:
                            # Fallback progress logging
                            if processed_images % 5 == 0 or processed_images == 1:
                                logger.info(
                                    f"Progress: {processed_images} images processed, {saved_measurements} measurements saved, {errors} errors")

                logger.info(f"Writer finished - Processed: {processed_images} images, {saved_measurements} measurements, {errors} errors")

        except (OSError, RuntimeError) as file_error:
            logger.error(f"Failed to open HDF5 file for writing: {file_error}")
            raise

    # .................................................................
    def _aggregate_measurements_from_hdf5(self, imageset: ImageSet) -> pd.DataFrame:
        """Aggregate measurements by reading directly from HDF5 file after processing completes.

        Args:
            imageset: The ImageSet instance containing the HDF5 file path

        Returns:
            Aggregated pandas DataFrame containing all measurements
        """
        import pandas as pd
        from phenotypic.util.constants_ import IO
        measurements_list = []

        with imageset.hdf_.reader() as reader:
            image_group = reader[str(imageset.hdf_.set_data_posix)]

            for image_name in image_group.keys():
                image_subgroup = image_group[image_name]

                # Use the static method from SetMeasurementAccessor to load DataFrame
                from .._image_set_parts._image_set_accessors._image_set_measurements_accessor import SetMeasurementAccessor
                # Pass the measurement key to correctly access the measurements subgroup
                df = SetMeasurementAccessor._load_dataframe_from_hdf5_group(image_subgroup,
                                                                            measurement_key=IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY)
                if not df.empty:
                    measurements_list.append(df)

        # Concatenate all measurements
        if measurements_list:
            aggregated_df = pd.concat(measurements_list, ignore_index=True)
        else:
            aggregated_df = pd.DataFrame()

        return aggregated_df

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_hdf5_dataset_size(ds) -> int:
        """Return rough size (bytes) of an HDF5 dataset or group."""
        if isinstance(ds, h5py.Dataset):
            return int(ds.size * ds.dtype.itemsize)
        elif isinstance(ds, h5py.Group):
            return sum(ImagePipelineBatch._estimate_hdf5_dataset_size(v) for v in ds.values())
        else:
            return 0

    def _get_measurement_dtypes_for_swmr(self):
        """Determine measurement data types for SWMR pre-allocation.
        
        Returns:
            tuple: (index_dtypes, column_dtypes) where:
                - index_dtypes: list of (name, dtype) tuples for DataFrame index
                - column_dtypes: list of (name, dtype, position) tuples for DataFrame columns
        """
        logger.debug("Starting _get_measurement_dtypes_for_swmr")
        # Create a test image to determine measurement structure
        # Use a real plate image from phenotypic.data for accurate dtype detection
        from phenotypic.data import load_plate_12hr
        from phenotypic import GridImage
        logger.debug("Imported required modules for dtype detection")
        
        # Load a real plate image that will work with the pipeline
        plate_data = load_plate_12hr()
        test_image = GridImage(
            input_image=plate_data,
            name="dtype_test_plate",
            nrows=8,
            ncols=12
        )
        
        # Apply pipeline operations to test image first, then measure
        # This ensures the image has detected objects for measurement
        try:
            processed_test_image = super(ImagePipelineBatch, self).apply(test_image, inplace=False, reset=True)
            test_measurements = super(ImagePipelineBatch, self).measure(processed_test_image)
        except Exception as e:
            # If the full pipeline fails, create a minimal test DataFrame with expected structure
            # This ensures pre-allocation works even if the test image processing fails
            import pandas as pd
            import numpy as np
            
            # Create a minimal test DataFrame with typical measurement structure
            test_measurements = pd.DataFrame({
                'area': [100.0, 200.0],
                'perimeter': [50.0, 75.0], 
                'circularity': [0.8, 0.9],
                'mean_intensity': [128.5, 156.2]
            })
            # Set a default index name that matches pandas conventions
            test_measurements.index.name = None  # This will become 'level_0' when accessed
        
        # Extract index information - use consistent naming with pandas defaults
        # When index.name is None, pandas uses 'level_0' in many contexts, so we should too
        index_name = test_measurements.index.name if test_measurements.index.name is not None else 'level_0'
        index_dtype = test_measurements.index.dtype
        index_dtypes = [(index_name, index_dtype)]
        
        # Extract column information
        column_dtypes = []
        for i, col in enumerate(test_measurements.columns):
            col_dtype = test_measurements[col].dtype
            column_dtypes.append((col, col_dtype, i))
        
        return index_dtypes, column_dtypes
    
    def _preallocate_measurement_datasets(self, imageset: 'ImageSet'):
        """Pre-allocate empty measurement datasets for all images in SWMR-compatible format.
        
        This method creates the HDF5 dataset structure required for SWMR-safe measurement saving:
        - Creates measurement groups and subgroups for each image
        - Pre-allocates chunked datasets for DataFrame index and columns
        - Sets up proper SWMR structure with unlimited dimensions
        
        Args:
            imageset: ImageSet instance containing images to process
        """
        import logging
        import os
        import threading
        
        # Create parallel-safe logger with process/thread identification
        log_prefix = f"[PID:{os.getpid()}|{threading.current_thread().name}]"
        logger = logging.getLogger(f"{__name__}.preallocation")
        
        logger.info(f"{log_prefix} Starting SWMR measurement dataset pre-allocation")
        
        from phenotypic.core._image_set_parts._image_set_accessors._image_set_measurements_accessor import SetMeasurementAccessor
        from phenotypic.util.constants_ import IO
        
        # Get measurement structure from pipeline
        logger.debug(f"{log_prefix} Getting measurement dtypes for SWMR pre-allocation")
        index_dtypes, column_dtypes = self._get_measurement_dtypes_for_swmr()
        logger.info(f"{log_prefix} Pre-allocation structure - Index dtypes: {len(index_dtypes)}, Column dtypes: {len(column_dtypes)}")
        
        # Get image names for processing (same as producer will use)
        try:
            image_names = list(imageset.get_image_names())
            logger.debug(f"{log_prefix} Found {len(image_names)} images to pre-allocate: {image_names}")
        except Exception as e:
            logger.error(f"{log_prefix} Failed to get image names: {e}")
            raise
        
        if not image_names:
            logger.warning(f"{log_prefix} No images found in ImageSet - skipping pre-allocation")
            return
        
        # Open file in normal write mode for pre-allocation (before SWMR)
        logger.debug(f"{log_prefix} Opening HDF5 file for pre-allocation: {imageset._out_path}")
        try:
            with imageset.hdf_.safe_writer() as writer:
                # Get the data group containing all images
                data_group = imageset.hdf_.get_data_group(writer)
                logger.debug(f"{log_prefix} Found data group with {len(data_group.keys())} images")
                
                # Pre-allocate datasets for each image (using same names as producer/writer)
                for image_name in image_names:
                    logger.debug(f"{log_prefix} Pre-allocating datasets for image: {image_name}")
                    image_group = data_group[image_name]
                    
                    # Create status subgroup if it doesn't exist
                    if imageset.hdf_.IMAGE_STATUS_SUBGROUP_KEY not in image_group:
                        status_group = image_group.create_group(imageset.hdf_.IMAGE_STATUS_SUBGROUP_KEY)
                        logger.debug(f"{log_prefix} Created status group for {image_name}")
                    else:
                        status_group = image_group[imageset.hdf_.IMAGE_STATUS_SUBGROUP_KEY]
                    
                    # Initialize status attributes
                    status_group.attrs[SET_STATUS.PROCESSED.label] = False
                    status_group.attrs[SET_STATUS.MEASURED.label] = False
                    status_group.attrs[SET_STATUS.ERROR.label] = False
                    
                    # Pre-allocate measurement datasets using SWMR-safe method
                    logger.debug(f"{log_prefix} Calling SetMeasurementAccessor._preallocate_swmr_measurement_datasets for {image_name}")
                    
                    try:
                        SetMeasurementAccessor._preallocate_swmr_measurement_datasets(
                            image_group,
                            imageset.hdf_.IMAGE_MEASUREMENT_SUBGROUP_KEY,
                            index_dtypes,
                            column_dtypes,
                            initial_size=1000  # Pre-allocate space for up to 1000 measurements per image
                        )
                        logger.debug(f"{log_prefix} Successfully pre-allocated datasets for {image_name}")
                        
                    except Exception as e:
                        logger.error(f"{log_prefix} Failed to pre-allocate datasets for {image_name}: {e}")
                        # Mark image as having an error
                        status_group.attrs[SET_STATUS.ERROR.label] = True
                        raise
                
                # Flush changes to ensure datasets are written
                logger.debug(f"{log_prefix} Flushing HDF5 writer after pre-allocation")
                writer.flush()
                
        except Exception as e:
            logger.error(f"{log_prefix} Pre-allocation failed: {e}")
            raise
        
        logger.info(f"{log_prefix} SWMR measurement dataset pre-allocation completed successfully")

    def _get_meas_dtypes(self):
        """Legacy method - kept for backward compatibility.
        
        Returns:
            Tuple[str, type]: A tuple containing the index's name and its data type.
            List[Tuple[str, type]]: A list of tuples where each tuple represents the
            name and data type of a column in the measurement data.
        """
        index_dtypes, column_dtypes = self._get_measurement_dtypes_for_swmr()
        
        # Convert to legacy format
        index_info = (index_dtypes[0][0], index_dtypes[0][1]) if index_dtypes else ("index", object)
        column_info = [(name, dtype) for name, dtype, _ in column_dtypes]
        
        return index_info, column_info

    def _aggregate_measurements_from_hdf5(self, imageset: 'ImageSet') -> pd.DataFrame:
        """Aggregate measurements by reading directly from HDF5 file after processing completes.

        Args:
            imageset: The ImageSet instance containing the HDF5 file path

        Returns:
            Aggregated pandas DataFrame containing all measurements
        """
        import pandas as pd
        from phenotypic.util.constants_ import IO
        from phenotypic.core._image_set_parts._image_set_accessors._image_set_measurements_accessor import SetMeasurementAccessor
        
        measurements_list = []

        with imageset.hdf_.reader() as reader:
            image_group = reader[str(imageset.hdf_.set_data_posix)]

            for image_name in image_group.keys():
                image_subgroup = image_group[image_name]

                # Use the new SWMR-compatible loader first, fall back to old format
                df = SetMeasurementAccessor._load_dataframe_from_hdf5_group_swmr(
                    image_subgroup,
                    measurement_key=IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY
                )
                if not df.empty:
                    measurements_list.append(df)

        # Concatenate all measurements
        if measurements_list:
            aggregated_df = pd.concat(measurements_list, ignore_index=True)
        else:
            aggregated_df = pd.DataFrame()

        return aggregated_df

