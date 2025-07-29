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


class ImagePipelineBatch(ImagePipelineCore):
    """Run an `ImagePipeline` on many images concurrently."""

    def __init__(self,
                 ops: List[ImageOperation] | Dict[str, ImageOperation] | None = None,
                 measurements: List[MeasureFeatures] | Dict[str, MeasureFeatures] | None = None,
                 num_workers: int = -1,
                 verbose: bool = True,
                 memblock_factor=1.25
                 ):
        super().__init__(ops, measurements)
        self.num_workers = num_workers
        self.verbose = verbose
        self.memblock_factor = memblock_factor

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
    def apply_and_measure(
            self,
            subject: Union[Image, ImageSet],
            *,
            inplace: bool = False,
            reset: bool = True,
            num_workers: Optional[int] = None,
            verbose: bool = False,
    ) -> pd.DataFrame:
        """Apply the pipeline either to a single `Image` **or** an `ImageSet`.

        Parameters
        ----------
        subject
            ``Image`` or ``ImageSet`` instance.
        inplace, reset
            Passed through when operating on a single ``Image``.
        num_workers
            How many worker processes to spawn when `subject` is an
            ``ImageSet``.  Defaults to ``os.cpu_count()``.

        Returns
        -------
        If *subject* is an ``Image`` the return value is identical to
        :py:meth:`ImagePipelineCore.apply_and_measure` – a tuple of the
        processed ``Image`` and its measurement ``DataFrame``.

        If *subject* is an ``ImageSet`` the method blocks until the whole
        set has been processed and returns **one** aggregated
        ``pandas.DataFrame`` (``pd.concat`` along ``index``) containing the
        measurements for all images.
        """
        # ------------------------------------------------------------------
        # Single image – just delegate to super-class.
        # ------------------------------------------------------------------
        import phenotypic
        if isinstance(subject, phenotypic.Image):
            return super().apply_and_measure(subject, inplace=inplace, reset=reset)

        # ------------------------------------------------------------------
        # ImageSet –  parallel batch execution.
        # ------------------------------------------------------------------
        if not isinstance(subject, ImageSet):
            raise TypeError(
                "subject must be an Image or ImageSet, got " f"{type(subject)}",
            )

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
        """Parallel processing entry-point used by apply/measure/apply_and_measure."""
        num_workers = num_workers or _mp.cpu_count() or 1

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=len(imageset.get_image_names()), desc="Images")
        else:
            pbar = None

        manager = _mp.Manager()
        work_q: "Queue[str]" = manager.Queue(maxsize=num_workers * 2)
        result_q: "Queue[Tuple[str, bytes, bytes]]" = manager.Queue()
        stop_event: Event = manager.Event()
        writer_ready: Event = manager.Event()  # Signals when writer has opened file

        # ------------------------------------------------------------------
        # Writer – single writer thread that runs in *this* process.
        # Must start first to open file for writing before producer reads
        # ------------------------------------------------------------------
        writer = Thread(
            target=self._writer,
            args=(imageset, result_q, num_workers, stop_event, writer_ready),
            daemon=True,
        )
        writer.start()

        # ------------------------------------------------------------------
        # Producer – feeds *names* into work queue while respecting RAM.
        # Waits for writer to be ready before starting
        # ------------------------------------------------------------------
        producer = Thread(
            target=self._producer,
            args=(imageset, work_q, stop_event, num_workers, writer_ready),
            daemon=True,
        )
        producer.start()

        # ------------------------------------------------------------------
        # Spawn worker **processes**.
        # ------------------------------------------------------------------
        workers = [
            _mp.Process(
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
    ) -> None:
        """Puts image-names onto *work_q* once sufficient free RAM is available."""
        logger = logging.getLogger(f"{__name__}.producer")
        logger.info(f"Producer started - PID: {os.getpid()}")
        
        # Wait for writer to open the file first
        logger.info("Waiting for writer to be ready...")
        writer_ready.wait()
        logger.info("Writer is ready, proceeding with production")

        image_names: List[str] = imageset.get_image_names()
        logger.info(f"Found {len(image_names)} images to process: {image_names[:5]}{'...' if len(image_names) > 5 else ''}")

        # Try to open with SWMR, fallback to regular read-only if not supported
        try:
            logger.info("Attempting to open HDF5 file with SWMR reader")
            reader = imageset._hdf.reader()
            logger.info("Successfully opened HDF5 file with SWMR reader")
        except (RuntimeError, ValueError) as e:
            logger.warning(f"SWMR reader failed ({e}), falling back to regular read-only mode")
            reader = h5py.File(imageset._out_path, "r", libver="latest")
            logger.info("Successfully opened HDF5 file in read-only mode")

        with reader:
            image_data = imageset._hdf.get_image_data_group(handle=reader)
            logger.info(f"Accessed image data group with {len(image_data)} entries")
            
            for i, name in enumerate(image_names):
                if stop_event.is_set():
                    logger.info(f"Stop event set, terminating producer after {i} images")
                    break

                # Estimate size on disk – used as proxy for RAM requirement.
                size_bytes = self._estimate_hdf5_dataset_size(image_data[name])
                logger.debug(f"Image {name}: estimated size {size_bytes:,} bytes")
                
                # Wait until enough free RAM is available.
                ram_required = size_bytes * self.memblock_factor
                while psutil.virtual_memory().available < ram_required:
                    if stop_event.is_set():
                        logger.info("Stop event set during RAM wait, terminating producer")
                        return
                    available_ram = psutil.virtual_memory().available
                    logger.debug(f"Waiting for RAM: need {ram_required:,} bytes, have {available_ram:,} bytes")
                    time.sleep(0.5)

                logger.debug(f"Queuing image {name} for processing ({i+1}/{len(image_names)})")
                work_q.put(name)

        # Signal the end of work – one *None* sentinel per worker.
        sentinels = num_workers
        logger.info(f"Sending {sentinels} sentinel values to terminate workers")
        for i in range(sentinels):
            work_q.put(None)  # type: ignore[arg-type]
            logger.debug(f"Sent sentinel {i+1}/{sentinels}")
        
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
                name = work_q.get(timeout=1)
                logger.debug(f"Worker {worker_pid}: Got work item from queue")
            except _queue.Empty:
                logger.debug(f"Worker {worker_pid}: Queue timeout, checking stop event")
                continue

            if name is None:
                # Sentinel received – terminate.
                logger.info(f"Worker {worker_pid}: Received sentinel, terminating after processing {processed_count} images")
                break

            logger.info(f"Worker {worker_pid}: Processing image '{name}' (#{processed_count + 1})")
            
            try:
                # Load image
                logger.debug(f"Worker {worker_pid}: Loading image '{name}' from ImageSet")
                image: Image = imageset.get_image(name)
                logger.debug(f"Worker {worker_pid}: Successfully loaded image '{name}', shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
                
                processed_img = None
                measurement = None
                
                if mode == "apply":
                    logger.debug(f"Worker {worker_pid}: Applying pipeline to '{name}'")
                    processed_img = super(ImagePipelineBatch, self).apply(image, inplace=False, reset=True)
                    logger.debug(f"Worker {worker_pid}: Pipeline applied to '{name}', result shape: {processed_img.shape if hasattr(processed_img, 'shape') else 'unknown'}")
                elif mode == "measure":
                    logger.debug(f"Worker {worker_pid}: Measuring image '{name}'")
                    measurement = super(ImagePipelineBatch, self).measure(image)
                    logger.debug(f"Worker {worker_pid}: Measurements completed for '{name}', rows: {len(measurement) if measurement is not None else 0}")
                else:  # apply_and_measure
                    logger.debug(f"Worker {worker_pid}: Applying pipeline to '{name}'")
                    processed_img = super(ImagePipelineBatch, self).apply(image, inplace=False, reset=True)
                    logger.debug(f"Worker {worker_pid}: Pipeline applied to '{name}', measuring processed image")
                    measurement = super(ImagePipelineBatch, self).measure(processed_img)
                    logger.debug(f"Worker {worker_pid}: Measurements completed for '{name}', rows: {len(measurement) if measurement is not None else 0}")

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
    ) -> None:
        """Single writer thread – runs in main process, writes to HDF5 (SWMR)."""
        logger = logging.getLogger(f"{__name__}.writer")
        logger.info(f"Writer started - PID: {os.getpid()}, expecting {num_workers} workers")
        
        finished_workers = 0
        processed_images = 0
        saved_measurements = 0
        errors = 0

        # Open file in append mode to avoid conflicts with existing readers
        # Create with proper version for SWMR compatibility
        logger.info(f"Opening HDF5 file for writing: {imageset._out_path}")
        with imageset._hdf.writer() as writer:
            logger.info("HDF5 file opened successfully for writing")
            
            # Only enable SWMR if file version supports it
            try:
                writer.swmr_mode = True  # enable SWMR once file is opened for writing
                logger.info("SWMR mode enabled successfully")
            except RuntimeError as e:
                # File version doesn't support SWMR, continue without it
                logger.warning(f"SWMR mode not supported ({e}), continuing without it")

            logger.info("Accessing image data group")
            image_group = imageset._hdf.get_image_data_group(handle=writer)
            logger.info(f"Image data group accessed, current keys: {list(image_group.keys())[:10]}{'...' if len(image_group.keys()) > 10 else ''}")

            # Signal that writer is ready and file is open for SWMR
            logger.info("Signaling writer ready")
            writer_ready.set()

            while finished_workers < num_workers and not stop_event.is_set():
                try:
                    logger.debug(f"Writer: Waiting for results (finished workers: {finished_workers}/{num_workers})")
                    name, img_bytes, meas_bytes = result_q.get(timeout=1)
                    logger.debug(f"Writer: Received result for '{name}', img_bytes: {len(img_bytes):,} bytes, meas_bytes: {len(meas_bytes):,} bytes")
                except _queue.Empty:
                    logger.debug("Writer: Result queue timeout, checking stop event")
                    continue

                if name == "__worker_done__":
                    finished_workers += 1
                    logger.info(f"Writer: Worker finished ({finished_workers}/{num_workers} completed)")
                    continue

                logger.info(f"Writer: Processing results for image '{name}'")
                status_group = imageset._hdf.get_image_status_subgroup(handle=writer, image_name=name)
                logger.debug(f"Writer: Got status group for '{name}'")

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
                        logger.debug(f"Writer: Successfully unpickled measurements for '{name}', shape: {measurement.shape if hasattr(measurement, 'shape') else 'unknown'}")
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
                        if name in image_group:
                            logger.debug(f"Writer: Deleting existing image group for '{name}'")
                            del image_group[name]
                        processed_img._save_image2hdfgroup(grp=image_group, compression="gzip", compression_opts=4)
                        processed_images += 1
                        logger.info(f"Writer: Successfully saved processed image for '{name}' ({processed_images} total)")
                    except Exception as save_error:
                        logger.error(f"Writer: Failed to save processed image for '{name}': {save_error}")
                        status_group.attrs[SET_STATUS.ERROR.label] = True
                        errors += 1
                        continue

                # Save measurements if available
                if measurement is not None:
                    try:
                        logger.debug(f"Writer: Saving measurements for '{name}' to HDF5")
                        # Get or create the image group for this specific image
                        if name in image_group:
                            img_group = image_group[name]
                            logger.debug(f"Writer: Found existing image group for '{name}'")
                        else:
                            # This shouldn't happen if processed_img was saved, but handle it gracefully
                            logger.warning(f"Writer: Image group for {name} not found when saving measurements. Creating new group.")
                            warnings.warn(f"Image group for {name} not found when saving measurements. Creating new group.")
                            img_group = image_group.create_group(name)

                        # Use the static method from SetMeasurementAccessor to save DataFrame
                        from .._image_set_parts._image_set_accessors._image_set_measurements_accessor import SetMeasurementAccessor
                        from phenotypic.util.constants_ import IO
                        logger.debug(f"Writer: Calling SetMeasurementAccessor to save measurements for '{name}'")
                        SetMeasurementAccessor._save_dataframe_to_hdf5_group(df=measurement, group=img_group,
                                                                             measurement_key=IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY)
                        saved_measurements += 1
                        logger.info(f"Writer: Successfully saved measurements for '{name}' ({saved_measurements} total)")
                    except Exception as save_error:
                        logger.error(f"Writer: Failed to save measurements for '{name}': {save_error}")
                        status_group.attrs[SET_STATUS.ERROR.label] = True
                        errors += 1
                        continue

                logger.debug(f"Writer: Flushing HDF5 file after processing '{name}'")
                writer.flush()
                logger.debug(f"Writer: Completed processing '{name}'")
        
        logger.info(f"Writer finished - Processed: {processed_images} images, {saved_measurements} measurements, {errors} errors")

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

        with imageset._hdf.reader() as reader:
            image_group = reader[str(imageset._hdf.set_images_posix)]

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
