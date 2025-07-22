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

import pandas as pd
import pickle
import warnings
import psutil
import h5py


from .._image_set import ImageSet
from phenotypic.abstract import Measurements
from ._image_pipeline_core import ImagePipelineCore


class ImagePipelineBatch(ImagePipelineCore):
    """Run an `ImagePipeline` on many images concurrently."""

    def __init__(self,
                 ops: List[ImageOperation] | Dict[str, ImageOperation] | None = None,
                 measurements: List[MeasureFeatures] | Dict[str, MeasureFeatures] | None = None,
                 num_workers: int = -1,
                 verbose: bool = True,
                 ):
        super().__init__(ops, measurements)
        self.num_workers = num_workers
        self.verbose = verbose

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
        # Wait for writer to open the file first
        writer_ready.wait()

        image_names: List[str] = imageset.get_image_names()

        # Try to open with SWMR, fallback to regular read-only if not supported
        try:
            reader = h5py.File(imageset._out_path, "r", libver="latest", swmr=True)
        except (RuntimeError, ValueError):
            reader = h5py.File(imageset._out_path, "r", libver="latest")

        with reader:
            img_group = imageset._get_hdf5_group(reader, imageset._hdf5_image_group_key)
            for name in image_names:
                if stop_event.is_set():
                    break

                # Estimate size on disk – used as proxy for RAM requirement.
                size_bytes = self._estimate_hdf5_dataset_size(img_group[name])
                # Wait until enough free RAM is available.
                while psutil.virtual_memory().available < size_bytes * 1.25:
                    if stop_event.is_set():
                        return
                    time.sleep(0.5)

                work_q.put(name)

        # Signal the end of work – one *None* sentinel per worker.
        sentinels = num_workers
        for _ in range(sentinels):
            work_q.put(None)  # type: ignore[arg-type]

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
        while not stop_event.is_set():
            try:
                name = work_q.get(timeout=1)
            except _queue.Empty:
                continue

            if name is None:
                # Sentinel received – terminate.
                break

            try:
                image: Image = imageset.get_image(name)
                processed_img = None
                measurement = None
                if mode == "apply":
                    processed_img = super(ImagePipelineBatch, self).apply(image, inplace=False, reset=True)
                elif mode == "measure":
                    measurement = super(ImagePipelineBatch, self).measure(image)
                else:  # apply_and_measure
                    processed_img, measurement = super(ImagePipelineBatch, self).apply_and_measure(image, inplace=False, reset=True)
                result_q.put((
                    name,
                    pickle.dumps(processed_img) if processed_img is not None else b"",
                    pickle.dumps(measurement) if measurement is not None else b""
                ))
            except Exception as exc:
                # Forward the exception details to the writer to decide.
                result_q.put((name, pickle.dumps(exc), b""))

        # Indicate this worker is done.
        result_q.put(("__worker_done__", b"", b""))

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
        finished_workers = 0

        # Open file in append mode to avoid conflicts with existing readers
        # Create with proper version for SWMR compatibility
        with h5py.File(imageset._out_path, "a", libver="latest") as writer:
            set_group = imageset._get_hdf5_group(writer, str(imageset._hdf5_image_group_key))
            # Only enable SWMR if file version supports it
            try:
                writer.swmr_mode = True  # enable SWMR once file is opened for writing
            except RuntimeError:
                # File version doesn't support SWMR, continue without it
                pass

            # Signal that writer is ready and file is open for SWMR
            writer_ready.set()

            while finished_workers < num_workers and not stop_event.is_set():
                try:
                    name, img_bytes, meas_bytes = result_q.get(timeout=1)
                except _queue.Empty:
                    continue

                if name == "__worker_done__":
                    finished_workers += 1
                    continue

                # Process exceptions first
                if img_bytes:
                    try:
                        maybe_exc = pickle.loads(img_bytes)
                        if isinstance(maybe_exc, Exception):
                            warnings.warn(f"Worker failed processing {name}: {maybe_exc}")
                            continue
                    except Exception as unpickle_error:
                        warnings.warn(f"Worker failed processing {name}: Could not unpickle exception - {unpickle_error}")
                        continue

                processed_img = pickle.loads(img_bytes) if img_bytes else None
                measurement = pickle.loads(meas_bytes) if meas_bytes else None

                if processed_img is not None:
                    if name in set_group:
                        del set_group[name]
                    processed_img._save_image2hdf5(grp=set_group, compression="gzip", compression_opts=4)

                if measurement is not None:
                    # Store measurement data using utility function
                    meas_group_name = f"{imageset._hdf5_image_group_key}/{name}"
                    if meas_group_name in writer:
                        meas_group = writer[meas_group_name]
                    else:
                        meas_group = writer.create_group(meas_group_name)

                    # Use utility function to save DataFrame with preserved data types
                    Measurements._save_dataframe_to_hdf5_group(measurement, meas_group)

                writer.flush()

    # .................................................................
    def _aggregate_measurements_from_hdf5(self, imageset: ImageSet) -> pd.DataFrame:
        """Aggregate measurements by reading directly from HDF5 file after processing completes.
        
        Args:
            imageset: The ImageSet instance containing the HDF5 file path
            
        Returns:
            Aggregated pandas DataFrame containing all measurements
        """
        measurements_list = []
        
        with h5py.File(imageset._out_path, "r", libver="latest", swmr=True) as reader:
            image_group = reader[str(imageset._hdf5_image_group_key)]
            
            for image_name in image_group.keys():
                image_subgroup = image_group[image_name]
                
                # Use utility function to load DataFrame with preserved data types
                df = Measurements._load_dataframe_from_hdf5_group(image_subgroup)
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
