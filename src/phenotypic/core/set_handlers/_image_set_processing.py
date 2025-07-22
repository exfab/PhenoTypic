from ._image_set_core import ImageSetCore
from __future__ import annotations

# ── std‑lib & third‑party ─────────────────────────────────────────────
import multiprocessing as mp
from multiprocessing import Queue, Event
from threading import Thread
import time
import pickle
import queue as _queue
from pathlib import Path
import warnings
import os

import numpy as np
import pandas as pd
import h5py
import psutil
from phenotypic import ImagePipeline

# Not in use in favor of deferring the processing to ImagePipeline
class ImageSetProcessing(ImageSetCore):
    """Parallel processing and measurement of the images contained in an :class:`ImageSet`.

    This class builds on :class:`~phenotypic.core.set_handlers._image_set_core.ImageSetCore` and
    adds a *process* method which will:

    1. Dispatch image names to a configurable pool of worker *processes*.
    2. Let each worker open the HDF5 file in **read-only SWMR** mode, load the image, run
       the provided ``ImagePipelineCore`` and return the processed `Image` object together with the
       measurement ``DataFrame``.
    3. A dedicated **writer thread** (single writer) consumes the results queue and writes the
       processed images and measurements back into the HDF5 file which is opened in **write
       SWMR** mode.  Using a single writer avoids the *single-writer* limitation of the HDF5
       SWMR specification while still allowing many concurrent readers.

    Memory-awareness: before a worker loads an image it checks the dataset's size inside the
    HDF5 file.  Only if the machine has *available* RAM > `ram_ratio` × image_size will it
    proceed, otherwise the task is returned to the queue and the worker sleeps briefly.  This
    ensures that very large images do not get loaded in parallel by multiple workers and
    overwhelm the node's memory (important on shared HPC nodes).

    The implementation is completely platform-independent – it relies on Python's
    :pymod:`multiprocessing` with the *spawn* start-method which works on Linux, macOS and
    Windows as well as within typical HPC batch systems.
    """

    # -------------------- public API --------------------
    def process(
        self,
        pipeline: ImagePipelineCore,
        *,
        num_workers: int | None = None,
        ram_ratio: float = 1.25,
        queue_maxsize: int = 0,
        progress: bool = True,
    ) -> None:
        """Run *pipeline* on every image in the set in parallel.

        Parameters
        ----------
        pipeline
            The :class:`~phenotypic.core._image_pipeline.ImagePipelineCore` to apply.
        num_workers
            Number of worker *processes* to spawn.  Defaults to ``cpu_count() - 1`` (leaving one
            CPU for the writer / OS).
        ram_ratio
            A worker only loads an image if
            ``psutil.virtual_memory().available > ram_ratio * image_size``.
        queue_maxsize
            Maximum task / result queue length.  ``0`` means *unlimited*.
        progress
            If *True* a tiny textual progress will be printed – useful when running interactively
            but harmless on HPC clusters where stdout is captured.
        """
        if num_workers is None or num_workers < 1:
            num_workers = max(mp.cpu_count() - 1, 1)

        # Expose paths for child processes via environment variables
        os.environ["PHENOTYPIC_IMAGESET_PATH"] = str(self._out_path)
        os.environ["PHENOTYPIC_IMAGE_GROUP_KEY"] = str(self._hdf5_image_group_key)

        # Serialize the pipeline once so it can be sent to every worker cheaply.
        try:
            pipeline_bytes = pickle.dumps(pipeline)
        except Exception as e:  # pragma: no cover – pipeline should be picklable but be safe.
            raise ValueError(
                "`pipeline` must be picklable in order to be broadcast to the worker processes."  # noqa: E501
            ) from e

        # ---- multiprocessing primitives ----------------------------------------------------
        ctx = mp.get_context("spawn")  # portable across OSes / HPC clusters
        task_q: Queue = ctx.Queue(maxsize=queue_maxsize)
        result_q: Queue = ctx.Queue(maxsize=queue_maxsize)
        stop_event: Event = ctx.Event()

        # Enqueue the image names *before* starting the workers so they can start immediately.
        for name in self.get_image_names():
            task_q.put(name)
        # Sentinel for graceful shutdown – one per worker so everyone exits.
        #  Using None as sentinel value.
        for _ in range(num_workers):
            task_q.put(None)

        # ---- writer thread -----------------------------------------------------------------
        writer_thread = Thread(
            target=self._writer,
            args=(result_q, stop_event, ram_ratio),
            daemon=True,
        )
        writer_thread.start()

        # ---- worker processes --------------------------------------------------------------
        workers: list[mp.Process] = []
        for wid in range(num_workers):
            p = ctx.Process(
                name=f"ImageWorker-{wid}",
                target=self._worker,
                args=(
                    pipeline_bytes,
                    task_q,
                    result_q,
                    stop_event,
                    ram_ratio,
                ),
            )
            p.start()
            workers.append(p)

        if progress:
            print(
                f"Started {num_workers} worker processes + writer thread to process "
                f"{task_q.qsize()} images …"
            )

        # Wait for all workers to finish.
        for p in workers:
            p.join()

        # Tell the writer we are done and wait for it.
        stop_event.set()
        writer_thread.join()

        if progress:
            print("Parallel processing completed.")

    # -------------------- worker & writer helpers --------------------
    @staticmethod
    def _worker(
        pipeline_bytes: bytes,
        task_q: Queue,
        result_q: Queue,
        stop_event: Event,
        ram_ratio: float,
    ) -> None:
        """Worker function executed in a *child* process."""
        # Re-instantiate the pipeline inside the subprocess.
        pipeline: ImagePipelineCore = pickle.loads(pipeline_bytes)

        # Import heavy libs *after* fork/spawn – avoids unnecessary memory use on Windows.
        import h5py  # noqa: WPS433 – local import inside subprocess
        from phenotypic import Image  # noqa: WPS433
        import os

        # The first argument to `pickle.loads` is bytes, so the pipeline is now ready.
        # Because we cannot access `self` in a staticmethod, we pass required paths via env.
        # We rely on the fact that the *task* items are just image names so we can rebuild
        # HDF5 access lazily here.
        set_path = Path(os.environ["PHENOTYPIC_IMAGESET_PATH"])
        image_group_key = os.environ["PHENOTYPIC_IMAGE_GROUP_KEY"]

        # Open the HDF5 file in **read-only SWMR** mode once per worker.
        reader = h5py.File(set_path, mode="r", libver="latest", swmr=True)
        image_group = reader[image_group_key]

        while True:
            if stop_event.is_set():
                break
            try:
                image_name = task_q.get(timeout=0.5)
            except _queue.Empty:
                continue

            if image_name is None:  # sentinel -> graceful shutdown
                break

            # Estimate dataset size before loading the image.
            if image_name not in image_group:
                warnings.warn(f"Image {image_name} not found – skipping.")
                continue

            ds = image_group[image_name]
            # Rough estimate – bytes on disk ≈ nbytes, may slightly underestimate RAM usage.
            image_size = ds[...].nbytes if hasattr(ds, "nbytes") else ds.size * ds.dtype.itemsize

            # Wait for RAM if needed – push task back so another worker can try later.
            if psutil.virtual_memory().available < ram_ratio * image_size:
                task_q.put(image_name)  # re-queue for later
                time.sleep(1.0)
                continue

            # Load, process, measure.
            img: Image = Image()._load_from_hdf5_group(ds)  # type: ignore[attr-defined]
            processed_img, measurement_df = pipeline.apply_and_measure(img)

            # Put results – the writer will serialise again.
            result_q.put(
                (
                    image_name,
                    pickle.dumps(processed_img, protocol=pickle.HIGHEST_PROTOCOL),
                    pickle.dumps(measurement_df, protocol=pickle.HIGHEST_PROTOCOL),
                )
            )

        reader.close()

    def _writer(self, result_q: Queue, stop_event: Event, ram_ratio: float) -> None:
        """Single writer thread – writes images *and* measurement tables to HDF5 with SWMR."""
        # Thread runs in the *main* process so we can access self easily.
        with h5py.File(self._out_path, mode="r+", libver="latest") as writer:
            writer.swmr_mode = True  # enable SWMR on the writer
            set_group = self._get_hdf5_group(writer, self._hdf5_set_group_key)
            img_group = self._get_hdf5_group(writer, self._hdf5_image_group_key)

            # Separate group for measurements
            meas_group = self._get_hdf5_group(writer, self._hdf5_set_group_key / "measurements")

            while not (stop_event.is_set() and result_q.empty()):
                try:
                    image_name, img_bytes, meas_bytes = result_q.get(timeout=0.5)
                except _queue.Empty:
                    continue

                img = pickle.loads(img_bytes)
                df: pd.DataFrame = pickle.loads(meas_bytes)

                # Overwrite existing datasets if present.
                if image_name in img_group:
                    del img_group[image_name]
                img._save_image2hdf5(grp=img_group, compression="gzip", compression_opts=4)

                meas_data = df.to_numpy()
                if image_name in meas_group:
                    del meas_group[image_name]
                meas_group.create_dataset(
                    name=image_name,
                    data=meas_data,
                    compression="gzip",
                    compression_opts=4,
                    dtype=meas_data.dtype,
                )

                # Flush so readers (if any) can see the new data immediately.
                writer.flush()
