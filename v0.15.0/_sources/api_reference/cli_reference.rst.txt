CLI Reference
=============

PhenoTypic provides a command-line interface for batch processing plate images.

Usage
-----

.. code-block:: bash

   python -m phenotypic PIPELINE_JSON INPUT_DIR OUTPUT_DIR [OPTIONS]

Positional Arguments
--------------------

``PIPELINE_JSON``
   Path to a pipeline configuration file created with ``pipeline.to_json()``.

``INPUT_DIR``
   Directory containing plate images to process.

``OUTPUT_DIR``
   Directory where results (overlays, measurements, checkpoints) are saved.

Image Options
-------------

``--image-type {Image,GridImage}``
   Image class to use for loading. Default: ``Image``.

``--nrows N``
   Number of grid rows (GridImage only). Default: 8.

``--ncols N``
   Number of grid columns (GridImage only). Default: 12.

``--bit-depth {8,16}``
   Force bit depth. Default: auto-detect.

``--detect-mode MODE``
   Detection matrix source channel. Default: ``gray``.

``--ext EXT``
   File extension filter (e.g., ``.png``, ``.jpg``). Default: all supported formats.

Execution Options
-----------------

``--n-jobs N``
   Number of parallel worker processes. Default: all available CPUs.

``--force-local``
   Run locally even if SLURM is available.

``--dry-run``
   Validate pipeline and list images without processing.

``--sample N``
   Process only N random images (for testing).

``--random-seed SEED``
   Random seed for ``--sample`` reproducibility.

Resume and Recovery
-------------------

``--resume``
   Continue from a previous run. Skips already-processed images.

``--retry-failures``
   Re-process only images that failed. Requires ``--resume``.

``--restart``
   Clear all state and start fresh. Mutually exclusive with ``--resume``.

``--overwrite``
   Reprocess all images. Mutually exclusive with ``--resume``.

``--checkpoint-interval N``
   Save checkpoint state every N images. Default: 100.

SLURM Options
-------------

``--slurm-args KEY=VALUE``
   Pass SLURM scheduling parameters (e.g., ``time=04:00:00``, ``partition=gpu``).

``--wait``
   Wait for SLURM jobs to complete before returning.

Output Options
--------------

``--overlay-alpha FLOAT``
   Opacity of detection overlay in saved images. Default: 0.3.

``--include-dataset-column``
   Add a dataset column to measurement CSV output.

``--metadata-csv PATH``
   Path to a CSV file with additional metadata to merge into results.

``--skip-validation``
   Skip pipeline validation before processing.
