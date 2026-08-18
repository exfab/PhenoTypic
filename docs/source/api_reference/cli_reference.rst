CLI Reference
=============

PhenoTypic provides a command-line interface for batch processing plate images.

For a guided walkthrough of the execution modes and the parameters worth
knowing, see the :doc:`CLI Execution Modes tutorial </tutorials/pages/cli_modes>`.

Usage
-----

.. code-block:: bash

   python -m phenotypic --mode full --pipeline PIPELINE_JSON --input INPUT_DIR --output OUTPUT_DIR [OPTIONS]

Path Options
------------

``-p, --pipeline PIPELINE_JSON``
   Path to a pipeline configuration file created with ``pipeline.to_json()``.
   Required for ``full``, ``measure``, and ``process`` modes. Rejected in
   ``recompile`` mode, which reloads the saved pipeline from the output root.

``-i, --input INPUT_DIR``
   Directory containing plate images to process. Required for ``full`` and
   ``process`` modes. Rejected in ``measure`` and ``recompile`` modes, which
   discover prior outputs from the output root.

``-o, --output OUTPUT_DIR``
   Directory where results (measurements, overlays under ``deliverables/overlays/<dataset>/``, checkpoints) are saved.
   Required for every mode.

Image Options
-------------

``--image-type {Image,GridImage}``
   Image class to use for loading. Case-insensitive. Default: ``GridImage``.

``--nrows N``
   Number of grid rows (GridImage only). Overrides any pipeline-level preset;
   when omitted, the pipeline preset is used, falling back to 8.

``--ncols N``
   Number of grid columns (GridImage only). Overrides any pipeline-level preset;
   when omitted, the pipeline preset is used, falling back to 12.

``--bit-depth {8,16}``
   Force bit depth. Default: auto-detect.

``--detect-mode MODE``
   Detection matrix source channel. One of ``gray``, ``red``, ``green``,
   ``blue``, ``MinRGB``, ``HsvS``, ``HsvV``, ``InvS``, ``LabL``, ``LabA``,
   ``LabB``. Default: ``gray``.

``--ext EXT``
   Deprecated for HDF output. Forward runs write a single ``.h5`` per image;
   only overlay PNG rendering still consults this value. Default: ``tiff``.

Execution Options
-----------------

``-m, --mode {full,measure,recompile,process}``
   Select the execution mode. Default: ``full``.

   ``full``
      Apply the pipeline, measure, and emit all deliverables.

   ``measure``
      Re-run measurements from HDF files in an existing output root. Requires
      ``--pipeline`` and ``--output``. Rejects ``--input``, ``--dry-run``,
      ``--resume``, ``--restart``, ``--retry-failures``, ``--overwrite``, and
      ``--sample``.

   ``recompile``
      Rebuild aggregate deliverables from an existing output root. Requires
      ``--output``. Rejects ``--pipeline``, ``--input``, and ``--dry-run``.

   ``process``
      Apply the pipeline and export one image layer per input, mirroring the
      input tree. Requires ``--pipeline``, ``--input``, ``--output``, and
      ``--layer``. Warns that ``--metadata``, ``--no-qc``, and
      ``--no-dataset-column`` are ignored.

``--layer {rgb,gray,detect_mat,objmap}``
   Image layer exported by ``--mode process``; required there and rejected in
   other modes. ``rgb`` writes an integer TIFF at the source bit depth,
   ``gray``/``detect_mat`` a float TIFF, ``objmap`` a 16-bit raw-label PNG.

``--njobs N``
   Number of parallel jobs for local execution. Default: ``-1`` (all cores).

``--force-local``
   Run locally even if SLURM is available.

``--dry-run``
   Validate pipeline and list images without processing. Supported by
   ``full`` and ``process`` modes; rejected by ``measure`` and ``recompile``.

``--sample N``
   Process only N random images per dataset (for testing).

``--random-seed SEED``
   Random seed for ``--sample`` reproducibility.

Resume and Recovery
-------------------

``--resume``
   Continue from a previous run. Staged GPU runs select Stage 1, 2, or 3 from
   valid HDF, sidecar, and terminal-marker artifacts and automatically include
   intermediate-stage failures.

``--retry-failures``
   Include recorded CPU or legacy single-pass failures in addition to unfinished
   images. Staged GPU failures are already included by ``--resume``. Requires
   ``--resume``.

``--restart``
   Clear all state and start fresh. Mutually exclusive with ``--resume``.

``--overwrite``
   Reprocess all images. Mutually exclusive with ``--resume``.

``--checkpoint-interval N``
   Insert checkpoint tasks every N images in SLURM arrays. Default:
   auto-estimate.

SLURM Options
-------------

``--slurm KEY=VALUE``
   Pass SLURM scheduling parameters as repeated key-value pairs (e.g.,
   ``--slurm slurm_partition=compute --slurm mem_gb=16 --slurm time=120``).
   Use the ``slurm_`` prefix for standard SBATCH directives, or the convenience
   keys ``mem_gb`` and ``time``. ``time`` is an **integer number of minutes**.
   The deprecated ``time_min`` key is auto-migrated to ``time``.

``--wait``
   Wait for SLURM jobs and their dependent finalizer to complete. For staged GPU
   runs, success requires the finalizer completion marker. Without this flag,
   the CLI prints ``PROCESSING SUBMITTED`` and returns without running local
   aggregation or claiming completion. Ctrl+C detaches monitoring without
   cancelling the active epoch.

GPU Staging Options
-------------------

A pipeline containing a ``GpuDetector`` runs as three stages (CPU preprocess →
resident-model GPU detect → CPU measure). These flags tune Stage 2.

On SLURM, an epoch-fenced dependent controller submits additional Stage-2
arrays while sidecar-less retryable images remain. It replaces worker
signal/self-requeue behavior. Dynamic controller, array, Stage-3, and finalizer
job IDs are recorded in the run ledger for monitoring and cancellation.

``--gpu-slurm KEY=VALUE``
   Stage-2 SBATCH resources. Inherits and deltas over ``--slurm`` (the CPU
   profile for Stages 1 and 3); auto-adds ``slurm_gpus_per_node=1``.

``--gpu-shards N``
   Parallel Stage-2 GPU tasks, one whole GPU each. SLURM-only; ignored locally.
   Default: 1.

``--gpu-workers-per-gpu W``
   Reserved for future per-GPU replica packing. The current staged worker runs
   one resident model per GPU shard. Default: 1.

Output Options
--------------

``--overlay-alpha FLOAT``
   Alpha transparency of the label overlay (0.0-1.0). Default: 0.3.

``--no-dataset-column``
   Exclude the ``Metadata_Dataset`` column from
   ``master_measurements.csv``. The column is included by default.

``--metadata PATH``
   CSV file to left-join onto the measurements mirror on shared columns. Every
   CSV row survives: one that matches no measured object is kept with null
   measurements and ``QC_MetadataOnly`` set to ``true``, so strains that were
   never detected stay visible instead of being silently dropped. Measurement
   rows with no matching CSV row *are* dropped. The join lands on
   ``deliverables/measurements.csv`` and its derivatives —
   ``master_measurements.csv`` stays a clean, metadata-free archive.
   Metadata headers are normalized in memory to ``Metadata_<Label>``. Recompile
   never mutates this external file; its regenerated bundle-owned metadata copy
   is canonical.

``--study PATH``
   Optional ``study.yaml`` of REMBI Study-level fields (Title, License, Author,
   ...) folded into ``deliverables/rembi.yaml``; overrides constant
   ``Metadata_*`` columns. Applies to ``full`` and ``measure`` modes only —
   it is silently ignored by ``--mode recompile``.

``--no-qc``
   Skip the QC compute step in finalize. QC otherwise runs whenever the pipeline
   has a non-empty ``qc`` section, writing the ``qc/`` artifact and resetting
   GUI review progress.

``--skip-validation``
   Skip pipeline validation before processing.

Tuning CLI
==========

PhenoTypic also ships a hyperparameter-tuning engine, ``python -m phenotypic.tune``,
which searches an ``ImagePipeline``'s parameters to maximize a scorer. It has two
subcommands: ``run`` (the search engine) and ``auto-space`` (infer a reviewable search
space from a pipeline). The ``tpe``, ``cmaes``, ``gp``, and ``nsga2`` strategies require
the optional ``tune`` extra (``uv sync --extras tune``); ``grid`` and ``random`` work out
of the box.

See the :doc:`tuning how-to </how_to/pages/tuning>` for an end-to-end walkthrough and
the :doc:`distributed HPCC guide </how_to/pages/tune_distributed_hpcc>` for SLURM and
Postgres fan-out.

``run`` — run a tuning spec
---------------------------

.. code-block:: bash

   python -m phenotypic.tune run SPEC_JSON -i INPUT_DIR [OPTIONS]

Path Options
~~~~~~~~~~~~

``SPEC_JSON``
   Positional. Path to a ``tuning_spec.json`` describing the search space, scorer,
   and budget.

``-i, --input INPUT_DIR``
   Directory of plate images to tune against. Required.

``-o, --output OUTPUT_DIR``
   Directory where tuning results are written.

Search Options
~~~~~~~~~~~~~~

``--strategy {grid,random,tpe,cmaes,gp,nsga2}``
   Override the spec's strategy. ``grid``/``random`` use the built-in configs;
   ``tpe``/``cmaes``/``gp``/``nsga2`` build an ``OptunaConfig`` (needs the ``tune``
   extra).

``--n-trials N``
   Trial-budget override.

``--screen`` / ``--no-screen``
   Enable or disable the two-round screening freeze. Default: ``--no-screen``.

Storage Options
~~~~~~~~~~~~~~~

``--storage-url URL``
   Optuna storage URL: ``sqlite:///…`` (local single node) or a password-less
   ``postgresql+psycopg://USER@HOST:PORT/DB`` (distributed; libpq reads the password
   from ``~/.pgpass`` or ``$PGPASSWORD``, so it never enters argv or the worker script).
   Falls back to ``$PHENOTYPIC_TUNE_STORAGE_URL``.

Distributed Options
~~~~~~~~~~~~~~~~~~~

``--slurm``
   Submit a distributed worker fleet over SLURM instead of running locally.

``--n-workers N``
   Number of SLURM array workers in the fleet (``--slurm`` only). When unset,
   defaults to ``min(8, n_trials)`` (or 4 if no trial budget is known). The
   fleet shares the one ``--n-trials`` budget rather than multiplying it.

``--slurm-partition NAME``
   SLURM partition for the worker fleet (``--slurm`` only). When unset the
   ``#SBATCH --partition`` directive is omitted (cluster default).

``--slurm-mem MEM``
   SLURM ``--mem`` for each worker (``--slurm`` only), e.g. ``8G``.

``--slurm-time HMS``
   SLURM ``--time`` wall-clock limit for each worker (``--slurm`` only), e.g.
   ``04:00:00``.

Robust-Eval Options
~~~~~~~~~~~~~~~~~~~

``--held-out-fraction F``
   Override the spec's held-out fraction: the target share of plates reserved for the
   generalization pass.

``--cv-group COL``
   Override the held-out grouping column. When unset, the spec value (then the count
   scorer's ``groupby[0]``) is inferred.

``auto-space`` — infer a search space
-------------------------------------

.. code-block:: bash

   python -m phenotypic.tune auto-space PIPELINE_JSON [OPTIONS]

``PIPELINE_JSON``
   Positional. Path to a pipeline JSON created with ``pipeline.to_json()``.

``-o, --output OUTPUT_DIR``
   Directory where the inferred search space is written.

``--unattended``
   Reserved: skip the interactive review prompt (currently a no-op).
