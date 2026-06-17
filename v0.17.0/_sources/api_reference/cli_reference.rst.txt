CLI Reference
=============

PhenoTypic provides a command-line interface for batch processing plate images.

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
   Directory where results (overlays, measurements, checkpoints) are saved.
   Required for every mode.

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

``-m, --mode {full,measure,recompile,process}``
   Select the execution mode. Default: ``full``.

   ``full``
      Apply the pipeline, measure, and emit all deliverables.

   ``measure``
      Re-run measurements from HDF files in an existing output root. Requires
      ``--pipeline`` and ``--output``. Rejects ``--input`` and ``--dry-run``.

   ``recompile``
      Rebuild aggregate deliverables from an existing output root. Requires
      ``--output``. Rejects ``--pipeline``, ``--input``, and ``--dry-run``.

   ``process``
      Apply the pipeline and export one image layer per input. Requires
      ``--pipeline``, ``--input``, ``--output``, and ``--layer``.

``--layer {rgb,gray,detect_mat,objmap}``
   Image layer exported by ``--mode process``. Rejected in other modes.

``--njobs N``
   Number of parallel worker processes. Default: all available CPUs.

``--force-local``
   Run locally even if SLURM is available.

``--dry-run``
   Validate pipeline and list images without processing. Supported by
   ``full`` and ``process`` modes; rejected by ``measure`` and ``recompile``.

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

``--slurm KEY=VALUE``
   Pass SLURM scheduling parameters as repeated key-value pairs (e.g.,
   ``--slurm time=04:00:00 --slurm partition=gpu``).

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
