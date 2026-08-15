Getting Started
===============

Prerequisites
-------------

Before installing Phenotypic, ensure you have the following prerequisites:

* Python 3.10 or higher
* pip (Python package installer)
* uv (recommended; see https://docs.astral.sh/uv/)

Installation Methods
------------------

From PyPi
+++++++++

Using uv (recommended)
+++++++++
.. code-block:: bash

   uv add phenotypic

Using pip
+++++++++

.. code-block:: bash

    pip install phenotypic

From Source
-----------

To install from source:


.. code-block:: bash

  git clone https://github.com/exfab/PhenoTypic.git
  cd PhenoTypic && uv sync


Optional Extras
---------------

PhenoTypic provides optional extras for different use cases:

- ``[gui]`` — Browser-based GUI hub: Plotly dashboards, Dash apps, and Jupyter
  integration. Does **not** include napari.
- ``[napari]`` — The interactive napari desktop viewers (pulls napari + PyQt6).
  Required for ``image.rgb.napari()`` and related viewer methods, the point
  picker, and the napari sweep viewer (``python -m phenotypic.gui.sweep``).
- ``[torch]`` — PyTorch + SAM2 for ``Sam2`` (Linux/macOS only).

.. code-block:: bash

   # Browser GUI hub (Plotly, Dash, Jupyter)
   uv add "phenotypic[gui]"

   # napari desktop viewers
   uv add "phenotypic[napari]"

   # SAM2 GPU detector (Linux/macOS)
   uv add "phenotypic[torch]"

``micro_sam`` (used by ``MicroSamDetector``) is only published on
conda-forge and is **not** included in any PhenoTypic extra. See
`Deep Learning Detectors`_ below for a self-service recipe that combines
PhenoTypic and ``micro_sam`` in a single ``pixi`` environment.

Optional libvips acceleration
+++++++++++++++++++++++++++++

On macOS and Windows, the ``[gui]`` extra installs the official
`pyvips binary distribution <https://pypi.org/project/pyvips/>`_, including a
self-contained native libvips. No separate system installation or loader-path
configuration is needed for the normal GUI installation.

Linux and HPC installations use the smaller ``pyvips`` binding. Provide native
`libvips <https://www.libvips.org/install.html>`_ through the operating system
or a site environment module when the faster DZI path is wanted:

.. code-block:: bash

   # Debian or Ubuntu
   sudo apt install libvips-dev --no-install-recommends

On a cluster, load the site's libvips module before launching the GUI when one
is available.

Verify that the native library is visible from the same shell that will launch
PhenoTypic:

.. code-block:: bash

   uv run python -c "import pyvips; print('pyvips', pyvips.__version__, 'libvips', '.'.join(str(pyvips.version(i)) for i in range(3)))"

The bundled build includes common image loaders but omits optional PDF and
OpenSlide support. Users who intentionally prefer a fuller Homebrew libvips can
install it with ``brew install vips``. If that system install is present but the
dynamic loader cannot find ``libvips.42.dylib``, set the fallback library path
before running either the verification command or the GUI:

.. code-block:: bash

   export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix vips)/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"

If ``pyvips`` cannot load its native library, PhenoTypic automatically uses its
Pillow implementation. All viewer features remain available, although large
DZI pyramids can take longer to prepare and require more memory.


Deep Learning Detectors
-----------------------

PhenoTypic ships several deep-learning colony detectors —
``Sam2``, ``Sam3``, ``DinoSam2Detector``, ``Insid3Detector``,
``FssDinoDetector``, and the conda-only ``MicroSamDetector``. This section
covers **installing** the detector packages and **pre-downloading** their model
weights. For detector usage, parameter tuning, device selection, and SLURM
staging, see the :doc:`GPU Detection Setup </how_to/pages/gpu_detection_setup>`
how-to guide.

Installing the detector packages
++++++++++++++++++++++++++++++++

The PyPI-published detectors are grouped into optional extras:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Extra
     - Detectors
     - Notes
   * - ``[torch]``
     - ``Sam2``
     - PyTorch + ``sam2`` from PyPI. Linux/macOS only — no Windows wheels
       (use WSL2).
   * - ``[foundation]``
     - ``Sam3``, ``DinoSam2Detector``, ``Insid3Detector``,
       ``FssDinoDetector``
     - ``transformers`` (>=5.2.0) + ``huggingface_hub`` (pulls ``torch``).
   * - ``[gpu]``
     - all of the above
     - Umbrella extra — every GPU detector at once (recommended for a dev env).

.. code-block:: bash

   # SAM2 only
   uv add "phenotypic[torch]"

   # SAM3 + DINO-based foundation detectors
   uv add "phenotypic[foundation]"

   # Everything, inside a uv-managed checkout
   uv sync --extra gpu

``MicroSamDetector`` depends on ``micro_sam``, which is published **only on
conda-forge** and so is not part of any PhenoTypic extra. Manage the combined
stack yourself — for example with `pixi <https://pixi.sh>`_, which resolves
conda-forge and PyPI in one lockfile, or with conda:

.. code-block:: bash

   pip install phenotypic                   # or phenotypic[torch] on non-Windows
   conda install -c conda-forge micro_sam   # adds MicroSamDetector

Model weights and licenses
++++++++++++++++++++++++++

PhenoTypic **does not redistribute model weights** — each is downloaded from its
upstream source under that model's license. Most weights are ungated (SAM2,
DINOv2), but **SAM3 and DINOv3 weights are gated** and need a one-time license
handshake on the Hugging Face Hub:

1. Create a Hugging Face account.
2. Accept the gate on the model page (this *is* the license acceptance):
   SAM3 → https://huggingface.co/facebook/sam3, DINOv3 →
   https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m.
3. Authenticate locally once: ``uv run hf auth login`` (or export ``HF_TOKEN``).

Downloading and caching checkpoints
+++++++++++++++++++++++++++++++++++

Every detector downloads its weights automatically on first use, but you can
pre-download them with the ``phenotypic.detect.nn`` CLI — essential on SLURM
clusters whose compute nodes have no internet access:

.. code-block:: bash

   # SAM2 (default tiny; --all for every size)
   uv run python -m phenotypic.detect.nn download --model-type sam2 --model-size tiny

   # micro-sam
   uv run python -m phenotypic.detect.nn download --model-type microsam --model-name vit_b_lm

   # Gated foundation weights — --accept-license acknowledges the model license
   uv run python -m phenotypic.detect.nn download --model-type sam3 --accept-license
   uv run python -m phenotypic.detect.nn download --model-type dinov3 --dino-size base --accept-license

   # Ungated DINOv2 (no token needed)
   uv run python -m phenotypic.detect.nn download --model-type dinov2 --dino-size base

   # Inspect or clear the cache
   uv run python -m phenotypic.detect.nn list
   uv run python -m phenotypic.detect.nn clear --model-type sam2

Cache locations are controlled by environment variables — point them at shared
storage so SLURM compute nodes can read pre-staged weights:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Variable
     - Controls
   * - ``TORCH_HOME``
     - SAM2 checkpoint cache (default ``~/.cache/torch/hub/checkpoints/``).
   * - ``MICROSAM_CACHEDIR``
     - micro-sam checkpoint cache.
   * - ``HF_HOME`` / ``HF_HUB_CACHE``
     - Hugging Face cache for SAM3 + DINO weights.
   * - ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1``
     - Force load-from-cache on offline compute nodes.
   * - ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``
     - Auth for gated downloads (alternative to ``hf auth login``).
   * - ``PHENOTYPIC_ACCEPT_MODEL_LICENSE``
     - Non-interactive license acknowledgement for batch/SLURM jobs
       (e.g. ``sam3,dinov3``).

.. note::

   On a cluster, accept the gates and pre-stage weights **on the login node**
   (which has internet), then run the compute job offline with the same
   ``HF_HOME`` plus ``HF_HUB_OFFLINE=1`` and
   ``PHENOTYPIC_ACCEPT_MODEL_LICENSE``. Acceptance never happens silently
   inside a batch job. See the
   :doc:`GPU Detection Setup </how_to/pages/gpu_detection_setup>` guide for the
   full SLURM staging workflow.


Development Installation
========================

For development of new modules, sync with the ``dev`` (and optionally
``docs``) dependency groups:

.. code-block:: bash

    git clone https://github.com/exfab/PhenoTypic.git
    cd PhenoTypic
    uv sync --group dev --group docs


Verification
------------

To verify the installation, run:

.. code-block:: python

   import phenotypic
   print(phenotypic.__version__)


Launching the GUI
-----------------

The unified GUI hub bundles the pipeline builder, results viewer, and
run console under one URL. Two equivalent entry points:

.. code-block:: bash

   # Console script (preferred)
   uv run phenotypic-gui --root ./images --port 8050

   # Module entry (works in environments without the console script on PATH)
   uv run python -m phenotypic.gui --root ./images --port 8050

``--root`` freezes the sandbox the GUI's file browser is allowed to see
(defaults to the current working directory). ``--host 127.0.0.1`` (the
default) keeps the server loopback-only — pair with SSH port forwarding
for remote workstations:

.. code-block:: bash

   ssh -L 8050:localhost:8050 user@cluster

Then open ``http://localhost:8050/`` in your browser. If you launch the
GUI from inside a Slurm allocation (``srun``/``salloc``), the server
binds to the compute node rather than the login node, and the
single-hop tunnel above will return ``connect failed: Connection
refused``. See
:ref:`Running the GUI on a Slurm compute node <gui-hub-slurm-tunnel>`
in the GUI hub guide for the two-hop tunnel pattern. That guide also
covers the file browser, pipeline builder, run console, and results
viewer.

For Open OnDemand-style proxies, pass the browser path as a prefix, not
the full URL:

.. code-block:: bash

   uv run phenotypic-gui --root /rhome/ejaco020 --host 0.0.0.0 --port 30099 --url-prefix /node/hz01/30099/

Then open the full proxy URL, for example
``https://ondemand.hpcc.ucr.edu/node/hz01/30099/``.

.. note::

   ``phenotypic gui`` (no hyphen, as a subcommand) is **not supported**.
   Use ``phenotypic-gui`` or ``python -m phenotypic.gui``. The existing
   ``phenotypic`` CLI is reserved for batch pipeline execution with explicit
   path options, not subcommands.
