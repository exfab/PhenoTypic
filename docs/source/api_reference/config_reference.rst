Configuration Reference
=======================

PhenoTypic's behavior can be configured through the ``phenotypic.settings_``
module. Import settings before other PhenoTypic modules when modifying them.

.. code-block:: python

   import phenotypic.settings_
   # Modify settings here, before importing other modules

Global Settings
---------------

Settings are accessed via the ``phenotypic.settings_`` module. Key
configuration options include:

**Image Defaults**

- ``default_bit_depth`` — Default bit depth for new images (8 or 16)
- ``default_gamma`` — Default gamma encoding (``GAMMA_ENCODINGS.SRGB``)
- ``default_illuminant`` — Default illuminant (``"D65"``)

**Processing**

- ``detect_mode`` — Default detection matrix source channel (``"gray"``)
- ``n_jobs`` — Default parallelism for batch operations

**Visualization**

- ``plotly_template`` — Default Plotly template for ``.dash()`` figures
- ``matplotlib_backend`` — Matplotlib backend selection

Constants
---------

Key constants are available in ``phenotypic.sdk_.constants_``:

.. code-block:: python

   from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS, METADATA

   GAMMA_ENCODINGS.SRGB     # Standard sRGB gamma encoding
   GAMMA_ENCODINGS.LINEAR   # Linear RGB (no gamma)

Pipeline JSON Format
--------------------

Pipeline configurations are stored as JSON with the following structure:

.. code-block:: json

   {
     "phenotypic_version": "0.x.y",
     "name": "pipeline_name",
     "description": "...",
     "ops": [
       {
         "class": "GaussianBlur",
         "module": "phenotypic.enhance",
         "params": {"sigma": 2.0, "mode": "reflect"}
       }
     ],
     "meas": [
       {
         "class": "MeasureSize",
         "module": "phenotypic.measure",
         "params": {}
       }
     ]
   }

All operation classes and their parameters are captured. The PhenoTypic
version is recorded to warn about compatibility issues when loading
pipelines saved with a different version.
