Configuration Reference
=======================

PhenoTypic exposes process-wide runtime switches through the
``phenotypic.settings`` module. These settings are intentionally narrow:
algorithm defaults and pipeline parameters are serialized on the operation
models themselves, not stored in global settings.

.. code-block:: python

   import phenotypic.settings as settings

   settings.set_validate_ops(True)

   with settings.validation(False):
       ...  # temporarily disable operation integrity validation

Global Settings
---------------

Settings are accessed via the ``phenotypic.settings`` module.

**Validation**

- ``VALIDATE_OPS`` — Enables operation and measurement integrity checks in the
  current Python process. Defaults to ``False``.
- ``set_validate_ops(enabled)`` — Sets ``VALIDATE_OPS`` explicitly.
- ``validation(enabled)`` — Context manager for temporary validation changes.

The legacy ``phenotypic.settings_`` import path has been removed.

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
         "class": "BlurGauss",
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
