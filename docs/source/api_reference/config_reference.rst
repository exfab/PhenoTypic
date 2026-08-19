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

   from phenotypic.schema import IMAGE
   from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS

   GAMMA_ENCODINGS.SRGB     # Standard sRGB gamma encoding
   GAMMA_ENCODINGS.LINEAR   # Linear RGB (no gamma)
   IMAGE.IMAGE_NAME         # Metadata_ImageName

Metadata Schema and Ownership
-----------------------------

The public metadata enums are ``IMAGE``, ``GENETIC``, ``SAMPLE``, ``PLATE``,
``CONDITION``, ``CULTURE``, ``EXPERIMENT``, ``STUDY``, and ``ACQUISITION``.
Every enum inherits ``MetadataInfo`` and emits the shared
``Metadata_<Label>`` namespace. Use owner lookup instead of parsing a header:

.. code-block:: python

   from phenotypic.schema import GENETIC
   from phenotypic.sdk_ import metadata_owner_for_header

   assert metadata_owner_for_header("Metadata_Strain") is GENETIC

``metadata_member_for_header`` and ``metadata_owner_for_header`` accept bare,
canonical, and exact historical spellings. Label-oriented equivalents are also
public. ``normalize_metadata_columns`` returns a normalized pandas or Polars
copy and rejects conflicting duplicate aliases. Stored historical headers are
readable permanently; the previous Python enum names warn for one transition
release and are not exported by ``phenotypic.schema.__all__``.

Metadata Migration
------------------

Use ``preflight_metadata_schema`` before an explicit standalone file or bundle
migration. Pass its optimistic fingerprint to the matching mutation API:

.. code-block:: python

   from phenotypic.sdk_ import (
       migrate_metadata_bundle,
       migrate_metadata_file,
       preflight_metadata_schema,
       rollback_metadata_migration,
   )

   file_report = preflight_metadata_schema("metadata.csv")
   file_result = migrate_metadata_file(
       "metadata.csv",
       expected_source_fingerprint=file_report.source_fingerprint,
   )

   bundle_report = preflight_metadata_schema("out")
   bundle_result = migrate_metadata_bundle(
       "out",
       expected_plan_fingerprint=bundle_report.plan_fingerprint,
   )

   if bundle_result.receipt_path:
       rollback_metadata_migration(bundle_result.receipt_path)

Migration is fingerprint-gated and journaled with prepared/applied receipts.
HDF changes use a validated sibling copy and do not change the independent HDF
layout version. Local and SLURM recompiles invoke bundle migration before
aggregation. External ``--metadata`` files are normalized in memory and remain
byte-for-byte unchanged; only bundle-owned regenerated outputs are canonicalized.

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
