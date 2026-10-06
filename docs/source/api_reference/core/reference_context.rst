ReferenceContext
================

.. currentmodule:: phenotypic

Overview
--------

A :class:`ReferenceContext` holds one experiment metadata table and the means to
resolve the reference images it names. Activating it with ``with ctx:`` makes it
visible to every :class:`~phenotypic.abc_.RefMetadata` operation that runs inside
the block, such as :class:`~phenotypic.enhance.SubtractBlank`, at any nesting depth.

For the task-oriented view, see :doc:`/how_to/pages/reference_metadata`.

.. autoclass:: ReferenceContext
   :members: lookup, has_column, resolve_image, load_image, reference_image_digest, narrow, current, table, columns, table_sha256

Errors
------

All are ``ValueError`` subclasses of :py:exc:`~phenotypic.sdk_.ReferenceContextError`
and are importable from :mod:`phenotypic.sdk_`; their pages are generated with that
module.

- :py:exc:`~phenotypic.sdk_.ReferenceContextError`: base class of the rest.
- :py:exc:`~phenotypic.sdk_.RefMetadataUnavailableError`: an operation ran with
  no active context.
- :py:exc:`~phenotypic.sdk_.ReferenceTableError`: the table is missing,
  unreadable, or lacks a needed column.
- :py:exc:`~phenotypic.sdk_.ReferenceLookupError`: an image has no row, or its
  rows are empty, disagree, or name the image itself.
- :py:exc:`~phenotypic.sdk_.ReferenceImageError`: a reference image cannot be
  resolved, read, or matched to its target.
- :py:exc:`~phenotypic.sdk_.StaleDetectMatError`: ``SubtractBlank`` ran on a
  ``detect_mat`` or image its raw blank does not match.
