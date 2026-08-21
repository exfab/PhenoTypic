OME-Zarr store layout
=====================

.. currentmodule:: phenotypic.sdk_.ngff_

Everything about the on-disk shape of a per-image store lives in
``phenotypic.sdk_.ngff_``: the directory layout, the pyramid geometry, the
chunk/shard/codec policy, the ``attributes.phenotypic`` contract, the
write-only OME projection, and the rename-promote commit primitive.

Nothing in this module reads or writes an :class:`~phenotypic.Image`. Keeping
the geometry free of the image model is what lets the committed
logic-validation script re-derive every numeric claim from numpy alone.

For the user-facing view of the same thing — what a store looks like, how to
open one in napari or QuPath, and the CLI flags that govern it — see
:doc:`/how_to/pages/zarr_storage`.

Layout constants
----------------

.. autodata:: STORE_SUFFIX
.. autodata:: STORE_ROOT_JSON
.. autodata:: STORE_SCHEMA_VERSION
.. autodata:: NGFF_VERSION
.. autodata:: BIOFORMATS2RAW_LAYOUT
.. autodata:: SERIES_ORDER
.. autodata:: OBJMAP_LABEL
.. autodata:: PYRAMID_STOP_PX

Pyramid geometry
----------------

Pyramid depth is a pure function of the level-0 shape — there is no user lever,
which is what makes mixed-geometry drift within one output tree unreachable
rather than merely unlikely.

.. autofunction:: pyramid_level_count
.. autofunction:: pyramid_level_shapes
.. autofunction:: level_scale_vector
.. autofunction:: downsample_image
.. autofunction:: downsample_label
.. autofunction:: build_pyramid

Reading a store
---------------

.. autofunction:: require_readable_store
.. autofunction:: read_phenotypic_attributes
.. autofunction:: store_level0_shape
.. autofunction:: primary_series
.. autofunction:: objmap_path
.. autofunction:: valid_staged_store

The commit protocol
-------------------

A store is never written in place. Each publisher builds a ``.part`` sibling,
writes arrays and chunks first and the root ``zarr.json`` **last**, then
promotes the directory by rename. An interrupted write therefore leaves no
valid root and reads as absent rather than partial.

.. warning::

   **Nothing may write into a promoted store.** Both the per-image completion
   marker and the results viewer's staleness scan identify a store by its root
   ``zarr.json`` alone, which is sound only because the promote writes that
   root last and replaces the directory wholesale. A code path that opens a
   promoted store for writing makes both report stale data as fresh, and
   neither can detect it. The guard is
   ``tests/unit/sdk_/test_ngff_promote.py::test_nothing_writes_into_a_promoted_store``.

.. autofunction:: new_part_path
.. autofunction:: promote_store
.. autofunction:: discard_parts_for
.. autofunction:: sweep_orphan_parts

Durability
----------

``fsync`` before promote is a tri-state: unset auto-detects (on under SLURM,
off locally), and ``--durable-writes`` / ``--no-durable-writes`` overrides it.
The resolution happens in exactly one place so the flag and the sentence
describing it cannot drift.

.. autofunction:: durable_writes_enabled
.. autofunction:: describe_durability
.. autofunction:: fsync_tree
