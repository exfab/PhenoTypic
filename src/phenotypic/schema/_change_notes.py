"""Release notes rendered above measurement tables by ``MeasurementInfo.change_note``.

One constant per public-column change. Each is RST, rendered verbatim in the
measurer's class docs, the enum's API page, and the Measurements reference
page. Never copy these into ``Entry.desc``: descs are published into every
run's README, so a release note there would ship with every future run.
"""

import inspect


def append_change_note(doc: str | None, note: str) -> str:
    """Append a change note to an enum's class docstring.

    The docstring is dedented first. On Python < 3.13 a class docstring keeps
    its source indentation, and a note appended at column 0 sets the common
    margin to 0, so Sphinx would strip nothing and render the body as a
    block quote.

    Args:
        doc: The class's ``__doc__``.
        note: RST from the enum's ``change_note()``.

    Returns:
        str: The dedented docstring, a blank line, then *note*.
    """
    return f"{inspect.cleandoc(doc or '')}\n\n{note}"


SIZE_SHAPE_SPLIT_NOTE = """\
.. versionchanged:: 0.20.0
   Colony size magnitudes moved from :class:`~phenotypic.schema.SHAPE` to
   :class:`~phenotypic.schema.SIZE`: :class:`~phenotypic.measure.MeasureSize`
   is now their only source, and :class:`~phenotypic.measure.MeasureShape`
   emits form descriptors only. The radius columns were rebuilt so each name
   matches its value. Retired columns are not aliased, and stores written by
   earlier versions keep their old column names. A run started before 0.20.0
   must be re-run with ``--overwrite``, not resumed. Resuming reuses the
   images it already finished, with their old column names.

   ==============================  ================================
   Retired column                  Successor
   ==============================  ================================
   ``Shape_Area``                  ``Size_Area``
   ``Shape_Perimeter``             ``Size_Perimeter``
   ``Shape_ConvexArea``            ``Size_ConvexArea``
   ``Shape_BboxArea``              ``Size_BboxArea``
   ``Shape_MajorAxisLength``       ``Size_MajorAxisLength``
   ``Shape_MinorAxisLength``       ``Size_MinorAxisLength``
   ``Shape_MinFeretDiameter``      ``Size_MinFeretDiameter``
   ``Shape_MaxFeretDiameter``      ``Size_MaxFeretDiameter``
   ``Shape_MaxRadius``             ``Size_InscribedRadius``
   ``Shape_MeanRadius``            ``Shape_MeanBoundaryDist``
   ``Shape_MedianRadius``          ``Shape_MedianBoundaryDist``
   ==============================  ================================

   New columns: ``Size_MedianRadius``, ``Size_MeanRadius``,
   ``Size_RobustMeanRadius`` and ``Size_MaxRadius``, all measured from one
   center, the centroid of the distance-transform peak.

   **Same name, different value:** ``Size_MedianRadius``, ``Size_MeanRadius``
   and ``Size_MaxRadius`` are *not* the retired ``Shape_MedianRadius``,
   ``Shape_MeanRadius`` and ``Shape_MaxRadius``. Compare old data against the
   successor in the table above (identical except for colonies touching
   another colony or the image border, which now measure to that edge),
   never against the same-named ``Size_`` column.
   Model outputs are named by the stripped label, so ``<Model>_MaxRadius_*``,
   ``<Model>_MeanRadius_*`` and ``<Model>_MedianRadius_*`` fitted before and
   after 0.20.0 share a name but not a meaning.
"""
