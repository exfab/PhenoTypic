"""Release notes rendered above measurement tables by ``MeasurementInfo.change_note``.

One constant per public-column change. Each is RST, rendered verbatim in the
measurer's class docs, the enum's API page, and the Measurements reference
page. Never copy these into ``Entry.desc``: descs are published into every
run's README, so a release note there would ship with every future run.
"""

SIZE_SHAPE_SPLIT_NOTE = """\
.. versionchanged:: 0.20.0
   Colony size magnitudes moved from :class:`~phenotypic.schema.SHAPE` to
   :class:`~phenotypic.schema.SIZE`: :class:`~phenotypic.measure.MeasureSize`
   is now their only source, and :class:`~phenotypic.measure.MeasureShape`
   emits form descriptors only. The radius columns were rebuilt so each name
   matches its value. Retired columns are not aliased, and stores written by
   earlier versions keep their old column names.

   ==============================  ================================
   Retired column                  Successor
   ==============================  ================================
   ``Shape_Area``                  ``Size_Area``
   ``Shape_Perimeter``             ``Size_Perimeter``
   ``Shape_ConvexArea``            ``Size_ConvexArea``
   ``Shape_BboxArea``              ``Size_BboxArea``
   ``Shape_MajorAxisLength``       ``Size_MajorAxisLength``
   ``Shape_MinorAxisLength``       ``Size_MinorAxisLength``
   ``Shape_MaxRadius``             ``Size_InscribedRadius``
   ``Shape_MeanRadius``            ``Shape_MeanBoundaryDist``
   ``Shape_MedianRadius``          ``Shape_MedianBoundaryDist``
   ==============================  ================================

   New columns: ``Size_MedianRadius``, ``Size_MeanRadius``,
   ``Size_RobustMeanRadius`` and ``Size_MaxRadius``, all measured from one
   center inside the colony.

   **Same name, different value:** ``Size_MedianRadius``, ``Size_MeanRadius``
   and ``Size_MaxRadius`` are *not* the retired ``Shape_MedianRadius``,
   ``Shape_MeanRadius`` and ``Shape_MaxRadius``. Compare old data against the
   successor in the table above, never against the same-named ``Size_`` column.
"""
