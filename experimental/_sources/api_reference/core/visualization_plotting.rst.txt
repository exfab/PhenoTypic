Plotting capabilities
=====================

.. currentmodule:: phenotypic.abc_.plotting

Pipeline plots use fieldless capability mixins. They can be added to existing
Pydantic operation or analysis classes without changing those classes' constructors
or serialized settings.

Lifecycle mixins
----------------

.. autoclass:: PhtPlot
   :members: inspect, report, iter_figures, figures

.. autoclass:: PlotImage

.. autoclass:: PlotMeas

.. autoclass:: PlotAnalysis

.. autoclass:: PlotQc

Figure composition
------------------

.. autoclass:: Control

.. autoclass:: FigureSpec

.. autofunction:: figure

.. autoclass:: BoundFigures
   :members:

Built-in plots
--------------

.. currentmodule:: phenotypic.plotting

.. autoclass:: PlotDiagnostics
   :members: inspect, report

.. autoclass:: PlotDetectModes
   :members: inspect, report

.. autoclass:: PlotMeasTimeSeries
   :members: inspect, report

.. autoclass:: PlotColonyMetricOverTime
   :members: inspect, report

.. autoclass:: AnalysisInput

.. autoclass:: MeasurementInput

.. autoclass:: PlotOutput

.. autoclass:: PlotPage
