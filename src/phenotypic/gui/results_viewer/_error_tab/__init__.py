"""Error-analysis tab for the results viewer.

For the focused error category, this tab runs
:class:`phenotypic.analysis.ErrorCutoffFinder` against a good baseline
(all-unlabeled or verified-only) and surfaces a ranked cutoff table, a
good-vs-error distribution figure with a draggable cutoff line, a live
recall/specificity readout, and a copy-able filter spec — recomputed as
the user marks objects on the other tabs.

The public callbacks + layout factory are wired in Task 6; for now this
package exposes only its pure, Dash-free data and figure layers so the
unit tests can exercise the load-bearing logic without booting an app.
"""
from __future__ import annotations
