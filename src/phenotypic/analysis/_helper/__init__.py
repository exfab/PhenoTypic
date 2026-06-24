"""Private helper modules for the analysis package.

Pure utilities with no public operation class: robust-statistics math
(:mod:`._qc_math`), error-report rendering (:mod:`._error_report`), and the
softplus inoculum-prior helper (:mod:`._inoculum_prior`). The public
error-report functions are re-exported here so :mod:`phenotypic.analysis`
can surface them from a single private home.
"""

from ._error_report import (
    filter_spec_json,
    filter_spec_query,
    render_error_analysis_html,
    render_error_analysis_report,
)

__all__ = [
    "filter_spec_json",
    "filter_spec_query",
    "render_error_analysis_html",
    "render_error_analysis_report",
]
