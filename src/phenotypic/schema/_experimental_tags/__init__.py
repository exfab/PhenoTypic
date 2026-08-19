"""Standardized biological / experimental metadata-tag vocabulary.

Eight ``MetadataInfo`` subclasses group recommended metadata tags for arrayed
colony phenotyping. Every member renders in the shared ``Metadata_<Label>``
namespace; the concrete owner retains the semantic topic for schema queries and
the grouping gives users canonical names, descriptions, and auto-generated
documentation tables that drop straight into the ``--metadata`` CSV join and the
``post/`` metadata operations.

This is a *recommended vocabulary, not a validator* — arbitrary metadata columns are
still accepted everywhere. Re-exported from :mod:`phenotypic.schema`:

    from phenotypic.schema import SAMPLE, CONDITION
"""

import sys
import warnings

from ._acquisition import ACQUISITION
from ._condition import CONDITION
from ._culture import CULTURE
from ._experiment import EXPERIMENT
from ._genetic import GENETIC
from ._plate import PLATE
from ._sample import SAMPLE
from ._study import STUDY

__all__ = [
    "ACQUISITION",
    "CONDITION",
    "CULTURE",
    "EXPERIMENT",
    "GENETIC",
    "PLATE",
    "SAMPLE",
    "STUDY",
]

_LEGACY_NAMES = {
    "ACQUISITION_METADATA": ACQUISITION,
    "CONDITION_METADATA": CONDITION,
    "CULTURE_METADATA": CULTURE,
    "EXPERIMENT_METADATA": EXPERIMENT,
    "GENETIC_METADATA": GENETIC,
    "PLATE_METADATA": PLATE,
    "SAMPLE_METADATA": SAMPLE,
    "STUDY_METADATA": STUDY,
}


def __getattr__(name: str):
    """Resolve one-release compatibility names for direct package imports."""
    value = _LEGACY_NAMES.get(name)
    if value is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    caller = sys._getframe(1)
    is_fromlist_probe = (
        caller.f_code.co_name == "_handle_fromlist"
        and caller.f_globals.get("__name__") == "importlib._bootstrap"
    )
    if not is_fromlist_probe:
        warnings.warn(
            f"{name} is deprecated; use {value.__name__} instead",
            DeprecationWarning,
            stacklevel=2,
        )
    return value
