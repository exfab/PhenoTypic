"""Compatibility path for the experimental-tag owners.

The owners moved to ``phenotypic.schema._metadata._experimental_tags`` in the
2026-09 schema reorganization. Some callers imported this private path directly
to reach the one-release transition aliases (the previous-release owner class
names), so it keeps resolving them, with the same ``DeprecationWarning``, for as
long as those aliases exist. Delete this module together with them.

New code imports the owners from ``phenotypic.schema``.
"""

from ._metadata._experimental_tags import (  # noqa: F401 - re-exported
    ACQUISITION,
    CONDITION,
    CULTURE,
    EXPERIMENT,
    GENETIC,
    PLATE,
    SAMPLE,
    STUDY,
    __all__,
    __getattr__,
)
