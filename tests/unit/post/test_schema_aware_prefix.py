"""Tests for schema-aware metadata prefixing (Task B4).

``ensure_metadata_prefix`` lives in ``phenotypic.sdk_`` (no post->core import
dependency) and is re-exported from ``phenotypic.post._utils``. Decouple-then-flip
discipline: the category-prefixed cases derive from the **live** enum value so the
assertions hold both while categories return ``"Metadata"`` and after the flip to
``Metadata<Topic>``.
"""

from __future__ import annotations

from phenotypic.schema import EXPERIMENT, GENETIC
from phenotypic.sdk_ import ensure_metadata_prefix


def test_known_label_gets_category_prefix():
    assert ensure_metadata_prefix("Strain") == str(GENETIC.STRAIN)
    assert ensure_metadata_prefix("Dataset") == str(EXPERIMENT.DATASET)


def test_unknown_label_gets_generic_prefix():
    assert ensure_metadata_prefix("MyCustomTag") == "Metadata_MyCustomTag"


def test_already_category_prefixed_passthrough():
    # str(GENETIC.STRAIN) is the live, self-describing header.
    live = str(GENETIC.STRAIN)
    assert ensure_metadata_prefix(live) == live


def test_generic_prefixed_passthrough():
    assert ensure_metadata_prefix("Metadata_MyCustomTag") == "Metadata_MyCustomTag"


def test_post_utils_reexports_the_sdk_function():
    from phenotypic.post import _utils

    assert _utils.ensure_metadata_prefix is ensure_metadata_prefix
