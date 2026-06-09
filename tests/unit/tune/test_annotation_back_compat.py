"""Wave-0 contract: the back-compat regression lock for the ``Field``-second stage.

A corpus of ``pipeline.json`` fixtures (``tests/fixtures/tune/
back_compat_pipelines/``) — one per soon-to-be-``Field``-bounded op carrying a
**legal-today** boundary value, plus realistic multi-op pipelines — must keep
loading via ``ImagePipeline.from_json`` **without** a ``ValidationError`` after a
validity bound is migrated from a ``field_validator`` to ``Field(ge=, le=)``.

This is the guardrail behind the staged rollout: adding a ``Field`` bound is the
one move that can change validation behaviour. Each fixture's value sits at the
tightest legal-today edge (``n_iter=1``, ``min_size=1``, ``sigma_onf=0.1``/``1.0``,
``cutoff=0.5``, ...), so a too-tight bound that excluded a previously-valid config
would trip this lock immediately.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic import ImagePipeline

_CORPUS_DIR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "tune"
    / "back_compat_pipelines"
)

_FIXTURES = sorted(_CORPUS_DIR.glob("*.json"))


def test_corpus_is_non_empty():
    """Guard against a silently-empty corpus (which would pass vacuously)."""
    assert _FIXTURES, f"no back-compat fixtures found under {_CORPUS_DIR}"


@pytest.mark.parametrize("fixture", _FIXTURES, ids=lambda p: p.stem)
def test_legacy_pipeline_json_still_loads(fixture: Path):
    """Each serialized pipeline deserializes cleanly (no ``ValidationError``)."""
    pipe = ImagePipeline.from_json(fixture)
    # A successful load yields a usable pipeline with at least one op.
    assert len(pipe.get_ops()) >= 1
