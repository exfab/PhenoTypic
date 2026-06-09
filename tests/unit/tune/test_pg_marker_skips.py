"""The ``@pytest.mark.postgres`` autoskip — gated on ``PHENOTYPIC_TEST_PG_URL``.

Postgres-backed study tests (a later chunk) need a live server. The marker +
``tests/conftest.py`` autoskip means a marked test is silently skipped unless
``PHENOTYPIC_TEST_PG_URL`` is set, so the default suite stays green without a
database.
"""
from __future__ import annotations

import os

import pytest


@pytest.mark.postgres
def test_postgres_marked_test_is_skipped_without_env(request):
    # If the env var is unset, the conftest autoskip must have prevented us from
    # ever running. Reaching the body means the gate failed to skip.
    if not os.environ.get("PHENOTYPIC_TEST_PG_URL"):
        pytest.fail("postgres-marked test ran despite PHENOTYPIC_TEST_PG_URL unset")


def test_postgres_marker_is_registered(request):
    # The marker must be declared so --strict-markers does not error on it.
    markers = request.config.getini("markers")
    assert any(m.startswith("postgres:") for m in markers)
