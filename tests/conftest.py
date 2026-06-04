"""Top-level test configuration.

Ensures that calling ``.show()`` on plotly or matplotlib figures during tests
does not spawn browser tabs or GUI windows, and autoskips ``@pytest.mark.postgres``
tests unless a live database URL is configured.
"""

import os

import matplotlib
import pytest

matplotlib.use("Agg")

try:
    import plotly.io as pio

    pio.renderers.default = "json"
except ImportError:
    pass


#: Env var carrying a live Postgres URL for the tune study-DB tests (a later
#: chunk). When unset, every ``@pytest.mark.postgres`` test is skipped so the
#: default suite needs no database.
PG_URL_ENV = "PHENOTYPIC_TEST_PG_URL"


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.postgres`` tests when ``PHENOTYPIC_TEST_PG_URL`` is unset.

    Args:
        config: The pytest config (unused; required by the hook signature).
        items: The collected test items, mutated in place with a skip marker.
    """
    if os.environ.get(PG_URL_ENV):
        return
    skip_pg = pytest.mark.skip(reason=f"requires a Postgres server via ${PG_URL_ENV}")
    for item in items:
        if "postgres" in item.keywords:
            item.add_marker(skip_pg)
