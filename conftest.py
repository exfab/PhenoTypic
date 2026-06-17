# conftest.py
import importlib.util
import logging

# The shared fixtures plugin (tests/unit/test_fixtures.py) imports numpy and
# walks the entire ``phenotypic`` package at import time, and ``pytest_configure``
# below imports ``phenotypic.settings_``. Both require the project and its full
# dependency stack to be installed. Guard on that so dependency-light pytest
# invocations -- notably the packaging-integrity CI job, which runs the
# build-artifact tests in a bare ``pytest``-only env (``uv run --no-project
# --with pytest``) -- can still collect and run conftest-free tests instead of
# crashing with ``ModuleNotFoundError: No module named 'numpy'`` during initial
# conftest loading. In every real test environment ``phenotypic`` is installed,
# so this is a no-op.
_PHENOTYPIC_AVAILABLE = importlib.util.find_spec("phenotypic") is not None

# Share test fixtures defined in tests/unit/test_fixtures.py across the suite.
if _PHENOTYPIC_AVAILABLE:
    pytest_plugins = ["tests.unit.test_fixtures"]


def pytest_configure(config):
    if not _PHENOTYPIC_AVAILABLE:
        return

    import phenotypic.settings_

    phenotypic.settings_.VALIDATE_OPS = True

    # Enable specific loggers
    logging.getLogger("ImagePipeline").setLevel(logging.DEBUG)
    logging.getLogger("ImagePipeline.coordinator").setLevel(logging.DEBUG)
    logging.getLogger("ImagePipeline.parallel").setLevel(logging.DEBUG)
    logging.getLogger("ImagePipeline.producer").setLevel(logging.DEBUG)
    logging.getLogger("ImagePipeline.writer").setLevel(logging.DEBUG)
    logging.getLogger("ImagePipeline.worker").setLevel(logging.DEBUG)
    logging.getLogger("ImageSet.get_measurement").setLevel(logging.DEBUG)
