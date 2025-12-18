# conftest.py
import logging

# Share test fixtures defined in tests/test_fixtures.py across the suite.
pytest_plugins = ["tests.test_fixtures"]


def pytest_configure(config):
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
