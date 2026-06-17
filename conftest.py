# conftest.py
import logging
import os

# Share test fixtures defined in tests/test_fixtures.py across the suite.
pytest_plugins = ["tests.unit.test_fixtures"]


def pytest_xdist_auto_num_workers(config):
    """Resolve ``pytest -n auto`` to the cores this process was *allocated*.

    On the HPCC (SLURM cgroup/cpuset) a compute node may expose hundreds of
    cores while the job is granted only a handful. pytest-xdist's default
    ``auto`` counts every physical core and oversubscribes the allocation,
    which thrashes memory and gets the job killed. ``os.sched_getaffinity``
    reflects the scheduler's actual grant. Returning ``None`` falls back to
    xdist's default on platforms without affinity support (macOS/Windows).
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return None


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
