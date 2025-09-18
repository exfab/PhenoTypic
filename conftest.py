# conftest.py
import logging

def pytest_configure(config):
    # Enable specific loggers
    logging.getLogger('ImagePipeline').setLevel(logging.DEBUG)
    logging.getLogger('ImagePipeline.coordinator').setLevel(logging.DEBUG)
    logging.getLogger('ImagePipeline.parallel').setLevel(logging.DEBUG)
    logging.getLogger('ImagePipeline.producer').setLevel(logging.DEBUG)
    logging.getLogger('ImagePipeline.writer').setLevel(logging.DEBUG)
    logging.getLogger('ImagePipeline.worker').setLevel(logging.DEBUG)
    logging.getLogger("ImageSet.get_measurement").setLevel(logging.DEBUG)