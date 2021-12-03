import os
import sys
import time
import logging


def setup_logger(name, output_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')

    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setFormatter(formatter)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    if output_dir is not None:
        file_handler = logging.FileHandler(os.path.join(output_dir, "log"))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger
