import logging


def configure_logging(level=logging.DEBUG, logfile=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s DeepSdf - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)
