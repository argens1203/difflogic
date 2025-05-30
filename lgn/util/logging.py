import logging
import os

from constant import DEBUG

LOG_FILE_PATH = "main.log"
INFO_LOG_FILE_PATH = "main.log.i"
default_formatter = logging.Formatter(
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    "%(message)s"
)


def setup_file_logger_and_remove_log(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    return logging.FileHandler(file_path)


def setup_logger(args=None):
    if not args.verbose:
        return
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(default_formatter)

    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.addHandler(console_handler)

    file_handler = setup_file_logger_and_remove_log(LOG_FILE_PATH)
    file_handler.setFormatter(default_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    if DEBUG:
        info_file_handler = setup_file_logger_and_remove_log(INFO_LOG_FILE_PATH)
        info_file_handler.setFormatter(default_formatter)
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
