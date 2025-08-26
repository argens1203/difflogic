import logging
import os

from experiment.args import ExperimentArgs

LOG_FILE_PATH = "main.log"
INFO_LOG_FILE_PATH = "main.log.i"
default_formatter = logging.Formatter(
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    "%(message)s"
)

console_handler = None
file_handler = None
info_file_handler = None


def setup_file_logger_and_remove_log(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    return logging.FileHandler(file_path)


def get_log_level(args: ExperimentArgs):
    if args.verbose == "debug":
        return logging.DEBUG
    elif args.verbose == "info":
        return logging.INFO
    elif args.verbose == "warn":
        return logging.WARN
    return logging.INFO


def setup_logger(args: ExperimentArgs):
    global console_handler, file_handler, info_file_handler

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if console_handler is not None:
        logger.removeHandler(console_handler)
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(default_formatter)
    console_handler.setLevel(get_log_level(args))
    logger.addHandler(console_handler)

    if file_handler is not None:
        logger.removeHandler(file_handler)
    # Add detailed file log
    file_handler = setup_file_logger_and_remove_log(LOG_FILE_PATH)
    file_handler.setFormatter(default_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    if info_file_handler is not None:
        logger.removeHandler(info_file_handler)
    # Add info file log
    info_file_handler = setup_file_logger_and_remove_log(INFO_LOG_FILE_PATH)
    info_file_handler.setFormatter(default_formatter)
    info_file_handler.setLevel(logging.INFO)
    logger.addHandler(info_file_handler)
