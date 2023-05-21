import logging
from typing import Literal


def format(message: str, color: Literal["red", "green", "blue", "yellow", "purple", "cyan"]):
    color_formatting = {
        "red": "\033[91m{message}\033[0m",
        "green": "\033[92m{message}\033[0m",
        "yellow": "\033[93m{message}\033[0m",
        "blue": "\033[94m{message}\033[0m",
        "purple": "\033[95m{message}\033[0m",
        "cyan": "\033[96m{message}\033[0m",
    }

    return color_formatting[color].format(message=message)


def get_logger(name: str, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()

        formatter = logging.Formatter(f"\033[94m{name}\033[0m: %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger
