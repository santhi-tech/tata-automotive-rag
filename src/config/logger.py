import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger instance with timestamp.
    Supports INFO, WARNING, ERROR, DEBUG, CRITICAL.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers
        # Capture everything from DEBUG upwards
        logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # Formatter with timestamp
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger
