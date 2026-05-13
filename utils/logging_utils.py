import logging
import os
from datetime import datetime

_current_log_file: str = ""


def setup_logging(log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    global _current_log_file
    logger = logging.getLogger("ImageLabelingStudio")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        _current_log_file = os.path.join(
            log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(_current_log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_log_file() -> str:
    """Return the path to the current session log file."""
    return _current_log_file


def get_log_dir() -> str:
    """Return the log directory (parent of current log file)."""
    return os.path.dirname(_current_log_file) if _current_log_file else ""


def get_logger(name: str = "ImageLabelingStudio") -> logging.Logger:
    return logging.getLogger(name)
