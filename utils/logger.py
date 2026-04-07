"""utils.logger

Project logger configuration utilities.
"""

from __future__ import annotations

import logging
import os


class ContextFormatter(logging.Formatter):
    """Formatter that keeps objective/method context visible in every line."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "objective"):
            record.objective = "-"
        if not hasattr(record, "method"):
            record.method = "-"
        return super().format(record)


def setup_logger(
    name: str = "pso_logger",
    log_dir: str = "logs",
    log_file: str = "pso.log",
) -> logging.Logger:
    """
    Create and configure a logger for the project.

    The logger writes only to file, not to console, so terminal output
    can stay clean and be managed separately.

    Parameters
    ----------
    name : str, default="pso_logger"
        Logger name.
    log_dir : str, default="logs"
        Directory where the log file will be stored.
    log_file : str, default="pso.log"
        Log filename.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    full_log_path = os.path.abspath(os.path.join(log_dir, log_file))

    if logger.handlers:
        existing_paths = {
            os.path.abspath(getattr(handler, "baseFilename", ""))
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        }
        if existing_paths == {full_log_path}:
            return logger

        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    os.makedirs(log_dir, exist_ok=True)

    formatter = ContextFormatter(
        "%(asctime)s | %(levelname)-5s | %(objective)s | %(method)s | %(message)s"
    )

    file_handler = logging.FileHandler(full_log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
