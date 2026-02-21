import logging
import os

def setup_logger(log_dir: str = "logs", log_file: str = "pso.log") -> logging.Logger:
    """
    Setup a logger to save PSO and grid search logs.

    Args:
        log_dir (str): Directory to save logs.
        log_file (str): Log file name.

    Returns:
        Logger object.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("PSO")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(fh)

    return logger
