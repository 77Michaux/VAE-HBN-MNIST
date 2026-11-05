import logging
import os
from pathlib import Path


def setup_logging(main_name: str = "script", mode: str = "w"):
    """
    Creates a logging folder and sets up a parent logger that can log other scripts into one single shared log.
    :return: parent logger that will share all children loggers in other scripts.
    """
    log_dir = Path("logs/")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = Path(log_dir / f"{main_name}.log")

    parent_logger = logging.getLogger(f"{main_name}")
    parent_logger.setLevel(logging.INFO)

    # Prevent messages from propagating to the root logger if it has handlers
    parent_logger.propagate = False
    # Create a FileHandler
    file_handler = logging.FileHandler(log_file, mode=mode)
    # Create a Formatter that includes the logger's name.
    # Crucial for identifying the source of the log.
    formatter = logging.Formatter("%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s")
    file_handler.setFormatter(formatter)
    # Add the handler to the logger, but only if it doesn't have one already
    if not parent_logger.handlers:
        parent_logger.addHandler(file_handler)

    return parent_logger
