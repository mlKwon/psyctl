"""Simple logging setup."""
import logging
import sys
from pathlib import Path

from psyctl import config


def setup_logging():
    """Setup basic logging."""
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    # Console handler
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # File handler for DEBUG
    if config.LOG_LEVEL.upper() == 'DEBUG':
        debug_file = config.OUTPUT_DIR / 'debug.log'
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(debug_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)

    # Optional log file
    if config.LOG_FILE:
        config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name or __name__)
