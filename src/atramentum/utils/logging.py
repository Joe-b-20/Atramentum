# src/atramentum/utils/logging.py
"""
Logging utilities. Document the descent.
"""

import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional


def setup_logging(config_path: Optional[str] = None) -> None:
    """Setup logging from config file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        # Default config - because sometimes we can't even configure properly
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance. Name your suffering."""
    return logging.getLogger(name)


# Initialize on import
setup_logging('configs/logging.yaml')