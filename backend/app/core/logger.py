import os
import logging
import sys
from typing import Optional


def setup_logger(name: str = "wasserstoff", level: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger with the specified name and level.
    
    Args:
        name: The name of the logger
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
              If None, it will use the LOG_LEVEL environment variable or default to INFO
    
    Returns:
        A configured logger instance
    """
    # Get log level from environment or use default
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    numeric_level = getattr(logging, level, logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create console handler if not already added
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    
    return logger