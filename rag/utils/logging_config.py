# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Logging Configuration

Provides consistent logging setup across all RAG pipeline modules.
Replaces print statements with proper structured logging.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}
_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT
) -> None:
    """
    Configure logging for the RAG pipeline.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Log message format
        date_format: Date format for timestamps
    """
    global _initialized
    
    if _initialized:
        return
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Configure root logger for our package
    root_logger = logging.getLogger("rag")
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        
    Returns:
        Configured logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
        logger.error("An error occurred", exc_info=True)
    """
    # Ensure logging is set up
    if not _initialized:
        setup_logging()
    
    # Prefix with 'rag.' if not already
    if not name.startswith("rag."):
        name = f"rag.{name}"
    
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    
    return _loggers[name]


class LogContext:
    """
    Context manager for logging with indentation/context.
    
    Usage:
        logger = get_logger(__name__)
        with LogContext(logger, "Building index"):
            # logs will show context
            logger.info("Step 1...")
    """
    
    def __init__(self, logger: logging.Logger, context: str) -> None:
        self.logger = logger
        self.context = context
    
    def __enter__(self) -> "LogContext":
        self.logger.info(f"{'='*60}")
        self.logger.info(f"{self.context}")
        self.logger.info(f"{'='*60}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            self.logger.error(f"{self.context} failed: {exc_val}")
        else:
            self.logger.info(f"{self.context} completed")


def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    message: str = "Progress",
    interval: int = 25
) -> None:
    """
    Log progress at regular intervals.
    
    Args:
        logger: Logger instance
        current: Current item number
        total: Total items
        message: Progress message prefix
        interval: Log every N items
    """
    if current % interval == 0 or current == total:
        percentage = (100 * current) // total
        logger.info(f"{message}: {current}/{total} ({percentage}%)")
