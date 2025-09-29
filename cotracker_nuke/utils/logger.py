#!/usr/bin/env python3
"""
Logging utilities for CoTracker Nuke App
========================================

Provides centralized logging configuration with proper verbosity levels
and clean output formatting.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class CoTrackerLogger:
    """Centralized logger for CoTracker Nuke App with configurable verbosity."""
    
    def __init__(self, name: str = "cotracker", debug_dir: Optional[Path] = None):
        """
        Initialize logger with proper configuration.
        
        Args:
            name: Logger name (default: "cotracker")
            debug_dir: Directory for log files (default: temp/)
        """
        self.name = name
        self.debug_dir = debug_dir or Path("temp")
        self.debug_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup file and console handlers with proper formatting."""
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler (detailed logging)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.debug_dir / f"cotracker_debug_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler (cleaner output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Default to INFO for console
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info("="*60)
        self.logger.info("CoTracker Debug Session Started")
        self.logger.info(f"Debug directory: {self.debug_dir.absolute()}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("="*60)
    
    def set_console_level(self, level: str):
        """
        Set console logging level.
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        
        if level.upper() not in level_map:
            self.logger.warning(f"Invalid log level: {level}. Using INFO.")
            level = 'INFO'
        
        # Update console handler level
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(level_map[level.upper()])
                self.logger.info(f"Console log level set to: {level.upper()}")
                break
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger


def setup_logger(name: str = "cotracker", 
                debug_dir: Optional[Path] = None,
                console_level: str = "INFO") -> logging.Logger:
    """
    Convenience function to setup and get a logger.
    
    Args:
        name: Logger name
        debug_dir: Directory for log files
        console_level: Console logging level
    
    Returns:
        Configured logger instance
    """
    logger_manager = CoTrackerLogger(name, debug_dir)
    logger_manager.set_console_level(console_level)
    return logger_manager.get_logger()


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    logger = setup_logger("test", console_level="DEBUG")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\nTesting different console levels:")
    
    # Test different console levels
    logger_manager = CoTrackerLogger("test2")
    logger2 = logger_manager.get_logger()
    
    for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        print(f"\n--- Console level: {level} ---")
        logger_manager.set_console_level(level)
        
        logger2.debug("Debug message")
        logger2.info("Info message")
        logger2.warning("Warning message")
        logger2.error("Error message")
