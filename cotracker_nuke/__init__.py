#!/usr/bin/env python3
"""
CoTracker Nuke Integration Package
==================================

A modular package for integrating CoTracker point tracking with Nuke VFX workflows.

Main Components:
- core: Core tracking, video processing, and application logic
- exporters: Nuke export functionality
- ui: Gradio web interface
- cli: Command-line interface
- utils: Logging and utility functions
"""

from .core.app import CoTrackerNukeApp
from .ui.gradio_interface import create_gradio_interface
from .utils.logger import setup_logger

__version__ = "1.0.0"
__author__ = "AI Assistant"
__license__ = "MIT"

__all__ = [
    "CoTrackerNukeApp",
    "create_gradio_interface", 
    "setup_logger"
]
