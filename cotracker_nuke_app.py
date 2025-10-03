#!/usr/bin/env python3
"""
CoTracker Nuke Integration App - Refactored
===========================================

This is the new modular version of the CoTracker Nuke integration app.
The original monolithic file has been refactored into a clean modular structure.

Features:
- Modular architecture with separate concerns
- Proper logging with configurable verbosity levels
- Clean separation of UI, core logic, and exporters
- Support for both GUI and CLI interfaces

Usage:
    python cotracker_nuke_app_new.py [--log-level DEBUG|INFO|WARNING|ERROR]

Author: AI Assistant (under human supervision)
License: MIT
"""

import os
import multiprocessing as mp

import argparse
import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cotracker_nuke import create_gradio_interface


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="CoTracker Nuke Integration - Modular Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is the new modular version with improved architecture:

🏗️  Modular Structure:
  - core/: Core tracking and video processing logic
  - exporters/: Nuke export functionality  
  - ui/: Gradio web interface
  - cli/: Command-line interface
  - utils/: Logging and utility functions

📊 Logging Levels:
  - DEBUG: Detailed debugging information
  - INFO: General information (default)
  - WARNING: Warning messages only
  - ERROR: Error messages only

🚀 Usage:
  python cotracker_nuke_app_new.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug file logging"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="CoTracker Nuke Integration v1.0.0 (Modular)"
    )
    
    args = parser.parse_args()
    
    try:
        print("CoTracker Nuke Integration - Modular Version")
        print(f"Console log level: {args.log_level}")
        print(f"Debug logging: {'Disabled' if args.no_debug else 'Enabled'}")
        print()
        
        # Create and launch Gradio interface
        debug_mode = not args.no_debug
        interface = create_gradio_interface(debug_mode, args.log_level)
        
        # Launch the interface
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True
        )
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
