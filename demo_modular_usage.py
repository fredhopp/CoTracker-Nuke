#!/usr/bin/env python3
"""
Demo: CoTracker Nuke App - Modular Usage Examples
==================================================

This script demonstrates the new modular capabilities and clean API
of the refactored CoTracker Nuke application.
"""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cotracker_nuke import CoTrackerNukeApp, setup_logger
from cotracker_nuke.core import CoTrackerEngine, VideoProcessor, MaskHandler
from cotracker_nuke.exporters import NukeExporter


def demo_logging_system():
    """Demo: New logging system with configurable levels."""
    print("🔧 Demo: Configurable Logging System")
    print("=" * 50)
    
    # Create loggers with different levels
    debug_logger = setup_logger("demo_debug", console_level="DEBUG")
    info_logger = setup_logger("demo_info", console_level="INFO") 
    warning_logger = setup_logger("demo_warning", console_level="WARNING")
    
    print("\n📊 DEBUG Level (shows all messages):")
    debug_logger.debug("This is a debug message")
    debug_logger.info("This is an info message")
    debug_logger.warning("This is a warning message")
    debug_logger.error("This is an error message")
    
    print("\n📊 INFO Level (shows info, warning, error):")
    info_logger.debug("This debug message won't appear")
    info_logger.info("This info message will appear")
    info_logger.warning("This warning will appear")
    info_logger.error("This error will appear")
    
    print("\n📊 WARNING Level (shows warning, error only):")
    warning_logger.debug("This debug message won't appear")
    warning_logger.info("This info message won't appear")
    warning_logger.warning("This warning will appear")
    warning_logger.error("This error will appear")
    
    print("\n✅ Logging demo completed!\n")


def demo_modular_components():
    """Demo: Using individual modules independently."""
    print("🧩 Demo: Modular Components")
    print("=" * 50)
    
    # Demo 1: Video Processor
    print("\n📹 VideoProcessor Module:")
    video_processor = VideoProcessor()
    print(f"   - Initialized VideoProcessor")
    print(f"   - Can load videos independently")
    print(f"   - Provides clean video info API")
    
    # Demo 2: Mask Handler  
    print("\n🎨 MaskHandler Module:")
    mask_handler = MaskHandler()
    print(f"   - Initialized MaskHandler")
    print(f"   - Handles mask processing independently")
    print(f"   - Provides mask validation utilities")
    
    # Demo 3: CoTracker Engine
    print("\n🎯 CoTrackerEngine Module:")
    tracker = CoTrackerEngine()
    print(f"   - Initialized CoTrackerEngine on {tracker.device}")
    print(f"   - Can track points independently")
    print(f"   - Handles model loading and inference")
    
    # Demo 4: Nuke Exporter
    print("\n📤 NukeExporter Module:")
    exporter = NukeExporter()
    print(f"   - Initialized NukeExporter")
    print(f"   - Handles CSV generation independently")
    print(f"   - Manages .nk file creation")
    
    print("\n✅ Modular components demo completed!\n")


def demo_main_app_api():
    """Demo: Clean main application API."""
    print("🚀 Demo: Main Application API")
    print("=" * 50)
    
    # Create app with custom settings
    app = CoTrackerNukeApp(debug_mode=True, console_log_level="INFO")
    
    print(f"\n📱 Application initialized:")
    print(f"   - Device: {app.device}")
    print(f"   - Debug mode: enabled")
    print(f"   - Console log level: INFO")
    print(f"   - Components: tracker, video_processor, mask_handler, exporter")
    
    # Show clean API methods
    print(f"\n🔧 Available API methods:")
    methods = [
        "load_video(path)",
        "set_reference_frame(frame_idx)",
        "get_reference_frame_image()",
        "process_mask_from_editor(edited_image)",
        "track_points(grid_size, use_mask)",
        "export_to_nuke(output_path, frame_offset)",
        "get_corner_pin_points()",
        "get_video_info()",
        "get_tracking_info()"
    ]
    
    for method in methods:
        print(f"   - app.{method}")
    
    print("\n✅ Main application API demo completed!\n")


def demo_import_patterns():
    """Demo: Different import patterns available."""
    print("📦 Demo: Import Patterns")
    print("=" * 50)
    
    print("\n🎯 Package-level imports:")
    print("   from cotracker_nuke import CoTrackerNukeApp, setup_logger")
    print("   from cotracker_nuke import create_gradio_interface")
    
    print("\n🧩 Module-specific imports:")
    print("   from cotracker_nuke.core import CoTrackerEngine, VideoProcessor")
    print("   from cotracker_nuke.exporters import NukeExporter")
    print("   from cotracker_nuke.utils import CoTrackerLogger")
    
    print("\n⚙️ CLI imports:")
    print("   from cotracker_nuke.cli import main")
    
    print("\n🖥️ UI imports:")
    print("   from cotracker_nuke.ui import GradioInterface")
    
    print("\n✅ Import patterns demo completed!\n")


def main():
    """Run all demonstrations."""
    print("🎬 CoTracker Nuke App - Modular Architecture Demo")
    print("=" * 60)
    print("This demo showcases the new modular capabilities!")
    print()
    
    try:
        # Run demos
        demo_logging_system()
        demo_modular_components()
        demo_main_app_api()
        demo_import_patterns()
        
        print("🎉 All demos completed successfully!")
        print("\n💡 Key Benefits Demonstrated:")
        print("   ✅ Configurable logging with multiple verbosity levels")
        print("   ✅ Independent, testable modules with clear responsibilities")
        print("   ✅ Clean, intuitive API for the main application")
        print("   ✅ Flexible import patterns for different use cases")
        print("   ✅ Professional architecture supporting CLI and GUI")
        
        print(f"\n🚀 Ready to use! Try:")
        print(f"   python cotracker_nuke_app_new.py --log-level DEBUG")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
