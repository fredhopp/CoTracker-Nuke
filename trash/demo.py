#!/usr/bin/env python3
"""
Demo script to launch the CoTracker Nuke App
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the CoTracker Nuke App."""
    
    print("🎬 CoTracker Nuke Integration App")
    print("=" * 50)
    print()
    print("This app will:")
    print("• Track points in your video using CoTracker")
    print("• Automatically select 4 optimal corner points")
    print("• Export tracking data for Nuke CornerPin2D")
    print()
    print("Starting Gradio interface...")
    print("📱 Open your browser to: http://127.0.0.1:7860")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        from cotracker_nuke_app import create_gradio_interface
        
        interface = create_gradio_interface()
        interface.launch(
            share=False, 
            server_name="127.0.0.1", 
            server_port=7860,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the virtual environment:")
        print("   source .venv/Scripts/activate  # Windows")
        print("   source .venv/bin/activate      # macOS/Linux")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check that port 7860 is available")
        sys.exit(1)

if __name__ == "__main__":
    main()
