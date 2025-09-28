#!/usr/bin/env python3
"""
Demo script to launch the Simple Mask Drawing Tool
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the Simple Mask Drawing Tool."""
    
    print("🎨 Simple Mask Drawing Tool")
    print("=" * 50)
    print()
    print("This tool will:")
    print("• Load video and select reference frame")
    print("• Draw masks with Gradio's built-in brush tools")
    print("• Extract black/white mask from your drawing")
    print("• Export mask as PNG file")
    print()
    print("Starting Gradio interface...")
    print("📱 Open your browser to: http://127.0.0.1:7862")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        from simple_mask_tool import create_simple_mask_interface
        
        interface = create_simple_mask_interface()
        interface.launch(
            share=False, 
            server_name="127.0.0.1", 
            server_port=7862,
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
        print("3. Check that port 7862 is available")
        sys.exit(1)

if __name__ == "__main__":
    main()
