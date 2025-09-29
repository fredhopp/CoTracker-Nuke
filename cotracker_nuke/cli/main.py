#!/usr/bin/env python3
"""
CLI interface for CoTracker Nuke App
=====================================

Provides command-line interface for batch processing and automation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..core.app import CoTrackerNukeApp


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CoTracker Nuke Integration - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tracking with default settings
  cotracker-nuke video.mp4 output.nk
  
  # Custom grid size and reference frame
  cotracker-nuke video.mp4 output.nk --grid-size 15 --reference-frame 10
  
  # With frame offset for image sequences
  cotracker-nuke video.mp4 output.nk --frame-offset 1001
  
  # Enable debug logging
  cotracker-nuke video.mp4 output.nk --log-level DEBUG
  
  # Use mask file
  cotracker-nuke video.mp4 output.nk --mask mask.png
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "input_video",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "output_nuke",
        help="Path to output Nuke .nk file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--grid-size", "-g",
        type=int,
        default=10,
        help="Grid size for point generation (default: 10)"
    )
    
    parser.add_argument(
        "--reference-frame", "-r",
        type=int,
        default=0,
        help="Reference frame index (0-based, default: 0)"
    )
    
    parser.add_argument(
        "--frame-offset", "-f",
        type=int,
        default=1001,
        help="Frame offset for image sequences (default: 1001)"
    )
    
    parser.add_argument(
        "--mask", "-m",
        type=str,
        help="Path to mask image file (optional)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug file logging"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="CoTracker Nuke Integration v1.0.0"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input files
    input_path = Path(args.input_video)
    if not input_path.exists():
        print(f"❌ Error: Input video file not found: {input_path}")
        sys.exit(1)
    
    # Validate mask file if provided
    mask_path = None
    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            print(f"❌ Error: Mask file not found: {mask_path}")
            sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output_nuke)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize application
        debug_mode = not args.no_debug
        app = CoTrackerNukeApp(debug_mode, args.log_level)
        
        print(f"🎬 CoTracker Nuke Integration - CLI Mode")
        print(f"📹 Input: {input_path}")
        print(f"📁 Output: {output_path}")
        print(f"🔢 Grid size: {args.grid_size}")
        print(f"🎯 Reference frame: {args.reference_frame}")
        print(f"📊 Frame offset: {args.frame_offset}")
        if mask_path:
            print(f"🎨 Mask: {mask_path}")
        print()
        
        # Load video
        print("📹 Loading video...")
        app.load_video(str(input_path))
        
        # Set reference frame
        print(f"🎯 Setting reference frame to {args.reference_frame}...")
        app.set_reference_frame(args.reference_frame)
        
        # Load mask if provided
        if mask_path:
            print(f"🎨 Loading mask from {mask_path}...")
            import numpy as np
            from PIL import Image
            
            mask_image = Image.open(mask_path)
            mask_array = np.array(mask_image.convert('L'))  # Convert to grayscale
            # Convert to binary mask (assuming white areas are trackable)
            mask_binary = (mask_array > 128).astype(np.uint8) * 255
            app.mask_handler.current_mask = mask_binary
            
            stats = app.mask_handler.get_mask_stats(mask_binary)
            print(f"   Mask coverage: {stats['coverage_percent']:.1f}%")
        
        # Track points
        print("🚀 Processing video with CoTracker...")
        app.track_points(args.grid_size, use_mask=mask_path is not None)
        
        # Get tracking info
        info = app.get_tracking_info()
        print(f"   ✅ Tracked {info['num_points']} points across {info['num_frames']} frames")
        print(f"   👁️ Overall visibility: {info['visibility_rate']:.1f}%")
        
        # Export to Nuke
        print("📤 Exporting to Nuke...")
        nuke_file = app.export_to_nuke(str(output_path), args.frame_offset)
        
        print(f"✅ Export completed successfully!")
        print(f"📁 Nuke file: {nuke_file}")
        print()
        print("🎉 Ready to import in Nuke!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
