#!/usr/bin/env python3
"""
Test Debug Functionality
========================

This script tests the new debug functionality to ensure it works correctly.
"""

import numpy as np
import cv2
from cotracker_nuke_app import CoTrackerNukeApp
from pathlib import Path
import json


def create_simple_test_video():
    """Create a simple test video for debugging."""
    frames = []
    width, height = 160, 120
    num_frames = 20
    
    for t in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a moving dot
        x = int(20 + t * 5)
        y = int(height // 2)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        
        # Add frame number
        cv2.putText(frame, str(t), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        frames.append(frame)
    
    return np.array(frames)


def test_debug_functionality():
    """Test the debug functionality."""
    print("=== TESTING DEBUG FUNCTIONALITY ===")
    
    # Create test video
    video = create_simple_test_video()
    print(f"Created test video: {video.shape}")
    
    # Initialize app with debug mode
    app = CoTrackerNukeApp(debug_mode=True)
    
    # Test video info logging
    app.log_video_info(video, "test_video.mp4")
    
    # Test tracking parameters logging
    grid_size = 3
    reference_frame = 10
    preview_downsample = 2
    
    app.log_tracking_params(grid_size, reference_frame, preview_downsample)
    
    # Test tracking with debug
    print("\nRunning tracking with debug enabled...")
    try:
        tracks, visibility = app.track_points(video, grid_size=grid_size, reference_frame=reference_frame)
        
        # Test coordinate export
        app.export_coordinates(tracks, visibility, grid_size, preview_downsample, reference_frame)
        
        print("✓ Debug functionality test completed successfully!")
        
        # Check if files were created
        debug_dir = Path("Z:/Dev/Cotracker/temp")
        print(f"\nDebug files in {debug_dir}:")
        
        if debug_dir.exists():
            files = list(debug_dir.glob("*"))
            for file in sorted(files):
                if file.is_file():
                    size_kb = file.stat().st_size / 1024
                    print(f"  {file.name} ({size_kb:.1f} KB)")
            
            # Test reading one of the JSON files
            json_files = list(debug_dir.glob("full_grid_coords_*.json"))
            if json_files:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                
                print(f"\nSample JSON data structure:")
                print(f"  Metadata keys: {list(data['metadata'].keys())}")
                print(f"  Grid size: {data['metadata']['grid_size']}")
                print(f"  Reference frame: {data['metadata']['reference_frame']}")
                print(f"  Total points: {data['metadata']['total_points']}")
                print(f"  Total frames: {data['metadata']['total_frames']}")
                
                # Show sample coordinates
                frame_0_coords = data['coordinates']['frame_0'][:3]  # First 3 points
                print(f"  Sample frame 0 coordinates: {frame_0_coords}")
        
        return True
        
    except Exception as e:
        print(f"⚠ Debug functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_debug_mode_toggle():
    """Test that debug mode can be toggled on/off."""
    print("\n=== TESTING DEBUG MODE TOGGLE ===")
    
    # Test with debug mode off
    app_no_debug = CoTrackerNukeApp(debug_mode=False)
    print(f"Debug mode OFF - has logger: {hasattr(app_no_debug, 'logger')}")
    
    # Test with debug mode on
    app_debug = CoTrackerNukeApp(debug_mode=True)
    print(f"Debug mode ON - has logger: {hasattr(app_debug, 'logger')}")
    
    if hasattr(app_debug, 'logger'):
        print("✓ Debug mode toggle works correctly")
        return True
    else:
        print("⚠ Debug mode toggle may not be working")
        return False


def main():
    """Run all debug functionality tests."""
    print("DEBUG FUNCTIONALITY TEST SUITE")
    print("=" * 50)
    
    # Test 1: Debug functionality
    debug_works = test_debug_functionality()
    
    # Test 2: Debug mode toggle
    toggle_works = test_debug_mode_toggle()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Debug functionality: {'✓ PASS' if debug_works else '⚠ FAIL'}")
    print(f"Debug mode toggle: {'✓ PASS' if toggle_works else '⚠ FAIL'}")
    
    if debug_works and toggle_works:
        print("\n🎉 ALL DEBUG TESTS PASSED!")
        print("Debug functionality is ready for use.")
    else:
        print("\n⚠ Some debug tests failed.")
    
    print(f"\nDebug output location: Z:/Dev/Cotracker/temp/")


if __name__ == "__main__":
    main()
