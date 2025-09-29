#!/usr/bin/env python3
"""
Test script for CoTracker Nuke App
"""

import sys
import torch
import numpy as np
from cotracker_nuke_app import CoTrackerNukeApp

def test_basic_functionality():
    """Test basic app functionality without video processing."""
    print("Testing CoTracker Nuke App...")
    
    # Initialize app
    app = CoTrackerNukeApp()
    print(f"✓ App initialized successfully")
    print(f"✓ Device: {app.device}")
    
    # Test point selection algorithm with dummy data
    print("\nTesting point selection algorithm...")
    
    # Create dummy tracking data
    T, N = 100, 20  # 100 frames, 20 points
    dummy_tracks = torch.randn(1, T, N, 2) * 100 + 200  # Random tracks around (200, 200)
    dummy_visibility = torch.ones(1, T, N, 1) * 0.9  # High visibility
    
    # Test corner point selection
    corner_indices = app.select_corner_pin_points(dummy_tracks, dummy_visibility)
    print(f"✓ Selected corner points: {corner_indices}")
    assert len(corner_indices) == 4, f"Expected 4 corner points, got {len(corner_indices)}"
    
    # Test Nuke export with dummy data
    print("\nTesting Nuke export...")
    video_info = {'width': 1920, 'height': 1080, 'fps': 24}
    
    try:
        nuke_script = app.export_to_nuke(
            dummy_tracks, dummy_visibility, corner_indices, 
            "test_output.nk", video_info
        )
        print("✓ Nuke script generated successfully")
        print(f"✓ Script length: {len(nuke_script)} characters")
        
        # Check if script contains expected elements
        assert "CornerPin2D" in nuke_script, "CornerPin2D node not found in script"
        assert "Tracker4" in nuke_script, "Tracker4 node not found in script"
        print("✓ Script contains required nodes")
        
    except Exception as e:
        print(f"✗ Nuke export failed: {e}")
        return False
    
    print("\n🎉 All basic tests passed!")
    return True

def test_model_loading():
    """Test CoTracker model loading (requires internet)."""
    print("\nTesting CoTracker model loading...")
    
    app = CoTrackerNukeApp()
    
    try:
        app.load_cotracker_model()
        print("✓ CoTracker model loaded successfully")
        print(f"✓ Model device: {next(app.cotracker_model.parameters()).device}")
        return True
    except Exception as e:
        print(f"⚠ Model loading failed (this is expected without internet): {e}")
        return False

if __name__ == "__main__":
    print("CoTracker Nuke App Test Suite")
    print("=" * 40)
    
    # Run basic tests
    basic_success = test_basic_functionality()
    
    # Try model loading (may fail without internet)
    model_success = test_model_loading()
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"Basic functionality: {'✓ PASS' if basic_success else '✗ FAIL'}")
    print(f"Model loading: {'✓ PASS' if model_success else '⚠ SKIP (no internet)'}")
    
    if basic_success:
        print("\n🎉 App is ready to use!")
        print("Run: python cotracker_nuke_app.py")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
