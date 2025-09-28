#!/usr/bin/env python3
"""
Debug CoTracker API to understand the correct query format
"""

import torch
import numpy as np


def debug_cotracker2_api():
    """Debug the CoTracker2 API to understand query format."""
    print("=== DEBUGGING COTRACKER2 API ===")
    
    # Load CoTracker2 model
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    
    # Create test video
    video = torch.randn(1, 10, 3, 100, 100)  # B, T, C, H, W
    
    print(f"Video shape: {video.shape}")
    
    # Test 1: Default grid tracking
    print("\n1. Testing default grid tracking...")
    try:
        tracks, visibility = model(video, grid_size=3)
        print(f"✓ Success! Tracks: {tracks.shape}, Visibility: {visibility.shape}")
    except Exception as e:
        print(f"⚠ Failed: {e}")
    
    # Test 2: Try different query formats
    print("\n2. Testing different query formats...")
    
    # Format 1: [N, 3] - (frame, x, y)
    queries_format1 = torch.tensor([
        [0, 10, 10],
        [0, 50, 50],
        [0, 90, 90]
    ], dtype=torch.float32)
    
    print(f"Query format 1 shape: {queries_format1.shape}")
    try:
        tracks, visibility = model(video, queries=queries_format1)
        print(f"✓ Format 1 works! Tracks: {tracks.shape}, Visibility: {visibility.shape}")
    except Exception as e:
        print(f"⚠ Format 1 failed: {e}")
    
    # Format 2: [B, N, 3] - batch dimension added
    queries_format2 = queries_format1.unsqueeze(0)  # Add batch dimension
    
    print(f"Query format 2 shape: {queries_format2.shape}")
    try:
        tracks, visibility = model(video, queries=queries_format2)
        print(f"✓ Format 2 works! Tracks: {tracks.shape}, Visibility: {visibility.shape}")
    except Exception as e:
        print(f"⚠ Format 2 failed: {e}")
    
    # Test 3: Check model documentation
    print("\n3. Checking model documentation...")
    print(f"Model type: {type(model)}")
    
    # Try to get help or docstring
    if hasattr(model, 'forward'):
        print("Forward method signature:")
        import inspect
        sig = inspect.signature(model.forward)
        print(f"  {sig}")
        
        if model.forward.__doc__:
            print("Forward method docstring:")
            print(f"  {model.forward.__doc__}")
    
    # Test 4: Check if there are any example usages in the model
    print("\n4. Model attributes and methods:")
    methods = [attr for attr in dir(model) if not attr.startswith('_')]
    for method in methods[:10]:  # Show first 10 methods
        print(f"  {method}")
    
    return model


def test_fixed_query_format():
    """Test with the corrected query format."""
    print("\n=== TESTING FIXED QUERY FORMAT ===")
    
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    video = torch.randn(1, 10, 3, 100, 100)
    
    # Based on the error, it seems queries should be [B, N, 3]
    # Let's try the correct format
    queries = torch.tensor([[
        [5, 25, 25],  # frame 5, position (25, 25)
        [5, 50, 50],  # frame 5, position (50, 50)
        [5, 75, 75],  # frame 5, position (75, 75)
    ]], dtype=torch.float32)
    
    print(f"Corrected query shape: {queries.shape}")
    
    try:
        tracks, visibility = model(video, queries=queries)
        print(f"✓ Corrected format works! Tracks: {tracks.shape}, Visibility: {visibility.shape}")
        
        # Check if reference frame is correctly used
        print(f"Tracks at frame 5 (reference): {tracks[0, 5]}")
        print(f"Expected coordinates: [[25, 25], [50, 50], [75, 75]]")
        
        return True
    except Exception as e:
        print(f"⚠ Corrected format failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the debugging."""
    model = debug_cotracker2_api()
    success = test_fixed_query_format()
    
    if success:
        print("\n✓ Found the correct query format!")
        print("Queries should be shape [B, N, 3] where:")
        print("  B = batch size (usually 1)")
        print("  N = number of query points")
        print("  3 = (frame_index, x_coordinate, y_coordinate)")
    else:
        print("\n⚠ Still having issues with query format")


if __name__ == "__main__":
    main()
