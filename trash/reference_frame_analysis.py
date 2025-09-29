#!/usr/bin/env python3
"""
Detailed Analysis of Reference Frame Issues
===========================================

Based on the test results, this script provides a detailed analysis
of the issues found with reference frame implementation.
"""

import numpy as np
import torch
from cotracker_nuke_app import CoTrackerNukeApp
import matplotlib.pyplot as plt


def analyze_query_generation_issue():
    """Analyze the issue with custom query generation."""
    print("=== ANALYZING QUERY GENERATION ISSUE ===")
    
    app = CoTrackerNukeApp()
    
    # Create a simple test video
    video = np.random.rand(10, 100, 100, 3) * 255
    video = video.astype(np.uint8)
    
    # Test query generation
    try:
        queries = app._generate_grid_queries(video, grid_size=3, reference_frame=5)
        print(f"✓ Query generation successful")
        print(f"Query shape: {queries.shape}")
        print(f"Sample queries:\n{queries[:5]}")
        
        # Check query format
        if queries.shape[1] == 3:
            print("✓ Query format is correct: [frame_idx, x, y]")
            
            # Check if reference frame is correctly set
            ref_frames = queries[:, 0].unique()
            print(f"Reference frames in queries: {ref_frames.cpu().numpy()}")
            
            if len(ref_frames) == 1 and ref_frames[0] == 5:
                print("✓ Reference frame correctly set in all queries")
            else:
                print("⚠ Reference frame issue in queries")
        else:
            print(f"⚠ Query format issue: expected 3 columns, got {queries.shape[1]}")
            
    except Exception as e:
        print(f"⚠ Query generation failed: {e}")
        import traceback
        traceback.print_exc()


def analyze_tracking_api_issue():
    """Analyze the API issue causing the tracking error."""
    print("\n=== ANALYZING TRACKING API ISSUE ===")
    
    app = CoTrackerNukeApp()
    
    # Load the model
    app.load_cotracker_model()
    
    # Create test data
    video = np.random.rand(10, 100, 100, 3) * 255
    video = video.astype(np.uint8)
    video_tensor = torch.tensor(video).permute(0, 3, 1, 2)[None].float().to(app.device)
    
    print(f"Video tensor shape: {video_tensor.shape}")
    
    # Test 1: Default grid tracking (should work)
    try:
        print("\nTesting default grid tracking...")
        with torch.no_grad():
            pred_tracks, pred_visibility = app.cotracker_model(video_tensor, grid_size=3)
        print(f"✓ Default tracking successful")
        print(f"Tracks shape: {pred_tracks.shape}")
        print(f"Visibility shape: {pred_visibility.shape}")
    except Exception as e:
        print(f"⚠ Default tracking failed: {e}")
    
    # Test 2: Custom queries tracking (this is where the error occurs)
    try:
        print("\nTesting custom queries tracking...")
        queries = app._generate_grid_queries(video, grid_size=3, reference_frame=5)
        print(f"Generated queries shape: {queries.shape}")
        
        with torch.no_grad():
            result = app.cotracker_model(video_tensor, queries=queries)
            
        print(f"Result type: {type(result)}")
        if isinstance(result, tuple):
            print(f"Result tuple length: {len(result)}")
            for i, item in enumerate(result):
                if hasattr(item, 'shape'):
                    print(f"  Item {i} shape: {item.shape}")
                else:
                    print(f"  Item {i} type: {type(item)}")
        
    except Exception as e:
        print(f"⚠ Custom queries tracking failed: {e}")
        print("This explains the 'not enough values to unpack' error!")
        import traceback
        traceback.print_exc()


def test_coordinate_alignment():
    """Test if coordinates are actually aligned at the reference frame."""
    print("\n=== TESTING COORDINATE ALIGNMENT ===")
    
    app = CoTrackerNukeApp()
    
    # Create a test video with known pattern
    video = create_pattern_video()
    
    # Test tracking from different reference frames
    ref_frames = [0, 15, 30]
    
    for ref_frame in ref_frames:
        print(f"\nTesting reference frame: {ref_frame}")
        
        try:
            tracks, visibility = app.track_points(video, grid_size=3, reference_frame=ref_frame)
            
            # Get coordinates at the reference frame
            ref_coords = tracks[0, ref_frame].cpu().numpy()
            
            print(f"Coordinates at reference frame {ref_frame}:")
            for i, (x, y) in enumerate(ref_coords):
                print(f"  Point {i}: ({x:.1f}, {y:.1f})")
            
            # Check if coordinates form expected grid pattern
            expected_positions = get_expected_grid_positions(video.shape[2], video.shape[1], 3)
            
            # Compare with expected positions
            distances = []
            for i, (expected_x, expected_y) in enumerate(expected_positions):
                if i < len(ref_coords):
                    actual_x, actual_y = ref_coords[i]
                    dist = np.sqrt((actual_x - expected_x)**2 + (actual_y - expected_y)**2)
                    distances.append(dist)
                    print(f"  Point {i}: expected ({expected_x:.1f}, {expected_y:.1f}), "
                          f"actual ({actual_x:.1f}, {actual_y:.1f}), distance: {dist:.1f}")
            
            avg_distance = np.mean(distances) if distances else float('inf')
            print(f"Average alignment error: {avg_distance:.1f} pixels")
            
            if avg_distance < 5.0:
                print("✓ Grid alignment looks good")
            else:
                print("⚠ Grid alignment may be poor")
                
        except Exception as e:
            print(f"⚠ Tracking failed: {e}")


def create_pattern_video(width=160, height=120, num_frames=45):
    """Create a video with a clear pattern for testing."""
    frames = []
    
    for t in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a grid pattern
        grid_spacing = 20
        for y in range(0, height, grid_spacing):
            for x in range(0, width, grid_spacing):
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        
        # Add frame number
        cv2.putText(frame, str(t), (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        frames.append(frame)
    
    return np.array(frames)


def get_expected_grid_positions(width, height, grid_size):
    """Get expected grid positions for a given grid size."""
    positions = []
    
    x_step = width / (grid_size - 1) if grid_size > 1 else width / 2
    y_step = height / (grid_size - 1) if grid_size > 1 else height / 2
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = j * x_step if grid_size > 1 else width // 2
            y = i * y_step if grid_size > 1 else height // 2
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            positions.append((x, y))
    
    return positions


def main():
    """Run the detailed analysis."""
    print("REFERENCE FRAME DETAILED ANALYSIS")
    print("=" * 50)
    
    # Import cv2 here to avoid issues if not available
    global cv2
    import cv2
    
    analyze_query_generation_issue()
    analyze_tracking_api_issue()
    test_coordinate_alignment()
    
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print("Key findings from the test:")
    print("1. Custom query tracking has API compatibility issues")
    print("2. The fallback to default grid works but ignores reference frame")
    print("3. Grid structure varies inconsistently with reference frame")
    print("4. Bidirectional tracking works but may not use true reference frame")
    print("\nRecommendation: Fix the custom query API usage")


if __name__ == "__main__":
    main()
