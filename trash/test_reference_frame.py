#!/usr/bin/env python3
"""
Test Reference Frame Functionality
==================================

This script tests whether the reference frame functionality in CoTracker
is working correctly by:
1. Creating a simple test video with a known pattern
2. Testing tracking from different reference frames
3. Verifying grid alignment at the reference frame
4. Checking bidirectional tracking behavior
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from cotracker_nuke_app import CoTrackerNukeApp
import tempfile
import os


def create_test_video(width=320, height=240, num_frames=60):
    """Create a test video with moving objects and grid patterns."""
    frames = []
    
    for t in range(num_frames):
        # Create a frame with a checkerboard pattern that moves
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a checkerboard background
        checker_size = 20
        for y in range(0, height, checker_size):
            for x in range(0, width, checker_size):
                if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                    frame[y:y+checker_size, x:x+checker_size] = [50, 50, 50]
        
        # Add a moving circle (predictable motion)
        circle_x = int(width * 0.2 + (width * 0.6) * t / num_frames)
        circle_y = int(height * 0.3 + (height * 0.4) * np.sin(2 * np.pi * t / 30))
        cv2.circle(frame, (circle_x, circle_y), 15, (255, 100, 100), -1)
        
        # Add a moving square
        square_x = int(width * 0.8 - (width * 0.6) * t / num_frames)
        square_y = int(height * 0.7)
        cv2.rectangle(frame, (square_x-10, square_y-10), (square_x+10, square_y+10), (100, 255, 100), -1)
        
        # Add corner markers for reference
        cv2.circle(frame, (20, 20), 5, (255, 255, 255), -1)  # Top-left
        cv2.circle(frame, (width-20, 20), 5, (255, 255, 255), -1)  # Top-right
        cv2.circle(frame, (20, height-20), 5, (255, 255, 255), -1)  # Bottom-left
        cv2.circle(frame, (width-20, height-20), 5, (255, 255, 255), -1)  # Bottom-right
        
        frames.append(frame)
    
    return np.array(frames)


def save_test_video(frames, filename):
    """Save frames as a video file."""
    import imageio.v3 as iio
    iio.imwrite(filename, frames, plugin="FFMPEG", fps=24)
    return filename


def test_reference_frame_alignment(app, video, reference_frames=[0, 15, 30, 45]):
    """Test if grid points are correctly aligned at different reference frames."""
    results = {}
    
    print("\n=== Testing Reference Frame Alignment ===")
    
    for ref_frame in reference_frames:
        print(f"\nTesting reference frame: {ref_frame}")
        
        # Track with a small grid for easier analysis
        tracks, visibility = app.track_points(video, grid_size=5, reference_frame=ref_frame)
        
        # Extract coordinates at the reference frame
        ref_frame_coords = tracks[0, ref_frame].cpu().numpy()  # Shape: (N, 2)
        
        print(f"Reference frame {ref_frame} coordinates:")
        print(f"Shape: {ref_frame_coords.shape}")
        
        # Check if coordinates form a reasonable grid
        x_coords = ref_frame_coords[:, 0]
        y_coords = ref_frame_coords[:, 1]
        
        print(f"X coordinates: {sorted(set(np.round(x_coords).astype(int)))}")
        print(f"Y coordinates: {sorted(set(np.round(y_coords).astype(int)))}")
        
        # Store results
        results[ref_frame] = {
            'tracks': tracks,
            'visibility': visibility,
            'ref_coords': ref_frame_coords,
            'x_coords': x_coords,
            'y_coords': y_coords
        }
        
        # Verify grid structure
        unique_x = len(set(np.round(x_coords).astype(int)))
        unique_y = len(set(np.round(y_coords).astype(int)))
        expected_grid_size = 5
        
        print(f"Grid structure: {unique_x} x {unique_y} (expected: {expected_grid_size} x {expected_grid_size})")
        
        if unique_x == expected_grid_size and unique_y == expected_grid_size:
            print("✓ Grid structure looks correct")
        else:
            print("⚠ Grid structure may be incorrect")
    
    return results


def test_bidirectional_tracking(app, video, reference_frame=30):
    """Test if tracking works bidirectionally from a middle reference frame."""
    print(f"\n=== Testing Bidirectional Tracking (ref frame: {reference_frame}) ===")
    
    # Track from middle frame
    tracks, visibility = app.track_points(video, grid_size=4, reference_frame=reference_frame)
    
    # Analyze a single point's trajectory
    point_idx = 0
    point_track = tracks[0, :, point_idx].cpu().numpy()  # Shape: (T, 2)
    point_visibility = visibility[0, :, point_idx].cpu().numpy()  # Shape: (T,)
    
    print(f"Analyzing point {point_idx} trajectory:")
    print(f"Total frames: {len(point_track)}")
    print(f"Reference frame: {reference_frame}")
    
    # Check coordinates at key frames
    ref_coord = point_track[reference_frame]
    first_coord = point_track[0]
    last_coord = point_track[-1]
    
    print(f"Coordinates at frame 0: ({first_coord[0]:.1f}, {first_coord[1]:.1f})")
    print(f"Coordinates at ref frame {reference_frame}: ({ref_coord[0]:.1f}, {ref_coord[1]:.1f})")
    print(f"Coordinates at last frame: ({last_coord[0]:.1f}, {last_coord[1]:.1f})")
    
    # Check if tracking exists before and after reference frame
    frames_before_ref = reference_frame
    frames_after_ref = len(point_track) - reference_frame - 1
    
    visible_before = np.sum(point_visibility[:reference_frame] > 0.5)
    visible_after = np.sum(point_visibility[reference_frame+1:] > 0.5)
    
    print(f"Visible frames before ref: {visible_before}/{frames_before_ref}")
    print(f"Visible frames after ref: {visible_after}/{frames_after_ref}")
    
    # Test if tracking is truly bidirectional
    if visible_before > 0 and visible_after > 0:
        print("✓ Bidirectional tracking appears to work")
        
        # Check for reasonable motion
        motion_before = np.linalg.norm(ref_coord - first_coord)
        motion_after = np.linalg.norm(last_coord - ref_coord)
        
        print(f"Motion magnitude before ref: {motion_before:.1f} pixels")
        print(f"Motion magnitude after ref: {motion_after:.1f} pixels")
        
        return True
    else:
        print("⚠ Bidirectional tracking may not be working")
        return False


def visualize_tracking_results(results, video, save_path="reference_frame_test.png"):
    """Visualize tracking results for different reference frames."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    ref_frames = list(results.keys())[:4]  # Take first 4 reference frames
    
    for i, ref_frame in enumerate(ref_frames):
        ax = axes[i]
        
        # Show the reference frame image
        ref_image = video[ref_frame]
        ax.imshow(ref_image)
        
        # Overlay the grid points at reference frame
        coords = results[ref_frame]['ref_coords']
        ax.scatter(coords[:, 0], coords[:, 1], c='red', s=50, alpha=0.8, marker='x')
        
        # Number the points
        for j, (x, y) in enumerate(coords):
            ax.text(x+5, y+5, str(j), color='yellow', fontsize=8, fontweight='bold')
        
        ax.set_title(f'Reference Frame {ref_frame}\nGrid Points Overlay')
        ax.set_xlim(0, video.shape[2])
        ax.set_ylim(video.shape[1], 0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    return save_path


def test_tracking_consistency(app, video):
    """Test if the same points tracked from different reference frames give consistent results."""
    print("\n=== Testing Tracking Consistency ===")
    
    # Track from frame 0 and frame 30
    tracks_ref0, _ = app.track_points(video, grid_size=3, reference_frame=0)
    tracks_ref30, _ = app.track_points(video, grid_size=3, reference_frame=30)
    
    # Compare coordinates at frame 15 (should be similar regardless of reference frame)
    test_frame = 15
    coords_from_ref0 = tracks_ref0[0, test_frame].cpu().numpy()
    coords_from_ref30 = tracks_ref30[0, test_frame].cpu().numpy()
    
    print(f"Coordinates at frame {test_frame}:")
    print(f"When tracked from ref frame 0:")
    for i, (x, y) in enumerate(coords_from_ref0[:5]):  # Show first 5 points
        print(f"  Point {i}: ({x:.1f}, {y:.1f})")
    
    print(f"When tracked from ref frame 30:")
    for i, (x, y) in enumerate(coords_from_ref30[:5]):  # Show first 5 points
        print(f"  Point {i}: ({x:.1f}, {y:.1f})")
    
    # Calculate differences
    differences = np.linalg.norm(coords_from_ref0 - coords_from_ref30, axis=1)
    avg_difference = np.mean(differences)
    max_difference = np.max(differences)
    
    print(f"Average coordinate difference: {avg_difference:.2f} pixels")
    print(f"Maximum coordinate difference: {max_difference:.2f} pixels")
    
    if avg_difference < 5.0:  # Threshold for "close enough"
        print("✓ Tracking consistency looks good")
        return True
    else:
        print("⚠ Tracking results are inconsistent between reference frames")
        return False


def main():
    """Run all reference frame tests."""
    print("Creating test video...")
    video = create_test_video(width=320, height=240, num_frames=60)
    
    # Save test video for inspection
    temp_video = save_test_video(video, "test_video_reference_frame.mp4")
    print(f"Test video saved: {temp_video}")
    
    # Initialize app
    print("Initializing CoTracker app...")
    app = CoTrackerNukeApp()
    
    try:
        # Test 1: Reference frame alignment
        results = test_reference_frame_alignment(app, video, reference_frames=[0, 15, 30, 45])
        
        # Test 2: Bidirectional tracking
        bidirectional_works = test_bidirectional_tracking(app, video, reference_frame=30)
        
        # Test 3: Tracking consistency
        consistency_ok = test_tracking_consistency(app, video)
        
        # Visualize results
        viz_path = visualize_tracking_results(results, video)
        
        # Summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Bidirectional tracking: {'✓ PASS' if bidirectional_works else '⚠ FAIL'}")
        print(f"Tracking consistency: {'✓ PASS' if consistency_ok else '⚠ FAIL'}")
        print(f"Visualization: {viz_path}")
        print(f"Test video: {temp_video}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
