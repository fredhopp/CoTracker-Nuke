#!/usr/bin/env python3
"""
Test the Reference Frame Fix
============================

This script tests whether the reference frame fix works correctly.
"""

import numpy as np
import torch
import cv2
from cotracker_nuke_app import CoTrackerNukeApp
import matplotlib.pyplot as plt


def create_test_video_with_markers(width=320, height=240, num_frames=30):
    """Create a test video with clear markers at specific positions."""
    frames = []
    
    for t in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a moving pattern - diagonal line that moves
        line_offset = t * 2
        for i in range(min(width, height)):
            x = (i + line_offset) % width
            y = i
            if y < height:
                frame[y, x] = [255, 255, 255]
        
        # Add static corner markers for reference
        cv2.circle(frame, (20, 20), 8, (255, 0, 0), -1)  # Red - top-left
        cv2.circle(frame, (width-20, 20), 8, (0, 255, 0), -1)  # Green - top-right
        cv2.circle(frame, (20, height-20), 8, (0, 0, 255), -1)  # Blue - bottom-left
        cv2.circle(frame, (width-20, height-20), 8, (255, 255, 0), -1)  # Yellow - bottom-right
        
        # Add frame number
        cv2.putText(frame, f"F{t}", (width//2-10, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return np.array(frames)


def test_reference_frame_fix():
    """Test if the reference frame fix works correctly."""
    print("=== TESTING REFERENCE FRAME FIX ===")
    
    # Create test video
    video = create_test_video_with_markers()
    print(f"Created test video: {video.shape}")
    
    # Initialize app
    app = CoTrackerNukeApp()
    
    # Test different reference frames
    test_frames = [0, 10, 20]
    results = {}
    
    for ref_frame in test_frames:
        print(f"\nTesting reference frame: {ref_frame}")
        
        try:
            # Track with small grid for easier analysis
            tracks, visibility = app.track_points(video, grid_size=3, reference_frame=ref_frame)
            
            print(f"✓ Tracking successful!")
            print(f"  Tracks shape: {tracks.shape}")
            print(f"  Visibility shape: {visibility.shape}")
            
            # Get coordinates at the reference frame
            ref_coords = tracks[0, ref_frame].cpu().numpy()
            
            print(f"  Coordinates at reference frame {ref_frame}:")
            for i, (x, y) in enumerate(ref_coords):
                print(f"    Point {i}: ({x:.1f}, {y:.1f})")
            
            # Verify the grid is properly positioned
            expected_positions = get_expected_grid_positions(video.shape[2], video.shape[1], 3)
            
            # Calculate alignment error
            alignment_errors = []
            for i, (expected_x, expected_y) in enumerate(expected_positions):
                if i < len(ref_coords):
                    actual_x, actual_y = ref_coords[i]
                    error = np.sqrt((actual_x - expected_x)**2 + (actual_y - expected_y)**2)
                    alignment_errors.append(error)
            
            avg_error = np.mean(alignment_errors) if alignment_errors else float('inf')
            print(f"  Average alignment error: {avg_error:.1f} pixels")
            
            # Store results
            results[ref_frame] = {
                'tracks': tracks,
                'visibility': visibility,
                'ref_coords': ref_coords,
                'alignment_error': avg_error,
                'success': True
            }
            
        except Exception as e:
            print(f"⚠ Tracking failed for reference frame {ref_frame}: {e}")
            results[ref_frame] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def test_bidirectional_tracking_fixed():
    """Test if bidirectional tracking works correctly with the fix."""
    print("\n=== TESTING BIDIRECTIONAL TRACKING (FIXED) ===")
    
    video = create_test_video_with_markers(num_frames=30)
    app = CoTrackerNukeApp()
    
    # Use middle frame as reference
    ref_frame = 15
    
    try:
        tracks, visibility = app.track_points(video, grid_size=2, reference_frame=ref_frame)
        
        # Analyze first point's trajectory
        point_track = tracks[0, :, 0].cpu().numpy()  # Shape: (T, 2)
        point_visibility = visibility[0, :, 0].cpu().numpy()  # Shape: (T,)
        
        print(f"Point trajectory analysis (reference frame: {ref_frame}):")
        print(f"  Frame 0: ({point_track[0, 0]:.1f}, {point_track[0, 1]:.1f})")
        print(f"  Frame {ref_frame} (ref): ({point_track[ref_frame, 0]:.1f}, {point_track[ref_frame, 1]:.1f})")
        print(f"  Frame {len(point_track)-1}: ({point_track[-1, 0]:.1f}, {point_track[-1, 1]:.1f})")
        
        # Check visibility before and after reference frame
        visible_before = np.sum(point_visibility[:ref_frame] > 0.5)
        visible_after = np.sum(point_visibility[ref_frame+1:] > 0.5)
        
        print(f"  Visible frames before reference: {visible_before}/{ref_frame}")
        print(f"  Visible frames after reference: {visible_after}/{len(point_track)-ref_frame-1}")
        
        if visible_before > 0 and visible_after > 0:
            print("✓ Bidirectional tracking confirmed!")
            
            # Check if coordinates actually change (indicating real tracking)
            motion_before = np.linalg.norm(point_track[ref_frame] - point_track[0])
            motion_after = np.linalg.norm(point_track[-1] - point_track[ref_frame])
            
            print(f"  Motion before reference: {motion_before:.1f} pixels")
            print(f"  Motion after reference: {motion_after:.1f} pixels")
            
            return True
        else:
            print("⚠ Bidirectional tracking may not be working")
            return False
            
    except Exception as e:
        print(f"⚠ Bidirectional test failed: {e}")
        return False


def get_expected_grid_positions(width, height, grid_size):
    """Get expected grid positions for comparison."""
    positions = []
    
    x_step = width / (grid_size - 1) if grid_size > 1 else width / 2
    y_step = height / (grid_size - 1) if grid_size > 1 else height / 2
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = j * x_step if grid_size > 1 else width // 2
            y = i * y_step if grid_size > 1 else height // 2
            
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            positions.append((x, y))
    
    return positions


def visualize_fix_results(results, video):
    """Visualize the results of the reference frame fix."""
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, (ref_frame, result) in enumerate(results.items()):
        ax = axes[i]
        
        if result['success']:
            # Show the reference frame
            frame_img = video[ref_frame]
            ax.imshow(frame_img)
            
            # Overlay tracked points
            coords = result['ref_coords']
            ax.scatter(coords[:, 0], coords[:, 1], c='red', s=100, alpha=0.8, marker='x')
            
            # Number the points
            for j, (x, y) in enumerate(coords):
                ax.text(x+5, y+5, str(j), color='yellow', fontsize=12, fontweight='bold')
            
            ax.set_title(f'Ref Frame {ref_frame}\nError: {result["alignment_error"]:.1f}px')
        else:
            ax.text(0.5, 0.5, f'Failed\n{result["error"]}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Ref Frame {ref_frame} - FAILED')
        
        ax.set_xlim(0, video.shape[2])
        ax.set_ylim(video.shape[1], 0)
    
    plt.tight_layout()
    plt.savefig('reference_frame_fix_test.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: reference_frame_fix_test.png")


def main():
    """Run the reference frame fix test."""
    print("REFERENCE FRAME FIX TEST")
    print("=" * 50)
    
    # Test 1: Reference frame alignment
    results = test_reference_frame_fix()
    
    # Test 2: Bidirectional tracking
    bidirectional_works = test_bidirectional_tracking_fixed()
    
    # Create test video for visualization
    video = create_test_video_with_markers()
    
    # Visualize results
    visualize_fix_results(results, video)
    
    # Summary
    print("\n" + "=" * 50)
    print("FIX TEST SUMMARY")
    print("=" * 50)
    
    successful_tests = sum(1 for r in results.values() if r.get('success', False))
    total_tests = len(results)
    
    print(f"Reference frame tests: {successful_tests}/{total_tests} passed")
    print(f"Bidirectional tracking: {'✓ PASS' if bidirectional_works else '⚠ FAIL'}")
    
    if successful_tests == total_tests and bidirectional_works:
        print("🎉 ALL TESTS PASSED - Reference frame fix is working!")
    else:
        print("⚠ Some tests failed - more work needed")
    
    # Show alignment errors
    print("\nAlignment errors:")
    for ref_frame, result in results.items():
        if result.get('success', False):
            error = result['alignment_error']
            status = "✓ GOOD" if error < 5.0 else "⚠ HIGH"
            print(f"  Frame {ref_frame}: {error:.1f} pixels {status}")


if __name__ == "__main__":
    main()
