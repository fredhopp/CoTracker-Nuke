#!/usr/bin/env python3
"""
Test video loading functionality
"""

import numpy as np
import imageio.v3 as iio
import imageio
import cv2

def test_video_loading_methods():
    """Test different video loading methods."""
    
    # Create a simple test video
    print("Creating test video...")
    
    # Generate test frames
    frames = []
    for i in range(10):
        # Create a simple colored frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 25) % 255  # Red channel varies
        frame[:, :, 1] = 128  # Green constant
        frame[:, :, 2] = 255 - (i * 25) % 255  # Blue channel varies
        frames.append(frame)
    
    frames = np.array(frames)
    
    # Save test video
    test_video_path = "test_video.mp4"
    iio.imwrite(test_video_path, frames, plugin="FFMPEG", fps=24)
    print(f"✓ Created test video: {test_video_path}")
    
    # Test Method 1: imageio.v3.imread
    print("\nTesting Method 1: imageio.v3.imread")
    try:
        loaded_frames = iio.imread(test_video_path, plugin="FFMPEG")
        print(f"✓ Method 1 success: {loaded_frames.shape}")
    except Exception as e:
        print(f"✗ Method 1 failed: {e}")
    
    # Test Method 2: imageio.get_reader
    print("\nTesting Method 2: imageio.get_reader")
    try:
        reader = imageio.get_reader(test_video_path, 'ffmpeg')
        loaded_frames = []
        for frame in reader:
            loaded_frames.append(frame)
        reader.close()
        loaded_frames = np.array(loaded_frames)
        print(f"✓ Method 2 success: {loaded_frames.shape}")
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")
    
    # Test Method 3: OpenCV
    print("\nTesting Method 3: OpenCV")
    try:
        cap = cv2.VideoCapture(test_video_path)
        loaded_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            loaded_frames.append(frame)
        cap.release()
        loaded_frames = np.array(loaded_frames)
        print(f"✓ Method 3 success: {loaded_frames.shape}")
    except Exception as e:
        print(f"✗ Method 3 failed: {e}")
    
    print("\n🎉 Video loading tests completed!")
    
    # Clean up
    import os
    if os.path.exists(test_video_path):
        os.remove(test_video_path)
        print(f"✓ Cleaned up test file: {test_video_path}")

if __name__ == "__main__":
    test_video_loading_methods()
