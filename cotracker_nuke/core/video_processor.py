#!/usr/bin/env python3
"""
Video processing utilities
==========================

Handles video loading, frame extraction, and video-related operations.
"""

import numpy as np
import imageio
import imageio.v3 as iio
from typing import Optional
import logging
from pathlib import Path


class VideoProcessor:
    """Handles video loading and processing operations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize video processor.
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.current_video = None
        self.video_path = None
    
    def load_video(self, video_path: str) -> np.ndarray:
        """
        Load video file and return as numpy array.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Video array with shape (T, H, W, C)
        """
        try:
            self.logger.info(f"Loading video: {video_path}")
            
            # Use imageio to load video
            video_data = []
            
            # Try different readers for better compatibility
            try:
                # First try with imageio v3 (newer API)
                with iio.imopen(video_path, 'r') as video_file:
                    for frame in video_file:
                        # Convert to RGB if needed
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            video_data.append(frame)
                        elif len(frame.shape) == 3 and frame.shape[2] == 4:
                            # Convert RGBA to RGB
                            video_data.append(frame[:, :, :3])
                        else:
                            # Handle grayscale
                            if len(frame.shape) == 2:
                                frame = np.stack([frame] * 3, axis=-1)
                            video_data.append(frame)
                            
            except Exception as e:
                self.logger.warning(f"imageio v3 failed: {e}, trying fallback...")
                
                # Fallback to older imageio API
                video_reader = imageio.get_reader(video_path)
                for frame in video_reader:
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        video_data.append(frame)
                    elif len(frame.shape) == 3 and frame.shape[2] == 4:
                        video_data.append(frame[:, :, :3])
                    else:
                        if len(frame.shape) == 2:
                            frame = np.stack([frame] * 3, axis=-1)
                        video_data.append(frame)
                video_reader.close()
            
            if not video_data:
                raise ValueError("No frames could be loaded from video")
            
            # Convert to numpy array
            video_array = np.array(video_data)
            
            # Store for future reference
            self.current_video = video_array
            self.video_path = video_path
            
            # Log video information
            self._log_video_info(video_array, video_path)
            
            return video_array
            
        except Exception as e:
            self.logger.error(f"Error loading video {video_path}: {str(e)}")
            raise
    
    def _log_video_info(self, video: np.ndarray, video_path: str = None):
        """Log detailed video information."""
        if video_path:
            self.logger.info(f"VIDEO INFORMATION:")
            self.logger.info(f"  Source: {video_path}")
        
        self.logger.info(f"  Shape: {video.shape}")
        self.logger.info(f"  Frames: {video.shape[0]}")
        self.logger.info(f"  Resolution: {video.shape[2]}x{video.shape[1]}")
        self.logger.info(f"  Channels: {video.shape[3]}")
        self.logger.info(f"  Data type: {video.dtype}")
        
        # Calculate memory usage
        memory_mb = video.nbytes / (1024 * 1024)
        self.logger.info(f"  Memory usage: {memory_mb:.1f} MB")
    
    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the loaded video.
        
        Args:
            frame_index: Frame index to retrieve
        
        Returns:
            Frame array or None if invalid index
        """
        if self.current_video is None:
            self.logger.error("No video loaded")
            return None
        
        if frame_index < 0 or frame_index >= len(self.current_video):
            self.logger.error(f"Frame index {frame_index} out of range (0-{len(self.current_video)-1})")
            return None
        
        return self.current_video[frame_index].copy()
    
    def get_video_info(self) -> dict:
        """
        Get information about the currently loaded video.
        
        Returns:
            Dictionary with video information
        """
        if self.current_video is None:
            return {}
        
        return {
            'path': self.video_path,
            'shape': self.current_video.shape,
            'frames': self.current_video.shape[0],
            'height': self.current_video.shape[1],
            'width': self.current_video.shape[2],
            'channels': self.current_video.shape[3],
            'dtype': str(self.current_video.dtype),
            'memory_mb': self.current_video.nbytes / (1024 * 1024)
        }
    
    def validate_reference_frame(self, reference_frame: int) -> int:
        """
        Validate and clamp reference frame to valid range.
        
        Args:
            reference_frame: Requested reference frame
        
        Returns:
            Valid reference frame index
        """
        if self.current_video is None:
            self.logger.warning("No video loaded, using frame 0")
            return 0
        
        max_frame = len(self.current_video) - 1
        
        if reference_frame < 0:
            self.logger.warning(f"Reference frame {reference_frame} < 0, using frame 0")
            return 0
        elif reference_frame > max_frame:
            self.logger.warning(f"Reference frame {reference_frame} > max frame {max_frame}, using frame {max_frame}")
            return max_frame
        
        return reference_frame
