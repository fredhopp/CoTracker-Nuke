#!/usr/bin/env python3
"""
Main CoTracker Nuke Application
===============================

Orchestrates all components for the CoTracker-Nuke integration.
"""

import torch
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any, List
import logging
from pathlib import Path
import os

from .tracker import CoTrackerEngine
from .video_processor import VideoProcessor
from .mask_handler import MaskHandler
from ..exporters.nuke_exporter import NukeExporter
from ..utils.logger import setup_logger


class CoTrackerNukeApp:
    """Main application class orchestrating all CoTracker-Nuke functionality."""
    
    def __init__(self, debug_mode: bool = True, console_log_level: str = "INFO"):
        """
        Initialize the CoTracker Nuke application.
        
        Args:
            debug_mode: Enable debug mode with file logging
            console_log_level: Console logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        # Set up directories
        self.debug_dir = Path("temp")
        self.debug_dir.mkdir(exist_ok=True)
        
        # Set up logging
        if debug_mode:
            self.logger = setup_logger("cotracker", self.debug_dir, console_log_level)
        else:
            self.logger = logging.getLogger("cotracker")
            self.logger.setLevel(logging.WARNING)
        
        # Initialize device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Device: {self.device}")
        
        # Initialize components
        self.tracker = CoTrackerEngine(self.device, self.logger)
        self.video_processor = VideoProcessor(self.logger)
        self.mask_handler = MaskHandler(self.debug_dir, self.logger)
        self.exporter = NukeExporter(self.debug_dir, self.logger)
        
        # Application state
        self.current_video = None
        self.tracking_results = None
        self.reference_frame = 0
        
        self.logger.info("CoTracker Nuke App initialized successfully")
    
    def load_video(self, video_path: str) -> np.ndarray:
        """
        Load video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Video array
        """
        self.current_video = self.video_processor.load_video(video_path)
        return self.current_video
    
    def set_reference_frame(self, frame_index: int) -> int:
        """
        Set and validate reference frame.
        
        Args:
            frame_index: Reference frame index
        
        Returns:
            Validated reference frame index
        """
        self.reference_frame = self.video_processor.validate_reference_frame(frame_index)
        self.exporter.set_reference_frame(self.reference_frame)
        return self.reference_frame
    
    def get_reference_frame_image(self) -> Optional[np.ndarray]:
        """Get the reference frame image for mask drawing."""
        if self.current_video is None:
            return None
        return self.video_processor.get_frame(self.reference_frame)
    
    def process_mask_from_editor(self, edited_image: Any) -> Tuple[str, np.ndarray]:
        """
        Process mask from Gradio ImageEditor.
        
        Args:
            edited_image: Output from Gradio ImageEditor
        
        Returns:
            Tuple of (status_message, mask_array)
        """
        try:
            mask = self.mask_handler.process_mask_from_editor(edited_image)
            
            if self.mask_handler.is_mask_empty(mask):
                return "⚠️ Mask is empty or too small. Please draw a larger mask area.", mask
            
            mask_path = self.mask_handler.save_mask(mask)
            stats = self.mask_handler.get_mask_stats(mask)
            
            message = (f"✅ Mask saved successfully!\n"
                      f"📊 Coverage: {stats['coverage_percent']:.1f}% "
                      f"({stats['white_pixels']} pixels)\n"
                      f"💾 Saved to: {mask_path}")
            
            return message, mask
            
        except Exception as e:
            error_msg = f"❌ Error processing mask: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, None
    
    def track_points(self, grid_size: int = 10, use_mask: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track points in the loaded video.
        
        Args:
            grid_size: Grid size for point generation
            use_mask: Whether to apply the current mask
        
        Returns:
            Tuple of (tracks, visibility) tensors
        """
        if self.current_video is None:
            raise ValueError("No video loaded")
        
        # Get mask if available and requested
        mask = self.mask_handler.current_mask if use_mask else None
        
        # Log tracking parameters
        self.logger.info(f"TRACKING PARAMETERS:")
        self.logger.info(f"  Grid size: {grid_size}")
        self.logger.info(f"  Reference frame: {self.reference_frame}")
        self.logger.info(f"  Using mask: {mask is not None}")
        if mask is not None:
            stats = self.mask_handler.get_mask_stats(mask)
            self.logger.info(f"  Mask coverage: {stats['coverage_percent']:.1f}%")
        
        # Perform tracking
        tracks, visibility = self.tracker.track_points(
            self.current_video, 
            grid_size, 
            self.reference_frame, 
            mask
        )
        
        # Store results
        self.tracking_results = (tracks, visibility)
        
        # Log results
        self.exporter.log_tracking_results(tracks, visibility)
        
        return tracks, visibility
    
    def export_to_nuke(self, output_path: str, frame_offset: int = 1001) -> str:
        """
        Export tracking results to Nuke .nk file.
        
        Args:
            output_path: Output path for .nk file
            frame_offset: Frame offset for image sequence start
        
        Returns:
            Path to generated .nk file
        """
        if self.tracking_results is None:
            raise ValueError("No tracking results available. Run track_points() first.")
        
        tracks, visibility = self.tracking_results
        
        # Generate CSV
        csv_path = self.exporter.generate_csv_for_nuke_export(tracks, visibility)
        
        # Export to Nuke
        nuke_path = self.exporter.export_to_nuke(csv_path, output_path, frame_offset)
        
        return nuke_path
    
    def get_corner_pin_points(self) -> list:
        """
        Get optimal points for corner pin tracking.
        
        Returns:
            List of point indices
        """
        if self.tracking_results is None:
            raise ValueError("No tracking results available. Run track_points() first.")
        
        tracks, visibility = self.tracking_results
        return self.exporter.select_corner_pin_points(tracks, visibility)
    
    def get_video_info(self) -> dict:
        """Get information about the loaded video."""
        return self.video_processor.get_video_info()
    
    def get_tracking_info(self) -> dict:
        """Get information about the current tracking results."""
        if self.tracking_results is None:
            return {}
        
        tracks, visibility = self.tracking_results
        tracks_np = tracks[0].cpu().numpy()
        visibility_np = visibility[0].cpu().numpy()
        
        if len(visibility_np.shape) == 3:
            visibility_np = visibility_np[:, :, 0]
        
        num_frames, num_points = tracks_np.shape[0], tracks_np.shape[1]
        total_detections = np.sum(visibility_np > 0.5)
        possible_detections = num_frames * num_points
        visibility_rate = (total_detections / possible_detections) * 100
        
        return {
            'num_points': num_points,
            'num_frames': num_frames,
            'total_detections': int(total_detections),
            'possible_detections': int(possible_detections),
            'visibility_rate': float(visibility_rate),
            'reference_frame': self.reference_frame
        }
    
    def create_preview_video(self, max_preview_points: int = 75) -> Optional[str]:
        """Create preview video showing tracked points over time."""
        if self.tracking_results is None or self.current_video is None:
            return None
            
        tracks, visibility = self.tracking_results
        
        try:
            self.logger.info(f"Creating preview video with {max_preview_points} points...")
            preview_path = self._create_preview_video(
                self.current_video, tracks, visibility, 
                max_preview_points, self.reference_frame
            )
            self.logger.info(f"Preview video created: {preview_path}")
            return str(preview_path)
        except Exception as e:
            self.logger.error(f"Failed to create preview video: {e}")
            return None
    
    def _create_preview_video(self, video: np.ndarray, tracks: torch.Tensor, 
                             visibility: torch.Tensor, max_preview_points: int = 75, reference_frame: int = 0) -> str:
        """Create preview video showing tracked points over time."""
        tracks_np = tracks[0].cpu().numpy()  # (T, N, 2)
        visibility_np = visibility[0].cpu().numpy()  # (T, N)
        
        # Handle different visibility shapes
        if len(visibility_np.shape) == 3:
            visibility_np = visibility_np[:, :, 0]
        
        # Select consistent points for preview
        total_points = tracks_np.shape[1]
        max_points_to_show = min(total_points, max_preview_points)
        
        # Use reference frame to select points with good spatial distribution
        ref_frame_tracks = tracks_np[reference_frame]  # (N, 2)
        ref_frame_visibility = visibility_np[reference_frame]  # (N,)
        
        # Get points visible at reference frame
        visible_mask = ref_frame_visibility > 0.5
        visible_indices = np.where(visible_mask)[0]
        
        if len(visible_indices) <= max_points_to_show:
            selected_point_indices = visible_indices.tolist()
        else:
            # Sample evenly distributed points
            step = len(visible_indices) // max_points_to_show
            selected_point_indices = visible_indices[::step][:max_points_to_show].tolist()
        
        self.logger.info(f"Selected {len(selected_point_indices)} points for preview")
        
        # Create frames with tracking overlays
        preview_frames = []
        
        # Limit frames for performance
        num_frames = min(video.shape[0], 150)
        step = max(1, video.shape[0] // num_frames)
        
        # Resize frames for better performance (max 720p)
        original_height, original_width = video.shape[1], video.shape[2]
        if original_width > 1280:
            scale_factor = 1280 / original_width
            new_width = 1280
            new_height = int(original_height * scale_factor)
        else:
            scale_factor = 1.0
            new_width = original_width
            new_height = original_height
        
        for i, frame_idx in enumerate(range(0, video.shape[0], step)):
            if len(preview_frames) >= num_frames:
                break
                
            frame = video[frame_idx].copy()
            
            # Resize frame if needed
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Add frame number overlay
            frame_text = f'Frame: {frame_idx}'
            cv2.putText(frame, frame_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)  # Black outline
            cv2.putText(frame, frame_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text
            
            # Draw tracking points
            if frame_idx < tracks_np.shape[0]:
                frame_tracks = tracks_np[frame_idx].copy()  # (N, 2)
                frame_visibility = visibility_np[frame_idx]  # (N,)
                
                # Scale tracking points if frame was resized
                if scale_factor != 1.0:
                    frame_tracks = frame_tracks * scale_factor
                
                # Draw selected points
                colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
                    (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128)
                ]
                
                for idx, j in enumerate(selected_point_indices):
                    if j < len(frame_tracks):
                        track = frame_tracks[j]
                        vis = frame_visibility[j]
                        
                        if vis > 0.5:  # Only draw visible points
                            x, y = int(track[0]), int(track[1])
                            if 0 <= x < new_width and 0 <= y < new_height:
                                color = colors[j % len(colors)]
                                cv2.circle(frame, (x, y), 2, color, -1)  # No outline - just the colored dot
            
            preview_frames.append(frame)
        
        # Save as temporary video file
        temp_video_path = self.debug_dir / f"cotracker_preview_{os.getpid()}.mp4"
        
        try:
            import imageio.v3 as iio
            iio.imwrite(
                temp_video_path, 
                np.array(preview_frames), 
                plugin="FFMPEG", 
                fps=24,
                codec='libx264',
                quality=8,
                ffmpeg_params=[
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart'
                ]
            )
            return str(temp_video_path)
        except Exception as e:
            self.logger.error(f"Error creating preview video: {e}")
            # Fallback to simpler settings
            try:
                import imageio.v3 as iio
                iio.imwrite(temp_video_path, np.array(preview_frames), plugin="FFMPEG", fps=24)
                return str(temp_video_path)
            except Exception as e2:
                self.logger.error(f"Fallback video creation failed: {e2}")
                raise
