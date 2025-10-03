#!/usr/bin/env python3
"""
STMap export functionality
=========================

Generates animated STMap sequences from CoTracker tracking data.
STMap (ST mapping) creates UV maps where pixel values represent source coordinates
for geometric transformations in compositing software like Nuke.
"""

import torch
import numpy as np
import cv2
from typing import Optional, Tuple, Union
import logging
from pathlib import Path
from datetime import datetime
from scipy.interpolate import griddata
import OpenEXR
import Imath


class STMapExporter:
    """Handles export of tracking data to animated STMap sequences."""
    
    def __init__(self, debug_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize STMap exporter.
        
        Args:
            debug_dir: Directory for export files
            logger: Logger instance (optional)
        """
        self.debug_dir = debug_dir or Path("outputs")
        self.debug_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.reference_frame = 0
        self.video_height = 1080
        self.video_width = 1920
    
    def set_reference_frame(self, reference_frame: int):
        """Set the reference frame for STMap generation."""
        self.reference_frame = reference_frame
    
    def set_video_dimensions(self, width: int, height: int):
        """Set video dimensions for coordinate conversion."""
        self.video_width = width
        self.video_height = height
    
    def generate_stmap_sequence(self, 
                              tracks: torch.Tensor, 
                              visibility: torch.Tensor,
                              mask: Optional[np.ndarray] = None,
                              interpolation_method: str = "linear",
                              bit_depth: int = 32,
                              frame_start: int = 0,
                              frame_end: Optional[int] = None,
                              filename_pattern: str = "stmap_%04d.exr",
                              frame_offset: int = 0,
                              progress_callback: Optional[callable] = None) -> str:
        """
        Generate animated STMap sequence from tracking data.
        
        Args:
            tracks: Tracking data tensor (1, T, N, 2)
            visibility: Visibility data tensor (1, T, N) or (1, T, N, 1)
            mask: Optional mask for limiting STMap generation
            interpolation_method: "linear", "cubic", or "delaunay"
            bit_depth: 16 or 32 for EXR bit depth
            frame_start: Starting frame for sequence (video-relative)
            frame_end: Ending frame for sequence (video-relative, None for all frames)
            filename_pattern: Pattern for output filenames (use %04d for frame number)
            frame_offset: Offset to add to frame numbers in filenames
            progress_callback: Optional callback function for progress updates (current_frame, total_frames)
        
        Returns:
            Path to output directory containing EXR sequence
        """
        self.logger.info("Starting STMap sequence generation...")
        self.logger.info(f"Interpolation method: {interpolation_method}")
        self.logger.info(f"Bit depth: {bit_depth}")
        self.logger.info(f"Frame range: {frame_start} to {frame_end or 'end'}")
        
        # Convert tensors to numpy
        tracks_np = tracks[0].cpu().numpy()  # Shape: (T, N, 2)
        visibility_np = visibility[0].cpu().numpy()  # Shape: (T, N) or (T, N, 1)
        
        # Handle different visibility shapes
        if len(visibility_np.shape) == 3:
            visibility_np = visibility_np[:, :, 0]
        
        T, N, _ = tracks_np.shape
        
        # Set frame range
        if frame_end is None:
            frame_end = T - 1
        
        # Create output directory if it doesn't exist
        self.debug_dir.mkdir(exist_ok=True)
        output_dir = self.debug_dir
        
        self.logger.info(f"Output directory: {output_dir}")
        
        # Calculate total frames to process
        total_frames = min(frame_end + 1, T) - frame_start
        processed_frames = 0
        
        # Generate STMap for each frame
        for frame_idx in range(frame_start, min(frame_end + 1, T)):
            self.logger.debug(f"Processing frame {frame_idx}/{T-1}")
            
            # Check if this is the reference frame
            if frame_idx == self.reference_frame:
                # For reference frame: create perfect identity gradient (ignore mask)
                stmap = self._generate_reference_frame_stmap(None)
            else:
                # For other frames: map from current positions to reference positions
                # Get tracking data for this frame
                frame_tracks = tracks_np[frame_idx]  # Shape: (N, 2)
                frame_visibility = visibility_np[frame_idx]  # Shape: (N,)
                
                # Filter visible points
                visible_mask = frame_visibility > 0.5
                if not np.any(visible_mask):
                    self.logger.warning(f"No visible points in frame {frame_idx}, skipping")
                    continue
                
                visible_tracks = frame_tracks[visible_mask]
                
                # Get reference frame tracks (user's selected reference frame)
                reference_frame_tracks = tracks_np[self.reference_frame]  # User's reference frame
                reference_visible_tracks = reference_frame_tracks[visible_mask]
                
                # Generate STMap for this frame (ignore mask)
                stmap = self._generate_frame_stmap(
                    visible_tracks, 
                    reference_visible_tracks,  # Use actual reference frame tracks
                    None,  # Ignore mask
                    interpolation_method
                )
            
            # Save as EXR using custom filename pattern with frame offset
            actual_frame_number = frame_idx + frame_offset
            frame_filename = filename_pattern % actual_frame_number
            frame_path = output_dir / frame_filename
            
            # Create metadata for this frame using standard OpenEXR keys
            frame_metadata = {
                'software': 'CoTracker Nuke Integration',
                'comment': f'ReferenceFrame:{self.reference_frame + frame_offset} CurrentFrame:{actual_frame_number} Interpolation:{interpolation_method} BitDepth:{bit_depth}-bit Points:{tracks_np.shape[1]}'
            }
            
            self._save_exr(stmap, frame_path, bit_depth, frame_metadata)
            
            # Update progress
            processed_frames += 1
            if progress_callback:
                progress_callback(processed_frames, total_frames)
        
        self.logger.info(f"STMap sequence generated: {output_dir}")
        return str(output_dir)
    
    def _generate_frame_stmap(self, 
                            current_tracks: np.ndarray,
                            reference_tracks: np.ndarray,
                            mask: Optional[np.ndarray],
                            interpolation_method: str) -> np.ndarray:
        """
        Generate STMap for a single frame.
        
        Args:
            current_tracks: Current frame track positions (N, 2) - where points are now
            reference_tracks: Reference frame track positions (N, 2) - where points were in user's reference frame
            mask: Optional mask for limiting generation
            interpolation_method: Interpolation method to use
        
        Returns:
            STMap array (H, W, 2) with normalized coordinates
        """
        # Create output grid
        height, width = self.video_height, self.video_width
        stmap = np.zeros((height, width, 2), dtype=np.float32)
        
        # Create coordinate grids for all pixels
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        grid_points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        # Apply mask if provided
        if mask is not None:
            # Resize mask to match video dimensions if needed
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Create mask for valid interpolation areas
            valid_mask = mask > 128  # White areas in mask
            valid_indices = np.where(valid_mask.ravel())[0]
            
            if len(valid_indices) == 0:
                self.logger.warning("No valid mask areas found, generating full STMap")
                valid_indices = np.arange(len(grid_points))
        else:
            valid_indices = np.arange(len(grid_points))
        
        # For STMap: we want to map from current frame positions to reference frame positions
        # Each pixel in the STMap should contain the reference frame coordinates to sample from
        
        # Interpolate reference frame coordinates for each current frame pixel
        if interpolation_method == "cubic":
            stmap_coords = self._interpolate_cubic(
                current_tracks, reference_tracks,
                grid_points[valid_indices]
            )
        else:  # linear (default and fallback)
            stmap_coords = self._interpolate_linear(
                current_tracks, reference_tracks,
                grid_points[valid_indices]
            )
        
        # Fill STMap with reference frame coordinates
        stmap.reshape(-1, 2)[valid_indices] = stmap_coords
        
        # Convert to Nuke coordinate system (bottom-left origin)
        stmap = self._convert_to_nuke_coordinates(stmap)
        
        return stmap
    
    def _generate_reference_frame_stmap(self, mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Generate perfect identity gradient STMap for reference frame.
        
        Args:
            mask: Optional mask for limiting generation
        
        Returns:
            STMap array (H, W, 2) with perfect gradient coordinates
        """
        # Create output grid
        height, width = self.video_height, self.video_width
        stmap = np.zeros((height, width, 2), dtype=np.float32)
        
        # Create perfect identity gradient
        # X coordinates: 0 to 1 horizontally
        for x in range(width):
            stmap[:, x, 0] = x / (width - 1) if width > 1 else 0.0
        
        # Y coordinates: 0 to 1 vertically (will be flipped for Nuke)
        for y in range(height):
            stmap[y, :, 1] = y / (height - 1) if height > 1 else 0.0
        
        # Apply mask if provided
        if mask is not None:
            # Resize mask to match video dimensions if needed
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Set areas outside mask to black
            valid_mask = mask > 128  # White areas in mask
            stmap[~valid_mask] = 0.0
        
        # Convert to Nuke coordinate system (bottom-left origin)
        # Coordinates are already normalized, so pass is_normalized=True
        stmap = self._convert_to_nuke_coordinates(stmap, is_normalized=True)
        
        return stmap
    
    def _interpolate_cubic(self, 
                         current_pos: np.ndarray, 
                         reference_pos: np.ndarray,
                         query_points: np.ndarray) -> np.ndarray:
        """Interpolate using cubic interpolation."""
        try:
            # Use griddata with cubic interpolation
            result = np.zeros((len(query_points), 2), dtype=np.float32)
            
            for i in range(2):  # X and Y coordinates
                interpolated = griddata(
                    current_pos, reference_pos[:, i], query_points,
                    method='cubic', fill_value=np.nan
                )
                result[:, i] = interpolated
            
            # Handle NaN values (areas outside convex hull) with nearest neighbor
            nan_mask = np.isnan(result).any(axis=1)
            if np.any(nan_mask):
                from scipy.spatial.distance import cdist
                nan_indices = np.where(nan_mask)[0]
                nan_points = query_points[nan_indices]
                distances = cdist(nan_points, current_pos)
                nearest_indices = np.argmin(distances, axis=1)
                result[nan_indices] = reference_pos[nearest_indices]
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Cubic interpolation failed: {e}, falling back to linear")
            return self._interpolate_linear(current_pos, reference_pos, query_points)
    
    def _interpolate_linear(self, 
                          current_pos: np.ndarray, 
                          reference_pos: np.ndarray,
                          query_points: np.ndarray) -> np.ndarray:
        """Interpolate using linear interpolation."""
        try:
            result = np.zeros((len(query_points), 2), dtype=np.float32)
            
            for i in range(2):  # X and Y coordinates
                interpolated = griddata(
                    current_pos, reference_pos[:, i], query_points,
                    method='linear', fill_value=np.nan
                )
                result[:, i] = interpolated
            
            # Handle NaN values (areas outside convex hull) with nearest neighbor
            nan_mask = np.isnan(result).any(axis=1)
            if np.any(nan_mask):
                from scipy.spatial.distance import cdist
                nan_indices = np.where(nan_mask)[0]
                nan_points = query_points[nan_indices]
                distances = cdist(nan_points, current_pos)
                nearest_indices = np.argmin(distances, axis=1)
                result[nan_indices] = reference_pos[nearest_indices]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Linear interpolation failed: {e}")
            # Fallback to nearest neighbor for all points
            from scipy.spatial.distance import cdist
            distances = cdist(query_points, current_pos)
            nearest_indices = np.argmin(distances, axis=1)
            return reference_pos[nearest_indices]
    
    def _convert_to_nuke_coordinates(self, stmap: np.ndarray, is_normalized: bool = False) -> np.ndarray:
        """
        Convert STMap coordinates to Nuke coordinate system.
        
        CoTracker uses top-left origin (0,0), Nuke uses bottom-left origin.
        
        Args:
            stmap: STMap array (H, W, 2) with coordinates
            is_normalized: If True, coordinates are already normalized (0-1)
        """
        height, width = stmap.shape[:2]
        
        if not is_normalized:
            # Normalize coordinates to 0-1 range
            stmap[:, :, 0] = stmap[:, :, 0] / width   # X coordinate (S)
            stmap[:, :, 1] = stmap[:, :, 1] / height  # Y coordinate (T)
        
        # Convert Y coordinate from top-left to bottom-left origin
        stmap[:, :, 1] = 1.0 - stmap[:, :, 1]
        
        return stmap
    
    def _save_exr(self, stmap: np.ndarray, filepath: Path, bit_depth: int, metadata: Optional[dict] = None):
        """
        Save STMap as EXR file with RGB channels for proper visualization.
        
        Args:
            stmap: STMap array (H, W, 2) with normalized coordinates
            filepath: Output file path
            bit_depth: 16 or 32 for bit depth
            metadata: Optional dictionary of metadata to add to EXR header
        """
        height, width = stmap.shape[:2]
        
        # Prepare RGB channels
        r_channel = stmap[:, :, 0].astype(np.float32)  # X coordinates (S) -> Red
        g_channel = stmap[:, :, 1].astype(np.float32)  # Y coordinates (T) -> Green
        b_channel = np.zeros((height, width), dtype=np.float32)  # Blue channel (black)
        
        # Set pixel type based on bit depth
        if bit_depth == 16:
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
            # Convert to half precision for 16-bit
            r_channel = r_channel.astype(np.float16)
            g_channel = g_channel.astype(np.float16)
            b_channel = b_channel.astype(np.float16)
        else:  # 32-bit
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Create EXR header with RGB channels
        header = OpenEXR.Header(width, height)
        header['channels'] = {
            'R': Imath.Channel(pixel_type),
            'G': Imath.Channel(pixel_type),
            'B': Imath.Channel(pixel_type)
        }
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                # Use proper OpenEXR metadata format
                header[key] = str(value)
        
        # Write EXR file
        exr_file = OpenEXR.OutputFile(str(filepath), header)
        exr_file.writePixels({
            'R': r_channel.tobytes(),
            'G': g_channel.tobytes(),
            'B': b_channel.tobytes()
        })
        exr_file.close()
        
        self.logger.debug(f"Saved EXR: {filepath}")
    
    def log_stmap_stats(self, tracks: torch.Tensor, visibility: torch.Tensor):
        """Log STMap generation statistics."""
        tracks_np = tracks[0].cpu().numpy()
        visibility_np = visibility[0].cpu().numpy()
        
        if len(visibility_np.shape) == 3:
            visibility_np = visibility_np[:, :, 0]
        
        num_frames, num_points = tracks_np.shape[0], tracks_np.shape[1]
        
        # Calculate per-frame visibility
        frame_visibility = np.mean(visibility_np > 0.5, axis=1) * 100
        
        self.logger.info("STMAP GENERATION STATS:")
        self.logger.info(f"  Total points: {num_points}")
        self.logger.info(f"  Total frames: {num_frames}")
        self.logger.info(f"  Average points per frame: {np.mean(frame_visibility) * num_points / 100:.1f}")
        self.logger.info(f"  Best frame visibility: {np.max(frame_visibility):.1f}%")
        self.logger.info(f"  Worst frame visibility: {np.min(frame_visibility):.1f}%")
        self.logger.info(f"  Video dimensions: {self.video_width}x{self.video_height}")
