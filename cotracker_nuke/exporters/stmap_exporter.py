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
    
    def generate_enhanced_stmap_sequence(self,
                                        tracks: torch.Tensor,
                                        visibility: torch.Tensor,
                                        mask: np.ndarray,
                                        interpolation_method: str = "linear",
                                        bit_depth: int = 32,
                                        frame_start: int = 0,
                                        frame_end: Optional[int] = None,
                                        frame_offset: int = 1001,
                                        progress_callback: Optional[callable] = None) -> str:
        """
        Generate enhanced STMap sequence with mask-aware intelligent interpolation.
        
        Args:
            tracks: Tracking data tensor (1, T, N, 2)
            visibility: Visibility tensor (1, T, N)
            mask: Original mask array (H, W)
            interpolation_method: "linear" or "cubic"
            bit_depth: 16 or 32 bit EXR output
            frame_start: First frame to export
            frame_end: Last frame to export (None for all)
            frame_offset: Frame number offset for filenames
            progress_callback: Progress callback function
            
        Returns:
            Path to output directory as string
        """
        try:
            self.logger.info(f"Starting enhanced STMap generation: tracks shape {tracks.shape}, visibility shape {visibility.shape}")
            
            # Convert tensors to numpy
            tracks_np = tracks[0].cpu().numpy()  # Shape: (T, N, 2)
            visibility_np = visibility[0].cpu().numpy()  # Shape: (T, N)
            
            # Handle different visibility shapes
            if len(visibility_np.shape) == 3:
                visibility_np = visibility_np[:, :, 0]
            
            T, N, _ = tracks_np.shape
            self.logger.info(f"Processed tracking data: {T} frames, {N} trackers")
            
            # Convert user frame numbers to 0-based indices
            # frame_start and frame_end are user input (e.g., 1001, 1089)
            # We need to convert them to 0-based indices for tracks_np (e.g., 0, 88)
            start_idx = max(0, frame_start - frame_offset)
            if frame_end is None:
                end_idx = T - 1
            else:
                end_idx = min(T - 1, frame_end - frame_offset)
            
            self.logger.info(f"User frame range: {frame_start} to {frame_end} (offset: {frame_offset})")
            self.logger.info(f"Tracking data range: {start_idx} to {end_idx} (0-based indices)")
            
            # Get reference frame tracks
            # self.reference_frame is already 0-based (set from UI)
            ref_frame_idx = self.reference_frame
            self.logger.info(f"Reference frame: 0-based index={ref_frame_idx}")
            
            # Ensure reference frame index is valid
            if ref_frame_idx < 0 or ref_frame_idx >= T:
                raise ValueError(f"Reference frame index {ref_frame_idx} is out of bounds for tracking data (0-{T-1})")
            
            reference_tracks = tracks_np[ref_frame_idx]
            reference_visibility = visibility_np[ref_frame_idx]
            
            # Filter visible reference trackers
            visible_mask = reference_visibility > 0.5
            if not np.any(visible_mask):
                raise ValueError("No visible trackers in reference frame")
            
            visible_reference_tracks = reference_tracks[visible_mask]
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.debug_dir / f"CoTracker_{timestamp}_enhanced_stmap"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            total_frames = end_idx - start_idx + 1
            processed_frames = 0
            
            # Generate enhanced STMap for each frame
            self.logger.info(f"Processing tracking frames {start_idx} to {end_idx} (total: {total_frames} frames)")
            for frame_idx in range(start_idx, end_idx + 1):
                if frame_idx % 10 == 0:  # Log every 10 frames
                    self.logger.info(f"Processing enhanced STMap frame {frame_idx}/{end_idx}")
                
                # Get current frame tracks
                current_tracks = tracks_np[frame_idx]
                current_visibility = visibility_np[frame_idx]
                
                # Filter visible current trackers (same indices as reference)
                visible_current_tracks = current_tracks[visible_mask]
                current_visibility_values = current_visibility[visible_mask]
                valid_trackers = current_visibility_values > 0.5
                
                if not np.any(valid_trackers):
                    # If no valid trackers, create identity STMap with original mask
                    stmap = self._generate_identity_stmap_with_mask(mask)
                else:
                    # Generate enhanced STMap with intelligent interpolation
                    stmap = self._generate_enhanced_frame_stmap(
                        mask, 
                        visible_reference_tracks, 
                        visible_current_tracks,
                        valid_trackers,
                        interpolation_method
                    )
                
                # Save as RGBA EXR
                actual_frame_number = frame_idx + frame_offset
                filename = f"CoTracker_{timestamp}_enhanced_stmap.{actual_frame_number:04d}.exr"
                frame_path = output_dir / filename
                
                self.logger.debug(f"Saving enhanced STMap frame {actual_frame_number} to {frame_path}")
                
                # Create metadata
                frame_metadata = {
                    'software': 'CoTracker Nuke Integration',
                    'comment': f'EnhancedSTMap ReferenceFrame:{self.reference_frame + frame_offset} CurrentFrame:{actual_frame_number} Interpolation:{interpolation_method} BitDepth:{bit_depth}-bit Points:{len(visible_reference_tracks)} MaskAware:True'
                }
                
                self._save_enhanced_exr(stmap, frame_path, bit_depth, frame_metadata)
                
                # Update progress
                processed_frames += 1
                if progress_callback:
                    progress_callback(processed_frames, total_frames)
            
            # Count generated files
            exr_files = list(output_dir.glob("*.exr"))
            self.logger.info(f"Enhanced STMap sequence generated: {output_dir} with {len(exr_files)} files")
            return str(output_dir)
            
        except Exception as e:
            self.logger.error(f"Enhanced STMap generation failed: {e}")
            raise
    
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
            
            # Handle NaN values (areas outside convex hull) with vectorized nearest neighbor
            nan_mask = np.isnan(result).any(axis=1)
            if np.any(nan_mask):
                nan_indices = np.where(nan_mask)[0]
                nan_points = query_points[nan_indices]
                
                # Vectorized approach: find closest tracker for all NaN points at once
                # Reshape for broadcasting: (N_nan, 1, 2) - (1, N_trackers, 2)
                nan_points_reshaped = nan_points[:, np.newaxis, :]  # (N_nan, 1, 2)
                current_pos_reshaped = current_pos[np.newaxis, :, :]  # (1, N_trackers, 2)
                
                # Calculate distances for all combinations at once
                distances = np.sqrt(np.sum((nan_points_reshaped - current_pos_reshaped)**2, axis=2))  # (N_nan, N_trackers)
                
                # Find closest tracker for each NaN point
                nearest_indices = np.argmin(distances, axis=1)  # (N_nan,)
                
                # Use reference position from closest tracker
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
            
            # Handle NaN values (areas outside convex hull) with vectorized nearest neighbor
            nan_mask = np.isnan(result).any(axis=1)
            if np.any(nan_mask):
                nan_indices = np.where(nan_mask)[0]
                nan_points = query_points[nan_indices]
                
                # Vectorized approach: find closest tracker for all NaN points at once
                # Reshape for broadcasting: (N_nan, 1, 2) - (1, N_trackers, 2)
                nan_points_reshaped = nan_points[:, np.newaxis, :]  # (N_nan, 1, 2)
                current_pos_reshaped = current_pos[np.newaxis, :, :]  # (1, N_trackers, 2)
                
                # Calculate distances for all combinations at once
                distances = np.sqrt(np.sum((nan_points_reshaped - current_pos_reshaped)**2, axis=2))  # (N_nan, N_trackers)
                
                # Find closest tracker for each NaN point
                nearest_indices = np.argmin(distances, axis=1)  # (N_nan,)
                
                # Use reference position from closest tracker
                result[nan_indices] = reference_pos[nearest_indices]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Linear interpolation failed: {e}")
            # Fallback to vectorized nearest neighbor for all points
            # Reshape for broadcasting: (N_points, 1, 2) - (1, N_trackers, 2)
            query_points_reshaped = query_points[:, np.newaxis, :]  # (N_points, 1, 2)
            current_pos_reshaped = current_pos[np.newaxis, :, :]  # (1, N_trackers, 2)
            
            # Calculate distances for all combinations at once
            distances = np.sqrt(np.sum((query_points_reshaped - current_pos_reshaped)**2, axis=2))  # (N_points, N_trackers)
            
            # Find closest tracker for each point
            nearest_indices = np.argmin(distances, axis=1)  # (N_points,)
            
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
    
    def _generate_identity_stmap_with_mask(self, mask: np.ndarray) -> np.ndarray:
        """Generate identity STMap with mask in alpha channel."""
        height, width = mask.shape
        
        # Create identity STMap (perfect gradient)
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Normalize coordinates to 0-1 range
        x_norm = x_coords.astype(np.float32) / (width - 1)
        y_norm = y_coords.astype(np.float32) / (height - 1)
        
        # Convert to Nuke coordinates (flip Y)
        y_nuke = 1.0 - y_norm
        
        # Create RGBA array: R=X, G=Y, B=0, A=mask
        stmap = np.zeros((height, width, 4), dtype=np.float32)
        stmap[:, :, 0] = x_norm  # R = X coordinates
        stmap[:, :, 1] = y_nuke  # G = Y coordinates (Nuke system)
        stmap[:, :, 2] = 0.0     # B = 0 (unused)
        stmap[:, :, 3] = mask.astype(np.float32) / 255.0  # A = mask (normalized)
        
        return stmap
    
    def _generate_enhanced_frame_stmap(self, 
                                     mask: np.ndarray,
                                     reference_tracks: np.ndarray,
                                     current_tracks: np.ndarray,
                                     valid_trackers: np.ndarray,
                                     interpolation_method: str) -> np.ndarray:
        """Generate enhanced STMap with intelligent interpolation."""
        try:
            # Use same dimensions as regular STMap
            height, width = self.video_height, self.video_width
            
            # Resize mask to match video dimensions if needed
            if mask.shape[:2] != (height, width):
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            
            # 1. First, warp the mask using the same logic as animated mask export
            warped_mask = self._warp_mask_with_trackers(mask_resized, reference_tracks, current_tracks)
            
            # 2. Create coordinate grids (same as regular STMap)
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
            
            # 3. Generate STMap coordinates using EXACT same logic as regular STMap
            # This ensures coordinates inside the hull are identical
            stmap_coords = np.zeros((len(points), 2), dtype=np.float32)
            
            if interpolation_method == "cubic":
                stmap_coords = self._interpolate_cubic(
                    current_tracks, reference_tracks, points
                )
            else:  # linear
                stmap_coords = self._interpolate_linear(
                    current_tracks, reference_tracks, points
                )
            
            # 4. Reshape to image dimensions (same as regular STMap)
            stmap_2d = stmap_coords.reshape(height, width, 2)
            
            # 5. Convert to Nuke coordinates (same as regular STMap)
            stmap_2d = self._convert_to_nuke_coordinates(stmap_2d)
            
            # 6. Create RGBA array: R=X, G=Y, B=0, A=warped_mask
            stmap = np.zeros((height, width, 4), dtype=np.float32)
            stmap[:, :, 0] = stmap_2d[:, :, 0]  # R = X coordinates (same as regular STMap)
            stmap[:, :, 1] = stmap_2d[:, :, 1]  # G = Y coordinates (same as regular STMap)
            stmap[:, :, 2] = 0.0                # B = 0 (unused)
            stmap[:, :, 3] = warped_mask.astype(np.float32) / 255.0  # A = warped mask (normalized)
            
            return stmap
            
        except Exception as e:
            self.logger.error(f"Enhanced frame STMap generation failed: {e}")
            # Fallback to identity STMap with original mask
            return self._generate_identity_stmap_with_mask(mask)
    
    def _warp_mask_with_trackers(self, mask: np.ndarray, reference_tracks: np.ndarray, current_tracks: np.ndarray) -> np.ndarray:
        """Warp mask based on tracker movement (same logic as animated mask export)."""
        try:
            height, width = mask.shape
            warped_mask = np.zeros_like(mask)
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
            
            # Calculate displacement vectors (reference - current) for backward mapping
            displacement_vectors = reference_tracks - current_tracks
            
            # Interpolate displacement vectors using linear interpolation
            interpolated_displacements = griddata(
                reference_tracks,
                displacement_vectors,
                points,
                method='linear',
                fill_value=np.nan
            )
            
            # Handle NaN values (outside convex hull) with vectorized nearest neighbor
            nan_mask = np.isnan(interpolated_displacements[:, 0])
            if np.any(nan_mask):
                nan_indices = np.where(nan_mask)[0]
                nan_points = points[nan_indices]
                
                # Vectorized approach: find closest tracker for all NaN points at once
                nan_points_reshaped = nan_points[:, np.newaxis, :]  # (N_nan, 1, 2)
                ref_tracks_reshaped = reference_tracks[np.newaxis, :, :]  # (1, N_trackers, 2)
                
                # Calculate distances for all combinations at once
                distances = np.sqrt(np.sum((nan_points_reshaped - ref_tracks_reshaped)**2, axis=2))  # (N_nan, N_trackers)
                
                # Find closest tracker for each NaN point
                closest_indices = np.argmin(distances, axis=1)  # (N_nan,)
                
                # Use displacement from closest tracker
                interpolated_displacements[nan_mask] = displacement_vectors[closest_indices]
            
            # Reshape interpolated displacements back to image shape
            dx = interpolated_displacements[:, 0].reshape(height, width)
            dy = interpolated_displacements[:, 1].reshape(height, width)
            
            # Create source coordinates
            source_x = x_coords + dx
            source_y = y_coords + dy
            
            # Warp the mask using vectorized bilinear interpolation
            valid_mask = (source_x >= 0) & (source_x < width-1) & (source_y >= 0) & (source_y < height-1)
            
            # For valid coordinates, use bilinear interpolation
            if np.any(valid_mask):
                # Get integer and fractional parts
                x1 = np.floor(source_x[valid_mask]).astype(int)
                y1 = np.floor(source_y[valid_mask]).astype(int)
                x2 = np.minimum(x1 + 1, width - 1)
                y2 = np.minimum(y1 + 1, height - 1)
                fx = source_x[valid_mask] - x1
                fy = source_y[valid_mask] - y1
                
                # Bilinear interpolation
                val = (mask[y1, x1] * (1-fx) * (1-fy) +
                       mask[y1, x2] * fx * (1-fy) +
                       mask[y2, x1] * (1-fx) * fy +
                       mask[y2, x2] * fx * fy)
                
                warped_mask[valid_mask] = val.astype(np.uint8)
            
            # For invalid coordinates, use original pixel
            warped_mask[~valid_mask] = mask[~valid_mask]
            
            return warped_mask
            
        except Exception as e:
            self.logger.error(f"Error warping mask: {e}")
            return mask  # Return original mask if warping fails
    
    def _save_enhanced_exr(self, stmap: np.ndarray, filepath: Path, bit_depth: int, metadata: Optional[dict] = None):
        """Save enhanced STMap as RGBA EXR file."""
        height, width = stmap.shape[:2]
        
        # Prepare RGBA channels
        r_channel = stmap[:, :, 0].astype(np.float32)  # X coordinates
        g_channel = stmap[:, :, 1].astype(np.float32)  # Y coordinates
        b_channel = stmap[:, :, 2].astype(np.float32)  # Blue channel (0)
        a_channel = stmap[:, :, 3].astype(np.float32)  # Alpha channel (warped mask)
        
        # Set pixel type based on bit depth
        if bit_depth == 16:
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
            r_channel = r_channel.astype(np.float16)
            g_channel = g_channel.astype(np.float16)
            b_channel = b_channel.astype(np.float16)
            a_channel = a_channel.astype(np.float16)
        else:  # 32-bit
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Create EXR header
        header = OpenEXR.Header(width, height)
        header['channels'] = {
            'R': Imath.Channel(pixel_type),
            'G': Imath.Channel(pixel_type),
            'B': Imath.Channel(pixel_type),
            'A': Imath.Channel(pixel_type)
        }
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                header[key] = str(value)
        
        # Write EXR file
        exr_file = OpenEXR.OutputFile(str(filepath), header)
        exr_file.writePixels({
            'R': r_channel.tobytes(),
            'G': g_channel.tobytes(),
            'B': b_channel.tobytes(),
            'A': a_channel.tobytes()
        })
        exr_file.close()
        
        self.logger.debug(f"Saved enhanced EXR: {filepath}")
