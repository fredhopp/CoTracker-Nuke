#!/usr/bin/env python3
"""
Nuke export functionality
=========================

Handles CSV generation and Nuke file creation for tracking data.
"""

import torch
import numpy as np
import csv
from typing import Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import os


class NukeExporter:
    """Handles export of tracking data to Nuke-compatible formats."""
    
    def __init__(self, debug_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize Nuke exporter.
        
        Args:
            debug_dir: Directory for export files
            logger: Logger instance (optional)
        """
        self.debug_dir = debug_dir or Path("temp")
        self.debug_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.reference_frame = 0  # Will be set during processing
    
    def set_reference_frame(self, reference_frame: int):
        """Set the reference frame for CSV generation."""
        self.reference_frame = reference_frame
    
    def generate_csv_for_nuke_export(self, tracks: torch.Tensor, visibility: torch.Tensor) -> str:
        """
        Generate CSV file with tracking data for Nuke export.
        
        Args:
            tracks: Tracking data tensor
            visibility: Visibility data tensor
        
        Returns:
            Path to generated CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.debug_dir / f"full_coords_{timestamp}_for_nuke.csv"
        
        # Convert tensors to numpy
        tracks_np = tracks[0].cpu().numpy()  # Shape: (T, N, 2)
        visibility_np = visibility[0].cpu().numpy()  # Shape: (T, N) or (T, N, 1)
        
        # Handle different visibility shapes
        if len(visibility_np.shape) == 3:
            visibility_np = visibility_np[:, :, 0]
        
        # Write CSV without frame offset (will be applied in .nk generation)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'point_id', 'x', 'y', 'visible', 'confidence', 'is_reference_frame'])
            
            num_frames, num_points = tracks_np.shape[0], tracks_np.shape[1]
            
            for frame_idx in range(num_frames):
                for point_id in range(num_points):
                    frame_number = frame_idx  # No frame offset applied here
                    x = tracks_np[frame_idx, point_id, 0]
                    y = tracks_np[frame_idx, point_id, 1]
                    visible = visibility_np[frame_idx, point_id] > 0.5
                    confidence = float(visibility_np[frame_idx, point_id])
                    is_reference_frame = frame_idx == self.reference_frame  # User's chosen reference frame
                    
                    writer.writerow([
                        frame_number, point_id, f"{x:.2f}", f"{y:.2f}", 
                        str(visible), f"{confidence:.3f}", str(is_reference_frame)
                    ])
        
        self.logger.info(f"Generated CSV for Nuke export (no frame offset applied): {csv_path}")
        return str(csv_path)
    
    def export_to_nuke(self, csv_path: str, output_path: str, frame_offset: int = 1001) -> str:
        """
        Export tracking data to Nuke .nk file.
        
        Args:
            csv_path: Path to CSV file with tracking data
            output_path: Output path for .nk file
            frame_offset: Frame offset for image sequence start
        
        Returns:
            Path to generated .nk file
        """
        try:
            self.logger.info(f"Exporting to Nuke file: {output_path}")
            self.logger.info(f"Using CSV: {csv_path}")
            self.logger.info(f"Frame offset: {frame_offset}")
            
            # Use the external script to generate the .nk file
            result = subprocess.run([
                'python', 
                'generate_exact_nuke_file.py',
                csv_path,
                output_path,
                str(frame_offset),
                str(self.reference_frame)
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            
            if result.returncode == 0:
                # Get the absolute path from the script output
                absolute_path = result.stdout.strip()
                self.logger.info(f"Successfully exported to Nuke: {absolute_path}")
                return absolute_path
            else:
                error_msg = f"Error generating Nuke file: {result.stderr}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            self.logger.error(f"Failed to export to Nuke: {str(e)}")
            raise
    
    def log_tracking_results(self, tracks: torch.Tensor, visibility: torch.Tensor):
        """Log tracking results statistics."""
        tracks_np = tracks[0].cpu().numpy()
        visibility_np = visibility[0].cpu().numpy()
        
        if len(visibility_np.shape) == 3:
            visibility_np = visibility_np[:, :, 0]
        
        num_frames, num_points = tracks_np.shape[0], tracks_np.shape[1]
        
        # Calculate visibility statistics
        total_detections = np.sum(visibility_np > 0.5)
        possible_detections = num_frames * num_points
        visibility_rate = (total_detections / possible_detections) * 100
        
        # Calculate per-point visibility
        per_point_visibility = np.mean(visibility_np > 0.5, axis=0) * 100
        
        self.logger.info("TRACKING RESULTS:")
        self.logger.info(f"  Points tracked: {num_points}")
        self.logger.info(f"  Frames: {num_frames}")
        self.logger.info(f"  Total detections: {total_detections}/{possible_detections}")
        self.logger.info(f"  Overall visibility: {visibility_rate:.1f}%")
        self.logger.info(f"  Best point visibility: {np.max(per_point_visibility):.1f}%")
        self.logger.info(f"  Worst point visibility: {np.min(per_point_visibility):.1f}%")
        self.logger.info(f"  Average point visibility: {np.mean(per_point_visibility):.1f}%")
    
    def select_corner_pin_points(self, tracks: torch.Tensor, visibility: torch.Tensor) -> list:
        """
        Select 4 optimal points for corner pin tracking.
        
        Args:
            tracks: Tracking data tensor
            visibility: Visibility data tensor
        
        Returns:
            List of point indices for corner pin
        """
        tracks_np = tracks[0].cpu().numpy()  # Shape: (T, N, 2)
        visibility_np = visibility[0].cpu().numpy()  # Shape: (T, N) or (T, N, 1)
        
        # Handle different visibility shapes
        if len(visibility_np.shape) == 3:
            visibility_np = visibility_np[:, :, 0]  # (T, N)
        
        T, N, _ = tracks_np.shape
        
        # Calculate visibility score for each point
        visibility_scores = np.mean(visibility_np > 0.5, axis=0)
        
        # Calculate position variance (stability)
        position_variance = np.var(tracks_np, axis=0)
        stability_scores = 1.0 / (1.0 + np.mean(position_variance, axis=1))
        
        # Combined score (visibility + stability)
        combined_scores = visibility_scores * stability_scores
        
        # Get top candidates
        top_candidates = np.argsort(combined_scores)[::-1][:min(12, N)]
        
        # Select 4 points that form a good quadrilateral
        if len(top_candidates) >= 4:
            selected_points = self._select_corner_points(
                tracks_np[self.reference_frame, top_candidates], 
                visibility_np[self.reference_frame, top_candidates],
                top_candidates
            )
        else:
            # Fallback: use all available points
            selected_points = top_candidates.tolist()
        
        self.logger.info(f"Selected corner pin points: {selected_points}")
        self.logger.info(f"Point visibility scores: {[f'{visibility_scores[i]:.3f}' for i in selected_points]}")
        
        return selected_points
    
    def _select_corner_points(self, positions: np.ndarray, visibility: np.ndarray, 
                            candidate_indices: np.ndarray) -> list:
        """
        Select 4 points that form a good quadrilateral.
        
        Args:
            positions: Point positions (N, 2)
            visibility: Point visibility (N,)
            candidate_indices: Original indices of candidate points
        
        Returns:
            List of 4 point indices
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy.optimize import linear_sum_assignment
        
        N = len(positions)
        if N <= 4:
            return candidate_indices.tolist()
        
        # Calculate pairwise distances
        distances = squareform(pdist(positions))
        
        # Find 4 points that maximize spread
        # Start with the two most distant points
        max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
        selected = list(max_dist_idx)
        
        # Add two more points to maximize area
        while len(selected) < 4:
            best_point = -1
            best_score = -1
            
            for i in range(N):
                if i in selected:
                    continue
                
                # Calculate minimum distance to already selected points
                min_dist = min(distances[i, j] for j in selected)
                
                # Prefer points with good visibility and distance
                score = min_dist * visibility[i]
                
                if score > best_score:
                    best_score = score
                    best_point = i
            
            if best_point >= 0:
                selected.append(best_point)
            else:
                break
        
        # Convert back to original indices
        return [candidate_indices[i] for i in selected[:4]]
