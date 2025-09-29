#!/usr/bin/env python3
"""
Core CoTracker functionality
============================

Handles CoTracker model loading, point tracking, and query generation.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import logging


class CoTrackerEngine:
    """Core CoTracker functionality for point tracking."""
    
    def __init__(self, device: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize CoTracker engine.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            logger: Logger instance (optional)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or logging.getLogger(__name__)
        self.cotracker_model = None
        
        self.logger.info(f"CoTracker engine initialized on device: {self.device}")
    
    def load_model(self):
        """Load CoTracker model from torch hub."""
        try:
            self.logger.info(f"Loading CoTracker model on {self.device}...")
            
            # Prioritize CoTracker3 offline for best bidirectional tracking
            try:
                self.cotracker_model = torch.hub.load(
                    "facebookresearch/co-tracker", 
                    "cotracker3_offline"
                ).to(self.device)
                self.logger.info("CoTracker3 offline model loaded successfully (best for bidirectional tracking)")
                
            except Exception as e:
                self.logger.warning(f"CoTracker3 offline failed: {e}")
                self.logger.info("Falling back to CoTracker2...")
                
                self.cotracker_model = torch.hub.load(
                    "facebookresearch/co-tracker", 
                    "cotracker2"
                ).to(self.device)
                self.logger.info("CoTracker2 model loaded successfully (with backward_tracking=True)")
                
        except Exception as e:
            self.logger.error(f"Error loading CoTracker model: {e}")
            raise
    
    def generate_grid_queries(self, video: np.ndarray, grid_size: int, 
                             reference_frame: int, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Generate grid query points for tracking.
        
        Args:
            video: Video array (T, H, W, C)
            grid_size: Grid size for point generation
            reference_frame: Reference frame index
            mask: Optional mask for filtering points
        
        Returns:
            Query tensor for CoTracker
        """
        # Ensure reference frame is within bounds
        reference_frame = max(0, min(reference_frame, video.shape[0] - 1))
        
        # Get video dimensions
        height, width = video.shape[1], video.shape[2]
        
        # Create grid points on the reference frame
        x_step = width / (grid_size - 1) if grid_size > 1 else width / 2
        y_step = height / (grid_size - 1) if grid_size > 1 else height / 2
        
        queries = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = int(j * x_step) if grid_size > 1 else width // 2
                y = int(i * y_step) if grid_size > 1 else height // 2
                
                # Ensure coordinates are within bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                
                # Add query: [frame_index, x, y]
                queries.append([reference_frame, x, y])
        
        # Convert to tensor format expected by CoTracker
        queries_tensor = torch.tensor([queries], dtype=torch.float32, device=self.device)
        
        self.logger.info(f"Generated {len(queries)} query points ({grid_size}x{grid_size}) on frame {reference_frame}")
        self.logger.debug(f"Query tensor shape before mask filtering: {queries_tensor.shape}")
        
        # Apply mask filtering if mask is available
        if mask is not None:
            queries_tensor = self._apply_mask_to_grid(queries_tensor, mask)
            self.logger.debug(f"Query tensor shape after mask filtering: {queries_tensor.shape}")
        
        return queries_tensor
    
    def _apply_mask_to_grid(self, queries: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        """
        Filter grid queries based on mask. Only keep points in white areas.
        
        Args:
            queries: Query tensor
            mask: Binary mask (white areas = trackable)
        
        Returns:
            Filtered query tensor
        """
        self.logger.debug(f"apply_mask_to_grid called - mask is None: {mask is None}")
        if mask is not None:
            self.logger.debug(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
            self.logger.debug(f"is_mask_empty result: {self._is_mask_empty(mask)}")
        
        if mask is None or self._is_mask_empty(mask):
            self.logger.debug("Returning original queries (no mask or empty mask)")
            return queries
        
        # Convert queries to numpy for processing
        queries_np = queries.cpu().numpy()
        
        # Get image dimensions
        H, W = mask.shape
        
        # Filter queries
        filtered_queries = []
        
        for batch_idx in range(queries_np.shape[0]):
            batch_queries = queries_np[batch_idx]  # Shape: (N, 3) - [t, x, y]
            
            valid_queries = []
            for query in batch_queries:
                t, x, y = query
                
                # Convert coordinates to mask indices
                mask_x = int(x)
                mask_y = int(y)
                
                # Check bounds
                if 0 <= mask_x < W and 0 <= mask_y < H:
                    # Check if point is in white area of mask
                    if mask[mask_y, mask_x] == 255:  # White pixel
                        valid_queries.append(query)
            
            if valid_queries:
                filtered_queries.append(np.array(valid_queries))
            else:
                # If no valid queries, keep at least one (center point)
                center_query = [batch_queries[0][0], W//2, H//2]
                filtered_queries.append(np.array([center_query]))
                self.logger.warning("No valid mask points found, using center point as fallback")
        
        # Convert back to tensor
        if filtered_queries:
            filtered_tensor = torch.tensor(filtered_queries, dtype=queries.dtype, device=queries.device)
        else:
            # Fallback to original if filtering failed
            filtered_tensor = queries
        
        original_points = queries.shape[1] if len(queries.shape) > 1 else 0
        filtered_points = filtered_tensor.shape[1] if len(filtered_tensor.shape) > 1 else 0
        self.logger.info(f"Mask filtering: {original_points} → {filtered_points} points")
        
        return filtered_tensor
    
    def _is_mask_empty(self, mask: np.ndarray) -> bool:
        """Check if mask is empty (no white pixels)."""
        if mask is None:
            return True
        
        white_pixels = np.sum(mask == 255)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage = (white_pixels / total_pixels) * 100
        
        # Consider mask empty if less than 0.1% coverage or less than 10 pixels
        min_pixels_threshold = 10
        percentage_threshold = 0.1
        
        is_empty = white_pixels < min_pixels_threshold and coverage < percentage_threshold
        self.logger.debug(f"is_mask_empty result: {is_empty} (white_pixels: {white_pixels}, coverage: {coverage:.2f}%)")
        
        return is_empty
    
    def track_points(self, video: np.ndarray, grid_size: int = 10, 
                    reference_frame: int = 0, mask: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track points in video using CoTracker.
        
        Args:
            video: Video array (T, H, W, C)
            grid_size: Grid size for point generation
            reference_frame: Reference frame for tracking
            mask: Optional mask for point filtering
        
        Returns:
            Tuple of (tracks, visibility) tensors
        """
        if self.cotracker_model is None:
            self.load_model()
        
        self.logger.info(f"Starting point tracking with grid_size={grid_size}, reference_frame={reference_frame}")
        
        # Convert video to tensor
        video_tensor = torch.tensor(video, dtype=torch.float32, device=self.device)
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)
        
        with torch.no_grad():
            if reference_frame == 0 and mask is None:
                # Use automatic grid on first frame with bidirectional tracking (only if no mask)
                pred_tracks, pred_visibility = self.cotracker_model(
                    video_tensor, 
                    grid_size=grid_size,
                    backward_tracking=True
                )
                self.logger.info("Used automatic grid generation (no mask)")
            else:
                # Use custom queries with mask filtering and bidirectional tracking
                mask_info = f" with mask ({np.sum(mask == 255)}/{mask.size} pixels)" if mask is not None else ""
                self.logger.info(f"Using custom reference frame: {reference_frame} with bidirectional tracking{mask_info}")
                
                queries = self.generate_grid_queries(video, grid_size, reference_frame, mask)
                self.logger.info(f"Generated {queries.shape[1] if len(queries.shape) > 1 else 0} query points for tracking")
                
                pred_tracks, pred_visibility = self.cotracker_model(
                    video_tensor, 
                    queries=queries,
                    backward_tracking=True
                )
        
        self.logger.info(f"Tracking completed - Tracks shape: {pred_tracks.shape}, Visibility shape: {pred_visibility.shape}")
        
        return pred_tracks, pred_visibility
