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
from typing import Optional, Tuple, Union, Callable
import logging
from pathlib import Path
from datetime import datetime
from scipy.interpolate import griddata
import OpenEXR
import Imath
import multiprocessing as mp
import psutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import os
import pickle

# OpenBLAS environment variables are set in the batch file to avoid threading issues

def _process_frame_standalone(frame_data: dict) -> Tuple[int, str]:
    """
    Standalone function for processing a single frame in a separate process.
    This function must be at module level to be picklable for ProcessPoolExecutor.
    """
    try:
        # Import here to avoid circular imports
        from .stmap_exporter import STMapExporter
        
        frame_idx = frame_data['frame_idx']
        mask = frame_data['mask']
        visible_reference_tracks = frame_data['visible_reference_tracks']
        visible_current_tracks = frame_data['visible_current_tracks']
        valid_trackers = frame_data['valid_trackers']
        interpolation_method = frame_data['interpolation_method']
        frame_offset = frame_data['frame_offset']
        output_dir = frame_data['output_dir']
        output_file_path = frame_data['output_file_path']
        timestamp = frame_data['timestamp']
        reference_frame = frame_data['reference_frame']
        
        # Create a temporary exporter instance for processing
        exporter = STMapExporter()
        # Set the required attributes from the frame data
        exporter.reference_frame = frame_data.get('reference_frame', 0)
        exporter.video_height = frame_data.get('video_height', 1080)
        exporter.video_width = frame_data.get('video_width', 1920)
        
        # Generate STMap for this frame
        stmap = exporter._generate_frame_stmap(
            mask, 
            visible_reference_tracks, 
            visible_current_tracks,
            valid_trackers,
            interpolation_method
        )
        
        # Save as RGBA EXR
        actual_frame_number = frame_idx + frame_offset
        if output_file_path and "%04d" in output_file_path:
            filename = Path(output_file_path).name % actual_frame_number
        else:
            filename = f"CoTracker_{timestamp}_stmap.{actual_frame_number:04d}.exr"
        frame_path = output_dir / filename
        
        # Save EXR file (no metadata needed)
        exporter._save_exr(stmap, frame_path, 32, None)
        
        return frame_idx, str(frame_path)
        
    except Exception as e:
        # Return error information
        return frame_data.get('frame_idx', -1), f"ERROR: {str(e)}"


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
    
    def _get_system_resources(self) -> dict:
        """Get current system resource information."""
        return {
            'cpu_count': mp.cpu_count(),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(interval=1)
        }
    
    def _analyze_single_frame_performance(self, 
                                        mask: np.ndarray,
                                        visible_reference_tracks: np.ndarray,
                                        visible_current_tracks: np.ndarray,
                                        valid_trackers: np.ndarray,
                                        interpolation_method: str) -> dict:
        """Analyze performance of processing a single frame to determine optimal parallelization."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        # Process one frame
        stmap = self._generate_frame_stmap(
            mask, 
            visible_reference_tracks, 
            visible_current_tracks,
            valid_trackers,
            interpolation_method
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        processing_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        return {
            'processing_time': processing_time,
            'memory_used_mb': memory_used,
            'frame_size_mb': stmap.nbytes / (1024**2)
        }
    
    def _calculate_optimal_parallelization(self, 
                                         total_frames: int,
                                         single_frame_analysis: dict,
                                         system_resources: dict) -> dict:
        """Calculate optimal number of parallel workers based on system resources."""
        # Use up to 80% of available resources
        max_memory_usage = system_resources['available_memory_gb'] * 0.8
        max_cpu_usage = system_resources['cpu_count'] * 0.8
        
        # Calculate memory-based limit
        memory_per_frame_gb = single_frame_analysis['memory_used_mb'] / 1024
        memory_limited_workers = int(max_memory_usage / memory_per_frame_gb) if memory_per_frame_gb > 0 else 1
        
        # Calculate CPU-based limit
        cpu_limited_workers = int(max_cpu_usage)
        
        # For ProcessPoolExecutor, use more conservative memory limits
        # Each process has its own Python interpreter and memory space
        process_memory_multiplier = 3  # Each process uses ~3x more memory than a thread
        process_memory_limited_workers = int(memory_limited_workers / process_memory_multiplier)
        
        # Use the more restrictive limit for process-based parallelization
        optimal_workers = min(process_memory_limited_workers, cpu_limited_workers, total_frames)
        optimal_workers = max(1, optimal_workers)  # At least 1 worker
        
        return {
            'optimal_workers': optimal_workers,
            'memory_limited_workers': memory_limited_workers,
            'cpu_limited_workers': cpu_limited_workers,
            'estimated_total_memory_gb': memory_per_frame_gb * optimal_workers,
            'estimated_processing_time': single_frame_analysis['processing_time'] * total_frames / optimal_workers if optimal_workers > 0 else single_frame_analysis['processing_time'] * total_frames
        }
    
    def _process_frame_parallel(self, 
                              frame_data: dict) -> Tuple[int, str]:
        """Process a single frame in parallel. Returns (frame_idx, output_path)."""
        try:
            frame_idx = frame_data['frame_idx']
            mask = frame_data['mask']
            visible_reference_tracks = frame_data['visible_reference_tracks']
            visible_current_tracks = frame_data['visible_current_tracks']
            valid_trackers = frame_data['valid_trackers']
            interpolation_method = frame_data['interpolation_method']
            frame_offset = frame_data['frame_offset']
            output_dir = frame_data['output_dir']
            output_file_path = frame_data['output_file_path']
            timestamp = frame_data['timestamp']
            reference_frame = frame_data['reference_frame']
            
            # Get current thread ID for debugging
            thread_id = threading.current_thread().ident
            
            # Debug logging for first few frames with thread ID
            if frame_idx <= 2:
                self.logger.debug(f"🔄 Thread {thread_id}: Processing frame {frame_idx} (0-based) with {len(visible_current_tracks)} trackers")
            
            # Generate STMap for this frame
            stmap = self._generate_frame_stmap(
                mask, 
                visible_reference_tracks, 
                visible_current_tracks,
                valid_trackers,
                interpolation_method
            )
            
            # Save as RGBA EXR
            actual_frame_number = frame_idx + frame_offset
            if output_file_path and "%04d" in output_file_path:
                filename = Path(output_file_path).name % actual_frame_number
            else:
                filename = f"CoTracker_{timestamp}_stmap.{actual_frame_number:04d}.exr"
            frame_path = output_dir / filename
            
            # Debug logging for first few frames with thread ID
            if frame_idx <= 2:
                self.logger.debug(f"💾 Thread {thread_id}: Saving frame {frame_idx} as {filename} (display frame {actual_frame_number})")
            
            # Save EXR file (no metadata needed)
            self._save_exr(stmap, frame_path, 32, None)
            
            # Debug logging for first few frames with thread ID
            if frame_idx <= 2:
                self.logger.debug(f"✅ Thread {thread_id}: Saved frame {frame_idx} to {frame_path}")
            
            return frame_idx, str(frame_path), thread_id
            
        except Exception as e:
            thread_id = threading.current_thread().ident
            self.logger.error(f"Thread {thread_id}: Error processing frame {frame_data.get('frame_idx', 'unknown')}: {e}")
            raise
    
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
            
            # Save EXR file (no metadata needed)
            self._save_exr(stmap, frame_path, bit_depth, None)
            
            # Update progress
            processed_frames += 1
            if progress_callback:
                progress_callback(processed_frames, total_frames)
        
        self.logger.info(f"STMap sequence generated: {output_dir}")
        return str(output_dir)
    
    def generate_stmap_sequence(self,
                               tracks: torch.Tensor,
                               visibility: torch.Tensor,
                               mask: np.ndarray,
                               interpolation_method: str = "linear",
                               bit_depth: int = 32,
                               frame_start: int = 0,
                               frame_end: Optional[int] = None,
                               frame_offset: int = 1001,
                               output_file_path: str = None,
                               progress_callback: Optional[callable] = None) -> str:
        """
        Generate STMap sequence with mask-aware intelligent interpolation.
        
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
            self.logger.info(f"Starting STMap generation: tracks shape {tracks.shape}, visibility shape {visibility.shape}")
            
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
            
            # Create output directory from provided path or generate default
            if output_file_path is None or output_file_path.strip() == "":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file_path = f"outputs/CoTracker_{timestamp}_stmap/CoTracker_{timestamp}_stmap.%04d.exr"
            else:
                # Generate timestamp for filename fallback
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Extract directory from file path pattern
            output_dir = Path(output_file_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            total_frames = end_idx - start_idx + 1
            processed_frames = 0
            
            # Stage 1: Show analyzing first frame message
            if progress_callback:
                progress_callback(0, total_frames, "🔍 Analyzing first frame performance to estimate processing time...")
            
            # Get system resources for parallel processing optimization
            self.logger.info("Analyzing system resources for parallel processing...")
            system_resources = self._get_system_resources()
            self.logger.info(f"System resources: {system_resources['cpu_count']} CPUs, "
                           f"{system_resources['available_memory_gb']:.1f}GB available memory, "
                           f"{system_resources['memory_percent']:.1f}% memory used")
            
            # Analyze single frame performance for optimization and save the first frame
            self.logger.info("🔍 Analyzing single frame performance to estimate total processing time...")
            self.logger.info("📊 This first frame will be processed to measure CPU/RAM usage and calculate optimal parallelization")
            test_frame_idx = start_idx
            current_tracks = tracks_np[test_frame_idx]
            current_visibility = visibility_np[test_frame_idx]
            visible_current_tracks = current_tracks[visible_mask]
            current_visibility_values = current_visibility[visible_mask]
            valid_trackers = current_visibility_values > 0.5
            
            single_frame_analysis = self._analyze_single_frame_performance(
                mask, visible_reference_tracks, visible_current_tracks, valid_trackers, interpolation_method
            )
            
            # Save the first frame that was already processed during performance analysis
            first_frame_stmap = self._generate_frame_stmap(
                mask, visible_reference_tracks, visible_current_tracks, valid_trackers, interpolation_method
            )
            
            # Save first frame EXR
            first_frame_number = test_frame_idx + frame_offset
            if output_file_path and "%04d" in output_file_path:
                first_filename = Path(output_file_path).name % first_frame_number
            else:
                first_filename = f"CoTracker_{timestamp}_stmap.{first_frame_number:04d}.exr"
            first_frame_path = output_dir / first_filename
            
            # Save EXR file (no metadata needed)
            self._save_exr(first_frame_stmap, first_frame_path, bit_depth, None)
            self.logger.info(f"💾 First frame written to disk: {first_frame_path}")
            self.logger.info(f"✅ Saved first frame {first_frame_number} during performance analysis: {first_frame_path}")
            processed_frames += 1
            
            # Calculate optimal parallelization
            parallelization_info = self._calculate_optimal_parallelization(
                total_frames, single_frame_analysis, system_resources
            )
            
            self.logger.info(f"Performance analysis: {single_frame_analysis['processing_time']:.2f}s per frame, "
                           f"{single_frame_analysis['memory_used_mb']:.1f}MB memory per frame")
            self.logger.info(f"Optimal parallelization: {parallelization_info['optimal_workers']} workers "
                           f"(memory-limited: {parallelization_info['memory_limited_workers']}, "
                           f"CPU-limited: {parallelization_info['cpu_limited_workers']})")
            self.logger.info(f"Estimated total memory usage: {parallelization_info['estimated_total_memory_gb']:.1f}GB")
            self.logger.info(f"Estimated processing time: {parallelization_info['estimated_processing_time']:.1f}s")
            
            # Calculate more realistic time estimate (accounting for I/O bottleneck)
            # EXR file writing is the main bottleneck, not CPU computation
            single_frame_time = single_frame_analysis['processing_time']
            remaining_frames = total_frames - 1  # First frame already processed
            optimal_workers = parallelization_info['optimal_workers']
            
            # For ProcessPoolExecutor, assume better parallelization for CPU-bound computation
            # STMap generation is CPU-intensive, EXR writing is I/O bound
            # Based on actual performance data: 2.8 fps with 51 workers = ~32% efficiency
            cpu_efficiency = 0.3  # 30% efficiency (realistic based on actual performance)
            parallel_processing_time = single_frame_time * remaining_frames / (optimal_workers * cpu_efficiency)
            
            # Total estimate includes first frame time + parallel processing time
            realistic_estimate = single_frame_time + parallel_processing_time
            realistic_minutes = int(realistic_estimate // 60)
            realistic_seconds = int(realistic_estimate % 60)
            self.logger.info(f"📊 Realistic processing estimate: {realistic_minutes:02d}:{realistic_seconds:02d} (includes parallel overhead)")
            
            # Show initial estimate after first frame analysis
            if progress_callback:
                # Convert to minutes:seconds format
                eta_minutes = int(realistic_estimate // 60)
                eta_secs = int(realistic_estimate % 60)
                eta_str = f"{eta_minutes:02d}:{eta_secs:02d}"
                
                # Stage 2: Show estimate after first frame analysis
                progress_callback(0, total_frames, f"🚀 Starting STMap generation... Estimated time: {eta_str}")
            
            # Prepare frame data for parallel processing (skip first frame - already processed)
            frame_data_list = []
            for frame_idx in range(start_idx + 1, end_idx + 1):  # Skip first frame (start_idx + 1)
                current_tracks = tracks_np[frame_idx]
                current_visibility = visibility_np[frame_idx]
                visible_current_tracks = current_tracks[visible_mask]
                current_visibility_values = current_visibility[visible_mask]
                valid_trackers = current_visibility_values > 0.5
                
                frame_data = {
                    'frame_idx': frame_idx,
                    'mask': mask,
                    'visible_reference_tracks': visible_reference_tracks,
                    'visible_current_tracks': visible_current_tracks,
                    'valid_trackers': valid_trackers,
                    'interpolation_method': interpolation_method,
                    'frame_offset': frame_offset,
                    'output_dir': output_dir,
                    'output_file_path': output_file_path,
                    'timestamp': timestamp,
                    'reference_frame': self.reference_frame,
                    'video_height': self.video_height,
                    'video_width': self.video_width
                }
                frame_data_list.append(frame_data)
            
            self.logger.info(f"Skipping first frame (already processed), processing remaining {len(frame_data_list)} frames in parallel")
            
            # Process frames in parallel
            self.logger.info(f"Processing {len(frame_data_list)} remaining frames with {parallelization_info['optimal_workers']} parallel workers...")
            self.logger.info(f"Frame range: {start_idx + 1} to {end_idx} (0-based), frame offset: {frame_offset}")
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=parallelization_info['optimal_workers']) as executor:
                # Submit all frame processing tasks
                future_to_frame = {
                    executor.submit(_process_frame_standalone, frame_data): frame_data['frame_idx'] 
                    for frame_data in frame_data_list
                }
                
            self.logger.info(f"🚀 Submitted {len(future_to_frame)} frame processing tasks to {parallelization_info['optimal_workers']} worker processes")
            self.logger.info(f"📊 Process pool will process frames in true parallel - no GIL limitations")
            self.logger.info(f"⏱️  Processing will start immediately with {parallelization_info['optimal_workers']} parallel processes")
            
            # Track process completion for debugging
            completed_processes = set()
            
            # Process completed frames as they finish (true parallel processing)
            
            for future in as_completed(future_to_frame):
                try:
                    frame_idx, output_path = future.result()
                    processed_frames += 1
                    
                    # Track process completion
                    completed_processes.add(frame_idx)
                    
                    # Check for errors in the result
                    if output_path.startswith("ERROR:"):
                        self.logger.error(f"Process error for frame {frame_idx}: {output_path}")
                        raise Exception(f"Frame {frame_idx} processing failed: {output_path}")
                    
                    # Log first few frames for debugging
                    if processed_frames <= 3:
                        current_frame_number = frame_idx + frame_offset
                        self.logger.info(f"✅ Process completed frame {processed_frames}/{total_frames} (0-based: {frame_idx}, display: {current_frame_number}) -> {output_path}")
                    
                    # Log progress (no UI updates during processing)
                    self.logger.info(f"📊 Progress: {processed_frames}/{total_frames} frames completed")
                    
                    if processed_frames % 10 == 0:  # Log every 10 frames
                        current_frame_number = frame_idx + frame_offset
                        self.logger.info(f"Completed {processed_frames}/{total_frames} frames (frame {current_frame_number}) - Active processes: {len(completed_processes)}")
                        
                except Exception as e:
                    frame_idx = future_to_frame[future]
                    self.logger.error(f"Error processing frame {frame_idx}: {e}")
                    raise
            
            end_time = time.time()
            total_processing_time = end_time - start_time
            self.logger.info(f"Parallel processing completed in {total_processing_time:.1f}s "
                           f"({total_frames/total_processing_time:.1f} frames/second)")
            self.logger.info(f"🔄 Process usage summary: {len(completed_processes)} frames processed using {parallelization_info['optimal_workers']} worker processes")
            
            # Count generated files
            exr_files = list(output_dir.glob("*.exr"))
            self.logger.info(f"STMap sequence generated: {output_dir} with {len(exr_files)} files")
            
            # Show final completion message
            if progress_callback:
                progress_callback(total_frames, total_frames, f"✅ STMap sequence generated in {total_processing_time:.1f} seconds.")
            
            return str(output_dir)
            
        except Exception as e:
            self.logger.error(f"STMap generation failed: {e}")
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
            if width > 0:
                stmap[:, :, 0] = stmap[:, :, 0] / width   # X coordinate (S)
            if height > 0:
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
        
        # Add standard OpenEXR attributes
        header['pixelAspectRatio'] = 1.0
        header['screenWindowCenter'] = Imath.V2f(0.0, 0.0)
        header['screenWindowWidth'] = 1.0
        header['lineOrder'] = OpenEXR.INCREASING_Y
        header['compression'] = OpenEXR.ZIP_COMPRESSION
        
        # Add custom metadata if provided
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
    
    def _generate_frame_stmap(self, 
                            mask: np.ndarray,
                            reference_tracks: np.ndarray,
                            current_tracks: np.ndarray,
                            valid_trackers: np.ndarray,
                            interpolation_method: str) -> np.ndarray:
        """Generate STMap with intelligent interpolation."""
        try:
            # Use same dimensions as regular STMap
            height, width = self.video_height, self.video_width
            
            # Resize mask to match video dimensions if needed
            if mask.shape[:2] != (height, width):
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            
            # 1. First, warp the mask using the same segment-based algorithm
            # Note: We warp the original mask, not the resized one, to avoid double-masking
            warped_mask = self._warp_mask_with_segment_algorithm(mask, reference_tracks, current_tracks)
            
            # 2. Calculate smart processing bounds using current frame trackers
            min_x, min_y, max_x, max_y = self._calculate_processing_bounds(mask, current_tracks)
            
            # 3. Create coordinate grids only for the bounding box area
            y_coords, x_coords = np.mgrid[min_y:max_y+1, min_x:max_x+1]
            points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
            
            # 4. Only process pixels inside the warped mask (much faster!)
            warped_mask_bool = warped_mask[min_y:max_y+1, min_x:max_x+1] > 0
            warped_mask_indices = np.where(warped_mask_bool.ravel())[0]
            
            
            # Initialize STMap coordinates (will remain 0 for pixels outside warped mask)
            stmap_coords = np.zeros((len(points), 2), dtype=np.float32)
            
            if len(warped_mask_indices) > 0:
                self.logger.debug(f"Processing {len(warped_mask_indices)} pixels inside warped mask")
                
                # Get pixels inside the warped mask
                warped_pixels = points[warped_mask_indices]
                
                # First, try regular interpolation for pixels inside hull
                if interpolation_method == "cubic":
                    interpolated_coords = self._interpolate_cubic(
                        current_tracks, reference_tracks, warped_pixels
                    )
                else:  # linear
                    interpolated_coords = self._interpolate_linear(
                        current_tracks, reference_tracks, warped_pixels
                    )
                
                # Identify pixels inside/outside hull using Delaunay triangulation
                inside_hull_mask = self._is_inside_delaunay_hull(warped_pixels, current_tracks)
                inside_hull_indices = np.where(inside_hull_mask)[0]
                outside_hull_indices = np.where(~inside_hull_mask)[0]
                
                self.logger.debug(f"Hull detection: {len(outside_hull_indices)} pixels outside hull, {len(inside_hull_indices)} pixels inside hull")
                
                # Set coordinates for pixels inside hull
                if len(inside_hull_indices) > 0:
                    stmap_coords[warped_mask_indices[inside_hull_indices]] = interpolated_coords[inside_hull_indices]
                
                # Apply segment-based algorithm to pixels outside hull
                if len(outside_hull_indices) > 0:
                    self.logger.debug(f"Processing {len(outside_hull_indices)} pixels outside hull with segment-based algorithm")
                    
                    outside_hull_pixels = warped_pixels[outside_hull_indices]
                    outside_hull_coords = self._calculate_fringe_coordinates(
                        outside_hull_pixels, reference_tracks, current_tracks
                    )
                    
                    # Update the coordinates for pixels outside hull
                    stmap_coords[warped_mask_indices[outside_hull_indices]] = outside_hull_coords
            
            # 4. Reshape to bounding box dimensions and place in full image
            stmap_2d = np.zeros((height, width, 2), dtype=np.float32)
            stmap_bbox = stmap_coords.reshape(max_y - min_y + 1, max_x - min_x + 1, 2)
            stmap_2d[min_y:max_y+1, min_x:max_x+1] = stmap_bbox
            
            # 5. Convert to Nuke coordinates
            stmap_2d = self._convert_to_nuke_coordinates(stmap_2d)
            
            # 6. Create RGBA array: R=X, G=Y, B=0, A=warped_mask
            stmap = np.zeros((height, width, 4), dtype=np.float32)
            
            # Set STMap coordinates for the bounding box area
            stmap[min_y:max_y+1, min_x:max_x+1, 0] = stmap_2d[min_y:max_y+1, min_x:max_x+1, 0]  # R = X coordinates
            stmap[min_y:max_y+1, min_x:max_x+1, 1] = stmap_2d[min_y:max_y+1, min_x:max_x+1, 1]  # G = Y coordinates
            stmap[min_y:max_y+1, min_x:max_x+1, 2] = 0.0                                        # B = 0 (unused)
            stmap[:, :, 3] = warped_mask.astype(np.float32) / 255.0                             # A = warped mask (all pixels)
            
            return stmap
            
        except Exception as e:
            self.logger.error(f"Frame STMap generation failed: {e}")
            # Fallback to identity STMap with original mask
            return self._generate_identity_stmap_with_mask(mask)
    
    def _calculate_fringe_coordinates(self, 
                                    pixel_coords: np.ndarray,
                                    reference_tracks: np.ndarray,
                                    current_tracks: np.ndarray) -> np.ndarray:
        """
        Calculate STMap coordinates for fringe pixels using segment-based algorithm.
        
        For each pixel C' in current frame:
        1. Find closest tracker segment A'B' (two nearest trackers)
        2. Project C' perpendicularly onto A'B' → get point P'
        3. Map P' to reference frame: P = A + t * (B - A) where t = A'P' / A'B'
        4. Construct perpendicular through P in reference frame
        5. Find C on perpendicular: PC = (|AB| / |A'B'|) * |P'C'|
        6. Return C's coordinates for STMap
        
        Args:
            pixel_coords: Array of pixel coordinates (N, 2) in current frame
            reference_tracks: Reference frame tracker positions (M, 2)
            current_tracks: Current frame tracker positions (M, 2)
            
        Returns:
            Array of corresponding coordinates in reference frame (N, 2)
        """
        try:
            num_pixels = len(pixel_coords)
            result_coords = np.zeros((num_pixels, 2), dtype=np.float32)
            
            for i, pixel in enumerate(pixel_coords):
                # Step 1: Given C', find closest segment A'B'
                distances = np.linalg.norm(current_tracks - pixel, axis=1)
                closest_indices = np.argsort(distances)[:2]  # Two closest trackers
                A_idx, B_idx = closest_indices[0], closest_indices[1]
                
                A_prime = current_tracks[A_idx]  # A' in current frame
                B_prime = current_tracks[B_idx]  # B' in current frame
                A = reference_tracks[A_idx]      # A in reference frame
                B = reference_tracks[B_idx]      # B in reference frame
                
                # Step 2a: Project C' onto the LINE through A'B' → get P'
                AB_prime = B_prime - A_prime
                AC_prime = pixel - A_prime
                
                # Calculate parameter t along the LINE through A'B' (not clamped to segment)
                AB_prime_dot = np.dot(AB_prime, AB_prime)
                if AB_prime_dot > 0:
                    t = np.dot(AC_prime, AB_prime) / AB_prime_dot
                    # No clamping - allow projections beyond segment endpoints
                    P_prime = A_prime + t * AB_prime
                else:
                    # If AB_prime is zero vector, use A_prime as projection point
                    P_prime = A_prime
                
                # Step 2b: Find P on AB (corresponding point in reference frame)
                if AB_prime_dot > 0:
                    P = A + t * (B - A)
                else:
                    # If AB_prime is zero vector, use A as reference point
                    P = A
                
                # Step 3: Find C along perpendicular line to AB going through P
                # Calculate perpendicular distance from C' to segment A'B'
                PC_prime_length = np.linalg.norm(pixel - P_prime)
                
                # Calculate perpendicular direction in reference frame
                AB = B - A
                AB_length = np.linalg.norm(AB)
                AB_prime_length = np.linalg.norm(AB_prime)
                
                if AB_length > 0 and AB_prime_length > 0:
                    # Perpendicular vector (rotated 90 degrees)
                    perp_direction = np.array([AB[1], -AB[0]]) / AB_length
                    
                    # Proportional distance: AB/A'B' * P'C'
                    scale_factor = AB_length / AB_prime_length
                    PC_length = scale_factor * PC_prime_length
                    
                    # Determine which side of the segment (same side as in current frame)
                    cross_product = np.cross(AC_prime, AB_prime)
                    side_sign = 1 if cross_product >= 0 else -1
                    
                    # Find C on perpendicular through P
                    C = P + side_sign * PC_length * perp_direction
                    result_coords[i] = C
                    
                else:
                    # Fallback: use nearest tracker position
                    result_coords[i] = A
                    
            return result_coords
            
        except Exception as e:
            self.logger.error(f"Block offset calculation failed: {e}")
            # Fallback: return nearest tracker positions
            distances = np.linalg.norm(current_tracks[:, np.newaxis] - pixel_coords, axis=2)
            closest_indices = np.argmin(distances, axis=0)
            return reference_tracks[closest_indices]

    def _calculate_processing_bounds(self, mask: np.ndarray, current_tracks: np.ndarray, padding_factor: float = 0.25) -> tuple:
        """
        Calculate smart bounding box based on current frame tracker hull with generous padding.
        Avoids circular dependency with mask warping by not using mask bounds.
        
        Args:
            mask: Original mask array (H, W) - not used for bounds calculation
            current_tracks: Current frame tracker positions (N, 2)
            padding_factor: Safety padding as fraction of bbox size (default 0.25 = 25%)
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y) for processing bounds
        """
        try:
            height, width = mask.shape
            
            # 1. Get tracker hull bounding box (ignore mask to avoid circular dependency)
            if len(current_tracks) < 3:
                # Not enough trackers for hull - use tracker bounds
                track_min_x, track_max_x = np.min(current_tracks[:, 0]), np.max(current_tracks[:, 0])
                track_min_y, track_max_y = np.min(current_tracks[:, 1]), np.max(current_tracks[:, 1])
            else:
                # Calculate convex hull of trackers
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(current_tracks)
                    hull_points = current_tracks[hull.vertices]
                    track_min_x, track_max_x = np.min(hull_points[:, 0]), np.max(hull_points[:, 0])
                    track_min_y, track_max_y = np.min(hull_points[:, 1]), np.max(hull_points[:, 1])
                except:
                    # Fallback to tracker bounds if hull fails
                    track_min_x, track_max_x = np.min(current_tracks[:, 0]), np.max(current_tracks[:, 0])
                    track_min_y, track_max_y = np.min(current_tracks[:, 1]), np.max(current_tracks[:, 1])
            
            # 2. Add generous padding (25% instead of 10%) to account for mask warping
            bbox_width = track_max_x - track_min_x
            bbox_height = track_max_y - track_min_y
            padding_x = max(1, int(bbox_width * padding_factor))
            padding_y = max(1, int(bbox_height * padding_factor))
            
            # 3. Clamp to image bounds
            min_x = max(0, int(track_min_x - padding_x))
            max_x = min(width-1, int(track_max_x + padding_x))
            min_y = max(0, int(track_min_y - padding_y))
            max_y = min(height-1, int(track_max_y + padding_y))
            
            self.logger.debug(f"Processing bounds: ({min_x}, {min_y}) to ({max_x}, {max_y}) - area: {(max_x-min_x+1)*(max_y-min_y+1)} pixels")
            
            return (min_x, min_y, max_x, max_y)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate processing bounds: {e}")
            # Fallback to full image bounds
            return (0, 0, width-1, height-1)

    def _is_inside_delaunay_hull(self, pixels: np.ndarray, trackers: np.ndarray) -> np.ndarray:
        """
        Check which pixels are inside the Delaunay triangulation of trackers.
        
        Args:
            pixels: Array of pixel coordinates (N, 2)
            trackers: Array of tracker positions (M, 2)
            
        Returns:
            Boolean array (N,) indicating which pixels are inside the hull
        """
        try:
            from scipy.spatial import Delaunay
            
            if len(trackers) < 3:
                # Not enough trackers for triangulation - use distance-based fallback
                distances = np.linalg.norm(trackers[:, np.newaxis] - pixels, axis=2)
                min_distances = np.min(distances, axis=0)
                # Consider inside if within reasonable distance of nearest tracker
                avg_tracker_distance = np.mean(np.linalg.norm(trackers[1:] - trackers[:-1], axis=1))
                return min_distances < (avg_tracker_distance * 1.5)
            
            # Create Delaunay triangulation
            tri = Delaunay(trackers)
            
            # Check which pixels are inside the triangulation
            inside_mask = tri.find_simplex(pixels) >= 0
            
            return inside_mask
            
        except Exception as e:
            self.logger.warning(f"Delaunay hull detection failed: {e}, using distance-based fallback")
            # Fallback to distance-based method
            distances = np.linalg.norm(trackers[:, np.newaxis] - pixels, axis=2)
            min_distances = np.min(distances, axis=0)
            if len(trackers) > 1:
                avg_tracker_distance = np.mean(np.linalg.norm(trackers[1:] - trackers[:-1], axis=1))
                return min_distances < (avg_tracker_distance * 1.5)
            else:
                return np.zeros(len(pixels), dtype=bool)

    def _warp_mask_with_segment_algorithm(self, mask: np.ndarray, reference_tracks: np.ndarray, current_tracks: np.ndarray) -> np.ndarray:
        """Warp mask using the same hull detection and processing logic as STMap coordinates."""
        try:
            height, width = mask.shape
            warped_mask = np.zeros_like(mask)
            
            # Calculate smart processing bounds using current frame trackers
            min_x, min_y, max_x, max_y = self._calculate_processing_bounds(mask, current_tracks)
            
            # Create coordinate grids only for the bounding box area
            y_coords, x_coords = np.mgrid[min_y:max_y+1, min_x:max_x+1]
            points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
            
            # Process only pixels inside the bounding box
            self.logger.debug(f"Warping {len(points)} pixels (bounding box) with hull-aware algorithm")
            
            # Use same hull detection as STMap processing
            inside_hull_mask = self._is_inside_delaunay_hull(points, current_tracks)
            inside_hull_indices = np.where(inside_hull_mask)[0]
            outside_hull_indices = np.where(~inside_hull_mask)[0]
            
            self.logger.debug(f"Mask warping: {len(outside_hull_indices)} pixels outside hull, {len(inside_hull_indices)} pixels inside hull")
            
            # Initialize result coordinates
            warped_coords = np.zeros((len(points), 2), dtype=np.float32)
            
            # Process pixels inside hull with interpolation
            if len(inside_hull_indices) > 0:
                inside_pixels = points[inside_hull_indices]
                interpolated_coords = self._interpolate_linear(current_tracks, reference_tracks, inside_pixels)
                warped_coords[inside_hull_indices] = interpolated_coords
            
            # Process pixels outside hull with fringe algorithm
            if len(outside_hull_indices) > 0:
                outside_pixels = points[outside_hull_indices]
                fringe_coords = self._calculate_fringe_coordinates(
                    outside_pixels, reference_tracks, current_tracks
                )
                warped_coords[outside_hull_indices] = fringe_coords
            
            # For each pixel in current frame, sample from reference frame mask
            for i, (current_idx, reference_coord) in enumerate(zip(range(len(points)), warped_coords)):
                # Get current pixel position (where we're writing to) - adjust for bounding box offset
                current_y, current_x = divmod(current_idx, max_x - min_x + 1)
                current_y += min_y
                current_x += min_x
                
                # Get reference coordinates (where to sample from in the reference mask)
                x, y = reference_coord
                
                # Clamp coordinates to image bounds
                x = np.clip(x, 0, width - 1)
                y = np.clip(y, 0, height - 1)
                
                # Bilinear interpolation for smooth sampling
                x0, y0 = int(x), int(y)
                x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
                
                # Get fractional parts
                fx, fy = x - x0, y - y0
                
                # Sample four neighboring pixels from the reference mask
                p00 = mask[y0, x0]
                p01 = mask[y1, x0]
                p10 = mask[y0, x1]
                p11 = mask[y1, x1]
                
                # Bilinear interpolation
                interpolated = (p00 * (1 - fx) * (1 - fy) +
                              p10 * fx * (1 - fy) +
                              p01 * (1 - fx) * fy +
                              p11 * fx * fy)
                
                # Set the warped pixel value at the current position
                warped_mask[current_y, current_x] = interpolated
            
            return warped_mask
            
        except Exception as e:
            self.logger.error(f"Mask warping with segment algorithm failed: {e}")
            return mask  # Return original mask if warping fails

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
    
    def _save_exr(self, stmap: np.ndarray, filepath: Path, bit_depth: int, metadata: Optional[dict] = None):
        """Save STMap as RGBA EXR file."""
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
        
        # Add standard OpenEXR attributes
        header['pixelAspectRatio'] = 1.0
        header['screenWindowCenter'] = Imath.V2f(0.0, 0.0)
        header['screenWindowWidth'] = 1.0
        header['lineOrder'] = OpenEXR.INCREASING_Y
        header['compression'] = OpenEXR.ZIP_COMPRESSION
        
        # Add custom metadata
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
        
        self.logger.debug(f"Saved EXR: {filepath}")
