#!/usr/bin/env python3
"""
CoTracker Nuke Integration App
=============================

This application leverages CoTracker to track points in video and exports
the tracking data in a format suitable for Nuke's CornerPin2D node.

Features:
- Load video files and track multiple points using CoTracker
- Automatically select 4 optimal points for corner pin tracking
- Export tracking data in Nuke-compatible format
- Interactive GUI for point selection and preview

Author: AI Assistant
License: MIT
"""

import os
import sys
import numpy as np
import cv2
import torch
import imageio
import imageio.v3 as iio
from typing import List, Tuple, Dict, Optional
import gradio as gr
from pathlib import Path
import json
import tempfile
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment


class CoTrackerNukeApp:
    """Main application class for CoTracker-Nuke integration."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cotracker_model = None
        self.current_video = None
        self.tracking_results = None
        self.selected_points = []
        
    def load_cotracker_model(self):
        """Load CoTracker model from torch hub."""
        try:
            print(f"Loading CoTracker model on {self.device}...")
            # Use CoTracker2 for better compatibility with custom queries
            # CoTracker3 offline doesn't properly support the queries parameter
            try:
                self.cotracker_model = torch.hub.load(
                    "facebookresearch/co-tracker", 
                    "cotracker2"
                ).to(self.device)
                print("CoTracker2 model loaded successfully (best for custom reference frames)")
            except Exception as e:
                print(f"CoTracker2 not available, trying CoTracker3 offline: {e}")
                try:
                    self.cotracker_model = torch.hub.load(
                        "facebookresearch/co-tracker", 
                        "cotracker3_offline"
                    ).to(self.device)
                    print("CoTracker3 offline model loaded successfully")
                except Exception as e2:
                    print(f"CoTracker3 offline not available, trying online: {e2}")
                    self.cotracker_model = torch.hub.load(
                        "facebookresearch/co-tracker", 
                        "cotracker3_online"
                    ).to(self.device)
                    print("CoTracker3 online model loaded successfully")
                
        except Exception as e:
            print(f"Error loading CoTracker model: {e}")
            raise
    
    def _generate_grid_queries(self, video: np.ndarray, grid_size: int, reference_frame: int) -> torch.Tensor:
        """Generate grid query points for a specific reference frame."""
        import torch
        
        # Ensure reference frame is within bounds
        reference_frame = max(0, min(reference_frame, video.shape[0] - 1))
        
        # Get video dimensions
        height, width = video.shape[1], video.shape[2]
        
        # Create grid points on the reference frame
        # Calculate step sizes for even distribution
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
        queries_tensor = torch.tensor(queries, dtype=torch.float32, device=self.device)
        print(f"Generated {len(queries)} query points on frame {reference_frame}")
        
        return queries_tensor
    
    def load_video(self, video_path: str) -> np.ndarray:
        """Load video file and return as numpy array."""
        try:
            # Use imageio.v3.imread for video files
            frames = iio.imread(video_path, plugin="FFMPEG")
            print(f"Loaded video: {frames.shape}")
            return frames
        except Exception as e:
            print(f"Error loading video with imageio.v3, trying alternative method: {e}")
            try:
                # Alternative method using imageio reader
                reader = imageio.get_reader(video_path, 'ffmpeg')
                frames = []
                for frame in reader:
                    frames.append(frame)
                reader.close()
                frames = np.array(frames)
                print(f"Loaded video (alternative method): {frames.shape}")
                return frames
            except Exception as e2:
                print(f"Error loading video with alternative method: {e2}")
                try:
                    # Third method using OpenCV as fallback
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    cap.release()
                    frames = np.array(frames)
                    print(f"Loaded video (OpenCV fallback): {frames.shape}")
                    return frames
                except Exception as e3:
                    print(f"All video loading methods failed: {e3}")
                    raise
    
    def track_points(self, video: np.ndarray, grid_size: int = 10, reference_frame: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Track points in video using CoTracker."""
        if self.cotracker_model is None:
            self.load_cotracker_model()
        
        # Convert video to tensor format (B, T, C, H, W)
        video_tensor = torch.tensor(video).permute(0, 3, 1, 2)[None].float().to(self.device)
        
        try:
            # Check if this is CoTracker3 online model (different API)
            if hasattr(self.cotracker_model, 'step'):
                # CoTracker3 online model - use proper sliding window processing
                print("Using CoTracker3 online API with proper sliding window")
                with torch.no_grad():
                    T = video_tensor.shape[1]  # Number of frames
                    window_size = self.cotracker_model.step * 2  # Window size (T')
                    step_size = self.cotracker_model.step  # Step size (T'/2)
                    
                    print(f"Processing {T} frames with window_size={window_size}, step_size={step_size}")
                    
                    # Initialize tracking results for the full video length
                    first_chunk = video_tensor[:, :min(window_size, T)]
                    self.cotracker_model(video_chunk=first_chunk, is_first_step=True, grid_size=grid_size)
                    pred_tracks, pred_visibility = self.cotracker_model(video_chunk=first_chunk)
                    
                    # Get the number of points and initialize full arrays
                    num_points = pred_tracks.shape[2]
                    full_tracks = torch.zeros((1, T, num_points, 2), device=self.device)
                    full_visibility = torch.zeros((1, T, num_points), device=self.device)
                    
                    # Store first window results
                    first_window_frames = min(step_size, T)
                    full_tracks[:, :first_window_frames] = pred_tracks[:, :first_window_frames]
                    full_visibility[:, :first_window_frames] = pred_visibility[:, :first_window_frames]
                    
                    # Process remaining windows with proper overlap handling
                    current_pos = step_size
                    while current_pos < T:
                        end_pos = min(current_pos + window_size, T)
                        chunk = video_tensor[:, current_pos:end_pos]
                        
                        # Pad chunk if needed
                        if chunk.shape[1] < window_size:
                            padding = torch.zeros((1, window_size - chunk.shape[1], chunk.shape[2], chunk.shape[3], chunk.shape[4]), device=self.device)
                            chunk = torch.cat([chunk, padding], dim=1)
                        
                        pred_tracks, pred_visibility = self.cotracker_model(video_chunk=chunk)
                        
                        # Only use non-overlapping part (first step_size frames of prediction)
                        frames_to_use = min(step_size, T - current_pos)
                        full_tracks[:, current_pos:current_pos + frames_to_use] = pred_tracks[:, :frames_to_use]
                        full_visibility[:, current_pos:current_pos + frames_to_use] = pred_visibility[:, :frames_to_use]
                        
                        current_pos += step_size
                        print(f"Processed window ending at frame {current_pos}")
                    
                    pred_tracks = full_tracks
                    pred_visibility = full_visibility
            else:
                # CoTracker2 or offline model
                print("Using CoTracker2/offline API")
                with torch.no_grad():
                    if reference_frame == 0:
                        # Use automatic grid on first frame
                        pred_tracks, pred_visibility = self.cotracker_model(
                            video_tensor, 
                            grid_size=grid_size
                        )
                    else:
                        # Use custom queries with specified reference frame
                        print(f"Using custom reference frame: {reference_frame}")
                        queries = self._generate_grid_queries(video, grid_size, reference_frame)
                        pred_tracks, pred_visibility = self.cotracker_model(
                            video_tensor, 
                            queries=queries
                        )
            
            print(f"Tracking completed. Tracks shape: {pred_tracks.shape}")
            return pred_tracks, pred_visibility
            
        except Exception as e:
            print(f"Error during tracking: {e}")
            # Try fallback to CoTracker2
            try:
                print("Trying fallback to CoTracker2...")
                self.cotracker_model = torch.hub.load(
                    "facebookresearch/co-tracker", 
                    "cotracker2"
                ).to(self.device)
                
                with torch.no_grad():
                    pred_tracks, pred_visibility = self.cotracker_model(
                        video_tensor, 
                        grid_size=grid_size
                    )
                
                print(f"Fallback successful. Tracks shape: {pred_tracks.shape}")
                return pred_tracks, pred_visibility
                
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise
    
    def select_corner_pin_points(self, tracks: torch.Tensor, visibility: torch.Tensor) -> List[int]:
        """
        Select 4 optimal points for corner pin from tracked points.
        
        Strategy:
        1. Filter points that are visible throughout most of the sequence
        2. Find points that form a good quadrilateral (spread out corners)
        3. Prioritize points with stable tracking
        """
        # Convert to numpy for easier processing
        tracks_np = tracks[0].cpu().numpy()  # Shape: (T, N, 2)
        visibility_np = visibility[0].cpu().numpy()  # Shape: (T, N) or (T, N, 1)
        
        # Handle different visibility shapes
        if len(visibility_np.shape) == 3:
            visibility_np = visibility_np[:, :, 0]  # (T, N)
        
        T, N, _ = tracks_np.shape
        
        # Calculate visibility score for each point
        visibility_scores = np.mean(visibility_np, axis=0)  # (N,)
        
        # Filter points with good visibility (>70% of frames)
        good_visibility_mask = visibility_scores > 0.7
        good_indices = np.where(good_visibility_mask)[0]
        
        if len(good_indices) < 4:
            # If not enough good points, use all points
            good_indices = np.arange(N)
        
        # Calculate motion stability (lower variance = more stable)
        motion_variance = np.var(tracks_np[:, good_indices, :], axis=0)
        stability_scores = 1.0 / (1.0 + np.mean(motion_variance, axis=1))
        
        # Get initial positions for spatial distribution
        initial_positions = tracks_np[0, good_indices, :]  # (M, 2)
        
        # Find 4 points that are well distributed spatially
        selected_indices = self._select_corner_points(
            initial_positions, 
            visibility_scores[good_indices], 
            stability_scores
        )
        
        # Convert back to original indices
        corner_indices = good_indices[selected_indices]
        
        return corner_indices.tolist()
    
    def _select_corner_points(self, positions: np.ndarray, visibility: np.ndarray, 
                            stability: np.ndarray) -> np.ndarray:
        """Select 4 points that form a good quadrilateral."""
        N = len(positions)
        if N <= 4:
            return np.arange(N)
        
        # Calculate combined score
        combined_scores = 0.4 * visibility + 0.6 * stability
        
        # Find the convex hull to get boundary points
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(positions)
            hull_indices = hull.vertices
            
            if len(hull_indices) >= 4:
                # Select 4 best points from convex hull
                hull_scores = combined_scores[hull_indices]
                best_hull_indices = np.argsort(hull_scores)[-4:]
                return hull_indices[best_hull_indices]
        except:
            pass
        
        # Fallback: select 4 points with maximum spread
        # Use k-means++ like selection for good spatial distribution
        selected = []
        remaining = list(range(N))
        
        # Start with the point with highest combined score
        first_idx = np.argmax(combined_scores)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Select remaining 3 points to maximize minimum distance
        for _ in range(3):
            if not remaining:
                break
                
            distances = []
            for candidate in remaining:
                min_dist = min([
                    np.linalg.norm(positions[candidate] - positions[sel]) 
                    for sel in selected
                ])
                # Weight by combined score
                weighted_dist = min_dist * (1 + combined_scores[candidate])
                distances.append(weighted_dist)
            
            best_candidate_idx = remaining[np.argmax(distances)]
            selected.append(best_candidate_idx)
            remaining.remove(best_candidate_idx)
        
        return np.array(selected)
    
    def export_all_tracks_to_nuke(self, tracks: torch.Tensor, visibility: torch.Tensor,
                                 output_path: str, video_info: Dict) -> str:
        """Export tracking data to Nuke-compatible format."""
        
        tracks_np = tracks[0].cpu().numpy()  # (T, N, 2)
        visibility_np = visibility[0].cpu().numpy()  # (T, N) or (T, N, 1)
        
        # Handle different visibility shapes
        if len(visibility_np.shape) == 3:
            visibility_np = visibility_np[:, :, 0]  # (T, N)
        
        T, N, _ = tracks_np.shape
        
        # Use all tracks (limit to reasonable number for Nuke)
        max_tracks = min(N, 20)  # Limit to 20 tracks for manageable Nuke file
        all_tracks = tracks_np[:, :max_tracks, :]  # (T, max_tracks, 2)
        all_visibility = visibility_np[:, :max_tracks]  # (T, max_tracks)
        
        # Generate Nuke script
        nuke_script = self._generate_nuke_script(
            all_tracks, all_visibility, video_info
        )
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(nuke_script)
        
        return nuke_script
    
    def _generate_nuke_script(self, tracks: np.ndarray, visibility: np.ndarray, 
                            video_info: Dict) -> str:
        """Generate Nuke script with CornerPin2D node."""
        
        T, num_tracks, _ = tracks.shape
        width = video_info.get('width', 1920)
        height = video_info.get('height', 1080)
        fps = video_info.get('fps', 24)
        
        # Nuke script header (version independent)
        script = f"""#! C4_8_0 v1
# CoTracker Generated Corner Pin
# Video: {width}x{height} @ {fps}fps
# Frames: {T}

"""
        
        # Create Read node (placeholder)
        script += """
Read {
 inputs 0
 file_type mov
 file "path/to/your/video.mov"
 first 1
 last """ + str(T) + """
 origfirst 1
 origlast """ + str(T) + """
 origset true
 name Read1
 selected true
 xpos 0
 ypos 0
}
"""
        
        # Create Tracker node with proper Nuke format based on real example
        script += f"""Tracker4 {{
 inputs 1
 tracks {{ {{ 1 31 2 }} 
{{ {{ 5 1 20 enable e 1 }} 
{{ 3 1 75 name name 1 }} 
{{ 2 1 58 track_x track_x 1 }} 
{{ 2 1 58 track_y track_y 1 }} 
{{ 2 1 63 offset_x offset_x 1 }} 
{{ 2 1 63 offset_y offset_y 1 }} 
{{ 4 1 27 T T 1 }} 
{{ 4 1 27 R R 1 }} 
{{ 4 1 27 S S 1 }} 
{{ 2 0 45 error error 1 }} 
{{ 1 1 0 error_min error_min 1 }} 
{{ 1 1 0 error_max error_max 1 }} 
{{ 1 1 0 pattern_x pattern_x 1 }} 
{{ 1 1 0 pattern_y pattern_y 1 }} 
{{ 1 1 0 pattern_r pattern_r 1 }} 
{{ 1 1 0 pattern_t pattern_t 1 }} 
{{ 1 1 0 search_x search_x 1 }} 
{{ 1 1 0 search_y search_y 1 }} 
{{ 1 1 0 search_r search_r 1 }} 
{{ 1 1 0 search_t search_t 1 }} 
{{ 2 1 0 key_track key_track 1 }} 
{{ 2 1 0 key_search_x key_search_x 1 }} 
{{ 2 1 0 key_search_y key_search_y 1 }} 
{{ 2 1 0 key_search_r key_search_r 1 }} 
{{ 2 1 0 key_search_t key_search_t 1 }} 
{{ 2 1 0 key_track_x key_track_x 1 }} 
{{ 2 1 0 key_track_y key_track_y 1 }} 
{{ 2 1 0 key_track_r key_track_r 1 }} 
{{ 2 1 0 key_track_t key_track_t 1 }} 
{{ 2 1 0 key_centre_offset_x key_centre_offset_x 1 }} 
{{ 2 1 0 key_centre_offset_y key_centre_offset_y 1 }} 
}} 
{{"""
        
        # Add each track in proper Nuke format
        for i in range(num_tracks):
            track_name = f"CoTracker_Point_{i+1}"
            
            # Build X coordinate curve with keyframes
            x_curve = f"{{curve x1 {tracks[0, i, 0]:.6f}"
            for frame in range(1, min(T, 50)):  # Limit frames for file size
                if frame < T and visibility[frame, i] > 0.5:
                    x_curve += f" {tracks[frame, i, 0]:.6f}"
            x_curve += "}"
            
            # Build Y coordinate curve with keyframes  
            y_curve = f"{{curve x1 {tracks[0, i, 1]:.6f}"
            for frame in range(1, min(T, 50)):
                if frame < T and visibility[frame, i] > 0.5:
                    y_curve += f" {tracks[frame, i, 1]:.6f}"
            y_curve += "}"
            
            # Build error curve (dummy values)
            error_curve = "{curve x1 0.0001"
            for frame in range(1, min(T, 50)):
                if frame < T and visibility[frame, i] > 0.5:
                    error_curve += " 0.0001"
            error_curve += "}"
            
            # Add track data in proper format
            script += f"""
 {{ {{curve K x1 1}} "{track_name}" {x_curve} {y_curve} {{curve K x1 0}} {{curve K x1 0}} 1 0 0 {error_curve} 0 0.0001 -29.6 -31 29.6 31 -11 -11 11 11 {{curve}} {{curve x1 {tracks[0, i, 0]:.0f}}} {{curve x1 {tracks[0, i, 1]:.0f}}} {{curve x1 {tracks[0, i, 0]+60:.0f}}} {{curve x1 {tracks[0, i, 1]+60:.0f}}} {{curve x1 {tracks[0, i, 0]:.0f}}} {{curve x1 {tracks[0, i, 1]:.0f}}} {{curve x1 {tracks[0, i, 0]+60:.0f}}} {{curve x1 {tracks[0, i, 1]+60:.0f}}} {{curve x1 {tracks[0, i, 0]:.1f}}} {{curve x1 {tracks[0, i, 1]:.1f}}}  }}"""
            
            if i < num_tracks - 1:
                script += " "
        
        script += f"""
}} 
}}
reference_frame 1
translate {{{{curve x1 0}}}} {{{{curve x1 0}}}}
center {{{{curve x1 {width/2}}}}} {{{{curve x1 {height/2}}}}}
selected_tracks {','.join(map(str, range(num_tracks)))}
name Tracker1
selected true
xpos 0
ypos 100
}}
"""
        
        return script


def create_gradio_interface():
    """Create Gradio interface for the application."""
    
    app = CoTrackerNukeApp()
    app.reference_frame = 0  # Default to frame 0
    
    def load_video_for_reference(reference_video):
        """Load video and display it for reference frame selection."""
        if reference_video is None:
            return None, "Please upload a video file first."
        
        try:
            video = app.load_video(reference_video)
            app.current_video = video
            
            # Create a simple preview video for reference frame selection
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_dir, f"reference_video_{os.getpid()}.mp4")
            
            # Save the full video for reference frame selection
            preview_frames = video  # Use full video for reference frame selection
            
            import imageio.v3 as iio
            iio.imwrite(
                temp_video_path, 
                preview_frames, 
                plugin="FFMPEG", 
                fps=24,
                codec='libx264',
                quality=8
            )
            
            return temp_video_path, f"Video loaded: {video.shape[0]} frames available"
            
        except Exception as e:
            return None, f"Error loading video: {str(e)}"
    
    def select_reference_frame(reference_video, video_data):
        """Select the current frame as reference frame."""
        if reference_video is None or app.current_video is None:
            return "Please load a video first."
        
        try:
            # Extract current time from video data
            # video_data is a tuple: (video_path, current_time)
            if isinstance(video_data, tuple) and len(video_data) >= 2:
                current_time = video_data[1]
            else:
                # Fallback: assume middle of video
                current_time = app.current_video.shape[0] / 2 / 24  # Convert frames to seconds at 24fps
            
            # Convert time to frame index (assuming 24fps)
            fps = 24
            frame_idx = int(current_time * fps)
            frame_idx = max(0, min(frame_idx, app.current_video.shape[0] - 1))
            
            app.reference_frame = frame_idx
            
            return f"Reference frame set to: Frame {frame_idx} (time: {current_time:.2f}s)"
            
        except Exception as e:
            return f"Error selecting reference frame: {str(e)}"
    
    def process_video(reference_video, grid_size, preview_downsample):
        """Process video and return tracking results."""
        if reference_video is None:
            return "Please upload a video file.", None
        
        try:
            # Load video (reference_video is now the direct video file)
            video = app.load_video(reference_video)
            app.current_video = video
            
            # Track points with reference frame
            tracks, visibility = app.track_points(video, grid_size, app.reference_frame)
            app.tracking_results = (tracks, visibility)
            
            # Calculate preview points from downsampling ratio
            # If grid_size=50 and downsample=4, show every 4th point -> 50/4 = ~12 points per axis
            preview_points_per_axis = max(1, grid_size // preview_downsample)
            # For square grids, this gives us preview_points_per_axis² total points
            max_preview_points = preview_points_per_axis * preview_points_per_axis
            
            # Create preview video with tracked points (try both methods)
            try:
                preview_video = app._create_preview_video(video, tracks, visibility, max_preview_points)
            except Exception as e:
                print(f"Video preview failed, trying image sequence: {e}")
                preview_video = app._create_preview_image_sequence(video, tracks, visibility)
            
            result_text = f"""
Tracking completed successfully!
- Video shape: {video.shape}
- Video frames: {video.shape[0]} (input)
- Reference frame: {app.reference_frame}
- Total tracked points: {tracks.shape[2]}
- Preview downsampling: 1/{preview_downsample} (showing ~{max_preview_points} points)
- Processing device: {app.device}
- CoTracker frames: {tracks.shape[1]} (internal processing)
- Note: CoTracker3 generates more frames for temporal precision
            """
            
            return result_text, preview_video
            
        except Exception as e:
            return f"Error processing video: {str(e)}", None
    
    def export_nuke_file(output_filename):
        """Export tracking data to Nuke file."""
        if app.tracking_results is None:
            return "Please process a video first."
        
        if not output_filename:
            output_filename = "cotracker_tracks.nk"
        
        if not output_filename.endswith('.nk'):
            output_filename += '.nk'
        
        try:
            tracks, visibility = app.tracking_results
            video_info = {
                'width': app.current_video.shape[2],
                'height': app.current_video.shape[1], 
                'fps': 24  # Default FPS
            }
            
            # Export all tracked points (not just corner selection)
            nuke_script = app.export_all_tracks_to_nuke(
                tracks, visibility, output_filename, video_info
            )
            
            return f"Nuke tracker file exported successfully to: {output_filename}"
            
        except Exception as e:
            return f"Error exporting Nuke file: {str(e)}"
    
    # Create Gradio interface
    with gr.Blocks(title="CoTracker Nuke Integration") as interface:
        gr.Markdown("# CoTracker Nuke Integration")
        gr.Markdown("Upload a video, track points with CoTracker, and export tracking data for Nuke.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Primary video upload/reference selection (replaces redundant upload widget)
                reference_video = gr.Video(
                    label="Upload Video & Select Reference Frame", 
                    interactive=True,
                    sources=["upload"],  # Only allow file upload, no webcam
                    height=300   # Half the default size
                )
                
                # Reference frame selection controls
                gr.Markdown("### Reference Frame Selection")
                reference_frame_info = gr.Textbox(
                    label="Reference Frame Info", 
                    value="No reference frame selected (will use frame 0)",
                    interactive=False,
                    lines=2
                )
                select_reference_btn = gr.Button("Select Current Frame as Reference", variant="secondary")
                
                # Tracking parameters
                gr.Markdown("### Tracking Parameters")
                grid_size = gr.Slider(
                    minimum=5, maximum=70, value=10, step=1,
                    label="Grid Size (number of points to track)"
                )
                preview_downsample = gr.Dropdown(
                    choices=[("1/10 (every 10th point)", 10), ("1/8 (every 8th point)", 8), ("1/6 (every 6th point)", 6), ("1/4 (every 4th point)", 4), ("1/3 (every 3rd point)", 3), ("1/2 (every 2nd point)", 2), ("1/1 (all points)", 1)],
                    value=4,
                    label="Preview Downsample (ratio of Grid Size to show)"
                )
                process_btn = gr.Button("Process Video", variant="primary")
                
            with gr.Column(scale=1):
                result_text = gr.Textbox(
                    label="Processing Results", 
                    lines=15, 
                    interactive=False
                )
        
        # Tracking preview moved below
        gr.Markdown("## Tracking Results")
        preview_video = gr.Video(
            label="Tracking Preview",
            height=400  # 2/3 the default size
        )
        
        with gr.Row():
            output_filename = gr.Textbox(
                label="Output Filename", 
                value="cotracker_tracks.nk",
                placeholder="Enter filename for Nuke script"
            )
            export_btn = gr.Button("Export to Nuke", variant="secondary")
            export_result = gr.Textbox(
                label="Export Status", 
                lines=3, 
                interactive=False
            )
        
        # Event handlers
        # Load video for reference frame selection (now using reference_video directly)
        reference_video.upload(
            fn=load_video_for_reference,
            inputs=[reference_video],
            outputs=[reference_video, reference_frame_info]
        )
        
        # Select reference frame
        select_reference_btn.click(
            fn=select_reference_frame,
            inputs=[reference_video, reference_video],
            outputs=[reference_frame_info]
        )
        
        # Process video
        process_btn.click(
            fn=process_video,
            inputs=[reference_video, grid_size, preview_downsample],
            outputs=[result_text, preview_video]
        )
        
        # Export to Nuke
        export_btn.click(
            fn=export_nuke_file,
            inputs=[output_filename],
            outputs=[export_result]
        )
    
    return interface


# Add preview video creation method to the class
def _create_preview_video(self, video: np.ndarray, tracks: torch.Tensor, 
                         visibility: torch.Tensor, max_preview_points: int = 75) -> str:
    """Create preview video showing tracked points over time."""
    import tempfile
    import os
    import numpy as np
    
    tracks_np = tracks[0].cpu().numpy()  # (T, N, 2)
    visibility_np = visibility[0].cpu().numpy()  # (T, N)
    
    # Handle different visibility shapes
    if len(visibility_np.shape) == 3:
        visibility_np = visibility_np[:, :, 0]
    
    # PRE-SELECT CONSISTENT POINTS ACROSS ALL FRAMES
    # This prevents "dancing" by ensuring we track the same physical points
    total_points = tracks_np.shape[1]
    max_points_to_show = min(total_points, max_preview_points)
    
    # Use first frame to select points with good spatial distribution
    first_frame_tracks = tracks_np[0]  # (N, 2)
    first_frame_visibility = visibility_np[0]  # (N,)
    
    # Get initially visible points
    visible_mask = first_frame_visibility > 0.5
    visible_indices = np.where(visible_mask)[0]
    
    if len(visible_indices) <= max_points_to_show:
        selected_point_indices = visible_indices.tolist()
    else:
        # Implement proper 2D grid sampling to ensure even distribution
        visible_tracks = first_frame_tracks[visible_indices]
        
        # Find the grid structure
        x_coords = visible_tracks[:, 0]
        y_coords = visible_tracks[:, 1]
        
        # Round to find unique grid positions
        x_rounded = np.round(x_coords).astype(int)
        y_rounded = np.round(y_coords).astype(int)
        
        unique_x = np.unique(x_rounded)
        unique_y = np.unique(y_rounded)
        
        grid_width = len(unique_x)
        grid_height = len(unique_y)
        
        # Calculate target grid dimensions for sampling
        # Maintain aspect ratio while getting close to max_points_to_show
        aspect_ratio = grid_width / grid_height
        target_points_sqrt = np.sqrt(max_points_to_show)
        
        if aspect_ratio >= 1.0:  # Wider than tall
            target_width = int(target_points_sqrt * np.sqrt(aspect_ratio))
            target_height = int(target_points_sqrt / np.sqrt(aspect_ratio))
        else:  # Taller than wide
            target_width = int(target_points_sqrt / np.sqrt(1/aspect_ratio))
            target_height = int(target_points_sqrt * np.sqrt(1/aspect_ratio))
        
        # Ensure we don't exceed grid dimensions
        target_width = min(target_width, grid_width)
        target_height = min(target_height, grid_height)
        
        print(f"Grid sampling: {grid_width}x{grid_height} -> {target_width}x{target_height} ({target_width*target_height} points)")
        
        # Sample grid positions evenly, ensuring we include edges
        if target_width == 1:
            selected_x_indices = [grid_width // 2]  # Center point
        elif target_width >= grid_width:
            selected_x_indices = list(range(grid_width))  # All points
        else:
            # Use linspace-like distribution to include first and last
            selected_x_indices = [int(i * (grid_width - 1) / (target_width - 1)) for i in range(target_width)]
        
        if target_height == 1:
            selected_y_indices = [grid_height // 2]  # Center point
        elif target_height >= grid_height:
            selected_y_indices = list(range(grid_height))  # All points
        else:
            # Use linspace-like distribution to include first and last
            selected_y_indices = [int(i * (grid_height - 1) / (target_height - 1)) for i in range(target_height)]
        
        # Find points that match the selected grid positions
        selected_point_indices = []
        for x_idx in selected_x_indices:
            target_x = unique_x[x_idx]
            for y_idx in selected_y_indices:
                target_y = unique_y[y_idx]
                
                # Find the point closest to this grid position
                distances = np.sqrt((x_rounded - target_x)**2 + (y_rounded - target_y)**2)
                closest_idx = np.argmin(distances)
                original_idx = visible_indices[closest_idx]
                
                if original_idx not in selected_point_indices:
                    selected_point_indices.append(original_idx)
        
        # If we didn't get enough points, fill with remaining points
        if len(selected_point_indices) < max_points_to_show:
            remaining_indices = [idx for idx in visible_indices if idx not in selected_point_indices]
            needed = max_points_to_show - len(selected_point_indices)
            step = max(1, len(remaining_indices) // needed)
            additional_indices = remaining_indices[::step][:needed]
            selected_point_indices.extend(additional_indices)
    
    print(f"Selected {len(selected_point_indices)} consistent points for tracking across all frames")
    
    # Verify the distribution of selected points
    if len(selected_point_indices) > 0:
        selected_tracks_frame0 = first_frame_tracks[selected_point_indices]
        sel_x = np.round(selected_tracks_frame0[:, 0]).astype(int)
        sel_y = np.round(selected_tracks_frame0[:, 1]).astype(int)
        sel_unique_x = len(np.unique(sel_x))
        sel_unique_y = len(np.unique(sel_y))
        print(f"Selected grid distribution: {sel_unique_x} columns x {sel_unique_y} rows")
        print(f"Total available points: {total_points}")
    
    # Create frames with tracking overlays
    preview_frames = []
    # For videos with reasonable length (< 200 frames), use all frames
    # For longer videos, sample to keep preview manageable
    if video.shape[0] <= 200:
        num_frames = video.shape[0]  # Use all frames
        step = 1
    else:
        # For very long videos, limit to 150 frames max
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
        
        # Map video frame index to tracking frame index
        # Direct 1:1 mapping since we fixed the sliding window processing
        track_frame_idx = frame_idx
        
        if track_frame_idx < tracks_np.shape[0]:
            frame_tracks = tracks_np[track_frame_idx].copy()  # (N, 2)
            frame_visibility = visibility_np[track_frame_idx]  # (N,)
            
            # Scale tracking points if frame was resized
            if scale_factor != 1.0:
                frame_tracks = frame_tracks * scale_factor
            
            # Use pre-selected consistent points
            points_drawn = 0
            for idx_pos, j in enumerate(selected_point_indices):
                track = frame_tracks[j]
                vis = frame_visibility[j]
                
                if vis > 0.5:  # Only draw visible points
                    x, y = int(track[0]), int(track[1])
                    # Ensure points are within frame bounds
                    if 0 <= x < new_width and 0 <= y < new_height:
                        # Use different colors for different points
                        color = [
                            (255, 0, 0),    # Red
                            (0, 255, 0),    # Green  
                            (0, 0, 255),    # Blue
                            (255, 255, 0),  # Cyan
                            (255, 0, 255),  # Magenta
                            (0, 255, 255),  # Yellow
                            (128, 255, 0),  # Lime
                            (255, 128, 0),  # Orange
                            (0, 128, 255),  # Light blue
                            (255, 0, 128),  # Pink
                            (128, 0, 255),  # Purple
                            (0, 255, 128),  # Sea green
                        ][j % 12]
                        
                        # Make circles more visible
                        cv2.circle(frame, (x, y), 4, color, -1)
                        cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)  # White outline
                        
                        # Add point number for visible points (up to 50 for better grid visualization)
                        if idx_pos < min(50, len(selected_point_indices)):  # Show numbers for up to 50 points
                            cv2.putText(frame, str(idx_pos + 1), (x+8, y-8), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                            cv2.putText(frame, str(idx_pos + 1), (x+8, y-8), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)  # Black outline
                        
                        points_drawn += 1
            
            print(f"Frame {i}: Video {frame_idx}/{video.shape[0]} -> Track {track_frame_idx}/{tracks_np.shape[0]}, Drew {points_drawn}/{max_points_to_show} points")
        
        preview_frames.append(frame)
    
    # Save as temporary video file with optimized settings
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, f"cotracker_preview_{os.getpid()}.mp4")
    
    try:
        import imageio.v3 as iio
        # Use better encoding settings for smooth playback
        iio.imwrite(
            temp_video_path, 
            np.array(preview_frames), 
            plugin="FFMPEG", 
            fps=24,  # Higher fps for smoother playback
            codec='libx264',  # Better codec
            quality=8,  # Good quality
            macro_block_size=16,  # Ensure compatibility
            ffmpeg_params=[
                '-preset', 'fast',  # Fast encoding
                '-crf', '18',  # High quality
                '-pix_fmt', 'yuv420p',  # Compatible pixel format
                '-movflags', '+faststart'  # Optimize for web playback
            ]
        )
        return temp_video_path
    except Exception as e:
        print(f"Error creating preview video with advanced settings: {e}")
        try:
            # Fallback to simpler settings
            iio.imwrite(temp_video_path, np.array(preview_frames), plugin="FFMPEG", fps=24)
            return temp_video_path
        except Exception as e2:
            print(f"Error with fallback video creation: {e2}")
            # Return first frame as final fallback
            return preview_frames[0] if preview_frames else video[0]

# Add image sequence preview method (alternative for better scrubbing)
def _create_preview_image_sequence(self, video: np.ndarray, tracks: torch.Tensor, 
                                  visibility: torch.Tensor) -> str:
    """Create preview as image sequence for better scrubbing performance."""
    import tempfile
    import os
    import zipfile
    
    tracks_np = tracks[0].cpu().numpy()  # (T, N, 2)
    visibility_np = visibility[0].cpu().numpy()  # (T, N)
    
    # Handle different visibility shapes
    if len(visibility_np.shape) == 3:
        visibility_np = visibility_np[:, :, 0]
    
    # Create fewer frames for image sequence (every 10th frame)
    preview_frames = []
    step = max(1, video.shape[0] // 20)  # 20 frames max
    
    # Resize for performance
    original_height, original_width = video.shape[1], video.shape[2]
    if original_width > 960:
        scale_factor = 960 / original_width
        new_width = 960
        new_height = int(original_height * scale_factor)
    else:
        scale_factor = 1.0
        new_width = original_width
        new_height = original_height
    
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    for i, frame_idx in enumerate(range(0, video.shape[0], step)):
        if i >= 20:  # Limit to 20 images
            break
            
        frame = video[frame_idx].copy()
        
        # Resize frame if needed
        if scale_factor != 1.0:
            frame = cv2.resize(frame, (new_width, new_height))
        
        if frame_idx < tracks_np.shape[0]:
            frame_tracks = tracks_np[frame_idx]
            frame_visibility = visibility_np[frame_idx]
            
            # Scale tracking points if frame was resized
            if scale_factor != 1.0:
                frame_tracks = frame_tracks * scale_factor
            
            # Draw tracking points
            for j, (track, vis) in enumerate(zip(frame_tracks[:15], frame_visibility[:15])):
                if vis > 0.5:
                    x, y = int(track[0]), int(track[1])
                    if 0 <= x < new_width and 0 <= y < new_height:
                        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                                (255, 255, 0), (255, 0, 255), (0, 255, 255)][j % 6]
                        cv2.circle(frame, (x, y), 4, color, -1)
                        cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
        
        # Save frame as image
        image_path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        image_paths.append(image_path)
    
    # Create a simple HTML viewer for the image sequence
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CoTracker Preview</title>
        <style>
            body {{ margin: 0; padding: 20px; background: #222; color: white; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .player {{ text-align: center; }}
            img {{ max-width: 100%; height: auto; }}
            .controls {{ margin: 20px 0; }}
            input[type="range"] {{ width: 100%; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="player">
                <img id="preview" src="data:image/jpeg;base64,{{}}" alt="Frame"/>
                <div class="controls">
                    <input type="range" id="scrubber" min="0" max="{len(image_paths)-1}" value="0"/>
                    <p>Frame: <span id="frameNum">0</span> / {len(image_paths)-1}</p>
                </div>
            </div>
        </div>
        <script>
            const images = {image_paths};
            const img = document.getElementById('preview');
            const scrubber = document.getElementById('scrubber');
            const frameNum = document.getElementById('frameNum');
            
            scrubber.oninput = function() {{
                const idx = parseInt(this.value);
                // Load image would go here in a real implementation
                frameNum.textContent = idx;
            }}
        </script>
    </body>
    </html>
    '''
    
    # For now, just return the first frame since HTML viewer is complex
    return image_paths[0] if image_paths else video[0]

# Add the methods to the class
CoTrackerNukeApp._create_preview_video = _create_preview_video
CoTrackerNukeApp._create_preview_image_sequence = _create_preview_image_sequence


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=False, server_name="127.0.0.1", server_port=7860)
