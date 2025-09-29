#!/usr/bin/env python3
"""
Gradio UI Interface
===================

Provides the web-based user interface for the CoTracker Nuke application.
"""

import gradio as gr
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Any
import logging
from pathlib import Path

from ..core.app import CoTrackerNukeApp


class GradioInterface:
    """Gradio web interface for CoTracker Nuke App."""
    
    def __init__(self, app: CoTrackerNukeApp):
        """
        Initialize Gradio interface.
        
        Args:
            app: CoTrackerNukeApp instance
        """
        self.app = app
        self.logger = app.logger
        
        # UI state
        self.preview_video_path = None
    
    def load_video_for_reference(self, reference_video) -> Tuple[str, Optional[str]]:
        """Load video and return status message + video path for player."""
        try:
            if reference_video is None:
                return "❌ No video file selected", None
            
            # Load video
            self.app.load_video(reference_video)
            self.preview_video_path = reference_video
            
            # Get video info
            info = self.app.get_video_info()
            
            status_msg = (f"✅ Video loaded successfully!\n"
                         f"📹 Frames: {info['frames']}\n"
                         f"📐 Resolution: {info['width']}x{info['height']}\n"
                         f"💾 Size: {info['memory_mb']:.1f} MB")
            
            return status_msg, reference_video
                   
        except Exception as e:
            error_msg = f"❌ Error loading video: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, None
    
    def set_manual_reference_frame(self, frame_number_with_offset: int, 
                                  start_frame_offset: int) -> Tuple[str, Optional[Image.Image]]:
        """Set manual reference frame and return image for masking."""
        try:
            # Validate input
            if frame_number_with_offset < start_frame_offset:
                return (f"❌ Error: Frame number {frame_number_with_offset} is less than "
                       f"start frame offset {start_frame_offset}"), None
            
            # Calculate 0-based video frame
            frame_number = frame_number_with_offset - start_frame_offset
            
            # Set reference frame
            actual_frame = self.app.set_reference_frame(frame_number)
            
            # Get reference frame image
            frame_image = self.app.get_reference_frame_image()
            if frame_image is None:
                return "❌ Error: Could not load reference frame", None
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_image.astype(np.uint8))
            
            info_msg = (f"✅ Reference frame set to {frame_number_with_offset} "
                       f"(video frame {actual_frame})\n"
                       f"🎯 Ready for mask drawing")
            
            return info_msg, pil_image
            
        except Exception as e:
            error_msg = f"❌ Error setting reference frame: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, None
    
    def process_video(self, reference_video, grid_size: int) -> Tuple[str, Optional[str]]:
        """Process video with tracking and return status + preview video."""
        try:
            if reference_video is None:
                return "❌ No video loaded", None
            
            if self.app.current_video is None:
                self.app.load_video(reference_video)
            
            # Track points
            tracks, visibility = self.app.track_points(grid_size)
            
            # Create preview video
            preview_points_per_axis = min(10, int(np.sqrt(grid_size * grid_size * 0.75)))
            max_preview_points = preview_points_per_axis * preview_points_per_axis
            
            self.logger.info(f"Creating preview with {max_preview_points} points...")
            preview_video_path = self.app.create_preview_video(max_preview_points)
            
            # Get tracking info
            info = self.app.get_tracking_info()
            
            status_msg = (f"✅ Tracking completed!\n"
                         f"🎯 Points tracked: {info['num_points']}\n"
                         f"📹 Frames: {info['num_frames']}\n"
                         f"👁️ Visibility: {info['visibility_rate']:.1f}%\n"
                         f"📊 Total detections: {info['total_detections']}/{info['possible_detections']}\n"
                         f"🎬 Preview: {'Created successfully' if preview_video_path else 'Failed to create'}")
            
            return status_msg, preview_video_path
                   
        except Exception as e:
            error_msg = f"❌ Error processing video: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, None
    
    def use_mask_from_editor(self, edited_image: Any) -> str:
        """Process and use mask from Gradio ImageEditor (non-blocking)."""
        try:
            if edited_image is None:
                return "❌ No mask drawn. Please draw a mask on the reference frame."
            
            # Process mask in a non-blocking way
            self.logger.info("Processing mask from editor...")
            message, mask = self.app.process_mask_from_editor(edited_image)
            
            # Return immediately to avoid UI freeze
            return message
            
        except Exception as e:
            error_msg = f"❌ Error processing mask: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def get_default_output_path(self) -> str:
        """Get default output file path."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        return f"outputs/CoTracker_{timestamp}.nk"
    
    def export_nuke_file(self, output_file_path, frame_offset: int) -> str:
        """Export to Nuke .nk file."""
        try:
            # Handle different input types from gr.File
            if output_file_path is None or output_file_path == "":
                # Use default path
                output_path = self.get_default_output_path()
            elif isinstance(output_file_path, str):
                output_path = output_file_path
            else:
                # Handle file object or other types
                output_path = str(output_file_path)
            
            if self.app.tracking_results is None:
                return "❌ No tracking data available. Please process video first."
            
            # Ensure .nk extension
            if not output_path.endswith('.nk'):
                output_path += '.nk'
            
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export to Nuke
            nuke_path = self.app.export_to_nuke(output_path, frame_offset)
            
            # Get tracking info for summary
            info = self.app.get_tracking_info()
            
            return (f"✅ Export completed!\n"
                   f"📁 File: {nuke_path}\n"
                   f"🎯 Points: {info['num_points']}\n"
                   f"📹 Frames: {info['num_frames']}\n"
                   f"🔢 Frame offset: {frame_offset}\n"
                   f"📂 Directory: {Path(nuke_path).parent}")
                   
        except Exception as e:
            error_msg = f"❌ Export failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface."""
        with gr.Blocks(
            title="CoTracker Nuke Integration",
            theme=gr.themes.Soft(),
            css=".gradio-container {max-width: 1200px; margin: auto; width: 100%;}"
        ) as interface:
            
            gr.Markdown("""
            # 🎬 CoTracker Nuke Integration
            
            Track points in video using CoTracker and export to Nuke for seamless VFX workflows.
            """)
            
            # === STEP 1: VIDEO UPLOAD ===
            gr.Markdown("## 📹 Step 1: Upload Video")
            reference_video = gr.File(
                label="📁 Upload Video File",
                file_types=[".mp4", ".mov", ".avi", ".mkv"],
                type="filepath"
            )
            
            # Video player for uploaded video
            video_player = gr.Video(
                label="📹 Uploaded Video",
                height=300
            )
            
            video_status = gr.Textbox(
                label="📊 Video Status",
                interactive=False,
                lines=3
            )
            
            # === STEP 2: IMAGE SEQUENCE START FRAME ===
            gr.Markdown("## 🎬 Step 2: Set Image Sequence Start Frame")
            
            image_sequence_start_frame = gr.Number(
                label="🎬 Image Sequence Start Frame",
                value=1001,
                info="Frame number where your image sequence starts in Nuke"
            )
            
            # === STEP 3: REFERENCE FRAME SELECTION ===
            gr.Markdown("## 🎯 Step 3: Set Reference Frame")
            
            with gr.Row():
                with gr.Column(scale=2):
                    manual_frame_input = gr.Number(
                        label="🎯 Frame #",
                        value=1001,
                        info="Frame number for tracking reference (includes start frame offset)"
                    )
                    
                with gr.Column(scale=1):
                    set_manual_frame_btn = gr.Button(
                        "📤 Set Reference Frame",
                        variant="primary",
                        size="lg"
                    )
            
            reference_frame_info = gr.Textbox(
                label="ℹ️ Reference Frame Info",
                interactive=False,
                lines=2
            )
            
            # === STEP 4: OPTIONAL MASK DRAWING ===
            gr.Markdown("""
            ## 🎨 Step 4: Optional Mask Drawing
            Draw on the reference frame to restrict tracking to specific areas. Only points in **white areas** will be tracked.
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    mask_editor = gr.ImageEditor(
                        label="🖼️ Reference Frame - Draw Mask",
                        type="pil",
                        brush=gr.Brush(colors=["#FFFFFF", "#000000"], default_size=20),
                        height=400
                    )
                
                with gr.Column(scale=1):
                    use_mask_btn = gr.Button(
                        "🎯 Use/Update Mask",
                        variant="primary",
                        size="lg"
                    )
                    
                    mask_result = gr.Textbox(
                        label="✅ Mask Status",
                        interactive=False,
                        lines=4
                    )
            
            # === STEP 5: PROCESS VIDEO ===
            gr.Markdown("## 🚀 Step 5: Process Video")
            
            with gr.Row():
                with gr.Column(scale=2):
                    grid_size = gr.Slider(
                        minimum=5,
                        maximum=100,
                        step=1,
                        value=40,
                        label="🔢 Grid Size (Points per Side)",
                        info="Higher values = more tracking points"
                    )
                    
                with gr.Column(scale=1):
                    process_btn = gr.Button(
                        "🚀 Process Video",
                        variant="primary",
                        size="lg"
                    )
            
            processing_status = gr.Textbox(
                label="⚙️ Processing Status",
                interactive=False,
                lines=4
            )
            
            # Tracking Results Preview
            gr.Markdown("### 🎬 Tracking Results")
            preview_video = gr.Video(
                label="📹 Tracking Preview",
                height=400
            )
            
            # === STEP 6: EXPORT TO NUKE ===
            gr.Markdown("## 📤 Step 6: Export to Nuke")
            
            with gr.Row():
                output_file_path = gr.Textbox(
                    label="📁 Output File Path",
                    value=self.get_default_output_path(),
                    info="Path where the .nk file will be saved",
                    scale=3
                )
                
                file_picker_btn = gr.Button(
                    "📂 Browse",
                    size="sm",
                    scale=1
                )
            
            export_btn = gr.Button(
                "📤 Export to Nuke",
                variant="primary",
                size="lg"
            )
            
            export_status = gr.Textbox(
                label="📋 Export Status",
                interactive=False,
                lines=4
            )
            
            # Event handlers
            reference_video.change(
                fn=self.load_video_for_reference,
                inputs=[reference_video],
                outputs=[video_status, video_player]
            )
            
            set_manual_frame_btn.click(
                fn=self.set_manual_reference_frame,
                inputs=[manual_frame_input, image_sequence_start_frame],
                outputs=[reference_frame_info, mask_editor]
            )
            
            process_btn.click(
                fn=self.process_video,
                inputs=[reference_video, grid_size],
                outputs=[processing_status, preview_video]
            )
            
            # Event handlers
            
            
            # Simplified mask processing without queue to avoid freezing
            def process_mask_simple(edited_image):
                try:
                    if edited_image is None:
                        return "❌ No mask drawn. Please draw a mask on the reference frame."
                    
                    self.logger.info("Processing mask...")
                    message, mask = self.app.process_mask_from_editor(edited_image)
                    return message
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    self.logger.error(error_msg)
                    return error_msg
            
            use_mask_btn.click(
                fn=process_mask_simple,
                inputs=[mask_editor],
                outputs=[mask_result]
            )
            
            file_picker_btn.click(
                fn=lambda: gr.update(value=self.get_default_output_path()),
                outputs=[output_file_path]
            )
            
            export_btn.click(
                fn=self.export_nuke_file,
                inputs=[output_file_path, image_sequence_start_frame],
                outputs=[export_status]
            )
        
        return interface


def create_gradio_interface(debug_mode: bool = True, console_log_level: str = "INFO") -> gr.Blocks:
    """
    Create and return the Gradio interface.
    
    Args:
        debug_mode: Enable debug mode
        console_log_level: Console logging level
    
    Returns:
        Gradio Blocks interface
    """
    app = CoTrackerNukeApp(debug_mode, console_log_level)
    ui = GradioInterface(app)
    return ui.create_interface()
