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
    
    def load_video_for_reference(self, reference_video, start_frame_offset) -> Tuple[str, Optional[str], dict]:
        """Load video and return status message + video path for player + slider update."""
        try:
            if reference_video is None:
                return "❌ No video file selected", None, gr.update()
            
            # Load video
            self.app.load_video(reference_video)
            self.preview_video_path = reference_video
            
            # Get video info including FPS
            info = self.app.get_video_info()
            
            # Get FPS from video metadata
            fps_info = self.get_video_fps(reference_video)
            
            status_msg = (f"✅ Video loaded successfully!\n"
                         f"📹 Frames: {info['frames']}\n"
                         f"📐 Resolution: {info['width']}x{info['height']}\n"
                         f"🎬 FPS: {fps_info}\n"
                         f"💾 Size: {info['memory_mb']:.1f} MB")
            
            # Calculate slider range
            start_offset = start_frame_offset if start_frame_offset is not None else 1001
            max_frame = start_offset + info['frames'] - 1
            
            self.logger.info(f"Initializing frame slider range: {start_offset} to {max_frame} (total frames: {info['frames']})")
            
            slider_update = gr.update(minimum=start_offset, maximum=max_frame, value=start_offset)
            
            return status_msg, reference_video, slider_update
                   
        except Exception as e:
            error_msg = f"❌ Error loading video: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, None, gr.update()
    
    def update_frame_slider_range(self, reference_video, start_frame_offset) -> dict:
        """Update frame slider range when video is loaded."""
        try:
            if reference_video is None or self.app.current_video is None:
                return gr.update()
            
            info = self.app.get_video_info()
            start_offset = start_frame_offset if start_frame_offset is not None else 1001
            max_frame = start_offset + info['frames'] - 1
            
            self.logger.info(f"Updating frame slider range: {start_offset} to {max_frame} (total frames: {info['frames']})")
            
            return gr.update(minimum=start_offset, maximum=max_frame, value=start_offset)
            
        except Exception as e:
            self.logger.error(f"Error updating slider range: {e}")
            return gr.update()
    
    def set_manual_reference_frame(self, frame_number_with_offset: int, 
                                  start_frame_offset: int) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Set manual reference frame and return frame preview + mask editor image."""
        try:
            # Validate input
            if frame_number_with_offset < start_frame_offset:
                self.logger.error(f"Frame number {frame_number_with_offset} is less than start frame offset {start_frame_offset}")
                return None, None
            
            # Calculate 0-based video frame
            frame_number = frame_number_with_offset - start_frame_offset
            
            # Set reference frame
            actual_frame = self.app.set_reference_frame(frame_number)
            
            # Get reference frame image
            frame_image = self.app.get_reference_frame_image()
            if frame_image is None:
                self.logger.error("Could not load reference frame")
                return None, None
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_image.astype(np.uint8))
            
            self.logger.info(f"Reference frame set to {frame_number_with_offset} (video frame {actual_frame})")
            
            # Return the same image for both frame preview and mask editor
            return pil_image, pil_image
            
        except Exception as e:
            error_msg = f"Error setting reference frame: {str(e)}"
            self.logger.error(error_msg)
            return None, None
    
    def process_video(self, reference_video, grid_size: int, image_sequence_start_frame: int = 1001) -> Tuple[str, Optional[str]]:
        """Process video with tracking and return status + preview video."""
        try:
            if reference_video is None:
                return "❌ No video loaded", None
            
            if self.app.current_video is None:
                self.app.load_video(reference_video)
            
            # Track points
            tracks, visibility = self.app.track_points(grid_size)
            
            # Create preview video with all points
            self.logger.info(f"Creating preview with all generated points...")
            preview_video_path = self.app.create_preview_video(frame_offset=image_sequence_start_frame)
            
            # Get tracking info
            info = self.app.get_tracking_info()
            
            # Get reference frame and mask info
            ref_frame_internal = self.app.reference_frame
            ref_frame_display = ref_frame_internal + image_sequence_start_frame
            has_mask = self.app.mask_handler.current_mask is not None
            mask_status = "✅ Used" if has_mask else "❌ None"
            
            status_msg = (f"✅ Tracking completed!\n"
                         f"🎯 Points tracked: {info['num_points']}\n"
                         f"📹 Frames: {info['num_frames']}\n"
                         f"🎬 Reference frame: {ref_frame_display}\n"
                         f"🎭 Mask: {mask_status}\n"
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
    
    def browse_output_folder(self) -> str:
        """Open file dialog to browse for output location."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            # Create a root window and hide it
            root = tk.Tk()
            root.withdraw()
            
            # Get current output path directory
            current_path = self.get_default_output_path()
            current_dir = os.path.dirname(os.path.abspath(current_path))
            
            # Open file dialog
            file_path = filedialog.asksaveasfilename(
                title="Save Nuke file as...",
                initialdir=current_dir,
                defaultextension=".nk",
                filetypes=[("Nuke files", "*.nk"), ("All files", "*.*")],
                initialfile=os.path.basename(current_path)
            )
            
            # Clean up the root window
            root.destroy()
            
            if file_path:
                # Convert to forward slashes and return relative path if possible
                file_path = file_path.replace('\\', '/')
                try:
                    # Try to make it relative to current working directory
                    rel_path = os.path.relpath(file_path)
                    return rel_path.replace('\\', '/')
                except ValueError:
                    # If relative path fails, return absolute path
                    return file_path
            else:
                # User cancelled, return current path
                return self.get_default_output_path()
                
        except ImportError:
            self.logger.warning("tkinter not available, using default path")
            return self.get_default_output_path()
        except Exception as e:
            self.logger.error(f"Error in file browser: {e}")
            return self.get_default_output_path()
    
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
            
            # Regular video player for realtime playback
            video_player = gr.Video(
                label="📹 Video Player",
                height=300
            )
            
            video_status = gr.Textbox(
                label="📊 Video Status",
                interactive=False,
                lines=4
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
                    frame_display = gr.Image(
                        label="🖼️ Reference Frame Preview",
                        height=300,
                        type="pil"
                    )
                
                with gr.Column(scale=1):
                    frame_slider = gr.Slider(
                        minimum=1001,
                        maximum=1100,  # Will be updated when video loads
                        step=1,
                        value=1001,
                        label="🎬 Frame #",
                        info="Frame number for tracking reference (includes start frame offset)"
                    )
                    
                    set_manual_frame_btn = gr.Button(
                        "📤 Set Reference Frame",
                        variant="primary",
                        size="lg"
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
                    
                    vram_warning = gr.Textbox(
                        label="⚠️ VRAM Warning",
                        interactive=False,
                        lines=2,
                        visible=False
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
                inputs=[reference_video, image_sequence_start_frame],
                outputs=[video_status, video_player, frame_slider]
            )
            
            image_sequence_start_frame.change(
                fn=self.update_frame_slider_range,
                inputs=[reference_video, image_sequence_start_frame],
                outputs=[frame_slider]
            )
            
            set_manual_frame_btn.click(
                fn=self.set_manual_reference_frame,
                inputs=[frame_slider, image_sequence_start_frame],
                outputs=[frame_display, mask_editor]
            )
            
            # Update frame display only on slider release (not during dragging)
            frame_slider.release(
                fn=self.update_frame_from_input,
                inputs=[frame_slider, image_sequence_start_frame],
                outputs=[frame_display]
            )
            
            # Check VRAM warning when grid size changes
            grid_size.change(
                fn=self.check_vram_warning,
                inputs=[grid_size],
                outputs=[vram_warning]
            )
            
            process_btn.click(
                fn=self.process_video,
                inputs=[reference_video, grid_size, image_sequence_start_frame],
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
            
            # Update VRAM warning when mask is used
            use_mask_btn.click(
                fn=self.check_vram_warning,
                inputs=[grid_size],
                outputs=[vram_warning]
            )
            
            file_picker_btn.click(
                fn=lambda: gr.update(value=self.browse_output_folder()),
                outputs=[output_file_path]
            )
            
            export_btn.click(
                fn=self.export_nuke_file,
                inputs=[output_file_path, image_sequence_start_frame],
                outputs=[export_status]
            )
            
        
        return interface
    
    def get_video_fps(self, video_path: str) -> str:
        """Get FPS information from video metadata."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return f"{fps:.2f}" if fps > 0 else "Unknown"
        except Exception as e:
            self.logger.warning(f"Could not get FPS: {e}")
            return "Unknown"
    
    def update_frame_from_input(self, frame_number_with_offset: int, start_frame_offset: int) -> Optional[Image.Image]:
        """Update frame display when Frame # input changes."""
        try:
            if self.app.current_video is None:
                return None
            
            # Calculate 0-based video frame
            if frame_number_with_offset < start_frame_offset:
                return None
            
            frame_number = frame_number_with_offset - start_frame_offset
            
            # Get the requested frame
            frame = self.app.video_processor.get_frame(int(frame_number))
            if frame is None:
                return None
            
            # Convert to PIL Image
            frame_pil = Image.fromarray(frame.astype(np.uint8))
            return frame_pil
            
        except Exception as e:
            self.logger.error(f"Error displaying frame {frame_number_with_offset}: {str(e)}")
            return None
    
    def check_vram_warning(self, grid_size: int) -> dict:
        """Check if VRAM warning should be displayed."""
        try:
            if grid_size > 50:
                # Check if mask is available
                has_mask = self.app.mask_handler.current_mask is not None
                
                if not has_mask:
                    warning_msg = (f"⚠️ High VRAM usage warning!\n"
                                 f"Grid size {grid_size} without mask = {grid_size * grid_size:,} points.\n"
                                 f"Consider using a mask or reducing grid size to avoid GPU memory issues.")
                    return gr.update(value=warning_msg, visible=True)
            
            # Hide warning if conditions not met
            return gr.update(visible=False)
            
        except Exception as e:
            self.logger.error(f"Error checking VRAM warning: {e}")
            return gr.update(visible=False)


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
