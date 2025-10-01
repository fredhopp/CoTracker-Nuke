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
import os
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
        self.last_exported_path = None  # Store last exported .nk file path
    
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
            root.wm_attributes('-topmost', 1)  # Keep dialog on top
            
            # Get current output path directory
            current_path = self.get_default_output_path()
            current_dir = os.path.dirname(os.path.abspath(current_path))
            
            # Ensure output directory exists
            os.makedirs(current_dir, exist_ok=True)
            
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
                self.logger.info("File dialog cancelled by user")
                return self.get_default_output_path()
                
        except ImportError:
            self.logger.warning("tkinter not available for file dialog, using default path")
            return self.get_default_output_path()
        except Exception as e:
            self.logger.error(f"Error in file browser: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self.get_default_output_path()
    
    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to clipboard using multiple fallback methods with Windows 11 fixes."""
        import sys
        
        try:
            # Method 1: Try pyperclip (most reliable, especially on Windows 11)
            try:
                import pyperclip
                pyperclip.copy(text)
                # Verify the copy worked
                if pyperclip.paste() == text:
                    self.logger.info("Clipboard copy successful via pyperclip")
                    return True
                else:
                    self.logger.warning("pyperclip copy verification failed")
            except ImportError:
                self.logger.debug("pyperclip not available")
            except Exception as e:
                self.logger.warning(f"pyperclip failed: {e}")
            
            # Method 2: Windows-specific methods (better for Windows 11)
            if sys.platform == "win32":
                # Try Windows PowerShell method (more reliable on Windows 11)
                try:
                    import subprocess
                    powershell_cmd = f'Set-Clipboard -Value "{text}"'
                    result = subprocess.run(
                        ['powershell', '-Command', powershell_cmd], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    if result.returncode == 0:
                        self.logger.info("Clipboard copy successful via PowerShell")
                        return True
                    else:
                        self.logger.warning(f"PowerShell clipboard failed: {result.stderr}")
                except Exception as e:
                    self.logger.warning(f"PowerShell method failed: {e}")
                
                # Try traditional Windows clip command
                try:
                    import subprocess
                    result = subprocess.run(['clip'], input=text, text=True, check=True, timeout=5)
                    self.logger.info("Clipboard copy successful via Windows clip")
                    return True
                except Exception as e:
                    self.logger.warning(f"Windows clip failed: {e}")
            
            # Method 3: Enhanced tkinter clipboard (with Windows 11 fixes)
            try:
                import tkinter as tk
                import time
                
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes('-topmost', 1)  # Windows 11 fix
                
                # Clear and set clipboard
                root.clipboard_clear()
                root.clipboard_append(text)
                
                # Multiple update calls for Windows 11 reliability
                for _ in range(3):
                    root.update_idletasks()
                    root.update()
                    time.sleep(0.01)  # Small delay for Windows 11
                
                # Verify clipboard content
                try:
                    clipboard_content = root.clipboard_get()
                    if clipboard_content == text:
                        self.logger.info("Clipboard copy successful via tkinter")
                        root.destroy()
                        return True
                    else:
                        self.logger.warning("tkinter clipboard verification failed")
                except tk.TclError:
                    self.logger.warning("Could not verify tkinter clipboard content")
                
                root.destroy()
                
            except Exception as e:
                self.logger.warning(f"tkinter clipboard failed: {e}")
            
            # Method 4: Platform-specific fallbacks
            if sys.platform == "darwin":
                # macOS
                try:
                    import subprocess
                    subprocess.run(['pbcopy'], input=text, text=True, check=True, timeout=5)
                    self.logger.info("Clipboard copy successful via pbcopy")
                    return True
                except Exception as e:
                    self.logger.warning(f"macOS pbcopy failed: {e}")
                    
            elif sys.platform.startswith("linux"):
                # Linux - try xclip first, then xsel
                try:
                    import subprocess
                    subprocess.run(['xclip', '-selection', 'clipboard'], input=text, text=True, check=True, timeout=5)
                    self.logger.info("Clipboard copy successful via xclip")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    self.logger.warning(f"xclip failed: {e}")
                    try:
                        subprocess.run(['xsel', '--clipboard', '--input'], input=text, text=True, check=True, timeout=5)
                        self.logger.info("Clipboard copy successful via xsel")
                        return True
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        self.logger.warning(f"xsel failed: {e}")
            
            self.logger.error("All clipboard methods failed")
            return False
            
        except Exception as e:
            self.logger.error(f"Clipboard copy failed with exception: {e}")
            return False

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
            
            # Store the exported path for the copy button
            self.last_exported_path = nuke_path
            
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
    
    def copy_exported_path(self) -> str:
        """Copy the last exported .nk file path to clipboard."""
        if self.last_exported_path is None:
            return "❌ No file has been exported yet. Please export a .nk file first."
        
        success = self.copy_to_clipboard(self.last_exported_path)
        if success:
            return f"📋 Copied to clipboard!\n{self.last_exported_path}"
        else:
            return f"⚠️ Could not copy to clipboard.\nPath: {self.last_exported_path}"
    
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
                        height=400,
                        interactive=True
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
                        maximum=400,
                        step=1,
                        value=40,
                        label="🔢 Grid Size (Points on Longest Side)",
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
                "📤 Generate Tracker Node as .nk",
                variant="primary",
                size="lg"
            )
            
            export_status = gr.Textbox(
                label="📋 Export Status",
                interactive=False,
                lines=4
            )
            
            copy_path_btn = gr.Button(
                "📋 Copy .nk Path to Clipboard",
                variant="primary",
                size="lg"
            )
            
            copy_status = gr.Textbox(
                label="📋 Copy Status",
                interactive=False,
                lines=2
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
            
            # Update grid info on slider release
            grid_size.release(
                fn=self.calculate_grid_info,
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
            
            # Update grid info when mask is used
            use_mask_btn.click(
                fn=self.calculate_grid_info,
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
            
            copy_path_btn.click(
                fn=self.copy_exported_path,
                outputs=[copy_status]
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
    
    def calculate_grid_info(self, grid_size: int) -> dict:
        """Calculate and display grid point information in VRAM warning area."""
        try:
            # Get video dimensions if available
            if self.app.video_processor.current_video is not None:
                height, width = self.app.video_processor.current_video.shape[1:3]
                
                # Calculate grid dimensions based on aspect ratio
                if width >= height:
                    grid_width = grid_size
                    grid_height = max(1, int(round(grid_size * height / width)))
                else:
                    grid_height = grid_size
                    grid_width = max(1, int(round(grid_size * width / height)))
                
                total_points = grid_width * grid_height
                
                # Check if mask is available
                has_mask = self.app.mask_handler.current_mask is not None
                
                if has_mask:
                    mask = self.app.mask_handler.current_mask
                    white_pixels = np.sum(mask == 255)
                    total_pixels = mask.shape[0] * mask.shape[1]
                    coverage = white_pixels / total_pixels
                    estimated_masked_points = int(total_points * coverage)
                    
                    info_text = f"📊 Grid: {grid_width}×{grid_height} = {total_points:,} points\n✅ With mask: ≈{estimated_masked_points:,} points ({coverage*100:.1f}% coverage)"
                    
                    # Add VRAM warning if masked points exceed 300
                    if estimated_masked_points > 300:
                        info_text += f"\n⚠️ High VRAM usage: {estimated_masked_points:,} points may cause GPU memory issues"
                else:
                    info_text = f"📊 Grid: {grid_width}×{grid_height} = {total_points:,} points (no mask)"
                    
                    # Add VRAM warning if total points exceed 300
                    if total_points > 300:
                        info_text += f"\n⚠️ High VRAM usage: {total_points:,} points may cause GPU memory issues"
                
                # Show in VRAM warning area
                return gr.update(value=info_text, visible=True)
            else:
                return gr.update(value="⚠️ Load video first to calculate points", visible=True)
                
        except Exception as e:
            self.logger.error(f"Error calculating grid info: {e}")
            return gr.update(value="❌ Error calculating points", visible=True)
    
    def check_vram_warning(self, grid_size: int) -> dict:
        """Check if VRAM warning should be displayed."""
        try:
            if grid_size > 50:
                # Check if mask is available
                has_mask = self.app.mask_handler.current_mask is not None
                
                if not has_mask:
                    # Estimate points (actual count depends on aspect ratio)
                    # For 16:9 (most common): grid_size * (grid_size * 9/16)
                    estimated_points = int(grid_size * grid_size * 0.56)  # Approximate for 16:9
                    warning_msg = (f"⚠️ High VRAM usage warning!\n"
                                 f"Grid size {grid_size} without mask ≈ {estimated_points:,} points (aspect-ratio adjusted).\n"
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
