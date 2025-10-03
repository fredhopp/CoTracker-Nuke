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
from datetime import datetime

from ..core.app import CoTrackerNukeApp
from ..exporters.stmap_exporter import STMapExporter


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
        self.last_stmap_path = None  # Store last exported STMap directory path
        self.stmap_output_path = None  # Store STMap output file path
        self.last_animated_mask_path = None  # Store last exported animated mask directory path
    
    def load_video_for_reference(self, reference_video, start_frame_offset) -> Tuple[str, Optional[str], dict, dict, dict]:
        """Load video and return status message + video path for player + slider update."""
        try:
            if reference_video is None:
                return "❌ No video file selected", None, gr.update(), gr.update(), gr.update()
            
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
            
            # Update STMap frame defaults
            stmap_start_update = gr.update(value=start_offset)
            stmap_end_update = gr.update(value=max_frame)
            
            return status_msg, reference_video, slider_update, stmap_start_update, stmap_end_update
                   
        except Exception as e:
            error_msg = f"❌ Error loading video: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, None, gr.update(), gr.update(), gr.update()
    
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
    
    def get_default_stmap_output_path(self) -> str:
        """Get default STMap output file path."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        return f"outputs/CoTracker_{timestamp}_stmap/CoTracker_{timestamp}_stmap.%04d.exr"
    
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
    
    def browse_stmap_output_folder(self) -> str:
        """Open file dialog to browse for STMap output location."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            # Create a root window and hide it
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)  # Keep dialog on top
            
            # Get current output path directory
            current_path = self.get_default_stmap_output_path()
            current_dir = os.path.dirname(os.path.abspath(current_path))
            
            # Ensure output directory exists
            os.makedirs(current_dir, exist_ok=True)
            
            # Open file dialog
            file_path = filedialog.asksaveasfilename(
                title="Save STMap sequence as...",
                initialdir=current_dir,
                defaultextension=".exr",
                filetypes=[("EXR files", "*.exr"), ("All files", "*.*")],
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
                self.logger.info("STMap file dialog cancelled by user")
                return self.get_default_stmap_output_path()
                
        except ImportError:
            self.logger.warning("tkinter not available for file dialog, using default path")
            return self.get_default_stmap_output_path()
        except Exception as e:
            self.logger.error(f"Error in STMap file browser: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self.get_default_stmap_output_path()
    
    def update_stmap_frame_defaults(self, reference_video, image_sequence_start_frame) -> Tuple[dict, dict]:
        """Update STMap frame defaults based on video and image sequence start frame."""
        try:
            if reference_video is None or self.app.current_video is None:
                return gr.update(), gr.update()
            
            # Get video info
            info = self.app.get_video_info()
            start_frame = image_sequence_start_frame if image_sequence_start_frame is not None else 1001
            end_frame = start_frame + info['frames'] - 1
            
            return gr.update(value=start_frame), gr.update(value=end_frame)
            
        except Exception as e:
            self.logger.error(f"Error updating STMap frame defaults: {e}")
            return gr.update(), gr.update()
    
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
    
    def export_stmap_sequence(self, 
                            interpolation_method: str,
                            bit_depth: int,
                            frame_start: int,
                            frame_end: Optional[int],
                            image_sequence_start_frame: int = 1001,
                            output_file_path: Optional[str] = None,
                            progress=gr.Progress()) -> str:
        """Export tracking data to animated STMap sequence."""
        try:
            if self.app.tracking_results is None:
                return "❌ No tracking data available. Please process video first."
            
            # Get tracking data
            tracks, visibility = self.app.tracking_results
            
            # Get video dimensions
            if self.app.video_processor.current_video is not None:
                height, width = self.app.video_processor.current_video.shape[1:3]
            else:
                return "❌ No video loaded. Please load a video first."
            
            # Get mask if available
            mask = self.app.mask_handler.current_mask
            
            # Determine output path
            if output_file_path is None or output_file_path == "":
                output_path = self.get_default_stmap_output_path()
            else:
                output_path = output_file_path
            
            # Extract directory and filename pattern from output path
            output_dir = Path(output_path).parent
            filename_pattern = Path(output_path).name
            
            # Create STMap exporter
            stmap_exporter = STMapExporter(
                debug_dir=output_dir,
                logger=self.app.logger
            )
            
            # Set parameters
            stmap_exporter.set_reference_frame(self.app.reference_frame)
            stmap_exporter.set_video_dimensions(width, height)
            
            # Convert frame range to 0-based video frames
            if frame_start is not None:
                video_frame_start = max(0, frame_start - image_sequence_start_frame)
            else:
                video_frame_start = 0  # Default to first frame
                
            if frame_end is not None:
                video_frame_end = frame_end - image_sequence_start_frame
            else:
                # Default to last frame
                video_frame_end = None
            
            # Progress tracking with Gradio
            def progress_callback(current, total):
                progress(current / total, desc=f"Processing frame {current}/{total}")
                self.logger.info(f"Processing frame {current}/{total}")
            
            # Generate STMap sequence
            output_dir = stmap_exporter.generate_stmap_sequence(
                tracks=tracks,
                visibility=visibility,
                mask=mask,
                interpolation_method=interpolation_method,
                bit_depth=bit_depth,
                frame_start=video_frame_start,
                frame_end=video_frame_end,
                filename_pattern=filename_pattern,
                frame_offset=image_sequence_start_frame,
                progress_callback=progress_callback
            )
            
            # Store the exported path (make it absolute)
            absolute_output_dir = str(Path(output_dir).resolve())
            self.last_stmap_path = absolute_output_dir
            
            # Copy mask PNG to output folder if available
            mask_path = None
            if mask is not None:
                try:
                    # Find the most recent mask file in the debug directory
                    debug_dir = Path(self.app.mask_handler.debug_dir)
                    mask_files = list(debug_dir.glob("drawn_mask_*.png"))
                    if mask_files:
                        # Get the most recent mask file
                        latest_mask = max(mask_files, key=lambda x: x.stat().st_mtime)
                        
                        # Create new filename with reference frame
                        reference_frame_display = self.app.reference_frame + image_sequence_start_frame
                        original_name = latest_mask.stem  # Remove .png extension
                        new_filename = f"{original_name}_{reference_frame_display}.png"
                        
                        # Convert mask to RGBA and save to output directory with new name
                        output_mask_path = Path(output_dir) / new_filename
                        self._convert_mask_to_rgba(latest_mask, output_mask_path)
                        
                        # Make path absolute
                        mask_path = str(output_mask_path.resolve())
                        self.logger.info(f"Copied and converted mask to: {mask_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to copy mask: {e}")
            
            # Get tracking info for summary
            info = self.app.get_tracking_info()
            
            # Count generated files
            output_path = Path(output_dir)
            exr_files = list(output_path.glob("*.exr"))
            
            # Get reference frame info
            reference_frame_display = self.app.reference_frame + image_sequence_start_frame
            
            status_msg = (f"✅ STMap sequence generated!\n"
                         f"📁 Directory: {absolute_output_dir}\n"
                         f"🎯 Points: {info['num_points']}\n"
                         f"📹 Frames: {len(exr_files)} EXR files\n"
                         f"🎬 Reference frame: {reference_frame_display}\n"
                         f"🔧 Interpolation: {interpolation_method}\n"
                         f"💾 Bit depth: {bit_depth}-bit float\n"
                         f"🎭 Mask: {'Used' if mask is not None else 'None'}")
            
            if mask_path:
                status_msg += f"\n🎭 Mask copied to: {mask_path}"
            
            return status_msg
                   
        except Exception as e:
            error_msg = f"❌ STMap export failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def _convert_mask_to_rgba(self, input_path: Path, output_path: Path):
        """
        Convert monochromatic mask PNG to RGBA format.
        
        Args:
            input_path: Path to input mask file
            output_path: Path to output RGBA mask file
        """
        try:
            from PIL import Image
            import numpy as np
            
            # Load the mask image
            mask_image = Image.open(input_path)
            
            # Convert to numpy array
            mask_array = np.array(mask_image)
            
            # Handle different input formats
            if len(mask_array.shape) == 2:  # Grayscale
                # Convert to RGBA where R=G=B=A=original_value
                rgba_array = np.stack([mask_array, mask_array, mask_array, mask_array], axis=-1)
            elif len(mask_array.shape) == 3 and mask_array.shape[2] == 3:  # RGB
                # Convert to RGBA where A=original_R (assuming grayscale input)
                alpha = mask_array[:, :, 0]  # Use red channel as alpha
                rgba_array = np.stack([mask_array[:, :, 0], mask_array[:, :, 1], mask_array[:, :, 2], alpha], axis=-1)
            elif len(mask_array.shape) == 3 and mask_array.shape[2] == 4:  # Already RGBA
                rgba_array = mask_array
            else:
                raise ValueError(f"Unsupported mask format: {mask_array.shape}")
            
            # Create RGBA image and save
            rgba_image = Image.fromarray(rgba_array.astype(np.uint8), 'RGBA')
            rgba_image.save(output_path)
            
            self.logger.debug(f"Converted mask to RGBA: {input_path} -> {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to convert mask to RGBA: {e}")
            # Fallback: just copy the original file
            import shutil
            shutil.copy2(input_path, output_path)
    
    def export_animated_mask_sequence(self, image_sequence_start_frame: int = 1001) -> str:
        """Export animated mask sequence that follows tracked points."""
        try:
            self.logger.info(f"Starting animated mask export with start frame: {image_sequence_start_frame}")
            
            if self.app.tracking_results is None:
                self.logger.warning("No tracking data available")
                return "❌ No tracking data available. Please process video first."
            
            if self.app.mask_handler.current_mask is None:
                self.logger.warning("No mask available")
                return "❌ No mask available. Please draw a mask first."
            
            self.logger.info("Tracking data and mask found, proceeding with export...")
            
            # Get tracking data
            tracks, visibility = self.app.tracking_results
            self.logger.info(f"Got tracking data: tracks shape {tracks.shape}, visibility shape {visibility.shape}")
            
            # Get video dimensions
            if self.app.video_processor.current_video is not None:
                height, width = self.app.video_processor.current_video.shape[1:3]
                self.logger.info(f"Video dimensions: {width}x{height}")
            else:
                self.logger.warning("No video loaded")
                return "❌ No video loaded. Please load a video first."
            
            # Get the original mask
            original_mask = self.app.mask_handler.current_mask
            self.logger.info(f"Original mask shape: {original_mask.shape}")
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs") / f"CoTracker_{timestamp}_animated_mask"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")
            
            # Convert tensors to numpy
            tracks_np = tracks[0].cpu().numpy()  # Shape: (T, N, 2)
            visibility_np = visibility[0].cpu().numpy()  # Shape: (T, N)
            
            # Handle different visibility shapes
            if len(visibility_np.shape) == 3:
                visibility_np = visibility_np[:, :, 0]
            
            T, N, _ = tracks_np.shape
            
            # Get reference frame tracks
            reference_frame = self.app.reference_frame
            reference_tracks = tracks_np[reference_frame]
            reference_visibility = visibility_np[reference_frame]
            
            # Filter visible reference trackers
            visible_mask = reference_visibility > 0.5
            if not np.any(visible_mask):
                return "❌ No visible trackers in reference frame."
            
            visible_reference_tracks = reference_tracks[visible_mask]
            
            # Generate animated mask for each frame
            self.logger.info(f"Processing {T} frames...")
            for frame_idx in range(T):
                if frame_idx % 10 == 0:  # Log every 10 frames
                    self.logger.info(f"Processing frame {frame_idx}/{T}")
                
                # Get current frame tracks
                current_tracks = tracks_np[frame_idx]
                current_visibility = visibility_np[frame_idx]
                
                # Filter visible trackers in current frame
                current_visible_mask = current_visibility > 0.5
                visible_count = np.sum(current_visible_mask)
                
                if not np.any(current_visible_mask):
                    # If no visible trackers in current frame, use reference mask
                    self.logger.warning(f"Frame {frame_idx}: No visible trackers, using reference mask")
                    animated_mask = original_mask.copy()
                else:
                    # Get visible trackers from current frame
                    visible_current_tracks = current_tracks[current_visible_mask]
                    
                    # Get corresponding reference trackers (same indices)
                    visible_reference_tracks_current = reference_tracks[current_visible_mask]
                    
                    self.logger.debug(f"Frame {frame_idx}: {visible_count} visible trackers")
                    
                    # Warp mask based on tracker movement
                    animated_mask = self._warp_mask_with_trackers(
                        original_mask, 
                        visible_reference_tracks_current, 
                        visible_current_tracks
                    )
                
                # Save as PNG
                actual_frame_number = frame_idx + image_sequence_start_frame
                filename = f"animated_mask_{actual_frame_number:04d}.png"
                filepath = output_dir / filename
                
                # Convert to PIL and save
                mask_image = Image.fromarray(animated_mask.astype(np.uint8))
                mask_image.save(filepath)
            
            # Store the exported path
            absolute_output_dir = str(output_dir.resolve())
            self.last_animated_mask_path = absolute_output_dir
            
            # Count generated files
            mask_files = list(output_dir.glob("animated_mask_*.png"))
            self.logger.info(f"Generated {len(mask_files)} mask files in {absolute_output_dir}")
            
            success_msg = (f"✅ Animated mask sequence generated!\n"
                          f"📁 Directory: {absolute_output_dir}\n"
                          f"📹 Frames: {len(mask_files)} PNG files\n"
                          f"🎬 Reference frame: {reference_frame + image_sequence_start_frame}\n"
                          f"🎯 Trackers used: {len(visible_reference_tracks)} visible points")
            
            self.logger.info("Animated mask export completed successfully")
            return success_msg
                   
        except Exception as e:
            error_msg = f"❌ Animated mask export failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def _warp_mask_with_trackers(self, mask: np.ndarray, reference_tracks: np.ndarray, current_tracks: np.ndarray) -> np.ndarray:
        """
        Hybrid mask warping: interpolation inside tracker bounds, block offset outside.
        Uses same logic as STMap for smooth areas, block movement for sparse areas.
        """
        try:
            self.logger.debug(f"Starting mask warping: mask shape {mask.shape}, ref tracks {reference_tracks.shape}, curr tracks {current_tracks.shape}")
            
            from scipy.interpolate import griddata
            
            height, width = mask.shape
            warped_mask = np.zeros_like(mask)
            
            # Create coordinate grids
            self.logger.debug("Creating coordinate grids...")
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
            self.logger.debug(f"Created {len(points)} coordinate points")
            
            # Calculate displacement vectors (reference - current) for backward mapping
            self.logger.debug("Calculating displacement vectors...")
            displacement_vectors = reference_tracks - current_tracks
            self.logger.debug(f"Displacement vectors shape: {displacement_vectors.shape}")
            
            # Interpolate displacement vectors using linear interpolation
            self.logger.debug("Starting griddata interpolation...")
            interpolated_displacements = griddata(
                reference_tracks,
                displacement_vectors,
                points,
                method='linear',
                fill_value=np.nan
            )
            self.logger.debug("Griddata interpolation completed")
            
            # Handle NaN values (outside convex hull) with block offset logic
            self.logger.debug("Handling NaN values...")
            nan_mask = np.isnan(interpolated_displacements[:, 0])
            nan_count = np.sum(nan_mask)
            self.logger.debug(f"Found {nan_count} NaN values to handle")
            
            if np.any(nan_mask):
                # For pixels outside convex hull, use closest tracker offset
                nan_points = points[nan_mask]
                self.logger.debug(f"Processing {len(nan_points)} NaN points...")
                
                # Vectorized approach: find closest tracker for all NaN points at once
                # Reshape for broadcasting: (N_nan, 1, 2) - (1, N_trackers, 2)
                nan_points_reshaped = nan_points[:, np.newaxis, :]  # (N_nan, 1, 2)
                ref_tracks_reshaped = reference_tracks[np.newaxis, :, :]  # (1, N_trackers, 2)
                
                # Calculate distances for all combinations at once
                distances = np.sqrt(np.sum((nan_points_reshaped - ref_tracks_reshaped)**2, axis=2))  # (N_nan, N_trackers)
                
                # Find closest tracker for each NaN point
                closest_indices = np.argmin(distances, axis=1)  # (N_nan,)
                
                # Use displacement from closest tracker
                interpolated_displacements[nan_mask] = displacement_vectors[closest_indices]
                self.logger.debug("Completed vectorized NaN handling")
            
            # Reshape interpolated displacements back to image shape
            self.logger.debug("Reshaping displacements...")
            dx = interpolated_displacements[:, 0].reshape(height, width)
            dy = interpolated_displacements[:, 1].reshape(height, width)
            
            # Create source coordinates
            source_x = x_coords + dx
            source_y = y_coords + dy
            
            # Warp the mask using vectorized bilinear interpolation
            self.logger.debug("Starting vectorized bilinear interpolation...")
            
            # Create masks for valid coordinates
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
            
            self.logger.debug("Mask warping completed successfully")
            return warped_mask
            
        except Exception as e:
            self.logger.error(f"Error warping mask: {e}")
            return mask  # Return original mask if warping fails
    
    def copy_stmap_path(self) -> str:
        """Copy the last exported STMap directory path to clipboard."""
        if self.last_stmap_path is None:
            return "❌ No STMap sequence has been exported yet. Please export STMap first."
        
        success = self.copy_to_clipboard(self.last_stmap_path)
        if success:
            return f"📋 Copied to clipboard!\n{self.last_stmap_path}"
        else:
            return f"⚠️ Could not copy to clipboard.\nPath: {self.last_stmap_path}"
    
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
            
            # === STEP 7: EXPORT STMAP SEQUENCE ===
            gr.Markdown("## 🗺️ Step 7: Export STMap Sequence")
            gr.Markdown("""
            Generate an animated STMap sequence for geometric transformations in Nuke.
            STMap uses UV coordinates where pixel values represent source positions for warping.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        stmap_interpolation = gr.Dropdown(
                            choices=["linear", "cubic"],
                            value="linear",
                            label="🔧 Interpolation",
                            info="Linear: Fast, Cubic: Smooth",
                            scale=1
                        )
                        
                        stmap_bit_depth = gr.Dropdown(
                            choices=[16, 32],
                            value=32,
                            label="💾 Bit Depth",
                            info="32-bit: Max precision, 16-bit: Smaller files",
                            scale=1
                        )
                
                with gr.Column(scale=1):
                    with gr.Row():
                        stmap_frame_start = gr.Number(
                            value=None,
                            label="🎬 Start Frame",
                            scale=1
                        )
                        
                        stmap_frame_end = gr.Number(
                            value=None,
                            label="🎬 End Frame",
                            scale=1
                        )
            
            with gr.Row():
                stmap_output_file_path = gr.Textbox(
                    label="📁 STMap Output File Path",
                    value=self.get_default_stmap_output_path(),
                    info="Path pattern for EXR sequence (use %04d for frame numbers)",
                    scale=3
                )
                
                stmap_file_picker_btn = gr.Button(
                    "📂 Browse",
                    size="sm",
                    scale=1
                )
            
            stmap_export_btn = gr.Button(
                "🗺️ Generate STMap Sequence",
                variant="primary",
                size="lg"
            )
            
            stmap_progress = gr.Progress()
            
            stmap_export_status = gr.Textbox(
                label="📋 STMap Export Status",
                interactive=False,
                lines=4
            )
            
            stmap_copy_path_btn = gr.Button(
                "📋 Copy STMap Directory Path",
                variant="primary",
                size="lg"
            )
            
            stmap_copy_status = gr.Textbox(
                label="📋 STMap Copy Status",
                interactive=False,
                lines=2
            )
            
            # === ANIMATED MASK EXPORT ===
            gr.Markdown("## 🎭 Animated Mask Export")
            gr.Markdown("""
            Export the mask as an animated sequence that follows the tracked points.
            The mask will move as coherent blocks based on the closest tracker points.
            """)
            
            animated_mask_export_btn = gr.Button(
                "🎭 Export Animated Mask Sequence",
                variant="primary",
                size="lg"
            )
            
            animated_mask_export_status = gr.Textbox(
                label="📋 Animated Mask Export Status",
                interactive=False,
                lines=4
            )
            
            # Event handlers
            reference_video.change(
                fn=self.load_video_for_reference,
                inputs=[reference_video, image_sequence_start_frame],
                outputs=[video_status, video_player, frame_slider, stmap_frame_start, stmap_frame_end]
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
            def process_mask_and_update_grid(edited_image, grid_size):
                try:
                    if edited_image is None:
                        return "❌ No mask drawn. Please draw a mask on the reference frame.", gr.update()
                    
                    self.logger.info("Processing mask...")
                    message, mask = self.app.process_mask_from_editor(edited_image)
                    
                    # Update grid info after mask processing
                    grid_info = self.calculate_grid_info(grid_size)
                    
                    return message, grid_info
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    self.logger.error(error_msg)
                    return error_msg, gr.update()
            
            use_mask_btn.click(
                fn=process_mask_and_update_grid,
                inputs=[mask_editor, grid_size],
                outputs=[mask_result, vram_warning]
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
            
            # STMap export event handlers
            stmap_file_picker_btn.click(
                fn=lambda: gr.update(value=self.browse_stmap_output_folder()),
                outputs=[stmap_output_file_path]
            )
            
            # Update STMap frame defaults when video loads
            reference_video.change(
                fn=self.update_stmap_frame_defaults,
                inputs=[reference_video, image_sequence_start_frame],
                outputs=[stmap_frame_start, stmap_frame_end]
            )
            
            # Update STMap frame defaults when image sequence start frame changes
            image_sequence_start_frame.change(
                fn=self.update_stmap_frame_defaults,
                inputs=[reference_video, image_sequence_start_frame],
                outputs=[stmap_frame_start, stmap_frame_end]
            )
            
            stmap_export_btn.click(
                fn=self.export_stmap_sequence,
                inputs=[stmap_interpolation, stmap_bit_depth, stmap_frame_start, stmap_frame_end, image_sequence_start_frame, stmap_output_file_path],
                outputs=[stmap_export_status]
            )
            
            stmap_copy_path_btn.click(
                fn=self.copy_stmap_path,
                outputs=[stmap_copy_status]
            )
            
            animated_mask_export_btn.click(
                fn=self.export_animated_mask_sequence,
                inputs=[image_sequence_start_frame],
                outputs=[animated_mask_export_status]
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
