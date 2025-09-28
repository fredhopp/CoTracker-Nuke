#!/usr/bin/env python3
"""
Simple Mask Drawing Tool
========================

A Gradio interface for drawing simple masks on video frames using Gradio's built-in ImageEditor.

Features:
- Load video and select reference frame
- Draw masks with Gradio's built-in brush tools
- Export as black/white PNG mask
"""

import gradio as gr
import numpy as np
import cv2
import imageio.v3 as iio
from pathlib import Path
import logging
from datetime import datetime
from PIL import Image


class SimpleMaskTool:
    """Simple mask drawing tool using Gradio's ImageEditor."""
    
    def __init__(self):
        self.current_video = None
        self.reference_frame = 0
        self.reference_frame_image = None
        self.debug_dir = Path("temp")
        self.debug_dir.mkdir(exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging for debug information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.debug_dir / f"simple_mask_debug_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 60)
        self.logger.info("Simple Mask Tool Debug Session Started")
        self.logger.info(f"Debug directory: {self.debug_dir}")
        self.logger.info("=" * 60)
    
    def load_video(self, video_path: str) -> np.ndarray:
        """Load video file and return as numpy array."""
        try:
            frames = iio.imread(video_path, plugin="FFMPEG")
            self.logger.info(f"Loaded video: {frames.shape}")
            return frames
        except Exception as e:
            self.logger.error(f"Error loading video: {e}")
            raise
    
    def get_reference_frame_image(self, video: np.ndarray, frame_idx: int) -> np.ndarray:
        """Extract a specific frame from the video."""
        frame_idx = max(0, min(frame_idx, video.shape[0] - 1))
        frame = video[frame_idx]
        self.logger.info(f"Extracted reference frame {frame_idx}: {frame.shape}")
        return frame
    
    def extract_mask_from_edited_image(self, edited_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """Extract a black and white mask from the edited image by comparing with original."""
        if edited_image is None:
            self.logger.warning("No edited image provided")
            return None
        
        self.logger.info(f"Extracting mask from edited image: {edited_image.shape}")
        self.logger.info(f"Original image shape: {original_image.shape}")
        
        # Convert to grayscale for comparison
        if len(edited_image.shape) == 3:
            edited_gray = cv2.cvtColor(edited_image, cv2.COLOR_RGB2GRAY)
        else:
            edited_gray = edited_image
            
        if len(original_image.shape) == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original_image
        
        # Find differences between original and edited image
        diff = cv2.absdiff(original_gray, edited_gray)
        
        # Threshold to create binary mask
        # Areas that were painted will have differences > threshold
        _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        
        self.logger.info(f"Mask created: {mask.shape}, unique values: {np.unique(mask)}")
        
        return mask
    
    def save_mask(self, mask: np.ndarray) -> str:
        """Save the mask as PNG and return the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_file = self.debug_dir / f"drawn_mask_{timestamp}.png"
        
        cv2.imwrite(str(mask_file), mask)
        self.logger.info(f"Mask saved to: {mask_file}")
        
        return str(mask_file)


def create_simple_mask_interface():
    """Create the Gradio interface for simple mask drawing."""
    
    tool = SimpleMaskTool()
    
    def load_video_for_reference(video_file):
        """Load video and display it for reference frame selection."""
        if video_file is None:
            return None, "Please upload a video file first.", None
        
        try:
            video = tool.load_video(video_file)
            tool.current_video = video
            
            # Create a preview video for reference frame selection
            temp_video_path = tool.debug_dir / f"reference_video_{datetime.now().strftime('%H%M%S')}.mp4"
            
            iio.imwrite(
                temp_video_path, 
                video, 
                plugin="FFMPEG", 
                fps=24,
                codec='libx264',
                quality=8
            )
            
            return str(temp_video_path), f"Video loaded: {video.shape[0]} frames available", None
            
        except Exception as e:
            return None, f"Error loading video: {str(e)}", None
    
    def select_reference_frame(reference_video, video_data):
        """Select the current frame as reference frame."""
        if reference_video is None or tool.current_video is None:
            return "Please load a video first.", None
        
        try:
            # For now, use middle frame as default (in real implementation, get from video player time)
            current_time = tool.current_video.shape[0] / 2 / 24  # Middle frame
            
            # Convert time to frame index
            fps = 24
            frame_idx = int(current_time * fps)
            frame_idx = max(0, min(frame_idx, tool.current_video.shape[0] - 1))
            
            tool.reference_frame = frame_idx
            tool.logger.info(f"Selected reference frame: {frame_idx}")
            
            # Get the reference frame
            ref_frame = tool.get_reference_frame_image(tool.current_video, frame_idx)
            tool.reference_frame_image = ref_frame
            
            # Convert to PIL Image for Gradio
            ref_frame_pil = Image.fromarray(ref_frame)
            
            return f"Reference frame set to: Frame {frame_idx}", ref_frame_pil
            
        except Exception as e:
            tool.logger.error(f"Error in select_reference_frame: {str(e)}")
            return f"Error selecting reference frame: {str(e)}", None
    
    def process_edited_image(edited_image):
        """Process the edited image and create a mask."""
        if edited_image is None:
            return "No edited image provided", None
        
        if tool.reference_frame_image is None:
            return "Please select a reference frame first", None
        
        try:
            tool.logger.info(f"Received edited image type: {type(edited_image)}")
            
            # Handle Gradio ImageEditor format (dict with 'background' and 'layers')
            if isinstance(edited_image, dict):
                tool.logger.info(f"ImageEditor dict keys: {edited_image.keys()}")
                
                # Try to get the composite image or background
                if 'composite' in edited_image:
                    edited_pil = edited_image['composite']
                elif 'background' in edited_image:
                    edited_pil = edited_image['background']
                else:
                    # Try to get the first available image
                    for key, value in edited_image.items():
                        if isinstance(value, Image.Image):
                            edited_pil = value
                            break
                    else:
                        return f"Could not find image in ImageEditor data. Keys: {list(edited_image.keys())}", None
            elif isinstance(edited_image, Image.Image):
                edited_pil = edited_image
            else:
                return f"Unexpected image format: {type(edited_image)}", None
            
            # Convert PIL to numpy
            edited_array = np.array(edited_pil)
            tool.logger.info(f"Converted to array: {edited_array.shape}")
            
            # Extract mask
            mask = tool.extract_mask_from_edited_image(edited_array, tool.reference_frame_image)
            
            if mask is None:
                return "Failed to extract mask", None
            
            # Save mask
            mask_file = tool.save_mask(mask)
            
            # Convert mask to PIL for display
            mask_pil = Image.fromarray(mask)
            
            return f"Mask created and saved to: {mask_file}", mask_pil
            
        except Exception as e:
            tool.logger.error(f"Error processing edited image: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error processing image: {str(e)}", None
    
    # Create Gradio interface
    with gr.Blocks(title="Simple Mask Drawing Tool") as interface:
        gr.Markdown("# Simple Mask Drawing Tool")
        gr.Markdown("Draw masks on video frames using Gradio's built-in painting tools.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Video upload and reference frame selection
                reference_video = gr.Video(
                    label="Upload Video & Select Reference Frame",
                    interactive=True,
                    sources=["upload"],
                    height=300
                )
                
                reference_frame_info = gr.Textbox(
                    label="Reference Frame Info",
                    value="No reference frame selected",
                    interactive=False,
                    lines=2
                )
                
                select_reference_btn = gr.Button("Select Current Frame as Reference", variant="secondary")
            
            with gr.Column(scale=1):
                result_text = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False
                )
        
        # Drawing interface
        gr.Markdown("## Drawing Interface")
        gr.Markdown("**Instructions:** 1) Select reference frame above, 2) Adjust brush size, 3) Use brush tools below to draw your mask, 4) Click 'Create Mask'")
        
        # Brush size control
        brush_size_slider = gr.Slider(
            minimum=5,
            maximum=100,
            value=20,
            step=5,
            label="Brush Size",
            info="Adjust the brush size for drawing"
        )
        
        image_editor = gr.ImageEditor(
            label="Draw Your Mask Here",
            type="pil",
            height=500,
            brush=gr.Brush(default_size=20, colors=["#FFFFFF", "#000000"])
        )
        
        with gr.Row():
            create_mask_btn = gr.Button("Create Mask from Drawing", variant="primary")
        
        # Mask output
        with gr.Row():
            with gr.Column(scale=1):
                mask_display = gr.Image(
                    label="Generated Mask (White = Selected, Black = Ignored)",
                    interactive=False,
                    height=300
                )
            
            with gr.Column(scale=1):
                mask_result = gr.Textbox(
                    label="Mask Creation Result",
                    lines=5,
                    interactive=False
                )
        
        # Event handlers
        reference_video.upload(
            fn=load_video_for_reference,
            inputs=[reference_video],
            outputs=[reference_video, reference_frame_info, image_editor]
        )
        
        select_reference_btn.click(
            fn=select_reference_frame,
            inputs=[reference_video, reference_video],
            outputs=[reference_frame_info, image_editor]
        )
        
        # Update brush size when slider changes
        def update_brush_size(size):
            # Return updated ImageEditor with new brush size
            return gr.ImageEditor(
                label="Draw Your Mask Here",
                type="pil",
                height=500,
                brush=gr.Brush(default_size=int(size), colors=["#FFFFFF", "#000000"])
            )
        
        brush_size_slider.change(
            fn=update_brush_size,
            inputs=[brush_size_slider],
            outputs=[image_editor]
        )
        
        create_mask_btn.click(
            fn=process_edited_image,
            inputs=[image_editor],
            outputs=[mask_result, mask_display]
        )
    
    return interface


if __name__ == "__main__":
    interface = create_simple_mask_interface()
    interface.launch(
        share=False, 
        server_name="127.0.0.1", 
        server_port=7862,  # Different port
        show_error=True
    )
