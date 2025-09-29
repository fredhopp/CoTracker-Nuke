#!/usr/bin/env python3
"""
Mask handling utilities
=======================

Handles mask creation, processing, and validation for point filtering.
"""

import numpy as np
from PIL import Image
from typing import Optional, Union, List, Any
import logging
from pathlib import Path
from datetime import datetime


class MaskHandler:
    """Handles mask creation, processing, and validation."""
    
    def __init__(self, debug_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize mask handler.
        
        Args:
            debug_dir: Directory for saving mask files
            logger: Logger instance (optional)
        """
        self.debug_dir = debug_dir or Path("temp")
        self.debug_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.current_mask = None
    
    def extract_mask_from_edited_image(self, original: np.ndarray, edited: np.ndarray) -> np.ndarray:
        """
        Extract mask from edited image by comparing with original.
        
        Args:
            original: Original reference frame
            edited: Edited image with mask drawn
        
        Returns:
            Binary mask array
        """
        self.logger.debug("Extracting mask from edited image comparison")
        
        # Convert to grayscale for comparison
        if len(original.shape) == 3:
            orig_gray = np.mean(original, axis=2)
        else:
            orig_gray = original
            
        if len(edited.shape) == 3:
            edit_gray = np.mean(edited, axis=2)
        else:
            edit_gray = edited
        
        # Find differences
        diff = np.abs(edit_gray - orig_gray)
        
        # Threshold to create binary mask
        threshold = 10  # Adjust as needed
        mask = (diff > threshold).astype(np.uint8) * 255
        
        self.logger.debug(f"Extracted mask shape: {mask.shape}, unique values: {np.unique(mask)}")
        
        return mask
    
    def extract_mask_from_layers(self, layers_data: Any) -> np.ndarray:
        """
        Extract mask from Gradio ImageEditor layers data.
        
        Args:
            layers_data: Gradio ImageEditor layers data
        
        Returns:
            Binary mask array
        """
        self.logger.debug(f"Processing layers data: type={type(layers_data)}")
        
        if isinstance(layers_data, list) and len(layers_data) > 0:
            self.logger.debug(f"layers_data is a list with {len(layers_data)} items")
            
            # Process each layer
            combined_mask = None
            
            for i, layer in enumerate(layers_data):
                self.logger.debug(f"Processing layer {i}: {type(layer)}")
                
                if isinstance(layer, Image.Image):
                    # Convert PIL image to numpy
                    layer_array = np.array(layer)
                    self.logger.debug(f"Layer {i} size: {layer.size}")
                    self.logger.debug(f"layers_array shape: {layer_array.shape}, dtype: {layer_array.dtype}")
                    
                    # Check unique values for debugging
                    unique_vals = np.unique(layer_array.flatten())
                    self.logger.debug(f"layers_array unique values: {unique_vals}")
                    
                    if len(layer_array.shape) == 3:
                        if layer_array.shape[2] == 4:  # RGBA
                            self.logger.debug("Processing RGBA layers data using alpha channel")
                            # Use alpha channel as mask
                            alpha_channel = layer_array[:, :, 3]
                            self.logger.debug(f"Alpha channel shape: {alpha_channel.shape}, unique values: {np.unique(alpha_channel)}")
                            
                            # Convert alpha to binary mask (non-zero alpha = white mask)
                            mask = (alpha_channel > 0).astype(np.uint8) * 255
                        elif layer_array.shape[2] == 3:  # RGB
                            self.logger.debug("Processing RGB layers data")
                            # Convert to grayscale and threshold
                            gray = np.mean(layer_array, axis=2)
                            mask = (gray > 128).astype(np.uint8) * 255
                        else:
                            self.logger.warning(f"Unexpected layer shape: {layer_array.shape}")
                            continue
                    elif len(layer_array.shape) == 2:  # Grayscale
                        self.logger.debug("Processing grayscale layers data")
                        mask = (layer_array > 128).astype(np.uint8) * 255
                    else:
                        self.logger.warning(f"Unexpected layer shape: {layer_array.shape}")
                        continue
                    
                    # Combine with existing mask
                    if combined_mask is None:
                        combined_mask = mask
                    else:
                        combined_mask = np.logical_or(combined_mask > 0, mask > 0).astype(np.uint8) * 255
            
            if combined_mask is not None:
                self.logger.debug(f"Final mask shape: {combined_mask.shape}, unique values: {np.unique(combined_mask)}")
                return combined_mask
        
        # Fallback: create empty mask
        self.logger.warning("Could not extract mask from layers, creating empty mask")
        return np.zeros((480, 640), dtype=np.uint8)  # Default size
    
    def process_mask_from_editor(self, edited_image: Union[dict, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Process mask from Gradio ImageEditor output.
        
        Args:
            edited_image: Output from Gradio ImageEditor
        
        Returns:
            Binary mask array
        """
        self.logger.debug(f"Processing mask from editor, image type: {type(edited_image)}")
        
        if isinstance(edited_image, dict):
            self.logger.debug(f"ImageEditor dict keys: {edited_image.keys()}")
            
            # Check for layers data first (clean painted areas)
            if 'layers' in edited_image and edited_image['layers'] is not None:
                self.logger.debug("Using layers data for clean mask extraction")
                return self.extract_mask_from_layers(edited_image['layers'])
            
            # Fallback to composite image
            elif 'composite' in edited_image and edited_image['composite'] is not None:
                self.logger.debug("Using composite image for mask extraction")
                composite = edited_image['composite']
                if isinstance(composite, Image.Image):
                    composite_array = np.array(composite)
                    if len(composite_array.shape) == 3 and composite_array.shape[2] == 4:
                        # Use alpha channel
                        return (composite_array[:, :, 3] > 0).astype(np.uint8) * 255
                    else:
                        # Convert to grayscale and threshold
                        gray = np.mean(composite_array, axis=2) if len(composite_array.shape) == 3 else composite_array
                        return (gray > 128).astype(np.uint8) * 255
        
        elif isinstance(edited_image, Image.Image):
            self.logger.debug("Processing PIL Image directly")
            image_array = np.array(edited_image)
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                return (image_array[:, :, 3] > 0).astype(np.uint8) * 255
            else:
                gray = np.mean(image_array, axis=2) if len(image_array.shape) == 3 else image_array
                return (gray > 128).astype(np.uint8) * 255
        
        elif isinstance(edited_image, np.ndarray):
            self.logger.debug("Processing numpy array directly")
            if len(edited_image.shape) == 3 and edited_image.shape[2] == 4:
                return (edited_image[:, :, 3] > 0).astype(np.uint8) * 255
            else:
                gray = np.mean(edited_image, axis=2) if len(edited_image.shape) == 3 else edited_image
                return (gray > 128).astype(np.uint8) * 255
        
        # Fallback
        self.logger.warning("Could not process mask from editor input")
        return np.zeros((480, 640), dtype=np.uint8)
    
    def save_mask(self, mask: np.ndarray) -> str:
        """
        Save mask to file.
        
        Args:
            mask: Binary mask array
        
        Returns:
            Path to saved mask file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_path = self.debug_dir / f"drawn_mask_{timestamp}.png"
        
        # Save as PNG
        mask_image = Image.fromarray(mask.astype(np.uint8))
        mask_image.save(mask_path)
        
        # Store current mask
        self.current_mask = mask
        
        self.logger.debug(f"Mask saved to: {mask_path}")
        return str(mask_path)
    
    def is_mask_empty(self, mask: np.ndarray) -> bool:
        """
        Check if mask is effectively empty.
        
        Args:
            mask: Binary mask array
        
        Returns:
            True if mask is empty or has insufficient coverage
        """
        if mask is None:
            return True
        
        white_pixels = np.sum(mask == 255)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage = (white_pixels / total_pixels) * 100
        
        # Consider mask empty if less than 0.1% coverage or less than 10 pixels
        min_pixels_threshold = 10
        percentage_threshold = 0.1
        
        is_empty = white_pixels < min_pixels_threshold and coverage < percentage_threshold
        
        self.logger.debug(f"is_mask_empty: {white_pixels}/{total_pixels} white pixels = {coverage:.2f}% coverage")
        self.logger.debug(f"is_mask_empty result: {is_empty} (min_pixels: {white_pixels >= min_pixels_threshold}, percentage: {coverage >= percentage_threshold})")
        
        return is_empty
    
    def get_mask_stats(self, mask: np.ndarray) -> dict:
        """
        Get statistics about the mask.
        
        Args:
            mask: Binary mask array
        
        Returns:
            Dictionary with mask statistics
        """
        if mask is None:
            return {'error': 'No mask provided'}
        
        white_pixels = np.sum(mask == 255)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage = (white_pixels / total_pixels) * 100
        
        return {
            'shape': mask.shape,
            'white_pixels': int(white_pixels),
            'total_pixels': int(total_pixels),
            'coverage_percent': float(coverage),
            'is_empty': self.is_mask_empty(mask),
            'unique_values': np.unique(mask).tolist()
        }
