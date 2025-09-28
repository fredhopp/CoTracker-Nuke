# Simple Mask Drawing Tool

A standalone tool for drawing simple masks on video frames using Gradio's built-in ImageEditor.

## Features

- ✅ **Video Loading**: Upload and preview videos
- ✅ **Reference Frame Selection**: Pick specific frames for mask drawing
- ✅ **Brush Drawing**: Draw with Gradio's built-in brush tools
- ✅ **Mask Extraction**: Automatic black/white mask generation
- ✅ **PNG Export**: Save masks as PNG files
- ✅ **Debug Logging**: Comprehensive logging in `temp/`

## Usage

### Quick Start

```bash
# From the main CoTracker directory
cd draw
python simple_demo.py
```

This will launch the interface at `http://127.0.0.1:7862`

### Workflow

1. **Upload Video**: Use the video upload component
2. **Select Reference Frame**: Click "Select Current Frame as Reference" 
3. **Draw Mask**: Use the brush tools in the ImageEditor to paint your mask
4. **Create Mask**: Click "Create Mask from Drawing" to generate the black/white mask
5. **Save**: Mask is automatically saved as PNG in the `temp/` folder

### Output Files

All outputs are saved to `temp/`:
- `drawn_mask_TIMESTAMP.png` - Black/white mask image  
- `simple_mask_debug_TIMESTAMP.log` - Debug log
- `reference_video_TIMESTAMP.mp4` - Temporary reference video

## Technical Details

### Gradio ImageEditor Integration

The tool uses Gradio's built-in ImageEditor:
- Native brush tools with adjustable size
- Black and white brush colors
- Direct image editing capabilities
- PIL Image format support

### Mask Generation

- **White pixels**: Areas you painted (selected region)  
- **Black pixels**: Unpainted areas (ignored region)
- **Resolution**: Same as original video frame
- **Format**: PNG for compatibility
- **Method**: Difference detection between original and edited images

### Future CoTracker Integration

This tool is designed for easy integration into the main CoTracker project:
- Compatible video loading interface
- Same reference frame selection logic
- Mask output format suitable for point filtering
- Modular design for component reuse

## Development Notes

### Current Implementation

- ✅ Basic ellipse drawing
- ✅ 4-point control system
- ✅ SVG/PNG export
- ⏳ Bezier handle support (planned)
- ⏳ Multiple shape support (planned)

### Integration Plan

1. Test standalone functionality
2. Refine user interface
3. Add bezier handle support
4. Integrate into main CoTracker app
5. Add mask-based point filtering

## Dependencies

Uses the same dependencies as the main CoTracker project:
- `gradio` - Web interface
- `numpy` - Array operations
- `opencv-python` - Image processing
- `imageio` - Video I/O
- `pillow` - Image manipulation

## Browser Compatibility

Requires a modern browser with:
- HTML5 Canvas support
- JavaScript ES6 support
- Paper.js compatibility
