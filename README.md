# CoTracker Nuke Integration App

> **Development Note**: This project was developed with AI assistance (Claude) under human supervision, combining automated code generation with human oversight for architecture decisions, testing, and quality assurance.

A powerful application that leverages Facebook Research's CoTracker for point tracking in videos and exports the tracking data in a format compatible with Foundry Nuke's CornerPin2D node.

## Demo Video

🎥 **See CoTracker-Nuke in action:**

*Complete workflow demonstration: video upload, masking, tracking and Nuke export*

> **📺 [Watch the full demo on YouTube](https://youtu.be/zWHQrzYtVKw)** *(Ctrl + Click to open in new tab 😿)*

## Features

- 🎯 **Automatic Point Tracking**: Uses CoTracker3/CoTracker2 to track multiple points across video frames
- 🎨 **Interactive GUI**: User-friendly Gradio interface with modern, intuitive design
- 🎭 **Zone Masking**: Interactive mask drawing to restrict tracking to specific areas
- 🖼️ **Visual Preview**: Real-time preview of tracked points with frame-by-frame navigation
- 📤 **Smart Nuke Export**: Generates complete Nuke Tracker4 nodes with proper coordinate transformation
- 🗺️ **Advanced STMap Generation**: Complete animated STMap sequence with RGBA output and intelligent interpolation
- 🎭 **Animated Mask Integration**: Mask-aware STMap generation with embedded animated mask in alpha channel
- 🔍 **Delaunay Hull Detection**: Accurate inside/outside hull classification for optimal coordinate mapping
- ⚡ **Performance Optimization**: Smart bounding box processing for 10-20x speed improvements
- 📊 **Real-Time Progress Tracking**: 3-stage progress system with actual timing display for STMap generation
- 📋 **Clipboard Integration**: One-click copy of .nk file paths with Windows 11 support
- 🗂️ **File Management**: Built-in file browser and automatic output organization with absolute paths
- 🏗️ **Modular Architecture**: Clean, maintainable codebase with separated concerns
- 🎨 **Clean UI Design**: Organized interface with grouped browse buttons and dynamic path variables
- 🔧 **Dynamic Path Variables**: Support for %refFrame% in output paths for organized file management
- 🧹 **Project Cleanup**: Clean project structure with removed test/debug files
- ⚡ **ProcessPoolExecutor**: True CPU parallelism for STMap generation with optimal resource utilization
- 🛡️ **Robust Error Handling**: Division by zero protection and comprehensive error recovery

## Installation

1. **Clone and setup the project:**
   ```bash
   git clone <your-repo>
   cd Cotracker
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux  
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the modular application:**
   ```bash
   python cotracker_nuke_app.py
   ```

2. **Open your browser** and navigate to `http://127.0.0.1:7860`

### Using the Interface

1. **📹 Upload Video**: Click "📁 Upload Video File" and select your video file (.mp4, .mov, .avi, .mkv)

2. **🎬 Set Image Sequence Start Frame**: Configure the frame number where your image sequence starts in Nuke (default: 1001)

3. **🎯 Set Reference Frame**: Use the frame slider and preview to choose your tracking reference frame

4. **🎨 Optional Mask Drawing**: Draw on the reference frame to restrict tracking to specific areas (white areas = tracked)

5. **🚀 Process Video**: 
   - Adjust grid size (5-100 points) to control tracking density
   - Click "🚀 Process Video" to run CoTracker
   - Preview the tracking results in the generated video

6. **📤 Generate Tracker Node**: 
   - Use the default absolute path or click "📂 Browse" to choose a custom location
   - Click "📤 Generate Tracker Node as .nk" to create the Nuke file
   - Click "📋 Copy .nk Path to Clipboard" to copy the file path for easy import

7. **🗺️ Generate STMap Sequence** (Optional):
   - Choose interpolation method (linear or cubic) and bit depth (16-bit or 32-bit)
   - Set frame range for STMap export (defaults to full sequence)
   - Use dynamic path variables like %refFrame% for organized output
   - Click "🗺️ Generate STMap Sequence" to create animated RGBA EXR files
   - View 3-stage progress system: analyzing → ETA estimation → completion with actual timing
   - Click "📋 Copy STMap Directory Path" to copy the output folder path

### Importing into Nuke

1. **Open Nuke** and create a new composition

2. **Import the Tracker4 node**: 
   - Use the copied path: File → Open → Paste (`Ctrl+V`) the file path
   - Or drag and drop the .nk file into Nuke
   - The Tracker4 node will load with all tracking data and proper coordinate transformation

3. **Using STMap sequences** (if generated):
   - Import the EXR sequence using Read node
   - Connect to STMap node for geometric transformations
   - STMap files contain RGBA channels: Red=X coordinates, Green=Y coordinates, Blue=black, Alpha=animated mask
   - Reference frame shows perfect gradient, other frames show coordinate mapping with animated mask

## How It Works

### Point Selection Algorithm

The app uses a sophisticated algorithm to select the 4 best corner points:

1. **Visibility Filtering**: Only considers points visible in >70% of frames
2. **Stability Analysis**: Calculates motion variance to prefer stable tracks  
3. **Spatial Distribution**: Uses convex hull and distance maximization to ensure good corner spread
4. **Combined Scoring**: Weights visibility (40%) and stability (60%) for final selection

### Nuke Export Format

The generated Nuke script includes:
- **Tracker4 node**: Complete tracking data with proper coordinate transformation
- **Automatic coordinate conversion**: From CoTracker's top-left origin to Nuke's bottom-left origin
- **Frame offset support**: Matches your image sequence start frame
- **Reference frame preservation**: Maintains your chosen reference frame
- **Track visibility data**: Only includes confident tracking points

### STMap Generation

The advanced STMap export system provides:
- **RGBA EXR sequences**: Red=X coordinates, Green=Y coordinates, Blue=black, Alpha=animated mask
- **Delaunay hull detection**: Accurate inside/outside hull classification for optimal processing
- **Unified processing logic**: Mask warping and STMap generation use identical algorithms
- **Segment-based fringe algorithm**: Geometric projection for pixels outside tracker hull
- **Smart bounding box optimization**: 10-20x performance improvement by processing only relevant pixels
- **Perfect reference frame**: Identity gradient (Red=0-1 horizontal, Green=0-1 vertical)
- **Coordinate mapping**: Each pixel shows where to sample from in the reference frame
- **Interpolation options**: Linear or cubic interpolation between tracking points
- **Bit depth support**: 16-bit or 32-bit float precision
- **Real-time progress tracking**: 3-stage progress system with actual timing display
- **ProcessPoolExecutor**: True CPU parallelism for optimal performance
- **Robust error handling**: Division by zero protection and comprehensive error recovery

## Technical Details

### Dependencies

- **PyTorch**: For running CoTracker models
- **CoTracker**: Facebook Research's point tracking model  
- **OpenCV**: Video processing and visualization
- **Gradio**: Modern web-based user interface
- **NumPy/SciPy**: Numerical computations and spatial algorithms
- **pyperclip**: Cross-platform clipboard functionality
- **imageio**: Video loading and processing
- **OpenEXR**: High-precision EXR file format support for STMap generation

### Supported Formats

- **Input Videos**: MP4, MOV, AVI, MKV
- **Output**: Nuke script files (.nk) and EXR sequences for STMap
- **CoTracker Models**: CoTracker3 (preferred) or CoTracker2 (fallback)
- **STMap Output**: 16-bit or 32-bit float EXR files with RGB channels

### Performance

- **GPU Acceleration**: Automatically uses CUDA if available (tested with RTX PRO 6000 Blackwell)
- **CUDA 12.8 Support**: Compatible with latest Blackwell architecture GPUs
- **Speed**: ~10x faster processing on GPU vs CPU (3 seconds vs 30+ seconds)
- **Memory Efficient**: Processes videos in chunks for large files
- **Real-time Preview**: Fast preview generation with tracking visualization
- **Scalable**: Handles anywhere from 10 to 1000+ tracking points efficiently

### Architecture

- **Modular Design**: Clean separation between core logic, UI, and export functionality
- **Error Handling**: Comprehensive error handling and user feedback
- **Logging**: Detailed logging for debugging and monitoring
- **Cross-Platform**: Works on Windows, macOS, and Linux

### Common Issues

1. **CUDA Out of Memory**: Reduce video resolution, use masking to limit tracking area, or use CPU mode
2. **Video Loading Errors**: Ensure video format is supported and file isn't corrupted
3. **Model Loading Issues**: Check internet connection for torch.hub downloads
4. **Coordinate Misalignment**: The app automatically handles coordinate system conversion
5. **Clipboard Issues**: Multiple fallback methods ensure clipboard functionality works across platforms

### Performance Tips

- Use GPU when available for significantly faster processing
- Start with smaller grid sizes (10-20) for testing, scale up as needed
- Use masking to focus tracking on relevant areas and improve performance
- Ensure good lighting and contrast in source videos for better tracking accuracy
- **VRAM Usage**: Memory consumption depends on total pixels to be tracked (video resolution × tracking area), not the number of trackers
- For high-resolution videos or large tracking areas, consider using masks to limit the processing region

## License

CoTracker-Nuke is wrapper application and is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

**Important Note**: CoTracker-Nuke uses CoTracker3/CoTracker2 by Facebook Research, which is licensed for **non-commercial use only**. Please review CoTracker's license terms at: [CoTracker3 LICENSE.md](https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md)

## Acknowledgments

- Facebook Research for CoTracker - [CoTracker3 Project Page](https://cotracker3.github.io/)
- Foundry for Nuke integration specifications
- The open-source community for supporting libraries
