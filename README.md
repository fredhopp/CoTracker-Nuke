# CoTracker Nuke Integration App

> **Development Note**: This project was developed with AI assistance (Claude) under human supervision, combining automated code generation with human oversight for architecture decisions, testing, and quality assurance.

A powerful application that leverages Facebook Research's CoTracker for point tracking in videos and exports the tracking data in a format compatible with Foundry Nuke's CornerPin2D node.

## Features

- 🎯 **Automatic Point Tracking**: Uses CoTracker3/CoTracker2 to track multiple points across video frames
- 🎨 **Interactive GUI**: User-friendly Gradio interface for video upload and processing
- 🎭 **Zone masking**: User can mask out a specific are on a reference frame to have trackers restricted to that zone
- 🖼️ **Visual Preview**: Shows tracked points
- 📤 **Nuke Export**: Generates complete Nuke scripts (.nk files)

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

1. **Start the Gradio interface:**
   ```bash
   python cotracker_nuke_app.py
   ```

2. **Open your browser** and navigate to `http://127.0.0.1:7860`

### Using the Interface

1. **Upload Video**: Click "Upload Video" and select your video file (.mp4, .mov, .avi, .mkv)

2. **Set Grid Size**: Adjust the slider to control how many points CoTracker will track (5-50 points)

3. **Process Video**: Click "Process Video" to run CoTracker and automatically select corner points

4. **Preview Results**: Double check that the mask and reference have been correctly considered.

5. **Export to Nuke**: 
   - Enter a filename for your Nuke script
   - Click "Export to Nuke" to generate the .nk file

### Importing into Nuke

1. **Open Nuke** and create a new composition

2. **Import the script**: 
   - File → Import → Select your generated .nk file
   - Or drag and drop the .nk file into Nuke

## How It Works

### Point Selection Algorithm

The app uses a sophisticated algorithm to select the 4 best corner points:

1. **Visibility Filtering**: Only considers points visible in >70% of frames
2. **Stability Analysis**: Calculates motion variance to prefer stable tracks  
3. **Spatial Distribution**: Uses convex hull and distance maximization to ensure good corner spread
4. **Combined Scoring**: Weights visibility (40%) and stability (60%) for final selection

### Nuke Export Format

The generated Nuke script includes:
- **Read node**: Placeholder for your source video
- **Tracker4 node**: Contains all tracking data with keyframes

## Technical Details

### Dependencies

- **PyTorch**: For running CoTracker models
- **CoTracker**: Facebook Research's point tracking model
- **OpenCV**: Video processing and visualization
- **Gradio**: Web-based user interface
- **NumPy/SciPy**: Numerical computations and spatial algorithms

### Supported Formats

- **Input Videos**: MP4, MOV, AVI, MKV
- **Output**: Nuke script files (.nk)
- **CoTracker Models**: CoTracker3 (preferred) or CoTracker2 (fallback)

### Performance

- **GPU Acceleration**: Automatically uses CUDA if available (tested with RTX PRO 6000 Blackwell)
- **CUDA 12.8 Support**: Compatible with latest Blackwell architecture GPUs
- **Speed**: ~10x faster processing on GPU vs CPU (3 seconds vs 30+ seconds)
- **Memory Efficient**: Processes videos in chunks for large files
- **Real-time Preview**: Fast preview generation for immediate feedback


### Common Issues

1. **CUDA Out of Memory**: Reduce grid size or use CPU mode
2. **Video Loading Errors**: Ensure video format is supported and file isn't corrupted
3. **Model Loading Issues**: Check internet connection for torch.hub downloads

### Performance Tips

- Use GPU when available for faster processing
- Start with smaller grid sizes (10-15) for testing
- Ensure good lighting and contrast in source videos for better tracking

## License

MIT License - See LICENSE file for details


## Acknowledgments

- Facebook Research for CoTracker - [CoTracker3 Project Page](https://cotracker3.github.io/)
- Foundry for Nuke integration specifications
- The open-source community for supporting libraries
