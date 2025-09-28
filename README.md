# CoTracker Nuke Integration App

A powerful application that leverages Facebook Research's CoTracker for point tracking in videos and exports the tracking data in a format compatible with Foundry Nuke's CornerPin2D node.

## Features

- 🎯 **Automatic Point Tracking**: Uses CoTracker3/CoTracker2 to track multiple points across video frames
- 🔍 **Smart Corner Selection**: Automatically selects 4 optimal points for corner pin tracking based on:
  - Point visibility throughout the sequence
  - Spatial distribution (forming a good quadrilateral)
  - Tracking stability (low motion variance)
- 🎨 **Interactive GUI**: User-friendly Gradio interface for video upload and processing
- 📤 **Nuke Export**: Generates complete Nuke scripts (.nk files) with CornerPin2D nodes
- 🖼️ **Visual Preview**: Shows tracked points and selected corners on the first frame

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

4. **Preview Results**: View the tracking preview showing:
   - Blue dots: All tracked points
   - Red dots: Selected corner points (numbered)

5. **Export to Nuke**: 
   - Enter a filename for your Nuke script
   - Click "Export to Nuke" to generate the .nk file

### Importing into Nuke

1. **Open Nuke** and create a new composition

2. **Import the script**: 
   - File → Import → Select your generated .nk file
   - Or drag and drop the .nk file into Nuke

3. **Update the Read node**: 
   - Select the Read node in the generated script
   - Update the file path to point to your original video

4. **Connect your replacement footage** to the CornerPin2D node input

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
- **CornerPin2D node**: Pre-configured with the 4 selected corner tracks

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

## System Rules

⚠️ **Important Development Guidelines:**
- Never push to git before confirming with the user
- Always use the `.venv` virtual environment for dependencies
- Install packages only within the virtual environment, not system-wide

## Troubleshooting

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

## Contributing

1. Follow the system rules in `SYSTEM_RULES.md`
2. Test thoroughly with various video formats
3. Document any new features or changes

## Acknowledgments

- Facebook Research for CoTracker
- Foundry for Nuke integration specifications
- The open-source community for supporting libraries
