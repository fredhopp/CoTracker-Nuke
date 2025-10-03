# CoTracker Nuke App - TODO List

## Status Overview
- **Current Status**: ✅ Production ready with optimized STMap generation and unified processing
- **Recent Major Features**: ✅ Delaunay hull detection + unified mask/STMap processing + performance optimization
- **Architecture**: ✅ Fully modular structure with clean separation of concerns
- **Current Priority**: Feature complete - ready for production use with advanced STMap and optimized processing

## Current Tasks

### 🚀 Future Enhancements (Optional)
- [ ] **Export Formats** - Add support for other compositing software (Flame, etc.)
- [ ] **Desktop App Conversion** - Convert Gradio web interface to PySide6 desktop app with native UI
- [ ] **Cross-Platform Packaging** - Set up PyInstaller with spec file for Windows, macOS, and Linux builds
- [ ] **Installer Creation** - Create installers for Windows (Inno Setup), macOS, and Linux distributions
- [ ] **Rocky Linux Build** - Set up GitHub Actions workflow to build for Rocky Linux using Docker container
- [ ] **Docker Build Environment** - Create Dockerfile.rocky with Rocky Linux 9 base image and dependencies
- [ ] **Build Testing** - Test Rocky Linux build process and verify executable works on target system
- [x] **Grid Point Spacing** - Check if the grid of points is taking the ratio of image as a parameter to have checkerboard like spacing
- [ ] **Frame Rate Handling** - Check if the video is working at a set 24fps or properly considering 23.976 fps
- [ ] **STMap Advanced Interpolation** - Implement memory-efficient bicubic interpolation for STMap generation (Delaunay removed due to memory issues)
- [x] **STMap Performance Optimization** - Apply vectorized NaN handling to STMap generation for better performance
- [x] **Enhanced STMap Block Offset Technique** - Implemented proper segment-based algorithm for fringe pixels with Delaunay hull detection
- [x] **Consolidate EXR Output** - Merged enhanced STMap functionality into regular STMap export:
  - Regular STMap sequence outputs RGBA EXR with embedded animated mask in alpha channel
  - Removed separate "Enhanced STMap" section from UI (regular STMap handles everything)
  - Removed "Animated Mask Export" section from UI (mask is embedded in STMap alpha channel)
  - Single unified export provides both STMap coordinates and animated mask in one EXR sequence
- [x] **STMap Frame Range Integration** - Start and end frame need to take the Image Sequence Start Frame into account. The parameter should default to first and last frame
- [x] **STMap Output Path Handling** - The output path for EXR needs to be absolute in the status and clipboard
- [x] **STMap Filename Convention** - The EXR filename needs to conform to our convention: the user needs to be able to pick a path the same way he does for the .nk file. It should default to CoTracker_date_time.%04d.exr ("." not "_" as a separator)
- [x] **STMap Black Output Fix** - The exported EXR files are black (investigate coordinate mapping issue)
- [x] **STMap Reference Frame Fix** - STMap now uses the actual user-selected reference frame instead of hardcoded frame 0
- [x] **STMap UI Layout Optimization** - Condensed UI layout to use less vertical space
- [x] **STMap Output Folder Structure** - Updated to use nested folder structure: outputs/CoTracker_date_time_stmap/


### 🔧 Maintenance Tasks
- [ ] **Testing** - Comprehensive testing across different video formats and resolutions
- [ ] **Documentation** - Add video tutorials and advanced usage examples
- [ ] **Gradio Update** - Monitor Gradio GitHub issue #7529 for ImageEditor drawing interactivity fix and update when resolved


## Completed Tasks ✅

### Latest Updates (October 2, 2025)
- [x] **Delaunay Hull Detection** - Replaced unreliable NaN-based hull detection with accurate Delaunay triangulation
- [x] **Unified Processing Logic** - Mask warping and STMap processing now use identical hull detection and processing methods
- [x] **Segment-Based Fringe Algorithm** - Renamed and optimized fringe coordinate calculation with proper geometric projection
- [x] **Smart Bounding Box Optimization** - Only process pixels inside calculated bounds for 10-20x performance improvement
- [x] **UI Consolidation** - Removed duplicate STMap sections, unified into single clean interface
- [x] **Enhanced STMap Generation System** - Complete mask-aware STMap with RGBA output and intelligent interpolation
- [x] **Enhanced STMap UI Integration** - New export button and status tracking for mask-aware STMap generation
- [x] **Enhanced STMap Coordinate Consistency** - R/G channels now match regular STMap coordinates inside tracker hull
- [x] **Enhanced STMap Frame Math Fix** - Corrected frame range conversion and reference frame handling
- [x] **Enhanced STMap Debug Logging** - Comprehensive logging for troubleshooting and performance monitoring
- [x] **Animated Mask Export System** - Complete animated mask sequence generation with hybrid warping
- [x] **Hybrid Warping Algorithm** - Interpolation inside tracker bounds, block offset outside convex hull
- [x] **Vectorized Performance Optimization** - Dramatic speed improvement using vectorized NaN handling
- [x] **Mask Animation Logic** - Proper backward mapping with negated displacement vectors
- [x] **STMap Generation System** - Complete STMap generation with proper coordinate mapping and RGB channel output
- [x] **STMap Progress Tracking** - Real-time progress bar in Gradio UI showing frame processing status
- [x] **STMap Metadata Integration** - EXR files include embedded metadata with export parameters
- [x] **STMap RGBA Mask Conversion** - Automatic conversion of monochromatic masks to RGBA format
- [x] **STMap Reference Frame Integration** - Proper reference frame handling with frame offset support
- [x] **STMap Filename Convention** - Reference frame included in copied mask filenames with absolute paths
- [x] **STMap UI Optimization** - Condensed layout and intelligent frame range defaults

### Previous Updates (September 30, 2025)
- [x] **Coordinate System Fix** - Fixed Y coordinate offset issue by using actual video height instead of hardcoded 1080p
- [x] **Clipboard Functionality** - Added robust clipboard support with Windows 11 compatibility and multiple fallback methods
- [x] **UI Improvements** - Enhanced interface with better button layout, naming, and user experience
- [x] **Browse Button Fix** - Fixed file picker functionality with proper imports and error handling
- [x] **Modular Architecture** - Completed transition to clean modular structure with separate packages

### Integration Completed (September 29, 2025)
- [x] **generate_exact_nuke_file.py Integration** - Successfully integrated exact Nuke file generation into main app
- [x] **Outputs Directory Structure** - Created `outputs/` directory for organized .nk file storage
- [x] **Configurable Parameters** - Added frame offset, tracker node naming, and reference frame support
- [x] **Gradio Default Path** - Added automatic default path generation in UI
- [x] **Double Frame Offset Fix** - Resolved frame offset being applied twice by fixing script output parsing
- [x] **Hardcoded Values Removal** - Made Root name, tracker node naming, and positioning configurable

### Core Functionality (September 28, 2025)
- [x] **Bidirectional Tracking Fix** - Fixed CoTracker to use `backward_tracking=True` for proper tracking from reference frames
- [x] **Grid Visibility Fix** - Fixed "empty zones" in preview videos by using reference frame for point selection  
- [x] **Model Priority Updates** - Updated to use CoTracker3 offline as primary model
- [x] **Application Status Check** - Verified all core functionality is working properly
- [x] **Testing Validation** - Confirmed bidirectional tracking and grid visibility fixes are functioning

## Technical Notes

### Modular Architecture
```
cotracker_nuke/
├── core/                    # Core application logic
│   ├── app.py              # Main CoTrackerNukeApp orchestrator
│   ├── tracker.py          # CoTracker engine wrapper
│   ├── video_processor.py  # Video loading and processing
│   └── mask_handler.py     # Mask processing and management
├── exporters/              # Export functionality
│   └── nuke_exporter.py    # Nuke .nk file generation
├── ui/                     # User interface
│   └── gradio_interface.py # Web-based Gradio interface
├── utils/                  # Utilities
│   └── logger.py           # Logging configuration
└── cli/                    # Command-line interface
    └── main.py             # CLI entry point
```

### Key Features Implemented
- **Delaunay Hull Detection**: Accurate inside/outside hull classification using scipy.spatial.Delaunay triangulation
- **Unified Processing Logic**: Mask warping and STMap processing use identical hull detection and coordinate calculation
- **Segment-Based Fringe Algorithm**: Geometric projection algorithm for pixels outside tracker hull with proper coordinate mapping
- **Smart Bounding Box Optimization**: 10-20x performance improvement by processing only relevant pixels
- **Enhanced STMap Generation**: Mask-aware STMap with RGBA output (R=X, G=Y, B=0, A=warped mask)
- **Intelligent Interpolation**: Coordinates inside tracker hull match regular STMap, animated mask in alpha channel
- **Animated Mask Export**: Complete animated mask sequence generation with hybrid warping algorithm
- **Hybrid Warping**: Interpolation inside tracker bounds, block offset outside convex hull for coherent movement
- **Vectorized Performance**: Dramatic speed improvements using vectorized operations for NaN handling
- **STMap Generation**: Complete animated STMap sequence generation with proper coordinate mapping
- **Progress Tracking**: Real-time progress bar in Gradio UI with frame-by-frame status updates
- **EXR Metadata**: Embedded metadata in EXR files with export parameters and software information
- **RGBA Mask Conversion**: Automatic conversion of monochromatic masks to RGBA format
- **Reference Frame Integration**: Proper handling of user-selected reference frames with offset support
- **Coordinate System**: Proper transformation from CoTracker (top-left origin) to Nuke (bottom-left origin)
- **Clipboard Integration**: Cross-platform clipboard with Windows 11 support and multiple fallback methods
- **Dynamic Video Height**: Uses actual video dimensions instead of hardcoded values
- **Robust File Export**: Clean .nk file generation with proper error handling
- **Enhanced UI**: Improved button layout, naming, and user feedback

## Development Guidelines
- Follow system rules in `SYSTEM_RULES.md`
- Always use CoTracker3 offline with `backward_tracking=True`
- Never push to git without user confirmation
- Use `.venv` virtual environment for all dependencies
- Maintain modular architecture with clear separation of concerns
- Use proper logging and error handling throughout

---
*Last Updated: October 2, 2025 - Production Ready with Optimized STMap Generation and Unified Processing Systems*
