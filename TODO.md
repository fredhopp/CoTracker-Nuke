# CoTracker Nuke App - TODO List

## Status Overview
- **Current Status**: ✅ Production ready with modular architecture
- **Recent Major Fixes**: ✅ Coordinate transformation, clipboard functionality, UI improvements
- **Architecture**: ✅ Fully modular structure with clean separation of concerns
- **Current Priority**: Feature complete - ready for production use

## Current Tasks

### 🚀 Future Enhancements (Optional)
- [ ] **Batch Processing** - Add support for processing multiple videos at once
- [ ] **Advanced Mask Tools** - Add more sophisticated masking options
- [ ] **Export Formats** - Add support for other compositing software (After Effects, etc.)
- [ ] **Performance Optimization** - Further optimize for very large numbers of tracks

### 🔧 Maintenance Tasks
- [ ] **Testing** - Comprehensive testing across different video formats and resolutions
- [ ] **Documentation** - Add video tutorials and advanced usage examples

## Completed Tasks ✅

### Latest Updates (September 30, 2025)
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
*Last Updated: September 30, 2025 - Production Ready with Full Feature Set*
