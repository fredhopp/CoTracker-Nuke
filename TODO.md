# CoTracker Nuke App - TODO List

## Status Overview
- **Recent Major Fixes**: ✅ Bidirectional tracking and grid visibility issues resolved
- **Integration Status**: ✅ generate_exact_nuke_file.py successfully integrated into main app
- **Current Priority**: Fix double frame offset application issue
- **Next Phase**: Testing and documentation updates

## Current Tasks

### 🐛 Critical Fixes
- [ ] **Fix Double Frame Offset** - Frame offset is being applied twice:
  1. In `generate_csv_with_frame_offset()` when creating CSV
  2. In `generate_exact_nuke_file()` when processing frame numbers
  - **Solution**: Remove intermediate CSV with added offset, apply frame offset only in .nk generation
  - **Impact**: Frame values in .nk file are currently offset by 2x the intended amount

### 🔧 Integration Tasks  
- [x] **Integrate Exact Nuke Export** - ✅ Successfully integrated `generate_exact_nuke_file.py` into main app
- [x] **User File Path Support** - ✅ Added default path `outputs/CoTracker_YYMMDD_HHMMSS.nk` in Gradio interface
- [x] **Reference Frame Calculation** - ✅ Fixed to use user_reference_frame + image_sequence_start_frame
- [x] **Remove Translate/Center Curves** - ✅ Removed unnecessary translate and center lines from .nk output

### 📚 Documentation & Maintenance
- [ ] **Update Documentation** - Update README.md and other documentation to reflect all the recent progress and fixes made

## Completed Tasks ✅

### Integration Completed (September 29, 2025)
- [x] **generate_exact_nuke_file.py Integration** - Successfully integrated exact Nuke file generation into main app
- [x] **Outputs Directory Structure** - Created `outputs/` directory for organized .nk file storage
- [x] **Configurable Parameters** - Added frame offset, tracker node naming, and reference frame support
- [x] **Gradio Default Path** - Added automatic default path generation in UI
- [x] **CSV Generation with Frame Offset** - Added `generate_csv_with_frame_offset()` method (needs revision)
- [x] **Hardcoded Values Removal** - Made Root name, tracker name, and positioning configurable

### Recent Major Fixes (September 28, 2025)
- [x] **Bidirectional Tracking Fix** - Fixed CoTracker to use `backward_tracking=True` for proper tracking from reference frames
- [x] **Grid Visibility Fix** - Fixed "empty zones" in preview videos by using reference frame for point selection  
- [x] **Model Priority Updates** - Updated to use CoTracker3 offline as primary model
- [x] **Application Status Check** - Verified all core functionality is working properly
- [x] **Testing Validation** - Confirmed bidirectional tracking and grid visibility fixes are functioning

## Technical Notes

### Key Files Modified
- `cotracker_nuke_app.py` - Main application with integrated Nuke export functionality
- `generate_exact_nuke_file.py` - Enhanced with configurable parameters and frame offset support
- `outputs/` - New directory for organized .nk file storage
- `TODO.md` - Updated to reflect integration status and current priorities

### Current Architecture
- **CSV Generation**: `generate_csv_with_frame_offset()` creates intermediate CSV (needs revision)
- **Nuke Export**: `generate_exact_nuke_file()` processes CSV and generates .nk files
- **Frame Offset Issue**: Currently applied twice - once in CSV generation, once in .nk processing
- **Default Paths**: Automatic timestamp-based naming: `outputs/CoTracker_YYMMDD_HHMMSS.nk`

## Development Guidelines
- Follow system rules in `SYSTEM_RULES.md`
- Always use CoTracker3 offline with `backward_tracking=True`
- Never push to git without user confirmation
- Use `.venv` virtual environment for all dependencies

---
*Last Updated: September 29, 2025 - Integration Complete, Double Frame Offset Issue Identified*
