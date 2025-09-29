# CoTracker Nuke App - TODO List

## Status Overview
- **Recent Major Fixes**: ✅ Bidirectional tracking and grid visibility issues resolved
- **Current Priority**: Integrate exact Nuke file generation functionality
- **Next Phase**: Testing and documentation updates

## Current Tasks

### 🧪 Testing & Validation
- [ ] **Test Exact Nuke Generator** - Test `Z:/Dev/Cotracker/generate_exact_nuke_file.py` separately again before implementing in main cotracker app

### 🔧 Integration Tasks  
- [ ] **Integrate Exact Nuke Export** - Use the code from `generate_exact_nuke_file.py` for the "Export to Nuke" button functionality
- [ ] **User File Path Support** - Modify the export functionality to write the .nk file to the user-specified output file path instead of default location

### 📚 Documentation & Maintenance
- [ ] **Update Documentation** - Update README.md and other documentation to reflect all the recent progress and fixes made

## Completed Tasks ✅

### Recent Major Fixes (September 28, 2025)
- [x] **Bidirectional Tracking Fix** - Fixed CoTracker to use `backward_tracking=True` for proper tracking from reference frames
- [x] **Grid Visibility Fix** - Fixed "empty zones" in preview videos by using reference frame for point selection  
- [x] **Model Priority Updates** - Updated to use CoTracker3 offline as primary model
- [x] **Application Status Check** - Verified all core functionality is working properly
- [x] **Testing Validation** - Confirmed bidirectional tracking and grid visibility fixes are functioning

## Technical Notes

### Key Files Modified
- `cotracker_nuke_app.py` - Main application with bidirectional tracking and grid visibility fixes
- `SYSTEM_RULES.md` - Updated with model selection guidelines and backward_tracking requirements
- `temp/BIDIRECTIONAL_TRACKING_FIX_REPORT.md` - Detailed fix documentation
- `temp/GRID_VISIBILITY_FIX_REPORT.md` - Grid display issue resolution

### Integration Requirements
- The `generate_exact_nuke_file.py` contains the precise Nuke export functionality needed
- Must integrate with existing "Export to Nuke" button in main app
- User-specified output file path support needed
- Maintain all existing functionality while adding new export capabilities

## Development Guidelines
- Follow system rules in `SYSTEM_RULES.md`
- Always use CoTracker3 offline with `backward_tracking=True`
- Never push to git without user confirmation
- Use `.venv` virtual environment for all dependencies

---
*Last Updated: September 29, 2025*
