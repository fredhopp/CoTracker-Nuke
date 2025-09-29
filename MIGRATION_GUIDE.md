# 🔄 Migration Guide: Monolithic → Modular CoTracker Nuke App

## 🎯 **Quick Start**

### **Option 1: Use New Modular Version (Recommended)**
```bash
# Launch the new modular GUI
python cotracker_nuke_app_new.py

# With custom logging level
python cotracker_nuke_app_new.py --log-level DEBUG

# CLI batch processing (NEW!)
python -m cotracker_nuke.cli video.mp4 output.nk --grid-size 15
```

### **Option 2: Keep Using Original (Fallback)**
```bash
# Original monolithic version still available
python cotracker_nuke_app.py
```

---

## 📊 **What Changed**

### **Files Added** ✅
```
cotracker_nuke/                    # New modular package
├── core/
│   ├── app.py                     # Main orchestrator (87 lines)
│   ├── tracker.py                 # CoTracker logic (280 lines)  
│   ├── video_processor.py         # Video handling (120 lines)
│   └── mask_handler.py            # Mask processing (230 lines)
├── exporters/
│   └── nuke_exporter.py           # Export logic (250 lines)
├── ui/
│   └── gradio_interface.py        # UI components (280 lines)
├── cli/
│   └── main.py                    # CLI interface (150 lines)
└── utils/
    └── logger.py                  # Logging system (120 lines)

cotracker_nuke_app_new.py          # New modular entry point
demo_modular_usage.py              # Usage examples
REFACTORING_SUMMARY.md             # Detailed comparison
MIGRATION_GUIDE.md                 # This file
```

### **Files Unchanged** 🔒
- `cotracker_nuke_app.py` - Original monolithic version (preserved)
- `generate_exact_nuke_file.py` - Nuke generation script (unchanged)
- All other project files remain exactly the same

---

## 🚀 **New Features Available**

### **1. Professional Logging System**
```python
# Before: Mixed print/logging statements
print("Loading video...")
self.logger.info("Video loaded")

# After: Consistent, configurable logging
logger = setup_logger("cotracker", console_level="DEBUG")
logger.debug("Detailed debugging info")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### **2. Command-Line Interface**
```bash
# Batch processing support (NEW!)
python -m cotracker_nuke.cli video.mp4 output.nk \
    --grid-size 15 \
    --reference-frame 10 \
    --frame-offset 1001 \
    --mask mask.png \
    --log-level DEBUG
```

### **3. Modular API**
```python
# Use individual components
from cotracker_nuke.core import CoTrackerEngine, VideoProcessor
from cotracker_nuke.exporters import NukeExporter

# Or use the main app
from cotracker_nuke import CoTrackerNukeApp
app = CoTrackerNukeApp(console_log_level="DEBUG")
```

---

## 🔧 **For Developers**

### **Testing Individual Components**
```python
# Test video processing independently
from cotracker_nuke.core import VideoProcessor
processor = VideoProcessor()
video = processor.load_video("test.mp4")

# Test tracking independently  
from cotracker_nuke.core import CoTrackerEngine
tracker = CoTrackerEngine()
tracks, visibility = tracker.track_points(video, grid_size=10)

# Test export independently
from cotracker_nuke.exporters import NukeExporter
exporter = NukeExporter()
csv_path = exporter.generate_csv_for_nuke_export(tracks, visibility)
```

### **Custom Logging Setup**
```python
from cotracker_nuke.utils import setup_logger

# Create custom logger
logger = setup_logger(
    name="my_custom_tracker",
    debug_dir=Path("my_logs"),
    console_level="WARNING"  # Only show warnings/errors on console
)
```

### **Extending Functionality**
```python
# Easy to extend with custom exporters
class CustomExporter(NukeExporter):
    def export_to_maya(self, tracks, output_path):
        # Custom Maya export logic
        pass

# Easy to add new UI components
class CustomInterface(GradioInterface):
    def add_custom_controls(self):
        # Custom UI elements
        pass
```

---

## 🎯 **Migration Strategies**

### **Strategy 1: Gradual Migration (Recommended)**
1. **Week 1**: Test new version alongside existing workflow
2. **Week 2**: Use new version for new projects
3. **Week 3**: Migrate existing workflows gradually
4. **Week 4+**: Fully adopt new version

### **Strategy 2: Immediate Switch**
- Replace `cotracker_nuke_app.py` with `cotracker_nuke_app_new.py` in your scripts
- All functionality remains identical
- Benefit from improved logging and architecture immediately

### **Strategy 3: Hybrid Approach**
- Keep using original GUI for familiar workflows
- Use new CLI for batch processing
- Gradually explore modular components for custom scripts

---

## ⚠️ **Compatibility Notes**

### **✅ Fully Compatible**
- All original functionality preserved
- Same Gradio interface layout and behavior
- Same output .nk file format
- Same tracking quality and performance
- Same dependencies and requirements

### **🆕 Additional Benefits**
- Better error messages with proper logging
- Configurable verbosity levels
- Cleaner console output
- Better debugging information
- CLI support for automation

### **📁 File Structure**
- Original `cotracker_nuke_app.py` preserved as fallback
- New modular version available as `cotracker_nuke_app_new.py`
- All existing scripts and workflows continue to work

---

## 🛠️ **Troubleshooting**

### **Import Issues**
```python
# If you get import errors, make sure you're in the right directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cotracker_nuke import CoTrackerNukeApp
```

### **Logging Issues**
```bash
# If logging is too verbose, reduce the level
python cotracker_nuke_app_new.py --log-level WARNING

# If you need more debugging info
python cotracker_nuke_app_new.py --log-level DEBUG
```

### **Performance**
```bash
# If startup is slower, disable debug logging
python cotracker_nuke_app_new.py --no-debug
```

---

## 🎉 **Success Indicators**

You'll know the migration is successful when:

✅ **Startup**: Clean initialization with proper logging  
✅ **Interface**: Same familiar Gradio interface  
✅ **Functionality**: All features work as before  
✅ **Logging**: Better error messages and debugging info  
✅ **CLI**: New batch processing capabilities available  
✅ **Development**: Easy to modify and extend individual components  

---

## 📞 **Need Help?**

- **Check logs**: Look in `temp/cotracker_debug_*.log` for detailed information
- **Test original**: Fall back to `cotracker_nuke_app.py` if needed
- **Start simple**: Begin with `python cotracker_nuke_app_new.py` (no arguments)
- **Use CLI**: Try batch processing with the new CLI interface

---

**🎯 The modular architecture provides the same functionality with better maintainability, testability, and extensibility while preserving full backward compatibility.**
