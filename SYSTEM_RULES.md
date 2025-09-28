# System Rules for CoTracker Nuke App Project

## Development Guidelines

### Git Management
- **NEVER push to git before confirming with the user**
- Always ask for permission before any git operations (commit, push, etc.)

### Environment Management
- **Create and use a virtual environment called `.venv` in the project root**
- **Install all dependencies in the virtual environment, NOT system-wide**
- Use `python -m venv .venv` to create the virtual environment
- Activate with `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)
- Always install packages using `pip install` within the activated virtual environment

### Model Selection
- **ALWAYS use CoTracker3 offline as the primary model**
- CoTracker3 offline provides superior temporal consistency and bidirectional tracking
- **CRITICAL: Always use backward_tracking=True parameter for proper bidirectional tracking**
- Falls back to CoTracker2 (with backward_tracking=True) if CoTracker3 offline is not available
- CoTracker3 online does NOT support backward_tracking parameter and should be last resort

### General Development
- Follow these rules consistently throughout the project
- Reference this file when making environment or git-related decisions
