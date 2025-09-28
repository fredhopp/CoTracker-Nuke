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
- **ALWAYS use CoTracker3 offline model only**
- The offline model behaves much better than the online model
- Provides superior temporal consistency and no sliding window artifacts
- Eliminates repeating animation cycles and frame synchronization issues

### General Development
- Follow these rules consistently throughout the project
- Reference this file when making environment or git-related decisions
