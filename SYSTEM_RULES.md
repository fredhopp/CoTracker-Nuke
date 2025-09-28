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
- **Use CoTracker2 model for best compatibility with custom reference frames**
- CoTracker3 offline doesn't properly support the queries parameter for custom reference frames
- CoTracker2 provides reliable tracking with both automatic grid and custom query support
- Falls back to CoTracker3 offline if CoTracker2 is not available

### General Development
- Follow these rules consistently throughout the project
- Reference this file when making environment or git-related decisions
