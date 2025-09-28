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

### General Development
- Follow these rules consistently throughout the project
- Reference this file when making environment or git-related decisions
