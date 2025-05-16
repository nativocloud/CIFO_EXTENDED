# Notebook Initialization and Environment Setup Guide

This guide explains how to properly initialize Jupyter notebooks in the CIFO_EXTENDED project to ensure consistent behavior across different environments.

## Table of Contents
- [Quick Start](#quick-start)
- [Why Initialization is Important](#why-initialization-is-important)
- [The `init_notebook.py` Script](#the-init_notebookpy-script)
- [Manual Configuration (Alternative)](#manual-configuration-alternative)
- [Troubleshooting](#troubleshooting)

## Quick Start

Add this to the first code cell of your notebook:

```python
# Initialize notebook environment
%run ../utils/init_notebook.py
```

For notebooks in the `phase_1_sp` directory, use:

```python
# Initialize notebook environment
%run ../utils/init_notebook.py
```

## Why Initialization is Important

Proper notebook initialization solves several common issues:

1. **Path Resolution**: Ensures modules in `src/` can be imported regardless of where the notebook is located
2. **Auto-reloading**: Changes to Python modules are automatically reloaded without restarting the kernel
3. **Environment Consistency**: Same behavior across different machines and environments
4. **Directory Structure**: Automatically creates necessary output directories
5. **Data Verification**: Checks if required data files exist

## The `init_notebook.py` Script

The initialization script provides:

### Automatic Features
- **Path Detection**: Automatically finds the project root directory
- **Python Path Configuration**: Adds project root to `sys.path` for imports
- **Directory Creation**: Creates necessary output directories if they don't exist
- **Environment Information**: Displays Python version and key directories
- **Data Verification**: Checks if required data files exist

### Pre-configured Settings
- **Matplotlib**: Sets up consistent plot styling
- **Common Imports**: Includes frequently used libraries
- **Path Variables**: Defines standard paths for data, results, etc.

## Manual Configuration (Alternative)

If you prefer to manually configure your notebook, add these cells:

```python
# Enable auto-reloading of modules
%load_ext autoreload
%autoreload 2

# Add project root to Python path
import sys
import os
from pathlib import Path

# Find project root (adjust as needed)
project_root = Path(os.getcwd()).parent
sys.path.insert(0, str(project_root))

# Define key directories
DATA_DIR = project_root / "data"
RESULTS_DIR = project_root / "results"
```

## Troubleshooting

### Imports Not Working
- Verify the project structure is correct
- Check that `src/` directory exists
- Print `sys.path` to ensure project root is included

### Paths Not Resolving Correctly
- Use `print(project_root)` to check detected project root
- Manually set project root if auto-detection fails:
  ```python
  project_root = Path('/absolute/path/to/CIFO_EXTENDED_Project')
  ```

### Changes to Modules Not Reflected
- Ensure `%autoreload 2` is active
- Try restarting the kernel if changes still aren't reflected
- Check that the module is being imported after initialization

### Missing Output Directories
- The script should create them automatically
- If not, manually create with:
  ```python
  from pathlib import Path
  Path("results/phase_1_sp/images").mkdir(parents=True, exist_ok=True)
  ```
