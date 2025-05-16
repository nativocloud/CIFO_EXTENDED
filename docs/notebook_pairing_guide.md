# Jupyter Notebook and Python Script Pairing Guide

This guide explains how to set up and maintain synchronization between Jupyter notebooks (`.ipynb`) and Python scripts (`.py`) in the CIFO_EXTENDED project.

## Table of Contents
- [Quick Start](#quick-start)
- [Available Tools](#available-tools)
- [Basic Pairing Setup](#basic-pairing-setup)
- [Advanced DevOps Workflow](#advanced-devops-workflow)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Option 1: Simple Pairing (Recommended for Individual Work)

Run this command to pair all notebooks in the project:

```bash
python -m notebooks.utils.pair_notebooks --all
```

Or for a specific notebook:

```bash
python -m notebooks.utils.pair_notebooks path/to/notebook.ipynb
```

### Option 2: Initialize Notebook Environment

Add this to the first cell of your notebook:

```python
# Initialize notebook environment
%run ../utils/init_notebook.py
```

## Available Tools

The project provides several tools for notebook-script synchronization:

1. **`set_jupytext_pairing.py`**: Basic script that configures Jupytext pairing with relative paths
2. **`pair_notebooks.py`**: More advanced script that can pair individual or all notebooks
3. **`init_notebook.py`**: Environment initialization script for consistent notebook execution

## Basic Pairing Setup

### Step 1: Pair Your Notebooks

```bash
python -m notebooks.utils.pair_notebooks --all
```

This will:
- Find all notebooks in the project
- Configure them for Jupytext pairing with Python scripts
- Create paired `.py` files if they don't exist

### Step 2: Initialize Your Notebook Environment

Add this to the first cell of each notebook:

```python
# Initialize notebook environment
%run ../utils/init_notebook.py
```

This ensures:
- Proper path configuration
- Module auto-reloading
- Consistent environment across machines

### Step 3: Working with Paired Files

- Edit either the `.ipynb` or `.py` file
- Save the file you're editing
- The paired file will update automatically
- Use Git to track changes in the `.py` files

## Advanced DevOps Workflow

For team environments or CI/CD pipelines, a more robust setup is available:

### Step 1: Install Development Requirements

Create a `requirements-dev.txt` file:

```
jupyter>=1.0.0
jupytext>=1.13.0
pre-commit>=2.0.0
```

Install with:

```bash
pip install -r requirements-dev.txt
```

### Step 2: Configure Jupyter

Create or edit `~/.jupyter/jupyter_notebook_config.py`:

```python
c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"
c.ContentsManager.default_jupytext_formats = "ipynb,py:percent"
c.ContentsManager.default_notebook_metadata_filter = "all,-language_info"
c.ContentsManager.default_cell_metadata_filter = "-all"
```

### Step 3: Set Up Pre-commit Hook

Create `.pre-commit-config.yaml`:

```yaml
repos:
-   repo: local
    hooks:
    -   id: sync-notebooks
        name: Sync Jupyter notebooks
        entry: python -m jupytext --sync
        language: system
        types: [jupyter]
        always_run: true
        pass_filenames: false
```

Install the hook:

```bash
pre-commit install
```

## Troubleshooting

### Notebooks Not Syncing

1. Check if Jupytext is installed:
   ```bash
   pip install jupytext
   ```

2. Verify notebook metadata:
   ```bash
   python -m notebooks.utils.pair_notebooks path/to/notebook.ipynb
   ```

3. Try manual sync:
   ```bash
   jupytext --sync path/to/notebook.ipynb
   ```

### Path Issues

If you encounter path-related errors:

1. Make sure to run the initialization script:
   ```python
   %run ../utils/init_notebook.py
   ```

2. Check the project structure:
   ```
   CIFO_EXTENDED_Project/
   ├── notebooks/
   │   ├── phase_1_sp/
   │   └── utils/
   ├── src/
   └── data/
   ```

### Kernel Issues

If the kernel can't find modules:

1. Restart the kernel
2. Run the initialization script first
3. Check that the project root is in your Python path:
   ```python
   import sys
   print(sys.path)
   ```
