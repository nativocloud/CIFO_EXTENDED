"""
init_notebook.py - Initialization script for Jupyter notebooks in the CIFO_EXTENDED project

This script sets up the Python environment for notebooks, ensuring:
1. Proper path configuration for imports
2. Module auto-reloading
3. Standard imports
4. Consistent environment across different machines

Usage:
    Add the following to the first cell of your notebook:
    %run init_notebook.py
"""

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure matplotlib
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# Enable auto-reloading of modules
try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    print(
        "✅ Autoreload enabled: Changes to Python modules will be automatically reloaded"
    )
except:
    print("⚠️ Running outside IPython/Jupyter environment - autoreload not available")

# Determine project root directory
notebook_dir = Path(os.getcwd())
project_root = None

# Try to find the project root by looking for key directories/files
for parent in [notebook_dir] + list(notebook_dir.parents):
    if (parent / "src").exists() and (parent / "data").exists():
        project_root = parent
        break

if project_root is None:
    # If we couldn't find the project root, use the current directory's parent
    project_root = notebook_dir.parent
    print(f"⚠️ Project root not detected, using: {project_root}")
else:
    print(f"✅ Project root detected: {project_root}")

# Add project root to Python path for imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"✅ Added {project_root} to Python path")

# Define common paths
DATA_DIR = project_root / "data"
SRC_DIR = project_root / "src"
RESULTS_DIR = project_root / "results"

# Create results directory structure if it doesn't exist
PHASE_1_DIR = RESULTS_DIR / "phase_1_sp"
IMAGES_DIR = PHASE_1_DIR / "images"
DATA_DIR_RESULTS = PHASE_1_DIR / "data"

for dir_path in [RESULTS_DIR, PHASE_1_DIR, IMAGES_DIR, DATA_DIR_RESULTS]:
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
        print(f"✅ Created directory: {dir_path}")

# Print environment information
print("\n=== Environment Information ===")
print(f"Python version: {sys.version.split()[0]}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Data directory: {DATA_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print("===============================\n")

# Check if required data files exist
players_file = DATA_DIR / "players.csv"
if players_file.exists():
    print(f"✅ Data file found: {players_file}")
else:
    print(f"❌ Data file not found: {players_file}")

print(
    "\nNotebook initialization complete. You can now import project modules and run your analysis."
)
