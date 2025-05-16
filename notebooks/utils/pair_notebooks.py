"""
pair_notebooks.py - Script for pairing Jupyter notebooks with Python scripts

This script configures Jupytext pairing between .ipynb notebooks and .py files,
ensuring proper synchronization for version control and collaboration.

Usage:
    python -m notebooks.utils.pair_notebooks [notebook_path]
    python -m notebooks.utils.pair_notebooks --all
"""

import json
import argparse
import os
import glob
from pathlib import Path

def pair_notebook(notebook_path, script_path=None):
    """
    Pair a notebook with a Python script using Jupytext.
    
    Args:
        notebook_path: Path to the notebook file
        script_path: Optional path to the Python script (if None, uses same name with .py extension)
    
    Returns:
        bool: True if successful, False otherwise
    """
    nb_path = Path(notebook_path)
    
    if not nb_path.exists():
        print(f"Error: Notebook {nb_path} does not exist")
        return False
    
    if script_path is None:
        script_path = nb_path.with_suffix('.py')
    else:
        script_path = Path(script_path)
    
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {nb_path}. The file might be corrupted.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while loading the notebook: {e}")
        return False
    
    # Ensure metadata field exists
    if 'metadata' not in nb:
        nb['metadata'] = {}
    
    # Ensure jupytext field exists within metadata
    if 'jupytext' not in nb['metadata']:
        nb['metadata']['jupytext'] = {}
    
    # Set the formats for pairing
    nb['metadata']['jupytext'] = {
        'formats': 'ipynb,py:percent',
        'text_representation': {
            'extension': '.py',
            'format_name': 'percent',
            'format_version': '1.3'
        }
    }
    
    try:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"✅ Paired {nb_path.name} with {script_path.name}")
        return True
    except Exception as e:
        print(f"Error saving notebook: {e}")
        return False

def pair_all_notebooks(directory=None):
    """
    Pair all notebooks in the given directory and its subdirectories.
    
    Args:
        directory: Directory to search for notebooks (default: current project)
    """
    if directory is None:
        # Try to find project root
        current_dir = Path(os.getcwd())
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / 'src').exists() and (parent / 'data').exists():
                directory = parent
                break
        
        if directory is None:
            directory = current_dir
    else:
        directory = Path(directory)
    
    print(f"Searching for notebooks in {directory}...")
    
    # Find all notebooks, excluding checkpoints
    notebooks = []
    for pattern in ['**/*.ipynb']:
        notebooks.extend([nb for nb in directory.glob(pattern) 
                         if '.ipynb_checkpoints' not in str(nb)])
    
    if not notebooks:
        print("No notebooks found.")
        return False
    
    print(f"Found {len(notebooks)} notebooks:")
    for nb in notebooks:
        print(f"  - {nb.relative_to(directory)}")
    
    print("\nPairing notebooks...")
    success_count = 0
    for nb in notebooks:
        if pair_notebook(nb):
            success_count += 1
    
    print(f"\n✅ Successfully paired {success_count}/{len(notebooks)} notebooks.")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description='Pair Jupyter notebooks with Python scripts')
    parser.add_argument('notebook', nargs='?', help='Path to the notebook')
    parser.add_argument('--all', action='store_true', help='Pair all notebooks in the project')
    
    args = parser.parse_args()
    
    if args.all:
        pair_all_notebooks()
    elif args.notebook:
        pair_notebook(args.notebook)
    else:
        print("Please specify a notebook or use --all")
        print("Usage: python -m notebooks.utils.pair_notebooks [notebook_path]")
        print("       python -m notebooks.utils.pair_notebooks --all")

if __name__ == "__main__":
    main()
