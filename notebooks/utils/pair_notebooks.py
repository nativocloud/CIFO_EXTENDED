#!/usr/bin/env python3
"""
pair_notebooks.py - Pair Jupyter notebooks with Python scripts and set project kernel

This script provides robust notebook-script pairing that works across different environments
and with different kernel configurations. It can detect existing kernels or use a specified
kernel name, making it resilient to environment differences.

Usage:
    # Pair a single notebook with default kernel
    python pair_notebooks.py path/to/notebook.ipynb

    # Pair a single notebook with specific kernel
    python pair_notebooks.py path/to/notebook.ipynb --kernel custom_kernel_name

    # Pair all notebooks in the project
    python pair_notebooks.py --all

    # Pair all notebooks with specific kernel
    python pair_notebooks.py --all --kernel custom_kernel_name
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Default kernel settings - aligned with setup_notebooks.py
DEFAULT_KERNEL_NAME = "cifo_project_kernel"
DEFAULT_KERNEL_DISPLAY = "CIFO Project Kernel"


def detect_available_kernels():
    """Detect available Jupyter kernels in the system."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        kernels = json.loads(result.stdout)
        return kernels.get("kernelspecs", {})
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        print("⚠️ Could not detect available kernels. Using default kernel settings.")
        return {}


def get_best_kernel(kernel_name=None):
    """Get the best available kernel or create settings for the specified one."""
    if kernel_name:
        # User specified a kernel name, use it
        return {
            "name": kernel_name,
            "display_name": f"{kernel_name.replace('_', ' ').title()}",
        }

    # Try to detect available kernels
    available_kernels = detect_available_kernels()

    # Check if our default kernel exists
    if DEFAULT_KERNEL_NAME in available_kernels:
        kernel_info = available_kernels[DEFAULT_KERNEL_NAME]
        return {
            "name": DEFAULT_KERNEL_NAME,
            "display_name": kernel_info.get("spec", {}).get(
                "display_name", DEFAULT_KERNEL_DISPLAY
            ),
        }

    # Check if python3 kernel exists (common default)
    if "python3" in available_kernels:
        kernel_info = available_kernels["python3"]
        return {
            "name": "python3",
            "display_name": kernel_info.get("spec", {}).get("display_name", "Python 3"),
        }

    # Fallback to default settings
    return {"name": DEFAULT_KERNEL_NAME, "display_name": DEFAULT_KERNEL_DISPLAY}


def pair_notebook(notebook_path, script_path=None, kernel_name=None):
    """Pair a notebook with a Python script and set project kernel."""
    nb_path = Path(notebook_path)

    if not nb_path.exists():
        print(f"❌ Error: Notebook {nb_path} does not exist")
        return False

    if script_path is None:
        script_path = nb_path.with_suffix(".py")
    else:
        script_path = Path(script_path)

    script_path.parent.mkdir(parents=True, exist_ok=True)

    # Get kernel information
    kernel_info = get_best_kernel(kernel_name)

    try:
        if nb_path.exists():
            try:
                with open(nb_path, "r", encoding="utf-8") as f:
                    nb = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ Error: Notebook {nb_path} has invalid JSON: {e}")
                print("   This might be due to corruption or incomplete edits.")
                return False
        else:
            nb = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}

        if "metadata" not in nb:
            nb["metadata"] = {}

        # Add Jupytext configuration - more explicit for Windows compatibility
        nb["metadata"]["jupytext"] = {
            "formats": "ipynb,py:percent",
            "text_representation": {
                "extension": ".py",
                "format_name": "percent",
                "format_version": "1.3",
            },
            # Add explicit Windows-friendly settings
            "notebook_metadata_filter": "all",
            "cell_metadata_filter": "all",
        }

        # Ensure the notebook uses the project kernel
        nb["metadata"]["kernelspec"] = {
            "name": "project_kernel",
            "display_name": "Project Kernel",
            "language": "python",
        }

        # Write with proper line endings for cross-platform compatibility
        with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=2)

        if not script_path.exists():
            # Create script with Windows-friendly line endings
            with open(script_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(
                    """# %% [markdown]
# # Notebook Title
# This is a paired Jupyter notebook

# %%
# Initialize notebook environment
import sys
import os
from pathlib import Path

# Try to find and run the initialization script
try:
    # Look for init_notebook.py in various locations
    notebook_dir = Path(os.getcwd())
    init_paths = [
        notebook_dir / "utils" / "init_notebook.py",
        notebook_dir / "../utils" / "init_notebook.py",
        notebook_dir.parent / "utils" / "init_notebook.py",
        notebook_dir.parent / "notebooks" / "utils" / "init_notebook.py"
    ]
    
    init_script = None
    for path in init_paths:
        if path.exists():
            init_script = path
            break
    
    if init_script:
        print(f"Running initialization script: {init_script}")
        %run {init_script}
    else:
        print("Initialization script not found. You may need to set up paths manually.")
except Exception as e:
    print(f"Error initializing notebook: {e}")
    print("You may need to set up paths manually.")

# Your code here
print("Hello, World!")
"""
                )

        print(f"✅ Paired {nb_path.name} with {script_path.name}")
        print(f"   Kernel: Project Kernel (project_kernel)")
        return True

    except Exception as e:
        print(f"❌ Error pairing notebook: {e}")
        return False


def pair_all_notebooks(directory=".", kernel_name=None):
    """Pair all notebooks in the given directory and set project kernel."""
    directory = Path(directory)
    notebooks = list(directory.glob("**/*.ipynb"))
    notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]

    if not notebooks:
        print("No notebooks found in the project")
        return

    # Get kernel information once to use for all notebooks
    kernel_info = get_best_kernel(kernel_name)
    print(f"Using kernel: {kernel_info['display_name']} ({kernel_info['name']})")

    print(f"Found {len(notebooks)} notebooks. Pairing...\n")
    success_count = 0

    for nb in notebooks:
        if pair_notebook(nb, kernel_name=kernel_info["name"]):
            success_count += 1

    print(f"\n✅ Successfully paired {success_count} of {len(notebooks)} notebooks")

    if success_count < len(notebooks):
        print("⚠️ Some notebooks could not be paired. Check the errors above.")

    print("\nNext steps:")
    print("1. Make sure the kernel exists in your environment")
    print(f"   - If using '{DEFAULT_KERNEL_NAME}', run setup_notebooks.py to create it")
    print("   - Or select an existing kernel when opening notebooks")
    print("2. When editing notebooks, changes will sync to .py files automatically")
    print(
        "3. When editing .py files, restart the kernel to see changes in the notebook"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pair Jupyter notebooks with Python scripts and set project kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n")[1],  # Use the usage section from the docstring
    )
    parser.add_argument("notebook", nargs="?", help="Path to the notebook")
    parser.add_argument(
        "script", nargs="?", help="Path to the Python script (optional)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Pair all notebooks in the project"
    )
    parser.add_argument("--kernel", help="Kernel name to use (default: auto-detect)")

    args = parser.parse_args()

    if args.all:
        pair_all_notebooks(kernel_name=args.kernel)
    elif args.notebook:
        pair_notebook(args.notebook, args.script, kernel_name=args.kernel)
    else:
        parser.print_help()
