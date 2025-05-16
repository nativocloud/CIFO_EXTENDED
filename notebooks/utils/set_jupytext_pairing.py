import json
import os
import sys
import glob

def set_jupytext_pairing(notebook_path=None):
    """
    Set Jupytext pairing for a notebook to enable synchronization between .ipynb and .py files.
    
    Args:
        notebook_path: Path to the notebook file. If None, will try to find notebooks in standard locations.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # If no path provided, try to find notebooks in standard locations
    if notebook_path is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to find notebooks in parent directory structure
        search_paths = [
            # Look in notebooks/phase_1_sp directory (new structure)
            os.path.join(os.path.dirname(script_dir), "phase_1_sp", "*.ipynb"),
            # Look in notebooks directory (old structure)
            os.path.join(os.path.dirname(script_dir), "*.ipynb"),
            # Look in current directory
            os.path.join(os.getcwd(), "*.ipynb")
        ]
        
        # Find all notebook files in search paths
        notebook_files = []
        for path in search_paths:
            notebook_files.extend(glob.glob(path))
        
        if not notebook_files:
            print("Error: No notebook files found in standard locations.")
            return False
        
        # Use the first notebook found
        notebook_path = notebook_files[0]
        print(f"Auto-detected notebook: {notebook_path}")
    
    print(f"Attempting to set Jupytext pairing for: {notebook_path}")
    
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file not found at {notebook_path}")
        return False
    
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_data = json.load(f)
        print("Notebook JSON loaded successfully.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {notebook_path}. The file might be corrupted.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while loading the notebook: {e}")
        return False
    
    # Ensure metadata field exists
    if "metadata" not in notebook_data:
        notebook_data["metadata"] = {}
    
    # Ensure jupytext field exists within metadata
    if "jupytext" not in notebook_data["metadata"]:
        notebook_data["metadata"]["jupytext"] = {}
    
    # Set the formats for pairing
    notebook_data["metadata"]["jupytext"]["formats"] = "ipynb,py:percent"
    
    print(f"Jupytext formats metadata set to: {notebook_data['metadata']['jupytext']['formats']}")
    
    try:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook_data, f, indent=1) # Using indent=1 for readability
        print(f"Notebook {notebook_path} updated with Jupytext pairing metadata and saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the notebook with Jupytext metadata: {e}")
        return False
    
    print("Jupytext pairing configuration script finished.")
    return True

if __name__ == "__main__":
    # If notebook path is provided as command line argument, use it
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
        set_jupytext_pairing(notebook_path)
    else:
        # Otherwise try to auto-detect
        set_jupytext_pairing()
