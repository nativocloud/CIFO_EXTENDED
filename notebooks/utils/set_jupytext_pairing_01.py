import json
import os

notebook_path = "/home/ubuntu/CIFO_EXTENDED_Project/notebooks/01_single_processor_analysis.ipynb"
print(f"Attempting to set Jupytext pairing for: {notebook_path}")

if not os.path.exists(notebook_path):
    print(f"Error: Notebook file not found at {notebook_path}")
    exit(1)

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_data = json.load(f)
    print("Notebook JSON loaded successfully.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {notebook_path}. The file might be corrupted.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading the notebook: {e}")
    exit(1)

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
    exit(1)

print("Jupytext pairing configuration script finished.")

