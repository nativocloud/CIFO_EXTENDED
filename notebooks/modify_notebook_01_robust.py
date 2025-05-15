import json
import os

notebook_path = "/home/ubuntu/CIFO_EXTENDED_Project/notebooks/01_single_processor_analysis.ipynb"
print(f"Attempting to modify notebook: {notebook_path}")

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

cell_1f869fde_modified = False
cell_32f66742_modified = False
modification_made = False

# Target source for cell 1f869fde
target_source_1f869fde = [
    "all_results_summary = []\n",
    "# Define problem parameters\n",
    "NUM_TEAMS = 5\n",
    "TEAM_SIZE = 7\n",
    "MAX_BUDGET = 750\n",
    "\n",
    "# Define number of runs for stochastic algorithms (and now Hill Climbing)\n",
    "NUM_RUNS = 30 # Parameter for number of runs (e.g., 10, 30)"
]

for cell_index, cell in enumerate(notebook_data.get("cells", [])):
    cell_id = cell.get("id")
    cell_type = cell.get("cell_type")

    if cell_id == "1f869fde" and cell_type == "code":
        print(f"Found cell 1f869fde (index {cell_index}). Current source: {cell.get('source')}")
        # Ensure the source is exactly what we want
        if cell.get("source") != target_source_1f869fde:
            cell["source"] = target_source_1f869fde
            cell_1f869fde_modified = True
            modification_made = True
            print("Cell 1f869fde source has been corrected.")
        else:
            print("Cell 1f869fde source is already correct.")

    elif cell_id == "32f66742" and cell_type == "code":
        print(f"Found cell 32f66742 (index {cell_index}).")
        original_source = cell.get("source", [])
        # Remove the specific line "    all_results_summary = []\n"
        # or "    all_results_summary = []" if it's the last line in the list without a newline char
        new_source = [line for line in original_source if not ("all_results_summary = []" in line and line.strip() == "all_results_summary = []")]

        if len(new_source) < len(original_source):
            cell["source"] = new_source
            cell_32f66742_modified = True
            modification_made = True
            print("Removed local 'all_results_summary = []' from cell 32f66742.")
        else:
            print("Local 'all_results_summary = []' not found or already removed in cell 32f66742.")

if modification_made:
    try:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook_data, f, indent=1) # Using indent=1 for readability
        print(f"Notebook {notebook_path} modified and saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the notebook: {e}")
        exit(1)
else:
    print("No modifications were necessary or made to the notebook.")

print("Script finished.")

