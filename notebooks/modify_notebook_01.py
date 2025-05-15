import json

notebook_path = "/home/ubuntu/CIFO_EXTENDED_Project/notebooks/01_single_processor_analysis.ipynb"

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_data = json.load(f)
    
    cell_1f869fde_modified = False
    cell_32f66742_modified = False
    
    for cell in notebook_data.get("cells", []):
        if cell.get("id") == "1f869fde" and cell.get("cell_type") == "code":
            # This cell's source might have been corrupted by previous string replacements.
            # We reset it to the correct state with all_results_summary at the top.
            current_source_tuple = tuple(cell.get("source", []))
            target_source_tuple = tuple([
                "all_results_summary = []\n",
                "# Define problem parameters\n",
                "NUM_TEAMS = 5\n",
                "TEAM_SIZE = 7\n",
                "MAX_BUDGET = 750\n",
                "\n",
                "# Define number of runs for stochastic algorithms (and now Hill Climbing)\n",
                "NUM_RUNS = 30 # Parameter for number of runs (e.g., 10, 30)"
            ])
            if current_source_tuple != target_source_tuple:
                cell["source"] = list(target_source_tuple)
                cell_1f869fde_modified = True
                print("Cell 1f869fde source has been reset and corrected.")
            else:
                print("Cell 1f869fde source is already correct.")
        
        elif cell.get("id") == "32f66742" and cell.get("cell_type") == "code":
            # Remove "    all_results_summary = []\n" from this cell's source
            original_source = cell.get("source", [])
            # The line to remove is specifically '    all_results_summary = []\n' or similar with slight variations in whitespace
            # if it's the last line, it might be '    all_results_summary = []"\n' in the raw JSON string from file_read, 
            # but in the parsed list of strings, it should be '    all_results_summary = []\n'
            new_source = [line for line in original_source if not ("all_results_summary = []" in line and line.strip().startswith("all_results_summary = []"))]
            
            if len(new_source) < len(original_source):
                cell["source"] = new_source
                cell_32f66742_modified = True
                print("Removed local 'all_results_summary = []' from cell 32f66742.")
            else:
                print("Local 'all_results_summary = []' not found in cell 32f66742 for removal, or already removed.")

    if cell_1f869fde_modified or cell_32f66742_modified:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook_data, f, indent=1) # Standard indent for .ipynb files is often 1 for cells array, 2 for top level.
                                                # Using 1 for the whole thing for simplicity, should be readable by Jupyter.
        print(f"Notebook {notebook_path} modified and saved successfully.")
    else:
        print("No modifications made to the notebook as target cells/lines were not found or already correct.")

except FileNotFoundError:
    print(f"Error: Notebook file not found at {notebook_path}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {notebook_path}. The file might be corrupted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

