import json
import os

notebook_path = "/home/ubuntu/CIFO_EXTENDED_Project/notebooks/01_single_processor_analysis.ipynb"
print(f"Attempting to correct indentation in notebook: {notebook_path}")

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

cells_to_deindent_ids = ["809abdcd", "24921359", "65583723"]
modifications_made = False

for cell_index, cell in enumerate(notebook_data.get("cells", [])):
    cell_id = cell.get("id")
    cell_type = cell.get("cell_type")

    if cell_id in cells_to_deindent_ids and cell_type == "code":
        print(f"Found cell {cell_id} (index {cell_index}) for de-indentation.")
        original_source = cell.get("source", [])
        new_source = []
        cell_modified_this_iteration = False
        for line in original_source:
            if line.startswith("    "): # 4 spaces
                new_source.append(line[4:])
                cell_modified_this_iteration = True
            else:
                new_source.append(line)
        
        if cell_modified_this_iteration:
            cell["source"] = new_source
            modifications_made = True
            print(f"De-indented cell {cell_id}.")
        else:
            print(f"Cell {cell_id} did not require de-indentation or was already de-indented.")

if modifications_made:
    try:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook_data, f, indent=1) # Using indent=1 for readability
        print(f"Notebook {notebook_path} modified (de-indented) and saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the de-indented notebook: {e}")
        exit(1)
else:
    print("No de-indentation modifications were necessary or made to the notebook.")

print("Indentation correction script finished.")

