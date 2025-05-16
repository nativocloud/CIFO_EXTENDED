import json

notebook_path = '/home/ubuntu/CIFO_EXTENDED_Project/notebooks/03_multi_processor_analysis_rebuilt.ipynb'
script_path = '03_multi_processor_analysis_rebuilt.py'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook_content = json.load(f)

if 'metadata' not in notebook_content:
    notebook_content['metadata'] = {}
if 'jupytext' not in notebook_content['metadata']:
    notebook_content['metadata']['jupytext'] = {}

notebook_content['metadata']['jupytext']['formats'] = 'ipynb,py:percent'
# Ensure the paired .py file is correctly referenced if it's in the same directory
# Jupytext usually handles this automatically if the .py file has the same base name
# but explicitly setting text_representation.extension can be done if needed.
# For now, relying on the formats string and --sync command.

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)

print(f"Jupytext pairing metadata updated in {notebook_path} to pair with {script_path}")
