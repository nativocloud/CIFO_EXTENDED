import json

notebook_path = '/home/ubuntu/CIFO_EXTENDED_Project/notebooks/02_multi_processor_analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook_content = json.load(f)

notebook_content['metadata']['jupytext'] = {
    "formats": "ipynb,py:percent",
    "text_representation": {
        "extension": ".py",
        "format_name": "percent",
        "format_version": "1.3",
        "jupytext_version": "1.15.2" # Use a recent version, adjust if needed
    }
}

# Ensure kernelspec is present, Jupytext often adds it or expects it
if 'kernelspec' not in notebook_content['metadata']:
    notebook_content['metadata']['kernelspec'] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    }

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)
    f.write('\n') # Add a newline at the end of the file, common practice

print(f"Jupytext metadata added to {notebook_path}")
