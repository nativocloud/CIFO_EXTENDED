import json

notebook_path = "/home/ubuntu/CIFO_EXTENDED_Project/notebooks/04_multi_processor_analysis_clean.ipynb"

# Configuração do emparelhamento Jupytext
metadata = {
    "jupytext": {
        "formats": "ipynb,py:percent"
    }
}

# Carregar o notebook existente (se existir)
try:
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Atualizar os metadados
    if 'metadata' not in notebook:
        notebook['metadata'] = {}
    
    notebook['metadata'].update(metadata)
    
    # Guardar o notebook atualizado
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Jupytext pairing configurado para {notebook_path}")
except FileNotFoundError:
    print(f"O ficheiro {notebook_path} ainda não existe. Será criado com o emparelhamento configurado quando for convertido.")
