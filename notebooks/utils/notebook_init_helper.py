"""
notebook_init_helper.py - Código auxiliar para inicialização de notebooks

Este módulo fornece um snippet de código para ser colocado no topo de cada notebook
para garantir que o módulo init_notebook.py possa ser importado corretamente,
independentemente da localização do notebook na estrutura de diretórios.
"""

INIT_SNIPPET = '''
import sys
import os

# Obtém o caminho absoluto para o diretório utils
utils_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'utils'))
if utils_path not in sys.path:
    sys.path.append(utils_path)

# Agora importa o módulo de inicialização
from init_notebook import *

print("✅ Ambiente do notebook inicializado com sucesso!")
'''

def print_init_snippet():
    """Imprime o snippet de inicialização para ser copiado para notebooks."""
    print(INIT_SNIPPET)
    
if __name__ == "__main__":
    print_init_snippet()
