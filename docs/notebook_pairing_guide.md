# Guia de Emparelhamento Notebook-Script

Este guia explica como configurar e utilizar o emparelhamento entre notebooks Jupyter (.ipynb) e scripts Python (.py) no projeto CIFO_EXTENDED.

## Visão Geral

O emparelhamento notebook-script oferece várias vantagens:

- **Controle de versão eficiente**: Os scripts Python são mais amigáveis para git diff/merge
- **Colaboração melhorada**: Facilita o trabalho em equipe com diferentes ambientes
- **Portabilidade**: Funciona em qualquer ambiente com configuração mínima
- **Reprodutibilidade**: Garante consistência entre execuções e ambientes

## Configuração Inicial

### 1. Instalar Dependências

Certifique-se de que seu ambiente tem as dependências necessárias:

```bash
# Usando conda
conda env create -f environment.yml

# Ou manualmente
pip install jupytext ipykernel
```

### 2. Configurar o Kernel do Projeto

Execute o script de configuração para criar um kernel específico para o projeto:

```bash
python -m notebooks.utils.setup_notebooks
```

Este script:
- Instala pacotes necessários
- Configura o Jupyter para usar Jupytext
- Cria um kernel específico para o projeto com PYTHONPATH correto
- Configura hooks de pre-commit (opcional)

### 3. Emparelhar Notebooks

Para emparelhar todos os notebooks do projeto:

```bash
python -m notebooks.utils.pair_notebooks --all
```

Para emparelhar um notebook específico:

```bash
python -m notebooks.utils.pair_notebooks caminho/para/notebook.ipynb
```

## Uso Diário

### Ao Criar Novos Notebooks

1. Crie o notebook normalmente no Jupyter
2. Execute o script de emparelhamento para configurá-lo:
   ```bash
   python -m notebooks.utils.pair_notebooks caminho/para/novo_notebook.ipynb
   ```
3. Selecione o kernel "Project Kernel" ao abrir o notebook

### Inicialização do Notebook

Adicione esta linha na primeira célula de cada notebook:

```python
%run ../utils/init_notebook.py
```

Isso garante:
- Configuração correta de caminhos para imports
- Auto-recarregamento de módulos
- Criação de diretórios de resultados
- Informações do ambiente

### Edição e Sincronização

- **Edições no notebook (.ipynb)**: Salvas automaticamente no arquivo .py
- **Edições no script (.py)**: Requer reinicialização do kernel para ver no notebook

## Resolução de Problemas

### Kernel Não Encontrado

Se o kernel "Project Kernel" não estiver disponível:

1. Execute novamente o script de configuração:
   ```bash
   python -m notebooks.utils.setup_notebooks
   ```
2. Verifique se o kernel foi criado:
   ```bash
   jupyter kernelspec list
   ```

### Imports Não Funcionam

Se os imports de módulos do projeto não funcionarem:

1. Verifique se a primeira célula executa o script de inicialização
2. Confirme que o projeto tem a estrutura esperada (src/, data/)
3. Execute manualmente:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path.cwd().parent))
   ```

### Emparelhamento Quebrado

Se o emparelhamento entre .ipynb e .py parar de funcionar:

1. Verifique se Jupytext está instalado:
   ```bash
   pip install jupytext
   ```
2. Reaplique o emparelhamento:
   ```bash
   python -m notebooks.utils.pair_notebooks caminho/para/notebook.ipynb
   ```

## Melhores Práticas

1. **Sempre use o kernel do projeto**: Garante consistência de ambiente
2. **Execute o script de inicialização**: Primeira célula de cada notebook
3. **Mantenha ambos .ipynb e .py no controle de versão**: Facilita colaboração
4. **Reinicie o kernel após editar .py**: Para ver as mudanças no notebook
5. **Use caminhos relativos**: Melhora a portabilidade entre ambientes

## Referências

- [Documentação Jupytext](https://jupytext.readthedocs.io/)
- [Jupyter Notebook Best Practices](https://jupyter.org/jupyter-book/guide/advanced/advanced.html)
