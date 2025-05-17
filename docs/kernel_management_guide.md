# Guia de Gestão de Kernel para o Projeto CIFO_EXTENDED

Este guia explica como configurar e gerir o kernel Jupyter para o projeto CIFO_EXTENDED, garantindo compatibilidade entre diferentes ambientes.

## Visão Geral

O kernel Jupyter é essencial para a execução correta dos notebooks do projeto. Um kernel bem configurado garante:

- **Acesso aos módulos do projeto**: Através da configuração correta do PYTHONPATH
- **Consistência entre ambientes**: Mesmo comportamento em diferentes máquinas
- **Dependências corretas**: Acesso a todas as bibliotecas necessárias
- **Emparelhamento notebook-script funcional**: Compatibilidade com Jupytext

## Configuração do Kernel do Projeto

### Método Automático (Recomendado)

Execute o script de configuração:

```bash
python -m notebooks.utils.setup_notebooks
```

Este script:
1. Cria um kernel chamado "cifo_project_kernel"
2. Configura o PYTHONPATH para incluir a raiz do projeto
3. Define o nome de exibição como "CIFO Project Kernel"
4. Instala dependências necessárias

### Método Manual

Se precisar configurar manualmente:

1. Instale o pacote ipykernel:
   ```bash
   pip install ipykernel
   ```

2. Crie um kernel específico para o projeto:
   ```bash
   python -m ipykernel install --user --name=project_kernel --display-name="Project Kernel"
   ```

3. Configure o PYTHONPATH manualmente:
   - Localize o diretório do kernel:
     ```bash
     jupyter kernelspec list
     ```
   - Edite o arquivo `kernel.json` para adicionar a variável de ambiente PYTHONPATH

## Uso do Kernel

### Em Notebooks Novos

Ao criar um novo notebook:
1. Selecione o kernel "Project Kernel" no menu suspenso
2. Verifique o nome do kernel no canto superior direito

### Em Notebooks Existentes

Para notebooks existentes:
1. Use o menu "Kernel > Change kernel" para selecionar "Project Kernel"
2. Ou execute o script de emparelhamento para configurar automaticamente:
   ```bash
   python -m notebooks.utils.pair_notebooks caminho/para/notebook.ipynb
   ```

## Resolução de Problemas

### Kernel Não Aparece na Lista

Se o kernel "Project Kernel" não aparecer:

1. Verifique se foi instalado:
   ```bash
   jupyter kernelspec list
   ```

2. Se não estiver listado, reinstale:
   ```bash
   python -m notebooks.utils.setup_notebooks
   ```

3. Reinicie o servidor Jupyter:
   ```bash
   jupyter notebook stop
   jupyter notebook
   ```

### Imports Falham Mesmo com o Kernel Correto

Se os imports de módulos do projeto falharem:

1. Verifique se o PYTHONPATH está configurado corretamente:
   ```python
   import sys
   print(sys.path)
   ```

2. A raiz do projeto deve estar na lista. Se não estiver:
   ```python
   import sys
   from pathlib import Path
   project_root = Path.cwd().parent  # Ajuste conforme necessário
   sys.path.insert(0, str(project_root))
   ```

3. Execute o script de inicialização:
   ```python
   %run ../utils/init_notebook.py
   ```

### Kernel Muda ao Converter entre .ipynb e .py

Se o kernel mudar durante a conversão:

1. Verifique o metadata do notebook:
   ```bash
   jupyter nbconvert --to script --stdout notebook.ipynb | head -n 20
   ```

2. Reaplique o emparelhamento:
   ```bash
   python -m notebooks.utils.pair_notebooks notebook.ipynb
   ```

## Melhores Práticas

1. **Use sempre o kernel do projeto**: Evite o kernel Python padrão
2. **Não altere manualmente o kernelspec**: Use os scripts fornecidos
3. **Verifique o kernel antes de executar**: Confirme no canto superior direito
4. **Reinicie o kernel após alterações de ambiente**: Para garantir consistência
5. **Mantenha o ambiente atualizado**: Sincronize com environment.yml

## Referências

- [Documentação Jupyter Kernels](https://jupyter-client.readthedocs.io/en/stable/kernels.html)
- [IPython Kernel Documentation](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)
