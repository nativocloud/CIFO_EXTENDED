# Guia de Configuração do Ambiente para CIFO_EXTENDED

Este guia explica como configurar o ambiente de desenvolvimento para o projeto CIFO_EXTENDED, garantindo que todas as dependências necessárias estejam instaladas e configuradas corretamente.

## Visão Geral

O projeto CIFO_EXTENDED utiliza três arquivos principais para gerenciar dependências:

1. **environment.yml**: Define o ambiente Conda principal com dependências essenciais
2. **requirements-dev.txt**: Lista dependências adicionais para desenvolvimento
3. **setup.sh**: Script de automação para configuração do ambiente Conda

## Configuração Rápida (Recomendada)

### 1. Configuração do Ambiente Conda

Execute o script de configuração:

```bash
./setup.sh
```

Este script:
- Verifica se o Conda está instalado
- Cria um ambiente Conda chamado `cifo_extended` com as dependências básicas
- Fornece instruções para ativação do ambiente

### 2. Instalação de Dependências de Desenvolvimento

Após ativar o ambiente Conda, instale as dependências de desenvolvimento:

```bash
conda activate cifo_extended
pip install -r requirements-dev.txt
```

### 3. Configuração do Kernel e Emparelhamento

Configure o kernel do projeto e o emparelhamento notebook-script:

```bash
python -m notebooks.utils.setup_notebooks
python -m notebooks.utils.pair_notebooks --all
```

## Configuração Manual

Se preferir configurar manualmente ou se encontrar problemas com o script automático:

### 1. Criar Ambiente Conda

```bash
conda env create -f environment.yml
conda activate cifo_extended
```

### 2. Instalar Dependências de Desenvolvimento

```bash
pip install -r requirements-dev.txt
```

### 3. Configurar Kernel do Projeto

```bash
python -m ipykernel install --user --name=project_kernel --display-name="Project Kernel"
```

### 4. Emparelhar Notebooks

```bash
python -m notebooks.utils.pair_notebooks --all
```

## Detalhes dos Arquivos de Dependências

### environment.yml

Contém as dependências principais do projeto:
- Python 3.11
- pandas, numpy, matplotlib (análise de dados)
- jupyterlab (interface de notebook)
- jupytext (emparelhamento notebook-script)
- ipykernel (suporte a kernels Jupyter)

Use este arquivo para configurar o ambiente Conda básico.

### requirements-dev.txt

Contém dependências adicionais para desenvolvimento:
- Ferramentas de qualidade de código (black, flake8, isort)
- Ferramentas de teste (pytest, pytest-cov)
- Integração com controle de versão (pre-commit, nbdime)
- Extensões Jupyter e documentação

Use este arquivo para instalar ferramentas de desenvolvimento após configurar o ambiente Conda.

## Verificação da Instalação

Para verificar se o ambiente está configurado corretamente:

```bash
# Verificar ambiente Conda
conda info --envs

# Verificar kernel do projeto
jupyter kernelspec list

# Verificar emparelhamento notebook-script
python -c "import jupytext; print(f'Jupytext version: {jupytext.__version__}')"
```

## Resolução de Problemas

### Conflitos de Dependências

Se encontrar conflitos de dependências:

```bash
# Remover ambiente e recriar
conda remove --name cifo_extended --all
./setup.sh
```

### Kernel Não Encontrado

Se o kernel "Project Kernel" não estiver disponível:

```bash
# Reinstalar kernel
python -m notebooks.utils.setup_notebooks
```

### Jupytext Não Funciona

Se o emparelhamento notebook-script não funcionar:

```bash
# Reinstalar jupytext
pip install --force-reinstall jupytext
python -m notebooks.utils.pair_notebooks --all
```

## Melhores Práticas

1. **Sempre ative o ambiente antes de trabalhar**: `conda activate cifo_extended`
2. **Mantenha environment.yml atualizado**: Adicione novas dependências principais aqui
3. **Use requirements-dev.txt para ferramentas de desenvolvimento**: Mantenha separado do ambiente principal
4. **Execute setup_notebooks.py após alterações no ambiente**: Garante configuração correta do kernel
5. **Documente novas dependências**: Atualize este guia ao adicionar dependências significativas

## Referências

- [Documentação Conda](https://docs.conda.io/)
- [Documentação Jupytext](https://jupytext.readthedocs.io/)
- [Guia de Emparelhamento Notebook-Script](notebook_pairing_guide.md)
- [Guia de Gestão de Kernel](kernel_management_guide.md)
