#!/bin/bash

# Script para configurar o ambiente miniconda para o projeto CIFO_EXTENDED

echo "Iniciando configuração do ambiente para CIFO_EXTENDED..."

# Verificar se o Miniconda/Conda está instalado
if ! command -v conda &> /dev/null; then
    echo "Miniconda/Conda não encontrado. Por favor, instale o Miniconda antes de executar este script."
    echo "Instruções de instalação: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Criar o ambiente a partir do ficheiro environment.yml
echo "Criando ambiente conda a partir do ficheiro environment.yml..."
conda env create -f environment.yml

# Verificar se a criação do ambiente foi bem-sucedida
if [ $? -ne 0 ]; then
    echo "Erro ao criar o ambiente conda. Por favor, verifique o ficheiro environment.yml."
    exit 1
fi

echo "Ambiente 'cifo_extended' criado com sucesso!"
echo "Para ativar o ambiente, execute: conda activate cifo_extended"
echo "Para iniciar o Jupyter Lab, execute: jupyter lab"
