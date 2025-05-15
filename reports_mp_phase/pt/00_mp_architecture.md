# Documento de Arquitetura da Fase de Multiprocessamento (MP)

**Versão:** 1.0
**Data:** 15 de maio de 2025

## 1. Introdução

Este documento descreve a arquitetura de software implementada para a fase de multiprocessamento (MP) do projeto CIFO EXTENDED. O objetivo principal desta fase foi estender a arquitetura da fase de processador único (SP) para permitir a execução paralela de múltiplas instâncias dos algoritmos meta-heurísticos, visando obter resultados estatisticamente mais robustos e acelerar o processo de experimentação.

## 2. Visão Geral da Arquitetura MP

A arquitetura da fase MP baseia-se nos módulos desenvolvidos na fase SP (`solution.py`, `evolution.py`, `operators.py`), introduzindo um novo script principal (`main_script_mp.py` e suas variantes como `main_script_mp_30_runs.py`, `main_script_mp_param_var.py`, `main_script_mp_final_param_var.py`) que orquestra as execuções paralelas utilizando o módulo `multiprocessing` do Python.

Os principais componentes adicionais ou modificados na arquitetura MP são:

*   **Scripts Principais de Multiprocessamento (ex: `main_script_mp.py`):** Responsáveis por configurar e gerir a execução paralela de múltiplas execuções de um ou mais algoritmos. Utilizam um `Pool` de processos para distribuir o trabalho.
*   **Função de Execução do Trabalhador (Worker Function):** Uma função wrapper que encapsula a lógica de uma única execução de um algoritmo, tornando-a adequada para ser executada por um processo num `Pool`.
*   **Agregação de Resultados:** Mecanismos para recolher e agregar os resultados (fitness, tempo de execução, etc.) de todas as execuções paralelas para análise subsequente.

## 3. Detalhamento dos Componentes MP

### 3.1. Scripts Principais de Multiprocessamento

Estes scripts são o ponto central da fase MP.

*   **Exemplo (`main_script_mp.py`):**
    *   **Configuração dos Algoritmos e Execuções:** Define quais algoritmos e com que parâmetros serão executados, e quantas vezes cada um deve ser repetido (ex: 5 execuções, 30 execuções).
    *   **Criação do `Pool` de Processos:** Utiliza `multiprocessing.Pool()` para criar um conjunto de processos trabalhadores, geralmente correspondendo ao número de núcleos de CPU disponíveis ou a um valor configurável.
    *   **Preparação das Tarefas:** Para cada execução de algoritmo necessária, os parâmetros são empacotados (ex: numa tupla ou dicionário) para serem passados à função do trabalhador.
    *   **Distribuição das Tarefas:** Utiliza métodos como `pool.map()` ou `pool.apply_async()` para submeter as tarefas aos processos do `Pool`.
    *   **Recolha de Resultados:** Aguarda a conclusão de todas as tarefas e recolhe os resultados retornados por cada processo trabalhador.
    *   **Análise e Apresentação dos Resultados Agregados:** Calcula métricas estatísticas (média, desvio padrão, melhor geral) a partir dos resultados individuais e apresenta-os ou guarda-os num ficheiro de sumário (ex: CSV).

### 3.2. Função de Execução do Trabalhador (Worker Function)

Esta função é executada por cada processo no `Pool`.

*   **Estrutura Típica:**
    ```python
    def run_algorithm_worker(params):
        # Desempacotar parâmetros (nome do algoritmo, configuração, dados dos jogadores, etc.)
        algorithm_name, config, player_data, num_teams, ... = params

        # Gerar solução inicial (se necessário)
        initial_solution = LeagueSolution(player_data, ...)

        # Executar o algoritmo (ex: hill_climbing, genetic_algorithm)
        start_time = time.time()
        if algorithm_name == 'HC':
            best_solution_data = evolution.hill_climbing(initial_solution, ...)
        elif algorithm_name == 'GA':
            best_solution_data = evolution.genetic_algorithm(player_data, ...)
        # ... outros algoritmos
        end_time = time.time()

        execution_time = end_time - start_time
        fitness = best_solution_data['best_fitness'] # ou similar

        # Retornar resultados relevantes (fitness, tempo, talvez a melhor solução)
        return (fitness, execution_time, config_name) # Exemplo
    ```
*   **Isolamento:** Cada chamada a esta função é executada num processo separado, garantindo que as execuções são independentes e não partilham memória de forma problemática (a menos que explicitamente gerido com mecanismos de `multiprocessing`).

### 3.3. Módulos da Fase SP (`solution.py`, `evolution.py`, `operators.py`)

Estes módulos são largamente reutilizados sem modificações significativas na sua lógica interna. A principal mudança é como são invocados (através da função do trabalhador em processos separados).

*   **Considerações:**
    *   **Estado Global:** Deve-se ter cuidado para que estes módulos não dependam de estado global que possa ser modificado de forma concorrente e não segura entre processos. Na prática do projeto, a passagem explícita de todos os dados necessários (como `player_data`) para as funções ajuda a mitigar isto.
    *   **Serialização:** Objetos passados entre processos (como a `initial_solution` ou os parâmetros) devem ser serializáveis (picklable) pelo Python.

## 4. Fluxo de Dados na Fase MP

1.  O script principal MP (ex: `main_script_mp.py`) carrega os dados dos jogadores e define as configurações de algoritmos e o número de repetições.
2.  Cria uma lista de tuplas de parâmetros, onde cada tupla contém tudo o que é necessário para uma única execução de algoritmo.
3.  Um `multiprocessing.Pool` é criado.
4.  As tuplas de parâmetros são passadas para a função do trabalhador usando `pool.map()` (ou similar).
5.  Cada processo no `Pool` executa a função do trabalhador com um conjunto de parâmetros:
    a.  A função do trabalhador invoca o algoritmo apropriado de `evolution.py`.
    b.  O algoritmo utiliza `solution.py` e `operators.py` como na fase SP.
    c.  A função do trabalhador retorna o resultado da sua execução (ex: fitness, tempo).
6.  O script principal recolhe a lista de resultados de todas as execuções.
7.  Os resultados são agregados (calculando médias, desvios padrão, etc.) e guardados ou apresentados.

## 5. Decisões de Design e Justificativas para MP

*   **Utilização de `multiprocessing.Pool`:** Fornece uma forma conveniente e de alto nível para gerir um conjunto de processos trabalhadores e distribuir tarefas, abstraindo muitos dos detalhes de baixo nível da gestão de processos.
*   **Função Wrapper para Trabalhador:** Isola a lógica de uma única execução, tornando o código mais limpo e facilitando a paralelização.
*   **Passagem Explícita de Dados:** Minimiza problemas com estado partilhado, pois cada processo trabalhador recebe a sua própria cópia dos dados necessários (ou referências a dados imutáveis).
*   **Reutilização de Código SP:** A arquitetura MP foi construída sobre a base sólida da fase SP, reutilizando os módulos de algoritmos e solução, o que acelerou o desenvolvimento.
*   **Ficheiros de Sumário CSV:** A utilização de ficheiros CSV para guardar os resultados agregados facilita a análise posterior e a geração de gráficos/relatórios.

## 6. Desafios e Considerações na Arquitetura MP

*   **Overhead de Multiprocessamento:** A criação de processos e a comunicação entre processos (serialização/desserialização de dados) introduzem algum overhead. Para tarefas muito curtas, este overhead pode diminuir os ganhos da paralelização.
*   **Consumo de Memória:** Cada processo pode ter a sua própria cópia de alguns dados, o que pode aumentar o consumo total de memória, especialmente para grandes conjuntos de dados ou populações grandes em AGs.
*   **Debugging:** Depurar aplicações multiprocesso pode ser mais complexo do que aplicações sequenciais.
*   **Número Ótimo de Processos:** Determinar o número ideal de processos no `Pool` (geralmente relacionado com o número de núcleos de CPU) pode requerer alguma experimentação para otimizar o desempenho.

## 7. Evolução da Arquitetura MP no Projeto

Ao longo do projeto, a arquitetura MP evoluiu através de diferentes scripts principais para acomodar diferentes fases de experimentação:

*   `main_script_mp.py`: Para as 5 execuções iniciais.
*   `main_script_mp_30_runs.py`: Para as 30 execuções dos algoritmos promissores.
*   `main_script_mp_param_var.py`: Para testar variações de parâmetros dos AGs (com 5 execuções por variação).
*   `main_script_mp_final_param_var.py`: Para a análise final com 30 execuções das variações de parâmetros mais promissoras.

A lógica central de multiprocessamento (uso de `Pool`, função trabalhadora) permaneceu consistente, com variações principalmente na forma como as configurações dos algoritmos e os loops de experimentação eram definidos.

Esta arquitetura permitiu uma exploração muito mais vasta e estatisticamente significativa do espaço de soluções e do comportamento dos algoritmos em comparação com a fase de processador único.

