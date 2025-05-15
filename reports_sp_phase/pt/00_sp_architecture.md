# Documento de Arquitetura da Fase de Processador Único (SP)

**Versão:** 1.0
**Data:** 15 de maio de 2025

## 1. Introdução

Este documento descreve a arquitetura de software implementada para a fase de processador único (SP) do projeto CIFO EXTENDED. O objetivo desta fase foi estabelecer uma base sólida para a implementação e teste dos algoritmos meta-heurísticos (Hill Climbing, Simulated Annealing e Algoritmos Genéticos) para o problema de otimização da formação de equipas desportivas.

## 2. Visão Geral da Arquitetura

A arquitetura da fase SP foi concebida de forma modular, procurando separar as diferentes responsabilidades do sistema em componentes distintos para facilitar o desenvolvimento, a manutenção e a testabilidade. A interação entre os módulos é orquestrada por um script principal.

Os principais componentes da arquitetura são:

*   **Módulo de Solução (`solution.py`):** Responsável pela representação de uma solução candidata, cálculo da sua qualidade (fitness) e validação das restrições do problema.
*   **Módulo de Evolução (`evolution.py`):** Contém a lógica central dos algoritmos meta-heurísticos implementados (Hill Climbing, Simulated Annealing, Algoritmo Genético).
*   **Módulo de Operadores (`operators.py`):** Fornece os operadores genéticos específicos (seleção, cruzamento, mutação) utilizados pelo Algoritmo Genético.
*   **Script Principal (`main_script_sp.py`):** Ponto de entrada da aplicação, responsável por carregar os dados de entrada, configurar os parâmetros dos algoritmos, instanciar e executar os algoritmos, e apresentar/guardar os resultados.
*   **Dados de Entrada:** Ficheiro CSV contendo os dados dos jogadores (habilidade, posição, salário, etc.).

## 3. Detalhamento dos Componentes

### 3.1. Módulo de Solução (`solution.py`)

Este módulo encapsula toda a lógica relacionada com uma solução individual para o problema da formação de equipas.

*   **Classe `LeagueSolution`:**
    *   **Representação Interna:** Uma solução é representada como uma atribuição de jogadores a equipas. Inicialmente, uma lista onde o índice representa o ID do jogador e o valor o ID da equipa. Posteriormente, otimizações podem levar ao uso de arrays NumPy para eficiência.
    *   **Funcionalidades Principais:**
        *   `__init__(self, player_data, num_teams, team_size, budget_per_team, initial_assignment=None)`: Construtor que inicializa uma solução, podendo gerar uma atribuição aleatória válida ou utilizar uma fornecida.
        *   `calculate_fitness(self)`: Calcula a função objetivo, tipicamente a minimização da variância da média de habilidade entre as equipas.
        *   `is_valid(self)`: Verifica se a solução atual satisfaz todas as restrições do problema (orçamento por equipa, número de jogadores por equipa, restrições posicionais).
        *   `get_random_neighbor(self)`: Gera uma solução vizinha através de uma pequena modificação na solução atual (ex: trocar um jogador de equipa).
        *   Outros métodos auxiliares para manipulação e avaliação da solução.

### 3.2. Módulo de Evolução (`evolution.py`)

Este módulo contém as implementações dos algoritmos de otimização.

*   **Função `hill_climbing(initial_solution, max_iterations, player_data, num_teams, team_size, budget_per_team)`:**
    *   Implementa o algoritmo Hill Climbing. Começa com uma solução inicial e tenta iterativamente encontrar melhores vizinhos até que um máximo local seja atingido ou o número máximo de iterações seja alcançado.
*   **Função `simulated_annealing(initial_solution, max_iterations, initial_temperature, cooling_rate, player_data, num_teams, team_size, budget_per_team)`:**
    *   Implementa o algoritmo Simulated Annealing. Permite movimentos para soluções piores com uma probabilidade que diminui ao longo do tempo (temperatura), ajudando a escapar de ótimos locais.
*   **Função `genetic_algorithm(player_data, num_teams, team_size, budget_per_team, population_size, num_generations, tournament_size, crossover_rate, mutation_rate, elite_size, selection_operator, crossover_operator, mutation_operator)`:**
    *   Implementa o Algoritmo Genético. Mantém uma população de soluções que evolui através de operadores de seleção, cruzamento e mutação ao longo de várias gerações.

### 3.3. Módulo de Operadores (`operators.py`)

Este módulo fornece as funções para os operadores genéticos.

*   **Operadores de Seleção:**
    *   `selection_tournament(population, fitnesses, tournament_size)`
    *   `selection_roulette_wheel(population, fitnesses)`
    *   `selection_ranking(population, fitnesses)`
    *   `selection_boltzmann(population, fitnesses, temperature)`
*   **Operadores de Cruzamento:**
    *   `crossover_one_point(parent1, parent2, player_data, num_teams, team_size, budget_per_team)`
    *   `crossover_uniform(parent1, parent2, player_data, num_teams, team_size, budget_per_team)`
    *   Versões `_prefer_valid` que tentam gerar descendentes válidos diretamente.
*   **Operadores de Mutação:**
    *   `mutation_random_player_team_change(solution, player_data, num_teams, team_size, budget_per_team)`
    *   `mutation_swap_players_between_teams(solution, player_data, num_teams, team_size, budget_per_team)`

### 3.4. Script Principal (`main_script_sp.py`)

Este script é o orquestrador da fase SP.

*   **Carregamento de Dados:** Lê os dados dos jogadores do ficheiro CSV (ex: `players.csv`).
*   **Configuração:** Define os parâmetros para cada algoritmo (ex: número de iterações, tamanho da população, taxas de mutação/cruzamento, etc.).
*   **Execução dos Algoritmos:**
    *   Instancia `LeagueSolution` para gerar soluções iniciais.
    *   Chama as funções dos algoritmos do módulo `evolution.py`, passando os parâmetros e operadores configurados.
*   **Recolha e Apresentação de Resultados:**
    *   Guarda a melhor solução encontrada, o seu fitness, e o tempo de execução.
    *   Imprime os resultados no console ou guarda-os em ficheiros de log/sumário.
*   **Profiling (Opcional):** Pode incluir código para profiling (ex: `cProfile`) para identificar gargalos de desempenho.

## 4. Fluxo de Dados

1.  O `main_script_sp.py` carrega os dados dos jogadores.
2.  Para cada algoritmo a ser executado:
    a.  Uma solução inicial é criada (ou uma população de soluções para AGs) usando `LeagueSolution`.
    b.  O `main_script_sp.py` invoca a função do algoritmo correspondente em `evolution.py`.
    c.  Os algoritmos em `evolution.py` utilizam `LeagueSolution` para avaliar fitness, validar soluções e gerar vizinhos/descendentes.
    d.  O AG utiliza operadores de `operators.py` para seleção, cruzamento e mutação.
    e.  A melhor solução e as métricas de desempenho são retornadas ao `main_script_sp.py`.
3.  O `main_script_sp.py` apresenta ou guarda os resultados.

## 5. Decisões de Design e Justificativas

*   **Modularidade:** A separação em `solution.py`, `evolution.py`, e `operators.py` permite que cada componente seja desenvolvido e testado de forma mais independente. Facilita também a adição de novos algoritmos ou operadores.
*   **Abstração da Solução:** A classe `LeagueSolution` abstrai os detalhes da representação da solução e das suas operações, tornando o código dos algoritmos mais limpo.
*   **Configurabilidade:** O `main_script_sp.py` centraliza a configuração dos algoritmos, permitindo fácil experimentação com diferentes parâmetros.
*   **Foco na Clareza Inicial:** A primeira versão da arquitetura priorizou a clareza e a correção funcional sobre a otimização prematura. Otimizações (como vetorização e redução de `deepcopy`) foram aplicadas posteriormente, após profiling.

## 6. Limitações da Arquitetura SP

*   **Processamento Sequencial:** Por definição, esta arquitetura executa os algoritmos e as suas repetições de forma sequencial, o que pode ser demorado para análises extensivas.
*   **Escalabilidade:** A gestão de múltiplas execuções e a agregação de resultados para análise estatística são feitas manualmente ou com scripts simples, o que não escala bem para um grande número de experiências.

Esta arquitetura serviu como base para as fases subsequentes do projeto, onde o multiprocessamento foi introduzido para endereçar estas limitações.

