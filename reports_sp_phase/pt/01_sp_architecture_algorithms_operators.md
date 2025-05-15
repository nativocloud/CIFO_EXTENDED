# Relatório da Fase de Processador Único - Otimização de Algoritmos Meta-heurísticos para Seleção de Equipas

**Equipa CIFO**

**Data:** 15 de maio de 2025

## 1. Introdução

Este documento detalha a primeira fase do projeto desenvolvido pela equipa CIFO, focada na implementação e análise de algoritmos meta-heurísticos para o problema de otimização de seleção de equipas desportivas num contexto de processador único. O objetivo principal desta fase foi estabelecer uma base sólida, implementando diferentes algoritmos e operadores, e identificar os seus desempenhos e características antes de explorar otimizações mais avançadas e paralelização.

O problema consiste em formar um conjunto de equipas a partir de um leque de jogadores disponíveis, respeitando um conjunto de restrições (orçamento, número de jogadores por equipa, requisitos posicionais) e visando otimizar um determinado critério (neste caso, minimizar a variância da média de habilidade entre as equipas formadas).

Este relatório descreverá a arquitetura inicial do código, a representação da solução, as estruturas de dados escolhidas, os algoritmos implementados (Hill Climbing, Simulated Annealing e Algoritmos Genéticos com diversos operadores), o racional por detrás das escolhas de design e os resultados iniciais de desempenho.

## 2. Arquitetura do Código de Processador Único (Versão Inicial)

A arquitetura do código para a fase de processador único foi concebida de forma modular para facilitar a implementação, teste e substituição de diferentes componentes dos algoritmos meta-heurísticos. Os principais ficheiros Python envolvidos nesta arquitetura são:

*   `solution.py`: Define a representação de uma solução para o problema (uma atribuição de jogadores a equipas) e os métodos para avaliar a sua validade e qualidade (fitness). Inclui também métodos para gerar soluções vizinhas ou modificar soluções existentes.
*   `evolution.py`: Contém as implementações dos principais algoritmos meta-heurísticos: Hill Climbing, Simulated Annealing e o Algoritmo Genético genérico.
*   `operators.py`: Implementa os diferentes operadores genéticos utilizados pelo Algoritmo Genético, tais como operadores de seleção, cruzamento e mutação.
*   `main_script_sp.py`: O script principal que orquestra a execução dos algoritmos, carrega os dados dos jogadores, configura os parâmetros dos algoritmos e recolhe/apresenta os resultados.

### 2.1. Representação da Solução e Estruturas de Dados

O racional por detrás da escolha da representação da solução e das estruturas de dados foi a simplicidade e a eficiência para as operações necessárias.

*   **Representação da Solução (`LeagueSolution` em `solution.py`):** Uma solução é representada como uma lista (ou, posteriormente, um array NumPy) onde o índice corresponde ao ID de um jogador e o valor nesse índice corresponde ao ID da equipa à qual esse jogador foi atribuído. Por exemplo, `assignment[player_id] = team_id`.
    *   **Racional:** Esta representação é direta, fácil de manipular e permite um acesso rápido à equipa de qualquer jogador. É também compacta.

*   **Dados dos Jogadores:** Os dados dos jogadores (ID, nome, posição, salário, habilidade) são inicialmente carregados a partir de um ficheiro CSV para uma estrutura de dados conveniente (como uma lista de dicionários ou um DataFrame Pandas) no script principal. Para as operações internas das classes de solução e algoritmos, estes dados são frequentemente passados ou convertidos para formatos mais otimizados para cálculo (como arrays NumPy para salários, habilidades e posições numéricas, como foi feito na fase de vetorização).
    *   **Racional:** O CSV é um formato comum e fácil de usar para entrada de dados. Internamente, a conversão para arrays NumPy (especialmente após a vetorização) visa acelerar os cálculos numéricos.

*   **Estrutura das Equipas:** Implicitamente, as equipas são coleções de jogadores. A representação da solução permite reconstruir facilmente a composição de cada equipa, filtrando os jogadores com base no `team_id` que lhes foi atribuído.

### 2.2. Algoritmos Implementados (Versão Inicial, Não Otimizada)

Foram selecionados três tipos principais de algoritmos meta-heurísticos para esta fase inicial, devido à sua popularidade e diferentes abordagens à exploração do espaço de soluções:

1.  **Hill Climbing (HC):**
    *   **Descrição:** Um algoritmo de busca local iterativo que começa com uma solução arbitrária e tenta encontrar uma solução melhor movendo-se para um vizinho com melhor fitness. Termina quando atinge um ótimo local, onde nenhum vizinho tem um fitness melhor.
    *   **Racional da Seleção:** É um dos algoritmos de otimização mais simples de implementar e entender. Serve como uma boa linha de base para comparar com algoritmos mais complexos. É rápido, mas propenso a ficar preso em ótimos locais.

2.  **Simulated Annealing (SA):**
    *   **Descrição:** Uma técnica de otimização probabilística inspirada no processo de recozimento em metalurgia. Permite movimentos para soluções piores com uma certa probabilidade, que diminui à medida que o algoritmo progride (a "temperatura" baixa). Isto ajuda a escapar de ótimos locais.
    *   **Racional da Seleção:** Oferece uma melhor capacidade de exploração global do que o Hill Climbing, com a capacidade de evitar ótimos locais. A sua eficácia depende da correta parametrização do esquema de arrefecimento.

3.  **Algoritmos Genéticos (AGs):**
    *   **Descrição:** Algoritmos de busca inspirados na evolução biológica. Mantêm uma população de soluções candidatas que evoluem ao longo de gerações através de operadores genéticos como seleção, cruzamento (recombinação) e mutação.
    *   **Racional da Seleção:** São poderosos para problemas de otimização complexos e podem explorar eficazmente grandes espaços de soluções. A sua natureza baseada em população permite manter diversidade e explorar múltiplas regiões do espaço de busca simultaneamente.

### 2.3. Seleção de Operadores para Teste (Algoritmos Genéticos)

Para os Algoritmos Genéticos, foi implementada uma variedade de operadores para permitir testar diferentes estratégias evolutivas:

*   **Operadores de Seleção:** Determinam quais indivíduos da população atual são escolhidos para se reproduzirem.
    *   `selection_tournament`: Seleção por torneio. Vários indivíduos são escolhidos aleatoriamente da população, e o melhor entre eles é selecionado.
        *   **Racional:** Simples de implementar, eficiente e permite controlar a pressão seletiva através do tamanho do torneio.
    *   `selection_roulette_wheel`: Seleção por roleta. A probabilidade de um indivíduo ser selecionado é proporcional ao seu fitness.
        *   **Racional:** Um método clássico que dá uma hipótese a todos os indivíduos, mas favorece os mais aptos.
    *   `selection_ranking`: Seleção baseada no ranking. Os indivíduos são ordenados pelo seu fitness, e a probabilidade de seleção é baseada no seu ranking, não no valor absoluto do fitness.
        *   **Racional:** Evita problemas de convergência prematura que podem ocorrer com a roleta se houver indivíduos com fitness muito superior aos restantes.
    *   `selection_boltzmann` (implementado, mas pode necessitar de ajuste de temperatura para ser eficaz):
        *   **Racional:** Inspirado na exploração de Boltzmann, ajusta a pressão seletiva ao longo do tempo.

*   **Operadores de Cruzamento (Crossover):** Combinam o material genético de dois pais para criar um ou mais filhos.
    *   `crossover_one_point`: Cruzamento de um ponto. Um ponto de corte é escolhido aleatoriamente, e os segmentos dos pais são trocados para formar os filhos.
    *   `crossover_uniform`: Cruzamento uniforme. Para cada gene (posição no array de atribuição), decide-se aleatoriamente de qual pai o filho herdará esse gene.
    *   Versões `_prefer_valid`: Foram também exploradas variantes destes operadores que tentam gerar filhos válidos mais diretamente ou que têm mecanismos para reparar filhos inválidos (embora a validação principal ocorra após a geração do filho).
        *   **Racional da Variedade:** Diferentes operadores de cruzamento exploram o espaço de soluções de maneiras distintas. O cruzamento de um ponto tende a preservar blocos de genes, enquanto o uniforme promove uma maior mistura.

*   **Operadores de Mutação:** Introduzem pequenas alterações aleatórias nos indivíduos para manter a diversidade genética e permitir a exploração de novas áreas do espaço de soluções.
    *   `mutation_random_player_team_change`: Muda aleatoriamente a equipa de um jogador selecionado aleatoriamente.
    *   `mutation_swap_players_between_teams`: Troca dois jogadores entre duas equipas diferentes.
    *   `mutation_swap_players_different_teams_specific_roles` (mais complexo, pode ter sido simplificado ou focado em trocas mais genéricas inicialmente): Tenta trocar jogadores de papéis específicos entre equipas.
        *   **Racional da Variedade:** Diferentes tipos de mutação permitem diferentes tipos de "saltos" no espaço de soluções. Mutações simples são menos disruptivas, enquanto mutações mais complexas podem permitir escapar de ótimos locais mais facilmente, mas também podem ser mais difíceis de garantir que resultem em soluções válidas rapidamente.

A seleção inicial destes algoritmos e operadores pela equipa CIFO visou cobrir um espectro de abordagens meta-heurísticas, desde as mais simples (HC) até às mais complexas e baseadas em população (AGs com múltiplos operadores), para avaliar a sua adequação e desempenho no problema específico da formação de equipas.




## 3. Resultados das Otimizações Iniciais (Processador Único)

Após a implementação inicial dos algoritmos, a equipa CIFO procedeu a uma fase de profiling e otimização do código de processador único. O objetivo era identificar e mitigar os principais gargalos de desempenho antes de considerar a paralelização. As duas principais áreas de otimização focadas foram a redução do uso de `deepcopy` e a vetorização de funções críticas.

### 3.1. Profiling Inicial e Identificação de Gargalos

A execução do script `main_script_sp_timing_estimate.py` (configurado para 1 execução por algoritmo e 1 geração para os Algoritmos Genéticos) revelou que o tempo total de execução era de aproximadamente **85.66 segundos**. A análise de profiling com `cProfile` destacou os seguintes pontos como os maiores consumidores de tempo:

*   **Simulated Annealing (SA):** Era o algoritmo individualmente mais lento, consumindo cerca de **73.17 segundos** (aproximadamente 85% do tempo total do script).
*   **`copy.deepcopy`:** Esta operação foi identificada como a principal causa da lentidão, especialmente dentro do Simulated Annealing. Foi chamada mais de 23 milhões de vezes, totalizando **53.49 segundos** do tempo de execução.
*   **Funções de Solução:** As funções `solution.py:178(get_random_neighbor)` (usada pelo SA), `solution.py:122(fitness)`, e `solution.py:85(is_valid)` também consumiam parcelas significativas do tempo, principalmente devido ao elevado número de chamadas no SA e, no caso de `get_random_neighbor`, ao uso interno de `deepcopy`.

### 3.2. Otimização do Uso de `deepcopy` no Simulated Annealing

Dado o impacto massivo de `deepcopy` no desempenho do SA, a primeira otimização implementada pela equipa CIFO visou reduzir a sua utilização.

*   **Alteração Implementada:** Na função `simulated_annealing` (em `evolution.py`), as chamadas a `deepcopy(neighbor_solution)` para atualizar a `current_solution` (tanto quando uma solução melhor era encontrada, como quando uma pior era aceite probabilisticamente) foram substituídas por atribuições diretas (`current_solution = neighbor_solution`).
*   **Racional:** Esta alteração foi considerada segura porque a função `get_random_neighbor` (em `solution.py`) já criava e retornava um *novo objeto de solução* (vizinho) que era uma instância independente. A chamada a `deepcopy` para atualizar a `best_solution` global foi mantida para garantir a sua integridade.
*   **Impacto:** Esta otimização resultou numa melhoria de desempenho muito significativa:
    *   O tempo total do script foi reduzido de 85.66 segundos para **41.42 segundos** (uma melhoria de aproximadamente 51.7%).
    *   O tempo específico da função `simulated_annealing` diminuiu de 73.17 segundos para **28.91 segundos** (uma melhoria de aproximadamente 60.5%).
    *   O número total de chamadas a `deepcopy` no script diminuiu drasticamente.
    *   A qualidade da solução (fitness) alcançada pelos algoritmos manteve-se a mesma.

### 3.3. Vetorização das Funções `is_valid()` e `fitness()`

A segunda tentativa de otimização focou-se na vetorização das funções `is_valid()` e `fitness()` em `solution.py`, utilizando NumPy para substituir loops Python por operações de array mais eficientes.

*   **Alterações Implementadas:**
    1.  **Pré-processamento de Dados:** No construtor da classe `LeagueSolution`, os dados dos jogadores (salários, habilidades, posições) foram convertidos e armazenados como arrays NumPy.
    2.  **Reescrita de `is_valid()`:** As verificações de tamanho de equipa, orçamento e requisitos posicionais foram reescritas para usar operações NumPy como `np.bincount`, máscaras booleanas e `np.array_equal`.
    3.  **Reescrita de `fitness()`:** O cálculo da média de habilidades por equipa foi otimizado usando `np.bincount` com o argumento `weights`, e o desvio padrão foi calculado com `np.std`.
*   **Racional:** A intenção era acelerar estas funções, que são chamadas frequentemente, aproveitando a eficiência do NumPy para operações numéricas em arrays.
*   **Impacto:** Após a implementação da vetorização e as correções necessárias nos scripts `evolution.py`, `operators.py` e `main_script_sp_timing_estimate.py` para compatibilidade, a execução completa do script (com as otimizações de `deepcopy` no SA e a vetorização) resultou num tempo total de aproximadamente **40.96 segundos**.
    *   Comparando com o tempo após a otimização do `deepcopy` (41.42s), a vetorização, no contexto atual de chamadas (uma solução de cada vez para estas funções na maioria dos casos), não trouxe uma melhoria adicional significativa no tempo de execução global. As funções `is_valid` e `fitness` em si tornaram-se mais rápidas isoladamente, mas o seu contributo para o tempo total já tinha sido reduzido pela diminuição do número de chamadas ou pela predominância de outros fatores no SA.
    *   No entanto, a implementação vetorizada é considerada mais robusta e potencialmente mais escalável se o número de jogadores ou equipas aumentasse significativamente, ou se os algoritmos fossem adaptados para avaliar lotes de soluções.

Estes resultados da fase de processador único forneceram informações valiosas sobre o comportamento dos algoritmos e a eficácia das estratégias de otimização, preparando o terreno para a exploração do multiprocessamento.




## 4. Conclusões da Fase de Processador Único e Próximos Passos

A fase de implementação e otimização em processador único, conduzida pela equipa CIFO, foi crucial para estabelecer uma linha de base sólida para o problema de seleção de equipas. A implementação inicial dos algoritmos Hill Climbing, Simulated Annealing e Algoritmos Genéticos com diversos operadores permitiu uma compreensão aprofundada do seu comportamento e dos desafios computacionais inerentes.

A análise de profiling revelou-se uma ferramenta indispensável, identificando o uso excessivo de `deepcopy` no Simulated Annealing como o principal gargalo de desempenho. A otimização direcionada a esta questão resultou numa redução significativa do tempo de execução (aproximadamente 51.7% no tempo total do script), sem comprometer a qualidade das soluções encontradas. A subsequente vetorização das funções `is_valid()` e `fitness()`, embora não tenha proporcionado ganhos de tempo adicionais substanciais neste contexto específico de chamadas, modernizou a base de código e preparou-a para cenários potencialmente mais exigentes.

Os resultados desta fase indicam que, mesmo com otimizações, o Simulated Annealing permanece o algoritmo mais demorado, principalmente devido à sua natureza iterativa intensiva e às operações ainda dispendiosas na geração de vizinhos. Os Algoritmos Genéticos, mesmo com apenas uma geração, mostraram-se relativamente rápidos, e o Hill Climbing, como esperado, foi o mais rápido, embora com maior risco de ficar preso em ótimos locais.

Com esta base de código de processador único otimizada e bem compreendida, a equipa CIFO está agora preparada para avançar para as próximas fases do projeto, que incluem a documentação detalhada do racional da vetorização, a implementação e análise do multiprocessamento para explorar ganhos de desempenho através da paralelização, e a subsequente seleção e teste exaustivo de algoritmos promissores com diferentes configurações.

As aprendizagens desta fase serão fundamentais para guiar as decisões nas etapas futuras, visando alcançar soluções de alta qualidade de forma eficiente.

