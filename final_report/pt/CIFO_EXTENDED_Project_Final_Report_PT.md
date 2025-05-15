# Relatório Final Abrangente do Projeto CIFO EXTENDED

**Equipa CIFO**

**Data:** 15 de maio de 2025

## Sumário Executivo

Este relatório final consolida todas as fases do projeto CIFO EXTENDED, que visou a otimização da seleção de equipas desportivas através da aplicação e análise de algoritmos meta-heurísticos. O projeto evoluiu desde implementações em processador único, passando por otimizações de código, exploração de multiprocessamento com diferentes números de execuções, até uma análise aprofundada da variação de parâmetros para os algoritmos mais promissores. As principais conclusões apontam para o Hill Climbing (especificamente com 500 iterações) como a abordagem mais eficiente em termos de equilíbrio entre qualidade da solução, consistência e tempo de execução. Para cenários que exigem a garantia da melhor fitness possível com consistência perfeita, uma configuração otimizada do Algoritmo Genético (GA_Config_4 com taxa de mutação de 0.15) demonstrou ser superior, embora com um custo computacional mais elevado. Este documento detalha a metodologia, os resultados e as conclusões de cada fase, culminando com recomendações para trabalhos futuros e um diagrama do processo de análise empregue.

## 1. Introdução Geral ao Projeto

O desafio central do projeto CIFO EXTENDED foi desenvolver e avaliar métodos computacionais eficazes para o problema complexo de formar múltiplas equipas desportivas a partir de um conjunto de jogadores disponíveis. Este problema de otimização combinatória é caracterizado por diversas restrições, incluindo orçamentos por equipa, número de jogadores por equipa e requisitos posicionais específicos. O objetivo principal da otimização foi minimizar a variância da média de habilidade entre as equipas formadas, promovendo assim o equilíbrio competitivo.

Ao longo do projeto, foram explorados diversos algoritmos meta-heurísticos, incluindo Hill Climbing (HC), Simulated Annealing (SA) e Algoritmos Genéticos (AGs) com uma variedade de operadores. A investigação progrediu através de várias fases distintas:

1.  **Fase de Processador Único:** Implementação inicial dos algoritmos, profiling para identificação de gargalos e otimizações de código (redução de `deepcopy` e vetorização).
2.  **Fase de Multiprocessamento - 5 Execuções:** Avaliação inicial do desempenho dos algoritmos em paralelo para obter dados estatisticamente mais robustos e selecionar candidatos para análises mais profundas.
3.  **Fase de Multiprocessamento - 30 Execuções com Algoritmos Promissores:** Análise estatística mais rigorosa dos algoritmos selecionados (HC, SA, e duas configurações de AG) com os seus parâmetros base.
4.  **Fase de Variação de Parâmetros - 30 Execuções por Variação:** Investigação detalhada do impacto de diferentes hiperparâmetros (número de iterações para HC; taxa de mutação, tamanho da população, número de gerações para os AGs selecionados) no desempenho.

Este relatório visa apresentar uma visão consolidada de todo o processo, desde a conceção inicial até às conclusões finais e recomendações.





## 2. Fase de Processador Único: Implementação e Otimizações Iniciais




### 2.1. Arquitetura do Código de Processador Único (Versão Inicial)

A arquitetura do código para a fase de processador único foi concebida de forma modular para facilitar a implementação, teste e substituição de diferentes componentes dos algoritmos meta-heurísticos. Os principais ficheiros Python envolvidos nesta arquitetura são:

*   `solution.py`: Define a representação de uma solução para o problema (uma atribuição de jogadores a equipas) e os métodos para avaliar a sua validade e qualidade (fitness). Inclui também métodos para gerar soluções vizinhas ou modificar soluções existentes.
*   `evolution.py`: Contém as implementações dos principais algoritmos meta-heurísticos: Hill Climbing, Simulated Annealing e o Algoritmo Genético genérico.
*   `operators.py`: Implementa os diferentes operadores genéticos utilizados pelo Algoritmo Genético, tais como operadores de seleção, cruzamento e mutação.
*   `main_script_sp.py`: O script principal que orquestra a execução dos algoritmos, carrega os dados dos jogadores, configura os parâmetros dos algoritmos e recolhe/apresenta os resultados.

#### 2.1.1. Representação da Solução e Estruturas de Dados

O racional por detrás da escolha da representação da solução e das estruturas de dados foi a simplicidade e a eficiência para as operações necessárias.

*   **Representação da Solução (`LeagueSolution` em `solution.py`):** Uma solução é representada como uma lista (ou, posteriormente, um array NumPy) onde o índice corresponde ao ID de um jogador e o valor nesse índice corresponde ao ID da equipa à qual esse jogador foi atribuído. Por exemplo, `assignment[player_id] = team_id`.
    *   **Racional:** Esta representação é direta, fácil de manipular e permite um acesso rápido à equipa de qualquer jogador. É também compacta.

*   **Dados dos Jogadores:** Os dados dos jogadores (ID, nome, posição, salário, habilidade) são inicialmente carregados a partir de um ficheiro CSV para uma estrutura de dados conveniente (como uma lista de dicionários ou um DataFrame Pandas) no script principal. Para as operações internas das classes de solução e algoritmos, estes dados são frequentemente passados ou convertidos para formatos mais otimizados para cálculo (como arrays NumPy para salários, habilidades e posições numéricas, como foi feito na fase de vetorização).
    *   **Racional:** O CSV é um formato comum e fácil de usar para entrada de dados. Internamente, a conversão para arrays NumPy (especialmente após a vetorização) visa acelerar os cálculos numéricos.

*   **Estrutura das Equipas:** Implicitamente, as equipas são coleções de jogadores. A representação da solução permite reconstruir facilmente a composição de cada equipa, filtrando os jogadores com base no `team_id` que lhes foi atribuído.

#### 2.1.2. Algoritmos Implementados (Versão Inicial, Não Otimizada)

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

#### 2.1.3. Seleção de Operadores para Teste (Algoritmos Genéticos)

Para os Algoritmos Genéticos, foi implementada uma variedade de operadores para permitir testar diferentes estratégias evolutivas:

*   **Operadores de Seleção:** Determinam quais indivíduos da população atual são escolhidos para se reproduzirem.
    *   `selection_tournament`: Seleção por torneio.
    *   `selection_roulette_wheel`: Seleção por roleta.
    *   `selection_ranking`: Seleção baseada no ranking.
    *   `selection_boltzmann`.
*   **Operadores de Cruzamento (Crossover):** Combinam o material genético de dois pais para criar um ou mais filhos.
    *   `crossover_one_point`: Cruzamento de um ponto.
    *   `crossover_uniform`: Cruzamento uniforme.
    *   Versões `_prefer_valid` para tentar gerar filhos válidos mais diretamente.
*   **Operadores de Mutação:** Introduzem pequenas alterações aleatórias nos indivíduos.
    *   `mutation_random_player_team_change`: Muda aleatoriamente a equipa de um jogador.
    *   `mutation_swap_players_between_teams`: Troca dois jogadores entre duas equipas.

A seleção inicial destes algoritmos e operadores visou cobrir um espectro de abordagens meta-heurísticas.

### 2.2. Resultados das Otimizações Iniciais (Processador Único)

Após a implementação inicial, procedeu-se a uma fase de profiling e otimização. As duas principais áreas de otimização focadas foram a redução do uso de `deepcopy` e a vetorização de funções críticas.

#### 2.2.1. Profiling Inicial e Identificação de Gargalos

A execução inicial do script (`main_script_sp_timing_estimate.py`) revelou que o tempo total era de aproximadamente **85.66 segundos**. O `cProfile` destacou:

*   **Simulated Annealing (SA):** O mais lento, consumindo cerca de **73.17 segundos**.
*   **`copy.deepcopy`:** Principal causa da lentidão, chamada mais de 23 milhões de vezes, totalizando **53.49 segundos**.
*   **Funções de Solução:** `get_random_neighbor`, `fitness`, e `is_valid` também consumiam tempo significativo.

#### 2.2.2. Otimização do Uso de `deepcopy` no Simulated Annealing

*   **Alteração:** No SA, chamadas a `deepcopy(neighbor_solution)` para atualizar `current_solution` foram substituídas por atribuições diretas. A chamada a `deepcopy` para `best_solution` foi mantida.
*   **Impacto:** Redução do tempo total do script para **41.42 segundos** (melhoria de ~51.7%). Tempo do SA diminuiu para **28.91 segundos** (melhoria de ~60.5%).

#### 2.2.3. Vetorização das Funções `is_valid()` e `fitness()`

*   **Alterações:** Funções `is_valid()` e `fitness()` em `solution.py` foram reescritas usando NumPy.
*   **Impacto:** Tempo total do script com todas as otimizações foi de aproximadamente **40.96 segundos**. A vetorização não trouxe melhoria adicional significativa no tempo global neste contexto, mas tornou o código mais robusto.

### 2.3. Conclusões da Fase de Processador Único

A fase de processador único foi crucial. A otimização de `deepcopy` resultou numa redução significativa do tempo de execução. A vetorização modernizou a base de código, preparando-a para cenários mais exigentes. Estes resultados forneceram informações valiosas para a exploração do multiprocessamento.




## 3. Fase de Multiprocessamento - 5 Execuções: Avaliação Inicial

Esta secção do projeto focou-se na avaliação do desempenho dos algoritmos de otimização (Hill Climbing, Simulated Annealing e quatro configurações de Algoritmos Genéticos) quando executados múltiplas vezes em paralelo. O objetivo principal desta fase foi obter dados estatisticamente mais robustos sobre a eficácia e eficiência de cada abordagem, executando cada algoritmo 5 vezes. A utilização de multiprocessamento permitiu realizar estas execuções concorrentes, otimizando o tempo total de experimentação.

### 3.1. Metodologia Experimental

Os algoritmos avaliados foram: Hill Climbing (HC), Simulated Annealing (SA), e quatro configurações de Algoritmos Genéticos (GA_Config_1, GA_Config_2, GA_Config_3, GA_Config_4), variando operadores de mutação, cruzamento e seleção. Cada algoritmo/configuração foi executado 5 vezes em paralelo. As métricas coletadas incluíram melhor fitness geral, fitness média, desvio padrão da fitness e tempo médio de execução.

### 3.2. Resultados e Discussão (5 Execuções)

Após a execução do script `main_script_mp.py` para 5 execuções de cada algoritmo, os resultados foram:

*   **Melhor Fitness Geral:** Todos os algoritmos (SA, GA_Config_1, GA_Config_2, GA_Config_3, e GA_Config_4) conseguiram alcançar o mesmo valor de Melhor Fitness Geral de aproximadamente **0.057143**. O Hill Climbing, embora tenha atingido este valor, apresentou uma Fitness Média ligeiramente superior (0.0671).
*   **Consistência (Desvio Padrão da Fitness):** O Simulated Annealing e a GA_Config_4 destacaram-se com um Desvio Padrão da Fitness de **0.0**, indicando que convergiram para o mesmo valor ótimo em todas as 5 execuções.
*   **Eficiência (Tempo Médio de Execução):**
    *   Hill Climbing: ~0.47 segundos (mais rápido).
    *   GA_Config_1: ~6.05 segundos.
    *   GA_Config_4: ~8.19 segundos.
    *   Simulated Annealing: ~17.60 segundos.
    *   GA_Config_2 e GA_Config_3: ~61-62 segundos (mais lentas).

### 3.3. Conclusões Preliminares (5 Execuções)

Com base nos resultados de 5 execuções:

*   O **Hill Climbing** mostrou-se extremamente rápido.
*   O **Simulated Annealing** encontrou consistentemente a melhor solução, mas com custo de tempo moderado.
*   A **GA_Config_4** emergiu como uma candidata forte, igualando o SA na melhor fitness e consistência, mas sendo mais rápida.
*   A **GA_Config_1** foi notável pela sua rapidez entre os AGs, embora com menor consistência.

Estes resultados forneceram a base para selecionar algoritmos para a fase de 30 execuções.




## 4. Fase de Multiprocessamento - 30 Execuções com Algoritmos Promissores: Análise Estatística

Esta secção do projeto apresenta uma análise estatística detalhada dos quatro algoritmos identificados como mais promissores na fase anterior de 5 execuções (Hill Climbing, Simulated Annealing, GA_Config_1 e GA_Config_4). Cada algoritmo foi executado 30 vezes em paralelo para obter resultados estatisticamente mais robustos.

### 4.1. Metodologia

Os algoritmos foram executados com os seus parâmetros base (os mesmos da fase de 5 execuções). As métricas analisadas incluíram fitness média, desvio padrão da fitness, melhor fitness geral e tempo médio de execução.

### 4.2. Resultados e Análise Estatística (30 Execuções Iniciais)

*   **Melhor Fitness Geral:** Todos os quatro algoritmos (HC, SA, GA_Config_1, GA_Config_4) conseguiram alcançar o mesmo valor de Melhor Fitness Geral de aproximadamente **0.0571**.
*   **Fitness Média e Consistência:**
    *   HC, SA, e GA_Config_4 apresentaram a mesma fitness média (0.0605) e o mesmo desvio padrão (0.0124).
    *   GA_Config_1 teve um desempenho ligeiramente inferior em fitness média (0.0688) e maior variabilidade (desvio padrão 0.0210).
*   **Eficiência (Tempo Médio de Execução):**
    *   Hill Climbing: ~0.63 segundos (mais rápido).
    *   GA_Config_1: ~5.97 segundos.
    *   GA_Config_4: ~8.30 segundos.
    *   Simulated Annealing: ~18.21 segundos (mais lento).

### 4.3. Conclusões (30 Execuções Iniciais)

*   **Hill Climbing** emergiu como o algoritmo mais eficiente, alcançando a mesma qualidade de solução que SA e GA_Config_4, mas numa fração do tempo.
*   HC, SA e GA_Config_4 foram igualmente consistentes.
*   O HC ofereceu o melhor equilíbrio entre qualidade, consistência e eficiência.

Esta análise levou à decisão de focar a fase final de variação de parâmetros no Hill Climbing e nas duas configurações de AG mais promissoras (GA_Config_1 e GA_Config_4), descartando o SA devido à sua lentidão comparativa sem ganho de qualidade sobre o HC ou o GA_Config_4 otimizado.




## 5. Fase de Variação de Parâmetros - 30 Execuções por Variação: Análise Estatística Detalhada

Esta secção detalha a análise estatística dos resultados obtidos após a execução de 30 repetições para cada uma das 19 variações de parâmetros aplicadas ao Hill Climbing (HC) e às duas configurações de Algoritmos Genéticos (GA_Config_1 e GA_Config_4). O objetivo desta fase foi identificar as configurações de hiperparâmetros mais eficazes para cada algoritmo.

### 5.1. Justificação da Escolha dos Algoritmos Genéticos para Variação de Parâmetros

As configurações `GA_Config_1_SwapConst1PtPreferVTournVarK` e `GA_Config_4_TargetExchUnifPreferVTournVarK_k5` foram selecionadas para a fase de variação de parâmetros detalhada com base no seu desempenho promissor na ronda anterior de 30 execuções (com parâmetros base). 

*   `GA_Config_1` foi a configuração de AG mais rápida nessa fase anterior que também demonstrou capacidade de alcançar a melhor fitness, embora com menor consistência que outras. A sua arquitetura, com `mutate_swap_constrained` e `crossover_one_point_prefer_valid`, representava uma abordagem mais clássica e menos disruptiva, cujo desempenho poderia ser afinado.
*   `GA_Config_4` destacou-se por alcançar a melhor fitness com uma consistência muito boa e um tempo de execução competitivo. A sua combinação de operadores (mutação direcionada `mutate_targeted_player_exchange` e cruzamento uniforme `crossover_uniform_prefer_valid`) sugeria um bom potencial para exploração e explotação do espaço de soluções, justificando uma investigação mais aprofundada dos seus parâmetros.

A expectativa era que, através da variação de parâmetros como taxa de mutação, tamanho da população e número de gerações, pudéssemos refinar ainda mais o desempenho destas duas arquiteturas de AG distintas.

### 5.2. Metodologia da Variação de Parâmetros

*   **Hill Climbing (HC):** Variou-se o **Número Máximo de Iterações**: [500, 1000 (base), 1500].
*   **GA_Config_1 e GA_Config_4:** Para cada uma, variou-se individualmente (mantendo os outros parâmetros base nos seus valores originais: PopSize=50, NumGen=100, MutRate=0.1):
    *   **Taxa de Mutação (MutRate):** [0.05, 0.15, 0.25]
    *   **Tamanho da População (PopSize):** [30, 75]
    *   **Número de Gerações (NumGen):** [75, 150]
Cada uma das 19 variações resultantes foi executada 30 vezes.

### 5.3. Resultados e Análise da Variação de Parâmetros

**Observação Chave Global:** Todas as 19 configurações testadas, ao longo das suas 30 execuções, foram capazes de encontrar pelo menos uma vez a mesma melhor fitness global de **0.0571**.

#### 5.3.1. Hill Climbing (HC) Otimizado

*   **HC_Iter_500 (Máx. Iterações = 500):**
    *   Fitness Média: 0.0588, Desvio Padrão: 0.0089, Tempo Médio: 0.41s.
*   **Conclusão para HC:** A configuração `HC_Iter_500` foi a mais eficaz, com a melhor fitness média, menor desvio padrão e tempo de execução extremamente baixo. Aumentar as iterações não trouxe melhorias.

#### 5.3.2. Algoritmo Genético - Configuração 1 (GA_Config_1) Otimizado

*   **Melhores Variações:**
    *   `GA_Config_1_mutation_rate_0.25` (Pop=50, Gen=100): Fitness Média: 0.0588, Desvio Padrão: 0.0089, Tempo: 6.13s.
    *   `GA_Config_1_population_size_75` (MutRate=0.1, Gen=100): Fitness Média: 0.0588, Desvio Padrão: 0.0089, Tempo: 9.08s.
*   **Conclusão para GA_Config_1:** Ambas as variações igualaram a qualidade e consistência do `HC_Iter_500`, mas com tempos de execução consideravelmente maiores. A variação com taxa de mutação de 0.25 foi mais eficiente em tempo.

#### 5.3.3. Algoritmo Genético - Configuração 4 (GA_Config_4) Otimizado

*   **Melhores Variações (atingiram consistentemente fitness média de 0.0571 com desvio padrão ~0.0):**
    *   `GA_Config_4_mutation_rate_0.15` (Pop=50, Gen=100): Tempo: 8.55s.
    *   `GA_Config_4_mutation_rate_0.25` (Pop=50, Gen=100): Tempo: 8.90s.
    *   `GA_Config_4_population_size_75` (MutRate=0.1, Gen=100): Tempo: 12.91s.
    *   `GA_Config_4_generations_150` (MutRate=0.1, Pop=50): Tempo: 12.44s.
*   **Conclusão para GA_Config_4:** Demonstrou um potencial notável, com várias variações alcançando consistentemente a melhor fitness média de 0.0571 com desvio padrão nulo. A `GA_Config_4_mutation_rate_0.15` foi a mais eficiente em tempo entre estas configurações de topo.

### 5.4. Conclusões Gerais da Variação de Parâmetros

1.  **Melhor Fitness Global Atingível:** 0.0571, alcançado por todas as variações.
2.  **Hill Climbing Otimizado (`HC_Iter_500`):** Destacou-se pela extrema eficiência (0.41s), excelente fitness média (0.0588) e boa consistência.
3.  **GA_Config_1 Otimizado (`GA_Config_1_mutation_rate_0.25`):** Igualou a qualidade do HC otimizado, mas mais lento (6.13s).
4.  **GA_Config_4 Otimizado (`GA_Config_4_mutation_rate_0.15`):** Atingiu consistentemente a melhor fitness média (0.0571) com desvio padrão nulo, sendo a mais rápida (8.55s) entre as que alcançaram este nível de perfeição.

**Recomendação Final de Algoritmos e Configurações Pós-Variação:**

*   **Para Eficiência Máxima com Excelente Qualidade:** `HC_Iter_500`.
*   **Para Garantia da Melhor Fitness com Consistência Perfeita:** `GA_Config_4_mutation_rate_0.15`.



## 6. Conclusões Finais e Recomendações

### 6.1. Síntese dos Resultados

Este projeto explorou a aplicação de algoritmos meta-heurísticos ao problema de otimização de seleção de equipas desportivas, progredindo desde implementações em processador único até análises detalhadas de variação de parâmetros. Os principais resultados podem ser sintetizados da seguinte forma:

1. **Otimizações de Código:** A redução do uso de `deepcopy` no Simulated Annealing resultou numa melhoria de desempenho de aproximadamente 51.7% no tempo total de execução.

2. **Multiprocessamento - 5 Execuções:** Todos os algoritmos conseguiram alcançar a mesma melhor fitness geral (0.057143), mas com diferenças significativas em consistência e tempo de execução. O Hill Climbing foi o mais rápido (0.47s), enquanto o Simulated Annealing e a GA_Config_4 foram os mais consistentes.

3. **Multiprocessamento - 30 Execuções:** Hill Climbing, Simulated Annealing e GA_Config_4 apresentaram desempenhos estatisticamente equivalentes em termos de qualidade da solução (fitness média 0.0605, desvio padrão 0.0124), mas com o Hill Climbing sendo significativamente mais rápido (0.63s vs. 18.21s para SA).

4. **Variação de Parâmetros - 30 Execuções por Variação:** 
   - O Hill Climbing com 500 iterações (`HC_Iter_500`) emergiu como a configuração mais eficiente, com excelente fitness média (0.0588), boa consistência (desvio padrão 0.0089) e tempo extremamente baixo (0.41s).
   - A GA_Config_4 com taxa de mutação de 0.15 (`GA_Config_4_mutation_rate_0.15`) alcançou consistentemente a melhor fitness média (0.0571) com desvio padrão nulo, sendo a mais rápida (8.55s) entre as configurações que garantiram este nível de perfeição.

### 6.2. Recomendações para Aplicação Prática

Com base nos resultados obtidos, recomendamos:

1. **Para Cenários com Restrições de Tempo:** Utilizar o Hill Climbing com 500 iterações (`HC_Iter_500`), que oferece um excelente equilíbrio entre qualidade da solução e eficiência computacional.

2. **Para Cenários que Exigem Garantia da Melhor Solução:** Utilizar o Algoritmo Genético GA_Config_4 com taxa de mutação de 0.15 (`GA_Config_4_mutation_rate_0.15`), que garante consistentemente a melhor fitness possível, embora com um custo computacional mais elevado.

3. **Para Aplicações de Larga Escala:** Considerar a implementação de uma abordagem híbrida, iniciando com Hill Climbing para obter rapidamente uma boa solução, seguido de refinamento com GA_Config_4 otimizado para casos específicos onde a qualidade máxima é crítica.

### 6.3. Sugestões para Trabalhos Futuros

1. **Exploração de Hibridização de Algoritmos:** Investigar a combinação de Hill Climbing com Algoritmos Genéticos, potencialmente utilizando HC para refinar soluções geradas por AGs.

2. **Análise de Escalabilidade:** Testar o desempenho dos algoritmos otimizados em problemas de maior dimensão, com mais equipas ou mais jogadores.

3. **Paralelização Interna dos Algoritmos:** Além do multiprocessamento para múltiplas execuções, explorar a paralelização interna dos algoritmos, especialmente para os Algoritmos Genéticos (avaliação paralela de indivíduos).

4. **Exploração de Técnicas de Machine Learning:** Investigar o uso de técnicas de aprendizagem para prever boas configurações de parâmetros com base nas características do problema.

## 7. Diagrama do Processo de Análise

O diagrama abaixo ilustra o processo de análise seguido neste projeto, desde a implementação inicial até às recomendações finais:

```
┌─────────────────────────┐
│ Implementação Inicial   │
│ (Processador Único)     │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Profiling e Otimização  │
│ - Redução de deepcopy   │
│ - Vetorização           │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Multiprocessamento      │
│ (5 Execuções)           │
│ - HC, SA, 4 configs AG  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Seleção de Algoritmos   │
│ Promissores             │
│ - HC, SA, GA1, GA4      │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Análise Estatística     │
│ (30 Execuções)          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Seleção para Variação   │
│ de Parâmetros           │
│ - HC, GA1, GA4          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Variação de Parâmetros  │
│ (30 Execuções/Variação) │
│ - 19 configurações      │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Análise Final e         │
│ Recomendações           │
│ - HC_Iter_500           │
│ - GA4_MutRate_0.15      │
└─────────────────────────┘
```

Este processo metodológico permitiu uma exploração sistemática do espaço de algoritmos e parâmetros, culminando em recomendações bem fundamentadas para a aplicação prática.
