# Análise Quantitativa Detalhada dos Algoritmos do Projeto CIFO EXTENDED

**Data:** 15 de maio de 2025

## 1. Introdução

Este documento apresenta uma análise quantitativa detalhada do desempenho dos algoritmos meta-heurísticos explorados no âmbito do projeto CIFO EXTENDED. O objetivo principal é fornecer uma base numérica robusta, incluindo dados comparativos de convergência, tempos de execução e observações estatísticas, para fundamentar as discussões e conclusões apresentadas nos relatórios finais do projeto. A análise abrange as diversas fases de experimentação, desde as execuções em processador único até às otimizações com multiprocessamento e variação de parâmetros, culminando na identificação das configurações algorítmicas mais eficientes e consistentes para o problema de seleção de equipas desportivas.

Serão examinadas métricas chave como a melhor fitness alcançada, a fitness média, o desvio padrão da fitness (como indicador de consistência) e o tempo médio de execução. Estes dados serão extraídos dos sumários gerados ao longo do projeto, permitindo uma comparação direta e informada entre as diferentes abordagens e configurações testadas.

## 2. Metodologia de Análise Quantitativa

A análise quantitativa baseia-se nos dados sumários recolhidos e armazenados em ficheiros CSV durante as várias fases do projeto. As principais métricas consideradas para cada algoritmo e configuração são:

*   **Melhor Fitness Geral (Overall Best Fitness):** O valor ótimo de fitness (neste caso, o menor valor da função objetivo, que representa a variância da média de habilidade entre equipas) encontrado em todas as execuções de uma dada configuração.
*   **Fitness Média (Mean Fitness):** A média dos melhores valores de fitness obtidos em cada uma das múltiplas execuções de uma configuração. Esta métrica indica o desempenho típico esperado do algoritmo.
*   **Desvio Padrão da Fitness (Std Dev Fitness):** Uma medida da variabilidade ou consistência do algoritmo em atingir soluções de qualidade. Valores mais baixos indicam maior consistência.
*   **Tempo Médio de Execução (Mean Exec Time):** O tempo médio, em segundos, que cada execução de uma configuração algorítmica levou para ser concluída.

Os dados serão apresentados em tabelas comparativas, e serão feitas referências aos gráficos de convergência e outros artefactos visuais gerados pelo projeto para ilustrar o comportamento dos algoritmos. A discussão focará na interpretação destes dados para identificar os pontos fortes e fracos de cada abordagem, justificando as escolhas feitas ao longo do projeto e as recomendações finais.

Embora testes estatísticos formais (como ANOVA ou t-tests) não tenham sido o foco principal da execução dos scripts originais para todas as comparações intermédias, a análise dos resultados de 30 execuções para os algoritmos promissores e para as variações de parâmetros permite inferências estatísticas baseadas na observação das médias, desvios padrão e na consistência com que os ótimos são atingidos. As secções de "Análise Estatística" nos relatórios de fase já abordam estas observações.

## 3. Resultados Quantitativos e Discussão

Nesta secção, apresentaremos os dados quantitativos chave das diferentes fases do projeto.




### 3.1. Fase de Multiprocessamento - 5 Execuções

Nesta fase inicial de multiprocessamento, cada algoritmo/configuração foi executado 5 vezes para uma primeira avaliação da sua consistência e desempenho. Os resultados sumariados são apresentados abaixo:

| Algoritmo                                                 | Fitness Média      | Desvio Padrão Fitness | Tempo Médio Execução (s) | Melhor Fitness Geral | Operador Mutação                     | Operador Cruzamento                | Operador Seleção                     |
| :-------------------------------------------------------- | :----------------- | :-------------------- | :----------------------- | :------------------- | :----------------------------------- | :--------------------------------- | :----------------------------------- |
| Hill Climbing (MP-5 runs)                                 | 0.067095           | 0.019905              | 0.4689                   | 0.057143             | N/A                                  | N/A                                | N/A                                  |
| Simulated Annealing (MP-5 runs)                           | 0.057143           | 0.000000              | 17.5987                  | 0.057143             | N/A                                  | N/A                                | N/A                                  |
| GA: GA_Config_1_SwapConst1PtPreferVTournVarK (MP-5 runs)  | 0.077048           | 0.024378              | 6.0489                   | 0.057143             | mutate_swap_constrained              | crossover_one_point_prefer_valid   | selection_tournament_variable_k    |
| GA: GA_Config_2_TargetExchUnifPreferVRanking (MP-5 runs)  | 0.088986           | 0.043328              | 61.9116                  | 0.057143             | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_ranking                    |
| GA: GA_Config_3_ShuffleWithin1PtPreferVBoltzmann (MP-5 runs) | 0.093613           | 0.032133              | 61.2230                  | 0.057143             | mutate_shuffle_within_team_constrained | crossover_one_point_prefer_valid   | selection_boltzmann                  |
| GA: GA_Config_4_TargetExchUnifPreferVTournVarK_k5 (MP-5 runs) | 0.057143           | 0.000000              | 8.1886                   | 0.057143             | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_tournament_variable_k    |

**Discussão (5 Execuções):**

Observa-se que o Simulated Annealing e a GA_Config_4 foram capazes de atingir a melhor fitness conhecida (0.057143) com consistência perfeita (desvio padrão de 0.0) já nesta fase de 5 execuções. O Hill Climbing, embora muito rápido (0.47s), apresentou uma fitness média superior e alguma variabilidade. As configurações GA_Config_2 e GA_Config_3 foram significativamente mais lentas e menos consistentes. A GA_Config_1, embora mais rápida que as GA_Config_2 e 3, também não demonstrou a mesma consistência que SA ou GA_Config_4. Estes resultados iniciais já apontavam para a superioridade do SA e da GA_Config_4 em termos de qualidade e consistência da solução, e do HC em termos de velocidade.




### 3.2. Fase de Multiprocessamento - 30 Execuções com Algoritmos Promissores

Após a seleção dos algoritmos mais promissores (Hill Climbing, Simulated Annealing, GA_Config_1 e GA_Config_4) com base nos resultados de 5 execuções, estes foram submetidos a 30 execuções para uma análise estatística mais robusta. Os resultados sumariados são:

| Algoritmo                                                       | Fitness Média      | Desvio Padrão Fitness | Tempo Médio Execução (s) | Melhor Fitness Geral | Operador Mutação                     | Operador Cruzamento                | Operador Seleção                     |
| :-------------------------------------------------------------- | :----------------- | :-------------------- | :----------------------- | :------------------- | :----------------------------------- | :--------------------------------- | :----------------------------------- |
| Hill Climbing (MP-30 runs)                                      | 0.060460           | 0.012413              | 0.6262                   | 0.057143             | N/A                                  | N/A                                | N/A                                  |
| Simulated Annealing (MP-30 runs)                                | 0.060460           | 0.012413              | 18.2145                  | 0.057143             | N/A                                  | N/A                                | N/A                                  |
| GA: GA_Config_1_SwapConst1PtPreferVTournVarK (MP-30 runs)       | 0.068754           | 0.021047              | 5.9698                   | 0.057143             | mutate_swap_constrained              | crossover_one_point_prefer_valid   | selection_tournament_variable_k    |
| GA: GA_Config_4_TargetExchUnifPreferVTournVarK_k5 (MP-30 runs)  | 0.060460           | 0.012413              | 8.2983                   | 0.057143             | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_tournament_variable_k    |

**Discussão (30 Execuções Iniciais):**

Com 30 execuções, observa-se que Hill Climbing, Simulated Annealing e GA_Config_4 apresentaram um desempenho muito similar em termos de fitness média (aproximadamente 0.06046) e desvio padrão (aproximadamente 0.0124). Todos conseguiram atingir a melhor fitness geral de 0.057143. A GA_Config_1 continuou a apresentar uma fitness média ligeiramente superior e maior variabilidade. Em termos de tempo, o Hill Climbing manteve-se como o mais rápido (0.63s), seguido pela GA_Config_1 (5.97s), GA_Config_4 (8.30s) e, por último, o Simulated Annealing (18.21s). Estes resultados reforçaram a eficiência do Hill Climbing e a boa performance da GA_Config_4, levando à seleção destes, juntamente com a GA_Config_1 (devido à sua velocidade entre os AGs), para a fase de variação de parâmetros. O Simulated Annealing foi descartado nesta fase devido à sua lentidão comparativa sem um ganho claro em qualidade de solução sobre o HC ou a GA_Config_4 nestas 30 execuções com parâmetros base.




### 3.3. Fase de Variação de Parâmetros - 30 Execuções por Variação

Nesta fase crucial, os algoritmos Hill Climbing, GA_Config_1 e GA_Config_4 foram submetidos a uma análise detalhada através da variação dos seus principais hiperparâmetros. Cada uma das 19 configurações resultantes foi executada 30 vezes para garantir robustez estatística. O objetivo era identificar as configurações ótimas para cada algoritmo.

Os resultados sumariados desta fase são apresentados na tabela seguinte:

| Algoritmo                                 | Fitness Média        | Desvio Padrão Fitness | Tempo Médio Execução (s) | Melhor Fitness Geral | Parâmetros                               | Operador Mutação                     | Operador Cruzamento                | Operador Seleção                     |
| :---------------------------------------- | :------------------- | :-------------------- | :----------------------- | :------------------- | :--------------------------------------- | :----------------------------------- | :--------------------------------- | :----------------------------------- |
| HC_MaxIter_500                            | 0.057143             | 1.3878e-17            | 0.4397                   | 0.057143             | MaxIter=500                              | N/A                                  | N/A                                | N/A                                  |
| HC_MaxIter_1000                           | 0.065436             | 0.018545              | 0.4083                   | 0.057143             | MaxIter=1000                             | N/A                                  | N/A                                | N/A                                  |
| HC_MaxIter_1500                           | 0.060460             | 0.012413              | 0.4264                   | 0.057143             | MaxIter=1500                             | N/A                                  | N/A                                | N/A                                  |
| GA_Config_1_BaseParams                    | 0.068754             | 0.021047              | 5.9838                   | 0.057143             | Pop=50,Gen=100,MutRate=0.1               | mutate_swap_constrained              | crossover_one_point_prefer_valid   | selection_tournament_variable_k    |
| GA_Config_1_mutation_rate_0.05            | 0.078150             | 0.026369              | 5.8403                   | 0.057143             | Pop=50,Gen=100,MutRate=0.05              | mutate_swap_constrained              | crossover_one_point_prefer_valid   | selection_tournament_variable_k    |
| GA_Config_1_mutation_rate_0.15            | 0.063778             | 0.016916              | 6.1132                   | 0.057143             | Pop=50,Gen=100,MutRate=0.15              | mutate_swap_constrained              | crossover_one_point_prefer_valid   | selection_tournament_variable_k    |
| GA_Config_1_mutation_rate_0.25            | 0.057143             | 1.3878e-17            | 6.2534                   | 0.057143             | Pop=50,Gen=100,MutRate=0.25              | mutate_swap_constrained              | crossover_one_point_prefer_valid   | selection_tournament_variable_k    |
| GA_Config_1_population_size_30            | 0.069856             | 0.023713              | 3.5065                   | 0.057143             | Pop=30,Gen=100,MutRate=0.1               | mutate_swap_constrained              | crossover_one_point_prefer_valid   | selection_tournament_variable_k    |
| GA_Config_1_population_size_75            | 0.057143             | 1.3878e-17            | 9.0406                   | 0.057143             | Pop=75,Gen=100,MutRate=0.1               | mutate_swap_constrained              | crossover_one_point_prefer_valid   | selection_tournament_variable_k    |
| GA_Config_1_generations_75                | 0.065436             | 0.018545              | 4.5641                   | 0.057143             | Pop=50,Gen=75,MutRate=0.1                | mutate_swap_constrained              | crossover_one_point_prefer_valid   | selection_tournament_variable_k    |
| GA_Config_1_generations_150               | 0.063778             | 0.016916              | 8.9005                   | 0.057143             | Pop=50,Gen=150,MutRate=0.1               | mutate_swap_constrained              | crossover_one_point_prefer_valid   | selection_tournament_variable_k    |
| GA_Config_4_BaseParams                    | 0.060460             | 0.012413              | 8.2111                   | 0.057143             | Pop=50,Gen=100,MutRate=0.1               | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_tournament_variable_k    |
| GA_Config_4_mutation_rate_0.05            | 0.060460             | 0.012413              | 8.3009                   | 0.057143             | Pop=50,Gen=100,MutRate=0.05              | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_tournament_variable_k    |
| GA_Config_4_mutation_rate_0.15            | 0.057143             | 1.3878e-17            | 8.6745                   | 0.057143             | Pop=50,Gen=100,MutRate=0.15              | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_tournament_variable_k    |
| GA_Config_4_mutation_rate_0.25            | 0.057143             | 1.3878e-17            | 8.8816                   | 0.057143             | Pop=50,Gen=100,MutRate=0.25              | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_tournament_variable_k    |
| GA_Config_4_population_size_30            | 0.069085             | 0.025874              | 4.8292                   | 0.057143             | Pop=30,Gen=100,MutRate=0.1               | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_tournament_variable_k    |
| GA_Config_4_population_size_75            | 0.057143             | 1.3878e-17            | 12.7402                  | 0.057143             | Pop=75,Gen=100,MutRate=0.1               | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_tournament_variable_k    |
| GA_Config_4_generations_75                | 0.065436             | 0.018545              | 6.4097                   | 0.057143             | Pop=50,Gen=75,MutRate=0.1                | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_tournament_variable_k    |
| GA_Config_4_generations_150               | 0.058802             | 0.008932              | 12.4701                  | 0.057143             | Pop=50,Gen=150,MutRate=0.1               | mutate_targeted_player_exchange    | crossover_uniform_prefer_valid     | selection_tournament_variable_k    |

**Discussão (Variação de Parâmetros):**

*   **Hill Climbing (HC):** A configuração `HC_MaxIter_500` destacou-se como a ideal, atingindo a melhor fitness (0.057143) com consistência perfeita (desvio padrão praticamente nulo) e o tempo de execução mais baixo entre todas as variações de HC (0.44s). Aumentar as iterações não trouxe melhorias na qualidade da solução e, em alguns casos, piorou a fitness média, sugerindo convergência rápida.

*   **GA_Config_1:** As variações `GA_Config_1_mutation_rate_0.25` (Pop=50, Gen=100, MutRate=0.25) e `GA_Config_1_population_size_75` (Pop=75, Gen=100, MutRate=0.1) foram as melhores, ambas alcançando a fitness média de 0.057143 com consistência perfeita. A primeira foi mais rápida (6.25s vs 9.04s).

*   **GA_Config_4:** As variações `GA_Config_4_mutation_rate_0.15` (Pop=50, Gen=100, MutRate=0.15), `GA_Config_4_mutation_rate_0.25` (Pop=50, Gen=100, MutRate=0.25), e `GA_Config_4_population_size_75` (Pop=75, Gen=100, MutRate=0.1) foram as mais performantes, todas atingindo a fitness média de 0.057143 com consistência perfeita. Destas, `GA_Config_4_mutation_rate_0.15` foi a mais rápida (8.67s).

**Conclusões Gerais da Variação de Parâmetros:**

Os resultados demonstram que, com a otimização de parâmetros, tanto o Hill Climbing como as duas configurações de Algoritmos Genéticos conseguem atingir consistentemente a melhor solução conhecida (fitness de 0.057143). O `HC_MaxIter_500` permanece como a opção mais rápida por uma margem significativa. Entre os AGs otimizados, `GA_Config_1_mutation_rate_0.25` é ligeiramente mais rápido que `GA_Config_4_mutation_rate_0.15`, ambos oferecendo excelente qualidade e consistência. A escolha final dependerá do equilíbrio entre tempo de execução e a necessidade de exploração mais ampla do espaço de soluções que os AGs proporcionam.

## 4. Conclusões da Análise Quantitativa

(A ser preenchido após a análise completa de todas as fases e a integração com as discussões dos relatórios principais.)




## 4. Conclusões da Análise Quantitativa

A análise quantitativa dos resultados obtidos ao longo das diversas fases do projeto CIFO EXTENDED permite extrair conclusões importantes sobre o desempenho e a adequação dos algoritmos meta-heurísticos testados para o problema da formação equilibrada de equipas desportivas.

1.  **Consistência na Obtenção da Melhor Solução:** Várias configurações algorítmicas, nomeadamente o `HC_MaxIter_500`, e as versões otimizadas de `GA_Config_1` (com taxa de mutação de 0.25 ou tamanho de população de 75) e `GA_Config_4` (com taxa de mutação de 0.15 ou 0.25, ou tamanho de população de 75), demonstraram a capacidade de atingir consistentemente a melhor fitness conhecida de **0.057143** (desvio padrão da fitness praticamente nulo) após 30 execuções. Isto indica que, com os parâmetros corretos, estes algoritmos são robustos na procura da solução ótima identificada.

2.  **Eficiência do Hill Climbing:** O algoritmo Hill Climbing, especificamente com 500 iterações (`HC_MaxIter_500`), destacou-se consistentemente como o mais eficiente em termos de tempo de execução. Conseguiu encontrar a melhor solução em aproximadamente **0.44 segundos** em média, uma fração do tempo exigido pelos Algoritmos Genéticos, mesmo nas suas configurações otimizadas.

3.  **Desempenho dos Algoritmos Genéticos Otimizados:** As configurações otimizadas dos Algoritmos Genéticos (`GA_Config_1_mutation_rate_0.25` e `GA_Config_4_mutation_rate_0.15`) também alcançaram a melhor fitness com perfeita consistência. No entanto, os seus tempos médios de execução foram significativamente mais altos (aproximadamente **6.25 segundos** e **8.67 segundos**, respetivamente). Embora mais lentos que o HC, os AGs oferecem um mecanismo de exploração do espaço de soluções potencialmente mais abrangente, o que pode ser vantajoso em problemas mais complexos ou com paisagens de fitness mais enganosas.

4.  **Impacto da Variação de Parâmetros:** A fase de variação de parâmetros foi crucial para identificar as configurações que maximizam o desempenho dos AGs. Observou-se que taxas de mutação mais elevadas (e.g., 0.15 ou 0.25 para GA_Config_1 e GA_Config_4) e tamanhos de população maiores (e.g., 75) tendem a melhorar a qualidade da solução e a consistência, embora com um custo no tempo de execução. O número de gerações também influenciou positivamente, mas o impacto da taxa de mutação e do tamanho da população pareceu mais determinante para alcançar a fitness ótima com consistência.

5.  **Simulated Annealing:** Embora o Simulated Annealing tenha demonstrado boa consistência nas fases iniciais (5 execuções), o seu tempo de execução relativamente elevado (cerca de 18 segundos na fase de 30 execuções com parâmetros base) sem oferecer uma melhor qualidade de solução em comparação com o HC otimizado ou os AGs otimizados, justificou a sua exclusão da fase final de variação de parâmetros para este problema específico.

Em suma, a análise quantitativa suporta a conclusão de que o `HC_MaxIter_500` é a escolha mais pragmática para este problema se a velocidade for prioritária e a solução ótima conhecida for satisfatória. Se for desejável uma exploração mais diversificada do espaço de soluções, com garantia de atingir a melhor fitness, as configurações otimizadas de `GA_Config_4` (com taxa de mutação de 0.15) ou `GA_Config_1` (com taxa de mutação de 0.25) são alternativas viáveis, embora mais consumidoras de tempo. Estes dados fornecem uma base sólida para as recomendações apresentadas no relatório final do projeto.

