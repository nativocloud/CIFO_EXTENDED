# Análise Estatística dos Resultados da Variação de Parâmetros (30 Execuções)

Esta secção detalha a análise estatística dos resultados obtidos após a execução de 30 repetições para cada uma das 19 variações de parâmetros dos algoritmos Hill Climbing (HC), GA_Config_1 e GA_Config_4. O objetivo é identificar as configurações de parâmetros que otimizam o desempenho de cada algoritmo e, subsequentemente, comparar os melhores algoritmos otimizados entre si.

Os dados completos desta fase de experimentação foram guardados em `/home/ubuntu/CIFO_EXTENDED_Project/images_mp/final_param_var_results/all_algorithms_summary_final_param_var_30runs.csv`.

## Metodologia de Análise

A análise focou-se em três métricas principais para cada variação de algoritmo:

1.  **Melhor Fitness Geral (Overall Best Fitness):** A melhor solução (menor desvio padrão) encontrada em todas as 30 execuções.
2.  **Fitness Média (Mean Fitness):** A média das melhores fitness encontradas em cada uma das 30 execuções. Indica o desempenho típico do algoritmo.
3.  **Desvio Padrão da Fitness (Std Dev Fitness):** A medida da consistência do algoritmo. Valores menores indicam que o algoritmo encontra consistentemente soluções de qualidade semelhante.
4.  **Tempo Médio de Execução (Mean Exec Time):** O tempo médio que cada execução do algoritmo levou para ser concluída.

Devido à natureza dos dados e ao objetivo de identificar as configurações mais performantes, a análise comparativa será primariamente qualitativa, observando as tendências e os valores extremos nas métricas acima. Para uma análise formal, seriam aplicados testes de normalidade (e.g., Shapiro-Wilk) seguidos de testes paramétricos (ANOVA, teste t) ou não paramétricos (Kruskal-Wallis, Mann-Whitney U) com correções post-hoc, conforme a distribuição dos dados. No entanto, para este relatório, focaremos na interpretação direta dos resultados sumariados.

## Resultados e Análise por Algoritmo

### 1. Hill Climbing (HC)

Foram testadas três variações para o número máximo de iterações:

*   **HC_MaxIter_500:**
    *   Fitness Média: 0.0571
    *   Desvio Padrão da Fitness: 1.38e-17 (essencialmente zero, indicando consistência perfeita)
    *   Melhor Fitness Geral: 0.0571
    *   Tempo Médio: 0.44s
*   **HC_MaxIter_1000:**
    *   Fitness Média: 0.0654
    *   Desvio Padrão da Fitness: 0.0185
    *   Melhor Fitness Geral: 0.0571
    *   Tempo Médio: 0.41s
*   **HC_MaxIter_1500:**
    *   Fitness Média: 0.0605
    *   Desvio Padrão da Fitness: 0.0124
    *   Melhor Fitness Geral: 0.0571
    *   Tempo Médio: 0.43s

**Conclusão para HC:** A configuração `HC_MaxIter_500` demonstrou ser a ideal. Atingiu a melhor fitness possível (0.0571) com consistência perfeita (desvio padrão praticamente nulo) e um tempo de execução muito baixo. Aumentar o número de iterações para 1000 ou 1500 não melhorou a qualidade da melhor solução encontrada e, em alguns casos (como MaxIter_1000), até piorou a fitness média e a consistência, sugerindo que o HC converge muito rapidamente para o ótimo (ou um ótimo local de alta qualidade) e iterações adicionais não trazem benefício.

### 2. Algoritmo Genético - Configuração 1 (GA_Config_1_SwapConst1PtPreferVTournVarK)

O GA_Config_1 base (Pop=50, Gen=100, MutRate=0.1) apresentou:
*   Fitness Média: 0.0688
*   Desvio Padrão da Fitness: 0.0210
*   Melhor Fitness Geral: 0.0571
*   Tempo Médio: 5.98s

Analisando as variações:

*   **Variação da Taxa de Mutação (MutRate):**
    *   `MutRate=0.05`: Fitness Média 0.0781 (pior), StdDev 0.0264, Tempo 5.84s.
    *   `MutRate=0.15`: Fitness Média 0.0638 (melhor que base), StdDev 0.0169, Tempo 6.11s.
    *   `MutRate=0.25`: Fitness Média 0.0571 (excelente), StdDev 1.38e-17 (consistência perfeita), Tempo 6.25s.
    A taxa de mutação de 0.25 resultou num desempenho notavelmente superior, atingindo a melhor fitness com consistência perfeita.

*   **Variação do Tamanho da População (PopSize):**
    *   `PopSize=30`: Fitness Média 0.0699 (ligeiramente pior que base), StdDev 0.0237, Tempo 3.51s (mais rápido).
    *   `PopSize=75`: Fitness Média 0.0571 (excelente), StdDev 1.38e-17 (consistência perfeita), Tempo 9.04s (mais lento).
    Aumentar o tamanho da população para 75 melhorou drasticamente a qualidade e consistência, ao custo de maior tempo de execução.

*   **Variação do Número de Gerações (NumGen):**
    *   `NumGen=75`: Fitness Média 0.0654 (ligeiramente melhor que base), StdDev 0.0185, Tempo 4.56s (mais rápido).
    *   `NumGen=150`: Fitness Média 0.0638 (melhor que base), StdDev 0.0169, Tempo 8.90s (mais lento).
    Aumentar as gerações para 150 melhorou a fitness média e a consistência em relação à base.

**Melhores Configurações para GA_Config_1:**
As variações `GA_Config_1_mutation_rate_0.25` (Pop=50, Gen=100, MutRate=0.25) e `GA_Config_1_population_size_75` (Pop=75, Gen=100, MutRate=0.1) destacaram-se, ambas atingindo a fitness média de 0.0571 com consistência perfeita. A primeira é mais rápida (6.25s vs 9.04s).

### 3. Algoritmo Genético - Configuração 4 (GA_Config_4_TargetExchUnifPreferVTournVarK_k5)

O GA_Config_4 base (Pop=50, Gen=100, MutRate=0.1) apresentou:
*   Fitness Média: 0.0605
*   Desvio Padrão da Fitness: 0.0124
*   Melhor Fitness Geral: 0.0571
*   Tempo Médio: 8.21s

Analisando as variações:

*   **Variação da Taxa de Mutação (MutRate):**
    *   `MutRate=0.05`: Fitness Média 0.0605 (igual à base), StdDev 0.0124, Tempo 8.30s.
    *   `MutRate=0.15`: Fitness Média 0.0571 (excelente), StdDev 1.38e-17 (consistência perfeita), Tempo 8.67s.
    *   `MutRate=0.25`: Fitness Média 0.0571 (excelente), StdDev 1.38e-17 (consistência perfeita), Tempo 8.88s.
    Taxas de mutação de 0.15 e 0.25 foram ótimas, com 0.15 sendo ligeiramente mais rápida.

*   **Variação do Tamanho da População (PopSize):**
    *   `PopSize=30`: Fitness Média 0.0691 (pior que base), StdDev 0.0259, Tempo 4.83s (mais rápido).
    *   `PopSize=75`: Fitness Média 0.0571 (excelente), StdDev 1.38e-17 (consistência perfeita), Tempo 12.74s (mais lento).
    Aumentar a população para 75 foi benéfico para qualidade e consistência, mas aumentou o tempo.

*   **Variação do Número de Gerações (NumGen):**
    *   `NumGen=75`: Fitness Média 0.0654 (pior que base), StdDev 0.0185, Tempo 6.41s (mais rápido).
    *   `NumGen=150`: Fitness Média 0.0588 (melhor que base), StdDev 0.0089 (boa consistência), Tempo 12.47s (mais lento).
    Aumentar as gerações para 150 melhorou a fitness média e a consistência.

**Melhores Configurações para GA_Config_4:**
As variações `GA_Config_4_mutation_rate_0.15` (Pop=50, Gen=100, MutRate=0.15), `GA_Config_4_mutation_rate_0.25` (Pop=50, Gen=100, MutRate=0.25), e `GA_Config_4_population_size_75` (Pop=75, Gen=100, MutRate=0.1) foram as melhores, todas atingindo fitness média de 0.0571 com consistência perfeita. Entre estas, `GA_Config_4_mutation_rate_0.15` foi a mais rápida (8.67s).

## Conclusões Gerais da Variação de Parâmetros

1.  **Hill Climbing:** A configuração `HC_MaxIter_500` é extremamente eficiente e robusta, consistentemente encontrando a melhor solução conhecida (0.0571) em menos de 0.5 segundos.

2.  **GA_Config_1 Otimizada:** A configuração `GA_Config_1_mutation_rate_0.25` (Pop=50, Gen=100, MutRate=0.25) demonstrou ser a melhor para esta arquitetura de AG, atingindo a melhor fitness (0.0571) com consistência perfeita e um tempo médio de 6.25s.

3.  **GA_Config_4 Otimizada:** A configuração `GA_Config_4_mutation_rate_0.15` (Pop=50, Gen=100, MutRate=0.15) foi a mais equilibrada para esta arquitetura, também atingindo a melhor fitness (0.0571) com consistência perfeita, num tempo médio de 8.67s.

Comparando os melhores desempenhos otimizados:

*   **HC_MaxIter_500:** Fitness Média 0.0571, StdDev ~0, Tempo ~0.44s.
*   **GA_Config_1_mutation_rate_0.25:** Fitness Média 0.0571, StdDev ~0, Tempo ~6.25s.
*   **GA_Config_4_mutation_rate_0.15:** Fitness Média 0.0571, StdDev ~0, Tempo ~8.67s.

Todos os três algoritmos, com os seus parâmetros otimizados, conseguem encontrar consistentemente a melhor solução conhecida. O Hill Climbing continua a ser ordens de magnitude mais rápido. Entre os AGs, o GA_Config_1 otimizado é ligeiramente mais rápido que o GA_Config_4 otimizado, ambos oferecendo excelente qualidade de solução e consistência.

A escolha final do algoritmo dependerá do equilíbrio desejado entre tempo de execução e a confiança de explorar diferentes partes do espaço de soluções que os AGs podem oferecer em comparação com a natureza mais local do HC. No entanto, para este problema específico e com os resultados obtidos, o **HC_MaxIter_500** apresenta-se como a solução mais pragmática e eficiente.

